// backend_vulkan_hpp.cpp — Vulkan inference backend.
// Uses Vulkan C API (via vulkan.h) with the ZINC SPIR-V shaders.
// Integrated into BackendManager + DynamicRouter for unified Vulkan+ROCm+NPU routing.

#include "backend.h"
#include "backend_vulkan_hpp.h"
#include "../engine/npu/src/onebp_loader.cpp"

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <memory>
#include <chrono>
#include <fstream>
#include <cstring>
#include <cstdlib>

static constexpr float EPS = 1e-6f;

// ═══════════════════════════════════════════════════════════════════════════
// Vulkan context (C-style, proven from ZINC backend)
// ═══════════════════════════════════════════════════════════════════════════
struct VkCtx {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physDev = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t qf = VK_QUEUE_FAMILY_IGNORED;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    bool ok = false;
    char devName[256] = {};

    ~VkCtx() {
        if (descPool) vkDestroyDescriptorPool(device, descPool, nullptr);
        if (cmdPool) vkDestroyCommandPool(device, cmdPool, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

static VkCtx* create_vk_context() {
    auto* ctx = new VkCtx();

    // Instance
    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "1bit", 1, "Zaya", 1, VK_API_VERSION_1_3};
    const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
    const char* exts[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &appInfo,
                                1, layers, 1, exts};
    if (vkCreateInstance(&ici, nullptr, &ctx->instance) != VK_SUCCESS) {
        fprintf(stderr, "[vk] instance failed\n"); delete ctx; return nullptr;
    }

    // Physical device
    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &nd, nullptr);
    if (nd == 0) { fprintf(stderr, "[vk] no devices\n"); delete ctx; return nullptr; }
    std::vector<VkPhysicalDevice> devs(nd);
    vkEnumeratePhysicalDevices(ctx->instance, &nd, devs.data());

    for (auto pd : devs) {
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qps(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qps.data());
        for (uint32_t i = 0; i < nq; i++) {
            if (qps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                ctx->physDev = pd;
                ctx->qf = i;
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(pd, &props);
                strncpy(ctx->devName, props.deviceName, 255);
                break;
            }
        }
        if (ctx->qf != VK_QUEUE_FAMILY_IGNORED) break;
    }
    if (ctx->qf == VK_QUEUE_FAMILY_IGNORED) {
        fprintf(stderr, "[vk] no compute queue\n"); delete ctx; return nullptr;
    }
    fprintf(stderr, "[vk] device: %s\n", ctx->devName);

    // Device
    float qp = 1.0f;
    VkDeviceQueueCreateInfo dqci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, ctx->qf, 1, &qp};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &dqci, 0, nullptr, 0, nullptr, nullptr};
    if (vkCreateDevice(ctx->physDev, &dci, nullptr, &ctx->device) != VK_SUCCESS) {
        fprintf(stderr, "[vk] device creation failed\n"); delete ctx; return nullptr;
    }
    vkGetDeviceQueue(ctx->device, ctx->qf, 0, &ctx->queue);

    // Command pool
    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
                                     VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, ctx->qf};
    if (vkCreateCommandPool(ctx->device, &cpci, nullptr, &ctx->cmdPool) != VK_SUCCESS) {
        fprintf(stderr, "[vk] cmd pool failed\n"); delete ctx; return nullptr;
    }

    // Descriptor pool
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 65536};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr,
                                        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, 65536, 1, &ps};
    if (vkCreateDescriptorPool(ctx->device, &dpci, nullptr, &ctx->descPool) != VK_SUCCESS) {
        fprintf(stderr, "[vk] desc pool failed\n"); delete ctx; return nullptr;
    }

    ctx->ok = true;
    return ctx;
}

// ═══════════════════════════════════════════════════════════════════════════
// SPIR-V shader loader
// ═══════════════════════════════════════════════════════════════════════════
struct VkShader {
    VkShaderModule module = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

static std::vector<uint32_t> read_spv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    size_t n = f.tellg() / sizeof(uint32_t);
    f.seekg(0);
    std::vector<uint32_t> code(n);
    f.read((char*)code.data(), n * sizeof(uint32_t));
    return code;
}

static VkShader load_vk_shader(VkDevice dev, const std::string& dir, const std::string& name,
                                const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
    VkShader s;
    auto code = read_spv(dir + "/" + name + ".spv");
    if (code.empty()) { fprintf(stderr, "[vk] missing: %s\n", name.c_str()); return s; }

    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                      code.size() * sizeof(uint32_t), code.data()};
    if (vkCreateShaderModule(dev, &smci, nullptr, &s.module) != VK_SUCCESS) return s;

    VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0,
                                              (uint32_t)bindings.size(), bindings.data()};
    if (vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &s.descLayout) != VK_SUCCESS) return s;

    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0,
                                       1, &s.descLayout, 1, &pcr};
    if (vkCreatePipelineLayout(dev, &plci, nullptr, &s.pipeLayout) != VK_SUCCESS) return s;

    VkPipelineShaderStageCreateInfo pssci = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                                              VK_SHADER_STAGE_COMPUTE_BIT, s.module, "main", nullptr};
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0, pssci, s.pipeLayout, VK_NULL_HANDLE, 0};
    if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &s.pipeline) != VK_SUCCESS) return s;

    return s;
}

static void destroy_vk_shader(VkDevice dev, VkShader& s) {
    if (s.pipeline) vkDestroyPipeline(dev, s.pipeline, nullptr);
    if (s.pipeLayout) vkDestroyPipelineLayout(dev, s.pipeLayout, nullptr);
    if (s.descLayout) vkDestroyDescriptorSetLayout(dev, s.descLayout, nullptr);
    if (s.module) vkDestroyShaderModule(dev, s.module, nullptr);
}

// ═══════════════════════════════════════════════════════════════════════════
// Vulkan storage buffer (device-local, host-visible)
// ═══════════════════════════════════════════════════════════════════════════
struct VkBuf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* mapped = nullptr;
    size_t size = 0;
};

static VkBuf create_vk_buf(VkDevice dev, VkPhysicalDevice physDev, size_t bytes,
                            VkBufferUsageFlags usage, VkMemoryPropertyFlags props) {
    VkBuf b;
    if (bytes == 0) return b;
    b.size = bytes;
    if (bytes > 100000000) fprintf(stderr, "[vk] creating large buf: %zu MB\n", bytes/1000000);

    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, bytes, usage, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    VkResult r = vkCreateBuffer(dev, &bci, nullptr, &b.buf);
    if (r != VK_SUCCESS) { fprintf(stderr, "[vk] createBuffer(%zu) failed: %d\n", bytes, r); return b; }

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, b.buf, &mr);

    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physDev, &mp);
    uint32_t memType = VK_MAX_MEMORY_TYPES;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) {
            memType = i; break;
        }
    }
    if (memType == VK_MAX_MEMORY_TYPES) { fprintf(stderr, "[vk] no mem type\n"); return b; }

    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size, memType};
    r = vkAllocateMemory(dev, &mai, nullptr, &b.mem);
    if (r != VK_SUCCESS) { fprintf(stderr, "[vk] allocMemory(%zu) failed: %d\n", bytes, r); return b; }
    vkBindBufferMemory(dev, b.buf, b.mem, 0);
    r = vkMapMemory(dev, b.mem, 0, bytes, 0, &b.mapped);
    if (r != VK_SUCCESS) { fprintf(stderr, "[vk] mapMemory(%zu) failed: %d\n", bytes, r); return b; }
    return b;
}

static void destroy_vk_buf(VkDevice dev, VkBuf& b) {
    if (b.mapped) vkUnmapMemory(dev, b.mem);
    if (b.buf) vkDestroyBuffer(dev, b.buf, nullptr);
    if (b.mem) vkFreeMemory(dev, b.mem, nullptr);
}

// ═══════════════════════════════════════════════════════════════════════════
// VulkanBackend — unified inference engine
// ═══════════════════════════════════════════════════════════════════════════
struct VulkanBackend : Backend {
    VkCtx* vk = nullptr;
    std::vector<VkShader> shaders;
    std::string shaderDir;
    bool vk_ok = false;

    int H = 0, NC = 0, NH = 0, NKV = 0, HD_ = 128, IM = 0, VOCAB = 0;
    float rope_theta = 10000.0f;
    int max_seq = 4096, pos = 0;

    VkBuf bufEmbed, bufFinalNorm, bufOutput;
    struct LBuf { VkBuf wq, wk, wv, wo, w1, w2, w3, pn, pon; };
    std::vector<LBuf> layers;

    VkBuf bHidden, bResid, bQ, bK, bV, bQKV, bAttn;
    VkBuf bGate, bUp, bGateAll, bSilu, bFFNOut, bLogits;
    VkBuf kCache, vCache;
    VkBuf bKVCacheAll, bDummy;

    // CPU weight copies (for lm_head + CPU fallback)
    std::vector<float> cpuEmbed, cpuFinalNorm, cpuOutput;
    struct CpuL { std::vector<float> wq,wk,wv,wo,w1,w2,w3,pn,pon; };
    std::vector<CpuL> cpuL;

    VulkanBackend() { type = BackendType::GENERIC; name = "Vulkan GPU"; }
    ~VulkanBackend() override { destroy(); }

    bool init(const ModelConfig& cfg, const std::string&) override {
        this->cfg = cfg;
        H = cfg.hidden_size; NC = cfg.num_layers; NH = cfg.num_heads;
        NKV = cfg.num_kv_heads; HD_ = cfg.head_dim; IM = cfg.intermediate_size;
        VOCAB = cfg.vocab_size;
        rope_theta = cfg.rope_theta > 0 ? cfg.rope_theta : 10000.0f;
        if (NKV == 0) NKV = NH; if (HD_ == 0) HD_ = 128;
        printf("[vk] H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d V=%d\n", H, NC, NH, NKV, HD_, IM, VOCAB);

        vk = create_vk_context();
        if (!vk || !vk->ok) { fprintf(stderr, "[vk] context failed\n"); return false; }

        shaderDir = "build/zinc_cpp_build/shaders";
        fprintf(stderr, "[vk] shader dir: %s\n", shaderDir.c_str());

        // Load shaders
        VkDescriptorSetLayoutBinding ssb = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};
        auto ldShader = [&](const std::string& name, int nbufs) {
            // Allocate max 6 bindings — covers all shaders. Unused bindings are harmless.
            std::vector<VkDescriptorSetLayoutBinding> bs;
            int maxBind = (nbufs > 6) ? nbufs : 6;
            for (int i = 0; i < maxBind; i++) { ssb.binding = i; bs.push_back(ssb); }
            return load_vk_shader(vk->device, shaderDir, name, bs);
        };

        shaders.push_back(ldShader("gemv_f32", 3));       // 0
        shaders.push_back(ldShader("rms_norm", 2));        // 1
        shaders.push_back(ldShader("rope_fused", 2));      // 2
        shaders.push_back(ldShader("embed", 2));           // 3
        shaders.push_back(ldShader("vadd_f32", 3));        // 4
        shaders.push_back(ldShader("copy_buffer", 2));     // 5
        shaders.push_back(ldShader("fused_gate_up", 4));   // 6
        shaders.push_back(ldShader("swiglu", 3));          // 7
        shaders.push_back(ldShader("fused_silu_down", 4)); // 8
        shaders.push_back(ldShader("flash_attn", 6));      // 9 — Q,K,V_cache,out,seq,scale
        shaders.push_back(ldShader("argmax", 3));          // 10
        shaders.push_back(ldShader("rms_norm_mul", 2));    // 11

        for (auto& s : shaders)
            if (!s.pipeline) { fprintf(stderr, "[vk] shader load failed\n"); return false; }
        fprintf(stderr, "[vk] %zu shaders loaded\n", shaders.size());

        // Create buffers
        auto dev = vk->device;
        auto phys = vk->physDev;
        auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        auto props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        auto mkBuf = [&](VkBuf& b, size_t n, const char* tag) {
            if (n == 0) return;
            b = create_vk_buf(dev, phys, n * sizeof(float), usage, props);
        };

        // Scratch and cache buffers (all host-visible for UMA access)
        int qkvSize = (NH + 2 * NKV) * HD_;
        mkBuf(bHidden,  H,       "hidden");
        mkBuf(bResid,   H,       "resid");
        mkBuf(bQ,       NH*HD_,  "bQ");
        mkBuf(bK,       NKV*HD_, "bK");
        mkBuf(bV,       NKV*HD_, "bV");
        mkBuf(bQKV,     qkvSize, "bQKV");
        mkBuf(bAttn,    NH*HD_,  "bAttn");
        mkBuf(bGate,    IM,      "bGate");
        mkBuf(bUp,      IM,      "bUp");
        mkBuf(bGateAll, 2*IM,    "bGateAll");
        mkBuf(bSilu,    IM,      "bSilu");
        mkBuf(bFFNOut,  H,       "bFFNOut");
        mkBuf(bLogits,  VOCAB,   "bLogits");
        // Full layered KV cache for flash_attn: [2][NC][max_seq][NKV*HD]
        size_t kvcTotal = (size_t)2 * NC * max_seq * NKV * HD_;
        mkBuf(bKVCacheAll, kvcTotal, "bKVCacheAll");
        if (bKVCacheAll.mapped) memset(bKVCacheAll.mapped, 0, kvcTotal * sizeof(float));
        // Dummy buffer for unused descriptor set bindings
        mkBuf(bDummy, 64, "dummy");
        // Per-layer simple KV cache (kept for reset())
        mkBuf(kCache, (size_t)max_seq * NKV * HD_, "k_cache");
        mkBuf(vCache, (size_t)max_seq * NKV * HD_, "v_cache");
        memset(kCache.mapped, 0, (size_t)max_seq * NKV * HD_ * sizeof(float));
        memset(vCache.mapped, 0, (size_t)max_seq * NKV * HD_ * sizeof(float));

        // Load model (also resizes layers and uploads weights)
        if (!load_1bp(cfg.model_path)) return false;

        vk_ok = true; initialized = true;
        printf("[vk] ✅ Vulkan backend ready\n");
        return true;
    }

    bool load_1bp(const std::string& path) {
        fprintf(stderr, "[vk] Loading: %s\n", path.c_str());
        OnebpModel mdl;
        if (!mdl.open(path.c_str())) { fprintf(stderr, "[vk] open fail\n"); return false; }
        fprintf(stderr, "[vk] model opened\n");

        auto ld = [&](const char* n, std::vector<float>& v) { return mdl.get_tensor_f32(n, v); };
        ld("token_embd.weight", cpuEmbed);
        fprintf(stderr, "[vk] token_embd: %zu\n", cpuEmbed.size());
        if (!ld("output_norm.weight", cpuFinalNorm)) ld("token_embd_norm.weight", cpuFinalNorm);
        fprintf(stderr, "[vk] final_norm: %zu\n", cpuFinalNorm.size());
        if (!ld("output.weight", cpuOutput)) ld("lm_head.weight", cpuOutput);
        fprintf(stderr, "[vk] output: %zu\n", cpuOutput.size());

        // Load all weights into CPU memory first, then upload to GPU
        cpuL.resize(NC);
        char buf[128];

        for (int l = 0; l < NC; l++) {
            auto& cl = cpuL[l];
            auto ldW = [&](const char* bk, const char* lg, std::vector<float>& dst, int n) {
                snprintf(buf, sizeof(buf), "blk.%d.%s", l, bk);
                if (!mdl.get_tensor_f32(buf, dst)) {
                    snprintf(buf, sizeof(buf), "model.layers.%d.%s", l, lg);
                    mdl.get_tensor_f32(buf, dst);
                }
            };
            ldW("attn_q.weight", "self_attn.q_proj.weight", cl.wq, H*NH*HD_);
            ldW("attn_k.weight", "self_attn.k_proj.weight", cl.wk, H*NKV*HD_);
            ldW("attn_v.weight", "self_attn.v_proj.weight", cl.wv, H*NKV*HD_);
            ldW("attn_output.weight", "self_attn.o_proj.weight", cl.wo, NH*HD_*H);
            ldW("ffn_gate.weight", "mlp.gate_proj.weight", cl.w1, H*IM);
            ldW("ffn_up.weight", "mlp.up_proj.weight", cl.w2, H*IM);
            ldW("ffn_down.weight", "mlp.down_proj.weight", cl.w3, IM*H);
            ldW("attn_norm.weight", "input_layernorm.weight", cl.pn, H);
            ldW("ffn_norm.weight", "post_attention_layernorm.weight", cl.pon, H);
        }
        fprintf(stderr, "[vk] 1BP weights loaded — uploading to GPU\n");

        // Upload all weights into device-accessible VkBufs
        auto usage2 = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        auto props2 = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        auto mkW = [&](VkBuf& b, std::vector<float>& v) {
            if (v.empty()) { b = {}; return; }
            b = create_vk_buf(vk->device, vk->physDev, v.size() * sizeof(float), usage2, props2);
            if (b.mapped) memcpy(b.mapped, v.data(), v.size() * sizeof(float));
            v.clear(); v.shrink_to_fit();
        };

        // Global buffers
        mkW(bufEmbed, cpuEmbed);
        mkW(bufFinalNorm, cpuFinalNorm);
        mkW(bufOutput, cpuOutput);

        // Per-layer
        layers.resize(NC);
        for (int l = 0; l < NC; l++) {
            auto& cl = cpuL[l]; auto& gl = layers[l];
            mkW(gl.wq, cl.wq); mkW(gl.wk, cl.wk); mkW(gl.wv, cl.wv); mkW(gl.wo, cl.wo);
            mkW(gl.w1, cl.w1); mkW(gl.w2, cl.w2); mkW(gl.w3, cl.w3);
            mkW(gl.pn, cl.pn); mkW(gl.pon, cl.pon);
        }
        cpuL.clear();
        fprintf(stderr, "[vk] 1BP weights on GPU — %d layers\n", NC);
        return true;
    }

    bool reset() override {
        pos = 0;
        size_t kvSize = (size_t)max_seq * NKV * HD_;
        if (kCache.mapped) memset(kCache.mapped, 0, kvSize * sizeof(float));
        if (vCache.mapped) memset(vCache.mapped, 0, kvSize * sizeof(float));
        size_t kvcTotal = (size_t)2 * NC * max_seq * NKV * HD_;
        if (bKVCacheAll.mapped) memset(bKVCacheAll.mapped, 0, kvcTotal * sizeof(float));
        return true;
    }

    // ── Dispatch a compute shader ──
    void dispatch(int shaderIdx, const std::vector<VkBuf*>& bufs, uint32_t gx = 1, uint32_t gy = 1, uint32_t gz = 1) {
        if (shaderIdx < 0 || shaderIdx >= (int)shaders.size()) return;
        auto& s = shaders[shaderIdx];
        auto dev = vk->device;

        // Allocate descriptor set
        VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, vk->descPool, 1, &s.descLayout};
        VkDescriptorSet ds;
        if (vkAllocateDescriptorSets(dev, &dsai, &ds) != VK_SUCCESS) return;

        // Write descriptors
        std::vector<VkDescriptorBufferInfo> binfo;
        std::vector<VkWriteDescriptorSet> writes;
        for (uint32_t i = 0; i < (uint32_t)bufs.size(); i++) {
            if (!bufs[i] || !bufs[i]->buf) return;
            binfo.push_back({bufs[i]->buf, 0, bufs[i]->size});
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &binfo.back(), nullptr});
        }
        vkUpdateDescriptorSets(dev, (uint32_t)writes.size(), writes.data(), 0, nullptr);

        // Command buffer
        VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, vk->cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        VkCommandBuffer cb;
        vkAllocateCommandBuffers(dev, &cbai, &cb);

        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        vkBeginCommandBuffer(cb, &cbbi);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipeLayout, 0, 1, &ds, 0, nullptr);
        vkCmdDispatch(cb, gx, gy, gz);
        vkEndCommandBuffer(cb);

        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb, 0, nullptr};
        vkQueueSubmit(vk->queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(vk->queue);

        vkFreeCommandBuffers(dev, vk->cmdPool, 1, &cb);
        vkFreeDescriptorSets(dev, vk->descPool, 1, &ds);
    }

    // ── GEMV via Vulkan compute (gemv_f32 shader) ──
    // Groups of 64 threads per workgroup, each handling one output row
    void gemv(VkBuf& y, const VkBuf& W, const VkBuf& x, int M, int N) {
        if (!W.buf) return;
        dispatch(0, {&y, const_cast<VkBuf*>(&W), const_cast<VkBuf*>(&x)}, (M + 63) / 64, 1, 1);
    }

    // ── Push constants (matches all ZINC shaders) ──
    struct PC {
        uint32_t M = 0, N = 0, K = 0, s = 0;
        float sc = 0, eps = 0, th = 0, p = 0;
        int32_t tok = 0, l = 0, h = 0, pos = 0;
    };

    // ── Dispatch with push constants ──
    void dispatch_pc(int shaderIdx, const std::vector<VkBuf*>& bufs,
                     uint32_t gx, uint32_t gy, uint32_t gz, const PC& pc) {
        if (shaderIdx < 0 || shaderIdx >= (int)shaders.size()) return;
        auto& s = shaders[shaderIdx];
        auto dev = vk->device;

        VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                            nullptr, vk->descPool, 1, &s.descLayout};
        VkDescriptorSet ds;
        if (vkAllocateDescriptorSets(dev, &dsai, &ds) != VK_SUCCESS) return;

        std::vector<VkDescriptorBufferInfo> binfo;
        std::vector<VkWriteDescriptorSet> writes;
        for (uint32_t i = 0; i < (uint32_t)bufs.size(); i++) {
            if (!bufs[i] || !bufs[i]->buf) { vkFreeDescriptorSets(dev, vk->descPool, 1, &ds); return; }
            binfo.push_back({bufs[i]->buf, 0, bufs[i]->size});
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, i, 0, 1,
                              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &binfo.back(), nullptr});
        }
        vkUpdateDescriptorSets(dev, (uint32_t)writes.size(), writes.data(), 0, nullptr);

        VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                            nullptr, vk->cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        VkCommandBuffer cb;
        vkAllocateCommandBuffers(dev, &cbai, &cb);

        VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                         VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        vkBeginCommandBuffer(cb, &cbbi);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipeLayout, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cb, s.pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
        vkCmdDispatch(cb, gx, gy, gz);
        vkEndCommandBuffer(cb);

        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb, 0, nullptr};
        vkQueueSubmit(vk->queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(vk->queue);

        vkFreeCommandBuffers(dev, vk->cmdPool, 1, &cb);
        vkFreeDescriptorSets(dev, vk->descPool, 1, &ds);
    }

    // ── GPU GEMV: y[M] = W[M,K] @ x[K]  — bindings: {x, y, W} ──
    void gpu_gemv(VkBuf& x, VkBuf& y, VkBuf& W, int M, int K, const PC& base_pc) {
        PC pc = base_pc;
        pc.M = (uint32_t)M; pc.K = (uint32_t)K;
        uint32_t gx = (uint32_t)std::min(M, 65535);
        uint32_t gy = ((uint32_t)M + gx - 1) / gx;
        dispatch_pc(0, {&x, &y, &W}, gx, gy, 1, pc);
    }

    // ── CPU GEMV fallback (lm_head only) ──
    static void cpu_gemv(float* y, const float* W, const float* x, int M, int N) {
        for (int i = 0; i < M; i++) {
            double s = 0;
            for (int j = 0; j < N; j++) s += (double)x[j] * W[(size_t)i * N + j];
            y[i] = (float)s;
        }
    }

    bool forward(int token_id, float* hidden_out) override {
        if (!vk_ok) return false;
        PC pc{};
        pc.eps = EPS;

        // ── Embedding ──
        pc.M = (uint32_t)H; pc.tok = token_id;
        dispatch_pc(3, {&bufEmbed, &bHidden, &bDummy, &bDummy, &bDummy, &bDummy}, 1, 1, 1, pc);

        for (int l = 0; l < NC; l++) {
            auto& lw = layers[l];

            // ── Save residual for attention ──
            pc.M = (uint32_t)H;
            dispatch_pc(5, {&bHidden, &bResid, &bDummy, &bDummy, &bDummy, &bDummy}, 1, 1, 1, pc);

            // ── Pre-attention RMSNorm (in-place, shader 11 = rms_norm_mul) ──
            pc.M = (uint32_t)H; pc.eps = EPS;
            dispatch_pc(11, {&bHidden, &bDummy, &lw.pn, &bDummy, &bDummy, &bDummy},
                        ((uint32_t)H + 255) / 256, 1, 1, pc);

            // ── QKV GEMV (bindings: X=input, Y=output, W=weights) ──
            pc.K = (uint32_t)H;
            pc.M = (uint32_t)(NH * HD_);
            gpu_gemv(bHidden, bQ, lw.wq, NH * HD_, H, pc);
            pc.M = (uint32_t)(NKV * HD_);
            gpu_gemv(bHidden, bK, lw.wk, NKV * HD_, H, pc);
            gpu_gemv(bHidden, bV, lw.wv, NKV * HD_, H, pc);

            // ── Assemble QKV buffer (CPU — HOST_VISIBLE, safe after vkQueueWaitIdle) ──
            {
                float* qkv = (float*)bQKV.mapped;
                memcpy(qkv,                          bQ.mapped, (size_t)NH * HD_ * 4);
                memcpy(qkv + NH * HD_,               bK.mapped, (size_t)NKV * HD_ * 4);
                memcpy(qkv + NH * HD_ + NKV * HD_,   bV.mapped, (size_t)NKV * HD_ * 4);
            }

            // ── RoPE in-place on bQKV (M=NH, N=NKV, K=HD, th=theta, pos=pos) ──
            pc.M = (uint32_t)NH; pc.N = (uint32_t)NKV; pc.K = (uint32_t)HD_;
            pc.th = rope_theta; pc.pos = pos;
            dispatch_pc(2, {&bQKV, &bDummy, &bDummy, &bDummy, &bDummy, &bDummy},
                        (uint32_t)(NH + NKV), 1, 1, pc);

            // ── Flash attention (M=NH, N=pos+1, K=HD, s=NKV, l=layer, tok=max_seq, h=NC, pos) ──
            pc.M = (uint32_t)NH; pc.N = (uint32_t)(pos + 1); pc.K = (uint32_t)HD_;
            pc.s = (uint32_t)NKV; pc.l = l; pc.tok = max_seq; pc.h = NC; pc.pos = pos;
            dispatch_pc(9, {&bQKV, &bAttn, &bKVCacheAll, &bDummy, &bDummy, &bDummy},
                        (uint32_t)NH, 1, 1, pc);

            // ── Output projection: bAttn → bHidden (M=H, K=NH*HD) ──
            pc.M = (uint32_t)H; pc.K = (uint32_t)(NH * HD_);
            gpu_gemv(bAttn, bHidden, lw.wo, H, NH * HD_, pc);

            // ── Attention residual: bHidden += bResid ──
            pc.M = (uint32_t)H;
            dispatch_pc(4, {&bResid, &bHidden, &bDummy, &bDummy, &bDummy, &bDummy},
                        ((uint32_t)H + 255) / 256, 1, 1, pc);

            // ── Save residual for FFN ──
            dispatch_pc(5, {&bHidden, &bResid, &bDummy, &bDummy, &bDummy, &bDummy}, 1, 1, 1, pc);

            // ── Post-attention RMSNorm ──
            pc.eps = EPS;
            dispatch_pc(11, {&bHidden, &bDummy, &lw.pon, &bDummy, &bDummy, &bDummy},
                        ((uint32_t)H + 255) / 256, 1, 1, pc);

            // ── Gate + Up GEMV ──
            pc.M = (uint32_t)IM; pc.K = (uint32_t)H;
            gpu_gemv(bHidden, bGate, lw.w1, IM, H, pc);
            gpu_gemv(bHidden, bUp,   lw.w2, IM, H, pc);

            // ── Assemble bGateAll = [gate | up] (CPU, HOST_VISIBLE) ──
            {
                float* ga = (float*)bGateAll.mapped;
                memcpy(ga,        bGate.mapped, (size_t)IM * 4);
                memcpy(ga + IM,   bUp.mapped,   (size_t)IM * 4);
            }

            // ── SwiGLU: bGateAll → bSilu (M=IM) ──
            pc.M = (uint32_t)IM;
            dispatch_pc(7, {&bGateAll, &bSilu, &bDummy, &bDummy, &bDummy, &bDummy},
                        ((uint32_t)IM + 255) / 256, 1, 1, pc);

            // ── Down projection: bSilu → bHidden (M=H, K=IM) ──
            pc.M = (uint32_t)H; pc.K = (uint32_t)IM;
            gpu_gemv(bSilu, bHidden, lw.w3, H, IM, pc);

            // ── FFN residual: bHidden += bResid ──
            pc.M = (uint32_t)H;
            dispatch_pc(4, {&bResid, &bHidden, &bDummy, &bDummy, &bDummy, &bDummy},
                        ((uint32_t)H + 255) / 256, 1, 1, pc);
        }

        // ── Final RMSNorm ──
        pc.M = (uint32_t)H; pc.eps = EPS;
        dispatch_pc(11, {&bHidden, &bDummy, &bufFinalNorm, &bDummy, &bDummy, &bDummy},
                    ((uint32_t)H + 255) / 256, 1, 1, pc);

        memcpy(hidden_out, bHidden.mapped, (size_t)H * sizeof(float));
        pos++;
        return true;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        // Upload hidden to bHidden, run GEMV on GPU
        if (bHidden.mapped) memcpy(bHidden.mapped, hidden, (size_t)H * sizeof(float));
        VkBuf* wBuf = bufOutput.buf ? &bufOutput : (bufEmbed.buf ? &bufEmbed : nullptr);
        if (!wBuf) {
            memset(logits, 0, VOCAB * 4); logits[0] = 1; if (argmax) *argmax = 0; return true;
        }
        PC pc{};
        pc.M = (uint32_t)VOCAB; pc.K = (uint32_t)H;
        uint32_t gx = (uint32_t)std::min(VOCAB, 65535);
        uint32_t gy = ((uint32_t)VOCAB + gx - 1) / gx;
        dispatch_pc(0, {&bHidden, &bLogits, wBuf, &bDummy, &bDummy, &bDummy}, gx, gy, 1, pc);
        memcpy(logits, bLogits.mapped, (size_t)VOCAB * sizeof(float));
        if (argmax) { *argmax = 0; for (int v = 1; v < VOCAB; v++) if (logits[v] > logits[*argmax]) *argmax = v; }
        return true;
    }

    int generate(int token_id) override {
        std::vector<float> h(H);
        if (!forward(token_id, h.data())) return -1;
        std::vector<float> l(VOCAB); int n = -1;
        if (!lm_head(h.data(), l.data(), &n)) return -1;
        return n;
    }

    float benchmark(int tokens) override {
        if (!initialized) return -1;
        reset();
        auto t0 = std::chrono::steady_clock::now();
        int tok = 1;
        for (int i = 0; i < tokens; i++) { tok = generate(tok); if (tok < 0) break; }
        auto t1 = std::chrono::steady_clock::now();
        return (float)(std::chrono::duration<double, std::milli>(t1 - t0).count() / tokens);
    }

    void destroy() override {
        auto dev = vk ? vk->device : VK_NULL_HANDLE;
        auto db = [&](VkBuf& b) { if (dev) destroy_vk_buf(dev, b); };
        db(bHidden); db(bResid); db(bQ); db(bK); db(bV); db(bQKV); db(bAttn);
        db(bGate); db(bUp); db(bGateAll); db(bSilu); db(bFFNOut); db(bLogits);
        db(kCache); db(vCache); db(bKVCacheAll); db(bDummy);
        db(bufEmbed); db(bufFinalNorm); db(bufOutput);
        for (auto& l : layers) {
            db(l.wq); db(l.wk); db(l.wv); db(l.wo);
            db(l.w1); db(l.w2); db(l.w3); db(l.pn); db(l.pon);
        }
        layers.clear();
        for (auto& s : shaders) destroy_vk_shader(dev, s);
        shaders.clear();
        delete vk; vk = nullptr;
        vk_ok = false; initialized = false;
    }
};

Backend* create_vulkan_hpp_backend() { return static_cast<Backend*>(new VulkanBackend()); }
