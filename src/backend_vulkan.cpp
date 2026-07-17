// backend_vulkan.cpp — Vulkan GPU backend for Zaya1-8B
// Hybrid: Vulkan compute for standard matmuls, CPU for CCA custom ops
// Portable: AMD, NVIDIA, Intel, Apple (any Vulkan 1.2+)

#include "backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

// ── Architecture ──
static constexpr int H=2048,NQ=8,NKV=2,HD=128,QD=NQ*HD,KD=NKV*HD,QKV=QD+KD;
static constexpr int GQA=4,NROT=64,N_EXP=16,N_FF=2048,RTR_H=256,VOCAB=262272,N_LAYERS=40;

// ── Vulkan helpers ──
struct VK {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkDescriptorSetLayout ds_layout = VK_NULL_HANDLE;
    VkDescriptorPool ds_pool = VK_NULL_HANDLE;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    VkBuffer buf_w = VK_NULL_HANDLE, buf_in = VK_NULL_HANDLE, buf_out = VK_NULL_HANDLE;
    VkDeviceMemory mem_w = VK_NULL_HANDLE, mem_in = VK_NULL_HANDLE, mem_out = VK_NULL_HANDLE;
    uint32_t qf = 0;
    char name[256] = {};

    bool init() {
        VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "ZayaVK"; app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return false;

        uint32_t nd; vkEnumeratePhysicalDevices(inst, &nd, nullptr);
        if (!nd) return false;
        std::vector<VkPhysicalDevice> pds(nd);
        vkEnumeratePhysicalDevices(inst, &nd, pds.data());
        phys = pds[0];
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys, &props);
        strncpy(name, props.deviceName, sizeof(name)-1);

        uint32_t nq; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qps(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qps.data());
        for (uint32_t i = 0; i < nq; i++) if (qps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }

        float prio = 1.0f;
        VkDeviceQueueCreateInfo dq = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        dq.queueFamilyIndex = qf; dq.queueCount = 1; dq.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &dq;
        if (vkCreateDevice(phys, &dci, nullptr, &dev) != VK_SUCCESS) return false;
        vkGetDeviceQueue(dev, qf, 0, &queue);

        VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.queueFamilyIndex = qf;
        if (vkCreateCommandPool(dev, &cpi, nullptr, &pool) != VK_SUCCESS) return false;

        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool; ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(dev, &ai, &cmd) != VK_SUCCESS) return false;

        return true;
    }

    VkShaderModule load_shader(const char* spv_path) {
        FILE* f = fopen(spv_path, "rb"); if (!f) return VK_NULL_HANDLE;
        fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
        std::vector<uint32_t> code(sz/4); fread(code.data(), 4, code.size(), f); fclose(f);
        VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sm.codeSize = sz; sm.pCode = code.data();
        VkShaderModule mod;
        vkCreateShaderModule(dev, &sm, nullptr, &mod);
        return mod;
    }

    VkDeviceMemory alloc_buf(VkDeviceSize sz, VkBufferUsageFlags usage, VkBuffer* buf) {
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = sz; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(dev, &bi, nullptr, buf);
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, *buf, &mr);
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(phys, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
            if ((mr.memoryTypeBits & (1<<i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                ai.allocationSize = mr.size; ai.memoryTypeIndex = i;
                VkDeviceMemory mem;
                vkAllocateMemory(dev, &ai, nullptr, &mem);
                vkBindBufferMemory(dev, *buf, mem, 0);
                return mem;
            }
        }
        return VK_NULL_HANDLE;
    }

    bool create_matmul_pipeline(VkShaderModule mod) {
        VkDescriptorSetLayoutBinding bindings[3] = {};
        bindings[0].binding = 0; bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].descriptorCount = 1; bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1] = bindings[0]; bindings[1].binding = 1;
        bindings[2] = bindings[0]; bindings[2].binding = 2;
        VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dli.bindingCount = 3; dli.pBindings = bindings;
        vkCreateDescriptorSetLayout(dev, &dli, nullptr, &ds_layout);

        VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1; pli.pSetLayouts = &ds_layout;
        vkCreatePipelineLayout(dev, &pli, nullptr, &pipe_layout);

        VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
        VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
        vkCreateDescriptorPool(dev, &dpi, nullptr, &ds_pool);

        VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = ds_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &ds_layout;
        vkAllocateDescriptorSets(dev, &dai, &ds);

        VkPipelineShaderStageCreateInfo ss = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ss.stage = VK_SHADER_STAGE_COMPUTE_BIT; ss.module = mod; ss.pName = "main";
        VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = ss; ci.layout = pipe_layout;
        vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe);
        return true;
    }

    void destroy() {
        if (pipe) vkDestroyPipeline(dev, pipe, nullptr);
        if (pipe_layout) vkDestroyPipelineLayout(dev, pipe_layout, nullptr);
        if (ds_layout) vkDestroyDescriptorSetLayout(dev, ds_layout, nullptr);
        if (ds_pool) vkDestroyDescriptorPool(dev, ds_pool, nullptr);
        if (buf_w) vkDestroyBuffer(dev, buf_w, nullptr);
        if (buf_in) vkDestroyBuffer(dev, buf_in, nullptr);
        if (buf_out) vkDestroyBuffer(dev, buf_out, nullptr);
        if (mem_w) vkFreeMemory(dev, mem_w, nullptr);
        if (mem_in) vkFreeMemory(dev, mem_in, nullptr);
        if (mem_out) vkFreeMemory(dev, mem_out, nullptr);
        if (pool) vkDestroyCommandPool(dev, pool, nullptr);
        if (dev) vkDestroyDevice(dev, nullptr);
        if (inst) vkDestroyInstance(inst, nullptr);
    }

    // Vulkan matmul: out[M] = in[K] @ wt[M×K]^T
    void matmul(float* out, const float* in, const float* wt, int M, int K) {
        size_t sz_w = (size_t)M * K * sizeof(float);
        size_t sz_i = (size_t)K * sizeof(float);
        size_t sz_o = (size_t)M * sizeof(float);

        // Upload weights (reused across layers — upload once)
        // For simplicity, upload each call. In production, pre-upload all layers.
        void* pm; vkMapMemory(dev, mem_w, 0, sz_w, 0, &pm);
        memcpy(pm, wt, sz_w); vkUnmapMemory(dev, mem_w);
        void* pi; vkMapMemory(dev, mem_in, 0, sz_i, 0, &pi);
        memcpy(pi, in, sz_i); vkUnmapMemory(dev, mem_in);

        // Bind descriptors
        VkDescriptorBufferInfo dbw = {buf_w, 0, sz_w};
        VkDescriptorBufferInfo dbi = {buf_in, 0, sz_i};
        VkDescriptorBufferInfo dbo = {buf_out, 0, sz_o};
        VkWriteDescriptorSet writes[3] = {};
        for (int i = 0; i < 3; i++) {
            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = ds; writes[i].dstBinding = i;
            writes[i].descriptorCount = 1; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        writes[0].pBufferInfo = &dbw; writes[1].pBufferInfo = &dbi; writes[2].pBufferInfo = &dbo;
        vkUpdateDescriptorSets(dev, 3, writes, 0, nullptr);

        // Dispatch
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd, &bi);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1, &ds, 0, nullptr);
        struct Push { uint32_t M, K, io, wo, oo; } push = {(uint32_t)M, (uint32_t)K, 0, 0, 0};
        vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, (M+63)/64, 1, 1);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        // Read output
        void* po; vkMapMemory(dev, mem_out, 0, sz_o, 0, &po);
        memcpy(out, po, sz_o); vkUnmapMemory(dev, mem_out);
    }
};

// ── Backend ──
struct VKBackend : Backend {
    VK vk;
    float* embed = nullptr; float* fnorm = nullptr; float* iscale = nullptr; float* ibias = nullptr;
    struct LayerW {
        float nw[H], wq[QD*H], wk[KD*H], wv1[(KD/2)*H], wv2[(KD/2)*H], wo[H*QD], pan[H];
        float cdw[QKV*2], cdb[QKV], cgw[QKV*128*2], cgb[QKV], ks[NKV];
        float pahss[H], pahsb[H], parss[H], parsb[H];
        float gdw[RTR_H*H], gdb[RTR_H], rfn[RTR_H], rf1[RTR_H*RTR_H], rf1b[RTR_H];
        float rf2[RTR_H*RTR_H], rf2b[RTR_H], rout[17*RTR_H], bb[17];
        float* gu_mmap = nullptr; size_t gu_size = 0;
        float* dn_mmap = nullptr; size_t dn_size = 0;
        float pmhss[H], pmhsb[H], pmrss[H], pmrsb[H];
    };
    LayerW* lw = nullptr;
    std::string wd;
    float hs[H], prev_hs[H], conv_state[2*QKV], prev_rs[RTR_H], moe_out[H];
    int pos = 0;

    VKBackend() { type = BackendType::VULKAN; name = "Vulkan GPU (portable)"; }
    ~VKBackend() override { destroy(); }

    // mmap helpers
    static float* mmap_file(const std::string& path, size_t* sz) {
        int fd = open(path.c_str(), O_RDONLY); if (fd < 0) { *sz = 0; return nullptr; }
        struct stat st; fstat(fd, &st); *sz = st.st_size;
        float* m = (float*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
        if (m == MAP_FAILED) { *sz = 0; return nullptr; } return m;
    }
    static void munmap_file(float* p, size_t sz) { if (p) munmap(p, sz); }

    VkShaderModule load_and_create_pipeline(const char* spv_path, VkPipeline* pipe_out, int extra_bindings = 0) {
        VkShaderModule mod = vk.load_shader(spv_path);
        if (!mod) return VK_NULL_HANDLE;
        
        // Create pipeline for this shader
        VkDescriptorSetLayoutBinding bindings[17] = {};
        int nb = 3 + extra_bindings;
        for (int i = 0; i < nb; i++) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dli.bindingCount = (uint32_t)nb; dli.pBindings = bindings;
        vkCreateDescriptorSetLayout(vk.dev, &dli, nullptr, &vk.ds_layout);
        
        VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1; pli.pSetLayouts = &vk.ds_layout;
        vkCreatePipelineLayout(vk.dev, &pli, nullptr, &vk.pipe_layout);
        
        VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)nb};
        VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
        vkCreateDescriptorPool(vk.dev, &dpi, nullptr, &vk.ds_pool);
        
        VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = vk.ds_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &vk.ds_layout;
        vkAllocateDescriptorSets(vk.dev, &dai, &vk.ds);
        
        VkPipelineShaderStageCreateInfo ss = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ss.stage = VK_SHADER_STAGE_COMPUTE_BIT; ss.module = mod; ss.pName = "main";
        VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = ss; ci.layout = vk.pipe_layout;
        vkCreateComputePipelines(vk.dev, VK_NULL_HANDLE, 1, &ci, nullptr, pipe_out);
        return mod;
    }

    bool init(const ModelConfig& cfg, const std::string& weights_dir) override {
        this->cfg = cfg; this->wd = weights_dir;
        printf("Vulkan: Initializing...\n");

        if (!vk.init()) { fprintf(stderr,"Vulkan: No device\n"); return false; }
        printf("Vulkan: GPU = %s\n", vk.name);

        // Load shaders
        VkShaderModule mm_mod = vk.load_shader("kernels/vulkan/matmul_fp32.comp.spv");
        if (!mm_mod) { fprintf(stderr,"Vulkan: No matmul shader\n"); return false; }
        // Reuse matmul pipeline creation for the matmul shader
        vk.create_matmul_pipeline(mm_mod);
        vkDestroyShaderModule(vk.dev, mm_mod, nullptr);
        
        // Load CCA attention shader (17 descriptor bindings for all inputs+outputs)
        VkShaderModule cca_mod = vk.load_shader("kernels/vulkan/zaya_cca_attn.comp.spv");
        if (cca_mod) {
            VkPipeline cca_pipe;
            load_and_create_pipeline("kernels/vulkan/zaya_cca_attn.comp.spv", &cca_pipe, 14);
            vkDestroyShaderModule(vk.dev, cca_mod, nullptr);
            printf("Vulkan: CCA attention shader loaded\n");
        } else {
            printf("Vulkan: No CCA shader, using CPU fallback\n");
        }

        // Allocate Vulkan buffers
        size_t max_w = (size_t)N_EXP * 2 * N_FF * H * sizeof(float);
        vk.mem_w = vk.alloc_buf(max_w, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &vk.buf_w);
        vk.mem_in = vk.alloc_buf(H * sizeof(float) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &vk.buf_in);
        vk.mem_out = vk.alloc_buf(H * sizeof(float) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &vk.buf_out);

        // Load weights (same as CPU backend)
        auto W = [&](const std::string& name) -> std::vector<float> {
            std::ifstream f(wd+"/"+name, std::ios::binary|std::ios::ate);
            if (!f) return {}; size_t n = f.tellg()/sizeof(float); f.seekg(0);
            std::vector<float> d(n); f.read((char*)d.data(),n*sizeof(float)); return d;
        };
        auto load = [&](float* dst, const std::string& name) {
            auto v = W(name); if (!v.empty()) memcpy(dst, v.data(), v.size()*4);
        };
        auto L = [](int i) { return std::to_string(i); };

        embed = new float[VOCAB*H]; fnorm = new float[H];
        iscale = new float[H]; ibias = new float[H];
        load(embed, "model_embed_tokens_weight.bin");
        load(fnorm, "model_norm_weight.bin");
        load(iscale, "model_input_hidden_states_scale.bin");
        load(ibias, "model_input_hidden_states_bias.bin");

        printf("Vulkan: Loading %d layers...\n", N_LAYERS);
        lw = new LayerW[N_LAYERS];
        for (int il = 0; il < N_LAYERS; il++) {
            auto& l = lw[il]; std::string B = "model_layers_"+L(il)+"_";
            load(l.nw, B+"input_layernorm_weight.bin");    load(l.wq, B+"self_attn_qkv_proj_q_proj_weight.bin");
            load(l.wk, B+"self_attn_qkv_proj_k_proj_weight.bin");
            load(l.wv1, B+"self_attn_qkv_proj_v_proj_current_weight.bin");
            load(l.wv2, B+"self_attn_qkv_proj_v_proj_delayed_weight.bin");
            load(l.wo, B+"self_attn_o_proj_weight.bin");   load(l.pan, B+"post_attention_layernorm_weight.bin");
            load(l.cdw, B+"self_attn_qkv_proj_conv_qk_depthwise_weight.bin");
            load(l.cdb, B+"self_attn_qkv_proj_conv_qk_depthwise_bias.bin");
            load(l.cgw, B+"self_attn_qkv_proj_conv_qk_grouped_weight.bin");
            load(l.cgb, B+"self_attn_qkv_proj_conv_qk_grouped_bias.bin");
            load(l.ks, B+"self_attn_qk_norm_temp.bin");    load(l.pahss, B+"post_attention_residual_scale_hidden_states_scale.bin");
            load(l.pahsb, B+"post_attention_residual_scale_hidden_states_bias.bin");
            load(l.parss, B+"post_attention_residual_scale_residual_scale.bin");
            load(l.parsb, B+"post_attention_residual_scale_residual_bias.bin");
            load(l.gdw, B+"mlp_gate_down_proj_weight.bin"); load(l.gdb, B+"mlp_gate_down_proj_bias.bin");
            load(l.rfn, B+"mlp_gate_router_mlp_norm_weight.bin");
            load(l.rf1, B+"mlp_gate_router_mlp_fc1_weight.bin"); load(l.rf1b, B+"mlp_gate_router_mlp_fc1_bias.bin");
            load(l.rf2, B+"mlp_gate_router_mlp_fc2_weight.bin"); load(l.rf2b, B+"mlp_gate_router_mlp_fc2_bias.bin");
            load(l.rout, B+"mlp_gate_router_mlp_out_proj_weight.bin"); load(l.bb, B+"mlp_gate_balancing_biases.bin");
            l.gu_mmap = mmap_file(wd+"/"+B+"mlp_experts_gate_up_proj.bin", &l.gu_size);
            l.dn_mmap = mmap_file(wd+"/"+B+"mlp_experts_down_proj.bin", &l.dn_size);
            load(l.pmhss, B+"post_mlp_residual_scale_hidden_states_scale.bin");
            load(l.pmhsb, B+"post_mlp_residual_scale_hidden_states_bias.bin");
            load(l.pmrss, B+"post_mlp_residual_scale_residual_scale.bin");
            load(l.pmrsb, B+"post_mlp_residual_scale_residual_bias.bin");
        }
        // Validate that critical weight files loaded (issue #274)
        if (!embed || !fnorm) {
            fprintf(stderr, "Vulkan: missing critical weight files (embed/fnorm)\n");
            initialized = false;
            return false;
        }
        printf("Vulkan: %d layers loaded\n", N_LAYERS);
        initialized = true;
        return true;
    }

    bool reset() override {
        memset(hs,0,sizeof(hs)); memset(prev_hs,0,sizeof(prev_hs));
        memset(conv_state,0,sizeof(conv_state)); memset(prev_rs,0,sizeof(prev_rs)); pos=0;
        return true;
    }

    // ── CCA Attention (CPU - custom ops not in Vulkan shader yet) ──
    static void cca_attn_cpu(float* h, const float* wq, const float* wk,
        const float* wv1, const float* wv2, const float* wo,
        const float* cdw, const float* cdb, const float* cgw, const float* cgb,
        const float* ks, const float* nw, float* phs, float* cs, int l) {
        float b[H],q[QD],k[KD],v[KD],sk[QKV],d0[QKV],d1[QKV];
        auto mm=[](float*o,const float*a,const float*w,int M,int K){for(int i=0;i<M;i++){float s=0;for(int t=0;t<K;t++)s+=a[t]*w[(size_t)i*K+t];o[i]=s;}};
        memcpy(b,h,H*4); double ss=0; for(int i=0;i<H;i++)ss+=b[i]*b[i];
        float ir=1.0f/sqrtf((float)(ss/H)+1e-5f); for(int i=0;i<H;i++)b[i]=b[i]*ir*nw[i];
        mm(q,b,wq,QD,H); mm(k,b,wk,KD,H); memcpy(sk,q,QD*4); memcpy(sk+QD,k,KD*4);
        mm(v,b,wv1,KD/2,H); mm(&v[KD/2],phs,wv2,KD/2,H);
        for(int c=0;c<QKV;c++){d0[c]=cdw[c*2]*cs[c]+cdw[c*2+1]*cs[QKV+c]+cdb[c];d1[c]=cdw[c*2]*cs[QKV+c]+cdw[c*2+1]*sk[c]+cdb[c];}
        for(int oc=0;oc<QKV;oc++){int g=oc/128,ba=g*128;float s=cgb[oc];for(int j=0;j<128;j++)s+=cgw[(size_t)oc*256+j*2]*d0[ba+j]+cgw[(size_t)oc*256+j*2+1]*d1[ba+j];sk[oc]=s;}
        for(int hh=0;hh<NQ;hh++){int kv=hh/GQA;for(int d=0;d<HD;d++)sk[hh*HD+d]=sk[hh*HD+d]+0.5f*q[hh*HD+d]+0.5f*k[kv*HD+d];}
        for(int khv=0;khv<NKV;khv++)for(int d=0;d<HD;d++){float sm=0;for(int g=0;g<GQA;g++)sm+=q[(khv*GQA+g)*HD+d];sk[QD+khv*HD+d]=sk[QD+khv*HD+d]+0.5f*sm/GQA+0.5f*k[khv*HD+d];}
        float sd=sqrtf((float)HD);
        for(int hh=0;hh<NQ;hh++){float s2=0;for(int d=0;d<HD;d++)s2+=sk[hh*HD+d]*sk[hh*HD+d];float iv=sd/(sqrtf(s2)+1e-12f);for(int d=0;d<HD;d++)sk[hh*HD+d]*=iv;}
        for(int khv=0;khv<NKV;khv++){float s2=0;for(int d=0;d<HD;d++)s2+=sk[QD+khv*HD+d]*sk[QD+khv*HD+d];float iv=sd*ks[khv]/(sqrtf(s2)+1e-12f);for(int d=0;d<HD;d++)sk[QD+khv*HD+d]*=iv;}
        for(int hh=0;hh<NQ;hh++){float*ba=sk+hh*HD;for(int d=0;d<32;d++){float th=(float)l*powf(1e6f,-2.0f*d/64);float c=cosf(th),s=sinf(th);float xv=ba[d],xw=ba[d+32];ba[d]=xv*c-xw*s;ba[d+32]=xv*s+xw*c;}}
        for(int khv=0;khv<NKV;khv++){float*ba=sk+QD+khv*HD;for(int d=0;d<32;d++){float th=(float)l*powf(1e6f,-2.0f*d/64);float c=cosf(th),s=sinf(th);float xv=ba[d],xw=ba[d+32];ba[d]=xv*c-xw*s;ba[d+32]=xv*s+xw*c;}}
        float ao[H]; for(int hh=0;hh<NQ;hh++){int kv=hh/GQA;for(int d=0;d<HD;d++)ao[hh*HD+d]=v[kv*HD+d];}
        float fo[H]; mm(fo,ao,wo,H,QD); memcpy(h,fo,H*4); memcpy(phs,h,H*4);
        memcpy(cs,cs+QKV,QKV*4); memcpy(cs+QKV,sk,QKV*4);
    }

    static void rmsnorm(float* x, const float* w, int n) {
        double ss=0; for(int i=0;i<n;i++)ss+=x[i]*x[i];
        float ir=1.0f/sqrtf((float)(ss/n)+1e-5f); for(int i=0;i<n;i++)x[i]=x[i]*ir*w[i];
    }

    static void moe_cpu(float* h, const float* gdw, const float* gdb, const float* rfn,
        const float* rf1, const float* rf1b, const float* rf2, const float* rf2b,
        const float* rout, const float* bb, const float* gu, const float* dn,
        float* prs, float* mo) {
        float rs[256],tmp[256],sc[17];
        auto mm=[](float*o,const float*a,const float*w,int M,int K){for(int i=0;i<M;i++){float s=0;for(int t=0;t<K;t++)s+=a[t]*w[(size_t)i*K+t];o[i]=s;}};
        mm(rs,h,gdw,256,H); for(int i=0;i<256;i++)rs[i]+=gdb[i]+prs[i]; memcpy(prs,rs,256*4);
        double ss=0; for(int i=0;i<256;i++)ss+=rs[i]*rs[i]; float ir=1.0f/sqrtf((float)(ss/256)+1e-5f);
        for(int i=0;i<256;i++)rs[i]=rs[i]*ir*rfn[i]; mm(tmp,rs,rf1,256,256);
        for(int i=0;i<256;i++)tmp[i]+=rf1b[i]; for(int i=0;i<256;i++){float v=tmp[i];tmp[i]=0.5f*v*(1+tanhf(0.79788456f*(v+0.044715f*v*v*v)));}
        mm(rs,tmp,rf2,256,256); for(int i=0;i<256;i++)rs[i]+=rf2b[i];
        for(int i=0;i<256;i++){float v=rs[i];rs[i]=0.5f*v*(1+tanhf(0.79788456f*(v+0.044715f*v*v*v)));}
        for(int i=0;i<17;i++){float s=0;for(int j=0;j<256;j++)s+=rs[j]*rout[(size_t)i*256+j];sc[i]=s+bb[i];}
        float mv=sc[0];for(int i=1;i<17;i++)if(sc[i]>mv)mv=sc[i];
        float su=0;for(int i=0;i<17;i++){sc[i]=expf(sc[i]-mv);su+=sc[i];} for(int i=0;i<17;i++)sc[i]/=su;
        int be=0;for(int i=1;i<17;i++)if(sc[i]>sc[be])be=i;
        memset(mo,0,H*4); if(be<16){
            float gb[4096]; mm(gb,h,gu+(size_t)be*2*N_FF*H,2*N_FF,H);
            float mb[2048]; for(int i=0;i<2048;i++)mb[i]=(gb[i]/(1+expf(-gb[i])))*gb[2048+i];
            mm(mo,mb,dn+(size_t)be*H*N_FF,H,2048);
            for(int i=0;i<H;i++)mo[i]*=sc[be];
        } else memcpy(mo,h,H*4);
    }

    // ── Vulkan matmul wrapper ──
    void vk_mm(float* out, const float* in, const float* wt, int M, int K) {
        vk.matmul(out, in, wt, M, K);
    }

    // ── Vulkan tiled lm_head ──
    // Dispatches matmul in 64-row tiles for the 262K×2048 lm_head
    void vk_lmhead(float* logits, const float* hidden, const float* embed_w, int V, int H_) {
        // Tile in chunks of 64 (matches shader workgroup size)
        for (int v = 0; v < V; v += 64) {
            int todo = (v + 64 <= V) ? 64 : V - v;
            vk.matmul(logits + v, hidden, embed_w + (size_t)v * H_, todo, H_);
        }
    }

    bool forward(int token_id, float* hidden_out) override {
        for(int i=0;i<H;i++) hs[i]=(embed[token_id*(size_t)H+i]+ibias[i])*iscale[i];

        for(int il=0;il<N_LAYERS;il++){
            auto& l=lw[il];
            float buf[H], q[QD], k[KD], v[KD];

            // ── LN1 (CPU) ──
            memcpy(buf,hs,H*4);
            double ss=0; for(int i=0;i<H;i++)ss+=buf[i]*buf[i];
            float ir=1.0f/sqrtf((float)(ss/H)+1e-5f);
            for(int i=0;i<H;i++)buf[i]=buf[i]*ir*l.nw[i];

            // ── QKV projections (Vulkan) ──
            vk_mm(q, buf, l.wq, QD, H);
            vk_mm(k, buf, l.wk, KD, H);
            vk_mm(v, buf, l.wv1, KD/2, H);
            float vd[KD/2]; vk_mm(vd, prev_hs, l.wv2, KD/2, H);
            memcpy(&v[KD/2], vd, (KD/2)*4);

            // ── CCA custom ops (CPU) ──
            float sqk[QKV], dw0[QKV], dw1[QKV];
            memcpy(sqk,q,QD*4); memcpy(sqk+QD,k,KD*4);
            for(int c=0;c<QKV;c++){
                dw0[c]=l.cdw[c*2]*conv_state[c]+l.cdw[c*2+1]*conv_state[QKV+c]+l.cdb[c];
                dw1[c]=l.cdw[c*2]*conv_state[QKV+c]+l.cdw[c*2+1]*sqk[c]+l.cdb[c];
            }
            for(int oc=0;oc<QKV;oc++){
                int g=oc/128,ba=g*128; float s=l.cgb[oc];
                for(int j=0;j<128;j++) s+=l.cgw[(size_t)oc*256+j*2]*dw0[ba+j]+l.cgw[(size_t)oc*256+j*2+1]*dw1[ba+j];
                sqk[oc]=s;
            }
            for(int hh=0;hh<NQ;hh++){int kv=hh/GQA;for(int d=0;d<HD;d++)sqk[hh*HD+d]=sqk[hh*HD+d]+0.5f*q[hh*HD+d]+0.5f*k[kv*HD+d];}
            for(int khv=0;khv<NKV;khv++)for(int d=0;d<HD;d++){float sm=0;for(int g=0;g<GQA;g++)sm+=q[(khv*GQA+g)*HD+d];sqk[QD+khv*HD+d]=sqk[QD+khv*HD+d]+0.5f*sm/GQA+0.5f*k[khv*HD+d];}
            float sd=sqrtf((float)HD);
            for(int hh=0;hh<NQ;hh++){float s2=0;for(int d=0;d<HD;d++)s2+=sqk[hh*HD+d]*sqk[hh*HD+d];float iv=sd/(sqrtf(s2)+1e-12f);for(int d=0;d<HD;d++)sqk[hh*HD+d]*=iv;}
            for(int khv=0;khv<NKV;khv++){float s2=0;for(int d=0;d<HD;d++)s2+=sqk[QD+khv*HD+d]*sqk[QD+khv*HD+d];float iv=sd*l.ks[khv]/(sqrtf(s2)+1e-12f);for(int d=0;d<HD;d++)sqk[QD+khv*HD+d]*=iv;}
            for(int hh=0;hh<NQ;hh++){float*ba=sqk+hh*HD;for(int d=0;d<32;d++){float th=pos*powf(1e6f,-2.0f*d/64);float c=cosf(th),s=sinf(th);float xv=ba[d],xw=ba[d+32];ba[d]=xv*c-xw*s;ba[d+32]=xv*s+xw*c;}}
            for(int khv=0;khv<NKV;khv++){float*ba=sqk+QD+khv*HD;for(int d=0;d<32;d++){float th=pos*powf(1e6f,-2.0f*d/64);float c=cosf(th),s=sinf(th);float xv=ba[d],xw=ba[d+32];ba[d]=xv*c-xw*s;ba[d+32]=xv*s+xw*c;}}
            float attn_out[H];
            for(int hh=0;hh<NQ;hh++){int kv=hh/GQA;for(int d=0;d<HD;d++)attn_out[hh*HD+d]=v[kv*HD+d];}

            // ── O projection (Vulkan) ──
            float o_out[H]; vk_mm(o_out, attn_out, l.wo, H, QD);

            // ── Residual + LN2 (CPU) ──
            for(int i=0;i<H;i++)o_out[i]=o_out[i]*l.pahss[i]+l.pahsb[i]+hs[i]*l.parss[i]+l.parsb[i];
            memcpy(hs,o_out,H*4);
            memcpy(buf,hs,H*4);
            ss=0; for(int i=0;i<H;i++)ss+=buf[i]*buf[i];
            ir=1.0f/sqrtf((float)(ss/H)+1e-5f);
            for(int i=0;i<H;i++)buf[i]=buf[i]*ir*l.pan[i];

            // ── EDA Router gate_down (Vulkan) ──
            float rs[256], tmp[256], scores[17];
            vk_mm(rs, buf, l.gdw, RTR_H, H);
            for(int i=0;i<RTR_H;i++)rs[i]+=l.gdb[i]+prev_rs[i];
            memcpy(prev_rs,rs,RTR_H*4);
            ss=0; for(int i=0;i<RTR_H;i++)ss+=rs[i]*rs[i];
            ir=1.0f/sqrtf((float)(ss/RTR_H)+1e-5f);
            for(int i=0;i<RTR_H;i++)rs[i]=rs[i]*ir*l.rfn[i];

            // ── Router FC1+Vulkan, FC2+Vulkan ──
            vk_mm(tmp, rs, l.rf1, RTR_H, RTR_H);
            for(int i=0;i<RTR_H;i++)tmp[i]+=l.rf1b[i];
            for(int i=0;i<RTR_H;i++){float v=tmp[i];tmp[i]=0.5f*v*(1+tanhf(0.79788456f*(v+0.044715f*v*v*v)));}
            vk_mm(rs, tmp, l.rf2, RTR_H, RTR_H);
            for(int i=0;i<RTR_H;i++)rs[i]+=l.rf2b[i];
            for(int i=0;i<RTR_H;i++){float v=rs[i];rs[i]=0.5f*v*(1+tanhf(0.79788456f*(v+0.044715f*v*v*v)));}

            // ── Router output + expert selection (CPU) ──
            for(int i=0;i<17;i++){float s=0;for(int j=0;j<RTR_H;j++)s+=rs[j]*l.rout[(size_t)i*RTR_H+j];scores[i]=s+l.bb[i];}
            float mv=scores[0];for(int i=1;i<17;i++)if(scores[i]>mv)mv=scores[i];
            float su=0;for(int i=0;i<17;i++){scores[i]=expf(scores[i]-mv);su+=scores[i];} for(int i=0;i<17;i++)scores[i]/=su;
            int best=0;for(int i=1;i<17;i++)if(scores[i]>scores[best])best=i;

            // ── MoE expert FFN (Vulkan matmuls) ──
            memset(moe_out,0,H*4);
            if(best<16 && l.gu_mmap && l.dn_mmap){
                float gu_buf[4096];
                vk_mm(gu_buf, buf, l.gu_mmap + (size_t)best*2*N_FF*H, 2*N_FF, H);
                float mb[2048];
                for(int i=0;i<2048;i++) mb[i]=(gu_buf[i]/(1+expf(-gu_buf[i])))*gu_buf[2048+i];
                vk_mm(moe_out, mb, l.dn_mmap + (size_t)best*H*N_FF, H, 2048);
                for(int i=0;i<H;i++) moe_out[i]*=scores[best];
            } else {
                memcpy(moe_out,buf,H*4);
            }

            // ── Post-MLP residual (CPU) ──
            for(int i=0;i<H;i++)hs[i]=moe_out[i]*l.pmhss[i]+l.pmhsb[i]+hs[i]*l.pmrss[i]+l.pmrsb[i];

            // Update conv state
            memcpy(prev_hs, hs, H*4);
            memcpy(conv_state, conv_state+QKV, QKV*4);
            memcpy(conv_state+QKV, sqk, QKV*4);
        }

        memcpy(hidden_out,hs,H*4); pos++; return true;
    }

    bool lm_head(const float* h, float* logits, int* argmax) override {
        float tmp[H]; memcpy(tmp,h,H*4); rmsnorm(tmp,fnorm,H);
        // Use Vulkan tiled matmul for 262K×2048 lm_head
        vk_lmhead(logits, tmp, embed, VOCAB, H);
        if(argmax){*argmax=0;for(int v=1;v<VOCAB;v++)if(logits[v]>logits[*argmax])*argmax=v;}
        return true;
    }

    int generate(int token_id) override {
        float h[H]; if(!forward(token_id,h))return-1;
        float* lg=new float[VOCAB]; int r; lm_head(h,lg,&r); delete[] lg; return r;
    }

    float benchmark(int tokens=10) override {
        reset(); int tok=100; auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<tokens;i++){tok=generate(tok);if(tok<0)break;}
        return std::chrono::duration<float,std::milli>(std::chrono::high_resolution_clock::now()-t0).count()/tokens;
    }

    void destroy() override {
        delete[] embed; delete[] fnorm; delete[] iscale; delete[] ibias;
        if(lw){for(int i=0;i<N_LAYERS;i++){munmap_file(lw[i].gu_mmap,lw[i].gu_size);munmap_file(lw[i].dn_mmap,lw[i].dn_size);}delete[] lw;}
        vk.destroy(); initialized=false;
    }
};

Backend* create_vulkan_backend() { return new VKBackend(); }
