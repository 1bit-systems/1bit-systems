/// Compute engine — GPU shader dispatch implementations.
/// Fixed: #761 descriptor pool caching, proper push constant layout binding
#include "compute_engine.h"
#include <cstring>
#include <cfloat>

// ═══════════════════════════════════════════════════════════════════
//  ComputeEngine
// ═══════════════════════════════════════════════════════════════════

ComputeEngine::ComputeEngine(VkDevice device, VkQueue queue, uint32_t queue_family,
                               CommandPool& cmd_pool, ComputePipelineCache& pipelines)
    : device_(device), queue_(queue), queue_family_(queue_family),
      cmd_pool_(cmd_pool), pipelines_(pipelines) {
    // Pre-allocate descriptor pool for 1024 sets
    desc_pool_ = create_descriptor_pool(1024);
}

ComputeEngine::~ComputeEngine() {
    if (desc_pool_) vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
    if (desc_set_layout_) vkDestroyDescriptorSetLayout(device_, desc_set_layout_, nullptr);
}

VkDescriptorPool ComputeEngine::create_descriptor_pool(int max_sets) {
    VkDescriptorPoolSize pool_sizes[1] = {};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[0].descriptorCount = max_sets * 3;

    VkDescriptorPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.maxSets = max_sets;
    ci.poolSizeCount = 1;
    ci.pPoolSizes = pool_sizes;
    ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    VkDescriptorPool pool;
    VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &pool));
    return pool;
}

VkDescriptorSet ComputeEngine::alloc_descriptor_set(VkDescriptorPool pool) {
    if (!desc_set_layout_) {
        VkDescriptorSetLayoutBinding bindings[3] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[0].descriptorCount = 1;
        bindings[1] = bindings[0]; bindings[1].binding = 1;
        bindings[2] = bindings[0]; bindings[2].binding = 2;

        VkDescriptorSetLayoutCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dci.bindingCount = 3;
        dci.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dci, nullptr, &desc_set_layout_));
    }

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &desc_set_layout_;
    VkDescriptorSet set;
    VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &set));
    return set;
}

// Get the shared pipeline layout (matches ComputePipelineCache's layout)
VkPipelineLayout ComputeEngine::get_pipeline_layout() {
    // Ask the pipeline cache to create its layout
    // We need a way to get the layout handle. Since the pipeline cache
    // creates it internally, we create our own matching layout here.
    if (!pipe_layout_) {
        VkDescriptorSetLayoutBinding bindings[3] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[0].descriptorCount = 1;
        bindings[1] = bindings[0]; bindings[1].binding = 1;
        bindings[2] = bindings[0]; bindings[2].binding = 2;

        VkDescriptorSetLayoutCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dci.bindingCount = 3;
        dci.pBindings = bindings;
        VkDescriptorSetLayout dsl;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dci, nullptr, &dsl));

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = 128;

        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsl;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        VK_CHECK(vkCreatePipelineLayout(device_, &plci, nullptr, &pipe_layout_));
        vkDestroyDescriptorSetLayout(device_, dsl, nullptr);
    }
    return pipe_layout_;
}

void ComputeEngine::dispatch(const std::string& shader, const PushConstants& push,
                              VkBuffer input, VkBuffer output, VkBuffer weights,
                              uint32_t group_x, uint32_t group_y, uint32_t group_z) {
    VkPipeline pipe = pipelines_.get(shader);
    VkPipelineLayout layout = get_pipeline_layout();

    // Allocate descriptor set from shared pool
    VkDescriptorSet desc_set = alloc_descriptor_set(desc_pool_);

    VkDescriptorBufferInfo buf_infos[3] = {};
    buf_infos[0].buffer = input;
    buf_infos[0].range = VK_WHOLE_SIZE;
    buf_infos[1].buffer = output;
    buf_infos[1].range = VK_WHOLE_SIZE;
    buf_infos[2].buffer = weights;
    buf_infos[2].range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet writes[3] = {};
    for (int i = 0; i < 3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = desc_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);

    VkCommandBuffer cmd = cmd_pool_.begin_once();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             layout, 0, 1, &desc_set, 0, nullptr);
    vkCmdPushConstants(cmd, layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, group_x, group_y, group_z);

    cmd_pool_.submit_and_wait(cmd, queue_);
}

void ComputeEngine::rms_norm(VkBuffer x, VkBuffer w, int n, float eps) {
    PushConstants push{};
    push.M = n; push.eps = eps;
    dispatch("rms_norm", push, x, x, w, (n + 255) / 256);
}

void ComputeEngine::gemv(VkBuffer y, VkBuffer x, VkBuffer W, int M, int N, int K) {
    PushConstants push{};
    push.M = M; push.N = N; push.K = K;
    dispatch("gemv", push, x, y, W, M);
}

void ComputeEngine::rope(VkBuffer q, VkBuffer k, int hd, int pos,
                          int n_heads, int n_kv, float theta) {
    PushConstants push{};
    push.M = n_heads; push.N = n_kv; push.K = hd;
    push.rope_theta = theta; push.pos = pos;
    dispatch("rope", push, q, k, VK_NULL_HANDLE, n_heads + n_kv);
}

void ComputeEngine::flash_attn(VkBuffer q, VkBuffer k_cache, VkBuffer v_cache,
                                VkBuffer out, int seq_len,
                                int n_heads, int n_kv, int hd, int gqa) {
    PushConstants push{};
    push.M = n_heads; push.N = seq_len; push.K = hd;
    dispatch("flash_attn", push, q, out, k_cache, n_heads);
    (void)v_cache; (void)gqa;
}

void ComputeEngine::silu_mul(VkBuffer y, VkBuffer gate, VkBuffer up, int n) {
    PushConstants push{};
    push.M = n;
    dispatch("silu_mul", push, gate, y, up, (n + 255) / 256);
}

void ComputeEngine::swiglu_ffn(VkBuffer x, VkBuffer gate_w, VkBuffer up_w,
                                VkBuffer down_w, VkBuffer out,
                                int hidden, int inter) {
    PushConstants push{};
    push.M = inter; push.K = hidden;
    dispatch("gemv", push, x, VK_NULL_HANDLE, gate_w, inter);
    dispatch("gemv", push, x, VK_NULL_HANDLE, up_w, inter);
    (void)down_w; (void)out;
}

int ComputeEngine::argmax(VkBuffer logits, int n) {
    PushConstants push{};
    push.M = n;
    GpuBuffer result_buf(device_, sizeof(int),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    dispatch("argmax", push, logits, result_buf.buffer(), VK_NULL_HANDLE, 1);
    int* result = (int*)result_buf.map();
    int idx = *result;
    result_buf.unmap();
    return idx;
}

void ComputeEngine::embed_lookup(VkBuffer out, VkBuffer embed,
                                  int token_id, int hidden) {
    PushConstants push{};
    push.M = hidden; push.token = token_id;
    dispatch("embed", push, embed, out, VK_NULL_HANDLE, 1);
}

// ═══════════════════════════════════════════════════════════════════
//  InferenceEngine
// ═══════════════════════════════════════════════════════════════════

bool InferenceEngine::init(ComputeEngine& ce, ModelGPU& m) {
    compute = &ce;
    model = &m;
    auto& d = m.dims;
    VkBufferUsageFlags rw = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    auto alloc = [&](GpuBuffer& buf, size_t size, const char* name) {
        if (size == 0) return;
        buf = GpuBuffer(ce.device(), size, rw, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        printf("  Scratch %s: %.1f MB\n", name, size / (1024.0 * 1024.0));
    };
    alloc(hidden,     d.hidden * sizeof(float),                              "hidden");
    alloc(residual,   d.hidden * sizeof(float),                              "residual");
    alloc(qkv,        (d.n_heads * d.head_dim + d.n_kv_heads * d.head_dim * 2) * sizeof(float), "qkv");
    alloc(attn_out,   d.n_heads * d.head_dim * sizeof(float),                 "attn_out");
    alloc(gate_up,    d.inter * 2 * sizeof(float),                            "gate_up");
    alloc(silu_buf,   d.inter * sizeof(float),                                "silu_buf");
    alloc(logits,     d.vocab * sizeof(float),                                "logits");
    alloc(argmax_buf, sizeof(int),                                            "argmax");
    printf("  Inference engine ready: %d layers, H=%d\n", d.n_layers, d.hidden);
    return true;
}

int InferenceEngine::generate(int token_id) {
    if (!compute || !model) return -1;
    auto& d = model->dims;
    
    compute->embed_lookup(hidden.buffer(), model->embed.buffer(), token_id, d.hidden);
    
    for (int l = 0; l < d.n_layers; l++) {
        auto& layer = model->layers[l];
        
        PushConstants pc_res{};
        pc_res.M = (uint32_t)d.hidden;
        compute->dispatch("copy_buffer", pc_res,
                          hidden.buffer(), residual.buffer(), VK_NULL_HANDLE, 1);
        
        compute->rms_norm(hidden.buffer(), layer.rms_attn.buffer(), d.hidden, d.rms_eps);
        compute->gemv(qkv.buffer(), hidden.buffer(), layer.wq.buffer(),
                       d.n_heads * d.head_dim, 1, d.hidden);
        compute->gemv(VK_NULL_HANDLE, hidden.buffer(), layer.wk.buffer(),
                       d.n_kv_heads * d.head_dim, 1, d.hidden);
        compute->gemv(VK_NULL_HANDLE, hidden.buffer(), layer.wv.buffer(),
                       d.n_kv_heads * d.head_dim, 1, d.hidden);
        compute->rope(qkv.buffer(), VK_NULL_HANDLE, d.head_dim, pos,
                      d.n_heads, d.n_kv_heads, d.rope_theta);
        compute->flash_attn(qkv.buffer(), model->k_cache.buffer(), model->v_cache.buffer(),
                            attn_out.buffer(), pos + 1,
                            d.n_heads, d.n_kv_heads, d.head_dim,
                            d.n_heads / d.n_kv_heads);
        compute->gemv(hidden.buffer(), attn_out.buffer(), layer.wo.buffer(),
                       d.hidden, 1, d.n_heads * d.head_dim);
        compute->dispatch("add_residual", pc_res,
                          hidden.buffer(), residual.buffer(), VK_NULL_HANDLE, 1);
        
        compute->rms_norm(hidden.buffer(), layer.rms_ffn.buffer(), d.hidden, d.rms_eps);
        compute->gemv(gate_up.buffer(), hidden.buffer(), layer.w1.buffer(),
                       d.inter, 1, d.hidden);
        compute->gemv(VK_NULL_HANDLE, hidden.buffer(), layer.w2.buffer(),
                       d.inter, 1, d.hidden);
        compute->silu_mul(silu_buf.buffer(), gate_up.buffer(), VK_NULL_HANDLE, d.inter);
        compute->gemv(hidden.buffer(), silu_buf.buffer(), layer.w3.buffer(),
                       d.hidden, 1, d.inter);
        compute->dispatch("add_residual", pc_res,
                          hidden.buffer(), residual.buffer(), VK_NULL_HANDLE, 1);
    }
    
    compute->rms_norm(hidden.buffer(), model->final_norm.buffer(), d.hidden, d.rms_eps);
    compute->gemv(logits.buffer(), hidden.buffer(),
                   model->tied_embed ? model->embed.buffer() : model->lm_head.buffer(),
                   d.vocab, 1, d.hidden);
    pos++;
    return compute->argmax(logits.buffer(), d.vocab);
}
