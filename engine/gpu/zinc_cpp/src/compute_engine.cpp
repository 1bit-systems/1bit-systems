/// Compute engine — GPU shader dispatch implementations.
/// Fixed: #774 descriptor set leak, #777 shared layouts, #776 fence reuse
#include "compute_engine.h"
#include <map>
#include <cstring>
#include <cfloat>

// ═══════════════════════════════════════════════════════════════════
//  ComputeEngine
// ═══════════════════════════════════════════════════════════════════

ComputeEngine::ComputeEngine(VkDevice device, VkQueue queue, uint32_t queue_family,
                               CommandPool& cmd_pool, ComputePipelineCache& pipelines)
    : device_(device), queue_(queue), queue_family_(queue_family),
      cmd_pool_(cmd_pool), pipelines_(pipelines) {
    // Pre-allocate descriptor pool for 65536 sets (enough for multiple tokens)
    VkDescriptorPoolSize ps[1] = {};
    ps[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps[0].descriptorCount = 65536 * 3;
    VkDescriptorPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.maxSets = 65536;
    ci.poolSizeCount = 1;
    ci.pPoolSizes = ps;
    ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &desc_pool_));
    
    // Use the shared layout from pipeline cache
    desc_set_layout_ = pipelines.desc_set_layout();
}

ComputeEngine::~ComputeEngine() {
    if (desc_pool_) vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
}

void ComputeEngine::reset_descriptors() {
    // Reset pool to free all allocated descriptor sets (fix #774)
    if (desc_pool_) {
        vkResetDescriptorPool(device_, desc_pool_, 0);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Batched dispatch thread-local state
// ═══════════════════════════════════════════════════════════════════
static thread_local VkCommandBuffer s_batch_cmd = VK_NULL_HANDLE;
static thread_local bool s_batching = false;

// Shader name map — inference op names to .spv filenames
// Uses production ZINC shaders from engine/gpu/shaders/
static const std::map<std::string,std::string> shader_map = {
    {"gemv", "dmmv_f32"},           // FP32 GEMV (production)
    {"rms_norm", "rms_norm_mul"},   // RMS norm + weight multiply (fused)
    {"rope", "rope_fused"},          // fused RoPE
    {"flash_attn", "flash_attn"},    // flash attention
    {"silu_mul", "swiglu"},          // SiLU gate multiply
    {"argmax", "argmax"},            // argmax reduction
    {"add_residual", "vadd"},        // vector add
    {"copy_buffer", "vadd"},         // copy via add
    {"embed", "embed"},              // embedding lookup
};

void ComputeEngine::dispatch(const std::string& shader, const PushConstants& push,
                              VkBuffer input, VkBuffer output, VkBuffer weights,
                              uint32_t group_x, uint32_t group_y, uint32_t group_z) {
    std::string actual_shader = shader;
    auto it = shader_map.find(shader);
    if (it != shader_map.end()) actual_shader = it->second;
    
    VkPipeline pipe = pipelines_.get(actual_shader);
    VkPipelineLayout layout = pipelines_.pipeline_layout();

    // Allocate descriptor set from shared pool
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = desc_pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &desc_set_layout_;
    VkDescriptorSet desc_set;
    VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &desc_set));

    VkDescriptorBufferInfo buf_infos[3] = {};
    VkWriteDescriptorSet writes[3] = {};
    int nwrites = 0;
    
    if (input) {
        buf_infos[nwrites].buffer = input; buf_infos[nwrites].range = VK_WHOLE_SIZE;
        writes[nwrites].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[nwrites].dstSet = desc_set;
        writes[nwrites].dstBinding = nwrites;
        writes[nwrites].descriptorCount = 1;
        writes[nwrites].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[nwrites].pBufferInfo = &buf_infos[nwrites];
        nwrites++;
    }
    if (output) {
        buf_infos[nwrites].buffer = output; buf_infos[nwrites].range = VK_WHOLE_SIZE;
        writes[nwrites].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[nwrites].dstSet = desc_set;
        writes[nwrites].dstBinding = nwrites;
        writes[nwrites].descriptorCount = 1;
        writes[nwrites].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[nwrites].pBufferInfo = &buf_infos[nwrites];
        nwrites++;
    }
    if (weights) {
        buf_infos[nwrites].buffer = weights; buf_infos[nwrites].range = VK_WHOLE_SIZE;
        writes[nwrites].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[nwrites].dstSet = desc_set;
        writes[nwrites].dstBinding = nwrites;
        writes[nwrites].descriptorCount = 1;
        writes[nwrites].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[nwrites].pBufferInfo = &buf_infos[nwrites];
        nwrites++;
    }
    if (nwrites > 0) vkUpdateDescriptorSets(device_, nwrites, writes, 0, nullptr);

    VkCommandBuffer cmd = cmd_pool_.begin_once();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             layout, 0, 1, &desc_set, 0, nullptr);
    vkCmdPushConstants(cmd, layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, group_x, group_y, group_z);
    cmd_pool_.submit_and_wait(cmd, queue_);
}




void ComputeEngine::begin_batch() {
    if (s_batching) return;
    s_batch_cmd = cmd_pool_.begin_once();
    s_batching = true;
}

void ComputeEngine::end_batch() {
    if (!s_batching) return;
    cmd_pool_.submit_and_wait(s_batch_cmd, queue_);
    s_batching = false;
    s_batch_cmd = VK_NULL_HANDLE;
    reset_descriptors();
}

void ComputeEngine::dispatch_batch(const std::string& shader, const PushConstants& push,
                                    VkBuffer input, VkBuffer output, VkBuffer weights,
                                    uint32_t group_x, uint32_t group_y, uint32_t group_z) {
    // Fall back to regular dispatch if not batching
    if (!s_batching) {
        dispatch(shader, push, input, output, weights, group_x, group_y, group_z);
        return;
    }
    
    // Resolve shader name
    auto it = shader_map.find(shader);
    std::string actual_shader = (it != shader_map.end()) ? it->second : shader;
    VkPipeline pipe = pipelines_.get(actual_shader);
    VkPipelineLayout layout = pipelines_.pipeline_layout();

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = desc_pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &desc_set_layout_;
    VkDescriptorSet desc_set;
    VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &desc_set));

    VkDescriptorBufferInfo buf_infos[3] = {};

    VkWriteDescriptorSet writes[3] = {};
    int nwrites = 0;
    auto add_desc = [&](VkBuffer buf) {
        if (!buf) return;
        buf_infos[nwrites].buffer = buf; buf_infos[nwrites].range = VK_WHOLE_SIZE;
        writes[nwrites].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[nwrites].dstSet = desc_set;
        writes[nwrites].dstBinding = nwrites;
        writes[nwrites].descriptorCount = 1;
        writes[nwrites].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[nwrites].pBufferInfo = &buf_infos[nwrites];
        nwrites++;
    };
    add_desc(input); add_desc(output); add_desc(weights);
    if (nwrites > 0) vkUpdateDescriptorSets(device_, nwrites, writes, 0, nullptr);

    vkCmdBindPipeline(s_batch_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(s_batch_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             layout, 0, 1, &desc_set, 0, nullptr);
    vkCmdPushConstants(s_batch_cmd, layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(s_batch_cmd, group_x, group_y, group_z);
}

// Use batch dispatch if batching, else regular dispatch
static inline void batch_dispatch(ComputeEngine* ce, const std::string& s, const PushConstants& p,
                              VkBuffer in, VkBuffer out, VkBuffer w,
                              uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
    if (s_batching) ce->dispatch_batch(s, p, in, out, w, gx, gy, gz);
    else ce->dispatch(s, p, in, out, w, gx, gy, gz);
}

void ComputeEngine::rms_norm(VkBuffer x, VkBuffer w, int n, float eps) {
    PushConstants push{}; push.M = n; push.eps = eps;
    batch_dispatch(this, "rms_norm", push, x, x, w, (n + 255) / 256);
}

void ComputeEngine::gemv(VkBuffer y, VkBuffer x, VkBuffer W, int M, int N, int K) {
    PushConstants push{}; push.M = M; push.N = N; push.K = K;
    batch_dispatch(this, "gemv", push, x, y, W, M);
}

void ComputeEngine::rope(VkBuffer q, VkBuffer k, int hd, int pos,
                          int n_heads, int n_kv, float theta) {
    PushConstants push{}; push.M = n_heads; push.N = n_kv; push.K = hd;
    push.rope_theta = theta; push.pos = pos;
    batch_dispatch(this, "rope", push, q, k, VK_NULL_HANDLE, n_heads + n_kv);
}

void ComputeEngine::flash_attn(VkBuffer q, VkBuffer k_cache, VkBuffer v_cache,
                                VkBuffer out, int seq_len,
                                int n_heads, int n_kv, int hd, int gqa) {
    PushConstants push{}; push.M = n_heads; push.N = seq_len; push.K = hd;
    batch_dispatch(this, "flash_attn", push, q, out, k_cache, n_heads);
    (void)v_cache; (void)gqa;
}

void ComputeEngine::silu_mul(VkBuffer y, VkBuffer gate, VkBuffer up, int n) {
    PushConstants push{}; push.M = n;
    batch_dispatch(this, "silu_mul", push, gate, y, up, (n + 255) / 256);
}

int ComputeEngine::argmax(VkBuffer logits, int n) {
    PushConstants push{}; push.M = n;
    GpuBuffer result_buf(device_, sizeof(int),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    batch_dispatch(this, "argmax", push, logits, result_buf.buffer(), VK_NULL_HANDLE, 1);
    int* result = (int*)result_buf.map();
    int idx = *result;
    result_buf.unmap();
    return idx;
}

void ComputeEngine::embed_lookup(VkBuffer out, VkBuffer embed,
                                  int token_id, int hidden) {
    PushConstants push{}; push.M = hidden; push.token = token_id;
    batch_dispatch(this, "embed", push, embed, out, VK_NULL_HANDLE, 1);
}

// ═══════════════════════════════════════════════════════════════════
//  InferenceEngine
// ═══════════════════════════════════════════════════════════════════

bool InferenceEngine::init(ComputeEngine& ce, ModelGPU& m) {
    compute = &ce; model = &m;
    auto& d = m.dims;
    VkBufferUsageFlags rw = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    auto alloc = [&](GpuBuffer& buf, size_t sz, const char* name) {
        if (sz == 0) return;
        buf = GpuBuffer(ce.device(), sz, rw, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        printf("  Scratch %s: %.1f MB\n", name, sz / (1024.0 * 1024.0));
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

void InferenceEngine::reset() {
    pos = 0;
    if (compute) compute->reset_descriptors();
}

int InferenceEngine::generate(int token_id) {
    if (!compute || !model) return -1;
    auto& d = model->dims;
    
    compute->embed_lookup(hidden.buffer(), model->embed.buffer(), token_id, d.hidden);
    
    // Batch all layer dispatches into one command buffer
    compute->begin_batch();
    
    for (int l = 0; l < d.n_layers; l++) {
        auto& layer = model->layers[l];
        
        PushConstants pc_res{};
        pc_res.M = (uint32_t)d.hidden;
        compute->dispatch_batch("copy_buffer", pc_res,
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
        compute->flash_attn(qkv.buffer(), model->kv_cache.buffer(), model->kv_cache.buffer(),
                            attn_out.buffer(), pos + 1,
                            d.n_heads, d.n_kv_heads, d.head_dim,
                            d.n_heads / d.n_kv_heads);
        compute->gemv(hidden.buffer(), attn_out.buffer(), layer.wo.buffer(),
                       d.hidden, 1, d.n_heads * d.head_dim);
        compute->dispatch_batch("add_residual", pc_res,
                          hidden.buffer(), residual.buffer(), VK_NULL_HANDLE, 1);
        
        compute->rms_norm(hidden.buffer(), layer.rms_ffn.buffer(), d.hidden, d.rms_eps);
        compute->gemv(gate_up.buffer(), hidden.buffer(), layer.w1.buffer(),
                       d.inter, 1, d.hidden);
        compute->gemv(VK_NULL_HANDLE, hidden.buffer(), layer.w2.buffer(),
                       d.inter, 1, d.hidden);
        compute->silu_mul(silu_buf.buffer(), gate_up.buffer(), VK_NULL_HANDLE, d.inter);
        compute->gemv(hidden.buffer(), silu_buf.buffer(), layer.w3.buffer(),
                       d.hidden, 1, d.inter);
        compute->dispatch_batch("add_residual", pc_res,
                          hidden.buffer(), residual.buffer(), VK_NULL_HANDLE, 1);
    }
    
    compute->rms_norm(hidden.buffer(), model->final_norm.buffer(), d.hidden, d.rms_eps);
    compute->gemv(logits.buffer(), hidden.buffer(),
                   model->embed.buffer(), d.vocab, 1, d.hidden);
    
    // End batch — submits all recorded dispatches
    compute->end_batch();
    
    // Argmax reads back to CPU — separate sync
    pos++;
    return compute->argmax(logits.buffer(), d.vocab);
}

