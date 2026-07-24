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
    {"gemv", "dmmv_q4k"},           // Q4_K quantized GEMV
    {"gemv_batch", "dmmv_q4k_batch"}, // Q4_K batched GEMV (multi-prompt)
    {"fused_qkv", "fused_qkv"},     // Fused QKV projection
    {"fused_gate_up", "fused_gate_up"}, // Fused gate+up projection
    {"rms_norm", "rms_norm_mul"},   // RMS norm + weight multiply (fused)
    {"rope", "rope_fused"},          // fused RoPE
    {"flash_attn", "flash_attn"},    // flash attention
    {"silu_mul", "swiglu"},          // SiLU gate multiply
    {"argmax", "argmax"},            // argmax reduction
    {"add_residual", "vadd"},        // vector add
    {"copy_buffer", "copy_buffer"},  // out = in
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

// Async fence for double-buffered dispatch
static thread_local VkFence s_async_fence = VK_NULL_HANDLE;

void ComputeEngine::end_batch() {
    if (!s_batching) return;
    
    if (s_async_fence == VK_NULL_HANDLE) {
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(device_, &fci, nullptr, &s_async_fence);
    } else {
        // Wait for previous token's batch to finish (overlaps CPU prep with GPU exec)
        vkWaitForFences(device_, 1, &s_async_fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device_, 1, &s_async_fence);
    }
    
    cmd_pool_.submit(s_batch_cmd, queue_, s_async_fence);
    s_batching = false;
    s_batch_cmd = VK_NULL_HANDLE;
    reset_descriptors();
}

// Called before shutdown to drain last async batch
void ComputeEngine::sync() {
    if (s_async_fence) {
        vkWaitForFences(device_, 1, &s_async_fence, VK_TRUE, UINT64_MAX);
        vkDestroyFence(device_, s_async_fence, nullptr);
        s_async_fence = VK_NULL_HANDLE;
    }
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

    // Serialize this dispatch's shader writes before the next dispatch reads
    // them. Every op in the batch feeds the next (rms_norm -> qkv -> attn ->
    // ...), so without this barrier consecutive dispatches race (RAW hazard)
    // and the result is undefined once more than one layer runs.
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(s_batch_cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
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
    PushConstants push{}; push.M = M; push.N = (batch_size_ > 1) ? batch_size_ : N; push.K = K;
    std::string shader = (batch_size_ > 1) ? "gemv_batch" : "gemv";
    batch_dispatch(this, shader, push, x, y, W, M);
}

void ComputeEngine::gemv_f32(VkBuffer y, VkBuffer x, VkBuffer W, int M, int N, int K) {
    PushConstants push{}; push.M = M; push.N = N; push.K = K;
    // Tile into a 2D grid when M exceeds maxComputeWorkGroupCount[0] (65535 on
    // RADV) — the lm_head has M = vocab (~152k). The shader recovers the row as
    // x + y * NumWorkGroups.x.
    uint32_t gx = (uint32_t)M, gy = 1;
    if (gx > 65535u) { gy = (gx + 65534u) / 65535u; gx = 65535u; }
    batch_dispatch(this, "gemv_f32", push, x, y, W, gx, gy, 1);
}

void ComputeEngine::rope(VkBuffer q, VkBuffer k, int hd, int pos,
                          int n_heads, int n_kv, float theta) {
    PushConstants push{}; push.M = n_heads; push.N = n_kv; push.K = hd;
    push.rope_theta = theta; push.pos = pos;
    batch_dispatch(this, "rope", push, q, k, VK_NULL_HANDLE, n_heads + n_kv);
}

void ComputeEngine::flash_attn(VkBuffer q, VkBuffer k_cache, VkBuffer v_cache,
                                VkBuffer out, int seq_len,
                                int n_heads, int n_kv, int hd, int gqa,
                                int layer, int max_seq, int n_layers) {
    PushConstants push{};
    push.M = (uint32_t)n_heads; push.N = (uint32_t)seq_len; push.K = (uint32_t)hd;
    push.stride = (uint32_t)n_kv;      // shader reads this as 'n_kv'
    push.layer = layer;                // 'layer'
    push.token = max_seq;              // shader reads this as 'max_seq'
    push.head = n_layers;              // shader reads this as 'n_layers'
    push.pos = seq_len - 1;
    batch_dispatch(this, "flash_attn", push, q, out, k_cache, n_heads);
    (void)v_cache; (void)gqa;
}

void ComputeEngine::silu_mul(VkBuffer y, VkBuffer gate, VkBuffer up, int n) {
    PushConstants push{}; push.M = n;
    batch_dispatch(this, "silu_mul", push, gate, y, up, (n + 255) / 256);
}

int ComputeEngine::argmax(VkBuffer logits, int n) {
    PushConstants push{}; push.M = n;
    GpuBuffer result_buf(device_, sizeof(int) * 2,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    batch_dispatch(this, "argmax", push, logits, result_buf.buffer(), VK_NULL_HANDLE, 1);
    int* result = (int*)result_buf.map();
    int idx = *result;
    static const bool dbg = getenv("ZINC_DEBUG") != nullptr;
    if (dbg) { float mx; memcpy(&mx, &result[1], 4); fprintf(stderr, "[zinc] argmax idx=%d maxlogit=%g\n", idx, mx); }
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

    static const char* nl_env = getenv("ZINC_NUM_LAYERS");
    int n_layers_run = nl_env ? atoi(nl_env) : d.n_layers;
    if (n_layers_run > d.n_layers) n_layers_run = d.n_layers;

    // Batch all layer dispatches into one command buffer
    compute->begin_batch();
    
    for (int l = 0; l < n_layers_run; l++) {
        auto& layer = model->layers[l];
        
        PushConstants pc_res{};
        pc_res.M = (uint32_t)d.hidden;
        compute->dispatch_batch("copy_buffer", pc_res,
                          hidden.buffer(), residual.buffer(), VK_NULL_HANDLE, 1);
        
        compute->rms_norm(hidden.buffer(), layer.rms_attn.buffer(), d.hidden, d.rms_eps);
        // Fused QKV: single dispatch for Q, K, V projections
        {
            uint32_t qkv_rows = d.n_heads * d.head_dim + 2 * d.n_kv_heads * d.head_dim;
            // F32 fused QKV projection: qkv = W_qkv @ hidden.
            compute->gemv_f32(qkv.buffer(), hidden.buffer(), layer.qkv_fused.buffer(),
                              (int)qkv_rows, 1, d.hidden);
            // Add QKV bias (Qwen2/ZR1) in-place: qkv += bias.
            if (layer.has_bias) {
                PushConstants pc_b{}; pc_b.M = qkv_rows;
                compute->dispatch_batch("add_residual", pc_b,
                                  qkv.buffer(), layer.qkv_bias_fused.buffer(), VK_NULL_HANDLE, 1);
            }
        }
        compute->rope(qkv.buffer(), VK_NULL_HANDLE, d.head_dim, pos,
                      d.n_heads, d.n_kv_heads, d.rope_theta);
        compute->flash_attn(qkv.buffer(), model->kv_cache.buffer(), model->kv_cache.buffer(),
                            attn_out.buffer(), pos + 1,
                            d.n_heads, d.n_kv_heads, d.head_dim,
                            d.n_heads / d.n_kv_heads,
                            l, d.max_seq, d.n_layers);
        compute->gemv_f32(hidden.buffer(), attn_out.buffer(), layer.wo.buffer(),
                       d.hidden, 1, d.n_heads * d.head_dim);
        compute->dispatch_batch("add_residual", pc_res,
                          hidden.buffer(), residual.buffer(), VK_NULL_HANDLE, 1);

        // Refresh the residual base to the post-attention hidden before the FFN
        // block. rms_norm() is in-place, so without this the FFN residual would
        // add the original layer input (losing the attention contribution).
        compute->dispatch_batch("copy_buffer", pc_res,
                          hidden.buffer(), residual.buffer(), VK_NULL_HANDLE, 1);

        compute->rms_norm(hidden.buffer(), layer.rms_ffn.buffer(), d.hidden, d.rms_eps);
        // Fused gate+up: single dispatch for gate and up projections
        {
            uint32_t gu_rows = 2 * d.inter;
            // F32 fused gate+up projection: gate_up = W_gu @ hidden.
            compute->gemv_f32(gate_up.buffer(), hidden.buffer(), layer.gate_up_fused.buffer(),
                              (int)gu_rows, 1, d.hidden);
        }
        compute->silu_mul(silu_buf.buffer(), gate_up.buffer(), VK_NULL_HANDLE, d.inter);
        compute->gemv_f32(hidden.buffer(), silu_buf.buffer(), layer.w3.buffer(),
                       d.hidden, 1, d.inter);
        compute->dispatch_batch("add_residual", pc_res,
                          hidden.buffer(), residual.buffer(), VK_NULL_HANDLE, 1);
    }

    if (getenv("ZINC_DEBUG_HID")) {
        compute->end_batch();
        int hi = compute->argmax(hidden.buffer(), d.hidden);  // prints max|hidden| via [zinc] argmax
        (void)hi;
        compute->begin_batch();
    }

    compute->rms_norm(hidden.buffer(), model->final_norm.buffer(), d.hidden, d.rms_eps);
    // lm_head: untied output.weight if present, else tied embeddings (both F32).
    VkBuffer head = model->has_lm_head ? model->lm_head.buffer() : model->embed.buffer();
    compute->gemv_f32(logits.buffer(), hidden.buffer(), head, d.vocab, 1, d.hidden);
    
    // End batch — submits all recorded dispatches
    compute->end_batch();
    
    pos++;
    return compute->argmax(logits.buffer(), d.vocab);
}

