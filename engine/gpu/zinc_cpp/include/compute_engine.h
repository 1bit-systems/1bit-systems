#pragma once
/// Compute engine — dispatches GPU shaders for transformer inference.
/// Port of ZINC's compute/forward.zig + compute/dmmv.zig + compute/elementwise.zig
/// + compute/graph.zig + compute/attention.zig to C++.
///
/// Loads pre-compiled .spv shaders and dispatches them via Vulkan compute.
/// Each operation (dmmv_q4k, rms_norm, flash_attn, rope, swiglu, etc.)
/// is a compute shader dispatch with push constants + buffer bindings.
#include "vulkan_wrapper.h"
#include "model_loader.h"
#include <vector>
#include <cstdint>
#include <cstdio>

// ─── Push constants for shader dispatch ──────────────────────────
// Layout matches ZINC's shader push constant structure:
//   vec4 dims   — M, N, K, stride (model dimensions)
//   vec4 params — scale, eps, rope_theta, etc.
//   ivec4 ids   — token_id, layer, head, position
struct alignas(16) PushConstants {
    uint32_t M = 0, N = 0, K = 0, stride = 0;  // matmul dims
    float    scale = 1.0f, eps = 1e-6f, rope_theta = 10000.0f, pad0 = 0;
    int32_t  token = 0, layer = 0, head = 0, pos = 0;
};

// ─── Descriptor set for compute shader ───────────────────────────
// Each dispatch binds a descriptor set with:
//   binding 0: input buffer  (activations)
//   binding 1: output buffer (result)
//   binding 2: weights buffer (model weights)
struct ComputeBindings {
    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
};

// ─── Compute engine ──────────────────────────────────────────────
class ComputeEngine {
public:
    ComputeEngine(VkDevice device, VkQueue queue, uint32_t queue_family,
                   CommandPool& cmd_pool, ComputePipelineCache& pipelines);
    ~ComputeEngine();

    ComputeEngine(const ComputeEngine&) = delete;
    ComputeEngine& operator=(const ComputeEngine&) = delete;

    // ─── Dispatch helpers ──
    void dispatch(const std::string& shader, const PushConstants& push,
                  VkBuffer input, VkBuffer output, VkBuffer weights,
                  uint32_t group_x, uint32_t group_y = 1, uint32_t group_z = 1);

    void rms_norm(VkBuffer x, VkBuffer w, int n, float eps);
    void gemv(VkBuffer y, VkBuffer x, VkBuffer W, int M, int N, int K);
    void rope(VkBuffer q, VkBuffer k, int hd, int pos, int n_heads, int n_kv, float theta);
    void flash_attn(VkBuffer q, VkBuffer k_cache, VkBuffer v_cache, VkBuffer out,
                    int seq_len, int n_heads, int n_kv, int hd, int gqa);
    void silu_mul(VkBuffer y, VkBuffer gate, VkBuffer up, int n);
    int argmax(VkBuffer logits, int n);
    void embed_lookup(VkBuffer out, VkBuffer embed, int token_id, int hidden);

    VkDevice device() const { return device_; }
    // Reset descriptor pool for new inference sequence
    void reset_descriptors();

private:
    VkDevice device_;
    VkQueue queue_;
    uint32_t queue_family_;
    CommandPool& cmd_pool_;
    ComputePipelineCache& pipelines_;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_set_layout_ = VK_NULL_HANDLE;
};

// ─── Inference engine — orchestrates model layers on GPU ───────────────
struct InferenceEngine {
    ComputeEngine* compute = nullptr;
    ModelGPU* model = nullptr;
    int pos = 0;
    GpuBuffer hidden, residual, qkv, attn_out, gate_up, silu_buf, logits, argmax_buf;

    bool init(ComputeEngine& ce, ModelGPU& m);
    void reset();
    int generate(int token_id);
};
