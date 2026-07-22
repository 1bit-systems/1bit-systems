#pragma once
/// Model loader — parses GGUF files and uploads weights to GPU.
/// Port of ZINC's model/*.zig to C++.
/// Reuses the existing src/gguf_reader.cpp for GGUF parsing,
/// then uploads tensors to Vulkan storage buffers.
#include "vulkan_wrapper.h"
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <map>

// ─── Model dimensions (parsed from GGUF header) ──────────────────
struct ModelDims {
    int n_layers = 0;
    int hidden = 0;       // hidden_size
    int n_heads = 0;      // num_attention_heads
    int n_kv_heads = 0;   // num_kv_heads
    int head_dim = 0;     // head_dim (computed: hidden / n_heads, or explicit)
    int inter = 0;        // intermediate_size (FFN)
    int vocab = 0;        // vocab_size
    int max_seq = 4096;   // max_position_embeddings
    float rope_theta = 10000.0f;
    float rms_eps = 1e-6f;
    int n_experts = 0;    // MoE experts (0 = dense)
    int n_experts_top = 0; // MoE top-k
    std::string arch;     // architecture name (llama, qwen2, gemma, etc.)
};

// ─── Weight tensor descriptor ────────────────────────────────────
struct WeightTensor {
    std::string name;
    std::vector<uint32_t> shape;
    uint32_t dtype = 0;  // GGUF dtype enum
    uint64_t offset = 0; // byte offset in file
    uint64_t numel = 0;
    
    // Uploaded GPU buffer (after load_model)
    GpuBuffer gpu_buf;
    bool on_gpu = false;
};

// ─── Per-layer weights on GPU ────────────────────────────────────
struct LayerWeightsGPU {
    GpuBuffer wq, wk, wv, wo;     // attention [out, in]
    GpuBuffer w1, w2, w3;         // FFN gate/up/down [out, in]
    GpuBuffer rms_attn, rms_ffn;  // RMS norm [hidden]
    GpuBuffer bq, bk, bv;         // optional QKV biases
    GpuBuffer qkv_fused;          // fused QKV weights [Q+K+V, hidden]
    GpuBuffer gate_up_fused;      // fused gate+up weights [2*inter, hidden]
    bool has_bias = false;
};

// ─── Model on GPU ────────────────────────────────────────────────
struct ModelGPU {
    ModelDims dims;
    
    // Shared weights
    GpuBuffer embed;       // [vocab, hidden] — token embeddings
    GpuBuffer final_norm;  // [hidden]
    
    // KV cache: single contiguous buffer [2 * n_layers * max_seq * n_kv * hd] floats
    GpuBuffer kv_cache;
    int kv_cache_capacity = 0;
    
    // Per-layer weights
    std::vector<LayerWeightsGPU> layers;
};

// ─── Model Loader ────────────────────────────────────────────────
class ModelLoader {
public:
    ModelLoader(VkDevice device, VkQueue queue, uint32_t queue_family,
                 CommandPool& cmd_pool);
    ~ModelLoader() = default;

    /// Load a GGUF model and upload all weights to GPU.
    /// @param gguf_path  Path to .gguf file
    /// @param model      Output: populated ModelGPU with GPU buffers
    /// @return true on success
    bool load(const std::string& gguf_path, ModelGPU& model);

private:
    VkDevice device_;
    VkQueue queue_;
    uint32_t queue_family_;
    CommandPool& cmd_pool_;

    /// Parse GGUF header for tensor info
    bool scan_gguf(const std::string& path, ModelDims& dims,
                   std::vector<WeightTensor>& tensors);
};
