#ifndef GPU_INFER_ENGINE_H
#define GPU_INFER_ENGINE_H

#include "common.h"
#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <hipblas/hipblas.h>

// GPU Inference Engine for Qwen3-0.6B using ROCm/hipBLAS
class GpuInferenceEngine {
public:
    GpuInferenceEngine();
    ~GpuInferenceEngine();

    // Initialize engine and load model weights from .q4nx file
    // Returns true on success, false on failure
    bool init(const char* model_path);

    // Generate tokens from input_ids
    // input_ids: array of input token IDs
    // input_len: number of input tokens
    // output_ids: array to store generated token IDs (must have max_output entries)
    // max_output: maximum number of tokens to generate
    // Returns: number of tokens generated
    int generate(const int* input_ids, int input_len,
                 int* output_ids, int max_output);

    // Check if engine is initialized
    bool is_initialized() const { return initialized_; }

private:
    // Model configuration (Qwen3-0.6B)
    static constexpr int H = 1024;        // hidden_size
    static constexpr int NC = 28;         // num_layers
    static constexpr int NH = 16;         // num_attention_heads
    static constexpr int NKV = 8;         // num_key_value_heads
    static constexpr int HD = 128;       // head_dim
    static constexpr int IM = 3072;       // intermediate_size
    static constexpr int NV = 151936;     // vocab_size
    static constexpr int GQA = 2;         // Q:KV ratio
    static constexpr float EPS = 1e-6f;
    static constexpr int MAX_SEQ_LEN = 4096;

    bool initialized_ = false;

    // hipBLAS handle
    hipblasHandle_t hipblas_handle_ = nullptr;

    // Memory-mapped model file
    uint8_t* model_data_ = nullptr;
    size_t model_size_ = 0;
    int fd_ = -1;

    // JSON metadata size (parsed at init)
    uint64_t json_size_ = 0;

    // Dequantized weights (FP32, owned by this class)
    struct LayerWeights {
        std::vector<float> qkv_w;   // Q+K+V combined: H × (Q+K+V) = 1024 × 4096
        std::vector<float> o_w;     // O projection: NH*HD × H = 2048 × 1024
        std::vector<float> gate_w;  // Gate projection: H × IM = 1024 × 3072
        std::vector<float> up_w;    // Up projection: H × IM = 1024 × 3072
        std::vector<float> down_w;   // Down projection: IM × H = 3072 × 1024
    };
    std::vector<LayerWeights> layer_weights_;

    // RMSNorm weights (FP16 BF16, stored as uint16_t)
    std::vector<uint16_t> input_layernorm_weights_;  // NC × H
    std::vector<uint16_t> post_attention_layernorm_weights_;  // NC × H
    std::vector<uint16_t> q_norm_weights_;  // NC × HD
    std::vector<uint16_t> k_norm_weights_;   // NC × HD
    uint16_t* final_norm_weight_ = nullptr;  // H

    // LM head weight (dequantized to float, stored as H × NV for efficient GEMM)
    std::unique_ptr<float[]> lm_head_dequant_;

    // Embedding table (FP16 BF16)
    uint16_t* embed_tokens_ = nullptr;  // NV × H

    // KV cache: [layer][2 (K/V)][seq_len × NKV × HD]
    std::vector<std::vector<float>> kv_cache_k_;
    std::vector<std::vector<float>> kv_cache_v_;
    std::vector<int> kv_cache_len_;  // current length per layer

    // GPU buffers (HIP allocations)
    float* d_q_ = nullptr;        // Q activation
    float* d_k_ = nullptr;        // K activation
    float* d_v_ = nullptr;        // V activation
    float* d_qk_ = nullptr;       // Q×K^T scores
    float* d_attn_ = nullptr;     // Attention output
    float* d_h1_ = nullptr;       // Hidden state temp
    float* d_h2_ = nullptr;       // Hidden state temp
    float* d_gate_out_ = nullptr; // Gate projection output
    float* d_up_out_ = nullptr;   // Up projection output
    float* d_swiglu_ = nullptr;   // SwiGLU result
    float* d_down_out_ = nullptr; // Down projection output
    float* d_logits_ = nullptr;   // Final logits
    float* d_k_cache_ = nullptr;  // KV cache K on GPU
    float* d_v_cache_ = nullptr;  // KV cache V on GPU

    // Host-side working buffers
    std::vector<float> h_q_;           // NH * HD = 2048
    std::vector<float> h_k_;           // NKV * HD = 1024
    std::vector<float> h_v_;           // NKV * HD = 1024
    std::vector<float> h_q_out_;       // QKV output: NH*HD + NKV*HD + NKV*HD = 4096
    std::vector<float> h_attn_out_;    // NH * HD = 2048
    std::vector<float> h_s1_;          // Hidden state temp (H = 1024)
    std::vector<float> h_s2_;          // Hidden state temp (H = 1024)
    std::vector<float> h_gate_out_;    // Gate projection output (IM = 3072)
    std::vector<float> h_up_out_;      // Up projection output (IM = 3072)
    std::vector<float> h_swiglu_out_;  // SwiGLU result (IM = 3072)
    std::vector<float> h_logits_;      // Final logits (NV = 151936)
    std::vector<float> h_scor_;        // Attention scores (MAX_SEQ_LEN)

    // Allocate GPU memory
    bool allocate_gpu_buffers();

    // Free GPU memory
    void free_gpu_buffers();

    // Copy data to GPU
    bool copy_weights_to_gpu();

    // Helper: parse JSON offset
    static uint64_t parse_json_offset(const char* js, size_t jl, const char* name);

    // Helper: BF16 decode
    static inline float bf16_to_float(uint16_t v) {
        uint32_t b = ((uint32_t)v) << 16;
        float f;
        memcpy(&f, &b, 4);
        return f;
    }

    // RMSNorm: y = x / (sqrt(sum(x^2)/n + eps)) * w
    void rms_norm(float* out, const float* in, const uint16_t* weight, int n);

    // Softmax (in-place)
    void softmax(float* scores, int n);

    // Compute one layer
    void compute_layer(int layer_idx, float* h, int seq_pos);

    // Compute final norm and logits
    void compute_logits(const float* hidden, float* logits);

    // Sample from logits
    int sample_token(const float* logits);
};

#endif // GPU_INFER_ENGINE_H
