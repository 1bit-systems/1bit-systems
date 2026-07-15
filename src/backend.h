// backend.h — Universal backend abstraction for inference
//
// This is the CANONICAL backend interface. The tests/backends/backend.h
// (InferenceBackend) is a separate parallel interface that should eventually
// be unified with this one (see TODO in that file).
//
// Each backend implements: init, forward, lm_head, generate, benchmark, destroy
// Auto-detected and selected at runtime by BackendManager.

#pragma once
#include <cstdint>
#include <string>
#include <vector>

// ── Common types ──
struct ModelConfig {
    int hidden = 2048;
    int n_heads = 8;
    int n_kv_heads = 2;
    int head_dim = 128;
    int n_layers = 40;
    int n_experts = 16;
    int n_ff = 2048;
    int vocab = 262272;
    int router_hidden = 256;
    int qkv_dim = 1280;  // QD + KD = 1024 + 256
};

// ── Backend type ──
enum class BackendType : uint8_t {
    NONE = 0,
    HIP_GPU = 1,      // AMD ROCm GPU via HIP
    VULKAN = 2,       // Any Vulkan 1.2+ GPU (AMD, NVIDIA, Intel, Apple)
    NPU_XRT = 3,      // AMD XDNA NPU via XRT
    CPU_AVX512 = 4,   // CPU with AVX-512
    CPU_SCALAR = 5,   // CPU fallback (any x86)
};

inline const char* backend_name(BackendType t) {
    switch(t) {
        case BackendType::HIP_GPU: return "HIP GPU (ROCm)";
        case BackendType::VULKAN: return "Vulkan GPU (portable)";
        case BackendType::NPU_XRT: return "NPU XDNA (XRT)";
        case BackendType::CPU_AVX512: return "CPU AVX-512";
        case BackendType::CPU_SCALAR: return "CPU (scalar)";
        default: return "none";
    }
}

// ── Backend interface ──
// All backends must implement this. Ops return true on success.
struct Backend {
    BackendType type = BackendType::NONE;
    std::string name;
    ModelConfig cfg;
    bool initialized = false;

    virtual ~Backend() = default;

    /// Initialize backend: detect hardware, load weights, allocate memory.
    /// weights_dir = path to /tmp/zaya_weights/ or equivalent
    virtual bool init(const ModelConfig& cfg, const std::string& weights_dir) = 0;

    /// Reset KV cache and router state for a new sequence.
    virtual bool reset() = 0;

    /// Run one token through all 40 layers.
    /// token_id = input token, hidden_out[hidden] = output hidden state
    virtual bool forward(int token_id, float* hidden_out) = 0;

    /// Compute lm_head: logits[vocab] = hidden[hidden] @ embed[vocab×hidden]^T
    virtual bool lm_head(const float* hidden, float* logits, int* argmax) = 0;

    /// Generate one token (forward + lm_head in one call).
    /// Returns the predicted token ID, -1 on error.
    virtual int generate(int token_id) = 0;

    /// Clean up resources.
    virtual void destroy() = 0;

    /// Benchmark: run N iterations, return ms/token.
    virtual float benchmark(int tokens = 10) = 0;

    /// True if this backend can actually run inference (forward/lm_head/generate).
    /// Defaults to true; stub backends that only detect hardware override to false
    /// so BackendManager discovers them but never selects them for inference (#82).
    virtual bool can_infer() const { return true; }
};

// ── Factory: auto-detect and create best available backend ──
Backend* create_best_backend();
Backend* create_backend(BackendType type);

// ── CPU backend ──
Backend* create_cpu_backend();

// ── Vulkan backend ──
Backend* create_vulkan_backend();

// ── HIP backend ──
Backend* create_hip_backend();

// ── NPU backend ──
Backend* create_npu_backend();

// ── Auto-detect ──
BackendType detect_backends();
