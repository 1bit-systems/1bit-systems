#pragma once
#include <cstdint>
#include <string>
#include <vector>

enum class BackendType : uint8_t {
    NONE = 0,
    HIP_GPU = 1,
    VULKAN = 2,
    NPU_XRT = 3,
    CPU_AVX512 = 4,
    CPU_SCALAR = 5,
    GENERIC = 6,
    ZAMBA2 = 7,   // Zamba2 hybrid Mamba2+attention (CPU ref)
    ZAMBA2_GPU = 8, // Zamba2 with HIP acceleration
    ZINC_GPU = 9,   // General GGUF backend via engine/gpu (ZINC), multi-arch/multi-quant
    NPU_FLM = 10,   // NPU inference via FastFlowLM subprocess (in-process kernels are broken, see docs/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md)
    Q4NX_FUSION = 11, // Q4NX format via engine/fusion, forced cpu_only policy
};

inline const char* backend_name(BackendType t) {
    switch(t) {
        case BackendType::HIP_GPU: return "HIP GPU (ROCm)";
        case BackendType::VULKAN: return "Vulkan GPU (portable)";
        case BackendType::NPU_XRT: return "NPU XDNA (XRT)";
        case BackendType::CPU_AVX512: return "CPU AVX-512";
        case BackendType::CPU_SCALAR: return "CPU (scalar)";
        case BackendType::GENERIC: return "Generic CPU (GGUF)";
        case BackendType::ZAMBA2: return "Zamba2 (Mamba2 CPU)";
        case BackendType::ZAMBA2_GPU: return "Zamba2 (Mamba2 GPU)";
        case BackendType::ZINC_GPU: return "ZINC GPU (Vulkan, multi-arch)";
        case BackendType::NPU_FLM: return "NPU via FastFlowLM";
        case BackendType::Q4NX_FUSION: return "Q4NX Fusion (CPU)";
        default: return "none";
    }
}

enum class ModelFormat : uint8_t {
    UNKNOWN = 0,
    GGUF = 1,
    H1B = 2,
    Q4NX = 3,
    SAFETENSORS = 4,
    RAW_BIN = 5,
};

struct ModelConfig {
    int hidden            = 2048;
    int n_heads           = 8;
    int n_kv_heads        = 2;
    int head_dim          = 128;
    int n_layers          = 40;
    int n_experts         = 16;
    int n_ff              = 2048;
    int vocab             = 262272;
    int router_hidden     = 256;
    int qkv_dim           = 1280;

    int hidden_size       = 2048;
    int num_heads         = 8;
    int num_kv_heads      = 2;
    int num_layers        = 40;
    int vocab_size        = 262272;
    int intermediate_size = 2048;
    int num_experts       = 16;
    int num_experts_top   = 2;
    int num_attention_heads = 8;
    bool has_q_norm = false;
    bool has_k_norm = false;
    bool gu_split = false;
    int max_seq_len       = 2048;
    float rope_theta      = 500000.0f;
    float rms_norm_eps    = 1e-5f;
    std::string model_name = "unknown";
    std::string model_path;
    std::string weights_dir;
    std::string lora_path;   // optional .lora file for adapter merge

    ModelFormat format = ModelFormat::UNKNOWN;
    std::string architecture;   // e.g. "llama", "qwen2", "qwen3", "gemma", "phi3", "zaya1" — from
                                 // general.architecture (GGUF) or format-specific header, NOT model_name
    std::string quantization;   // best-effort, e.g. "Q4_K_M", "Q8_0", "F16", "ternary"
};