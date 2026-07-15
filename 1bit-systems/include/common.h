#pragma once
#include <cstdint>
#include <string>
#include <vector>

enum class BackendType : uint8_t {
    NONE = 0,
    HIP_GPU = 1,
    HIP = 1,
    VULKAN = 2,
    Vulkan = 2,
    NPU_XRT = 3,
    NPU = 3,
    CPU_AVX512 = 4,
    CPU = 4,
    CPU_SCALAR = 5,
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
};