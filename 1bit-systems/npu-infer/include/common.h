#ifndef NPU_INFER_COMMON_H
#define NPU_INFER_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPU_MAX_CTX 16

// Model configuration for Qwen3-0.6B (NPU-verified values)
typedef struct {
    int32_t hidden_size;
    int32_t num_layers;
    int32_t num_attention_heads;
    int32_t num_key_value_heads;
    int32_t intermediate_size;
    int32_t head_dim;
    int32_t vocab_size;
    int32_t max_position_embeddings;
    int32_t max_seq_len;
    float rms_norm_eps;
    bool tie_word_embeddings;
    
    // NPU-specific blocked format constants
    uint32_t npu_block_cols;   // 1024 — each BO holds this many columns
    uint32_t npu_block_rows;   // 256 — each BO holds this many rows
    uint32_t npu_weight_bo_size; // 1048576 — 1MB per weight BO
    uint32_t npu_activation_bo_size; // 1048576 — 1MB per activation BO
    uint32_t npu_kv_cache_bo_size; // 134217728 — 128MB per KV cache BO
} ModelConfig;

static const ModelConfig QWEN3_0_6B_CONFIG = {
    .hidden_size = 1024,
    .num_layers = 28,
    .num_attention_heads = 16,
    .num_key_value_heads = 8,
    .intermediate_size = 3072,
    .head_dim = 128,
    .vocab_size = 151936,
    .max_position_embeddings = 40960,
    .max_seq_len = 4096,
    .rms_norm_eps = 1e-6f,
    .tie_word_embeddings = true,
    // NPU blocked format
    .npu_block_cols = 1024,
    .npu_block_rows = 256,
    .npu_weight_bo_size = 1048576,
    .npu_activation_bo_size = 1048576,
    .npu_kv_cache_bo_size = 134217728,
};

// XCLBIN types
typedef enum {
    XCLBIN_MM = 0,       // GEMM operations
    XCLBIN_ATTN,         // Attention
    XCLBIN_LAYER,        // Full layer
    XCLBIN_DEQUANT,      // Dequantization
    XCLBIN_COUNT
} XclbinType;

static const char* XCLBIN_NAMES[XCLBIN_COUNT] = {
    "mm.xclbin",
    "attn.xclbin",
    "layer.xclbin",
    "dequant.xclbin"
};

static const char* XCLBIN_PATHS[XCLBIN_COUNT] = {
    "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/mm.xclbin",
    "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/attn.xclbin",
    "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/layer.xclbin",
    "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/dequant.xclbin",
};

// NPU kernel argument indices
typedef enum {
    KERN_ARG_OPCODE = 0,    // uint32_t
    KERN_ARG_INSTR_PTR,     // uint64_t (0 for pre-compiled)
    KERN_ARG_NINSTR,        // uint32_t (0 for pre-compiled)
    KERN_ARG_ACTIVATION,    // idx=3: shared activation buffer (1MB)
    KERN_ARG_ACT_WORKSPACE, // idx=4: activation workspace (10MB)
    KERN_ARG_WEIGHT1,       // idx=5: weight buffer #1 (1MB)
    KERN_ARG_WEIGHT2,       // idx=6: weight buffer #2 (1MB)
    KERN_ARG_KVCACHE,       // idx=7: KV cache (128MB)
    KERN_ARG_COUNT
} KernelArgIndex;

#define OPCODE_GEMM  3  // mm.xclbin GEMM kernel
#define OPCODE_ATTN  4  // attn.xclbin attention kernel (example)
#define OPCODE_LAYER 5  // layer.xclbin layer kernel
#define OPCODE_DEQUANT 2 // dequant.xclbin dequant kernel

// Block quantization constants for Q4NX format
#define GROUP_SIZE 32
#define MAX_GROUPS 32768

// Logging
#define LOG_INFO(fmt, ...)  fprintf(stderr, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) fprintf(stderr, "[DEBUG] " fmt "\n", ##__VA_ARGS__)
#define LOG_WARNING(fmt, ...) fprintf(stderr, "[WARN]  " fmt "\n", ##__VA_ARGS__)

#endif // NPU_INFER_COMMON_H
