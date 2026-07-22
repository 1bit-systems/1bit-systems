#pragma once
#ifndef ROCM_CPP_BITNET_MODEL_H
#define ROCM_CPP_BITNET_MODEL_H

#ifndef ROCM_CPP_NO_SHERRY
#include "rocm_cpp/ck_gemm.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define H1B_FLAG_HADAMARD_ROTATED 0x1u
#define H1B_FLAG_SHERRY_FP16      0x2u
#define H1B_FLAG_BONSAI_Q1        0x4u
#define H1B_FLAG_BONSAI_TQ2       0x8u
#define H1B_FLAG_BLOCK_SCALED     0x10u

typedef enum {
    RCPP_WEIGHT_FORMAT_HALO_V2    = 0,
    RCPP_WEIGHT_FORMAT_SHERRY_I8  = 1,
    RCPP_WEIGHT_FORMAT_TQ1        = 2,
    RCPP_WEIGHT_FORMAT_SHERRY_FP16 = 3,
    RCPP_WEIGHT_FORMAT_BONSAI_Q1  = 4,
    RCPP_WEIGHT_FORMAT_BONSAI_TQ2 = 5,
    RCPP_WEIGHT_FORMAT_WMMA_I8    = 6,
    RCPP_WEIGHT_FORMAT_BLOCK_SCALED_TERNARY = 7,
} rcpp_weight_format_t;

typedef enum {
    RCPP_ARCH_BITNET  = 0,
    RCPP_ARCH_QWEN3   = 1,
    RCPP_ARCH_LLAMA   = 2,
    RCPP_ARCH_MISTRAL = 3,
    RCPP_ARCH_QWEN2   = 4,
    RCPP_ARCH_GEMMA   = 5,
    RCPP_ARCH_PHI     = 6,
    RCPP_ARCH_ZAMBA2  = 7,
    RCPP_ARCH_ZAMBA   = 8,   // Zamba-7B-v1 (Mamba1 + shared attn)
    RCPP_ARCH_MAMBA   = 9,   // BlackMamba (Mamba1 + MoE)
    RCPP_ARCH_LAGUNA  = 10,
    RCPP_ARCH_FALCON  = 11,  // Falcon (tiiuae) — parallel attn+ffn, MQA
    RCPP_ARCH_OLMO    = 12,  // OLMo (AI2) — LayerNorm, no RoPE  // Poolside Laguna (sigmoid-routed MoE, hybrid SWA/global attn)
} rcpp_arch_t;

#include <string.h>

static inline rcpp_arch_t rcpp_arch_from_string(const char* s) {
    if (!s) return RCPP_ARCH_BITNET;
    if (strcmp(s, "qwen3")   == 0) return RCPP_ARCH_QWEN3;
    if (strcmp(s, "llama")   == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "mistral") == 0) return RCPP_ARCH_MISTRAL;
    if (strcmp(s, "qwen2")   == 0) return RCPP_ARCH_QWEN2;
    if (strcmp(s, "gemma")   == 0) return RCPP_ARCH_GEMMA;
    if (strcmp(s, "phi")     == 0) return RCPP_ARCH_PHI;
    if (strcmp(s, "zamba2")  == 0) return RCPP_ARCH_ZAMBA2;
    if (strcmp(s, "zamba")   == 0) return RCPP_ARCH_ZAMBA;
    if (strcmp(s, "mamba")   == 0) return RCPP_ARCH_MAMBA;
    if (strcmp(s, "laguna")  == 0) return RCPP_ARCH_LAGUNA;
    if (strcmp(s, "falcon")  == 0) return RCPP_ARCH_FALCON;
    if (strcmp(s, "falcon3") == 0) return RCPP_ARCH_FALCON;
    if (strcmp(s, "olmo")    == 0) return RCPP_ARCH_OLMO;
    if (strcmp(s, "olmo2")   == 0) return RCPP_ARCH_OLMO;
    if (strcmp(s, "olmoe")   == 0) return RCPP_ARCH_OLMO;
    return RCPP_ARCH_BITNET;
}

typedef struct {
    void* input_norm_dev;
    void* post_attn_norm_dev;
    void* attn_sub_norm_dev;
    void* ffn_sub_norm_dev;
    void* attn_q_norm_dev;
    void* attn_k_norm_dev;

    // Ternary linear layers — halo-encoded uint8 packed + per-row FP32 scales
    void* q_packed_dev;     float* q_scales_dev;
    void* k_packed_dev;     float* k_scales_dev;
    void* v_packed_dev;     float* v_scales_dev;
    void* o_packed_dev;     float* o_scales_dev;
    void* gate_packed_dev;  float* gate_scales_dev;
    void* up_packed_dev;    float* up_scales_dev;
    void* down_packed_dev;  float* down_scales_dev;

    // WMMA_I8 path: Hadamard-rotated INT8 weights + per-row fp32 scales
    void* q_i8_dev;          float* q_i8_scales_dev;
    void* k_i8_dev;          float* k_i8_scales_dev;
    void* v_i8_dev;          float* v_i8_scales_dev;
    void* o_i8_dev;          float* o_i8_scales_dev;
    void* gate_i8_dev;       float* gate_i8_scales_dev;
    void* up_i8_dev;         float* up_i8_scales_dev;
    void* down_i8_dev;       float* down_i8_scales_dev;

    // Block-Scaled Ternary path: block-scaled ternary packed (5 bytes/block)
    // See include/block_scaled_ternary.h for format
    void* bst_q_packed_dev;     void* bst_q_scales_dev;
    void* bst_k_packed_dev;     void* bst_k_scales_dev;
    void* bst_v_packed_dev;     void* bst_v_scales_dev;
    void* bst_o_packed_dev;     void* bst_o_scales_dev;
    void* bst_gate_packed_dev;  void* bst_gate_scales_dev;
    void* bst_up_packed_dev;    void* bst_up_scales_dev;
    void* bst_down_packed_dev;  void* bst_down_scales_dev;
} rcpp_bitnet_layer_t;

typedef struct {
    int hidden_size;
    int intermediate_size;
    int num_layers;
    int num_heads;
    int num_kv_heads;
    int vocab_size;
    int max_seq_len;
    int tie_embeddings;
    float rope_theta;
    float rms_norm_eps;
    int format_version;
    unsigned int flags;
    rcpp_weight_format_t weight_format;
    int is_qwen3;
    rcpp_arch_t arch;
    void* embedding_dev;
    void* embedding_packed_dev;
    void* final_norm_weight_dev;
    rcpp_bitnet_layer_t* layers;
} rcpp_bitnet_model_t;

rcpp_status_t rcpp_bitnet_load_h1b(const char* path, rcpp_bitnet_model_t* out_model);
rcpp_status_t rcpp_bitnet_load_gguf(const char* path, rcpp_bitnet_model_t* out_model);
rcpp_status_t rcpp_bitnet_load_onnx(const char* path, rcpp_bitnet_model_t* out_model);
void rcpp_bitnet_free(rcpp_bitnet_model_t* model);

#ifdef __cplusplus
}
#endif
#endif
