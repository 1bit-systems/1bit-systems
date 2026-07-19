#pragma once
// include/rocm_cpp/oscar.h — OSCAR rotation-aware INT2 KV cache compression
//
// Per the OSCAR paper (arXiv:2605.17757), replaces the fixed Hadamard rotation
// in the KV cache quantizer with an attention-aware rotation derived from
// offline calibration.
//
// Pipeline:
//   1. Offline: oscar_calib computes per-layer rotation matrices (R_K, R_V)
//      from model weight statistics and exports to a binary file.
//   2. Load: oscar_load_rotations("oscar_rots.bin") fills an OscarRots struct.
//   3. Quantize: rcpp_kv_quantize_oscar applies R_K/R_V rotation, then INT2
//      quantization with per-token clipping.
//   4. Decode: rcpp_kv_cache_attn_decode_oscar dequantizes, inverse-rotates,
//      and computes attention.
//
// INT2 format: 2 bits per value, 4 levels: {-3/scale, -1/scale, 1/scale, 3/scale}
// Packed: 4 values per byte. Per-token scale stored as fp16.
//
// Bytes per token per layer:
//   K: num_kv_heads * head_dim / 4  +  num_kv_heads * sizeof(half)
//   V: same
// vs fp16: num_kv_heads * head_dim * 2
// Reduction: ~8x on the cache data (neglecting scales)

#ifndef ROCM_CPP_OSCAR_H
#define ROCM_CPP_OSCAR_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward-declare HIP stream as void* to avoid pulling HIP headers
typedef void* oscar_stream_t;

// ── Rotation matrix set (one per layer, for K and V) ───────────────────────
// Precomputed by offline calibration tool (oscar_calib).
typedef struct {
    int32_t head_dim;           // 128 for Qwen3
    int32_t n_layers;
    int32_t n_heads_q;
    int32_t n_heads_kv;
    // Rotation matrices: R_K[l] and R_V[l], each [head_dim x head_dim] row-major
    // Total: 2 * n_layers * head_dim * head_dim floats
    float* data;                // [R_K_0, R_V_0, R_K_1, R_V_1, ...]
} OscarRots;

// Load rotation matrices from binary file (exported by oscar_calib).
// Allocates internal storage. Call oscar_free_rotations() to release.
// Returns 0 on success, -1 on error.
int oscar_load_rotations(const char* path, OscarRots* rots);

// Free rotation matrices allocated by oscar_load_rotations.
void oscar_free_rotations(OscarRots* rots);

// Get R_K for a given layer (returns pointer to head_dim x head_dim float matrix)
static inline const float* oscar_Rk(const OscarRots* rots, int layer) {
    if (!rots || !rots->data) return nullptr;
    int stride = rots->head_dim * rots->head_dim;
    return rots->data + (size_t)layer * 2 * stride;
}

// Get R_V for a given layer
static inline const float* oscar_Rv(const OscarRots* rots, int layer) {
    if (!rots || !rots->data) return nullptr;
    int stride = rots->head_dim * rots->head_dim;
    return rots->data + (size_t)layer * 2 * stride + stride;
}

// ── INT2 quantization API ──────────────────────────────────────────────────
// Layout: K_idx [seq_len, num_kv_heads, (head_dim/4)] uint8
//         K_scale [seq_len, num_kv_heads] half (fp16 scale)
// Same for V.

// Quantize K: apply OSCAR rotation, then INT2 quantization with per-token clip.
// Requires precomputed rotation matrices in rots.
void rcpp_kv_quantize_oscar_k(
    const void* K_fp16_dev,          // [seq_len, num_kv_heads, head_dim] fp16
    void* K_idx_dev,                 // [seq_len, num_kv_heads, head_dim/4] uint8
    void* K_scale_dev,               // [seq_len, num_kv_heads] half
    int seq_len, int num_kv_heads, int head_dim, int layer_idx,
    const OscarRots* rots, oscar_stream_t stream);

// Same for V.
void rcpp_kv_quantize_oscar_v(
    const void* V_fp16_dev,
    void* V_idx_dev,
    void* V_scale_dev,
    int seq_len, int num_kv_heads, int head_dim, int layer_idx,
    const OscarRots* rots, oscar_stream_t stream);

// Flash-decoding attention with inline INT2 dequant + inverse OSCAR rotation.
// Drop-in replacement for rcpp_kv_cache_attn_decode_fd.
int rcpp_kv_cache_attn_decode_oscar(
    const void* Q_dev,               // [num_q_heads, head_dim] fp16
    const void* K_idx_dev,           // [seq_len, num_kv_heads, head_dim/4] uint8
    const void* K_scale_dev,         // [seq_len, num_kv_heads] half
    const void* V_idx_dev,           // [seq_len, num_kv_heads, head_dim/4] uint8
    const void* V_scale_dev,         // [seq_len, num_kv_heads] half
    void* out_dev,                   // [num_q_heads, head_dim] fp16
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, int layer_idx,
    const OscarRots* rots, float scale, oscar_stream_t stream);

#ifdef __cplusplus
}
#endif

#endif // ROCM_CPP_OSCAR_H
