// rotorquant PlanarQuant-3 (PQ3) KV-cache compression — public C API.
//
// Scheme:
//   1. Apply a layer-seeded deterministic 2D-Givens rotation to K (and V) along
//      the head_dim axis (dim pairs (2i, 2i+1)). The rotation whitens the
//      intra-head distribution so a *single shared* 3-bit Lloyd-Max codebook
//      for N(0,1) (scaled by 1/sqrt(head_dim)) is near-optimal across all heads
//      and positions.
//   2. Quantize each post-rotation scalar to a 3-bit index via binary search
//      over the shared 8-entry LUT.
//   3. Pack 8 indices into 3 bytes (24 bits). Row stride (one (pos, kv_head)
//      row) = (head_dim * 3) / 8 bytes. head_dim must be a multiple of 8
//      (BitNet-2B-4T's HD=128 satisfies this).
//
// No per-token fp16 scale is stored — the rotation makes the distribution
// stationary, and the codebook bakes in the 1/sqrt(head_dim) scale.
//
// Bytes per token per layer: 2 * num_kv_heads * head_dim * 3/8  vs fp16's
//   2 * num_kv_heads * head_dim * 2  =  5.33x reduction.

#ifndef ROCM_CPP_KV_ROTORQUANT_H
#define ROCM_CPP_KV_ROTORQUANT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward-declare hipStream_t as a void* to avoid pulling a HIP header here.
// The .cpp/.hip side casts back to hipStream_t.
typedef void* rcpp_pq3_stream_t;

// Shared 3-bit Lloyd-Max codebook for N(0,1). 8 entries, MSE ~= 0.03455 for
// a unit-variance Gaussian source (Max 1960, Lloyd 1982 optimal centroids).
// The kernel applies a final 1/sqrt(head_dim) scale when dequantizing.
//
// Exposed via extern so the kernel TU and the host TU see the *same* values
// at link time (no duplicate definitions).
extern const float RCPP_PQ3_LUT[8];

// Host-visible copy of the per-layer Givens rotation angle-seed. The kernel
// derives all head_dim/2 rotation angles from (layer_idx, dim_pair_idx) via a
// cheap splitmix32; this helper reports the constant seed basis so callers
// can log / reproduce.
uint32_t rcpp_pq3_layer_seed(int layer_idx);

// ---- Requantize: rotate K (or V) fp16 -> pq3 packed bytes ------------------
// K_fp16_dev layout  : FP16 [seq_len, num_kv_heads, head_dim]
// K_idx_dev  layout  : uint8 [seq_len, num_kv_heads, head_dim * 3 / 8]
//                      (row-contiguous over the packed bytes; each row covers
//                       one (pos, kv_head) pair)
// layer_idx          : used to seed the Givens rotation. Required for the
//                      inverse rotation inside the attention kernel.
// head_dim must be a multiple of 8.
void rcpp_kv_requantize_pq3(const void* K_fp16_dev, void* K_idx_dev,
                            int seq_len, int num_kv_heads, int head_dim,
                            int layer_idx, rcpp_pq3_stream_t stream);

// Same shape, identical code path (K and V share the rotation + codebook).
// Kept as a distinct symbol so future extensions (e.g. asymmetric V handling)
// can diverge without breaking callers.
void rcpp_kv_requantize_pq3_v(const void* V_fp16_dev, void* V_idx_dev,
                              int seq_len, int num_kv_heads, int head_dim,
                              int layer_idx, rcpp_pq3_stream_t stream);

// ---- Split-KV Flash-Decoding attention with inline PQ3 dequant --------------
// Drop-in replacement for rcpp_kv_cache_attn_decode_fd, except K/V are the
// packed-index buffers produced by rcpp_kv_requantize_pq3 / _v above.
//
// Layouts:
//   Q_dev           : FP16 [num_q_heads, head_dim]
//   K_idx_dev       : uint8 [seq_len, num_kv_heads, head_dim * 3 / 8]
//   V_idx_dev       : uint8 [seq_len, num_kv_heads, head_dim * 3 / 8]
//   out_dev         : FP16 [num_q_heads, head_dim]
//
// layer_idx must match the value used to write the caches (or the inverse
// rotation is wrong and output is garbage — deterministic, not UB).
//
// Returns 0 on success, nonzero on invalid-arg / HIP failure.
int rcpp_kv_cache_attn_decode_fd_pq3(const void* Q_dev,
                                     const void* K_idx_dev, const void* V_idx_dev,
                                     void* out_dev,
                                     int num_q_heads, int num_kv_heads, int head_dim,
                                     int seq_len, int layer_idx, float scale,
                                     rcpp_pq3_stream_t stream);

#ifdef __cplusplus
}
#endif
#endif  // ROCM_CPP_KV_ROTORQUANT_H
