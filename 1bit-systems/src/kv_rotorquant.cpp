// Host-side wrappers for the rotorquant PlanarQuant-3 KV compression kernels.
// Keeps the GEMM TU (src/ck_gemm.cpp) clean and gives the kernels a single
// owning translation unit for the constant codebook definition.

#include "rocm_cpp/kv_rotorquant.h"

#include <cstdint>

// ---- Shared 3-bit Lloyd-Max codebook for N(0,1) ---------------------------
// Optimal centroids for a unit-variance Gaussian source at 8 levels. MSE
// ~= 0.03455 (Lloyd 1982 / Max 1960). Symmetric around 0, monotonic — the
// quantize_to_3bit binary-search in rotorquant_pack.hip depends on both
// properties.
//
// The kernel applies the 1/sqrt(head_dim) scale on top of this LUT inside
// each launch (so changing head_dim doesn't require a new codebook).
extern "C" const float RCPP_PQ3_LUT[8] = {
    -2.1519143f,
    -1.3439092f,
    -0.7560054f,
    -0.2451275f,
     0.2451275f,
     0.7560054f,
     1.3439092f,
     2.1519143f,
};

// splitmix32 — public-facing helper that lets callers log the per-layer seed.
// Exact arithmetic matches the kernel-side splitmix32 (verified by identical
// constants in kernels/rotorquant_pack.hip and src/kv_cache_attn_fd_rotor.hip).
extern "C" uint32_t rcpp_pq3_layer_seed(int layer_idx) {
    uint32_t x = (uint32_t)layer_idx * 0x9E3779B9u;
    x += 0x9E3779B9u;
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    x =  x ^ (x >> 16);
    return x;
}

// -- HIP-side launcher prototypes (defined in the .hip TUs) ------------------
extern "C" void rcpp_pq3_requantize_launch(
    const void* in_fp16, void* out_idx,
    int seq_len, int num_kv_heads, int head_dim,
    int layer_idx, void* stream);

extern "C" int rcpp_pq3_fd_decode_launch(
    const void* Q_dev, const void* K_idx_dev, const void* V_idx_dev,
    void* out_dev,
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, int layer_idx, float scale, void* stream);

// -- Public C API ------------------------------------------------------------
extern "C" void rcpp_kv_requantize_pq3(
    const void* K_fp16_dev, void* K_idx_dev,
    int seq_len, int num_kv_heads, int head_dim,
    int layer_idx, rcpp_pq3_stream_t stream)
{
    if (!K_fp16_dev || !K_idx_dev) return;
    if (seq_len <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;
    if ((head_dim & 7) != 0) return;     // head_dim must be multiple of 8
    if ((head_dim & 1) != 0) return;     // rotation requires even dim (redundant w/ %8)
    rcpp_pq3_requantize_launch(K_fp16_dev, K_idx_dev,
                               seq_len, num_kv_heads, head_dim,
                               layer_idx, stream);
}

extern "C" void rcpp_kv_requantize_pq3_v(
    const void* V_fp16_dev, void* V_idx_dev,
    int seq_len, int num_kv_heads, int head_dim,
    int layer_idx, rcpp_pq3_stream_t stream)
{
    // Identical code path. Separate symbol keeps future asymmetric V handling
    // (e.g. different codebook) binary-compatible.
    rcpp_kv_requantize_pq3(V_fp16_dev, V_idx_dev,
                           seq_len, num_kv_heads, head_dim,
                           layer_idx, stream);
}

extern "C" int rcpp_kv_cache_attn_decode_fd_pq3(
    const void* Q_dev, const void* K_idx_dev, const void* V_idx_dev,
    void* out_dev,
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, int layer_idx, float scale, rcpp_pq3_stream_t stream)
{
    return rcpp_pq3_fd_decode_launch(
        Q_dev, K_idx_dev, V_idx_dev, out_dev,
        num_q_heads, num_kv_heads, head_dim,
        seq_len, layer_idx, scale, stream);
}
