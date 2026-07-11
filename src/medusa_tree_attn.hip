// medusa_tree_attn.hip — tree-parallel split-KV Flash-Decoding (SKELETON).
//
// See include/rocm_cpp/medusa.h for the contract this file will eventually
// implement. THIS PASS lands only:
//
//   * The C-linkage entry point declared in medusa.h, returning
//     RCPP_NOT_IMPLEMENTED so callers can link + smoke-test the dispatch
//     plumbing without exercising any device code.
//   * A compile-clean __global__ tile placeholder so build coverage in CI
//     verifies the .hip TU lives inside the rocm_cpp HIP source set, but
//     the kernel never launches in this pass.
//
// What lands in the next chunk (~3-5 days)
// ----------------------------------------
//   * Pass-1 grid (tree_size, num_q_heads, num_kv_tiles): each block runs
//     the online-softmax recipe from src/kv_cache_attn_fd.hip but with
//     the per-candidate causal mask applied BEFORE the exponent (positions
//     not in candidate m's ancestor set get q·k = -FLT_MAX, mask is read
//     once into LDS at block entry — uniform across the wave for the
//     committed-prefix portion, divergent only in the speculative tail).
//   * Pass-2 grid (tree_size, num_q_heads): combine per-tile (m, l, partial)
//     triples with the same broadcast-reduce recipe used by the M=1 path.
//     The cached partials buffer is sized [tree_size × num_q_heads ×
//     num_kv_tiles × (head_dim + 2)] f32; allocate via the same lazy-grow
//     mutex pattern as kv_cache_attn_fd.hip.
//   * Bit-exact differential test against a scalar reference that walks
//     the same accumulation order — this is the gate for landing the
//     kernel into bitnet_decode's HALO_MEDUSA=1 path.
//
// gfx1151 / wave32 first-class. No hipBLAS, no Python.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#include "rocm_cpp/medusa.h"

namespace {

// Placeholder kernel — exists only so the compiler / CI verifies the TU is
// part of the rocm_cpp HIP source set. Never launched in this pass; the
// host launcher early-returns with RCPP_NOT_IMPLEMENTED before any
// hipLaunchKernelGGL would fire. Marked __launch_bounds__(128) to mirror
// the production kv_cache_attn_decode_split_kernel BLOCK=128 / 4-wave
// shape this kernel will eventually grow into.
__global__ __launch_bounds__(128)
void medusa_tree_attn_decode_fd_stub_kernel(
    const __half* __restrict__ /*Q*/,
    const __half* __restrict__ /*K*/,
    const __half* __restrict__ /*V*/,
    const uint8_t* __restrict__ /*tree_mask*/,
    __half*       __restrict__ /*out*/,
    int /*tree_size*/, int /*num_q_heads*/, int /*num_kv_heads*/,
    int /*head_dim*/, int /*seq_len*/, float /*scale*/)
{
    // Intentionally empty — see file header for the staged plan.
}

}  // namespace

extern "C"
rcpp_status_t
rcpp_medusa_tree_attn_decode_fd(const void* /*Q_dev*/,
                                const void* /*K_dev*/,
                                const void* /*V_dev*/,
                                const void* /*tree_mask_dev*/,
                                void*       /*out_dev*/,
                                int         /*tree_size*/,
                                int         /*num_q_heads*/,
                                int         /*num_kv_heads*/,
                                int         /*head_dim*/,
                                int         /*seq_len*/,
                                float       /*scale*/,
                                void*       /*stream*/)
{
    // Suppress unused-fn warnings on the placeholder kernel under -Wall/-Werror
    // configurations — taking the address keeps it ODR-used without ever
    // launching it.
    (void)&medusa_tree_attn_decode_fd_stub_kernel;
    return RCPP_NOT_IMPLEMENTED;
}
