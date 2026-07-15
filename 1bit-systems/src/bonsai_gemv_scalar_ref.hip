// bonsai_gemv_scalar_ref.hip — 1-thread-per-row Bonsai GEMV reference.
//
// Purpose: known-correct baseline for differential testing of the fast
// Bonsai GEMV kernels. Covers both block formats:
//   Q1_0_g128  (18 B per g128 block, fp16 d first + 128 sign bits)
//   TQ2_0_g128 (34 B per g128 block, 32 B codes + fp16 d last)
//
// Accumulation order:
//   Ascending over blocks × ascending over weights-within-block (0..127),
//   FMA into a single fp32 register. Same fp16 clamp+cast epilogue as
//   the fast kernels. Bit-exactness expected on 1-weight-per-block shapes;
//   real shapes diverge by a handful of bf16 ULP because the fast kernel
//   performs a tree reduce across 128 lanes (non-associative fp32 add).
//
// Do NOT benchmark. 50–100× slower than the fast kernels by design.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>

#include "rocm_cpp/bonsai.h"

namespace {

constexpr int kWeightsPerB   = 128;
constexpr int kQ1BlockBytes  = 18;
constexpr int kTQ2BlockBytes = 34;
// TQ2_0_g128 on disk is [fp16 d : 2][qs : 32] — d FIRST, NOT d-last as
// the original Bonsai-Kernel-Spec.md claimed. Verified against raw
// PrismML GGUF: with d@0 no block emits the reserved 0b11 code; with
// d@32 1–2 per block do. See bonsai_tq2_gemv.hip header comment.
constexpr int kTQ2ScaleOfs   = 0;
constexpr int kTQ2CodesOfs   = 2;

__device__ __forceinline__
__half ref_load_fp16(const uint8_t* __restrict__ p) {
    uint16_t u;
    __builtin_memcpy(&u, p, sizeof(u));
    __half h;
    __builtin_memcpy(&h, &u, sizeof(h));
    return h;
}

}  // namespace

// ── Q1 scalar reference ─────────────────────────────────────────────────────

__attribute__((amdgpu_flat_work_group_size(64, 64)))
__global__ void bonsai_q1_gemv_scalar_ref_kernel(
    const uint8_t* __restrict__ packed,
    const __half*  __restrict__ act,
    __half*        __restrict__ out,
    int N_out, int K_in)
{
    const int row = blockIdx.x * 64 + threadIdx.x;
    if (row >= N_out) return;

    const int num_blocks   = K_in >> 7;
    const size_t row_bytes = (size_t)num_blocks * (size_t)kQ1BlockBytes;
    const uint8_t* __restrict__ wrow = packed + (size_t)row * row_bytes;

    float acc = 0.0f;
    for (int b = 0; b < num_blocks; ++b) {
        const uint8_t* __restrict__ blk = wrow + (size_t)b * kQ1BlockBytes;
        const float d = __half2float(ref_load_fp16(blk));
        const uint8_t* __restrict__ qs = blk + 2;
        for (int j = 0; j < kWeightsPerB; ++j) {
            const uint32_t bit = ((uint32_t)qs[j >> 3] >> (j & 7)) & 1u;
            const float sign = bit ? 1.0f : -1.0f;
            const float a = __half2float(act[b * kWeightsPerB + j]);
            acc = fmaf(sign * d, a, acc);
        }
    }

    constexpr float FP16_MAX = 65504.0f;
    if (acc >  FP16_MAX) acc =  FP16_MAX;
    if (acc < -FP16_MAX) acc = -FP16_MAX;
    out[row] = __float2half(acc);
}

// ── TQ2 scalar reference ────────────────────────────────────────────────────

__attribute__((amdgpu_flat_work_group_size(64, 64)))
__global__ void bonsai_tq2_gemv_scalar_ref_kernel(
    const uint8_t* __restrict__ packed,
    const __half*  __restrict__ act,
    __half*        __restrict__ out,
    int N_out, int K_in)
{
    const int row = blockIdx.x * 64 + threadIdx.x;
    if (row >= N_out) return;

    const int num_blocks   = K_in >> 7;
    const size_t row_bytes = (size_t)num_blocks * (size_t)kTQ2BlockBytes;
    const uint8_t* __restrict__ wrow = packed + (size_t)row * row_bytes;

    float acc = 0.0f;
    for (int b = 0; b < num_blocks; ++b) {
        const uint8_t* __restrict__ blk = wrow + (size_t)b * kTQ2BlockBytes;
        const float d = __half2float(ref_load_fp16(blk + kTQ2ScaleOfs));
        for (int j = 0; j < kWeightsPerB; ++j) {
            const uint32_t qbyte = (uint32_t)blk[kTQ2CodesOfs + (j >> 2)];
            const uint32_t code  = (qbyte >> ((j & 3) << 1)) & 3u;
            const float tern = (float)code - 1.0f;   // 00->-1, 01->0, 10->+1, 11->+2
            const float a = __half2float(act[b * kWeightsPerB + j]);
            acc = fmaf(tern * d, a, acc);
        }
    }

    constexpr float FP16_MAX = 65504.0f;
    if (acc >  FP16_MAX) acc =  FP16_MAX;
    if (acc < -FP16_MAX) acc = -FP16_MAX;
    out[row] = __float2half(acc);
}

// ── C API ───────────────────────────────────────────────────────────────────

extern "C" void bonsai_q1_gemv_scalar_ref_launch(
    const uint8_t*  packed_weights,
    const uint16_t* act_fp16,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream)
{
    if (N_out <= 0 || K_in <= 0)          return;
    if ((K_in & (kWeightsPerB - 1)) != 0) return;

    dim3 block(64, 1, 1);
    dim3 grid((N_out + 63) / 64, 1, 1);
    bonsai_q1_gemv_scalar_ref_kernel<<<grid, block, 0, (hipStream_t)stream>>>(
        packed_weights,
        reinterpret_cast<const __half*>(act_fp16),
        reinterpret_cast<__half*>(out_fp16),
        N_out, K_in);
}

extern "C" void bonsai_tq2_gemv_scalar_ref_launch(
    const uint8_t*  packed_weights,
    const uint16_t* act_fp16,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream)
{
    if (N_out <= 0 || K_in <= 0)          return;
    if ((K_in & (kWeightsPerB - 1)) != 0) return;

    dim3 block(64, 1, 1);
    dim3 grid((N_out + 63) / 64, 1, 1);
    bonsai_tq2_gemv_scalar_ref_kernel<<<grid, block, 0, (hipStream_t)stream>>>(
        packed_weights,
        reinterpret_cast<const __half*>(act_fp16),
        reinterpret_cast<__half*>(out_fp16),
        N_out, K_in);
}
