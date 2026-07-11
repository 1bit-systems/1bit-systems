// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
// Sherry — see LICENSE-SHERRY.md and SHERRY-FILES.txt at the repo root.
// Commercial use requires a separate license.
//
// sherry_gemv_scalar_ref.hip — 1 thread per row Sherry reference.
//
// Purpose: a known-correct implementation the test harness can diff the fast
// kernel against. One HIP thread computes one output row via a scalar K-loop.
// Same fp32 accumulation order and final fp16 clamp+cast as the fast kernel,
// so bit-exactness is expected on shape-regular inputs; test tolerates ≤ 1
// bf16 ULP to stay robust against fp16-cast rounding corner cases.
//
// Do NOT benchmark. 50–100x slower than the fast kernel by design.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>

#include "rocm_cpp/sherry.h"

namespace {

__device__ __forceinline__
uint32_t ref_fetch_code_5b(const uint8_t* __restrict__ row, int g) {
    const int bit_ofs  = g * 5;
    const int byte_idx = bit_ofs >> 3;
    const int shift    = bit_ofs & 0x7;
    uint16_t two;
    __builtin_memcpy(&two, row + byte_idx, sizeof(two));
    return ((uint32_t)two >> shift) & 0x1Fu;
}

}  // namespace

__attribute__((amdgpu_flat_work_group_size(64, 64)))
__global__ void sherry_ternary_gemv_scalar_ref_kernel(
    const uint8_t* __restrict__ packed,
    const __half*  __restrict__ act,
    __half*        __restrict__ out,
    int N_out, int K_in)
{
    const int row = blockIdx.x * 64 + threadIdx.x;
    if (row >= N_out) return;

    const int row_bytes = (K_in * 5) / 32;
    const uint8_t* __restrict__ wrow = packed + (size_t)row * (size_t)row_bytes;

    float acc = 0.0f;
    const int groups = K_in >> 2;
    for (int g = 0; g < groups; ++g) {
        const uint32_t code  = ref_fetch_code_5b(wrow, g);
        const uint32_t zp    = (code >> 3) & 0x3u;
        const uint32_t signs =  code       & 0x7u;

        const int a_base = g * 4;
        const float a0 = __half2float(act[a_base + 0]);
        const float a1 = __half2float(act[a_base + 1]);
        const float a2 = __half2float(act[a_base + 2]);
        const float a3 = __half2float(act[a_base + 3]);

        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        uint32_t si = 0;
        if (zp == 0u) { s0 = 0.0f; } else { s0 = ((signs >> si) & 1u) ? 1.0f : -1.0f; ++si; }
        if (zp == 1u) { s1 = 0.0f; } else { s1 = ((signs >> si) & 1u) ? 1.0f : -1.0f; ++si; }
        if (zp == 2u) { s2 = 0.0f; } else { s2 = ((signs >> si) & 1u) ? 1.0f : -1.0f; ++si; }
        if (zp == 3u) { s3 = 0.0f; } else { s3 = ((signs >> si) & 1u) ? 1.0f : -1.0f; }

        acc = fmaf(a0, s0, acc);
        acc = fmaf(a1, s1, acc);
        acc = fmaf(a2, s2, acc);
        acc = fmaf(a3, s3, acc);
    }

    constexpr float FP16_MAX = 65504.0f;
    if (acc >  FP16_MAX) acc =  FP16_MAX;
    if (acc < -FP16_MAX) acc = -FP16_MAX;
    out[row] = __float2half(acc);
}

extern "C" void sherry_ternary_gemv_scalar_ref_launch(
    const uint8_t*  packed_weights,
    const uint16_t* act_fp16,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream)
{
    if (N_out <= 0 || K_in <= 0)            return;
    if ((K_in & 0x3) != 0)                  return;
    if ((K_in & 0x1F) != 0)                 return;

    dim3 block(64, 1, 1);
    dim3 grid((N_out + 63) / 64, 1, 1);
    sherry_ternary_gemv_scalar_ref_kernel<<<grid, block, 0, (hipStream_t)stream>>>(
        packed_weights,
        reinterpret_cast<const __half*>(act_fp16),
        reinterpret_cast<__half*>(out_fp16),
        N_out, K_in);
}
