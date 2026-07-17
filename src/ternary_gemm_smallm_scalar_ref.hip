// ternary_gemm_smallm_scalar_ref.hip — Scalar reference for Medusa tree-verify.
//
// Purpose
// -------
// Naive numerical reference that matches the tiled kernel's ABI. One HIP
// thread per (m, n) output element. Scalar K-loop. No LDS, no dot4, no
// WMMA. 50-100x slower than the tiled kernel by design — this exists only
// so the test harness can diff the tiled output against a known-correct
// implementation compiled with the same floating-point semantics.
//
// Signature matches onebit_ternary_gemm_smallm_launch (see
// ternary_gemm_smallm.hip):
//
//   x_i8           : [M, K] int8 activations, per-tensor x_scale.
//   w_packed       : [K, N] ternary codes, 2-bit packed, 16 vals per u32.
//                    Layout: w_packed[(k/16) * N + n], code at bits
//                    2*(k%16)..2*(k%16)+1. Codes: 0→-1, 1→0, 2→+1, 3→0.
//   w_row_scales   : [N] one fp32 scale per output column.
//   y_out          : [M, N] bf16 output.
//
// Ternary accumulation runs in int32 (bit-accurate across sdot4 and this
// reference at the integer level); the final epilogue cast to bf16 is the
// only place where rounding can differ. The test harness tolerates ≤ 1
// bf16 ULP, never bit-exact.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>

namespace {

// Halo-1bit encoding: code 0 -> -1, code 1 -> 0, code 2 -> +1, code 3 -> 0.
// Same convention as kernels/ternary_gemv_phase5_halo.hip (line 40).
static __device__ __forceinline__ int unpack_ternary_code(uint32_t code) {
    return (int)(code == 2u) - (int)(code == 0u);
}

} // namespace

extern "C" {

// One thread per (m, n). Grid = (ceil(N/64), M). Block = 64 threads.
// Intentionally naive; do NOT benchmark this as a kernel of record.
__global__
__attribute__((amdgpu_flat_work_group_size(64, 64)))
void onebit_ternary_gemm_smallm_scalar_kernel(
    const int8_t*   __restrict__ x_i8,
    float                        x_scale,
    const uint32_t* __restrict__ w_packed,
    const float*    __restrict__ w_row_scales,
    hip_bfloat16* __restrict__ y_out,
    int M, int N, int K)
{
    const int m = blockIdx.y;
    const int n = blockIdx.x * 64 + threadIdx.x;
    if (m >= M || n >= N) return;

    // Kernel precondition (shared with the tiled path): K is a multiple of 16.
    // The packer in the test harness (pack_ternary) relies on this too.
    const int k_blocks = K / 16;
    int32_t acc = 0;

    // Scalar K-loop. For each 16-k block at this n column, decode 16
    // ternary codes and multiply-accumulate with the x_i8 activations.
    // Inner loop has no runtime-dependent break, so -Wpass-failed=transform
    // is silent and the compiler fully unrolls.
    for (int kb = 0; kb < k_blocks; ++kb) {
        const uint32_t word = w_packed[(size_t)kb * (size_t)N + (size_t)n];
        const int k_base = kb * 16;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int k = k_base + j;
            const uint32_t code = (word >> (j * 2)) & 0x3u;
            const int s = unpack_ternary_code(code);
            if (s != 0) {
                const int xv = (int)x_i8[(size_t)m * (size_t)K + (size_t)k];
                acc += (s > 0) ? xv : -xv;
            }
        }
    }

    const float fp = (float)acc * x_scale * w_row_scales[n];
    y_out[(size_t)m * (size_t)N + (size_t)n] = hip_bfloat16(fp);
}

// Host wrapper. Stream-aware; no hipDeviceSynchronize.
void onebit_ternary_gemm_smallm_scalar_launch(
    const int8_t*   x_i8,
    float           x_scale,
    const uint32_t* w_packed,
    const float*    w_row_scales,
    hip_bfloat16* y_out,
    int M, int N, int K,
    hipStream_t stream)
{
    if (M <= 0 || N <= 0 || K <= 0 || M > 16) return;
    if (K % 16 != 0) return;  // packer + unrolled inner loop require this
    dim3 block(64, 1, 1);
    dim3 grid((N + 63) / 64, M, 1);
    onebit_ternary_gemm_smallm_scalar_kernel<<<grid, block, 0, stream>>>(
        x_i8, x_scale, w_packed, w_row_scales, y_out, M, N, K);
}

} // extern "C"
