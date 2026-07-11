// BitNet v2 — H-BitLinear Hadamard rotation (wave32 butterfly kernel).
//
// Companion to src/hadamard_rotate.hip (O(B^2) scalar reference / oracle).
// This file is the O(B log B) production candidate: 7 butterfly stages on a
// 128-element block, one workgroup per block, one lane per element.
//
// Sylvester decomposition: H_{2n} = [[H_n, H_n], [H_n, -H_n]]. At stage s
// we pair lanes i and i^(1<<s); the "low" half of each pair (bit s of i = 0)
// takes a + b, the "high" half takes a - b. After 7 stages we have applied
// H_128 exactly (up to the 1/sqrt(B) scale folded into the final store).
//
// Wave topology on gfx1151 (wave32, 128-thread workgroup = 4 waves):
//   stages 0..4 (strides 1,2,4,8,16): intra-wave, __shfl_xor (no LDS).
//   stages 5..6 (strides 32,64):      cross-wave, LDS exchange + barrier.
// LDS footprint: 128 * 4 B = 512 B per block (plan doc matches).
//
// Launch contract: identical to the reference kernel — grid = (K/128,),
// block = (128,). In-place permitted. K must be a multiple of 128.
//
// Verify (caller responsibility, not wired into CMakeLists.txt yet):
//   /opt/rocm/lib/llvm/bin/clang++ -x hip --offload-arch=gfx1151 -O3 \
//       -std=c++20 -c kernels/hadamard_rotate_butterfly.hip \
//       -o /tmp/hadamard_rotate_butterfly.o
//
// Target: sub-microsecond per 128-element block on gfx1151. Paper budget for
// 112 rotates/token at ~1-2 us each is ~150-200 us/token (≤2% of decode).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cmath>

namespace {

constexpr int HADAMARD_BLOCK = 128;   // B; power of 2; log2 = 7 stages
constexpr int WAVE           = 32;    // wave32 on gfx1151 (RDNA3.5)

// Butterfly partner combine: `self` plus or minus `partner`, chosen by
// which half of the pair we are at stage `s`. Low-half (bit s == 0) → add,
// high-half → subtract. Compiler should lower this to a single v_cndmask.
__device__ __forceinline__ float combine(float self, float partner, bool high) {
    return high ? (partner - self) : (self + partner);
}

// ============================================================================
// Butterfly kernel. One workgroup per 128-element block; one lane per element.
// Stages 0..4 (intra-wave) are unrolled with __shfl_xor. Stages 5..6
// (cross-wave) go via LDS with a single __syncthreads each.
// ============================================================================
__global__ __launch_bounds__(HADAMARD_BLOCK)
void hadamard_rotate_fp16_butterfly_kernel(
    const __half* __restrict__ x_in,    // [K]
    __half*       __restrict__ y_out,   // [K]
    int K)
{
    const int block_idx = blockIdx.x;
    const int tid       = threadIdx.x;        // 0..127
    const int base      = block_idx * HADAMARD_BLOCK;
    const int gidx      = base + tid;

    if (gidx >= K) return;

    // Coalesced fp16 load → fp32 working precision. RMSNorm invariant is
    // preserved because H/sqrt(B) is unitary; accumulator error budget fits
    // in fp32 comfortably (7 stages of ±1 adds on values in roughly [-8,8]).
    float v = __half2float(x_in[gidx]);

    // --- Intra-wave stages (strides 1, 2, 4, 8, 16) ---------------------------
    // __shfl_xor does a butterfly permutation within the wave. For wave32, a
    // stride of 1<<s with s in [0..4] stays inside a single wave. "high" is
    // bit s of the LOCAL lane id (tid & 31).
    const int lane = tid & (WAVE - 1);

    #pragma unroll
    for (int s = 0; s < 5; ++s) {
        const int stride = 1 << s;
        const float partner = __shfl_xor(v, stride, WAVE);
        const bool high     = (lane & stride) != 0;
        v = combine(v, partner, high);
    }

    // --- Cross-wave stages (strides 32, 64) -----------------------------------
    // Exchange via LDS. 128 floats = 512 B; at 4 workgroups per CU that's
    // 2 KiB LDS per CU, well under the 64 KiB budget (plan doc §2).
    __shared__ float s_buf[HADAMARD_BLOCK];

    // Stage 5: stride 32. Partner lane = tid ^ 32; "high" = bit5 of tid.
    s_buf[tid] = v;
    __syncthreads();
    {
        const float partner = s_buf[tid ^ 32];
        const bool  high    = (tid & 32) != 0;
        v = combine(v, partner, high);
    }
    __syncthreads();

    // Stage 6: stride 64. Partner lane = tid ^ 64; "high" = bit6 of tid.
    s_buf[tid] = v;
    __syncthreads();
    {
        const float partner = s_buf[tid ^ 64];
        const bool  high    = (tid & 64) != 0;
        v = combine(v, partner, high);
    }
    // No trailing __syncthreads needed: s_buf is not read again this launch.

    // Fold orthogonal normalization into the single final store.
    constexpr float inv_sqrt_b = 0.088388347648318440550f;  // 1 / sqrt(128)
    y_out[gidx] = __float2half(v * inv_sqrt_b);
}

}  // namespace

// -----------------------------------------------------------------------------
// C-linkage launcher. Same signature shape as rcpp_hadamard_rotate_fp16_ref_launch
// so the public-header promotion later is a one-line swap. Not yet in
// include/rocm_cpp/ck_gemm.h — callers currently prototype at their site.
//
// Preconditions (caller-checked; kernel does not branch on them):
//   - K % HADAMARD_BLOCK == 0
//   - x_in, y_out are device pointers; aliased in-place is permitted
//   - stream is a hipStream_t (or nullptr for the default stream)
// -----------------------------------------------------------------------------
extern "C" void rcpp_hadamard_rotate_fp16_butterfly_launch(
    const void* x_in, void* y_out, int K, void* stream)
{
    const int num_blocks = K / HADAMARD_BLOCK;
    dim3 grid(num_blocks, 1, 1);
    dim3 block(HADAMARD_BLOCK, 1, 1);
    hipLaunchKernelGGL(hadamard_rotate_fp16_butterfly_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(x_in),
                       static_cast<__half*>(y_out),
                       K);
}

// Status-returning wrapper — validates preconditions before launching.
extern "C" int rcpp_hadamard_rotate_fp16(
    const void* x_in, void* y_out, int K, void* stream)
{
    if (!x_in || !y_out || K <= 0 || (K % HADAMARD_BLOCK) != 0)
        return 1;  // RCPP_INVALID_ARG
    rcpp_hadamard_rotate_fp16_butterfly_launch(x_in, y_out, K, stream);
    return 0;  // RCPP_OK
}
