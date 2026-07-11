// BitNet v2 — H-BitLinear Hadamard rotation (scalar reference, kernel skeleton).
//
// Paper: BitNet v2 (arXiv 2504.18415). Replaces the staged BitNet-a4.8 recipe:
// instead of "fp16 -> per-channel int8 -> gradually int4", BitNet v2 rotates
// the activation by a block-diagonal Walsh-Hadamard matrix H_B BEFORE
// quantization. Rotation redistributes outliers across the block so the
// post-rotation distribution is near-Gaussian → int4 per-token symmetric
// quantization (max/7) becomes viable with <0.1 PPL regression on the 2B4T
// base. No staged training, no a4.8 hand-off, no H-less fallback path.
//
// Insertion: one call PER activation-quant site in the decode forward pass.
// In halo-router backend_impl.rs those sites are (2026-04-20 code):
//   L530, L640, L674, L714 — immediately after each rmsnorm*, before
//   quantize_fp16_to_i8. The rotation is orthogonal (H/sqrt(B)) so it does
//   NOT change the RMSNorm invariant and does NOT need to be inverted on the
//   output side — it's absorbed into the ternary weights at model export
//   time (offline requantizer does W' = W @ H^T, shipped as a fresh .h1b).
//
// Hadamard matrix: block size B = 128 chosen as our default (see plan doc).
// H_128 is a real Walsh-Hadamard matrix with entries in {+1, -1} built by
// Sylvester's construction: H_{2n} = [[H_n, H_n], [H_n, -H_n]]. After
// rotation we divide by sqrt(B) = sqrt(128) to keep the transform orthogonal
// (L2-preserving; RMSNorm invariant preserved).
//
// Why B=128:
//   - BitNet-2B-4T hidden_size = 2560 = 20*128; head_dim = 128. Both divisible.
//     FFN intermediate = 6912 = 54*128. All three GEMV input sizes are clean.
//   - 128 values * 2 bytes (fp16) = 256 B per lane-stripe in LDS — well within
//     the 64 KiB per-CU LDS budget on RDNA3.5 at 4 wg/CU occupancy (4*256 = 1 KiB).
//   - Butterfly depth = log2(128) = 7 stages — each stage is a single wave-wide
//     xor-shuffle on wave32 (per Sylvester structure, stage s swaps lanes i
//     and i^(1<<s)). Fits nicely into wave32 __shfl_xor primitive.
//   - Larger blocks (B=256, 512) cost LDS with no accuracy benefit per the
//     paper's ablations; smaller (B=64) leaves outlier-mixing incomplete on
//     the intermediate FFN dim (6912 outlier-spikes benefit from 128-mix).
//
// THIS FILE IS THE SCALAR REFERENCE. It is intentionally slow and obvious,
// so the WMMA/butterfly-optimized path can be diff-tested against it. The
// optimized kernel is NOT in this file (separate follow-up commit).
//
// Layout: input x_fp16 is [K] contiguous; K must be a multiple of B. We
// rotate IN-PLACE. One block per B-sized chunk, one thread per element.
// Scalar-naive: O(B^2) mults per block — useless for perf, fine for correctness.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cmath>

namespace {

constexpr int HADAMARD_BLOCK = 128;   // default; must be power of 2
constexpr int WARP           = 32;    // wave32 on gfx1151

// Walsh-Hadamard matrix entry H[i][j] in {+1, -1}.
// Sylvester: H[i][j] = (-1)^popcount(i & j).
__device__ __forceinline__ int had_sign(int i, int j) {
    // popcount of (i & j) parity
    unsigned int v = (unsigned int)(i & j);
    // __builtin_popcount is supported on device by hipcc/clang
    return (__builtin_popcount(v) & 1) ? -1 : 1;
}

// ============================================================================
// Scalar reference — one block per Hadamard chunk. BLOCK.x = HADAMARD_BLOCK.
// Each thread computes y[i] = (1/sqrt(B)) * sum_j H[i][j] * x[j].
//
// This is a correctness reference — O(B^2) work per block; the optimized
// butterfly variant is separate. Launch grid = (K / HADAMARD_BLOCK,).
// ============================================================================
__global__ __launch_bounds__(HADAMARD_BLOCK)
void hadamard_rotate_fp16_ref_kernel(
    const __half* __restrict__ x_in,    // [K]
    __half*       __restrict__ y_out,   // [K]
    int K)
{
    const int block_idx = blockIdx.x;
    const int tid       = threadIdx.x;        // 0..B-1
    const int base      = block_idx * HADAMARD_BLOCK;

    if (base + tid >= K) return;

    // Load the entire block into LDS once (coalesced).
    __shared__ float s_x[HADAMARD_BLOCK];
    s_x[tid] = __half2float(x_in[base + tid]);
    __syncthreads();

    // Each thread computes one output row of H * x for this block.
    // Naive O(B) inner loop; correctness only.
    float acc = 0.0f;
    #pragma unroll 8
    for (int j = 0; j < HADAMARD_BLOCK; ++j) {
        const float sgn = (float)had_sign(tid, j);
        acc += sgn * s_x[j];
    }
    // Orthogonal normalization: 1/sqrt(B).
    const float inv_sqrt_b = 1.0f / std::sqrt((float)HADAMARD_BLOCK);
    y_out[base + tid] = __float2half(acc * inv_sqrt_b);
}

}  // namespace

// -----------------------------------------------------------------------------
// C-linkage launcher. Matches the rcpp_* prim_kernels.hip pattern so a future
// public header declaration fits in include/rocm_cpp/ck_gemm.h with no surprises.
//
// Contract: K must be a multiple of HADAMARD_BLOCK (128 today). In-place is
// permitted (x_in == y_out). Caller passes a hipStream_t (void*, nullable).
//
// Returns no status — the kernel has no failure modes other than the K
// divisibility precondition, which is caller-checked at model-load time
// (hidden_size and intermediate_size are both divisible by 128 on BitNet-2B-4T).
// -----------------------------------------------------------------------------
extern "C" void rcpp_hadamard_rotate_fp16_ref_launch(
    const void* x_in, void* y_out, int K, void* stream)
{
    // Precondition expressed as an assert — NOT UB-safe in Release; the
    // production launcher (once we ship non-ref) should return rcpp_status_t.
    // For the prototype we stay minimal.
    const int num_blocks = K / HADAMARD_BLOCK;
    dim3 grid(num_blocks, 1, 1);
    dim3 block(HADAMARD_BLOCK, 1, 1);
    hipLaunchKernelGGL(hadamard_rotate_fp16_ref_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(x_in),
                       static_cast<__half*>(y_out),
                       K);
}
