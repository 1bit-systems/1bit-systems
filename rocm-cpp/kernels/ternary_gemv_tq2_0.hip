// Ternary GEMV consuming ggml TQ2_0 weight format + Q8_1 activations.
//
// TQ2_0 block layout (ggml-common.h):
//   struct block_tq2_0 {
//       uint8_t  qs[64];   // 256 2-bit values, stride-32 interleave
//       ggml_half d;       // fp16 per-block scale
//   };                     // => 66 bytes per 256 weights (2.0625 bpw effective)
//
// Stride-32 decoding: within sub-block j ∈ {0, 32}, byte qs[j+m] holds
//   q_at_k[m + 0*32] = (qs[j+m] >> 0) & 3    (k offset = j*4 + m + 0)
//   q_at_k[m + 1*32] = (qs[j+m] >> 2) & 3    (k offset = j*4 + m + 32)
//   q_at_k[m + 2*32] = (qs[j+m] >> 4) & 3    (k offset = j*4 + m + 64)
//   q_at_k[m + 3*32] = (qs[j+m] >> 6) & 3    (k offset = j*4 + m + 96)
// signed value = q - 1  (so {0,1,2} → {-1,0,+1})
//
// Q8_1 activation block: 32 int8 values + fp16 scale + fp16 sum (ggml-common.h).
// 256 weights cover 8 consecutive q8_1 blocks on the x side.
//
// SCAFFOLD — kernel body pending. See docs/wiki/TQ2_0-HIP-Port-Plan.md.
// This compilation unit is intentionally a stub until the plan's landing
// step 1 is implemented + unit-tested. Dispatch glue in ggml-hip must not
// be enabled until this stub is replaced with a real kernel.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#define TQ2_0_BLOCK_SIZE 128
#define TQ2_0_WARP_SIZE  32
#define TQ2_0_ROWS_PER_BLOCK 8
#define TQ2_0_ROWS_PER_WARP  2
#define TQ2_0_QK_K 256
#define TQ2_0_QK8_1 32

// ggml block layouts — keep in sync with ggml-common.h. We intentionally
// redeclare here so this TU does not pull in ggml headers (Rule B: rocm-cpp
// is self-contained).
struct block_tq2_0 {
    uint8_t qs[TQ2_0_QK_K / 4];
    __half  d;
};

struct block_q8_1 {
    __half  d;
    __half  s;
    int8_t  qs[TQ2_0_QK8_1];
};

// TODO(task-9): implement.
// Per-row output = sum_{b=0..K/256} d_w_b * (sum_{k in block b} w_k * x_k)
// where w_k = (packed_bits - 1) ∈ {-1,0,+1} and x_k is the int8 activation
// scaled by its own q8_1 block d. Final scaled by d_x_block from q8_1.
//
// Expected perf target (gfx1151, K=4096, M=4096): ≥50 tok/s MMVQ decode.
__attribute__((amdgpu_flat_work_group_size(TQ2_0_BLOCK_SIZE, TQ2_0_BLOCK_SIZE)))
__global__ void ternary_gemv_tq2_0_q8_1_kernel(
    const block_tq2_0* __restrict__ vx,
    const block_q8_1*  __restrict__ vy,
    float*             __restrict__ dst,
    int ncols,
    int nrows)
{
    // Stub: zero the output so callers fail loud on unit test diff
    // rather than silently running the halo kernel by accident.
    const int tid = threadIdx.x;
    const int row_base = blockIdx.x * TQ2_0_ROWS_PER_BLOCK;
    if (tid < TQ2_0_ROWS_PER_BLOCK && row_base + tid < nrows) {
        dst[row_base + tid] = 0.0f;
    }
    (void)vx; (void)vy; (void)ncols;
}

extern "C" void ternary_gemv_tq2_0_q8_1_launch(
    const void* vx,
    const void* vy,
    void*       dst,
    int32_t     ncols,
    int32_t     nrows,
    void*       stream)
{
    dim3 grid((nrows + TQ2_0_ROWS_PER_BLOCK - 1) / TQ2_0_ROWS_PER_BLOCK, 1, 1);
    dim3 block(TQ2_0_BLOCK_SIZE, 1, 1);
    hipLaunchKernelGGL(ternary_gemv_tq2_0_q8_1_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const block_tq2_0*>(vx),
                       static_cast<const block_q8_1*>(vy),
                       static_cast<float*>(dst),
                       ncols, nrows);
}
