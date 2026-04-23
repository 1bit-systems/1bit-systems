// Ternary GEMV consuming ggml TQ2_0 weight format + Q8_1 activations.
//
// TQ2_0 block layout (ggml-common.h):
//   struct block_tq2_0 {
//       uint8_t  qs[64];   // 256 2-bit values, stride-32 interleave
//       ggml_half d;       // fp16 per-block scale
//   };                     // => 66 bytes per 256 weights (2.0625 bpw effective)
//
// Stride-32 decoding: within sub-block j ∈ {0, 32}, byte qs[j+m] holds
//   q_at_k[m + 0*32] = (qs[j+m] >> 0) & 3    (k offset = j*4 + m + 0*32)
//   q_at_k[m + 1*32] = (qs[j+m] >> 2) & 3    (k offset = j*4 + m + 1*32)
//   q_at_k[m + 2*32] = (qs[j+m] >> 4) & 3    (k offset = j*4 + m + 2*32)
//   q_at_k[m + 3*32] = (qs[j+m] >> 6) & 3    (k offset = j*4 + m + 3*32)
// signed value = q - 1  (so {0,1,2} → {-1,0,+1}; 3 is illegal)
//
// Q8_1 activation block: fp16 scale + fp16 sum + 32 int8 values.
// 256 weights cover 8 consecutive q8_1 blocks on the x side.
// Sub-block j=0  (k ∈ [0,128))   aligns with q8_1 blocks {0,1,2,3}
// Sub-block j=32 (k ∈ [128,256)) aligns with q8_1 blocks {4,5,6,7}
// Within sub-block j, slot l ∈ [0,4) aligns exactly with q8_1 block (j/8 + l),
// which covers k = j*4 + l*32 .. j*4 + l*32 + 31. Each slot therefore uses
// exactly one q8_1 d scale, so the per-slot int32 dot stays scale-uniform.
//
// The q8_1 row-sum `s` field is ignored: {-1,0,+1} is symmetric so the
// bias-correction path used by asymmetric quant formats (Qk_0/K_M) is
// unnecessary. Same story as the halo kernel.
//
// See docs/wiki/TQ2_0-HIP-Port-Plan.md for the full design discussion.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#define TQ2_0_BLOCK_SIZE      128
#define TQ2_0_WARP_SIZE       32
#define TQ2_0_ROWS_PER_BLOCK  8
#define TQ2_0_ROWS_PER_WARP   2
#define TQ2_0_QK_K            256
#define TQ2_0_QK8_1           32
#define TQ2_0_BLOCKS_PER_K    (TQ2_0_QK_K / TQ2_0_QK8_1)   // 8 q8_1 per weight block

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

// Shared inner body: returns fp32 accumulator for `my_row` across all K.
// Keeping this in a helper avoids copy-paste between fp32 and fp16 variants
// and lets the compiler make a single inlined specialization per call site.
static __device__ __forceinline__ float tq2_0_gemv_row_body(
    const block_tq2_0* __restrict__ vx,
    const block_q8_1*  __restrict__ vy,
    int  my_row,
    int  ncols,
    int  nrows,
    int  lane)
{
    if (my_row >= nrows) return 0.0f;

    const int nblocks_k = ncols / TQ2_0_QK_K;            // weight blocks along K
    const block_tq2_0* row_wb = vx + (size_t)my_row * nblocks_k;

    float row_acc = 0.0f;

    // Iterate 256-element weight blocks.
    for (int b = 0; b < nblocks_k; ++b) {
        const block_tq2_0& wb = row_wb[b];
        const float d_w = __half2float(wb.d);

        // Each of the 8 q8_1 blocks covering this weight block lives at
        // y_base + s, for s ∈ [0,8). Preload their fp16 scales as fp32
        // (ignore the row-sum `s` field — symmetric alphabet).
        const block_q8_1* y_base = vy + (size_t)b * TQ2_0_BLOCKS_PER_K;

        // Per-slot int32 accumulators. Each slot is one q8_1 block, and
        // slots must stay separate because their d_x scales differ.
        int32_t acc_slot[TQ2_0_BLOCKS_PER_K] = {0,0,0,0, 0,0,0,0};

        // Two sub-blocks j ∈ {0, 32} split the 64-byte qs[] in half.
        // Sub-block j covers the four q8_1 blocks (j/8 + {0,1,2,3}).
        #pragma unroll
        for (int jhalf = 0; jhalf < 2; ++jhalf) {
            const int j = jhalf * 32;                    // 0 or 32
            const int slot_base = jhalf * 4;             // 0 or 4

            // Lane `lane` ∈ [0,32) owns exactly one byte qs[j + lane].
            // That byte encodes 4 weights at k = j*4 + lane + slot*32,
            // one weight per slot ∈ [0,4).
            const uint32_t byte = wb.qs[j + lane];

            // For each slot within this half, compute lane-local product
            // and fold into per-slot int32 accumulator.
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int slot = slot_base + l;          // 0..7
                const int q    = (int)((byte >> (l * 2)) & 0x3);
                const int w_s  = q - 1;                  // {-1, 0, +1}

                // x_i8 lives in the slot'th q8_1 block, at local offset `lane`.
                const int8_t x = y_base[slot].qs[lane];

                acc_slot[slot] += w_s * (int)x;
            }
        }

        // Wave-reduce each slot (32-lane butterfly) and fold fp32 result
        // into row accumulator, scaling by d_w * d_x_slot.
        #pragma unroll
        for (int slot = 0; slot < TQ2_0_BLOCKS_PER_K; ++slot) {
            int32_t v = acc_slot[slot];
            #pragma unroll
            for (int offset = TQ2_0_WARP_SIZE / 2; offset > 0; offset >>= 1) {
                v += __shfl_xor(v, offset);
            }
            if (lane == 0) {
                const float d_x = __half2float(y_base[slot].d);
                row_acc += d_w * d_x * (float)v;
            }
        }
    }

    return row_acc;
}

__attribute__((amdgpu_flat_work_group_size(TQ2_0_BLOCK_SIZE, TQ2_0_BLOCK_SIZE)))
__global__ void ternary_gemv_tq2_0_q8_1_kernel(
    const block_tq2_0* __restrict__ vx,
    const block_q8_1*  __restrict__ vy,
    float*             __restrict__ dst,
    int ncols,
    int nrows)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (TQ2_0_WARP_SIZE - 1);
    const int wave = tid >> 5;                           // 0..3 (four waves)

    const int row_base      = blockIdx.x * TQ2_0_ROWS_PER_BLOCK;
    const int warp_row_base = row_base + wave * TQ2_0_ROWS_PER_WARP;

    #pragma unroll
    for (int r = 0; r < TQ2_0_ROWS_PER_WARP; ++r) {
        const int my_row = warp_row_base + r;
        const float y = tq2_0_gemv_row_body(vx, vy, my_row, ncols, nrows, lane);
        if (lane == 0 && my_row < nrows) {
            dst[my_row] = y;
        }
    }
}

__attribute__((amdgpu_flat_work_group_size(TQ2_0_BLOCK_SIZE, TQ2_0_BLOCK_SIZE)))
__global__ void ternary_gemv_tq2_0_q8_1_f16_kernel(
    const block_tq2_0* __restrict__ vx,
    const block_q8_1*  __restrict__ vy,
    __half*            __restrict__ dst,
    int ncols,
    int nrows)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (TQ2_0_WARP_SIZE - 1);
    const int wave = tid >> 5;

    const int row_base      = blockIdx.x * TQ2_0_ROWS_PER_BLOCK;
    const int warp_row_base = row_base + wave * TQ2_0_ROWS_PER_WARP;

    #pragma unroll
    for (int r = 0; r < TQ2_0_ROWS_PER_WARP; ++r) {
        const int my_row = warp_row_base + r;
        const float y = tq2_0_gemv_row_body(vx, vy, my_row, ncols, nrows, lane);
        if (lane == 0 && my_row < nrows) {
            dst[my_row] = __float2half(y);
        }
    }
}

extern "C" void ternary_gemv_tq2_0_q8_1_launch(
    const void* vx,
    const void* vy,
    void*       dst,
    int32_t     ncols,
    int32_t     nrows,
    void*       stream)
{
    // ncols must be a multiple of QK_K=256. Callers higher up the stack
    // guarantee this (ggml enforces row alignment for TQ2_0), but we
    // advertise the precondition loudly: any remainder K silently drops
    // weights. If a model with ncols % 256 != 0 appears, add a padded
    // tail-block path here.
    dim3 grid((nrows + TQ2_0_ROWS_PER_BLOCK - 1) / TQ2_0_ROWS_PER_BLOCK, 1, 1);
    dim3 block(TQ2_0_BLOCK_SIZE, 1, 1);
    hipLaunchKernelGGL(ternary_gemv_tq2_0_q8_1_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const block_tq2_0*>(vx),
                       static_cast<const block_q8_1*>(vy),
                       static_cast<float*>(dst),
                       ncols, nrows);
}

extern "C" void ternary_gemv_tq2_0_q8_1_f16_launch(
    const void* vx,
    const void* vy,
    void*       dst,
    int32_t     ncols,
    int32_t     nrows,
    void*       stream)
{
    dim3 grid((nrows + TQ2_0_ROWS_PER_BLOCK - 1) / TQ2_0_ROWS_PER_BLOCK, 1, 1);
    dim3 block(TQ2_0_BLOCK_SIZE, 1, 1);
    hipLaunchKernelGGL(ternary_gemv_tq2_0_q8_1_f16_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const block_tq2_0*>(vx),
                       static_cast<const block_q8_1*>(vy),
                       static_cast<__half*>(dst),
                       ncols, nrows);
}
