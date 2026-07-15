// WMMA-based ternary GEMV for gfx1151 decode path — v3 LDS-tiled.
//
// Key insight from benchmarking: v2 was memory-latency bound because each K-tile
// iteration did an uncoalesced global load (each lane loads from a different row).
// v3 prefetches WEIGHT_TILES tiles of weights into LDS at once, then processes
// them from LDS — no global memory access during the inner WMMA loop.
//
// WMMA fragment layout (gfx11 i8):
//   A frag: lane i holds row (i%16) of A (16 i8 values). Lanes 16..31 duplicate 0..15.
//   B frag: lane i holds COLUMN (i%16) of B (16 i8 values). Lanes 16..31 duplicate 0..15.
//   C output: lane i, slot r → C[2*r + (i>=16), i%16]
//
// For GEMV, A is broadcast (identical activation for all lanes), so all rows of C
// are identical. Each lane i (0..15) holds the dot product for weight row i, and
// all 8 slots in that lane are the same value.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#define WARP_SIZE 32

using int8x16_t = __attribute__((ext_vector_type(16))) int8_t;
using int32x8_t = __attribute__((ext_vector_type(8))) int32_t;

// Number of K-tiles to batch into one LDS prefetch.
// Each tile = 16 K values. With WEIGHT_TILES_PER_BATCH=32, we process 512 K values
// per LDS load. Each wave stores 16 rows × 32 tiles × 4 bytes = 2KB in LDS.
// With 2 waves and double buffering = 8KB total — well within 64KB LDS on gfx1151.
#define WEIGHT_TILES_PER_BATCH 32

// ─── Packed ternary unpack ──────────────────────────────────────────────────
static __device__ __forceinline__ int8x16_t unpack_ternary_4(uint32_t word) {
    int8x16_t result;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t bits = (word >> (i * 2)) & 0x3u;
        result[i] = (int8_t)((int)(bits == 2) - (int)(bits == 0));
    }
    return result;
}

// ─── Main kernel: 32 rows per block (2 waves), LDS-tiled ────────────────────
__attribute__((amdgpu_flat_work_group_size(64, 64)))
__global__ void ternary_gemv_wmma_kernel(
    const uint32_t* __restrict__ packed_weights,  // [M, K/16] row-major
    const int8_t*   __restrict__ x_i8,            // [K] INT8 activations
    float                        x_scale,
    const float*    __restrict__ row_scales,      // [M] per-row FP32 scale
    __half*         __restrict__ y_fp16,           // [M] FP16 output
    int M, int K)
{
    const int wave_id = threadIdx.x >> 5;       // 0 or 1
    const int lane    = threadIdx.x & 31;        // 0..31
    const int row_base = blockIdx.x * 32 + wave_id * 16;
    if (row_base >= M) return;

    const int u32_per_row = K / 16;
    const int n_tiles = K / 16;

    // ─── LDS for weight tiles ───
    // layout: weights_lds[wave][row_in_wave][tile_within_batch] = packed uint32
    // Two LDS banks per wave (not used for double buffering yet, but reserved).
    __shared__ uint32_t weights_lds[2][2][16][WEIGHT_TILES_PER_BATCH];

    int32x8_t wmma_acc = {};

    // ─── Iterate over batches of WEIGHT_TILES_PER_BATCH tiles ───
    for (int batch_start = 0; batch_start < n_tiles; batch_start += WEIGHT_TILES_PER_BATCH) {
        const int tiles_this_batch = min(WEIGHT_TILES_PER_BATCH, n_tiles - batch_start);

        // ── Prefetch this batch's weights into LDS (all lanes participate) ──
        // Each lane loads 1 uint32 per tile for its assigned row, but we batch
        // the loads: lane i loads tile t from row (row_base + i%16).
        // With 32 lanes and 16 rows, each row has 2 lanes loading = 2 uint32s/tile.
        // We interleave: lane i loads tiles where (tile & 1) == (i >> 4).
        // Lane 0..15 load tile 0,2,4,... Lane 16..31 load tile 1,3,5,...
        const int local_row = lane & 0xF;
        const int global_row = row_base + local_row;
        const int parity = lane >> 4;  // 0 for lanes 0..15, 1 for lanes 16..31

        for (int t = 0; t < tiles_this_batch; t++) {
            if ((t & 1) == parity) {
                int k_u32 = batch_start + t;
                uint32_t word = 0;
                if (global_row < M && k_u32 < u32_per_row) {
                    word = packed_weights[(size_t)global_row * u32_per_row + k_u32];
                }
                weights_lds[wave_id][0][local_row][t] = word;
            }
        }
        __syncthreads();

        // ── Process all tiles in this batch from LDS ──
        for (int t = 0; t < tiles_this_batch; t++) {
            const int k_base = (batch_start + t) * 16;

            // ── Broadcast activations via __shfl (same as v2) ──
            int8_t act_reg = 0;
            if (lane < 16) {
                int k = k_base + lane;
                act_reg = (k < K) ? x_i8[k] : 0;
            }
            int8x16_t a_frag;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                a_frag[i] = __shfl(act_reg, i);
            }

            // ── Load weight from LDS instead of global memory ──
            uint32_t packed_word = weights_lds[wave_id][0][local_row][t];
            int8x16_t b_frag = unpack_ternary_4(packed_word);

            // ── WMMA accumulate ──
            wmma_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                true, a_frag, true, b_frag, wmma_acc, false);
        }

        __syncthreads();  // ensure LDS is done before next prefetch overwrites it
    }

    // ── Extract outputs ──
    constexpr float FP16_MAX = 65504.0f;
    if (lane < 16) {
        int gr = row_base + lane;
        if (gr < M) {
            float out = (float)wmma_acc[0] * x_scale * row_scales[gr];
            out = fminf(fmaxf(out, -FP16_MAX), FP16_MAX);
            y_fp16[gr] = __float2half(out);
        }
    }
}

// ─── 16-row fallback for M < 32 ──────────────────────────────────────────────
__attribute__((amdgpu_flat_work_group_size(WARP_SIZE, WARP_SIZE)))
__global__ void ternary_gemv_wmma_kernel_16row(
    const uint32_t* __restrict__ packed_weights,
    const int8_t*   __restrict__ x_i8,
    float                        x_scale,
    const float*    __restrict__ row_scales,
    __half*         __restrict__ y_fp16,
    int M, int K)
{
    const int row_base = blockIdx.x * 16;
    if (row_base >= M) return;

    const int lane = threadIdx.x;
    const int u32_per_row = K / 16;
    const int n_tiles = K / 16;
    const int local_row = lane & 0xF;
    const int global_row = row_base + local_row;
    const int parity = lane >> 4;

    // LDS for 16-row single-wave variant (smaller, no double buffering needed)
    __shared__ uint32_t weights_lds[16][WEIGHT_TILES_PER_BATCH];

    int32x8_t wmma_acc = {};

    for (int batch_start = 0; batch_start < n_tiles; batch_start += WEIGHT_TILES_PER_BATCH) {
        const int tiles_this_batch = min(WEIGHT_TILES_PER_BATCH, n_tiles - batch_start);

        // ── Prefetch into LDS ──
        for (int t = 0; t < tiles_this_batch; t++) {
            if ((t & 1) == parity) {
                int k_u32 = batch_start + t;
                weights_lds[local_row][t] =
                    (global_row < M && k_u32 < u32_per_row)
                        ? packed_weights[(size_t)global_row * u32_per_row + k_u32]
                        : 0;
            }
        }
        __syncthreads();

        // ── Process from LDS ──
        for (int t = 0; t < tiles_this_batch; t++) {
            int k_base = (batch_start + t) * 16;

            int8_t act_reg = 0;
            if (lane < 16) {
                act_reg = (k_base + lane < K) ? x_i8[k_base + lane] : 0;
            }
            int8x16_t a_frag;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                a_frag[i] = __shfl(act_reg, i);
            }

            int8x16_t b_frag = unpack_ternary_4(weights_lds[local_row][t]);
            wmma_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                true, a_frag, true, b_frag, wmma_acc, false);
        }

        __syncthreads();
    }

    constexpr float FP16_MAX = 65504.0f;
    if (lane < 16) {
        int gr = row_base + lane;
        if (gr < M) {
            float out = (float)wmma_acc[0] * x_scale * row_scales[gr];
            out = fminf(fmaxf(out, -FP16_MAX), FP16_MAX);
            y_fp16[gr] = __float2half(out);
        }
    }
}

// ─── V2-style baseline kept as reference (no LDS tiling, single wave) ───────
// Used when K is small enough that LDS tiling overhead doesn't pay off.
// ────────────────────────────────────────────────────────────────────────────
__attribute__((amdgpu_flat_work_group_size(WARP_SIZE, WARP_SIZE)))
__global__ void ternary_gemv_wmma_kernel_v2(
    const uint32_t* __restrict__ packed_weights,
    const int8_t*   __restrict__ x_i8,
    float                        x_scale,
    const float*    __restrict__ row_scales,
    __half*         __restrict__ y_fp16,
    int M, int K)
{
    const int row_base = blockIdx.x * 16;
    if (row_base >= M) return;

    const int lane = threadIdx.x;
    const int u32_per_row = K / 16;
    const int n_tiles = K / 16;

    int32x8_t wmma_acc = {};

    for (int tile = 0; tile < n_tiles; ++tile) {
        const int k_base = tile * 16;

        int8_t act_val = 0;
        if (lane < 16) {
            int k = k_base + lane;
            act_val = (k < K) ? x_i8[k] : 0;
        }
        int8x16_t a_frag;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            a_frag[i] = __shfl(act_val, i);
        }

        int local_row = lane & 0xF;
        int global_row = row_base + local_row;
        uint32_t packed_word = 0;
        if (global_row < M && tile < u32_per_row) {
            packed_word = packed_weights[(size_t)global_row * u32_per_row + tile];
        }
        int8x16_t b_frag = unpack_ternary_4(packed_word);

        wmma_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true, a_frag, true, b_frag, wmma_acc, false);
    }

    constexpr float FP16_MAX = 65504.0f;
    if (lane < 16) {
        int gr = row_base + lane;
        if (gr < M) {
            float out = (float)wmma_acc[0] * x_scale * row_scales[gr];
            out = fminf(fmaxf(out, -FP16_MAX), FP16_MAX);
            y_fp16[gr] = __float2half(out);
        }
    }
}

// ============================================================================
// Launcher — auto-selects best variant
// ============================================================================

extern "C" void ternary_gemv_wmma_launch(
    const void* packed, const void* x_i8, float x_scale,
    const void* scales, void* y_fp16,
    int M, int K, void* stream)
{
    if (K < 16 || (K % 16) != 0) {
        extern void ternary_gemv_phase5_halo_f16_launch(
            const void*, const void*, float, const void*, void*, int, int, void*);
        ternary_gemv_phase5_halo_f16_launch(
            packed, static_cast<const int8_t*>(x_i8), x_scale,
            static_cast<const float*>(scales),
            static_cast<__half*>(y_fp16), M, K, stream);
        return;
    }

    // For small K (< 256), the LDS tiling overhead doesn't pay off — use v2 baseline.
    // For larger K, the LDS-tiled v3 kernel reduces global memory accesses by
    // WEIGHT_TILES_PER_BATCH×.
    if (K < 256) {
        dim3 grid((M + 15) / 16, 1, 1);
        dim3 block(WARP_SIZE, 1, 1);
        hipLaunchKernelGGL(ternary_gemv_wmma_kernel_v2,
                           grid, block, 0, (hipStream_t)stream,
                           static_cast<const uint32_t*>(packed),
                           static_cast<const int8_t*>(x_i8),
                           x_scale,
                           static_cast<const float*>(scales),
                           static_cast<__half*>(y_fp16),
                           M, K);
    } else if (M >= 32) {
        dim3 grid((M + 31) / 32, 1, 1);
        dim3 block(64, 1, 1);
        hipLaunchKernelGGL(ternary_gemv_wmma_kernel,
                           grid, block, 0, (hipStream_t)stream,
                           static_cast<const uint32_t*>(packed),
                           static_cast<const int8_t*>(x_i8),
                           x_scale,
                           static_cast<const float*>(scales),
                           static_cast<__half*>(y_fp16),
                           M, K);
    } else {
        dim3 grid((M + 15) / 16, 1, 1);
        dim3 block(WARP_SIZE, 1, 1);
        hipLaunchKernelGGL(ternary_gemv_wmma_kernel_16row,
                           grid, block, 0, (hipStream_t)stream,
                           static_cast<const uint32_t*>(packed),
                           static_cast<const int8_t*>(x_i8),
                           x_scale,
                           static_cast<const float*>(scales),
                           static_cast<__half*>(y_fp16),
                           M, K);
    }
}
