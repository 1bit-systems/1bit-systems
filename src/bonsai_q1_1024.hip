#include <cstdio>
#define HIP_CHECK(e) do { hipError_t _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP Error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_s)); abort(); } } while(0)

// bonsai_q1_1024.hip — Q1_1024: 1-bit GEMV with 1024-weight / 128-byte blocks
//
// Problem: the original Q1_0_g128 format uses 18-byte blocks (2B scale + 16B
// sign bits for 128 weights). On a GPU with 128-byte cache lines, every
// 18-byte block fetch wastes 110 bytes (86% of bandwidth).
//
// Solution — Q1_1024:
//   1024 weights × 1 bit = 128 bytes sign bits + 2 bytes FP16 scale = 130 B
//   Stored tile-interleaved (8 rows contiguous) so a workgroup reads
//   8 × 130 = 1040 contiguous bytes = 9 cache lines, 1040/1152 = 90% efficiency
//   vs the original 18/128 = 14%.
//
// Block layout (130 bytes):
//   [0..127] qs[128]   — 1024 sign bits, LSB-first
//   [128..129] d : FP16 — block scale
//   w[i] = (qs[i/8] >> (i%8)) & 1 ? +d : -d
//
// Thread model (wave32 gfx1151):
//   Same 8-rows-per-workgroup topology as the original kernel. Each wave
//   handles 2 output rows; per-row accumulator lives in registers.
//   Activations are staged into 1024-entry LDS (2048 bytes) once per block,
//   shared by all 8 rows.
//
// Key difference: 32 lanes each load 4 contiguous bytes from the 128-byte
// sign-bit region — a single coalesced 128-byte transaction. No nibble/pair
// unpacking needed. Each lane decodes 32 sign bits in a tight unrolled loop.
//
// LDS usage: 1024 × fp16 = 2048 B activation slab (per block, reloaded).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>

#include "rocm_cpp/bonsai.h"

namespace {

constexpr int kBlockSize      = 128;
constexpr int kWaveSize       = 32;
constexpr int kWaves          = kBlockSize / kWaveSize;      // 4
constexpr int kRowsPerWarp    = 2;
constexpr int kRowsPerWG      = kWaves * kRowsPerWarp;       // 8
constexpr int kWeightsPerB    = 1024;                        // weights per block
constexpr int kWeightsPerLane = kWeightsPerB / kWaveSize;    // 32 weights/lane
constexpr int kQ1024BlockBytes = 130;                        // 128B sign + 2B scale
constexpr int kScaleOffset    = 128;                         // d at offset 128

} // namespace

__attribute__((amdgpu_flat_work_group_size(kBlockSize, kBlockSize)))
__global__ void bonsai_q1_1024_gemv_kernel(
    const uint8_t* __restrict__ packed,
    const __half*  __restrict__ act,
    __half*        __restrict__ out,
    int N_out, int K_in)
{
    const int tid      = threadIdx.x;
    const int lane     = tid & (kWaveSize - 1);
    const int wave     = tid >> 5;
    const int tile     = blockIdx.x;
    const int row_base = tile * kRowsPerWG;

    const int num_blocks     = K_in >> 10;  // K / 1024
    const size_t tile_blk_stride = (size_t)kRowsPerWG * kQ1024BlockBytes; // 8 * 130 = 1040
    const size_t tile_stride     = (size_t)num_blocks * tile_blk_stride;

    __shared__ __half x_lds[kWeightsPerB];

    float acc[kRowsPerWarp];
    #pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) acc[r] = 0.0f;

    // Each lane reads 4 consecutive bytes from the 128-byte sign-bit region.
    // Lane l → bytes l*4 .. l*4+3 (contiguous, coalesced across all 32 lanes).
    const int qs_lane_offset = lane * 4;

    for (int b = 0; b < num_blocks; ++b) {
        // Cooperative load: 128 threads × 8 fp16 = 1024 activations into LDS
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            x_lds[tid * 8 + t] = act[b * kWeightsPerB + tid * 8 + t];
        }
        __syncthreads();

        // Base of this tile-block in the interleaved layout
        const uint8_t* __restrict__ tile_blk_base =
            packed + (size_t)tile * tile_stride + (size_t)b * tile_blk_stride;

        #pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            const int row = row_base + wave * kRowsPerWarp + r;
            if (row >= N_out) continue;

            const int row_in_wg = wave * kRowsPerWarp + r;
            const uint8_t* __restrict__ blk =
                tile_blk_base + (size_t)row_in_wg * kQ1024BlockBytes;

            const float d = __half2float(
                *reinterpret_cast<const __half*>(blk + kScaleOffset));

            // Load 4 contiguous bytes = 32 sign bits for this lane
            uint32_t bits;
            __builtin_memcpy(&bits, blk + qs_lane_offset, sizeof(bits));

            // Decode 32 sign bits and MAC with activations
            float partial = 0.0f;
            const int act_base = lane * kWeightsPerLane;
            #pragma unroll
            for (int j = 0; j < kWeightsPerLane; ++j) {
                const float a = __half2float(x_lds[act_base + j]);
                const float sign = (bits & 1) ? 1.0f : -1.0f;
                partial = fmaf(sign, a, partial);
                bits >>= 1;
            }

            acc[r] = fmaf(d, partial, acc[r]);
        }
    }

    // Intra-wave reduction
    #pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        float v = acc[r];
        #pragma unroll
        for (int off = kWaveSize / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off);
        }
        if (lane == 0) {
            const int row = row_base + wave * kRowsPerWarp + r;
            if (row < N_out) {
                constexpr float FP16_MAX = 65504.0f;
                if (v >  FP16_MAX) v =  FP16_MAX;
                if (v < -FP16_MAX) v = -FP16_MAX;
                out[row] = __float2half(v);
            }
        }
    }
}

// ── C API ───────────────────────────────────────────────────────────────────

extern "C" void bonsai_q1_1024_gemv_launch(
    const uint8_t*  packed_weights,
    const uint16_t* act_fp16,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream)
{
    if (N_out <= 0 || K_in <= 0)            return;
    if ((K_in & (kWeightsPerB - 1)) != 0)   return;

    const int num_wgs = (N_out + kRowsPerWG - 1) / kRowsPerWG;
    dim3 grid(num_wgs, 1, 1);
    dim3 block(kBlockSize, 1, 1);
    hipLaunchKernelGGL(bonsai_q1_1024_gemv_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       packed_weights,
                       reinterpret_cast<const __half*>(act_fp16),
                       reinterpret_cast<__half*>(out_fp16),
                       N_out, K_in);
}

// ── Converter: old Q1_0_g128 row-major → new Q1_1024 tile8-interleaved ──────
// Each old block (18B, 128 weights) maps to 1/8 of a new block (the sign bits
// are distributed across 8 new blocks since 1 new block = 1024 weights = 8 old
// blocks of 128 weights).
//
// For each tile of 8 consecutive rows:
//   For each new block b (covering weights b*1024 .. b*1024+1023):
//     For each row in the tile:
//       The 8 old blocks' data is repacked into 1 new 130-byte block.
//
// The old format per 128-weight block: [d:2B][qs:16B]
// The new format per 1024-weight block: [qs:128B][d:2B]

extern "C" void bonsai_q1_convert_to_1024(
    const uint8_t* packed_aos,
    uint8_t**      packed_1024,
    int            N_out,
    int            K_in)
{
    const int old_blocks_per_row = K_in >> 7;       // K / 128
    const int new_blocks_per_row = K_in >> 10;      // K / 1024
    const int num_tiles = (N_out + kRowsPerWG - 1) / kRowsPerWG;
    const size_t tile_stride = (size_t)new_blocks_per_row * kRowsPerWG * kQ1024BlockBytes;
    const size_t total_bytes = (size_t)num_tiles * tile_stride;
    const size_t old_row_stride = (size_t)old_blocks_per_row * 18;

    uint8_t* dst = nullptr;
    HIP_CHECK(hipMalloc(&dst, total_bytes));
    HIP_CHECK(hipMemset(dst, 0, total_bytes)); // zero-fill in case N_out not multiple of 8

    for (int t = 0; t < num_tiles; ++t) {
        for (int nb = 0; nb < new_blocks_per_row; ++nb) {
            for (int r = 0; r < kRowsPerWG; ++r) {
                const int row = t * kRowsPerWG + r;
                if (row >= N_out) continue;

                // Gather 8 old blocks (128 weights each) into 1 new block
                // Old blocks nb*8 .. nb*8+7 → 1 new block of 1024 weights
                uint8_t new_blk[kQ1024BlockBytes] = {0};
                __half scale = 0.0f;

                for (int ob = 0; ob < 8; ++ob) {
                    const uint8_t* old_blk = packed_aos
                        + (size_t)row * old_row_stride
                        + (size_t)(nb * 8 + ob) * 18;

                    // Copy 16 bytes of sign bits to the correct position
                    // Old block ob → new block's bytes ob*16 .. ob*16+15
                    for (int i = 0; i < 16; ++i) {
                        new_blk[ob * 16 + i] = old_blk[2 + i]; // skip old scale
                    }

                    // Accumulate scale (mean of old block scales)
                    __half old_d;
                    __builtin_memcpy(&old_d, old_blk, sizeof(old_d));
                    scale = __hadd(scale, old_d);
                }

                // Average scale
                scale = __hdiv(scale, __float2half(8.0f));

                // Write scale at offset 128
                __builtin_memcpy(new_blk + kScaleOffset, &scale, sizeof(scale));

                // Write to destination
                uint8_t* dst_blk = dst
                    + (size_t)t * tile_stride
                    + (size_t)nb * kRowsPerWG * kQ1024BlockBytes
                    + (size_t)r * kQ1024BlockBytes;

                for (int i = 0; i < kQ1024BlockBytes; ++i) {
                    dst_blk[i] = new_blk[i];
                }
            }
        }
    }

    *packed_1024 = dst;
}
