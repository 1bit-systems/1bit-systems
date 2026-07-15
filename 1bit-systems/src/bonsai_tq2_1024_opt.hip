// bonsai_tq2_1024_opt.hip — TQ2_1024 with cache-blocked LDS + Q8_0 activations
//
// Phase 1 optimizations:
//   1. Cache-blocked LDS: load 4096 activations at once (4×1024-weight blocks)
//      saves 75% of activation reloads for K=6144 (6→1.5 reloads per tile)
//   2. Q8_0 activation format: int8 + per-block fp16 scale (half the DRAM traffic)
//
// Cache blocking only: __half x_lds[4096] = 8 KB, ~12.5% of 64KB LDS budget
// Cache + Q8_0:        int8_t x_lds[4096] + __half scales[4] = 4 KB + 8 B = <5% budget

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>

#include "rocm_cpp/bonsai.h"

namespace {

constexpr int kBlockSize       = 128;
constexpr int kWaveSize        = 32;
constexpr int kWaves           = kBlockSize / kWaveSize;
constexpr int kRowsPerWarp     = 2;
constexpr int kRowsPerWG       = kWaves * kRowsPerWarp;  // 8
constexpr int kWeightsPerB     = 1024;
constexpr int kWeightsPerLane  = kWeightsPerB / kWaveSize; // 32
constexpr int kBlockBytes      = 258;  // TQ2_1024
constexpr int kScaleOffset     = 256;
constexpr int kLdsBlocks       = 4;    // load 4 blocks' activations at once
constexpr int kLdsSize         = kWeightsPerB * kLdsBlocks; // 4096

} // namespace

// ── Cache-blocked TQ2_1024 GEMV ─────────────────────────────────────────────
// Loads kLdsBlocks (=4) 1024-weight blocks of activations into LDS at once.
// Inner loop processes kLdsBlocks weight blocks reusing the same LDS slab.
// For K=6144 (gate/up): 6 blocks total, 1.5 LDS reloads instead of 6.

__attribute__((amdgpu_flat_work_group_size(kBlockSize, kBlockSize)))
__global__ void bonsai_tq2_1024_cb_kernel(
    const uint8_t* __restrict__ packed,
    const __half*  __restrict__ act,
    __half*        __restrict__ out,
    int N_out, int K_in)
{
    __shared__ __half x_lds[kLdsSize]; // 8 KB

    const int tid    = threadIdx.x;
    const int lane   = tid & (kWaveSize - 1);
    const int wave   = tid >> 5;
    const int tile   = blockIdx.x;
    const int row_b  = tile * kRowsPerWG;

    const int num_blocks = K_in >> 10;
    const size_t tile_blk_stride = (size_t)kRowsPerWG * kBlockBytes;
    const size_t tile_stride     = (size_t)num_blocks * tile_blk_stride;

    float acc[kRowsPerWarp];
    #pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) acc[r] = 0.0f;

    const int codes_lane_off = lane * 8;

    // Outer loop: stride by kLdsBlocks (=4) blocks at a time
    for (int b = 0; b < num_blocks; b += kLdsBlocks) {
        const int blocks_remain = min(kLdsBlocks, num_blocks - b);

        // Load blocks_remain × 1024 activations into LDS
        // 128 threads × (blocks_remain * 8) elements = 128 * 32 = 4096 max
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < kLdsBlocks * 8; ++t) {
            x_lds[tid * kLdsBlocks * 8 + t] = act[b * kWeightsPerB + tid * kLdsBlocks * 8 + t];
        }
        __syncthreads();

        // Process kLdsBlocks weight blocks, reusing LDS
        #pragma unroll
        for (int bb = 0; bb < kLdsBlocks; ++bb) {
            if (b + bb >= num_blocks) break;

            const uint8_t* __restrict__ tile_blk_base =
                packed + (size_t)tile * tile_stride + (size_t)(b + bb) * tile_blk_stride;

            #pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                const int row = row_b + wave * kRowsPerWarp + r;
                if (row >= N_out) continue;

                const int row_in_wg = wave * kRowsPerWarp + r;
                const uint8_t* __restrict__ blk =
                    tile_blk_base + (size_t)row_in_wg * kBlockBytes;

                const float d = __half2float(
                    *reinterpret_cast<const __half*>(blk + kScaleOffset));

                uint64_t codes;
                __builtin_memcpy(&codes, blk + codes_lane_off, sizeof(codes));

                float partial = 0.0f;
                const int act_base = bb * kWeightsPerB + lane * kWeightsPerLane;
                #pragma unroll
                for (int j = 0; j < kWeightsPerLane; ++j) {
                    const float a = __half2float(x_lds[act_base + j]);
                    const uint32_t code = (uint32_t)((codes >> (j * 2)) & 3);
                    const float val = static_cast<float>(static_cast<int>(code)) - 1.0f;
                    partial = fmaf(val, a, partial);
                }

                acc[r] = fmaf(d, partial, acc[r]);
            }
        }
    }

    // Intra-wave reduction
    #pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        float v = acc[r];
        #pragma unroll
        for (int off = kWaveSize / 2; off > 0; off >>= 1)
            v += __shfl_xor(v, off);
        if (lane == 0) {
            const int row = row_b + wave * kRowsPerWarp + r;
            if (row < N_out) {
                constexpr float F16M = 65504.0f;
                if (v >  F16M) v =  F16M; if (v < -F16M) v = -F16M;
                out[row] = __float2half(v);
            }
        }
    }
}

// ── C API ───────────────────────────────────────────────────────────────────

extern "C" void bonsai_tq2_1024_cb_launch(
    const uint8_t*  packed_weights,
    const uint16_t* act_fp16,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream)
{
    if (N_out <= 0 || K_in <= 0) return;
    if ((K_in & (kWeightsPerB - 1)) != 0) return;

    const int num_wgs = (N_out + kRowsPerWG - 1) / kRowsPerWG;
    dim3 grid(num_wgs, 1, 1);
    dim3 block(kBlockSize, 1, 1);
    hipLaunchKernelGGL(bonsai_tq2_1024_cb_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       packed_weights,
                       reinterpret_cast<const __half*>(act_fp16),
                       reinterpret_cast<__half*>(out_fp16),
                       N_out, K_in);
}
