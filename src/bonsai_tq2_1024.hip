#include <cstdio>
#define HIP_CHECK(e) do { hipError_t _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP Error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_s)); abort(); } } while(0)

// bonsai_tq2_1024.hip — TQ2_1024: ternary 2-bit GEMV with 1024-weight blocks
//
// Problem: original TQ2_0_g128 uses 34-byte blocks (2B scale + 32B 2-bit codes
// for 128 weights). Bus efficiency: 34/128 = 27%.
//
// Solution — TQ2_1024:
//   1024 weights × 2 bits = 256 bytes codes + 2 bytes FP16 scale = 258 B
//   Bus efficiency: 258/(3×128) = 67% (2.5× better)
//   Tile-8 interleaved: 8 × 258 = 2064 contiguous bytes, 95% bus efficiency
//
// Block layout (258 bytes):
//   [0..255] qs[256]    — 1024 × 2-bit codes, 4/byte LSB-first
//   [256..257] d : FP16 — block scale
//   Code map: 0→-1, 1→0, 2→+1, 3→+2  (value = (int)code - 1)
//   Weight reconstruction: w[j] = d * ((int)(qs[j/4] >> (j%4)*2 & 3) - 1)
//
// Thread model:
//   8 rows/workgroup, 2 rows/wave. 32 lanes each load 8 contiguous bytes
//   (64 bits = 32 × 2-bit codes) from the 256-byte code region —
//   a single coalesced 256-byte transaction across all lanes (2 cache lines).
//   Activation LDS: 1024 fp16 entries shared by all 8 rows.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>

#include "rocm_cpp/bonsai.h"

namespace {

constexpr int kBlockSize       = 128;
constexpr int kWaveSize        = 32;
constexpr int kWaves           = kBlockSize / kWaveSize;       // 4
constexpr int kRowsPerWarp     = 2;
constexpr int kRowsPerWG       = kWaves * kRowsPerWarp;        // 8
constexpr int kWeightsPerB     = 1024;                         // weights per block
constexpr int kWeightsPerLane  = kWeightsPerB / kWaveSize;     // 32 weights/lane
constexpr int kTQ1024BlockBytes = 258;                         // 256B codes + 2B scale
constexpr int kScaleOffset     = 256;                          // d at offset 256

} // namespace

__attribute__((amdgpu_flat_work_group_size(kBlockSize, kBlockSize)))
__global__ void bonsai_tq2_1024_gemv_kernel(
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

    const int num_blocks     = K_in >> 10;   // K / 1024
    const size_t tile_blk_stride = (size_t)kRowsPerWG * kTQ1024BlockBytes; // 8*258 = 2064
    const size_t tile_stride     = (size_t)num_blocks * tile_blk_stride;

    __shared__ __half x_lds[kWeightsPerB];

    float acc[kRowsPerWarp];
    #pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) acc[r] = 0.0f;

    // Each lane reads 8 bytes = 32 × 2-bit codes from the 256-byte region
    const int codes_lane_offset = lane * 8;

    for (int b = 0; b < num_blocks; ++b) {
        // Load 1024 activations into LDS (128 threads × 8 fp16)
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            x_lds[tid * 8 + t] = act[b * kWeightsPerB + tid * 8 + t];
        }
        __syncthreads();

        const uint8_t* __restrict__ tile_blk_base =
            packed + (size_t)tile * tile_stride + (size_t)b * tile_blk_stride;

        #pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            const int row = row_base + wave * kRowsPerWarp + r;
            if (row >= N_out) continue;

            const int row_in_wg = wave * kRowsPerWarp + r;
            const uint8_t* __restrict__ blk =
                tile_blk_base + (size_t)row_in_wg * kTQ1024BlockBytes;

            const float d = __half2float(
                *reinterpret_cast<const __half*>(blk + kScaleOffset));

            // Load 8 bytes = 32 × 2-bit codes for this lane
            uint64_t codes;
            __builtin_memcpy(&codes, blk + codes_lane_offset, sizeof(codes));

            // Decode 32 ternary codes and MAC with activations
            // Code map: 0→-1, 1→0, 2→+1, 3→+2 (value = (int)code - 1)
            float partial = 0.0f;
            const int act_base = lane * kWeightsPerLane;
            #pragma unroll
            for (int j = 0; j < kWeightsPerLane; ++j) {
                const float a = __half2float(x_lds[act_base + j]);
                const uint32_t code = (uint32_t)((codes >> (j * 2)) & 3);
                // Branch-free: (int)code - 1 → -1, 0, +1, +2
                const float val = static_cast<float>(static_cast<int>(code)) - 1.0f;
                partial = fmaf(val, a, partial);
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

extern "C" void bonsai_tq2_1024_gemv_launch(
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
    hipLaunchKernelGGL(bonsai_tq2_1024_gemv_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       packed_weights,
                       reinterpret_cast<const __half*>(act_fp16),
                       reinterpret_cast<__half*>(out_fp16),
                       N_out, K_in);
}

// ── Converter: old TQ2 row-major → new TQ2_1024 tile8-interleaved ────────────
// Each old block (34B, 128 weights × 2-bit) → 1/8 of a new block.
// 8 old blocks = 1024 weights = 1 new block.
// Old: [d:2B][qs:32B]
// New: [qs:256B][d:2B]

extern "C" void bonsai_tq2_convert_to_1024(
    const uint8_t* packed_aos,
    uint8_t**      packed_1024,
    int            N_out,
    int            K_in)
{
    const int old_blocks_per_row = K_in >> 7;        // K / 128
    const int new_blocks_per_row = K_in >> 10;       // K / 1024
    const int num_tiles = (N_out + kRowsPerWG - 1) / kRowsPerWG;
    const size_t tile_stride = (size_t)new_blocks_per_row * kRowsPerWG * kTQ1024BlockBytes;
    const size_t total_bytes = (size_t)num_tiles * tile_stride;
    const size_t old_row_stride = (size_t)old_blocks_per_row * 34; // old block = 34B

    uint8_t* dst = nullptr;
    hipMalloc(&dst, total_bytes);
    hipMemset(dst, 0, total_bytes);

    for (int t = 0; t < num_tiles; ++t) {
        for (int nb = 0; nb < new_blocks_per_row; ++nb) {
            for (int r = 0; r < kRowsPerWG; ++r) {
                const int row = t * kRowsPerWG + r;
                if (row >= N_out) continue;

                // Gather 8 old blocks (128 × 2-bit each) into 1 new block
                uint8_t new_blk[kTQ1024BlockBytes] = {0};

                for (int ob = 0; ob < 8; ++ob) {
                    const uint8_t* old_blk = packed_aos
                        + (size_t)row * old_row_stride
                        + (size_t)(nb * 8 + ob) * 34;

                    // Copy 32 bytes of 2-bit codes to new block
                    // Old block ob → new block's bytes ob*32 .. ob*32+31
                    for (int i = 0; i < 32; ++i) {
                        new_blk[ob * 32 + i] = old_blk[2 + i]; // skip old scale
                    }
                }

                // Average scale from old blocks
                __half scale = __float2half(0.0f);
                for (int ob = 0; ob < 8; ++ob) {
                    const uint8_t* old_blk = packed_aos
                        + (size_t)row * old_row_stride
                        + (size_t)(nb * 8 + ob) * 34;
                    __half old_d;
                    __builtin_memcpy(&old_d, old_blk, sizeof(old_d));
                    scale = __hadd(scale, old_d);
                }
                scale = __hdiv(scale, __float2half(8.0f));

                // Write scale at offset 256
                __builtin_memcpy(new_blk + kScaleOffset, &scale, sizeof(scale));

                // Write to destination
                uint8_t* dst_blk = dst
                    + (size_t)t * tile_stride
                    + (size_t)nb * kRowsPerWG * kTQ1024BlockBytes
                    + (size_t)r * kTQ1024BlockBytes;

                for (int i = 0; i < kTQ1024BlockBytes; ++i) {
                    dst_blk[i] = new_blk[i];
                }
            }
        }
    }

    *packed_1024 = dst;
}
