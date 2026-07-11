// bonsai_q1_gemv.hip — Bonsai Q1_0_g128 1-bit GEMV for gfx1151 (wave32).
//
// API (see include/rocm_cpp/bonsai.h):
//   bonsai_q1_gemv_launch(packed, act_fp16, out_fp16, N_out, K_in, stream)
//
// Layout (AoS, block-interleaved, 18 bytes per g128 block):
//   [FP16 d : 2 bytes][qs[16] : 16 bytes, 128 sign bits LSB-first]
//   w[i] = (qs[i/8] >> (i%8)) & 1 ? +d : -d
//
// Thread model (wave32 gfx1151) — multi-row-per-workgroup, same topology
// as bonsai_tq2_gemv.hip (uplifted from the original one-row/workgroup +
// two-stage-reduction design; see that file's header for the full
// rationale, repeated briefly here):
//   - One workgroup = 128 threads (4 waves × 32 lanes) = 8 output rows.
//   - Wave `w` owns rows `ROWS_PER_WG * w / kWaves .. +ROWS_PER_WARP`,
//     i.e. 2 rows/wave. Per-row accumulator lives in registers; the two
//     rows share the same 32 lanes, so no inter-wave fold is needed.
//   - Activations for the current 128-weight K-block are staged into
//     128-entry LDS once per block by all 128 threads cooperatively;
//     all 8 rows then read from that slab — 8x less activation DRAM
//     traffic than the old one-row-per-block topology.
//   - Inner loop: each lane covers 4 consecutive weights in the block
//     (lane `l` → weights 4l .. 4l+3). Unlike TQ2 (2-bit codes, 4/byte,
//     so 1 lane = 1 byte), Q1 packs 8 sign bits/byte, so 4 weights is
//     only half a byte: lane `l` reads qs byte `l>>1` and extracts the
//     low nibble (l even) or high nibble (l odd). 2 lanes share each
//     byte read — still a single coalesced 16-byte transaction across
//     the 32 lanes, just with each byte address hit by 2 lanes instead
//     of 1. Scale `d` is a warp-uniform broadcast as in TQ2.
//
// 1-bit decode — branch-free:
//   sign(bit) = bit ? +1 : -1 = 2*bit - 1. No branch, no LUT.
//
// LDS usage: 128 × fp16 (256 B) act slab. Per-row reductions live in
// registers (2 row-accumulators per lane) and fold within each wave via
// __shfl_xor — no inter-wave scratch, no extra __syncthreads past the
// cooperative activation load barriers. (272 B wave_sums scratch from the
// old two-stage reduction is gone — no longer needed.)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>

#include "rocm_cpp/bonsai.h"

namespace {

constexpr int kBlockSize     = 128;
constexpr int kWaveSize      = 32;
constexpr int kWaves         = kBlockSize / kWaveSize;      // 4
constexpr int kRowsPerWarp   = 2;
constexpr int kRowsPerWG     = kWaves * kRowsPerWarp;       // 8
constexpr int kWeightsPerB   = 128;
constexpr int kWeightsPerLane = kWeightsPerB / kWaveSize;   // 4
constexpr int kQ1BlockBytes  = 18;
constexpr int kScaleOffset   = 0;    // `d` first, then 16 bytes of sign bits
constexpr int kQsOffset      = 2;    // qs[0..16) live at blk + 2

__device__ __forceinline__
__half load_scale_fp16(const uint8_t* __restrict__ block_ptr) {
    uint16_t u;
    __builtin_memcpy(&u, block_ptr + kScaleOffset, sizeof(u));
    __half h;
    __builtin_memcpy(&h, &u, sizeof(h));
    return h;
}

// Branch-free decode of the 4 sign bits packed in a nibble (half a byte).
// Returns sum_{j=0..3} sign(bit_j) * a[j], sign(bit) = 2*bit - 1 in {-1,+1}.
__device__ __forceinline__
float decode4_mac(uint32_t nibble, float a0, float a1, float a2, float a3) {
    const uint32_t b0 =  nibble        & 1u;
    const uint32_t b1 = (nibble >> 1)  & 1u;
    const uint32_t b2 = (nibble >> 2)  & 1u;
    const uint32_t b3 = (nibble >> 3)  & 1u;

    auto sj = [](uint32_t b) __attribute__((always_inline)) -> float {
        return 2.0f * static_cast<float>(b) - 1.0f;
    };
    float acc = sj(b0) * a0;
    acc = fmaf(sj(b1), a1, acc);
    acc = fmaf(sj(b2), a2, acc);
    acc = fmaf(sj(b3), a3, acc);
    return acc;
}

}  // namespace

__attribute__((amdgpu_flat_work_group_size(kBlockSize, kBlockSize)))
__global__ void bonsai_q1_gemv_kernel(
    const uint8_t* __restrict__ packed,
    const __half*  __restrict__ act,
    __half*        __restrict__ out,
    int N_out, int K_in)
{
    const int tid      = threadIdx.x;
    const int lane     = tid & (kWaveSize - 1);
    const int wave     = tid >> 5;
    const int row_base = blockIdx.x * kRowsPerWG;

    const int num_blocks   = K_in >> 7;
    const size_t row_bytes = (size_t)num_blocks * (size_t)kQ1BlockBytes;

    __shared__ __half x_lds[kWeightsPerB];

    // One fp32 accumulator per row owned by this wave.
    float acc[kRowsPerWarp];
    #pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) acc[r] = 0.0f;

    // Nibble split: lane l -> qs byte (l>>1), shift (l&1)*4.
    const int qs_byte_idx = lane >> 1;
    const int nibble_shift = (lane & 1) * 4;

    for (int b = 0; b < num_blocks; ++b) {
        // Cooperative load: 128 threads each write one fp16 activation.
        __syncthreads();
        x_lds[tid] = act[b * kWeightsPerB + tid];
        __syncthreads();

        // This lane's 4 activations (4 consecutive weights = one nibble).
        const int a_base = lane * kWeightsPerLane;
        const float a0 = __half2float(x_lds[a_base + 0]);
        const float a1 = __half2float(x_lds[a_base + 1]);
        const float a2 = __half2float(x_lds[a_base + 2]);
        const float a3 = __half2float(x_lds[a_base + 3]);

        #pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            const int row = row_base + wave * kRowsPerWarp + r;
            const bool row_valid = (row < N_out);
            const int row_eff    = row_valid ? row : 0;
            const uint8_t* __restrict__ blk =
                packed + (size_t)row_eff * row_bytes + (size_t)b * kQ1BlockBytes;

            const float d = __half2float(load_scale_fp16(blk));

            // Lanes 0..31 cover qs bytes 0..15 (2 lanes/byte) — still one
            // coalesced 16-byte transaction per row per block.
            const uint32_t qbyte = (uint32_t)blk[kQsOffset + qs_byte_idx];
            const uint32_t nibble = (qbyte >> nibble_shift) & 0xFu;

            const float partial = decode4_mac(nibble, a0, a1, a2, a3);
            if (row_valid) acc[r] = fmaf(d, partial, acc[r]);
        }
    }

    // Intra-wave xor butterfly per row. Each wave's 2 rows reduce
    // independently within the same 32 lanes.
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

extern "C" void bonsai_q1_gemv_launch(
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
    hipLaunchKernelGGL(bonsai_q1_gemv_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       packed_weights,
                       reinterpret_cast<const __half*>(act_fp16),
                       reinterpret_cast<__half*>(out_fp16),
                       N_out, K_in);
}
