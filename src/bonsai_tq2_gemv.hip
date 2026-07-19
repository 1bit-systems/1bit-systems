// bonsai_tq2_gemv.hip — Bonsai TQ2_0_g128 ternary GEMV for gfx1151 (wave32).
//
// API (see include/rocm_cpp/bonsai.h):
//   bonsai_tq2_gemv_launch(packed, act_fp16, out_fp16, N_out, K_in, stream)
//
// Layout (AoS, block-interleaved, 34 bytes per g128 block):
//   [FP16 d : 2 bytes][qs[32] : 32 bytes of 2-bit codes]
//   (d-first, NOT d-last — verified against the raw PrismML
//   Ternary-Bonsai-1.7B-Q2_0.gguf file. The Bonsai-Kernel-Spec.md doc
//   originally claimed d-last; it was wrong. Empirical: with d@0 no
//   block emits the reserved 0b11 code; with d@32 1-2 per block do,
//   and d@32 varies wildly across 10 orders of magnitude while d@0
//   stays in a tight per-row band.)
//   Code map per 2-bit code: value(code) = code - 1, i.e. 00→-1, 01→0,
//   10→+1, 11→+2. Verified bit-exact (cos=1.000000 vs F16) via
//   q2_0_decode.py; code 3 does not occur in real Bonsai-1.7B weight data
//   but is not a reserved/unused code, it's a genuine 4th quant level.
//   4 codes per byte, LSB-first. Weight `j` within the block:
//     byte_idx = 2 + j/4, shift = (j%4)*2, code = (qs[byte_idx] >> shift) & 3.
//
// Thread model (wave32 gfx1151) — multi-row-per-workgroup:
//   - One workgroup = 128 threads (4 waves × 32 lanes) = 8 output rows.
//   - Wave `w` owns rows `ROWS_PER_WG * w / kWaves .. +ROWS_PER_WARP`,
//     i.e. 2 rows/wave. Per-row accumulator lives in registers; the two
//     rows share the same 32 lanes, so no inter-wave fold is needed.
//   - Activations for the current 128-weight K-block are staged into
//     128-entry LDS once per block by all 128 threads cooperatively;
//     all 8 rows (all 4 waves × 2 rows) then read from that slab.
//     This amortizes activation DRAM traffic by 8× over the previous
//     one-row-per-block topology — the dominant win at N_out ≫ K_in.
//   - Inner loop: each lane covers 4 consecutive weights in the block
//     (lane `l` → weights 4l .. 4l+3). That is exactly the 4 codes in
//     a single qs byte (byte index = l), which collapses each wave's
//     per-row per-block weight fetch to one 32-byte coalesced load.
//     Scale `d` is a warp-uniform broadcast (lane 0 reads, __shfl out).
//
// 2-bit decode — branch-free, no LUT:
//   For code c ∈ {0, 1, 2, 3} with the mapping above, the ternary value
//   v(c) ∈ {-1, 0, +1, 0}. Key observation:
//     v(c) = (c == 0) ? -1 : ((c == 2) ? +1 : 0)
//          = float(c == 2) - float(c == 0)
//   Each `c == k` lowers to a single v_cmp_u32 + v_cvt_f32_u32 pair on
//   RDNA3.5 and the compiler fuses the subtraction into one FMA. The
//   0b11 reserved code decodes to 0 naturally (both comparisons false).
//   No branch, no cndmask storm, no LUT.
//
// LDS usage: 128 × fp16 (256 B) act slab. Per-row reductions live in
// registers (2 row-accumulators per lane) and fold within each wave via
// __shfl_xor — no inter-wave scratch, no extra __syncthreads past the
// cooperative activation load barriers.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>

#include "rocm_cpp/bonsai.h"
#ifndef HIP_CHECK
#define HIP_CHECK(e) do { hipError_t _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP Error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_s)); abort(); } } while(0)
#endif

namespace {

constexpr int kBlockSize     = 128;
constexpr int kWaveSize      = 32;
constexpr int kWaves         = kBlockSize / kWaveSize;      // 4
constexpr int kRowsPerWarp   = 2;
constexpr int kRowsPerWG     = kWaves * kRowsPerWarp;       // 8
constexpr int kWeightsPerB   = 128;
constexpr int kWeightsPerLane = kWeightsPerB / kWaveSize;   // 4
constexpr int kTQ2BlockBytes = 34;
constexpr int kScaleOffset   = 0;    // `d` first, then 32 bytes of codes
constexpr int kCodesOffset   = 2;    // codes[0..32) live at blk + 2

__device__ __forceinline__
__half load_scale_fp16(const uint8_t* __restrict__ block_ptr) {
    uint16_t u;
    __builtin_memcpy(&u, block_ptr + kScaleOffset, sizeof(u));
    __half h;
    __builtin_memcpy(&h, &u, sizeof(h));
    return h;
}

// Branch-free 2-bit decode of ALL 4 codes packed in a single byte at once.
// Returns the linear-code sum contribution as an fp32 — i.e.
//   sum_{j=0..3} v(code_j) * a[j]
// where v(c) = c - 1 ∈ {-1, 0, +1, +2}. Verified bit-exact (cos=1.000000)
// against the real Ternary-Bonsai-1.7B-Q2_0.gguf + F16 reference via
// q2_0_decode.py; code 3 never occurs in real data so this and the older
// "reserved 0b11 -> 0" assumption are numerically identical on data seen
// so far, but this is the mathematically verified mapping. Inlined so the
// compiler can hoist the FMA chain directly into the per-block accumulate
// loop.
__device__ __forceinline__
float decode4_mac(uint32_t qbyte, float a0, float a1, float a2, float a3) {
    // code_j = (qbyte >> (2*j)) & 3
    const uint32_t c0 =  qbyte        & 3u;
    const uint32_t c1 = (qbyte >> 2)  & 3u;
    const uint32_t c2 = (qbyte >> 4)  & 3u;
    const uint32_t c3 = (qbyte >> 6)  & 3u;

    // v(c) = c - 1.
    auto vj = [](uint32_t c) __attribute__((always_inline)) -> float {
        return static_cast<float>(c) - 1.0f;
    };
    float acc = vj(c0) * a0;
    acc = fmaf(vj(c1), a1, acc);
    acc = fmaf(vj(c2), a2, acc);
    acc = fmaf(vj(c3), a3, acc);
    return acc;
}

}  // namespace

__attribute__((amdgpu_flat_work_group_size(kBlockSize, kBlockSize)))
__global__ void bonsai_tq2_gemv_kernel(
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
    const size_t row_bytes = (size_t)num_blocks * (size_t)kTQ2BlockBytes;

    __shared__ __half x_lds[kWeightsPerB];

    // One fp32 accumulator per row owned by this wave.
    float acc[kRowsPerWarp];
    #pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) acc[r] = 0.0f;

    for (int b = 0; b < num_blocks; ++b) {
        // Cooperative load: 128 threads each write one fp16 activation.
        __syncthreads();
        x_lds[tid] = act[b * kWeightsPerB + tid];
        __syncthreads();

        // This lane's 4 activations (contiguous 4 fp16 = 8 B, distinct LDS
        // banks across the 32 lanes — no conflicts).
        const int a_base = lane * kWeightsPerLane;
        const float a0 = __half2float(x_lds[a_base + 0]);
        const float a1 = __half2float(x_lds[a_base + 1]);
        const float a2 = __half2float(x_lds[a_base + 2]);
        const float a3 = __half2float(x_lds[a_base + 3]);

        // For each of the 2 rows this wave owns: one coalesced 32-byte
        // load of qs (32 lanes × 1 byte) and one scalar load of `d`.
        #pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            const int row = row_base + wave * kRowsPerWarp + r;
            // Degenerate guard: tail workgroup may cover fewer than
            // kRowsPerWG rows. Use a predicated address so out-of-range
            // waves still participate in the syncthreads above (no
            // divergent sync) but do not corrupt memory.
            const bool row_valid = (row < N_out);
            const int row_eff    = row_valid ? row : 0;
            const uint8_t* __restrict__ blk =
                packed + (size_t)row_eff * row_bytes + (size_t)b * kTQ2BlockBytes;

            // `d` is uniform across the wave — compiler hoists to a
            // scalar load via uniform-address propagation.
            const float d = __half2float(load_scale_fp16(blk));

            // Each lane loads its own unique byte from qs[] — lanes 0..31
            // cover bytes 0..31, the full 32-byte qs region, in one
            // coalesced global-memory transaction per row.
            const uint32_t qbyte =
                (uint32_t)blk[kCodesOffset + lane];

            const float partial = decode4_mac(qbyte, a0, a1, a2, a3);
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

extern "C" void bonsai_tq2_gemv_launch(
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
    hipLaunchKernelGGL(bonsai_tq2_gemv_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       packed_weights,
                       reinterpret_cast<const __half*>(act_fp16),
                       reinterpret_cast<__half*>(out_fp16),
                       N_out, K_in);
    HIP_CHECK(hipGetLastError());
}
