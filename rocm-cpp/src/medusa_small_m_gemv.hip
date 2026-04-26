// medusa_small_m_gemv.hip — small-M ternary GEMV (halo-1bit byte pack).
//
// Body for the contract declared in include/rocm_cpp/medusa.h. M activations
// against halo-1bit packed ternary weights → fp16 output. The on-device
// weight layout is the SAME buffer the production M=1 GEMV consumes
// (kernels/ternary_gemv_phase5_halo.hip): uint8 [N, (K+3)/4] byte-pack with
// 4 ternary codes per byte (codes 0→-1, 1→0, 2→+1, 3→0) plus per-row fp32
// scales. Reusing the layout means Medusa verify and the M=1 decode share
// weight memory regions verbatim — no second copy in the model loader.
//
// Why this kernel exists
// ----------------------
// Calling ternary_gemv_phase5_halo M times in a host loop re-streams every
// weight byte M times from HBM. At M=tree_size and 92% of LPDDR5 peak
// already on the M=1 path, that's M× the bandwidth. This kernel fuses M
// activation rows so each weight byte is loaded once and FMA'd against M
// activation lanes from LDS.
//
// Tile shape (gfx1151 / wave32, RDNA 3.5)
// ---------------------------------------
//   BLOCK_M_MAX = 64  — matches RCPP_MEDUSA_MAX_TREE_SIZE in medusa.h.
//                       The acc[64] register footprint is the dominant
//                       occupancy gate (see "Register pressure" below).
//   BLOCK_N     = 64  — one block per N-tile, lane-cyclic across waves.
//   BLOCK_K     = 128 — 32 packed bytes per N-row per K-tile, exactly one
//                       byte per (lane_lo, lane_hi) pair → no idle lanes.
//   THREADS     = 128 — 4 waves × 32 lanes (wave32, gfx1151 first-class).
//
// Lane → output mapping per wave
// ------------------------------
//   Each wave owns N_PER_WAVE = BLOCK_N / 4 = 16 output cols.
//   Lane decomposition (wave32):
//     lane_lo = lane & 15           (0..15)  → output col within wave
//     lane_hi = lane >> 4           (0..1)   → which K-half this lane owns
//   So a (wave, lane_lo) pair { (wave, lane_lo+0), (wave, lane_lo+16) }
//   shares one output col and splits the K-axis 50/50. The two partial
//   accumulators reduce via __shfl_xor(acc, 16) at the epilogue.
//
//   Per K-tile (BLOCK_K=128 K-values = 32 packed bytes per N-row):
//     bytes_per_K_half = BLOCK_K / 8 = 16 packed bytes per lane_hi value.
//     Each lane walks ALL 16 of its K-half's bytes inside the K-tile —
//     lane_lo only selects the OUTPUT COLUMN, never the K-byte. Together
//     the lane_hi pair consumes all 32 bytes of weights for that output
//     column for the tile (= the full BLOCK_K K-positions). 16 lanes
//     sharing the same lane_hi all hit the SAME LDS u32 cell each step
//     → free broadcast on wave32 LDS (no bank conflict).
//
//     byte_in_row(kt, lane_hi, b) =
//         kt * (BLOCK_K / 4)              ── starting byte for this K-tile
//       + lane_hi * (BLOCK_K / 8)         ── which half of the K-tile
//       + b                                ── byte within the half (inner loop)
//
// LDS
// ---
//   x_lds[BLOCK_M_MAX][BLOCK_K] = 64 × 128 = 8192 B activations.
//   No weight LDS — weights stream directly from HBM via __restrict__
//   loads. Same pattern as ternary_gemv_phase5_halo (M=1) which is at
//   92% of LPDDR5 peak; the HBM controller's burst behavior already
//   handles weight streaming well, LDS staging there would only burn
//   waves on the load side.
//
//   Bank-conflict note: within a wave, all 16 lanes sharing the same
//   lane_hi access the SAME LDS u32 cell (x_u32_idx = lane_hi * 16 + b
//   is lane_lo-independent). That's a broadcast — wave32 LDS handles
//   this conflict-free. Across the inner b-loop the cell index advances
//   one u32 at a time (4 bytes), wave-uniform. Across the m-loop, the
//   row index changes; BLOCK_K=128 B / row = 32 u32 = 32 banks, so
//   adjacent rows alias the same bank set, but only one m is read per
//   cycle per lane, so no conflict.
//
// Bit-exactness vs scalar reference
// ---------------------------------
// The integer accumulator is bit-exact across this kernel and the scalar
// reference: sdot4 is a deterministic 4-way s8 dot product, identical
// element-wise to the unrolled scalar accumulator, and integer addition
// is associative so reduction order is irrelevant. The fp16 cast at the
// epilogue is the only place rounding can differ from a different
// reference order; tests/test_medusa_small_m_gemv.cpp tolerates ≤ 1 fp16
// ULP at the output stage (in practice it sees 0 ULPs because the
// epilogue fp32 → fp16 rounding chain matches the scalar reference's
// rounding chain exactly: same `(float)v * x_scale * row_scale →
// __float2half` fold).
//
// Register pressure
// -----------------
// acc[64] int32 + ww (u32) + scratch ≈ ~80-90 vregs/thread at -O3.
// gfx1151 SIMD32 has 1024 vregs/SIMD; with 128 threads/block (4 waves)
// occupancy lands at ~3-4 wavefronts/CU. Memory-bound at the BW ceiling
// on the M=1 baseline so 2-3 waves/CU is enough latency-hiding. If we
// observe spill traffic in -save-temps (`v_writelane`/`v_readlane`
// storms or `scratch_*`), the next tuning pass M-tiles into outer
// chunks of 16 and re-streams the weight byte across chunks — that
// trades a 4× weight-byte read amplification (still cached in L1)
// for half the register footprint.
//
// Why not WMMA?
// -------------
// gfx1151 has WMMA i8 16x16x32 fragments (wave32). Tempting for M ≤ 16,
// but for THIS kernel:
//   1. The B matrix (weights) lives in halo-1bit byte pack, not the
//      WMMA-permuted i8 layout. Repacking per K-tile costs more LDS
//      bandwidth than the matrix-core saves on a BW-bound kernel.
//   2. The M=1 reference (kernels/ternary_gemv_phase5_halo.hip) is at
//      92% of LPDDR5 peak today. The win for small-M is amortizing
//      weight reads across M, not raising compute throughput — sdot4
//      is already overkill for ternary × int8.
//   3. M ≤ 8 doesn't fill a 16x16 fragment; padding with zeros wastes
//      the 256-bit fragment register storage.
// If a future M ≥ 16 launcher needs raw FLOPs (e.g. for prefill cubed),
// a separate WMMA-based path lives in src/ternary_gemm_smallm.hip with
// its own 2-bit-per-u32 K-major pack.
//
// gfx1151 / wave32 first-class. No hipBLAS, no Python, stream-based
// sync only.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#include "rocm_cpp/medusa.h"

namespace {

constexpr int BLOCK_M_MAX = 64;
constexpr int BLOCK_N     = 64;
constexpr int BLOCK_K     = 128;
constexpr int THREADS     = 128;
constexpr int WAVE_SIZE   = 32;
constexpr int WAVES       = THREADS / WAVE_SIZE;       // 4
constexpr int N_PER_WAVE  = BLOCK_N / WAVES;           // 16

// Halo encoding: code 0 → -1, code 1 → 0, code 2 → +1, code 3 → 0.
// Pack 4 such codes (one byte) into 4 signed-int8 lanes of a u32 in the
// little-endian byte order sdot4 expects (lane-0 = bits 0..7, lane-1 =
// bits 8..15, …).
//
// The same routine lives in src/ternary_gemm_smallm.hip and
// kernels/ternary_gemv_phase5_halo.hip; intentionally duplicated here
// because both upstream copies are __forceinline__ in their own .hip TU
// and the cross-TU inlining story across HIP source sets is fragile.
// The bit-exact tests in test_medusa_small_m_gemv.cpp catch any
// divergence between halo kernels at integration time.
__device__ __forceinline__
uint32_t unpack4_halo_codes_to_signs_u32(uint8_t byte) {
    const uint32_t c0 =  byte        & 0x3u;
    const uint32_t c1 = (byte >> 2)  & 0x3u;
    const uint32_t c2 = (byte >> 4)  & 0x3u;
    const uint32_t c3 = (byte >> 6)  & 0x3u;
    const uint8_t s0 = (uint8_t)((int8_t)((int)(c0 == 2u) - (int)(c0 == 0u)));
    const uint8_t s1 = (uint8_t)((int8_t)((int)(c1 == 2u) - (int)(c1 == 0u)));
    const uint8_t s2 = (uint8_t)((int8_t)((int)(c2 == 2u) - (int)(c2 == 0u)));
    const uint8_t s3 = (uint8_t)((int8_t)((int)(c3 == 2u) - (int)(c3 == 0u)));
    return  (uint32_t)s0
         | ((uint32_t)s1 << 8)
         | ((uint32_t)s2 << 16)
         | ((uint32_t)s3 << 24);
}

__global__ __launch_bounds__(THREADS)
void medusa_small_m_gemv_kernel(
    const int8_t*  __restrict__ x_i8,
    float                       x_scale,
    const uint8_t* __restrict__ w_packed,
    const float*   __restrict__ w_row_scales,
    __half*        __restrict__ y_fp16_out,
    int M, int N, int K)
{
    const int tid     = threadIdx.x;
    const int lane    = tid & (WAVE_SIZE - 1);
    const int wave    = tid >> 5;

    const int lane_lo = lane & 15;             // 0..15  (output col idx)
    const int lane_hi = lane >> 4;             // 0..1   (K-half idx)

    const int n_local  = wave * N_PER_WAVE + lane_lo;
    const int n_global = blockIdx.x * BLOCK_N + n_local;

    const int packed_cols = K >> 2;            // K / 4 bytes per N-row

    // Out-of-N-bounds clamp: every lane runs the inner loop to keep
    // the wave uniform. The launcher rejects N % BLOCK_N != 0 so all
    // BLOCK_N lane positions in this block ARE in-bounds whenever the
    // block's lane-0 col is in-bounds. n_safe is defensive: it keeps
    // any lane that somehow receives n_global ≥ N pointed at row 0
    // (re-reading row 0 weights, accumulator discarded at epilogue).
    const int n_safe = (n_global < N) ? n_global : 0;

    __shared__ int8_t x_lds[BLOCK_M_MAX][BLOCK_K];

    int32_t acc[BLOCK_M_MAX] = {0};

    const int k_tiles = K / BLOCK_K;

    for (int kt = 0; kt < k_tiles; ++kt) {
        const int k_tile_start = kt * BLOCK_K;

        // ── Cooperative load of [M, BLOCK_K] activations into LDS ──
        //
        // Total slots per tile = BLOCK_M_MAX * (BLOCK_K / 4) = 64 * 32 =
        // 2048 u32, dispatched THREADS-cyclic. For M < BLOCK_M_MAX we
        // skip the upper (m_idx >= M) rows entirely — the inner m-loop
        // is gated on `m < M` and the LDS contents for those rows are
        // never read, so leaving them undefined (whatever the previous
        // K-tile wrote) is correct.
        //
        // u32 layout per LDS row: 32 u32 cells (BLOCK_K / 4 = 32).
        // flat = ii * THREADS + tid, ii ∈ [0, 16):
        //   m_idx  = flat / 32   → 0..63
        //   k4_idx = flat & 31   → 0..31  (u32 column within the row)
        //
        // Coalescing: within a wavefront (32 lanes), sequential lanes
        // hit sequential k4_idx values for the same m_idx — adjacent
        // 4-byte loads from x_i8 are coalesced by the L1 line packer.
        //
        // M-aware skip: for M=1 we hit only 32 u32 of LDS load (1 wave),
        // not all 2048 — the M=1 path stays close to the M=1 reference's
        // LDS bandwidth profile. The branch is uniform across the wave
        // when M is a multiple of 2 (which it always is for tree_size
        // values 1, 2, 4, ..., 64) so divergence cost is negligible.
        #pragma unroll
        for (int ii = 0; ii < (BLOCK_M_MAX * (BLOCK_K / 4)) / THREADS; ++ii) {
            const int flat   = ii * THREADS + tid;        // 0..2047
            const int m_idx  = flat >> 5;                 // 0..63
            const int k4_idx = flat & 31;                 // 0..31  (k4 = k/4)
            if (m_idx >= M) continue;
            const int k_global = k_tile_start + k4_idx * 4;
            const uint32_t val = *reinterpret_cast<const uint32_t*>(
                      &x_i8[(size_t)m_idx * (size_t)K + (size_t)k_global]);
            // Write through u32 view of the LDS row. Each LDS row is
            // BLOCK_K=128 bytes = 32 u32, so k4_idx is the u32 column.
            reinterpret_cast<uint32_t*>(&x_lds[m_idx][0])[k4_idx] = val;
        }
        __syncthreads();

        // ── Inner K-loop: this lane walks its 16 weight bytes (one K-half)
        // for its single output column (n_safe). lane_lo selects the
        // OUTPUT COLUMN; the b-loop walks the K-bytes. lane_hi splits the
        // K-tile in two and the partner reduce at the epilogue glues the
        // halves back together for each output column.
        //
        //   bytes_per_K_half = BLOCK_K / 8 = 16 bytes per lane_hi value.
        //   total bytes per (lane_hi pair, output col) per K-tile = 32
        //   = the full BLOCK_K = 128 K-values per output col per tile. ✓
        //
        // byte_in_row(b) = kt * (BLOCK_K / 4)     (32 bytes per K-tile)
        //                + lane_hi * (BLOCK_K / 8) (which half: 0 or 16)
        //                + b                       (0..15 within the half)
        //
        // The activation u32 paired with that byte sits at the
        // (lane_hi * (BLOCK_K / 8) + b)'th u32 in the LDS row — same K
        // position as the weight byte. lane_lo never indexes the K
        // axis, so all 16 lanes-per-lane_hi broadcast-read the same
        // LDS cell each iteration (free on wave32 LDS).
        constexpr int BYTES_PER_K_HALF = BLOCK_K / 8;   // 16
        const int byte_base  = kt * (BLOCK_K / 4)
                             + lane_hi * BYTES_PER_K_HALF;
        const int x_u32_base = lane_hi * BYTES_PER_K_HALF;
        const size_t w_row_base = (size_t)n_safe * (size_t)packed_cols;

        // Partial unroll of the b-loop (4×) keeps the unrolled body small
        // enough to avoid scratch spills under acc[64] register pressure.
        // Full unroll × full inner unroll (16 × 64 = 1024 sudot4 bodies)
        // overflows VGPRs; 4 × 8 = 32 bodies stays within the 256 VGPR
        // budget at 2-3 waves/SIMD occupancy on gfx1151.
        #pragma unroll 4
        for (int b = 0; b < BYTES_PER_K_HALF; ++b) {
            const uint8_t  byte = w_packed[w_row_base + (size_t)(byte_base + b)];
            const uint32_t ww   = unpack4_halo_codes_to_signs_u32(byte);
            const int x_u32_idx = x_u32_base + b;        // 0..15 or 16..31

            #pragma unroll 8
            for (int m = 0; m < BLOCK_M_MAX; ++m) {
                if (m >= M) break;
                const uint32_t xw = reinterpret_cast<const uint32_t*>(
                                        &x_lds[m][0])[x_u32_idx];
                // sudot4 (signed × signed → int32) — matches the rest of
                // the rocm-cpp ternary path (kernels/ternary_gemv_phase5_*,
                // src/ternary_gemm_smallm) which all use sudot4 with
                // both-args-signed=true. gfx1151 lacks the dot1-insts
                // sdot4 variant; sudot4 is the right intrinsic. Clamp=false
                // → wrap on overflow, deterministic. With ternary weights
                // and int8 activations, max |acc| at K=2560 is bounded by
                // 2560 * 127 ≪ 2^31, no overflow risk; even at K=131072
                // we have headroom (131072 * 127 = 16.6M ≪ 2^31).
                acc[m] = __builtin_amdgcn_sudot4(true, ww, true, xw, acc[m], false);
            }
        }
        __syncthreads();
    }

    // ─── Epilogue: cross-lane reduce + scale + fp16 store ───────────────
    //
    // Reduce across the lane_hi=0 vs lane_hi=1 partner pair so each
    // lane_lo ends up with the full sum. __shfl_xor(16) on wave32
    // swaps acc[m] between lane and lane^16 — exactly the (lane_hi,
    // lane_lo) → (1-lane_hi, lane_lo) partner. After the xor each
    // lane holds the SUM of its pair's halves.
    //
    // Bit-exact vs the M=1 reference: integer addition is associative,
    // so swapping the K-axis reduction order between the two paths
    // (lane-cyclic 16-vals chunks vs lane_hi-pair-then-lane_lo) leaves
    // the int32 accumulator unchanged.
    #pragma unroll 8
    for (int m = 0; m < BLOCK_M_MAX; ++m) {
        if (m >= M) break;
        acc[m] += __shfl_xor(acc[m], 16);
    }

    // Only lane_hi=0 writes (avoids redundant store + makes the store
    // address sequential within the wavefront). Out-of-bounds n_global
    // lanes also skip the write.
    if (lane_hi != 0) return;
    if (n_global >= N) return;

    const float row_scale = w_row_scales[n_global];

    #pragma unroll 8
    for (int m = 0; m < BLOCK_M_MAX; ++m) {
        if (m >= M) break;
        // Same scaling order as kernels/ternary_gemv_phase5_halo.hip's
        // ternary_gemv_phase5_halo_f16_kernel epilogue:
        //   y = __float2half((float)v * x_scale * scales[my_row]);
        // Matching the multiplication order is what gives M=1 fp16-bit
        // exactness vs the M=1 reference path (the test in
        // tests/test_medusa_small_m_gemv.cpp has an `M=1 sanity` shape
        // explicitly for this).
        const float fp = (float)acc[m] * x_scale * row_scale;
        y_fp16_out[(size_t)m * (size_t)N + (size_t)n_global] = __float2half(fp);
    }
}

}  // namespace

extern "C"
rcpp_status_t
rcpp_medusa_small_m_gemv(const void* x_i8_dev,
                         float       x_scale,
                         const void* w_packed_halo_dev,
                         const void* w_row_scales_dev,
                         void*       y_fp16_out_dev,
                         int         M,
                         int         N,
                         int         K,
                         void*       stream)
{
    if (!x_i8_dev || !w_packed_halo_dev || !w_row_scales_dev || !y_fp16_out_dev)
        return RCPP_INVALID_ARG;
    if (M <= 0 || N <= 0 || K <= 0)            return RCPP_INVALID_ARG;
    if (M > BLOCK_M_MAX)                       return RCPP_UNSUPPORTED;
    if (M > RCPP_MEDUSA_MAX_TREE_SIZE)         return RCPP_UNSUPPORTED;
    if (N % BLOCK_N != 0)                      return RCPP_INVALID_ARG;
    if (K % BLOCK_K != 0)                      return RCPP_INVALID_ARG;

    dim3 grid(N / BLOCK_N, 1, 1);
    dim3 block(THREADS, 1, 1);
    hipLaunchKernelGGL(medusa_small_m_gemv_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const int8_t*>(x_i8_dev),
                       x_scale,
                       static_cast<const uint8_t*>(w_packed_halo_dev),
                       static_cast<const float*>(w_row_scales_dev),
                       static_cast<__half*>(y_fp16_out_dev),
                       M, N, K);
    return RCPP_OK;
}
