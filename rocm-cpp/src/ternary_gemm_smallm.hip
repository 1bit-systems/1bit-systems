// ternary_gemm_smallm.hip — Tiled small-M ternary GEMM for Medusa tree-verify.
//
// Why this kernel exists
// ----------------------
// Medusa-1 speculative decoding verifies a tree of candidate tokens in one
// backbone pass, turning the batch=1 GEMV into a small-M GEMM (M ∈ [1,16]).
// The production GEMV at kernels/ternary_gemv_phase5_halo.hip is at 92% of
// LPDDR5 peak at M=1; calling it M times re-streams the weight tensor M
// times and eats the speculative win. This kernel fuses the M queries so
// each ternary weight u32 is loaded from HBM once and reused across all M
// activations via LDS staging.
//
// Layout contract (matches the architect's pinned signature)
// ----------------------------------------------------------
//   x_i8           : [M, K]       int8, row-major.
//   x_scale        :               float32 per-tensor activation scale.
//   w_packed       : [K/16, N]     uint32, 16 ternary codes packed per u32.
//                                  Index: w_packed[(k/16) * N + n]; code_j
//                                  = (word >> (2*j)) & 0x3 corresponds to
//                                  k = (k/16)*16 + j at output column n.
//                                  Codes: 0→-1, 1→0, 2→+1, 3→0 (halo-1bit).
//   w_row_scales   : [N]           one fp32 scale per output column.
//   y_out          : [M, N]        hip_bfloat16, row-major.
//
// The scalar reference (ternary_gemm_smallm_scalar_ref.hip) shares this ABI
// exactly. The test harness diffs this kernel against that reference.
//
// Tile shape (architect-pinned)
// -----------------------------
//   BLOCK_M = 16  BLOCK_N = 64  BLOCK_K = 64
//   128 threads / block = 4 waves × 32 lanes (wave32, gfx1151).
//   Each wave owns a 16×16 output tile → 8 int32 accumulators per lane.
//   LDS: 1024 B activations + 1024 B weights = 2 KiB per block.
//   Target occupancy: 2 waves/SIMD — fits in gfx1151 VGPR/LDS budget.
//
// Inner loop: sdot4 on packed int8 (4 K's per op), 16 k4-chunks × 8
// accumulators = 128 sudot4 per lane per K-tile. K-tile count = K/BLOCK_K
// (e.g. 2560/64 = 40), so total 5120 sudot4 per lane for a 2560-K problem.
//
// Preconditions
// -------------
//   K % 64 == 0, N % 64 == 0, 1 ≤ M ≤ 16. 2560×2560 BitNet-b1.58-2B satisfies
//   both. Caller must pad if the Medusa head picks odd shapes; this kernel
//   does not handle ragged tiles.
//
// Non-goals
// ---------
//   * Not wired into Engine::forward_token — stays standalone until the
//     Medusa backend is enabled. (Was halo-router.forward_token; router
//     retired 2026-04-25, Engine lives in-process inside lemond.)
//   * Not a WMMA kernel. gfx1151 WMMA int8 is 16x16x32 which would fit,
//     but the VGPR cost for 4 WMMA accumulator tiles at 2-waves/SIMD is
//     higher than sdot4's 256-VGPR total. sdot4 also stays bit-compatible
//     with the existing GEMV accumulator precision (i32).

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>

namespace {

constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 64;
constexpr int THREADS = 128;          // 4 waves × 32 lanes
constexpr int WAVE_SIZE = 32;
constexpr int WAVES_PER_BLOCK = THREADS / WAVE_SIZE;
constexpr int N_PER_WAVE = BLOCK_N / WAVES_PER_BLOCK;  // 16
constexpr int ACCS_PER_LANE = 8;       // 16×16 tile ÷ 32 lanes
constexpr int K_CHUNKS_PER_TILE = BLOCK_K / 4;  // 16 sdot4 steps per K-tile

// Unpack 4 halo-1bit ternary codes packed in one byte (2 bits × 4) into a
// u32 holding 4 signed int8 values at positions 0,8,16,24.
// Halo encoding: sign = (code == 2) - (code == 0); code 1 → 0; code 3 → 0.
static __device__ __forceinline__ uint32_t unpack4_codes_to_signs(uint8_t byte) {
    const uint32_t c0 =  byte        & 0x3u;
    const uint32_t c1 = (byte >> 2)  & 0x3u;
    const uint32_t c2 = (byte >> 4)  & 0x3u;
    const uint32_t c3 = (byte >> 6)  & 0x3u;
    // (c == 2) - (c == 0) produces {-1, 0, +1}. Cast to uint8 then pack so
    // negative values become 0xFF (int8 sign-extended), matching sudot4's
    // signed-a interpretation.
    const uint8_t s0 = (uint8_t)((int8_t)((int)(c0 == 2u) - (int)(c0 == 0u)));
    const uint8_t s1 = (uint8_t)((int8_t)((int)(c1 == 2u) - (int)(c1 == 0u)));
    const uint8_t s2 = (uint8_t)((int8_t)((int)(c2 == 2u) - (int)(c2 == 0u)));
    const uint8_t s3 = (uint8_t)((int8_t)((int)(c3 == 2u) - (int)(c3 == 0u)));
    return  (uint32_t)s0
         | ((uint32_t)s1 << 8)
         | ((uint32_t)s2 << 16)
         | ((uint32_t)s3 << 24);
}

} // namespace

extern "C" {

__global__
__attribute__((amdgpu_flat_work_group_size(THREADS, THREADS)))
void onebit_ternary_gemm_smallm_kernel(
    const int8_t*   __restrict__ x_i8,
    float                        x_scale,
    const uint32_t* __restrict__ w_packed,
    const float*    __restrict__ w_row_scales,
    hip_bfloat16* __restrict__ y_out,
    int M, int N, int K)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (WAVE_SIZE - 1);
    const int wave = tid >> 5;

    const int block_n_base = blockIdx.x * BLOCK_N;
    // Block y is not used — we fuse all M into one block per N-tile.

    // ── Lane → (m, n_tile) mapping within the wave's 16×16 tile ──────────────
    //   m = lane & 15                       (row within M)
    //   lane_half = lane >> 4               (0 or 1)
    //   each lane owns ACCS_PER_LANE n columns starting at n_base_in_wave:
    //       n_base_in_wave = lane_half * ACCS_PER_LANE   (0 or 8)
    //   Absolute output n for accumulator `a`:
    //       n_abs = block_n_base + wave * N_PER_WAVE + n_base_in_wave + a
    const int m_lane         = lane & 15;
    const int n_base_in_wave = (lane >> 4) * ACCS_PER_LANE;
    const int n_wave_base    = wave * N_PER_WAVE;

    int32_t acc[ACCS_PER_LANE] = {0};

    // ── LDS: 1 KiB activations + 1 KiB weight codes ─────────────────────────
    __shared__ int8_t   x_lds[BLOCK_M][BLOCK_K];              // 1024 B
    __shared__ uint32_t w_lds[BLOCK_K / 16][BLOCK_N];         // 1024 B

    const int k_tiles = K / BLOCK_K;                          // K multiple of 64

    for (int kt = 0; kt < k_tiles; ++kt) {
        const int k_tile_start = kt * BLOCK_K;

        // ─ Stage weights: 4 × 64 = 256 u32, 128 threads → 2 u32 each ────────
        #pragma unroll
        for (int ii = 0; ii < 2; ++ii) {
            const int flat    = ii * THREADS + tid;           // 0..255
            const int kb_off  = flat >> 6;                    // 0..3
            const int n_local = flat & (BLOCK_N - 1);         // 0..63
            const int k_block_global = (k_tile_start >> 4) + kb_off;
            const int n_global = block_n_base + n_local;
            // N and K multiples of BLOCK_N/BLOCK_K by precondition.
            w_lds[kb_off][n_local] =
                w_packed[(size_t)k_block_global * (size_t)N + (size_t)n_global];
        }

        // ─ Stage activations: 16 × 64 = 1024 B = 256 u32, 2 per thread ──────
        // Each u32 = 4 consecutive int8 at (m_idx, k_tile_start + k4_idx*4).
        // Zero-fill rows m_idx >= M so invalid lanes do harmless work.
        #pragma unroll
        for (int ii = 0; ii < 2; ++ii) {
            const int flat   = ii * THREADS + tid;            // 0..255
            const int m_idx  = flat >> 4;                     // 0..15
            const int k4_idx = flat & 15;                     // 0..15
            uint32_t val = 0u;
            if (m_idx < M) {
                const int k_global = k_tile_start + k4_idx * 4;
                val = *reinterpret_cast<const uint32_t*>(
                          &x_i8[(size_t)m_idx * (size_t)K + (size_t)k_global]);
            }
            // Store as u32 into the LDS row. int8[16][64] row stride = 64 bytes.
            reinterpret_cast<uint32_t*>(&x_lds[m_idx][0])[k4_idx] = val;
        }

        __syncthreads();

        // ─ Inner compute: 16 k4-chunks × 8 accumulators = 128 sudot4 /lane ──
        // For each 4-K chunk within BLOCK_K:
        //   k4 ∈ [0, 16):  k_lo = k4*4, hits byte `k4 & 3` of w_lds[k4>>2][*].
        // x_lds row is m_lane (same across all accumulators of this lane).
        const uint32_t* __restrict__ x_row_u32 =
            reinterpret_cast<const uint32_t*>(&x_lds[m_lane][0]);

        #pragma unroll
        for (int k4 = 0; k4 < K_CHUNKS_PER_TILE; ++k4) {
            const uint32_t xw = x_row_u32[k4];
            const int kb_off  = k4 >> 2;         // 0..3 : which 16-K block
            const int chunk   = k4 & 3;          // 0..3 : byte within u32
            const int shift   = chunk * 8;

            #pragma unroll
            for (int a = 0; a < ACCS_PER_LANE; ++a) {
                const int n_local = n_wave_base + n_base_in_wave + a;
                const uint32_t wword = w_lds[kb_off][n_local];
                const uint8_t  byte  = (uint8_t)((wword >> shift) & 0xFFu);
                const uint32_t ww    = unpack4_codes_to_signs(byte);
                // sudot4(signed a, signed b, acc, clamp=false)
                acc[a] = __builtin_amdgcn_sudot4(true, ww, true, xw, acc[a], false);
            }
        }

        __syncthreads();
    }

    // ── Epilogue: i32 → fp32 → bf16, store if in-bounds ─────────────────────
    if (m_lane >= M) return;      // invalid lanes contribute no output

    #pragma unroll
    for (int a = 0; a < ACCS_PER_LANE; ++a) {
        const int n_local = n_wave_base + n_base_in_wave + a;
        const int n_abs   = block_n_base + n_local;
        if (n_abs >= N) continue;
        const float fp = (float)acc[a] * x_scale * w_row_scales[n_abs];
        y_out[(size_t)m_lane * (size_t)N + (size_t)n_abs] = hip_bfloat16(fp);
    }
}

// Host wrapper — stream-aware, no hipDeviceSynchronize, no allocations.
// Caller is responsible for error checking via hipGetLastError() after.
void onebit_ternary_gemm_smallm_launch(
    const int8_t*   x_i8,
    float           x_scale,
    const uint32_t* w_packed,
    const float*    w_row_scales,
    hip_bfloat16* y_out,
    int M, int N, int K,
    hipStream_t stream)
{
    if (M <= 0 || N <= 0 || K <= 0) return;
    if (M > BLOCK_M) return;           // architect-pinned contract: M ≤ 16
    if (N % BLOCK_N != 0) return;      // see preconditions comment
    if (K % BLOCK_K != 0) return;

    dim3 block(THREADS, 1, 1);
    dim3 grid(N / BLOCK_N, 1, 1);

    onebit_ternary_gemm_smallm_kernel<<<grid, block, 0, stream>>>(
        x_i8, x_scale, w_packed, w_row_scales, y_out, M, N, K);
}

} // extern "C"
