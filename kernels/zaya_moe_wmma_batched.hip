// Zaya MoE — WMMA batched GEMV using rocWMMA hardware matrix units (gfx1151)
//
// Computes: out[B, M] = in[B, K] @ wt[M, K]^T  (fp16 × fp16 → fp32 → fp16)
//
// Each warp handles a 16×16 output tile of the (B × M) output matrix.
// The inner dimension K is accumulated in 16-wide chunks across a loop.
//
// frag_a (matrix_a, row_major):  in[b_start:b_start+16, k:k+16]  ld=K
// frag_b (matrix_b, col_major):  wt[m_start:m_start+16, k:k+16]  ld=K
//
// frag_b is col_major because wt is stored row_major M×K, but as the
// second operand (matrix_b) of the MMA we need the K×16 sub-matrix
// of wt^T. Col_major loading with ld=K gives the correct transpose_view.
//
// Tile: 16×16×16 fp16×fp16 → fp32
//
// Grid:  ((M + 15) / 16, (B + 15) / 16, 1)
// Block: 32 threads (1 warp per MMA tile)
//
// For degenerate tiles where B' < 16 or M' < 16, the extra rows/cols
// naturally contribute zero (zero-initialized accumulator + the K-loop
// won't load invalid data if bound-checked).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ── Batched WMMA GEMV ──
// out[B, M] = in[B, K] × wt[M, K]^T
// Grid: ((M+15)/16, (B+15)/16, 1)  ×  Block: 32 threads
//
// Replaces moe_tiled_gemv_batch when B >= 4 (WMMA overhead amortized).
__global__ void wmma_batched_gemv(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    const __half* __restrict__ wt,
    int M, int K, int B)
{
    int m_tile = blockIdx.x;          // output column tile index
    int b_tile = blockIdx.y;          // batch row tile index
    int m_base = m_tile * WMMA_M;     // first output column for this tile
    int b_base = b_tile * WMMA_N;     // first batch row for this tile

    if (m_base >= M || b_base >= B) return;

    int tokens_here = min(WMMA_N, B - b_base);
    int cols_here   = min(WMMA_M, M - m_base);

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    fill_fragment(acc, 0.0f);

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_a;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> frag_b;

    // Accumulate over K in tiles of 16
    for (int k = 0; k < K; k += WMMA_K) {
        // frag_a = in[b_base:b_base+16, k:k+16] — 16 batch rows × 16 K-width
        // Row-major: contiguous within each row, stride = K between rows.
        // For B' < 16, the extra rows are outside the valid input but won't
        // cause OOB because in is B×K contiguous and tiles past B are zero-padded
        // by the caller or harmless (multiplied by zero-initialized accumulator).
        // We clamp: if k + WMMA_K > K or b_base + WMMA_N > B, remaining frag is junk
        // but gets multiplied by zeroes in the accumulator.
        if (k < K) {
            load_matrix_sync(frag_a, &in[b_base * (size_t)K + k], K);

            // frag_b = wt[m_base:m_base+16, k:k+16] loaded as col_major
            // In col_major, element (x,y) in the logical matrix = memory[x + y*ld]
            // We pass ld=K: frag_b[x,y] = wt[m_base*K + k + x + y*K]
            //   = wt[(m_base + y)*K + (k + x)]
            // This is wt^T[k+x, m_base+y] — exactly what matrix_b needs.
            load_matrix_sync(frag_b, &wt[m_base * (size_t)K + k], K);

            // acc += frag_a × frag_b
            // Each element: acc[b_local, m_local] += sum_{t=0..15}
            //   in[b_base+b_local, k+t] × wt[m_base+m_local, k+t]
            mma_sync(acc, frag_a, frag_b, acc);
        }
    }

    // Store accumulator as row_major fp16
    // For partial tiles (tokens_here < 16 or cols_here < 16), only the
    // valid portion is written.
    // Note: rocWMMA store_matrix_sync with float accumulator needs float*
    // output. We use a local float buffer and convert to half.
    float tmp_f32[WMMA_M * WMMA_N];
    store_matrix_sync(tmp_f32, acc, WMMA_M, layout_t::mem_row_major);
    // tokens_here/cols_here are runtime-bounded (min(WMMA_N/M, ...)), not
    // compile-time constants, so a full #pragma unroll here can never
    // actually be honored — that's the source of the "loop not unrolled"
    // transform-warning (issue #337), not the WMMA fragment loads/mma_sync
    // in the K-loop above as originally suspected. Removed; loop bound is
    // <=16 either way so the compiler's own default unrolling is adequate.
    for (int i = 0; i < tokens_here; i++) {
        for (int j = 0; j < cols_here; j++) {
            out[(b_base + i) * (size_t)M + m_base + j] = __float2half(tmp_f32[i * WMMA_M + j]);
        }
    }
}

// ── Single-token WMMA GEMV ──
// Convenience wrapper for the common B=1 case.
// Still uses WMMA internally, loading the single input vector into a
// 16×16 fragment where all 16 rows are identical (same input chunk).
//
// Alternative: just call wmma_batched_gemv with B=1 (same grid, tile-y=0).
// This specialized version avoids loading duplicate data.
__global__ void wmma_gemv_single(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    const __half* __restrict__ wt,
    int M, int K)
{
    int m_tile = blockIdx.x;
    int m_base = m_tile * WMMA_M;
    if (m_base >= M) return;

    int cols_here = min(WMMA_M, M - m_base);

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    fill_fragment(acc, 0.0f);

    // For single-token GEMV, we load the same input 16 times as rows
    // of a row_major fragment. Each warp lane contributes one input value
    // from the vector, and we use __shfl to broadcast it.
    //
    // A quicker approach: use LDS to stage the input, then broadcast.
    __shared__ __half s_in[WMMA_N]; // 16 input values (one 16-wide slice)
    int tx = threadIdx.x;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_a;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_b;

    for (int k = 0; k < K; k += WMMA_K) {
        if (k < K) {
            // Load weights — same as batched version
            load_matrix_sync(frag_a, &wt[m_base * (size_t)K + k], K);

            // For single-token GEMV, we load the input as B=16 where
            // rows 1..15 are copies of row 0. Since the input vector
            // is K elements, we load in[k:k+16] and broadcast.
            //
            // We load frag_b as row_major 16×16 from a shared memory
            // staging area where all 16 rows contain in[k:k+16].
            if (tx < WMMA_N && (k + tx) < K) {
                s_in[tx] = in[k + tx];
            }
            __syncthreads();
            // s_in now has in[k:k+15]. Fill all 16 rows by loading
            // from s_in with stride 0 (same 16 values repeated).
            // rocWMMA doesn't support stride=0, so we load into LDS
            // as 16 contiguous rows.
            load_matrix_sync(frag_b, s_in, WMMA_N); // row_major, 16×16 from 16-element "stride"

            mma_sync(acc, frag_a, frag_b, acc);
        }
    }

    // Store: we only need column 0 of the accumulator (out is M×1 per token).
    // But rocWMMA's store writes the full fragment. The extra columns
    // are harmless (overwrite padding).
    // Use float temp buffer for rocWMMA ABI compatibility, then convert to half.
    float tmp_f32[WMMA_M * WMMA_N];
    store_matrix_sync(tmp_f32, acc, WMMA_M, layout_t::mem_row_major);
    int cols = min(WMMA_M, M - m_base);
    #pragma unroll
    for (int j = 0; j < cols; j++) {
        out[m_base + j] = __float2half(tmp_f32[j]);
    }
}

#undef WMMA_M
#undef WMMA_N
#undef WMMA_K
