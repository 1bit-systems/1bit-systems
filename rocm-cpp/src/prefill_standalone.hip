// Standalone FP16 x pk_i4(ternary) -> FP16 GEMM for gfx1151.
// No CK headers, no CK symbols. Correctness-first; performance arrives in
// Phases 2-4.
//
// Layout (matches CK path exactly):
//   A      : FP16 row-major [M, K]   stride K
//   B_pack : pk_i4 bytes, WMMA-permuted for the same CK-style layout:
//              block-reshape K -> [K0, N, K1]  with K1 = 32
//              within-8 nibble permute (01234567 -> 20643175)
//              HIGH nibble of each byte holds the first (lower-index) element
//              (matches CK_USE_PK4_LAYOUT_SHUFFLE = 1 in ck.hpp)
//   C      : FP16 row-major [M, N]   stride N
//
// Nibble-to-FP16 decode matches CK: value = nibble - 8 (so 0x7->-1, 0x8->0, 0x9->+1)
//
// This TU intentionally has zero ck/ includes. Link: only HIP runtime.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#include "rocm_cpp/ck_gemm.h"   // C API — shared between CK and standalone backends

namespace {

constexpr int K1       = 32;  // KPerBlock in CK parlance
constexpr int K1_BYTES = K1 / 2;

// Forward permute (output slot -> input slot), per the CK upstream pk_i4 example:
//   slot 0 <- input 2   slot 4 <- input 3
//   slot 1 <- input 0   slot 5 <- input 1
//   slot 2 <- input 6   slot 6 <- input 7
//   slot 3 <- input 4   slot 7 <- input 5
// Inverse (input slot -> output slot) = {1, 5, 0, 4, 3, 7, 2, 6}.
__device__ __constant__ int kInversePerm[8] = {1, 5, 0, 4, 3, 7, 2, 6};

// Naive GEMM: one thread per output element. A loop of K inside each thread.
// BLOCK_M x BLOCK_N = 16 x 16 output tile per block, 256 threads per block.
__global__ __launch_bounds__(256)
void standalone_ternary_gemm_naive(
    const __half* __restrict__ A,       // [M, K] row-major
    const uint8_t* __restrict__ B,      // [K*N/2] WMMA-permuted pk_i4
    __half* __restrict__ C,             // [M, N] row-major
    int M, int N, int K)
{
    const int m = blockIdx.y * 16 + threadIdx.y;
    const int n = blockIdx.x * 16 + threadIdx.x;
    if(m >= M || n >= N) return;

    const int K0 = K / K1;
    float acc = 0.0f;

    for(int j = 0; j < K0; ++j) {
        // Base offset into B for (block j, column n): j*N*K1/2 + n*K1/2 bytes.
        const int b_base = j * N * K1_BYTES + n * K1_BYTES;

        for(int jj = 0; jj < K1; ++jj) {
            const int k_orig = j * K1 + jj;
            const int group_base = (jj / 8) * 8;
            const int slot_in    = jj & 7;
            const int perm_slot  = kInversePerm[slot_in];
            const int perm_jj    = group_base + perm_slot;

            // Byte inside this (j, n) block of K1/2 bytes, plus nibble select.
            const int byte_idx  = b_base + (perm_jj >> 1);
            const uint8_t byte  = B[byte_idx];
            const uint8_t nib   = ((perm_jj & 1) == 0)   // even perm_jj -> HIGH nibble
                                    ? ((byte >> 4) & 0xF)
                                    : (byte & 0xF);
            const float bval = (float)((int)nib - 8);   // -8 bias

            acc += (float)A[m * K + k_orig] * bval;
        }
    }

    C[m * N + n] = __float2half(acc);
}

// Phase 2: LDS-tiled GEMM.
//   Block tile: BLOCK_M x BLOCK_N = 16 x 16 output elements
//   K tile     : K1 = 32 per step (matches the block-permute K1)
//   Threads    : 256 per block (16 x 16, one output per thread)
//   LDS        : A_tile [16, 32] FP16 = 1 KiB
//                B_tile [32, 16] FP16 = 1 KiB  (decoded on-the-fly at load)
// Each outer K step corresponds exactly to one "j" block in CK's K0 iteration.
// The within-8 nibble permute is unwound during the LDS load so the LDS-side
// B_tile[kk, n] is naturally K-contiguous for the accumulator loop.
__global__ __launch_bounds__(256)
void standalone_ternary_gemm_lds(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    constexpr int TILE_K = 32;   // == K1

    __shared__ __half A_lds[TILE_M * TILE_K];   // [TILE_M, TILE_K] row-major
    __shared__ __half B_lds[TILE_K * TILE_N];   // [TILE_K, TILE_N] row-major

    const int tx = threadIdx.x;       // 0..TILE_N-1
    const int ty = threadIdx.y;       // 0..TILE_M-1
    const int tid = ty * TILE_N + tx; // 0..255

    const int m0 = blockIdx.y * TILE_M;   // block's row origin
    const int n0 = blockIdx.x * TILE_N;   // block's col origin
    const int m  = m0 + ty;
    const int n  = n0 + tx;

    float acc = 0.0f;
    const int K0 = K / TILE_K;

    // 512 LDS elements per side, 256 threads -> 2 elements per thread per side.
    // Strided cooperative load keeps coalescing-friendly lanes.
    for(int j = 0; j < K0; ++j) {
        // --- Load A tile: [TILE_M, TILE_K] = 16 x 32 ---
        // Covers A[m0..m0+16, j*32..j*32+32]. Load 2 elements per thread.
        {
            const int base = m0 * K + j * TILE_K;
            // Layout: row r in 0..15, col c in 0..31 -> index r*32 + c
            const int slot0 = tid;             // 0..255
            const int slot1 = tid + 256;       // 256..511
            const int r0 = slot0 / TILE_K;
            const int c0 = slot0 % TILE_K;
            const int r1 = slot1 / TILE_K;
            const int c1 = slot1 % TILE_K;
            A_lds[slot0] = ((m0 + r0) < M)
                ? A[(m0 + r0) * K + j * TILE_K + c0] : (__half)0;
            A_lds[slot1] = ((m0 + r1) < M)
                ? A[(m0 + r1) * K + j * TILE_K + c1] : (__half)0;
        }

        // --- Load B tile: [TILE_K, TILE_N] = 32 x 16 (decoded FP16) ---
        // B on-device layout for block j, columns [n0..n0+16]:
        //   byte base = j * N * K1_BYTES + col * K1_BYTES, K1_BYTES = 16
        // Each column brings 32 nibbles = 16 bytes.
        // 16 cols x 16 bytes = 256 bytes — perfect for 256 threads (1 byte each).
        {
            const int col_in_tile = tid / K1_BYTES;    // 0..15 -> which of TILE_N cols
            const int byte_in_col = tid % K1_BYTES;    // 0..15 -> which of 16 bytes
            const int global_col  = n0 + col_in_tile;
            if(global_col < N) {
                const int byte_idx = j * N * K1_BYTES + global_col * K1_BYTES + byte_in_col;
                const uint8_t b    = B[byte_idx];
                // Byte holds two nibbles at perm_jj = byte_in_col*2 (HIGH) and *2+1 (LOW).
                // Reverse the within-8 permute to map perm_jj back to original jj,
                // which is the K-axis index (= kk in the LDS layout).
                //   forward_perm[s] = input slot for output slot s — {2,0,6,4,3,1,7,5}
                const int forward_perm[8] = {2, 0, 6, 4, 3, 1, 7, 5};
                #pragma unroll
                for(int nibble = 0; nibble < 2; ++nibble) {
                    const int perm_jj = byte_in_col * 2 + nibble;  // 0..31
                    const int group_base = (perm_jj / 8) * 8;
                    const int slot_out   = perm_jj & 7;
                    const int orig_slot  = forward_perm[slot_out];
                    const int kk         = group_base + orig_slot;  // original K within block
                    const uint8_t nib = (nibble == 0)
                        ? ((b >> 4) & 0xF)   // HIGH nibble = first element in byte
                        : (b & 0xF);
                    const float v = (float)((int)nib - 8);
                    B_lds[kk * TILE_N + col_in_tile] = __float2half(v);
                }
            } else {
                #pragma unroll
                for(int nibble = 0; nibble < 2; ++nibble) {
                    const int perm_jj = byte_in_col * 2 + nibble;
                    const int group_base = (perm_jj / 8) * 8;
                    const int forward_perm[8] = {2, 0, 6, 4, 3, 1, 7, 5};
                    const int kk = group_base + forward_perm[perm_jj & 7];
                    B_lds[kk * TILE_N + col_in_tile] = (__half)0;
                }
            }
        }

        __syncthreads();

        // --- Accumulate TILE_K FMAs from LDS ---
        if(m < M && n < N) {
            #pragma unroll
            for(int kk = 0; kk < TILE_K; ++kk) {
                const float a = (float)A_lds[ty * TILE_K + kk];
                const float b = (float)B_lds[kk * TILE_N + tx];
                acc += a * b;
            }
        }

        __syncthreads();
    }

    if(m < M && n < N) {
        C[m * N + n] = __float2half(acc);
    }
}

// Phase 3: WMMA-backed GEMM on gfx1151.
//   One wave (32 threads) = one block = one 16x16 output tile.
//   K tile = 16 per WMMA step.
//   Accumulator: FP32 (f32-accum variant avoids the OpSel dance).
//
// RDNA 3 Wave32 WMMA fragment layout (documented in AMD ISA):
//   A[16, 16] FP16: each lane holds half16_t. Lanes 0..15 hold row (lane_id)
//     of A (all 16 cols). Lanes 16..31 duplicate the same data.
//   B[16, 16] FP16: each lane holds half16_t. Lanes 0..15 hold col (lane_id)
//     of B (all 16 rows). Lanes 16..31 duplicate.
//   C[16, 16] FP32: each lane holds float8_t = 8 outputs.
//     lane i, slot r in 0..7 -> C[2*r + (i >= 16), i % 16]
//     (interleaved rows: lanes 0..15 own even rows, lanes 16..31 own odd rows.)
//
// Builtin:
//   float8_t __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(half16_t A, half16_t B, float8_t C);
using half16_t  = __attribute__((ext_vector_type(16))) __fp16;
using float8_t  = __attribute__((ext_vector_type(8))) float;

__global__ __launch_bounds__(32)
void standalone_ternary_gemm_wmma(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    constexpr int WMMA_K = 16;  // WMMA K step; also K1/2

    const int lane = threadIdx.x;      // 0..31
    const int row_lane = lane & 15;    // 0..15 — row index for A / col index for B
    const int half = lane >> 4;        // 0 or 1 — duplicate half

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    // Accumulator
    float8_t acc = {};

    const int K0 = K / K1;             // number of j blocks
    const int steps_per_block = K1 / WMMA_K;  // 2 WMMA steps per K1-block
    const int jj_per_step     = WMMA_K;       // 16 original K per step

    const int forward_perm[8] = {2, 0, 6, 4, 3, 1, 7, 5};

    for(int j = 0; j < K0; ++j) {
        // Each K1 block (32 elements along K axis) = 2 WMMA steps of 16.
        for(int step = 0; step < steps_per_block; ++step) {
            const int jj_lo = step * WMMA_K;            // 0 or 16
            // Original K range for this step: [j*K1 + jj_lo, j*K1 + jj_lo + 16)

            // ------- Build A fragment -------
            // Each of 32 lanes holds row (lane % 16) of A, all 16 cols for
            // this K step. Lanes 16..31 duplicate lanes 0..15.
            half16_t A_frag;
            {
                const int m_row = m0 + row_lane;
                if(m_row < M) {
                    #pragma unroll
                    for(int c = 0; c < 16; ++c) {
                        const int k_orig = j * K1 + jj_lo + c;
                        A_frag[c] = (__fp16)(float)A[m_row * K + k_orig];
                    }
                } else {
                    #pragma unroll
                    for(int c = 0; c < 16; ++c) A_frag[c] = (__fp16)0;
                }
            }

            // ------- Build B fragment -------
            // Each of 32 lanes holds col (lane % 16) of B for this K step.
            // B on device: for block j, col (n0 + row_lane), 16 bytes per col.
            half16_t B_frag;
            {
                const int n_col = n0 + row_lane;
                if(n_col < N) {
                    const int b_base = j * N * K1_BYTES + n_col * K1_BYTES;
                    // We need the 16 original-K values [jj_lo, jj_lo+16) from the
                    // permuted bytes. Reverse the within-8 permute.
                    #pragma unroll
                    for(int r = 0; r < 16; ++r) {
                        const int jj_orig  = jj_lo + r;                     // 0..31
                        const int grp_base = (jj_orig / 8) * 8;
                        const int slot_out = jj_orig & 7;
                        const int orig_s   = forward_perm[slot_out];        // WRONG direction; see below
                        (void)orig_s;
                        // B_lds we built in Phase 2 used forward_perm (output->input). Here
                        // we want: given desired ORIGINAL position jj_orig, where is it in
                        // the permuted byte buffer?
                        const int inv_perm[8] = {1, 5, 0, 4, 3, 7, 2, 6};
                        const int perm_slot = inv_perm[slot_out];
                        const int perm_jj   = grp_base + perm_slot;         // position in permuted bytes
                        const uint8_t byte  = B[b_base + (perm_jj >> 1)];
                        const uint8_t nib   = ((perm_jj & 1) == 0)
                            ? ((byte >> 4) & 0xF)
                            : (byte & 0xF);
                        B_frag[r] = (__fp16)((float)((int)nib - 8));
                    }
                } else {
                    #pragma unroll
                    for(int r = 0; r < 16; ++r) B_frag[r] = (__fp16)0;
                }
            }

            // ------- WMMA step -------
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A_frag, B_frag, acc);
        }
    }

    // ------- Store C -------
    // lane i slot r in 0..7 -> C[r + 8*(i >= 16), i % 16]
    #pragma unroll
    for(int r = 0; r < 8; ++r) {
        const int m_out = m0 + 2 * r + half;
        const int n_out = n0 + row_lane;
        if(m_out < M && n_out < N) {
            C[m_out * N + n_out] = __float2half(acc[r]);
        }
    }
}

// Phase 4b: 1 wave per block, 16x64 output tile.
// Each wave accumulates FOUR 16x16 WMMA output tiles along N, amortizing
// the A fragment load across 4x the output. A stays read once per K step,
// B is loaded 4 times (one per sub-tile N offset), 4 WMMAs per K step.
// No LDS — direct global loads. Keeps launch overhead low (1 wave/block
// = high block count = good scheduler saturation).
__global__ __launch_bounds__(32)
void standalone_ternary_gemm_wmma_4x(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 64;   // 4 x 16
    constexpr int WMMA_K = 16;

    const int lane     = threadIdx.x;
    const int row_lane = lane & 15;
    const int half     = lane >> 4;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    float8_t acc[4] = {};  // four 16x16 accumulators along N

    const int forward_perm[8] = {2, 0, 6, 4, 3, 1, 7, 5};
    (void)forward_perm;
    const int inv_perm[8] = {1, 5, 0, 4, 3, 7, 2, 6};

    const int K0 = K / K1;

    for(int j = 0; j < K0; ++j) {
        #pragma unroll
        for(int step = 0; step < K1 / WMMA_K; ++step) {
            const int jj_lo = step * WMMA_K;

            // ------- A fragment (shared across all 4 B fragments) -------
            half16_t A_frag;
            {
                const int m_row = m0 + row_lane;
                if(m_row < M) {
                    #pragma unroll
                    for(int c = 0; c < 16; ++c) {
                        A_frag[c] = (__fp16)(float)A[m_row * K + j * K1 + jj_lo + c];
                    }
                } else {
                    #pragma unroll
                    for(int c = 0; c < 16; ++c) A_frag[c] = (__fp16)0;
                }
            }

            // ------- 4 B fragments + 4 WMMAs -------
            #pragma unroll
            for(int sub = 0; sub < 4; ++sub) {
                const int n_sub = n0 + sub * 16;
                half16_t B_frag;
                const int n_col = n_sub + row_lane;
                if(n_col < N) {
                    const int b_base = j * N * K1_BYTES + n_col * K1_BYTES;
                    #pragma unroll
                    for(int r = 0; r < 16; ++r) {
                        const int jj_orig  = jj_lo + r;
                        const int grp_base = (jj_orig / 8) * 8;
                        const int slot_out = jj_orig & 7;
                        const int perm_slot = inv_perm[slot_out];
                        const int perm_jj   = grp_base + perm_slot;
                        const uint8_t byte  = B[b_base + (perm_jj >> 1)];
                        const uint8_t nib   = ((perm_jj & 1) == 0)
                            ? ((byte >> 4) & 0xF)
                            : (byte & 0xF);
                        B_frag[r] = (__fp16)((float)((int)nib - 8));
                    }
                } else {
                    #pragma unroll
                    for(int r = 0; r < 16; ++r) B_frag[r] = (__fp16)0;
                }
                acc[sub] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A_frag, B_frag, acc[sub]);
            }
        }
    }

    // ------- Store: 4 sub-tiles along N -------
    #pragma unroll
    for(int sub = 0; sub < 4; ++sub) {
        const int n_sub = n0 + sub * 16;
        #pragma unroll
        for(int r = 0; r < 8; ++r) {
            const int m_out = m0 + 2 * r + half;
            const int n_out = n_sub + row_lane;
            if(m_out < M && n_out < N) {
                C[m_out * N + n_out] = __float2half(acc[sub][r]);
            }
        }
    }
}

// Phase 4c: 1 wave per block, 32x64 output tile.
// 2 A fragments x 4 B fragments = 8 accumulators per wave.
// Per K step: 2 A loads + 4 B loads + 8 WMMAs.
// Higher arithmetic intensity than 4b: 8 WMMAs per 6 frag loads vs 4 per 5.
__global__ __launch_bounds__(32)
void standalone_ternary_gemm_wmma_2x4(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_M = 32;   // 2 x 16
    constexpr int TILE_N = 64;   // 4 x 16
    constexpr int WMMA_K = 16;

    const int lane     = threadIdx.x;
    const int row_lane = lane & 15;
    const int half     = lane >> 4;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    float8_t acc[2][4] = {};   // 2 M sub-tiles x 4 N sub-tiles

    const int inv_perm[8] = {1, 5, 0, 4, 3, 7, 2, 6};
    const int K0 = K / K1;

    for(int j = 0; j < K0; ++j) {
        #pragma unroll
        for(int step = 0; step < K1 / WMMA_K; ++step) {
            const int jj_lo = step * WMMA_K;

            // ------- 2 A fragments, one per M sub-tile -------
            half16_t A_frags[2];
            #pragma unroll
            for(int m_sub = 0; m_sub < 2; ++m_sub) {
                const int m_row = m0 + m_sub * 16 + row_lane;
                if(m_row < M) {
                    #pragma unroll
                    for(int c = 0; c < 16; ++c) {
                        A_frags[m_sub][c] = (__fp16)(float)A[m_row * K + j * K1 + jj_lo + c];
                    }
                } else {
                    #pragma unroll
                    for(int c = 0; c < 16; ++c) A_frags[m_sub][c] = (__fp16)0;
                }
            }

            // ------- 4 B fragments, one per N sub-tile, then 8 WMMAs -------
            #pragma unroll
            for(int n_sub = 0; n_sub < 4; ++n_sub) {
                const int n_col = n0 + n_sub * 16 + row_lane;
                half16_t B_frag;
                if(n_col < N) {
                    const int b_base = j * N * K1_BYTES + n_col * K1_BYTES;
                    #pragma unroll
                    for(int r = 0; r < 16; ++r) {
                        const int jj_orig   = jj_lo + r;
                        const int grp_base  = (jj_orig / 8) * 8;
                        const int slot_out  = jj_orig & 7;
                        const int perm_slot = inv_perm[slot_out];
                        const int perm_jj   = grp_base + perm_slot;
                        const uint8_t byte  = B[b_base + (perm_jj >> 1)];
                        const uint8_t nib   = ((perm_jj & 1) == 0)
                            ? ((byte >> 4) & 0xF)
                            : (byte & 0xF);
                        B_frag[r] = (__fp16)((float)((int)nib - 8));
                    }
                } else {
                    #pragma unroll
                    for(int r = 0; r < 16; ++r) B_frag[r] = (__fp16)0;
                }
                // Cross product: this B fragment feeds both M sub-tiles.
                #pragma unroll
                for(int m_sub = 0; m_sub < 2; ++m_sub) {
                    acc[m_sub][n_sub] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        A_frags[m_sub], B_frag, acc[m_sub][n_sub]);
                }
            }
        }
    }

    // ------- Store 2x4 grid of 16x16 sub-tiles -------
    #pragma unroll
    for(int m_sub = 0; m_sub < 2; ++m_sub) {
        #pragma unroll
        for(int n_sub = 0; n_sub < 4; ++n_sub) {
            const int mo = m0 + m_sub * 16;
            const int no = n0 + n_sub * 16;
            #pragma unroll
            for(int r = 0; r < 8; ++r) {
                const int m_out = mo + 2 * r + half;
                const int n_out = no + row_lane;
                if(m_out < M && n_out < N) {
                    C[m_out * N + n_out] = __float2half(acc[m_sub][n_sub][r]);
                }
            }
        }
    }
}

// Phase 4d: 1 wave per block, 32x64 output.
// Key change vs 4c: B is decoded to FP16 into LDS ONCE per j block, then all
// 8 WMMAs (2 steps x 2 A x 4 B) read from LDS. Eliminates the hot-path
// pk_i4 nibble decode that dominated Phase 4c's 74%-of-CK ceiling.
// LDS: B_tile [32, 64] FP16 = 4 KiB per block.
__global__ __launch_bounds__(32)
void standalone_ternary_gemm_wmma_2x4_ldsB(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_M  = 32;
    constexpr int TILE_N  = 64;
    constexpr int WMMA_K  = 16;

    // B tile: [K1, TILE_N] FP16, K-contiguous per column
    __shared__ __half B_lds[K1 * TILE_N];

    const int lane     = threadIdx.x;
    const int row_lane = lane & 15;
    const int half     = lane >> 4;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    float8_t acc[2][4] = {};
    const int forward_perm[8] = {2, 0, 6, 4, 3, 1, 7, 5};
    const int K0 = K / K1;

    for(int j = 0; j < K0; ++j) {
        // ---- Cooperatively decode+load B tile into LDS ----
        // 64 cols x 16 bytes/col = 1024 bytes. 32 lanes -> 32 bytes/lane
        // = 64 nibbles/lane -> 64 half writes/lane to B_lds.
        #pragma unroll
        for(int bb = 0; bb < 32; ++bb) {
            const int byte_off  = lane * 32 + bb;            // 0..1023
            const int col       = byte_off / K1_BYTES;       // 0..63
            const int byte_in_col = byte_off % K1_BYTES;     // 0..15
            const int n_col     = n0 + col;
            if(n_col < N) {
                const int b_idx = j * N * K1_BYTES + n_col * K1_BYTES + byte_in_col;
                const uint8_t byte = B[b_idx];
                #pragma unroll
                for(int nibble = 0; nibble < 2; ++nibble) {
                    const int perm_jj   = byte_in_col * 2 + nibble;
                    const int grp_base  = (perm_jj / 8) * 8;
                    const int orig_slot = forward_perm[perm_jj & 7];
                    const int kk        = grp_base + orig_slot;
                    const uint8_t nib = (nibble == 0)
                        ? ((byte >> 4) & 0xF)
                        : (byte & 0xF);
                    B_lds[kk * TILE_N + col] = __float2half((float)((int)nib - 8));
                }
            } else {
                #pragma unroll
                for(int nibble = 0; nibble < 2; ++nibble) {
                    const int perm_jj   = byte_in_col * 2 + nibble;
                    const int grp_base  = (perm_jj / 8) * 8;
                    const int orig_slot = forward_perm[perm_jj & 7];
                    const int kk        = grp_base + orig_slot;
                    B_lds[kk * TILE_N + col] = (__half)0;
                }
            }
        }
        __syncthreads();

        // ---- 2 WMMA steps per K1 block ----
        #pragma unroll
        for(int step = 0; step < K1 / WMMA_K; ++step) {
            const int kk_base = step * WMMA_K;

            // Load 2 A fragments (directly from global — A reuse across N is still via registers)
            half16_t A_frags[2];
            #pragma unroll
            for(int m_sub = 0; m_sub < 2; ++m_sub) {
                const int m_row = m0 + m_sub * 16 + row_lane;
                if(m_row < M) {
                    #pragma unroll
                    for(int c = 0; c < 16; ++c) {
                        A_frags[m_sub][c] = (__fp16)(float)A[m_row * K + j * K1 + kk_base + c];
                    }
                } else {
                    #pragma unroll
                    for(int c = 0; c < 16; ++c) A_frags[m_sub][c] = (__fp16)0;
                }
            }

            // 4 B fragments from LDS + 8 WMMAs
            #pragma unroll
            for(int n_sub = 0; n_sub < 4; ++n_sub) {
                half16_t B_frag;
                const int n_col_in_tile = n_sub * 16 + row_lane;  // 0..63
                #pragma unroll
                for(int r = 0; r < 16; ++r) {
                    const int kk = kk_base + r;
                    B_frag[r] = (__fp16)(float)B_lds[kk * TILE_N + n_col_in_tile];
                }
                #pragma unroll
                for(int m_sub = 0; m_sub < 2; ++m_sub) {
                    acc[m_sub][n_sub] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        A_frags[m_sub], B_frag, acc[m_sub][n_sub]);
                }
            }
        }
        __syncthreads();
    }

    // Store 2x4 grid
    #pragma unroll
    for(int m_sub = 0; m_sub < 2; ++m_sub) {
        #pragma unroll
        for(int n_sub = 0; n_sub < 4; ++n_sub) {
            const int mo = m0 + m_sub * 16;
            const int no = n0 + n_sub * 16;
            #pragma unroll
            for(int r = 0; r < 8; ++r) {
                const int m_out = mo + 2 * r + half;
                const int n_out = no + row_lane;
                if(m_out < M && n_out < N) {
                    C[m_out * N + n_out] = __float2half(acc[m_sub][n_sub][r]);
                }
            }
        }
    }
}

// Phase 4g: 1 wave per block, 64x64 output.
// 4 A fragments x 4 B fragments = 16 accumulators per wave.
// Per K step: 4 A loads + 4 B loads + 16 WMMAs => 2.0 WMMA/load arithmetic
// intensity (1.5x vs Phase 4c's 1.33). Register pressure estimate:
//   16 f8 acc  = 128 f32 per lane  = 128 VGPRs
//    4 A frags = 64 halves per lane = 32 VGPRs
//    1 B frag  = 16 halves per lane =  8 VGPRs
// Fits the 256-VGPR/thread gfx11 budget with room for live loop state.
// Block count stays high: 4320 for 2560x6912, 27x the 160 wave slots.
__global__ __launch_bounds__(32)
void standalone_ternary_gemm_wmma_4x4(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_M = 64;   // 4 x 16
    constexpr int TILE_N = 64;   // 4 x 16
    constexpr int WMMA_K = 16;

    const int lane     = threadIdx.x;
    const int row_lane = lane & 15;
    const int half     = lane >> 4;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    float8_t acc[4][4] = {};  // 4 M sub-tiles x 4 N sub-tiles

    const int inv_perm[8] = {1, 5, 0, 4, 3, 7, 2, 6};
    const int K0 = K / K1;

    for(int j = 0; j < K0; ++j) {
        #pragma unroll
        for(int step = 0; step < K1 / WMMA_K; ++step) {
            const int jj_lo = step * WMMA_K;

            // ------- 4 A fragments, one per M sub-tile -------
            half16_t A_frags[4];
            #pragma unroll
            for(int m_sub = 0; m_sub < 4; ++m_sub) {
                const int m_row = m0 + m_sub * 16 + row_lane;
                if(m_row < M) {
                    #pragma unroll
                    for(int c = 0; c < 16; ++c) {
                        A_frags[m_sub][c] = (__fp16)(float)A[m_row * K + j * K1 + jj_lo + c];
                    }
                } else {
                    #pragma unroll
                    for(int c = 0; c < 16; ++c) A_frags[m_sub][c] = (__fp16)0;
                }
            }

            // ------- 4 B fragments + 16 WMMAs (cross product) -------
            #pragma unroll
            for(int n_sub = 0; n_sub < 4; ++n_sub) {
                const int n_col = n0 + n_sub * 16 + row_lane;
                half16_t B_frag;
                if(n_col < N) {
                    const int b_base = j * N * K1_BYTES + n_col * K1_BYTES;
                    #pragma unroll
                    for(int r = 0; r < 16; ++r) {
                        const int jj_orig   = jj_lo + r;
                        const int grp_base  = (jj_orig / 8) * 8;
                        const int slot_out  = jj_orig & 7;
                        const int perm_slot = inv_perm[slot_out];
                        const int perm_jj   = grp_base + perm_slot;
                        const uint8_t byte  = B[b_base + (perm_jj >> 1)];
                        const uint8_t nib   = ((perm_jj & 1) == 0)
                            ? ((byte >> 4) & 0xF)
                            : (byte & 0xF);
                        B_frag[r] = (__fp16)((float)((int)nib - 8));
                    }
                } else {
                    #pragma unroll
                    for(int r = 0; r < 16; ++r) B_frag[r] = (__fp16)0;
                }
                #pragma unroll
                for(int m_sub = 0; m_sub < 4; ++m_sub) {
                    acc[m_sub][n_sub] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        A_frags[m_sub], B_frag, acc[m_sub][n_sub]);
                }
            }
        }
    }

    // ------- Store 4x4 grid of 16x16 sub-tiles -------
    #pragma unroll
    for(int m_sub = 0; m_sub < 4; ++m_sub) {
        #pragma unroll
        for(int n_sub = 0; n_sub < 4; ++n_sub) {
            const int mo = m0 + m_sub * 16;
            const int no = n0 + n_sub * 16;
            #pragma unroll
            for(int r = 0; r < 8; ++r) {
                const int m_out = mo + 2 * r + half;
                const int n_out = no + row_lane;
                if(m_out < M && n_out < N) {
                    C[m_out * N + n_out] = __float2half(acc[m_sub][n_sub][r]);
                }
            }
        }
    }
}

// Phase 4h: 1 wave, 64x64 output — A and B loaded ONCE per K1 block
// (covering both WMMA K steps at 16 each), then 32 WMMAs reuse the cached
// fragments. Halves per-block memory traffic vs 4g.
//
// Register pressure:
//   A: 4 m_sub x 2 step x half16_t  = 128 halves = 64 VGPRs/lane
//   B:         2 step x half16_t   =  32 halves = 16 VGPRs/lane (one n_sub at a time)
//   acc: 4x4 float8_t              = 128 f32    = 128 VGPRs/lane
// Plus loop vars + decode temps — ~215 VGPRs/lane, inside 256 budget.
__global__ __launch_bounds__(32)
void standalone_ternary_gemm_wmma_4x4_cached(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;

    const int lane     = threadIdx.x;
    const int row_lane = lane & 15;
    const int half     = lane >> 4;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    float8_t acc[4][4] = {};
    const int inv_perm[8] = {1, 5, 0, 4, 3, 7, 2, 6};
    const int K0 = K / K1;

    for(int j = 0; j < K0; ++j) {
        // ------- Load 4 A fragments, each holding BOTH K steps (32 halves) -------
        // A_cache[m_sub][step] holds 16 halves along K. Loaded once per j.
        half16_t A_cache[4][2];
        #pragma unroll
        for(int m_sub = 0; m_sub < 4; ++m_sub) {
            const int m_row = m0 + m_sub * 16 + row_lane;
            if(m_row < M) {
                #pragma unroll
                for(int c = 0; c < K1; ++c) {
                    const float v = (float)A[m_row * K + j * K1 + c];
                    A_cache[m_sub][c / 16][c & 15] = (__fp16)v;
                }
            } else {
                #pragma unroll
                for(int c = 0; c < K1; ++c) A_cache[m_sub][c / 16][c & 15] = (__fp16)0;
            }
        }

        // ------- For each N sub-tile: decode B ONCE (both steps), then 4 WMMAs x 2 steps -------
        #pragma unroll
        for(int n_sub = 0; n_sub < 4; ++n_sub) {
            const int n_col = n0 + n_sub * 16 + row_lane;

            half16_t B_cache[2];
            if(n_col < N) {
                const int b_base = j * N * K1_BYTES + n_col * K1_BYTES;
                #pragma unroll
                for(int c = 0; c < K1; ++c) {
                    const int jj_orig   = c;
                    const int grp_base  = (jj_orig / 8) * 8;
                    const int slot_out  = jj_orig & 7;
                    const int perm_slot = inv_perm[slot_out];
                    const int perm_jj   = grp_base + perm_slot;
                    const uint8_t byte  = B[b_base + (perm_jj >> 1)];
                    const uint8_t nib   = ((perm_jj & 1) == 0)
                        ? ((byte >> 4) & 0xF)
                        : (byte & 0xF);
                    B_cache[c / 16][c & 15] = (__fp16)((float)((int)nib - 8));
                }
            } else {
                #pragma unroll
                for(int c = 0; c < K1; ++c) B_cache[c / 16][c & 15] = (__fp16)0;
            }

            // 2 K steps, 4 WMMAs each, all from registers
            #pragma unroll
            for(int step = 0; step < 2; ++step) {
                #pragma unroll
                for(int m_sub = 0; m_sub < 4; ++m_sub) {
                    acc[m_sub][n_sub] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        A_cache[m_sub][step], B_cache[step], acc[m_sub][n_sub]);
                }
            }
        }
    }

    // ------- Store 4x4 grid -------
    #pragma unroll
    for(int m_sub = 0; m_sub < 4; ++m_sub) {
        #pragma unroll
        for(int n_sub = 0; n_sub < 4; ++n_sub) {
            const int mo = m0 + m_sub * 16;
            const int no = n0 + n_sub * 16;
            #pragma unroll
            for(int r = 0; r < 8; ++r) {
                const int m_out = mo + 2 * r + half;
                const int n_out = no + row_lane;
                if(m_out < M && n_out < N) {
                    C[m_out * N + n_out] = __float2half(acc[m_sub][n_sub][r]);
                }
            }
        }
    }
}

// Phase 4i: Phase 4h + vectorized A loads.
// Phase 4h did 16 scalar FP16 loads per A fragment half. 4i replaces those
// with 4 int2 (64-bit = 4 FP16) loads, cutting A-load instructions 4x.
using half4_t = __attribute__((ext_vector_type(4))) __fp16;

__global__ __launch_bounds__(32)
void standalone_ternary_gemm_wmma_4x4_vec(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;

    const int lane     = threadIdx.x;
    const int row_lane = lane & 15;
    const int half     = lane >> 4;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    float8_t acc[4][4] = {};
    const int inv_perm[8] = {1, 5, 0, 4, 3, 7, 2, 6};
    const int K0 = K / K1;

    for(int j = 0; j < K0; ++j) {
        // ------- Vectorized A load: 8x half4 per m_sub (covers 32 K, both steps) -------
        half16_t A_cache[4][2];
        #pragma unroll
        for(int m_sub = 0; m_sub < 4; ++m_sub) {
            const int m_row = m0 + m_sub * 16 + row_lane;
            if(m_row < M) {
                const __half* arow = A + m_row * K + j * K1;  // 32 halves
                const half4_t* arow4 = reinterpret_cast<const half4_t*>(arow);
                // 32 halves = 8 half4 = 2 half16 (one per WMMA step)
                #pragma unroll
                for(int v = 0; v < 4; ++v) {
                    half4_t q = arow4[v];              // step 0, positions v*4..v*4+3
                    A_cache[m_sub][0][v * 4 + 0] = q[0];
                    A_cache[m_sub][0][v * 4 + 1] = q[1];
                    A_cache[m_sub][0][v * 4 + 2] = q[2];
                    A_cache[m_sub][0][v * 4 + 3] = q[3];
                }
                #pragma unroll
                for(int v = 0; v < 4; ++v) {
                    half4_t q = arow4[4 + v];          // step 1, positions 16..31
                    A_cache[m_sub][1][v * 4 + 0] = q[0];
                    A_cache[m_sub][1][v * 4 + 1] = q[1];
                    A_cache[m_sub][1][v * 4 + 2] = q[2];
                    A_cache[m_sub][1][v * 4 + 3] = q[3];
                }
            } else {
                #pragma unroll
                for(int c = 0; c < K1; ++c) A_cache[m_sub][c / 16][c & 15] = (__fp16)0;
            }
        }

        // ------- Per N sub-tile: decode B (both steps), 8 WMMAs -------
        #pragma unroll
        for(int n_sub = 0; n_sub < 4; ++n_sub) {
            const int n_col = n0 + n_sub * 16 + row_lane;
            half16_t B_cache[2];
            if(n_col < N) {
                const int b_base = j * N * K1_BYTES + n_col * K1_BYTES;
                #pragma unroll
                for(int c = 0; c < K1; ++c) {
                    const int grp_base  = (c / 8) * 8;
                    const int perm_slot = inv_perm[c & 7];
                    const int perm_jj   = grp_base + perm_slot;
                    const uint8_t byte  = B[b_base + (perm_jj >> 1)];
                    const uint8_t nib   = ((perm_jj & 1) == 0)
                        ? ((byte >> 4) & 0xF)
                        : (byte & 0xF);
                    B_cache[c / 16][c & 15] = (__fp16)((float)((int)nib - 8));
                }
            } else {
                #pragma unroll
                for(int c = 0; c < K1; ++c) B_cache[c / 16][c & 15] = (__fp16)0;
            }

            #pragma unroll
            for(int step = 0; step < 2; ++step) {
                #pragma unroll
                for(int m_sub = 0; m_sub < 4; ++m_sub) {
                    acc[m_sub][n_sub] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        A_cache[m_sub][step], B_cache[step], acc[m_sub][n_sub]);
                }
            }
        }
    }

    #pragma unroll
    for(int m_sub = 0; m_sub < 4; ++m_sub) {
        #pragma unroll
        for(int n_sub = 0; n_sub < 4; ++n_sub) {
            const int mo = m0 + m_sub * 16;
            const int no = n0 + n_sub * 16;
            #pragma unroll
            for(int r = 0; r < 8; ++r) {
                const int m_out = mo + 2 * r + half;
                const int n_out = no + row_lane;
                if(m_out < M && n_out < N) {
                    C[m_out * N + n_out] = __float2half(acc[m_sub][n_sub][r]);
                }
            }
        }
    }
}

// Phase 4i-FP16: 4i kernel but B is an FP16 buffer (no pk_i4 decode on the hot path).
// For consumers willing to materialize FP16 B once (one decode kernel per weight
// tensor at model load), this skips the per-step nibble-extract + -8 bias and
// closes the rocBLAS gap on square shapes. Same 64x64 tile as 4i.
// B layout: col-major [K, N] FP16, stride K. Same as rocBLAS.
__global__ __launch_bounds__(32)
void standalone_gemm_fp16b(
    const __half* __restrict__ A,
    const __half* __restrict__ B,       // col-major [K, N], stride K
    __half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;

    const int lane     = threadIdx.x;
    const int row_lane = lane & 15;
    const int half     = lane >> 4;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    float8_t acc[4][4] = {};
    const int K0 = K / K1;

    for(int j = 0; j < K0; ++j) {
        // ------- Vectorized A load: 4 half4_t per m_sub, covers both K steps -------
        half16_t A_cache[4][2];
        #pragma unroll
        for(int m_sub = 0; m_sub < 4; ++m_sub) {
            const int m_row = m0 + m_sub * 16 + row_lane;
            if(m_row < M) {
                const __half* arow = A + m_row * K + j * K1;
                const half4_t* arow4 = reinterpret_cast<const half4_t*>(arow);
                #pragma unroll
                for(int v = 0; v < 4; ++v) {
                    half4_t q = arow4[v];
                    A_cache[m_sub][0][v*4+0] = q[0];
                    A_cache[m_sub][0][v*4+1] = q[1];
                    A_cache[m_sub][0][v*4+2] = q[2];
                    A_cache[m_sub][0][v*4+3] = q[3];
                }
                #pragma unroll
                for(int v = 0; v < 4; ++v) {
                    half4_t q = arow4[4+v];
                    A_cache[m_sub][1][v*4+0] = q[0];
                    A_cache[m_sub][1][v*4+1] = q[1];
                    A_cache[m_sub][1][v*4+2] = q[2];
                    A_cache[m_sub][1][v*4+3] = q[3];
                }
            } else {
                #pragma unroll
                for(int c = 0; c < K1; ++c) A_cache[m_sub][c/16][c&15] = (__fp16)0;
            }
        }

        // ------- Per N sub-tile: B fragments loaded directly from FP16 buffer -------
        #pragma unroll
        for(int n_sub = 0; n_sub < 4; ++n_sub) {
            const int n_col = n0 + n_sub * 16 + row_lane;
            half16_t B_cache[2];
            if(n_col < N) {
                // B col-major: col n_col starts at B + n_col*K, j-block offset = j*K1
                const __half* bcol = B + (size_t)n_col * K + j * K1;
                const half4_t* bcol4 = reinterpret_cast<const half4_t*>(bcol);
                #pragma unroll
                for(int v = 0; v < 4; ++v) {
                    half4_t q = bcol4[v];
                    B_cache[0][v*4+0] = q[0];
                    B_cache[0][v*4+1] = q[1];
                    B_cache[0][v*4+2] = q[2];
                    B_cache[0][v*4+3] = q[3];
                }
                #pragma unroll
                for(int v = 0; v < 4; ++v) {
                    half4_t q = bcol4[4+v];
                    B_cache[1][v*4+0] = q[0];
                    B_cache[1][v*4+1] = q[1];
                    B_cache[1][v*4+2] = q[2];
                    B_cache[1][v*4+3] = q[3];
                }
            } else {
                #pragma unroll
                for(int c = 0; c < K1; ++c) B_cache[c/16][c&15] = (__fp16)0;
            }

            #pragma unroll
            for(int step = 0; step < 2; ++step) {
                #pragma unroll
                for(int m_sub = 0; m_sub < 4; ++m_sub) {
                    acc[m_sub][n_sub] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        A_cache[m_sub][step], B_cache[step], acc[m_sub][n_sub]);
                }
            }
        }
    }

    #pragma unroll
    for(int m_sub = 0; m_sub < 4; ++m_sub) {
        #pragma unroll
        for(int n_sub = 0; n_sub < 4; ++n_sub) {
            const int mo = m0 + m_sub * 16;
            const int no = n0 + n_sub * 16;
            #pragma unroll
            for(int r = 0; r < 8; ++r) {
                const int m_out = mo + 2 * r + half;
                const int n_out = no + row_lane;
                if(m_out < M && n_out < N) {
                    C[m_out * N + n_out] = __float2half(acc[m_sub][n_sub][r]);
                }
            }
        }
    }
}

// Simple B decoder: pk_i4 WMMA-permuted bytes -> col-major FP16 [K, N], stride K.
// Run once per weight tensor, before the first GEMM that uses it.
// 64 threads/block, 1 block per N column for parallelism.
__global__ __launch_bounds__(64)
void decode_pk_i4_to_fp16(
    const uint8_t* __restrict__ B_packed,
    __half* __restrict__ B_fp16,
    int K, int N)
{
    const int n = blockIdx.x;
    if(n >= N) return;
    const int tid = threadIdx.x;
    const int K0  = K / K1;

    // Each thread handles (K / 64) K1-blocks along K axis for this column.
    const int forward_perm[8] = {2, 0, 6, 4, 3, 1, 7, 5};

    for(int j_step = tid; j_step < K0; j_step += 64) {
        const int j = j_step;
        const int b_base = j * N * K1_BYTES + n * K1_BYTES;
        #pragma unroll
        for(int c = 0; c < K1; ++c) {
            const int grp_base = (c / 8) * 8;
            const int slot_out = c & 7;
            const int orig_s   = forward_perm[slot_out];
            const int kk       = grp_base + orig_s;     // ORIGINAL K position
            // The byte at perm_jj=c holds two nibbles; read each and map to its
            // original K slot.
            const uint8_t byte = B_packed[b_base + (c >> 1)];
            const uint8_t nib  = ((c & 1) == 0)
                ? ((byte >> 4) & 0xF)
                : (byte & 0xF);
            B_fp16[(size_t)n * K + j * K1 + kk] = __float2half((float)((int)nib - 8));
        }
    }
}

// Phase 4j: software-pipelined double buffer on A across j iterations.
// Geometry: 1 wave / 32x64 output (same as Phase 4c — 2A x 4B = 8 accumulators)
//           so the VGPR budget fits the 2x A buffers.
// Idea: at iteration j, while computing WMMAs over A_cache[cur], simultaneously
// issue the global loads for A_cache[nxt] covering j+1. Compiler + hardware's
// scoreboarding overlaps the load with compute, hiding global-load latency
// behind the 8 WMMAs per K step.
//
// Register budget (per lane):
//   acc[2][4]       = 8 x float8_t      =  64 VGPRs
//   A_cache[2][2][2] = 2 buf x 2 msub x 2 step x half16_t = 128 halves = 64 VGPRs
//   B_cache[2]      = 2 step x half16_t =  16 halves = 8 VGPRs
//   Total ~ 144 VGPRs + loop state — same occupancy class as 4c/4i (1 wave/SIMD).
//
__global__ __launch_bounds__(32)
void standalone_gemm_4c_pipelined(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_M = 32;
    constexpr int TILE_N = 64;

    const int lane     = threadIdx.x;
    const int row_lane = lane & 15;
    const int half     = lane >> 4;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    float8_t acc[2][4] = {};
    const int inv_perm[8] = {1, 5, 0, 4, 3, 7, 2, 6};
    const int K0 = K / K1;

    // Loader helper: fills A_cache[buf] (2 m_sub x 2 step x half16_t) for given j
    half16_t A_cache[2][2][2];  // [buf][m_sub][step]
    auto loadA = [&](int buf, int j) {
        #pragma unroll
        for(int m_sub = 0; m_sub < 2; ++m_sub) {
            const int m_row = m0 + m_sub * 16 + row_lane;
            if(m_row < M) {
                const __half* arow = A + m_row * K + j * K1;
                const half4_t* arow4 = reinterpret_cast<const half4_t*>(arow);
                #pragma unroll
                for(int v = 0; v < 4; ++v) {
                    half4_t q = arow4[v];
                    A_cache[buf][m_sub][0][v*4+0] = q[0];
                    A_cache[buf][m_sub][0][v*4+1] = q[1];
                    A_cache[buf][m_sub][0][v*4+2] = q[2];
                    A_cache[buf][m_sub][0][v*4+3] = q[3];
                }
                #pragma unroll
                for(int v = 0; v < 4; ++v) {
                    half4_t q = arow4[4+v];
                    A_cache[buf][m_sub][1][v*4+0] = q[0];
                    A_cache[buf][m_sub][1][v*4+1] = q[1];
                    A_cache[buf][m_sub][1][v*4+2] = q[2];
                    A_cache[buf][m_sub][1][v*4+3] = q[3];
                }
            } else {
                #pragma unroll
                for(int c = 0; c < K1; ++c) A_cache[buf][m_sub][c/16][c&15] = (__fp16)0;
            }
        }
    };

    // Bootstrap: load j=0 into buffer 0
    if(K0 > 0) loadA(0, 0);

    for(int j = 0; j < K0; ++j) {
        const int cur = j & 1;
        const int nxt = 1 - cur;
        // Prefetch next j into the other buffer (overlaps with WMMAs below)
        if(j + 1 < K0) {
            loadA(nxt, j + 1);
        }

        // Compute with cur buffer: 4 B fragments + 8 WMMAs per K step x 2 steps
        #pragma unroll
        for(int n_sub = 0; n_sub < 4; ++n_sub) {
            const int n_col = n0 + n_sub * 16 + row_lane;
            half16_t B_frags[2];
            if(n_col < N) {
                const int b_base = j * N * K1_BYTES + n_col * K1_BYTES;
                #pragma unroll
                for(int c = 0; c < K1; ++c) {
                    const int grp_base  = (c / 8) * 8;
                    const int perm_slot = inv_perm[c & 7];
                    const int perm_jj   = grp_base + perm_slot;
                    const uint8_t byte  = B[b_base + (perm_jj >> 1)];
                    const uint8_t nib   = ((perm_jj & 1) == 0)
                        ? ((byte >> 4) & 0xF)
                        : (byte & 0xF);
                    B_frags[c / 16][c & 15] = (__fp16)((float)((int)nib - 8));
                }
            } else {
                #pragma unroll
                for(int c = 0; c < K1; ++c) B_frags[c/16][c&15] = (__fp16)0;
            }
            #pragma unroll
            for(int step = 0; step < 2; ++step) {
                #pragma unroll
                for(int m_sub = 0; m_sub < 2; ++m_sub) {
                    acc[m_sub][n_sub] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        A_cache[cur][m_sub][step], B_frags[step], acc[m_sub][n_sub]);
                }
            }
        }
    }

    #pragma unroll
    for(int m_sub = 0; m_sub < 2; ++m_sub) {
        #pragma unroll
        for(int n_sub = 0; n_sub < 4; ++n_sub) {
            const int mo = m0 + m_sub * 16;
            const int no = n0 + n_sub * 16;
            #pragma unroll
            for(int r = 0; r < 8; ++r) {
                const int m_out = mo + 2 * r + half;
                const int n_out = no + row_lane;
                if(m_out < M && n_out < N) {
                    C[m_out * N + n_out] = __float2half(acc[m_sub][n_sub][r]);
                }
            }
        }
    }
}

// Phase 4k: LDS ping-pong with 2 waves per block. 64x32 output per block,
// split into two 32x32 sub-tiles (wave 0 = upper 32 rows, wave 1 = lower).
// Both waves share A_lds (64x32 FP16) and B_lds (32x32 FP16, decoded).
// Double-buffered so next-j global loads overlap with current-j WMMAs.
//
// Why 2 waves (not 4 like Phase 4f): Phase 4f's 128-thread blocks cut block
// count below saturation on gfx1151. 64-thread blocks (2 waves) keep block
// count high AND enable LDS sharing. Register pressure target: ~100 VGPR/lane
// to hit 2 waves per SIMD (true occupancy win vs 4i's single-wave).
//
// LDS footprint:
//   A_lds[2][64][32] FP16 = 8 KiB
//   B_lds[2][32][32] FP16 = 4 KiB
//   Total: 12 KiB per block — fits gfx1151's 64 KiB/CU with many blocks/CU.
__global__ __launch_bounds__(64)
void standalone_gemm_lds_pingpong(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 32;   // 2 N sub-tiles of 16 each, both waves cover full TILE_N
    constexpr int WMMA_K = 16;

    __shared__ __half A_lds[2][TILE_M * K1];   // [2 bufs][64 * 32 halves]
    __shared__ __half B_lds[2][K1 * TILE_N];   // [2 bufs][32 * 32 halves]

    const int tid  = threadIdx.x;          // 0..63
    const int wave = tid >> 5;             // 0 (upper) or 1 (lower 32 rows)
    const int lane = tid & 31;
    const int row_lane = lane & 15;
    const int half     = lane >> 4;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    // Each wave owns a 32x32 output sub-tile = 2 M sub x 2 N sub = 4 WMMA tiles.
    float8_t acc[2][2] = {};
    const int forward_perm[8] = {2, 0, 6, 4, 3, 1, 7, 5};
    const int K0 = K / K1;

    // Cooperative load helpers. 64 threads, K1=32.
    // A tile: 64 rows x 32 cols = 2048 halves. 32 halves per thread.
    auto load_A_buf = [&](int buf, int j) {
        #pragma unroll
        for(int e = 0; e < 32; ++e) {
            const int slot = tid * 32 + e;       // 0..2047
            const int r = slot / 32;             // 0..63
            const int c = slot & 31;             // 0..31
            const int m_row = m0 + r;
            A_lds[buf][slot] = (m_row < M) ? A[m_row * K + j * K1 + c] : (__half)0;
        }
    };
    // B tile: 32 cols x 16 bytes per col = 512 bytes. 8 bytes per thread.
    auto load_B_buf = [&](int buf, int j) {
        #pragma unroll
        for(int bb = 0; bb < 8; ++bb) {
            const int byte_off    = tid * 8 + bb;
            const int col         = byte_off / K1_BYTES;     // 0..63
            const int byte_in_col = byte_off % K1_BYTES;
            const int n_col       = n0 + col;
            uint8_t byte = 0;
            if(n_col < N)
                byte = B[j * N * K1_BYTES + n_col * K1_BYTES + byte_in_col];
            #pragma unroll
            for(int nibble = 0; nibble < 2; ++nibble) {
                const int perm_jj   = byte_in_col * 2 + nibble;
                const int grp_base  = (perm_jj / 8) * 8;
                const int orig_slot = forward_perm[perm_jj & 7];
                const int kk        = grp_base + orig_slot;
                const uint8_t nib = (nibble == 0) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                B_lds[buf][kk * TILE_N + col] = (n_col < N)
                    ? __float2half((float)((int)nib - 8))
                    : (__half)0;
            }
        }
    };

    // ==== Bootstrap: load j=0 into buf 0 ====
    if(K0 > 0) {
        load_A_buf(0, 0);
        load_B_buf(0, 0);
    }
    __syncthreads();

    for(int j = 0; j < K0; ++j) {
        const int cur = j & 1;
        const int nxt = 1 - cur;

        // Issue next-j loads into the other buffer. These go to LDS directly;
        // compute below reads from cur buffer. HW scoreboards overlap load
        // progress with compute WMMAs.
        if(j + 1 < K0) {
            load_A_buf(nxt, j + 1);
            load_B_buf(nxt, j + 1);
        }

        // Compute from cur buffer. Each wave: 2 A_frags + 2 B_frags + 2 K steps = 8 WMMAs.
        #pragma unroll
        for(int step = 0; step < K1 / WMMA_K; ++step) {
            const int kk_base = step * WMMA_K;

            // 2 A fragments for this wave's M rows.
            half16_t A_frags[2];
            #pragma unroll
            for(int m_sub = 0; m_sub < 2; ++m_sub) {
                const int row_in_lds = wave * 32 + m_sub * 16 + row_lane;
                #pragma unroll
                for(int c = 0; c < 16; ++c) {
                    A_frags[m_sub][c] = (__fp16)(float)A_lds[cur][row_in_lds * K1 + kk_base + c];
                }
            }

            // 2 B fragments covering TILE_N = 32 cols (16 per n_sub).
            #pragma unroll
            for(int n_sub = 0; n_sub < 2; ++n_sub) {
                const int col_in_lds = n_sub * 16 + row_lane;    // wave-independent N sweep
                // Each wave still wants to produce a 32x32 output, so both waves
                // compute n_sub=0..1 → cols 0..31 and 32..63. Each wave's accumulators
                // track both n_sub halves.
                half16_t B_frag;
                #pragma unroll
                for(int r = 0; r < 16; ++r) {
                    B_frag[r] = (__fp16)(float)B_lds[cur][(kk_base + r) * TILE_N + col_in_lds];
                }
                #pragma unroll
                for(int m_sub = 0; m_sub < 2; ++m_sub) {
                    acc[m_sub][n_sub] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        A_frags[m_sub], B_frag, acc[m_sub][n_sub]);
                }
            }
        }

        __syncthreads();
    }

    // Store: each wave owns rows [wave*32 .. wave*32+31], cols [0..TILE_N-1].
    const int wave_m0 = m0 + wave * 32;
    #pragma unroll
    for(int m_sub = 0; m_sub < 2; ++m_sub) {
        #pragma unroll
        for(int n_sub = 0; n_sub < 2; ++n_sub) {
            const int mo = wave_m0 + m_sub * 16;
            const int no = n0 + n_sub * 16;
            #pragma unroll
            for(int r = 0; r < 8; ++r) {
                const int m_out = mo + 2 * r + half;
                const int n_out = no + row_lane;
                if(m_out < M && n_out < N) {
                    C[m_out * N + n_out] = __float2half(acc[m_sub][n_sub][r]);
                }
            }
        }
    }
}

// Phase 4f: 4 waves per block (128 threads) in 2x2 layout, 64x64 output.
// LDS shared by all 4 waves — this is where LDS finally pays off (amortizes
// A and B loads across 4x the WMMA work per load).
//   waves arranged (wm, wn) for wm, wn in {0, 1}
//   each wave: 2 A frags (32 rows) x 2 B frags (32 cols) = 4 accumulators
//   LDS: A_tile[64, 32] = 4 KiB (K1=32, shared by 2 N-column waves each M row)
//        B_tile[32, 64] = 4 KiB (K1=32, shared by 2 M-row waves each N column)
//   Per K1 block: 4 waves x 2 K steps x 4 WMMAs = 32 WMMAs from LDS
__global__ __launch_bounds__(128)
void standalone_ternary_gemm_wmma_2x2wave(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    constexpr int WMMA_K = 16;

    __shared__ __half A_lds[TILE_M * K1];   // [64, 32] row-major
    __shared__ __half B_lds[K1 * TILE_N];   // [32, 64] row-major (decoded)

    const int tid  = threadIdx.x;
    const int wave = tid >> 5;            // 0..3
    const int wm   = wave >> 1;           // 0 or 1 (M index)
    const int wn   = wave & 1;            // 0 or 1 (N index)
    const int lane = tid & 31;
    const int row_lane = lane & 15;
    const int half     = lane >> 4;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    float8_t acc[2][2] = {};   // 2 M sub-tiles x 2 N sub-tiles per wave
    const int forward_perm[8] = {2, 0, 6, 4, 3, 1, 7, 5};
    const int K0 = K / K1;

    for(int j = 0; j < K0; ++j) {
        // ---- Cooperative A load: 64 rows x 32 K = 2048 halves, 128 threads
        // -> 16 halves per thread. ----
        #pragma unroll
        for(int e = 0; e < 16; ++e) {
            const int slot = tid * 16 + e;       // 0..2047
            const int r    = slot / 32;          // 0..63
            const int c    = slot % 32;          // 0..31
            const int m_row = m0 + r;
            A_lds[slot] = (m_row < M) ? A[m_row * K + j * K1 + c] : (__half)0;
        }

        // ---- Cooperative B decode+load: 64 cols x 16 bytes/col = 1024 bytes,
        // 128 threads -> 8 bytes/thread = 16 nibbles/thread -> 16 half writes. ----
        #pragma unroll
        for(int bb = 0; bb < 8; ++bb) {
            const int byte_off    = tid * 8 + bb;
            const int col         = byte_off / K1_BYTES;
            const int byte_in_col = byte_off % K1_BYTES;
            const int n_col       = n0 + col;
            uint8_t byte = 0;
            if(n_col < N) {
                byte = B[j * N * K1_BYTES + n_col * K1_BYTES + byte_in_col];
            }
            #pragma unroll
            for(int nibble = 0; nibble < 2; ++nibble) {
                const int perm_jj   = byte_in_col * 2 + nibble;
                const int grp_base  = (perm_jj / 8) * 8;
                const int orig_slot = forward_perm[perm_jj & 7];
                const int kk        = grp_base + orig_slot;
                const uint8_t nib = (nibble == 0)
                    ? ((byte >> 4) & 0xF)
                    : (byte & 0xF);
                B_lds[kk * TILE_N + col] = (n_col < N)
                    ? __float2half((float)((int)nib - 8))
                    : (__half)0;
            }
        }

        __syncthreads();

        // ---- 2 K steps per K1 block, 4 WMMAs per step per wave ----
        #pragma unroll
        for(int step = 0; step < K1 / WMMA_K; ++step) {
            const int kk_base = step * WMMA_K;

            // 2 A fragments for this wave's M rows
            half16_t A_frags[2];
            #pragma unroll
            for(int m_sub = 0; m_sub < 2; ++m_sub) {
                const int row_in_lds = wm * 32 + m_sub * 16 + row_lane;
                #pragma unroll
                for(int c = 0; c < 16; ++c) {
                    A_frags[m_sub][c] = (__fp16)(float)A_lds[row_in_lds * K1 + kk_base + c];
                }
            }

            // 2 B fragments for this wave's N cols + 4 WMMAs
            #pragma unroll
            for(int n_sub = 0; n_sub < 2; ++n_sub) {
                const int col_in_lds = wn * 32 + n_sub * 16 + row_lane;
                half16_t B_frag;
                #pragma unroll
                for(int r = 0; r < 16; ++r) {
                    B_frag[r] = (__fp16)(float)B_lds[(kk_base + r) * TILE_N + col_in_lds];
                }
                #pragma unroll
                for(int m_sub = 0; m_sub < 2; ++m_sub) {
                    acc[m_sub][n_sub] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        A_frags[m_sub], B_frag, acc[m_sub][n_sub]);
                }
            }
        }

        __syncthreads();
    }

    // ---- Store 2x2 grid of this wave's 16x16 sub-tiles ----
    #pragma unroll
    for(int m_sub = 0; m_sub < 2; ++m_sub) {
        #pragma unroll
        for(int n_sub = 0; n_sub < 2; ++n_sub) {
            const int mo = m0 + wm * 32 + m_sub * 16;
            const int no = n0 + wn * 32 + n_sub * 16;
            #pragma unroll
            for(int r = 0; r < 8; ++r) {
                const int m_out = mo + 2 * r + half;
                const int n_out = no + row_lane;
                if(m_out < M && n_out < N) {
                    C[m_out * N + n_out] = __float2half(acc[m_sub][n_sub][r]);
                }
            }
        }
    }
}

// Phase 4: 2 waves per block (64 threads) computing a 32x16 output tile.
// LDS staging for A and B — both waves share the K loads:
//   A_lds : [32, 32] FP16 = 2 KiB  (two 16-row halves, one per wave)
//   B_lds : [32, 16] FP16 = 1 KiB  (shared by both waves)
// Each wave then loads its fragments from LDS and does 2 WMMAs per K1 step.
__global__ __launch_bounds__(64)
void standalone_ternary_gemm_wmma_2wave(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_M = 32;   // 2 waves x 16 rows
    constexpr int TILE_N = 16;
    constexpr int WMMA_K = 16;

    __shared__ __half A_lds[TILE_M * K1];   // [32, 32] FP16 row-major
    __shared__ __half B_lds[K1 * TILE_N];   // [32, 16] FP16 row-major

    const int tid  = threadIdx.x;       // 0..63
    const int wave = tid >> 5;          // 0 or 1
    const int lane = tid & 31;          // 0..31
    const int row_lane = lane & 15;
    const int half     = lane >> 4;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    const int forward_perm[8] = {2, 0, 6, 4, 3, 1, 7, 5};
    const int K0 = K / K1;

    // Two float8 accumulators per wave, one per WMMA step along K1.
    // Waves both target their own 16x16 output rows; N is shared.
    float8_t acc = {};

    for(int j = 0; j < K0; ++j) {
        // ---- Cooperatively load A tile [32, 32] into LDS ----
        // 32*32 = 1024 halves, 64 threads -> 16 halves per thread.
        // Use a simple row-strided pattern for coalescing.
        #pragma unroll
        for(int e = 0; e < 16; ++e) {
            const int slot = tid * 16 + e;   // 0..1023
            const int r = slot / 32;         // 0..31
            const int c = slot % 32;         // 0..31
            const int m_row = m0 + r;
            A_lds[slot] = (m_row < M)
                ? A[m_row * K + j * K1 + c]
                : (__half)0;
        }

        // ---- Cooperatively load+decode B tile [32, 16] into LDS ----
        // 32*16 = 512 halves, 64 threads -> 8 halves per thread.
        // The byte layout supplies 16 cols x 16 bytes per j-block (256 bytes).
        // With 64 threads, each loads 4 bytes / 8 nibbles.
        {
            const int bytes_per_col = K1_BYTES;               // 16
            const int bytes_total   = TILE_N * bytes_per_col; // 256
            const int my_byte0 = tid * 4;                     // 0, 4, 8, ..., 252
            if(my_byte0 < bytes_total) {
                const int col_start_byte = my_byte0 % bytes_per_col;
                const int col_idx        = my_byte0 / bytes_per_col;
                (void)col_start_byte;
                (void)col_idx;
                #pragma unroll
                for(int bb = 0; bb < 4; ++bb) {
                    const int byte_off   = my_byte0 + bb;
                    const int col        = byte_off / bytes_per_col;
                    const int byte_in_col = byte_off % bytes_per_col;
                    const int n_col      = n0 + col;
                    if(n_col < N) {
                        const int b_idx = j * N * bytes_per_col + n_col * bytes_per_col + byte_in_col;
                        const uint8_t bb_val = B[b_idx];
                        #pragma unroll
                        for(int nibble = 0; nibble < 2; ++nibble) {
                            const int perm_jj    = byte_in_col * 2 + nibble;
                            const int grp_base   = (perm_jj / 8) * 8;
                            const int orig_slot  = forward_perm[perm_jj & 7];
                            const int kk         = grp_base + orig_slot;     // original K (0..31)
                            const uint8_t nib = (nibble == 0)
                                ? ((bb_val >> 4) & 0xF)
                                : (bb_val & 0xF);
                            B_lds[kk * TILE_N + col] = __float2half((float)((int)nib - 8));
                        }
                    } else {
                        // zero-fill guard for out-of-range cols handled on read
                    }
                }
            }
        }

        __syncthreads();

        // ---- 2 WMMAs per j block (K1 = 32 = 2 * WMMA_K) ----
        #pragma unroll
        for(int step = 0; step < K1 / WMMA_K; ++step) {
            const int kk0 = step * WMMA_K;

            half16_t A_frag;
            half16_t B_frag;

            // A: this wave's 16 rows [wave*16 .. wave*16+15] of A_lds,
            //    16 K cols [kk0 .. kk0+15].
            const int my_a_row = wave * 16 + row_lane;  // 0..31 in A_lds
            #pragma unroll
            for(int c = 0; c < 16; ++c) {
                A_frag[c] = (__fp16)(float)A_lds[my_a_row * K1 + kk0 + c];
            }

            // B: 16 K rows [kk0 .. kk0+15], col row_lane.
            #pragma unroll
            for(int r = 0; r < 16; ++r) {
                B_frag[r] = (__fp16)(float)B_lds[(kk0 + r) * TILE_N + row_lane];
            }

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A_frag, B_frag, acc);
        }

        __syncthreads();
    }

    // Store: each wave owns rows [wave*16 .. wave*16+15], interleaved within
    //        the 16x16 subtile.
    const int wave_m0 = m0 + wave * 16;
    #pragma unroll
    for(int r = 0; r < 8; ++r) {
        const int m_out = wave_m0 + 2 * r + half;
        const int n_out = n0 + row_lane;
        if(m_out < M && n_out < N) {
            C[m_out * N + n_out] = __float2half(acc[r]);
        }
    }
}

}  // namespace

// Exposed via an internal extern "C" entry; the C API (ck_gemm.cpp) will add a
// backend-switch once this kernel is correct. For Phase 1 we expose a direct
// launcher so tests/test_standalone.cpp can call it without modifying the C API.
extern "C" {

// Phase 1 naive launcher — kept as the correctness reference.
void rcpp_standalone_launch(
    const void* A_dev, const void* B_dev, void* C_dev,
    int M, int N, int K, void* stream)
{
    dim3 block(16, 16, 1);
    dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
    hipLaunchKernelGGL(standalone_ternary_gemm_naive,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_dev),
                       static_cast<const uint8_t*>(B_dev),
                       static_cast<__half*>(C_dev),
                       M, N, K);
}

// Phase 2 LDS-tiled launcher.
void rcpp_standalone_launch_lds(
    const void* A_dev, const void* B_dev, void* C_dev,
    int M, int N, int K, void* stream)
{
    dim3 block(16, 16, 1);
    dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
    hipLaunchKernelGGL(standalone_ternary_gemm_lds,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_dev),
                       static_cast<const uint8_t*>(B_dev),
                       static_cast<__half*>(C_dev),
                       M, N, K);
}

// Phase 3 WMMA launcher — one wave per 16x16 output tile.
void rcpp_standalone_launch_wmma(
    const void* A_dev, const void* B_dev, void* C_dev,
    int M, int N, int K, void* stream)
{
    dim3 block(32, 1, 1);
    dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
    hipLaunchKernelGGL(standalone_ternary_gemm_wmma,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_dev),
                       static_cast<const uint8_t*>(B_dev),
                       static_cast<__half*>(C_dev),
                       M, N, K);
}

// Phase 4 WMMA + LDS + 2 waves per block (32x16 output tile).
void rcpp_standalone_launch_wmma_2wave(
    const void* A_dev, const void* B_dev, void* C_dev,
    int M, int N, int K, void* stream)
{
    dim3 block(64, 1, 1);
    dim3 grid((N + 15) / 16, (M + 31) / 32, 1);
    hipLaunchKernelGGL(standalone_ternary_gemm_wmma_2wave,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_dev),
                       static_cast<const uint8_t*>(B_dev),
                       static_cast<__half*>(C_dev),
                       M, N, K);
}

// Phase 4b: 1 wave, 16x64 output tile (4 B fragments per K step).
void rcpp_standalone_launch_wmma_4x(
    const void* A_dev, const void* B_dev, void* C_dev,
    int M, int N, int K, void* stream)
{
    dim3 block(32, 1, 1);
    dim3 grid((N + 63) / 64, (M + 15) / 16, 1);
    hipLaunchKernelGGL(standalone_ternary_gemm_wmma_4x,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_dev),
                       static_cast<const uint8_t*>(B_dev),
                       static_cast<__half*>(C_dev),
                       M, N, K);
}

// Phase 4c: 1 wave, 32x64 output tile (2 A x 4 B = 8 accumulators).
void rcpp_standalone_launch_wmma_2x4(
    const void* A_dev, const void* B_dev, void* C_dev,
    int M, int N, int K, void* stream)
{
    dim3 block(32, 1, 1);
    dim3 grid((N + 63) / 64, (M + 31) / 32, 1);
    hipLaunchKernelGGL(standalone_ternary_gemm_wmma_2x4,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_dev),
                       static_cast<const uint8_t*>(B_dev),
                       static_cast<__half*>(C_dev),
                       M, N, K);
}

// Phase 4g: 1 wave, 64x64 output (4A x 4B = 16 accumulators).
void rcpp_standalone_launch_wmma_4x4(
    const void* A_dev, const void* B_dev, void* C_dev,
    int M, int N, int K, void* stream)
{
    dim3 block(32, 1, 1);
    dim3 grid((N + 63) / 64, (M + 63) / 64, 1);
    hipLaunchKernelGGL(standalone_ternary_gemm_wmma_4x4,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_dev),
                       static_cast<const uint8_t*>(B_dev),
                       static_cast<__half*>(C_dev),
                       M, N, K);
}

// Phase 4h: 1 wave, 64x64 output, A and B cached across K steps.
void rcpp_standalone_launch_wmma_4x4_cached(
    const void* A_dev, const void* B_dev, void* C_dev,
    int M, int N, int K, void* stream)
{
    dim3 block(32, 1, 1);
    dim3 grid((N + 63) / 64, (M + 63) / 64, 1);
    hipLaunchKernelGGL(standalone_ternary_gemm_wmma_4x4_cached,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_dev),
                       static_cast<const uint8_t*>(B_dev),
                       static_cast<__half*>(C_dev),
                       M, N, K);
}

// Phase 4i: 4h + vectorized A loads.
void rcpp_standalone_launch_wmma_4x4_vec(
    const void* A_dev, const void* B_dev, void* C_dev,
    int M, int N, int K, void* stream)
{
    dim3 block(32, 1, 1);
    dim3 grid((N + 63) / 64, (M + 63) / 64, 1);
    hipLaunchKernelGGL(standalone_ternary_gemm_wmma_4x4_vec,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_dev),
                       static_cast<const uint8_t*>(B_dev),
                       static_cast<__half*>(C_dev),
                       M, N, K);
}

// FP16-B variant: skip pk_i4 decode. Consumer holds materialized FP16 B.
void rcpp_standalone_launch_fp16b(
    const void* A_dev, const void* B_fp16_dev, void* C_dev,
    int M, int N, int K, void* stream)
{
    dim3 block(32, 1, 1);
    dim3 grid((N + 63) / 64, (M + 63) / 64, 1);
    hipLaunchKernelGGL(standalone_gemm_fp16b,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_dev),
                       static_cast<const __half*>(B_fp16_dev),
                       static_cast<__half*>(C_dev),
                       M, N, K);
}

// Phase 4k: 2 waves, 64x32 output, LDS ping-pong.
void rcpp_standalone_launch_lds_pingpong(
    const void* A_dev, const void* B_dev, void* C_dev,
    int M, int N, int K, void* stream)
{
    dim3 block(64, 1, 1);
    dim3 grid((N + 31) / 32, (M + 63) / 64, 1);
    hipLaunchKernelGGL(standalone_gemm_lds_pingpong,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_dev),
                       static_cast<const uint8_t*>(B_dev),
                       static_cast<__half*>(C_dev),
                       M, N, K);
}

// Phase 4j: 1 wave, 32x64 output, A double-buffered across j iterations.
void rcpp_standalone_launch_4c_pipelined(
    const void* A_dev, const void* B_dev, void* C_dev,
    int M, int N, int K, void* stream)
{
    dim3 block(32, 1, 1);
    dim3 grid((N + 63) / 64, (M + 31) / 32, 1);
    hipLaunchKernelGGL(standalone_gemm_4c_pipelined,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_dev),
                       static_cast<const uint8_t*>(B_dev),
                       static_cast<__half*>(C_dev),
                       M, N, K);
}

// Decode pk_i4 WMMA-permuted bytes to col-major FP16 on device.
void rcpp_decode_pk_i4_to_fp16_launch(
    const void* B_packed_dev, void* B_fp16_dev,
    int K, int N, void* stream)
{
    dim3 block(64, 1, 1);
    dim3 grid((unsigned)N, 1, 1);
    hipLaunchKernelGGL(decode_pk_i4_to_fp16,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const uint8_t*>(B_packed_dev),
                       static_cast<__half*>(B_fp16_dev),
                       K, N);
}

// Phase 4f: 4 waves (2x2), 64x64 output, A+B both in LDS.
void rcpp_standalone_launch_wmma_2x2wave(
    const void* A_dev, const void* B_dev, void* C_dev,
    int M, int N, int K, void* stream)
{
    dim3 block(128, 1, 1);
    dim3 grid((N + 63) / 64, (M + 63) / 64, 1);
    hipLaunchKernelGGL(standalone_ternary_gemm_wmma_2x2wave,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_dev),
                       static_cast<const uint8_t*>(B_dev),
                       static_cast<__half*>(C_dev),
                       M, N, K);
}

// Phase 4d: 1 wave, 32x64 output, B decoded to LDS once per j block.
void rcpp_standalone_launch_wmma_2x4_ldsB(
    const void* A_dev, const void* B_dev, void* C_dev,
    int M, int N, int K, void* stream)
{
    dim3 block(32, 1, 1);
    dim3 grid((N + 63) / 64, (M + 31) / 32, 1);
    hipLaunchKernelGGL(standalone_ternary_gemm_wmma_2x4_ldsB,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_dev),
                       static_cast<const uint8_t*>(B_dev),
                       static_cast<__half*>(C_dev),
                       M, N, K);
}

}  // extern "C"
