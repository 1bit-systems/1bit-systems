/** gemm_npu_instructions.cpp
 *  Open-source GEMM NPU instruction sequence generator.
 *  Drop-in replacement for FastFlowLM's proprietary libgemm.so.
 *
 *  Uses npu_sequence public API (rtp_write, npu_dma_memcpy_nd, npu_dma_wait)
 *  from FastFlowLM's open-source headers (npu_instr_utils.hpp).
 *
 *  Generates the exact same NPU instruction sequences as the proprietary binary
 *  does — XAIE_IO_BLOCKWRITE, XAIE_IO_WRITE, XAIE_IO_CUSTOM_OP_DDR_PATCH,
 *  XAIE_IO_MASKWRITE, XAIE_IO_CUSTOM_OP_TCT.
 *
 *  NPU2 (Strix Halo): 6 rows × 8 cols AIE tile array
 *    Row 0: IT0-IT7   Shim tiles (DDR DMA)
 *    Row 1: MT0-MT7   Memory tiles (L2 scratchpad)
 *    Rows 2-5: CT00-CT37  Compute tiles (AIE cores)
 */

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cassert>
#include <climits>
#include <stdexcept>

// FastFlowLM open-source headers (included in the 1bit-systems repo)
#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"

// ─── Constants (from reverse-engineered libgemm.so) ────────────────────
static constexpr uint32_t NPU_COLS       = 8;
static constexpr uint32_t FIRST_CT_ROW   = 2;  // compute tile rows start here

// Shim tile array (matches proprietary Gemm::Impl::shim_tiles)
static constexpr npu_tiles SHIM_TILES[8] = {
    IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7
};

// RTP register addresses (from proprietary disassembly)
static constexpr uint32_t REG_M          = 0x1000;  // output rows
static constexpr uint32_t REG_K          = 0x1004;  // inner reduction size
static constexpr uint32_t REG_N          = 0x1008;  // output cols
static constexpr uint32_t REG_ACT        = 0x100c;  // activation: 0=none,1=GeLU,2=SiLU
static constexpr uint32_t REG_BIAS       = 0x1010;  // bias enable
static constexpr uint32_t REG_KICK       = 0x1f0a0; // kernel kick-off (write 1)

// Queue push register (0x1d204 = MM2S queue)
static constexpr uint32_t REG_QUEUE_PUSH = 0x1d204;

// Tile sizes (from proprietary kernel config)
static constexpr uint32_t BLOCK_K = 32;   // K-block size
static constexpr uint32_t BLOCK_N = 128;  // N-block size

// ─── Helpers ───────────────────────────────────────────────────────────
static inline uint32_t div_ceil(uint32_t x, uint32_t y) {
    return (x + y - 1) / y;
}

static inline npu_tiles tile_at(uint32_t row, uint32_t col) {
    return static_cast<npu_tiles>((row << 4) | col);
}

// ─── GEMM NPU Instruction Generator ────────────────────────────────────
//
// Generates the NPU instruction sequence for:
//   C[M][N] = A[M][K] × B[K][N] + bias[N]  (optional activation)
//
// Weight layout (Q4NX reordered):
//   A weights at weight_offset: [num_k_blocks][M][block_k] bf16
//   B weights at offset + K*M: [num_k_blocks][block_k][N] bf16
//
void gemm_generate_sequence(
    npu_sequence*           seq,
    uint32_t                M,              // output rows
    uint32_t                K,              // inner dimension
    uint32_t                N,              // output cols
    uint32_t                weight_offset,  // DDR byte offset to A weights
    bool                    add_bias,
    int                     activation,     // 0=none, 1=GeLU, 2=SiLU
    uint32_t                bias_offset,    // DDR byte offset to bias
    uint32_t                output_offset   // DDR byte offset for output (0=no writeback)
) {
    uint32_t num_col_tiles = div_ceil(N, BLOCK_N);
    uint32_t num_k_blocks  = div_ceil(K, BLOCK_K);

    // ── Preemption point ──
    seq->npu_preemption(0);

    // ── K-loop: each iteration processes one K-block ──
    for (uint32_t kb = 0; kb < num_k_blocks; kb++) {
        uint32_t k_start = kb * BLOCK_K;
        uint32_t k_size  = std::min(BLOCK_K, K - k_start);

        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col     = tc % NPU_COLS;
            uint32_t n_start = tc * BLOCK_N;
            uint32_t n_size  = std::min(BLOCK_N, N - n_start);

            // ── DMA: Load A tile (M × k_size) from DDR via SHIM tile ──
            uint32_t a_off = weight_offset + k_start * M * 2;  // bf16 = 2 bytes
            seq->npu_dma_memcpy_nd(
                2,                      // elem_size (bf16)
                0,                      // arg_idx
                S2MM,                   // direction: DDR → AIE
                tile_at(0, col),        // SHIM tile
                bd_0,                   // BD ID
                it_channel_0,           // channel
                {0, 0, 0, a_off},       // offsets
                {1, 1, k_size, M},      // sizes
                {0, 0, 0, 2},           // strides (bf16)
                -1, 0, false, normal_cache
            );

            // ── DMA: Load B tile (k_size × n_size) from DDR ──
            uint32_t b_off = weight_offset +
                             (K * M + k_start * N + n_start) * 2;
            seq->npu_dma_memcpy_nd(
                2,
                1,
                S2MM,
                tile_at(0, col),
                bd_1,
                it_channel_0,
                {0, 0, 0, b_off},
                {1, 1, n_size, k_size},
                {0, 0, 0, 2},
                -1, 0, false, normal_cache
            );

            // ── DMA: Load bias (if enabled) ──
            if (add_bias && kb == 0) {
                seq->npu_dma_memcpy_nd(
                    4,                      // elem_size (float32)
                    2,
                    S2MM,
                    tile_at(0, col),
                    bd_2,
                    it_channel_0,
                    {0, 0, 0, bias_offset + n_start * 4},
                    {1, 1, 1, n_size},
                    {0, 0, 0, 4},
                    -1, 0, false, normal_cache
                );
            }

            // ── RTP writes: configure AIE kernel on compute tile ──
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_M,    M);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_K,    k_size);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_N,    n_size);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_ACT,  activation);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_BIAS, add_bias ? 1 : 0);
        }

        // ── Push DMA queues on all SHIM tiles ──
        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col = tc % NPU_COLS;
            // Queue push for BD 0 (A-load), MM2S, ch0
            seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH,
                           (0 << 0) | (0 << 3) | 0x10 |
                           (0 << 31));  // bd=0, ch=0, MM2S, no issue_token
            // Queue push for BD 1 (B-load)
            seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH,
                           (1 << 0) | (0 << 3) | 0x10);
            if (add_bias && kb == 0) {
                seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH,
                               (2 << 0) | (0 << 3) | 0x10);
            }
        }

        // ── Wait for DMA completion on all SHIM tiles ──
        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col = tc % NPU_COLS;
            seq->npu_dma_wait(tile_at(0, col), S2MM, it_channel_0);
        }

        // ── Kick off compute on CT tiles ──
        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col = tc % NPU_COLS;
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_KICK, 1);
        }

        // ── Wait for compute completion ──
        // (CT tiles issue completion tokens via the DMA engine)
    }

    // ── Optional: write output back to DDR ──
    if (output_offset != 0) {
        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col     = tc % NPU_COLS;
            uint32_t n_size  = std::min(BLOCK_N, N - tc * BLOCK_N);
            uint32_t out_off = output_offset + tc * BLOCK_N * 2;

            seq->npu_dma_memcpy_nd(
                2,
                3,                      // arg slot for output
                S2MM,
                tile_at(0, col),
                bd_3,
                it_channel_0,
                {0, 0, 0, out_off},
                {1, 1, n_size, M},
                {0, 0, 0, 2},
                15, 0,                  // packet for sync
                true,                   // issue token
                normal_cache
            );
            seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH,
                           (3 << 0) | (0 << 3) | 0x10);
            seq->npu_dma_wait(tile_at(0, col), S2MM, it_channel_0);
        }
    }

    // ── Finalize: serialize command list → NPU instruction sequence ──
    seq->cmds2seq();
}

// ─── GEMM NPU Instruction Generator (INT8) ──────────────────────────────
//
// Identical structure to gemm_generate_sequence() but for int8 weights:
//   - DMA elem_size = 1 (int8 byte vs bf16 2-byte)
//   - Weight offset calculations use 1-byte elements instead of 2-byte
//   - Weight stride is 1 byte instead of 2 bytes
//   - Tile sizes (BLOCK_K=32, BLOCK_N=128), register addresses, and
//     RTP writes are all unchanged.
//
void gemm_generate_sequence_i8(
    npu_sequence*           seq,
    uint32_t                M,              // output rows
    uint32_t                K,              // inner dimension
    uint32_t                N,              // output cols
    uint32_t                weight_offset,  // DDR byte offset to A weights
    bool                    add_bias,
    int                     activation,     // 0=none, 1=GeLU, 2=SiLU
    uint32_t                bias_offset,    // DDR byte offset to bias
    uint32_t                output_offset   // DDR byte offset for output (0=no writeback)
) {
    uint32_t num_col_tiles = div_ceil(N, BLOCK_N);
    uint32_t num_k_blocks  = div_ceil(K, BLOCK_K);

    // ── Preemption point ──
    seq->npu_preemption(0);

    // ── K-loop: each iteration processes one K-block ──
    for (uint32_t kb = 0; kb < num_k_blocks; kb++) {
        uint32_t k_start = kb * BLOCK_K;
        uint32_t k_size  = std::min(BLOCK_K, K - k_start);

        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col     = tc % NPU_COLS;
            uint32_t n_start = tc * BLOCK_N;
            uint32_t n_size  = std::min(BLOCK_N, N - n_start);

            // ── DMA: Load A tile (M × k_size) from DDR via SHIM tile ──
            // int8 = 1 byte per element
            uint32_t a_off = weight_offset + k_start * M * 1;
            seq->npu_dma_memcpy_nd(
                1,                      // elem_size (int8)
                0,                      // arg_idx
                S2MM,                   // direction: DDR → AIE
                tile_at(0, col),        // SHIM tile
                bd_0,                   // BD ID
                it_channel_0,           // channel
                {0, 0, 0, a_off},       // offsets
                {1, 1, k_size, M},      // sizes
                {0, 0, 0, 1},           // strides (int8)
                -1, 0, false, normal_cache
            );

            // ── DMA: Load B tile (k_size × n_size) from DDR ──
            uint32_t b_off = weight_offset +
                             (K * M + k_start * N + n_start) * 1;
            seq->npu_dma_memcpy_nd(
                1,
                1,
                S2MM,
                tile_at(0, col),
                bd_1,
                it_channel_0,
                {0, 0, 0, b_off},
                {1, 1, n_size, k_size},
                {0, 0, 0, 1},
                -1, 0, false, normal_cache
            );

            // ── DMA: Load bias (if enabled) ──
            if (add_bias && kb == 0) {
                seq->npu_dma_memcpy_nd(
                    4,                      // elem_size (float32)
                    2,
                    S2MM,
                    tile_at(0, col),
                    bd_2,
                    it_channel_0,
                    {0, 0, 0, bias_offset + n_start * 4},
                    {1, 1, 1, n_size},
                    {0, 0, 0, 4},
                    -1, 0, false, normal_cache
                );
            }

            // ── RTP writes: configure AIE kernel on compute tile ──
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_M,    M);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_K,    k_size);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_N,    n_size);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_ACT,  activation);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_BIAS, add_bias ? 1 : 0);
        }

        // ── Push DMA queues on all SHIM tiles ──
        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col = tc % NPU_COLS;
            // Queue push for BD 0 (A-load), MM2S, ch0
            seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH,
                           (0 << 0) | (0 << 3) | 0x10 |
                           (0 << 31));  // bd=0, ch=0, MM2S, no issue_token
            // Queue push for BD 1 (B-load)
            seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH,
                           (1 << 0) | (0 << 3) | 0x10);
            if (add_bias && kb == 0) {
                seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH,
                               (2 << 0) | (0 << 3) | 0x10);
            }
        }

        // ── Wait for DMA completion on all SHIM tiles ──
        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col = tc % NPU_COLS;
            seq->npu_dma_wait(tile_at(0, col), S2MM, it_channel_0);
        }

        // ── Kick off compute on CT tiles ──
        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col = tc % NPU_COLS;
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_KICK, 1);
        }

        // ── Wait for compute completion ──
        // (CT tiles issue completion tokens via the DMA engine)
    }

    // ── Optional: write output back to DDR ──
    if (output_offset != 0) {
        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col     = tc % NPU_COLS;
            uint32_t n_size  = std::min(BLOCK_N, N - tc * BLOCK_N);
            uint32_t out_off = output_offset + tc * BLOCK_N * 1;

            seq->npu_dma_memcpy_nd(
                1,
                3,                      // arg slot for output
                S2MM,
                tile_at(0, col),
                bd_3,
                it_channel_0,
                {0, 0, 0, out_off},
                {1, 1, n_size, M},
                {0, 0, 0, 1},
                15, 0,                  // packet for sync
                true,                   // issue token
                normal_cache
            );
            seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH,
                           (3 << 0) | (0 << 3) | 0x10);
            seq->npu_dma_wait(tile_at(0, col), S2MM, it_channel_0);
        }
    }

    // ── Finalize: serialize command list → NPU instruction sequence ──
    seq->cmds2seq();
}

// ─── GEMM NPU Instruction Generator (INT8) — Split A/B offsets ────────
//
// Variant of gemm_generate_sequence_i8() for the hybrid FLM engine where
// activations (A) and weights (B) reside in separate NPU address spaces
// (different kernel BO args). The existing gemm_generate_sequence_i8()
// applies weight_offset to BOTH A and B, which is incompatible with having
// A in a per-inference BO and B in a single contiguous weight BO.
//
// This variant takes separate a_ddr_offset and b_base_offset parameters:
//   a_ddr_offset: DDR byte offset for A within the activation BO (bA)
//   b_base_offset: DDR byte offset for the START of B within weight BO (bW)
//                  per-layer: b_base_offset = layer * K * N
//   (K*M is NO longer added to B — caller controls the base offset directly)
//
void gemm_generate_sequence_i8_split(
    npu_sequence*           seq,
    uint32_t                M,              // output rows
    uint32_t                K,              // inner dimension
    uint32_t                N,              // output cols
    uint32_t                a_ddr_offset,   // DDR byte offset for A (within arg_idx=0 BO)
    uint32_t                b_base_offset,  // DDR byte offset for B start (within arg_idx=1 BO)
    bool                    add_bias,
    int                     activation,     // 0=none, 1=GeLU, 2=SiLU
    uint32_t                bias_offset,    // DDR byte offset to bias
    uint32_t                output_offset   // DDR byte offset for output (0=no writeback)
) {
    uint32_t num_col_tiles = div_ceil(N, BLOCK_N);
    uint32_t num_k_blocks  = div_ceil(K, BLOCK_K);

    // ── Preemption point ──
    seq->npu_preemption(0);

    // ── K-loop: each iteration processes one K-block ──
    for (uint32_t kb = 0; kb < num_k_blocks; kb++) {
        uint32_t k_start = kb * BLOCK_K;
        uint32_t k_size  = std::min(BLOCK_K, K - k_start);

        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col     = tc % NPU_COLS;
            uint32_t n_start = tc * BLOCK_N;
            uint32_t n_size  = std::min(BLOCK_N, N - n_start);

            // ── DMA: Load A tile (M × k_size) from DDR via SHIM tile ──
            // A is at a_ddr_offset + k_start * M (within bA, arg_idx=0)
            uint32_t a_off = a_ddr_offset + k_start * M * 1;
            seq->npu_dma_memcpy_nd(
                1, 0, S2MM, tile_at(0, col), bd_0, it_channel_0,
                {0, 0, 0, a_off},
                {1, 1, k_size, M},
                {0, 0, 0, 1},
                -1, 0, false, normal_cache
            );

            // ── DMA: Load B tile (k_size × n_size) from DDR ──
            // B is at b_base_offset + k_start * N + n_start (within bW, arg_idx=1)
            // Note: K*M is NOT added — caller controls the full offset.
            uint32_t b_off = b_base_offset + k_start * N * 1 + n_start * 1;
            seq->npu_dma_memcpy_nd(
                1, 1, S2MM, tile_at(0, col), bd_1, it_channel_0,
                {0, 0, 0, b_off},
                {1, 1, n_size, k_size},
                {0, 0, 0, 1},
                -1, 0, false, normal_cache
            );

            // ── DMA: Load bias (if enabled) ──
            if (add_bias && kb == 0) {
                seq->npu_dma_memcpy_nd(
                    4, 2, S2MM, tile_at(0, col), bd_2, it_channel_0,
                    {0, 0, 0, bias_offset + n_start * 4},
                    {1, 1, 1, n_size},
                    {0, 0, 0, 4},
                    -1, 0, false, normal_cache
                );
            }

            // ── RTP writes: configure AIE kernel on compute tile ──
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_M,    M);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_K,    k_size);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_N,    n_size);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_ACT,  activation);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_BIAS, add_bias ? 1 : 0);
        }

        // ── Push DMA queues on all SHIM tiles ──
        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col = tc % NPU_COLS;
            seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH,
                           (0 << 0) | (0 << 3) | 0x10 | (0 << 31));
            seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH,
                           (1 << 0) | (0 << 3) | 0x10);
            if (add_bias && kb == 0) {
                seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH,
                               (2 << 0) | (0 << 3) | 0x10);
            }
        }

        // ── Wait for DMA completion on all SHIM tiles ──
        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col = tc % NPU_COLS;
            seq->npu_dma_wait(tile_at(0, col), S2MM, it_channel_0);
        }

        // ── Kick off compute on CT tiles ──
        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col = tc % NPU_COLS;
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_KICK, 1);
        }
    }

    // ── Optional: write output back to DDR ──
    if (output_offset != 0) {
        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col     = tc % NPU_COLS;
            uint32_t n_size  = std::min(BLOCK_N, N - tc * BLOCK_N);
            uint32_t out_off = output_offset + tc * BLOCK_N * 1;
            seq->npu_dma_memcpy_nd(
                1, 3, S2MM, tile_at(0, col), bd_3, it_channel_0,
                {0, 0, 0, out_off},
                {1, 1, n_size, M},
                {0, 0, 0, 1},
                15, 0, true, normal_cache
            );
            seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH,
                           (3 << 0) | (0 << 3) | 0x10);
            seq->npu_dma_wait(tile_at(0, col), S2MM, it_channel_0);
        }
    }

    seq->cmds2seq();
}

// ─── GEMM NPU Instruction Generator (TQ2 Ternary) ──────────────────────
//
// Generates NPU instructions for TQ2 ternary GEMM with on-tile LUT decode.
// DDR stores TQ2 packed data (2-bit codes + BF16 scales) — 4× less DDR
// traffic than INT8. The AIE tile unpacks TQ2→bf16 via LUT before mac.
//
// Weight BO layout (per layer):
//   [TQ2 codes: K×N/4 bytes] [BF16 scales: N×8×2 bytes]
//   Total: K*N/4 + N*16 bytes per layer
//
// vs INT8: K*N bytes —> TQ2 is ~4× smaller DDR footprint.
//
// The AIE tile reads:
//   A (activations): INT8 (same as i8 path)
//   B (TQ2 codes): uint8_t — unpacked via LUT to bf16 internally
//   Scales: bfloat16 — applied after LUT decode
//
void gemm_generate_sequence_tq2(
    npu_sequence*           seq,
    uint32_t                M,              // output rows (batch)
    uint32_t                K,              // inner dimension
    uint32_t                N,              // output cols
    uint32_t                weight_offset,  // DDR byte offset to TQ2 codes
    bool                    add_bias,
    int                     activation,     // 0=none, 1=GeLU, 2=SiLU
    uint32_t                bias_offset,
    uint32_t                output_offset
) {
    uint32_t num_col_tiles = div_ceil(N, BLOCK_N);
    uint32_t num_k_blocks  = div_ceil(K, BLOCK_K);

    // TQ2 tile bytes per K×N block:
    //   Codes: K * N / 4 bytes (4 codes per byte)
    //   Scales: N * 8 * 2 bytes (8 groups of bf16 per N column, for K=64 block)
    uint32_t tq2_codes_bytes   = K * N / 4;
    uint32_t tq2_scales_bytes  = N * 8 * 2;
    uint32_t tq2_block_bytes   = tq2_codes_bytes + tq2_scales_bytes;

    seq->npu_preemption(0);

    for (uint32_t kb = 0; kb < num_k_blocks; kb++) {
        uint32_t k_start = kb * BLOCK_K;
        uint32_t k_size  = std::min(BLOCK_K, K - k_start);

        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col     = tc % NPU_COLS;
            uint32_t n_start = tc * BLOCK_N;
            uint32_t n_size  = std::min(BLOCK_N, N - n_start);

            // ── DMA: Load A (activations, INT8) ──
            uint32_t a_off = weight_offset + k_start * M * 1;  // A at start of weight BO
            seq->npu_dma_memcpy_nd(
                1, 0, S2MM, tile_at(0, col), bd_0, it_channel_0,
                {0, 0, 0, a_off}, {1, 1, k_size, M}, {0, 0, 0, 1},
                -1, 0, false, normal_cache
            );

            // ── DMA: Load B TQ2 codes (packed 2-bit) ──
            // Codes are at weight_offset + K*M (right after activations)
            uint32_t b_codes_off = weight_offset + K * M * 1
                                   + kb * (BLOCK_K * N / 4)
                                   + n_start * (BLOCK_K / 4);
            seq->npu_dma_memcpy_nd(
                1, 1, S2MM, tile_at(0, col), bd_1, it_channel_0,
                {0, 0, 0, b_codes_off},
                {1, 1, n_size * BLOCK_K / 4, 1},
                {0, 0, 0, 1},
                -1, 0, false, normal_cache
            );

            // ── DMA: Load B TQ2 scales (BF16, 8 groups per N-column) ──
            uint32_t b_scales_off = weight_offset + K * M * 1
                                    + num_k_blocks * (BLOCK_K * N / 4)
                                    + n_start * 8 * 2;
            seq->npu_dma_memcpy_nd(
                2, 2, S2MM, tile_at(0, col), bd_2, it_channel_0,
                {0, 0, 0, b_scales_off},
                {1, 1, n_size * 8 * 2, 1},
                {0, 0, 0, 2},
                -1, 0, false, normal_cache
            );

            // ── DMA: Load bias (if enabled) ──
            if (add_bias && kb == 0) {
                seq->npu_dma_memcpy_nd(
                    4, 3, S2MM, tile_at(0, col), bd_3, it_channel_0,
                    {0, 0, 0, bias_offset + n_start * 4},
                    {1, 1, 1, n_size}, {0, 0, 0, 4},
                    -1, 0, false, normal_cache
                );
            }

            // ── RTP: configure AIE ternary kernel ──
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_M,    M);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_K,    k_size);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_N,    n_size);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_ACT,  activation);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_BIAS, add_bias ? 1 : 0);
        }

        // ── Push DMA queues on all SHIM tiles ──
        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col = tc % NPU_COLS;
            seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH,
                           (0 << 0) | (0 << 3) | 0x10 | (0 << 31));
            seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH,
                           (1 << 0) | (0 << 3) | 0x10);
            seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH,
                           (2 << 0) | (0 << 3) | 0x10);
            if (add_bias && kb == 0)
                seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH,
                               (3 << 0) | (0 << 3) | 0x10);
        }

        // ── Wait for DMA ──
        for (uint32_t tc = 0; tc < num_col_tiles; tc++)
            seq->npu_dma_wait(tile_at(0, tc % NPU_COLS), S2MM, it_channel_0);

        // ── Kick compute ──
        for (uint32_t tc = 0; tc < num_col_tiles; tc++)
            seq->rtp_write(tile_at(FIRST_CT_ROW, tc % NPU_COLS), REG_KICK, 1);
    }

    // ── Write output ──
    if (output_offset != 0) {
        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col = tc % NPU_COLS;
            uint32_t n_size = std::min(BLOCK_N, N - tc * BLOCK_N);
            uint32_t out_off = output_offset + tc * BLOCK_N * 1;
            seq->npu_dma_memcpy_nd(
                1, 4, S2MM, tile_at(0, col), bd_4, it_channel_0,
                {0, 0, 0, out_off}, {1, 1, n_size, M}, {0, 0, 0, 1},
                15, 0, true, normal_cache
            );
            seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH,
                           (4 << 0) | (0 << 3) | 0x10);
            seq->npu_dma_wait(tile_at(0, col), S2MM, it_channel_0);
        }
    }

    seq->cmds2seq();
}

// ─── MHA Attention NPU Instruction Generator ──────────────────────────
//
// Generates NPU sequence for multi-head attention:
//   O = softmax(Q × K^T / sqrt(d)) × V
//
// Supports all head_dim × quantization variants from proprietary libmha.so:
//   head_dim: 64, 128, 256
//   quant:    q2, q3, q4
//
void mha_generate_sequence(
    npu_sequence*           seq,
    uint32_t                head_dim,       // 64, 128, or 256
    int                     quant_bits,     // 2, 3, or 4
    uint32_t                num_heads,
    uint32_t                seq_len,        // current context length
    uint32_t                kv_offset,      // DDR offset to K/V cache
    uint32_t                q_offset,       // DDR offset to Q buffer
    uint32_t                out_offset      // DDR offset for output
) {
    // Validate parameters (same check as proprietary MHA)
    if (head_dim != 64 && head_dim != 128 && head_dim != 256) {
        throw std::runtime_error("MHA parameter check failed: head_dim must be 64, 128, or 256");
    }
    if (quant_bits < 2 || quant_bits > 4) {
        throw std::runtime_error("MHA parameter check failed: quant_bits must be 2-4");
    }

    // ── Step 1: Load Q projection from DDR ──
    seq->npu_dma_memcpy_nd(
        2, 0, S2MM, tile_at(0, 0), bd_0, it_channel_0,
        {0, 0, 0, q_offset},
        {1, 1, head_dim * num_heads, 1},
        {0, 0, 0, 2},
        -1, 0, false, normal_cache
    );

    // ── Step 2: Load K cache from DDR ──
    uint32_t k_cache_size = seq_len * head_dim * num_heads * 2;
    seq->npu_dma_memcpy_nd(
        2, 1, S2MM, tile_at(0, 0), bd_1, it_channel_0,
        {0, 0, 0, kv_offset},
        {1, 1, seq_len * head_dim * num_heads, 1},
        {0, 0, 0, 2},
        -1, 0, false, normal_cache
    );

    // ── Step 3: Load V cache from DDR ──
    uint32_t v_offset = kv_offset + seq_len * head_dim * num_heads * 2;
    seq->npu_dma_memcpy_nd(
        2, 2, S2MM, tile_at(0, 0), bd_2, it_channel_0,
        {0, 0, 0, v_offset},
        {1, 1, seq_len * head_dim * num_heads, 1},
        {0, 0, 0, 2},
        -1, 0, false, normal_cache
    );

    // ── Step 4: Push DMA queues ──
    for (uint32_t bd = 0; bd < 3; bd++) {
        seq->rtp_write(tile_at(0, 0), REG_QUEUE_PUSH,
                       (bd << 0) | (0 << 3) | 0x10);
    }

    // ── Step 5: Wait for DMA ──
    seq->npu_dma_wait(tile_at(0, 0), S2MM, it_channel_0);

    // ── Step 6: Configure MHA kernel on compute tiles ──
    // CT00-07: QK^T matmul
    // CT10-17: Scale + causal mask + softmax
    // CT20-37: Score × V matmul
    for (uint32_t col = 0; col < 8; col++) {
        seq->rtp_write(tile_at(FIRST_CT_ROW, col),     0x1000, head_dim);
        seq->rtp_write(tile_at(FIRST_CT_ROW, col),     0x1004, seq_len);
        seq->rtp_write(tile_at(FIRST_CT_ROW, col),     0x1008, num_heads);
        seq->rtp_write(tile_at(FIRST_CT_ROW, col),     0x100c, quant_bits);
        seq->rtp_write(tile_at(FIRST_CT_ROW, col + 1), 0x1000, seq_len);  // softmax tile
        seq->rtp_write(tile_at(FIRST_CT_ROW, col + 1), 0x1004, head_dim);
        seq->rtp_write(tile_at(FIRST_CT_ROW + 2, col), 0x1000, head_dim); // score*V tile
        seq->rtp_write(tile_at(FIRST_CT_ROW + 2, col), 0x1004, seq_len);
    }

    // ── Step 7: Kick off compute ──
    for (uint32_t col = 0; col < 8; col++) {
        seq->rtp_write(tile_at(FIRST_CT_ROW, col), REG_KICK, 1);
    }

    // ── Step 8: Write output back ──
    seq->npu_dma_memcpy_nd(
        2, 3, S2MM, tile_at(0, 0), bd_3, it_channel_0,
        {0, 0, 0, out_offset},
        {1, 1, head_dim * num_heads, 1},
        {0, 0, 0, 2},
        15, 0, true, normal_cache
    );
    seq->rtp_write(tile_at(0, 0), REG_QUEUE_PUSH,
                   (3 << 0) | (0 << 3) | 0x10);
    seq->npu_dma_wait(tile_at(0, 0), S2MM, it_channel_0);

    seq->cmds2seq();
}
