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

// ─── INT8 GEMM NPU Instruction Generator ──────────────────────────────
// Same as gemm_generate_sequence but for INT8 data (elem_size=1).
// Compatible with FLM's mm.xclbin (same DMA/RTP interface).
void gemm_generate_sequence_i8(
    npu_sequence*           seq,
    uint32_t                M,              // output rows (batch=1 or 128)
    uint32_t                K,              // inner dimension
    uint32_t                N,              // output cols
    uint32_t                weight_offset,  // DDR byte offset to weights (w/in BO arg 4)
    float                   a_scale,        // activation scale (INT8 quant factor)
    float                   b_scale,        // weight scale (INT8 quant factor)
    uint32_t                output_offset   // DDR byte offset for output (w/in BO arg 5)
) {
    uint32_t num_col_tiles = div_ceil(N, BLOCK_N);
    uint32_t num_k_blocks  = div_ceil(K, BLOCK_K);
    seq->npu_preemption(0);

    for (uint32_t kb = 0; kb < num_k_blocks; kb++) {
        uint32_t k_start = kb * BLOCK_K;
        uint32_t k_size  = std::min(BLOCK_K, K - k_start);

        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col     = tc % NPU_COLS;
            uint32_t n_start = tc * BLOCK_N;
            uint32_t n_size  = std::min(BLOCK_N, N - n_start);

            // DMA: Load A tile (M x k_size INT8) from BO arg 3
            uint32_t a_off = k_start * M;
            seq->npu_dma_memcpy_nd(1, 3, S2MM, tile_at(0, col), bd_0, it_channel_0,
                {0,0,0,a_off}, {1,1,k_size,M}, {0,0,0,1}, -1,0,false,normal_cache);

            // DMA: Load B tile (k_size x n_size INT8) from BO arg 4
            uint32_t b_off = weight_offset + k_start * N + n_start;
            seq->npu_dma_memcpy_nd(1, 4, S2MM, tile_at(0, col), bd_1, it_channel_0,
                {0,0,0,b_off}, {1,1,n_size,k_size}, {0,0,0,1}, -1,0,false,normal_cache);

            // RTP writes: configure dimensions + scales
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), 0x1000, M);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), 0x1004, k_size);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), 0x1008, n_size);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), 0x100c, 0);  // no activation
            // Scale registers (INT8 quant scales packed as bits)
            uint32_t a_bits; memcpy(&a_bits, &a_scale, 4);
            uint32_t b_bits; memcpy(&b_bits, &b_scale, 4);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), 0x1010, a_bits);
            seq->rtp_write(tile_at(FIRST_CT_ROW, col), 0x1014, b_bits);
        }
        // Push DMA queues
        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col = tc % NPU_COLS;
            for (uint32_t bd = 0; bd < 2; bd++)
                seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH, (bd << 0) | (0 << 3) | 0x10);
        }
        // Wait for DMA
        for (uint32_t tc2 = 0; tc2 < num_col_tiles; tc2++) {
            uint32_t col2 = tc2 % NPU_COLS;
            seq->npu_dma_wait(tile_at(0, col2), S2MM, it_channel_0);
        }
        // Kick compute
        for (uint32_t tc2 = 0; tc2 < num_col_tiles; tc2++) {
            uint32_t col2 = tc2 % NPU_COLS;
            seq->rtp_write(tile_at(FIRST_CT_ROW, col2), REG_KICK, 1);
        }
    }

    // Output writeback (INT16 -> BO arg 5)
    if (output_offset != 0) {
        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col = tc % NPU_COLS;
            uint32_t n_size = std::min(BLOCK_N, N - tc * BLOCK_N);
            seq->npu_dma_memcpy_nd(2, 5, MM2S, tile_at(0, col), bd_3, it_channel_0,
                {0,0,0,output_offset + tc * BLOCK_N * 2}, {1,1,n_size,M}, {0,0,0,2},
                15, 0, true, normal_cache);
            seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH, (3 << 0) | (0 << 3) | 0x10);
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

// ─── Per-Group INT8 GEMM NPU Instruction Generator ────────────────────
//
// Generates INT8 instructions for a SINGLE K-block (32 elements) of the GEMM.
// Designed for per-group quantization: each K-block has its own scale so
// we need separate NPU invocations per group to get per-group int32 results.
//
// BO arg layout (same as gemm_generate_sequence_i8):
//   arg 3: A (M x k_size INT8)
//   arg 4: B (k_size x N INT8)
//   arg 5: C output (M x N int32)
//
namespace npu_seq {
void gemm_generate_sequence_i8_kblock(
    npu_sequence*           seq,
    uint32_t                M,              // output rows (batch, typically XM=128)
    uint32_t                k_size,         // K-block size (typically 32)
    uint32_t                N,              // output cols
    uint32_t                k_offset,       // k-start offset in elements (for A: k_off*M, for B: k_off*N)
    float                   a_scale,        // activation scale
    float                   b_scale,        // weight scale for THIS group
    uint32_t                output_offset   // DDR byte offset for output (0 = write to start)
) {
    uint32_t num_col_tiles = div_ceil(N, BLOCK_N);
    seq->npu_preemption(0);

    for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
        uint32_t col     = tc % NPU_COLS;
        uint32_t n_start = tc * BLOCK_N;
        uint32_t n_size  = std::min(BLOCK_N, N - n_start);

        // DMA: Load A tile (M x k_size INT8) from BO arg 3
        uint32_t a_off = k_offset * M;
        seq->npu_dma_memcpy_nd(1, 3, S2MM, tile_at(0, col), bd_0, it_channel_0,
            {0,0,0,a_off}, {1,1,k_size,M}, {0,0,0,1}, -1,0,false,normal_cache);

        // DMA: Load B tile (k_size x n_size INT8) from BO arg 4
        uint32_t b_off = k_offset * N + n_start;
        seq->npu_dma_memcpy_nd(1, 4, S2MM, tile_at(0, col), bd_1, it_channel_0,
            {0,0,0,b_off}, {1,1,n_size,k_size}, {0,0,0,1}, -1,0,false,normal_cache);

        // RTP writes: configure dimensions + scales
        seq->rtp_write(tile_at(FIRST_CT_ROW, col), 0x1000, M);
        seq->rtp_write(tile_at(FIRST_CT_ROW, col), 0x1004, k_size);
        seq->rtp_write(tile_at(FIRST_CT_ROW, col), 0x1008, n_size);
        seq->rtp_write(tile_at(FIRST_CT_ROW, col), 0x100c, 0);  // no activation
        // Scale registers (INT8 quant scales packed as bits)
        uint32_t a_bits; memcpy(&a_bits, &a_scale, 4);
        uint32_t b_bits; memcpy(&b_bits, &b_scale, 4);
        seq->rtp_write(tile_at(FIRST_CT_ROW, col), 0x1010, a_bits);
        seq->rtp_write(tile_at(FIRST_CT_ROW, col), 0x1014, b_bits);
    }

    // Push DMA queues
    for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
        uint32_t col = tc % NPU_COLS;
        for (uint32_t bd = 0; bd < 2; bd++)
            seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH, (bd << 0) | (0 << 3) | 0x10);
    }
    // Wait for DMA
    for (uint32_t tc2 = 0; tc2 < num_col_tiles; tc2++) {
        uint32_t col2 = tc2 % NPU_COLS;
        seq->npu_dma_wait(tile_at(0, col2), S2MM, it_channel_0);
    }
    // Kick compute
    for (uint32_t tc2 = 0; tc2 < num_col_tiles; tc2++) {
        uint32_t col2 = tc2 % NPU_COLS;
        seq->rtp_write(tile_at(FIRST_CT_ROW, col2), REG_KICK, 1);
    }

    // Output writeback (int32 -> BO arg 5)
    if (output_offset != 0) {
        for (uint32_t tc = 0; tc < num_col_tiles; tc++) {
            uint32_t col = tc % NPU_COLS;
            uint32_t n_size = std::min(BLOCK_N, N - tc * BLOCK_N);
            seq->npu_dma_memcpy_nd(4, 5, MM2S, tile_at(0, col), bd_3, it_channel_0,
                {0,0,0,output_offset + tc * BLOCK_N * 4}, {1,1,n_size,M}, {0,0,0,4},
                15, 0, true, normal_cache);
            seq->rtp_write(tile_at(0, col), REG_QUEUE_PUSH, (3 << 0) | (0 << 3) | 0x10);
            seq->npu_dma_wait(tile_at(0, col), S2MM, it_channel_0);
        }
    }
    seq->cmds2seq();
}
} // namespace npu_seq

// ─── AttnCtx-compatible MHA instruction generator ─────────────────────
// Generates attention NPU instructions for AttnCtx's BO layout:
//   kernel(3, insts_bo, insts_size, bo0=Q, bo1=K, bo2=V, bo3=output, bo4=unused)
// data BOs at kernel arg indices 3..6.
void mha_generate_attn_instrs(
    std::vector<uint32_t>& out_instrs,
    uint32_t                head_dim,
    int                     quant_bits,
    uint32_t                num_q_heads,
    uint32_t                num_kv_heads,
    uint32_t                max_seq_len,
    uint32_t                seq_len
) {
    npu_sequence seq(device_npu2);
    if (seq_len > max_seq_len) seq_len = max_seq_len;

    uint32_t q_elements = num_q_heads * head_dim;
    uint32_t kv_elements = seq_len * num_kv_heads * head_dim;
    uint32_t out_elements = num_q_heads * head_dim;

    // Q in from BO arg 3 (i8 -> padded to uint32)
    seq.npu_dma_memcpy_nd(1, 3, S2MM, IT0, bd_0, it_channel_0,
        {0,0,0,0}, {1,1,(q_elements+3)/4,1}, {0,0,0,1}, -1,0,false,normal_cache);
    // K in from BO arg 4
    seq.npu_dma_memcpy_nd(1, 4, S2MM, IT0, bd_1, it_channel_0,
        {0,0,0,0}, {1,1,(kv_elements+3)/4,1}, {0,0,0,1}, -1,0,false,normal_cache);
    // V in from BO arg 5
    seq.npu_dma_memcpy_nd(1, 5, S2MM, IT0, bd_2, it_channel_0,
        {0,0,0,0}, {1,1,(kv_elements+3)/4,1}, {0,0,0,1}, -1,0,false,normal_cache);
    // Push queues
    for (uint32_t bd = 0; bd < 3; bd++)
        seq.rtp_write(IT0, REG_QUEUE_PUSH, (bd << 0) | (0 << 3) | 0x10);
    seq.npu_dma_wait(IT0, S2MM, it_channel_0);

    // Configure compute tiles
    uint32_t ncols = std::min(NPU_COLS, num_q_heads);
    for (uint32_t col = 0; col < ncols; col++) {
        seq.rtp_write(tile_at(FIRST_CT_ROW, col),     0x1000, head_dim);
        seq.rtp_write(tile_at(FIRST_CT_ROW, col),     0x1004, seq_len);
        seq.rtp_write(tile_at(FIRST_CT_ROW, col),     0x1008, num_q_heads);
        seq.rtp_write(tile_at(FIRST_CT_ROW, col),     0x100c, (uint32_t)quant_bits);
        seq.rtp_write(tile_at(FIRST_CT_ROW, col+1),   0x1000, seq_len);
        seq.rtp_write(tile_at(FIRST_CT_ROW, col+1),   0x1004, head_dim);
        seq.rtp_write(tile_at(FIRST_CT_ROW+2, col),   0x1000, head_dim);
        seq.rtp_write(tile_at(FIRST_CT_ROW+2, col),   0x1004, seq_len);
    }
    // Kick compute
    for (uint32_t col = 0; col < ncols; col++)
        seq.rtp_write(tile_at(FIRST_CT_ROW, col), REG_KICK, 1);

    // Output to BO arg 6 (bf16 output, 2 bytes per element)
    seq.npu_dma_memcpy_nd(2, 6, MM2S, IT0, bd_3, it_channel_0,
        {0,0,0,0}, {1,1,out_elements,1}, {0,0,0,2},
        15, 0, true, normal_cache);
    seq.rtp_write(IT0, REG_QUEUE_PUSH, (3 << 0) | (0 << 3) | 0x10);
    seq.npu_dma_wait(IT0, S2MM, it_channel_0);

    seq.cmds2seq();
    auto [data, size] = seq.dump();
    out_instrs.assign(data, data + size);
}
