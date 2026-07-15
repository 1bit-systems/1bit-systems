/*!
 *  Copyright (c) 2025 by Contributors (adapted from FastFlowLM MIT source)
 * \file npu_instr.hpp
 * \brief NPU instruction sequence system — dynamically generates NPU instruction
 *        streams from high-level commands. Replaces pre-compiled .insts files.
 *
 * Architecture:
 *   npu_sequence holds a list of npu_cmd objects. Each cmd is a typed instruction
 *   (DMA block write, register write, DDR address patch, sync token, wait).
 *   cmds2seq() compiles commands to a linear uint32_t[] instruction stream.
 *   seq2cmds() parses a pre-existing instruction stream back to commands.
 *
 * This is the core of FLM's dynamic instruction generation. The compiled
 * instruction stream is loaded as an xrt::elf → xrt::module → xrt::ext::kernel,
 * then executed with BO arguments patched in via DDR_PATCH commands.
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <cassert>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <xrt/xrt_bo.h>

// ---------------------------------------------------------------------------
// NPU device parameters
// ---------------------------------------------------------------------------

/// NPU generation enum — mirrors XDNA hardware
enum class npu_device : uint32_t {
    npu1 = 1,  // Phoenix / Hawk Point (XDNA 1, 4 cols)
    npu2 = 4,  // Strix / Strix Halo (XDNA 2, 8 cols)
};

/// NPU tile addressing — row<<4 | col
/// Row 0 = Shim tiles (DMA), Row 1 = Mem tiles, Rows 2-5 = Compute tiles
enum class npu_tile : uint32_t {
    // Shim tiles (row 0)
    ST0 = 0x00, ST1 = 0x01, ST2 = 0x02, ST3 = 0x03,
    ST4 = 0x04, ST5 = 0x05, ST6 = 0x06, ST7 = 0x07,
    // Mem tiles (row 1)
    MT0 = 0x10, MT1 = 0x11, MT2 = 0x12, MT3 = 0x13,
    MT4 = 0x14, MT5 = 0x15, MT6 = 0x16, MT7 = 0x17,
    // Compute tiles (rows 2-5)
    CT00 = 0x20, CT01 = 0x21, CT02 = 0x22, CT03 = 0x23,
    CT04 = 0x24, CT05 = 0x25, CT06 = 0x26, CT07 = 0x27,
    CT10 = 0x30, CT11 = 0x31, CT12 = 0x32, CT13 = 0x33,
    CT14 = 0x34, CT15 = 0x35, CT16 = 0x36, CT17 = 0x37,
    CT20 = 0x40, CT21 = 0x41, CT22 = 0x42, CT23 = 0x43,
    CT24 = 0x44, CT25 = 0x45, CT26 = 0x46, CT27 = 0x47,
    CT30 = 0x50, CT31 = 0x51, CT32 = 0x52, CT33 = 0x53,
    CT34 = 0x54, CT35 = 0x55, CT36 = 0x56, CT37 = 0x57,
};

inline npu_tile make_tile(uint32_t row, uint32_t col) {
    assert(col < 8 && row < 6);
    return static_cast<npu_tile>((row << 4) | col);
}

// ---------------------------------------------------------------------------
// Instruction opcode headers
// ---------------------------------------------------------------------------

enum : uint32_t {
    XAIE_IO_WRITE        = 0x00,  // Register write
    XAIE_IO_BLOCKWRITE   = 0x01,  // DMA buffer descriptor write
    XAIE_IO_BLOCKSET     = 0x02,  // Block set
    XAIE_IO_MASKWRITE    = 0x03,  // Masked write (issue token)
    XAIE_IO_MASKPOLL     = 0x04,  // Mask poll
    XAIE_IO_NOOP         = 0x05,
    XAIE_IO_PREEMPT      = 0x06,
    XAIE_IO_MASKPOLL_BUSY = 0x07,
    XAIE_IO_LOADPDI      = 0x08,
    XAIE_IO_LOAD_PM_START = 0x09,
    XAIE_IO_CREATE_SCRATCHPAD   = 0x0A,
    XAIE_IO_UPDATE_STATE_TABLE  = 0x0B,
    XAIE_IO_UPDATE_REG   = 0x0C,
    XAIE_IO_UPDATE_SCRATCH = 0x0D,
    XAIE_CONFIG_SHIMDMA_BD         = 0x0E,
    XAIE_CONFIG_SHIMDMA_DMABUF_BD  = 0x0F,
    XAIE_IO_CUSTOM_OP_TCT      = 0x80,  // Token / wait sync
    XAIE_IO_CUSTOM_OP_DDR_PATCH = 0x81, // DDR address patching
    XAIE_IO_CUSTOM_OP_READ_REGS = 0x82,
    XAIE_IO_CUSTOM_OP_RECORD_TIMER = 0x83,
    XAIE_IO_CUSTOM_OP_MERGE_SYNC  = 0x84,
    XAIE_IO_CUSTOM_OP_NEXT     = 0x85,
};

// ---------------------------------------------------------------------------
// DMA direction
// ---------------------------------------------------------------------------

enum class dma_dir : uint32_t {
    S2MM = 0,  // Stream-to-Memory (read from tile)
    MM2S = 1,  // Memory-to-Stream (write to tile)
};

// ---------------------------------------------------------------------------
// Cache flags
// ---------------------------------------------------------------------------

enum cache_flag : uint32_t {
    CACHE_NONE      = 0x00,
    CACHE_NORMAL    = 0x02,
    CACHE_AGGRESSIVE = 0x0E,
};

// ---------------------------------------------------------------------------
// Base command — virtual interface for all NPU instruction commands
// ---------------------------------------------------------------------------

struct npu_cmd {
    virtual ~npu_cmd() = default;
    /// Serialize this command into the uint32_t[] sequence
    virtual void to_npu(std::vector<uint32_t>& seq) = 0;
    /// Number of uint32_t words this command occupies
    virtual int word_count() const = 0;
};

// ---------------------------------------------------------------------------
// npu_dma_block_cmd — DMA buffer descriptor write (XAIE_IO_BLOCKWRITE)
//
/// Writes a buffer descriptor (BD) into a tile's DMA controller. The BD describes
/// a data movement: where in DDR to read/write, the dimensions/strides, packet
/// control, locking, and chaining to the next BD.
//
// BD word layout (12 words total):
//   0: XAIE_IO_BLOCKWRITE
//   1: 0 (unused/reserved)
//   2: (row<<20) | (col<<25) | (bd_id<<5) | 0x1D000  (BD register address)
//   3: op_size * 4  (48 for full BD, bytes)
//   4: buffer_length  (bytes per transfer)
//   5: buffer_offset  (byte offset within BO)
//   6: (packet_enable<<30) | (out_of_order_id<<24) | (packet_id<<19) | (packet_type<<16)
//   7: (dim0_size<<20) | ((dim0_stride-1)<<0)  — D0 dimension
//   8: 0xC0000000 | (dim1_size<<20) | ((dim1_stride-1)<<0)  — D1 dimension + burst
//   9: (cache_flag<<24) | ((dim2_stride-1)<<0)  — D2 stride + cache
//  10: ((iter_size-1)<<20) | ((iter_stride-1)<<0)  — iteration dimension
//  11: (next_bd_id<<27) | (valid_bd<<25) | lock fields
// ---------------------------------------------------------------------------

struct npu_dma_block_cmd : public npu_cmd {
    // Location
    uint32_t col = 0;     // tile column (0-7)
    uint32_t row = 0;     // tile row (0=shim, 1=mem, 2-5=compute)
    uint32_t bd_id = 0;   // buffer descriptor ID (0-15 per tile)

    // Buffer
    uint32_t buffer_length = 0;  // bytes
    uint32_t buffer_offset = 0;  // byte offset from BO start

    // Packet control
    uint32_t packet_enable = 0;
    uint32_t out_of_order_id = 0;
    uint32_t packet_id = 0;
    uint32_t packet_type = 0;

    // Dimensions (0=linear if dim0_size==0)
    bool     is_linear = true;
    uint32_t dim0_size = 0;
    uint32_t dim0_stride = 0;
    uint32_t dim1_size = 0;
    uint32_t dim1_stride = 0;
    uint32_t dim2_size = 0;
    uint32_t dim2_stride = 0;
    uint32_t burst_size = 0xC0000000 >> 30;  // constant = 3

    // Iteration (repeat BD for multi-dimensional access)
    uint32_t iter_size = 0;    // number of iterations (0 = no iteration)
    uint32_t iter_stride = 0;  // stride between iterations (bytes)

    // BD chaining
    uint32_t next_bd_id = 0;
    uint32_t use_next_bd = 0;
    uint32_t valid_bd = 1;    // 1 = this BD is valid

    // Lock acquire/release (for tile-to-tile sync, not used on NPU2)
    uint32_t get_lock_rel_val = 0;
    uint32_t get_lock_rel_id = 0;
    uint32_t get_lock_acq_enable = 0;
    uint32_t get_lock_acq_val = 0;
    uint32_t get_lock_acq_id = 0;

    cache_flag cache = CACHE_NORMAL;

    static constexpr int OP_WORDS = 12;

    void to_npu(std::vector<uint32_t>& seq) override {
        seq.push_back(XAIE_IO_BLOCKWRITE);
        seq.push_back(0x0);
        seq.push_back((row << 20) | (col << 25) | (bd_id << 5) | 0x1D000);
        seq.push_back(OP_WORDS * 4);
        seq.push_back(buffer_length);
        seq.push_back(buffer_offset);
        seq.push_back(
            (packet_enable  << 30) |
            (out_of_order_id << 24) |
            (packet_id     << 19) |
            (packet_type   << 16));
        seq.push_back(
            (dim0_size << 20) |
            ((dim0_stride - 1) << 0));
        seq.push_back(
            0xC0000000 |
            (dim1_size << 20) |
            ((dim1_stride - 1) << 0));
        seq.push_back(
            (cache << 24) |
            ((dim2_stride - 1) << 0));
        seq.push_back(
            ((iter_size  - 1) << 20) |
            ((iter_stride - 1) << 0));
        seq.push_back(
            (next_bd_id       << 27) |
            (valid_bd         << 25) |
            (get_lock_rel_val << 18) |
            (get_lock_rel_id  << 13) |
            (get_lock_acq_enable << 12) |
            (get_lock_acq_val << 5) |
            (get_lock_acq_id  << 0));
    }

    int word_count() const override { return OP_WORDS; }
};

// ---------------------------------------------------------------------------
// npu_write_cmd — Register write (XAIE_IO_WRITE)
//
/// Writes a value to a tile's register. Used for:
///   - RTP (run-time parameter) writes: setting dynamic params on compute tiles
///   - Push to BD queue: triggering a BD on a shim DMA channel
//
// Word layout (6 words):
//   0: XAIE_IO_WRITE
//   1: 0
//   2: (row<<20) | (col<<25) | reg_addr
//   3: 0
//   4: value
//   5: op_size * 4
//
// For push-to-queue: reg_addr = 0x1D204 (or +8 for channel 1)
//                    value = (bd_id) | (repeat_count << 16) | (issue_token << 31)
// ---------------------------------------------------------------------------

struct npu_write_cmd : public npu_cmd {
    npu_tile tile = npu_tile::ST0;
    uint32_t reg_addr = 0;
    uint32_t value = 0;

    // Queue push fields (for push-to-queue mode)
    bool     is_queue_push = false;
    dma_dir  channel_dir = dma_dir::S2MM;
    uint32_t channel_id = 0;
    uint32_t repeat_count = 0;
    bool     issue_token = false;
    uint32_t bd_id = 0;

    static constexpr int OP_WORDS = 6;

    /// Create a push-to-queue command (triggers a BD on a shim DMA channel)
    static npu_write_cmd push_queue(npu_tile t, dma_dir dir, uint32_t ch,
                                     uint32_t bid, uint32_t rep = 1, bool token = false) {
        npu_write_cmd cmd;
        cmd.tile = t;
        cmd.is_queue_push = true;
        cmd.channel_dir = dir;
        cmd.channel_id = ch;
        cmd.bd_id = bid;
        cmd.repeat_count = rep;
        cmd.issue_token = token;
        uint32_t row = static_cast<uint32_t>(t) >> 4;
        uint32_t col = static_cast<uint32_t>(t) & 0xF;
        cmd.reg_addr = 0x1D204;
        if (dir == dma_dir::MM2S) cmd.reg_addr |= 0x10;
        if (ch == 1) cmd.reg_addr += 8;
        cmd.value = (bid & 0xF) | ((rep & 0xFF) << 16) | ((token ? 1u : 0u) << 31);
        return cmd;
    }

    /// Create an RTP write (sets a run-time parameter on a compute tile)
    static npu_write_cmd rtp_write(npu_tile t, uint32_t addr, uint32_t val) {
        npu_write_cmd cmd;
        cmd.tile = t;
        cmd.reg_addr = addr;
        cmd.value = val;
        cmd.is_queue_push = false;
        return cmd;
    }

    void to_npu(std::vector<uint32_t>& seq) override {
        uint32_t row = static_cast<uint32_t>(tile) >> 4;
        uint32_t col = static_cast<uint32_t>(tile) & 0xF;
        seq.push_back(XAIE_IO_WRITE);
        seq.push_back(0x0);
        seq.push_back((row << 20) | (col << 25) | reg_addr);
        seq.push_back(0x0);
        seq.push_back(value);
        seq.push_back(OP_WORDS * 4);
    }

    int word_count() const override { return OP_WORDS; }
};

// ---------------------------------------------------------------------------
// npu_ddr_cmd — DDR address patch (XAIE_IO_CUSTOM_OP_DDR_PATCH)
//
/// Patches a BO's physical address into a BD's address field at runtime.
/// The BD register address is computed from the tile location and BD ID.
/// The argument index refers to which kernel arg (BO) the address comes from,
/// and the argument offset is added to the address.
//
// Word layout (12 words):
//   0: XAIE_IO_CUSTOM_OP_DDR_PATCH
//   1: op_size * 4
//   2-5: 0
//   6: (col<<25) | (row<<20) | (bd_id<<5) | 0x1D004  (BD address register)
//   7: 0
//   8: arg_idx  (which kernel argument/BO provides the address)
//   9: 0
//  10: arg_offset (byte offset added to BO base address)
//  11: 0
// ---------------------------------------------------------------------------

struct npu_ddr_cmd : public npu_cmd {
    uint32_t col = 0;
    uint32_t row = 0;
    uint32_t bd_id = 0;
    uint32_t arg_idx = 0;     // kernel argument index (3 = first BO arg)
    uint32_t arg_offset = 0;  // byte offset within the BO

    static constexpr int OP_WORDS = 12;

    void to_npu(std::vector<uint32_t>& seq) override {
        seq.push_back(XAIE_IO_CUSTOM_OP_DDR_PATCH);
        seq.push_back(OP_WORDS * 4);
        seq.push_back(0x0);
        seq.push_back(0x0);
        seq.push_back(0x0);
        seq.push_back(0x0);
        seq.push_back(
            (col << 25) |
            (row << 20) |
            (bd_id << 5) |
            0x1D004);
        seq.push_back(0x0);
        seq.push_back(arg_idx);
        seq.push_back(0x0);
        seq.push_back(arg_offset);
        seq.push_back(0x0);
    }

    int word_count() const override { return OP_WORDS; }
};

// ---------------------------------------------------------------------------
// npu_issue_token_cmd — Issue DMA completion token (XAIE_IO_MASKWRITE)
//
/// Issues a token on a shim DMA channel's controller packet queue. This signals
/// that a DMA transfer is complete and allows dependent operations to proceed.
//
// Word layout (7 words):
//   0: XAIE_IO_MASKWRITE
//   1: 0
//   2: 0x1D200 | (ch_id*8) | (dir*0x10) | (row<<20) | (col<<25)
//   3: 0
//   4: controller_packet_id << 8
//   5: mask (0x00001f00)
//   6: op_size * 4
// ---------------------------------------------------------------------------

struct npu_issue_token_cmd : public npu_cmd {
    npu_tile tile = npu_tile::ST0;
    dma_dir  channel_dir = dma_dir::S2MM;
    uint32_t channel_id = 0;
    uint32_t controller_packet_id = 0;

    static constexpr int OP_WORDS = 7;

    void to_npu(std::vector<uint32_t>& seq) override {
        uint32_t row = static_cast<uint32_t>(tile) >> 4;
        uint32_t col = static_cast<uint32_t>(tile) & 0xF;
        seq.push_back(XAIE_IO_MASKWRITE);
        seq.push_back(0x0);
        seq.push_back(
            0x1D200 |
            (channel_id * 8) |
            (static_cast<uint32_t>(channel_dir) * 0x10) |
            (row << 20) |
            (col << 25));
        seq.push_back(0x0);
        seq.push_back(controller_packet_id << 8);
        seq.push_back(0x00001f00);
        seq.push_back(OP_WORDS * 4);
    }

    int word_count() const override { return OP_WORDS; }
};

// ---------------------------------------------------------------------------
// npu_wait_cmd — Wait for DMA completion (XAIE_IO_CUSTOM_OP_TCT)
//
/// Blocks until a specific DMA channel completes. The token/transaction control
/// (TCT) instruction waits for a sync signal from the given tile/channel.
//
// Word layout (4 words):
//   0: XAIE_IO_CUSTOM_OP_TCT
//   1: op_size * 4
//   2: (wait_row<<8) | (wait_col<<16) | direction
//   3: (wait_channel<<24) | 0x10100
// ---------------------------------------------------------------------------

struct npu_wait_cmd : public npu_cmd {
    uint32_t wait_row = 0;
    uint32_t wait_col = 0;
    uint32_t wait_channel = 0;
    dma_dir  channel_dir = dma_dir::S2MM;

    static constexpr int OP_WORDS = 4;

    void to_npu(std::vector<uint32_t>& seq) override {
        seq.push_back(XAIE_IO_CUSTOM_OP_TCT);
        seq.push_back(OP_WORDS * 4);
        seq.push_back(
            (wait_row << 8) |
            (wait_col << 16) |
            (static_cast<uint32_t>(channel_dir) << 0));
        seq.push_back(
            (wait_channel << 24) |
            0x10100);
    }

    int word_count() const override { return OP_WORDS; }
};

// ---------------------------------------------------------------------------
// npu_sequence — holds commands and compiles to instruction stream
// ---------------------------------------------------------------------------

class npu_sequence {
public:
    npu_sequence(npu_device dev = npu_device::npu2, bool enable_preemption = false)
        : device_(dev), enable_preemption_(enable_preemption)
    {
        setup_device(dev);
        clear();
    }

    /// Clear all commands
    void clear() {
        cmds_.clear();
        seq_.clear();
        is_valid_ = false;
        version_++;
    }

    /// Add a command (takes ownership)
    template<typename T, typename... Args>
    T& add_cmd(Args&&... args) {
        auto ptr = std::make_unique<T>(std::forward<Args>(args)...);
        T& ref = *ptr;
        cmds_.push_back(std::move(ptr));
        is_valid_ = false;
        return ref;
    }

    /// Add a pre-constructed command
    void add_cmd(std::unique_ptr<npu_cmd> cmd) {
        cmds_.push_back(std::move(cmd));
        is_valid_ = false;
    }

    /// Compile commands to instruction stream
    void cmds2seq() {
        seq_.clear();
        seq_.reserve(4 + cmds_.size() * 12);  // worst case per cmd

        // Header
        seq_.push_back(
            (npu_major_ << 24) |
            (npu_minor_ << 16) |
            (dev_gen_   << 12) |
            (npu_rows_  << 0));
        seq_.push_back(
            (npu_cols_        << 0) |
            (npu_mem_tile_rows_ << 16));
        seq_.push_back(static_cast<uint32_t>(cmds_.size()));  // instruction count
        seq_.push_back(0);  // placeholder for instruction lines (bytes), filled below

        // Serialize each command
        for (auto& cmd : cmds_) {
            cmd->to_npu(seq_);
        }

        // Fix up instruction lines field
        seq_[3] = static_cast<uint32_t>(seq_.size() * 4);  // total bytes

        is_valid_ = true;
        version_++;
    }

    /// Parse a pre-existing binary instruction sequence back to commands
    void seq2cmds(const uint32_t* data, size_t word_count) {
        clear();
        seq_.assign(data, data + word_count);

        // Parse header
        npu_major_ = (data[0] >> 24) & 0xFF;
        npu_minor_ = (data[0] >> 16) & 0xFF;
        dev_gen_   = (data[0] >> 12) & 0xF;
        npu_rows_  = (data[0] >> 0)  & 0xFFF;
        npu_cols_  = (data[1] >> 0)  & 0xFFFF;
        npu_mem_tile_rows_ = (data[1] >> 16) & 0xFF;

        // Walk instructions starting at word 4
        size_t i = 4;
        while (i < word_count) {
            uint32_t op = data[i];
            if (op == XAIE_IO_BLOCKWRITE) {
                auto cmd = std::make_unique<npu_dma_block_cmd>();
                // Blockwrite words: 0(BLOCKWRITE),1(0),2(reg),3(op_size),4(len),5(off)
                //                    6(pkt),7(dim0),8(dim1),9(dim2+cache),10(iter),11(next)
                cmd->row  = (data[i+2] >> 20) & 0x1F;
                cmd->col  = (data[i+2] >> 25) & 0x7F;
                cmd->bd_id = (data[i+2] >> 5) & 0xF;
                cmd->buffer_length  = data[i+4];
                cmd->buffer_offset  = data[i+5];
                cmd->packet_enable  = (data[i+6] >> 30) & 1;
                cmd->out_of_order_id = (data[i+6] >> 24) & 0x3F;
                cmd->packet_id      = (data[i+6] >> 19) & 0x1F;
                cmd->packet_type    = (data[i+6] >> 16) & 7;
                cmd->dim0_size   = (data[i+7] >> 20) & 0x3FF;
                cmd->dim0_stride = ((data[i+7] >> 0) & 0xFFFFF) + 1;
                cmd->dim1_size   = (data[i+8] >> 20) & 0x3FF;
                cmd->dim1_stride = ((data[i+8] >> 0) & 0xFFFFF) + 1;
                cmd->dim2_stride = ((data[i+9] >> 0) & 0xFFFFF) + 1;
                cmd->cache       = static_cast<cache_flag>((data[i+9] >> 24) & 0xF);
                cmd->iter_size   = ((data[i+10] >> 20) & 0x3FF) + 1;
                cmd->iter_stride = ((data[i+10] >> 0) & 0xFFFFF) + 1;
                cmd->next_bd_id  = (data[i+11] >> 27) & 0xF;
                cmd->valid_bd    = (data[i+11] >> 25) & 1;
                i += cmd->word_count();
                cmds_.push_back(std::move(cmd));
            }
            else if (op == XAIE_IO_CUSTOM_OP_DDR_PATCH) {
                auto cmd = std::make_unique<npu_ddr_cmd>();
                cmd->col  = (data[i+6] >> 25) & 0x7F;
                cmd->row  = (data[i+6] >> 20) & 0x1F;
                cmd->bd_id = ((data[i+6] - 0x04) >> 5) & 0x1F;
                cmd->arg_idx    = data[i+8];
                cmd->arg_offset = data[i+10];
                i += cmd->word_count();
                cmds_.push_back(std::move(cmd));
            }
            else if (op == XAIE_IO_MASKWRITE) {
                auto cmd = std::make_unique<npu_issue_token_cmd>();
                uint32_t addr = data[i+2];
                cmd->channel_id = (addr >> 3) & 1;
                cmd->channel_dir = (addr & 0x10) ? dma_dir::MM2S : dma_dir::S2MM;
                cmd->controller_packet_id = data[i+4] >> 8;
                cmd->tile = make_tile(
                    (addr >> 20) & 0x1F,
                    (addr >> 25) & 0x7F);
                i += cmd->word_count();
                cmds_.push_back(std::move(cmd));
            }
            else if (op == XAIE_IO_WRITE) {
                auto cmd = std::make_unique<npu_write_cmd>();
                cmd->reg_addr = data[i+2] & 0xFFFFF;
                cmd->value = data[i+4];
                cmd->tile = make_tile(
                    (data[i+2] >> 20) & 0x1F,
                    (data[i+2] >> 25) & 0x7F);
                cmd->is_queue_push = ((cmd->reg_addr & 0x1FE00) == 0x1D200);
                i += cmd->word_count();
                cmds_.push_back(std::move(cmd));
            }
            else if (op == XAIE_IO_CUSTOM_OP_TCT) {
                auto cmd = std::make_unique<npu_wait_cmd>();
                cmd->wait_row    = (data[i+2] >> 8)  & 0xFF;
                cmd->wait_col    = (data[i+2] >> 16) & 0xFF;
                cmd->channel_dir = (data[i+2] & 1) ? dma_dir::MM2S : dma_dir::S2MM;
                cmd->wait_channel = (data[i+3] >> 24) & 0xFF;
                i += cmd->word_count();
                cmds_.push_back(std::move(cmd));
            }
            else {
                // Unknown opcode — skip past it
                i++;
            }
        }

        is_valid_ = true;
        version_++;
    }

    /// Get the compiled instruction stream
    const uint32_t* data() const { return seq_.data(); }
    size_t size() const { return seq_.size(); }
    size_t size_bytes() const { return seq_.size() * sizeof(uint32_t); }

    /// Dump to a vector (for ELF generation)
    std::pair<uint32_t*, size_t> dump() {
        return { seq_.data(), seq_.size() };
    }

    /// Write instructions to binary file
    void write_binary(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(seq_.data()), seq_.size() * sizeof(uint32_t));
    }

    /// Load instructions from binary file
    void read_binary(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        size_t bytes = f.tellg();
        f.seekg(0);
        std::vector<uint32_t> buf(bytes / sizeof(uint32_t));
        f.read(reinterpret_cast<char*>(buf.data()), bytes);
        seq2cmds(buf.data(), buf.size());
    }

    bool valid() const { return is_valid_; }
    uint32_t version() const { return version_; }
    size_t cmd_count() const { return cmds_.size(); }
    const std::vector<std::unique_ptr<npu_cmd>>& cmds() const { return cmds_; }

    // Device info
    uint32_t npu_rows() const { return npu_rows_; }
    uint32_t npu_cols() const { return npu_cols_; }

private:
    void setup_device(npu_device dev) {
        if (dev == npu_device::npu2) {
            npu_major_ = 0;
            npu_minor_ = 1;
            dev_gen_   = 4;     // XDNA 2
            npu_rows_  = 6;
            npu_cols_  = 8;
            npu_mem_tile_rows_ = 1;
        } else {
            npu_major_ = 0;
            npu_minor_ = 1;
            dev_gen_   = 1;     // XDNA 1
            npu_rows_  = 6;
            npu_cols_  = 4;
            npu_mem_tile_rows_ = 1;
        }
    }

    npu_device device_ = npu_device::npu2;
    bool enable_preemption_ = false;

    // Parsed header
    uint32_t npu_major_ = 0;
    uint32_t npu_minor_ = 1;
    uint32_t dev_gen_   = 4;
    uint32_t npu_rows_  = 6;
    uint32_t npu_cols_  = 8;
    uint32_t npu_mem_tile_rows_ = 1;

    std::vector<std::unique_ptr<npu_cmd>> cmds_;
    std::vector<uint32_t> seq_;
    bool is_valid_ = false;
    uint32_t version_ = 0;
};

// ---------------------------------------------------------------------------
// GEMM sequence generator — builds NPU instruction sequence for a matmul
//
/// Generates the DMA commands needed for activation upload, weight DMA (if needed),
/// compute kernel launch, and result download. The generated sequence is compatible
/// with xrt::ext::kernel execution.
//
// Tile assignment (8-column NPU, GEMM on 8 compute tiles):
//   Cols 0-7: compute tiles, each handling K/8 of the matrix
//   Each tile: computes C[m, n] += A[m, k_block] * B[k_block, n]
//   Requires: A broadcast (all tiles get same M rows), B distributed (each tile gets K/8)
//
// For pre-compiled xclbins (like the custom torch2aie ones), the BD chain is
// minimal: just patch BO addresses into the pre-compiled instructions.
//
// For dynamically generated instructions (FLM-style), we generate the full BD chain.
// ---------------------------------------------------------------------------

/// Configuration for a single GEMM operation
struct gemm_config {
    uint32_t M;        // rows of A / rows of C
    uint32_t K;        // columns of A / rows of B
    uint32_t N;        // columns of B / columns of C
    uint32_t tile_M;   // tile size in M dimension (typically 128)
    uint32_t tile_K;   // tile size in K dimension (typically 64)
    uint32_t tile_N;   // tile size in N dimension (typically 128)

    uint32_t num_cols_used;  // number of NPU columns used (1-8)

    // BO assignment for DDR_PATCH
    uint32_t arg_act;     // kernel arg index for activation BO (typically 3)
    uint32_t arg_weight;  // kernel arg index for weight BO   (typically 4)
    uint32_arg_out;      // kernel arg index for output BO   (typically 5)
};

/// Generate a GEMM instruction sequence for a pre-compiled xclbin.
/// This creates the DDR_PATCH commands needed to patch the BO addresses
/// into the pre-existing instruction stream, plus issue-token and wait
/// commands for synchronization.
///
/// For dynamically generated sequences (full FLM-style), use generate_gemm_seq_full().
inline void generate_gemm_seq_patches(
    npu_sequence& seq,
    const gemm_config& cfg,
    const uint32_t* base_instrs,   // pre-compiled instruction base
    size_t base_instr_words,
    const std::vector<std::pair<uint32_t, uint32_t>>& patch_locs
    // patch_locs: list of (instruction_word_index, arg_idx) for each DDR_PATCH needed
) {
    // Parse the base instructions to find BD locations and add DDR_PATCH commands
    for (auto& [word_idx, arg_idx] : patch_locs) {
        // The base instruction at word_idx is a BLOCKWRITE BD
        // We need to patch the BD's address field with the BO address
        uint32_t bd_reg = base_instrs[word_idx + 2];  // the register address word
        uint32_t col = (bd_reg >> 25) & 0x7F;
        uint32_t row = (bd_reg >> 20) & 0x1F;
        uint32_t bd_id = (bd_reg >> 5) & 0xF;

        auto& ddr = seq.add_cmd<npu_ddr_cmd>();
        ddr.col = col;
        ddr.row = row;
        ddr.bd_id = bd_id;
        ddr.arg_idx = arg_idx;
        ddr.arg_offset = 0;  // may be adjusted per-BD
    }
}

/// Generate a complete GEMM sequence with dynamic BD chain (FLM-compatible)
inline void generate_gemm_seq_full(
    npu_sequence& seq,
    const gemm_config& cfg,
    uint32_t act_bo_offset,    // byte offset into activation BO
    uint32_t weight_bo_offset, // byte offset into weight BO
    uint32_t out_bo_offset     // byte offset into output BO
) {
    const uint32_t ncols = cfg.num_cols_used;
    const uint32_t K_per_col = cfg.K / ncols;

    // BD chain: for each column, create a set of BDs:
    //   1. BD for activation DMA (MM2S) into tile SRAM
    //   2. BD(s) for weight DMA (MM2S) into tile SRAM
    //   3. BD for output DMA (S2MM) from tile SRAM
    //
    // Then issue tokens and wait for completion.

    // For simplicity, use 2 BDs per column:
    //   BD 0: activation + weight DMA (MM2S, chained)
    //   BD 1: output readback (S2MM)
    //
    // Weight DMA uses a separate BD on the compute tile's mem tile.

    for (uint32_t c = 0; c < ncols; c++) {
        uint32_t col = c;  // physical column

        // ---- BD 0: Activation DMA (shim tile, MM2S) ----
        // Reads activations from DDR BO, sends to compute tiles
        {
            auto& bd = seq.add_cmd<npu_dma_block_cmd>();
            bd.col = col;
            bd.row = 0;  // shim tile
            bd.bd_id = 0;
            bd.buffer_length = cfg.M * cfg.K * sizeof(int8_t);  // quantized activations
            bd.buffer_offset = act_bo_offset;
            bd.is_linear = true;
            bd.valid_bd = 1;
            bd.next_bd_id = 1;   // chain to BD 1
            bd.use_next_bd = 1;
            bd.dim0_size = 0;     // linear
            bd.cache = CACHE_NORMAL;
            bd.iter_size = 1;     // single shot
        }

        // ---- DDR_PATCH for BD 0 address ----
        {
            auto& ddr = seq.add_cmd<npu_ddr_cmd>();
            ddr.col = col;
            ddr.row = 0;
            ddr.bd_id = 0;
            ddr.arg_idx = cfg.arg_act;
            ddr.arg_offset = act_bo_offset;
        }

        // ---- BD 1: Weight DMA (mem tile, MM2S) ----
        // Each compute tile gets K/ncols columns of weights
        {
            auto& bd = seq.add_cmd<npu_dma_block_cmd>();
            bd.col = col;
            bd.row = 1;  // mem tile
            bd.bd_id = 1;
            bd.buffer_length = K_per_col * cfg.N * sizeof(int8_t);
            bd.buffer_offset = weight_bo_offset + c * K_per_col * cfg.N * sizeof(int8_t);
            bd.is_linear = true;
            bd.valid_bd = 1;
            bd.cache = CACHE_NORMAL;
        }

        // ---- DDR_PATCH for BD 1 address ----
        {
            auto& ddr = seq.add_cmd<npu_ddr_cmd>();
            ddr.col = col;
            ddr.row = 1;
            ddr.bd_id = 1;
            ddr.arg_idx = cfg.arg_weight;
            ddr.arg_offset = weight_bo_offset + c * K_per_col * cfg.N * sizeof(int8_t);
        }

        // ---- BD 2: Output readback (shim tile, S2MM) ----
        {
            auto& bd = seq.add_cmd<npu_dma_block_cmd>();
            bd.col = col;
            bd.row = 0;  // shim tile
            bd.bd_id = 2;
            bd.buffer_length = cfg.M * cfg.N * 2;  // BF16 output
            bd.buffer_offset = out_bo_offset;
            bd.is_linear = true;
            bd.valid_bd = 1;
            bd.cache = CACHE_NORMAL;
        }

        // ---- DDR_PATCH for BD 2 address ----
        {
            auto& ddr = seq.add_cmd<npu_ddr_cmd>();
            ddr.col = col;
            ddr.row = 0;
            ddr.bd_id = 2;
            ddr.arg_idx = cfg.arg_out;
            ddr.arg_offset = out_bo_offset;
        }

        // ---- Push BD 0 to DMA queue (triggers the chain) ----
        {
            auto& push = seq.add_cmd<npu_write_cmd>(
                npu_write_cmd::push_queue(
                    make_tile(0, col), dma_dir::MM2S, 0, 0, 1, true));
            (void)push;
        }
    }

    // ---- Wait for all DMA to complete ----
    // Wait on each shim tile's S2MM channel
    for (uint32_t c = 0; c < ncols; c++) {
        auto& wait = seq.add_cmd<npu_wait_cmd>();
        wait.wait_col = c;
        wait.wait_row = 0;
        wait.wait_channel = 1;  // S2MM channel on shim tile
        wait.channel_dir = dma_dir::S2MM;
    }

    // ---- Issue final token ----
    {
        auto& tok = seq.add_cmd<npu_issue_token_cmd>();
        tok.tile = make_tile(0, 0);
        tok.channel_dir = dma_dir::S2MM;
        tok.channel_id = 0;
        tok.controller_packet_id = 1;
    }
}

// ---------------------------------------------------------------------------
// Helper: generate layer sequence for Qwen3-0.6B
//
/// Generates the full instruction sequence for one transformer layer:
///   QKV projection → attention → O projection → Gate/Up → SiLU → Down
// ---------------------------------------------------------------------------

inline void generate_attn_seq(npu_sequence& seq, int layer, int pos) {
    // RTP: set RoPE position for attention
    for (int col = 0; col < 8; col++) {
        auto& rtp = seq.add_cmd<npu_write_cmd>(
            npu_write_cmd::rtp_write(make_tile(2, col), 0x1D004, pos));
        (void)rtp;
    }
}

#endif // __NPU_INSTR_HPP__
