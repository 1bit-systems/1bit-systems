//! PM4 packet builder shared by direct AMD ZINC_RT tiers.
//!
//! This is intentionally syntax-only: it does not know about model shapes or
//! IR op semantics. M1 lowering hands already-decided register writes and
//! dispatch dimensions to this builder, then T2/T1 copy the resulting dwords
//! into their user queue rings.
//! @section Inference Runtime
const std = @import("std");

/// Failure modes returned by `PacketBuilder` operations.
/// `OutOfSpace` means the caller-provided dword buffer cannot fit another
/// PM4 packet without overrunning its bounds.
pub const Error = error{OutOfSpace};

/// PM4 type-3 opcodes used by the direct ZINC_RT submission paths.
/// Each value matches the raw hardware opcode the command processor expects
/// in the PKT3 header field.
pub const Opcode = enum(u8) {
    nop = 0x10,
    dispatch_direct = 0x15,
    write_data = 0x37,
    wait_reg_mem = 0x3c,
    copy_data = 0x40,
    release_mem = 0x49,
    acquire_mem = 0x58,
    set_context_reg = 0x69,
    set_sh_reg = 0x76,
    set_uconfig_reg = 0x79,
};

// SH register offsets, expressed as `(byte_addr - 0xB000) >> 2`.
// The direct CS path programs these before a raw DISPATCH_DIRECT.
/// SH register offset for `COMPUTE_NUM_THREAD_X` (workgroup X dimension).
pub const sh_reg_num_thread_x: u32 = 0x207;
/// SH register offset for `COMPUTE_PGM_LO` (low 32 bits of the shader address).
pub const sh_reg_pgm_lo: u32 = 0x20c;
/// SH register offset for `COMPUTE_PGM_RSRC1` (VGPR/SGPR counts and float mode).
pub const sh_reg_pgm_rsrc1: u32 = 0x212;
/// SH register offset for `COMPUTE_RESOURCE_LIMITS` (waves-per-CU and locking).
pub const sh_reg_resource_limits: u32 = 0x215;
/// SH register offset for `COMPUTE_PGM_RSRC3` (extra GFX11+ shader resource bits).
pub const sh_reg_pgm_rsrc3: u32 = 0x228;
/// SH register offset for `COMPUTE_USER_DATA_0`; subsequent slots are
/// contiguous and used to pass kernel argument pointers.
pub const compute_user_data_0: u32 = 0x240;
/// Default `DISPATCH_INITIATOR` value enabling the compute pipeline.
pub const dispatch_initiator_compute: u32 = 5;

/// Cursor-style writer that emits PM4 type-3 packets into a caller-owned
/// dword buffer.
/// The builder is allocation-free and stateless beyond a write cursor, so
/// callers can reuse the same backing buffer across submissions by calling
/// `reset`.
pub const PacketBuilder = struct {
    words: []u32,
    len: usize = 0,

    /// Wrap a pre-allocated dword buffer.
    /// @param words Backing storage; the builder writes packets starting at
    ///     index 0 and never grows the slice.
    /// @returns A builder pointing at `words` with an empty write cursor.
    pub fn init(words: []u32) PacketBuilder {
        return .{ .words = words };
    }

    /// Rewind the write cursor without touching the backing buffer.
    /// @param self Builder to reset; subsequent writes overwrite previous dwords.
    pub fn reset(self: *PacketBuilder) void {
        self.len = 0;
    }

    /// Borrowed view of the dwords emitted so far.
    /// @param self Builder to inspect.
    /// @returns Slice of finalized packet words, ready to copy into a ring.
    pub fn written(self: *const PacketBuilder) []const u32 {
        return self.words[0..self.len];
    }

    /// Emit a PM4 `NOP` packet that consumes `payload_dwords` body dwords.
    /// @param self Builder to append to.
    /// @param payload_dwords Number of zero payload dwords; clamped to a
    ///     minimum of 1 to satisfy the PKT3 body-size encoding.
    pub fn writeNop(self: *PacketBuilder, payload_dwords: u32) Error!void {
        const body_dwords = @max(payload_dwords, 1);
        const start = try self.reservePacket(body_dwords);
        for (0..body_dwords) |i| self.words[start + 1 + i] = 0;
        self.publishPkt3Header(start, .nop, body_dwords);
    }

    /// Emit `SET_SH_REG` writing `values` into consecutive SH register slots.
    /// @param self Builder to append to.
    /// @param reg_offset Starting SH register offset (dword units from 0xB000).
    /// @param values Register values written in order; an empty slice is a no-op.
    pub fn setShReg(self: *PacketBuilder, reg_offset: u32, values: []const u32) Error!void {
        if (values.len == 0) return;
        const body_dwords: u32 = @intCast(values.len + 1);
        const start = try self.reservePacket(body_dwords);
        self.words[start + 1] = reg_offset;
        for (values, 0..) |value, i| self.words[start + 2 + i] = value;
        self.publishPkt3Header(start, .set_sh_reg, body_dwords);
    }

    /// Convenience helper that writes a single SH register.
    /// @param self Builder to append to.
    /// @param reg_offset SH register offset.
    /// @param value Value to write into that register.
    pub fn setShRegOne(self: *PacketBuilder, reg_offset: u32, value: u32) Error!void {
        const values = [_]u32{value};
        try self.setShReg(reg_offset, &values);
    }

    /// Write a 64-bit value into a pair of contiguous `COMPUTE_USER_DATA_*`
    /// slots, little-endian (low dword first).
    /// @param self Builder to append to.
    /// @param slot Zero-based index added to `compute_user_data_0`.
    /// @param value 64-bit kernel argument (typically a GPU virtual address).
    pub fn setUserData64(self: *PacketBuilder, slot: u32, value: u64) Error!void {
        const values = [_]u32{ lo32(value), hi32(value) };
        try self.setShReg(compute_user_data_0 + slot, &values);
    }

    /// Emit `DISPATCH_DIRECT` with a zero dispatch-initiator field.
    /// Prefer `dispatchDirectInitiator` when `COMPUTE_DISPATCH_INITIATOR` bits
    /// must be set explicitly (e.g. `dispatch_initiator_compute`).
    /// @param self Builder to append to.
    /// @param dim_x Workgroup count on X.
    /// @param dim_y Workgroup count on Y.
    /// @param dim_z Workgroup count on Z.
    pub fn dispatchDirect(self: *PacketBuilder, dim_x: u32, dim_y: u32, dim_z: u32) Error!void {
        try self.dispatchDirectInitiator(dim_x, dim_y, dim_z, 0);
    }

    /// Emit `DISPATCH_DIRECT` with a caller-supplied dispatch initiator value.
    /// @param self Builder to append to.
    /// @param dim_x Workgroup count on X.
    /// @param dim_y Workgroup count on Y.
    /// @param dim_z Workgroup count on Z.
    /// @param dispatch_initiator Raw `COMPUTE_DISPATCH_INITIATOR` bits; pass 0
    ///     to take the firmware default or use `dispatch_initiator_compute`
    ///     to force-enable the compute pipeline.
    pub fn dispatchDirectInitiator(
        self: *PacketBuilder,
        dim_x: u32,
        dim_y: u32,
        dim_z: u32,
        dispatch_initiator: u32,
    ) Error!void {
        const start = try self.reservePacket(4);
        self.words[start + 1] = dim_x;
        self.words[start + 2] = dim_y;
        self.words[start + 3] = dim_z;
        self.words[start + 4] = dispatch_initiator;
        self.publishPkt3Header(start, .dispatch_direct, 4);
    }

    /// Emit a GFX10+ `RELEASE_MEM` end-of-pipe fence that writes `value` to
    /// `gpu_addr` after prior shader work and global-memory writes complete.
    /// @param self Builder to append to.
    /// @param gpu_addr 64-bit GPU virtual address to receive the fence value.
    /// @param value Fence payload (typically a monotonically increasing seqno).
    /// @note This uses the GFX10+ release-mem packet layout used by amdgpu for
    ///     compute-ring fences; older ASICs are not a ZINC_RT direct target.
    pub fn releaseMemSignal(self: *PacketBuilder, gpu_addr: u64, value: u64) Error!void {
        const cache_flush_and_inv_ts_event: u32 = 0x14;
        const event_index_end_of_pipe: u32 = 5;
        const gcr_glm_wb: u32 = 1 << 12;
        const gcr_glm_inv: u32 = 1 << 13;
        const gcr_glv_inv: u32 = 1 << 14;
        const gcr_gl1_inv: u32 = 1 << 15;
        const gcr_gl2_inv: u32 = 1 << 20;
        const gcr_gl2_wb: u32 = 1 << 21;
        const gcr_seq: u32 = 1 << 22;
        const cache_policy_bypass: u32 = 3 << 25;
        const event_dw = cache_flush_and_inv_ts_event |
            (event_index_end_of_pipe << 8) |
            gcr_glm_wb |
            gcr_glm_inv |
            gcr_glv_inv |
            gcr_gl1_inv |
            gcr_gl2_inv |
            gcr_gl2_wb |
            gcr_seq |
            cache_policy_bypass;
        const data_sel_64: u32 = 2 << 29;
        const int_sel_none: u32 = 0 << 24;
        const dst_sel_mc: u32 = 0 << 16;
        const start = try self.reservePacket(7);
        self.words[start + 1] = event_dw;
        self.words[start + 2] = data_sel_64 | int_sel_none | dst_sel_mc;
        self.words[start + 3] = lo32(gpu_addr);
        self.words[start + 4] = hi32(gpu_addr);
        self.words[start + 5] = lo32(value);
        self.words[start + 6] = hi32(value);
        self.words[start + 7] = 0;
        self.publishPkt3Header(start, .release_mem, 7);
    }

    /// Emit `WRITE_DATA` that stores `value` (64 bits) at `gpu_addr` via the
    /// ME (micro-engine) with WR_CONFIRM set.
    /// @param self Builder to append to.
    /// @param gpu_addr Destination GPU virtual address.
    /// @param value 64-bit payload written little-endian.
    /// @note Uses `dst_sel=5` (memory async/direct) and `engine_sel=0` (ME).
    ///     The ME stalls until the write lands in memory (WR_CONFIRM=1), so
    ///     callers observe the value as soon as the packet retires.
    pub fn writeData64(self: *PacketBuilder, gpu_addr: u64, value: u64) Error!void {
        // PKT3_WRITE_DATA, dst_sel=5 (memory async/direct), WR_CONFIRM=1,
        // engine_sel=0 (ME). This is the simplest in-band memory scribble for
        // validating CS-submitted fence/output-ring writes before real kernels.
        const dst_sel_memory_async: u32 = 5 << 8;
        const wr_confirm: u32 = 1 << 20;
        const engine_sel_me: u32 = 0 << 30;
        const start = try self.reservePacket(5);
        self.words[start + 1] = dst_sel_memory_async | wr_confirm | engine_sel_me;
        self.words[start + 2] = lo32(gpu_addr);
        self.words[start + 3] = hi32(gpu_addr);
        self.words[start + 4] = lo32(value);
        self.words[start + 5] = hi32(value);
        self.publishPkt3Header(start, .write_data, 5);
    }

    /// Emit `COPY_DATA` that copies a single 32-bit dword from one GPU memory
    /// address to another with WR_CONFIRM set.
    /// @param self Builder to append to.
    /// @param src_gpu_addr Source GPU virtual address (memory, src_sel=1).
    /// @param dst_gpu_addr Destination GPU virtual address (memory, dst_sel=5).
    /// @note Copies exactly 32 bits (count_sel=0). WR_CONFIRM=1 means the
    ///     command processor stalls until the destination write completes.
    pub fn copyData32(self: *PacketBuilder, src_gpu_addr: u64, dst_gpu_addr: u64) Error!void {
        // PKT3_COPY_DATA, src_sel=1 (memory), dst_sel=5 (memory),
        // count_sel=0 (32 bits), WR_CONFIRM=1. This is the smallest
        // command-processor dataflow primitive available before shader
        // dispatch lowering is wired.
        const src_sel_memory: u32 = 1;
        const dst_sel_memory: u32 = 5 << 8;
        const count_sel_32: u32 = 0 << 16;
        const wr_confirm: u32 = 1 << 20;
        const start = try self.reservePacket(5);
        self.words[start + 1] = src_sel_memory | dst_sel_memory | count_sel_32 | wr_confirm;
        self.words[start + 2] = lo32(src_gpu_addr);
        self.words[start + 3] = hi32(src_gpu_addr);
        self.words[start + 4] = lo32(dst_gpu_addr);
        self.words[start + 5] = hi32(dst_gpu_addr);
        self.publishPkt3Header(start, .copy_data, 5);
    }

    /// Pad the buffer with `NOP` packets until the current write cursor is a
    /// multiple of `dword_alignment` dwords.
    /// @param self Builder to pad.
    /// @param dword_alignment Required alignment in dwords (0 is a no-op).
    /// @note Each emitted NOP carries enough payload to land on the alignment
    ///     in a single packet rather than spinning out many minimum-size NOPs.
    pub fn padToAlignment(self: *PacketBuilder, dword_alignment: usize) Error!void {
        if (dword_alignment == 0) return;
        while (self.len % dword_alignment != 0) {
            var packet_dwords: usize = 2;
            while ((self.len + packet_dwords) % dword_alignment != 0) : (packet_dwords += 1) {}
            try self.writeNop(@intCast(packet_dwords - 1));
        }
    }

    fn reservePacket(self: *PacketBuilder, body_dwords: u32) Error!usize {
        std.debug.assert(body_dwords > 0);
        const total = @as(usize, body_dwords) + 1;
        if (self.len + total > self.words.len) return error.OutOfSpace;
        const start = self.len;
        self.words[start] = 0;
        self.len += total;
        return start;
    }

    fn publishPkt3Header(self: *PacketBuilder, start: usize, opcode: Opcode, body_dwords: u32) void {
        const count = body_dwords - 1;
        const header = (@as(u32, 3) << 30) | ((count & 0x3fff) << 16) | (@as(u32, @intFromEnum(opcode)) << 8);
        const header_ptr: *volatile u32 = @ptrCast(&self.words[start]);
        header_ptr.* = header;
    }
};

/// Extract the low 32 bits of a 64-bit value for little-endian dword writes.
/// @param value 64-bit input.
/// @returns Bits [31:0] of `value`.
pub fn lo32(value: u64) u32 {
    return @truncate(value);
}

/// Extract the high 32 bits of a 64-bit value for little-endian dword writes.
/// @param value 64-bit input.
/// @returns Bits [63:32] of `value`.
pub fn hi32(value: u64) u32 {
    return @truncate(value >> 32);
}

test "packet builder emits PM4 type-3 dispatch packet" {
    var words = [_]u32{0} ** 8;
    var builder = PacketBuilder.init(&words);
    try builder.dispatchDirect(7, 2, 1);

    const out = builder.written();
    try std.testing.expectEqual(@as(usize, 5), out.len);
    try std.testing.expectEqual((@as(u32, 3) << 30) | (@as(u32, 3) << 16) | (@as(u32, 0x15) << 8), out[0]);
    try std.testing.expectEqual(@as(u32, 7), out[1]);
    try std.testing.expectEqual(@as(u32, 2), out[2]);
    try std.testing.expectEqual(@as(u32, 1), out[3]);
    try std.testing.expectEqual(@as(u32, 0), out[4]);
}

test "packet builder emits contiguous user data register writes" {
    var words = [_]u32{0} ** 8;
    var builder = PacketBuilder.init(&words);
    try builder.setUserData64(4, 0x11223344_aabbccdd);

    const out = builder.written();
    try std.testing.expectEqual(@as(usize, 4), out.len);
    try std.testing.expectEqual((@as(u32, 3) << 30) | (@as(u32, 2) << 16) | (@as(u32, 0x76) << 8), out[0]);
    try std.testing.expectEqual(compute_user_data_0 + 4, out[1]);
    try std.testing.expectEqual(@as(u32, 0xaabbccdd), out[2]);
    try std.testing.expectEqual(@as(u32, 0x11223344), out[3]);
}

test "packet builder emits write-data memory signal" {
    var words = [_]u32{0} ** 8;
    var builder = PacketBuilder.init(&words);
    try builder.writeData64(0x11223344_aabbccdd, 0x01020304_05060708);

    const out = builder.written();
    try std.testing.expectEqual(@as(usize, 6), out.len);
    try std.testing.expectEqual((@as(u32, 3) << 30) | (@as(u32, 4) << 16) | (@as(u32, 0x37) << 8), out[0]);
    try std.testing.expectEqual(@as(u32, (5 << 8) | (1 << 20)), out[1]);
    try std.testing.expectEqual(@as(u32, 0xaabbccdd), out[2]);
    try std.testing.expectEqual(@as(u32, 0x11223344), out[3]);
    try std.testing.expectEqual(@as(u32, 0x05060708), out[4]);
    try std.testing.expectEqual(@as(u32, 0x01020304), out[5]);
}

test "packet builder emits release-mem eop signal" {
    var words = [_]u32{0} ** 8;
    var builder = PacketBuilder.init(&words);
    try builder.releaseMemSignal(0x11223344_aabbccdd, 0x01020304_05060708);

    const out = builder.written();
    try std.testing.expectEqual(@as(usize, 8), out.len);
    try std.testing.expectEqual((@as(u32, 3) << 30) | (@as(u32, 6) << 16) | (@as(u32, 0x49) << 8), out[0]);
    try std.testing.expectEqual(@as(u32, 0x0670f514), out[1]);
    try std.testing.expectEqual(@as(u32, 2 << 29), out[2]);
    try std.testing.expectEqual(@as(u32, 0xaabbccdd), out[3]);
    try std.testing.expectEqual(@as(u32, 0x11223344), out[4]);
    try std.testing.expectEqual(@as(u32, 0x05060708), out[5]);
    try std.testing.expectEqual(@as(u32, 0x01020304), out[6]);
    try std.testing.expectEqual(@as(u32, 0), out[7]);
}

test "packet builder emits copy-data memory to memory dword" {
    var words = [_]u32{0} ** 8;
    var builder = PacketBuilder.init(&words);
    try builder.copyData32(0x1000_0040, 0x1000_0080);

    const out = builder.written();
    try std.testing.expectEqual(@as(usize, 6), out.len);
    try std.testing.expectEqual((@as(u32, 3) << 30) | (@as(u32, 4) << 16) | (@as(u32, 0x40) << 8), out[0]);
    try std.testing.expectEqual(@as(u32, 1 | (5 << 8) | (1 << 20)), out[1]);
    try std.testing.expectEqual(@as(u32, 0x1000_0040), out[2]);
    try std.testing.expectEqual(@as(u32, 0), out[3]);
    try std.testing.expectEqual(@as(u32, 0x1000_0080), out[4]);
    try std.testing.expectEqual(@as(u32, 0), out[5]);
}

test "packet builder reports fixed buffer exhaustion" {
    var words = [_]u32{0} ** 2;
    var builder = PacketBuilder.init(&words);
    try std.testing.expectError(error.OutOfSpace, builder.dispatchDirect(1, 1, 1));
}

test "packet builder pads to dword alignment with valid NOP packets" {
    var words = [_]u32{0} ** 16;
    var builder = PacketBuilder.init(&words);
    try builder.dispatchDirect(1, 1, 1);
    try builder.padToAlignment(4);
    try std.testing.expectEqual(@as(usize, 8), builder.written().len);
}
