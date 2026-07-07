//! Bridge between the unified KvPagePool scheduler and NPU XRT buffer objects.
//! Translates logical KV page IDs (managed by the scheduler's KvPagePool) into
//! byte offsets within NPU XRT buffer objects for attention read/write.
//!
//! Layout within a single token slot:
//!   [K_head0 (hd*4), K_head1 (hd*4), ..., K_headN (hd*4),
//!    V_head0 (hd*4), V_head1 (hd*4), ..., V_headN (hd*4)]
const std = @import("std");
const xrt = @import("xrt.zig");

const log = std.log.scoped(.npu_page_table);

// ============================================================
// NpuPageTable
// ============================================================

pub const NpuPageTable = struct {
    kv_bo: xrt.XrtBuffer,
    mapped: []u8,
    bo_size: u64,
    page_size_tokens: u32,
    n_kv_heads: u32,
    n_layers: u32,
    head_dim: u32,
    total_pages: u32,

    /// Per-token byte size (K heads + V heads).
    fn bytesPerToken(self: *const NpuPageTable) u64 {
        return @as(u64, self.n_kv_heads) * self.head_dim * 4 * 2; // K + V
    }

    fn bytesPerPage(self: *const NpuPageTable) u64 {
        return @as(u64, self.page_size_tokens) * self.bytesPerToken();
    }

    fn bytesPerLayer(self: *const NpuPageTable) u64 {
        return @as(u64, self.total_pages) * self.bytesPerPage();
    }

    pub fn init(
        page_size_tokens: u32,
        n_kv_heads: u32,
        n_layers: u32,
        head_dim: u32,
    ) NpuPageTable {
        return .{
            .kv_bo = xrt.XrtBuffer.init(null),
            .mapped = &.{},
            .bo_size = 0,
            .page_size_tokens = page_size_tokens,
            .n_kv_heads = n_kv_heads,
            .n_layers = n_layers,
            .head_dim = head_dim,
            .total_pages = 0,
        };
    }

    pub fn allocKVBO(self: *NpuPageTable, device: *xrt.XrtDevice, total_pages: u32) !void {
        self.total_pages = total_pages;

        const total_bytes = self.bytesPerLayer() * self.n_layers;

        log.info("KV BO: {d} layers × {d} pages × {d} tokens × {d} heads × {d} dim × 4 × 2 (K+V) = {d} MB", .{
            self.n_layers,   total_pages,   self.page_size_tokens,
            self.n_kv_heads, self.head_dim, total_bytes / (1024 * 1024),
        });

        const bo = try device.allocBO(total_bytes, xrt.XRT_BO_FLAGS_HOST_ONLY, 3);
        self.kv_bo = xrt.XrtBuffer.init(bo);
        self.bo_size = total_bytes;

        self.mapped = try self.kv_bo.map(total_bytes);
        @memset(self.mapped, 0);
        try self.kv_bo.sync(xrt.XCL_BO_SYNC_BO_TO_DEVICE, 0, total_bytes);
    }

    /// Byte offset for a specific K or V head within the BO.
    fn offsetKV(self: *const NpuPageTable, layer: u32, page_id: u32, token_in_page: u32, head: u32, is_v: bool) u64 {
        const head_bytes: u64 = self.head_dim * 4;
        const k_heads_bytes: u64 = self.n_kv_heads * head_bytes; // all K heads
        const token_bytes = k_heads_bytes * 2; // K + V

        var off = self.bytesPerLayer() * layer;
        off += @as(u64, page_id) * self.bytesPerPage();
        off += @as(u64, token_in_page) * token_bytes;
        off += @as(u64, head) * head_bytes;
        if (is_v) off += k_heads_bytes;
        return off;
    }

    /// Write one token's K at [n_kv_heads × head_dim].
    pub fn writeK(self: *NpuPageTable, layer: u32, page_id: u32, token_in_page: u32, k_data: []const f32) void {
        const head_bytes: u64 = self.head_dim * 4;
        for (0..self.n_kv_heads) |kvh| {
            const off = self.offsetKV(layer, page_id, token_in_page, @intCast(kvh), false);
            const slice = self.mapped[off .. off + head_bytes];
            const dst = @as([*]f32, @ptrCast(@alignCast(slice.ptr)));
            @memcpy(dst[0..self.head_dim], k_data[kvh * self.head_dim .. (kvh + 1) * self.head_dim]);
        }
    }

    /// Write one token's V at [n_kv_heads × head_dim].
    pub fn writeV(self: *NpuPageTable, layer: u32, page_id: u32, token_in_page: u32, v_data: []const f32) void {
        const head_bytes: u64 = self.head_dim * 4;
        for (0..self.n_kv_heads) |kvh| {
            const off = self.offsetKV(layer, page_id, token_in_page, @intCast(kvh), true);
            const slice = self.mapped[off .. off + head_bytes];
            const dst = @as([*]f32, @ptrCast(@alignCast(slice.ptr)));
            @memcpy(dst[0..self.head_dim], v_data[kvh * self.head_dim .. (kvh + 1) * self.head_dim]);
        }
    }

    /// Convenience: write both K and V in one call.
    pub fn writeKV(self: *NpuPageTable, layer: u32, page_id: u32, token_in_page: u32, k_data: []const f32, v_data: []const f32) void {
        self.writeK(layer, page_id, token_in_page, k_data);
        self.writeV(layer, page_id, token_in_page, v_data);
    }

    /// Linearize all KV data from pages into flat arrays.
    /// k_out/v_out: [n_tokens × n_kv_heads × head_dim].
    pub fn readAllKV(
        self: *NpuPageTable,
        layer: u32,
        page_ids: []const u32,
        n_tokens: u32,
        k_out: []f32,
        v_out: []f32,
    ) void {
        var token_idx: u32 = 0;

        page_loop: for (page_ids) |page_id| {
            for (0..self.page_size_tokens) |t_in_page| {
                if (token_idx >= n_tokens) break :page_loop;

                for (0..self.n_kv_heads) |kvh| {
                    // Read K
                    const k_off = self.offsetKV(layer, page_id, @intCast(t_in_page), @intCast(kvh), false);
                    const k_src = @as([*]const f32, @ptrCast(@alignCast(&self.mapped[k_off])));
                    const k_dst = k_out[token_idx * self.n_kv_heads * self.head_dim + kvh * self.head_dim ..];
                    @memcpy(k_dst[0..self.head_dim], k_src[0..self.head_dim]);

                    // Read V
                    const v_off = self.offsetKV(layer, page_id, @intCast(t_in_page), @intCast(kvh), true);
                    const v_src = @as([*]const f32, @ptrCast(@alignCast(&self.mapped[v_off])));
                    const v_dst = v_out[token_idx * self.n_kv_heads * self.head_dim + kvh * self.head_dim ..];
                    @memcpy(v_dst[0..self.head_dim], v_src[0..self.head_dim]);
                }
                token_idx += 1;
            }
        }
    }

    /// Zero-fill all data for a given page (used after H2O eviction).
    pub fn zeroFillPage(self: *NpuPageTable, layer: u32, page_id: u32) void {
        const off = self.bytesPerLayer() * layer + @as(u64, page_id) * self.bytesPerPage();
        const end = off + self.bytesPerPage();
        if (end <= self.mapped.len) @memset(self.mapped[off..end], 0);
    }

    pub fn deinit(self: *NpuPageTable) void {
        self.kv_bo.free();
        self.mapped = &.{};
        self.bo_size = 0;
    }
};

// ============================================================
// PageMapping — tracks which pages a request owns
// ============================================================

pub const PageMapping = struct {
    page_ids: []u32,
    token_count: u32,
    capacity: u32,

    pub fn init(allocator: std.mem.Allocator, max_pages: u32) !PageMapping {
        const page_ids = try allocator.alloc(u32, max_pages);
        @memset(page_ids, 0);
        return .{
            .page_ids = page_ids,
            .token_count = 0,
            .capacity = max_pages,
        };
    }

    pub fn addPage(self: *PageMapping, page_id: u32) !u32 {
        if (self.pageCount() >= self.capacity) return error.PageMappingFull;
        const idx = self.pageCount();
        self.page_ids[idx] = page_id;
        return @intCast(idx);
    }

    pub fn pageCount(self: *const PageMapping) u32 {
        // pages needed for current tokens
        return @min((self.token_count + 15) / 16, self.capacity);
    }

    pub fn currentPage(self: *const PageMapping) u32 {
        const pc = self.pageCount();
        if (pc == 0) return 0;
        return self.page_ids[pc - 1];
    }

    pub fn currentTokenInPage(self: *const PageMapping) u32 {
        if (self.token_count == 0) return 0;
        return (self.token_count - 1) % 16;
    }

    pub fn recordToken(self: *PageMapping) void {
        self.token_count += 1;
    }

    pub fn deinit(self: *PageMapping, allocator: std.mem.Allocator) void {
        allocator.free(self.page_ids);
    }
};
