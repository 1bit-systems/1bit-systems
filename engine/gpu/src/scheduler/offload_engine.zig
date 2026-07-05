//! CPU memory offloading for KV cache pages.
//! @section Scheduler
//!
//! When GPU memory is under pressure and H2O eviction would lose too many
//! tokens, cold KV cache pages can be swapped to CPU RAM. The page table
//! for offloaded pages points to a dedicated "offload staging" page — the
//! GPU reads zeros from it (same effect as eviction but the data is
//! recoverable).
//!
//! On subsequent access, the page is faulted back to the GPU via an async
//! blit copy. This is transparent to the attention kernel: the page table
//! is updated atomically once the copy completes.
//!
//! ## Page State Machine
//!
//! ┌──────────┐     evict     ┌──────────┐     complete    ┌───────────┐
//! │ RESIDENT │──────────────→│OFFLOADING│───────────────→│ OFFLOADED │
//! └──────────┘               └──────────┘               └───────────┘
//!       ↑                                                    │
//!       │                   complete     ┌──────────┐        │ fault
//!       └───────────────────────────────│ RELOADING│←───────┘
//!                                       └──────────┘
//!
//! TRANSITIONING states (OFFLOADING, RELOADING) are non-resident and
//! non-faultable — the page table temporarily points to the zero page.
const std = @import("std");

const log = std.log.scoped(.offload);

/// Offload state for a single KV cache page.
pub const OffloadState = enum(u8) {
    /// Page is resident in GPU memory. Normal operation.
    resident = 0,
    /// Page is being copied from GPU to CPU. Not yet offloaded.
    offloading = 1,
    /// Page data is on CPU. GPU access goes to zero page.
    offloaded = 2,
    /// Page is being copied from CPU back to GPU. Not yet resident.
    reloading = 3,
};

/// A page-sized buffer on the CPU side for offloaded KV data.
pub const CpuPageBuffer = struct {
    /// Page ID this buffer holds (or null if free).
    page_id: ?u32,
    /// CPU-side allocation for the full page content.
    /// Size = page_size * kv_dim * 2 (K + V) in bytes.
    data: []u8,
};

/// Async offload engine managing GPU↔CPU page swaps.
///
/// Uses a ring of pre-allocated CPU buffers to avoid runtime allocation
/// during the hot path. Offload/load requests are posted to queues and
/// processed in FIFO order.
pub const OffloadEngine = struct {
    /// Pre-allocated CPU page buffers (pool of N).
    cpu_buffers: []CpuPageBuffer,
    /// Number of active offloads (pages currently resident on CPU).
    active_offloads: u32,
    /// Maximum simultaneous offloads (size of cpu_buffers).
    max_offloads: u32,
    /// Bytes per KV page (page_size * kv_dim * 2 * sizeof(element)).
    page_bytes: u32,
    /// Queue of pending offload requests: page IDs to copy GPU→CPU.
    offload_queue: std.ArrayList(u32),
    /// Queue of pending reload requests: page IDs to copy CPU→GPU.
    reload_queue: std.ArrayList(u32),
    /// Allocator.
    allocator: std.mem.Allocator,

    /// Initialize the offload engine with N CPU buffers.
    pub fn init(allocator: std.mem.Allocator, max_offloads: u32, page_bytes: u32) !OffloadEngine {
        const cpu_buffers = try allocator.alloc(CpuPageBuffer, max_offloads);
        for (0..max_offloads) |i| {
            cpu_buffers[i] = .{
                .page_id = null,
                .data = try allocator.alloc(u8, page_bytes),
            };
        }
        log.info("Offload engine: {d} CPU buffers × {d} bytes = {d} KB", .{
            max_offloads, page_bytes, max_offloads * page_bytes / 1024,
        });
        return OffloadEngine{
            .cpu_buffers = cpu_buffers,
            .active_offloads = 0,
            .max_offloads = max_offloads,
            .page_bytes = page_bytes,
            .offload_queue = std.ArrayList(u32).empty,
            .reload_queue = std.ArrayList(u32).empty,
            .allocator = allocator,
        };
    }

    /// Request offload for a page (GPU → CPU).
    /// The request is queued and processed when a CPU buffer is available.
    pub fn requestOffload(self: *OffloadEngine, page_id: u32) !void {
        try self.offload_queue.append(self.allocator, page_id);
    }

    /// Request reload for a page (CPU → GPU).
    pub fn requestReload(self: *OffloadEngine, page_id: u32) !void {
        try self.reload_queue.append(self.allocator, page_id);
    }

    /// Process the next pending offload. Assigns a CPU buffer and
    /// returns the (page_id, cpu_buffer) for the engine to issue
    /// a GPU blit copy. Returns null if nothing to offload.
    pub fn processNextOffload(self: *OffloadEngine) ?struct { page_id: u32, cpu_data: []u8 } {
        if (self.offload_queue.items.len == 0) return null;

        // Find a free CPU buffer.
        for (self.cpu_buffers) |*buf| {
            if (buf.page_id == null) {
                const page_id = self.offload_queue.orderedRemove(0);
                buf.page_id = page_id;
                self.active_offloads += 1;
                log.debug("Offload: page {d} → CPU buffer", .{page_id});
                return .{
                    .page_id = page_id,
                    .cpu_data = buf.data,
                };
            }
        }
        return null; // all buffers busy
    }

    /// Complete an offload previously initiated by processNextOffload.
    /// The GPU→CPU copy is assumed to have finished.
    pub fn completeOffload(self: *OffloadEngine, page_id: u32) void {
        _ = self;
        log.debug("Offload complete: page {d}", .{page_id});
    }

    /// Find the CPU buffer containing an offloaded page.
    /// Returns null if the page is not on CPU.
    pub fn findCpuBuffer(self: *const OffloadEngine, page_id: u32) ?[]const u8 {
        for (self.cpu_buffers) |buf| {
            if (buf.page_id == page_id) return buf.data;
        }
        return null;
    }

    /// Process next reload. Returns (page_id, cpu_data) for the engine
    /// to issue a CPU→GPU blit copy. The CPU buffer remains reserved
    /// until `completeReload` is called after the blit finishes.
    /// Returns null if nothing to reload.
    pub fn processNextReload(self: *OffloadEngine) ?struct { page_id: u32, cpu_data: []const u8 } {
        if (self.reload_queue.items.len == 0) return null;
        const page_id = self.reload_queue.orderedRemove(0);
        const data = self.findCpuBuffer(page_id) orelse return null;

        log.debug("Reload: page {d} → GPU (buffer reserved until completeReload)", .{page_id});
        return .{ .page_id = page_id, .cpu_data = data };
    }

    /// Complete a reload previously initiated by processNextReload.
    /// Marks the CPU buffer as free after the GPU blit finishes.
    pub fn completeReload(self: *OffloadEngine, page_id: u32) void {
        for (self.cpu_buffers) |*buf| {
            if (buf.page_id == page_id) {
                buf.page_id = null;
                self.active_offloads -= 1;
                log.debug("Reload complete: page {d}, CPU buffer freed", .{page_id});
                return;
            }
        }
    }

    /// Number of pending offload requests.
    pub fn pendingOffloads(self: *const OffloadEngine) u32 {
        return @intCast(self.offload_queue.items.len);
    }

    /// Number of pending reload requests.
    pub fn pendingReloads(self: *const OffloadEngine) u32 {
        return @intCast(self.reload_queue.items.len);
    }

    /// Release all resources.
    pub fn deinit(self: *OffloadEngine) void {
        for (self.cpu_buffers) |*buf| {
            self.allocator.free(buf.data);
        }
        self.allocator.free(self.cpu_buffers);
        self.offload_queue.deinit(self.allocator);
        self.reload_queue.deinit(self.allocator);
    }
};

// ── Tests ──────────────────────────────────────────────────────────────────

test "OffloadEngine init and deinit" {
    const allocator = std.testing.allocator;
    var eng = try OffloadEngine.init(allocator, 4, 4096);
    defer eng.deinit();

    try std.testing.expectEqual(@as(u32, 4), eng.max_offloads);
    try std.testing.expectEqual(@as(u32, 0), eng.active_offloads);
}

test "OffloadEngine offload and reload cycle" {
    const allocator = std.testing.allocator;
    var eng = try OffloadEngine.init(allocator, 2, 256);
    defer eng.deinit();

    // Request offload for pages 0 and 1.
    try eng.requestOffload(0);
    try eng.requestOffload(1);
    try std.testing.expectEqual(@as(u32, 2), eng.pendingOffloads());

    // Process first offload.
    const off1 = eng.processNextOffload().?;
    try std.testing.expectEqual(@as(u32, 0), off1.page_id);
    try std.testing.expectEqual(@as(usize, 256), off1.cpu_data.len);

    // Process second offload.
    const off2 = eng.processNextOffload().?;
    try std.testing.expectEqual(@as(u32, 1), off2.page_id);

    // Both CPU buffers active.
    try std.testing.expectEqual(@as(u32, 2), eng.active_offloads);

    // Complete offloads.
    eng.completeOffload(0);
    eng.completeOffload(1);

    // Verify CPU data is accessible.
    const buf0 = eng.findCpuBuffer(0);
    try std.testing.expect(buf0 != null);

    // Reload page 0 back.
    try eng.requestReload(0);
    const reload = eng.processNextReload().?;
    try std.testing.expectEqual(@as(u32, 0), reload.page_id);

    // Complete the reload (simulates blit finish).
    eng.completeReload(0);

    // Active offloads decreased after completion.
    try std.testing.expectEqual(@as(u32, 1), eng.active_offloads);
}

test "OffloadEngine buffer exhaustion" {
    const allocator = std.testing.allocator;
    var eng = try OffloadEngine.init(allocator, 1, 64);
    defer eng.deinit();

    try eng.requestOffload(0);
    _ = eng.processNextOffload();

    // Second offload should fail.
    try eng.requestOffload(1);
    try std.testing.expect(eng.processNextOffload() == null);
}

test "OffloadEngine processNext in FIFO order" {
    const allocator = std.testing.allocator;
    var eng = try OffloadEngine.init(allocator, 4, 64);
    defer eng.deinit();

    try eng.requestOffload(10);
    try eng.requestOffload(20);
    try eng.requestOffload(30);

    const r1 = eng.processNextOffload().?;
    try std.testing.expectEqual(@as(u32, 10), r1.page_id);
    const r2 = eng.processNextOffload().?;
    try std.testing.expectEqual(@as(u32, 20), r2.page_id);
    const r3 = eng.processNextOffload().?;
    try std.testing.expectEqual(@as(u32, 30), r3.page_id);
}
