//! Continuous-batching scheduler with H2O KV-cache eviction support.
//! @section Scheduler
//! Owns request slot accounting, state collection, and KV page lifecycle.
//! The continuous-batching hot path: enqueue → admitNext → pendingPrefill →
//! prefill → transition to decoding → activeDecoding → decode step → release.
const std = @import("std");
pub const Request = @import("request.zig").Request;
pub const RequestState = @import("request.zig").RequestState;
pub const GenerationParams = @import("request.zig").GenerationParams;
pub const KvPagePool = @import("kv_cache.zig").KvPagePool;
pub const EvictionPolicy = @import("kv_cache.zig").EvictionPolicy;

const log = std.log.scoped(.scheduler);

/// Fixed-capacity pool of request slots for concurrent inference requests.
pub const Scheduler = struct {
    slots: []?Request,
    pending: std.ArrayList(Request),
    scratch: []u32,
    max_parallel: u32,
    next_id: u64,
    allocator: std.mem.Allocator,
    kv_page_pool: ?*KvPagePool,
    pages_per_request: u32,

    pub fn init(allocator: std.mem.Allocator, max_parallel: u32, kv_page_pool: ?*KvPagePool, pages_per_request: u32) !Scheduler {
        const slots = try allocator.alloc(?Request, max_parallel);
        @memset(slots, null);
        const scratch = try allocator.alloc(u32, max_parallel);
        log.info("Scheduler: {d} slots, KV pages: {d}", .{ max_parallel, if (kv_page_pool) |p| p.total_pages else 0 });
        return .{
            .slots = slots,
            .pending = std.ArrayList(Request).empty,
            .scratch = scratch,
            .max_parallel = max_parallel,
            .next_id = 1,
            .allocator = allocator,
            .kv_page_pool = kv_page_pool,
            .pages_per_request = pages_per_request,
        };
    }

    pub fn enqueue(self: *Scheduler, prompt_tokens: []const u32, params: GenerationParams) !u64 {
        const id = self.next_id;
        self.next_id += 1;
        const req = Request.init(self.allocator, id, prompt_tokens, params);
        try self.pending.append(self.allocator, req);
        return id;
    }

    pub fn admitNext(self: *Scheduler) !?u32 {
        if (self.pending.items.len == 0) return null;
        for (self.slots, 0..) |*slot, i| {
            if (slot.* == null) {
                var req = self.pending.orderedRemove(0);
                req.slot_id = @intCast(i);
                if (self.kv_page_pool) |pool| {
                    if (self.pages_per_request > 0) {
                        req.kv_page_ids = try pool.allocOrEvict(req.id, self.pages_per_request);
                    }
                }
                try req.transition(.prefilling);
                slot.* = req;
                return @intCast(i);
            }
        }
        return null;
    }

    pub fn hasFreeSlot(self: *const Scheduler) bool {
        for (self.slots) |s| if (s == null) return true;
        return false;
    }

    pub fn isIdle(self: *const Scheduler) bool {
        return self.pending.items.len == 0 and self.activeCount() == 0;
    }

    pub fn submit(self: *Scheduler, prompt_tokens: []const u32, params: GenerationParams) !u32 {
        for (self.slots, 0..) |*slot, i| {
            if (slot.* == null) {
                const id = self.next_id;
                self.next_id += 1;
                var req = Request.init(self.allocator, id, prompt_tokens, params);
                req.slot_id = @intCast(i);
                slot.* = req;
                return @intCast(i);
            }
        }
        return error.AllSlotsBusy;
    }

    pub fn isFull(self: *const Scheduler) bool { return self.activeCount() >= self.max_parallel; }

    pub fn activeCount(self: *const Scheduler) u32 {
        var c: u32 = 0;
        for (self.slots) |s| {
            if (s != null) c += 1;
        }
        return c;
    }

    pub fn transition(self: *Scheduler, slot_id: u32, new_state: RequestState) !void {
        if (slot_id >= self.slots.len or self.slots[slot_id] == null) return error.InvalidSlot;
        try self.slots[slot_id].?.transition(new_state);
    }

    pub fn collectByState(self: *const Scheduler, state: RequestState, out: []u32) []u32 {
        var n: usize = 0;
        for (self.slots, 0..) |slot, i| {
            if (n == out.len) break;
            if (slot) |req| if (req.state == state) { out[n] = @intCast(i); n += 1; };
        }
        return out[0..n];
    }

    pub fn pendingPrefill(self: *Scheduler) []u32 {
        var n: usize = 0;
        for (self.slots, 0..) |slot, i| {
            if (slot) |req| if (req.state == .prefilling) { self.scratch[n] = @intCast(i); n += 1; };
        }
        return self.scratch[0..n];
    }

    pub fn activeDecoding(self: *Scheduler) []u32 {
        var n: usize = 0;
        for (self.slots, 0..) |slot, i| {
            if (slot) |req| if (req.state == .decoding) { self.scratch[n] = @intCast(i); n += 1; };
        }
        return self.scratch[0..n];
    }

    pub fn release(self: *Scheduler, slot_id: u32) void {
        if (slot_id < self.slots.len) {
            if (self.slots[slot_id]) |*req| {
                if (self.kv_page_pool) |pool| {
                    if (req.kv_page_ids) |ids| {
                        pool.freePages(req.id);
                        self.allocator.free(ids);
                        req.kv_page_ids = null;
                    }
                }
                req.deinit();
                self.slots[slot_id] = null;
            }
        }
    }

    pub fn deinit(self: *Scheduler) void {
        for (self.slots) |*slot| if (slot.*) |*r| r.deinit();
        for (self.pending.items) |*r| r.deinit();
        self.pending.deinit(self.allocator);
        self.allocator.free(self.scratch);
        self.allocator.free(self.slots);
    }
};

test "Scheduler submit and release" {
    const a = std.testing.allocator;
    var s = try Scheduler.init(a, 4, null, 0);
    defer s.deinit();
    _ = try s.submit(&.{1}, .{});
    try std.testing.expectEqual(@as(u32, 1), s.activeCount());
    s.release(0);
    try std.testing.expectEqual(@as(u32, 0), s.activeCount());
}

test "Scheduler full" {
    const a = std.testing.allocator;
    var s = try Scheduler.init(a, 1, null, 0);
    defer s.deinit();
    _ = try s.submit(&.{1}, .{});
    try std.testing.expectError(error.AllSlotsBusy, s.submit(&.{2}, .{}));
}
