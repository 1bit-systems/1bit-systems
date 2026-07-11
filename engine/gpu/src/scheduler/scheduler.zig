//! Continuous-batching scheduler groundwork for concurrent inference requests.
//! @section Scheduler
//! Today this module owns request slot accounting and state collection only.
//! The HTTP serving hot path still serializes generation behind
//! ServerState.generation_mutex; the batched prefill/decode dispatch loop is
//! not wired yet.
const std = @import("std");
const Request = @import("request.zig").Request;
const RequestState = @import("request.zig").RequestState;
const GenerationParams = @import("request.zig").GenerationParams;

const log = std.log.scoped(.scheduler);

/// Fixed-capacity pool of request slots used to track concurrent inference requests.
/// Each slot holds at most one active `Request`; slots are reused once released.
pub const Scheduler = struct {
    /// Active requests indexed by slot ID (prefilling or decoding).
    slots: []?Request,
    /// FIFO of admitted-but-waiting requests (state `.pending`, no slot yet).
    /// Front = index 0. Drained into free slots by `admitNext`.
    pending: std.ArrayList(Request),
    /// Backing storage for the slot-ID slices returned by `pendingPrefill` /
    /// `activeDecoding` (sized `max_parallel`). Each call overwrites it, so a
    /// returned slice is only valid until the next such call.
    scratch: []u32,
    /// Maximum number of concurrent requests.
    max_parallel: u32,
    /// Next request ID counter.
    next_id: u64,
    /// Allocator for owned resources.
    allocator: std.mem.Allocator,

    /// Initialize the scheduler with a fixed number of concurrent request slots.
    /// @param allocator Allocator for the slot array.
    /// @param max_parallel Maximum number of concurrent requests.
    /// @returns A Scheduler with all slots initially empty.
    pub fn init(allocator: std.mem.Allocator, max_parallel: u32) !Scheduler {
        const slots = try allocator.alloc(?Request, max_parallel);
        @memset(slots, null);
        const scratch = try allocator.alloc(u32, max_parallel);
        log.info("Scheduler ready: {d} slots", .{max_parallel});
        return .{
            .slots = slots,
            .pending = .{},
            .scratch = scratch,
            .max_parallel = max_parallel,
            .next_id = 1,
            .allocator = allocator,
        };
    }

    /// Enqueue a new request without assigning a slot (continuous-batching path).
    /// The request sits in `pending` (state `.pending`) until `admitNext` moves it
    /// into a free slot. Unlike `submit`, this never fails on a full slot array —
    /// arrivals queue and are admitted as slots free, which is what lets a running
    /// batch admit/evict sequences between decode steps.
    /// @returns The new request's unique id.
    pub fn enqueue(self: *Scheduler, prompt_tokens: []const u32, params: GenerationParams) !u64 {
        const id = self.next_id;
        self.next_id += 1;
        const req = Request.init(self.allocator, id, prompt_tokens, params);
        try self.pending.append(self.allocator, req);
        log.info("Request {d} enqueued ({d} prompt tokens, {d} waiting)", .{ id, prompt_tokens.len, self.pending.items.len });
        return id;
    }

    /// Admit the oldest pending request into the first free slot, if any.
    /// Moves it out of the `pending` queue, assigns `slot_id`, and transitions it
    /// to `.prefilling`. The caller then runs prefill for every slot reported by
    /// `pendingPrefill` and transitions those to `.decoding`.
    /// @returns The assigned slot index, or null if no pending request or no free slot.
    pub fn admitNext(self: *Scheduler) !?u32 {
        if (self.pending.items.len == 0) return null;
        for (self.slots, 0..) |*slot, i| {
            if (slot.* == null) {
                var req = self.pending.orderedRemove(0);
                req.slot_id = @intCast(i);
                try req.transition(.prefilling);
                slot.* = req;
                log.info("Request {d} admitted to slot {d}", .{ req.id, i });
                return @intCast(i);
            }
        }
        return null; // all slots busy — request stays queued
    }

    /// True if at least one slot is free.
    pub fn hasFreeSlot(self: *const Scheduler) bool {
        for (self.slots) |slot| {
            if (slot == null) return true;
        }
        return false;
    }

    /// True if there is no outstanding work: every slot empty and no waiters.
    pub fn isIdle(self: *const Scheduler) bool {
        return self.pending.items.len == 0 and self.activeCount() == 0;
    }

    /// Submit a new request and assign it to the first free slot.
    /// @param self Scheduler to submit to.
    /// @param prompt_tokens Tokenized prompt for the request.
    /// @param params Generation parameters (max_tokens, temperature, etc.).
    /// @returns The slot index that was assigned; pass this value to `release` when the request completes.
    /// @note Returns `error.AllSlotsBusy` if every slot is occupied.
    pub fn submit(self: *Scheduler, prompt_tokens: []const u32, params: GenerationParams) !u32 {
        // Find a free slot
        for (self.slots, 0..) |*slot, i| {
            if (slot.* == null) {
                const id = self.next_id;
                self.next_id += 1;
                var req = Request.init(self.allocator, id, prompt_tokens, params);
                req.slot_id = @intCast(i);
                slot.* = req;
                log.info("Request {d} assigned to slot {d} ({d} prompt tokens)", .{ id, i, prompt_tokens.len });
                return @intCast(i);
            }
        }
        return error.AllSlotsBusy;
    }

    /// Check if all slots are occupied.
    /// @param self Scheduler to query.
    /// @returns True if every slot holds an active request.
    pub fn isFull(self: *const Scheduler) bool {
        return self.activeCount() >= self.max_parallel;
    }

    /// Get the number of active (non-null) requests.
    /// @param self Scheduler to query.
    /// @returns Count of occupied slots.
    pub fn activeCount(self: *const Scheduler) u32 {
        var count: u32 = 0;
        for (self.slots) |slot| {
            if (slot != null) count += 1;
        }
        return count;
    }

    /// Transition a live slot through the request state machine.
    /// @param self Scheduler to query.
    /// @param slot_id Slot index to update.
    /// @param new_state Target request state.
    /// @returns error.InvalidSlot if the slot is out of range or empty.
    pub fn transition(self: *Scheduler, slot_id: u32, new_state: RequestState) !void {
        if (slot_id >= self.slots.len) return error.InvalidSlot;
        if (self.slots[slot_id]) |*req| {
            try req.transition(new_state);
            return;
        }
        return error.InvalidSlot;
    }

    /// Collect slot IDs whose request currently has `state`.
    /// @param self Scheduler to query.
    /// @param state Request state to match.
    /// @param out Caller-owned scratch buffer for slot IDs.
    /// @returns A slice of `out` containing the collected slot IDs.
    pub fn collectByState(self: *const Scheduler, state: RequestState, out: []u32) []u32 {
        var count: usize = 0;
        for (self.slots, 0..) |slot, i| {
            if (count == out.len) break;
            if (slot) |req| {
                if (req.state == state) {
                    out[count] = @intCast(i);
                    count += 1;
                }
            }
        }
        return out[0..count];
    }

    /// Slot IDs of requests in the `.prefilling` state (admitted, prompt not yet
    /// processed). The driver runs prefill for each, then transitions it to
    /// `.decoding`.
    /// @returns A slice into `self.scratch`, valid until the next pendingPrefill /
    ///   activeDecoding call.
    pub fn pendingPrefill(self: *Scheduler) []u32 {
        var n: usize = 0;
        for (self.slots, 0..) |slot, i| {
            if (slot) |req| {
                if (req.state == .prefilling) {
                    self.scratch[n] = @intCast(i);
                    n += 1;
                }
            }
        }
        return self.scratch[0..n];
    }

    /// Slot IDs of requests in the `.decoding` state (the running decode batch).
    /// The driver gathers (token, position, slot) per id and issues ONE batched
    /// decode step over them.
    /// @returns A slice into `self.scratch`, valid until the next pendingPrefill /
    ///   activeDecoding call.
    pub fn activeDecoding(self: *Scheduler) []u32 {
        var n: usize = 0;
        for (self.slots, 0..) |slot, i| {
            if (slot) |req| {
                if (req.state == .decoding) {
                    self.scratch[n] = @intCast(i);
                    n += 1;
                }
            }
        }
        return self.scratch[0..n];
    }

    /// Release a completed or cancelled request's slot, freeing its resources.
    /// @param self Scheduler to release from.
    /// @param slot_id Slot index to free (the value returned by `submit`).
    /// @note Silently does nothing if `slot_id` is out of range or the slot is already empty.
    pub fn release(self: *Scheduler, slot_id: u32) void {
        if (slot_id < self.slots.len) {
            if (self.slots[slot_id]) |*req| {
                req.deinit();
                self.slots[slot_id] = null;
                log.info("Released slot {d}", .{slot_id});
            }
        }
    }

    /// Tear down all active and pending requests and free owned buffers.
    /// @param self Scheduler to destroy.
    pub fn deinit(self: *Scheduler) void {
        for (self.slots) |*slot| {
            if (slot.*) |*req| req.deinit();
        }
        for (self.pending.items) |*req| req.deinit();
        self.pending.deinit(self.allocator);
        self.allocator.free(self.scratch);
        self.allocator.free(self.slots);
    }
};

test "Scheduler submit and release" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 4);
    defer sched.deinit();

    try std.testing.expectEqual(@as(u32, 0), sched.activeCount());

    const slot0 = try sched.submit(&.{ 1, 2, 3 }, .{});
    try std.testing.expectEqual(@as(u32, 0), slot0);
    try std.testing.expectEqual(@as(u32, 1), sched.activeCount());

    const slot1 = try sched.submit(&.{ 4, 5 }, .{});
    try std.testing.expectEqual(@as(u32, 1), slot1);
    try std.testing.expectEqual(@as(u32, 2), sched.activeCount());

    sched.release(0);
    try std.testing.expectEqual(@as(u32, 1), sched.activeCount());
}

test "Scheduler full" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 2);
    defer sched.deinit();

    _ = try sched.submit(&.{1}, .{});
    _ = try sched.submit(&.{2}, .{});
    try std.testing.expectError(error.AllSlotsBusy, sched.submit(&.{3}, .{}));
}

test "Scheduler isFull" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 2);
    defer sched.deinit();

    try std.testing.expect(!sched.isFull());
    _ = try sched.submit(&.{1}, .{});
    try std.testing.expect(!sched.isFull());
    _ = try sched.submit(&.{2}, .{});
    try std.testing.expect(sched.isFull());
    sched.release(0);
    try std.testing.expect(!sched.isFull());
}

test "Scheduler release and reuse slot" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 1);
    defer sched.deinit();

    const s1 = try sched.submit(&.{10}, .{});
    try std.testing.expectEqual(@as(u32, 0), s1);
    sched.release(s1);

    // Same slot should be reusable
    const s2 = try sched.submit(&.{20}, .{});
    try std.testing.expectEqual(@as(u32, 0), s2);
    sched.release(s2);
}

test "Scheduler continuous-batching admit and reuse" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 2); // 2 slots, 3 requests → forces reuse
    defer sched.deinit();

    _ = try sched.enqueue(&.{ 1, 2 }, .{ .max_tokens = 4 });
    _ = try sched.enqueue(&.{3}, .{ .max_tokens = 4 });
    _ = try sched.enqueue(&.{ 4, 5 }, .{ .max_tokens = 4 }); // waits — no free slot

    // Admit as many as fit: 2 fill the slots, the 3rd stays pending.
    try std.testing.expect((try sched.admitNext()) != null);
    try std.testing.expect((try sched.admitNext()) != null);
    try std.testing.expectEqual(@as(?u32, null), try sched.admitNext());
    try std.testing.expectEqual(@as(usize, 1), sched.pending.items.len);

    // Both admitted requests are prefilling, none decoding yet.
    try std.testing.expectEqual(@as(usize, 2), sched.pendingPrefill().len);
    try std.testing.expectEqual(@as(usize, 0), sched.activeDecoding().len);

    // Transition both to decoding (driver does this after prefill).
    for (sched.slots) |*s| {
        if (s.*) |*r| try r.transition(.decoding);
    }
    try std.testing.expectEqual(@as(usize, 0), sched.pendingPrefill().len);
    try std.testing.expectEqual(@as(usize, 2), sched.activeDecoding().len);

    // Evict slot 0 → the waiter must now admit into the freed slot.
    sched.release(0);
    try std.testing.expect(sched.hasFreeSlot());
    const reused = (try sched.admitNext()).?;
    try std.testing.expectEqual(@as(u32, 0), reused);
    try std.testing.expectEqual(@as(usize, 0), sched.pending.items.len);
    try std.testing.expect(!sched.isIdle());
}

test "Scheduler EOS-driven eviction frees a slot for a waiter (variable lengths)" {
    const allocator = std.testing.allocator;
    const EOS: u32 = 42;
    var sched = try Scheduler.init(allocator, 2); // 2 slots, 3 requests → reuse on eviction
    defer sched.deinit();

    _ = try sched.enqueue(&.{1}, .{ .max_tokens = 8 }); // will EOS early
    _ = try sched.enqueue(&.{2}, .{ .max_tokens = 8 }); // runs longer
    _ = try sched.enqueue(&.{3}, .{ .max_tokens = 8 }); // waits for a free slot

    // Admit the first two; the third stays pending (no free slot).
    const a = (try sched.admitNext()).?;
    const b = (try sched.admitNext()).?;
    try std.testing.expectEqual(@as(?u32, null), try sched.admitNext());
    for (sched.slots) |*s| {
        if (s.*) |*r| try r.transition(.decoding);
    }

    // Slot `a` emits EOS after 2 tokens → shouldStop → release → admit the waiter.
    try sched.slots[a].?.appendToken(100);
    try sched.slots[a].?.appendToken(EOS);
    try std.testing.expect(sched.slots[a].?.shouldStop(EOS));
    try std.testing.expect(!sched.slots[b].?.shouldStop(EOS)); // still decoding
    try std.testing.expectEqual(@as(usize, 2), sched.slots[a].?.generated_tokens.items.len);

    try sched.slots[a].?.transition(.completed);
    sched.release(a);
    const reused = (try sched.admitNext()).?;
    try std.testing.expectEqual(a, reused); // waiter takes the freed slot
    try std.testing.expectEqual(@as(usize, 0), sched.pending.items.len);
    // `b` keeps decoding alongside the freshly-admitted request → ragged batch.
    try std.testing.expectEqual(RequestState.decoding, sched.slots[b].?.state);
}

test "Scheduler concurrent enqueue under external mutex assigns unique slots" {
    // Effort 28 inc 3 (3a): the concurrent serving harness drives enqueue from N
    // producer threads guarded by one external mutex (the worker owns admit/decode).
    // Prove that pattern yields exactly N pending requests with unique, contiguous
    // ids and no lost/duplicated entries — i.e. enqueue is safe under that locking.
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 2);
    defer sched.deinit();

    const N: u32 = 6;
    const Ctx = struct {
        sched: *Scheduler,
        mutex: *std.Thread.Mutex,
        fn run(self: *@This()) void {
            self.mutex.lock();
            defer self.mutex.unlock();
            _ = self.sched.enqueue(&.{ 1, 2, 3 }, .{ .max_tokens = 4 }) catch {};
        }
    };
    var mutex = std.Thread.Mutex{};
    var ctx = Ctx{ .sched = &sched, .mutex = &mutex };

    var threads: [N]std.Thread = undefined;
    for (&threads) |*t| t.* = try std.Thread.spawn(.{}, Ctx.run, .{&ctx});
    for (&threads) |t| t.join();

    try std.testing.expectEqual(@as(usize, N), sched.pending.items.len);
    try std.testing.expectEqual(@as(u64, N + 1), sched.next_id);
    // Ids are exactly the set {1..N}, each once.
    var seen = [_]bool{false} ** (N + 1);
    for (sched.pending.items) |req| {
        try std.testing.expect(req.id >= 1 and req.id <= N);
        try std.testing.expect(!seen[req.id]);
        seen[req.id] = true;
    }
}

test "Scheduler request IDs increment" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 4);
    defer sched.deinit();

    _ = try sched.submit(&.{1}, .{});
    _ = try sched.submit(&.{2}, .{});

    // Request IDs should be 0 and 1 (or some incrementing sequence)
    // Check slots have different request objects
    try std.testing.expect(sched.slots[0] != null);
    try std.testing.expect(sched.slots[1] != null);
    try std.testing.expect(sched.slots[0].?.id != sched.slots[1].?.id);
}

test "Scheduler collects pending prefill and active decoding slots" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 4);
    defer sched.deinit();

    const prefill_slot = try sched.submit(&.{1}, .{});
    const decode_slot = try sched.submit(&.{2}, .{});
    const other_slot = try sched.submit(&.{3}, .{});

    try sched.transition(decode_slot, .prefilling);
    try sched.transition(decode_slot, .decoding);
    try sched.transition(other_slot, .cancelled);

    var scratch: [4]u32 = undefined;
    const pending = sched.collectByState(.pending, &scratch);
    try std.testing.expectEqual(@as(usize, 1), pending.len);
    try std.testing.expectEqual(prefill_slot, pending[0]);

    const decoding = sched.collectByState(.decoding, &scratch);
    try std.testing.expectEqual(@as(usize, 1), decoding.len);
    try std.testing.expectEqual(decode_slot, decoding[0]);
}

test "Scheduler state collection respects scratch capacity" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 3);
    defer sched.deinit();

    _ = try sched.submit(&.{1}, .{});
    _ = try sched.submit(&.{2}, .{});
    _ = try sched.submit(&.{3}, .{});

    var scratch: [2]u32 = undefined;
    const pending = sched.collectByState(.pending, &scratch);
    try std.testing.expectEqual(@as(usize, 2), pending.len);
    try std.testing.expectEqual(@as(u32, 0), pending[0]);
    try std.testing.expectEqual(@as(u32, 1), pending[1]);
}
