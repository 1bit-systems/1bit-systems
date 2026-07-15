//! Low-overhead worker pool for the T-CPU decode matvec fan-out.
//! @section Inference Runtime
//!
//! Replaces std.Thread.Pool's spawnWg/waitAndWork pattern for the matvec hot
//! path. Each dispatch posts up to N typed tasks to per-worker atomic slots
//! and spins on a per-slot done-sequence — no heap allocation, no mutex, no
//! condvar. Workers are persistent and spin (with brief yields) between
//! dispatches; total CPU footprint is bounded by `n_workers` cores.
//!
//! Why bother: the Qwen3.6-MoE+SSM decode dispatches ~200 matvec/fan-out
//! barriers per token through std.Thread.Pool. Each spawnWg does a heap
//! allocation on the global smp_allocator (closure container), takes the
//! pool mutex twice, and signals a condvar; each waitAndWork takes the
//! mutex and waits on a ResetEvent futex. The atomic-only path here is
//! orders of magnitude cheaper per barrier (~tens of ns vs ~µs).
//!
//! API: build a small `Task` array, call `dispatchAndRun(&tasks)`. The
//! caller runs `tasks[0]` on the main thread; `tasks[1..n]` are posted to
//! workers 0..n-1. Returns when all tasks have completed.

const std = @import("std");

/// Upper bound on worker threads supported by a single `FastPool`.
/// The slot table is sized to this constant so dispatches stay branch-free
/// and cache-friendly; this matches the decode matvec scheduler's current
/// maximum so high-core hosts do not silently fall back to std.Thread.Pool.
pub const max_workers: usize = 16;

const Slot = struct {
    /// Worker's task fn. Set by the dispatcher before bumping `seq`.
    run_fn: std.atomic.Value(?*const fn (*anyopaque) void) align(64),
    /// Opaque context pointer for the worker. Stored atomically with
    /// release ordering; the publish barrier is the subsequent `seq.fetchAdd`.
    ctx: std.atomic.Value(?*anyopaque),
    /// Monotonically incremented by the dispatcher each time this slot
    /// receives new work. The worker observes a change in `seq` to wake.
    seq: std.atomic.Value(u64) align(64),
    /// Worker writes the seq it just completed. Dispatcher spins until
    /// `done_seq >= target` (where `target` is the seq value it posted).
    done_seq: std.atomic.Value(u64) align(64),
};

/// Single unit of work posted into the pool.
/// `fn_` is invoked with `ctx` on either the calling thread (task 0) or a
/// worker thread (tasks 1..). The caller owns the storage `ctx` points at
/// and must keep it alive until `dispatchAndRun` returns.
pub const Task = struct {
    fn_: *const fn (*anyopaque) void,
    ctx: *anyopaque,
};

/// Persistent worker pool that fans matvec barriers out across N threads.
/// Slots are cache-line aligned and communicated via release/acquire atomics;
/// see the module doc for the rationale and benchmark numbers.
pub const FastPool = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    threads: []std.Thread,
    n_workers: usize,
    slots: [max_workers]Slot align(64),
    shutdown: std.atomic.Value(bool) align(64),

    /// Spawn `n_workers` persistent worker threads bound to this pool.
    /// Initializes the slot table, then launches each worker on `workerMain`.
    /// @param self Pool storage; written in place so callers can keep it on
    /// the stack.
    /// @param allocator Used for the `threads` array only.
    /// @param n_workers Worker thread count; must be in `1..=max_workers`.
    /// @returns `error.InvalidWorkerCount` when `n_workers` is out of range,
    /// or a spawn error from `std.Thread.spawn`.
    pub fn init(self: *Self, allocator: std.mem.Allocator, n_workers: usize) !void {
        if (n_workers == 0 or n_workers > max_workers) return error.InvalidWorkerCount;
        self.* = .{
            .allocator = allocator,
            .threads = &.{},
            .n_workers = n_workers,
            .slots = undefined,
            .shutdown = std.atomic.Value(bool).init(false),
        };
        for (&self.slots) |*s| {
            s.* = .{
                .run_fn = std.atomic.Value(?*const fn (*anyopaque) void).init(null),
                .ctx = std.atomic.Value(?*anyopaque).init(null),
                .seq = std.atomic.Value(u64).init(0),
                .done_seq = std.atomic.Value(u64).init(0),
            };
        }
        self.threads = try allocator.alloc(std.Thread, n_workers);
        errdefer allocator.free(self.threads);

        var spawned: usize = 0;
        errdefer {
            self.shutdown.store(true, .release);
            for (self.threads[0..spawned]) |t| t.join();
        }
        for (self.threads, 0..) |*t, i| {
            t.* = try std.Thread.spawn(.{}, workerMain, .{ self, i });
            spawned += 1;
        }
    }

    /// Signal shutdown, wake every worker, join all threads, and free state.
    /// Bumps each slot's `seq` after raising the shutdown flag so workers
    /// observing a spin-loop step out and exit promptly.
    pub fn deinit(self: *Self) void {
        self.shutdown.store(true, .release);
        // Bump every slot's seq so workers that are spinning notice and
        // observe the shutdown flag.
        for (self.slots[0..self.n_workers]) |*s| {
            _ = s.seq.fetchAdd(1, .release);
        }
        for (self.threads) |t| t.join();
        self.allocator.free(self.threads);
        self.threads = &.{};
        self.n_workers = 0;
    }

    /// Post `tasks[1..]` to worker slots and run `tasks[0]` on the calling thread.
    /// Spins on each worker's done-sequence until all tasks have completed, then returns.
    /// @param tasks Slice of tasks to execute. `tasks[0]` runs on the caller;
    /// `tasks[1..n]` are posted to workers 0..n-1 via atomic slot writes.
    /// @note `tasks.len` must be `<= n_workers + 1`; the assertion is a hard
    /// assert, not a returned error. Passing an empty slice is a no-op.
    pub fn dispatchAndRun(self: *Self, tasks: []const Task) void {
        if (tasks.len == 0) return;
        if (tasks.len == 1) {
            tasks[0].fn_(tasks[0].ctx);
            return;
        }
        const n_to_post: usize = tasks.len - 1;
        std.debug.assert(n_to_post <= self.n_workers);

        var targets: [max_workers]u64 = undefined;
        for (tasks[1..], 0..) |t, i| {
            const slot = &self.slots[i];
            slot.ctx.store(t.ctx, .release);
            slot.run_fn.store(t.fn_, .release);
            // fetchAdd publishes the new run_fn/ctx (the .release on the
            // increment is paired with the worker's .acquire on `seq`).
            const prev = slot.seq.fetchAdd(1, .release);
            targets[i] = prev + 1;
        }

        // Main thread runs task 0 while the workers churn through the rest.
        tasks[0].fn_(tasks[0].ctx);

        // 16384 spinLoopHints on Zen 4 ≈ 150-300 µs of spin time. The decode loop
        // dispatches every ~50-200 µs, so a single matvec wait rarely exceeds the
        // spin budget — keeping the dispatcher off the kernel's sched_yield path
        // saves the ~100-500 ns syscall + wake-up cost per dispatch. Across ~220
        // dispatches/token this is the difference between paying that cost ~3-10x
        // per matvec (old 1024-spin budget = ~10-25 µs, yields multiple times per
        // wait) versus essentially never.
        for (0..n_to_post) |i| {
            const slot = &self.slots[i];
            const target = targets[i];
            var spins: usize = 0;
            while (slot.done_seq.load(.acquire) < target) {
                if (spins < 16384) {
                    std.atomic.spinLoopHint();
                    spins += 1;
                } else {
                    std.Thread.yield() catch {};
                    spins = 0;
                }
            }
        }
    }

    /// Return the total number of execution contexts available: worker threads plus the calling thread.
    /// Use this value to size task arrays passed to `dispatchAndRun`.
    /// @returns `n_workers + 1`.
    pub fn executorCount(self: *const Self) usize {
        return self.n_workers + 1;
    }
};

fn workerMain(pool: *FastPool, slot_idx: usize) void {
    const slot = &pool.slots[slot_idx];
    var last: u64 = 0;
    var idle_spins: usize = 0;
    while (true) {
        const cur = slot.seq.load(.acquire);
        if (cur == last) {
            if (pool.shutdown.load(.acquire)) return;
            // Match the dispatcher's spin budget so workers stay hot across the
            // inter-dispatch idle window — saves the wake-up + cache-migration
            // cost when the next dispatch hits the slot.
            if (idle_spins < 16384) {
                std.atomic.spinLoopHint();
                idle_spins += 1;
            } else {
                std.Thread.yield() catch {};
                idle_spins = 0;
            }
            continue;
        }
        idle_spins = 0;
        if (pool.shutdown.load(.acquire)) return;
        const run_fn = slot.run_fn.load(.acquire);
        const ctx = slot.ctx.load(.acquire);
        if (run_fn) |f| f(ctx.?);
        slot.done_seq.store(cur, .release);
        last = cur;
    }
}

test "fast pool dispatches and runs" {
    if (@import("builtin").single_threaded) return;
    var pool: FastPool = undefined;
    try pool.init(std.testing.allocator, 3);
    defer pool.deinit();

    var counter: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);
    const Ctx = struct {
        counter: *std.atomic.Value(u32),
        delta: u32,
        fn run(ctx: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            _ = self.counter.fetchAdd(self.delta, .acq_rel);
        }
    };
    var c0 = Ctx{ .counter = &counter, .delta = 1 };
    var c1 = Ctx{ .counter = &counter, .delta = 2 };
    var c2 = Ctx{ .counter = &counter, .delta = 4 };
    var c3 = Ctx{ .counter = &counter, .delta = 8 };
    const tasks = [_]Task{
        .{ .fn_ = Ctx.run, .ctx = @ptrCast(&c0) },
        .{ .fn_ = Ctx.run, .ctx = @ptrCast(&c1) },
        .{ .fn_ = Ctx.run, .ctx = @ptrCast(&c2) },
        .{ .fn_ = Ctx.run, .ctx = @ptrCast(&c3) },
    };
    pool.dispatchAndRun(&tasks);
    try std.testing.expectEqual(@as(u32, 15), counter.load(.seq_cst));

    // Second dispatch reuses the same slots — exercise the seq advance.
    counter.store(0, .seq_cst);
    pool.dispatchAndRun(&tasks);
    try std.testing.expectEqual(@as(u32, 15), counter.load(.seq_cst));
}
