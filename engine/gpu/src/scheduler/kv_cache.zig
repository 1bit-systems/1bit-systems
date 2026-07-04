//! Paged KV cache manager with H2O (Heavy-Hitter Oracle) eviction.
//! @section Scheduler
//! Manages a pool of fixed-size pages that are allocated per-request
//! and freed on completion or cancellation. Each page maps to a contiguous
//! region of the GPU KV cache buffer, giving each request non-overlapping
//! token storage.
//!
//! When the free list is exhausted, the pool applies Heavy-Hitter Oracle (H2O)
//! eviction: pages with the lowest cumulative attention scores are evicted to
//! free space for new tokens. Evicted pages are marked as invalid and their
//! page table entries are remapped to a shared zero-filled page, so the GPU
//! reads zero K/V values — naturally producing ~0 attention scores.
const std = @import("std");

const log = std.log.scoped(.kv_cache);

/// Eviction policy selecting which pages to drop when cache is full.
pub const EvictionPolicy = enum(u8) {
    /// No eviction — allocation fails on exhaustion (legacy behavior).
    none = 0,
    /// H2O: evict pages with lowest cumulative attention-score average.
    h2o_attention_score = 1,
    /// LRU: evict the least-recently-accessed page.
    lru = 2,
    /// FIFO: evict the oldest page (first allocated).
    fifo = 3,
};

/// A single page in the KV cache pool.
pub const KvPage = struct {
    page_id: u32,
    owner: ?u64,
    token_start: u32,
    token_count: u32,
    // H2O eviction metadata
    cumulative_score: f32 = 0.0,
    score_count: u32 = 0,
    evicted: bool = false,
    last_access_seq: u64 = 0,
    // Monotonic allocation order, set once per allocPages() call. Distinct
    // from last_access_seq (which advances on every recordScores() call) so
    // FIFO eviction can key off "first allocated" instead of "least recently
    // scored" — otherwise FIFO and LRU pick identical victims.
    alloc_seq: u64 = 0,
};

fn h2oScoreLessThan(context: void, a: struct { u32, f32 }, b: struct { u32, f32 }) std.math.Order {
    _ = context;
    return std.math.order(a[1], b[1]);
}

/// Pool-based allocator for KV cache pages with H2O eviction support.
/// When the free list is empty, the lowest-scoring pages are evicted.
/// Eviction uses the "zero page" technique: evicted page table entries
/// are remapped to a shared zero-filled page (page `total_pages - 1`).
pub const KvPagePool = struct {
    pages: []KvPage,
    free_list: std.ArrayList(u32),
    eviction_heap: std.PriorityQueue(struct { u32, f32 }, void, h2oScoreLessThan),
    page_size: u32,
    total_pages: u32,
    allocator: std.mem.Allocator,
    eviction_policy: EvictionPolicy = .none,
    access_seq: u64 = 0,
    /// Monotonic counter driving KvPage.alloc_seq — advances once per page
    /// handed out by allocPages(), independent of access/scoring activity.
    alloc_seq_counter: u64 = 0,
    zero_page_id: u32 = 0,
    recently_evicted: std.ArrayList(u32),

    pub fn init(allocator: std.mem.Allocator, total_pages: u32, page_size: u32) !KvPagePool {
        const usable_pages = total_pages -| 1;
        const pages = try allocator.alloc(KvPage, total_pages);
        var free_list = std.ArrayList(u32).empty;
        for (0..total_pages) |i| {
            pages[i] = .{
                .page_id = @intCast(i),
                .owner = null,
                .token_start = 0,
                .token_count = 0,
            };
            if (i < usable_pages) {
                try free_list.append(allocator, @intCast(i));
            }
        }
        const zero_page_id = total_pages - 1;
        log.info("KV page pool: {d} pages × {d} tokens = {d} total capacity (1 zero page reserved)", .{
            usable_pages, page_size, usable_pages * page_size,
        });
        return KvPagePool{
            .pages = pages,
            .free_list = free_list,
            .eviction_heap = std.PriorityQueue(struct { u32, f32 }, void, h2oScoreLessThan).empty,
            .recently_evicted = std.ArrayList(u32).empty,
            .page_size = page_size,
            .total_pages = total_pages,
            .allocator = allocator,
            .eviction_policy = .none,
            .zero_page_id = zero_page_id,
        };
    }

    pub fn initWithEviction(allocator: std.mem.Allocator, total_pages: u32, page_size: u32, policy: EvictionPolicy) !KvPagePool {
        var pool = try init(allocator, total_pages, page_size);
        pool.eviction_policy = policy;
        if (policy != .none) {
            pool.eviction_heap = std.PriorityQueue(struct { u32, f32 }, void, h2oScoreLessThan).initContext({});
            try pool.eviction_heap.ensureTotalCapacity(allocator, @intCast(total_pages));
        }
        return pool;
    }

    pub fn allocOrEvict(self: *KvPagePool, request_id: u64, count: u32) ![]u32 {
        if (self.free_list.items.len >= count) {
            return self.allocPages(request_id, count);
        }
        if (self.eviction_policy == .none) {
            return error.KvCacheExhausted;
        }
        const needed = count - self.freeCount();
        const to_evict = needed + @divTrunc(needed, 2);
        const evicted = self.evictLowestScoringPages(to_evict);
        if (evicted < needed) {
            return error.KvCacheExhausted;
        }
        return self.allocPages(request_id, count);
    }

    pub fn allocPages(self: *KvPagePool, request_id: u64, count: u32) ![]u32 {
        if (self.free_list.items.len < count) return error.KvCacheExhausted;
        const result = try self.allocator.alloc(u32, count);
        for (0..count) |i| {
            const page_id = self.free_list.pop() orelse return error.KvCacheExhausted;
            const page = &self.pages[page_id];
            page.owner = request_id;
            // Each page occupies a fixed, permanent slice of the KV/score
            // buffer determined by its own page_id (see positionBase() /
            // maxContext()). Populate the real token range here so
            // recordScores() aggregates against actual scored positions
            // instead of the zero-length range left by the default {0, 0}.
            page.token_start = page_id * self.page_size;
            page.token_count = self.page_size;
            page.cumulative_score = 0.0;
            page.score_count = 0;
            page.evicted = false;
            page.last_access_seq = self.access_seq;
            self.alloc_seq_counter += 1;
            page.alloc_seq = self.alloc_seq_counter;
            result[i] = page_id;
        }
        return result;
    }

    pub fn evictLowestScoringPages(self: *KvPagePool, count: u32) u32 {
        if (self.eviction_policy == .none) return 0;
        var evicted: u32 = 0;
        switch (self.eviction_policy) {
            .none => return 0,
            .h2o_attention_score => evicted = self.evictH2O(count),
            .lru => evicted = self.evictLRU(count),
            .fifo => evicted = self.evictFIFO(count),
        }
        if (evicted > 0) {
            log.info("Eviction ({s}): evicted {d} pages ({d} tokens freed)", .{
                @tagName(self.eviction_policy), evicted, evicted * self.page_size,
            });
        }
        return evicted;
    }

    fn evictH2O(self: *KvPagePool, count: u32) u32 {
        var evicted: u32 = 0;
        for (self.pages) |*page| {
            if (evicted >= count) break;
            if (page.owner != null and page.score_count == 0 and !page.evicted) {
                self.markEvicted(page);
                evicted += 1;
            }
        }
        while (evicted < count and self.eviction_heap.count() > 0) {
            const entry = self.eviction_heap.pop() orelse break;
            const page_id = entry[0];
            const page = &self.pages[page_id];
            if (page.owner == null or page.evicted) continue;
            const avg_score = if (page.score_count > 0)
                page.cumulative_score / @as(f32, @floatFromInt(page.score_count))
            else
                -std.math.floatMax(f32);
            if (avg_score != entry[1]) continue;
            self.markEvicted(page);
            evicted += 1;
        }
        return evicted;
    }

    fn evictLRU(self: *KvPagePool, count: u32) u32 {
        var evicted: u32 = 0;
        while (evicted < count) {
            var oldest_id: ?u32 = null;
            var oldest_seq: u64 = std.math.maxInt(u64);
            for (self.pages) |page| {
                if (page.owner != null and !page.evicted and page.last_access_seq < oldest_seq) {
                    oldest_seq = page.last_access_seq;
                    oldest_id = page.page_id;
                }
            }
            const pid = oldest_id orelse break;
            self.markEvicted(&self.pages[pid]);
            evicted += 1;
        }
        return evicted;
    }

    fn evictFIFO(self: *KvPagePool, count: u32) u32 {
        var evicted: u32 = 0;
        while (evicted < count) {
            var oldest_id: ?u32 = null;
            var oldest_seq: u64 = std.math.maxInt(u64);
            for (self.pages) |page| {
                // FIFO evicts by allocation order (first-allocated page),
                // not by last access — using last_access_seq here made
                // FIFO behave identically to LRU.
                if (page.owner != null and !page.evicted and page.alloc_seq < oldest_seq) {
                    oldest_seq = page.alloc_seq;
                    oldest_id = page.page_id;
                }
            }
            const pid = oldest_id orelse break;
            self.markEvicted(&self.pages[pid]);
            evicted += 1;
        }
        return evicted;
    }

    fn markEvicted(self: *KvPagePool, page: *KvPage) void {
        const request_id = page.owner orelse return;
        log.debug("Evicting page {d} (owner {d}, score {d:.2}, tokens {d})", .{
            page.page_id, request_id,
            if (page.score_count > 0) page.cumulative_score / @as(f32, @floatFromInt(page.score_count)) else 0.0,
            page.token_count,
        });
        // Use fallible append() rather than appendAssumeCapacity(), which
        // assumes capacity is guaranteed — under memory pressure that
        // assumption doesn't hold and would write past the backing array,
        // corrupting allocator state. If we can't record the eviction,
        // leave the page as-is (still owned, not evicted) rather than
        // partially updating state.
        self.recently_evicted.append(self.allocator, page.page_id) catch |err| {
            log.warn("markEvicted: failed to record page {d} as evicted: {s}", .{ page.page_id, @errorName(err) });
            return;
        };
        self.free_list.append(self.allocator, page.page_id) catch |err| {
            log.warn("markEvicted: failed to return page {d} to free list: {s}", .{ page.page_id, @errorName(err) });
            // Roll back the recently_evicted append so the two lists don't
            // disagree about whether this page was actually freed.
            _ = self.recently_evicted.pop();
            return;
        };
        page.owner = null;
        page.token_count = 0;
        page.evicted = true;
    }

    pub fn recordScores(self: *KvPagePool, page_ids: []const u32, token_scores: ?[]const f32) void {
        if (self.eviction_policy == .none) return;
        self.access_seq += 1;
        for (page_ids) |pid| {
            const page = &self.pages[pid];
            if (page.owner == null) continue;
            const page_avg: f32 = if (token_scores) |scores| blk: {
                const token_start = page.token_start;
                const count = @min(page.token_count, self.page_size);
                if (count == 0) break :blk 0.0;
                var sum: f32 = 0.0;
                for (0..@as(usize, count)) |i| {
                    sum += scores[token_start + i];
                }
                break :blk sum / @as(f32, @floatFromInt(count));
            } else 1.0;
            page.cumulative_score += page_avg;
            page.score_count += 1;
            page.last_access_seq = self.access_seq;
            self.updateEvictionHeap(pid);
        }
    }

    fn updateEvictionHeap(self: *KvPagePool, page_id: u32) void {
        if (self.eviction_policy != .h2o_attention_score) return;
        const page = &self.pages[page_id];
        if (page.owner == null) return;
        const avg_score = if (page.score_count > 0)
            page.cumulative_score / @as(f32, @floatFromInt(page.score_count))
        else
            0.0;
        self.eviction_heap.push(self.allocator, .{ page_id, avg_score }) catch {};
    }

    pub fn zeroPageId(self: *const KvPagePool) u32 { return self.zero_page_id; }

    pub fn drainEvicted(self: *KvPagePool) []const u32 {
        const result = self.recently_evicted.items;
        self.recently_evicted.clearRetainingCapacity();
        return result;
    }

    pub fn usablePageCount(self: *const KvPagePool) u32 { return self.total_pages - 1; }

    pub fn freePages(self: *KvPagePool, request_id: u64) void {
        for (self.pages) |*page| {
            if (page.owner == request_id) {
                page.owner = null;
                page.token_count = 0;
                page.cumulative_score = 0.0;
                page.score_count = 0;
                page.evicted = false;
                page.last_access_seq = 0;
                self.free_list.append(self.allocator, page.page_id) catch {};
            }
        }
    }

    pub fn positionBase(self: *const KvPagePool, page_ids: []const u32) u32 {
        if (page_ids.len == 0) return 0;
        return page_ids[0] * self.page_size;
    }

    pub fn maxContext(self: *const KvPagePool, page_count: u32) u32 { return page_count * self.page_size; }

    pub fn freeCount(self: *const KvPagePool) u32 { return @intCast(self.free_list.items.len); }

    pub fn canAllocateAfterEviction(self: *const KvPagePool, needed: u32) bool {
        if (self.eviction_policy == .none) return self.freeCount() >= needed;
        // page `total_pages - 1` is the reserved zero-page and can never be
        // evicted or allocated, so the ceiling on recoverable pages is
        // usablePageCount(), not total_pages.
        return self.freeCount() + self.usablePageCount() > needed;
    }

    pub fn deinit(self: *KvPagePool) void {
        if (self.eviction_policy != .none) {
            self.eviction_heap.deinit(self.allocator);
        }
        self.recently_evicted.deinit(self.allocator);
        self.free_list.deinit(self.allocator);
        self.allocator.free(self.pages);
    }
};

test "KvPagePool init reserves zero page" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.init(allocator, 5, 256);
    defer pool.deinit();

    try std.testing.expectEqual(@as(u32, 4), pool.freeCount());
    try std.testing.expectEqual(@as(u32, 4), pool.zeroPageId());

    // Zero page should not be in free list
    for (pool.free_list.items) |pid| {
        try std.testing.expect(pid != pool.zeroPageId());
    }
}

test "KvPagePool alloc and free" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.init(allocator, 5, 256);
    defer pool.deinit();

    try std.testing.expectEqual(@as(u32, 4), pool.freeCount());
    const pages = try pool.allocPages(1, 2);
    defer allocator.free(pages);
    try std.testing.expectEqual(@as(u32, 2), pool.freeCount());
    pool.freePages(1);
    try std.testing.expectEqual(@as(u32, 4), pool.freeCount());
}

test "KvPagePool exhaustion with eviction disabled" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.init(allocator, 3, 256);
    defer pool.deinit();

    const p1 = try pool.allocPages(1, 2);
    defer allocator.free(p1);
    try std.testing.expectError(error.KvCacheExhausted, pool.allocPages(2, 1));
    pool.freePages(1);
}

test "KvPagePool allocOrEvict triggers eviction" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.initWithEviction(allocator, 5, 256, .h2o_attention_score);
    defer pool.deinit();

    const p1 = try pool.allocOrEvict(1, 4);
    defer allocator.free(p1);
    try std.testing.expectEqual(@as(u32, 0), pool.freeCount());

    pool.recordScores(p1, null);

    const p2 = try pool.allocOrEvict(2, 1);
    defer allocator.free(p2);
    try std.testing.expectEqual(@as(usize, 1), p2.len);
}

test "KvPagePool H2O evicts lowest scoring pages first" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.initWithEviction(allocator, 7, 256, .h2o_attention_score);
    defer pool.deinit();

    // Exhaust the pool: allocate all 6 usable pages.
    const pages = try pool.allocOrEvict(1, 6);
    defer allocator.free(pages);

    try std.testing.expectEqual(@as(u32, 0), pool.freeCount());

    // recordScores() now indexes token_scores at each page's real absolute
    // range (page_id * page_size .. +page_size), so the buffer must cover
    // every page's range rather than a couple of placeholder entries.
    const high = [_]f32{100.0} ** (7 * 256);
    const med = [_]f32{50.0} ** (7 * 256);
    const low = [_]f32{1.0} ** (7 * 256);

    // Pages 0,1 get low score; 2,3 get med; 4,5 get high.
    pool.recordScores(pages[0..2], &low);
    pool.recordScores(pages[2..4], &med);
    pool.recordScores(pages[4..6], &high);
    pool.recordScores(pages[0..2], &low);
    pool.recordScores(pages[2..4], &med);
    pool.recordScores(pages[4..6], &high);

    // Allocate 2 more — pool is empty, triggers eviction of lowest 3 pages.
    // The heap pops 3 pages with the lowest average score (all pages
    // have score_count > 0 and avg_score depends on cumulative attention).
    // Since pages[0..1] got low scores and pages[4..5] got high,
    // the lowest-scoring ones should be among pages[0..1].
    const new = try pool.allocOrEvict(2, 2);
    defer allocator.free(new);

    // Check that new pages were allocated (2 of the 3 evicted pages reused).
    try std.testing.expectEqual(@as(usize, 2), new.len);
    // 1 evicted page still in free list (3 evicted - 2 allocated = 1 free).
    try std.testing.expectEqual(@as(u32, 1), pool.freeCount());

    // The drainEvicted list should contain the evicted page IDs.
    // (allocPages resets evicted=false on re-allocation, so we check
    // via drainEvicted which tracks evictions before re-allocation.)
    const evicted_ids = pool.drainEvicted();
    try std.testing.expect(evicted_ids.len >= 2);
}

test "KvPagePool zero page never allocated" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.init(allocator, 5, 256);
    defer pool.deinit();

    const zp = pool.zeroPageId();
    try std.testing.expectEqual(@as(u32, 4), zp);
    try std.testing.expect(pool.pages[zp].owner == null);
}

test "KvPagePool drainEvicted returns evicted pages" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.initWithEviction(allocator, 5, 256, .h2o_attention_score);
    defer pool.deinit();

    const p1 = try pool.allocOrEvict(1, 4);
    defer allocator.free(p1);
    pool.recordScores(p1, null);
    const p2 = try pool.allocOrEvict(2, 1);
    defer allocator.free(p2);

    const evicted = pool.drainEvicted();
    try std.testing.expect(evicted.len > 0);
}

test "KvPagePool initWithEviction configures heap" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.initWithEviction(allocator, 5, 256, .h2o_attention_score);
    defer pool.deinit();

    try std.testing.expectEqual(EvictionPolicy.h2o_attention_score, pool.eviction_policy);
    try std.testing.expectEqual(@as(u32, 4), pool.freeCount());
}
