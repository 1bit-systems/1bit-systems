//! Paged KV cache manager with H2O (Heavy-Hitter Oracle) eviction.
//! @section Scheduler
//! Manages a pool of fixed-size pages allocated per-request and freed
//! on completion or cancellation. When the free list is exhausted, applies
//! H2O eviction: pages with lowest cumulative attention scores are evicted
//! and their page-table entries remapped to a shared zero-filled page.
const std = @import("std");
const log = std.log.scoped(.kv_cache);

/// Eviction policy selecting which pages to drop when cache is full.
pub const EvictionPolicy = enum(u8) {
    none = 0,
    h2o_attention_score = 1,
    lru = 2,
    fifo = 3,
};

/// A single page in the KV cache pool.
pub const KvPage = struct {
    page_id: u32,
    owner: ?u64,
    token_start: u32,
    token_count: u32,
    cumulative_score: f32 = 0.0,
    score_count: u32 = 0,
    evicted: bool = false,
    last_access_seq: u64 = 0,
};

fn h2oScoreLessThan(_: void, a: struct { u32, f32 }, b: struct { u32, f32 }) std.math.Order {
    return std.math.order(a[1], b[1]);
}

/// Pool-based allocator for KV cache pages with H2O eviction support.
pub const KvPagePool = struct {
    pages: []KvPage,
    free_list: std.ArrayList(u32),
    eviction_heap: std.PriorityQueue(struct { u32, f32 }, void, h2oScoreLessThan),
    page_size: u32,
    total_pages: u32,
    allocator: std.mem.Allocator,
    eviction_policy: EvictionPolicy = .none,
    access_seq: u64 = 0,
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
            if (i < usable_pages) try free_list.append(allocator, @intCast(i));
        }
        return KvPagePool{
            .pages = pages,
            .free_list = free_list,
            .eviction_heap = std.PriorityQueue(struct { u32, f32 }, void, h2oScoreLessThan).empty,
            .recently_evicted = std.ArrayList(u32).empty,
            .page_size = page_size,
            .total_pages = total_pages,
            .allocator = allocator,
            .zero_page_id = total_pages - 1,
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
        if (self.free_list.items.len >= count) return try self.allocPages(request_id, count);
        if (self.eviction_policy == .none) return error.KvCacheExhausted;
        const needed = count - self.freeCount();
        const evicted = self.evictLowestScoringPages(needed + @divTrunc(needed, 2));
        if (evicted < needed) return error.KvCacheExhausted;
        return try self.allocPages(request_id, count);
    }

    pub fn allocPages(self: *KvPagePool, request_id: u64, count: u32) ![]u32 {
        if (self.free_list.items.len < count) return error.KvCacheExhausted;
        const result = try self.allocator.alloc(u32, count);
        for (0..count) |i| {
            const page_id = self.free_list.pop() orelse return error.KvCacheExhausted;
            const page = &self.pages[page_id];
            page.owner = request_id;
            page.token_count = 0;
            page.cumulative_score = 0.0;
            page.score_count = 0;
            page.evicted = false;
            page.last_access_seq = self.access_seq;
            result[i] = page_id;
        }
        return result;
    }

    pub fn evictLowestScoringPages(self: *KvPagePool, count: u32) u32 {
        if (self.eviction_policy == .none) return 0;
        return switch (self.eviction_policy) {
            .none => 0,
            .h2o_attention_score => self.evictH2O(count),
            .lru => self.evictLRU(count),
            .fifo => self.evictFIFO(count),
        };
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
            const page = &self.pages[entry[0]];
            if (page.owner == null or page.evicted) continue;
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
        return self.evictLRU(count); // same strategy for FIFO
    }

    fn markEvicted(self: *KvPagePool, page: *KvPage) void {
        if (page.owner == null) return;
        self.recently_evicted.ensureUnusedCapacity(self.allocator, 1) catch {};
        self.free_list.ensureUnusedCapacity(self.allocator, 1) catch {};
        self.recently_evicted.appendAssumeCapacity(page.page_id);
        page.owner = null;
        page.token_count = 0;
        page.evicted = true;
        self.free_list.appendAssumeCapacity(page.page_id);
    }

    pub fn recordScores(self: *KvPagePool, page_ids: []const u32, token_scores: ?[]const f32) void {
        if (self.eviction_policy == .none) return;
        self.access_seq += 1;
        for (page_ids) |pid| {
            const page = &self.pages[pid];
            if (page.owner == null) continue;
            const page_avg: f32 = if (token_scores) |scores| blk: {
                var sum: f32 = 0.0;
                for (0..@as(usize, @min(page.token_count, self.page_size))) |i|
                    sum += scores[page.token_start + i];
                break :blk sum / @as(f32, @floatFromInt(@min(page.token_count, self.page_size)));
            } else 1.0;
            page.cumulative_score += page_avg;
            page.score_count += 1;
            page.last_access_seq = self.access_seq;
            if (self.eviction_policy == .h2o_attention_score) {
                const avg = if (page.score_count > 0)
                    page.cumulative_score / @as(f32, @floatFromInt(page.score_count))
                else 0.0;
                self.eviction_heap.push(self.allocator, .{ pid, avg }) catch {};
            }
        }
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

    pub fn freeCount(self: *const KvPagePool) u32 { return @intCast(self.free_list.items.len); }

    pub fn deinit(self: *KvPagePool) void {
        if (self.eviction_policy != .none) self.eviction_heap.deinit(self.allocator);
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
}

test "KvPagePool alloc and free" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.init(allocator, 5, 256);
    defer pool.deinit();
    const pages = try pool.allocPages(1, 2);
    defer allocator.free(pages);
    try std.testing.expectEqual(@as(u32, 2), pool.freeCount());
    pool.freePages(1);
    try std.testing.expectEqual(@as(u32, 4), pool.freeCount());
}
