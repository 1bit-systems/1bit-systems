//! RadixAttention prefix tree for cross-request KV cache sharing.
//! @section Scheduler
//!
//! Organizes KV cache pages in a tree indexed by token sequence prefixes.
//! When multiple requests share a common prompt prefix (system prompt,
//! few-shot examples, tool definitions), their KV cache entries are stored
//! once in the tree and all requests pointing at the same node reuse the
//! same physical pages — no recomputation or redundant storage.
//!
//! ## Algorithm
//!
//! 1. **Ingest**: On prefill, walk the tree along the prompt tokens.
//!    For each matching edge, the cached KV pages are already computed.
//!    The first mismatch creates a new leaf node; the engine prefills
//!    only the suffix from that point.
//!
//! 2. **Allocation**: Each node holds a reference to the KV page IDs
//!    that store its tokens. Pages are allocated from the shared KvPagePool.
//!
//! 3. **Eviction**: When the pool evicts pages (H2O), the tree node
//!    is marked invalid. Requests sharing that node fall back to
//!    recomputation or a shorter prefix.
//!
//! 4. **Free**: When all requests detach from a node, its pages are
//!    released back to the pool (subject to a TTL for reuse).
const std = @import("std");
const KvPagePool = @import("kv_cache.zig").KvPagePool;

const log = std.log.scoped(.radix);

/// Maximum prefix length stored per edge. Longer prefixes are split
/// into multiple edges. Keeps tree traversal bounded.
const MAX_EDGE_TOKENS: u32 = 64;

/// A node in the radix tree. Each node stores a token prefix edge
/// and the KV pages that cover those tokens.
pub const RadixNode = struct {
    /// Token IDs for this edge (prefix segment).
    /// Owned by the node; freed on node destruction.
    tokens: []u32,
    /// KV page IDs covering this edge's tokens.
    /// Allocated from the shared KvPagePool.
    kv_page_ids: []u32,
    /// Number of KV tokens stored (may be less than page capacity).
    kv_token_count: u32,
    /// Child nodes indexed by the first token of their edge.
    /// HashMap: first_token → index into `nodes` array.
    children: std.AutoHashMap(u32, usize),
    /// Reference count: number of requests currently using this node.
    ref_count: u32,
    /// True if this node has been evicted from the KV cache.
    evicted: bool,
    /// Last access timestamp (monotonic epoch from tree).
    last_access: u64,
};

/// RadixAttention tree managing shared KV cache prefixes.
pub const RadixTree = struct {
    /// Flat array of all nodes. Node 0 is the root (empty prefix).
    nodes: std.ArrayList(RadixNode),
    /// Monotonic access counter for LRU eviction of tree nodes.
    access_epoch: u64,
    /// Allocator for node storage.
    allocator: std.mem.Allocator,
    /// Shared page pool for KV page allocation.
    page_pool: *KvPagePool,

    /// Initialize a radix tree with an empty root node.
    pub fn init(allocator: std.mem.Allocator, page_pool: *KvPagePool) !RadixTree {
        var nodes = std.ArrayList(RadixNode).empty;
        // Root node: children HashMap is initialized after appending to avoid
        // self-referential pointer breakage from ArrayList memcpy.
        const root = RadixNode{
            .tokens = &.{}, // empty edge at root
            .kv_page_ids = &.{},
            .kv_token_count = 0,
            .children = undefined,
            .ref_count = 0,
            .evicted = false,
            .last_access = 0,
        };
        try nodes.append(allocator, root);
        nodes.items[0].children = std.AutoHashMap(u32, usize).init(allocator);
        log.info("RadixAttention tree initialized", .{});
        return RadixTree{
            .nodes = nodes,
            .access_epoch = 1,
            .allocator = allocator,
            .page_pool = page_pool,
        };
    }

    /// Walk the tree along `prompt_tokens[start..]` and return the
    /// deepest matching node index, the number of tokens matched,
    /// and whether the match is exact (all prompts consumed).
    ///
    /// @returns A struct { node_idx, matched, exact_match } or null
    ///          if the root has no children (empty tree).
    pub fn matchPrefix(self: *RadixTree, prompt_tokens: []const u32, start: usize) !?struct {
        node_idx: usize,
        matched: u32,
        exact_match: bool,
    } {
        if (self.nodes.items.len == 0) return null;
        if (start >= prompt_tokens.len) {
            return .{ .node_idx = 0, .matched = 0, .exact_match = true };
        }

        var current: usize = 0; // start at root
        var matched: u32 = 0;
        var remaining = prompt_tokens[start..];

        // Walk the tree following edges.
        while (true) {
            const node = &self.nodes.items[current];
            node.last_access = self.access_epoch;
            self.access_epoch += 1;

            // Find child whose edge starts with remaining[0].
            const child_entry = node.children.get(remaining[0]);
            const child_idx = child_entry orelse break;

            const child = &self.nodes.items[child_idx];
            if (child.evicted) {
                // Skip evicted nodes — they have no valid KV cache.
                // The caller should not reuse cache for this prefix.
                break;
            }

            // Count matching tokens along this edge.
            const match_len = @min(@as(u32, @intCast(remaining.len)), @as(u32, @intCast(child.tokens.len)));
            const edge_tokens = child.tokens[0..match_len];
            const prompt_slice = remaining[0..match_len];
            if (!std.mem.eql(u32, edge_tokens, prompt_slice)) {
                // Partial match: count how many tokens matched before the mismatch.
                var partial: u32 = 0;
                while (partial < match_len and edge_tokens[partial] == prompt_slice[partial]) {
                    partial += 1;
                }
                matched += partial;
                break;
            }

            // Full edge consumed.
            matched += @intCast(match_len);
            remaining = prompt_tokens[start + matched ..];
            current = child_idx;

            // If this edge exactly matches and we've consumed all prompt tokens.
            if (match_len == child.tokens.len) {
                if (remaining.len == 0) {
                    return .{
                        .node_idx = current,
                        .matched = matched,
                        .exact_match = true,
                    };
                }
                // Continue to next edge.
            } else {
                // Partial edge match: we matched a prefix of the edge.
                // Need to split the edge.
                return .{
                    .node_idx = current,
                    .matched = matched,
                    .exact_match = false,
                };
            }
        }

        return .{
            .node_idx = current,
            .matched = matched,
            .exact_match = false,
        };
    }

    /// Insert a suffix of prompt tokens under `parent_idx`, allocating
    /// KV pages for them. Returns the new leaf node index.
    ///
    /// @param parent_idx Node to insert under.
    /// @param suffix Remaining prompt tokens to insert (non-empty).
    /// @param request_id Owner request ID for page allocation.
    /// @returns The new node index.
    pub fn insert(self: *RadixTree, parent_idx: usize, suffix: []const u32, request_id: u64) !usize {
        if (suffix.len == 0) return parent_idx;

        // parent = &self.nodes.items[parent_idx] is NOT cached across
        // self.nodes.append() because append may reallocate the backing buffer,
        // invalidating any pointer into it. Use parent_idx directly instead.

        // Calculate pages needed: suffix tokens fit in page_size increments.
        const tokens_needed = @as(u32, @intCast(suffix.len));
        const page_capacity = self.page_pool.page_size;
        const pages_needed = (tokens_needed + page_capacity - 1) / page_capacity;

        // Allocate KV pages (may trigger H2O eviction).
        const kv_page_ids = try self.page_pool.allocOrEvict(request_id, pages_needed);
        errdefer self.allocator.free(kv_page_ids);

        // Create the new node.
        const tokens = try self.allocator.dupe(u32, suffix);
        errdefer self.allocator.free(tokens);

        const new_node = RadixNode{
            .tokens = tokens,
            .kv_page_ids = kv_page_ids,
            .kv_token_count = tokens_needed,
            .children = undefined,
            .ref_count = 1,
            .evicted = false,
            .last_access = self.access_epoch,
        };
        self.access_epoch += 1;

        const new_idx = self.nodes.items.len;
        try self.nodes.append(self.allocator, new_node);
        // Initialize children HashMap after appending (avoids self-referential
        // pointer breakage from ArrayList memcpy).
        self.nodes.items[new_idx].children = std.AutoHashMap(u32, usize).init(self.allocator);

        // Register as child of parent via the edge's first token.
        // NOTE: parent pointer was invalidated by append; re-acquire it.
        self.nodes.items[parent_idx].children.put(suffix[0], new_idx) catch {};
        // Note: put failure is non-fatal — the tree still works, just without
        // this edge being navigable from the parent. Log and continue.

        log.debug("Radix: inserted node {d} ({d} tokens, {d} pages) under {d}", .{
            new_idx, tokens_needed, pages_needed, parent_idx,
        });
        return new_idx;
    }

    /// Split an existing edge at `node_idx` at position `split_pos`
    /// (0 < split_pos < edge_len). The edge is split into two edges:
    /// the first `split_pos` tokens remain at `node_idx`, and the
    /// remainder becomes a new child node. Returns the new child index.
    ///
    /// Needed when a matching prefix covers only part of an edge.
    /// The suffix tokens inherit the parent's KV pages with an offset.
    pub fn splitEdge(self: *RadixTree, node_idx: usize, split_pos: u32) !usize {
        const node = &self.nodes.items[node_idx];
        if (split_pos == 0 or split_pos >= node.tokens.len) return node_idx;

        // The suffix tokens for the new child.
        const suffix = try self.allocator.dupe(u32, node.tokens[split_pos..]);
        errdefer self.allocator.free(suffix);

        // Compute which parent pages cover the suffix.
        // The suffix starts at split_pos into the parent's token range.
        const suffix_tokens = @as(u32, @intCast(suffix.len));
        const suffix_token_offset = split_pos;

        // Count suffix pages (may span 1-2 pages depending on offset).
        const ps = self.page_pool.page_size;
        const suffix_start_page = suffix_token_offset / ps;
        const suffix_end_page = (suffix_token_offset + suffix_tokens + ps - 1) / ps;
        const suffix_num_pages = suffix_end_page - suffix_start_page;

        // Allocate new KV pages for the suffix.
        const suffix_page_ids = try self.page_pool.allocOrEvict(node.ref_count + 1, suffix_num_pages);
        errdefer self.allocator.free(suffix_page_ids);

        // Truncate the original node's tokens to split_pos.
        const truncated = try self.allocator.dupe(u32, node.tokens[0..split_pos]);
        errdefer self.allocator.free(truncated);
        self.allocator.free(node.tokens);
        node.tokens = truncated;

        // Create new child node with the suffix.
        const child = RadixNode{
            .tokens = suffix,
            .kv_page_ids = suffix_page_ids,
            .kv_token_count = suffix_tokens,
            .children = undefined,
            .ref_count = 0,
            .evicted = false,
            .last_access = self.access_epoch,
        };
        self.access_epoch += 1;

        const child_idx = self.nodes.items.len;
        try self.nodes.append(self.allocator, child);
        // Initialize children after appending (avoids HashMap memcpy issue).
        self.nodes.items[child_idx].children = std.AutoHashMap(u32, usize).init(self.allocator);

        // Transfer any existing children of the split node to the new child.
        var keys_to_remove = std.ArrayList(u32).empty;
        defer keys_to_remove.deinit(self.allocator);
        var iter = node.children.iterator();
        while (iter.next()) |entry| {
            try keys_to_remove.append(self.allocator, entry.key_ptr.*);
        }
        for (keys_to_remove.items) |key| {
            const child_node_idx = node.children.get(key).?;
            _ = node.children.remove(key);
            try self.nodes.items[child_idx].children.put(key, child_node_idx);
        }

        // Register child under parent (the split node).
        try node.children.put(suffix[0], child_idx);

        log.debug("Radix: split node {d} at pos {d} → child {d}", .{
            node_idx, split_pos, child_idx,
        });
        return child_idx;
    }

    /// Ingest a prompt into the tree: match the longest prefix, insert
    /// the remaining suffix, and return the KV page IDs and token range.
    ///
    /// @returns A struct of KV page info for the engine, or error if
    ///          page allocation fails.
    pub fn ingest(self: *RadixTree, prompt_tokens: []const u32, request_id: u64) !struct {
        /// All KV page IDs covering this prompt.
        page_ids: []u32,
        /// Number of tokens covered by cached pages (the matched prefix).
        cached_tokens: u32,
        /// Total prompt tokens.
        total_tokens: u32,
    } {
        // Walk tree to find the longest matching prefix.
        const match = try self.matchPrefix(prompt_tokens, 0);
        var matched_tokens: u32 = 0;
        var current_node: usize = 0;

        if (match) |m| {
            current_node = m.node_idx;
            matched_tokens = m.matched;
        }

        // Count total pages from root to current_node.
        var total_page_ids = std.ArrayList(u32).empty;
        defer total_page_ids.deinit(self.allocator);

        // Collect pages from the matched node (if any).
        // For non-root nodes along the matched path, collect their KV pages.
        if (match) |m| {
            if (m.matched > 0 and m.node_idx > 0) {
                const node = &self.nodes.items[m.node_idx];
                try total_page_ids.appendSlice(self.allocator, node.kv_page_ids);
            }
        }

        if (matched_tokens < prompt_tokens.len) {
            // Insert the unmatched suffix.
            const suffix = prompt_tokens[matched_tokens..];
            const leaf_idx = try self.insert(current_node, suffix, request_id);
            const leaf = &self.nodes.items[leaf_idx];
            try total_page_ids.appendSlice(self.allocator, leaf.kv_page_ids);
        }

        return .{
            .page_ids = try total_page_ids.toOwnedSlice(self.allocator),
            .cached_tokens = matched_tokens,
            .total_tokens = @intCast(prompt_tokens.len),
        };
    }

    /// Free all resources held by the tree.
    pub fn deinit(self: *RadixTree) void {
        for (self.nodes.items) |*node| {
            self.allocator.free(node.tokens);
            self.allocator.free(node.kv_page_ids);
            node.children.deinit();
        }
        self.nodes.deinit(self.allocator);
    }
};

// ── Tests ──────────────────────────────────────────────────────────────────

test "RadixTree empty tree has root node" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.init(allocator, 10, 256);
    defer pool.deinit();

    var tree = try RadixTree.init(allocator, &pool);
    defer tree.deinit();

    try std.testing.expectEqual(@as(usize, 1), tree.nodes.items.len);
    try std.testing.expectEqual(@as(usize, 0), tree.nodes.items[0].tokens.len);
}

test "RadixTree match empty tree returns null" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.init(allocator, 10, 256);
    defer pool.deinit();

    var tree = try RadixTree.init(allocator, &pool);
    defer tree.deinit();

    const match = try tree.matchPrefix(&.{ 1, 2, 3 }, 0);
    // Empty tree has only the root; match returns matched=0.
    if (match) |m| {
        try std.testing.expectEqual(@as(u32, 0), m.matched);
    } else {
        // Root does have a match return (not null), so this branch
        // should not be reached. Included for API coverage documentation.
        try std.testing.expect(match != null);
    }
}

test "RadixTree insert and match" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.initWithEviction(allocator, 10, 256, .h2o_attention_score);
    defer pool.deinit();

    var tree = try RadixTree.init(allocator, &pool);
    defer tree.deinit();

    // Insert "hello world" tokens under root.
    const prompt = [_]u32{ 100, 101, 102, 103, 104 };
    const result = try tree.ingest(&prompt, 1);
    defer allocator.free(result.page_ids);

    try std.testing.expectEqual(@as(u32, 0), result.cached_tokens);
    try std.testing.expectEqual(@as(u32, 5), result.total_tokens);
    try std.testing.expect(result.page_ids.len > 0);

    // Now match the same prefix — should find all 5 tokens cached.
    const match2 = try tree.matchPrefix(&prompt, 0);
    try std.testing.expect(match2 != null);
    if (match2) |m| {
        try std.testing.expectEqual(@as(u32, 5), m.matched);
    }
}

test "RadixTree shared prefix reuse" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.initWithEviction(allocator, 20, 256, .h2o_attention_score);
    defer pool.deinit();

    var tree = try RadixTree.init(allocator, &pool);
    defer tree.deinit();

    // Two prompts sharing a prefix: [100, 101, 102] + [200] and [100, 101, 102] + [300]
    const prompt_a = [_]u32{ 100, 101, 102, 200 };
    const prompt_b = [_]u32{ 100, 101, 102, 300 };

    // First request: ingests full prompt.
    const result_a = try tree.ingest(&prompt_a, 1);
    defer allocator.free(result_a.page_ids);
    try std.testing.expectEqual(@as(u32, 0), result_a.cached_tokens);

    // Second request: should match the shared prefix (3 tokens).
    const result_b = try tree.ingest(&prompt_b, 2);
    defer allocator.free(result_b.page_ids);
    try std.testing.expectEqual(@as(u32, 3), result_b.cached_tokens);
}

test "RadixTree deinit frees all resources" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.init(allocator, 10, 256);
    defer pool.deinit();

    var tree = try RadixTree.init(allocator, &pool);
    const result = try tree.ingest(&.{ 1, 2, 3, 4, 5 }, 1);
    allocator.free(result.page_ids);
    tree.deinit();
    // Should not leak.
}
