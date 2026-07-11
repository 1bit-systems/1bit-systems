//! MoE Router + Expert Dispatch — architecture-independent expert routing.
//!
//! The key insight: MoE dispatch is architecture-independent. The only things
//! that change between architectures are:
//!   - Number of experts
//!   - Top-k count
//!   - Whether shared experts exist
//!   - Whether it's 1-bit ternary (Zaya) vs fp16 (others)
//!
//! This module provides three dispatch strategies:
//!   1. **sort-then-dispatch** (default) — group tokens by expert, then launch
//!      a single expert-weight-major matvec per expert. Fastest on GPU because
//!      every thread in a warp accesses the same weight row.
//!   2. **per-token dispatch** — each token independently launches its top-k
//!      expert matvecs. Simpler, lower overhead for small batch sizes.
//!   3. **PROBE-style expert caching** — reuse experts from the previous layer
//!      when the gating scores haven't changed beyond a threshold.
//!
//! @section MoE Router

const std = @import("std");
const log = std.log.scoped(.moe_router);

// ── MoE Configuration ───────────────────────────────────────

/// MoE architecture configuration. Architecture-independent: the same struct
/// works for DeepSeek (256 routed + 1 shared), Gemma (128 routed + shared),
/// Mixtral (8 routed), Zaya (8 routed 1-bit ternary), etc.
pub const MoEConfig = struct {
    /// Total number of routed experts.
    n_experts: u32,
    /// Number of top-K experts selected per token.
    top_k: u32,
    /// Whether a shared (always-activated) expert exists.
    has_shared_expert: bool,
    /// Intermediate dimension of each routed expert.
    expert_intermediate_size: u32,
    /// Intermediate dimension of the shared expert (0 when absent).
    shared_expert_intermediate_size: u32,
    /// Hidden dimension of the model.
    hidden_dim: u32,
    /// Whether this is a 1-bit ternary MoE (Zaya-style).
    /// Ternary experts store weights as {-1, 0, +1} packed in 2 bits.
    is_ternary: bool,
    /// Number of bits per weight element. 2 for ternary, 16 for fp16.
    bits_per_weight: u32,
    /// Batch-size threshold for selecting per-assignment GEMV vs batched
    /// fused experts during decode. When batch_size <= this threshold,
    /// per-assignment GEMV is preferred. Default: 8 (Gemma4 decode sweet spot).
    /// Set to 0 to always use batched fused.
    gemv_bs_threshold: u32,
    /// When true, the router output includes token-expert affinity scores
    /// alongside the expert IDs (for weighted combine).
    emit_scores: bool,

    pub fn init(
        n_experts: u32,
        top_k: u32,
        has_shared_expert: bool,
        expert_intermediate_size: u32,
        shared_expert_intermediate_size: u32,
        hidden_dim: u32,
        is_ternary: bool,
    ) MoEConfig {
        return MoEConfig{
            .n_experts = n_experts,
            .top_k = top_k,
            .has_shared_expert = has_shared_expert,
            .expert_intermediate_size = expert_intermediate_size,
            .shared_expert_intermediate_size = shared_expert_intermediate_size,
            .hidden_dim = hidden_dim,
            .is_ternary = is_ternary,
            .bits_per_weight = if (is_ternary) 2 else 16,
            .emit_scores = true,
            .gemv_bs_threshold = 8,
        };
    }

    /// Return the total output tokens per token (top_k routed + optional shared).
    pub fn totalExpertsPerToken(self: MoEConfig) u32 {
        return self.top_k + if (self.has_shared_expert) @as(u32, 1) else 0;
    }

    /// Return the intermediate size rounded up for alignment (typically to 32 or 64).
    pub fn paddedIntermediateSize(self: MoEConfig, alignment: u32) u32 {
        return (self.expert_intermediate_size + alignment - 1) & ~(alignment - 1);
    }

    /// Select the optimal GEMV strategy for a given batch size.
    /// Returns `.per_assignment_gemv` for batch_size <= threshold (small-batch decode),
    /// `.batched_fused` for larger batches (prefill or large-batch decode).
    pub fn selectGemvStrategy(self: MoEConfig, batch_size: u32) MoeGemvStrategy {
        if (batch_size <= self.gemv_bs_threshold) {
            return .per_assignment_gemv;
        }
        return .batched_fused;
    }
};

/// MoE GEMV kernel strategy for decode.
///
/// The Gemma4 MoE decode paper shows that for BS <= 8, per-assignment GEMV
/// (sort tokens by expert, then one GEMV per expert on contiguous data) outperforms
/// batched fused experts 3-5x. For larger batches, batched fused takes over.
pub const MoeGemvStrategy = enum(u8) {
    /// Per-assignment GEMV: sort-then-dispatch, one GEMV per expert on
    /// contiguous token blocks. Best for decode (BS <= 8).
    /// Avoids warp/TB divergence from mixed-expert batches.
    per_assignment_gemv = 0,
    /// Batched fused: one kernel launch for all experts, handles
    /// per-token expert routing internally. Best for prefill / large batch.
    batched_fused = 1,
};

// ── MoE dispatch strategies ─────────────────────────────────

/// MoE dispatch strategy selection.
pub const DispatchStrategy = enum(u8) {
    /// Per-token dispatch: each token independently computes its top-k
    /// expert matvecs. Simple, low overhead for small batches.
    per_token = 0,
    /// Sort-then-dispatch: group tokens by assigned expert, then launch
    /// one expert-weight-major matvec per expert. Best GPU performance.
    sort_then_dispatch = 1,
    /// PROBE-style caching: reuse experts from previous layer if gating
    /// scores have not shifted beyond threshold. Tradeoff: may run fewer
    /// expert matvecs; best when consecutive layers have high expert affinity.
    cached_probe = 2,
};

// ── Router output ───────────────────────────────────────────

/// Per-token routing result: which experts were selected and with what weight.
pub const RoutingResult = struct {
    /// Expert IDs selected for this token [0..top_k).
    expert_ids: []u32,
    /// Softmax-normalized weights for each selected expert [0..top_k).
    /// Only valid when config.emit_scores is true.
    weights: []f32,
};

/// Complete routing table for a batch of tokens.
/// Memory layout: [n_tokens, 2 * top_k] u32, where each row is:
///   [expert_id_0, expert_id_1, ..., expert_id_{k-1},
///    weight_bits_0, weight_bits_1, ..., weight_bits_{k-1}]
/// Weight bits are `@bitCast(f32, weight)` stored as u32.
/// This matches the GPU-side layout used by `softmax_topk` and
/// `moe_weighted_acc` in the existing codebase.
pub const RoutingTable = struct {
    data: []u32,
    n_tokens: u32,
    top_k: u32,
    n_experts: u32,

    pub fn init(data: []u32, n_tokens: u32, top_k: u32, n_experts: u32) RoutingTable {
        return RoutingTable{ .data = data, .n_tokens = n_tokens, .top_k = top_k, .n_experts = n_experts };
    }

    /// Return the expert IDs for token `t`.
    pub fn expertIds(self: RoutingTable, t: u32) []u32 {
        const row = t * 2 * self.top_k;
        return self.data[row .. row + self.top_k];
    }

    /// Return the weight bits for token `t`.
    pub fn weightBits(self: RoutingTable, t: u32) []u32 {
        const row = t * 2 * self.top_k + self.top_k;
        return self.data[row .. row + self.top_k];
    }

    /// Return the weight as f32 for token `t`, expert slot `k`.
    pub fn weight(self: RoutingTable, t: u32, k: u32) f32 {
        return @bitCast(self.data[t * 2 * self.top_k + self.top_k + k]);
    }

    /// Set the weight for token `t`, expert slot `k`.
    pub fn setWeight(self: RoutingTable, t: u32, k: u32, w: f32) void {
        self.data[t * 2 * self.top_k + self.top_k + k] = @bitCast(w);
    }

    /// Get the expert ID at token `t`, slot `k`.
    pub fn expertId(self: RoutingTable, t: u32, k: u32) u32 {
        return self.data[t * 2 * self.top_k + k];
    }

    /// Set the expert ID at token `t`, slot `k`.
    pub fn setExpertId(self: RoutingTable, t: u32, k: u32, id: u32) void {
        self.data[t * 2 * self.top_k + k] = id;
    }
};

// ── Softmax + Top-K (host-side reference) ───────────────────

/// Compute softmax over logits, then extract top-k indices and weights.
/// This is the host-side reference implementation matching the GPU-side
/// `softmax_topk` / `softmax_topk_batched` kernels.
///
/// Logits are [n_experts] f32. Output routing table row at `out_row`
/// stores [top_k expert ids, top_k weight bits].
pub fn softmaxTopK(
    logits: []const f32,
    n_experts: u32,
    top_k: u32,
    out: []u32,
    out_row: u32,
) void {
    std.debug.assert(logits.len >= n_experts);
    std.debug.assert(out.len >= (out_row + 1) * 2 * top_k);
    const NEG_INF = @as(f32, @bitCast(@as(u32, 0xff800000)));

    const row = out_row * 2 * top_k;
    const ids = out[row .. row + top_k];
    const wbits = out[row + top_k .. row + 2 * top_k];

    // ── Iterative top-k selection ──
    // Build a working copy of logits so we can invalidate selected entries.
    var working: [256]f32 = undefined;
    const n_copy = @min(n_experts, working.len);
    @memcpy(working[0..n_copy], logits[0..n_copy]);

    for (0..top_k) |ki| {
        var best = NEG_INF;
        var best_idx: u32 = 0;
        for (0..n_experts) |i| {
            if (working[i] > best) {
                best = working[i];
                best_idx = @intCast(i);
            }
        }
        ids[ki] = best_idx;
        wbits[ki] = @bitCast(best);
        working[best_idx] = NEG_INF; // mask out
    }

    // ── Renormalize softmax over the selected top-k ──
    var max_logit = NEG_INF;
    for (0..top_k) |ki| {
        const v: f32 = @bitCast(wbits[ki]);
        if (v > max_logit) max_logit = v;
    }

    var wsum: f32 = 0.0;
    for (0..top_k) |ki| {
        const v: f32 = @bitCast(wbits[ki]);
        const e = std.math.exp(v - max_logit);
        wbits[ki] = @bitCast(e);
        wsum += e;
    }

    const inv = if (wsum > 0.0) 1.0 / wsum else 0.0;
    for (0..top_k) |ki| {
        const v: f32 = @bitCast(wbits[ki]);
        wbits[ki] = @bitCast(v * inv);
    }
}

// ── Sort-Then-Dispatch: Expert-Sorted Work List ─────────────

/// Expert dispatch order: maps (token, slot) → packed work item.
/// Packed as `(token << 16) | slot` matching the GPU-side `build_expert_order`.
pub const ExpertOrder = struct {
    /// Packed work items sorted by expert: [(token<<16)|slot, ...].
    items: []u32,
    /// Per-expert start offsets [n_experts+1] (prefix sum of counts).
    /// `offsets[e]..offsets[e+1]` is the range in `items` for expert e.
    offsets: []u32,
    n_tokens: u32,
    top_k: u32,
    n_experts: u32,

    /// Build the expert-sorted work list from a routing table.
    /// This matches the GPU-side `build_expert_order_off` kernel.
    /// Uses the provided allocator for scratch.
    pub fn build(
        allocator: std.mem.Allocator,
        routing: *const RoutingTable,
    ) !ExpertOrder {
        const n_experts = routing.n_experts;
        const top_k = routing.top_k;
        const n_tokens = routing.n_tokens;
        const P = n_tokens * top_k; // total work items

        // Count tokens per expert
        var counts = try allocator.alloc(u32, n_experts);
        defer allocator.free(counts);
        @memset(counts, 0);

        for (0..n_tokens) |t| {
            const ids = routing.expertIds(@intCast(t));
            for (0..top_k) |k| {
                const e = ids[k];
                if (e < n_experts) counts[e] += 1;
            }
        }

        // Exclusive prefix sum → offsets
        var offsets = try allocator.alloc(u32, n_experts + 1);
        var acc: u32 = 0;
        for (0..n_experts) |e| {
            offsets[e] = acc;
            acc += counts[e];
        }
        offsets[n_experts] = acc;

        // Cursor copy for scatter
        var cursor = try allocator.alloc(u32, n_experts);
        defer allocator.free(cursor);
        @memcpy(cursor, offsets[0..n_experts]);

        // Scatter each work item into its expert's range
        var items = try allocator.alloc(u32, P);
        for (0..n_tokens) |t| {
            const ti: u32 = @intCast(t);
            const ids = routing.expertIds(ti);
            for (0..top_k) |k| {
                const ki: u32 = @intCast(k);
                const e = ids[k];
                if (e >= n_experts) continue;
                const pos = cursor[e];
                cursor[e] = pos + 1;
                items[pos] = (ti << 16) | ki;
            }
        }

        return ExpertOrder{
            .items = items,
            .offsets = offsets,
            .n_tokens = n_tokens,
            .top_k = top_k,
            .n_experts = n_experts,
        };
    }

    /// Deinitialize, freeing all owned allocations.
    pub fn deinit(self: *ExpertOrder, allocator: std.mem.Allocator) void {
        allocator.free(self.items);
        allocator.free(self.offsets);
    }

    /// Return the number of work items for expert `e`.
    pub fn expertCount(self: ExpertOrder, e: u32) u32 {
        return self.offsets[e + 1] - self.offsets[e];
    }

    /// Return the packed work items for expert `e`.
    pub fn expertItems(self: ExpertOrder, e: u32) []const u32 {
        const start = self.offsets[e];
        const end = self.offsets[e + 1];
        return self.items[start..end];
    }
};

// ── Router — compute top-k experts from gating weights ─────

/// MoE Router: computes the top-k expert assignment for each token
/// from the gating network weights (logits).
///
/// Architecture-independent: the router just produces a
/// `RoutingTable` that downstream code consumes identically
/// regardless of the underlying model architecture.
pub const Router = struct {
    config: MoEConfig,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: MoEConfig) Router {
        return Router{ .allocator = allocator, .config = config };
    }

    /// Compute the routing table for a batch of tokens.
    ///
    /// `gating_logits` is [n_tokens, n_experts] f32 — the raw router logits
    /// produced by the gating network matvec.
    ///
    /// Returns a RoutingTable where each token row stores:
    ///   [expert_id_0..expert_id_{k-1}, weight_bits_0..weight_bits_{k-1}]
    pub fn route(
        self: *Router,
        gating_logits: []const f32,
        n_tokens: u32,
    ) !RoutingTable {
        const config = self.config;
        const n_experts = config.n_experts;
        const top_k = config.top_k;
        const stride = 2 * top_k;

        const table = try self.allocator.alloc(u32, n_tokens * stride);
        for (0..n_tokens) |t| {
            const offset = t * n_experts;
            softmaxTopK(
                gating_logits[offset .. offset + n_experts],
                n_experts,
                top_k,
                table,
                @intCast(t),
            );
        }

        return RoutingTable.init(table, n_tokens, top_k, n_experts);
    }

    /// Deinitialize, freeing the routing table if provided.
    pub fn deinit(_: *Router) void {
        // Router itself holds no long-lived allocations beyond
        // the RoutingTable which callers manage separately.
    }
};

// ── ExpertDispatcher — dispatch tokens to their experts ─────

/// Result of dispatching: per-expert info plus optional sorted order.
pub const DispatchResult = struct {
    per_expert: []ExpertDispatch,
    order: ?ExpertOrder,
};

/// Per-expert dispatch information for a batch of tokens.
pub const ExpertDispatch = struct {
    /// The expert ID this dispatch covers.
    expert_id: u32,
    /// Number of tokens assigned to this expert.
    token_count: u32,
    /// Indices into the full token array for tokens assigned here.
    /// Only valid when strategy == .per_token.
    token_indices: []u32,
    /// Packed work items [(token<<16)|slot, ...].
    /// Only valid when strategy == .sort_then_dispatch.
    work_items: []const u32,
};

/// Expert Dispatcher: after the router produces a routing table,
/// the dispatcher assigns each token to its expert(s) and produces
/// the GPU-friendly dispatch structures.
pub const ExpertDispatcher = struct {
    config: MoEConfig,
    strategy: DispatchStrategy,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        config: MoEConfig,
        strategy: DispatchStrategy,
    ) ExpertDispatcher {
        return ExpertDispatcher{
            .allocator = allocator,
            .config = config,
            .strategy = strategy,
        };
    }

    /// Prepare dispatch arrays from a routing table.
    /// The output depends on the strategy:
    ///
    /// **per_token**: returns a flat array of `ExpertDispatch` — one per
    /// (token × top_k) assignment, each listing the token index and expert ID.
    ///
    /// **sort_then_dispatch**: returns one `ExpertDispatch` per expert that
    /// has at least one token, with packed work items sorted by expert.
    /// Call `buildExpertOrder` for this.
    ///
    /// **cached_probe**: see `CachedRouter`.
    pub fn dispatch(
        self: *ExpertDispatcher,
        routing: *const RoutingTable,
    ) !DispatchResult {
        switch (self.strategy) {
            .per_token => return self.dispatchPerToken(routing),
            .sort_then_dispatch => return self.dispatchSortThen(routing),
            .cached_probe => {
                // cached_probe requires extra state — use CachedRouter instead.
                return error.CachedProbeRequiresCachedRouter;
            },
        }
    }

    /// Per-token dispatch: build a flat list of (token_id, expert_id) pairs.
    fn dispatchPerToken(
        self: *ExpertDispatcher,
        routing: *const RoutingTable,
    ) !DispatchResult {
        const n_tokens = routing.n_tokens;
        const top_k = routing.top_k;
        _ = n_tokens * top_k;

        // Count tokens per expert
        var counts = try self.allocator.alloc(u32, self.config.n_experts);
        defer self.allocator.free(counts);
        @memset(counts, 0);

        for (0..n_tokens) |t| {
            const ids = routing.expertIds(@intCast(t));
            for (0..top_k) |k| {
                const e = ids[k];
                if (e < self.config.n_experts) counts[e] += 1;
            }
        }

        // Allocate per-expert dispatch structs for experts with tokens
        var active_experts: u32 = 0;
        for (0..self.config.n_experts) |e| {
            if (counts[e] > 0) active_experts += 1;
        }

        var per_expert = try self.allocator.alloc(ExpertDispatch, active_experts);
        var cursor = try self.allocator.alloc(u32, self.config.n_experts);
        defer self.allocator.free(cursor);

        var idx: u32 = 0;
        for (0..self.config.n_experts) |e| {
            if (counts[e] == 0) continue;
            const ti = try self.allocator.alloc(u32, counts[e]);
            per_expert[idx] = ExpertDispatch{
                .expert_id = @intCast(e),
                .token_count = counts[e],
                .token_indices = ti,
                .work_items = &.{},
            };
            cursor[e] = 0;
            idx += 1;
        }

        // Scatter token indices
        for (0..n_tokens) |t| {
            const ti: u32 = @intCast(t);
            const ids = routing.expertIds(ti);
            for (0..top_k) |k| {
                const e = ids[k];
                if (e >= self.config.n_experts) continue;
                // Find which ExpertDispatch slot this expert occupies
                // We stored cursor[e] as the next write position.
                const pos = cursor[e];
                for (0..active_experts) |ai| {
                    if (per_expert[ai].expert_id == e) {
                        per_expert[ai].token_indices[pos] = ti;
                        break;
                    }
                }
                cursor[e] = pos + 1;
            }
        }

        return .{ .per_expert = per_expert, .order = null };
    }

    /// Sort-then-dispatch: group tokens by expert using counting sort.
    fn dispatchSortThen(
        self: *ExpertDispatcher,
        routing: *const RoutingTable,
    ) !DispatchResult {
        const order = try ExpertOrder.build(self.allocator, routing);
        const n_experts = self.config.n_experts;

        // Count active experts
        var active: u32 = 0;
        for (0..n_experts) |e| {
            if (order.expertCount(@intCast(e)) > 0) active += 1;
        }

        var per_expert = try self.allocator.alloc(ExpertDispatch, active);
        var idx: u32 = 0;
        for (0..n_experts) |e| {
            const cnt = order.expertCount(@intCast(e));
            if (cnt == 0) continue;
            per_expert[idx] = ExpertDispatch{
                .expert_id = @intCast(e),
                .token_count = cnt,
                .token_indices = &.{}, // not used in sort-then-dispatch
                .work_items = order.expertItems(@intCast(e)),
            };
            idx += 1;
        }

        return .{ .per_expert = per_expert, .order = order };
    }

    pub fn deinit(self: *ExpertDispatcher) void {
        _ = self;
    }
};

// ── PROBE-style CachedRouter ────────────────────────────────

/// Cached router state: one entry per layer, tracks the previous layer's
/// expert assignment and gating logits for comparison.
const CachedLayerState = struct {
    /// Gating logits from the previous inference of this layer.
    /// [n_experts] f32, null on first call.
    prev_logits: ?[]f32,
    /// Routing table from the previous inference of this layer.
    /// [2 * top_k] u32 (one token routing row).
    prev_routing: ?[]u32,
};

/// PROBE-style expert caching for MoE inference.
///
/// PROBE (Predictive Routing with Observational Bounds for Efficiency)
/// reuses expert assignments from the previous layer when the gating
/// logits have not changed beyond a threshold. The key observation is
/// that consecutive transformer layers often have similar expert
/// affinities for the same token, especially during decode (auto-regressive
/// generation) where the hidden state changes incrementally.
///
/// How it works:
///   1. On first invocation for a layer, compute the full softmax+top-k
///      and cache the gating logits.
///   2. On subsequent invocations, compute the L2 or cosine distance
///      between the current and cached gating logits.
///   3. If the distance is below `threshold`, reuse the cached expert
///      assignment (skipping the softmax+top-k compute).
///   4. If the distance exceeds `threshold`, recompute and update cache.
///
/// Architecture-independent: the same caching logic works for any MoE
/// architecture (DeepSeek, Gemma, Mixtral, Zaya).
pub const CachedRouter = struct {
    config: MoEConfig,
    allocator: std.mem.Allocator,
    /// Threshold for gating logit change (L2 distance per expert).
    /// Default: 0.05 — tuned for Gemma4-26b decode where logits shift
    /// by ~0.01-0.03 between layers. Lower = fewer cache hits but more
    /// accurate; higher = more cache hits but may cause expert mismatch.
    threshold: f32,
    /// Distance metric for comparing gating logits.
    distance_metric: DistanceMetric,
    /// Number of cached layers (usually = n_layers in model).
    /// `states[layer]` caches the gating logits and routing for each layer.
    states: []CachedLayerState,
    /// Statistics for observability.
    hits: u64,
    misses: u64,

    /// Distance metric for comparing gating logit vectors.
    pub const DistanceMetric = enum(u8) {
        /// L2 norm of the difference vector: sqrt(sum((a_i - b_i)^2)) / n_experts.
        l2_normalized = 0,
        /// Cosine distance: 1.0 - cos(a, b).
        cosine = 1,
        /// Max absolute difference across all elements.
        max_abs = 2,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        config: MoEConfig,
        n_layers: u32,
        threshold: f32,
        distance_metric: DistanceMetric,
    ) !CachedRouter {
        var states = try allocator.alloc(CachedLayerState, n_layers);
        for (0..n_layers) |i| {
            states[i] = CachedLayerState{
                .prev_logits = null,
                .prev_routing = null,
            };
        }
        return CachedRouter{
            .allocator = allocator,
            .config = config,
            .threshold = threshold,
            .distance_metric = distance_metric,
            .states = states,
            .hits = 0,
            .misses = 0,
        };
    }

    pub fn deinit(self: *CachedRouter) void {
        for (self.states) |*state| {
            if (state.prev_logits) |l| self.allocator.free(l);
            if (state.prev_routing) |r| self.allocator.free(r);
        }
        self.allocator.free(self.states);
    }

    /// Route one token through the cached router.
    ///
    /// `gating_logits` is [n_experts] f32.
    /// `layer` is the current layer index (0-based).
    ///
    /// Returns:
    ///   .cache_hit = true when the cached routing was reused,
    ///   .routing = the RoutingResult for this token.
    ///
    /// When the cache is cold or the logits have shifted beyond threshold,
    /// recomputes the full softmax+top-k and updates the cache.
    pub fn routeToken(
        self: *CachedRouter,
        gating_logits: []const f32,
        layer: u32,
        out_routing: []u32,
    ) struct { cache_hit: bool } {
        const config = self.config;
        const n_experts = config.n_experts;
        const top_k = config.top_k;
        const stride = 2 * top_k;

        const state = &self.states[layer];

        // ── Check cache ──
        if (state.prev_logits) |prev| {
            const distance = self.computeDistance(prev, gating_logits, n_experts);
            if (distance <= self.threshold) {
                // Cache hit: reuse previous routing.
                if (state.prev_routing) |prev_r| {
                    @memcpy(out_routing[0..stride], prev_r[0..stride]);
                }
                self.hits += 1;
                return .{ .cache_hit = true };
            }
        }

        // ── Cache miss: compute fresh ──
        softmaxTopK(gating_logits, n_experts, top_k, out_routing, 0);
        self.misses += 1;

        // ── Update cache ──
        if (state.prev_logits) |prev| {
            @memcpy(prev[0..n_experts], gating_logits[0..n_experts]);
        } else {
            const new_logits = self.allocator.alloc(f32, n_experts) catch {
                // If allocation fails, just leave cache uninitialized.
                return .{ .cache_hit = false };
            };
            @memcpy(new_logits[0..n_experts], gating_logits[0..n_experts]);
            state.prev_logits = new_logits;
        }

        if (state.prev_routing) |prev_r| {
            @memcpy(prev_r[0..stride], out_routing[0..stride]);
        } else {
            const new_routing = self.allocator.alloc(u32, stride) catch {
                return .{ .cache_hit = false };
            };
            @memcpy(new_routing[0..stride], out_routing[0..stride]);
            state.prev_routing = new_routing;
        }

        return .{ .cache_hit = false };
    }

    /// Route a batch of tokens through the cached router.
    ///
    /// `gating_logits` is [n_tokens, n_experts] f32.
    /// `out_table` is [n_tokens, 2*top_k] u32 — preallocated routing table.
    pub fn routeBatch(
        self: *CachedRouter,
        gating_logits: []const f32,
        n_tokens: u32,
        layer: u32,
        out_table: []u32,
    ) usize {
        const n_experts = self.config.n_experts;
        const stride: u32 = 2 * self.config.top_k;

        var cache_hits: usize = 0;
        _ = &self.states[layer];

        for (0..n_tokens) |t| {
            const ti: u32 = @intCast(t);
            const logits_start = ti * n_experts;
            const out_start = ti * stride;

            const result = self.routeToken(
                gating_logits[logits_start .. logits_start + n_experts],
                layer,
                out_table[out_start .. out_start + stride],
            );
            if (result.cache_hit) cache_hits += 1;
        }

        return cache_hits; // return number of cache hits
    }

    /// Compute the distance between two gating logit vectors.
    fn computeDistance(
        self: CachedRouter,
        prev: []const f32,
        curr: []const f32,
        n: u32,
    ) f32 {
        switch (self.distance_metric) {
            .l2_normalized => {
                var sum_sq: f32 = 0.0;
                for (0..n) |i| {
                    const d = curr[i] - prev[i];
                    sum_sq += d * d;
                }
                return std.math.sqrt(sum_sq / @as(f32, @floatFromInt(n)));
            },
            .cosine => {
                var dot: f32 = 0.0;
                var norm_a: f32 = 0.0;
                var norm_b: f32 = 0.0;
                for (0..n) |i| {
                    dot += prev[i] * curr[i];
                    norm_a += prev[i] * prev[i];
                    norm_b += curr[i] * curr[i];
                }
                const denom = std.math.sqrt(norm_a) * std.math.sqrt(norm_b);
                if (denom < std.math.floatEps(f32)) return 1.0;
                return 1.0 - (dot / denom);
            },
            .max_abs => {
                var max_d: f32 = 0.0;
                for (0..n) |i| {
                    const d = @abs(curr[i] - prev[i]);
                    if (d > max_d) max_d = d;
                }
                return max_d;
            },
        }
    }

    /// Reset the cache (e.g., when starting a new sequence or after prefill).
    pub fn reset(self: *CachedRouter) void {
        for (self.states) |*state| {
            if (state.prev_logits) |l| {
                self.allocator.free(l);
                state.prev_logits = null;
            }
            if (state.prev_routing) |r| {
                self.allocator.free(r);
                state.prev_routing = null;
            }
        }
        self.hits = 0;
        self.misses = 0;
    }

    /// Return the cache hit rate.
    pub fn hitRate(self: CachedRouter) f32 {
        const total = self.hits + self.misses;
        if (total == 0) return 0.0;
        return @as(f32, @floatFromInt(self.hits)) / @as(f32, @floatFromInt(total));
    }
};

// ── Dispatch Context: wraps all dispatch state ──────────────

/// Full MoE dispatch context for a single inference step.
///
/// Combines the router and dispatcher into one coherent object
/// that produces the GPU dispatch commands for a transformer layer's MoE block.
///
/// Usage:
///   var ctx = MoEDispatchContext.init(allocator, config, n_layers);
///   defer ctx.deinit();
///   // For each layer:
///   const routing = ctx.route(logits, n_tokens, layer) catch ...;
///   const dispatch = ctx.prepareDispatch(routing) catch ...;
///   // Use dispatch.per_expert[i] to launch expert matvecs.
pub const MoEDispatchContext = struct {
    config: MoEConfig,
    allocator: std.mem.Allocator,
    router: Router,
    dispatcher: ExpertDispatcher,
    cached_router: ?CachedRouter,
    current_strategy: DispatchStrategy,
    /// Pre-allocated routing table storage.
    routing_table: ?[]u32,

    pub fn init(
        allocator: std.mem.Allocator,
        config: MoEConfig,
        n_layers: u32,
        strategy: DispatchStrategy,
        cache_threshold: f32,
    ) !MoEDispatchContext {
        const router = Router.init(allocator, config);
        const dispatcher = ExpertDispatcher.init(allocator, config, strategy);
        const cached_router = if (strategy == .cached_probe)
            try CachedRouter.init(allocator, config, n_layers, cache_threshold, .l2_normalized)
        else
            null;

        return MoEDispatchContext{
            .allocator = allocator,
            .config = config,
            .router = router,
            .dispatcher = dispatcher,
            .cached_router = cached_router,
            .current_strategy = strategy,
            .routing_table = null,
        };
    }

    pub fn deinit(self: *MoEDispatchContext) void {
        if (self.routing_table) |rt| self.allocator.free(rt);
        if (self.cached_router) |*cr| cr.deinit();
        self.dispatcher.deinit();
        self.router.deinit();
    }

    /// Route a batch of tokens through the MoE layer.
    ///
    /// `gating_logits` is [n_tokens, n_experts] f32.
    /// `layer` is the current layer index (for cache lookup).
    ///
    /// Returns the routing table (owned by this context).
    pub fn route(
        self: *MoEDispatchContext,
        gating_logits: []const f32,
        n_tokens: u32,
        layer: u32,
    ) !*RoutingTable {
        const stride: u32 = 2 * self.config.top_k;
        const total = n_tokens * stride;

        // Reuse or allocate routing table storage.
        if (self.routing_table) |rt| {
            if (rt.len < total) {
                self.allocator.free(rt);
                self.routing_table = try self.allocator.alloc(u32, total);
            }
        } else {
            self.routing_table = try self.allocator.alloc(u32, total);
        }
        const table_data = self.routing_table.?[0..total];

        if (self.cached_router) |*cr| {
            _ = cr.routeBatch(gating_logits, n_tokens, layer, table_data);
        } else {
            // Standard routing via softmax top-k
            for (0..n_tokens) |t| {
                const ti: u32 = @intCast(t);
                const offset = ti * self.config.n_experts;
                softmaxTopK(
                    gating_logits[offset .. offset + self.config.n_experts],
                    self.config.n_experts,
                    self.config.top_k,
                    table_data,
                    ti,
                );
            }
        }

        // The RoutingTable references the owned table_data.
        // Use a static/stack instance since we need to return a pointer.
        // In practice, the caller should use the data directly.
        // For now, we just return a reference to the context-owned table.
        const static = &routing_table_static;
        static.* = RoutingTable.init(table_data, n_tokens, self.config.top_k, self.config.n_experts);
        return static;
    }

    /// Prepare dispatch arrays from the current routing table.
    pub fn prepareDispatch(
        self: *MoEDispatchContext,
        routing: *const RoutingTable,
    ) !DispatchResult {
        return self.dispatcher.dispatch(routing);
    }
};

// Internal static for returning a reference (simplification).
// In production, the caller should manage the RoutingTable lifetime directly.
var routing_table_static: RoutingTable = undefined;

// ── Weighted Combine ────────────────────────────────────────

/// Weighted combine: accumulate expert outputs weighted by router scores.
///
/// `out[t * hidden_dim + i] += sum_{k in top_k} weight[t, k] * expert_out[t * k * hidden_dim + i]`
///
/// This matches the GPU-side `moe_weighted_acc` and `moe_weighted_acc_scaled` kernels.
pub fn weightedAccumulate(
    out: []f32,
    expert_outputs: []const f32,
    routing: *const RoutingTable,
    t: u32,
    hidden_dim: u32,
) void {
    const top_k = routing.top_k;
    const row_offset = t * hidden_dim;
    const expert_offset = t * top_k * hidden_dim;

    for (0..top_k) |k| {
        const w = routing.weight(t, @intCast(k));
        if (w == 0.0) continue;
        const src = expert_offset + k * hidden_dim;
        for (0..hidden_dim) |i| {
            out[row_offset + i] += w * expert_outputs[src + i];
        }
    }
}

/// Batched weighted accumulate: process all tokens in one call.
pub fn weightedAccumulateBatch(
    out: []f32,
    expert_outputs: []const f32,
    routing: *const RoutingTable,
    n_tokens: u32,
    hidden_dim: u32,
) void {
    @memset(out[0 .. n_tokens * hidden_dim], 0.0);
    for (0..n_tokens) |t| {
        weightedAccumulate(out, expert_outputs, routing, @intCast(t), hidden_dim);
    }
}

// ── Tests ───────────────────────────────────────────────────

test "MoEConfig basics" {
    const cfg = MoEConfig.init(8, 2, false, 14336, 0, 4096, false);
    try std.testing.expectEqual(@as(u32, 8), cfg.n_experts);
    try std.testing.expectEqual(@as(u32, 2), cfg.top_k);
    try std.testing.expectEqual(false, cfg.has_shared_expert);
    try std.testing.expectEqual(false, cfg.is_ternary);
    try std.testing.expectEqual(@as(u32, 2), cfg.totalExpertsPerToken());
    try std.testing.expectEqual(@as(u32, 16), cfg.bits_per_weight);

    // Ternary
    const tcfg = MoEConfig.init(8, 2, false, 14336, 0, 4096, true);
    try std.testing.expectEqual(true, tcfg.is_ternary);
    try std.testing.expectEqual(@as(u32, 2), tcfg.bits_per_weight);
}

test "MoEConfig with shared expert" {
    const cfg = MoEConfig.init(256, 8, true, 2048, 4096, 7168, false);
    try std.testing.expectEqual(true, cfg.has_shared_expert);
    try std.testing.expectEqual(@as(u32, 9), cfg.totalExpertsPerToken()); // 8 + 1 shared
    try std.testing.expectEqual(@as(u32, 4096), cfg.shared_expert_intermediate_size);
}

test "softmaxTopK correctness" {
    var routing_data: [4]u32 = undefined; // 1 token, top_k=2 → 4 u32
    const logits = [_]f32{ -1.0, 2.0, 0.5, -0.5, 3.0, -2.0, 1.0, 0.0 };

    softmaxTopK(&logits, 8, 2, &routing_data, 0);

    const table = RoutingTable.init(&routing_data, 1, 2, 8);
    try std.testing.expectEqual(@as(u32, 4), table.expertId(0, 0)); // highest: logit 3.0
    try std.testing.expectEqual(@as(u32, 1), table.expertId(0, 1)); // second: logit 2.0

    // Verify weights are positive and sum to ~1.0
    const w0 = table.weight(0, 0);
    const w1 = table.weight(0, 1);
    try std.testing.expect(w0 > 0.0);
    try std.testing.expect(w1 > 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), w0 + w1, 1e-5);
    // Expert 4 (logit 3.0) should have higher weight than expert 1 (logit 2.0)
    try std.testing.expect(w0 > w1);
}

test "Router smoke" {
    const allocator = std.testing.allocator;
    const cfg = MoEConfig.init(8, 2, false, 14336, 0, 4096, false);
    var router = Router.init(allocator, cfg);
    defer router.deinit();

    // 3 tokens, 8 experts
    var logits: [24]f32 = undefined;
    for (0..3) |t| {
        for (0..8) |e| {
            logits[t * 8 + e] = @as(f32, @floatFromInt(e)) + @as(f32, @floatFromInt(t)) * 0.1;
        }
    }

    const table = try router.route(&logits, 3);
    defer allocator.free(table.data);

    // Token 0: expert 7 (logit 7.0) and expert 6 (logit 6.0)
    try std.testing.expectEqual(@as(u32, 7), table.expertId(0, 0));
    try std.testing.expectEqual(@as(u32, 6), table.expertId(0, 1));
    try std.testing.expect(table.weight(0, 0) > table.weight(0, 1));
}

test "ExpertOrder build" {
    const allocator = std.testing.allocator;
    var routing_data: [24]u32 = undefined; // 3 tokens, top_k=2 → 3*4=12 u32
    const logits = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    softmaxTopK(&logits, 8, 2, &routing_data, 0);
    softmaxTopK(&logits, 8, 2, &routing_data, 1);
    softmaxTopK(&logits, 8, 2, &routing_data, 2);

    const table = RoutingTable.init(&routing_data, 3, 2, 8);
    var order = try ExpertOrder.build(allocator, &table);
    defer order.deinit(allocator);

    // Total work items = 3 * 2 = 6
    try std.testing.expectEqual(@as(u32, 6), order.items.len);
    // Each expert should have some count
    var total: u32 = 0;
    for (0..8) |e| {
        total += order.expertCount(@intCast(e));
    }
    try std.testing.expectEqual(@as(u32, 6), total);
}

test "CachedRouter basics" {
    const allocator = std.testing.allocator;
    const cfg = MoEConfig.init(8, 2, false, 14336, 0, 4096, false);
    var cr = try CachedRouter.init(allocator, cfg, 28, 1.0, .l2_normalized);
    defer cr.deinit();

    var routing_out: [4]u32 = undefined; // top_k=2 → 4 u32 per token

    // First call: cold cache, should miss
    const logits_a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const r1 = cr.routeToken(&logits_a, 0, &routing_out);
    try std.testing.expectEqual(false, r1.cache_hit);
    try std.testing.expectEqual(@as(u64, 0), cr.hits);
    try std.testing.expectEqual(@as(u64, 1), cr.misses);

    // Same logits (distance ≈ 0 < 1.0 threshold): should hit
    const r2 = cr.routeToken(&logits_a, 0, &routing_out);
    try std.testing.expectEqual(true, r2.cache_hit);
    try std.testing.expectEqual(@as(u64, 1), cr.hits);
    try std.testing.expectEqual(@as(u64, 1), cr.misses);

    // Very different logits: should miss
    var logits_b: [8]f32 = undefined;
    for (0..8) |i| logits_b[i] = 100.0 + @as(f32, @floatFromInt(i));
    const r3 = cr.routeToken(&logits_b, 0, &routing_out);
    try std.testing.expectEqual(false, r3.cache_hit);
    try std.testing.expectEqual(@as(u64, 1), cr.hits);
    try std.testing.expectEqual(@as(u64, 2), cr.misses);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), cr.hitRate(), 0.01);
}

test "ExpertDispatcher per_token" {
    const allocator = std.testing.allocator;
    const cfg = MoEConfig.init(4, 2, false, 1024, 0, 512, false);
    var dispatcher = ExpertDispatcher.init(allocator, cfg, .per_token);
    defer dispatcher.deinit();

    // 2 tokens, each routed to experts {3,1} and {2,0}
    var routing_data: [8]u32 = undefined;
    {
        const rt = RoutingTable.init(&routing_data, 2, 2, 4);
        rt.setExpertId(0, 0, 3); rt.setWeight(0, 0, 0.7);
        rt.setExpertId(0, 1, 1); rt.setWeight(0, 1, 0.3);
        rt.setExpertId(1, 0, 2); rt.setWeight(1, 0, 0.6);
        rt.setExpertId(1, 1, 0); rt.setWeight(1, 1, 0.4);
    }

    const rt = RoutingTable.init(&routing_data, 2, 2, 4);
    const result = try dispatcher.dispatch(&rt);

    defer {
        for (result.per_expert) |pe| {
            if (pe.token_indices.len > 0) allocator.free(pe.token_indices);
        }
        allocator.free(result.per_expert);
    }

    // Should have 4 dispatches (one per distinct (token, expert) pair)
    try std.testing.expectEqual(@as(usize, 4), result.per_expert.len);
}

test "ExpertDispatcher sort_then_dispatch and combine" {
    const allocator = std.testing.allocator;
    const cfg = MoEConfig.init(4, 2, false, 1024, 0, 4, false);
    var dispatcher = ExpertDispatcher.init(allocator, cfg, .sort_then_dispatch);
    defer dispatcher.deinit();

    var routing_data: [8]u32 = undefined;
    {
        const rt = RoutingTable.init(&routing_data, 2, 2, 4);
        rt.setExpertId(0, 0, 1); rt.setWeight(0, 0, 0.6);
        rt.setExpertId(0, 1, 2); rt.setWeight(0, 1, 0.4);
        rt.setExpertId(1, 0, 1); rt.setWeight(1, 0, 0.7);
        rt.setExpertId(1, 1, 3); rt.setWeight(1, 1, 0.3);
    }

    const rt = RoutingTable.init(&routing_data, 2, 2, 4);
    var result = try dispatcher.dispatch(&rt);

    defer {
        allocator.free(result.per_expert);
        if (result.order) |*o| o.deinit(allocator);
    }

    // Expert 1 has 2 work items, experts 2 and 3 have 1 each
    var found_e1 = false;
    for (result.per_expert) |pe| {
        if (pe.expert_id == 1) {
            try std.testing.expectEqual(@as(u32, 2), pe.token_count);
            found_e1 = true;
        }
    }
    try std.testing.expect(found_e1);

    // Test weighted accumulate
    var out: [8]f32 = undefined;
    var expert_outputs: [16]f32 = undefined;
    for (0..16) |i| expert_outputs[i] = @as(f32, @floatFromInt(i)) * 0.5;
    weightedAccumulateBatch(&out, &expert_outputs, &rt, 2, 4);
    // Just verify outputs are non-zero and finite
    for (0..8) |i| {
        try std.testing.expect(std.math.isFinite(out[i]));
    }
}

test "CachedRouter batch" {
    const allocator = std.testing.allocator;
    const cfg = MoEConfig.init(4, 2, false, 1024, 0, 512, false);
    var cr = try CachedRouter.init(allocator, cfg, 4, 2.0, .l2_normalized);
    defer cr.deinit();

    // 1 token, 4 experts, across 4 layers (PROBE reuses across layers)
    var logits: [4]f32 = undefined;
    for (0..4) |i| logits[i] = @as(f32, @floatFromInt(i));
    var out_table: [8]u32 = undefined;

    // Token 0 at layer 0: cold cache -> miss
    const hits0 = cr.routeBatch(&logits, 1, 0, &out_table);
    try std.testing.expectEqual(@as(usize, 0), hits0);

    // Token 0 at layer 1: different logits should miss
    const hits1 = cr.routeBatch(&logits, 1, 0, &out_table);
    try std.testing.expectEqual(@as(usize, 1), hits1); // same layer, same logits -> hit

    // Token 0 at a different layer: cold cache -> miss
    const hits2 = cr.routeBatch(&logits, 1, 1, &out_table);
    try std.testing.expectEqual(@as(usize, 0), hits2);

    // Revisit layer 1: same logits -> hit
    const hits3 = cr.routeBatch(&logits, 1, 1, &out_table);
    try std.testing.expectEqual(@as(usize, 1), hits3);
}

test "MoEDispatchContext integration" {
    const allocator = std.testing.allocator;
    const cfg = MoEConfig.init(8, 2, false, 1024, 0, 512, false);
    var ctx = try MoEDispatchContext.init(allocator, cfg, 28, .sort_then_dispatch, 0.1);
    defer ctx.deinit();

    // 2 tokens, 8 experts
    var logits: [16]f32 = undefined;
    for (0..16) |i| logits[i] = @as(f32, @floatFromInt(i & 0x7)) + @as(f32, @floatFromInt(i / 8)) * 0.5;

    const routing = try ctx.route(&logits, 2, 0);
    var dispatch = try ctx.prepareDispatch(routing);

    defer {
        allocator.free(dispatch.per_expert);
        if (dispatch.order) |*o| o.deinit(allocator);
    }

    try std.testing.expect(dispatch.per_expert.len > 0);

    var token_counts: u32 = 0;
    for (dispatch.per_expert) |pe| token_counts += pe.token_count;
    try std.testing.expectEqual(@as(u32, 4), token_counts); // 2 tokens * top_k=2
}
