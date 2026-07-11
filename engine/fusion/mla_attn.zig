//! Multi-Head Latent Attention (MLA) for DeepSeek V2/V3.
//!
//! MLA is the key architectural innovation in DeepSeek models. Instead of
//! projecting Q/K/V to the full head_dim, it uses a compressed KV latent
//! representation (`kv_lora_rank`) with low-rank (LoRA-style) decompression
//! at attention time.
//!
//! ## Dimensions
//!
//! | Symbol               | Description                              | V2 Lite | V3      |
//! |----------------------|------------------------------------------|---------|---------|
//! | `H`                  | hidden_dim                               | 2048    | 7168    |
//! | `NH`                 | n_heads                                  | 16      | 56      |
//! | `NKV`                | n_kv_heads                               | 2       | 8       |
//! | `qk_nope_head_dim`   | Non-RoPE portion of Q/K (std attention)  | 128     | 128     |
//! | `qk_rope_head_dim`   | RoPE portion of Q/K                      | 64      | 64      |
//! | `qk_head_dim`        | Total Q/K head dim (nope + rope)         | 192     | 192     |
//! | `v_head_dim`         | Value head dimension                     | 128     | 128     |
//! | `kv_lora_rank`       | Compressed KV latent rank                | 256     | 512     |
//!
//! ## Weight layout
//!
//! ```
//! q_proj:     [NH × qk_head_dim, H]             — query projection (full)
//! kv_a_proj:  [kv_lora_rank, H]                  — KV compression (shared)
//! k_b_proj:   [NKV × qk_head_dim, kv_lora_rank]  — K decompression (up-proj)
//! v_b_proj:   [NKV × v_head_dim, kv_lora_rank]   — V decompression (up-proj)
//! o_proj:     [H, NH × v_head_dim]                — attention output projection
//! ```
//!
//! ## Absorbed attention (inference optimization)
//!
//! Instead of decompressing KV to full dim then attending, we _absorb_ the
//! K-nope up-projection into Q and only decompress the K- rope portion:
//!
//! ```
//! q_absorbed = q @ K_b_nope_T       → [n_heads × kv_lora_rank]
//! k_rope     = c @ K_b_rope_T       → [n_kv_heads × qk_rope_head_dim] (then apply RoPE)
//! score      = q_absorbed @ c_T     (nope part) + rope(q_rope) @ rope(k_rope)_T
//! output     = softmax(score) @ v   with v = c @ V_b_T
//! ```
//!
//! ## KV cache modes
//!
//! - **flat**: contiguous `[max_seq_len × kv_lora_rank]` buffer, no paging.
//!   Used for prefill or small-batch decode without eviction.
//! - **paged**: page-table-indexed cache with `page_size` tokens per page.
//!   Each page holds `[page_size × kv_lora_rank]` floats. Compatible with
//!   the existing `KvPagePool` eviction infrastructure.
//!
//! ## References
//!
//! - DeepSeek-V2: https://arxiv.org/abs/2405.04434
//! - DeepSeek-V3: https://arxiv.org/abs/2412.19437
//!
//! @section Fused Engine

const std = @import("std");
const math = std.math;
const log = std.log.scoped(.mla_attn);

// ── Configuration ────────────────────────────────────────────

/// MLA-specific configuration beyond the base ArchConfig fields.
///
/// Short field names match the fused engine convention (see arch_configs.zig).
pub const MLAConfig = struct {
    /// Number of query heads.
    NH: u32,
    /// Number of KV heads (GQA groups).
    NKV: u32,
    /// Hidden dimension (input/output size).
    H: u32,

    /// Non-RoPE portion of Q/K head dimension (typically 128).
    qk_nope_head_dim: u32 = 128,
    /// RoPE portion of Q/K head dimension (typically 64).
    qk_rope_head_dim: u32 = 64,
    /// Value head dimension (typically 128).
    v_head_dim: u32 = 128,

    /// Compressed KV latent rank.
    /// V2-Lite: 256, V3: 512, V2: 576.
    kv_lora_rank: u32,

    /// RoPE theta base (default 10000.0 for DeepSeek).
    rope_theta: f32 = 10000.0,

    /// Maximum sequence length (for precomputed RoPE tables).
    max_seq_len: u32,

    /// Attention softmax scaling factor: 1.0 / sqrt(qk_head_dim).
    /// Precomputed at init for speed.
    attn_scale: f32,

    /// Derived: total Q/K head dimension (nope + rope).
    qk_head_dim: u32,
    /// Derived: total Q output dimension (NH × qk_head_dim).
    q_out_dim: u32,
    /// Derived: total K decompressed dimension (NKV × qk_head_dim).
    k_out_dim: u32,
    /// Derived: total V decompressed dimension (NKV × v_head_dim).
    v_out_dim: u32,
    /// Derived: GQA ratio (NH / NKV).
    gqa_ratio: u32,

    pub fn init(NH: u32, NKV: u32, H: u32, kv_lora_rank: u32, max_seq_len: u32, opts: struct {
        qk_nope_head_dim: u32 = 128,
        qk_rope_head_dim: u32 = 64,
        v_head_dim: u32 = 128,
        rope_theta: f32 = 10000.0,
    }) MLAConfig {
        const qk_head_dim = opts.qk_nope_head_dim + opts.qk_rope_head_dim;
        return MLAConfig{
            .NH = NH,
            .NKV = NKV,
            .H = H,
            .qk_nope_head_dim = opts.qk_nope_head_dim,
            .qk_rope_head_dim = opts.qk_rope_head_dim,
            .v_head_dim = opts.v_head_dim,
            .kv_lora_rank = kv_lora_rank,
            .rope_theta = opts.rope_theta,
            .max_seq_len = max_seq_len,
            .attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(qk_head_dim))),
            .qk_head_dim = qk_head_dim,
            .q_out_dim = NH * qk_head_dim,
            .k_out_dim = NKV * qk_head_dim,
            .v_out_dim = NKV * opts.v_head_dim,
            .gqa_ratio = if (NKV > 0) NH / NKV else 1,
        };
    }

    /// Validate that the configuration is internally consistent.
    pub fn validate(self: *const MLAConfig) !void {
        if (self.NH == 0) return error.MLAZeroHeads;
        if (self.NKV == 0) return error.MLAZeroKvHeads;
        if (self.H == 0) return error.MLAZeroHidden;
        if (self.kv_lora_rank == 0) return error.MLAZeroLoraRank;
        if (self.NH % self.NKV != 0) return error.MLAGqaNotDivisible;
        if (self.qk_nope_head_dim == 0 and self.qk_rope_head_dim == 0) return error.MLAZeroQkDim;
        if (self.v_head_dim == 0) return error.MLAZeroVHeadDim;
        if (self.max_seq_len == 0) return error.MLAZeroSeqLen;
    }
};

// ── Weight slice descriptors ─────────────────────────────────

/// Pointers to the raw weight matrices for an MLA layer.
///
/// All slices are owned by the caller (loaded from the Q4NX file or
/// provided by the NPU/CPU backend). The MLA module reads them at
/// attention time — it does not take ownership.
pub const MLAWeights = struct {
    /// Query projection: [NH * qk_head_dim, H] row-major.
    /// Stored [q_out_dim][H].
    q_proj: []const f32,

    /// KV compression: [kv_lora_rank, H] row-major.
    kv_a_proj: []const f32,

    /// K decompression (up-projection): [NKV * qk_head_dim, kv_lora_rank].
    k_b_proj: []const f32,

    /// V decompression (up-projection): [NKV * v_head_dim, kv_lora_rank].
    v_b_proj: []const f32,

    /// Output projection: [H, NH * v_head_dim] row-major.
    o_proj: []const f32,
};

// ── Absorbed weights (precomputed for inference) ─────────────

/// Pre-absorbed weight matrices for fast inference.
///
/// These are computed once at load time by fusing the K-nope up-projection
/// into the query projection. This avoids decompressing K at every step.
pub const MLAAbsorbedWeights = struct {
    /// Absorbed Q weight: [NH * kv_lora_rank, H].
    /// q_absorbed = x @ q_absorbed_weight_T
    q_absorbed: []const f32,

    /// K-RoPE up-projection (needs decompression for RoPE): [NKV * qk_rope_head_dim, kv_lora_rank].
    k_b_rope: []const f32,

    /// V decompression: [NKV * v_head_dim, kv_lora_rank] (same as v_b_proj).
    v_b: []const f32,

    /// Output projection: [H, NH * v_head_dim] (same as o_proj).
    o_proj: []const f32,

    /// Deinit any owned memory.
    pub fn deinit(self: *MLAAbsorbedWeights, allocator: std.mem.Allocator) void {
        allocator.free(self.q_absorbed);
        // k_b_rope, v_b, o_proj are views/subslices of original weights — not freed here.
        self.* = undefined;
    }
};

// ── RoPE table for MLA ───────────────────────────────────────

/// Precomputed sin/cos tables for the RoPE portion of MLA (qk_rope_head_dim).
pub const MLARopeTables = struct {
    sin: []const f32, // [max_seq_len * qk_rope_head_dim]
    cos: []const f32, // [max_seq_len * qk_rope_head_dim]
    qk_rope_head_dim: u32,
    max_seq_len: u32,

    /// Precompute sin/cos tables for the given config.
    pub fn init(allocator: std.mem.Allocator, config: *const MLAConfig) !MLARopeTables {
        const rope_dim = config.qk_rope_head_dim;
        const max_seq_len = config.max_seq_len;
        const total = max_seq_len * rope_dim;
        const sin_table = try allocator.alloc(f32, total);
        errdefer allocator.free(sin_table);
        const cos_table = try allocator.alloc(f32, total);
        errdefer allocator.free(cos_table);

        const theta = config.rope_theta;
        const inv_scale: f32 = @as(f32, @floatFromInt(rope_dim));

        var pos: u32 = 0;
        while (pos < max_seq_len) : (pos += 1) {
            const p = @as(f32, @floatFromInt(pos));
            const base_idx = pos * rope_dim;
            var i: u32 = 0;
            while (i < rope_dim / 2) : (i += 1) {
                const freq = p * std.math.pow(f32, theta, -2.0 * @as(f32, @floatFromInt(i)) / inv_scale);
                const s = @sin(freq);
                const c = @cos(freq);
                sin_table[base_idx + 2 * i] = s;
                sin_table[base_idx + 2 * i + 1] = s;
                cos_table[base_idx + 2 * i] = c;
                cos_table[base_idx + 2 * i + 1] = c;
            }
        }

        return MLARopeTables{
            .sin = sin_table,
            .cos = cos_table,
            .qk_rope_head_dim = rope_dim,
            .max_seq_len = max_seq_len,
        };
    }

    pub fn deinit(self: *MLARopeTables, allocator: std.mem.Allocator) void {
        allocator.free(self.sin);
        allocator.free(self.cos);
        self.* = undefined;
    }

    /// Look up sin/cos for a position, returning slices of length qk_rope_head_dim.
    pub fn get(self: *const MLARopeTables, pos: u32) struct { sin: []const f32, cos: []const f32 } {
        const idx = @min(pos, self.max_seq_len - 1) * self.qk_rope_head_dim;
        const end = idx + self.qk_rope_head_dim;
        return .{
            .sin = self.sin[idx..end],
            .cos = self.cos[idx..end],
        };
    }
};

// ── KV cache interface ───────────────────────────────────────

/// Flat contiguous KV cache: [max_seq_len, kv_lora_rank] row-major.
pub const FlatKvCache = struct {
    data: []f32,
    kv_lora_rank: u32,
    max_seq_len: u32,
    seq_len: u32, // current filled length

    pub fn init(allocator: std.mem.Allocator, kv_lora_rank: u32, max_seq_len: u32) !FlatKvCache {
        const data = try allocator.alloc(f32, @as(usize, @intCast(max_seq_len)) * @as(usize, @intCast(kv_lora_rank)));
        @memset(data, 0);
        return FlatKvCache{ .data = data, .kv_lora_rank = kv_lora_rank, .max_seq_len = max_seq_len, .seq_len = 0 };
    }

    pub fn deinit(self: *FlatKvCache, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        self.* = undefined;
    }

    /// Write one token's compressed KV latent at position `pos`.
    pub fn write(self: *FlatKvCache, pos: u32, latent: []const f32) void {
        if (pos >= self.max_seq_len) return;
        const dst = self.data[@as(usize, @intCast(pos)) * @as(usize, @intCast(self.kv_lora_rank)) ..];
        @memcpy(dst[0..@as(usize, @intCast(self.kv_lora_rank))], latent);
        self.seq_len = @max(self.seq_len, pos + 1);
    }

    /// Read one token's compressed KV latent.
    pub fn read(self: *const FlatKvCache, pos: u32) []const f32 {
        const idx = @min(pos, self.seq_len - 1) * self.kv_lora_rank;
        return self.data[@as(usize, @intCast(idx)) .. @as(usize, @intCast(idx + self.kv_lora_rank))];
    }

    /// Read a batch of latents for a contiguous range of positions.
    pub fn readRange(self: *const FlatKvCache, start: u32, count: u32) []const f32 {
        const s = @min(start, self.seq_len);
        const c = @min(count, self.seq_len - s);
        return self.data[@as(usize, @intCast(s)) * @as(usize, @intCast(self.kv_lora_rank)) .. @as(usize, @intCast(s + c)) * @as(usize, @intCast(self.kv_lora_rank))];
    }

    pub fn reset(self: *FlatKvCache) void {
        self.seq_len = 0;
    }
};

/// Paged KV cache: pages of [page_size × kv_lora_rank] each.
///
/// Compatible with `KvPagePool` from sched/kv_cache.zig — each page
/// stores `page_size` consecutive compressed KV latents.
pub const PagedKvCache = struct {
    /// Flattened page pool: [n_pages * page_size * kv_lora_rank].
    data: []f32,
    kv_lora_rank: u32,
    page_size: u32,
    n_pages: u32,
    /// Page table: [n_tokens] → page_id (managed externally).
    page_table: []const u32,
    page_table_owned: bool,

    pub fn init(allocator: std.mem.Allocator, kv_lora_rank: u32, page_size: u32, n_pages: u32) !PagedKvCache {
        const total = @as(usize, @intCast(n_pages)) * @as(usize, @intCast(page_size)) * @as(usize, @intCast(kv_lora_rank));
        const data = try allocator.alloc(f32, total);
        @memset(data, 0);
        const pt = try allocator.alloc(u32, 0);
        return PagedKvCache{
            .data = data,
            .kv_lora_rank = kv_lora_rank,
            .page_size = page_size,
            .n_pages = n_pages,
            .page_table = pt,
            .page_table_owned = true,
        };
    }

    pub fn deinit(self: *PagedKvCache, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        if (self.page_table_owned) allocator.free(self.page_table);
        self.* = undefined;
    }

    /// Set the page table (borrowed reference — caller must keep alive).
    pub fn setPageTable(self: *PagedKvCache, pt: []const u32) void {
        self.page_table = pt;
        self.page_table_owned = false;
    }

    /// Write a compressed latent at (page_id, offset_within_page).
    pub fn write(self: *PagedKvCache, page_id: u32, offset: u32, latent: []const f32) void {
        const base = @as(usize, @intCast(page_id)) * @as(usize, @intCast(self.page_size)) * @as(usize, @intCast(self.kv_lora_rank));
        const slot = base + @as(usize, @intCast(offset)) * @as(usize, @intCast(self.kv_lora_rank));
        @memcpy(self.data[slot .. slot + @as(usize, @intCast(self.kv_lora_rank))], latent);
    }

    /// Read a compressed latent by page_id and offset.
    pub fn read(self: *const PagedKvCache, page_id: u32, offset: u32) []const f32 {
        const base = @as(usize, @intCast(page_id)) * @as(usize, @intCast(self.page_size)) * @as(usize, @intCast(self.kv_lora_rank));
        const slot = base + @as(usize, @intCast(offset)) * @as(usize, @intCast(self.kv_lora_rank));
        return self.data[slot .. slot + @as(usize, @intCast(self.kv_lora_rank))];
    }

    /// Read a compressed latent for a token using the page table.
    pub fn readByToken(self: *const PagedKvCache, token_idx: u32) []const f32 {
        if (token_idx >= self.page_table.len) {
            return self.data[0..0]; // empty / zero page
        }
        const page_id = self.page_table[@as(usize, @intCast(token_idx))];
        if (page_id >= self.n_pages) {
            return self.data[0..0];
        }
        const offset = token_idx % self.page_size;
        return self.read(page_id, offset);
    }
};

// ── Attention state (per-layer scratch) ──────────────────────

/// Per-layer scratch buffers for MLA computation.
///
/// Owned by the caller; reused across decode steps to minimize allocations.
pub const MLAScratch = struct {
    // Q buffer: [NH * qk_head_dim] flat
    q: []f32,

    // Q split into nope and rope portions
    q_nope: []f32, // [NH * qk_nope_head_dim]
    q_rope: []f32, // [NH * qk_rope_head_dim]

    // Absorbed Q: [NH * kv_lora_rank]
    q_absorbed: []f32,

    // K- rope after decompression and RoPE: [NKV * qk_rope_head_dim]
    k_rope: []f32,

    // V after decompression: [NKV * v_head_dim]
    v: []f32,

    // Output: [NH * v_head_dim] (before output projection)
    attn_out: []f32,

    // Score buffer: [NH * seq_len] (pre-softmax scores)
    scores: []f32,

    // Softmax output buffer: [NH * seq_len]
    softmax_out: []f32,

    pub fn init(allocator: std.mem.Allocator, config: *const MLAConfig, max_seq_len: u32) !MLAScratch {
        return MLAScratch{
            .q = try allocator.alloc(f32, @as(usize, @intCast(config.q_out_dim))),
            .q_nope = try allocator.alloc(f32, @as(usize, @intCast(config.NH)) * @as(usize, @intCast(config.qk_nope_head_dim))),
            .q_rope = try allocator.alloc(f32, @as(usize, @intCast(config.NH)) * @as(usize, @intCast(config.qk_rope_head_dim))),
            .q_absorbed = try allocator.alloc(f32, @as(usize, @intCast(config.NH)) * @as(usize, @intCast(config.kv_lora_rank))),
            .k_rope = try allocator.alloc(f32, @as(usize, @intCast(config.NKV)) * @as(usize, @intCast(config.qk_rope_head_dim))),
            .v = try allocator.alloc(f32, @as(usize, @intCast(config.v_out_dim))),
            .attn_out = try allocator.alloc(f32, @as(usize, @intCast(config.NH)) * @as(usize, @intCast(config.v_head_dim))),
            .scores = try allocator.alloc(f32, @as(usize, @intCast(config.NH)) * @as(usize, @intCast(max_seq_len))),
            .softmax_out = try allocator.alloc(f32, @as(usize, @intCast(config.NH)) * @as(usize, @intCast(max_seq_len))),
        };
    }

    pub fn deinit(self: *MLAScratch, allocator: std.mem.Allocator) void {
        allocator.free(self.q);
        allocator.free(self.q_nope);
        allocator.free(self.q_rope);
        allocator.free(self.q_absorbed);
        allocator.free(self.k_rope);
        allocator.free(self.v);
        allocator.free(self.attn_out);
        allocator.free(self.scores);
        allocator.free(self.softmax_out);
        self.* = undefined;
    }
};

// ── Core MLA attention operations ────────────────────────────

/// Apply RoPE to a (head, rope_dim) slice in-place.
/// `data` is [n_heads * rope_dim], flat row-major.
fn applyRoPE(data: []f32, n_heads: u32, rope_dim: u32, sin: []const f32, cos: []const f32) void {
    var h: u32 = 0;
    while (h < n_heads) : (h += 1) {
        const head_start = @as(usize, @intCast(h)) * @as(usize, @intCast(rope_dim));
        var i: u32 = 0;
        while (i < rope_dim / 2) : (i += 1) {
            const idx = head_start + 2 * i;
            const x0 = data[idx];
            const x1 = data[idx + 1];
            const s = sin[2 * i];
            const c = cos[2 * i];
            data[idx] = x0 * c - x1 * s;
            data[idx + 1] = x0 * s + x1 * c;
        }
    }
}

/// Split Q into nope and rope portions.
/// `q` is [NH * qk_head_dim], outputs are [NH * qk_nope_head_dim] and [NH * qk_rope_head_dim].
fn splitQRope(
    q: []const f32,
    q_nope: []f32,
    q_rope: []f32,
    NH: u32,
    qk_nope_head_dim: u32,
    qk_rope_head_dim: u32,
) void {
    const qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    var h: u32 = 0;
    while (h < NH) : (h += 1) {
        const q_off = @as(usize, @intCast(h)) * @as(usize, @intCast(qk_head_dim));
        const nope_off = @as(usize, @intCast(h)) * @as(usize, @intCast(qk_nope_head_dim));
        const rope_off = @as(usize, @intCast(h)) * @as(usize, @intCast(qk_rope_head_dim));

        // q_nope = q[0..qk_nope_head_dim]
        @memcpy(q_nope[nope_off..][0..@as(usize, @intCast(qk_nope_head_dim))], q[q_off..][0..@as(usize, @intCast(qk_nope_head_dim))]);
        // q_rope = q[qk_nope_head_dim..]
        @memcpy(q_rope[rope_off..][0..@as(usize, @intCast(qk_rope_head_dim))], q[q_off + @as(usize, @intCast(qk_nope_head_dim))..][0..@as(usize, @intCast(qk_rope_head_dim))]);
    }
}

/// Compute the absorbed Q: q_absorbed[h] = q_nope[h] @ K_b_nope^T
/// Where K_b_nope extracts the first qk_nope_head_dim rows of each KV head's
/// k_b_proj block.
///
/// `q_nope`: [NH * qk_nope_head_dim]
/// `q_absorbed`: [NH * kv_lora_rank]
/// `k_b_proj`: [NKV * qk_head_dim, kv_lora_rank] — full k_b_proj
fn computeAbsorbedQ(
    q_nope: []const f32,
    q_absorbed: []f32,
    k_b_proj: []const f32,
    NH: u32,
    NKV: u32,
    qk_nope_head_dim: u32,
    kv_lora_rank: u32,
) void {
    const gqa_ratio = NH / NKV;

    var h: u32 = 0;
    while (h < NH) : (h += 1) {
        const kv_head = h / gqa_ratio;
        const q_nope_off = @as(usize, @intCast(h)) * @as(usize, @intCast(qk_nope_head_dim));
        const out_off = @as(usize, @intCast(h)) * @as(usize, @intCast(kv_lora_rank));

        // k_b_nope for this KV head: [qk_nope_head_dim, kv_lora_rank]
        const k_b_row_start = @as(usize, @intCast(kv_head)) * @as(usize, @intCast(qk_nope_head_dim + 64)); // qk_head_dim
        // Actually we need the first qk_nope_head_dim rows of this KV head's block
        const k_b_base = k_b_row_start * @as(usize, @intCast(kv_lora_rank));

        // q_absorbed[h] = q_nope[h] @ k_b_nope^T  (where k_b_nope is [qk_nope_head_dim, kv_lora_rank])
        // = sum over i: q_nope[h][i] * k_b_nope[i][:]
        @memset(q_absorbed[out_off .. out_off + @as(usize, @intCast(kv_lora_rank))], 0);
        var i: u32 = 0;
        while (i < qk_nope_head_dim) : (i += 1) {
            const q_val = q_nope[q_nope_off + @as(usize, @intCast(i))];
            if (q_val == 0) continue;
            const k_b_row = k_b_proj[k_b_base + @as(usize, @intCast(i)) * @as(usize, @intCast(kv_lora_rank)) ..];
            var j: u32 = 0;
            while (j < kv_lora_rank) : (j += 1) {
                q_absorbed[out_off + @as(usize, @intCast(j))] += q_val * k_b_row[@as(usize, @intCast(j))];
            }
        }
    }
}

/// Decompress K- rope portion for a single KV latent.
/// `c`: [kv_lora_rank] — compressed latent
/// `k_rope_out`: [NKV * qk_rope_head_dim] — decompressed K- rope
/// `k_b_proj`: [NKV * qk_head_dim, kv_lora_rank]
fn decompressKRope(
    c: []const f32,
    k_rope_out: []f32,
    k_b_proj: []const f32,
    NKV: u32,
    qk_nope_head_dim: u32,
    qk_rope_head_dim: u32,
    kv_lora_rank: u32,
) void {
    const qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    var kv: u32 = 0;
    while (kv < NKV) : (kv += 1) {
        const rope_start = @as(usize, @intCast(kv)) * @as(usize, @intCast(qk_rope_head_dim));
        // K-b rope portion starts at qk_nope_head_dim within this KV head's block
        const k_b_rope_row_start = @as(usize, @intCast(kv)) * @as(usize, @intCast(qk_head_dim)) + @as(usize, @intCast(qk_nope_head_dim));
        const k_b_base = k_b_rope_row_start * @as(usize, @intCast(kv_lora_rank));

        @memset(k_rope_out[rope_start .. rope_start + @as(usize, @intCast(qk_rope_head_dim))], 0);
        var i: u32 = 0;
        while (i < qk_rope_head_dim) : (i += 1) {
            const k_b_row = k_b_proj[k_b_base + @as(usize, @intCast(i)) * @as(usize, @intCast(kv_lora_rank)) ..];
            var j: u32 = 0;
            while (j < kv_lora_rank) : (j += 1) {
                k_rope_out[rope_start + @as(usize, @intCast(i))] += k_b_row[@as(usize, @intCast(j))] * c[@as(usize, @intCast(j))];
            }
        }
    }
}

/// Decompress V from compressed latent.
/// `c`: [kv_lora_rank] — compressed latent
/// `v_out`: [NKV * v_head_dim] — decompressed V
/// `v_b_proj`: [NKV * v_head_dim, kv_lora_rank]
fn decompressV(
    c: []const f32,
    v_out: []f32,
    v_b_proj: []const f32,
    NKV: u32,
    v_head_dim: u32,
    kv_lora_rank: u32,
) void {
    var kv: u32 = 0;
    while (kv < NKV) : (kv += 1) {
        const v_off = @as(usize, @intCast(kv)) * @as(usize, @intCast(v_head_dim));
        const k_b_base = @as(usize, @intCast(kv)) * @as(usize, @intCast(v_head_dim)) * @as(usize, @intCast(kv_lora_rank));

        @memset(v_out[v_off .. v_off + @as(usize, @intCast(v_head_dim))], 0);
        var i: u32 = 0;
        while (i < v_head_dim) : (i += 1) {
            const k_b_row = v_b_proj[k_b_base + @as(usize, @intCast(i)) * @as(usize, @intCast(kv_lora_rank)) ..];
            var j: u32 = 0;
            while (j < kv_lora_rank) : (j += 1) {
                v_out[v_off + @as(usize, @intCast(i))] += k_b_row[@as(usize, @intCast(j))] * c[@as(usize, @intCast(j))];
            }
        }
    }
}

// ── Main MLA attention functions ─────────────────────────────

/// Compute attention scores (pre-softmax) for the absorbed formulation.
///
/// score[h][t] = q_absorbed[h] @ c[t] + rope(q_rope[h]) @ rope(k_rope[t][kv_h])
///
/// where kv_h = h / gqa_ratio maps each Q head to its KV head group.
///
/// Output: scores[NH * seq_len], row-major (h, t).
fn computeScores(
    scores: []f32,
    q_absorbed: []const f32,
    q_rope_rope: []const f32,
    kv_cache: *const FlatKvCache,
    k_rope_tables: []const f32, // [seq_len * NKV * qk_rope_head_dim] pre-roped K cache
    NH: u32,
    NKV: u32,
    seq_len: u32,
    qk_rope_head_dim: u32,
    kv_lora_rank: u32,
    attn_scale: f32,
) void {
    const gqa_ratio = NH / NKV;
    var h: u32 = 0;
    while (h < NH) : (h += 1) {
        const kv_h = h / gqa_ratio;
        const score_off = @as(usize, @intCast(h)) * @as(usize, @intCast(seq_len));
        const q_abs_off = @as(usize, @intCast(h)) * @as(usize, @intCast(kv_lora_rank));

        var t: u32 = 0;
        while (t < seq_len) : (t += 1) {
            const c_t = kv_cache.read(t);
            var dot: f32 = 0;

            // Nope part: q_absorbed @ c[t]
            var j: u32 = 0;
            while (j < kv_lora_rank) : (j += 1) {
                dot += q_absorbed[q_abs_off + @as(usize, @intCast(j))] * c_t[@as(usize, @intCast(j))];
            }

            // RoPE part: rope(q_rope[h]) @ rope(k_rope[t][kv_h])
            const q_rope_off = @as(usize, @intCast(h)) * @as(usize, @intCast(qk_rope_head_dim));
            const k_rope_off = @as(usize, @intCast(t)) * @as(usize, @intCast(NKV)) * @as(usize, @intCast(qk_rope_head_dim)) + @as(usize, @intCast(kv_h)) * @as(usize, @intCast(qk_rope_head_dim));

            var ri: u32 = 0;
            while (ri < qk_rope_head_dim) : (ri += 1) {
                dot += q_rope_rope[q_rope_off + @as(usize, @intCast(ri))] * k_rope_tables[k_rope_off + @as(usize, @intCast(ri))];
            }

            scores[score_off + @as(usize, @intCast(t))] = dot * attn_scale;
        }
    }
}

/// Compute attention scores for paged KV cache.
fn computeScoresPaged(
    scores: []f32,
    q_absorbed: []const f32,
    q_rope_rope: []const f32,
    kv_cache: *const PagedKvCache,
    k_rope_tables: []const f32, // [n_tokens * NKV * qk_rope_head_dim] pre-roped
    NH: u32,
    NKV: u32,
    n_tokens: u32,
    qk_rope_head_dim: u32,
    kv_lora_rank: u32,
    page_table: []const u32,
    attn_scale: f32,
) void {
    const gqa_ratio = NH / NKV;

    var h: u32 = 0;
    while (h < NH) : (h += 1) {
        const kv_h = h / gqa_ratio;
        const score_off = @as(usize, @intCast(h)) * @as(usize, @intCast(n_tokens));
        const q_abs_off = @as(usize, @intCast(h)) * @as(usize, @intCast(kv_lora_rank));

        var t: u32 = 0;
        while (t < n_tokens) : (t += 1) {
            const c_t = kv_cache.readByToken(t);
            if (c_t.len < kv_lora_rank) {
                scores[score_off + @as(usize, @intCast(t))] = -std.math.floatMax(f32);
                continue;
            }
            var dot: f32 = 0;

            var j: u32 = 0;
            while (j < kv_lora_rank) : (j += 1) {
                dot += q_absorbed[q_abs_off + @as(usize, @intCast(j))] * c_t[@as(usize, @intCast(j))];
            }

            const q_rope_off = @as(usize, @intCast(h)) * @as(usize, @intCast(qk_rope_head_dim));
            const k_rope_off = @as(usize, @intCast(t)) * @as(usize, @intCast(NKV)) * @as(usize, @intCast(qk_rope_head_dim)) + @as(usize, @intCast(kv_h)) * @as(usize, @intCast(qk_rope_head_dim));

            var ri: u32 = 0;
            while (ri < qk_rope_head_dim) : (ri += 1) {
                dot += q_rope_rope[q_rope_off + @as(usize, @intCast(ri))] * k_rope_tables[k_rope_off + @as(usize, @intCast(ri))];
            }

            scores[score_off + @as(usize, @intCast(t))] = dot * attn_scale;

            _ = page_table;
        }
    }
}

/// Compute softmax over the sequence dimension for each head.
/// `scores` and `output` are both [NH * seq_len]. Modified in place.
fn softmax(scores: []f32, output: []f32, NH: u32, seq_len: u32) void {
    var h: u32 = 0;
    while (h < NH) : (h += 1) {
        const off = @as(usize, @intCast(h)) * @as(usize, @intCast(seq_len));
        const row = scores[off .. off + @as(usize, @intCast(seq_len))];

        // Find max for numerical stability
        var max_val: f32 = -std.math.floatMax(f32);
        for (row) |v| {
            max_val = @max(max_val, v);
        }

        // Compute exp and sum
        var sum: f32 = 0;
        var i: usize = 0;
        while (i < row.len) : (i += 1) {
            const e = std.math.exp(row[i] - max_val);
            output[off + i] = e;
            sum += e;
        }

        // Normalize
        const inv_sum = 1.0 / sum;
        i = 0;
        while (i < row.len) : (i += 1) {
            output[off + i] *= inv_sum;
        }
    }
}

/// Weighted sum: attn_out[h] = sum_t softmax[h][t] * v[t][kv_h]
/// where v[t] is the decompressed V at position t.
fn weightedSum(
    attn_out: []f32,
    softmax_out: []const f32,
    kv_cache: *const FlatKvCache,
    v_b_proj: []const f32,
    NH: u32,
    NKV: u32,
    seq_len: u32,
    v_head_dim: u32,
    kv_lora_rank: u32,
) void {
    const gqa_ratio = NH / NKV;

    @memset(attn_out, 0);

    var h: u32 = 0;
    while (h < NH) : (h += 1) {
        const kv_h = h / gqa_ratio;
        const attn_off = @as(usize, @intCast(h)) * @as(usize, @intCast(v_head_dim));
        const sm_off = @as(usize, @intCast(h)) * @as(usize, @intCast(seq_len));

        var t: u32 = 0;
        while (t < seq_len) : (t += 1) {
            const w = softmax_out[sm_off + @as(usize, @intCast(t))];
            if (w < 1e-9) continue; // skip near-zero contributions

            // Decompress V for this token's compressed latent
            const c_t = kv_cache.read(t);

            // v_kvh = c_t @ v_b_proj[kv_h]^T
            const v_b_base = @as(usize, @intCast(kv_h)) * @as(usize, @intCast(v_head_dim)) * @as(usize, @intCast(kv_lora_rank));
            var i: u32 = 0;
            while (i < v_head_dim) : (i += 1) {
                const v_b_row = v_b_proj[v_b_base + @as(usize, @intCast(i)) * @as(usize, @intCast(kv_lora_rank)) ..];
                var j: u32 = 0;
                while (j < kv_lora_rank) : (j += 1) {
                    attn_out[attn_off + @as(usize, @intCast(i))] += w * v_b_row[@as(usize, @intCast(j))] * c_t[@as(usize, @intCast(j))];
                }
            }
        }
    }
}

/// Weighted sum variant for paged KV cache.
fn weightedSumPaged(
    attn_out: []f32,
    softmax_out: []const f32,
    kv_cache: *const PagedKvCache,
    v_b_proj: []const f32,
    NH: u32,
    NKV: u32,
    n_tokens: u32,
    v_head_dim: u32,
    kv_lora_rank: u32,
    page_table: []const u32,
) void {
    const gqa_ratio = NH / NKV;

    @memset(attn_out, 0);

    var h: u32 = 0;
    while (h < NH) : (h += 1) {
        const kv_h = h / gqa_ratio;
        const attn_off = @as(usize, @intCast(h)) * @as(usize, @intCast(v_head_dim));
        const sm_off = @as(usize, @intCast(h)) * @as(usize, @intCast(n_tokens));
        const v_b_base = @as(usize, @intCast(kv_h)) * @as(usize, @intCast(v_head_dim)) * @as(usize, @intCast(kv_lora_rank));

        var t: u32 = 0;
        while (t < n_tokens) : (t += 1) {
            const w = softmax_out[sm_off + @as(usize, @intCast(t))];
            if (w < 1e-9) continue;

            const c_t = kv_cache.readByToken(t);
            if (c_t.len < kv_lora_rank) continue;

            var i: u32 = 0;
            while (i < v_head_dim) : (i += 1) {
                const v_b_row = v_b_proj[v_b_base + @as(usize, @intCast(i)) * @as(usize, @intCast(kv_lora_rank)) ..];
                var j: u32 = 0;
                while (j < kv_lora_rank) : (j += 1) {
                    attn_out[attn_off + @as(usize, @intCast(i))] += w * v_b_row[@as(usize, @intCast(j))] * c_t[@as(usize, @intCast(j))];
                }
            }

            _ = page_table;
        }
    }
}

/// Apply the output projection.
/// `attn_out`: [NH * v_head_dim], `o_proj`: [H, NH * v_head_dim], output: [H].
fn outputProjection(output: []f32, attn_out: []const f32, o_proj: []const f32, H: u32, nhe_vdim: u32) void {
    @memset(output, 0);
    var i: u32 = 0;
    while (i < H) : (i += 1) {
        const o_row = o_proj[@as(usize, @intCast(i)) * @as(usize, @intCast(nhe_vdim)) ..];
        var j: u32 = 0;
        while (j < nhe_vdim) : (j += 1) {
            output[@as(usize, @intCast(i))] += o_row[@as(usize, @intCast(j))] * attn_out[@as(usize, @intCast(j))];
        }
    }
}

// ── Top-level decode (single query token) ────────────────────

/// Run MLA attention for a single query token (decode step).
///
/// Parameters:
///   config      — MLAConfig describing model dimensions
///   weights     — MLAWeights (q_proj, kv_a_proj, k_b_proj, v_b_proj, o_proj)
///   absorbed    — MLAAbsorbedWeights (precomputed q_absorbed, k_b_rope, v_b, o_proj)
///   scratch     — MLAScratch with pre-sized buffers
///   rope_tables — Precomputed sin/cos for the RoPE dimension
///   kv_cache    — Flat KV cache with prior latents
///   input       — Hidden state input [H] f32
///   output      — Output hidden state [H] f32
///   pos         — Current position (for RoPE lookup, only used if not absorbed)
pub fn decodeMlaFlat(
    config: *const MLAConfig,
    weights: *const MLAWeights,
    absorbed: *const MLAAbsorbedWeights,
    scratch: *MLAScratch,
    rope_tables: *const MLARopeTables,
    kv_cache: *FlatKvCache,
    input: []const f32,
    output: []f32,
    pos: u32,
) void {
    _ = absorbed; // reserved for future absorbed-weight path
    const NH = config.NH;
    const NKV = config.NKV;
    const H = config.H;
    const kv_lora_rank = config.kv_lora_rank;
    const qk_nope_head_dim = config.qk_nope_head_dim;
    const qk_rope_head_dim = config.qk_rope_head_dim;
    const qk_head_dim = config.qk_head_dim;
    const v_head_dim = config.v_head_dim;
    const seq_len = kv_cache.seq_len;
    const attn_scale = config.attn_scale;

    // ── 1. Q projection: q = input @ q_proj^T ──
    // q_proj is [q_out_dim, H], we compute q[h] = sum_i input[i] * q_proj[h][i]
    @memset(scratch.q, 0);
    var h: u32 = 0;
    while (h < NH) : (h += 1) {
        const q_off = @as(usize, @intCast(h)) * @as(usize, @intCast(qk_head_dim));
        var i: u32 = 0;
        while (i < qk_head_dim) : (i += 1) {
            const w_off = @as(usize, @intCast(h)) * @as(usize, @intCast(qk_head_dim)) + @as(usize, @intCast(i));
            const w_row_start = w_off * @as(usize, @intCast(H));
            var j: u32 = 0;
            while (j < H) : (j += 1) {
                scratch.q[q_off + @as(usize, @intCast(i))] += input[@as(usize, @intCast(j))] * weights.q_proj[w_row_start + @as(usize, @intCast(j))];
            }
        }
    }

    // ── 2. Split Q into nope and rope ──
    splitQRope(scratch.q, scratch.q_nope, scratch.q_rope, NH, qk_nope_head_dim, qk_rope_head_dim);

    // ── 3. Apply RoPE to q_rope ──
    const rope_at_pos = rope_tables.get(pos);
    applyRoPE(scratch.q_rope, NH, qk_rope_head_dim, rope_at_pos.sin, rope_at_pos.cos);

    // ── 4. Compute absorbed Q ──
    computeAbsorbedQ(scratch.q_nope, scratch.q_absorbed, weights.k_b_proj, NH, NKV, qk_nope_head_dim, kv_lora_rank);

    // ── 5. Compute scores over all cached tokens ──
    // We need K- rope cached for all prior positions. Since we compress K, we
    // keep a secondary cache of pre-roped K- rope values for efficiency.
    // For now, decompress-and-rope each token on the fly (simpler, correct).
    // TODO: maintain a pre-roped k_rope cache for speed.

    // Build k_rope_tables on the fly from the KV cache
    // This is the slow path — in production, maintain a persistent
    // k_rope_cache that's updated on each decode step.
    var k_rope_cache = scratch.k_rope; // reuse as temp for one token's k_rope

    // ── 5a. Compute scores per token (avoiding large k_rope_table) ──
    const gqa_ratio = NH / NKV;
    _ = gqa_ratio;

    var h_idx: u32 = 0;
    while (h_idx < NH) : (h_idx += 1) {
        const kv_h = h_idx / (NH / NKV);
        const score_off = @as(usize, @intCast(h_idx)) * @as(usize, @intCast(seq_len));
        const q_abs_off = @as(usize, @intCast(h_idx)) * @as(usize, @intCast(kv_lora_rank));
        const q_rope_off = @as(usize, @intCast(h_idx)) * @as(usize, @intCast(qk_rope_head_dim));

        var t: u32 = 0;
        while (t < seq_len) : (t += 1) {
            const c_t = kv_cache.read(t);

            // Decompress k_rope for this token (only once per KV head)
            // Since all Q heads sharing the same KV head get the same k_rope,
            // we could cache it per KV head, but for simplicity compute inline.
            decompressKRope(c_t, k_rope_cache, weights.k_b_proj, NKV, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank);

            // Apply RoPE to k_rope
            const rope_at_t = rope_tables.get(t);
            applyRoPE(k_rope_cache[0..@as(usize, @intCast(NKV * qk_rope_head_dim))], NKV, qk_rope_head_dim, rope_at_t.sin, rope_at_t.cos);

            // Compute score
            var dot: f32 = 0;

            // Nope part: q_absorbed @ c[t]
            var j: u32 = 0;
            while (j < kv_lora_rank) : (j += 1) {
                dot += scratch.q_absorbed[q_abs_off + @as(usize, @intCast(j))] * c_t[@as(usize, @intCast(j))];
            }

            // RoPE part
            const k_rope_start = @as(usize, @intCast(kv_h)) * @as(usize, @intCast(qk_rope_head_dim));
            var ri: u32 = 0;
            while (ri < qk_rope_head_dim) : (ri += 1) {
                dot += scratch.q_rope[q_rope_off + @as(usize, @intCast(ri))] * k_rope_cache[k_rope_start + @as(usize, @intCast(ri))];
            }

            scratch.scores[score_off + @as(usize, @intCast(t))] = dot * attn_scale;
        }
    }

    // ── 6. Softmax ──
    softmax(scratch.scores, scratch.softmax_out, NH, seq_len);

    // ── 7. Weighted sum to get attention output ──
    weightedSum(scratch.attn_out, scratch.softmax_out, kv_cache, weights.v_b_proj, NH, NKV, seq_len, v_head_dim, kv_lora_rank);

    // ── 8. Output projection ──
    outputProjection(output, scratch.attn_out, weights.o_proj, H, NH * v_head_dim);
}

/// Run MLA attention for a single query with paged KV cache.
pub fn decodeMlaPaged(
    config: *const MLAConfig,
    weights: *const MLAWeights,
    absorbed: *const MLAAbsorbedWeights,
    scratch: *MLAScratch,
    rope_tables: *const MLARopeTables,
    kv_cache: *PagedKvCache,
    page_table: []const u32,
    n_tokens: u32,
    input: []const f32,
    output: []f32,
    pos: u32,
) void {
    _ = absorbed; // reserved for future absorbed-weight path
    const NH = config.NH;
    const NKV = config.NKV;
    const H = config.H;
    const kv_lora_rank = config.kv_lora_rank;
    const qk_nope_head_dim = config.qk_nope_head_dim;
    const qk_rope_head_dim = config.qk_rope_head_dim;
    const qk_head_dim = config.qk_head_dim;
    const v_head_dim = config.v_head_dim;
    const attn_scale = config.attn_scale;

    // ── 1. Q projection ──
    @memset(scratch.q, 0);
    var h: u32 = 0;
    while (h < NH) : (h += 1) {
        const q_off = @as(usize, @intCast(h)) * @as(usize, @intCast(qk_head_dim));
        var i: u32 = 0;
        while (i < qk_head_dim) : (i += 1) {
            const w_off = @as(usize, @intCast(h)) * @as(usize, @intCast(qk_head_dim)) + @as(usize, @intCast(i));
            const w_row_start = w_off * @as(usize, @intCast(H));
            var j: u32 = 0;
            while (j < H) : (j += 1) {
                scratch.q[q_off + @as(usize, @intCast(i))] += input[@as(usize, @intCast(j))] * weights.q_proj[w_row_start + @as(usize, @intCast(j))];
            }
        }
    }

    // ── 2. Split Q ──
    splitQRope(scratch.q, scratch.q_nope, scratch.q_rope, NH, qk_nope_head_dim, qk_rope_head_dim);

    // ── 3. RoPE on q_rope ──
    const rope_at_pos = rope_tables.get(pos);
    applyRoPE(scratch.q_rope, NH, qk_rope_head_dim, rope_at_pos.sin, rope_at_pos.cos);

    // ── 4. Absorbed Q ──
    computeAbsorbedQ(scratch.q_nope, scratch.q_absorbed, weights.k_b_proj, NH, NKV, qk_nope_head_dim, kv_lora_rank);

    // ── 5. Scores over paged tokens ──
    const gqa_ratio = NH / NKV;

    var h_idx: u32 = 0;
    while (h_idx < NH) : (h_idx += 1) {
        const kv_h = h_idx / gqa_ratio;
        const score_off = @as(usize, @intCast(h_idx)) * @as(usize, @intCast(n_tokens));
        const q_abs_off = @as(usize, @intCast(h_idx)) * @as(usize, @intCast(kv_lora_rank));
        const q_rope_off = @as(usize, @intCast(h_idx)) * @as(usize, @intCast(qk_rope_head_dim));

        var t: u32 = 0;
        while (t < n_tokens) : (t += 1) {
            const c_t = kv_cache.readByToken(t);
            if (c_t.len < kv_lora_rank) {
                scratch.scores[score_off + @as(usize, @intCast(t))] = -std.math.floatMax(f32);
                continue;
            }

            // Decompress and RoPE k_rope
            decompressKRope(c_t, scratch.k_rope, weights.k_b_proj, NKV, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank);
            const rope_at_t = rope_tables.get(t);
            applyRoPE(scratch.k_rope[0..@as(usize, @intCast(NKV * qk_rope_head_dim))], NKV, qk_rope_head_dim, rope_at_t.sin, rope_at_t.cos);

            var dot: f32 = 0;
            var j: u32 = 0;
            while (j < kv_lora_rank) : (j += 1) {
                dot += scratch.q_absorbed[q_abs_off + @as(usize, @intCast(j))] * c_t[@as(usize, @intCast(j))];
            }

            const k_rope_start = @as(usize, @intCast(kv_h)) * @as(usize, @intCast(qk_rope_head_dim));
            var ri: u32 = 0;
            while (ri < qk_rope_head_dim) : (ri += 1) {
                dot += scratch.q_rope[q_rope_off + @as(usize, @intCast(ri))] * scratch.k_rope[k_rope_start + @as(usize, @intCast(ri))];
            }

            scratch.scores[score_off + @as(usize, @intCast(t))] = dot * attn_scale;
        }
    }

    // ── 6. Softmax ──
    softmax(scratch.scores, scratch.softmax_out, NH, n_tokens);

    // ── 7. Weighted sum ──
    weightedSumPaged(scratch.attn_out, scratch.softmax_out, kv_cache, weights.v_b_proj, NH, NKV, n_tokens, v_head_dim, kv_lora_rank, page_table);

    // ── 8. Output projection ──
    outputProjection(output, scratch.attn_out, weights.o_proj, H, NH * v_head_dim);
}

// ── Prefill (batched tokens) ─────────────────────────────────

/// Prefill MLA attention for a batch of tokens (seq_len >= 1).
///
/// Uses the full non-absorbed path for simplicity and correctness:
///   1. Decompress each token's KV latent
///   2. Store compressed latents in KV cache
///   3. Run causal attention
///
/// Parameters:
///   inputs   — [n_tokens, H] row-major input hidden states
///   output   — [n_tokens, H] row-major output (can alias inputs)
///   kv_latents — [n_tokens, kv_lora_rank] compressed KV latents (output)
///                Each token's compressed latent is written here.
pub fn prefillMla(
    config: *const MLAConfig,
    weights: *const MLAWeights,
    absorbed: *const MLAAbsorbedWeights,
    scratch: *MLAScratch,
    rope_tables: *const MLARopeTables,
    kv_cache: *FlatKvCache,
    inputs: []const f32,
    output: []f32,
    kv_latents: []f32,
    n_tokens: u32,
) void {
    _ = absorbed; // reserved for future absorbed-weight path
    const NH = config.NH;
    const NKV = config.NKV;
    const H = config.H;
    const kv_lora_rank = config.kv_lora_rank;
    const qk_nope_head_dim = config.qk_nope_head_dim;
    const qk_rope_head_dim = config.qk_rope_head_dim;
    const v_head_dim = config.v_head_dim;
    const qk_head_dim = config.qk_head_dim;
    const attn_scale = config.attn_scale;

    var t: u32 = 0;
    while (t < n_tokens) : (t += 1) {
        const inp = inputs[@as(usize, @intCast(t)) * @as(usize, @intCast(H)) ..];
        const out = output[@as(usize, @intCast(t)) * @as(usize, @intCast(H)) ..];
        const latent = kv_latents[@as(usize, @intCast(t)) * @as(usize, @intCast(kv_lora_rank)) ..];

        // ── 1. Q projection ──
        @memset(scratch.q, 0);
        var h: u32 = 0;
        while (h < NH) : (h += 1) {
            const q_off = @as(usize, @intCast(h)) * @as(usize, @intCast(qk_head_dim));
            var i: u32 = 0;
            while (i < qk_head_dim) : (i += 1) {
                const w_off = @as(usize, @intCast(h)) * @as(usize, @intCast(qk_head_dim)) + @as(usize, @intCast(i));
                const w_row_start = w_off * @as(usize, @intCast(H));
                var j: u32 = 0;
                while (j < H) : (j += 1) {
                    scratch.q[q_off + @as(usize, @intCast(i))] += inp[@as(usize, @intCast(j))] * weights.q_proj[w_row_start + @as(usize, @intCast(j))];
                }
            }
        }

        // ── 2. Compute compressed KV latent: c = inp @ kv_a_proj^T ──
        // kv_a_proj is [kv_lora_rank, H], so latent[k] = sum_i inp[i] * kv_a_proj[k][i]
        @memset(latent, 0);
        var k: u32 = 0;
        while (k < kv_lora_rank) : (k += 1) {
            const a_row_start = @as(usize, @intCast(k)) * @as(usize, @intCast(H));
            var j: u32 = 0;
            while (j < H) : (j += 1) {
                latent[@as(usize, @intCast(k))] += inp[@as(usize, @intCast(j))] * weights.kv_a_proj[a_row_start + @as(usize, @intCast(j))];
            }
        }

        // ── 3. Write latent to cache ──
        kv_cache.write(t, latent);

        // ── 4. Split Q ──
        splitQRope(scratch.q, scratch.q_nope, scratch.q_rope, NH, qk_nope_head_dim, qk_rope_head_dim);

        // ── 5. RoPE on q_rope ──
        const rope_at_t = rope_tables.get(t);
        applyRoPE(scratch.q_rope, NH, qk_rope_head_dim, rope_at_t.sin, rope_at_t.cos);

        // ── 6. Compute scores (causal: only positions <= t) ──
        const causal_len = t + 1;
        const gqa_ratio = NH / NKV;

        var h_idx: u32 = 0;
        while (h_idx < NH) : (h_idx += 1) {
            const kv_h = h_idx / gqa_ratio;
            const score_off = @as(usize, @intCast(h_idx)) * @as(usize, @intCast(causal_len));
            const q_abs_off = @as(usize, @intCast(h_idx)) * @as(usize, @intCast(kv_lora_rank));
            const q_rope_off = @as(usize, @intCast(h_idx)) * @as(usize, @intCast(qk_rope_head_dim));

            var s: u32 = 0;
            while (s < causal_len) : (s += 1) {
                const c_s = kv_cache.read(s);

                // Absorbed nope score
                var dot: f32 = 0;
                var j: u32 = 0;
                while (j < kv_lora_rank) : (j += 1) {
                    dot += scratch.q_absorbed[q_abs_off + @as(usize, @intCast(j))] * c_s[@as(usize, @intCast(j))];
                }

                // Decompress and RoPE k_rope for this source position
                decompressKRope(c_s, scratch.k_rope, weights.k_b_proj, NKV, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank);

                const rope_at_s = rope_tables.get(s);
                applyRoPE(scratch.k_rope[0..@as(usize, @intCast(NKV * qk_rope_head_dim))], NKV, qk_rope_head_dim, rope_at_s.sin, rope_at_s.cos);

                const k_rope_start = @as(usize, @intCast(kv_h)) * @as(usize, @intCast(qk_rope_head_dim));
                var ri: u32 = 0;
                while (ri < qk_rope_head_dim) : (ri += 1) {
                    dot += scratch.q_rope[q_rope_off + @as(usize, @intCast(ri))] * scratch.k_rope[k_rope_start + @as(usize, @intCast(ri))];
                }

                scratch.scores[score_off + @as(usize, @intCast(s))] = dot * attn_scale;
            }
        }

        // ── 7. Softmax (causal) ──
        softmax(scratch.scores[0 .. @as(usize, @intCast(NH * causal_len))], scratch.softmax_out, NH, causal_len);

        // ── 8. Weighted sum ──
        @memset(scratch.attn_out, 0);
        for (0..@as(usize, @intCast(NH))) |h_idx_usize| {
            const hi = @as(u32, @truncate(h_idx_usize));
            const kv_h = hi / (NH / NKV);
            const attn_off = h_idx_usize * @as(usize, @intCast(v_head_dim));
            const sm_off = h_idx_usize * @as(usize, @intCast(causal_len));

            for (0..@as(usize, @intCast(causal_len))) |s_usize| {
                const s = @as(u32, @truncate(s_usize));
                const w = scratch.softmax_out[sm_off + s_usize];
                if (w < 1e-9) continue;

                const c_s = kv_cache.read(s);
                const v_b_base = @as(usize, @intCast(kv_h)) * @as(usize, @intCast(v_head_dim)) * @as(usize, @intCast(kv_lora_rank));

                var i: u32 = 0;
                while (i < v_head_dim) : (i += 1) {
                    const v_b_row = weights.v_b_proj[v_b_base + @as(usize, @intCast(i)) * @as(usize, @intCast(kv_lora_rank)) ..];
                    var j: u32 = 0;
                    while (j < kv_lora_rank) : (j += 1) {
                        scratch.attn_out[attn_off + @as(usize, @intCast(i))] += w * v_b_row[@as(usize, @intCast(j))] * c_s[@as(usize, @intCast(j))];
                    }
                }
            }
        }

        // ── 9. Output projection ──
        outputProjection(out, scratch.attn_out, weights.o_proj, H, NH * v_head_dim);
    }
}

// ── Weight absorption precomputation ─────────────────────────

/// Precompute the absorbed Q weights from q_proj and k_b_proj.
///
/// The absorbed Q weight is:
///   q_absorbed_weight[h, :kv_lora_rank] = q_proj[h, :qk_nope] @ k_b_proj[kv_h, :qk_nope, :kv_lora_rank]
///
/// Where q_proj[h] is [qk_head_dim], we take the first qk_nope_head_dim elements
/// (the nope portion), and multiply by the nope portion of k_b_proj for the
/// corresponding KV head.
///
/// Returns: [NH * kv_lora_rank] owned slice.
pub fn computeAbsorbedWeights(
    allocator: std.mem.Allocator,
    config: *const MLAConfig,
    q_proj: []const f32,
    k_b_proj: []const f32,
) ![]f32 {
    const NH = config.NH;
    const NKV = config.NKV;
    const qk_nope_head_dim = config.qk_nope_head_dim;
    const qk_rope_head_dim = config.qk_rope_head_dim;
    const kv_lora_rank = config.kv_lora_rank;
    const qk_head_dim = config.qk_head_dim;
    const H = config.H;
    const gqa_ratio = NH / NKV;

    const result = try allocator.alloc(f32, @as(usize, @intCast(NH)) * @as(usize, @intCast(kv_lora_rank)));
    @memset(result, 0);

    var h: u32 = 0;
    while (h < NH) : (h += 1) {
        const kv_h = h / gqa_ratio;
        const q_base = @as(usize, @intCast(h)) * @as(usize, @intCast(qk_head_dim));
        const out_base = @as(usize, @intCast(h)) * @as(usize, @intCast(kv_lora_rank));

        // k_b_nope for this KV head: rows [kv_h * qk_head_dim .. kv_h * qk_head_dim + qk_nope_head_dim]
        const k_b_base = (@as(usize, @intCast(kv_h)) * @as(usize, @intCast(qk_head_dim))) * @as(usize, @intCast(kv_lora_rank));

        // For each hidden dimension, accumulate q_proj contribution × k_b_proj
        var i: u32 = 0;
        while (i < H) : (i += 1) {
            var nope_i: u32 = 0;
            while (nope_i < qk_nope_head_dim) : (nope_i += 1) {
                const q_val = q_proj[(q_base + @as(usize, @intCast(nope_i))) * @as(usize, @intCast(H)) + @as(usize, @intCast(i))];
                if (q_val == 0) continue;

                const k_b_row_start = k_b_base + @as(usize, @intCast(nope_i)) * @as(usize, @intCast(kv_lora_rank));
                var j: u32 = 0;
                while (j < kv_lora_rank) : (j += 1) {
                    result[out_base + @as(usize, @intCast(j))] += q_val * k_b_proj[k_b_row_start + @as(usize, @intCast(j))];
                }
            }
            _ = qk_rope_head_dim;
        }
    }

    return result;
}

/// Build the full MLAAbsorbedWeights from raw MLAWeights.
///
/// Owns q_absorbed; views k_b_rope, v_b, o_proj from the input weights
/// (caller must keep those alive).
pub fn buildAbsorbedWeights(
    allocator: std.mem.Allocator,
    config: *const MLAConfig,
    weights: *const MLAWeights,
) !MLAAbsorbedWeights {
    const q_absorbed = try computeAbsorbedWeights(allocator, config, weights.q_proj, weights.k_b_proj);

    const NKV = config.NKV;
    const qk_nope_head_dim = config.qk_nope_head_dim;
    const qk_rope_head_dim = config.qk_rope_head_dim;
    const kv_lora_rank = config.kv_lora_rank;

    // k_b_rope is the rope portion of k_b_proj: rows [qk_nope_head_dim .. qk_head_dim) per KV head
    const k_b_rope_start = @as(usize, @intCast(qk_nope_head_dim)) * @as(usize, @intCast(kv_lora_rank));
    const k_b_rope_len = @as(usize, @intCast(NKV)) * @as(usize, @intCast(qk_rope_head_dim)) * @as(usize, @intCast(kv_lora_rank));
    const k_b_rope = weights.k_b_proj[k_b_rope_start .. k_b_rope_start + k_b_rope_len];

    return MLAAbsorbedWeights{
        .q_absorbed = q_absorbed,
        .k_b_rope = k_b_rope,
        .v_b = weights.v_b_proj,
        .o_proj = weights.o_proj,
    };
}

// ── Unit tests ───────────────────────────────────────────────

test "MLAConfig init and validate" {
    const cfg = MLAConfig.init(16, 2, 2048, 256, 4096, .{});
    try cfg.validate();

    try std.testing.expectEqual(@as(u32, 192), cfg.qk_head_dim);
    try std.testing.expectEqual(@as(u32, 16 * 192), cfg.q_out_dim);
    try std.testing.expectEqual(@as(u32, 2 * 192), cfg.k_out_dim);
    try std.testing.expectEqual(@as(u32, 2 * 128), cfg.v_out_dim);
    try std.testing.expectEqual(@as(u32, 8), cfg.gqa_ratio);
    try std.testing.expect(cfg.attn_scale > 0);
}

test "MLAConfig validation errors" {
    const cfg1 = MLAConfig.init(0, 8, 7168, 512, 8192, .{});
    try std.testing.expectError(error.MLAZeroHeads, cfg1.validate());

    const cfg2 = MLAConfig.init(56, 0, 7168, 512, 8192, .{});
    try std.testing.expectError(error.MLAZeroKvHeads, cfg2.validate());

    const cfg3 = MLAConfig.init(56, 8, 0, 512, 8192, .{});
    try std.testing.expectError(error.MLAZeroHidden, cfg3.validate());

    const cfg4 = MLAConfig.init(56, 8, 7168, 0, 8192, .{});
    try std.testing.expectError(error.MLAZeroLoraRank, cfg4.validate());
}

test "MLAConfig DeepSeek-V3 dimensions" {
    const cfg = MLAConfig.init(56, 8, 7168, 512, 8192, .{});
    try std.testing.expectEqual(@as(u32, 56), cfg.NH);
    try std.testing.expectEqual(@as(u32, 8), cfg.NKV);
    try std.testing.expectEqual(@as(u32, 7168), cfg.H);
    try std.testing.expectEqual(@as(u32, 512), cfg.kv_lora_rank);
    try std.testing.expectEqual(@as(u32, 128), cfg.qk_nope_head_dim);
    try std.testing.expectEqual(@as(u32, 64), cfg.qk_rope_head_dim);
    try std.testing.expectEqual(@as(u32, 192), cfg.qk_head_dim);
    try std.testing.expectEqual(@as(u32, 7), cfg.gqa_ratio); // 56/8
}

test "MLAConfig DeepSeek-V2-Lite dimensions" {
    const cfg = MLAConfig.init(16, 2, 2048, 256, 4096, .{});
    try std.testing.expectEqual(@as(u32, 16), cfg.NH);
    try std.testing.expectEqual(@as(u32, 2), cfg.NKV);
    try std.testing.expectEqual(@as(u32, 2048), cfg.H);
    try std.testing.expectEqual(@as(u32, 256), cfg.kv_lora_rank);
    try std.testing.expectEqual(@as(u32, 8), cfg.gqa_ratio); // 16/2
}

test "MLARopeTables basic" {
    const allocator = std.testing.allocator;
    const cfg = MLAConfig.init(16, 2, 2048, 256, 64, .{ .qk_rope_head_dim = 4 });

    var tables = try MLARopeTables.init(allocator, &cfg);
    defer tables.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 4), tables.qk_rope_head_dim);
    try std.testing.expectEqual(@as(u32, 64), tables.max_seq_len);
    try std.testing.expectEqual(@as(usize, 256), tables.sin.len); // 64 * 4
    try std.testing.expectEqual(@as(usize, 256), tables.cos.len);

    // Position 0: sin=0, cos=1
    const pos0 = tables.get(0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), pos0.sin[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), pos0.cos[0], 0.001);
}

test "splitQRope smoke test" {
    const NH: u32 = 2;
    const qk_nope: u32 = 3;
    const qk_rope: u32 = 2;
    const qk_head = qk_nope + qk_rope;

    var q: [NH * qk_head]f32 = undefined;
    comptime var idx: usize = 0;
    inline while (idx < q.len) : (idx += 1) q[idx] = @floatFromInt(idx + 1);

    var q_nope: [NH * qk_nope]f32 = undefined;
    var q_rope: [NH * qk_rope]f32 = undefined;

    splitQRope(&q, &q_nope, &q_rope, NH, qk_nope, qk_rope);

    // Head 0: q_nope = q[0..2], q_rope = q[3..4]
    try std.testing.expectEqual(@as(f32, 1.0), q_nope[0]);
    try std.testing.expectEqual(@as(f32, 2.0), q_nope[1]);
    try std.testing.expectEqual(@as(f32, 3.0), q_nope[2]);
    try std.testing.expectEqual(@as(f32, 4.0), q_rope[0]);
    try std.testing.expectEqual(@as(f32, 5.0), q_rope[1]);

    // Head 1: starts at q[5]
    try std.testing.expectEqual(@as(f32, 6.0), q_nope[3]);
    try std.testing.expectEqual(@as(f32, 7.0), q_nope[4]);
    try std.testing.expectEqual(@as(f32, 8.0), q_nope[5]);
    try std.testing.expectEqual(@as(f32, 9.0), q_rope[2]);
    try std.testing.expectEqual(@as(f32, 10.0), q_rope[3]);
}

test "applyRoPE smoke test" {
    const n_heads: u32 = 1;
    const rope_dim: u32 = 4;

    var data: [4]f32 = .{ 1.0, 0.0, 0.0, 1.0 };
    const sin: [4]f32 = .{ 0.0, 0.0, 1.0, 1.0 };
    const cos: [4]f32 = .{ 1.0, 1.0, 0.0, 0.0 };

    applyRoPE(&data, n_heads, rope_dim, &sin, &cos);

    // Pair 0: (1,0) with (sin=0, cos=1) → (1,0)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[1], 0.001);

    // Pair 1: (0,1) with (sin=1, cos=0) → (-1, 0)
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[3], 0.001);
}

test "softmax basic" {
    var scores: [4]f32 = .{ 2.0, 1.0, 0.0, -1.0 };
    var output: [4]f32 = undefined;

    softmax(&scores, &output, 1, 4);

    // Check it sums to 1
    var sum: f32 = 0;
    for (output) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);

    // Ordering should be preserved
    try std.testing.expect(output[0] > output[1]);
    try std.testing.expect(output[1] > output[2]);
    try std.testing.expect(output[2] > output[3]);
}

test "softmax multi-head" {
    var scores: [6]f32 = .{ 1.0, 2.0, 3.0, 3.0, 2.0, 1.0 };
    var output: [6]f32 = undefined;

    softmax(&scores, &output, 2, 3);

    // Head 0: {1,2,3}
    var sum0: f32 = 0;
    for (output[0..3]) |v| sum0 += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum0, 0.001);
    try std.testing.expect(output[2] > output[1]);
    try std.testing.expect(output[1] > output[0]);

    // Head 1: {3,2,1}
    var sum1: f32 = 0;
    for (output[3..6]) |v| sum1 += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum1, 0.001);
    try std.testing.expect(output[3] > output[4]);
    try std.testing.expect(output[4] > output[5]);
}

test "FlatKvCache read/write" {
    const allocator = std.testing.allocator;
    var cache = try FlatKvCache.init(allocator, 4, 10);
    defer cache.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 0), cache.seq_len);

    const latent1: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
    cache.write(0, &latent1);

    try std.testing.expectEqual(@as(u32, 1), cache.seq_len);

    const read1 = cache.read(0);
    try std.testing.expectEqual(@as(f32, 1.0), read1[0]);
    try std.testing.expectEqual(@as(f32, 4.0), read1[3]);

    const latent2: [4]f32 = .{ 5.0, 6.0, 7.0, 8.0 };
    cache.write(1, &latent2);
    try std.testing.expectEqual(@as(u32, 2), cache.seq_len);

    const read2 = cache.read(1);
    try std.testing.expectEqual(@as(f32, 5.0), read2[0]);
    try std.testing.expectEqual(@as(f32, 8.0), read2[3]);

    cache.reset();
    try std.testing.expectEqual(@as(u32, 0), cache.seq_len);
}

test "PagedKvCache read/write" {
    const allocator = std.testing.allocator;
    var cache = try PagedKvCache.init(allocator, 4, 8, 10);
    defer cache.deinit(allocator);

    const latent: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
    cache.write(0, 0, &latent);

    const read = cache.read(0, 0);
    try std.testing.expectEqual(@as(f32, 1.0), read[0]);
    try std.testing.expectEqual(@as(f32, 4.0), read[3]);
}

test "MLAScratch init/deinit" {
    const allocator = std.testing.allocator;
    const cfg = MLAConfig.init(4, 2, 256, 32, 128, .{});
    var scratch = try MLAScratch.init(allocator, &cfg, 128);
    defer scratch.deinit(allocator);

    try std.testing.expectEqual(@as(usize, @intCast(4 * 192)), scratch.q.len);
    try std.testing.expectEqual(@as(usize, @intCast(4 * 32)), scratch.q_absorbed.len);
}

test "outputProjection basic" {
    // H=4, NH=1, v_head_dim=2 → nhe_vdim=2
    const H: u32 = 4;
    const vdim: u32 = 2;
    var attn: [2]f32 = .{ 1.0, 2.0 };
    var o_proj: [8]f32 = .{
        0.5, 0.0, // row 0
        0.0, 0.5, // row 1
        1.0, 0.0, // row 2
        0.0, 1.0, // row 3
    };
    var output: [4]f32 = undefined;

    outputProjection(&output, &attn, &o_proj, H, vdim);

    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), output[3], 0.001);
}

test "computeAbsorbedWeights smoke test" {
    const allocator = std.testing.allocator;
    const cfg = MLAConfig{ .NH = 1, .NKV = 1, .H = 2, .kv_lora_rank = 3, .max_seq_len = 10, .attn_scale = 0.1, .qk_head_dim = 4, .qk_nope_head_dim = 2, .qk_rope_head_dim = 2, .v_head_dim = 2, .q_out_dim = 4, .k_out_dim = 4, .v_out_dim = 2, .gqa_ratio = 1 };

    // q_proj: [1*4, 2] = [4,2]
    const q_proj: [8]f32 = .{
        1, 0, // h=0, nope_i=0 → [H=2]
        0, 1, // h=0, nope_i=1
        0, 0, // h=0, rope_i=0 (ignored in absorption)
        0, 0, // h=0, rope_i=1 (ignored)
    };

    // k_b_proj: [1*4, 3] = [4,3]
    // Only the first 2 rows (nope portion) matter for absorption
    const k_b_proj: [12]f32 = .{
        1, 2, 3, // kv=0, nope_i=0
        4, 5, 6, // kv=0, nope_i=1
        0, 0, 0, // kv=0, rope_i=0
        0, 0, 0, // kv=0, rope_i=1
    };

    var weight_cfg = cfg;
    weight_cfg.qk_nope_head_dim = 2;
    weight_cfg.qk_rope_head_dim = 2;
    weight_cfg.v_head_dim = 2;

    const result = try computeAbsorbedWeights(allocator, &weight_cfg, &q_proj, &k_b_proj);
    defer allocator.free(result);

    // q_absorbed[0] = q[0] * k_b[0,:] + q[1] * k_b[1,:]
    // q[0] = input[0]*1 + input[1]*0 = 1*1 + 0*0 = 1  (from q_proj[0,0]=1)
    // Wait, this is the WEIGHT absorption, not value absorption.
    // q_absorbed_weight[h, :kv_lora] = sum_i q_proj[h, nope_i, :H] × k_b_proj[kv_h, nope_i, :kv_lora]
    //
    // Let me compute:
    // For NH=1, H=2, kv_lora=3, qk_nope=2:
    // q_proj[h=0, nope_i=0] = [1, 0]  →  contributes q_val=1 to H=0, q_val=0 to H=1
    // q_proj[h=0, nope_i=1] = [0, 1]  →  contributes q_val=0 to H=0, q_val=1 to H=1
    //
    // For each H dimension:
    // H=0: q_proj[nope_i=0, H=0]*k_b[nope_i=0] + q_proj[nope_i=1, H=0]*k_b[nope_i=1]
    //     = 1 * [1,2,3] + 0 * [4,5,6] = [1,2,3]
    // H=1: q_proj[nope_i=0, H=1]*k_b[nope_i=0] + q_proj[nope_i=1, H=1]*k_b[nope_i=1]
    //     = 0 * [1,2,3] + 1 * [4,5,6] = [4,5,6]
    //
    // result = [1+4, 2+5, 3+6] = [5, 7, 9]

    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), result[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), result[2], 0.001);
}
