//! CPU-side math operations for the NPU inference engine.
//! These run on the host CPU when the NPU xclbin cannot accelerate the operation
//! (attention softmax, RoPE, RMS norm, activation functions, LM head).
const std = @import("std");

const math = std.math;
const FloatMax = f32;

// ============================================================
// RMS Normalization
// ============================================================

/// In-place RMS normalization: x[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
/// Sets NaN/Inf inputs to 0 before computing.
pub fn rmsNorm(x: []f32, weight: []const f32, eps: f32) void {
    const n = x.len;
    if (n == 0) return;
    if (weight.len < n) return;

    // Clean NaN/Inf
    for (x) |*val| {
        if (!math.isFinite(val.*)) val.* = 0.0;
    }

    // Sum of squares (double precision for stability)
    var sum_sq: f64 = 0.0;
    for (x) |val| {
        sum_sq += @as(f64, val) * @as(f64, val);
    }

    const mean = sum_sq / @as(f64, @floatFromInt(n));
    const inv_rms = 1.0 / @as(f64, @sqrt(mean + @as(f64, eps)));

    for (x, weight) |*val, w| {
        val.* = @as(f32, @floatCast(@as(f64, val.*) * inv_rms * @as(f64, w)));
    }
}

// ============================================================
// RoPE (Rotary Position Embedding)
// ============================================================

/// Apply rotary position embedding to a single head's query/key vector.
/// x is [hd] interleaved pairs. rc/rs are precomputed tables at offset pos*hd.
pub fn rope(x: []f32, hd: usize, pos: usize, rc: []const f32, rs: []const f32) void {
    const hd2 = hd / 2;
    const base = pos * hd;
    for (0..hd2) |d| {
        const a = x[d];
        const b = x[d + hd2];
        const c = rc[base + d];
        const s = rs[base + d];
        x[d] = a * c - b * s;
        x[d + hd2] = a * s + b * c;
    }
}

/// Precompute RoPE cos/sin tables for up to max_pos positions.
pub fn precomputeRoPE(allocator: std.mem.Allocator, hd: usize, max_pos: usize, theta: f32) !struct { rc: []f32, rs: []f32 } {
    const hd2: f32 = @floatFromInt(hd);
    const rc = try allocator.alloc(f32, max_pos * hd);
    const rs = try allocator.alloc(f32, max_pos * hd);

    for (0..max_pos) |pos| {
        const pf: f32 = @floatFromInt(pos);
        for (0..hd / 2) |d| {
            const df: f32 = @floatFromInt(d);
            const angle = pf / std.math.pow(f32, theta, 2.0 * df / hd2);
            const c = @cos(angle);
            const s = @sin(angle);
            rc[pos * hd + d] = c;
            rs[pos * hd + d] = s;
            // Mirror for second half
            rc[pos * hd + hd / 2 + d] = c;
            rs[pos * hd + hd / 2 + d] = s;
        }
    }

    return .{ .rc = rc, .rs = rs };
}

// ============================================================
// Softmax (numerically stable)
// ============================================================

/// In-place softmax with numeric stability (max subtraction).
/// Handles NaN/Inf and edge cases (all-zero → uniform distribution).
pub fn softmax(scores: []f32) void {
    const n = scores.len;
    if (n == 0) return;

    // Clean NaN/Inf and find max
    var mx: f32 = -math.floatMax(f32);
    for (scores) |*s| {
        if (!math.isFinite(s.*)) s.* = -math.floatMax(f32);
        if (s.* > mx) mx = s.*;
    }

    // If all -inf (masked), return uniform
    if (mx <= -math.floatMax(f32) / 2) {
        const inv_n = 1.0 / @as(f32, @floatFromInt(n));
        for (scores) |*s| s.* = inv_n;
        return;
    }

    // exp(x - max) and sum
    var sum: f64 = 0.0;
    for (scores) |*s| {
        const diff = s.* - mx;
        const e = if (diff < -80.0) 0.0 else @as(f64, @exp(diff));
        s.* = @as(f32, @floatCast(e));
        sum += e;
    }

    // Normalize
    if (sum <= 0.0) {
        const inv_n = 1.0 / @as(f32, @floatFromInt(n));
        for (scores) |*s| s.* = inv_n;
    } else {
        const inv_sum = @as(f32, @floatCast(1.0 / sum));
        for (scores) |*s| s.* *= inv_sum;
    }
}

// ============================================================
// Attention operations
// ============================================================

/// Compute Q*K^T attention scores for a single head.
/// q: [hd], k_cache: [cl * nkv * hd], scores out: [cl]
pub fn attentionQK(
    q: []const f32,
    k_cache: []const f32,
    scores: []f32,
    cl: usize,
    nkv: usize,
    kvh: usize,
    hd: usize,
    max_pos: ?usize,
) void {
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    const k_offset = kvh * hd;

    for (0..cl) |p| {
        if (max_pos) |mp| {
            if (p >= mp) {
                scores[p] = -math.floatMax(f32);
                continue;
            }
        }
        var s: f64 = 0.0;
        const kp = k_cache[p * nkv * hd + k_offset .. p * nkv * hd + k_offset + hd];
        for (0..hd) |d| {
            s += @as(f64, q[d]) * @as(f64, kp[d]);
        }
        scores[p] = @as(f32, @floatCast(s * @as(f64, scale)));
    }
}

/// Compute weighted sum of values by softmax scores.
pub fn attentionPV(
    scores: []const f32,
    v_cache: []const f32,
    output: []f32,
    cl: usize,
    nkv: usize,
    kvh: usize,
    hd: usize,
) void {
    const v_offset = kvh * hd;

    for (0..hd) |d| {
        var acc: f64 = 0.0;
        for (0..cl) |p| {
            acc += @as(f64, scores[p]) * @as(f64, v_cache[p * nkv * hd + v_offset + d]);
        }
        output[d] = @as(f32, @floatCast(acc));
    }
}

/// Full single-head attention: scores → softmax → weighted sum.
pub fn attentionHead(
    q: []const f32,
    k_cache: []const f32,
    v_cache: []const f32,
    output: []f32,
    cl: usize,
    hh: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    max_pos: ?usize,
) !void {
    const gqa = nh / nkv;
    const kvh = hh / gqa;

    // Scratch scores
    var scores_buf: [4096]f32 = undefined;
    if (cl > scores_buf.len) return error.ContextTooLong;
    const scores = scores_buf[0..cl];

    attentionQK(q, k_cache, scores, cl, nkv, kvh, hd, max_pos);
    softmax(scores);
    attentionPV(scores, v_cache, output, cl, nkv, kvh, hd);
}

// ============================================================
// Activation functions
// ============================================================

/// SiLU (Sigmoid Linear Unit): x / (1 + exp(-x))
pub fn silu(x: f32) f32 {
    if (!math.isFinite(x)) return if (x > 0) x else 0.0;
    return x / (1.0 + @exp(-x));
}

/// Find maximum absolute value in an array (for dynamic INT8 scaling).
pub fn findMaxAbs(x: []const f32) f32 {
    var amax: f32 = 0.0;
    for (x) |val| {
        if (math.isFinite(val)) {
            const a = @abs(val);
            if (a > amax) amax = a;
        }
    }
    return if (amax < 1e-12) 1.0 else amax;
}

/// Set NaN/Inf values to 0.
pub fn clipAndClean(x: []f32) void {
    for (x) |*val| {
        if (!math.isFinite(val.*)) val.* = 0.0;
    }
}

// ============================================================
// LM Head (vocabulary projection)
// ============================================================

/// Compute logits = hidden @ embeddings^T
/// embeddings is [nv * h] row-major, hidden is [h], logits out is [nv]
pub fn lmHead(hidden: []const f32, embeddings: []const f32, logits: []f32, nv: usize, h: usize) void {
    for (0..nv) |n| {
        var s: f64 = 0.0;
        const emb_row = embeddings[n * h .. n * h + h];
        for (0..h) |i| {
            s += @as(f64, hidden[i]) * @as(f64, emb_row[i]);
        }
        logits[n] = @as(f32, @floatCast(s));
    }
}

/// Compute logits, softmax, sample one token from distribution, and return top-k IDs.
/// top_ids[0] = sampled token, top_ids[1..k] = top-k alternatives.
pub fn lmHeadTopK(
    hidden: []const f32,
    embeddings: []const f32,
    top_ids: []u32,
    k: u32,
    nv: usize,
    h: usize,
) !void {
    var sfb = std.heap.stackFallback(@sizeOf(f32) * 151936, std.heap.page_allocator);
    const allocator = sfb.get();
    // Note: stack allocation for logits is too large for 151K vocab — use heap
    // For simplicity with default small vocabs, we stack-allocate a reasonable buffer
    const max_vocab: usize = 200000;
    var fixed_buf: [200000]f32 = undefined;
    const logits = if (nv <= max_vocab)
        fixed_buf[0..nv]
    else
        allocator.alloc(f32, nv) catch @panic("OOM for logits buffer");

    // Compute logits
    lmHead(hidden, embeddings, logits, nv, h);

    // Softmax with max subtraction
    var mx: f32 = -math.floatMax(f32);
    for (logits) |l| {
        if (l > mx) mx = l;
    }
    var sum: f64 = 0.0;
    for (logits) |*l| {
        const diff = l.* - mx;
        const e = if (diff < -80.0) 0.0 else @as(f64, @exp(diff));
        l.* = @as(f32, @floatCast(e));
        sum += e;
    }
    if (sum <= 0.0) {
        // Fallback: uniform
        for (logits) |*l| {
            l.* = 1.0 / @as(f32, @floatFromInt(nv));
        }
    } else {
        const inv_sum = @as(f32, @floatCast(1.0 / sum));
        for (logits) |*l| l.* *= inv_sum;
    }

    // Sample (greedy: pick highest probability)
    var best_id: u32 = 0;
    var best_p: f32 = -1.0;
    for (logits, 0..) |p, i| {
        if (p > best_p) {
            best_p = p;
            best_id = @intCast(i);
        }
    }
    top_ids[0] = best_id;

    // Top-k (including the sampled one)
    // Simple O(n*k) selection for small k
    if (nv > 4096) return error.VocabTooLarge;
    const k_usize = @as(usize, @min(k, @as(u32, @intCast(nv))));
    var used = std.mem.zeroes([4096]bool); // track selected indices
    for (0..k_usize) |ki| {
        var best: f32 = -1.0;
        var best_idx: usize = 0;
        for (logits, 0..) |p, i| {
            if (!used[i] and p > best) {
                best = p;
                best_idx = i;
            }
        }
        top_ids[ki] = @intCast(best_idx);
        used[best_idx] = true;
    }
}
