//! T-CPU flash attention (single-query decode) implementation.
//! Computes scaled dot-product attention over a KV cache for one query token.
//! @section Inference Runtime
const std = @import("std");

/// Inputs and outputs for one single-query flash attention call.
/// @param q Query vector packed `[n_heads, head_dim]` for the current decode token.
/// @param kv_k Key cache packed `[seq_len, n_kv_heads, head_dim]`.
/// @param kv_v Value cache packed `[seq_len, n_kv_heads, head_dim]`.
/// @param output Attention output packed `[n_heads, head_dim]`.
/// @param n_heads Number of query heads.
/// @param n_kv_heads Number of key/value heads (GQA: `n_heads / n_kv_heads` queries share each KV head).
/// @param head_dim Per-head feature dimension.
/// @param seq_len Number of cached key/value positions to attend over.
/// @param attn_sinks Per-head sink logits added to the softmax denominator; NaN disables a head's sink.
/// @param scratch_scores Caller-owned scratch of length `>= seq_len` for raw scores.
/// @param scratch_probs Caller-owned scratch of length `>= seq_len` for exponentiated weights.
pub const Params = struct {
    q: []const f32,
    kv_k: []const f32,
    kv_v: []const f32,
    output: []f32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    attn_sinks: []const f32,
    scratch_scores: []f32,
    scratch_probs: []f32,
};

/// Compute single-query scaled dot-product attention with optional per-head softmax sinks.
/// For each query head: dots `q` against the cached keys, applies a `1/sqrt(head_dim)` scale,
/// max-subtracted softmax (folding in the sink if finite), then writes the value-weighted sum
/// to the matching slot of `output`. GQA is supported via `q_per_kv = n_heads / n_kv_heads`.
/// @param params Query, KV cache, attention sinks, scratch buffers, and output slice; see `Params`.
/// @returns `error.EmptyInput` when query or output is empty, `error.ShapeMismatch` when scratch
/// slots are smaller than `seq_len` or `head_dim` is zero, otherwise void.
pub fn run(params: Params) !void {
    if (params.q.len == 0 or params.output.len == 0) return error.EmptyInput;
    if (params.head_dim == 0) return error.ShapeMismatch;
    if (params.scratch_scores.len < params.seq_len) return error.ShapeMismatch;
    if (params.scratch_probs.len < params.seq_len) return error.ShapeMismatch;

    const q_per_kv = @max(params.n_heads / @max(params.n_kv_heads, 1), 1);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(params.head_dim)));

    for (0..params.n_heads) |h| {
        const kv_head = h / q_per_kv;
        const q_head = params.q[h * params.head_dim ..][0..params.head_dim];
        var max_score: f32 = -std.math.inf(f32);

        for (0..params.seq_len) |pos| {
            const kv_off = pos * params.n_kv_heads * params.head_dim;
            const k_head = params.kv_k[kv_off + kv_head * params.head_dim ..][0..params.head_dim];
            var dot: f32 = 0;
            for (q_head, k_head) |qv, kv| dot += qv * kv;
            const score = dot * scale;
            params.scratch_scores[pos] = score;
            if (score > max_score) max_score = score;
        }

        if (params.attn_sinks.len > h) {
            const sink = params.attn_sinks[h];
            if (!std.math.isNan(sink) and sink > max_score) max_score = sink;
        }

        var sum_exp: f32 = 0;
        if (params.attn_sinks.len > h) {
            const sink = params.attn_sinks[h];
            if (!std.math.isNan(sink)) sum_exp += @exp(sink - max_score);
        }
        for (0..params.seq_len) |pos| {
            const p = @exp(params.scratch_scores[pos] - max_score);
            params.scratch_probs[pos] = p;
            sum_exp += p;
        }

        const inv_sum = if (sum_exp > 0) 1.0 / sum_exp else 0;
        const out_head = params.output[h * params.head_dim ..][0..params.head_dim];
        @memset(out_head, 0);
        for (0..params.seq_len) |pos| {
            const weight = params.scratch_probs[pos] * inv_sum;
            const kv_off = pos * params.n_kv_heads * params.head_dim;
            const v_head = params.kv_v[kv_off + kv_head * params.head_dim ..][0..params.head_dim];
            for (out_head, v_head) |*out, vv| out.* += weight * vv;
        }
    }
}

test "flash attention produces identity attention for matching q and k" {
    const head_dim: u32 = 4;
    const n_heads: u32 = 1;
    const n_kv_heads: u32 = 1;
    const seq_len: u32 = 1;

    const q = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const kv_k = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const kv_v = [_]f32{ 0.5, 1.0, 1.5, 2.0 };
    var output = [_]f32{0} ** 4;
    var scores = [_]f32{0} ** 1;
    var probs = [_]f32{0} ** 1;
    const sinks = [_]f32{std.math.nan(f32)};

    try run(.{
        .q = &q,
        .kv_k = &kv_k,
        .kv_v = &kv_v,
        .output = &output,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .seq_len = seq_len,
        .attn_sinks = &sinks,
        .scratch_scores = &scores,
        .scratch_probs = &probs,
    });

    const expected_weight = 1.0;
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, @as(f32, @floatFromInt(i + 1)) * 0.5 * expected_weight), output[i], 0.00001);
    }
}
