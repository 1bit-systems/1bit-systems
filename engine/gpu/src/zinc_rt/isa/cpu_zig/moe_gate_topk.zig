//! T-CPU MOE_GATE_TOPK implementation.
//! Computes router logits from a GGUF gate matrix, then selects and normalizes
//! the active expert weights using the same routing rules as the Vulkan path.
//! @section Inference Runtime
const std = @import("std");
const dequant = @import("dequant.zig");
const gguf = @import("gguf");

const max_experts = 256;

/// Selection rule applied after the router projection to convert logits into expert weights.
pub const RoutingRule = enum {
    /// Qwen/Gemma-style routing: softmax over all experts, select top-k,
    /// then renormalize the selected expert probabilities to sum to 1.
    softmax_all,
    /// GPT-OSS routing: select top-k raw logits, then softmax over that set.
    softmax_selected,
};

/// Inputs and outputs for one MoE gate + top-k call.
/// @param raw_data Raw GGUF tensor bytes for the router matrix `[num_experts, hidden_dim]`.
/// @param tensor_type GGML quantization format of `raw_data` (forwarded to `dequant.row`).
/// @param hidden Hidden state of length `hidden_dim`.
/// @param row_scratch Caller-owned scratch of length `>= hidden_dim` for one dequantized router row.
/// @param logits Destination router logits of length `num_experts`; capped at 256 experts.
/// @param k Number of experts to select; must satisfy `1 <= k <= logits.len`.
/// @param output_ids Destination expert indices of length `>= k`.
/// @param output_weights Destination per-expert weights of length `>= k`, summing to 1 after the call.
/// @param rule Routing rule that decides how the top-k weights are computed (see `RoutingRule`).
pub const Params = struct {
    raw_data: []const u8,
    tensor_type: gguf.GGMLType,
    hidden: []const f32,
    row_scratch: []f32,
    logits: []f32,
    k: u32,
    output_ids: []u32,
    output_weights: []f32,
    rule: RoutingRule = .softmax_all,
};

/// Project the hidden state through the router matrix, then select and normalize the top-k experts.
/// First fills `logits` row by row (matvec via `dequant.row`), then dispatches on `rule` to either
/// `softmax_all` (softmax across all experts, pick top-k, renormalize) or `softmax_selected`
/// (pick top-k by raw logit, softmax across that subset).
/// @param params Router weights, hidden state, scratch buffers, selection size, and outputs; see `Params`.
/// @returns `error.EmptyInput` for empty inputs, `error.TooManyExperts` when `logits.len > 256`,
/// `error.InvalidTopK` when `k` is zero or larger than the expert count, `error.ShapeMismatch` when
/// scratch or output slices are too small, otherwise void.
pub fn run(params: Params) !void {
    if (params.hidden.len == 0 or params.logits.len == 0) return error.EmptyInput;
    if (params.logits.len > max_experts) return error.TooManyExperts;
    if (params.k == 0 or params.k > params.logits.len) return error.InvalidTopK;
    if (params.row_scratch.len < params.hidden.len) return error.ShapeMismatch;
    if (params.output_ids.len < params.k or params.output_weights.len < params.k) return error.ShapeMismatch;

    const cols: u32 = @intCast(params.hidden.len);
    const scratch = params.row_scratch[0..params.hidden.len];
    for (params.logits, 0..) |*logit, row_index| {
        try dequant.row(params.raw_data, @intCast(row_index), cols, params.tensor_type, scratch);
        var acc: f32 = 0.0;
        for (scratch, params.hidden) |w, h| {
            acc += w * h;
        }
        logit.* = acc;
    }

    switch (params.rule) {
        .softmax_all => topKSoftmaxAll(
            params.logits,
            params.k,
            params.output_ids[0..params.k],
            params.output_weights[0..params.k],
        ),
        .softmax_selected => topKSoftmaxSelected(
            params.logits,
            params.k,
            params.output_ids[0..params.k],
            params.output_weights[0..params.k],
        ),
    }
}

fn topKSoftmaxAll(logits: []const f32, k: u32, out_ids: []u32, out_weights: []f32) void {
    var max_val: f32 = -std.math.inf(f32);
    for (logits) |value| {
        if (value > max_val) max_val = value;
    }

    var probs: [max_experts]f32 = undefined;
    var sum: f32 = 0.0;
    for (logits, 0..) |value, i| {
        probs[i] = @exp(value - max_val);
        sum += probs[i];
    }
    if (sum > 0.0) {
        for (0..logits.len) |i| probs[i] /= sum;
    }

    var used = [_]bool{false} ** max_experts;
    for (0..k) |ki| {
        var best_idx: u32 = 0;
        var best_val: f32 = -1.0;
        for (0..logits.len) |i| {
            if (!used[i] and probs[i] > best_val) {
                best_val = probs[i];
                best_idx = @intCast(i);
            }
        }
        out_ids[ki] = best_idx;
        out_weights[ki] = best_val;
        used[best_idx] = true;
    }

    var selected_sum: f32 = 0.0;
    for (0..k) |i| selected_sum += out_weights[i];
    if (selected_sum > 0.0) {
        for (0..k) |i| out_weights[i] /= selected_sum;
    }
}

fn topKSoftmaxSelected(logits: []const f32, k: u32, out_ids: []u32, out_weights: []f32) void {
    var used = [_]bool{false} ** max_experts;
    for (0..k) |ki| {
        var best_idx: u32 = 0;
        var best_val: f32 = -std.math.inf(f32);
        for (0..logits.len) |i| {
            if (!used[i] and logits[i] > best_val) {
                best_val = logits[i];
                best_idx = @intCast(i);
            }
        }
        out_ids[ki] = best_idx;
        out_weights[ki] = logits[best_idx];
        used[best_idx] = true;
    }

    var max_selected: f32 = -std.math.inf(f32);
    for (0..k) |i| {
        if (out_weights[i] > max_selected) max_selected = out_weights[i];
    }

    var sum: f32 = 0.0;
    for (0..k) |i| {
        out_weights[i] = @exp(out_weights[i] - max_selected);
        sum += out_weights[i];
    }
    if (sum > 0.0) {
        for (0..k) |i| out_weights[i] /= sum;
    }
}

test "moe_gate_topk projects router logits and normalizes selected experts" {
    const router_rows = [_]f32{
        1.0, 0.0,
        0.0, 3.0,
        2.0, 2.0,
    };
    const hidden = [_]f32{ 1.0, 2.0 };
    var scratch = [_]f32{ 0.0, 0.0 };
    var logits = [_]f32{ 0.0, 0.0, 0.0 };
    var ids = [_]u32{ 0, 0 };
    var weights = [_]f32{ 0.0, 0.0 };

    try run(.{
        .raw_data = std.mem.sliceAsBytes(&router_rows),
        .tensor_type = .f32,
        .hidden = &hidden,
        .row_scratch = &scratch,
        .logits = &logits,
        .k = 2,
        .output_ids = &ids,
        .output_weights = &weights,
    });

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), logits[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), logits[1], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), logits[2], 0.00001);
    try std.testing.expectEqual(@as(u32, 1), ids[0]);
    try std.testing.expectEqual(@as(u32, 2), ids[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), weights[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), weights[1], 0.00001);
}

test "moe_gate_topk supports selected-logit softmax routing" {
    const router_rows = [_]f32{
        1.0,
        5.0,
        3.0,
    };
    const hidden = [_]f32{1.0};
    var scratch = [_]f32{0.0};
    var logits = [_]f32{ 0.0, 0.0, 0.0 };
    var ids = [_]u32{ 0, 0 };
    var weights = [_]f32{ 0.0, 0.0 };

    try run(.{
        .raw_data = std.mem.sliceAsBytes(&router_rows),
        .tensor_type = .f32,
        .hidden = &hidden,
        .row_scratch = &scratch,
        .logits = &logits,
        .k = 2,
        .output_ids = &ids,
        .output_weights = &weights,
        .rule = .softmax_selected,
    });

    try std.testing.expectEqual(@as(u32, 1), ids[0]);
    try std.testing.expectEqual(@as(u32, 2), ids[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.880797), weights[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.119203), weights[1], 0.00001);
}
