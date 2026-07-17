//! T-CPU sigmoid-gated multiply implementation.
//! Computes: output[i] = sigmoid(gate[i]) * x[i]
//! Used for attention gating (Q-gate) and SSM gated norm.
//! @section Inference Runtime
const std = @import("std");

/// Inputs and outputs for one sigmoid-gated multiply.
/// @param gate Pre-activation gate values; sigmoid is applied element-wise.
/// @param x Companion values multiplied by the sigmoid of `gate`.
/// @param output Destination vector of length `>= gate.len`.
pub const Params = struct {
    gate: []const f32,
    x: []const f32,
    output: []f32,
};

/// Compute `output[i] = sigmoid(gate[i]) * x[i]` for every gate element.
/// Used by attention Q-gating and SSM gated-norm paths where a learned scalar selects how much of
/// `x` to pass through.
/// @param params Gate, value, and output slices; see `Params`.
/// @returns `error.EmptyInput` when `gate` is empty, `error.ShapeMismatch` when `output` or `x`
/// is shorter than `gate`, otherwise void.
pub fn run(params: Params) !void {
    if (params.gate.len == 0) return error.EmptyInput;
    if (params.output.len < params.gate.len) return error.ShapeMismatch;
    if (params.x.len < params.gate.len) return error.ShapeMismatch;
    for (0..params.gate.len) |i| {
        params.output[i] = sigmoid(params.gate[i]) * params.x[i];
    }
}

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

test "sigmoid_mul gates input by sigmoid of gate" {
    const gate = [_]f32{ 0.0, 100.0, -100.0 };
    const x = [_]f32{ 4.0, 2.0, 6.0 };
    var output = [_]f32{ 0.0, 0.0, 0.0 };

    try run(.{
        .gate = &gate,
        .x = &x,
        .output = &output,
    });

    try std.testing.expectApproxEqAbs(@as(f32, 0.5 * 4.0), output[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 * 2.0), output[1], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[2], 0.00001);
}
