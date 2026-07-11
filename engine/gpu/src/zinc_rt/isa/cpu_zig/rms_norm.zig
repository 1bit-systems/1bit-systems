//! T-CPU RMS_NORM implementation.
//! This is the scalar reference for RMS normalization and intentionally favors
//! exactness and readable shape checks over throughput.
//! @section Inference Runtime
const std = @import("std");

/// Inputs and outputs for one RMS_NORM call.
/// @param input Vector to normalize.
/// @param weight Per-channel learned scale; must match `input` in length.
/// @param output Destination vector; must match `input` in length.
/// @param eps Small constant added inside the square root to keep division numerically stable.
pub const Params = struct {
    input: []const f32,
    weight: []const f32,
    output: []f32,
    eps: f32,
};

/// Compute `output[i] = weight[i] * input[i] / sqrt(mean(input^2) + eps)`.
/// Scalar reference implementation; favors readable shape checks and bit-stable math over throughput.
/// @param params Input, learned scale, output slice, and `eps`; see `Params`.
/// @returns `error.EmptyInput` when `input` is zero-length, `error.ShapeMismatch` when `weight` or
/// `output` do not exactly match `input.len`, otherwise void.
pub fn run(params: Params) !void {
    if (params.input.len == 0) return error.EmptyInput;
    if (params.weight.len != params.input.len or params.output.len != params.input.len) {
        return error.ShapeMismatch;
    }

    var sum_sq: f32 = 0;
    for (params.input) |value| {
        sum_sq += value * value;
    }
    const mean_sq = sum_sq / @as(f32, @floatFromInt(params.input.len));
    const scale = 1.0 / @sqrt(mean_sq + params.eps);

    for (params.input, params.weight, params.output) |value, weight, *out| {
        out.* = value * scale * weight;
    }
}

test "rms_norm computes weighted normalization" {
    const input = [_]f32{ 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 0.5 };
    var output = [_]f32{ 0.0, 0.0 };

    try run(.{
        .input = &input,
        .weight = &weight,
        .output = &output,
        .eps = 0.0,
    });

    try std.testing.expectApproxEqAbs(@as(f32, 0.84852815), output[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5656854), output[1], 0.00001);
}
