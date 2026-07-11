//! T-CPU residual add + RMS norm implementation.
//! Computes: output = weight * rms_norm(x + residual)
//! @section Inference Runtime
const std = @import("std");

/// Inputs and outputs for one fused residual-add + RMS-norm call.
/// @param x Hidden state to normalize after residual addition.
/// @param residual Residual contribution added element-wise to `x` before normalization.
/// @param weight Per-channel learned scale applied after RMS normalization.
/// @param output Destination hidden state of length `>= x.len`.
/// @param eps Small constant added inside the square root to keep division numerically stable.
pub const Params = struct {
    x: []const f32,
    residual: []const f32,
    weight: []const f32,
    output: []f32,
    eps: f32,
};

/// Compute `output = weight * (x + residual) / sqrt(mean((x + residual)^2) + eps)`.
/// Fuses the post-attention/post-MLP residual add with the RMS norm so the sum lives only in
/// registers; matches the GPU kernel that the Vulkan path uses on the decode hot loop.
/// @param params Inputs, residual, learned scale, output slice, and `eps`; see `Params`.
/// @returns `error.EmptyInput` when inputs are zero-length, `error.ShapeMismatch` when any
/// companion slice is shorter than `x`, otherwise void.
pub fn run(params: Params) !void {
    if (params.x.len == 0 or params.output.len == 0) return error.EmptyInput;
    if (params.output.len < params.x.len) return error.ShapeMismatch;
    const n = params.x.len;
    if (params.residual.len < n or params.weight.len < n) return error.ShapeMismatch;

    var sq: f32 = 0;
    for (0..n) |i| {
        const v = params.x[i] + params.residual[i];
        sq += v * v;
    }
    const inv_rms = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(n)) + params.eps);
    for (0..n) |i| {
        const v = params.x[i] + params.residual[i];
        params.output[i] = v * inv_rms * params.weight[i];
    }
}

test "residual_rms_norm adds residual then normalizes" {
    const x = [_]f32{ 3.0, 4.0 };
    const residual = [_]f32{ 1.0, 0.0 };
    const weight = [_]f32{ 1.0, 1.0 };
    var output = [_]f32{ 0.0, 0.0 };

    try run(.{
        .x = &x,
        .residual = &residual,
        .weight = &weight,
        .output = &output,
        .eps = 0.0,
    });

    const expected_sq: f32 = 16.0 + 16.0;
    const inv_rms = 1.0 / @sqrt(expected_sq / 2.0);
    try std.testing.expectApproxEqAbs(4.0 * inv_rms, output[0], 0.00001);
    try std.testing.expectApproxEqAbs(4.0 * inv_rms, output[1], 0.00001);
}

test "residual_rms_norm with zero residual matches plain rms_norm" {
    const x = [_]f32{ 3.0, 4.0 };
    const residual = [_]f32{ 0.0, 0.0 };
    const weight = [_]f32{ 1.0, 0.5 };
    var output = [_]f32{ 0.0, 0.0 };

    try run(.{
        .x = &x,
        .residual = &residual,
        .weight = &weight,
        .output = &output,
        .eps = 0.0,
    });

    var sq: f32 = 0;
    for (x) |v| sq += v * v;
    const inv_rms = 1.0 / @sqrt(sq / 2.0);
    try std.testing.expectApproxEqAbs(3.0 * inv_rms * 1.0, output[0], 0.00001);
    try std.testing.expectApproxEqAbs(4.0 * inv_rms * 0.5, output[1], 0.00001);
}
