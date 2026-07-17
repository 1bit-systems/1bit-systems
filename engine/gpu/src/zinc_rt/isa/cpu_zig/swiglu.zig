//! T-CPU SwiGLU implementation.
//! This is the scalar reference activation used by MoE and dense MLP paths
//! before tier-specific kernels are trusted.
//! @section Inference Runtime
const std = @import("std");

/// Inputs and outputs for one SwiGLU activation.
/// @param gate Gate-projection vector fed through SiLU.
/// @param up Up-projection vector multiplied by the SiLU-gated values.
/// @param output Destination vector; all three slices must be the same length.
pub const Params = struct {
    gate: []const f32,
    up: []const f32,
    output: []f32,
};

/// Compute `output[i] = silu(gate[i]) * up[i]` where `silu(x) = x / (1 + exp(-x))`.
/// Reference SwiGLU used by MoE and dense MLP paths to validate tier-specific kernels.
/// @param params Gate, up, and output slices of equal length; see `Params`.
/// @returns `error.ShapeMismatch` when the three slices differ in length, otherwise void.
pub fn run(params: Params) !void {
    if (params.gate.len != params.up.len or params.output.len != params.gate.len) {
        return error.ShapeMismatch;
    }

    for (params.gate, params.up, params.output) |gate, up, *out| {
        const silu = gate / (1.0 + std.math.exp(-gate));
        out.* = silu * up;
    }
}

test "swiglu multiplies silu gate by up projection" {
    const gate = [_]f32{ 0.0, 2.0 };
    const up = [_]f32{ 3.0, 4.0 };
    var output = [_]f32{ 0.0, 0.0 };

    try run(.{ .gate = &gate, .up = &up, .output = &output });

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0463767), output[1], 0.00001);
}
