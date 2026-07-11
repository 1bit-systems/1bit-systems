//! T-CPU element-wise vector addition implementation.
//! Computes: output[i] = a[i] + b[i]
//! @section Inference Runtime
const std = @import("std");

/// Inputs and outputs for one element-wise vector addition.
/// @param a First operand vector.
/// @param b Second operand vector of length `>= a.len`.
/// @param output Destination vector of length `>= a.len`.
pub const Params = struct {
    a: []const f32,
    b: []const f32,
    output: []f32,
};

/// Compute `output[i] = a[i] + b[i]` for the first `a.len` elements.
/// Trailing elements of `b` and `output` are ignored, so callers may pass over-sized buffers.
/// @param params Operand and destination slices; see `Params`.
/// @returns `error.EmptyInput` when `a` is empty, `error.ShapeMismatch` when `output` or `b`
/// is shorter than `a`, otherwise void.
pub fn run(params: Params) !void {
    if (params.a.len == 0) return error.EmptyInput;
    if (params.output.len < params.a.len) return error.ShapeMismatch;
    if (params.b.len < params.a.len) return error.ShapeMismatch;
    for (0..params.a.len) |i| {
        params.output[i] = params.a[i] + params.b[i];
    }
}

test "vadd computes element-wise sum" {
    const a = [_]f32{ 1.0, -2.0, 3.0 };
    const b = [_]f32{ 4.0, 5.0, -1.0 };
    var output = [_]f32{ 0.0, 0.0, 0.0 };

    try run(.{
        .a = &a,
        .b = &b,
        .output = &output,
    });

    try std.testing.expectEqual(@as(f32, 5.0), output[0]);
    try std.testing.expectEqual(@as(f32, 3.0), output[1]);
    try std.testing.expectEqual(@as(f32, 2.0), output[2]);
}
