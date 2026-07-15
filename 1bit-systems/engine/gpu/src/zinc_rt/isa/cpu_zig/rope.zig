//! T-CPU RoPE (Rotary Positional Embedding) implementation.
//! Applies in-place rotary embeddings to query or key head data.
//! @section Inference Runtime
const std = @import("std");

/// Inputs for one in-place RoPE pass over a Q or K tensor.
/// @param data Mutable head-major buffer; head `h` lives at `data[h * stride ..][0..rope_dim]`.
/// @param stride Distance in floats between successive heads in `data`.
/// @param rope_dim Number of contiguous dimensions per head touched by RoPE (must be even).
/// @param n_heads Number of heads to rotate.
/// @param position Absolute token position used to compute the rotation angle.
/// @param inv_freq Inverse frequencies of length `rope_dim / 2`; missing entries default to zero (no rotation).
pub const Params = struct {
    data: []f32,
    stride: u32,
    rope_dim: u32,
    n_heads: u32,
    position: u32,
    inv_freq: []const f32,
};

/// Apply rotary positional embeddings in place to every head in `params.data`.
/// For each head and each frequency pair `(a, b) = (data[i], data[i + rope_dim/2])`, rotates by
/// `theta = position * inv_freq[i]`: writes `(a*cos - b*sin, a*sin + b*cos)`. Uses the half-rotation
/// layout (NeoX-style), matching the Vulkan kernels.
/// @param params Buffer, stride, rope dim, head count, position, and `inv_freq` table; see `Params`.
/// @returns `error.EmptyInput` when `data` is zero-length, otherwise void.
pub fn run(params: Params) !void {
    if (params.data.len == 0) return error.EmptyInput;
    apply(params.data, params.stride, params.rope_dim, params.n_heads, params.position, params.inv_freq);
}

fn apply(data: []f32, stride: u32, rope_dim: u32, n_heads: u32, position: u32, inv_freq: []const f32) void {
    const half = rope_dim / 2;
    for (0..n_heads) |h| {
        const base = h * stride;
        for (0..half) |i| {
            const freq = if (i < inv_freq.len) inv_freq[i] else 0.0;
            const theta = @as(f32, @floatFromInt(position)) * freq;
            const c = @cos(theta);
            const s = @sin(theta);
            const a = data[base + i];
            const b = data[base + i + half];
            data[base + i] = a * c - b * s;
            data[base + i + half] = a * s + b * c;
        }
    }
}

test "rope applies rotary embeddings to a single head" {
    const inv_freq = [_]f32{ 1.0, 0.5 };
    var data = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    try run(.{
        .data = &data,
        .stride = 2,
        .rope_dim = 2,
        .n_heads = 1,
        .position = 1,
        .inv_freq = &inv_freq,
    });
    const c0 = @cos(1.0);
    const s0 = @sin(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 * c0 - 0.0 * s0), data[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 * s0 + 0.0 * c0), data[1], 0.00001);
}

test "rope is identity at position zero" {
    const inv_freq = [_]f32{ 1.0, 0.5 };
    var data = [_]f32{ 3.0, -1.0, 2.0, 4.0 };
    try run(.{
        .data = &data,
        .stride = 2,
        .rope_dim = 2,
        .n_heads = 1,
        .position = 0,
        .inv_freq = &inv_freq,
    });
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), data[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), data[1], 0.00001);
}
