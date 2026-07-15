//! C dequantization function bindings for Q4NX → float conversion.
//! Wraps dequant_q4nx.c so Zig can call dequant_i8_to_float and
//! dequant_i8_to_float_ex, then properly free the returned buffer.
const std = @import("std");

// ============================================================
// Extern declarations from dequant_q4nx.c (compiled as C source)
// ============================================================

/// Dequantize Q4NX I8 tensor to float. in_features defaults to 1024.
/// Returns row-major [out_rows × out_cols] float array allocated with calloc.
/// Caller must free with c_free().
extern fn dequant_i8_to_float(
    data: [*]const u8,
    i8_rows: c_int,
    out_rows: *c_int,
    out_cols: *c_int,
) callconv(.c) ?[*]f32;

/// Extended version with explicit in_features.
extern fn dequant_i8_to_float_ex(
    data: [*]const u8,
    i8_rows: c_int,
    in_features: c_int,
    out_rows: *c_int,
    out_cols: *c_int,
) callconv(.c) ?[*]f32;

/// Free the buffer returned by dequant_i8_to_float.
extern "c" fn free(ptr: ?*anyopaque) callconv(.c) void;

// ============================================================
// Safe Zig wrappers
// ============================================================

/// Dequantize a Q4NX I8 tensor to a Zig-managed slice.
/// The returned slice must be freed with allocator.free().
/// Returns [out_rows × out_cols] row-major f32.
pub fn dequantToSlice(allocator: std.mem.Allocator, data: []const u8, i8_rows: u32, in_features: u32) !struct { data: []f32, rows: u32, cols: u32 } {
    var out_rows: c_int = 0;
    var out_cols: c_int = 0;

    const result = dequant_i8_to_float_ex(
        data.ptr,
        @intCast(i8_rows),
        @intCast(in_features),
        &out_rows,
        &out_cols,
    );
    if (result == null) return error.DequantFailed;

    const rows = @as(u32, @intCast(out_rows));
    const cols = @as(u32, @intCast(out_cols));
    const len = rows * cols;

    // Copy to a Zig-allocated slice, then free the C buffer
    const slice = try allocator.alloc(f32, len);
    @memcpy(slice, result.?[0..len]);
    free(@ptrCast(result.?));

    return .{ .data = slice, .rows = rows, .cols = cols };
}

/// Dequantize with default in_features=1024.
pub fn dequantToSliceDefault(allocator: std.mem.Allocator, data: []const u8, i8_rows: u32) !struct { data: []f32, rows: u32, cols: u32 } {
    return dequantToSlice(allocator, data, i8_rows, 1024);
}
