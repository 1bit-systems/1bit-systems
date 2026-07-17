//! T-CPU matrix-vector projection implementation.
//! Dequantizes one GGUF tensor row at a time and computes a scalar matvec.
//! @section Inference Runtime
const std = @import("std");
const dequant = @import("dequant.zig");
const gguf = @import("gguf");

/// Inputs and outputs for one matrix-vector projection.
/// @param raw_data Raw GGUF tensor bytes for the weight matrix `[rows, cols]`.
/// @param tensor_type GGML quantization format of `raw_data` (forwarded to `dequant.row`).
/// @param input Input vector of length `cols`.
/// @param row_scratch Caller-owned scratch of length `>= cols` used to materialize one dequantized row.
/// @param output Destination vector of length `rows`; row `i` of the matrix lands in `output[i]`.
/// @param accumulate When true, add into `output` instead of overwriting (e.g. for MoE expert mixing).
pub const Params = struct {
    raw_data: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    row_scratch: []f32,
    output: []f32,
    accumulate: bool = false,
};

/// Compute `output = W * input` (or `output += W * input` when `accumulate` is set) one row at a time.
/// Each row of the GGUF matrix is dequantized into `row_scratch` and dotted against `input`.
/// @param params Tensor data, input vector, scratch row, and output slice; see `Params`.
/// @returns `error.EmptyInput` when input or output is empty, `error.ShapeMismatch` when
/// `row_scratch` is shorter than `input`, otherwise void.
pub fn run(params: Params) !void {
    if (params.input.len == 0 or params.output.len == 0) return error.EmptyInput;
    if (params.row_scratch.len < params.input.len) return error.ShapeMismatch;

    const cols: u32 = @intCast(params.input.len);
    const scratch = params.row_scratch[0..params.input.len];
    for (params.output, 0..) |*out, row_index| {
        try dequant.row(params.raw_data, @intCast(row_index), cols, params.tensor_type, scratch);
        var acc: f32 = 0.0;
        for (scratch, params.input) |w, x| {
            acc += w * x;
        }
        if (params.accumulate) {
            out.* += acc;
        } else {
            out.* = acc;
        }
    }
}

test "matvec computes f32 projection" {
    const raw = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
    };
    const input = [_]f32{ 2.0, 3.0 };
    var scratch = [_]f32{ 0.0, 0.0 };
    var output = [_]f32{ 0.0, 0.0, 0.0 };

    try run(.{
        .raw_data = std.mem.sliceAsBytes(&raw),
        .tensor_type = .f32,
        .input = &input,
        .row_scratch = &scratch,
        .output = &output,
    });

    try std.testing.expectEqual(@as(f32, 2.0), output[0]);
    try std.testing.expectEqual(@as(f32, 3.0), output[1]);
    try std.testing.expectEqual(@as(f32, 5.0), output[2]);
}

test "matvec can accumulate into output" {
    const raw = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
    };
    const input = [_]f32{ 2.0, 3.0 };
    var scratch = [_]f32{ 0.0, 0.0 };
    var output = [_]f32{ 10.0, 20.0 };

    try run(.{
        .raw_data = std.mem.sliceAsBytes(&raw),
        .tensor_type = .f32,
        .input = &input,
        .row_scratch = &scratch,
        .output = &output,
        .accumulate = true,
    });

    try std.testing.expectEqual(@as(f32, 12.0), output[0]);
    try std.testing.expectEqual(@as(f32, 23.0), output[1]);
}
