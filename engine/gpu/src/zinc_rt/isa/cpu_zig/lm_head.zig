//! T-CPU LM_HEAD implementation.
//! Projects hidden state through a GGUF output matrix and writes logits.
//! @section Inference Runtime
const std = @import("std");
const dequant = @import("dequant.zig");
const gguf = @import("gguf");

/// Inputs and outputs for one LM_HEAD call.
/// @param raw_data Raw GGUF tensor bytes for the output matrix `[vocab_size, hidden_dim]`.
/// @param tensor_type GGML quantization format of `raw_data` (forwarded to `dequant.row`).
/// @param hidden Final hidden state of length `hidden_dim`.
/// @param row_scratch Caller-owned scratch buffer of length exactly `hidden_dim` for one dequantized row.
/// @param logits Destination vector of length `vocab_size`; row `i` of the matrix maps to `logits[i]`.
pub const Params = struct {
    raw_data: []const u8,
    tensor_type: gguf.GGMLType,
    hidden: []const f32,
    row_scratch: []f32,
    logits: []f32,
};

/// Project the hidden state through every row of the GGUF output matrix to produce vocab logits.
/// Rows are dequantized one at a time into `row_scratch` and dot-multiplied with `hidden`.
/// @param params Tensor data, hidden state, scratch row, and logits slice; see `Params`.
/// @returns `error.EmptyInput` when either `hidden` or `logits` is empty, `error.ShapeMismatch` when
/// `row_scratch` is not exactly `hidden.len`, otherwise void.
pub fn run(params: Params) !void {
    if (params.hidden.len == 0 or params.logits.len == 0) return error.EmptyInput;
    if (params.row_scratch.len != params.hidden.len) return error.ShapeMismatch;

    const cols: u32 = @intCast(params.hidden.len);
    for (params.logits, 0..) |*logit, row_index| {
        try dequant.row(params.raw_data, @intCast(row_index), cols, params.tensor_type, params.row_scratch);
        var acc: f32 = 0.0;
        for (params.row_scratch, params.hidden) |w, h| {
            acc += w * h;
        }
        logit.* = acc;
    }
}

test "lm_head computes f32 logits" {
    const raw = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
    };
    const hidden = [_]f32{ 2.0, 3.0 };
    var scratch = [_]f32{ 0.0, 0.0 };
    var logits = [_]f32{ 0.0, 0.0, 0.0 };

    try run(.{
        .raw_data = std.mem.sliceAsBytes(&raw),
        .tensor_type = .f32,
        .hidden = &hidden,
        .row_scratch = &scratch,
        .logits = &logits,
    });

    try std.testing.expectEqual(@as(f32, 2.0), logits[0]);
    try std.testing.expectEqual(@as(f32, 3.0), logits[1]);
    try std.testing.expectEqual(@as(f32, 5.0), logits[2]);
}
