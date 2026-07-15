//! T-CPU EMBED implementation.
//! Reads one token row from a GGUF tensor into f32 hidden state.
//! @section Inference Runtime
const dequant = @import("dequant.zig");
const gguf = @import("gguf");

/// Inputs and outputs for one EMBED call.
/// @param raw_data Raw GGUF tensor bytes for the embedding matrix `[vocab_size, hidden_dim]`.
/// @param tensor_type GGML quantization format of `raw_data` (forwarded to `dequant.row`).
/// @param token_id Row index to fetch; must be `< vocab_size`.
/// @param hidden_dim Number of columns per row; also the required length of `output`.
/// @param vocab_size Number of embedding rows in `raw_data`.
/// @param output Destination hidden-state slice of length exactly `hidden_dim`.
pub const Params = struct {
    raw_data: []const u8,
    tensor_type: gguf.GGMLType,
    token_id: u32,
    hidden_dim: u32,
    vocab_size: u32,
    output: []f32,
};

/// Dequantize the row at `params.token_id` of the embedding matrix into `params.output`.
/// Thin wrapper over `dequant.row` that validates the token index and output shape.
/// @param params Token id, matrix shape, and destination slice; see `Params`.
/// @returns `error.TokenOutOfRange` when the token id is past `vocab_size`, `error.ShapeMismatch` when
/// the output slice does not match `hidden_dim`, otherwise void.
pub fn run(params: Params) !void {
    if (params.token_id >= params.vocab_size) return error.TokenOutOfRange;
    if (params.output.len != params.hidden_dim) return error.ShapeMismatch;
    try dequant.row(params.raw_data, params.token_id, params.hidden_dim, params.tensor_type, params.output);
}

test "embed reads f32 token row" {
    const raw = [_]f32{
        1.0, 2.0,
        3.0, 4.0,
    };
    var output = [_]f32{ 0.0, 0.0 };
    try run(.{
        .raw_data = @import("std").mem.sliceAsBytes(&raw),
        .tensor_type = .f32,
        .token_id = 1,
        .hidden_dim = 2,
        .vocab_size = 2,
        .output = &output,
    });
    try @import("std").testing.expectEqual(@as(f32, 3.0), output[0]);
    try @import("std").testing.expectEqual(@as(f32, 4.0), output[1]);
}
