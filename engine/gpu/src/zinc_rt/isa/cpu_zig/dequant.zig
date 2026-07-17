//! Shared scalar GGML dequantization helpers for T-CPU kernels.
//! These helpers intentionally mirror the Vulkan backend's CPU diagnostic
//! dequantization so M0 can compare host-side ZINC_RT ops against it.
//! @section Inference Runtime
const std = @import("std");
const gguf = @import("gguf");

const GGMLType = gguf.GGMLType;

fn getScaleMinK4(j: usize, scales: []const u8) struct { sc: u8, m: u8 } {
    if (j < 4) {
        return .{ .sc = scales[j] & 63, .m = scales[j + 4] & 63 };
    }
    return .{
        .sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
        .m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
    };
}

/// Dequantize one row of a GGML tensor into f32 lanes.
/// Dispatches on `tensor_type` and writes the first `cols` entries of `output`. Supports the
/// formats used by ZINC weights today: `.f32`, `.f16`, `.bf16`, `.q8_0`, `.q4_0`, `.q5_1`,
/// `.q4_k`, `.q5_k`, `.q6_k`, and `.mxfp4`.
/// @param raw_data Raw tensor bytes for the full matrix.
/// @param row_index Zero-based row to materialize.
/// @param cols Number of columns per row; quantized formats require this to be a multiple of
/// their block size (32 for q4_0/q5_1/q8_0/mxfp4, 256 for q4_k/q5_k/q6_k).
/// @param tensor_type GGML quantization tag selecting the decode path.
/// @param output Destination slice; must be at least `cols` long.
/// @returns `error.OutputTooSmall` when `output.len < cols`, `error.EmptyInput` when `cols == 0`,
/// `error.InputTooSmall` when the row would overrun `raw_data`, `error.UnsupportedShape` on bad
/// alignment, or `error.UnsupportedTensorType` for formats not handled here.
pub fn row(raw_data: []const u8, row_index: u32, cols: u32, tensor_type: GGMLType, output: []f32) !void {
    if (output.len < cols) return error.OutputTooSmall;
    if (cols == 0) return error.EmptyInput;

    switch (tensor_type) {
        .f32 => {
            const row_bytes = @as(usize, cols) * @sizeOf(f32);
            const offset = @as(usize, row_index) * row_bytes;
            if (offset + row_bytes > raw_data.len) return error.InputTooSmall;
            const src: [*]const f32 = @ptrCast(@alignCast(raw_data[offset..].ptr));
            @memcpy(output[0..cols], src[0..cols]);
        },
        .f16, .bf16 => {
            const row_bytes = @as(usize, cols) * @sizeOf(u16);
            const offset = @as(usize, row_index) * row_bytes;
            if (offset + row_bytes > raw_data.len) return error.InputTooSmall;
            for (0..cols) |i| {
                const byte_off = offset + i * 2;
                const bits = std.mem.readInt(u16, raw_data[byte_off..][0..2], .little);
                output[i] = switch (tensor_type) {
                    .f16 => @floatCast(@as(f16, @bitCast(bits))),
                    .bf16 => @bitCast(@as(u32, bits) << 16),
                    else => unreachable,
                };
            }
        },
        .q8_0 => {
            if (cols % 32 != 0) return error.UnsupportedShape;
            const block_size: usize = 32;
            const bpb: usize = 34;
            const bpr = @as(usize, cols) / block_size;
            const row_off = @as(usize, row_index) * bpr * bpb;
            if (row_off + bpr * bpb > raw_data.len) return error.InputTooSmall;

            var out_i: usize = 0;
            for (0..bpr) |b| {
                const bo = row_off + b * bpb;
                const scale_bits = std.mem.readInt(u16, raw_data[bo..][0..2], .little);
                const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));
                for (0..block_size) |j| {
                    const v: i8 = @bitCast(raw_data[bo + 2 + j]);
                    output[out_i] = @as(f32, @floatFromInt(v)) * scale;
                    out_i += 1;
                }
            }
        },
        .q4_0 => {
            if (cols % 32 != 0) return error.UnsupportedShape;
            const bpb: usize = 18;
            const bpr = @as(usize, cols) / 32;
            const row_off = @as(usize, row_index) * bpr * bpb;
            if (row_off + bpr * bpb > raw_data.len) return error.InputTooSmall;

            var out_i: usize = 0;
            for (0..bpr) |b| {
                const bo = row_off + b * bpb;
                const scale_bits = std.mem.readInt(u16, raw_data[bo..][0..2], .little);
                const d: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));
                for (0..16) |j| {
                    const byte = raw_data[bo + 2 + j];
                    const q_lo: i32 = @as(i32, byte & 0x0F) - 8;
                    const q_hi: i32 = @as(i32, byte >> 4) - 8;
                    output[out_i + j] = @as(f32, @floatFromInt(q_lo)) * d;
                    output[out_i + j + 16] = @as(f32, @floatFromInt(q_hi)) * d;
                }
                out_i += 32;
            }
        },
        .q5_1 => {
            // Block layout: 2B delta (f16) + 2B min (f16) + 4B qh (1 bit/elt) +
            // 16B qs (nibbles); 32 5-bit unsigned quants per block.
            //   y[l]    = (qs[l] & 0xF | xh_0) * d + m
            //   y[l+16] = (qs[l] >> 4   | xh_1) * d + m
            // where xh_0 = bit l of qh shifted to bit 4, xh_1 = bit (l+16).
            if (cols % 32 != 0) return error.UnsupportedShape;
            const bpb: usize = 24;
            const bpr = @as(usize, cols) / 32;
            const row_off = @as(usize, row_index) * bpr * bpb;
            if (row_off + bpr * bpb > raw_data.len) return error.InputTooSmall;

            var out_i: usize = 0;
            for (0..bpr) |b| {
                const bo = row_off + b * bpb;
                const d_bits = std.mem.readInt(u16, raw_data[bo..][0..2], .little);
                const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
                const m_bits = std.mem.readInt(u16, raw_data[bo + 2 ..][0..2], .little);
                const m: f32 = @floatCast(@as(f16, @bitCast(m_bits)));
                const qh: u32 = std.mem.readInt(u32, raw_data[bo + 4 ..][0..4], .little);
                const qs = raw_data[bo + 8 .. bo + 24];
                for (0..16) |l| {
                    const xh_0: u8 = @intCast(((qh >> @intCast(l)) & 1) << 4);
                    const xh_1: u8 = @intCast(((qh >> @intCast(l + 16)) & 1) << 4);
                    const x0: u8 = (qs[l] & 0x0F) | xh_0;
                    const x1: u8 = (qs[l] >> 4) | xh_1;
                    output[out_i + l] = @as(f32, @floatFromInt(x0)) * d + m;
                    output[out_i + l + 16] = @as(f32, @floatFromInt(x1)) * d + m;
                }
                out_i += 32;
            }
        },
        .q4_k => {
            if (cols % 256 != 0) return error.UnsupportedShape;
            const bpb: usize = 144;
            const bpr = @as(usize, cols) / 256;
            const row_off = @as(usize, row_index) * bpr * bpb;
            if (row_off + bpr * bpb > raw_data.len) return error.InputTooSmall;

            var out_i: usize = 0;
            for (0..bpr) |bi| {
                const bb = row_off + bi * bpb;
                const d_bits = std.mem.readInt(u16, raw_data[bb..][0..2], .little);
                const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
                const dm_bits = std.mem.readInt(u16, raw_data[bb + 2 ..][0..2], .little);
                const dmin: f32 = @floatCast(@as(f16, @bitCast(dm_bits)));
                const scales = raw_data[bb + 4 .. bb + 16];
                const qs = raw_data[bb + 16 .. bb + 144];

                var is: usize = 0;
                var qo: usize = 0;
                for (0..4) |_| {
                    const sm0 = getScaleMinK4(is, scales);
                    const d1 = d * @as(f32, @floatFromInt(sm0.sc));
                    const m1 = dmin * @as(f32, @floatFromInt(sm0.m));
                    const sm1 = getScaleMinK4(is + 1, scales);
                    const d2 = d * @as(f32, @floatFromInt(sm1.sc));
                    const m2 = dmin * @as(f32, @floatFromInt(sm1.m));

                    for (0..32) |l| {
                        output[out_i] = d1 * @as(f32, @floatFromInt(qs[qo + l] & 0xF)) - m1;
                        out_i += 1;
                    }
                    for (0..32) |l| {
                        output[out_i] = d2 * @as(f32, @floatFromInt(qs[qo + l] >> 4)) - m2;
                        out_i += 1;
                    }
                    qo += 32;
                    is += 2;
                }
            }
        },
        .q5_k => {
            if (cols % 256 != 0) return error.UnsupportedShape;
            const bpb: usize = 176;
            const bpr = @as(usize, cols) / 256;
            const row_off = @as(usize, row_index) * bpr * bpb;
            if (row_off + bpr * bpb > raw_data.len) return error.InputTooSmall;

            var out_i: usize = 0;
            for (0..bpr) |bi| {
                const bb = row_off + bi * bpb;
                const d_bits = std.mem.readInt(u16, raw_data[bb..][0..2], .little);
                const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
                const dm_bits = std.mem.readInt(u16, raw_data[bb + 2 ..][0..2], .little);
                const dmin: f32 = @floatCast(@as(f16, @bitCast(dm_bits)));
                const scales = raw_data[bb + 4 .. bb + 16];
                const qh = raw_data[bb + 16 .. bb + 48];
                const qs = raw_data[bb + 48 .. bb + 176];

                var is: usize = 0;
                for (0..4) |j| {
                    const sm0 = getScaleMinK4(is, scales);
                    const d1 = d * @as(f32, @floatFromInt(sm0.sc));
                    const m1 = dmin * @as(f32, @floatFromInt(sm0.m));
                    const sm1 = getScaleMinK4(is + 1, scales);
                    const d2 = d * @as(f32, @floatFromInt(sm1.sc));
                    const m2 = dmin * @as(f32, @floatFromInt(sm1.m));

                    for (0..32) |l| {
                        const ql_lo = qs[j * 32 + l] & 0xF;
                        const ql_hi = qs[j * 32 + l] >> 4;
                        const hb_lo = (qh[l] >> @intCast(j * 2)) & 1;
                        const hb_hi = (qh[l] >> @intCast(j * 2 + 1)) & 1;
                        output[out_i + l] = d1 * @as(f32, @floatFromInt(ql_lo | (hb_lo << 4))) - m1;
                        output[out_i + 32 + l] = d2 * @as(f32, @floatFromInt(ql_hi | (hb_hi << 4))) - m2;
                    }
                    out_i += 64;
                    is += 2;
                }
            }
        },
        .q6_k => {
            if (cols % 256 != 0) return error.UnsupportedShape;
            const bpb: usize = 210;
            const bpr = @as(usize, cols) / 256;
            const row_off = @as(usize, row_index) * bpr * bpb;
            if (row_off + bpr * bpb > raw_data.len) return error.InputTooSmall;

            var out_i: usize = 0;
            for (0..bpr) |b| {
                const bb = row_off + b * bpb;
                const d_bits = std.mem.readInt(u16, raw_data[bb + 208 ..][0..2], .little);
                const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));

                var ql_o: usize = bb;
                var qh_o: usize = bb + 128;
                var sc_o: usize = bb + 192;

                for (0..2) |_| {
                    for (0..32) |l| {
                        const is = l / 16;
                        const ql_lo = raw_data[ql_o + l];
                        const ql_hi = raw_data[ql_o + l + 32];
                        const qh_v = raw_data[qh_o + l];

                        const rq1: u8 = (ql_lo & 0xF) | (((qh_v >> 0) & 3) << 4);
                        const rq2: u8 = (ql_hi & 0xF) | (((qh_v >> 2) & 3) << 4);
                        const rq3: u8 = (ql_lo >> 4) | (((qh_v >> 4) & 3) << 4);
                        const rq4: u8 = (ql_hi >> 4) | (((qh_v >> 6) & 3) << 4);

                        const q1: f32 = @floatFromInt(@as(i16, @intCast(rq1)) - 32);
                        const q2: f32 = @floatFromInt(@as(i16, @intCast(rq2)) - 32);
                        const q3: f32 = @floatFromInt(@as(i16, @intCast(rq3)) - 32);
                        const q4: f32 = @floatFromInt(@as(i16, @intCast(rq4)) - 32);

                        const s0: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_o + is])));
                        const s2: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_o + is + 2])));
                        const s4: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_o + is + 4])));
                        const s6: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_o + is + 6])));

                        output[out_i + l + 0] = d * s0 * q1;
                        output[out_i + l + 32] = d * s2 * q2;
                        output[out_i + l + 64] = d * s4 * q3;
                        output[out_i + l + 96] = d * s6 * q4;
                    }
                    ql_o += 64;
                    qh_o += 32;
                    sc_o += 8;
                    out_i += 128;
                }
            }
        },
        .mxfp4 => {
            if (cols % 32 != 0) return error.UnsupportedShape;
            const bpb: usize = 17;
            const bpr = @as(usize, cols) / 32;
            const row_off = @as(usize, row_index) * bpr * bpb;
            if (row_off + bpr * bpb > raw_data.len) return error.InputTooSmall;
            const lut = [16]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6 };

            var out_i: usize = 0;
            for (0..bpr) |b| {
                const bo = row_off + b * bpb;
                const exp_byte = raw_data[bo];
                const scale_bits: u32 = if (exp_byte == 0) 0x00400000 else @as(u32, @intCast(exp_byte)) << 23;
                const d: f32 = @bitCast(scale_bits);
                const qs = raw_data[bo + 1 .. bo + 17];
                for (0..16) |j| {
                    output[out_i + j] = d * lut[qs[j] & 0x0F];
                    output[out_i + j + 16] = d * lut[qs[j] >> 4];
                }
                out_i += 32;
            }
        },
        else => return error.UnsupportedTensorType,
    }
}

/// Dot one quantized row against an f32 input vector, dispatching on tensor type.
/// Hot formats (`f32`, `f16`, `bf16`, `q4_0`, `q8_0`, `q4_k`, `q5_k`, `q6_k`) take a fused vectorized
/// path that streams weights and folds the dequant scales into the FMAs. Every other format falls
/// back to dequantizing into `scratch` first and then dotting.
/// @param raw_data Raw tensor bytes for the full matrix.
/// @param row_index Zero-based row to dot.
/// @param cols Number of columns per row; subject to the same block-alignment constraints as `row`.
/// @param tensor_type GGML quantization tag selecting the decode path.
/// @param input f32 input vector of length `>= cols`.
/// @param scratch Caller-owned scratch of length `>= cols`; only consumed on the fallback path.
/// @returns The f32 dot product, or an error matching `row`'s shape and size diagnostics.
pub fn dotRow(
    raw_data: []const u8,
    row_index: u32,
    cols: u32,
    tensor_type: GGMLType,
    input: []const f32,
    scratch: []f32,
) !f32 {
    if (input.len < cols) return error.InputTooSmall;
    return switch (tensor_type) {
        .f32 => dotF32Row(raw_data, row_index, cols, input),
        .f16 => dotF16Row(raw_data, row_index, cols, input),
        .bf16 => dotBf16Row(raw_data, row_index, cols, input),
        .q4_0 => dotQ4_0Row(raw_data, row_index, cols, input),
        .q8_0 => dotQ8_0Row(raw_data, row_index, cols, input),
        .q4_k => dotQ4KRow(raw_data, row_index, cols, input),
        .q5_k => dotQ5KRow(raw_data, row_index, cols, input),
        .q6_k => dotQ6KRow(raw_data, row_index, cols, input),
        else => blk: {
            if (scratch.len < cols) return error.OutputTooSmall;
            const tmp = scratch[0..cols];
            try row(raw_data, row_index, cols, tensor_type, tmp);
            var acc: f32 = 0;
            for (tmp, input[0..cols]) |w, x| acc = @mulAdd(f32, w, x, acc);
            break :blk acc;
        },
    };
}

/// Dot one f32-packed row against `input` using a 16-wide AVX-512-friendly inner loop.
/// Uses four independent accumulators driven by a 4-way unroll so the FP-add chain stays short.
/// @param raw_data Raw f32 row-major tensor bytes.
/// @param row_index Zero-based row to dot.
/// @param cols Number of f32 columns in the row.
/// @param input f32 input vector of length `>= cols`.
/// @returns The f32 dot product, or `error.InputTooSmall` when `input` or `raw_data` is shorter than expected.
pub fn dotF32Row(raw_data: []const u8, row_index: u32, cols: u32, input: []const f32) !f32 {
    if (input.len < cols) return error.InputTooSmall;

    // 16-wide (AVX-512 zmm) with four independent accumulators driven by a 4-way
    // unroll. Under strict FP the single-accumulator form is bound by the
    // loop-carried mul+add latency (a fresh `w*x` every iteration, but each add
    // waits on the previous one); four parallel running sums keep many more
    // weight loads in flight — a ~2-4x per-core speedup that matters for the
    // serial F32 matvecs, chiefly the per-layer MoE router run on the caller
    // every layer (cf. dotQ8_0Row's 4-accumulator loop).
    const Vec16f = @Vector(16, f32);
    const Vec8f = @Vector(8, f32);
    const row_bytes = @as(usize, cols) * @sizeOf(f32);
    const offset = @as(usize, row_index) * row_bytes;
    if (offset + row_bytes > raw_data.len) return error.InputTooSmall;

    var acc0: Vec16f = @splat(0.0);
    var acc1: Vec16f = @splat(0.0);
    var acc2: Vec16f = @splat(0.0);
    var acc3: Vec16f = @splat(0.0);
    var i: usize = 0;
    while (i + 64 <= cols) : (i += 64) {
        const w0: Vec16f = @bitCast(@as([64]u8, raw_data[offset + (i + 0) * 4 ..][0..64].*));
        const w1: Vec16f = @bitCast(@as([64]u8, raw_data[offset + (i + 16) * 4 ..][0..64].*));
        const w2: Vec16f = @bitCast(@as([64]u8, raw_data[offset + (i + 32) * 4 ..][0..64].*));
        const w3: Vec16f = @bitCast(@as([64]u8, raw_data[offset + (i + 48) * 4 ..][0..64].*));
        const x0: Vec16f = input[i + 0 ..][0..16].*;
        const x1: Vec16f = input[i + 16 ..][0..16].*;
        const x2: Vec16f = input[i + 32 ..][0..16].*;
        const x3: Vec16f = input[i + 48 ..][0..16].*;
        acc0 = @mulAdd(Vec16f, w0, x0, acc0);
        acc1 = @mulAdd(Vec16f, w1, x1, acc1);
        acc2 = @mulAdd(Vec16f, w2, x2, acc2);
        acc3 = @mulAdd(Vec16f, w3, x3, acc3);
    }
    var acc_vec: Vec16f = (acc0 + acc1) + (acc2 + acc3);
    while (i + 16 <= cols) : (i += 16) {
        const w: Vec16f = @bitCast(@as([64]u8, raw_data[offset + i * 4 ..][0..64].*));
        const x: Vec16f = input[i..][0..16].*;
        acc_vec = @mulAdd(Vec16f, w, x, acc_vec);
    }

    var acc = @reduce(.Add, acc_vec);
    if (i + 8 <= cols) {
        const w: Vec8f = @bitCast(@as([32]u8, raw_data[offset + i * 4 ..][0..32].*));
        const x: Vec8f = input[i..][0..8].*;
        acc += @reduce(.Add, w * x);
        i += 8;
    }
    while (i < cols) : (i += 1) {
        const bits = std.mem.readInt(u32, raw_data[offset + i * 4 ..][0..4], .little);
        const w: f32 = @bitCast(bits);
        acc = @mulAdd(f32, w, input[i], acc);
    }
    return acc;
}

/// Dot one f16-packed row against an f32 input vector, promoting each weight to f32 on the fly.
/// @param raw_data Raw f16 row-major tensor bytes.
/// @param row_index Zero-based row to dot.
/// @param cols Number of f16 columns in the row.
/// @param input f32 input vector of length `>= cols`.
/// @returns The f32 dot product, or `error.InputTooSmall` when the input or row bytes are too short.
pub fn dotF16Row(raw_data: []const u8, row_index: u32, cols: u32, input: []const f32) !f32 {
    if (input.len < cols) return error.InputTooSmall;

    const row_bytes = @as(usize, cols) * @sizeOf(u16);
    const offset = @as(usize, row_index) * row_bytes;
    if (offset + row_bytes > raw_data.len) return error.InputTooSmall;

    var acc: f32 = 0;
    for (0..cols) |i| {
        const bits = std.mem.readInt(u16, raw_data[offset + i * 2 ..][0..2], .little);
        const w: f32 = @floatCast(@as(f16, @bitCast(bits)));
        acc = @mulAdd(f32, w, input[i], acc);
    }
    return acc;
}

/// Dot one bf16-packed row against an f32 input vector by zero-extending each weight into f32.
/// @param raw_data Raw bf16 row-major tensor bytes.
/// @param row_index Zero-based row to dot.
/// @param cols Number of bf16 columns in the row.
/// @param input f32 input vector of length `>= cols`.
/// @returns The f32 dot product, or `error.InputTooSmall` when the input or row bytes are too short.
pub fn dotBf16Row(raw_data: []const u8, row_index: u32, cols: u32, input: []const f32) !f32 {
    if (input.len < cols) return error.InputTooSmall;

    const row_bytes = @as(usize, cols) * @sizeOf(u16);
    const offset = @as(usize, row_index) * row_bytes;
    if (offset + row_bytes > raw_data.len) return error.InputTooSmall;

    var acc: f32 = 0;
    for (0..cols) |i| {
        const bits = std.mem.readInt(u16, raw_data[offset + i * 2 ..][0..2], .little);
        const w: f32 = @bitCast(@as(u32, bits) << 16);
        acc = @mulAdd(f32, w, input[i], acc);
    }
    return acc;
}

/// Dot one Q8_0-packed row against an f32 input vector.
/// Q8_0 stores 32 signed-int8 weights per block with one f16 scale; this entry point validates
/// block alignment and bounds, then delegates to the unchecked vectorized inner loop.
/// @param raw_data Raw Q8_0 tensor bytes (34 bytes per 32-element block).
/// @param row_index Zero-based row to dot.
/// @param cols Number of columns; must be a multiple of 32.
/// @param input f32 input vector of length `>= cols`.
/// @returns The f32 dot product, or `error.UnsupportedShape` / `error.InputTooSmall` on misuse.
pub fn dotQ8_0Row(raw_data: []const u8, row_index: u32, cols: u32, input: []const f32) !f32 {
    if (cols % 32 != 0) return error.UnsupportedShape;
    if (input.len < cols) return error.InputTooSmall;
    const block_size: usize = 32;
    const bpb: usize = 34;
    const bpr = @as(usize, cols) / block_size;
    const row_off = @as(usize, row_index) * bpr * bpb;
    if (row_off + bpr * bpb > raw_data.len) return error.InputTooSmall;

    return dotQ8_0RowUnchecked(raw_data, row_index, cols, input);
}

/// Dot one Q8_0-packed row against an f32 input vector without bounds checks.
/// Callers must have already validated `cols % 32 == 0`, `input.len >= cols`, and
/// that `raw_data` is long enough.  Uses a 16-wide vectorized inner loop with four
/// independent accumulators over a 2-block unroll to keep the FP-add chain short.
/// @param raw_data Raw Q8_0 tensor bytes (34 bytes per 32-element block).
/// @param row_index Zero-based row to dot.
/// @param cols Number of columns; must be a pre-validated multiple of 32.
/// @param input f32 input vector of length `>= cols`.
/// @returns The f32 dot product (no error path).
pub inline fn dotQ8_0RowUnchecked(raw_data: []const u8, row_index: u32, cols: u32, input: []const f32) f32 {
    // 16-wide inner loop (AVX-512 zmm), four independent accumulators driven by a
    // 2-block unroll. Q8_0's per-element work is tiny next to the weight stream,
    // so the dependent FP-add chain — not the ALU — is what serialises the loop;
    // splitting it 4 ways lets the core keep many more weight loads in flight.
    const Vec16f = @Vector(16, f32);
    const Vec16i8 = @Vector(16, i8);
    const Vec16i32 = @Vector(16, i32);
    const block_size: usize = 32;
    const bpb: usize = 34;
    const bpr = @as(usize, cols) / block_size;
    const row_off = @as(usize, row_index) * bpr * bpb;

    var acc0: Vec16f = @splat(0.0);
    var acc1: Vec16f = @splat(0.0);
    var acc2: Vec16f = @splat(0.0);
    var acc3: Vec16f = @splat(0.0);
    var in_i: usize = 0;
    var b: usize = 0;
    while (b + 2 <= bpr) : (b += 2) {
        const bo0 = row_off + b * bpb;
        const bo1 = bo0 + bpb;
        const s0: Vec16f = @splat(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo0..][0..2], .little)))));
        const s1: Vec16f = @splat(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo1..][0..2], .little)))));
        const q0a: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(@as(Vec16i8, @bitCast(@as([16]u8, raw_data[bo0 + 2 ..][0..16].*))))));
        const q0b: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(@as(Vec16i8, @bitCast(@as([16]u8, raw_data[bo0 + 18 ..][0..16].*))))));
        const q1a: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(@as(Vec16i8, @bitCast(@as([16]u8, raw_data[bo1 + 2 ..][0..16].*))))));
        const q1b: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(@as(Vec16i8, @bitCast(@as([16]u8, raw_data[bo1 + 18 ..][0..16].*))))));
        const x0a: Vec16f = input[in_i..][0..16].*;
        const x0b: Vec16f = input[in_i + 16 ..][0..16].*;
        const x1a: Vec16f = input[in_i + 32 ..][0..16].*;
        const x1b: Vec16f = input[in_i + 48 ..][0..16].*;
        acc0 = @mulAdd(Vec16f, q0a * s0, x0a, acc0);
        acc1 = @mulAdd(Vec16f, q0b * s0, x0b, acc1);
        acc2 = @mulAdd(Vec16f, q1a * s1, x1a, acc2);
        acc3 = @mulAdd(Vec16f, q1b * s1, x1b, acc3);
        in_i += 64;
    }
    if (b < bpr) {
        const bo = row_off + b * bpb;
        const s: Vec16f = @splat(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo..][0..2], .little)))));
        const qa: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(@as(Vec16i8, @bitCast(@as([16]u8, raw_data[bo + 2 ..][0..16].*))))));
        const qb: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(@as(Vec16i8, @bitCast(@as([16]u8, raw_data[bo + 18 ..][0..16].*))))));
        const xa: Vec16f = input[in_i..][0..16].*;
        const xb: Vec16f = input[in_i + 16 ..][0..16].*;
        acc0 = @mulAdd(Vec16f, qa * s, xa, acc0);
        acc1 = @mulAdd(Vec16f, qb * s, xb, acc1);
    }
    return @reduce(.Add, (acc0 + acc1) + (acc2 + acc3));
}

/// Dot one Q4_0-packed row against an f32 input vector.
/// Q4_0 stores 32 nibble weights with a `-8` bias per block plus an f16 scale; this entry point
/// validates block alignment and bounds, then delegates to the unchecked vectorized inner loop.
/// @param raw_data Raw Q4_0 tensor bytes (18 bytes per 32-element block).
/// @param row_index Zero-based row to dot.
/// @param cols Number of columns; must be a multiple of 32.
/// @param input f32 input vector of length `>= cols`.
/// @returns The f32 dot product, or `error.UnsupportedShape` / `error.InputTooSmall` on misuse.
pub fn dotQ4_0Row(raw_data: []const u8, row_index: u32, cols: u32, input: []const f32) !f32 {
    if (cols % 32 != 0) return error.UnsupportedShape;
    if (input.len < cols) return error.InputTooSmall;
    const bpb: usize = 18;
    const bpr = @as(usize, cols) / 32;
    const row_off = @as(usize, row_index) * bpr * bpb;
    if (row_off + bpr * bpb > raw_data.len) return error.InputTooSmall;

    return dotQ4_0RowUnchecked(raw_data, row_index, cols, input);
}

/// Dot one Q4_0-packed row against an f32 input vector without bounds checks.
/// Callers must have already validated `cols % 32 == 0`, `input.len >= cols`, and
/// that `raw_data` is long enough.  Uses a 16-wide vectorized inner loop with eight
/// independent accumulators over a 4-block unroll to break the FP-add dependency chain.
/// Nibbles are recentered to the signed [-8, 7] range before the FP scale to collapse
/// the constant bias term out of the inner FMA.
/// @param raw_data Raw Q4_0 tensor bytes (18 bytes per 32-element block).
/// @param row_index Zero-based row to dot.
/// @param cols Number of columns; must be a pre-validated multiple of 32.
/// @param input f32 input vector of length `>= cols`.
/// @returns The f32 dot product (no error path).
pub inline fn dotQ4_0RowUnchecked(raw_data: []const u8, row_index: u32, cols: u32, input: []const f32) f32 {
    // 16-wide inner loop (AVX-512 zmm) with eight independent accumulators driven
    // by a 4-block unroll, mirroring dotQ8_0Row. Q4_0's per-element work (nibble
    // extract + bias subtract + scale) is tiny next to the weight stream, so the
    // dependent FP-add chain — not the ALU — serialises a two-accumulator loop;
    // the wider unroll keeps more weight loads in flight, which is what
    // bounds the big re-quantized LM-head matvec read every decode token.
    const Vec16f = @Vector(16, f32);
    const Vec16u8 = @Vector(16, u8);
    const Vec16i32 = @Vector(16, i32);
    const bpb: usize = 18;
    const bpr = @as(usize, cols) / 32;
    const row_off = @as(usize, row_index) * bpr * bpb;

    // Convert nibbles to signed [-8, 7] lanes before the FP scale. This avoids
    // the inner dequant FMA in `(q*d - 8*d) * x + acc`; perf on the RDNA node
    // shows this loop is hot in the re-quantized Q4_0 LM head and projections.
    var acc0: Vec16f = @splat(0.0);
    var acc1: Vec16f = @splat(0.0);
    var acc2: Vec16f = @splat(0.0);
    var acc3: Vec16f = @splat(0.0);
    var acc4: Vec16f = @splat(0.0);
    var acc5: Vec16f = @splat(0.0);
    var acc6: Vec16f = @splat(0.0);
    var acc7: Vec16f = @splat(0.0);
    var in_i: usize = 0;
    var b: usize = 0;
    while (b + 4 <= bpr) : (b += 4) {
        const bo0 = row_off + b * bpb;
        const bo1 = bo0 + bpb;
        const bo2 = bo1 + bpb;
        const bo3 = bo2 + bpb;
        const d0s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo0..][0..2], .little))));
        const d1s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo1..][0..2], .little))));
        const d2s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo2..][0..2], .little))));
        const d3s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo3..][0..2], .little))));
        const d0: Vec16f = @splat(d0s);
        const d1: Vec16f = @splat(d1s);
        const d2: Vec16f = @splat(d2s);
        const d3: Vec16f = @splat(d3s);
        const q0: Vec16u8 = raw_data[bo0 + 2 ..][0..16].*;
        const q1: Vec16u8 = raw_data[bo1 + 2 ..][0..16].*;
        const q2: Vec16u8 = raw_data[bo2 + 2 ..][0..16].*;
        const q3: Vec16u8 = raw_data[bo3 + 2 ..][0..16].*;
        const q0lo_i = @as(Vec16i32, @intCast(q0 & @as(Vec16u8, @splat(0x0F)))) - @as(Vec16i32, @splat(8));
        const q0hi_i = @as(Vec16i32, @intCast(q0 >> @as(Vec16u8, @splat(4)))) - @as(Vec16i32, @splat(8));
        const q1lo_i = @as(Vec16i32, @intCast(q1 & @as(Vec16u8, @splat(0x0F)))) - @as(Vec16i32, @splat(8));
        const q1hi_i = @as(Vec16i32, @intCast(q1 >> @as(Vec16u8, @splat(4)))) - @as(Vec16i32, @splat(8));
        const q2lo_i = @as(Vec16i32, @intCast(q2 & @as(Vec16u8, @splat(0x0F)))) - @as(Vec16i32, @splat(8));
        const q2hi_i = @as(Vec16i32, @intCast(q2 >> @as(Vec16u8, @splat(4)))) - @as(Vec16i32, @splat(8));
        const q3lo_i = @as(Vec16i32, @intCast(q3 & @as(Vec16u8, @splat(0x0F)))) - @as(Vec16i32, @splat(8));
        const q3hi_i = @as(Vec16i32, @intCast(q3 >> @as(Vec16u8, @splat(4)))) - @as(Vec16i32, @splat(8));
        const q0lo: Vec16f = @floatFromInt(q0lo_i);
        const q0hi: Vec16f = @floatFromInt(q0hi_i);
        const q1lo: Vec16f = @floatFromInt(q1lo_i);
        const q1hi: Vec16f = @floatFromInt(q1hi_i);
        const q2lo: Vec16f = @floatFromInt(q2lo_i);
        const q2hi: Vec16f = @floatFromInt(q2hi_i);
        const q3lo: Vec16f = @floatFromInt(q3lo_i);
        const q3hi: Vec16f = @floatFromInt(q3hi_i);
        const x0lo: Vec16f = input[in_i..][0..16].*;
        const x0hi: Vec16f = input[in_i + 16 ..][0..16].*;
        const x1lo: Vec16f = input[in_i + 32 ..][0..16].*;
        const x1hi: Vec16f = input[in_i + 48 ..][0..16].*;
        const x2lo: Vec16f = input[in_i + 64 ..][0..16].*;
        const x2hi: Vec16f = input[in_i + 80 ..][0..16].*;
        const x3lo: Vec16f = input[in_i + 96 ..][0..16].*;
        const x3hi: Vec16f = input[in_i + 112 ..][0..16].*;
        acc0 = @mulAdd(Vec16f, q0lo * d0, x0lo, acc0);
        acc1 = @mulAdd(Vec16f, q0hi * d0, x0hi, acc1);
        acc2 = @mulAdd(Vec16f, q1lo * d1, x1lo, acc2);
        acc3 = @mulAdd(Vec16f, q1hi * d1, x1hi, acc3);
        acc4 = @mulAdd(Vec16f, q2lo * d2, x2lo, acc4);
        acc5 = @mulAdd(Vec16f, q2hi * d2, x2hi, acc5);
        acc6 = @mulAdd(Vec16f, q3lo * d3, x3lo, acc6);
        acc7 = @mulAdd(Vec16f, q3hi * d3, x3hi, acc7);
        in_i += 128;
    }
    while (b + 2 <= bpr) : (b += 2) {
        const bo0 = row_off + b * bpb;
        const bo1 = bo0 + bpb;
        const d0s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo0..][0..2], .little))));
        const d1s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo1..][0..2], .little))));
        const d0: Vec16f = @splat(d0s);
        const d1: Vec16f = @splat(d1s);
        const q0: Vec16u8 = raw_data[bo0 + 2 ..][0..16].*;
        const q1: Vec16u8 = raw_data[bo1 + 2 ..][0..16].*;
        const q0lo_i = @as(Vec16i32, @intCast(q0 & @as(Vec16u8, @splat(0x0F)))) - @as(Vec16i32, @splat(8));
        const q0hi_i = @as(Vec16i32, @intCast(q0 >> @as(Vec16u8, @splat(4)))) - @as(Vec16i32, @splat(8));
        const q1lo_i = @as(Vec16i32, @intCast(q1 & @as(Vec16u8, @splat(0x0F)))) - @as(Vec16i32, @splat(8));
        const q1hi_i = @as(Vec16i32, @intCast(q1 >> @as(Vec16u8, @splat(4)))) - @as(Vec16i32, @splat(8));
        const q0lo: Vec16f = @floatFromInt(q0lo_i);
        const q0hi: Vec16f = @floatFromInt(q0hi_i);
        const q1lo: Vec16f = @floatFromInt(q1lo_i);
        const q1hi: Vec16f = @floatFromInt(q1hi_i);
        const x0lo: Vec16f = input[in_i..][0..16].*;
        const x0hi: Vec16f = input[in_i + 16 ..][0..16].*;
        const x1lo: Vec16f = input[in_i + 32 ..][0..16].*;
        const x1hi: Vec16f = input[in_i + 48 ..][0..16].*;
        acc0 = @mulAdd(Vec16f, q0lo * d0, x0lo, acc0);
        acc1 = @mulAdd(Vec16f, q0hi * d0, x0hi, acc1);
        acc2 = @mulAdd(Vec16f, q1lo * d1, x1lo, acc2);
        acc3 = @mulAdd(Vec16f, q1hi * d1, x1hi, acc3);
        in_i += 64;
    }
    if (b < bpr) {
        const bo = row_off + b * bpb;
        const ds: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo..][0..2], .little))));
        const d: Vec16f = @splat(ds);
        const q: Vec16u8 = raw_data[bo + 2 ..][0..16].*;
        const qlo_i = @as(Vec16i32, @intCast(q & @as(Vec16u8, @splat(0x0F)))) - @as(Vec16i32, @splat(8));
        const qhi_i = @as(Vec16i32, @intCast(q >> @as(Vec16u8, @splat(4)))) - @as(Vec16i32, @splat(8));
        const qlo: Vec16f = @floatFromInt(qlo_i);
        const qhi: Vec16f = @floatFromInt(qhi_i);
        const x_lo: Vec16f = input[in_i..][0..16].*;
        const x_hi: Vec16f = input[in_i + 16 ..][0..16].*;
        acc0 = @mulAdd(Vec16f, qlo * d, x_lo, acc0);
        acc1 = @mulAdd(Vec16f, qhi * d, x_hi, acc1);
    }
    return @reduce(.Add, ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7)));
}

/// Dot one Q4_0-packed row against an f32 input vector using precomputed 32-element
/// block sums to accelerate the zero-point bias subtraction.
/// The Q4_0 zero-point correction is `-8 * d * sum(x_block)`; passing precomputed
/// block sums from `fillInputSum32` moves that reduction out of the inner loop.
/// Callers must have already validated sizes.  All other behavior matches
/// `dotQ4_0RowUnchecked`.
/// @param raw_data Raw Q4_0 tensor bytes (18 bytes per 32-element block).
/// @param row_index Zero-based row to dot.
/// @param cols Number of columns; must be a pre-validated multiple of 32.
/// @param input f32 input vector of length `>= cols`.
/// @param input_sum32 Per-32-element block sums of `input`, as produced by `fillInputSum32`.
/// @returns The f32 dot product (no error path).
pub inline fn dotQ4_0RowWithSum32Unchecked(raw_data: []const u8, row_index: u32, cols: u32, input: []const f32, input_sum32: []const f32) f32 {
    const Vec16f = @Vector(16, f32);
    const Vec16u8 = @Vector(16, u8);
    const Vec16i32 = @Vector(16, i32);
    const bpb: usize = 18;
    const bpr = @as(usize, cols) / 32;
    const row_off = @as(usize, row_index) * bpr * bpb;

    var acc0: Vec16f = @splat(0.0);
    var acc1: Vec16f = @splat(0.0);
    var acc2: Vec16f = @splat(0.0);
    var acc3: Vec16f = @splat(0.0);
    var acc4: Vec16f = @splat(0.0);
    var acc5: Vec16f = @splat(0.0);
    var acc6: Vec16f = @splat(0.0);
    var acc7: Vec16f = @splat(0.0);
    var zero_bias0: f32 = 0.0;
    var zero_bias1: f32 = 0.0;
    var zero_bias2: f32 = 0.0;
    var zero_bias3: f32 = 0.0;
    var in_i: usize = 0;
    var b: usize = 0;
    while (b + 4 <= bpr) : (b += 4) {
        const bo0 = row_off + b * bpb;
        const bo1 = bo0 + bpb;
        const bo2 = bo1 + bpb;
        const bo3 = bo2 + bpb;
        const d0s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo0..][0..2], .little))));
        const d1s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo1..][0..2], .little))));
        const d2s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo2..][0..2], .little))));
        const d3s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo3..][0..2], .little))));
        const d0: Vec16f = @splat(d0s);
        const d1: Vec16f = @splat(d1s);
        const d2: Vec16f = @splat(d2s);
        const d3: Vec16f = @splat(d3s);
        const q0: Vec16u8 = raw_data[bo0 + 2 ..][0..16].*;
        const q1: Vec16u8 = raw_data[bo1 + 2 ..][0..16].*;
        const q2: Vec16u8 = raw_data[bo2 + 2 ..][0..16].*;
        const q3: Vec16u8 = raw_data[bo3 + 2 ..][0..16].*;
        const q0lo: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q0 & @as(Vec16u8, @splat(0x0F)))));
        const q0hi: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q0 >> @as(Vec16u8, @splat(4)))));
        const q1lo: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q1 & @as(Vec16u8, @splat(0x0F)))));
        const q1hi: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q1 >> @as(Vec16u8, @splat(4)))));
        const q2lo: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q2 & @as(Vec16u8, @splat(0x0F)))));
        const q2hi: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q2 >> @as(Vec16u8, @splat(4)))));
        const q3lo: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q3 & @as(Vec16u8, @splat(0x0F)))));
        const q3hi: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q3 >> @as(Vec16u8, @splat(4)))));
        const x0lo: Vec16f = input[in_i..][0..16].*;
        const x0hi: Vec16f = input[in_i + 16 ..][0..16].*;
        const x1lo: Vec16f = input[in_i + 32 ..][0..16].*;
        const x1hi: Vec16f = input[in_i + 48 ..][0..16].*;
        const x2lo: Vec16f = input[in_i + 64 ..][0..16].*;
        const x2hi: Vec16f = input[in_i + 80 ..][0..16].*;
        const x3lo: Vec16f = input[in_i + 96 ..][0..16].*;
        const x3hi: Vec16f = input[in_i + 112 ..][0..16].*;
        acc0 = @mulAdd(Vec16f, q0lo * d0, x0lo, acc0);
        acc1 = @mulAdd(Vec16f, q0hi * d0, x0hi, acc1);
        acc2 = @mulAdd(Vec16f, q1lo * d1, x1lo, acc2);
        acc3 = @mulAdd(Vec16f, q1hi * d1, x1hi, acc3);
        acc4 = @mulAdd(Vec16f, q2lo * d2, x2lo, acc4);
        acc5 = @mulAdd(Vec16f, q2hi * d2, x2hi, acc5);
        acc6 = @mulAdd(Vec16f, q3lo * d3, x3lo, acc6);
        acc7 = @mulAdd(Vec16f, q3hi * d3, x3hi, acc7);
        zero_bias0 = @mulAdd(f32, d0s, input_sum32[b], zero_bias0);
        zero_bias1 = @mulAdd(f32, d1s, input_sum32[b + 1], zero_bias1);
        zero_bias2 = @mulAdd(f32, d2s, input_sum32[b + 2], zero_bias2);
        zero_bias3 = @mulAdd(f32, d3s, input_sum32[b + 3], zero_bias3);
        in_i += 128;
    }
    while (b + 2 <= bpr) : (b += 2) {
        const bo0 = row_off + b * bpb;
        const bo1 = bo0 + bpb;
        const d0s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo0..][0..2], .little))));
        const d1s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo1..][0..2], .little))));
        const d0: Vec16f = @splat(d0s);
        const d1: Vec16f = @splat(d1s);
        const q0: Vec16u8 = raw_data[bo0 + 2 ..][0..16].*;
        const q1: Vec16u8 = raw_data[bo1 + 2 ..][0..16].*;
        const q0lo: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q0 & @as(Vec16u8, @splat(0x0F)))));
        const q0hi: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q0 >> @as(Vec16u8, @splat(4)))));
        const q1lo: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q1 & @as(Vec16u8, @splat(0x0F)))));
        const q1hi: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q1 >> @as(Vec16u8, @splat(4)))));
        const x0lo: Vec16f = input[in_i..][0..16].*;
        const x0hi: Vec16f = input[in_i + 16 ..][0..16].*;
        const x1lo: Vec16f = input[in_i + 32 ..][0..16].*;
        const x1hi: Vec16f = input[in_i + 48 ..][0..16].*;
        acc0 = @mulAdd(Vec16f, q0lo * d0, x0lo, acc0);
        acc1 = @mulAdd(Vec16f, q0hi * d0, x0hi, acc1);
        acc2 = @mulAdd(Vec16f, q1lo * d1, x1lo, acc2);
        acc3 = @mulAdd(Vec16f, q1hi * d1, x1hi, acc3);
        zero_bias0 = @mulAdd(f32, d0s, input_sum32[b], zero_bias0);
        zero_bias1 = @mulAdd(f32, d1s, input_sum32[b + 1], zero_bias1);
        in_i += 64;
    }
    if (b < bpr) {
        const bo = row_off + b * bpb;
        const ds: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo..][0..2], .little))));
        const d: Vec16f = @splat(ds);
        const q: Vec16u8 = raw_data[bo + 2 ..][0..16].*;
        const qlo: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q & @as(Vec16u8, @splat(0x0F)))));
        const qhi: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q >> @as(Vec16u8, @splat(4)))));
        const x_lo: Vec16f = input[in_i..][0..16].*;
        const x_hi: Vec16f = input[in_i + 16 ..][0..16].*;
        acc0 = @mulAdd(Vec16f, qlo * d, x_lo, acc0);
        acc1 = @mulAdd(Vec16f, qhi * d, x_hi, acc1);
        zero_bias0 = @mulAdd(f32, ds, input_sum32[b], zero_bias0);
    }
    const zero_bias = (zero_bias0 + zero_bias1) + (zero_bias2 + zero_bias3);
    return @reduce(.Add, ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7))) - 8.0 * zero_bias;
}

/// Specialized Q4_0 dot for the dominant 2048-column decode shape.
/// Same math and accumulation order as `dotQ4_0RowWithSum32Unchecked` for
/// `cols == 2048`, but with a fixed 64-block row and no dynamic tail handling.
pub inline fn dotQ4_0RowWithSum32Cols2048Unchecked(raw_data: []const u8, row_index: u32, input: []const f32, input_sum32: []const f32) f32 {
    const Vec16f = @Vector(16, f32);
    const Vec16u8 = @Vector(16, u8);
    const Vec16i32 = @Vector(16, i32);
    const bpb: usize = 18;
    const bpr: usize = 64;
    const row_off = @as(usize, row_index) * bpr * bpb;

    var acc0: Vec16f = @splat(0.0);
    var acc1: Vec16f = @splat(0.0);
    var acc2: Vec16f = @splat(0.0);
    var acc3: Vec16f = @splat(0.0);
    var acc4: Vec16f = @splat(0.0);
    var acc5: Vec16f = @splat(0.0);
    var acc6: Vec16f = @splat(0.0);
    var acc7: Vec16f = @splat(0.0);
    var zero_bias0: f32 = 0.0;
    var zero_bias1: f32 = 0.0;
    var zero_bias2: f32 = 0.0;
    var zero_bias3: f32 = 0.0;

    inline for (0..16) |group| {
        const b = group * 4;
        const bo0 = row_off + b * bpb;
        const bo1 = bo0 + bpb;
        const bo2 = bo1 + bpb;
        const bo3 = bo2 + bpb;
        const in_i = group * 128;
        const d0s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo0..][0..2], .little))));
        const d1s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo1..][0..2], .little))));
        const d2s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo2..][0..2], .little))));
        const d3s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw_data[bo3..][0..2], .little))));
        const d0: Vec16f = @splat(d0s);
        const d1: Vec16f = @splat(d1s);
        const d2: Vec16f = @splat(d2s);
        const d3: Vec16f = @splat(d3s);
        const q0: Vec16u8 = raw_data[bo0 + 2 ..][0..16].*;
        const q1: Vec16u8 = raw_data[bo1 + 2 ..][0..16].*;
        const q2: Vec16u8 = raw_data[bo2 + 2 ..][0..16].*;
        const q3: Vec16u8 = raw_data[bo3 + 2 ..][0..16].*;
        const q0lo: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q0 & @as(Vec16u8, @splat(0x0F)))));
        const q0hi: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q0 >> @as(Vec16u8, @splat(4)))));
        const q1lo: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q1 & @as(Vec16u8, @splat(0x0F)))));
        const q1hi: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q1 >> @as(Vec16u8, @splat(4)))));
        const q2lo: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q2 & @as(Vec16u8, @splat(0x0F)))));
        const q2hi: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q2 >> @as(Vec16u8, @splat(4)))));
        const q3lo: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q3 & @as(Vec16u8, @splat(0x0F)))));
        const q3hi: Vec16f = @floatFromInt(@as(Vec16i32, @intCast(q3 >> @as(Vec16u8, @splat(4)))));
        const x0lo: Vec16f = input[in_i..][0..16].*;
        const x0hi: Vec16f = input[in_i + 16 ..][0..16].*;
        const x1lo: Vec16f = input[in_i + 32 ..][0..16].*;
        const x1hi: Vec16f = input[in_i + 48 ..][0..16].*;
        const x2lo: Vec16f = input[in_i + 64 ..][0..16].*;
        const x2hi: Vec16f = input[in_i + 80 ..][0..16].*;
        const x3lo: Vec16f = input[in_i + 96 ..][0..16].*;
        const x3hi: Vec16f = input[in_i + 112 ..][0..16].*;
        acc0 = @mulAdd(Vec16f, q0lo * d0, x0lo, acc0);
        acc1 = @mulAdd(Vec16f, q0hi * d0, x0hi, acc1);
        acc2 = @mulAdd(Vec16f, q1lo * d1, x1lo, acc2);
        acc3 = @mulAdd(Vec16f, q1hi * d1, x1hi, acc3);
        acc4 = @mulAdd(Vec16f, q2lo * d2, x2lo, acc4);
        acc5 = @mulAdd(Vec16f, q2hi * d2, x2hi, acc5);
        acc6 = @mulAdd(Vec16f, q3lo * d3, x3lo, acc6);
        acc7 = @mulAdd(Vec16f, q3hi * d3, x3hi, acc7);
        zero_bias0 = @mulAdd(f32, d0s, input_sum32[b], zero_bias0);
        zero_bias1 = @mulAdd(f32, d1s, input_sum32[b + 1], zero_bias1);
        zero_bias2 = @mulAdd(f32, d2s, input_sum32[b + 2], zero_bias2);
        zero_bias3 = @mulAdd(f32, d3s, input_sum32[b + 3], zero_bias3);
    }

    const zero_bias = (zero_bias0 + zero_bias1) + (zero_bias2 + zero_bias3);
    return @reduce(.Add, ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7))) - 8.0 * zero_bias;
}

/// Quantize one row of f32 weights into the GGML `Q4_0` block layout.
/// Each 32-element block is stored as one f16 scale followed by 16 packed nibble
/// pairs (low nibble = first weight, high nibble = second weight), where each nibble
/// encodes a value in [0, 15] representing the original weight offset by +8.
/// Mirrors the reference implementation's `quantize_row_q4_0_ref`.
/// @param src Source f32 values; length must be a positive multiple of 32.
/// @param dst Destination byte buffer; must be at least `(src.len / 32) * 18` bytes.
/// @note Asserts (debug builds only) that alignment and size preconditions hold.
pub fn quantizeRowToQ4_0(src: []const f32, dst: []u8) void {
    const block_size: usize = 32;
    const bpb: usize = 18;
    std.debug.assert(src.len % block_size == 0);
    std.debug.assert(dst.len >= (src.len / block_size) * bpb);

    var si: usize = 0;
    var di: usize = 0;
    while (si + block_size <= src.len) : (si += block_size) {
        var amax: f32 = 0;
        var vmax: f32 = 0;
        for (src[si..][0..block_size]) |w| {
            const a = @abs(w);
            if (a > amax) {
                amax = a;
                vmax = w;
            }
        }
        const d: f32 = vmax / -8.0;
        const id: f32 = if (d != 0) 1.0 / d else 0.0;
        std.mem.writeInt(u16, dst[di..][0..2], @bitCast(@as(f16, @floatCast(d))), .little);
        for (0..16) |j| {
            const x0 = src[si + j] * id;
            const x1 = src[si + j + 16] * id;
            const xi0: u8 = @intCast(std.math.clamp(@as(i32, @intFromFloat(x0 + 8.5)), 0, 15));
            const xi1: u8 = @intCast(std.math.clamp(@as(i32, @intFromFloat(x1 + 8.5)), 0, 15));
            dst[di + 2 + j] = xi0 | (xi1 << 4);
        }
        di += bpb;
    }
}

/// Quantize one row of f32 weights into the GGML `Q8_0` block layout.
/// Each 32-element block is stored as one f16 scale followed by 32 signed int8
/// values clamped to [-127, 127]; the scale is `max(|w|) / 127`.
/// Mirrors the reference implementation's `quantize_row_q8_0_ref`.
/// @param src Source f32 values; length must be a positive multiple of 32.
/// @param dst Destination byte buffer; must be at least `(src.len / 32) * 34` bytes.
/// @note Asserts (debug builds only) that alignment and size preconditions hold.
pub fn quantizeRowToQ8_0(src: []const f32, dst: []u8) void {
    const block_size: usize = 32;
    const bpb: usize = 34;
    std.debug.assert(src.len % block_size == 0);
    std.debug.assert(dst.len >= (src.len / block_size) * bpb);

    var si: usize = 0;
    var di: usize = 0;
    while (si + block_size <= src.len) : (si += block_size) {
        var amax: f32 = 0;
        for (src[si..][0..block_size]) |w| {
            const a = @abs(w);
            if (a > amax) amax = a;
        }
        const d: f32 = amax / 127.0;
        const id: f32 = if (d != 0) 1.0 / d else 0.0;
        std.mem.writeInt(u16, dst[di..][0..2], @bitCast(@as(f16, @floatCast(d))), .little);
        for (0..block_size) |j| {
            const x = src[si + j] * id;
            const xi: i8 = @intCast(std.math.clamp(@as(i32, @intFromFloat(@round(x))), -127, 127));
            dst[di + 2 + j] = @bitCast(xi);
        }
        di += bpb;
    }
}

/// Dot one Q4_K-packed row against an f32 input vector.
/// Q4_K stores 256 weights per super-block as 8 sub-blocks of 32 nibbles, each with its own 6-bit
/// scale and min packed into a 12-byte header (plus block-level f16 `d`/`dmin`); validates block
/// alignment and bounds, then delegates to the unchecked vectorized inner loop.
/// @param raw_data Raw Q4_K tensor bytes (144 bytes per 256-element super-block).
/// @param row_index Zero-based row to dot.
/// @param cols Number of columns; must be a multiple of 256.
/// @param input f32 input vector of length `>= cols`.
/// @returns The f32 dot product, or `error.UnsupportedShape` / `error.InputTooSmall` on misuse.
pub fn dotQ4KRow(raw_data: []const u8, row_index: u32, cols: u32, input: []const f32) !f32 {
    if (cols % 256 != 0) return error.UnsupportedShape;
    if (input.len < cols) return error.InputTooSmall;
    const bpb: usize = 144;
    const bpr = @as(usize, cols) / 256;
    const row_off = @as(usize, row_index) * bpr * bpb;
    if (row_off + bpr * bpb > raw_data.len) return error.InputTooSmall;

    return dotQ4KRowUnchecked(raw_data, row_index, cols, input);
}

/// Dot one Q4_K-packed row against an f32 input vector without bounds checks.
/// Callers must have already validated `cols % 256 == 0`, `input.len >= cols`, and
/// that `raw_data` is long enough.  Iterates over 256-element super-blocks; for each
/// super-block reads the block-level f16 `d`/`dmin` pair and 12 bytes of packed 6-bit
/// sub-scales/mins, then processes four pairs of 32-element chunks with four independent
/// vector accumulators to keep the FP-add chain short.
/// @param raw_data Raw Q4_K tensor bytes (144 bytes per 256-element super-block).
/// @param row_index Zero-based row to dot.
/// @param cols Number of columns; must be a pre-validated multiple of 256.
/// @param input f32 input vector of length `>= cols`.
/// @returns The f32 dot product (no error path).
pub inline fn dotQ4KRowUnchecked(raw_data: []const u8, row_index: u32, cols: u32, input: []const f32) f32 {
    // 16-wide so the inner dequant/dot loop maps onto AVX-512 (zmm) on
    // capable CPUs; falls back to 2x256-bit on AVX2-only hosts.
    const Vec16f = @Vector(16, f32);
    const Vec16u8 = @Vector(16, u8);
    const Vec16u32 = @Vector(16, u32);
    const bpb: usize = 144;
    const bpr = @as(usize, cols) / 256;
    const row_off = @as(usize, row_index) * bpr * bpb;

    // Four independent accumulators (one per low/high-nibble × chunk site) keep
    // the dependent FP-add chain short so the loop stays weight-stream bound.
    var acc_lo0: Vec16f = @splat(0.0);
    var acc_hi0: Vec16f = @splat(0.0);
    var acc_lo1: Vec16f = @splat(0.0);
    var acc_hi1: Vec16f = @splat(0.0);
    var in_i: usize = 0;
    for (0..bpr) |bi| {
        const bb = row_off + bi * bpb;
        const d_bits = std.mem.readInt(u16, raw_data[bb..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        const dm_bits = std.mem.readInt(u16, raw_data[bb + 2 ..][0..2], .little);
        const dmin: f32 = @floatCast(@as(f16, @bitCast(dm_bits)));
        const scales = raw_data[bb + 4 .. bb + 16];
        const qs = raw_data[bb + 16 .. bb + 144];

        var is: usize = 0;
        var qo: usize = 0;
        for (0..4) |_| {
            const sm0 = getScaleMinK4(is, scales);
            const d1 = d * @as(f32, @floatFromInt(sm0.sc));
            const m1 = dmin * @as(f32, @floatFromInt(sm0.m));
            const sm1 = getScaleMinK4(is + 1, scales);
            const d2 = d * @as(f32, @floatFromInt(sm1.sc));
            const m2 = dmin * @as(f32, @floatFromInt(sm1.m));

            const d1_vec: Vec16f = @splat(d1);
            const d2_vec: Vec16f = @splat(d2);
            const nm1_vec: Vec16f = @splat(-m1);
            const nm2_vec: Vec16f = @splat(-m2);
            inline for (0..2) |chunk| {
                const q_arr: [16]u8 = qs[qo + chunk * 16 ..][0..16].*;
                const q: Vec16u8 = q_arr;
                const low_u32: Vec16u32 = @intCast(q & @as(Vec16u8, @splat(0x0F)));
                const high_u32: Vec16u32 = @intCast(q >> @as(Vec16u8, @splat(4)));
                const low_f: Vec16f = @floatFromInt(low_u32);
                const high_f: Vec16f = @floatFromInt(high_u32);
                const x_low: Vec16f = input[in_i + chunk * 16 ..][0..16].*;
                const x_high: Vec16f = input[in_i + 32 + chunk * 16 ..][0..16].*;
                // (q*scale - min) * x + acc as two fused multiply-adds.
                if (chunk == 0) {
                    acc_lo0 = @mulAdd(Vec16f, @mulAdd(Vec16f, low_f, d1_vec, nm1_vec), x_low, acc_lo0);
                    acc_hi0 = @mulAdd(Vec16f, @mulAdd(Vec16f, high_f, d2_vec, nm2_vec), x_high, acc_hi0);
                } else {
                    acc_lo1 = @mulAdd(Vec16f, @mulAdd(Vec16f, low_f, d1_vec, nm1_vec), x_low, acc_lo1);
                    acc_hi1 = @mulAdd(Vec16f, @mulAdd(Vec16f, high_f, d2_vec, nm2_vec), x_high, acc_hi1);
                }
            }
            qo += 32;
            in_i += 64;
            is += 2;
        }
    }
    return @reduce(.Add, (acc_lo0 + acc_hi0) + (acc_lo1 + acc_hi1));
}

/// Precompute per-32-element sums of an input vector for the `WithSum32` Q4_K/Q5_K dot paths.
/// Those paths fold the asymmetric min subtraction `-m * sum(x_block)` out of the inner loop, so the
/// caller fills `sums[i] = sum(input[i*32 .. (i+1)*32])` once and reuses it across many rows.
/// @param input Input vector whose length must be a positive multiple of 32.
/// @param sums Destination of length `>= input.len / 32`; lane `i` receives the sum of input block `i`.
pub fn fillInputSum32(input: []const f32, sums: []f32) void {
    std.debug.assert(input.len % 32 == 0);
    std.debug.assert(sums.len >= input.len / 32);

    const Vec16f = @Vector(16, f32);
    var base: usize = 0;
    var out: usize = 0;
    while (base < input.len) : ({
        base += 32;
        out += 1;
    }) {
        const a: Vec16f = input[base..][0..16].*;
        const b: Vec16f = input[base + 16 ..][0..16].*;
        sums[out] = @reduce(.Add, a + b);
    }
}

/// Dot one Q4_K-packed row against an f32 input vector using precomputed 32-element
/// block sums to hoist the asymmetric min subtraction out of the inner loop.
/// The Q4_K correction term `-dmin * m * sum(x_block)` is accumulated from the
/// `input_sum32` array (filled by `fillInputSum32`) rather than being recomputed
/// per weight, reducing inner-loop FMA count.  Callers must have already validated sizes.
/// @param raw_data Raw Q4_K tensor bytes (144 bytes per 256-element super-block).
/// @param row_index Zero-based row to dot.
/// @param cols Number of columns; must be a pre-validated multiple of 256.
/// @param input f32 input vector of length `>= cols`.
/// @param input_sum32 Per-32-element block sums of `input`, as produced by `fillInputSum32`.
/// @returns The f32 dot product (no error path).
pub inline fn dotQ4KRowWithSum32Unchecked(
    raw_data: []const u8,
    row_index: u32,
    cols: u32,
    input: []const f32,
    input_sum32: []const f32,
) f32 {
    const Vec16f = @Vector(16, f32);
    const Vec16u8 = @Vector(16, u8);
    const Vec16u32 = @Vector(16, u32);
    const bpb: usize = 144;
    const bpr = @as(usize, cols) / 256;
    const row_off = @as(usize, row_index) * bpr * bpb;

    var acc_lo0: Vec16f = @splat(0.0);
    var acc_hi0: Vec16f = @splat(0.0);
    var acc_lo1: Vec16f = @splat(0.0);
    var acc_hi1: Vec16f = @splat(0.0);
    var min_acc: f32 = 0.0;
    var in_i: usize = 0;
    for (0..bpr) |bi| {
        const bb = row_off + bi * bpb;
        const d_bits = std.mem.readInt(u16, raw_data[bb..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        const dm_bits = std.mem.readInt(u16, raw_data[bb + 2 ..][0..2], .little);
        const dmin: f32 = @floatCast(@as(f16, @bitCast(dm_bits)));
        const scales = raw_data[bb + 4 .. bb + 16];
        const qs = raw_data[bb + 16 .. bb + 144];

        var is: usize = 0;
        var qo: usize = 0;
        for (0..4) |_| {
            const sm0 = getScaleMinK4(is, scales);
            const d1_vec: Vec16f = @splat(d * @as(f32, @floatFromInt(sm0.sc)));
            const m1 = dmin * @as(f32, @floatFromInt(sm0.m));
            const sm1 = getScaleMinK4(is + 1, scales);
            const d2_vec: Vec16f = @splat(d * @as(f32, @floatFromInt(sm1.sc)));
            const m2 = dmin * @as(f32, @floatFromInt(sm1.m));
            const sum_base = in_i / 32;
            min_acc -= m1 * input_sum32[sum_base] + m2 * input_sum32[sum_base + 1];

            inline for (0..2) |chunk| {
                const q_arr: [16]u8 = qs[qo + chunk * 16 ..][0..16].*;
                const q: Vec16u8 = q_arr;
                const low_u32: Vec16u32 = @intCast(q & @as(Vec16u8, @splat(0x0F)));
                const high_u32: Vec16u32 = @intCast(q >> @as(Vec16u8, @splat(4)));
                const low_f: Vec16f = @floatFromInt(low_u32);
                const high_f: Vec16f = @floatFromInt(high_u32);
                const x_low: Vec16f = input[in_i + chunk * 16 ..][0..16].*;
                const x_high: Vec16f = input[in_i + 32 + chunk * 16 ..][0..16].*;
                if (chunk == 0) {
                    acc_lo0 = @mulAdd(Vec16f, low_f * d1_vec, x_low, acc_lo0);
                    acc_hi0 = @mulAdd(Vec16f, high_f * d2_vec, x_high, acc_hi0);
                } else {
                    acc_lo1 = @mulAdd(Vec16f, low_f * d1_vec, x_low, acc_lo1);
                    acc_hi1 = @mulAdd(Vec16f, high_f * d2_vec, x_high, acc_hi1);
                }
            }
            qo += 32;
            in_i += 64;
            is += 2;
        }
    }
    return @reduce(.Add, (acc_lo0 + acc_hi0) + (acc_lo1 + acc_hi1)) + min_acc;
}

/// Dot one Q5_K-packed row against an f32 input vector.
/// Q5_K extends Q4_K with a 5th high-bit plane stored as 32 bytes (one bit per weight, eight 32-element
/// sub-blocks); validates block alignment and bounds, then delegates to the unchecked vectorized
/// inner loop.
/// @param raw_data Raw Q5_K tensor bytes (176 bytes per 256-element super-block).
/// @param row_index Zero-based row to dot.
/// @param cols Number of columns; must be a multiple of 256.
/// @param input f32 input vector of length `>= cols`.
/// @returns The f32 dot product, or `error.UnsupportedShape` / `error.InputTooSmall` on misuse.
pub fn dotQ5KRow(raw_data: []const u8, row_index: u32, cols: u32, input: []const f32) !f32 {
    if (cols % 256 != 0) return error.UnsupportedShape;
    if (input.len < cols) return error.InputTooSmall;
    const bpb: usize = 176;
    const bpr = @as(usize, cols) / 256;
    const row_off = @as(usize, row_index) * bpr * bpb;
    if (row_off + bpr * bpb > raw_data.len) return error.InputTooSmall;

    return dotQ5KRowUnchecked(raw_data, row_index, cols, input);
}

/// Dot one Q5_K-packed row against an f32 input vector without bounds checks.
/// Callers must have already validated `cols % 256 == 0`, `input.len >= cols`, and
/// that `raw_data` is long enough.  Layout matches Q4_K super-blocks but with an
/// additional 32-byte high-bit plane (one bit per weight) that provides the 5th bit;
/// the plane is merged with the 4-bit base to produce 5-bit unsigned weights before
/// applying the sub-scale and min.
/// @param raw_data Raw Q5_K tensor bytes (176 bytes per 256-element super-block).
/// @param row_index Zero-based row to dot.
/// @param cols Number of columns; must be a pre-validated multiple of 256.
/// @param input f32 input vector of length `>= cols`.
/// @returns The f32 dot product (no error path).
pub inline fn dotQ5KRowUnchecked(raw_data: []const u8, row_index: u32, cols: u32, input: []const f32) f32 {
    // 16-wide inner loop: AVX-512 (zmm) on capable CPUs, 2x256-bit otherwise.
    const Vec16f = @Vector(16, f32);
    const Vec16u3 = @Vector(16, u3);
    const Vec16u8 = @Vector(16, u8);
    const Vec16u32 = @Vector(16, u32);
    const bpb: usize = 176;
    const bpr = @as(usize, cols) / 256;
    const row_off = @as(usize, row_index) * bpr * bpb;

    // Four independent accumulators (one per low/high-nibble × chunk site) keep
    // the dependent FP-add chain short so the loop stays weight-stream bound.
    var acc_lo0: Vec16f = @splat(0.0);
    var acc_hi0: Vec16f = @splat(0.0);
    var acc_lo1: Vec16f = @splat(0.0);
    var acc_hi1: Vec16f = @splat(0.0);
    var in_i: usize = 0;
    for (0..bpr) |bi| {
        const bb = row_off + bi * bpb;
        const d_bits = std.mem.readInt(u16, raw_data[bb..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        const dm_bits = std.mem.readInt(u16, raw_data[bb + 2 ..][0..2], .little);
        const dmin: f32 = @floatCast(@as(f16, @bitCast(dm_bits)));
        const scales = raw_data[bb + 4 .. bb + 16];
        const qh = raw_data[bb + 16 .. bb + 48];
        const qs = raw_data[bb + 48 .. bb + 176];

        var is: usize = 0;
        for (0..4) |j| {
            const sm0 = getScaleMinK4(is, scales);
            const d1 = d * @as(f32, @floatFromInt(sm0.sc));
            const m1 = dmin * @as(f32, @floatFromInt(sm0.m));
            const sm1 = getScaleMinK4(is + 1, scales);
            const d2 = d * @as(f32, @floatFromInt(sm1.sc));
            const m2 = dmin * @as(f32, @floatFromInt(sm1.m));

            const d1_vec: Vec16f = @splat(d1);
            const d2_vec: Vec16f = @splat(d2);
            const nm1_vec: Vec16f = @splat(-m1);
            const nm2_vec: Vec16f = @splat(-m2);
            const shift_lo: Vec16u3 = @splat(@intCast(j * 2));
            const shift_hi: Vec16u3 = @splat(@intCast(j * 2 + 1));
            inline for (0..2) |chunk| {
                const q: Vec16u8 = qs[j * 32 + chunk * 16 ..][0..16].*;
                const qh_v: Vec16u8 = qh[chunk * 16 ..][0..16].*;
                const hb_lo = ((qh_v >> shift_lo) & @as(Vec16u8, @splat(1))) << @as(Vec16u3, @splat(4));
                const hb_hi = ((qh_v >> shift_hi) & @as(Vec16u8, @splat(1))) << @as(Vec16u3, @splat(4));
                const low_u32: Vec16u32 = @intCast((q & @as(Vec16u8, @splat(0x0F))) | hb_lo);
                const high_u32: Vec16u32 = @intCast((q >> @as(Vec16u3, @splat(4))) | hb_hi);
                const low_f: Vec16f = @floatFromInt(low_u32);
                const high_f: Vec16f = @floatFromInt(high_u32);
                const x_low: Vec16f = input[in_i + chunk * 16 ..][0..16].*;
                const x_high: Vec16f = input[in_i + 32 + chunk * 16 ..][0..16].*;
                // (q*scale - min) * x + acc as two fused multiply-adds.
                if (chunk == 0) {
                    acc_lo0 = @mulAdd(Vec16f, @mulAdd(Vec16f, low_f, d1_vec, nm1_vec), x_low, acc_lo0);
                    acc_hi0 = @mulAdd(Vec16f, @mulAdd(Vec16f, high_f, d2_vec, nm2_vec), x_high, acc_hi0);
                } else {
                    acc_lo1 = @mulAdd(Vec16f, @mulAdd(Vec16f, low_f, d1_vec, nm1_vec), x_low, acc_lo1);
                    acc_hi1 = @mulAdd(Vec16f, @mulAdd(Vec16f, high_f, d2_vec, nm2_vec), x_high, acc_hi1);
                }
            }
            in_i += 64;
            is += 2;
        }
    }
    return @reduce(.Add, (acc_lo0 + acc_hi0) + (acc_lo1 + acc_hi1));
}

/// Dot one Q5_K-packed row against an f32 input vector using precomputed 32-element
/// block sums to hoist the asymmetric min subtraction out of the inner loop.
/// Combines the Q5_K high-bit merge from `dotQ5KRowUnchecked` with the precomputed
/// min-correction strategy from `dotQ4KRowWithSum32Unchecked`.  Callers must have
/// already validated sizes.
/// @param raw_data Raw Q5_K tensor bytes (176 bytes per 256-element super-block).
/// @param row_index Zero-based row to dot.
/// @param cols Number of columns; must be a pre-validated multiple of 256.
/// @param input f32 input vector of length `>= cols`.
/// @param input_sum32 Per-32-element block sums of `input`, as produced by `fillInputSum32`.
/// @returns The f32 dot product (no error path).
pub inline fn dotQ5KRowWithSum32Unchecked(
    raw_data: []const u8,
    row_index: u32,
    cols: u32,
    input: []const f32,
    input_sum32: []const f32,
) f32 {
    const Vec16f = @Vector(16, f32);
    const Vec16u3 = @Vector(16, u3);
    const Vec16u8 = @Vector(16, u8);
    const Vec16u32 = @Vector(16, u32);
    const bpb: usize = 176;
    const bpr = @as(usize, cols) / 256;
    const row_off = @as(usize, row_index) * bpr * bpb;

    var acc_lo0: Vec16f = @splat(0.0);
    var acc_hi0: Vec16f = @splat(0.0);
    var acc_lo1: Vec16f = @splat(0.0);
    var acc_hi1: Vec16f = @splat(0.0);
    var min_acc: f32 = 0.0;
    var in_i: usize = 0;
    for (0..bpr) |bi| {
        const bb = row_off + bi * bpb;
        const d_bits = std.mem.readInt(u16, raw_data[bb..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        const dm_bits = std.mem.readInt(u16, raw_data[bb + 2 ..][0..2], .little);
        const dmin: f32 = @floatCast(@as(f16, @bitCast(dm_bits)));
        const scales = raw_data[bb + 4 .. bb + 16];
        const qh = raw_data[bb + 16 .. bb + 48];
        const qs = raw_data[bb + 48 .. bb + 176];

        var is: usize = 0;
        for (0..4) |j| {
            const sm0 = getScaleMinK4(is, scales);
            const d1_vec: Vec16f = @splat(d * @as(f32, @floatFromInt(sm0.sc)));
            const m1 = dmin * @as(f32, @floatFromInt(sm0.m));
            const sm1 = getScaleMinK4(is + 1, scales);
            const d2_vec: Vec16f = @splat(d * @as(f32, @floatFromInt(sm1.sc)));
            const m2 = dmin * @as(f32, @floatFromInt(sm1.m));
            const sum_base = in_i / 32;
            min_acc -= m1 * input_sum32[sum_base] + m2 * input_sum32[sum_base + 1];

            const shift_lo: Vec16u3 = @splat(@intCast(j * 2));
            const shift_hi: Vec16u3 = @splat(@intCast(j * 2 + 1));
            inline for (0..2) |chunk| {
                const q: Vec16u8 = qs[j * 32 + chunk * 16 ..][0..16].*;
                const qh_v: Vec16u8 = qh[chunk * 16 ..][0..16].*;
                const hb_lo = ((qh_v >> shift_lo) & @as(Vec16u8, @splat(1))) << @as(Vec16u3, @splat(4));
                const hb_hi = ((qh_v >> shift_hi) & @as(Vec16u8, @splat(1))) << @as(Vec16u3, @splat(4));
                const low_u32: Vec16u32 = @intCast((q & @as(Vec16u8, @splat(0x0F))) | hb_lo);
                const high_u32: Vec16u32 = @intCast((q >> @as(Vec16u3, @splat(4))) | hb_hi);
                const low_f: Vec16f = @floatFromInt(low_u32);
                const high_f: Vec16f = @floatFromInt(high_u32);
                const x_low: Vec16f = input[in_i + chunk * 16 ..][0..16].*;
                const x_high: Vec16f = input[in_i + 32 + chunk * 16 ..][0..16].*;
                if (chunk == 0) {
                    acc_lo0 = @mulAdd(Vec16f, low_f * d1_vec, x_low, acc_lo0);
                    acc_hi0 = @mulAdd(Vec16f, high_f * d2_vec, x_high, acc_hi0);
                } else {
                    acc_lo1 = @mulAdd(Vec16f, low_f * d1_vec, x_low, acc_lo1);
                    acc_hi1 = @mulAdd(Vec16f, high_f * d2_vec, x_high, acc_hi1);
                }
            }
            in_i += 64;
            is += 2;
        }
    }
    return @reduce(.Add, (acc_lo0 + acc_hi0) + (acc_lo1 + acc_hi1)) + min_acc;
}

/// Dot one Q6_K-packed row against an f32 input vector.
/// Q6_K packs 256 6-bit weights as a low-nibble plane plus a 2-bit-per-weight high plane, with one
/// f16 super-block scale and eight signed-int8 per-32 sub-scales; weights are recentered by `-32`.
/// Validates block alignment and bounds, then delegates to the unchecked vectorized inner loop.
/// @param raw_data Raw Q6_K tensor bytes (210 bytes per 256-element super-block).
/// @param row_index Zero-based row to dot.
/// @param cols Number of columns; must be a multiple of 256.
/// @param input f32 input vector of length `>= cols`.
/// @returns The f32 dot product, or `error.UnsupportedShape` / `error.InputTooSmall` on misuse.
pub fn dotQ6KRow(raw_data: []const u8, row_index: u32, cols: u32, input: []const f32) !f32 {
    if (cols % 256 != 0) return error.UnsupportedShape;
    if (input.len < cols) return error.InputTooSmall;
    const bpb: usize = 210;
    const bpr = @as(usize, cols) / 256;
    const row_off = @as(usize, row_index) * bpr * bpb;
    if (row_off + bpr * bpb > raw_data.len) return error.InputTooSmall;

    return dotQ6KRowUnchecked(raw_data, row_index, cols, input);
}

/// Dot one Q6_K-packed row against an f32 input vector without bounds checks.
/// Callers must have already validated `cols % 256 == 0`, `input.len >= cols`, and
/// that `raw_data` is long enough.  Each 256-element super-block is decoded by
/// combining a 128-byte low-nibble plane, a 64-byte 2-bit high plane, and 8 signed
/// int8 per-32 sub-scales; weights are recentered by subtracting 32 before scaling.
/// @param raw_data Raw Q6_K tensor bytes (210 bytes per 256-element super-block).
/// @param row_index Zero-based row to dot.
/// @param cols Number of columns; must be a pre-validated multiple of 256.
/// @param input f32 input vector of length `>= cols`.
/// @returns The f32 dot product (no error path).
pub inline fn dotQ6KRowUnchecked(raw_data: []const u8, row_index: u32, cols: u32, input: []const f32) f32 {
    // 16-wide inner loop: AVX-512 (zmm) on capable CPUs, 2x256-bit otherwise.
    const Vec16f = @Vector(16, f32);
    const Vec16u3 = @Vector(16, u3);
    const Vec16u8 = @Vector(16, u8);
    const Vec16u32 = @Vector(16, u32);
    const Vec16i32 = @Vector(16, i32);
    const bpb: usize = 210;
    const bpr = @as(usize, cols) / 256;
    const row_off = @as(usize, row_index) * bpr * bpb;

    // Four independent accumulators (one per dequantised quarter) keep the
    // dependent FP-add chain short so the loop stays weight-stream bound.
    var acc1v: Vec16f = @splat(0.0);
    var acc2v: Vec16f = @splat(0.0);
    var acc3v: Vec16f = @splat(0.0);
    var acc4v: Vec16f = @splat(0.0);
    var in_i: usize = 0;
    for (0..bpr) |bi| {
        const bb = row_off + bi * bpb;
        const d_bits = std.mem.readInt(u16, raw_data[bb + 208 ..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));

        var ql_o: usize = bb;
        var qh_o: usize = bb + 128;
        var sc_o: usize = bb + 192;

        for (0..2) |_| {
            inline for (0..2) |chunk| {
                const is = chunk;
                const ql_lo: Vec16u8 = raw_data[ql_o + chunk * 16 ..][0..16].*;
                const ql_hi: Vec16u8 = raw_data[ql_o + 32 + chunk * 16 ..][0..16].*;
                const qh_v: Vec16u8 = raw_data[qh_o + chunk * 16 ..][0..16].*;

                const rq1: Vec16u32 = @intCast((ql_lo & @as(Vec16u8, @splat(0x0F))) | (((qh_v >> @as(Vec16u3, @splat(0))) & @as(Vec16u8, @splat(3))) << @as(Vec16u3, @splat(4))));
                const rq2: Vec16u32 = @intCast((ql_hi & @as(Vec16u8, @splat(0x0F))) | (((qh_v >> @as(Vec16u3, @splat(2))) & @as(Vec16u8, @splat(3))) << @as(Vec16u3, @splat(4))));
                const rq3: Vec16u32 = @intCast((ql_lo >> @as(Vec16u3, @splat(4))) | (((qh_v >> @as(Vec16u3, @splat(4))) & @as(Vec16u8, @splat(3))) << @as(Vec16u3, @splat(4))));
                const rq4: Vec16u32 = @intCast((ql_hi >> @as(Vec16u3, @splat(4))) | (((qh_v >> @as(Vec16u3, @splat(6))) & @as(Vec16u8, @splat(3))) << @as(Vec16u3, @splat(4))));

                const q1_i: Vec16i32 = @as(Vec16i32, @intCast(rq1)) - @as(Vec16i32, @splat(32));
                const q2_i: Vec16i32 = @as(Vec16i32, @intCast(rq2)) - @as(Vec16i32, @splat(32));
                const q3_i: Vec16i32 = @as(Vec16i32, @intCast(rq3)) - @as(Vec16i32, @splat(32));
                const q4_i: Vec16i32 = @as(Vec16i32, @intCast(rq4)) - @as(Vec16i32, @splat(32));

                const s0: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_o + is])));
                const s2: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_o + is + 2])));
                const s4: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_o + is + 4])));
                const s6: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_o + is + 6])));

                const x0: Vec16f = input[in_i + chunk * 16 ..][0..16].*;
                const x1: Vec16f = input[in_i + 32 + chunk * 16 ..][0..16].*;
                const x2: Vec16f = input[in_i + 64 + chunk * 16 ..][0..16].*;
                const x3: Vec16f = input[in_i + 96 + chunk * 16 ..][0..16].*;
                acc1v = @mulAdd(Vec16f, @as(Vec16f, @floatFromInt(q1_i)) * @as(Vec16f, @splat(d * s0)), x0, acc1v);
                acc2v = @mulAdd(Vec16f, @as(Vec16f, @floatFromInt(q2_i)) * @as(Vec16f, @splat(d * s2)), x1, acc2v);
                acc3v = @mulAdd(Vec16f, @as(Vec16f, @floatFromInt(q3_i)) * @as(Vec16f, @splat(d * s4)), x2, acc3v);
                acc4v = @mulAdd(Vec16f, @as(Vec16f, @floatFromInt(q4_i)) * @as(Vec16f, @splat(d * s6)), x3, acc4v);
            }
            ql_o += 64;
            qh_o += 32;
            sc_o += 8;
            in_i += 128;
        }
    }
    return @reduce(.Add, (acc1v + acc2v) + (acc3v + acc4v));
}

test "dequant row copies f32 rows" {
    const raw = [_]f32{ 1.0, -2.0, 3.5, 4.0 };
    const bytes = std.mem.sliceAsBytes(&raw);
    var out = [_]f32{ 0, 0 };
    try row(bytes, 1, 2, .f32, &out);
    try std.testing.expectEqual(@as(f32, 3.5), out[0]);
    try std.testing.expectEqual(@as(f32, 4.0), out[1]);
}

test "dequant row handles q8_0" {
    var block = [_]u8{0} ** 34;
    const scale_bits: u16 = @bitCast(@as(f16, 0.5));
    std.mem.writeInt(u16, block[0..2], scale_bits, .little);
    block[2] = @bitCast(@as(i8, -2));
    block[3] = @bitCast(@as(i8, 4));

    var out = [_]f32{0} ** 32;
    try row(&block, 0, 32, .q8_0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[1], 0.00001);
}

test "dotRow q8_0 matches dequantized dot" {
    const cols: usize = 64;
    const blocks = cols / 32;
    var raw = [_]u8{0} ** (2 * blocks * 34);
    for (0..2) |row_index| {
        for (0..blocks) |block| {
            const bo = (row_index * blocks + block) * 34;
            const scale_bits: u16 = @bitCast(@as(f16, @floatCast(0.125 * @as(f32, @floatFromInt(row_index + block + 1)))));
            std.mem.writeInt(u16, raw[bo..][0..2], scale_bits, .little);
            for (0..32) |j| {
                const q: i8 = @intCast(@as(i32, @intCast((row_index * 13 + block * 7 + j * 3) % 29)) - 14);
                raw[bo + 2 + j] = @bitCast(q);
            }
        }
    }

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 5) % 17)) - 8.0) * 0.0625;
    var scratch = [_]f32{0} ** cols;

    for (0..2) |row_index| {
        try row(&raw, @intCast(row_index), cols, .q8_0, &scratch);
        var expected: f32 = 0;
        for (scratch, input) |w, x| expected += w * x;
        const actual = try dotQ8_0Row(&raw, @intCast(row_index), cols, &input);
        try std.testing.expectApproxEqAbs(expected, actual, 0.00001);
    }
}

test "direct scalar float dot rows match dequantized dot" {
    const cols: usize = 17;
    const rows: usize = 2;

    const raw_f32 = [_]f32{
        0.25,   -0.5,  1.0,    2.0,    -1.5,  3.0,   0.125,  -0.25, 0.75,
        -2.0,   1.25,  0.5,    -0.875, 1.75,  -3.0,  2.5,    0.375, -0.125,
        0.625,  -1.25, 1.5,    2.25,   -2.5,  3.5,   -3.75,  0.875, 1.125,
        -1.625, 2.875, -0.375, 0.0625, -0.75, 1.875, -2.125,
    };
    var raw_f16: [rows * cols * 2]u8 = undefined;
    var raw_bf16: [rows * cols * 2]u8 = undefined;
    for (raw_f32, 0..) |v, i| {
        const h_bits: u16 = @bitCast(@as(f16, @floatCast(v)));
        std.mem.writeInt(u16, raw_f16[i * 2 ..][0..2], h_bits, .little);
        const b_bits: u16 = @truncate(@as(u32, @bitCast(v)) >> 16);
        std.mem.writeInt(u16, raw_bf16[i * 2 ..][0..2], b_bits, .little);
    }

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 11) % 23)) - 10.0) * 0.125;

    var scratch = [_]f32{0} ** cols;
    inline for (.{ GGMLType.f32, GGMLType.f16, GGMLType.bf16 }) |tensor_type| {
        const raw = switch (tensor_type) {
            .f32 => std.mem.sliceAsBytes(&raw_f32),
            .f16 => &raw_f16,
            .bf16 => &raw_bf16,
            else => unreachable,
        };
        for (0..rows) |row_index| {
            try row(raw, @intCast(row_index), cols, tensor_type, &scratch);
            var expected: f32 = 0;
            for (scratch, input) |w, x| expected += w * x;
            const actual = try dotRow(raw, @intCast(row_index), cols, tensor_type, &input, &scratch);
            try std.testing.expectApproxEqAbs(expected, actual, 0.0001);
        }
    }
}

test "dotRow q5_k matches dequantized dot" {
    var raw = [_]u8{0} ** (2 * 176);
    for (0..2) |row_index| {
        const bb = row_index * 176;
        const d_bits: u16 = @bitCast(@as(f16, @floatCast(0.02734375 * @as(f32, @floatFromInt(row_index + 1)))));
        const dmin_bits: u16 = @bitCast(@as(f16, @floatCast(0.0068359375 * @as(f32, @floatFromInt(row_index + 1)))));
        std.mem.writeInt(u16, raw[bb..][0..2], d_bits, .little);
        std.mem.writeInt(u16, raw[bb + 2 ..][0..2], dmin_bits, .little);
        for (0..12) |i| raw[bb + 4 + i] = @intCast((i * 7 + row_index * 13) & 0xff);
        for (0..32) |i| raw[bb + 16 + i] = @intCast((i * 5 + row_index * 19) & 0xff);
        for (0..128) |i| raw[bb + 48 + i] = @intCast((i * 11 + row_index * 17) & 0xff);
    }

    var input: [256]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 13) % 37)) - 18.0) * 0.03125;
    var scratch = [_]f32{0} ** 256;
    var sums = [_]f32{0} ** 8;
    fillInputSum32(&input, &sums);

    for (0..2) |row_index| {
        try row(&raw, @intCast(row_index), 256, .q5_k, &scratch);
        var expected: f32 = 0;
        for (scratch, input) |w, x| expected += w * x;
        const actual = try dotQ5KRow(&raw, @intCast(row_index), 256, &input);
        try std.testing.expectApproxEqAbs(expected, actual, 0.0005);
        const summed = dotQ5KRowWithSum32Unchecked(&raw, @intCast(row_index), 256, &input, &sums);
        try std.testing.expectApproxEqAbs(expected, summed, 0.0005);
    }
}

test "dotRow q6_k matches dequantized dot" {
    var raw = [_]u8{0} ** (2 * 210);
    for (0..2) |row_index| {
        const bb = row_index * 210;
        for (0..128) |i| raw[bb + i] = @intCast((i * 11 + row_index * 3) & 0xff);
        for (0..64) |i| raw[bb + 128 + i] = @intCast((i * 17 + row_index * 7) & 0xff);
        for (0..16) |i| {
            const scale: i8 = @intCast(@as(i32, @intCast((i * 9 + row_index * 5) % 31)) - 15);
            raw[bb + 192 + i] = @bitCast(scale);
        }
        const d_bits: u16 = @bitCast(@as(f16, @floatCast(0.01953125 * @as(f32, @floatFromInt(row_index + 1)))));
        std.mem.writeInt(u16, raw[bb + 208 ..][0..2], d_bits, .little);
    }

    var input: [256]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 17) % 41)) - 20.0) * 0.03125;
    var scratch = [_]f32{0} ** 256;

    for (0..2) |row_index| {
        try row(&raw, @intCast(row_index), 256, .q6_k, &scratch);
        var expected: f32 = 0;
        for (scratch, input) |w, x| expected += w * x;
        const actual = try dotQ6KRow(&raw, @intCast(row_index), 256, &input);
        try std.testing.expectApproxEqAbs(expected, actual, 0.0005);
    }
}

test "dotRow q4_k matches dequantized dot" {
    var raw = [_]u8{0} ** (2 * 144);
    for (0..2) |row_index| {
        const bb = row_index * 144;
        const d_bits: u16 = @bitCast(@as(f16, @floatCast(0.03125 * @as(f32, @floatFromInt(row_index + 1)))));
        const dmin_bits: u16 = @bitCast(@as(f16, @floatCast(0.0078125 * @as(f32, @floatFromInt(row_index + 1)))));
        std.mem.writeInt(u16, raw[bb..][0..2], d_bits, .little);
        std.mem.writeInt(u16, raw[bb + 2 ..][0..2], dmin_bits, .little);
        for (0..12) |i| raw[bb + 4 + i] = @intCast((i * 9 + row_index * 5) & 0xff);
        for (0..128) |i| raw[bb + 16 + i] = @intCast((i * 11 + row_index * 17) & 0xff);
    }

    var input: [256]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 7) % 31)) - 15.0) * 0.03125;
    var scratch = [_]f32{0} ** 256;
    var sums = [_]f32{0} ** 8;
    fillInputSum32(&input, &sums);

    for (0..2) |row_index| {
        try row(&raw, @intCast(row_index), 256, .q4_k, &scratch);
        var expected: f32 = 0;
        for (scratch, input) |w, x| expected += w * x;
        const actual = try dotQ4KRow(&raw, @intCast(row_index), 256, &input);
        try std.testing.expectApproxEqAbs(expected, actual, 0.0001);
        const summed = dotQ4KRowWithSum32Unchecked(&raw, @intCast(row_index), 256, &input, &sums);
        try std.testing.expectApproxEqAbs(expected, summed, 0.0001);
    }
}

test "dequant row handles q4_0" {
    var block = [_]u8{0} ** 18;
    const d_bits: u16 = @bitCast(@as(f16, 0.5));
    std.mem.writeInt(u16, block[0..2], d_bits, .little);
    // q[0] low nibble = 6 -> (6-8)*0.5 = -1.0 ; q[16] high nibble = 10 -> (10-8)*0.5 = 1.0
    block[2] = 6 | (10 << 4);

    var out = [_]f32{0} ** 32;
    try row(&block, 0, 32, .q4_0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[16], 0.00001);
}

test "dotQ4_0Row matches dequantized dot" {
    const cols: usize = 64;
    const blocks = cols / 32;
    var raw = [_]u8{0} ** (2 * blocks * 18);
    for (0..2) |row_index| {
        for (0..blocks) |block| {
            const bo = (row_index * blocks + block) * 18;
            const d_bits: u16 = @bitCast(@as(f16, @floatCast(0.125 * @as(f32, @floatFromInt(row_index + block + 1)))));
            std.mem.writeInt(u16, raw[bo..][0..2], d_bits, .little);
            for (0..16) |j| {
                const lo: u8 = @intCast((row_index * 13 + block * 7 + j * 3) % 16);
                const hi: u8 = @intCast((row_index * 5 + block * 11 + j * 2) % 16);
                raw[bo + 2 + j] = lo | (hi << 4);
            }
        }
    }

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 5) % 17)) - 8.0) * 0.0625;
    var sums = [_]f32{0} ** blocks;
    fillInputSum32(&input, &sums);

    for (0..2) |row_index| {
        var scratch = [_]f32{0} ** cols;
        try row(&raw, @intCast(row_index), cols, .q4_0, &scratch);
        var expected: f32 = 0;
        for (scratch, input) |w, x| expected += w * x;
        const actual = try dotQ4_0Row(&raw, @intCast(row_index), cols, &input);
        try std.testing.expectApproxEqAbs(expected, actual, 0.0001);
        const summed = dotQ4_0RowWithSum32Unchecked(&raw, @intCast(row_index), cols, &input, &sums);
        try std.testing.expectApproxEqAbs(expected, summed, 0.0001);
    }
}

test "dotQ4_0RowWithSum32Cols2048 matches generic sum path" {
    const cols: usize = 2048;
    const blocks = cols / 32;
    var raw = [_]u8{0} ** (2 * blocks * 18);
    for (0..2) |row_index| {
        for (0..blocks) |block| {
            const bo = (row_index * blocks + block) * 18;
            const d_bits: u16 = @bitCast(@as(f16, @floatCast(0.03125 * @as(f32, @floatFromInt((row_index + block) % 11 + 1)))));
            std.mem.writeInt(u16, raw[bo..][0..2], d_bits, .little);
            for (0..16) |j| {
                const lo: u8 = @intCast((row_index * 17 + block * 5 + j * 7) % 16);
                const hi: u8 = @intCast((row_index * 11 + block * 3 + j * 13) % 16);
                raw[bo + 2 + j] = lo | (hi << 4);
            }
        }
    }

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 19) % 23)) - 11.0) * 0.03125;
    var sums = [_]f32{0} ** blocks;
    fillInputSum32(&input, &sums);

    for (0..2) |row_index| {
        const generic = dotQ4_0RowWithSum32Unchecked(&raw, @intCast(row_index), cols, &input, &sums);
        const specialized = dotQ4_0RowWithSum32Cols2048Unchecked(&raw, @intCast(row_index), &input, &sums);
        try std.testing.expectApproxEqAbs(generic, specialized, 0.0001);
    }
}

test "quantizeRowToQ4_0 round-trips f32 within Q4_0 resolution" {
    const cols: usize = 64;
    var src: [cols]f32 = undefined;
    for (&src, 0..) |*v, i| {
        // A spread of magnitudes so the per-block scale is exercised.
        const t = @as(f32, @floatFromInt((i * 37) % 97)) / 97.0 - 0.5;
        v.* = t * (1.0 + @as(f32, @floatFromInt(i / 32)));
    }
    var packed_q: [(cols / 32) * 18]u8 = undefined;
    quantizeRowToQ4_0(&src, &packed_q);

    var recon = [_]f32{0} ** cols;
    try row(&packed_q, 0, cols, .q4_0, &recon);
    for (0..cols) |i| {
        // Per-block max magnitude / 8 is the quantization step; tolerate ~1 step.
        const block = i / 32;
        var amax: f32 = 0;
        for (src[block * 32 ..][0..32]) |w| amax = @max(amax, @abs(w));
        const step = amax / 8.0 + 1e-6;
        try std.testing.expect(@abs(recon[i] - src[i]) <= step * 1.01);
    }

    // The dot product over a row should track the exact dot to within the
    // accumulated quantization noise.
    var x: [cols]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 11) % 23)) - 11.0) * 0.05;
    var exact: f32 = 0;
    for (recon, x) |w, xi| exact += w * xi;
    const got = try dotQ4_0Row(&packed_q, 0, cols, &x);
    try std.testing.expectApproxEqAbs(exact, got, 0.001);
}

test "quantizeRowToQ8_0 round-trips f32 within Q8_0 resolution" {
    const cols: usize = 64;
    var src: [cols]f32 = undefined;
    for (&src, 0..) |*v, i| {
        const t = @as(f32, @floatFromInt((i * 37) % 97)) / 97.0 - 0.5;
        v.* = t * (1.0 + @as(f32, @floatFromInt(i / 32)));
    }
    var packed_q: [(cols / 32) * 34]u8 = undefined;
    quantizeRowToQ8_0(&src, &packed_q);

    var recon = [_]f32{0} ** cols;
    try row(&packed_q, 0, cols, .q8_0, &recon);
    for (0..cols) |i| {
        const block = i / 32;
        var amax: f32 = 0;
        for (src[block * 32 ..][0..32]) |w| amax = @max(amax, @abs(w));
        const step = amax / 127.0 + 1e-6;
        try std.testing.expect(@abs(recon[i] - src[i]) <= step * 1.01);
    }

    var x: [cols]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 11) % 23)) - 11.0) * 0.05;
    var exact: f32 = 0;
    for (recon, x) |w, xi| exact += w * xi;
    const got = try dotQ8_0Row(&packed_q, 0, cols, &x);
    try std.testing.expectApproxEqAbs(exact, got, 0.001);
}

test "dotF32Row 4-way unrolled loop and all tails match the scalar dot" {
    // 90 cols hits the 64-element unrolled body, the 16-element body, the
    // 8-element tail and the scalar tail in one pass.
    const cols: usize = 90;
    const rows: usize = 2;
    var raw_f32: [rows * cols]f32 = undefined;
    for (&raw_f32, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 13) % 71)) - 35.0) * 0.03125;

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 7) % 53)) - 26.0) * 0.0625;

    var scratch = [_]f32{0} ** cols;
    const raw = std.mem.sliceAsBytes(&raw_f32);
    for (0..rows) |row_index| {
        try row(raw, @intCast(row_index), cols, .f32, &scratch);
        var expected: f32 = 0;
        for (scratch, input) |w, x| expected += w * x;
        const actual = try dotF32Row(raw, @intCast(row_index), cols, &input);
        try std.testing.expectApproxEqAbs(expected, actual, 0.0005);
    }
}

test "dotQ4_0Row three-block row exercises the 2-block unroll and the odd tail" {
    const cols: usize = 96; // 3 blocks: b=0 unrolled (blocks 0,1) + b=2 tail (block 2)
    const blocks = cols / 32;
    var raw = [_]u8{0} ** (2 * blocks * 18);
    for (0..2) |row_index| {
        for (0..blocks) |block| {
            const bo = (row_index * blocks + block) * 18;
            const d_bits: u16 = @bitCast(@as(f16, @floatCast(0.0625 * @as(f32, @floatFromInt(row_index + block + 1)))));
            std.mem.writeInt(u16, raw[bo..][0..2], d_bits, .little);
            for (0..16) |j| {
                const lo: u8 = @intCast((row_index * 13 + block * 7 + j * 3) % 16);
                const hi: u8 = @intCast((row_index * 5 + block * 11 + j * 2) % 16);
                raw[bo + 2 + j] = lo | (hi << 4);
            }
        }
    }

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 5) % 19)) - 9.0) * 0.0625;

    for (0..2) |row_index| {
        var scratch = [_]f32{0} ** cols;
        try row(&raw, @intCast(row_index), cols, .q4_0, &scratch);
        var expected: f32 = 0;
        for (scratch, input) |w, x| expected += w * x;
        const actual = try dotQ4_0Row(&raw, @intCast(row_index), cols, &input);
        try std.testing.expectApproxEqAbs(expected, actual, 0.0005);
    }
}

test "dotQ4_0Row five-block row exercises the 4-block unroll and tail" {
    const cols: usize = 160; // 5 blocks: 4-block body + single-block tail
    const blocks = cols / 32;
    var raw = [_]u8{0} ** (2 * blocks * 18);
    for (0..2) |row_index| {
        for (0..blocks) |block| {
            const bo = (row_index * blocks + block) * 18;
            const d_bits: u16 = @bitCast(@as(f16, @floatCast(0.046875 * @as(f32, @floatFromInt(row_index + block + 1)))));
            std.mem.writeInt(u16, raw[bo..][0..2], d_bits, .little);
            for (0..16) |j| {
                const lo: u8 = @intCast((row_index * 17 + block * 5 + j * 7) % 16);
                const hi: u8 = @intCast((row_index * 3 + block * 13 + j * 11) % 16);
                raw[bo + 2 + j] = lo | (hi << 4);
            }
        }
    }

    var input: [cols]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 13) % 43)) - 21.0) * 0.03125;

    for (0..2) |row_index| {
        var scratch = [_]f32{0} ** cols;
        try row(&raw, @intCast(row_index), cols, .q4_0, &scratch);
        var expected: f32 = 0;
        for (scratch, input) |w, x| expected += w * x;
        const actual = try dotQ4_0Row(&raw, @intCast(row_index), cols, &input);
        try std.testing.expectApproxEqAbs(expected, actual, 0.0005);
    }
}
