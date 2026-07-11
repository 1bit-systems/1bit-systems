//! ZINC_RT IR and T-CPU smoke tests.
//! These tests keep the M0 scaffold honest before a full forward_zinc_rt
//! decode loop exists.
//! @section Decode Planning
const std = @import("std");
const graph_mod = @import("../ir/graph.zig");
const verify = @import("../ir/verify.zig");
const ring = @import("../ring/mod.zig");
const cpu = @import("../ring/cpu.zig");

test "zinc_rt graph verifies a minimal decode chain" {
    var graph = graph_mod.Graph.init(std.testing.allocator);
    defer graph.deinit();

    const hidden = graph.addBuffer();
    const rms_weight = graph.addBuffer();
    const norm = graph.addBuffer();
    const logits = graph.addBuffer();
    const token = graph.addBuffer();

    _ = try graph.addNode(.rms_norm, &.{ hidden, rms_weight }, &.{norm});
    _ = try graph.addNode(.lm_head, &.{norm}, &.{logits});
    _ = try graph.addNode(.argmax, &.{logits}, &.{token});

    try verify.graph(&graph);
}

test "t_cpu ring executes basic opcode packets" {
    const input = [_]f32{ 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 0.5 };
    var norm = [_]f32{ 0.0, 0.0 };

    const gate = [_]f32{ 0.0, 2.0 };
    const up = [_]f32{ 3.0, 4.0 };
    var activated = [_]f32{ 0.0, 0.0 };

    const logits = [_]f32{ -1.0, 3.5, 2.0 };
    var token: u32 = 0;

    const packets = [_]ring.Packet{
        .{ .rms_norm = .{
            .input = &input,
            .weight = &weight,
            .output = &norm,
            .eps = 0.0,
        } },
        .{ .barrier = {} },
        .{ .swiglu = .{
            .gate = &gate,
            .up = &up,
            .output = &activated,
        } },
        .{ .argmax = .{
            .logits = &logits,
            .output = &token,
        } },
    };

    var cpu_ring = cpu.CpuRing.init();
    defer cpu_ring.deinit();
    try cpu_ring.submitAndWait(.{ .packets = &packets });

    try std.testing.expectApproxEqAbs(@as(f32, 0.84852815), norm[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0463767), activated[1], 0.00001);
    try std.testing.expectEqual(@as(u32, 1), token);
}

test "t_cpu ring executes no-layer forward packet chain" {
    const embed_rows = [_]f32{
        1.0, 0.0,
        0.0, 2.0,
    };
    const norm_weight = [_]f32{ 1.0, 1.0 };
    const lm_rows = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
    };

    var hidden = [_]f32{ 0.0, 0.0 };
    var norm = [_]f32{ 0.0, 0.0 };
    var row_scratch = [_]f32{ 0.0, 0.0 };
    var logits = [_]f32{ 0.0, 0.0, 0.0 };
    var token: u32 = 0;

    const packets = [_]ring.Packet{
        .{ .embed = .{
            .raw_data = std.mem.sliceAsBytes(&embed_rows),
            .tensor_type = .f32,
            .token_id = 1,
            .hidden_dim = 2,
            .vocab_size = 2,
            .output = &hidden,
        } },
        .{ .rms_norm = .{
            .input = &hidden,
            .weight = &norm_weight,
            .output = &norm,
            .eps = 0.0,
        } },
        .{ .lm_head = .{
            .raw_data = std.mem.sliceAsBytes(&lm_rows),
            .tensor_type = .f32,
            .hidden = &norm,
            .row_scratch = &row_scratch,
            .logits = &logits,
        } },
        .{ .argmax = .{
            .logits = &logits,
            .output = &token,
        } },
    };

    var cpu_ring = cpu.CpuRing.init();
    defer cpu_ring.deinit();
    try cpu_ring.submitAndWait(.{ .packets = &packets });

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), logits[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.4142135), logits[1], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.4142135), logits[2], 0.00001);
    try std.testing.expectEqual(@as(u32, 1), token);
}
