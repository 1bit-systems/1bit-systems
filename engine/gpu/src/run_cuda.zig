//! Standalone CUDA greedy-decode driver for the qwen35 forward pass.
//!
//! Loads the GGUF onto the GPU via loader_cuda, builds the forward state in
//! src/compute/forward_cuda.zig, and runs a single greedy decode step for a
//! fixed token, printing the predicted token id, a hidden-state sanity dump,
//! and the top-5 logits. Independent of the gpu/ dispatch interface so the
//! CUDA forward pass can be brought up incrementally.
//!
//! Rooted at src/ (module path) so it reaches both model/* and cuda/* and
//! compute/* — exactly like src/main.zig. The whole thing is one Zig module so
//! the loader's and forward pass's internal `../cuda/*` imports resolve.
//!
//! Build + run (on the box):
//!   cd ~/zinc5090 && CUDA_HOME=/usr/local/cuda \
//!     ~/zig-0.15.2/zig build cuda-run -Dbackend=cuda -- [token] [milestone] [model.gguf]
//!     milestone: v0 (tail only), v1 (one attn L3 + one ssm L0 + ffn), v2 (full 32 layers)
//! @section CUDA Runtime
const std = @import("std");
const device = @import("cuda/device.zig");
const loader = @import("model/loader_cuda.zig");
const forward = @import("compute/forward_cuda.zig");

const DEFAULT_MODEL = "models/Qwen3.5-9B-Q4_K_M.gguf";

const Milestone = enum { v0, v1, v2 };

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.next(); // exe name
    const token: u32 = blk: {
        const a = args.next() orelse break :blk 100;
        break :blk std.fmt.parseInt(u32, a, 10) catch 100;
    };
    const milestone: Milestone = blk: {
        const a = args.next() orelse break :blk .v2;
        break :blk std.meta.stringToEnum(Milestone, a) orelse .v2;
    };
    const model_path = args.next() orelse DEFAULT_MODEL;

    var dev = try device.CudaDevice.initBest(allocator);
    defer dev.deinit();
    var namebuf: [256]u8 = undefined;
    std.debug.print("=== CUDA greedy decode ({s}) ===\n", .{@tagName(milestone)});
    std.debug.print("device : {s}  cc={d}  SMs={d}\n", .{ dev.name(&namebuf), dev.computeCapability(), dev.smCount() });
    std.debug.print("model  : {s}\n", .{model_path});
    std.debug.print("token  : {d}  pos: 0\n", .{token});

    var model = try loader.Model.load(allocator, dev.ctx, model_path);
    defer model.deinit();

    var fwd = try forward.ForwardCuda.init(allocator, &model, 512);
    defer fwd.deinit();
    std.debug.print("forward init OK (n_embd={d}, n_layers={d}, vocab={d})\n", .{
        fwd.d.n_embd, fwd.d.n_layers, fwd.d.vocab,
    });

    const run_layers = milestone != .v0;

    // v1 sanity: run a couple of isolated blocks first to watch for explosion.
    if (milestone == .v1) {
        try runV1(&fwd, token);
        return;
    }

    const out_tok = try fwd.decodeStep(token, 0, run_layers);

    // hidden sanity
    const hid = try allocator.alloc(f32, fwd.d.n_embd);
    defer allocator.free(hid);
    fwd.readHidden(hid);
    var bad: usize = 0;
    var hmin: f32 = std.math.inf(f32);
    var hmax: f32 = -std.math.inf(f32);
    for (hid) |v| {
        if (!std.math.isFinite(v)) bad += 1;
        hmin = @min(hmin, v);
        hmax = @max(hmax, v);
    }
    std.debug.print("hidden[0..5] = {d:.4} {d:.4} {d:.4} {d:.4} {d:.4}\n", .{ hid[0], hid[1], hid[2], hid[3], hid[4] });
    std.debug.print("hidden: min={d:.4} max={d:.4} non-finite={d}/{d}\n", .{ hmin, hmax, bad, fwd.d.n_embd });

    // top-5 logits
    const logits = try allocator.alloc(f32, fwd.d.vocab);
    defer allocator.free(logits);
    fwd.readLogits(logits);
    printTop5(logits);

    std.debug.print("\nPREDICTED TOKEN: {d}\n", .{out_tok});
    if (bad != 0) {
        std.debug.print("RESULT: FAIL (non-finite hidden state)\n", .{});
        return error.NonFinite;
    }
    if (out_tok >= fwd.d.vocab) {
        std.debug.print("RESULT: FAIL (token out of range)\n", .{});
        return error.TokenOOB;
    }
    std.debug.print("RESULT: PASS\n", .{});
}

/// v1: embed, then run ONLY ssm layer 0 + ffn, dump; then attn layer 3 + ffn,
/// dump — to localize any explosion to a single block type.
fn runV1(fwd: *forward.ForwardCuda, token: u32) !void {
    const a = fwd.allocator;
    const hid = try a.alloc(f32, fwd.d.n_embd);
    defer a.free(hid);

    // embed only
    fwd.model.dequantEmbeddingRow(token, fwd.host_embed);
    @import("cuda/buffer.zig").upload(fwd.ctx, &fwd.hidden, std.mem.sliceAsBytes(fwd.host_embed));
    fwd.readHidden(hid);
    dumpHidden("after embed        ", hid);

    try fwd.ssmLayerPub(0);
    fwd.readHidden(hid);
    dumpHidden("after ssm L0        ", hid);
    try fwd.ffnBlockPub(0);
    fwd.readHidden(hid);
    dumpHidden("after ffn L0        ", hid);

    try fwd.attentionLayerPub(3, 0);
    fwd.readHidden(hid);
    dumpHidden("after attn L3      ", hid);
    try fwd.ffnBlockPub(3);
    fwd.readHidden(hid);
    dumpHidden("after ffn L3        ", hid);

    std.debug.print("\nv1 RESULT: blocks executed without crash (see explosion check above)\n", .{});
}

fn dumpHidden(label: []const u8, hid: []const f32) void {
    var bad: usize = 0;
    var hmin: f32 = std.math.inf(f32);
    var hmax: f32 = -std.math.inf(f32);
    for (hid) |v| {
        if (!std.math.isFinite(v)) bad += 1;
        hmin = @min(hmin, v);
        hmax = @max(hmax, v);
    }
    std.debug.print("{s}: [{d:.4} {d:.4} {d:.4}] min={d:.3} max={d:.3} bad={d}\n", .{ label, hid[0], hid[1], hid[2], hmin, hmax, bad });
}

fn printTop5(logits: []const f32) void {
    var idx: [5]usize = .{ 0, 0, 0, 0, 0 };
    var val: [5]f32 = .{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) };
    for (logits, 0..) |v, i| {
        if (v > val[4]) {
            // insertion into the top-5
            var j: usize = 4;
            while (j > 0 and v > val[j - 1]) : (j -= 1) {
                val[j] = val[j - 1];
                idx[j] = idx[j - 1];
            }
            val[j] = v;
            idx[j] = i;
        }
    }
    std.debug.print("top-5 logits:\n", .{});
    for (0..5) |i| std.debug.print("  #{d}: token {d}  logit {d:.4}\n", .{ i + 1, idx[i], val[i] });
}
