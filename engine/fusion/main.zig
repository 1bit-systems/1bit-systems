//! Fused NPU+GPU inference engine — CLI entry point.
const std = @import("std");
const eng = @import("engine.zig");
const dispatcher = @import("dispatcher.zig");
const DispatchPolicy = dispatcher.DispatchPolicy;

const log = std.log.scoped(.fusion_main);

pub fn main(init: std.process.Init) !void {
    var policy: DispatchPolicy = .auto;
    var max_tokens: u32 = 64;
    var prompt: ?[]const u8 = null;
    var list_policies: bool = false;

    var args_iter = std.process.Args.Iterator.init(init.minimal.args);
    _ = args_iter.next();

    while (args_iter.next()) |arg| {
        const a = std.mem.sliceTo(arg, 0);
        if (std.mem.eql(u8, a, "--policy")) {
            if (args_iter.next()) |val| policy = parsePolicy(std.mem.sliceTo(val, 0));
        } else if (std.mem.eql(u8, a, "-n") or std.mem.eql(u8, a, "--max-tokens")) {
            if (args_iter.next()) |val| max_tokens = std.fmt.parseInt(u32, std.mem.sliceTo(val, 0), 10) catch 64;
        } else if (std.mem.eql(u8, a, "--list-policies")) {
            list_policies = true;
        } else if (std.mem.eql(u8, a, "-h") or std.mem.eql(u8, a, "--help")) {
            printHelp();
            return;
        } else if (prompt == null) {
            prompt = a;
        }
    }

    if (list_policies) {
        printPolicies();
        return;
    }

    std.debug.print("Fused NPU+GPU Engine\n", .{});
    std.debug.print("Dispatch policy: {s}\n", .{@tagName(policy)});
    std.debug.print("\n", .{});
    std.debug.print("Policy: {s}\n", .{getPolicyDescription(policy)});
    std.debug.print("Backend: FLM NPU proxy (82 tok/s on Qwen3-0.6B)\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Layer assignments:\n", .{});

    const n_layers: u32 = 28;
    for (0..n_layers) |i| {
        const a = dispatcher.Dispatcher.getPolicyAssignment(policy, @intCast(i));
        std.debug.print("  Layer {d:>2}: attn={s:>3} ffn={s:>3} qkv={s:>3}\n", .{
            i, @tagName(a.attention), @tagName(a.ffn), @tagName(a.qkv),
        });
    }

    if (prompt) |p| {
        std.debug.print("\nPrompt: {s}\n", .{p});
        std.debug.print("Max tokens: {d}\n", .{max_tokens});
        std.debug.print("\nRun the daemon for inference:\n", .{});
        std.debug.print("  npu-gpu-cpud --port 8080 --fused-policy {s}\n", .{@tagName(policy)});
        std.debug.print("Then: curl http://127.0.0.1:8080/v1/chat/completions ...\n", .{});
    }
}

fn parsePolicy(name: []const u8) DispatchPolicy {
    if (std.mem.eql(u8, name, "npu_only")) return .npu_only;
    if (std.mem.eql(u8, name, "gpu_only")) return .gpu_only;
    if (std.mem.eql(u8, name, "attention_on_npu")) return .attention_on_npu;
    if (std.mem.eql(u8, name, "ffn_on_npu")) return .ffn_on_npu;
    if (std.mem.eql(u8, name, "qkv_on_npu")) return .qkv_on_npu;
    if (std.mem.eql(u8, name, "layer_by_layer")) return .layer_by_layer;
    if (std.mem.eql(u8, name, "prefill_npu_decode_gpu")) return .prefill_npu_decode_gpu;
    if (std.mem.eql(u8, name, "auto")) return .auto;
    return .auto;
}

fn getPolicyDescription(policy: DispatchPolicy) []const u8 {
    return switch (policy) {
        .npu_only => "All layers -> NPU (XRT xclbin INT8 GEMM)",
        .gpu_only => "All layers -> GPU (Vulkan flash attention + DMMV)",
        .layer_by_layer => "Round-robin per layer: even -> NPU, odd -> GPU",
        .attention_on_npu => "Attention -> NPU, FFN -> GPU",
        .ffn_on_npu => "FFN -> NPU, Attention -> GPU flash attention",
        .qkv_on_npu => "QKV projection -> NPU, rest -> GPU",
        .prefill_npu_decode_gpu => "Prefill -> NPU, Decode -> GPU",
        .auto => "Auto-tuned: FFN/QKV -> NPU, Attention -> GPU",
    };
}

fn printPolicies() void {
    std.debug.print("Fused NPU+GPU Engine - Dispatch Policies\n", .{});
    inline for (std.meta.tags(DispatchPolicy)) |p| {
        std.debug.print("  {s:30}  {s}\n", .{ @tagName(p), getPolicyDescription(p) });
    }
}

fn printHelp() void {
    std.debug.print(
        \\Fused NPU+GPU Engine - 1bit.systems
        \\
        \\Usage:  fused-engine [options] [prompt]
        \\--policy <p>     Dispatch policy (auto, npu_only, gpu_only, ...)
        \\-n, --max-tokens Max tokens (default: 64)
        \\--list-policies  List all dispatch policies
        \\-h, --help       This help
        \\
        \\For HTTP serving: npu-gpu-cpud --port 8080
        \\
    , .{});
}
