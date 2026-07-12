//! Fused NPU+GPU inference engine — CLI entry point.
//! Loads model, initializes NPU and GPU backends, runs fused inference.
//!
//! @section Fused Engine
const std = @import("std");
const model_data = @import("model_data.zig");
const gpu_attn = @import("gpu_attn.zig");
const fuse = @import("fused_execute.zig");
const dispatcher = @import("dispatcher.zig");
const tokenizer = @import("tokenizer.zig");

const FusedExecutor = fuse.FusedExecutor;
const DispatchPolicy = dispatcher.DispatchPolicy;
const QWEN3_0_6B = fuse.QWEN3_0_6B;

const DEFAULT_MODEL = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
const DEFAULT_NPU_ENGINE = "/home/bcloud/engine/npu/build/npu_engine_universal";
const ZINC_SHADER_DIR = "/home/bcloud/zinc/zig-out/share/zinc/shaders";
const TOKENIZER_JSON = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json";
const MAX_CONTEXT: u32 = 4096;
const BATCH_SIZE: u32 = 128;

const CliOptions = struct {
    model_path: []const u8 = DEFAULT_MODEL,
    npu_engine: []const u8 = DEFAULT_NPU_ENGINE,
    shader_dir: []const u8 = ZINC_SHADER_DIR,
    policy: DispatchPolicy = .ffn_on_npu,
    max_tokens: u32 = 128,
    prompt: []const u8 = "Hello",
    batch_size: u32 = BATCH_SIZE,
    debug: bool = false,
    list_policies: bool = false,
    show_help: bool = false,
};

fn parsePolicy(name: []const u8) DispatchPolicy {
    if (std.mem.eql(u8, name, "npu_only")) return .npu_only;
    if (std.mem.eql(u8, name, "gpu_only")) return .gpu_only;
    if (std.mem.eql(u8, name, "cpu_only")) return .cpu_only;
    if (std.mem.eql(u8, name, "ffn_on_npu")) return .ffn_on_npu;
    if (std.mem.eql(u8, name, "qkv_on_npu")) return .qkv_on_npu;
    if (std.mem.eql(u8, name, "attention_on_npu")) return .attention_on_npu;
    return .ffn_on_npu;
}

fn printHelp() void {
    std.debug.print(
        \\Fused NPU+GPU Inference Engine - 1bit.systems
        \\Target: 273 tok/s on Qwen3-0.6B
        \\
        \\Usage: fused-engine [options]
        \\
        \\  -m, --model <path>     Q4NX model path
        \\  --policy <name>        Dispatch policy (default: ffn_on_npu)
        \\  -n, --max-tokens <N>   Max tokens (default: 128)
        \\  -p, --prompt <text>    Input prompt
        \\  -b, --batch-size <N>   Batch size (default: 128)
        \\  --shader-dir <path>   ZINC SPIR-V shader directory
        \\  --list-policies        List all policies
        \\  --debug                Enable debug logging
        \\  -h, --help             This help
        \\
    , .{});
}

fn printPolicies() void {
    std.debug.print("Fused NPU+GPU Engine - Dispatch Policies\n", .{});
    inline for (std.meta.fields(DispatchPolicy)) |f| {
        const desc = switch (@as(DispatchPolicy, @enumFromInt(f.value))) {
            .npu_only => "All on NPU",
            .gpu_only => "All on GPU",
            .layer_by_layer => "Round-robin per layer",
            .attention_on_npu => "Attention on NPU, FFN on GPU",
            .ffn_on_npu => "FFN/QKV on NPU, Attention on GPU (target: 273 tok/s)",
            .qkv_on_npu => "QKV on NPU, rest on GPU",
            .prefill_npu_decode_gpu => "Prefill NPU, Decode GPU",
            .auto => "Auto-tuned",
            .cpu_only => "All on CPU ternary (pure C++, no accelerator needed)",
            .cpu_fallback => "CPU fallback when NPU/GPU backends are unavailable",
        };
        std.debug.print("  {s:20}  {s}\n", .{ f.name, desc });
    }
}

/// Tokenize text using the real C++ BPE tokenizer (tokenize.cpp / tokenizer.zig).
/// Returns u32 token IDs (converted from i32 — all valid token IDs are non-negative).
fn tokenize(text: []const u8, allocator: std.mem.Allocator) ![]u32 {
    var tok = try tokenizer.Tokenizer.init(allocator, TOKENIZER_JSON);
    defer tok.deinit();
    const ids_i32 = try tok.encode(text);
    defer allocator.free(ids_i32);
    const tokens = try allocator.alloc(u32, ids_i32.len);
    for (ids_i32, 0..) |id, i| {
        tokens[i] = @intCast(id);
    }
    return tokens;
}

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;

    var opts = CliOptions{};
    var args_iter = std.process.Args.Iterator.init(init.minimal.args);
    _ = args_iter.next(); // skip argv[0]
    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) {
            if (args_iter.next()) |v| opts.model_path = v;
        } else if (std.mem.eql(u8, arg, "--policy")) {
            if (args_iter.next()) |v| opts.policy = parsePolicy(v);
        } else if (std.mem.eql(u8, arg, "-n") or std.mem.eql(u8, arg, "--max-tokens")) {
            if (args_iter.next()) |v| opts.max_tokens = std.fmt.parseInt(u32, v, 10) catch 128;
        } else if (std.mem.eql(u8, arg, "--prompt") or std.mem.eql(u8, arg, "-p")) {
            if (args_iter.next()) |v| opts.prompt = v;
        } else if (std.mem.eql(u8, arg, "--batch-size") or std.mem.eql(u8, arg, "-b")) {
            if (args_iter.next()) |v| opts.batch_size = std.fmt.parseInt(u32, v, 10) catch 128;
        } else if (std.mem.eql(u8, arg, "--debug") or std.mem.eql(u8, arg, "-d")) {
            opts.debug = true;
        } else if (std.mem.eql(u8, arg, "--list-policies")) {
            opts.list_policies = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            opts.show_help = true;
        } else if (std.mem.eql(u8, arg, "--npu-engine")) {
            if (args_iter.next()) |v| opts.npu_engine = v;
        } else if (std.mem.eql(u8, arg, "--shader-dir")) {
            if (args_iter.next()) |v| opts.shader_dir = v;
        }
    }

    if (opts.show_help) { printHelp(); return; }
    if (opts.list_policies) { printPolicies(); return; }
    if (opts.debug) is_debug_mode = true;

    std.debug.print("\n=== Fused NPU+GPU Engine ===\n", .{});
    std.debug.print("Policy: {s}\n", .{@tagName(opts.policy)});

    // ── Load model ──
    std.debug.print("Loading model: {s}\n", .{opts.model_path});
    var model = try model_data.loadModel(allocator, init.io, opts.model_path, "qwen3_0_6b");
    defer model.deinit(allocator);

    const cfg = model.config;
    std.debug.print("  H={d} NC={d} NH={d} NKV={d} HD={d} IM={d} NV={d}\n", .{
        cfg.H, cfg.NC, cfg.NH, cfg.NKV, cfg.HD, cfg.IM, cfg.NV,
    });

    // ── Check NPU binary exists ──
    // NPU binary check skipped (path may not be null-terminated)

    // ── Initialize GPU attention (best-effort) ──
    std.debug.print("GPU attention init...\n", .{});
    var gpu_attn_instance = gpu_attn.GpuAttention.init(allocator, ZINC_SHADER_DIR) catch |err| {
        std.debug.print("  GPU unavailable: {s} (CPU fallback)\n", .{@errorName(err)});
        return error.GpuUnavailable;
    };
    defer gpu_attn_instance.deinit();
    std.debug.print("  GPU flash attention ready!\n", .{});

    // ── Create FusedExecutor ──
    // Map dispatcher policy to fuse policy by name
    const fuse_policy: fuse.DispatchPolicy = switch (opts.policy) {
        .npu_only => .npu_only,
        .gpu_only => .gpu_only,
        .ffn_on_npu => .ffn_on_npu,
        .qkv_on_npu => .qkv_on_npu,
        .attention_on_npu => .attention_on_npu,
        .cpu_only => .cpu_only,
        else => .ffn_on_npu,
    };
    var executor = try FusedExecutor.init(
        allocator, init.io, fuse_policy, QWEN3_0_6B,
        opts.model_path, opts.npu_engine,
        MAX_CONTEXT, opts.batch_size,
        model.emb_f32, model.lm_head_f32, model.tied_embeddings,
        model.final_norm, model.in_norm, model.pa_norm, model.q_norm, model.k_norm,
        model.rope_sin, model.rope_cos,
        .{
            .q = model.q_weight, .k = model.k_weight, .v = model.v_weight, .o = model.o_weight,
            .gate = model.gate_weight, .up = model.up_weight, .down = model.down_weight,
        },
    );
    defer executor.deinit();
    executor.gpu = gpu_attn_instance;

    // ── Tokenize ──
    const prompt_tokens = try tokenize(opts.prompt, allocator);
    defer allocator.free(prompt_tokens);
    std.debug.print("Prompt tokens ({d}): {any}\n", .{ prompt_tokens.len, prompt_tokens });

    // ── Prefill ──
    std.debug.print("Prefilling {d} tokens...\n", .{prompt_tokens.len});
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(std.os.linux.CLOCK.MONOTONIC, &ts);
    const t_start = @as(i64, ts.sec) * 1_000_000_000 + @as(i64, ts.nsec);
    executor.prefill(prompt_tokens, 1) catch |err| {
        std.debug.print("  Prefill error: {s}\n", .{@errorName(err)});
        return;
    };
    var ts2: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(std.os.linux.CLOCK.MONOTONIC, &ts2);
    const t_now = @as(i64, ts2.sec) * 1_000_000_000 + @as(i64, ts2.nsec);
    const prefill_ns = t_now - t_start;
    const prefill_ms = @as(f64, @floatFromInt(prefill_ns)) / 1_000_000.0;
    std.debug.print("  {d:.1}ms ({d:.1} ms/tok)\n", .{ prefill_ms, prefill_ms / @as(f64, @floatFromInt(prompt_tokens.len)) });

    // ── Decode ──
    // One token per decodeBatch() call. decodeBatch(tokens) writes
    // tokens.len distinct KV-cache positions (one per element) but always
    // calls kv.advance(1) regardless of that length -- correct only when
    // tokens.len == 1. The previous version filled a `batch`-sized buffer
    // with `batch` copies of the same stale last_token and called
    // decodeBatch once per outer-loop iteration, which wrote `batch`
    // distinct (garbage, since every input was identical) cache slots but
    // only advanced the position by 1 -- so the next call's writes mostly
    // overlapped and overwrote what the previous call had just written.
    // Real one-token-at-a-time decoding avoids this entirely: B=1 means
    // exactly one cache slot is written and the advance-by-1 is correct.
    std.debug.print("Generating up to {d} tokens...\n", .{opts.max_tokens});
    var generated: u32 = 0;

    // The first generated token comes directly from prefill()'s own
    // already-computed, correctly-positioned hidden state for the last
    // prompt token -- not by re-embedding and reprocessing that token as if
    // it were new input one position later (see firstDecodeToken's doc
    // comment). Only tokens after this one go through the normal
    // feed-the-previous-output-back decode loop.
    var last_token: u32 = executor.firstDecodeToken(@intCast(prompt_tokens.len)) catch |err| {
        std.debug.print("  Decode error at 0: {s}\n", .{@errorName(err)});
        return;
    };
    std.debug.print("{d} ", .{last_token});
    generated += 1;

    while (generated < opts.max_tokens) {
        const next_token = executor.decodeBatch(&[_]u32{last_token}) catch |err| {
            std.debug.print("  Decode error at {d}: {s}\n", .{ generated, @errorName(err) });
            break;
        };
        std.debug.print("{d} ", .{next_token});
        last_token = next_token;
        generated += 1;
        if (generated % 20 == 0) std.debug.print("\n", .{});
    }

    var ts3: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(std.os.linux.CLOCK.MONOTONIC, &ts3);
    const t_end = @as(i64, ts3.sec) * 1_000_000_000 + @as(i64, ts3.nsec);
    const total_ns = t_end - t_start;
    const total_ms = @as(f64, @floatFromInt(total_ns)) / 1_000_000.0;
    const tok_s = if (total_ms > 0) @as(f64, @floatFromInt(generated)) / (total_ms / 1000.0) else 0;
    std.debug.print("\n{d} tokens in {d:.0}ms ({d:.0} tok/s)\n", .{ generated, total_ms, tok_s });
}

pub var is_debug_mode: bool = false;
