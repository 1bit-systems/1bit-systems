//! NPU-only inference engine — independent backend.
//! Pure NPU inference via XRT. KV cache shared via zero-copy dma-buf.
//!
//! Build: zig build npu-engine
//! @section Fused Engine
const std = @import("std");
const model_data = @import("model_data.zig");
const fuse = @import("fused_execute.zig");

const FusedExecutor = fuse.FusedExecutor;

const DEFAULT_MODEL = "model.q4nx";  // override with -m/--model
const DEFAULT_NPU_ENGINE = "./npu_engine_universal";  // override with --npu-engine
const MAX_CONTEXT: u32 = 4096;
const BATCH_SIZE: u32 = 128;

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.c_allocator;

    var model_path: []const u8 = DEFAULT_MODEL;
    var npu_engine: []const u8 = DEFAULT_NPU_ENGINE;
    var max_tokens: u32 = 128;
    var args_iter = std.process.Args.Iterator.init(init.minimal.args);
    _ = args_iter.next(); // skip argv[0]
    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) {
            if (args_iter.next()) |v| model_path = v;
        } else if (std.mem.eql(u8, arg, "--npu-engine")) {
            if (args_iter.next()) |v| npu_engine = v;
        } else if (std.mem.eql(u8, arg, "-n") or std.mem.eql(u8, arg, "--max-tokens")) {
            if (args_iter.next()) |v| max_tokens = std.fmt.parseInt(u32, v, 10) catch 128;
        }
    }

    std.log.info("NPU-only engine starting: model={s}", .{model_path});
    // Ownership of every ModelData buffer (emb_f32, norms, RoPE tables, ...)
    // transfers into FusedExecutor below, which frees them all in its own
    // deinit() — do not also call model.deinit(), that double-frees them.
    const model = try model_data.loadModel(allocator, init.io, model_path, "qwen3_0_6b");

    const cfg = model.config;
    const fuse_config = fuse.ModelConfig{
        .hidden_dim = cfg.H,
        .n_layers = cfg.NC,
        .n_heads = cfg.NH,
        .n_kv_heads = cfg.NKV,
        .head_dim = cfg.HD,
        .inter_size = cfg.IM,
        .vocab_size = cfg.NV,
        .gqa_ratio = cfg.NH / cfg.NKV,
    };

    var executor = try FusedExecutor.init(
        allocator, init.io, .npu_only, fuse_config,
        model_path, npu_engine,
        MAX_CONTEXT, BATCH_SIZE,
        model.emb_f32, model.lm_head_f32, model.tied_embeddings,
        model.final_norm, model.in_norm, model.pa_norm,
        model.q_norm, model.k_norm,
        model.rope_sin, model.rope_cos,
    );
    defer executor.deinit();

    const prompt_tokens = [_]u32{151644}; // <|im_start|>
    try executor.prefill(&prompt_tokens, 1);

    std.log.info("Generating up to {d} tokens...", .{max_tokens});
    var cur_token: u32 = prompt_tokens[0];
    var generated: u32 = 0;
    while (generated < max_tokens) {
        // decodeBatch advances the KV cache position by exactly one token per
        // call, so it must be fed one token at a time — autoregressively, using
        // the just-generated token as the next input (not the original prompt).
        const next_token = try executor.decodeBatch(&[_]u32{cur_token});
        std.debug.print("{d} ", .{next_token});
        cur_token = next_token;
        generated += 1;
    }
    std.debug.print("\n", .{});
}
