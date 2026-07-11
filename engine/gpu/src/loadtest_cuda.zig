//! Standalone load-test for src/model/loader_cuda.zig.
//!
//! Drives only the CUDA loader path: select the best device, mmap + parse a
//! GGUF, upload every tensor to the GPU, and report the parsed config, the
//! tensor count, GPU free-mem before/after (to confirm ~model-size uploaded),
//! a spot-check of 3 tensors, and a CPU embedding-row dequant. Independent of
//! forward_cuda / gpu dispatch so the loader can be validated on its own.
//!
//! Rooted at src/ (module path) so it can import both model/* and cuda/* —
//! exactly like src/main.zig. The whole thing is one Zig module, which keeps
//! the loader's internal `../cuda/*` imports resolvable.
//!
//! Build + run (on the box):
//!   cd ~/zinc5090 && CUDA_HOME=/usr/local/cuda \
//!     ~/zig-0.15.2/zig build cuda-loadtest -Dbackend=cuda -- <model.gguf>
//! @section CUDA Runtime
const std = @import("std");
const device = @import("cuda/device.zig");
const loader = @import("model/loader_cuda.zig");

const DEFAULT_MODEL = "models/Qwen3.5-9B-Q4_K_M.gguf";

/// Load a GGUF onto the GPU via loader_cuda and print a structured report.
/// @returns Propagates loader errors (init/open/parse/upload) so the build step
/// surfaces a nonzero exit on failure.
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Model path from argv[1], else the default on the box.
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.next(); // exe name
    const model_path = args.next() orelse DEFAULT_MODEL;

    // --- Device selection ---
    var dev = try device.CudaDevice.initBest(allocator);
    defer dev.deinit();
    var namebuf: [256]u8 = undefined;
    const nm = dev.name(&namebuf);
    std.debug.print("=== CUDA loader load-test ===\n", .{});
    std.debug.print("device     : {s}  cc={d}  SMs={d}  vramGB={d}\n", .{
        nm, dev.computeCapability(), dev.smCount(), dev.totalMemory() / (1024 * 1024 * 1024),
    });

    const free_before = dev.freeMemory();
    std.debug.print("model      : {s}\n", .{model_path});
    std.debug.print("free before: {d} MB\n", .{free_before / (1024 * 1024)});

    // --- Load (mmap + parse + upload all tensors) ---
    var model = try loader.Model.load(allocator, dev.ctx, model_path);
    defer model.deinit();

    const free_after = dev.freeMemory();
    const used = if (free_before > free_after) free_before - free_after else 0;

    // --- Parsed config ---
    const c = model.config;
    std.debug.print("\n--- config ---\n", .{});
    std.debug.print("arch              : {s}\n", .{@tagName(c.architecture)});
    std.debug.print("n_layers          : {d}\n", .{c.n_layers});
    std.debug.print("hidden_dim        : {d}\n", .{c.hidden_dim});
    std.debug.print("intermediate (n_ff): {d}\n", .{c.intermediate_dim});
    std.debug.print("n_heads           : {d}\n", .{c.n_heads});
    std.debug.print("n_kv_heads        : {d}\n", .{c.n_kv_heads});
    std.debug.print("head_dim          : {d}\n", .{c.head_dim});
    std.debug.print("vocab_size        : {d}\n", .{c.vocab_size});
    std.debug.print("context_length    : {d}\n", .{c.context_length});
    std.debug.print("full_attn_interval: {d}\n", .{c.full_attn_interval});
    std.debug.print("rope_freq_base    : {d}\n", .{c.rope_freq_base});
    std.debug.print("rms_norm_eps      : {d}\n", .{c.rms_norm_eps});
    std.debug.print("ssm d_state={d} dt_rank={d} d_conv={d} d_inner={d} n_group={d}\n", .{
        c.ssm_d_state, c.ssm_dt_rank, c.ssm_d_conv, c.ssm_d_inner, c.ssm_n_group,
    });
    std.debug.print("n_experts         : {d} (used {d})\n", .{ c.n_experts, c.n_experts_used });

    // --- Tensors + GPU memory delta ---
    std.debug.print("\n--- tensors ---\n", .{});
    std.debug.print("tensor count      : {d}\n", .{model.tensors.items.len});
    std.debug.print("free after        : {d} MB\n", .{free_after / (1024 * 1024)});
    std.debug.print("GPU used by load  : {d} MB\n", .{used / (1024 * 1024)});

    // --- Spot-check 3 tensors ---
    std.debug.print("\n--- spot-check ---\n", .{});
    const checks = [_][]const u8{ "token_embd.weight", "output.weight", "blk.0.attn_norm.weight" };
    var all_ok = true;
    for (checks) |name| {
        if (model.get(name)) |t| {
            const ptr = t.gpu_buffer.devicePtr();
            const ok = ptr != 0;
            if (!ok) all_ok = false;
            std.debug.print("{s:<24} dims=[{d},{d},{d},{d}] type={s} devptr=0x{x} {s}\n", .{
                name,
                t.info.dims[0],
                t.info.dims[1],
                t.info.dims[2],
                t.info.dims[3],
                @tagName(t.info.type_),
                ptr,
                if (ok) "OK" else "NULL-PTR",
            });
        } else {
            all_ok = false;
            std.debug.print("{s:<24} MISSING\n", .{name});
        }
    }

    // --- getLayer sanity (same tensor via the formatted path) ---
    if (model.getLayer(0, "attn_norm.weight")) |t| {
        std.debug.print("getLayer(0,\"attn_norm.weight\") -> devptr=0x{x}\n", .{t.gpu_buffer.devicePtr()});
    } else {
        std.debug.print("getLayer(0,\"attn_norm.weight\") -> MISSING\n", .{});
        all_ok = false;
    }

    // --- CPU embedding dequant for token_id = 1 ---
    std.debug.print("\n--- embedding row (token_id=1) ---\n", .{});
    const emb = try allocator.alloc(f32, c.hidden_dim);
    defer allocator.free(emb);
    model.dequantEmbeddingRow(1, emb);
    var finite = true;
    for (emb) |v| {
        if (!std.math.isFinite(v)) finite = false;
    }
    std.debug.print("first 5: [{d}, {d}, {d}, {d}, {d}]  all-finite={}\n", .{
        emb[0], emb[1], emb[2], emb[3], emb[4], finite,
    });

    const pass = all_ok and finite and model.tensors.items.len > 0;
    std.debug.print("\nRESULT: {s}\n", .{if (pass) "PASS" else "FAIL"});
    if (!pass) return error.LoadTestFailed;
}
