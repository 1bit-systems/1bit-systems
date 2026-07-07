//! CLI entry point — reads config from environment variables.
const std = @import("std");
const NpuEngine = @import("npu_engine.zig").NpuEngine;

const DEFAULT_TOKENS: u32 = 32;
const DEFAULT_KV_PAGES: u32 = 1024;
const DEFAULT_XCLBIN_DIR = "/home/bcloud/npu-sandbox/npu-infer/build/int8";
const DEFAULT_MODEL = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
const TEST_TOKENS = [_]u32{ 151643, 872, 198, 11852, 151644, 198, 151643, 77091, 198 };

fn getEnv(comptime name: [:0]const u8) ?[]const u8 {
    const ptr = std.c.getenv(@ptrCast(name));
    return if (ptr) |p| std.mem.sliceTo(p, 0) else null;
}

pub fn main() !void {
    const a = std.heap.page_allocator;

    const model_path = getEnv("MODEL_PATH") orelse DEFAULT_MODEL;
    const tokens = blk: {
        const s = getEnv("TOKENS") orelse break :blk DEFAULT_TOKENS;
        break :blk std.fmt.parseInt(u32, s, 10) catch DEFAULT_TOKENS;
    };
    const kv_pages = blk: {
        const s = getEnv("KV_PAGES") orelse break :blk DEFAULT_KV_PAGES;
        break :blk std.fmt.parseInt(u32, s, 10) catch DEFAULT_KV_PAGES;
    };
    const xclbin_dir = getEnv("XCLBIN_DIR") orelse DEFAULT_XCLBIN_DIR;
    const model_tag = getEnv("MODEL_TAG") orelse "";

    // Detect model tag from path if not explicitly set
    const tag = blk: {
        if (model_tag.len > 0) break :blk model_tag;
        if (std.mem.lastIndexOfScalar(u8, model_path, '/')) |ls| {
            const parent = model_path[0..ls];
            if (std.mem.lastIndexOfScalar(u8, parent, '/')) |ss| {
                const raw = parent[ss + 1 ..];
                const r = try a.alloc(u8, raw.len);
                for (raw, 0..) |c, j| r[j] = switch (c) {
                    'A'...'Z' => c - 'A' + 'a',
                    '-', '.' => '_',
                    else => c,
                };
                for ([_][]const u8{ "_npu2", "_instruct", "_it", "_it_npu2" }) |sfx| {
                    if (std.mem.endsWith(u8, r, sfx)) {
                        break :blk r[0 .. r.len - sfx.len];
                    }
                }
                break :blk r;
            }
        }
        break :blk "qwen3_0_6b";
    };
    if (model_tag.len == 0) { defer a.free(tag); }

    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(@as(std.c.clockid_t, @enumFromInt(1)), &ts);
    const t0 = @as(i64, ts.sec) * 1_000_000_000 + @as(i64, ts.nsec);
    std.debug.print("Model: {s}\nTag: {s}\nXclbin: {s}\n\n", .{ model_path, tag, xclbin_dir });

    var eng = try NpuEngine.init(a, model_path, xclbin_dir, tag, 1, kv_pages);
    defer eng.deinit();

    _ = std.c.clock_gettime(@as(std.os.linux.clockid_t, @enumFromInt(1)), &ts);
    const init_ns = (@as(i64, ts.sec) * 1_000_000_000 + @as(i64, ts.nsec)) - t0;
    std.debug.print("Init: {d}ms\n\n", .{@divTrunc(init_ns, 1000000)});

    const out = try eng.runSimple(&TEST_TOKENS, tokens);
    defer a.free(out);

    _ = std.c.clock_gettime(@as(std.os.linux.clockid_t, @enumFromInt(1)), &ts);
    const total_ns = (@as(i64, ts.sec) * 1_000_000_000 + @as(i64, ts.nsec)) - t0;
    std.debug.print("\n=== {d} tokens ===\nIDs:", .{out.len});
    for (out) |t| std.debug.print(" {d}", .{t});
    const mspt = if (out.len > 0) @divTrunc(@divTrunc(total_ns, 1000000), @as(i64, @intCast(out.len))) else 0;
    const tps = if (mspt > 0) @divTrunc(1000, mspt) else 0;
    std.debug.print("\n\nTotal: {d}ms  ms/tok: {d}  tok/s: {d}\n", .{@divTrunc(total_ns, 1000000), mspt, tps});
}
