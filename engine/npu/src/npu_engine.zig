//! NPU inference engine wrapper (stub for build compatibility).
//! The real NPU engine is the npu_ternaryd C++ daemon.
//! This module provides the types expected by engine.zig.
const std = @import("std");

pub const Config = struct {
    NC: u32 = 28,
    H: u32 = 2048,
    NH: u32 = 16,
    NKV: u32 = 8,
    HD: u32 = 64,
    vocab_size: u32 = 151936,
    max_seq_len: u32 = 4096,
};

pub const NpuEngine = struct {
    allocator: std.mem.Allocator,
    config: Config,
    healthy: bool = false,

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8, xclbin_dir: []const u8, model_tag: []const u8, max_parallel: u32, total_kv_pages: u32) !NpuEngine {
        _ = model_path; _ = xclbin_dir; _ = model_tag; _ = max_parallel; _ = total_kv_pages;
        return .{ .allocator = allocator, .config = .{}, .healthy = false };
    }

    pub fn deinit(self: *NpuEngine) void { self.* = undefined; }

    pub fn setKvCacheDmaBuf(self: *NpuEngine, dma_buf_fd: i32) !void {
        _ = self; _ = dma_buf_fd;
        return error.DaemonUnhealthy;
    }
};

pub const NpuError = error{ DaemonNotFound, DaemonUnhealthy, ImportFailed };

test "type compiles" { try std.testing.expectEqual(@sizeOf(usize), @sizeOf(*NpuEngine)); }
test "Config defaults" { const c: Config = .{}; try std.testing.expectEqual(@as(u32, 28), c.NC); }
test "init returns config" {
    var e = try NpuEngine.init(std.testing.allocator, "", "", "", 4, 1024);
    defer e.deinit();
    try std.testing.expect(!e.healthy);
}
