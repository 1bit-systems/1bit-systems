//! GPU-only inference engine — independent backend.
//! Pure GPU inference via Vulkan/HIP. KV cache shared via zero-copy dma-buf.
//!
//! Build: zig build gpu-engine
//! @section Fused Engine
const std = @import("std");
const model_data = @import("model_data.zig");
const gpu_attn = @import("gpu_attn.zig");
const fuse = @import("fused_execute.zig");
const dispatcher = @import("dispatcher.zig");

const FusedExecutor = fuse.FusedExecutor;
const QWEN3_0_6B = fuse.QWEN3_0_6B;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    std.log.info("GPU-only engine starting", .{});
    var executor = try FusedExecutor.init(allocator, QWEN3_0_6B, .gpu_only);
    defer executor.deinit();
    try executor.loadModel();
    try executor.run();
}
