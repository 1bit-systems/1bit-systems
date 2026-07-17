//! video-lora-vk — Pure Zig Vulkan compute backend for SD1.5/AnimateDiff.
//!
//! Usage:
//!   zig build run -- --prompt "a cat walking" --frames 8
//!   zig build run -- --prompt "cinematic dolly zoom" --lora ./motion-lora.safetensors

const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const Instance = @import("vulkan/instance.zig").Instance;
const Tensor = @import("vulkan/buffer.zig").Tensor;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // ---- CLI args ----------------------------------------------------
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var prompt: []const u8 = "a cat walking, cinematic lighting";
    var frames: u32 = 8;
    var lora_path: ?[]const u8 = null;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--prompt") and i + 1 < args.len) {
            prompt = args[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--frames") and i + 1 < args.len) {
            frames = try std.fmt.parseInt(u32, args[i + 1], 10);
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--lora") and i + 1 < args.len) {
            lora_path = args[i + 1];
            i += 1;
        }
    }

    // ---- Init Vulkan -------------------------------------------------
    std.log.info("video-lora-vk initialising Vulkan compute...", .{});
    var inst = try Instance.create(allocator);
    defer inst.deinit();

    var props: c.VkPhysicalDeviceProperties = undefined;
    c.vkGetPhysicalDeviceProperties(inst.physical_device, &props);
    const device_name = @as([*:0]const u8, @ptrCast(&props.deviceName));
    std.log.info("GPU: {s}", .{device_name});

    // ---- Allocate tensors --------------------------------------------
    // SD1.5 latent: [1, 4, 64, 64] for a 512x512 image
    var latent = try Tensor.create(&inst, 1, 4, 64, 64);
    defer latent.deinit();
    latent.buffer.upload(@as([*]const u8, @ptrCast(&[_]f32{0.1} ** (1 * 4 * 64 * 64))));

    var text_emb = try Tensor.create(&inst, 1, 77, 768);
    defer text_emb.deinit();

    // ---- Generate ----------------------------------------------------
    const total_frames: usize = @intCast(frames);
    std.log.info("Generating {d} frames for prompt: {s}...", .{ total_frames, prompt });

    // For each frame, run the UNet denoising loop (50 steps each)
    for (0..total_frames) |frame_idx| {
        _ = frame_idx;
        // UNet step dispatches go here once pipelines are wired up.
        // For the scaffold: log progress.
        std.log.info("  frame {d}/{d} (scaffold — dispatch TBD)", .{ frame_idx + 1, total_frames });
    }

    std.log.info("Done! Generated {d} frames.", .{total_frames});
}
