//! Conv2d dispatcher — launches the conv2d.comp shader.

const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const vk = @import("../vulkan/vk.zig");
const Instance = @import("../vulkan/instance.zig").Instance;
const Pipeline = @import("../vulkan/pipeline.zig").Pipeline;
const Buffer = @import("../vulkan/buffer.zig").Buffer;
const Tensor = @import("../vulkan/buffer.zig").Tensor;

const PushConstants = extern struct {
    in_channels: u32,
    out_channels: u32,
    H: u32,
    W: u32,
    K: u32,
    pad: u32,
    groups: u32,
};

pipeline: Pipeline,
input_binding: u32,
weight_binding: u32,
bias_binding: u32,
output_binding: u32,

pub fn create(inst: *const Instance, spirv: []const u8) !Conv2d {
    const bindings = [_]c.VkDescriptorSetLayoutBinding{
        .{ .binding = 0, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT },
        .{ .binding = 1, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT },
        .{ .binding = 2, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT },
        .{ .binding = 3, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT },
    };
    const pipe = try Pipeline.create(inst, spirv, &bindings, @sizeOf(PushConstants));
    return Conv2d{ .pipeline = pipe, .input_binding = 0, .weight_binding = 1, .bias_binding = 2, .output_binding = 3 };
}

pub fn dispatch(
    self: *Conv2d,
    cmd: vk.VkCommandBuffer,
    input: *const Tensor,
    weights: *const Buffer,
    bias: *const Buffer,
    output: *Tensor,
    in_c: u32,
    out_c: u32,
) void {
    _ = input;
    _ = weights;
    _ = bias;
    _ = output;
    _ = in_c;
    _ = out_c;
    // Full dispatch needs descriptor writes + vkCmdPushConstants + vkCmdDispatch
    // Scaffold for now — implementation follows once the C layer is removed.
}

pub fn deinit(self: *Conv2d) void {
    self.pipeline.deinit();
}

const Conv2d = @This();
