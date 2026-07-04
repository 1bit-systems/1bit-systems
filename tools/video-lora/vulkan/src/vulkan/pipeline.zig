//! SPIR-V compute pipeline management.
//! Inspired by zinc/src/vulkan/pipeline.zig

const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const vk = @import("vk.zig");
const Instance = @import("instance.zig").Instance;

allocator: std.mem.Allocator,
device: vk.VkDevice,
shader_module: vk.VkShaderModule,
descriptor_set_layout: vk.VkDescriptorSetLayout,
pipeline_layout: vk.VkPipelineLayout,
pipeline: vk.VkPipeline,

pub fn create(
    inst: *const Instance,
    spirv_data: []const u8,
    descriptor_bindings: []const c.VkDescriptorSetLayoutBinding,
    push_constant_size: u32,
) !Pipeline {
    // ---- Shader module ------------------------------------------------
    const spirv = @as([*]const u32, @ptrCast(spirv_data.ptr));
    const spirv_word_count = @as(u32, @intCast(spirv_data.len / 4));

    var shader_module: vk.VkShaderModule = undefined;
    const sm_ci = c.VkShaderModuleCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv_data.len,
        .pCode = spirv,
    };
    try check(c.vkCreateShaderModule(inst.device, &sm_ci, null, &shader_module));

    // ---- Descriptor set layout ----------------------------------------
    var descriptor_set_layout: vk.VkDescriptorSetLayout = undefined;
    const dsl_ci = c.VkDescriptorSetLayoutCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = @intCast(descriptor_bindings.len),
        .pBindings = descriptor_bindings.ptr,
    };
    try check(c.vkCreateDescriptorSetLayout(inst.device, &dsl_ci, null, &descriptor_set_layout));

    // ---- Pipeline layout ----------------------------------------------
    var push_constant_range = c.VkPushConstantRange{
        .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = push_constant_size,
    };

    var pipeline_layout: vk.VkPipelineLayout = undefined;
    const pl_ci = c.VkPipelineLayoutCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
        .pushConstantRangeCount = if (push_constant_size > 0) 1 else 0,
        .pPushConstantRanges = if (push_constant_size > 0) &push_constant_range else null,
    };
    try check(c.vkCreatePipelineLayout(inst.device, &pl_ci, null, &pipeline_layout));

    // ---- Compute pipeline ---------------------------------------------
    const stage = c.VkPipelineShaderStageCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = c.VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main",
    };

    var pipeline: vk.VkPipeline = undefined;
    const cp_ci = c.VkComputePipelineCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage,
        .layout = pipeline_layout,
    };
    try check(c.vkCreateComputePipelines(inst.device, null, 1, &cp_ci, null, &pipeline));

    return Pipeline{
        .allocator = inst.allocator,
        .device = inst.device,
        .shader_module = shader_module,
        .descriptor_set_layout = descriptor_set_layout,
        .pipeline_layout = pipeline_layout,
        .pipeline = pipeline,
    };
}

pub fn deinit(self: *Pipeline) void {
    c.vkDestroyPipeline(self.device, self.pipeline, null);
    c.vkDestroyPipelineLayout(self.device, self.pipeline_layout, null);
    c.vkDestroyDescriptorSetLayout(self.device, self.descriptor_set_layout, null);
    c.vkDestroyShaderModule(self.device, self.shader_module, null);
}

const Pipeline = @This();

fn check(result: c.VkResult) !void {
    if (result != c.VK_SUCCESS) {
        std.log.err("Vulkan pipeline error: {d}", .{@intFromEnum(result)});
        return error.VulkanPipelineError;
    }
}
