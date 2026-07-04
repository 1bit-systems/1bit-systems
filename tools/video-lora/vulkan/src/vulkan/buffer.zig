//! Device buffer and tensor management.

const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const vk = @import("vk.zig");
const Instance = @import("instance.zig").Instance;

allocator: std.mem.Allocator,
device: vk.VkDevice,
buffer: vk.VkBuffer,
memory: vk.VkDeviceMemory,
size: usize,

/// Allocate a device buffer with the given size and usage flags.
pub fn create(inst: *const Instance, size: usize, usage: c.VkBufferUsageFlags) !Buffer {
    const buffer_ci = c.VkBufferCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
    };

    var buffer: vk.VkBuffer = undefined;
    try check(c.vkCreateBuffer(inst.device, &buffer_ci, null, &buffer));

    var mem_req: c.VkMemoryRequirements = undefined;
    c.vkGetBufferMemoryRequirements(inst.device, buffer, &mem_req);

    const mem_type = inst.findMemoryType(mem_req.memoryTypeBits, c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) orelse
        inst.findMemoryType(mem_req.memoryTypeBits, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) orelse
        return error.NoSuitableMemory;

    const alloc_ci = c.VkMemoryAllocateInfo{
        .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_req.size,
        .memoryTypeIndex = mem_type,
    };

    var memory: vk.VkDeviceMemory = undefined;
    try check(c.vkAllocateMemory(inst.device, &alloc_ci, null, &memory));
    try check(c.vkBindBufferMemory(inst.device, buffer, memory, 0));

    return Buffer{
        .allocator = inst.allocator,
        .device = inst.device,
        .buffer = buffer,
        .memory = memory,
        .size = size,
    };
}

/// Create a staging buffer (host-visible, for upload/download).
pub fn createStaging(inst: *const Instance, size: usize) !Buffer {
    return create(inst, size, c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | c.VK_BUFFER_USAGE_TRANSFER_DST_BIT);
}

/// Upload data to the device buffer.
pub fn upload(self: *Buffer, data: []const u8) !void {
    var ptr: ?*anyopaque = null;
    try check(c.vkMapMemory(self.device, self.memory, 0, self.size, 0, &ptr));
    defer c.vkUnmapMemory(self.device, self.memory);
    @memcpy(@as([*]u8, @ptrCast(ptr))[0..data.len], data);
}

/// Download data from the device buffer.
pub fn download(self: *Buffer, data: []u8) !void {
    var ptr: ?*anyopaque = null;
    try check(c.vkMapMemory(self.device, self.memory, 0, self.size, 0, &ptr));
    defer c.vkUnmapMemory(self.device, self.memory);
    @memcpy(data, @as([*]u8, @ptrCast(ptr))[0..data.len]);
}

pub fn deinit(self: *Buffer) void {
    c.vkDestroyBuffer(self.device, self.buffer, null);
    c.vkFreeMemory(self.device, self.memory, null);
}

// ---- Tensor (NCHW float buffer for UNet) ---------------------------

pub const Tensor = struct {
    buffer: Buffer,
    batch: u32,
    channels: u32,
    height: u32,
    width: u32,

    pub fn create(inst: *const Instance, b: u32, c: u32, h: u32, w: u32) !Tensor {
        const size = b * c * h * w * @sizeOf(f32);
        const buf = try Buffer.create(inst, size, c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | c.VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        return Tensor{ .buffer = buf, .batch = b, .channels = c, .height = h, .width = w };
    }

    pub fn elements(self: *const Tensor) u32 {
        return self.batch * self.channels * self.height * self.width;
    }

    pub fn sizeBytes(self: *const Tensor) usize {
        return self.elements() * @sizeOf(f32);
    }

    pub fn deinit(self: *Tensor) void {
        self.buffer.deinit();
    }
};

const Buffer = @This();

fn check(result: c.VkResult) !void {
    if (result != c.VK_SUCCESS) {
        std.log.err("Vulkan buffer error: {d}", .{@intFromEnum(result)});
        return error.VulkanBufferError;
    }
}
