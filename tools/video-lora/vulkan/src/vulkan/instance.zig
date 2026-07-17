//! Vulkan compute instance — device selection, queue, memory allocator.
//! Inspired by zinc/src/vulkan/instance.zig

const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const vk = @import("vk.zig");

allocator: std.mem.Allocator,
instance: vk.VkInstance,
physical_device: vk.VkPhysicalDevice,
device: vk.VkDevice,
queue: vk.VkQueue,
queue_family: u32,
memory_props: c.VkPhysicalDeviceMemoryProperties,

pub fn create(allocator: std.mem.Allocator) !Instance {
    // ---- Instance ------------------------------------------------------
    const app_info = c.VkApplicationInfo{
        .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "video-lora",
        .applicationVersion = c.VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "video-lora-vk",
        .engineVersion = c.VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = vk.API_VERSION,
    };

    var instance: vk.VkInstance = undefined;
    const inst_ci = c.VkInstanceCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = null,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = null,
    };
    try check(c.vkCreateInstance(&inst_ci, null, &instance));

    // ---- Physical device (first compute-capable) -----------------------
    var device_count: u32 = 0;
    try check(c.vkEnumeratePhysicalDevices(instance, &device_count, null));
    if (device_count == 0) return error.NoVulkanDevice;

    const devices = try allocator.alloc(vk.VkPhysicalDevice, device_count);
    defer allocator.free(devices);
    try check(c.vkEnumeratePhysicalDevices(instance, &device_count, devices.ptr));

    const physical_device = blk: {
        for (devices) |pd| {
            var props: c.VkPhysicalDeviceProperties = undefined;
            c.vkGetPhysicalDeviceProperties(pd, &props);
            if (props.deviceType == c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU or
                props.deviceType == c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            {
                break :blk pd;
            }
        }
        break :blk devices[0];
    };

    // ---- Queue family (compute) ---------------------------------------
    var queue_family_count: u32 = 0;
    c.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, null);
    const queue_families = try allocator.alloc(c.VkQueueFamilyProperties, queue_family_count);
    defer allocator.free(queue_families);
    c.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.ptr);

    const queue_family = blk: {
        for (queue_families, 0..) |qf, i| {
            if (qf.queueFlags & c.VK_QUEUE_COMPUTE_BIT != 0) break :blk @as(u32, @intCast(i));
        }
        return error.NoComputeQueue;
    };

    // ---- Logical device ------------------------------------------------
    const queue_priority = [_]f32{1.0};
    const queue_ci = c.VkDeviceQueueCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };

    var device: vk.VkDevice = undefined;
    const device_ci = c.VkDeviceCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_ci,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = null,
        .pEnabledFeatures = null,
    };
    try check(c.vkCreateDevice(physical_device, &device_ci, null, &device));

    // ---- Queue ---------------------------------------------------------
    var queue: vk.VkQueue = undefined;
    c.vkGetDeviceQueue(device, queue_family, 0, &queue);

    // ---- Memory properties --------------------------------------------
    var memory_props: c.VkPhysicalDeviceMemoryProperties = undefined;
    c.vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_props);

    // ---- Device properties (for logging) --------------------------------
    var props: c.VkPhysicalDeviceProperties = undefined;
    c.vkGetPhysicalDeviceProperties(physical_device, &props);
    std.log.info("vk: {s} — compute queue: {}", .{
        @as([*:0]const u8, @ptrCast(&props.deviceName)),
        .{queue_family},
    });

    return Instance{
        .allocator = allocator,
        .instance = instance,
        .physical_device = physical_device,
        .device = device,
        .queue = queue,
        .queue_family = queue_family,
        .memory_props = memory_props,
    };
}

pub fn deinit(self: *Instance) void {
    c.vkDestroyDevice(self.device, null);
    c.vkDestroyInstance(self.instance, null);
}

/// Find a memory type that satisfies the requested properties.
pub fn findMemoryType(self: *const Instance, type_filter: u32, properties: c.VkMemoryPropertyFlags) ?u32 {
    for (0..self.memory_props.memoryTypeCount) |i| {
        if (type_filter & (@as(u32, 1) << @truncate(i)) != 0 and
            self.memory_props.memoryTypes[i].propertyFlags & properties == properties)
        {
            return @truncate(i);
        }
    }
    return null;
}

fn check(result: c.VkResult) !void {
    if (result != c.VK_SUCCESS) {
        std.log.err("Vulkan call failed with result: {d}", .{@intFromEnum(result)});
        return error.VulkanError;
    }
}

const Instance = @This();
