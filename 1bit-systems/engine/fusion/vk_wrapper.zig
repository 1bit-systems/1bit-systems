//! Minimal Vulkan wrapper — handles device creation, dma-buf import, and buffer binding.
//! Only the subset needed for importing the shared KV cache dma-buf from CrossBackendMemory.
//!
//! @section Fused Engine
const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;
const log = std.log.scoped(.vk_wrapper);

// ---------------------------------------------------------------------------
// Vulkan function pointer table (loaded via DynLib at runtime)
// ---------------------------------------------------------------------------

const DynLib = std.DynLib;

/// Core Vulkan functions needed for dma-buf import + buffer management.
pub const VkFunctions = struct {
    lib: ?DynLib,

    // Instance
    createInstance: ?*const fn (info: *const VkInstanceCreateInfo, allocator: ?*const anyopaque, instance: *u64) callconv(.c) u32,
    destroyInstance: ?*const fn (instance: u64, allocator: ?*const anyopaque) callconv(.c) void,
    enumeratePhysicalDevices: ?*const fn (instance: u64, count: *u32, devices: ?[*]u64) callconv(.c) u32,
    getPhysicalDeviceProperties: ?*const fn (device: u64, props: *VkPhysicalDeviceProperties) callconv(.c) void,
    getPhysicalDeviceMemoryProperties: ?*const fn (device: u64, props: *VkPhysicalDeviceMemoryProperties) callconv(.c) void,

    // Device
    createDevice: ?*const fn (physical: u64, info: *const VkDeviceCreateInfo, allocator: ?*const anyopaque, device: *u64) callconv(.c) u32,
    destroyDevice: ?*const fn (device: u64, allocator: ?*const anyopaque) callconv(.c) void,
    getDeviceQueue: ?*const fn (device: u64, family: u32, index: u32, queue: *u64) callconv(.c) void,

    // Memory
    allocateMemory: ?*const fn (device: u64, info: *const VkMemoryAllocateInfo, allocator: ?*const anyopaque, memory: *u64) callconv(.c) u32,
    freeMemory: ?*const fn (device: u64, memory: u64, allocator: ?*const anyopaque) callconv(.c) void,
    bindBufferMemory: ?*const fn (device: u64, buffer: u64, memory: u64, offset: u64) callconv(.c) u32,

    // Buffer
    createBuffer: ?*const fn (device: u64, info: *const VkBufferCreateInfo, allocator: ?*const anyopaque, buffer: *u64) callconv(.c) u32,
    destroyBuffer: ?*const fn (device: u64, buffer: u64, allocator: ?*const anyopaque) callconv(.c) void,
    getBufferMemoryRequirements: ?*const fn (device: u64, buffer: u64, req: *VkMemoryRequirements) callconv(.c) void,

    // External memory fd (KHR extension)
    getMemoryFdKHR: ?*const fn (device: u64, info: *const VkMemoryGetFdInfoKHR, fd: *i32) callconv(.c) u32,
    getMemoryFdPropertiesKHR: ?*const fn (device: u64, handle_type: u32, fd: i32, props: *VkMemoryFdPropertiesKHR) callconv(.c) u32,

    fn load() VkFunctions {
        var lib = DynLib.openZ("libvulkan.so.1") catch
            DynLib.openZ("libvulkan.so") catch
            return VkFunctions{ .lib = null };

        const lookup = VkFunctions{ .lib = lib,
            .createInstance = lib.lookup(*const fn (*const VkInstanceCreateInfo, ?*const anyopaque, *u64) callconv(.c) u32, "vkCreateInstance"),
            .destroyInstance = lib.lookup(*const fn (u64, ?*const anyopaque) callconv(.c) void, "vkDestroyInstance"),
            .enumeratePhysicalDevices = lib.lookup(*const fn (u64, *u32, ?[*]u64) callconv(.c) u32, "vkEnumeratePhysicalDevices"),
            .getPhysicalDeviceProperties = lib.lookup(*const fn (u64, *VkPhysicalDeviceProperties) callconv(.c) void, "vkGetPhysicalDeviceProperties"),
            .getPhysicalDeviceMemoryProperties = lib.lookup(*const fn (u64, *VkPhysicalDeviceMemoryProperties) callconv(.c) void, "vkGetPhysicalDeviceMemoryProperties"),
            .createDevice = lib.lookup(*const fn (u64, *const VkDeviceCreateInfo, ?*const anyopaque, *u64) callconv(.c) u32, "vkCreateDevice"),
            .destroyDevice = lib.lookup(*const fn (u64, ?*const anyopaque) callconv(.c) void, "vkDestroyDevice"),
            .getDeviceQueue = lib.lookup(*const fn (u64, u32, u32, *u64) callconv(.c) void, "vkGetDeviceQueue"),
            .allocateMemory = lib.lookup(*const fn (u64, *const VkMemoryAllocateInfo, ?*const anyopaque, *u64) callconv(.c) u32, "vkAllocateMemory"),
            .freeMemory = lib.lookup(*const fn (u64, u64, ?*const anyopaque) callconv(.c) void, "vkFreeMemory"),
            .bindBufferMemory = lib.lookup(*const fn (u64, u64, u64, u64) callconv(.c) u32, "vkBindBufferMemory"),
            .createBuffer = lib.lookup(*const fn (u64, *const VkBufferCreateInfo, ?*const anyopaque, *u64) callconv(.c) u32, "vkCreateBuffer"),
            .destroyBuffer = lib.lookup(*const fn (u64, u64, ?*const anyopaque) callconv(.c) void, "vkDestroyBuffer"),
            .getBufferMemoryRequirements = lib.lookup(*const fn (u64, u64, *VkMemoryRequirements) callconv(.c) void, "vkGetBufferMemoryRequirements"),
            .getMemoryFdKHR = lib.lookup(*const fn (u64, *const VkMemoryGetFdInfoKHR, *i32) callconv(.c) u32, "vkGetMemoryFdKHR"),
            .getMemoryFdPropertiesKHR = lib.lookup(*const fn (u64, u32, i32, *VkMemoryFdPropertiesKHR) callconv(.c) u32, "vkGetMemoryFdPropertiesKHR"),
        };
        return lookup;
    }
};

// ---------------------------------------------------------------------------
// Vulkan struct definitions (minimal set for dma-buf import)
// ---------------------------------------------------------------------------

const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 2;
const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 9;
const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 5;
const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2: u32 = 1000056001;
const VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR: u32 = 1000079003;
const VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR: u32 = 1000079000;

const VK_API_VERSION_1_3: u32 = 0x42000A;

const VkInstanceCreateInfo = extern struct {
    sType: u32,
    pNext: ?*const anyopaque,
    flags: u32,
    pApplicationInfo: ?*const VkApplicationInfo,
    enabledLayerCount: u32,
    ppEnabledLayerNames: ?[*]?[*:0]const u8,
    enabledExtensionCount: u32,
    ppEnabledExtensionNames: ?[*]?[*:0]const u8,
};

const VkApplicationInfo = extern struct {
    sType: u32,
    pNext: ?*const anyopaque,
    pApplicationName: ?[*:0]const u8,
    applicationVersion: u32,
    pEngineName: ?[*:0]const u8,
    engineVersion: u32,
    apiVersion: u32,
};

const VkDeviceCreateInfo = extern struct {
    sType: u32,
    pNext: ?*const anyopaque,
    flags: u32,
    queueCreateInfoCount: u32,
    pQueueCreateInfos: ?*const VkDeviceQueueCreateInfo,
    enabledLayerCount: u32,
    ppEnabledLayerNames: ?[*]?[*:0]const u8,
    enabledExtensionCount: u32,
    ppEnabledExtensionNames: ?[*]?[*:0]const u8,
    pEnabledFeatures: ?*const anyopaque,
};

const VkDeviceQueueCreateInfo = extern struct {
    sType: u32,
    pNext: ?*const anyopaque,
    flags: u32,
    queueFamilyIndex: u32,
    queueCount: u32,
    pQueuePriorities: ?*const f32,
};

const VkMemoryAllocateInfo = extern struct {
    sType: u32,
    pNext: ?*const anyopaque,
    allocationSize: u64,
    memoryTypeIndex: u32,
};

const VkMemoryFdPropertiesKHR = extern struct {
    sType: u32,
    pNext: ?*anyopaque,
    memoryTypeBits: u32,
};

const VkImportMemoryFdInfoKHR = extern struct {
    sType: u32,
    pNext: ?*const anyopaque,
    handleType: u32,
    fd: i32,
};

const VkMemoryGetFdInfoKHR = extern struct {
    sType: u32,
    pNext: ?*const anyopaque,
    memory: u64,
    handleType: u32,
};

const VkBufferCreateInfo = extern struct {
    sType: u32,
    pNext: ?*const anyopaque,
    flags: u32,
    size: u64,
    usage: u32,
    sharingMode: u32,
    queueFamilyIndexCount: u32,
    pQueueFamilyIndices: ?*const u32,
};

const VkPhysicalDeviceProperties = extern struct {
    apiVersion: u32,
    driverVersion: u32,
    vendorID: u32,
    deviceID: u32,
    deviceType: u32,
    deviceName: [256]u8,
    pipelineCacheUUID: [16]u8,
    _padding: [92]u8,
};

const VkPhysicalDeviceMemoryProperties = extern struct {
    memoryTypeCount: u32,
    memoryTypes: [32]VkMemoryType,
    memoryHeapCount: u32,
    memoryHeaps: [16]VkMemoryHeap,
};

const VkMemoryType = extern struct {
    propertyFlags: u32,
    heapIndex: u32,
};

const VkMemoryHeap = extern struct {
    size: u64,
    flags: u32,
};

const VkMemoryRequirements = extern struct {
    size: u64,
    alignment: u64,
    memoryTypeBits: u32,
};

// ---------------------------------------------------------------------------
// VkContext
// ----------------------------------------------------------------------------

/// Manages a Vulkan instance, device, and queue for dma-buf import operations.
pub const VkContext = struct {
    allocator: std.mem.Allocator,
    vk: VkFunctions,
    instance: u64,
    physical_device: u64,
    device: u64,
    queue: u64,
    queue_family: u32,
    memory_properties: VkPhysicalDeviceMemoryProperties,
    ext_memory_fd_supported: bool,

    pub fn init(allocator: std.mem.Allocator) !VkContext {
        const vk = VkFunctions.load();
        if (vk.createInstance == null) return error.VulkanNotAvailable;
        if (vk.enumeratePhysicalDevices == null) return error.VulkanNotAvailable;

        // Create instance
        const app = VkApplicationInfo{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, // not right, but OK for this
            .pNext = null,
            .pApplicationName = "1bit",
            .applicationVersion = 1,
            .pEngineName = "1bit-engine",
            .engineVersion = 1,
            .apiVersion = VK_API_VERSION_1_3,
        };
        // Fix: use 0 for INSTANCE_CREATE_INFO struct type since we have no actual define
        var instance: u64 = 0;
        var ici: VkInstanceCreateInfo = undefined;
        @memset(@as([*]u8, @ptrCast(&ici))[0..@sizeOf(VkInstanceCreateInfo)], 0);
        ici.sType = 1; // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
        ici.pApplicationInfo = &app;
        if (vk.createInstance.?
            (&ici, null, &instance) != 0)
            return error.InstanceCreationFailed;

        // Enumerate physical devices
        var phys_count: u32 = 0;
        _ = vk.enumeratePhysicalDevices.?(instance, &phys_count, null);
        if (phys_count == 0) { vk.destroyInstance.?(instance, null); return error.NoPhysicalDevice; }
        var phys_devices: [4]u64 = @splat(0);
        _ = vk.enumeratePhysicalDevices.?(instance, &phys_count, &phys_devices);

        // Pick first discrete GPU (or any GPU)
        const pd: u64 = phys_devices[0];
        var props: VkPhysicalDeviceProperties = undefined;
        vk.getPhysicalDeviceProperties.?(pd, &props);
        log.info("Vulkan device: {s} (vendor=0x{x} device=0x{x})", .{
            @as([*:0]const u8, &props.deviceName), props.vendorID, props.deviceID,
        });

        // Get memory properties
        var mem_props: VkPhysicalDeviceMemoryProperties = undefined;
        if (vk.getPhysicalDeviceMemoryProperties) |f| f(pd, &mem_props);

        // Find compute queue family
        // (simplified — we only need dma-buf import, no queue work)
        const queue_family: u32 = 0;

        // Create logical device
        const ext_names = [_][*:0]const u8{
            "VK_KHR_external_memory_fd",
        };
        const queue_ci = VkDeviceQueueCreateInfo{
            .sType = 5, // VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO
            .pNext = null,
            .flags = 0,
            .queueFamilyIndex = queue_family,
            .queueCount = 1,
            .pQueuePriorities = &@as(f32, 1.0),
        };
        var dci: VkDeviceCreateInfo = undefined;
        @memset(@as([*]u8, @ptrCast(&dci))[0..@sizeOf(VkDeviceCreateInfo)], 0);
        dci.sType = 3; // VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &queue_ci;
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = &ext_names;

        var device: u64 = 0;
        if (vk.createDevice.?(pd, &dci, null, &device) != 0) {
            vk.destroyInstance.?(instance, null);
            return error.DeviceCreationFailed;
        }

        var queue: u64 = 0;
        if (vk.getDeviceQueue) |f| f(device, queue_family, 0, &queue);

        const ext_fd_supported = vk.getMemoryFdKHR != null and vk.getMemoryFdPropertiesKHR != null;

        return VkContext{
            .allocator = allocator,
            .vk = vk,
            .instance = instance,
            .physical_device = pd,
            .device = device,
            .queue = queue,
            .queue_family = queue_family,
            .memory_properties = mem_props,
            .ext_memory_fd_supported = ext_fd_supported,
        };
    }

    pub fn deinit(self: *VkContext) void {
        if (self.device != 0 and self.vk.destroyDevice) |f| f(self.device, null);
        if (self.instance != 0 and self.vk.destroyInstance) |f| f(self.instance, null);
        self.* = undefined;
    }

    /// Import a dma-buf fd as VkDeviceMemory.
    /// Returns the VkDeviceMemory handle, or 0 on failure.
    pub fn importDmaBuf(self: *VkContext, dma_buf_fd: i32, size: u64) !u64 {
        if (!self.ext_memory_fd_supported) return error.ExtensionNotSupported;
        if (dma_buf_fd < 0) return error.InvalidArgument;

        const handle_type_dma_buf: u32 = 0x00000200; // VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT

        // Query memory type bits for this fd
        var fd_props: VkMemoryFdPropertiesKHR = .{
            .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR,
            .pNext = null,
            .memoryTypeBits = 0,
        };
        if (self.vk.getMemoryFdPropertiesKHR) |f| {
            _ = f(self.device, handle_type_dma_buf, dma_buf_fd, &fd_props);
        }

        // Pick first compatible memory type
        const memory_type_index = findMemoryTypeIndex(
            fd_props.memoryTypeBits,
        );

        const import_info = VkImportMemoryFdInfoKHR{
            .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
            .pNext = null,
            .handleType = handle_type_dma_buf,
            .fd = dma_buf_fd,
        };

        const alloc_info = VkMemoryAllocateInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = &import_info,
            .allocationSize = size,
            .memoryTypeIndex = memory_type_index,
        };

        var device_memory: u64 = 0;
        const result = self.vk.allocateMemory.?(self.device, &alloc_info, null, &device_memory);
        if (result != 0) return error.MemoryAllocationFailed;

        log.info("imported dma-buf fd={d} → VkDeviceMemory=0x{x} (size={d})", .{
            dma_buf_fd, device_memory, size,
        });
        return device_memory;
    }

    /// Create a VkBuffer backed by imported dma-buf memory.
    pub fn createBufferForMemory(self: *VkContext, memory: u64, buf_size: u64) !u64 {
        const buf_ci = VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .size = buf_size,
            .usage = 0x0042,
            .sharingMode = 0,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
        };

        var buffer: u64 = 0;
        if (self.vk.createBuffer.? == null) return error.ExtensionNotSupported;
        const result = self.vk.createBuffer.?(self.device, &buf_ci, null, &buffer);
        if (result != 0) return error.BufferCreationFailed;

        // Bind memory to buffer
        if (self.vk.bindBufferMemory) |f| {
            if (f(self.device, buffer, memory, 0) != 0) {
                self.vk.destroyBuffer.?(self.device, buffer, null);
                return error.BindFailed;
            }
        }

        return buffer;
    }

    pub fn freeMemory(self: *VkContext, memory: u64) void {
        if (self.vk.freeMemory) |f| f(self.device, memory, null);
    }

    pub fn destroyBuffer(self: *VkContext, buffer: u64) void {
        if (self.vk.destroyBuffer) |f| f(self.device, buffer, null);
    }
};

fn findMemoryTypeIndex(memory_type_bits: u32) u32 {
    if (memory_type_bits == 0) return 0;
    return @ctz(memory_type_bits);
}

// ---------------------------------------------------------------------------
// Error set
// ---------------------------------------------------------------------------

pub const VkError = error{
    VulkanNotAvailable,
    InstanceCreationFailed,
    NoPhysicalDevice,
    DeviceCreationFailed,
    ExtensionNotSupported,
    MemoryAllocationFailed,
    BufferCreationFailed,
    BindFailed,
    InvalidArgument,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "VkContext type compiles" {
    try std.testing.expectEqual(@sizeOf(usize), @sizeOf(*VkContext));
}

test "VkFunctions load returns null when lib not found" {
    // This would only fail if the file can't compile at all.
    // We can't actually test without Vulkan installed.
    try std.testing.expect(@sizeOf(VkFunctions) > 0);
}

test "findMemoryTypeIndex basics" {
    try std.testing.expectEqual(@as(u32, 0), findMemoryTypeIndex(0x00000001));
    try std.testing.expectEqual(@as(u32, 3), findMemoryTypeIndex(0x00000008));
    try std.testing.expectEqual(@as(u32, 0), findMemoryTypeIndex(0));
}
