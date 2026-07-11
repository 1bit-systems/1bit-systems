//! Standalone GPU flash attention module wrapping Vulkan compute with ZINC's SPIR-V shader blobs.
//!
//! Owns a lightweight Vulkan runtime (instance, device, compute queue, command
//! resources) and loads ZINC's precompiled flash_attn.spv / flash_attn_batched.spv
//! shaders. The public API accepts CPU-side f32 slices, handles device upload,
//! dispatch, and output readback transparently.
//!
//! Push-constant layouts are exact mirrors of ZINC's `FlashAttnPush` and
//! `FlashAttnBatchedPush` so any ZINC-compiled flash-attention SPIR-V blob
//! works without modification.
//!
//! ## Example
//! ```
//! var gpu = try GpuAttention.init(allocator, "/path/to/zinc/shaders");
//! defer gpu.deinit();
//!
//! try gpu.flashAttention(q, k_cache, v_cache, page_table, output, sinks,
//!     n_heads, n_kv_heads, head_dim, seq_len, page_size, attn_scale, sink_offset);
//! ```
//!
//! @section Fused Engine

const std = @import("std");

// Vulkan bindings — re-exported through vk_wrapper wrapper module.
const vk = @import("vulkan_c");

const log = std.log.scoped(.gpu_attn);

// ─────────────────────────────────────────────────────────────────────────────
// Push-constant structs  (1:1 ABI mirrors of ZINC's attention.zig)
// ─────────────────────────────────────────────────────────────────────────────

/// Push constants for `flash_attn.spv` — single-query decode dispatch.
/// Each workgroup processes one query head: grid = (n_heads, 1, 1).
pub const FlashAttnPush = extern struct {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    seq_len: u32,
    page_size: u32,
    attn_scale_bits: u32,
    sink_offset: u32,
};

/// Push constants for `flash_attn_batched.spv` — N-query prefill or decode.
/// Grid = (n_heads, n_queries, 1); each (head, query) workgroup uses
/// causal_len = seq_start + query + 1.
pub const FlashAttnBatchedPush = extern struct {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    seq_start: u32,
    n_queries: u32,
    page_size: u32,
    attn_scale_bits: u32,
    sink_offset: u32,
};

// ─────────────────────────────────────────────────────────────────────────────
// Internal Vulkan wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// A compute pipeline owning its shader module, descriptor layout, pipeline
/// layout, and VkPipeline handle.
const Pipeline = struct {
    shader_module: vk.VkShaderModule,
    descriptor_set_layout: vk.VkDescriptorSetLayout,
    pipeline_layout: vk.VkPipelineLayout,
    handle: vk.VkPipeline,
    device: vk.VkDevice,

    fn deinit(self: *Pipeline) void {
        vk.vkDestroyPipeline(self.device, self.handle, null);
        vk.vkDestroyPipelineLayout(self.device, self.pipeline_layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.descriptor_set_layout, null);
        vk.vkDestroyShaderModule(self.device, self.shader_module, null);
        self.* = undefined;
    }
};

/// A device-memory-backed buffer with optional host mapping.
const Buffer = struct {
    handle: vk.VkBuffer,
    memory: vk.VkDeviceMemory,
    size: vk.VkDeviceSize,
    device: vk.VkDevice,
    mapped_ptr: ?[*]u8,

    fn map(self: *Buffer, offset: vk.VkDeviceSize, size: vk.VkDeviceSize) ![*]u8 {
        var ptr: ?*anyopaque = null;
        const result = vk.vkMapMemory(self.device, self.memory, offset, size, 0, &ptr);
        if (result != vk.VK_SUCCESS) return error.BufferMapFailed;
        self.mapped_ptr = @ptrCast(@alignCast(ptr));
        return self.mapped_ptr.?;
    }

    fn unmap(self: *Buffer) void {
        if (self.mapped_ptr != null) {
            vk.vkUnmapMemory(self.device, self.memory);
            self.mapped_ptr = null;
        }
    }

    fn deinit(self: *Buffer) void {
        self.unmap();
        vk.vkDestroyBuffer(self.device, self.handle, null);
        vk.vkFreeMemory(self.device, self.memory, null);
        self.* = undefined;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// StagingPool — persistent pre-allocated staging buffers
// ─────────────────────────────────────────────────────────────────────────────

/// Pool of reusable host-visible staging buffers for attention inputs/outputs.
/// Pre-allocates on first use, grows on demand, and avoids per-call
/// vkCreateBuffer / vkAllocateMemory overhead on the hot path.
///
/// Indices map to descriptor bindings:
///   0 = Q, 1 = K, 2 = V, 3 = page_table, 4 = output, 5 = sinks
const StagingPool = struct {
    const POOL_SIZE = 6;

    const PoolEntry = struct {
        buffer: Buffer = undefined,
        capacity: vk.VkDeviceSize = 0,
        initialized: bool = false,
    };

    entries: [POOL_SIZE]PoolEntry = undefined,
    device: vk.VkDevice = null,
    memory_type: u32 = 0,
    mem_props: *const vk.VkPhysicalDeviceMemoryProperties = undefined,

    /// Initialise the pool with a Vulkan device and the desired memory type.
    /// No buffers are allocated until the first `acquire()` call.
    fn init(self: *StagingPool, device: vk.VkDevice, memory_type: u32, mem_props: *const vk.VkPhysicalDeviceMemoryProperties) void {
        self.device = device;
        self.memory_type = memory_type;
        self.mem_props = mem_props;
        // Ensure all entries start as uninitialised (initialized = false).
        for (&self.entries) |*e| {
            e.* = .{};
        }
    }

    /// Return a pointer to the pool buffer at `index`, ensuring it can hold
    /// at least `required_size` bytes.  Grows (reallocates) if the current
    /// capacity is insufficient.  Sets `buf.size = required_size` so that
    /// descriptor-bind ranges reflect the active data size.
    fn acquire(self: *StagingPool, index: usize, required_size: vk.VkDeviceSize) !*Buffer {
        const entry = &self.entries[index];
        if (entry.initialized and entry.capacity >= required_size) {
            entry.buffer.size = required_size;
            return &entry.buffer;
        }
        // Grow: destroy old allocation if present, then create a larger one.
        if (entry.initialized) {
            entry.buffer.deinit();
        }
        entry.buffer = try createPoolBuffer(self.device, required_size, self.memory_type, self.mem_props);
        entry.capacity = required_size;
        entry.buffer.size = required_size;
        entry.initialized = true;
        return &entry.buffer;
    }

    /// Release all pool buffers.  Called from `GpuAttention.deinit()`.
    fn deinit(self: *StagingPool) void {
        for (&self.entries) |*e| {
            if (e.initialized) {
                e.buffer.deinit();
            }
        }
        self.* = undefined;
    }
};

/// Internal: create a host-visible staging buffer (used by StagingPool and
/// as fallback when on-demand allocation is needed).
fn createPoolBuffer(
    device: vk.VkDevice,
    size: vk.VkDeviceSize,
    memory_type: u32,
    mem_props: *const vk.VkPhysicalDeviceMemoryProperties,
) !Buffer {
    // Vulkan requires buffers to have a non-zero size. Use at least 4 bytes.
    const safe_size = @max(size, @as(vk.VkDeviceSize, 4));
    const buf_ci = vk.VkBufferCreateInfo{
        .sType = vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .size = safe_size,
        .usage = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = null,
    };
    var buf_handle: vk.VkBuffer = null;
    if (vk.vkCreateBuffer(device, &buf_ci, null, &buf_handle) != vk.VK_SUCCESS) {
        return error.BufferCreateFailed;
    }
    errdefer vk.vkDestroyBuffer(device, buf_handle, null);

    var req: vk.VkMemoryRequirements = undefined;
    vk.vkGetBufferMemoryRequirements(device, buf_handle, &req);

    // Find a memory type that is BOTH compatible with this buffer's
    // memoryTypeBits AND has the desired HOST_VISIBLE | HOST_COHERENT flags.
    // Using the caller-supplied memory_type as fallback.
    var actual_type = memory_type;
    {
        const mask = req.memoryTypeBits;
        for (0..32) |i| {
            if (i >= mem_props.memoryTypeCount) break;
            const bit: u5 = @truncate(i);
            if ((mask & (@as(u32, 1) << bit)) != 0 and
                (mem_props.memoryTypes[i].propertyFlags &
                (vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            {
                actual_type = @intCast(i);
                break;
            }
        }
    }

    const alloc_info = vk.VkMemoryAllocateInfo{
        .sType = vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = null,
        .allocationSize = req.size,
        .memoryTypeIndex = actual_type,
    };
    var mem: vk.VkDeviceMemory = null;
    if (vk.vkAllocateMemory(device, &alloc_info, null, &mem) != vk.VK_SUCCESS) {
        return error.BufferMemoryAllocFailed;
    }
    errdefer vk.vkFreeMemory(device, mem, null);

    if (vk.vkBindBufferMemory(device, buf_handle, mem, 0) != vk.VK_SUCCESS) {
        return error.BufferBindFailed;
    }

    return Buffer{
        .handle = buf_handle,
        .memory = mem,
        .size = size,
        .device = device,
        .mapped_ptr = null,
    };
}

/// Create a buffer with custom usage flags and memory type.
/// Used for device-local KV cache buffers.
fn createCustomBuffer(
    device: vk.VkDevice,
    size: vk.VkDeviceSize,
    usage: u32,
    memory_type: u32,
    mem_props: *const vk.VkPhysicalDeviceMemoryProperties,
) !Buffer {
    const safe_size = @max(size, @as(vk.VkDeviceSize, 4));
    const buf_ci = vk.VkBufferCreateInfo{
        .sType = vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .size = safe_size,
        .usage = usage,
        .sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = null,
    };
    var buf_handle: vk.VkBuffer = null;
    if (vk.vkCreateBuffer(device, &buf_ci, null, &buf_handle) != vk.VK_SUCCESS) {
        return error.BufferCreateFailed;
    }
    errdefer vk.vkDestroyBuffer(device, buf_handle, null);

    var req: vk.VkMemoryRequirements = undefined;
    vk.vkGetBufferMemoryRequirements(device, buf_handle, &req);

    // Find a compatible memory type with the requested property flags
    var actual_type = memory_type;
    {
        const mask = req.memoryTypeBits;
        const target_flags = mem_props.memoryTypes[memory_type].propertyFlags;
        for (0..32) |i| {
            if (i >= mem_props.memoryTypeCount) break;
            const bit: u5 = @truncate(i);
            if ((mask & (@as(u32, 1) << bit)) != 0 and
                (mem_props.memoryTypes[i].propertyFlags & target_flags) == target_flags)
            {
                actual_type = @intCast(i);
                break;
            }
        }
    }

    const alloc_info = vk.VkMemoryAllocateInfo{
        .sType = vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = null,
        .allocationSize = req.size,
        .memoryTypeIndex = actual_type,
    };
    var mem: vk.VkDeviceMemory = null;
    if (vk.vkAllocateMemory(device, &alloc_info, null, &mem) != vk.VK_SUCCESS) {
        return error.BufferMemoryAllocFailed;
    }
    errdefer vk.vkFreeMemory(device, mem, null);

    if (vk.vkBindBufferMemory(device, buf_handle, mem, 0) != vk.VK_SUCCESS) {
        return error.BufferBindFailed;
    }

    return Buffer{
        .handle = buf_handle,
        .memory = mem,
        .size = safe_size,
        .device = device,
        .mapped_ptr = null,
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuAttention — public module
// ─────────────────────────────────────────────────────────────────────────────

/// Standalone GPU flash attention module.
///
/// Manages a complete Vulkan compute runtime with two attention pipelines.
/// Use `init()` to create, call `flashAttention()` / `flashAttentionBatched()`
/// for inference, and `deinit()` to release all GPU resources.
pub const GpuAttention = struct {
    // ── Vulkan handles ──
    instance: vk.VkInstance,
    physical_device: vk.VkPhysicalDevice,
    device: vk.VkDevice,
    compute_queue: vk.VkQueue,
    compute_queue_family: u32,
    mem_props: vk.VkPhysicalDeviceMemoryProperties,

    // ── Pipelines ──
    pipeline: Pipeline,
    pipeline_batched: Pipeline,

    // ── Descriptor pool + a reusable descriptor set ──
    descriptor_pool: vk.VkDescriptorPool,
    descriptor_set: vk.VkDescriptorSet,

    // ── Command resources ──
    command_pool: vk.VkCommandPool,
    cmd_buffer: vk.VkCommandBuffer,
    fence: vk.VkFence,

    // ── Allocator ──
    allocator: std.mem.Allocator,

    // ── Cached memory-type indices ──
    /// Index of a memory type that is HOST_VISIBLE | HOST_COHERENT.
    host_coherent_type: u32,
    /// Index of a memory type that is DEVICE_LOCAL.
    device_local_type: u32,

    // ── Persistent staging pool ──
    staging_pool: StagingPool = .{},

    // ── Device-local KV cache buffers (persistent, avoid 938MB/step uploads) ──
    kv_k_buffer: Buffer = undefined,
    kv_v_buffer: Buffer = undefined,
    kv_k_capacity: vk.VkDeviceSize = 0,
    kv_v_capacity: vk.VkDeviceSize = 0,

    // ── Initialisation ──

    /// Create a Vulkan instance, select a compute-capable GPU, load both
    /// flash-attention shaders, and prepare command resources.
    ///
    /// Returns `error.VulkanUnavailable` when no Vulkan runtime or compute
    /// device is found. This is the graceful-degradation path — callers
    /// can fall back to a CPU attention implementation.
    pub fn init(
        allocator: std.mem.Allocator,
        shader_dir: []const u8,
    ) !GpuAttention {
        // ── 1. Create Vulkan instance ──
        const app_info = vk.VkApplicationInfo{
            .sType = vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = null,
            .pApplicationName = "fused-engine",
            .applicationVersion = vk.VK_MAKE_VERSION(0, 1, 0),
            .pEngineName = "fused-engine",
            .engineVersion = vk.VK_MAKE_VERSION(0, 1, 0),
            .apiVersion = vk.VK_API_VERSION_1_3,
        };
        const inst_info = vk.VkInstanceCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .pApplicationInfo = &app_info,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = 0,
            .ppEnabledExtensionNames = null,
        };
        var instance: vk.VkInstance = null;
        if (vk.vkCreateInstance(&inst_info, null, &instance) != vk.VK_SUCCESS) {
            log.warn("vkCreateInstance failed — Vulkan unavailable", .{});
            return error.VulkanUnavailable;
        }
        errdefer vk.vkDestroyInstance(instance, null);

        // ── 2. Enumerate physical devices ──
        var dev_count: u32 = 0;
        _ = vk.vkEnumeratePhysicalDevices(instance, &dev_count, null);
        if (dev_count == 0) {
            log.warn("No Vulkan physical devices found", .{});
            return error.VulkanUnavailable;
        }
        const phys_devices = try allocator.alloc(vk.VkPhysicalDevice, dev_count);
        defer allocator.free(phys_devices);
        _ = vk.vkEnumeratePhysicalDevices(instance, &dev_count, phys_devices.ptr);

        // ── 3. Select a compute-capable device (prefer discrete GPU) ──
        const phys_device = blk: {
            var best: vk.VkPhysicalDevice = null;
            var best_score: u32 = 0;
            for (phys_devices[0..dev_count]) |pdev| {
                var qf_count: u32 = 0;
                vk.vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qf_count, null);
                if (qf_count == 0) continue;
                const qf_props = try allocator.alloc(vk.VkQueueFamilyProperties, qf_count);
                defer allocator.free(qf_props);
                vk.vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qf_count, qf_props.ptr);
                var has_compute = false;
                for (qf_props[0..qf_count]) |qf| {
                    if (qf.queueFlags & vk.VK_QUEUE_COMPUTE_BIT != 0) {
                        has_compute = true;
                        break;
                    }
                }
                if (!has_compute) continue;
                var props: vk.VkPhysicalDeviceProperties = undefined;
                vk.vkGetPhysicalDeviceProperties(pdev, &props);
                const score: u32 = switch (props.deviceType) {
                    vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => 5,
                    vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => 4,
                    vk.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU => 3,
                    vk.VK_PHYSICAL_DEVICE_TYPE_CPU => 1,
                    else => 2,
                };
                if (best == null or score > best_score) {
                    best = pdev;
                    best_score = score;
                }
            }
            if (best == null) {
                log.warn("No compute-capable Vulkan device found", .{});
                return error.VulkanUnavailable;
            }
            break :blk best.?;
        };

        var device_props: vk.VkPhysicalDeviceProperties = undefined;
        vk.vkGetPhysicalDeviceProperties(phys_device, &device_props);
        const dev_name = std.mem.sliceTo(&device_props.deviceName, 0);
        log.info("Selected GPU: {s} (type={d})", .{ dev_name, device_props.deviceType });

        var mem_props: vk.VkPhysicalDeviceMemoryProperties = undefined;
        vk.vkGetPhysicalDeviceMemoryProperties(phys_device, &mem_props);

        // ── 4. Find compute queue family ──
        var qf_count: u32 = 0;
        vk.vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &qf_count, null);
        const qf_props = try allocator.alloc(vk.VkQueueFamilyProperties, qf_count);
        defer allocator.free(qf_props);
        vk.vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &qf_count, qf_props.ptr);

        const compute_family = blk2: {
            for (qf_props[0..qf_count], 0..) |qf, i| {
                if (qf.queueFlags & vk.VK_QUEUE_COMPUTE_BIT != 0 and
                    qf.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT == 0)
                {
                    break :blk2 @as(u32, @intCast(i));
                }
            }
            for (qf_props[0..qf_count], 0..) |qf, i| {
                if (qf.queueFlags & vk.VK_QUEUE_COMPUTE_BIT != 0) {
                    break :blk2 @as(u32, @intCast(i));
                }
            }
            return error.VulkanUnavailable;
        };

        // ── 5. Create logical device ──
        const queue_priority: f32 = 1.0;
        const queue_ci = vk.VkDeviceQueueCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueFamilyIndex = compute_family,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority,
        };
        const device_ci = vk.VkDeviceCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_ci,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = 0,
            .ppEnabledExtensionNames = null,
            .pEnabledFeatures = null,
        };
        var device: vk.VkDevice = null;
        if (vk.vkCreateDevice(phys_device, &device_ci, null, &device) != vk.VK_SUCCESS) {
            log.warn("vkCreateDevice failed", .{});
            return error.VulkanUnavailable;
        }
        errdefer vk.vkDestroyDevice(device, null);

        var compute_queue: vk.VkQueue = null;
        vk.vkGetDeviceQueue(device, compute_family, 0, &compute_queue);

        // ── 6. Cache memory type indices ──
        const host_coherent_type = findMemoryType(&mem_props,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) orelse {
            return error.NoSuitableMemoryType;
        };
        const device_local_type = findMemoryType(&mem_props,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) orelse host_coherent_type;

        // ── 6b. Initialise persistent staging pool (lazy allocation) ──
        var staging_pool = StagingPool{};
        staging_pool.init(device, host_coherent_type, &mem_props);

        // ── 6c. Create initial device-local KV cache buffers (minimum size) ──
        // Grow on first updateKVCache() call. One-time pre-alloc to avoid null handles.
        const kv_usage = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        const kv_init_size: vk.VkDeviceSize = 4;
        var kv_k = try createCustomBuffer(device, kv_init_size, kv_usage, host_coherent_type, &mem_props);
        errdefer kv_k.deinit();
        var kv_v = try createCustomBuffer(device, kv_init_size, kv_usage, host_coherent_type, &mem_props);
        errdefer kv_v.deinit();

        // ── 7. Load shaders and create pipelines ──
        var path_buf: [512]u8 = undefined;

        const attn_path = std.fmt.bufPrint(&path_buf, "{s}/flash_attn.spv", .{shader_dir}) catch unreachable;
        var pipeline = try createPipeline(device, attn_path, 6, @sizeOf(FlashAttnPush));
        errdefer pipeline.deinit();

        const attn_batched_path = std.fmt.bufPrint(&path_buf, "{s}/flash_attn_batched.spv", .{shader_dir}) catch unreachable;
        var pipeline_batched = try createPipeline(device, attn_batched_path, 6, @sizeOf(FlashAttnBatchedPush));
        errdefer pipeline_batched.deinit();

        // ── 8. Create descriptor pool ──
        const pool_size = vk.VkDescriptorPoolSize{
            .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 6,
        };
        const pool_ci = vk.VkDescriptorPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        var descriptor_pool: vk.VkDescriptorPool = null;
        if (vk.vkCreateDescriptorPool(device, &pool_ci, null, &descriptor_pool) != vk.VK_SUCCESS) {
            return error.DescriptorPoolCreateFailed;
        }
        errdefer vk.vkDestroyDescriptorPool(device, descriptor_pool, null);

        // ── 9. Pre-allocate one descriptor set ──
        const ds_alloc_info = vk.VkDescriptorSetAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null,
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &pipeline.descriptor_set_layout,
        };
        var descriptor_set: vk.VkDescriptorSet = null;
        if (vk.vkAllocateDescriptorSets(device, &ds_alloc_info, &descriptor_set) != vk.VK_SUCCESS) {
            return error.DescriptorSetAllocFailed;
        }

        // ── 10. Create command pool + command buffer + fence ──
        const cmd_pool_ci = vk.VkCommandPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = compute_family,
        };
        var command_pool: vk.VkCommandPool = null;
        if (vk.vkCreateCommandPool(device, &cmd_pool_ci, null, &command_pool) != vk.VK_SUCCESS) {
            return error.CommandPoolCreateFailed;
        }
        errdefer vk.vkDestroyCommandPool(device, command_pool, null);

        const cmd_alloc_info = vk.VkCommandBufferAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = null,
            .commandPool = command_pool,
            .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        var cmd_buffer: vk.VkCommandBuffer = null;
        if (vk.vkAllocateCommandBuffers(device, &cmd_alloc_info, &cmd_buffer) != vk.VK_SUCCESS) {
            return error.CommandBufferAllocFailed;
        }

        const fence_ci = vk.VkFenceCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
        };
        var fence: vk.VkFence = null;
        if (vk.vkCreateFence(device, &fence_ci, null, &fence) != vk.VK_SUCCESS) {
            return error.FenceCreateFailed;
        }
        errdefer vk.vkDestroyFence(device, fence, null);

        log.info("GpuAttention ready — loaded {s} and {s}",
            .{ attn_path, attn_batched_path });

        return GpuAttention{
            .instance = instance,
            .physical_device = phys_device,
            .device = device,
            .compute_queue = compute_queue,
            .compute_queue_family = compute_family,
            .mem_props = mem_props,
            .pipeline = pipeline,
            .pipeline_batched = pipeline_batched,
            .descriptor_pool = descriptor_pool,
            .descriptor_set = descriptor_set,
            .command_pool = command_pool,
            .cmd_buffer = cmd_buffer,
            .fence = fence,
            .allocator = allocator,
            .host_coherent_type = host_coherent_type,
            .device_local_type = device_local_type,
            .staging_pool = staging_pool,
            .kv_k_buffer = kv_k,
            .kv_v_buffer = kv_v,
            .kv_k_capacity = kv_init_size,
            .kv_v_capacity = kv_init_size,
        };
    }

    // ── Public API: single-query flash attention ──

    /// Run flash attention for a single decode step.
    ///
    /// Reads Q, K, V, and per-head sinks from the provided CPU slices,
    /// uploads them to the GPU, dispatches the compute shader, and copies
    /// the attention output back into `output`.
    ///
    /// Workgroup grid: `(n_heads, 1, 1)` — each workgroup processes one head.
    ///
    /// **Buffer layout (6 storage-buffer bindings):**
    ///   - Binding 0: Q       — shape `[n_heads * head_dim]` f32
    ///   - Binding 1: K cache — shape `[n_kv_heads * seq_len * head_dim]` f32 (paged)
    ///   - Binding 2: V cache — shape `[n_kv_heads * seq_len * head_dim]` f32 (paged)
    ///   - Binding 3: page_table — shape `[page_count]` u32 page indices
    ///   - Binding 4: output  — shape `[n_heads * head_dim]` f32
    ///   - Binding 5: sinks   — shape `[n_heads * ...]` f32 per-head sink terms
    ///
    /// `page_table` maps logical page index → physical page slot in the K/V
    /// cache buffers. The number of entries must be at least
    /// `(seq_len + page_size - 1) / page_size`.
    pub fn flashAttention(
        self: *GpuAttention,
        q: []const f32,
        k_cache: []const f32,
        v_cache: []const f32,
        page_table: []const u32,
        output: []f32,
        sinks: []const f32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
        page_size: u32,
        attn_scale: f32,
        sink_offset: u32,
    ) !void {
        const output_bytes = output.len * @sizeOf(f32);

        // ── 1. Upload K/V to device-local persistent cache ──
        try self.updateKVCache(k_cache, v_cache);

        // ── 2. Acquire staging buffers for the small per-step data ──
        const q_buf = try self.staging_pool.acquire(0, @intCast(q.len * @sizeOf(f32)));
        const pt_buf = if (page_table.len > 0) try self.staging_pool.acquire(3, @intCast(page_table.len * @sizeOf(u32))) else null;
        const out_buf = try self.staging_pool.acquire(4, output_bytes);
        const sinks_buf = if (sinks.len > 0) try self.staging_pool.acquire(5, @intCast(sinks.len * @sizeOf(f32))) else null;

        // ── 3. Upload small data ──
        uploadToBuffer(q_buf, std.mem.sliceAsBytes(q));
        if (pt_buf) |pt| uploadToBuffer(pt, std.mem.sliceAsBytes(page_table));
        if (sinks_buf) |sb| uploadToBuffer(sb, std.mem.sliceAsBytes(sinks));
        uploadToBuffer(out_buf, std.mem.sliceAsBytes(output));

        // ── 4. Build descriptor set writes (K/V from device-local buffers) ──
        const buffer_infos = [_]vk.VkDescriptorBufferInfo{
            makeBufferInfo(q_buf.handle, q_buf.size),
            makeBufferInfo(self.kv_k_buffer.handle, self.kv_k_capacity),
            makeBufferInfo(self.kv_v_buffer.handle, self.kv_v_capacity),
            makeBufferInfo(
                if (pt_buf) |pt| pt.handle else self.kv_k_buffer.handle,
                if (pt_buf) |pt| pt.size else 4,
            ),
            makeBufferInfo(out_buf.handle, out_buf.size),
            makeBufferInfo(
                if (sinks_buf) |sb| sb.handle else self.kv_k_buffer.handle,
                if (sinks_buf) |sb| sb.size else 4,
            ),
        };

        var writes: [6]vk.VkWriteDescriptorSet = undefined;
        for (&writes, 0..) |*w, i| {
            w.* = .{
                .sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = null,
                .dstSet = self.descriptor_set,
                .dstBinding = @intCast(i),
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null,
                .pBufferInfo = &buffer_infos[i],
                .pTexelBufferView = null,
            };
        }
        vk.vkUpdateDescriptorSets(self.device, 6, &writes, 0, null);

        // ── 4. Build push constants ──
        const push = FlashAttnPush{
            .head_dim = head_dim,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .seq_len = seq_len,
            .page_size = page_size,
            .attn_scale_bits = if (attn_scale != 0) @as(u32, @bitCast(attn_scale)) else 0,
            .sink_offset = sink_offset,
        };
        const push_bytes = std.mem.asBytes(&push);

        // ── 5. Record and submit ──
        try self.recordAndSubmit(push_bytes, n_heads, 1, 1);

        // ── 6. Read back output ──
        readbackFromBuffer(out_buf, std.mem.sliceAsBytes(output));
    }

    // ── Public API: batched flash attention ──

    /// Run batched flash attention for N query tokens (prefill or decode).
    ///
    /// Grid: `(n_heads, n_queries, 1)` — each (head, query) workgroup applies
    /// causal masking up to `seq_start + query + 1`.
    ///
    /// Parameters mirror ZINC's `FlashAttnBatchedPush`.  `seq_start` is the
    /// token position of the first query (0 for a fresh prefill).
    pub fn flashAttentionBatched(
        self: *GpuAttention,
        q: []const f32,
        k_cache: []const f32,
        v_cache: []const f32,
        page_table: []const u32,
        output: []f32,
        sinks: []const f32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        seq_start: u32,
        n_queries: u32,
        page_size: u32,
        attn_scale: f32,
        sink_offset: u32,
    ) !void {
        const output_bytes = output.len * @sizeOf(f32);

        // ── 1. Acquire persistent staging buffers from pool ──
        const q_buf = try self.staging_pool.acquire(0, @intCast(q.len * @sizeOf(f32)));
        const k_buf = try self.staging_pool.acquire(1, @intCast(k_cache.len * @sizeOf(f32)));
        const v_buf = try self.staging_pool.acquire(2, @intCast(v_cache.len * @sizeOf(f32)));
        const pt_buf = try self.staging_pool.acquire(3, @intCast(page_table.len * @sizeOf(u32)));
        const out_buf = try self.staging_pool.acquire(4, output_bytes);
        const sinks_buf = try self.staging_pool.acquire(5, @intCast(sinks.len * @sizeOf(f32)));

        // ── 2. Upload ──
        uploadToBuffer(q_buf, std.mem.sliceAsBytes(q));
        uploadToBuffer(k_buf, std.mem.sliceAsBytes(k_cache));
        uploadToBuffer(v_buf, std.mem.sliceAsBytes(v_cache));
        uploadToBuffer(pt_buf, std.mem.sliceAsBytes(page_table));
        uploadToBuffer(sinks_buf, std.mem.sliceAsBytes(sinks));
        uploadToBuffer(out_buf, std.mem.sliceAsBytes(output));

        // ── 3. Update descriptor set (re-use same slot layout) ──
        const buffer_infos = [_]vk.VkDescriptorBufferInfo{
            makeBufferInfo(q_buf.handle, q_buf.size),
            makeBufferInfo(k_buf.handle, k_buf.size),
            makeBufferInfo(v_buf.handle, v_buf.size),
            makeBufferInfo(pt_buf.handle, pt_buf.size),
            makeBufferInfo(out_buf.handle, out_buf.size),
            makeBufferInfo(sinks_buf.handle, sinks_buf.size),
        };
        var writes: [6]vk.VkWriteDescriptorSet = undefined;
        for (&writes, 0..) |*w, i| {
            w.* = .{
                .sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = null,
                .dstSet = self.descriptor_set,
                .dstBinding = @intCast(i),
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null,
                .pBufferInfo = &buffer_infos[i],
                .pTexelBufferView = null,
            };
        }
        vk.vkUpdateDescriptorSets(self.device, 6, &writes, 0, null);

        // ── 4. Push constants ──
        const push = FlashAttnBatchedPush{
            .head_dim = head_dim,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .seq_start = seq_start,
            .n_queries = n_queries,
            .page_size = page_size,
            .attn_scale_bits = if (attn_scale != 0) @as(u32, @bitCast(attn_scale)) else 0,
            .sink_offset = sink_offset,
        };
        const push_bytes = std.mem.asBytes(&push);

        // ── 5. Dispatch with batched pipeline ──
        try self.recordAndSubmitBatched(push_bytes, n_heads, n_queries, 1);

        // ── 6. Readback ──
        readbackFromBuffer(out_buf, std.mem.sliceAsBytes(output));
    }

    // ── KV cache management ──

    /// Ensure device-local K/V cache buffers can hold at least `k_bytes` and
    /// `v_bytes`. Grows by reallocating if needed. On integrated GPUs
    /// (Radeon 8060S unified memory) this is HOST_VISIBLE so memcpy is direct.
    fn ensureKVCacheSize(self: *GpuAttention, k_bytes: vk.VkDeviceSize, v_bytes: vk.VkDeviceSize) !void {
        const buf_usage = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        
        // Find a memory type that is DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT
        // (works on integrated GPUs with unified memory like Radeon 8060S)
        var dl_host_type = self.host_coherent_type;
        {
            for (0..self.mem_props.memoryTypeCount) |i| {
                const flags = self.mem_props.memoryTypes[i].propertyFlags;
                if (flags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0 and
                    flags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0 and
                    flags & vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT != 0)
                {
                    dl_host_type = @intCast(i);
                    break;
                }
            }
        }

        if (k_bytes > self.kv_k_capacity) {
            if (self.kv_k_capacity > 0) self.kv_k_buffer.deinit();
            const safe = @max(k_bytes, @as(vk.VkDeviceSize, 4));
            self.kv_k_buffer = try createCustomBuffer(self.device, safe, buf_usage, dl_host_type, &self.mem_props);
            self.kv_k_capacity = safe;
        }
        if (v_bytes > self.kv_v_capacity) {
            if (self.kv_v_capacity > 0) self.kv_v_buffer.deinit();
            const safe = @max(v_bytes, @as(vk.VkDeviceSize, 4));
            self.kv_v_buffer = try createCustomBuffer(self.device, safe, buf_usage, dl_host_type, &self.mem_props);
            self.kv_v_capacity = safe;
        }
    }

    /// Upload K and V cache data to device-local memory.
    /// On integrated GPUs this is a direct memcpy into mapped memory.
    pub fn updateKVCache(self: *GpuAttention, k_data: []const f32, v_data: []const f32) !void {
        const k_bytes = @as(vk.VkDeviceSize, k_data.len * @sizeOf(f32));
        const v_bytes = @as(vk.VkDeviceSize, v_data.len * @sizeOf(f32));
        try self.ensureKVCacheSize(k_bytes, v_bytes);

        // Map + memcpy for K
        {
            const ptr = try self.kv_k_buffer.map(0, k_bytes);
            defer self.kv_k_buffer.unmap();
            @memcpy(ptr[0..k_bytes], std.mem.sliceAsBytes(k_data));
        }
        // Map + memcpy for V
        {
            const ptr = try self.kv_v_buffer.map(0, v_bytes);
            defer self.kv_v_buffer.unmap();
            @memcpy(ptr[0..v_bytes], std.mem.sliceAsBytes(v_data));
        }
    }

    // ── Teardown ──

    /// Release all Vulkan resources.
    /// Safe to call even after a partial initialisation failure in `init()`
    /// (the struct is zero-initialised by the caller on error).
    pub fn deinit(self: *GpuAttention) void {
        _ = vk.vkDeviceWaitIdle(self.device);
        vk.vkDestroyFence(self.device, self.fence, null);
        vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, &self.cmd_buffer);
        vk.vkDestroyCommandPool(self.device, self.command_pool, null);
        _ = vk.vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, &self.descriptor_set);
        vk.vkDestroyDescriptorPool(self.device, self.descriptor_pool, null);
        self.staging_pool.deinit();
        if (self.kv_k_capacity > 0) self.kv_k_buffer.deinit();
        if (self.kv_v_capacity > 0) self.kv_v_buffer.deinit();
        self.pipeline.deinit();
        self.pipeline_batched.deinit();
        vk.vkDestroyDevice(self.device, null);
        vk.vkDestroyInstance(self.instance, null);
        self.* = undefined;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ─────────────────────────────────────────────────────────────────────────

    fn recordAndSubmit(
        self: *GpuAttention,
        push_bytes: []const u8,
        group_x: u32,
        group_y: u32,
        group_z: u32,
    ) !void {
        // Reset command buffer
        _ = vk.vkResetCommandBuffer(self.cmd_buffer, 0);

        const begin_info = vk.VkCommandBufferBeginInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = null,
        };
        if (vk.vkBeginCommandBuffer(self.cmd_buffer, &begin_info) != vk.VK_SUCCESS) {
            return error.BeginCommandBufferFailed;
        }

        // Bind pipeline
        vk.vkCmdBindPipeline(
            self.cmd_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline.handle,
        );

        // Push constants
        vk.vkCmdPushConstants(
            self.cmd_buffer,
            self.pipeline.pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            @intCast(push_bytes.len),
            push_bytes.ptr,
        );

        // Bind descriptor set
        vk.vkCmdBindDescriptorSets(
            self.cmd_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline.pipeline_layout,
            0,
            1,
            &self.descriptor_set,
            0,
            null,
        );

        // Dispatch
        vk.vkCmdDispatch(self.cmd_buffer, group_x, group_y, group_z);

        if (vk.vkEndCommandBuffer(self.cmd_buffer) != vk.VK_SUCCESS) {
            return error.EndCommandBufferFailed;
        }

        // Submit and wait
        const submit_info = vk.VkSubmitInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = null,
            .pWaitDstStageMask = null,
            .commandBufferCount = 1,
            .pCommandBuffers = &self.cmd_buffer,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = null,
        };
        if (vk.vkQueueSubmit(self.compute_queue, 1, &submit_info, self.fence) != vk.VK_SUCCESS) {
            return error.QueueSubmitFailed;
        }
        if (vk.vkWaitForFences(self.device, 1, &self.fence, vk.VK_TRUE, std.math.maxInt(u64)) != vk.VK_SUCCESS) {
            return error.FenceWaitFailed;
        }
        _ = vk.vkResetFences(self.device, 1, &self.fence);
    }

    fn recordAndSubmitBatched(
        self: *GpuAttention,
        push_bytes: []const u8,
        group_x: u32,
        group_y: u32,
        group_z: u32,
    ) !void {
        _ = vk.vkResetCommandBuffer(self.cmd_buffer, 0);

        const begin_info = vk.VkCommandBufferBeginInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = null,
        };
        if (vk.vkBeginCommandBuffer(self.cmd_buffer, &begin_info) != vk.VK_SUCCESS) {
            return error.BeginCommandBufferFailed;
        }

        vk.vkCmdBindPipeline(
            self.cmd_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline_batched.handle,
        );

        vk.vkCmdPushConstants(
            self.cmd_buffer,
            self.pipeline_batched.pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            @intCast(push_bytes.len),
            push_bytes.ptr,
        );

        vk.vkCmdBindDescriptorSets(
            self.cmd_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline_batched.pipeline_layout,
            0,
            1,
            &self.descriptor_set,
            0,
            null,
        );

        vk.vkCmdDispatch(self.cmd_buffer, group_x, group_y, group_z);

        if (vk.vkEndCommandBuffer(self.cmd_buffer) != vk.VK_SUCCESS) {
            return error.EndCommandBufferFailed;
        }

        const submit_info = vk.VkSubmitInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = null,
            .pWaitDstStageMask = null,
            .commandBufferCount = 1,
            .pCommandBuffers = &self.cmd_buffer,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = null,
        };
        if (vk.vkQueueSubmit(self.compute_queue, 1, &submit_info, self.fence) != vk.VK_SUCCESS) {
            return error.QueueSubmitFailed;
        }
        if (vk.vkWaitForFences(self.device, 1, &self.fence, vk.VK_TRUE, std.math.maxInt(u64)) != vk.VK_SUCCESS) {
            return error.FenceWaitFailed;
        }
        _ = vk.vkResetFences(self.device, 1, &self.fence);
    }

};

// ─────────────────────────────────────────────────────────────────────────────
// Standalone helpers (no self capture needed)
// ─────────────────────────────────────────────────────────────────────────────

/// Find the first memory type satisfying `required_flags`.
fn findMemoryType(
    mem_props: *const vk.VkPhysicalDeviceMemoryProperties,
    required_flags: vk.VkMemoryPropertyFlags,
) ?u32 {
    for (0..mem_props.memoryTypeCount) |i| {
        if (mem_props.memoryTypes[i].propertyFlags & required_flags == required_flags) {
            return @intCast(i);
        }
    }
    return null;
}

/// Create a compute pipeline from a SPIR-V file.  Builds a single descriptor
/// set layout with `binding_count` storage-buffer bindings followed by a
/// pipeline layout with push-constant range and the compute pipeline itself.
fn createPipeline(
    device: vk.VkDevice,
    spirv_path: []const u8,
    binding_count: u32,
    push_constant_size: u32,
) !Pipeline {
    // ── Read SPIR-V binary via posix ──
    var path_buf: [4096]u8 = undefined;
    @memcpy(path_buf[0..spirv_path.len], spirv_path);
    path_buf[spirv_path.len] = 0;
    const path_ptr: [*:0]u8 = @ptrCast(&path_buf);
    const fd_raw = std.os.linux.open(path_ptr, .{ .ACCMODE = .RDONLY }, 0);
    if (std.os.linux.errno(fd_raw) != .SUCCESS) {
        log.err("Cannot open SPIR-V '{s}'", .{spirv_path});
        return error.ShaderFileNotFound;
    }
    const fd: i32 = @intCast(fd_raw);
    defer _ = std.os.linux.close(fd);

    const file_end = std.os.linux.lseek(fd, 0, std.os.linux.SEEK.END);
    if (std.os.linux.errno(file_end) != .SUCCESS) return error.ShaderReadIncomplete;
    const file_size = @as(usize, @intCast(file_end));
    _ = std.os.linux.lseek(fd, 0, std.os.linux.SEEK.SET);

    // Read SPIR-V binary
    const spv_code = try std.heap.page_allocator.alloc(u8, file_size);
    defer std.heap.page_allocator.free(spv_code);
    const nread = std.os.linux.read(fd, spv_code.ptr, file_size);
    if (std.os.linux.errno(nread) != .SUCCESS or @as(usize, @intCast(nread)) != file_size) {
        return error.ShaderReadIncomplete;
    }

    // ── Create shader module ──
    const module_ci = vk.VkShaderModuleCreateInfo{
        .sType = vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .codeSize = file_size,
        .pCode = @ptrCast(@alignCast(spv_code.ptr)),
    };
    var shader_module: vk.VkShaderModule = null;
    if (vk.vkCreateShaderModule(device, &module_ci, null, &shader_module) != vk.VK_SUCCESS) {
        log.err("vkCreateShaderModule failed for '{s}'", .{spirv_path});
        return error.ShaderModuleCreateFailed;
    }
    errdefer vk.vkDestroyShaderModule(device, shader_module, null);

    // ── Descriptor set layout: `binding_count` storage buffers ──
    // Stack-allocate up to 16 bindings.
    const MaxBindings = 16;
    var binding_storage: [MaxBindings]vk.VkDescriptorSetLayoutBinding = undefined;
    const bindings = binding_storage[0..binding_count];
    for (bindings, 0..) |*b, i| {
        b.* = .{
            .binding = @intCast(i),
            .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = null,
        };
    }
    const ds_layout_ci = vk.VkDescriptorSetLayoutCreateInfo{
        .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .bindingCount = binding_count,
        .pBindings = bindings.ptr,
    };
    var ds_layout: vk.VkDescriptorSetLayout = null;
    if (vk.vkCreateDescriptorSetLayout(device, &ds_layout_ci, null, &ds_layout) != vk.VK_SUCCESS) {
        return error.DescriptorSetLayoutFailed;
    }
    errdefer vk.vkDestroyDescriptorSetLayout(device, ds_layout, null);

    // ── Pipeline layout with push constants ──
    const push_range = vk.VkPushConstantRange{
        .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = push_constant_size,
    };
    const pl_ci = vk.VkPipelineLayoutCreateInfo{
        .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &ds_layout,
        .pushConstantRangeCount = if (push_constant_size > 0) 1 else 0,
        .pPushConstantRanges = if (push_constant_size > 0) &push_range else null,
    };
    var pl_layout: vk.VkPipelineLayout = null;
    if (vk.vkCreatePipelineLayout(device, &pl_ci, null, &pl_layout) != vk.VK_SUCCESS) {
        return error.PipelineLayoutFailed;
    }
    errdefer vk.vkDestroyPipelineLayout(device, pl_layout, null);

    // ── Compute pipeline ──
    const stage_ci = vk.VkPipelineShaderStageCreateInfo{
        .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .stage = vk.VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main",
        .pSpecializationInfo = null,
    };
    const cp_ci = vk.VkComputePipelineCreateInfo{
        .sType = vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .stage = stage_ci,
        .layout = pl_layout,
        .basePipelineHandle = @as(vk.VkPipeline, null),
        .basePipelineIndex = -1,
    };
    var pipeline_handle: vk.VkPipeline = null;
    if (vk.vkCreateComputePipelines(device, @as(vk.VkPipelineCache, null), 1, &cp_ci, null, &pipeline_handle) != vk.VK_SUCCESS) {
        return error.ComputePipelineCreateFailed;
    }
    errdefer vk.vkDestroyPipeline(device, pipeline_handle, null);

    return Pipeline{
        .shader_module = shader_module,
        .descriptor_set_layout = ds_layout,
        .pipeline_layout = pl_layout,
        .handle = pipeline_handle,
        .device = device,
    };
}

/// Construct a `VkDescriptorBufferInfo` for a buffer of `size` bytes.
fn makeBufferInfo(buffer: vk.VkBuffer, size: vk.VkDeviceSize) vk.VkDescriptorBufferInfo {
    return .{
        .buffer = buffer,
        .offset = 0,
        .range = size,
    };
}

/// Copy host data into a mapped staging buffer.
fn uploadToBuffer(buf: *Buffer, data: []const u8) void {
    const ptr = buf.map(0, data.len) catch unreachable;
    defer buf.unmap();
    @memcpy(ptr[0..data.len], data);
}

/// Read back device buffer contents into a host slice.
fn readbackFromBuffer(buf: *Buffer, dst: []u8) void {
    const ptr = buf.map(0, dst.len) catch unreachable;
    defer buf.unmap();
    @memcpy(dst, ptr[0..dst.len]);
}
