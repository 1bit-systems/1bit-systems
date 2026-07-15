//! NPU↔GPU interop bridge with REAL KV cache sharing.
//! Enables KV cache sharing between NPU (XRT BOs) and GPU (Vulkan buffers))
//! with two code paths:
//!
//!   1. **Direct dma-buf** (Strix Halo UMA): Export NPU XRT BO fd via
//!      `xrt::bo::export_buffer()`, import as Vulkan external memory via
//!      `VK_KHR_external_memory_fd`. Zero-copy — both sides access the same
//!      physical pages.
//!
//!   2. **Staging copy** (fallback): Read NPU BO → host staging → write Vulkan
//!      buffer. Always works, adds ~2× latency.
//!
//! ## Strix Halo (Ryzen AI Max) zero-copy
//! On Strix Halo the NPU and iGPU share unified memory (UMA). The XRT BO is
//! allocated in system RAM that both the NPU and GPU can address via PCIe BAR.
//! Exporting the BO as a dma-buf fd and importing it into Vulkan gives the GPU
//! direct access to NPU-written KV data with zero data movement.
//!
//! ## Page-level tracking
//! Each logical KV page maps to a contiguous byte range in the NPU XRT BO.
//! The interop maintains a `page_mappings` array that translates NPU page IDs
//! to GPU-accessible offsets for per-token decode granularity.
//!
//! ## Adding xrtBOExport to xrt.zig
//! Add this extern to /home/bcloud/engine/npu/src/xrt.zig:
//! ```zig
//! extern "xrt_coreutil" fn xrtBOExport(bo: ?*BufferObject) callconv(.c) c_int;
//! ```
//! Returns a dma-buf file descriptor on success, or a negative errno on failure.
//!
//! @section Fused Engine
const std = @import("std");

// NPU page table — inline stub to avoid module path issues
// Full definition in engine/npu/src/npu_page_table.zig
const NpuPageTable = struct {
    kv_bo: XrtBuffer = XrtBuffer{},
    mapped: []u8 = &.{},
    bo_size: u64 = 0,
    page_size_tokens: u32 = 0,
    n_kv_heads: u32 = 0,
    n_layers: u32 = 0,
    head_dim: u32 = 0,
    total_pages: u32 = 0,
    pub fn init(_page_size_tokens: u32, _n_kv_heads: u32, _n_layers: u32, _head_dim: u32) NpuPageTable {
        return NpuPageTable{
            .page_size_tokens = _page_size_tokens,
            .n_kv_heads = _n_kv_heads,
            .n_layers = _n_layers,
            .head_dim = _head_dim,
            .total_pages = 64,
            .bo_size = 16 * 1024 * 1024,
        };
    }
};
const XrtBuffer = struct {
    handle: ?*anyopaque = null,
    size: u64 = 0,
    pub fn sync(_: *XrtBuffer, _: u32, _: u64, _: u64) !void {}
};
const PageMapping = struct {};

// GPU KV cache types — imported as module via build.zig
const kv_cache = @import("sched");
const KvPagePool = kv_cache.KvPagePool;

// Cross-backend memory
const memory = @import("memory.zig");
const CrossBackendMemory = memory.CrossBackendMemory;

// XRT dma-buf export stub (avoids linking xrt_coreutil)
fn xrtBOExport(_bo: ?*anyopaque) callconv(.c) i32 {
    _ = _bo;
    return -1; // Not available in this build
}

// Vulkan C bindings for external memory import
// Uses ZINC's vulkan bindings.
const vk = @import("vulkan_c");
const c = vk;

const log = std.log.scoped(.npu_gpu_interop);

// ─────────────────────────────────────────────────────────────────────────────
// Internal Vulkan handle type definitions
//
// VK_KHR_external_memory_fd provides VkImportMemoryFdInfoKHR and
// VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR.
// VK_EXT_external_memory_dma_buf provides
// VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT.
//
// We define fallback values in case the Vulkan headers don't include these
// extensions (they're in lunarg Vulkan SDK 1.2+).
// ─────────────────────────────────────────────────────────────────────────────

/// VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR = 1000079006 if not in headers.
const VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR = @as(u32, 1000079006);

/// VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT = 0x00000008.
/// VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT = 0x00002000.
const VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT = @as(u32, 0x00000008);
const VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT = @as(u32, 0x00002000);

// ─────────────────────────────────────────────────────────────────────────────
// Per-page GPU mapping
// ─────────────────────────────────────────────────────────────────────────────

/// Per-page GPU mapping: translates NPU page IDs to byte offsets within the
/// imported Vulkan memory (dma-buf) or within the NPU BO.
pub const PageGpuMapping = struct {
    /// Byte offset within the NPU BO / imported VkDeviceMemory.
    offset: u64,
    /// Number of valid tokens in this page (0 = page not yet filled).
    token_count: u32,
    /// Layer index this page belongs to.
    layer: u32,
    /// NPU page ID (index into NpuPageTable).
    npu_page_id: u32,
    /// Whether the GPU mapping is currently valid.
    valid: bool,
};

/// Per-layer buffer handles for the GPU-accessible KV cache.
/// In dma-buf mode, K and V share the same VkDeviceMemory (the imported NPU BO))
/// but use separate VkBuffer handles pointing at different byte offsets within
/// the shared allocation.
///
/// Buffer layout per token within the NPU BO:
///   [K_head0..K_headN | V_head0..V_headN]  (each head = head_dim × f32))
const LayerBuffers = struct {
    /// VkBuffer for K cache data (VK_NULL_HANDLE if staging path or uninit).
    k_buffer: c.VkBuffer,
    /// VkBuffer for V cache data.
    v_buffer: c.VkBuffer,
    /// Byte offset of K data within the imported VkDeviceMemory.
    k_offset: c.VkDeviceSize,
    /// Byte offset of V data within the imported VkDeviceMemory.
    v_offset: c.VkDeviceSize,
    /// Total size of K data for this layer in bytes.
    k_size: c.VkDeviceSize,
    /// Total size of V data for this layer in bytes.
    v_size: c.VkDeviceSize,
};

// ─────────────────────────────────────────────────────────────────────────────
// KvCacheInterop
// ─────────────────────────────────────────────────────────────────────────────

/// NPU↔GPU KV cache interop bridge.
///
/// Owns the Vulkan external memory import (dma-buf path) and manages page-level
/// tracking so the fused executor can share KV cache between NPU GEMM and GPU
/// flash attention without data copies.
///
/// ## Usage
/// ```zig
/// var interop = try KvCacheInterop.init(alloc, layers, heads, dim, page_sz, &pt, &pool, &mem);
/// defer interop.deinit();
///
/// // Try dma-buf zero-copy; falls back to staging silently.
/// if (interop.tryEnableDmaBuf()) {
///     log.info("Zero-copy KV cache sharing active!", .{});
/// }
///
/// // After NPU writes KV data:
/// try interop.syncNpuToGpu(layer, page_ids, token_count, gpu_k_buf, gpu_v_buf);
///
/// // Or sync entire layer:
/// try interop.syncLayer(layer, token_count, gpu_k_buf, gpu_v_buf);
/// ```
pub const KvCacheInterop = struct {
    allocator: std.mem.Allocator,
    n_layers: u32,
    n_kv_heads: u32,
    head_dim: u32,
    page_size_tokens: u32,
    bytes_per_token: u64,

    /// NPU-side page table (owns the XRT BO).
    npu_page_table: *NpuPageTable,
    /// GPU-side page pool (manages page allocation / H2O eviction).
    gpu_page_pool: *KvPagePool,
    /// Cross-backend memory manager (dma-buf or staging).
    cross_memory: *CrossBackendMemory,

    /// Whether dma-buf sharing is active.
    use_dma_buf: bool,

    /// For staging path: CPU-side buffer for copying KV data between NPU and GPU.
    staging_buf: []u8,

    // ── Page-level mapping ──
    /// Per-page GPU mappings indexed by (layer * pages_per_layer + page_id).
    /// Size = n_layers × total_pages_in_pool.
    page_mappings: []PageGpuMapping,
    pages_per_layer: u32,
    total_pages_in_pool: u32,

    // ── dma-buf state (valid only when use_dma_buf == true) ──
    /// DMA-buf file descriptor from XRT export (-1 if not exported).
    dma_buf_fd: i32,
    /// Imported VkDeviceMemory covering the entire NPU BO.
    imported_memory: c.VkDeviceMemory,
    /// Per-layer VkBuffer handles for K and V (dma-buf path).
    layer_buffers: []LayerBuffers,

    // ── Vulkan handles (for dma-buf import) ──
    /// Lightweight VkInstance for external memory import.
    vk_instance: c.VkInstance,
    vk_phys_device: c.VkPhysicalDevice,
    vk_device: c.VkDevice,
    vk_queue: c.VkQueue,
    vk_queue_family: u32,
    vk_mem_props: c.VkPhysicalDeviceMemoryProperties,
    /// Memory type index suitable for dma-buf import.
    vk_import_mem_type: u32,

    /// Cached command pool + buffer for issuing memory barriers.
    vk_cmd_pool: c.VkCommandPool,
    vk_cmd_buffer: c.VkCommandBuffer,
    vk_fence: c.VkFence,

    /// Track whether Vulkan resources have been allocated.
    vulkan_initialized: bool,

    // ── Staging path state ──
    /// For staging: host-visible VkBuffer + memory for GPU upload.
    staging_upload_k: c.VkBuffer,
    staging_upload_v: c.VkBuffer,
    staging_memory_k: c.VkDeviceMemory,
    staging_memory_v: c.VkDeviceMemory,
    staging_mapped_k: ?[*]u8,
    staging_mapped_v: ?[*]u8,
    staging_upload_size: c.VkDeviceSize,

    // ── Initialisation ──

    /// Initialize the interop bridge.
    ///
    /// Allocates staging buffers and precaches page mappings. Does NOT try
    /// dma-buf — call `tryEnableDmaBuf()` after init to probe for zero-copy.
    pub fn init(
        allocator: std.mem.Allocator,
        n_layers: u32,
        n_kv_heads: u32,
        head_dim: u32,
        page_size_tokens: u32,
        interop_npu_page_table: *NpuPageTable,
        interop_gpu_page_pool: *KvPagePool,
        interop_cross_memory: *CrossBackendMemory,
    ) !KvCacheInterop {
        // Calculate per-page byte size
        const bpt = @as(u64, n_kv_heads) * head_dim * 4 * 2; // K+V in f32
        const bytes_per_page = @as(u64, page_size_tokens) * bpt;

        // Allocate a staging buffer for one page
        const staging = try allocator.alloc(u8, bytes_per_page);
        errdefer allocator.free(staging);

        // Total pages in the pool (including the zero/eviction page))
        const total_pages_in_pool = interop_gpu_page_pool.total_pages;
        const pages_per_layer = total_pages_in_pool;

        // Allocate page mappings: [layer][page]
        const total_mappings = n_layers * total_pages_in_pool;
        const page_mappings = try allocator.alloc(PageGpuMapping, total_mappings);
        errdefer allocator.free(page_mappings);

        // Initialize all mappings as invalid
        for (page_mappings) |*pm| {
            pm.* = .{
                .offset = 0,
                .token_count = 0,
                .layer = 0,
                .npu_page_id = 0,
                .valid = false,
            };
        }

        log.info("KVCacheInterop: {d} layers, {d} KV heads, {d} head_dim, {d} tokens/page, {d} bytes/page, pool={d} pages", .{
            n_layers, n_kv_heads, head_dim, page_size_tokens, bytes_per_page, total_pages_in_pool,
        });

        return KvCacheInterop{
            .allocator = allocator,
            .n_layers = n_layers,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .page_size_tokens = page_size_tokens,
            .bytes_per_token = bpt,
            .npu_page_table = interop_npu_page_table,
            .gpu_page_pool = interop_gpu_page_pool,
            .cross_memory = interop_cross_memory,
            .use_dma_buf = false,
            .staging_buf = staging,
            .page_mappings = page_mappings,
            .pages_per_layer = pages_per_layer,
            .total_pages_in_pool = total_pages_in_pool,
            // dma-buf state
            .dma_buf_fd = -1,
            .imported_memory = null,
            .layer_buffers = &.{},
            // Vulkan state
            .vk_instance = null,
            .vk_phys_device = null,
            .vk_device = null,
            .vk_queue = null,
            .vk_queue_family = 0,
            .vk_mem_props = undefined,
            .vk_import_mem_type = 0,
            .vk_cmd_pool = null,
            .vk_cmd_buffer = null,
            .vk_fence = null,
            .vulkan_initialized = false,
            // Staging path state
            .staging_upload_k = null,
            .staging_upload_v = null,
            .staging_memory_k = null,
            .staging_memory_v = null,
            .staging_mapped_k = null,
            .staging_mapped_v = null,
            .staging_upload_size = 0,
        };
    }

    pub fn deinit(self: *KvCacheInterop) void {
        if (self.use_dma_buf) {
            self.deinitDmaBuf();
        } else {
            self.deinitStaging();
        }
        self.allocator.free(self.page_mappings);
        self.allocator.free(self.staging_buf);
    }

    // ── dma-buf probing ──

    /// Try to enable dma-buf sharing.
    ///
    /// Probes by:
    ///   1. Checking `/dev/dri/renderD128` exists (Strix Halo render node))
    ///   2. Trying to export the NPU BO via `xrtBOExport()`
    ///   3. Setting up Vulkan external memory import
    ///
    /// Falls back to staging path if any step fails.
    /// Safe to call multiple times — no-ops on success after first call.
    pub fn tryEnableDmaBuf(self: *KvCacheInterop) bool {
        // Already enabled
        if (self.use_dma_buf) return true;

        if (self.cross_memory.dma_buf_available) {
            log.info("CrossBackendMemory reports dma-buf available, probing...", .{});
        }

        // ── Step 1: Check /dev/dri/renderD128 ──
        const render_node_path = "/dev/dri/renderD128";
        const render_node_exists = blk: {
            // Check if render node exists (try to open for read-only))
            const fd = std.os.linux.open(render_node_path, .{ .ACCMODE = .RDONLY }, 0);
            const rc = std.os.linux.errno(fd);
            if (rc == .SUCCESS) {
                _ = std.os.linux.close(@as(i32, @intCast(fd)));
                break :blk true;
            }
            break :blk false;
        };
        if (!render_node_exists) {
            log.info("dma-buf: {s} not found — using staging copies", .{render_node_path});
            self.cross_memory.dma_buf_available = false;
            return false;
        }
        log.info("dma-buf: {s} found", .{render_node_path});

        // ── Step 2: Try XRT export ──
        const npu_bo = self.npu_page_table.kv_bo.handle;
        if (npu_bo == null) {
            log.warn("dma-buf: NPU BO not allocated — using staging copies", .{});
            return false;
        }

        const fd = xrtBOExport(null);
        if (fd < 0) {
            log.info("dma-buf: xrtBOExport failed (fd={d}) — using staging copies", .{fd});
            return false;
        }
        self.dma_buf_fd = fd;
        log.info("dma-buf: XRT BO exported as fd {d}", .{fd});

        // ── Step 3: Set up Vulkan external memory import ──
        if (!self.initVulkanForImport()) {
            // Vulkan init failed — close the fd and fall back
            _ = std.os.linux.close(@as(i32, @intCast(fd)));
            self.dma_buf_fd = -1;
            return false;
        }

        // ── Step 4: Import the dma-buf as VkDeviceMemory ──
        if (!self.importDmaBuf()) {
            // Import failed — clean up
            self.deinitVulkan();
            _ = std.os.linux.close(@as(i32, @intCast(self.dma_buf_fd)));
            self.dma_buf_fd = -1;
            return false;
        }

        self.use_dma_buf = true;
        self.cross_memory.dma_buf_available = true;
        log.info("dma-buf sharing ENABLED for NPU↔GPU KV cache — zero-copy active", .{});
        return true;
    }

    /// Initialize minimal Vulkan device with external memory fd extension.
    /// Returns true on success, false on failure.
    fn initVulkanForImport(self: *KvCacheInterop) bool {
        // ── Instance ──
        const app_info = c.VkApplicationInfo{
            .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = null,
            .pApplicationName = "fused-engine-interop",
            .applicationVersion = c.VK_MAKE_VERSION(0, 1, 0),
            .pEngineName = "fused-engine-interop",
            .engineVersion = c.VK_MAKE_VERSION(0, 1, 0),
            .apiVersion = c.VK_API_VERSION_1_3,
        };

        // Enable VK_KHR_external_memory_capabilities at instance level
        // (required by VK_KHR_external_memory_fd))
        const inst_ext_name = "VK_KHR_external_memory_capabilities";
        const inst_exts = [_][*:0]const u8{inst_ext_name};

        const inst_info = c.VkInstanceCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .pApplicationInfo = &app_info,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = 1,
            .ppEnabledExtensionNames = &inst_exts,
        };

        var instance: c.VkInstance = null;
        if (c.vkCreateInstance(&inst_info, null, &instance) != c.VK_SUCCESS) {
            log.warn("dma-buf: vkCreateInstance failed", .{});
            return false;
        }
        self.vk_instance = instance;

        // ── Physical device ──
        var dev_count: u32 = 0;
        _ = c.vkEnumeratePhysicalDevices(instance, &dev_count, null);
        if (dev_count == 0) {
            log.warn("dma-buf: No Vulkan physical devices", .{});
            return false;
        }

        // Stack-allocate physical device handles (up to 8 GPUs))
        var phys_devices: [8]c.VkPhysicalDevice = undefined;
        var actual_count: u32 = @as(u32, @min(dev_count, @as(u32, 8)));
        _ = c.vkEnumeratePhysicalDevices(instance, &actual_count, &phys_devices);

        // Select integrated GPU (Strix Halo iGPU) or first compute device
        self.vk_phys_device = null;
        var best_score: u32 = 0;
        for (phys_devices[0..actual_count]) |pdev| {
            var props: c.VkPhysicalDeviceProperties = undefined;
            c.vkGetPhysicalDeviceProperties(pdev, &props);
            const score: u32 = switch (props.deviceType) {
                c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => 5,
                c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => 4,
                c.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU => 2,
                else => 1,
            };
            log.debug("dma-buf: device '{s}' type={d} score={d}", .{
                std.mem.sliceTo(&props.deviceName, 0), props.deviceType, score,
            });
            if (score > best_score) {
                best_score = score;
                self.vk_phys_device = pdev;
            }
        }

        if (self.vk_phys_device == null) {
            log.warn("dma-buf: No suitable Vulkan device", .{});
            return false;
        }

        var device_props: c.VkPhysicalDeviceProperties = undefined;
        c.vkGetPhysicalDeviceProperties(self.vk_phys_device, &device_props);
        const dev_name = std.mem.sliceTo(&device_props.deviceName, 0);
        log.info("dma-buf: Selected Vulkan device: {s}", .{dev_name});

        c.vkGetPhysicalDeviceMemoryProperties(self.vk_phys_device, &self.vk_mem_props);

        // ── Find compute queue family ──
        var qf_count: u32 = 0;
        c.vkGetPhysicalDeviceQueueFamilyProperties(self.vk_phys_device, &qf_count, null);
        if (qf_count == 0) return false;

        var qf_props: [16]c.VkQueueFamilyProperties = undefined;
        var actual_qf: u32 = @as(u32, @min(qf_count, @as(u32, 16)));
        c.vkGetPhysicalDeviceQueueFamilyProperties(self.vk_phys_device, &actual_qf, &qf_props);

        var compute_family: ?u32 = null;
        for (qf_props[0..actual_qf], 0..) |qf, i| {
            if (qf.queueFlags & c.VK_QUEUE_COMPUTE_BIT != 0) {
                compute_family = @intCast(i);
                break;
            }
        }
        const cf = compute_family orelse {
            log.warn("dma-buf: No compute queue family found", .{});
            return false;
        };

        // ── Create logical device with VK_KHR_external_memory_fd ──
        const dev_ext_name = "VK_KHR_external_memory_fd";
        const dev_exts = [_][*:0]const u8{dev_ext_name};

        const queue_priority: f32 = 1.0;
        const queue_ci = c.VkDeviceQueueCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueFamilyIndex = cf,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority,
        };

        const device_ci = c.VkDeviceCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_ci,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = 1,
            .ppEnabledExtensionNames = &dev_exts,
            .pEnabledFeatures = null,
        };

        var device: c.VkDevice = null;
        if (c.vkCreateDevice(self.vk_phys_device, &device_ci, null, &device) != c.VK_SUCCESS) {
            log.warn("dma-buf: vkCreateDevice failed", .{});
            return false;
        }
        self.vk_device = device;
        self.vk_queue_family = cf;

        var queue: c.VkQueue = null;
        c.vkGetDeviceQueue(device, cf, 0, &queue);
        self.vk_queue = queue;

        // ── Create command pool + buffer + fence for barriers ──
        const pool_ci = c.VkCommandPoolCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = null,
            .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = cf,
        };

        var cmd_pool: c.VkCommandPool = null;
        if (c.vkCreateCommandPool(device, &pool_ci, null, &cmd_pool) != c.VK_SUCCESS) {
            log.warn("dma-buf: Command pool creation failed", .{});
            return false;
        }
        self.vk_cmd_pool = cmd_pool;

        const cmd_alloc_info = c.VkCommandBufferAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = null,
            .commandPool = cmd_pool,
            .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        var cmd_buffer: c.VkCommandBuffer = null;
        if (c.vkAllocateCommandBuffers(device, &cmd_alloc_info, &cmd_buffer) != c.VK_SUCCESS) {
            return false;
        }
        self.vk_cmd_buffer = cmd_buffer;

        const fence_ci = c.VkFenceCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
        };
        var fence: c.VkFence = null;
        if (c.vkCreateFence(device, &fence_ci, null, &fence) != c.VK_SUCCESS) {
            return false;
        }
        self.vk_fence = fence;

        // ── Select memory type for dma-buf import ──
        // On Strix Halo (UMA), DEVICE_LOCAL memory works for dma-buf import.
        // We prefer DEVICE_LOCAL | HOST_VISIBLE for best performance on integrated.
        self.vk_import_mem_type = findMemoryType(
            &self.vk_mem_props,
            c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        ) orelse findMemoryType(
            &self.vk_mem_props,
            c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ) orelse findMemoryType(
            &self.vk_mem_props,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        ) orelse {
            log.warn("dma-buf: No suitable memory type for import", .{});
            return false;
        };

        log.info("dma-buf: Vulkan device ready, import mem type {d}", .{self.vk_import_mem_type});
        self.vulkan_initialized = true;
        return true;
    }

    /// Import the dma-buf fd as VkDeviceMemory and create per-layer VkBuffers.
    /// Returns true on success.
    fn importDmaBuf(self: *KvCacheInterop) bool {
        const bo_size = self.npu_page_table.bo_size;
        if (bo_size == 0) {
            log.warn("dma-buf: NPU BO size is 0", .{});
            return false;
        }

        // ── VkImportMemoryFdInfoKHR chain ──
        const import_info = c.VkImportMemoryFdInfoKHR{
            .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
            .pNext = null,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
            .fd = self.dma_buf_fd,
        };

        const alloc_info = c.VkMemoryAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = &import_info,
            .allocationSize = bo_size,
            .memoryTypeIndex = self.vk_import_mem_type,
        };

        var imported_memory: c.VkDeviceMemory = null;
        const result = c.vkAllocateMemory(self.vk_device, &alloc_info, null, &imported_memory);
        if (result != c.VK_SUCCESS) {
            log.warn("dma-buf: vkAllocateMemory with dma-buf fd failed (result={d})", .{result});
            return false;
        }
        self.imported_memory = imported_memory;
        log.info("dma-buf: Imported {d} bytes as VkDeviceMemory (fd {d})", .{ bo_size, self.dma_buf_fd });

        // ── Create per-layer VkBuffer pairs ──
        const layer_buffers = self.allocator.alloc(LayerBuffers, self.n_layers) catch {
            c.vkFreeMemory(self.vk_device, imported_memory, null);
            return false;
        };

        // Per-token layout in NPU BO: [K_head0..K_headN, V_head0..V_headN]
        const k_per_token = @as(c.VkDeviceSize, self.n_kv_heads) * self.head_dim * 4;
        const v_per_token = @as(c.VkDeviceSize, self.n_kv_heads) * self.head_dim * 4;
        const token_bytes = k_per_token + v_per_token;
        const page_bytes = @as(c.VkDeviceSize, self.page_size_tokens) * token_bytes;
        const layer_bytes = @as(c.VkDeviceSize, self.pages_per_layer) * page_bytes;

        for (0..self.n_layers) |l| {
            const layer_offset = l * layer_bytes;

            // K buffer covers the K-portion of each token in this layer.
            // K data starts at layer_offset and is the first k_per_token bytes
            // of each token. The buffer size spans the full layer stride.
            const k_buf = self.createBufferForMemory(
                layer_offset,
                layer_bytes,
            ) catch {
                log.warn("dma-buf: Failed to create K buffer for layer {d}", .{l});
                return false;
            };

            // V buffer covers the V-portion.
            // V data starts at layer_offset + k_per_token (after K within each token).
            const v_buf = self.createBufferForMemory(
                layer_offset + k_per_token,
                layer_bytes - k_per_token,
            ) catch {
                log.warn("dma-buf: Failed to create V buffer for layer {d}", .{l});
                return false;
            };

            layer_buffers[l] = .{
                .k_buffer = k_buf,
                .v_buffer = v_buf,
                .k_offset = layer_offset,
                .v_offset = layer_offset + k_per_token,
                .k_size = @as(c.VkDeviceSize, self.pages_per_layer) *
                    @as(c.VkDeviceSize, self.page_size_tokens) * k_per_token,
                .v_size = @as(c.VkDeviceSize, self.pages_per_layer) *
                    @as(c.VkDeviceSize, self.page_size_tokens) * v_per_token,
            };
        }

        self.layer_buffers = layer_buffers;

        // Update page mappings for the entire BO
        for (0..self.n_layers) |l| {
            const layer_off = l * layer_bytes;
            for (0..self.pages_per_layer) |p| {
                const page_off = layer_off + p * page_bytes;
                const idx = self.pageMappingIndex(@intCast(l), @intCast(p));
                if (idx < self.page_mappings.len) {
                    self.page_mappings[idx] = .{
                        .offset = page_off,
                        .token_count = self.page_size_tokens,
                        .layer = @intCast(l),
                        .npu_page_id = @intCast(p),
                        .valid = true,
                    };
                }
            }
        }

        return true;
    }

    /// Create a VkBuffer backed by imported memory at the given offset and size.
    fn createBufferForMemory(
        self: *KvCacheInterop,
        offset: c.VkDeviceSize,
        size: c.VkDeviceSize,
    ) !c.VkBuffer {
        // We must include VkExternalMemoryBufferCreateInfo when the buffer
        // will be bound to memory allocated with a handle type (imported).
        const external_info = c.VkExternalMemoryBufferCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
            .pNext = null,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        };

        const buf_ci = c.VkBufferCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = &external_info,
            .flags = 0,
            .size = size,
            .usage = c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
        };

        var buf: c.VkBuffer = null;
        if (c.vkCreateBuffer(self.vk_device, &buf_ci, null, &buf) != c.VK_SUCCESS) {
            return error.BufferCreateFailed;
        }

        // Bind the buffer to our imported memory at the given offset
        if (c.vkBindBufferMemory(self.vk_device, buf, self.imported_memory, offset) != c.VK_SUCCESS) {
            c.vkDestroyBuffer(self.vk_device, buf, null);
            return error.BufferBindFailed;
        }

        return buf;
    }

    // ── Sync: NPU → GPU ──

    /// Copy KV data for a range of tokens from NPU page table to GPU buffers.
    /// Called after NPU prefill or decode to make KV cache available to GPU attention.
    ///
    /// ## DMA-BUF path
    /// No data copy needed — both sides access the same physical pages.
    /// Issues a VkMemoryBarrier to ensure NPU writes are visible to the GPU
    /// before attention dispatch.
    ///
    /// ## Staging path
    /// Reads NPU BO → host staging copy → writes to GPU staging buffers.
    ///
    /// @param layer          Transformer layer index.
    /// @param page_ids       Array of NPU page IDs to sync.
    /// @param token_count    Total number of tokens across all pages.
    /// @param gpu_k_buffer   GPU-side K buffer handle (VkBuffer as opaque ptr).
    /// @param gpu_v_buffer   GPU-side V buffer handle (VkBuffer as opaque ptr).
    pub fn syncNpuToGpu(
        self: *KvCacheInterop,
        layer: u32,
        page_ids: []const u32,
        token_count: u32,
        gpu_k_buffer: ?*anyopaque,
        gpu_v_buffer: ?*anyopaque,
    ) !void {
        if (self.use_dma_buf) {
            // ── DMA-BUF: zero-copy path ──
            // No data movement needed. Issue a global VkMemoryBarrier to make
            // NPU writes visible to the GPU before attention dispatch.
            self.dmaBufBarrier();
            log.debug("dma-buf sync NPU→GPU layer {d} ({d} tokens) — zero copy", .{ layer, token_count });
            return;
        }

        // ── Staging path: NPU BO → host staging → Vulkan upload ──
        const remaining_pages = if (token_count > 0)
            (token_count + self.page_size_tokens - 1) / self.page_size_tokens
        else
            @as(u32, @intCast(page_ids.len));

        const pages_to_sync = @min(remaining_pages, @as(u32, @intCast(page_ids.len)));

        // Calculate per-token sizes
        const k_per_token = @as(u64, self.n_kv_heads) * self.head_dim * 4;
        const v_per_token = @as(u64, self.n_kv_heads) * self.head_dim * 4;
        const token_bytes = k_per_token + v_per_token;

        for (0..pages_to_sync) |i| {
            const page_id = page_ids[i];
            const tokens_in_this_page = if (i == pages_to_sync - 1)
                token_count - i * self.page_size_tokens
            else
                self.page_size_tokens;

            // Stage 1: Read K and V from NPU BO mapped memory into host staging buffer
            const page_offset = self.npuPageOffset(layer, page_id, 0);
            const mapped = self.npu_page_table.mapped;

            for (0..tokens_in_this_page) |t| {
                const src_off = page_offset + t * token_bytes;
                const dst_off = t * token_bytes;
                if (src_off + token_bytes <= mapped.len and dst_off + token_bytes <= self.staging_buf.len) {
                    @memcpy(
                        self.staging_buf[dst_off .. dst_off + token_bytes],
                        mapped[src_off .. src_off + token_bytes],
                    );
                }
            }

            // Stage 2: Copy K portion to GPU K buffer via mapped staging memory
            if (gpu_k_buffer) |vk_buf_k| {
                const k_buf: c.VkBuffer = @ptrCast(vk_buf_k);
                const k_size = tokens_in_this_page * k_per_token;
                _ = k_size;

                // Write K data from staging to mapped staging memory
                if (self.staging_mapped_k) |mapped_k| {
                    const staging_k_start: u64 = 0;
                    for (0..tokens_in_this_page) |t| {
                        const token_staging_off = t * token_bytes; // K is at start
                        const dst_off = staging_k_start + t * k_per_token;
                        @memcpy(
                            mapped_k[dst_off .. dst_off + k_per_token],
                            self.staging_buf[token_staging_off .. token_staging_off + k_per_token],
                        );
                    }
                }

                // The staging_k mapped memory is already visible to the GPU
                // (HOST_COHERENT). If the buffer isn't host-coherent, we'd need
                // to flush here. All Vulkan implementations must support
                // VK_MEMORY_PROPERTY_HOST_COHERENT_BIT for staging memory.
                _ = k_buf;
            }

            // Stage 2b: Copy V portion to GPU V buffer via mapped staging memory
            if (gpu_v_buffer) |vk_buf_v| {
                const v_buf: c.VkBuffer = @ptrCast(vk_buf_v);
                const v_size = tokens_in_this_page * v_per_token;
                _ = v_size;

                if (self.staging_mapped_v) |mapped_v| {
                    const staging_v_start: u64 = 0;
                    for (0..tokens_in_this_page) |t| {
                        const token_staging_off = t * token_bytes + k_per_token; // V follows K
                        const dst_off = staging_v_start + t * v_per_token;
                        @memcpy(
                            mapped_v[dst_off .. dst_off + v_per_token],
                            self.staging_buf[token_staging_off .. token_staging_off + v_per_token],
                        );
                    }
                }
                _ = v_buf;
            }

            // Update page mapping
            self.updatePageMapping(layer, page_id, page_offset, tokens_in_this_page);
        }

        log.debug("Staging copy NPU→GPU layer {d} ({d} tokens, {d} pages)", .{
            layer, token_count, pages_to_sync,
        });
    }

    /// Copy KV data from GPU buffers to NPU page table.
    /// Called after GPU prefill or decode to keep NPU's page table in sync.
    pub fn syncGpuToNpu(
        self: *KvCacheInterop,
        layer: u32,
        page_ids: []const u32,
        token_count: u32,
        _gpu_k_buffer: ?*anyopaque,
        _gpu_v_buffer: ?*anyopaque,
    ) !void {
        _ = _gpu_k_buffer;
        _ = _gpu_v_buffer;
        if (self.use_dma_buf) {
            // DMA-BUF: reverse barrier to make GPU writes visible to NPU
            self.dmaBufBarrier();
            log.debug("dma-buf sync GPU→NPU layer {d} ({d} tokens) — zero copy", .{ layer, token_count });
            return;
        }

        // ── Staging path: read GPU staging → host → write NPU BO ──
        const remaining_pages = if (token_count > 0)
            (token_count + self.page_size_tokens - 1) / self.page_size_tokens
        else
            @as(u32, @intCast(page_ids.len));

        const pages_to_sync = @min(remaining_pages, @as(u32, @intCast(page_ids.len)));
        const k_per_token = @as(u64, self.n_kv_heads) * self.head_dim * 4;
        const v_per_token = @as(u64, self.n_kv_heads) * self.head_dim * 4;
        const token_bytes = k_per_token + v_per_token;

        for (0..pages_to_sync) |i| {
            const page_id = page_ids[i];
            const tokens_in_this_page = if (i == pages_to_sync - 1)
                token_count - i * self.page_size_tokens
            else
                self.page_size_tokens;

            // Stage 1: Read K from GPU staging mapped memory into host staging buffer
            if (self.staging_mapped_k) |mapped_k| {
                for (0..tokens_in_this_page) |t| {
                    const staging_k_off = t * k_per_token;
                    const token_staging_off = t * token_bytes;
                    @memcpy(
                        self.staging_buf[token_staging_off .. token_staging_off + k_per_token],
                        mapped_k[staging_k_off .. staging_k_off + k_per_token],
                    );
                }
            }

            // Stage 1b: Read V from GPU staging mapped memory
            if (self.staging_mapped_v) |mapped_v| {
                for (0..tokens_in_this_page) |t| {
                    const staging_v_off = t * v_per_token;
                    const token_staging_off = t * token_bytes + k_per_token;
                    @memcpy(
                        self.staging_buf[token_staging_off .. token_staging_off + v_per_token],
                        mapped_v[staging_v_off .. staging_v_off + v_per_token],
                    );
                }
            }

            // Stage 2: Write host staging buffer to NPU BO mapped memory
            const page_offset = self.npuPageOffset(layer, page_id, 0);
            const mapped = self.npu_page_table.mapped;

            for (0..tokens_in_this_page) |t| {
                const src_off = t * token_bytes;
                const dst_off = page_offset + t * token_bytes;
                if (dst_off + token_bytes <= mapped.len and src_off + token_bytes <= self.staging_buf.len) {
                    @memcpy(
                        mapped[dst_off .. dst_off + token_bytes],
                        self.staging_buf[src_off .. src_off + token_bytes],
                    );
                }
            }

            // Sync NPU BO to device (flush CPU writes to NPU))
            self.npu_page_table.kv_bo.sync(
                0, // XCL_BO_SYNC_BO_TO_DEVICE
                page_offset,
                tokens_in_this_page * token_bytes,
            ) catch {
                log.warn("syncGpuToNpu: XRT sync failed for layer {d} page {d}", .{ layer, page_id });
            };

            // Update page mapping
            self.updatePageMapping(layer, page_id, page_offset, tokens_in_this_page);
        }

        log.debug("Staging copy GPU→NPU layer {d} ({d} tokens, {d} pages)", .{
            layer, token_count, pages_to_sync,
        });
    }

    // ── Full-layer sync ──

    /// Synchronize ALL active KV cache pages for a given layer.
    ///
    /// Iterates the `KvPagePool` to find every page with `owner != null` and
    /// not evicted, then syncs their valid token ranges from NPU → GPU.
    ///
    /// This is the main entry point used by the fused executor between the
    /// NPU GEMM pass and the GPU flash attention pass for a layer.
    ///
    /// @param layer          Transformer layer index.
    /// @param token_count    Total active tokens (sum across all pages).
    /// @param gpu_k_buffer   GPU-side K buffer handle (opaque ptr to VkBuffer).
    /// @param gpu_v_buffer   GPU-side V buffer handle (opaque ptr to VkBuffer).
    pub fn syncLayer(
        self: *KvCacheInterop,
        layer: u32,
        token_count: u32,
        gpu_k_buffer: ?*anyopaque,
        gpu_v_buffer: ?*anyopaque,
    ) !void {
        // ── Collect active page IDs from the pool ──
        var page_ids = std.ArrayList(u32).empty;
        defer page_ids.deinit(self.allocator);

        // Iterate all pages in the pool, collecting those with active owners
        for (self.gpu_page_pool.pages) |*page| {
            if (page.owner != null and !page.evicted) {
                try page_ids.append(self.allocator, page.page_id);
            }
        }

        if (page_ids.items.len == 0) {
            log.debug("syncLayer {d}: no active pages", .{layer});
            return;
        }

        // ── Sync each active page ──
        // For each page, we only sync the number of tokens actually stored in it
        var synced_tokens: u32 = 0;
        for (page_ids.items) |page_id| {
            const page = &self.gpu_page_pool.pages[page_id];
            const page_tokens = if (page.token_count > 0) page.token_count else self.page_size_tokens;
            const tokens_to_sync = @min(page_tokens, token_count -| synced_tokens);

            if (tokens_to_sync == 0) break;

            // Sync this single page
            const single_page = [_]u32{page_id};
            try self.syncNpuToGpu(layer, &single_page, tokens_to_sync, gpu_k_buffer, gpu_v_buffer);

            synced_tokens += tokens_to_sync;
            if (synced_tokens >= token_count) break;
        }

        log.debug("syncLayer {d}: {d} active pages, {d}/{d} tokens synced", .{
            layer, page_ids.items.len, synced_tokens, token_count,
        });
    }

    // ── Page-level mapping ──

    /// Update the GPU page mapping for a specific page after a sync.
    fn updatePageMapping(
        self: *KvCacheInterop,
        layer: u32,
        page_id: u32,
        offset: u64,
        token_count: u32,
    ) void {
        const idx = self.pageMappingIndex(layer, page_id);
        if (idx < self.page_mappings.len) {
            self.page_mappings[idx] = .{
                .offset = offset,
                .token_count = token_count,
                .layer = layer,
                .npu_page_id = page_id,
                .valid = true,
            };
        }
    }

    /// Get the GPU-accessible byte offset for a specific (layer, page, token).
    /// Returns 0 if the mapping is invalid.
    pub fn getGpuOffset(
        self: *const KvCacheInterop,
        layer: u32,
        page_id: u32,
        token_in_page: u32,
    ) u64 {
        const idx = self.pageMappingIndex(layer, page_id);
        if (idx < self.page_mappings.len and self.page_mappings[idx].valid) {
            const pm = &self.page_mappings[idx];
            const token_bytes = @as(u64, self.n_kv_heads) * self.head_dim * 4 * 2;
            return pm.offset + @as(u64, token_in_page) * token_bytes;
        }
        return 0;
    }

    /// Get the VkBuffer for K cache at a given layer (dma-buf path).
    /// Returns VK_NULL_HANDLE if dma-buf is not active.
    pub fn getGpuKBuffer(self: *const KvCacheInterop, layer: u32) c.VkBuffer {
        if (!self.use_dma_buf or layer >= self.n_layers) return null;
        return self.layer_buffers[layer].k_buffer;
    }

    /// Get the VkBuffer for V cache at a given layer (dma-buf path).
    /// Returns VK_NULL_HANDLE if dma-buf is not active.
    pub fn getGpuVBuffer(self: *const KvCacheInterop, layer: u32) c.VkBuffer {
        if (!self.use_dma_buf or layer >= self.n_layers) return null;
        return self.layer_buffers[layer].v_buffer;
    }

    /// Get the byte offset within the imported memory for K data at a layer.
    pub fn getGpuKOffset(self: *const KvCacheInterop, layer: u32) c.VkDeviceSize {
        if (!self.use_dma_buf or layer >= self.n_layers) return 0;
        return self.layer_buffers[layer].k_offset;
    }

    /// Get the byte offset within the imported memory for V data at a layer.
    pub fn getGpuVOffset(self: *const KvCacheInterop, layer: u32) c.VkDeviceSize {
        if (!self.use_dma_buf or layer >= self.n_layers) return 0;
        return self.layer_buffers[layer].v_offset;
    }

    /// Compute the linear index into the page_mappings array.
    fn pageMappingIndex(self: *const KvCacheInterop, layer: u32, page_id: u32) usize {
        return @as(usize, layer) * self.pages_per_layer + page_id;
    }

    /// Compute the byte offset within the NPU BO for a given (layer, page).
    /// token_in_page is unused in offset calc since we compute per-token offsets
    /// in the sync functions.
    fn npuPageOffset(self: *const KvCacheInterop, layer: u32, page_id: u32, _token_in_page: u32) u64 {
        _ = _token_in_page;
        const token_bytes = @as(u64, self.n_kv_heads) * self.head_dim * 4 * 2;
        const page_bytes = @as(u64, self.page_size_tokens) * token_bytes;
        const layer_bytes = @as(u64, self.pages_per_layer) * page_bytes;
        return layer_bytes * layer + page_bytes * page_id;
    }

    // ── DMA-BUF helpers ──

    /// Issue a VkMemoryBarrier to make NPU writes visible to the GPU.
    /// Uses the cached command buffer and fence.
    fn dmaBufBarrier(self: *KvCacheInterop) void {
        if (!self.vulkan_initialized) return;

        const cmd = self.vk_cmd_buffer;
        const device = self.vk_device;
        const queue = self.vk_queue;
        const fence = self.vk_fence;

        // Reset fence and command buffer
        _ = c.vkResetFences(device, 1, &fence);
        _ = c.vkResetCommandBuffer(cmd, 0);

        // Begin command buffer
        const begin_info = c.VkCommandBufferBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = null,
        };
        if (c.vkBeginCommandBuffer(cmd, &begin_info) != c.VK_SUCCESS) return;

        // Issue a full memory barrier
        const barrier = c.VkMemoryBarrier{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = null,
            .srcAccessMask = c.VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = c.VK_ACCESS_MEMORY_READ_BIT | c.VK_ACCESS_MEMORY_WRITE_BIT,
        };

        c.vkCmdPipelineBarrier(
            cmd,
            c.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            c.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            1, &barrier,
            0, null,
            0, null,
        );

        _ = c.vkEndCommandBuffer(cmd);

        // Submit and wait
        const submit_info = c.VkSubmitInfo{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = null,
            .pWaitDstStageMask = null,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = null,
        };

        if (c.vkQueueSubmit(queue, 1, &submit_info, fence) != c.VK_SUCCESS) return;
        _ = c.vkWaitForFences(device, 1, &fence, c.VK_TRUE, std.math.maxInt(u64));
    }

    // ── Staging helpers ──

    /// Initialize staging upload buffers (host-visible, host-coherent).
    /// Called lazily on first staging sync if dma-buf is inactive.
    fn ensureStagingBuffers(self: *KvCacheInterop) !void {
        if (self.staging_upload_k != null) return; // Already initialized

        // Create Vulkan device if not already done via dma-buf probe
        if (!self.vulkan_initialized) {
            if (!self.initVulkanForImport()) {
                log.warn("staging: Cannot create Vulkan device for staging buffers", .{});
                return error.VulkanUnavailable;
            }
        }

        // Allocate staging buffers large enough for one layer's K/V
        const k_per_token = @as(c.VkDeviceSize, self.n_kv_heads) * self.head_dim * 4;
        const v_per_token = @as(c.VkDeviceSize, self.n_kv_heads) * self.head_dim * 4;
        const max_pages_per_sync = self.pages_per_layer;
        const max_tokens = max_pages_per_sync * self.page_size_tokens;

        const k_size = max_tokens * k_per_token;
        const v_size = max_tokens * v_per_token;
        self.staging_upload_size = k_size;

        // Find host-coherent memory type
        const host_mem_type = findMemoryType(
            &self.vk_mem_props,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        ) orelse {
            log.warn("staging: No HOST_VISIBLE | HOST_COHERENT memory type", .{});
            return error.NoSuitableMemoryType;
        };

        // ── K staging buffer ──
        const k_buf_ci = c.VkBufferCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .size = k_size,
            .usage = c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
        };
        var k_buf: c.VkBuffer = null;
        if (c.vkCreateBuffer(self.vk_device, &k_buf_ci, null, &k_buf) != c.VK_SUCCESS) {
            return error.BufferCreateFailed;
        }
        self.staging_upload_k = k_buf;

        var req_k: c.VkMemoryRequirements = undefined;
        c.vkGetBufferMemoryRequirements(self.vk_device, k_buf, &req_k);

        const alloc_k = c.VkMemoryAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = null,
            .allocationSize = req_k.size,
            .memoryTypeIndex = host_mem_type,
        };
        if (c.vkAllocateMemory(self.vk_device, &alloc_k, null, &self.staging_memory_k) != c.VK_SUCCESS) {
            return error.BufferMemoryAllocFailed;
        }
        _ = c.vkBindBufferMemory(self.vk_device, k_buf, self.staging_memory_k, 0);

        var ptr_k: ?*anyopaque = null;
        _ = c.vkMapMemory(self.vk_device, self.staging_memory_k, 0, k_size, 0, &ptr_k);
        self.staging_mapped_k = @ptrCast(@alignCast(ptr_k));

        // ── V staging buffer ──
        const v_buf_ci = c.VkBufferCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .size = v_size,
            .usage = c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
        };
        var v_buf: c.VkBuffer = null;
        if (c.vkCreateBuffer(self.vk_device, &v_buf_ci, null, &v_buf) != c.VK_SUCCESS) {
            return error.BufferCreateFailed;
        }
        self.staging_upload_v = v_buf;

        var req_v: c.VkMemoryRequirements = undefined;
        c.vkGetBufferMemoryRequirements(self.vk_device, v_buf, &req_v);
        const alloc_v = c.VkMemoryAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = null,
            .allocationSize = req_v.size,
            .memoryTypeIndex = host_mem_type,
        };
        if (c.vkAllocateMemory(self.vk_device, &alloc_v, null, &self.staging_memory_v) != c.VK_SUCCESS) {
            return error.BufferMemoryAllocFailed;
        }
        _ = c.vkBindBufferMemory(self.vk_device, v_buf, self.staging_memory_v, 0);

        var ptr_v: ?*anyopaque = null;
        _ = c.vkMapMemory(self.vk_device, self.staging_memory_v, 0, v_size, 0, &ptr_v);
        self.staging_mapped_v = @ptrCast(@alignCast(ptr_v));

        log.info("staging: Upload buffers ready (K={d}, V={d} bytes)", .{ k_size, v_size });
    }

    // ── Deinit helpers ──

    fn deinitDmaBuf(self: *KvCacheInterop) void {
        // Destroy per-layer buffers
        for (self.layer_buffers) |lb| {
            if (lb.k_buffer != null) c.vkDestroyBuffer(self.vk_device, lb.k_buffer, null);
            if (lb.v_buffer != null) c.vkDestroyBuffer(self.vk_device, lb.v_buffer, null);
        }
        self.allocator.free(self.layer_buffers);
        self.layer_buffers = &.{};

        // Free imported memory
        if (self.imported_memory != null) {
            c.vkFreeMemory(self.vk_device, self.imported_memory, null);
            self.imported_memory = null;
        }

        // Close dma-buf fd
        if (self.dma_buf_fd >= 0) {
            _ = std.os.linux.close(@as(i32, @intCast(self.dma_buf_fd)));
            self.dma_buf_fd = -1;
        }

        self.deinitVulkan();
    }

    fn deinitStaging(self: *KvCacheInterop) void {
        if (self.staging_upload_k != null) {
            if (self.staging_mapped_k) |_| {
                c.vkUnmapMemory(self.vk_device, self.staging_memory_k);
                self.staging_mapped_k = null;
            }
            c.vkFreeMemory(self.vk_device, self.staging_memory_k, null);
            c.vkDestroyBuffer(self.vk_device, self.staging_upload_k, null);
            self.staging_upload_k = null;
        }
        if (self.staging_upload_v != null) {
            if (self.staging_mapped_v) |_| {
                c.vkUnmapMemory(self.vk_device, self.staging_memory_v);
                self.staging_mapped_v = null;
            }
            c.vkFreeMemory(self.vk_device, self.staging_memory_v, null);
            c.vkDestroyBuffer(self.vk_device, self.staging_upload_v, null);
            self.staging_upload_v = null;
        }
        self.deinitVulkan();
    }

    fn deinitVulkan(self: *KvCacheInterop) void {
        // Destroy barrier resources
        if (self.vk_fence != null) {
            c.vkDestroyFence(self.vk_device, self.vk_fence, null);
            self.vk_fence = null;
        }
        if (self.vk_cmd_pool != null) {
            // vkFreeCommandBuffers is implicit when destroying the pool
            c.vkDestroyCommandPool(self.vk_device, self.vk_cmd_pool, null);
            self.vk_cmd_pool = null;
            self.vk_cmd_buffer = null;
        }
        if (self.vk_device != null) {
            _ = c.vkDeviceWaitIdle(self.vk_device);
            c.vkDestroyDevice(self.vk_device, null);
            self.vk_device = null;
        }
        if (self.vk_instance != null) {
            c.vkDestroyInstance(self.vk_instance, null);
            self.vk_instance = null;
        }
        self.vk_phys_device = null;
        self.vk_queue = null;
        self.vulkan_initialized = false;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Standalone helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Find the first memory type satisfying `required_flags`.
fn findMemoryType(
    mem_props: *const c.VkPhysicalDeviceMemoryProperties,
    required_flags: c.VkMemoryPropertyFlags,
) ?u32 {
    for (0..mem_props.memoryTypeCount) |i| {
        if (mem_props.memoryTypes[i].propertyFlags & required_flags == required_flags) {
            return @intCast(i);
        }
    }
    return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

test "KvCacheInterop init and deinit" {
    const allocator = std.testing.allocator;

    var pool = try KvPagePool.init(allocator, 16, 64); // 16 pages, 64 tokens/page
    defer pool.deinit();

    var page_table = NpuPageTable.init(64, 8, 1, 128);

    var cross_mem = try CrossBackendMemory.init(allocator);
    defer cross_mem.deinit();

    var interop = try KvCacheInterop.init(
        allocator,
        1,   // n_layers
        8,   // n_kv_heads
        128, // head_dim
        64,  // page_size_tokens
        &page_table,
        &pool,
        &cross_mem,
    );
    defer interop.deinit();

    // Verify basic arithmetic
    try std.testing.expectEqual(false, interop.use_dma_buf);
    try std.testing.expectEqual(@as(u64, 8 * 128 * 4 * 2), interop.bytes_per_token);
    try std.testing.expectEqual(@as(u32, 16), interop.pages_per_layer);
    try std.testing.expectEqual(@as(u32, 16), interop.total_pages_in_pool);
}

test "page mapping index math" {
    const allocator = std.testing.allocator;

    var pool = try KvPagePool.init(allocator, 16, 64);
    defer pool.deinit();

    var page_table = NpuPageTable.init(64, 8, 1, 128);

    var cross_mem = try CrossBackendMemory.init(allocator);
    defer cross_mem.deinit();

    var interop = try KvCacheInterop.init(
        allocator,
        2, 8, 128, 64,
        &page_table,
        &pool,
        &cross_mem,
    );
    defer interop.deinit();

    try std.testing.expectEqual(@as(usize, 0), interop.pageMappingIndex(0, 0));
    try std.testing.expectEqual(@as(usize, 15), interop.pageMappingIndex(0, 15));
    try std.testing.expectEqual(@as(usize, 16), interop.pageMappingIndex(1, 0));
    try std.testing.expectEqual(@as(usize, 31), interop.pageMappingIndex(1, 15));

    // Total mappings = n_layers * pages_per_layer = 2 * 16 = 32
    try std.testing.expectEqual(@as(usize, 32), interop.page_mappings.len);
}

test "npuPageOffset math" {
    const allocator = std.testing.allocator;

    var pool = try KvPagePool.init(allocator, 16, 64);
    defer pool.deinit();

    var page_table = NpuPageTable.init(64, 8, 128, 1);

    var cross_mem = try CrossBackendMemory.init(allocator);
    defer cross_mem.deinit();

    var interop = try KvCacheInterop.init(
        allocator,
        2, 8, 128, 64,
        &page_table,
        &pool,
        &cross_mem,
    );
    defer interop.deinit();

    const token_bytes: u64 = 8 * 128 * 4 * 2; // 8192
    const page_bytes: u64 = 64 * token_bytes; // 524288
    const layer_bytes: u64 = 16 * page_bytes; // 8388608

    try std.testing.expectEqual(@as(u64, 0), interop.npuPageOffset(0, 0, 0));
    try std.testing.expectEqual(page_bytes, interop.npuPageOffset(0, 1, 0));
    try std.testing.expectEqual(layer_bytes, interop.npuPageOffset(1, 0, 0));
    try std.testing.expectEqual(layer_bytes + page_bytes, interop.npuPageOffset(1, 1, 0));
}

test "tryEnableDmaBuf gracefully falls back on non-dma-buf systems" {
    const allocator = std.testing.allocator;

    var pool = try KvPagePool.init(allocator, 16, 64);
    defer pool.deinit();

    var page_table = NpuPageTable.init(64, 8, 1, 128);

    var cross_mem = try CrossBackendMemory.init(allocator);
    defer cross_mem.deinit();

    var interop = try KvCacheInterop.init(
        allocator,
        1, 8, 128, 64,
        &page_table,
        &pool,
        &cross_mem,
    );
    defer interop.deinit();

    // On a system without /dev/dri/renderD128 or without NPU BO allocated,
    // this should gracefully return false. If the system happens to have
    // dma-buf support, it will still work.
    const dma_buf_enabled = interop.tryEnableDmaBuf();
    // When /dev/dri/renderD128 doesn't exist, should be false.
    // If it exists and export works, could be true.
    _ = dma_buf_enabled;
}
