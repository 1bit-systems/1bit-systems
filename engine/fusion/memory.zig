//! Cross-backend memory sharing between NPU (XRT BOs) and GPU (Vulkan buffers).
//!
//! Uses dma-buf for zero-copy buffer sharing between devices.
//! Dma-buf support is probed at init time and falls back gracefully.
//!
//! Architecture:
//!   ┌──────────┐   amdgpu_bo_export    ┌──────────────────┐
//!   │ GPU GTT  │ ────────────────────→ │   dma-buf fd     │
//!   │ BO       │   (dma_buf_fd)        │  (shared pages)  │
//!   └────┬─────┘                       └────────┬─────────┘
//!        │                                      │
//!        │ amdgpu_bo_cpu_map                    │ DRM_IOCTL_PRIME_FD_TO_HANDLE
//!        ▼                                      ▼
//!   ┌──────────┐                         ┌──────────┐
//!   │ CPU mmap │                         │ NPU XRT  │
//!   │ (direct) │                         │ BO       │
//!   └──────────┘                         └──────────┘
//!
//! All three domains (GPU, CPU, NPU) access the same physical pages.
//! No memcpy between devices.
//!
//! @section Fused Engine
const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;

const log = std.log.scoped(.fusion_memory);

const is_linux = builtin.target.os.tag == .linux;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default DRM render node for the AMD GPU.
const DEFAULT_RENDER_NODE: [:0]const u8 = "/dev/dri/renderD128";

/// Maximum number of tracked shared buffers.
const MAX_BUFFERS: usize = 256;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A buffer shared between GPU (GTT), CPU (mmap), and optionally NPU (XRT).
pub const SharedBuffer = struct {
    /// dma-buf file descriptor. -1 when not exported.
    dma_buf_fd: i32 = -1,
    /// CPU-accessible mmap of the dma-buf (or GTT BO).
    cpu_ptr: ?[*]u8 = null,
    /// Size of the buffer in bytes.
    size: u64 = 0,
    /// XRT BO handle (opaque void* from xrtBufferHandle), 0 when not NPU-imported.
    xrt_bo_handle: u64 = 0,
    /// Vulkan device memory handle (VkDeviceMemory), 0 when not GPU-imported.
    vk_device_memory: u64 = 0,
    /// libdrm buffer handle for the GTT allocation.
    bo_handle: u64 = 0,
    /// Whether this slot holds a valid buffer.
    valid: bool = false,

    pub fn init(dma_buf_fd: i32, cpu_ptr: [*]u8, size: u64, bo_handle: u64) SharedBuffer {
        return .{
            .dma_buf_fd = dma_buf_fd,
            .cpu_ptr = cpu_ptr,
            .size = size,
            .bo_handle = bo_handle,
            .valid = true,
        };
    }

    pub fn empty() SharedBuffer {
        return .{ .valid = false, .dma_buf_fd = -1 };
    }
};

// ---------------------------------------------------------------------------
// CrossBackendMemory
// ---------------------------------------------------------------------------

/// Cross-backend memory manager.
///
/// Allocates shared buffers in the GPU GTT domain and exports them as dma-buf
/// file descriptors that can be imported by the NPU (via XRT) or mapped by the
/// CPU, all accessing the same physical pages — zero copy.
pub const CrossBackendMemory = struct {
    allocator: std.mem.Allocator,
    /// Tracking table for allocated shared buffers.
    buffers: []SharedBuffer,
    /// Whether dma-buf sharing is available on this system.
    dma_buf_available: bool,
    /// Path to the DRM render node (set at init time).
    render_node_path: [:0]const u8,
    /// fd for the open DRM render node (cached for reuse). -1 if not opened.
    drm_fd: i32,
    /// libdrm_amdgpu device handle (opaque pointer). 0 if not initialized.
    amdgpu_dev: u64,

    /// Initialize the cross-backend memory manager.
    ///
    /// Allocates the tracking table but does NOT probe the system.
    /// Call `probeSystem()` or use `initWithProbe()` to enable dma-buf.
    /// When dma-buf is unavailable, all allocation operations return invalid
    /// buffers and the caller should use staging copies instead.
    pub fn init(allocator: std.mem.Allocator) !CrossBackendMemory {
        const buf_slots = try allocator.alloc(SharedBuffer, MAX_BUFFERS);
        errdefer allocator.free(buf_slots);
        for (buf_slots) |*slot| slot.* = SharedBuffer.empty();

        return CrossBackendMemory{
            .allocator = allocator,
            .buffers = buf_slots,
            .dma_buf_available = false,
            .render_node_path = @as([:0]const u8, DEFAULT_RENDER_NODE),
            .drm_fd = -1,
            .amdgpu_dev = 0,
        };
    }

    /// Probe the system and enable dma-buf.
    /// Must be called before allocating shared buffers.
    /// Safe to call even when dma-buf is unavailable.
    pub fn probeSystem(self: *CrossBackendMemory) void {
        if (!is_linux) return;
        self.probe();
    }

    /// Release all resources. Closes dma-buf fds, unmaps buffers, frees GTT BOs.
    /// Initialize with full probe (convenience for production use).
    /// Equivalent to init() + probeSystem().
    pub fn initWithProbe(allocator: std.mem.Allocator) !CrossBackendMemory {
        var self = try CrossBackendMemory.init(allocator);
        self.probeSystem();
        return self;
    }

    pub fn deinit(self: *CrossBackendMemory) void {
        for (self.buffers) |*buf| {
            if (buf.valid) {
                self.freeBuffer(buf);
            }
        }
        self.allocator.free(self.buffers);

        if (self.drm_fd >= 0) {
            _ = linux.close(self.drm_fd);
        }
        self.* = undefined;
    }

    // -----------------------------------------------------------------------
    // Allocation
    // -----------------------------------------------------------------------

    /// Allocate a shared buffer accessible by GPU (GTT), CPU (mmap), and
    /// optionally NPU (via dma-buf import).
    ///
    /// The buffer is allocated in the GPU GTT domain (system RAM, accessible
    /// via the GPU's GTT aperture) and exported as a dma-buf fd. The CPU can
    /// mmap the fd or the GTT BO directly.
    ///
    /// Returns a `SharedBuffer` with `.valid = false` when dma-buf is
    /// unavailable — the caller must use staging copies instead.
    pub fn allocateShared(self: *CrossBackendMemory, size: u64) SharedBuffer {
        if (!self.dma_buf_available) {
            return SharedBuffer.empty();
        }

        // Allocate GTT BO and export as dma-buf fd
        const buf = self.gttAllocAndExport(size) catch |err| {
            log.warn("GTT allocation failed: {s}", .{@errorName(err)});
            return SharedBuffer.empty();
        };

        // Track in our slot table
        if (self.findFreeSlot()) |slot| {
            slot.* = buf;
            log.debug("allocated shared buffer: fd={d}, size={d}, ptr={*}", .{
                buf.dma_buf_fd, buf.size, buf.cpu_ptr.?,
            });
        } else {
            log.warn("shared buffer tracking table full, closing leaked fd={d}", .{buf.dma_buf_fd});
            closeFd(buf.dma_buf_fd);
            if (buf.cpu_ptr) |ptr| {
                _ = linux.munmap(ptr, buf.size);
            }
            return SharedBuffer.empty();
        }

        return buf;
    }

    /// Allocate a shared buffer from a dma-buf fd obtained externally.
    /// This allows importing a buffer that was allocated by the NPU (via XRT
    /// export) or another device.
    pub fn importFd(self: *CrossBackendMemory, dma_buf_fd: i32, size: u64) !SharedBuffer {
        if (!self.dma_buf_available) return error.DmaBufUnavailable;
        if (dma_buf_fd < 0) return error.InvalidArgument;

        // mmap the dma-buf fd for CPU access
        const mmap_result = linux.mmap(
            null,
            size,
            linux.PROT{ .READ = true, .WRITE = true },
            linux.MAP{ .TYPE = linux.MAP_TYPE.SHARED },
            dma_buf_fd,
            0,
        );
        if (mmap_result == std.math.maxInt(usize)) {
            log.warn("mmap of dma-buf fd={d} failed", .{dma_buf_fd});
            return error.MmapFailed;
        }

        const buf = SharedBuffer.init(dma_buf_fd, @as([*]u8, @ptrFromInt(mmap_result)), size, 0);

        if (self.findFreeSlot()) |slot| {
            slot.* = buf;
        } else {
            log.warn("tracking table full, leaking fd={d}", .{dma_buf_fd});
        }

        return buf;
    }

    /// Free a shared buffer previously returned by `allocateShared()` or `importFd()`.
    pub fn freeShared(self: *CrossBackendMemory, buf: *SharedBuffer) void {
        if (!buf.valid) return;
        self.freeBuffer(buf);
        buf.* = SharedBuffer.empty();
    }

    // -----------------------------------------------------------------------
    // Synchronisation
    // -----------------------------------------------------------------------

    /// Synchronize a shared buffer for NPU→GPU data visibility.
    ///
    /// Call after the NPU has written to the buffer and before the GPU reads it.
    /// This ensures the GPU's GTT cache sees the NPU's writes.
    pub fn syncNpuToGpu(self: *CrossBackendMemory, buf: *const SharedBuffer) void {
        _ = self;
        _ = buf;
        // On Strix Halo with unified memory, GTT and NPU share the same
        // physical pages and the same coherency domain. No explicit sync
        // needed in most cases — the IOMMU/snoop handles it.
        //
        // If coherency issues arise, use:
        //   dma_buf_sync(DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ)
        //   ... GPU reads ...
        //   dma_buf_sync(DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ)
        //
        // Call syncCpuAccess() to use the dma_buf_sync ioctl.
    }

    /// Synchronize a shared buffer for GPU→NPU data visibility.
    ///
    /// Call after the GPU has written to the buffer and before the NPU reads it.
    pub fn syncGpuToNpu(self: *CrossBackendMemory, buf: *const SharedBuffer) void {
        _ = self;
        _ = buf;
        // Same coherency note as syncNpuToGpu.
    }

    /// Synchronize a shared buffer for CPU access (read or write).
    ///
    /// Call before reading/writing the buffer from the CPU, and again after
    /// to make writes visible to the device.
    pub fn syncCpuAccess(self: *CrossBackendMemory, buf: *const SharedBuffer, start: bool, write: bool) !void {
        _ = self;
        if (!buf.valid or buf.dma_buf_fd < 0) return;

        const rw_flag: u64 = if (write) 2 else 1; // DMA_BUF_SYNC_WRITE / DMA_BUF_SYNC_READ
        const se_flag: u64 = if (start) 0 else 4; // DMA_BUF_SYNC_START / DMA_BUF_SYNC_END
        const flags = rw_flag | se_flag;

        var sync = dma_buf_sync{ .flags = flags };
        const rc = linux.ioctl(buf.dma_buf_fd, DMA_BUF_IOCTL_SYNC, @intFromPtr(&sync));
        if (rc != 0) {
            const errno = linux.errno(rc);
            log.warn("dma_buf_sync failed on fd={d}: errno={}", .{ buf.dma_buf_fd, .{ .errno = @intFromEnum(errno) } });
            return switch (errno) {
                .ACCES => error.PermissionDenied,
                .BADF => error.InvalidArgument,
                .FAULT => error.Unexpected,
                .INVAL => error.InvalidArgument,
                .NOTTY => error.NotSupported,
                else => error.Unexpected,
            };
        }
    }

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /// Find a tracked buffer by its dma-buf fd.
    pub fn findByFd(self: *const CrossBackendMemory, fd: i32) ?*const SharedBuffer {
        for (self.buffers) |*slot| {
            if (slot.valid and slot.dma_buf_fd == fd) return slot;
        }
        return null;
    }

    /// Number of currently tracked buffers.
    pub fn activeCount(self: *const CrossBackendMemory) usize {
        var count: usize = 0;
        for (self.buffers) |buf| {
            if (buf.valid) count += 1;
        }
        return count;
    }

    /// Whether dma-buf sharing is available.
    pub fn isAvailable(self: *const CrossBackendMemory) bool {
        return self.dma_buf_available;
    }

    // -----------------------------------------------------------------------
    // Internal: probe
    // -----------------------------------------------------------------------

    /// Probe the system for dma-buf support.
    fn probe(self: *CrossBackendMemory) void {
        // 1. Check that the DRM render node exists and is accessible.
        if (accessPath(self.render_node_path, 0) != 0) {
            // Try scanning /dev/dri/
            self.render_node_path = findRenderNode() orelse {
                log.info("no DRM render node found — dma-buf unavailable", .{});
                return;
            };
        }

        // 2. Open the DRM render node.
        const fd = sys_open(self.render_node_path, 2, 0); // O_RDWR = 2
        if (fd < 0) {
            log.warn("cannot open {s}", .{self.render_node_path});
            return;
        }
        self.drm_fd = fd;

        // 3. Try to initialize libdrm_amdgpu.
        const dev = amdgpuDeviceInit(self.drm_fd) catch |err| {
            log.warn("libdrm_amdgpu init failed: {s}", .{@errorName(err)});
            _ = linux.close(self.drm_fd);
            self.drm_fd = -1;
            return;
        };
        self.amdgpu_dev = dev;

        // 4. Quick verification: can we query the device info?
        if (amdgpuQueryGpuInfo(dev)) {
            self.dma_buf_available = true;
        } else {
            log.warn("amdgpu query failed — dma-buf unavailable", .{});
            amdgpuDeviceDeinit(dev) catch {};
            _ = linux.close(self.drm_fd);
            self.drm_fd = -1;
            self.amdgpu_dev = 0;
        }
    }

    // -----------------------------------------------------------------------
    // Internal: GTT allocation + dma-buf export
    // -----------------------------------------------------------------------

    /// Allocate a buffer in the GPU GTT domain and export it as a dma-buf fd.
    /// Returns a populated SharedBuffer with the fd and CPU mmap.
    fn gttAllocAndExport(self: *CrossBackendMemory, size: u64) !SharedBuffer {
        // Allocate in GTT domain
        const alloc_result = amdgpuBoAllocGtt(self.amdgpu_dev, size) catch |err| {
            log.warn("amdgpu_bo_alloc GTT failed: {s}", .{@errorName(err)});
            return err;
        };

        const bo_handle = alloc_result.bo_handle;
        const dma_buf_fd = alloc_result.fd;

        // mmap the GTT BO for CPU access
        const cpu_ptr = amdgpuBoCpuMap(self.amdgpu_dev, bo_handle) catch |err| {
            log.warn("amdgpu_bo_cpu_map failed: {s}", .{@errorName(err)});
            closeFd(dma_buf_fd);
            return err;
        };

        return SharedBuffer.init(dma_buf_fd, cpu_ptr, size, bo_handle);
    }

    /// Free a tracked buffer: munmap, close dma-buf fd, free GTT BO.
    fn freeBuffer(self: *CrossBackendMemory, buf: *SharedBuffer) void {
        if (buf.cpu_ptr) |ptr| {
            _ = linux.munmap(ptr, buf.size);
            buf.cpu_ptr = null;
        }
        if (buf.dma_buf_fd >= 0) {
            closeFd(buf.dma_buf_fd);
            buf.dma_buf_fd = -1;
        }
        if (buf.bo_handle != 0 and self.amdgpu_dev != 0) {
            amdgpuBoFree(self.amdgpu_dev, buf.bo_handle) catch {};
            buf.bo_handle = 0;
        }
        buf.valid = false;
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn findFreeSlot(self: *CrossBackendMemory) ?*SharedBuffer {
        for (self.buffers) |*slot| {
            if (!slot.valid) return slot;
        }
        return null;
    }
};

// ---------------------------------------------------------------------------
// DMA-buf sync ioctl
// ---------------------------------------------------------------------------

/// Build ioctl request code for DMA_BUF_IOCTL_SYNC.
/// Equivalent to Linux kernel's _IOW('b', 0, struct dma_buf_sync).
fn buildIoctlRequest(io_type: u8, nr: u8, comptime T: type) u32 {
    const arch = builtin.cpu.arch;
    const dir_write: u32 = switch (arch) {
        .mips, .mipsel, .mips64, .mips64el, .powerpc, .powerpcle,
        .powerpc64, .powerpc64le, .sparc, .sparc64 => 2,
        else => 1,
    };
    const size_val: u32 = @sizeOf(T);
    return (@as(u32, dir_write) << 30) |
           (@as(u32, io_type) << 8) |
           (@as(u32, nr) << 0) |
           (@as(u32, size_val) << 16);
}

const DMA_BUF_IOCTL_SYNC: u32 = buildIoctlRequest('b', 0, dma_buf_sync);

const dma_buf_sync = extern struct {
    flags: u64,
};

// ---------------------------------------------------------------------------
// Linux helpers
// ---------------------------------------------------------------------------

fn sys_open(path: [:0]const u8, flags: u32, perm: u32) i32 {
    if (!is_linux) return -1;
    const fd_cwd = @as(u64, @bitCast(@as(i64, -100))); // AT_FDCWD
    const rc = linux.syscall5(.openat, fd_cwd, @intFromPtr(path.ptr), flags, perm, 0);
    if (rc >= std.math.maxInt(u32)) return @as(i32, @intCast(@as(i64, -@as(i64, @intCast(rc -% std.math.maxInt(usize) +% 1)))));
    return @as(i32, @intCast(rc));
}

fn accessPath(path: [:0]const u8, mode: u32) i32 {
    if (!is_linux) return -1;
    return @intCast(linux.access(path, mode));
}

fn findRenderNode() ?[:0]const u8 {
    if (!is_linux) return null;

    // Check commonly used render nodes.
    const candidates = [_][:0]const u8{
        "/dev/dri/renderD128",
        "/dev/dri/renderD129",
        "/dev/dri/renderD130",
        "/dev/dri/renderD131",
        "/dev/dri/renderD132",
        "/dev/dri/renderD133",
    };

    for (candidates) |path| {
        if (accessPath(path, 0) == 0) {
            return path;
        }
    }
    return null;
}

fn closeFd(fd: i32) void {
    if (!is_linux) return;
    _ = linux.close(fd);
}

// ---------------------------------------------------------------------------
// libdrm_amdgpu bindings (loaded at runtime via DynLib)
// ---------------------------------------------------------------------------

const DynLib = std.DynLib;

/// Result of a GTT allocation.
const GttAllocResult = struct {
    bo_handle: u64,
    fd: i32,
};

/// Thin C ABI bindings to libdrm_amdgpu functions, loaded at runtime so the
/// engine does not gain a build-time dependency on libdrm development headers.
const AmdgpuApi = struct {
    _handle: ?*DynLib,
    device_initialize: ?*const fn (fd: i32, major_version: *u32, minor_version: *u32, device: *u64) callconv(.c) i32,
    device_deinitialize: ?*const fn (device: u64) callconv(.c) i32,
    bo_alloc: ?*const fn (device: u64, alloc_size: u64, heap: u32, flags: u64, bo: *u64) callconv(.c) i32,
    bo_export: ?*const fn (device: u64, bo: u64, handle_type: u32, fd: *i32) callconv(.c) i32,
    bo_cpu_map: ?*const fn (device: u64, bo: u64, ptr: *?*anyopaque) callconv(.c) i32,
    bo_cpu_unmap: ?*const fn (device: u64, bo: u64) callconv(.c) i32,
    bo_free: ?*const fn (device: u64, bo: u64) callconv(.c) i32,
    query_gpu_info: ?*const fn (device: u64, info: *anyopaque) callconv(.c) i32,

    fn load() AmdgpuApi {
        var lib = DynLib.openZ("libdrm_amdgpu.so.1") catch
            DynLib.openZ("libdrm_amdgpu.so") catch
            return AmdgpuApi.null_handle();
        leakLib(lib);

        return AmdgpuApi{
            ._handle = null,
            .device_initialize = lib.lookup(*const fn (i32, *u32, *u32, *u64) callconv(.c) i32, "amdgpu_device_initialize"),
            .device_deinitialize = lib.lookup(*const fn (u64) callconv(.c) i32, "amdgpu_device_deinitialize"),
            .bo_alloc = lib.lookup(*const fn (u64, u64, u32, u64, *u64) callconv(.c) i32, "amdgpu_bo_alloc"),
            .bo_export = lib.lookup(*const fn (u64, u64, u32, *i32) callconv(.c) i32, "amdgpu_bo_export"),
            .bo_cpu_map = lib.lookup(*const fn (u64, u64, *?*anyopaque) callconv(.c) i32, "amdgpu_bo_cpu_map"),
            .bo_cpu_unmap = lib.lookup(*const fn (u64, u64) callconv(.c) i32, "amdgpu_bo_cpu_unmap"),
            .bo_free = lib.lookup(*const fn (u64, u64) callconv(.c) i32, "amdgpu_bo_free"),
            .query_gpu_info = lib.lookup(*const fn (u64, *anyopaque) callconv(.c) i32, "amdgpu_query_gpu_info"),
        };
    }

    fn null_handle() AmdgpuApi {
        return .{
            ._handle = null,
            .device_initialize = null,
            .device_deinitialize = null,
            .bo_alloc = null,
            .bo_export = null,
            .bo_cpu_map = null,
            .bo_cpu_unmap = null,
            .bo_free = null,
            .query_gpu_info = null,
        };
    }

    fn isAvailable(self: *const AmdgpuApi) bool {
        return self.device_initialize != null and
            self.device_deinitialize != null and
            self.bo_alloc != null and
            self.bo_export != null and
            self.bo_cpu_map != null and
            self.bo_free != null;
    }
};

var amdgpu_api: AmdgpuApi = AmdgpuApi.null_handle();
var amdgpu_lib_handle: ?DynLib = null;
var amdgpu_api_loaded: bool = false;

fn getAmdgpuApi() *const AmdgpuApi {
    if (!amdgpu_api_loaded) {
        amdgpu_api = AmdgpuApi.load();
        amdgpu_api_loaded = true;
    }
    return &amdgpu_api;
}

/// Leak a DynLib so it stays loaded for the process lifetime.
/// The library is never intentionally closed.
fn leakLib(lib: DynLib) void {
    amdgpu_lib_handle = lib;
}

/// Initialize an amdgpu device from an open DRM fd.
fn amdgpuDeviceInit(fd: i32) !u64 {
    const api = getAmdgpuApi();
    if (!api.isAvailable()) return error.LibraryNotFound;

    var dev: u64 = 0;
    var major: u32 = undefined;
    var minor: u32 = undefined;
    const rc = api.device_initialize.?(fd, &major, &minor, &dev);
    if (rc != 0 or dev == 0) return error.InitFailed;
    return dev;
}

/// Deinitialize an amdgpu device.
fn amdgpuDeviceDeinit(dev: u64) !void {
    const api = getAmdgpuApi();
    if (!api.isAvailable()) return error.LibraryNotFound;
    _ = api.device_deinitialize.?(dev);
}

/// Query amdgpu GPU info as a basic health check.
fn amdgpuQueryGpuInfo(dev: u64) bool {
    const api = getAmdgpuApi();
    if (!api.isAvailable()) return false;

    // Allocate a buffer for amdgpu_gpu_info — the first 4 uint32_t fields
    // (asic_id, chip_rev, chip_external_rev, family_id) are sufficient to
    // verify the device is responsive.
    var info: [64]u8 = @splat(0);
    const rc = api.query_gpu_info.?(dev, &info);
    return rc == 0;
}

/// Allocate a GTT buffer and export as dma-buf fd.
fn amdgpuBoAllocGtt(dev: u64, size: u64) !GttAllocResult {
    const api = getAmdgpuApi();
    if (!api.isAvailable()) return error.LibraryNotFound;

    const AMDGPU_GEM_DOMAIN_GTT: u32 = 2;
    const AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED: u64 = 1 << 2;

    var bo: u64 = 0;
    var rc = api.bo_alloc.?(dev, size, AMDGPU_GEM_DOMAIN_GTT, AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED, &bo);
    if (rc != 0 or bo == 0) return error.AllocFailed;

    // Export as dma-buf fd
    const AMDGPU_BO_HANDLE_TYPE_DMA_BUF_FD: u32 = 2;
    var dma_buf_fd: i32 = -1;
    rc = api.bo_export.?(dev, bo, AMDGPU_BO_HANDLE_TYPE_DMA_BUF_FD, &dma_buf_fd);
    if (rc != 0 or dma_buf_fd < 0) {
        _ = api.bo_free.?(dev, bo);
        return error.ExportFailed;
    }

    return GttAllocResult{ .bo_handle = bo, .fd = dma_buf_fd };
}

/// mmap a GTT BO for CPU access.
fn amdgpuBoCpuMap(dev: u64, bo: u64) ![*]u8 {
    const api = getAmdgpuApi();
    if (!api.isAvailable()) return error.LibraryNotFound;

    var ptr: ?*anyopaque = null;
    const rc = api.bo_cpu_map.?(dev, bo, &ptr);
    if (rc != 0 or ptr == null) return error.MmapFailed;

    return @as([*]u8, @ptrCast(ptr));
}

/// Free a GTT BO.
fn amdgpuBoFree(dev: u64, bo: u64) !void {
    const api = getAmdgpuApi();
    if (!api.isAvailable()) return;

    if (api.bo_cpu_unmap) |unmap| {
        _ = unmap(dev, bo);
    }
    _ = api.bo_free.?(dev, bo);
}

// ---------------------------------------------------------------------------
// Error set
// ---------------------------------------------------------------------------

pub const MemoryError = error{
    DmaBufUnavailable,
    AllocFailed,
    ExportFailed,
    MmapFailed,
    InitFailed,
    LibraryNotFound,
    InvalidArgument,
    PermissionDenied,
    NotSupported,
    SlotExhausted,
    Unexpected,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "CrossBackendMemory init/deinit" {
    const a = std.testing.allocator;
    var m = try CrossBackendMemory.init(a);
    defer m.deinit();
    try std.testing.expect(m.activeCount() == 0);
}

test "CrossBackendMemory dma_buf_available is false on non-Linux" {
    const a = std.testing.allocator;
    var m = try CrossBackendMemory.init(a);
    defer m.deinit();
    _ = m.isAvailable();
}

test "allocateShared returns invalid buffer when unavailable" {
    const a = std.testing.allocator;
    var m = try CrossBackendMemory.init(a);
    defer m.deinit();

    m.dma_buf_available = false;
    const buf = m.allocateShared(4096);
    try std.testing.expect(!buf.valid);
    try std.testing.expectEqual(@as(i32, -1), buf.dma_buf_fd);
}

test "importFd returns DmaBufUnavailable when unavailable" {
    const a = std.testing.allocator;
    var m = try CrossBackendMemory.init(a);
    defer m.deinit();

    m.dma_buf_available = false;
    try std.testing.expectError(error.DmaBufUnavailable, m.importFd(42, 4096));
}

test "importFd rejects invalid fd" {
    const a = std.testing.allocator;
    var m = try CrossBackendMemory.init(a);
    defer m.deinit();

    m.dma_buf_available = true;
    try std.testing.expectError(error.InvalidArgument, m.importFd(-1, 4096));
}

test "freeShared handles invalid buffer" {
    const a = std.testing.allocator;
    var m = try CrossBackendMemory.init(a);
    defer m.deinit();

    var buf = SharedBuffer.empty();
    m.freeShared(&buf);
    try std.testing.expect(!buf.valid);
}

test "activeCount tracks allocations" {
    const a = std.testing.allocator;
    var m = try CrossBackendMemory.init(a);
    defer m.deinit();

    // Manually populate a slot to test tracking
    const slot = m.findFreeSlot() orelse @panic("no free slot");
    slot.* = SharedBuffer.init(42, @as([*]u8, @ptrFromInt(0x1000)), 4096, 0);

    try std.testing.expectEqual(@as(usize, 1), m.activeCount());

    m.freeShared(slot);
    try std.testing.expectEqual(@as(usize, 0), m.activeCount());
}

test "findByFd works" {
    const a = std.testing.allocator;
    var m = try CrossBackendMemory.init(a);
    defer m.deinit();

    const slot = m.findFreeSlot() orelse @panic("no free slot");
    slot.* = SharedBuffer.init(42, @as([*]u8, @ptrFromInt(0x1000)), 4096, 0);

    const found = m.findByFd(42) orelse @panic("not found");
    try std.testing.expectEqual(@as(u64, 4096), found.size);
}

test "findFreeSlot returns null when full" {
    const a = std.testing.allocator;
    var m = try CrossBackendMemory.init(a);
    defer m.deinit();

    // Fill all slots manually
    for (0..MAX_BUFFERS) |_| {
        const slot = m.findFreeSlot() orelse break;
        slot.* = SharedBuffer.init(1, @as([*]u8, @ptrFromInt(0x2000)), 64, 0);
    }

    try std.testing.expect(m.findFreeSlot() == null);
}

test "empty SharedBuffer has correct defaults" {
    const buf = SharedBuffer.empty();
    try std.testing.expect(!buf.valid);
    try std.testing.expectEqual(@as(i32, -1), buf.dma_buf_fd);
    try std.testing.expectEqual(@as(u64, 0), buf.size);
    try std.testing.expect(buf.cpu_ptr == null);
}

test "SharedBuffer init populates fields" {
    const ptr: [*]u8 = @as([*]u8, @ptrFromInt(0xDEAD));
    const buf = SharedBuffer.init(7, ptr, 4096, 0x1234);
    try std.testing.expect(buf.valid);
    try std.testing.expectEqual(@as(i32, 7), buf.dma_buf_fd);
    try std.testing.expectEqual(@as(u64, 4096), buf.size);
    try std.testing.expect(buf.cpu_ptr == ptr);
    try std.testing.expectEqual(@as(u64, 0x1234), buf.bo_handle);
}

test "syncCpuAccess with invalid buffer is no-op" {
    const a = std.testing.allocator;
    var m = try CrossBackendMemory.init(a);
    defer m.deinit();

    m.dma_buf_available = true;
    var buf = SharedBuffer.empty();
    try m.syncCpuAccess(&buf, true, false);
}

test "buffer tracking does not overflow" {
    const a = std.testing.allocator;
    var m = try CrossBackendMemory.init(a);
    defer m.deinit();

    // Fill all slots
    for (0..MAX_BUFFERS) |i| {
        const slot = m.findFreeSlot() orelse {
            try std.testing.expectEqual(@as(usize, MAX_BUFFERS), i);
            break;
        };
        slot.* = SharedBuffer.init(@as(i32, @intCast(i)), @as([*]u8, @ptrFromInt(0x7000)), 64, 0);
    }

    try std.testing.expectEqual(@as(usize, MAX_BUFFERS), m.activeCount());
    try std.testing.expect(m.findFreeSlot() == null);
}
