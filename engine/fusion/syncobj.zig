//! DRM syncobj → VkSemaphore bridge for NPU↔GPU ordering.
//!
//! The amdxdna NPU driver (accel, /dev/accel/accel0) supports DRM syncobjs.
//! When an NPU job completes, its associated dma_fence signals the syncobj.
//! By exporting the syncobj as a sync_file fd and importing it into Vulkan
//! as a VkSemaphore, the GPU can wait directly on NPU completion — no CPU
//! polling needed.
//!
//! ## Usage
//! ```zig
//! var sync = try NpuGpuSync.init(allocator, vk_device, vk_queue);
//! defer sync.deinit();
//!
//! // After NPU writes to the dma-buf, before GPU reads:
//! const sem = sync.makeWaitSemaphore() catch null; // sync_file → VkSemaphore
//! // Submit GPU compute with wait on sem
//! // sync.signal() advances the timeline point
//! ```
//!
//! ## Architecture
//! ```
//! amdxdna.ko                    amdgpu.ko                  Vulkan
//! ┌─────────────────┐          ┌──────────────────┐     ┌──────────┐
//! │ hwctx syncobj   │          │ sync_file fd     │     │ VkSema   │
//! │ (dma_fence sig) │─sync_fd─▶│ import as sema   │────▶│ wait     │
//! └─────────────────┘          └──────────────────┘     └──────────┘
//! ```
//!
//! ## References
//! - amdxdna UAPI: include/uapi/drm/amdxdna_accel.h
//! - DRM syncobj: drivers/gpu/drm/drm_syncobj.c
//! - VK_KHR_external_semaphore_fd: Vulkan semaphore import from sync_file
//!
//! @section Fused Engine

const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;

const log = std.log.scoped(.npu_gpu_syncobj);

const is_linux = builtin.target.os.tag == .linux;

// ════════════════════════════════════════════════════════════════════
// amdxdna accel device path
// ════════════════════════════════════════════════════════════════════

/// Default amdxdna acceler device node.
const AMDXDNA_ACCEL_PATH: [:0]const u8 = "/dev/accel/accel0";

/// DRM_ACCEL major number (261 on Linux 6.10+).
const DRM_ACCEL_MAJOR: u16 = 261;

// ════════════════════════════════════════════════════════════════════
// DRM ioctl constants (from drm.h / drm_mode.h)
// ════════════════════════════════════════════════════════════════════

/// DRM ioctl base type.
const DRM_IOCTL_BASE: u8 = 'd';

/// Build a DRM ioctl request code.
/// Equivalent to DRM_IOWR(nr, type).
fn drmIoctlReq(nr: u8, comptime T: type) u32 {
    const size = @sizeOf(T);
    // _IOC(_IOC_READ|_IOC_WRITE, 'd', nr, size)
    return (@as(u32, 3) << 30) | // read+write direction
        (@as(u32, 'd') << 8) |
        (@as(u32, nr) << 0) |
        (@as(u32, size) << 16);
}

fn drmIoctlReqW(nr: u8, comptime T: type) u32 {
    const size = @sizeOf(T);
    return (@as(u32, 1) << 30) | // write direction
        (@as(u32, 'd') << 8) |
        (@as(u32, nr) << 0) |
        (@as(u32, size) << 16);
}

fn drmIoctlReqR(nr: u8, comptime T: type) u32 {
    const size = @sizeOf(T);
    return (@as(u32, 2) << 30) | // read direction
        (@as(u32, 'd') << 8) |
        (@as(u32, nr) << 0) |
        (@as(u32, size) << 16);
}

/// DRM_IOCTL_SYNCOBJ_CREATE = DRM_IOWR(0xBF, struct drm_syncobj_create)
const DRM_IOCTL_SYNCOBJ_CREATE: u32 = drmIoctlReq(0xBF, DrmSyncobjCreate);

/// DRM_IOCTL_SYNCOBJ_DESTROY = DRM_IOWR(0xC0, struct drm_syncobj_destroy)
const DRM_IOCTL_SYNCOBJ_DESTROY: u32 = drmIoctlReq(0xC0, DrmSyncobjDestroy);

/// DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD = DRM_IOWR(0xC1, struct drm_syncobj_handle)
const DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD: u32 = drmIoctlReq(0xC1, DrmSyncobjHandle);

/// DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE = DRM_IOWR(0xC2, struct drm_syncobj_handle)
const DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE: u32 = drmIoctlReq(0xC2, DrmSyncobjHandle);

/// DRM_IOCTL_SYNCOBJ_SIGNAL = DRM_IOWR(0xC4, struct drm_syncobj_array)
const DRM_IOCTL_SYNCOBJ_SIGNAL: u32 = drmIoctlReqW(0xC4, DrmSyncobjArray);

/// DRM_IOCTL_SYNCOBJ_WAIT = DRM_IOWR(0xC6, struct drm_syncobj_array)
const DRM_IOCTL_SYNCOBJ_WAIT: u32 = drmIoctlReq(0xC6, DrmSyncobjArray);

/// DRM_IOCTL_SYNCOBJ_RESET = DRM_IOWR(0xC7, struct drm_syncobj_array)
const DRM_IOCTL_SYNCOBJ_RESET: u32 = drmIoctlReqW(0xC7, DrmSyncobjArray);

/// DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL = DRM_IOWR(0xCB, struct drm_syncobj_timeline_array)
const DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL: u32 = drmIoctlReqW(0xCB, DrmSyncobjTimelineArray);

/// DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT = DRM_IOWR(0xCC, struct drm_syncobj_timeline_array)
const DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT: u32 = drmIoctlReq(0xCC, DrmSyncobjTimelineArray);

/// DRM_IOCTL_SYNCOBJ_QUERY = DRM_IOWR(0xCD, struct drm_syncobj_timeline_array)
const DRM_IOCTL_SYNCOBJ_QUERY: u32 = drmIoctlReq(0xCD, DrmSyncobjTimelineArray);

// ════════════════════════════════════════════════════════════════════
// DRM ioctl argument structures (from drm.h)
// ════════════════════════════════════════════════════════════════════

/// DRM_CLOEXEC flag for DRM_IOCTL_SYNCOBJ_CREATE.
const DRM_SYNCOBJ_CREATE_SIGNALED: u32 = 1 << 0;

/// Flag for DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD: export as sync_file (not opaque fd).
/// Required for Vulkan import via VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT.
const DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE: u32 = 1 << 0;

const DrmSyncobjCreate = extern struct {
    handle: u32, // out
    flags: u32, // in
};

const DrmSyncobjDestroy = extern struct {
    handle: u32, // in
    padding: u32,
};

const DrmSyncobjHandle = extern struct {
    handle: u32, // in/out
    fd: i32, // in/out
    flags: u32, // in
    padding: u32,
};

const DrmSyncobjArray = extern struct {
    handles: u64, // pointer to array of u32 handles
    count: u32, // number of handles
    flags: u32, // wait flags (DRM_SYNCOBJ_WAIT_FLAGS_*)
    first_signaled: u32, // out: first signaled handle index
    pad: u32,
};

const DrmSyncobjTimelineArray = extern struct {
    handles: u64, // pointer to array of u32 handles
    points: u64, // pointer to array of u64 points (timeline values)
    count: u32, // number of handles
    flags: u32,
    first_signaled: u32,
    pad: u32,
};

/// DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL = 0x01
/// DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT = 0x02
/// DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE = 0x04

// ════════════════════════════════════════════════════════════════════
// NpuSyncobj
// ════════════════════════════════════════════════════════════════════

/// Manages a DRM syncobj on the amdxdna accel device, exported as sync_file
/// fd and importable into Vulkan as a VkSemaphore.
///
/// Current implementation uses a CPU-mediated signal (syncobj is signaled
/// after XRT confirms NPU job completion). In the future, the amdxdna
/// driver's built-in hwctx syncobj can directly signal NPU completion
/// without CPU involvement.
pub const NpuSyncobj = struct {
    /// DRM fd for the amdxdna accel device.
    drm_fd: i32,
    /// DRM syncobj handle (valid on drm_fd).
    syncobj_handle: u32,
    /// Current timeline point (monotonically increasing).
    timeline_point: u64,
    /// Whether this instance is initialized and valid.
    valid: bool,

    /// Initialize the syncobj.
    /// Opens the amdxdna accel device and creates a DRM syncobj.
    pub fn init() !NpuSyncobj {
        if (!is_linux) return error.NotSupported;

        // Try to open the amdxdna accel device.
        // If not found, scan /dev/accel/ for any accel device.
        const fd = openAccelDevice() catch |err| {
            log.warn("amdxdna accel device not found ({s}) — syncobj unavailable", .{@errorName(err)});
            return err;
        };

        // Create a binary DRM syncobj (not signaled initially).
        var create = DrmSyncobjCreate{
            .handle = 0,
            .flags = 0, // not pre-signaled
        };
        const rc = linux.ioctl(fd, DRM_IOCTL_SYNCOBJ_CREATE, @intFromPtr(&create));
        if (rc != 0) {
            const errno = linux.errno(rc);
            log.warn("DRM_IOCTL_SYNCOBJ_CREATE failed: errno={}", .{@intFromEnum(errno)});
            _ = linux.close(fd);
            return error.SyncobjCreateFailed;
        }

        log.info("NPU syncobj created: fd={d}, syncobj_handle={d}", .{ fd, create.handle });

        return NpuSyncobj{
            .drm_fd = fd,
            .syncobj_handle = create.handle,
            .timeline_point = 0,
            .valid = true,
        };
    }

    /// Export the syncobj as a sync_file fd.
    /// The returned fd can be imported into Vulkan via
    /// VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT.
    pub fn exportSyncFile(self: *NpuSyncobj) !i32 {
        if (!self.valid) return error.InvalidState;

        var h2f = DrmSyncobjHandle{
            .handle = self.syncobj_handle,
            .fd = -1,
            .flags = DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE, // sync_file, not opaque
            .padding = 0,
        };
        const rc = linux.ioctl(self.drm_fd, DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD, @intFromPtr(&h2f));
        if (rc != 0) {
            const errno = linux.errno(rc);
            log.warn("SYNCOBJ_HANDLE_TO_FD failed: errno={}", .{@intFromEnum(errno)});
            return error.ExportFailed;
        }
        log.debug("syncobj {d} exported as sync_file fd {d}", .{ self.syncobj_handle, h2f.fd });
        return h2f.fd;
    }

    /// Import a sync_file fd into this syncobj.
    /// This creates a new handle on our DRM fd that references the same
    /// underlying dma_fence as the sync_file.
    pub fn importSyncFile(self: *NpuSyncobj, sync_fd: i32) !void {
        if (!self.valid) return error.InvalidState;
        if (sync_fd < 0) return error.InvalidArgument;

        var f2h = DrmSyncobjHandle{
            .handle = 0,
            .fd = sync_fd,
            .flags = 0,
            .padding = 0,
        };
        const rc = linux.ioctl(self.drm_fd, DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE, @intFromPtr(&f2h));
        if (rc != 0) {
            const errno = linux.errno(rc);
            log.warn("SYNCOBJ_FD_TO_HANDLE failed: errno={}", .{@intFromEnum(errno)});
            return error.ImportFailed;
        }
        self.syncobj_handle = f2h.handle;
        log.debug("sync_file fd {d} imported as syncobj handle {d}", .{ sync_fd, f2h.handle });
    }

    /// Signal the syncobj (mark as completed).
    /// Call after NPU job has confirmed completion (e.g., xrtRunWait returns).
    /// Advances the timeline point.
    pub fn signal(self: *NpuSyncobj) !void {
        if (!self.valid) return error.InvalidState;

        const handle = self.syncobj_handle;
        var arr = DrmSyncobjArray{
            .handles = @intFromPtr(&handle),
            .count = 1,
            .flags = 0,
            .first_signaled = 0,
            .pad = 0,
        };
        const rc = linux.ioctl(self.drm_fd, DRM_IOCTL_SYNCOBJ_SIGNAL, @intFromPtr(&arr));
        if (rc != 0) {
            const errno = linux.errno(rc);
            log.warn("SYNCOBJ_SIGNAL failed: errno={}", .{@intFromEnum(errno)});
            return error.SignalFailed;
        }
        self.timeline_point += 1;
    }

    /// Reset the syncobj to unsignaled.
    pub fn reset(self: *NpuSyncobj) !void {
        if (!self.valid) return error.InvalidState;

        const handle = self.syncobj_handle;
        var arr = DrmSyncobjArray{
            .handles = @intFromPtr(&handle),
            .count = 1,
            .flags = 0,
            .first_signaled = 0,
            .pad = 0,
        };
        const rc = linux.ioctl(self.drm_fd, DRM_IOCTL_SYNCOBJ_RESET, @intFromPtr(&arr));
        if (rc != 0) {
            const errno = linux.errno(rc);
            log.warn("SYNCOBJ_RESET failed: errno={}", .{@intFromEnum(errno)});
            return error.ResetFailed;
        }
    }

    /// Wait for the syncobj to be signaled (CPU-side, for diagnostics).
    /// timeout_ms: timeout in milliseconds. 0 = infinite wait.
    pub fn wait(self: *NpuSyncobj, timeout_ms: u64) !void {
        _ = timeout_ms; // TODO: not yet threaded through to the ioctl below
        if (!self.valid) return error.InvalidState;

        const handle = self.syncobj_handle;
        var arr = DrmSyncobjArray{
            .handles = @intFromPtr(&handle),
            .count = 1,
            .flags = 0, // wait for ANY (not all), no timeout flags
            .first_signaled = 0,
            .pad = 0,
        };
        const rc = linux.ioctl(self.drm_fd, DRM_IOCTL_SYNCOBJ_WAIT, @intFromPtr(&arr));
        if (rc != 0) {
            const errno = linux.errno(rc);
            log.warn("SYNCOBJ_WAIT failed: errno={}", .{@intFromEnum(errno)});
            return switch (errno) {
                .AGAIN => error.Timeout,
                .INTR => error.Interrupted,
                else => error.WaitFailed,
            };
        }
    }

    /// Create a VkSemaphore from our syncobj's current state.
    /// The semaphore will be signaled when the NPU completes.
    ///
    /// vk_device: VkDevice handle
    /// vk_queue_family: queue family index for the semaphore
    pub fn makeVkSemaphore(self: *NpuSyncobj, vk_device: u64, vk_queue_family: u32) !u64 {
        _ = vk_device;
        _ = vk_queue_family;
        if (!self.valid) return error.InvalidState;

        // Export syncobj as sync_file fd
        const sync_fd = try self.exportSyncFile();

        // The caller must import this fd into Vulkan via
        // VkImportSemaphoreFdInfoKHR with VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT
        // The semaphore fd is returned as the "handle" for callers who want
        // to do the import themselves with raw Vulkan bindings.
        return @as(u64, @intCast(@as(u64, @bitCast(@as(i64, sync_fd)))));
    }

    /// Close the DRM fd and release resources.
    pub fn deinit(self: *NpuSyncobj) void {
        if (!is_linux) return;
        if (self.drm_fd >= 0) {
            // Destroy syncobj
            if (self.syncobj_handle != 0) {
                var destroy = DrmSyncobjDestroy{
                    .handle = self.syncobj_handle,
                    .padding = 0,
                };
                _ = linux.ioctl(self.drm_fd, DRM_IOCTL_SYNCOBJ_DESTROY, @intFromPtr(&destroy));
            }
            _ = linux.close(self.drm_fd);
        }
        self.* = undefined;
    }
};

// ════════════════════════════════════════════════════════════════════
// Internal helpers
// ════════════════════════════════════════════════════════════════════

/// Open the amdxdna accel device.
/// Tries /dev/accel/accel0 first, then scans /dev/accel/ for any device.
fn openAccelDevice() !i32 {
    // Try the default path
    const fd0 = tryOpen(AMDXDNA_ACCEL_PATH, 2); // O_RDWR
    if (fd0 >= 0) {
        log.debug("opened {s} fd={d}", .{ AMDXDNA_ACCEL_PATH, fd0 });
        return fd0;
    }

    // Scan /dev/accel/ directory
    const dir_fd = linux.openat(linux.AT.FDCWD, "/dev/accel", linux.O{ .DIRECTORY = true }, 0);
    if (linux.errno(dir_fd) != .SUCCESS) {
        return error.AccelDeviceNotFound;
    }
    defer _ = linux.close(@as(i32, @intCast(dir_fd)));

    var buf: [4096]u8 = undefined;
    var iter = linux.fd_iterator(@as(i32, @intCast(dir_fd)), &buf);
    while (iter.next()) |entry| {
        const name = std.mem.sliceTo(&entry.name, 0);
        if (name.len == 0) continue;

        // Check if it looks like an accel device (accel0, accel1, ...)
        if (std.ascii.startsWith(u8, name, "accel")) {
            var path_buf: [64]u8 = undefined;
            const path = std.fmt.bufPrintZ(&path_buf, "/dev/accel/{s}", .{name}) catch continue;
            const fd = tryOpen(path, 2);
            if (fd >= 0) {
                log.debug("opened {s} fd={d}", .{ path, fd });
                return fd;
            }
        }
    }

    return error.AccelDeviceNotFound;
}

fn tryOpen(path: [:0]const u8, _flags: u32) i32 {
    _ = _flags;
    const fd = linux.openat(linux.AT.FDCWD, @intFromPtr(path.ptr), @as(u32, @bitCast(linux.O{ .RDWR = true, .CLOEXEC = true })), 0);
    if (linux.errno(fd) == .SUCCESS) {
        return @as(i32, @intCast(fd));
    }
    return -1;
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

test "NpuSyncobj init/deinit no-op on non-Linux" {
    if (is_linux) return error.SkipZigTest;
    const sync = NpuSyncobj.init();
    try std.testing.expectError(error.NotSupported, sync);
}

test "NpuSyncobj exportSyncFile returns error on invalid state" {
    var sync = NpuSyncobj{
        .drm_fd = -1,
        .syncobj_handle = 0,
        .timeline_point = 0,
        .valid = false,
    };
    try std.testing.expectError(error.InvalidState, sync.exportSyncFile());
    try std.testing.expectError(error.InvalidState, sync.signal());
    try std.testing.expectError(error.InvalidState, sync.reset());
    try std.testing.expectError(error.InvalidState, sync.wait(100));
}
