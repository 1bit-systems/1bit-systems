//! Zig FFI bindings to the XRT C API (libxrt_coreutil).
//! Provides thin safe wrappers around XRT device, buffer, kernel, and run handles.
const std = @import("std");

// ============================================================
// XRT C API constants
// ============================================================

pub const XCL_BO_FLAGS_CACHEABLE: u64 = 0;
pub const XRT_BO_FLAGS_HOST_ONLY: u64 = 2;

pub const XCL_BO_SYNC_BO_TO_DEVICE: u32 = 0;
pub const XCL_BO_SYNC_BO_FROM_DEVICE: u32 = 1;

// ============================================================
// Opaque handle types (mirror C typedefs)
// ============================================================

pub const Device = opaque {};
pub const BufferObject = opaque {};
pub const KernelHandle = opaque {};
pub const RunHandle = opaque {};
pub const HwContext = opaque {};

/// XRT UUID — 16 bytes, matches struct xrt_uuid in C headers.
pub const Uuid = struct {
    data: [16]u8 = [_]u8{0} ** 16,
};

// ============================================================
// Extern declarations from libxrt_coreutil
// ============================================================

extern "xrt_coreutil" fn xrtDeviceOpen(index: u32) callconv(.c) ?*Device;
extern "xrt_coreutil" fn xrtDeviceClose(handle: ?*Device) callconv(.c) void;
extern "xrt_coreutil" fn xrtDeviceLoadXclbin(handle: ?*Device, xclbin_file: [*:0]const u8) callconv(.c) c_int;
extern "xrt_coreutil" fn xrtDeviceLoadXclbinFile(handle: ?*Device, xclbin_file: [*:0]const u8) callconv(.c) c_int;
extern "xrt_coreutil" fn xrtDeviceLoadXclbinHandle(handle: ?*Device, xclbin: ?*const anyopaque) callconv(.c) c_int;

extern "xrt_coreutil" fn xrtXclbinLock(handle: ?*Device, uuid: ?*const Uuid) callconv(.c) c_int;
extern "xrt_coreutil" fn xrtXclbinAlloc() callconv(.c) ?*anyopaque;
extern "xrt_coreutil" fn xrtXclbinFree(xclbin: ?*anyopaque) callconv(.c) void;
extern "xrt_coreutil" fn xrtXclbinGetUUID(xclbin: ?*const anyopaque, uuid: ?*Uuid) callconv(.c) c_int;

extern "xrt_coreutil" fn xrtHwContextCreate(device: ?*Device, uuid: ?*const Uuid) callconv(.c) ?*HwContext;
extern "xrt_coreutil" fn xrtHwContextDestroy(hwctx: ?*HwContext) callconv(.c) void;

extern "xrt_coreutil" fn xrtKernelOpen(device: ?*Device, uuid: ?*const Uuid, name: [*:0]const u8) callconv(.c) ?*KernelHandle;
extern "xrt_coreutil" fn xrtKernelClose(kernel: ?*KernelHandle) callconv(.c) void;

extern "xrt_coreutil" fn xrtKernelRun(
    kernel: ?*KernelHandle,
    nargs: c_int,
    arg1: ?*BufferObject,
    arg2: c_int,
    arg3: ?*BufferObject,
    arg4: ?*BufferObject,
    arg5: ?*BufferObject,
) callconv(.c) ?*RunHandle;

extern "xrt_coreutil" fn xrtRunClose(run: ?*RunHandle) callconv(.c) void;
extern "xrt_coreutil" fn xrtRunWait(run: ?*RunHandle) callconv(.c) c_int;

extern "xrt_coreutil" fn xrtBOAlloc(
    device: ?*Device,
    size: usize,
    flags: u64,
    group: u32,
) callconv(.c) ?*BufferObject;

extern "xrt_coreutil" fn xrtBOFree(bo: ?*BufferObject) callconv(.c) void;
extern "xrt_coreutil" fn xrtBOMap(bo: ?*BufferObject) callconv(.c) ?*anyopaque;
extern "xrt_coreutil" fn xrtBOSync(bo: ?*BufferObject, dir: u32, offset: u64, size: u64) callconv(.c) c_int;

// ============================================================
// Safe wrapper: Device
// ============================================================

pub const XrtDevice = struct {
    handle: ?*Device,

    /// Open XRT device by index (usually 0).
    pub fn open(index: u32) !XrtDevice {
        const h = xrtDeviceOpen(index);
        if (h == null) return error.XrtDeviceOpenFailed;
        return XrtDevice{ .handle = h };
    }

    /// Load an xclbin from a file path.
    /// Returns the xclbin's UUID on success.
    pub fn loadXclbin(self: XrtDevice, path: []const u8) !Uuid {
        // Create xclbin object from file
        const alloc = xrtXclbinAlloc();
        if (alloc == null) return error.XrtXclbinAllocFailed;
        defer xrtXclbinFree(alloc);

        const compat = @import("compat.zig");
        const file = compat.File.openAbsolute(path) catch |err| {
            std.log.err("Cannot open xclbin: {s}", .{path});
            return err;
        };
        defer file.close();

        const file_size = try file.getEndPos();
        const buf = try std.heap.page_allocator.alloc(u8, @intCast(file_size));
        defer std.heap.page_allocator.free(buf);
        _ = try file.readAll(buf);

        // Load from raw bytes
        const rc = xrtDeviceLoadXclbinHandle(self.handle, @ptrCast(buf.ptr));
        if (rc != 0) return error.XrtLoadXclbinFailed;

        // Get UUID
        var uuid: Uuid = .{};
        const rc2 = xrtXclbinGetUUID(alloc, &uuid);
        if (rc2 != 0) return error.XrtGetUUIDFailed;

        return uuid;
    }

    /// Allocate a buffer object.
    pub fn allocBO(self: XrtDevice, size: usize, flags: u64, group: u32) !*BufferObject {
        const bo = xrtBOAlloc(self.handle, size, flags, group);
        if (bo == null) return error.XrtBOAllocFailed;
        return bo.?;
    }

    /// Create a hardware context for the given UUID.
    pub fn createHwContext(self: XrtDevice, uuid: *const Uuid) !*HwContext {
        const ctx = xrtHwContextCreate(self.handle, uuid);
        if (ctx == null) return error.XrtHwContextCreateFailed;
        return ctx.?;
    }

    /// Open a kernel by name.
    pub fn createKernel(self: XrtDevice, uuid: *const Uuid, name: [:0]const u8) !*KernelHandle {
        const k = xrtKernelOpen(self.handle, uuid, name.ptr);
        if (k == null) return error.XrtKernelOpenFailed;
        return k.?;
    }

    /// Close the device.
    pub fn close(self: XrtDevice) void {
        if (self.handle) |h| xrtDeviceClose(h);
    }
};

// ============================================================
// Safe wrapper: BufferObject
// ============================================================

pub const XrtBuffer = struct {
    handle: ?*BufferObject,

    pub fn init(handle: ?*BufferObject) XrtBuffer {
        return .{ .handle = handle };
    }

    /// Map the buffer into host address space.
    pub fn map(self: XrtBuffer, size: usize) ![]u8 {
        const ptr = xrtBOMap(self.handle);
        if (ptr == null) return error.XrtBOMapFailed;
        return @as([*]u8, @ptrCast(ptr.?))[0..size];
    }

    /// Sync buffer between device and host.
    pub fn sync(self: XrtBuffer, dir: u32, offset: u64, size: u64) !void {
        const rc = xrtBOSync(self.handle, dir, offset, size);
        if (rc != 0) return error.XrtBOSyncFailed;
    }

    /// Free the buffer.
    pub fn free(self: XrtBuffer) void {
        if (self.handle) |h| xrtBOFree(h);
    }
};

// ============================================================
// Safe wrapper: KernelHandle
// ============================================================

pub const XrtKernel = struct {
    handle: ?*KernelHandle,

    pub fn init(handle: ?*KernelHandle) XrtKernel {
        return .{ .handle = handle };
    }

    /// Run the kernel with the standard 5-argument signature used by NPU xclbins:
    /// run(3, instr_bo, instr_count, act_bo, weight_bo, out_bo)
    pub fn run(
        self: XrtKernel,
        instr: *BufferObject,
        instr_count: i32,
        act: *BufferObject,
        weight: *BufferObject,
        out: *BufferObject,
    ) !*RunHandle {
        const r = xrtKernelRun(self.handle, 3, instr, instr_count, act, weight, out);
        if (r == null) return error.XrtKernelRunFailed;
        return r.?;
    }

    /// Close the kernel.
    pub fn close(self: XrtKernel) void {
        if (self.handle) |h| xrtKernelClose(h);
    }
};

// ============================================================
// Safe wrapper: RunHandle
// ============================================================

pub const XrtRun = struct {
    handle: ?*RunHandle,

    pub fn init(handle: ?*RunHandle) XrtRun {
        return .{ .handle = handle };
    }

    /// Wait for kernel execution to complete.
    pub fn wait(self: XrtRun) !void {
        const rc = xrtRunWait(self.handle);
        if (rc != 0) return error.XrtRunWaitFailed;
    }

    /// Close the run handle.
    pub fn close(self: XrtRun) void {
        if (self.handle) |h| xrtRunClose(h);
    }
};

// ============================================================
// Safe wrapper: HwContext
// ============================================================

pub const XrtHwContext = struct {
    handle: ?*HwContext,

    pub fn init(handle: ?*HwContext) XrtHwContext {
        return .{ .handle = handle };
    }

    pub fn destroy(self: XrtHwContext) void {
        if (self.handle) |h| xrtHwContextDestroy(h);
    }
};
