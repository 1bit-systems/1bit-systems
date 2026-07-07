//! Compatibility layer for file I/O in Zig 0.16.
//! Uses raw Linux syscalls (open/read/close/lseek) to provide the old std.fs.File API.
const std = @import("std");
const linux = std.os.linux;

/// Thin wrapper around a raw file descriptor.
/// Mimics the pre-Zig-0.16 std.fs.File API.
pub const File = struct {
    fd: i32,

    pub fn openAbsolute(path: []const u8) !File {
        const path_z = try std.posix.toPosixPath(path);
        const rc = linux.openat(linux.AT.FDCWD, &path_z, .{}, 0);
        const fd: i32 = @intCast(rc);
        if (fd < 0) return error.FileNotFound;
        return .{ .fd = fd };
    }

    pub fn close(self: File) void {
        _ = linux.close(self.fd);
    }

    /// Raw file descriptor for mmap, etc.
    pub fn handle(self: File) i32 {
        return self.fd;
    }

    pub fn getEndPos(self: File) !u64 {
        const offset = linux.lseek(self.fd, 0, linux.SEEK.END);
        if (offset == std.math.maxInt(usize)) return error.StatFailed;
        return offset;
    }

    pub fn readAll(self: File, buf: []u8) !usize {
        const n = linux.read(self.fd, buf.ptr, buf.len);
        const nr: isize = @bitCast(n);
        if (nr < 0) return error.ReadFailed;
        return @intCast(nr);
    }

    pub fn readToEndAlloc(self: File, allocator: std.mem.Allocator, max_size: usize) ![]u8 {
        const size = try self.getEndPos();
        const capped = @min(size, max_size);
        const buf = try allocator.alloc(u8, @intCast(capped));
        _ = try self.readAll(buf);
        return buf;
    }
};
