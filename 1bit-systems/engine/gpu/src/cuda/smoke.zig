//! Standalone smoke test for the ZINC CUDA backend Zig wrapper layer.
//! Drives the GPU entirely through device.zig / buffer.zig / pipeline.zig /
//! command.zig (which wrap cuda_shim.c) — proving the Zig<->CUDA seam:
//! device select, staged buffers + H2D/D2H, NVRTC runtime compile, the
//! buffers+push dispatch ABI, and both sync and async commit paths.
//!
//! Build (on the box):
//!   ~/zig-0.15.2/zig build-exe smoke.zig cuda_shim.c \
//!     -I. -I/usr/local/cuda/include -lc \
//!     -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcuda -lnvrtc \
//!     -rpath /usr/local/cuda/lib64 -femit-bin=smoke_zig
//! @section CUDA Runtime
const std = @import("std");
const device = @import("device.zig");
const buffer = @import("buffer.zig");
const pipeline = @import("pipeline.zig");
const command = @import("command.zig");

const KSRC =
    \\struct Push { int n; };
    \\extern "C" __global__ void vadd(const float* a, const float* b, float* c, struct Push pc){
    \\  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < pc.n) c[i] = a[i] + b[i];
    \\}
    \\extern "C" __global__ void dp4a_k(const int* a, const int* b, int* out, struct Push pc){
    \\  int acc = 0; acc = __dp4a(a[0], b[0], acc); out[0] = acc;
    \\}
;

const Push = extern struct { n: i32 };

/// Run the CUDA Zig<->shim smoke test end to end and report PASS/FAIL.
///
/// Selects the best device, then exercises the full seam: a `vadd` kernel via the
/// synchronous commit path and a `dp4a` kernel via the async commit path, checking
/// both results.
/// @returns `error.SmokeFailed` if any computed value mismatches its expected output.
pub fn main() !void {
    var dev = try device.CudaDevice.initBest(std.heap.page_allocator);
    defer dev.deinit();
    var namebuf: [128]u8 = undefined;
    const nm = dev.name(&namebuf);
    std.debug.print("device: {s}  cc={d}  SMs={d}  vramGB={d}\n", .{
        nm, dev.computeCapability(), dev.smCount(), dev.totalMemory() / (1024 * 1024 * 1024),
    });

    const ctx = dev.ctx;
    const N: usize = 1024;

    var a = try buffer.createBufferStaged(ctx, N * @sizeOf(f32));
    defer buffer.freeBuffer(&a);
    var b = try buffer.createBufferStaged(ctx, N * @sizeOf(f32));
    defer buffer.freeBuffer(&b);
    var out = try buffer.createBuffer(ctx, N * @sizeOf(f32));
    defer buffer.freeBuffer(&out);

    const ha: [*]f32 = @ptrCast(@alignCast(a.host_ptr.?));
    const hb: [*]f32 = @ptrCast(@alignCast(b.host_ptr.?));
    var i: usize = 0;
    while (i < N) : (i += 1) {
        ha[i] = @floatFromInt(i);
        hb[i] = @as(f32, @floatFromInt(i)) * 2.0;
    }
    buffer.upload(ctx, &a, std.mem.sliceAsBytes(ha[0..N]));
    buffer.upload(ctx, &b, std.mem.sliceAsBytes(hb[0..N]));

    var vadd = try pipeline.createPipeline(ctx, KSRC, "vadd");
    defer pipeline.freePipeline(&vadd);
    var dp = try pipeline.createPipeline(ctx, KSRC, "dp4a_k");
    defer pipeline.freePipeline(&dp);
    std.debug.print("nvrtc: compiled vadd (max_threads={d}) + dp4a_k for sm_{d}\n", .{
        vadd.max_threads, dev.computeCapability(),
    });

    const push = Push{ .n = @intCast(N) };
    var cmd = try command.beginCommand(ctx);
    const bufs = [_]*const buffer.CudaBuffer{ &a, &b, &out };
    cmd.dispatch(&vadd, .{ @intCast((N + 255) / 256), 1, 1 }, .{ 256, 1, 1 }, &bufs, &push, @sizeOf(Push), 0);
    cmd.commitAndWait();

    var hc: [N]f32 = undefined;
    buffer.download(ctx, &out, std.mem.sliceAsBytes(hc[0..]));
    std.debug.print("vadd: c[1]={d} (expect 3)  c[100]={d} (expect 300)\n", .{ hc[1], hc[100] });

    // dp4a through the abstraction via the ASYNC commit path.
    const pa: i32 = (1 & 0xff) | (2 << 8) | (3 << 16) | (4 << 24);
    const pb: i32 = (5 & 0xff) | (6 << 8) | (7 << 16) | (8 << 24);
    var da = try buffer.createBuffer(ctx, 4);
    defer buffer.freeBuffer(&da);
    var db = try buffer.createBuffer(ctx, 4);
    defer buffer.freeBuffer(&db);
    var dout = try buffer.createBuffer(ctx, 4);
    defer buffer.freeBuffer(&dout);
    buffer.upload(ctx, &da, std.mem.asBytes(&pa));
    buffer.upload(ctx, &db, std.mem.asBytes(&pb));

    const bufs2 = [_]*const buffer.CudaBuffer{ &da, &db, &dout };
    var cmd2 = try command.beginCommand(ctx);
    cmd2.dispatch(&dp, .{ 1, 1, 1 }, .{ 1, 1, 1 }, &bufs2, &push, @sizeOf(Push), 0);
    cmd2.commitAsync();
    cmd2.wait();
    var hr: i32 = 0;
    buffer.download(ctx, &dout, std.mem.asBytes(&hr));
    std.debug.print("dp4a (via abstraction, async path): {d} (expect 70)\n", .{hr});

    const ok = (hc[1] == 3.0 and hc[100] == 300.0 and hr == 70);
    std.debug.print("RESULT: {s}\n", .{if (ok) "PASS" else "FAIL"});
    if (!ok) return error.SmokeFailed;
}
