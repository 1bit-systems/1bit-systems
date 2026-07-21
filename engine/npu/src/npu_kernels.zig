//! NPU XRT xclbin kernel manager.
//! Manages one xclbin kernel (e.g., QKV, O, GU, D) and its associated buffer objects.
//! This is the Zig equivalent of the C++ I8Ctx struct.
const std = @import("std");
const xrt = @import("xrt.zig");

const log = std.log.scoped(.npu_kernel);

/// Manages one NPU xclbin kernel instance with its instruction, activation,
/// weight, and output buffer objects.
pub const XclbinKernel = struct {
    name: []const u8,
    md: u32, // M dimension (tile rows, typically 128)
    kd: u32, // K dimension (in_features)
    nd: u32, // N dimension (out_features)
    n_layers: u32, // Number of transformer layers
    allocator: std.mem.Allocator,

    // XRT resources
    device: ?*xrt.XrtDevice,
    hw_context: ?*xrt.HwContext,
    kernel: xrt.XrtKernel,

    // Buffer objects
    bo_instr: xrt.XrtBuffer,
    bo_act: xrt.XrtBuffer,
    bo_out: xrt.XrtBuffer,
    layer_bos: std.ArrayList(xrt.XrtBuffer),

    // Mapped pointers
    act_mapped: []i8,
    out_mapped: []i16,
    instr_mapped: []u32,
    instr_count: u32,

    // Initialization state
    initialized: bool,

    pub fn init(
        allocator: std.mem.Allocator,
        name: []const u8,
    ) XclbinKernel {
        return .{
            .name = name,
            .md = 0,
            .kd = 0,
            .nd = 0,
            .n_layers = 0,
            .allocator = allocator,
            .device = null,
            .hw_context = null,
            .kernel = xrt.XrtKernel.init(null),
            .bo_instr = xrt.XrtBuffer.init(null),
            .bo_act = xrt.XrtBuffer.init(null),
            .bo_out = xrt.XrtBuffer.init(null),
            .layer_bos = std.ArrayList(xrt.XrtBuffer).empty,
            .act_mapped = undefined,
            .out_mapped = undefined,
            .instr_mapped = undefined,
            .instr_count = 0,
            .initialized = false,
        };
    }

    /// Initialize the kernel: load xclbin, create hardware context, open kernel,
    /// allocate buffer objects, and load instructions.
    pub fn load(
        self: *XclbinKernel,
        device: *xrt.XrtDevice,
        xclbin_path: []const u8,
        insts_path: []const u8,
        md: u32,
        kd: u32,
        nd: u32,
        n_layers: u32,
        weight_group: u32,
    ) !void {
        self.device = device;
        self.md = md;
        self.kd = kd;
        self.nd = nd;
        self.n_layers = n_layers;

        // Load xclbin and get UUID
        const uuid = try device.loadXclbin(xclbin_path);
        log.info("Loaded xclbin {s}, uuid={any}", .{ self.name, uuid.data });

        // Create HW context
        const hwctx_handle = try device.createHwContext(&uuid);
        self.hw_context = hwctx_handle;

        // Open kernel
        const k_handle = try device.createKernel(&uuid, "MLIR_AIE");
        self.kernel = xrt.XrtKernel.init(k_handle);

        // Read instructions file
        const instrs = try readInstructionsFile(self.allocator, insts_path);
        defer self.allocator.free(instrs);
        self.instr_count = @intCast(instrs.len);

        // Allocate instruction BO (cacheable, group 1)
        const instr_bo = try device.allocBO(instrs.len * 4, xrt.XCL_BO_FLAGS_CACHEABLE, 1);
        self.bo_instr = xrt.XrtBuffer.init(instr_bo);

        // Map and copy instructions
        const instr_mem = try self.bo_instr.map(instrs.len * 4);
        self.instr_mapped = @as([*]u32, @ptrCast(@alignCast(instr_mem.ptr)))[0..instrs.len];
        @memcpy(self.instr_mapped, instrs);
        try self.bo_instr.sync(xrt.XCL_BO_SYNC_BO_TO_DEVICE, 0, instrs.len * 4);

        // Allocate activation BO (host_only, group 3)
        // md * kd bytes (int8_t)
        const act_size: usize = @as(usize, md) * @as(usize, kd);
        const act_bo = try device.allocBO(act_size, xrt.XRT_BO_FLAGS_HOST_ONLY, 3);
        self.bo_act = xrt.XrtBuffer.init(act_bo);
        const act_mem = try self.bo_act.map(act_size);
        self.act_mapped = @as([*]i8, @ptrCast(@alignCast(act_mem.ptr)))[0..act_size];

        // Allocate output BO (host_only, group 5)
        // md * nd * 2 bytes (int16_t)
        const out_size: usize = @as(usize, md) * @as(usize, nd) * 2;
        const out_bo = try device.allocBO(out_size, xrt.XRT_BO_FLAGS_HOST_ONLY, 5);
        self.bo_out = xrt.XrtBuffer.init(out_bo);
        const out_mem = try self.bo_out.map(out_size);
        self.out_mapped = @as([*]i16, @ptrCast(@alignCast(out_mem.ptr)))[0 .. md * nd];

        // Allocate per-layer weight BOs (host_only, weight_group)
        try self.layer_bos.ensureTotalCapacity(self.allocator, n_layers);
        const weight_size: usize = @as(usize, kd) * @as(usize, nd);
        for (0..n_layers) |_| {
            const w_bo = try device.allocBO(weight_size, xrt.XRT_BO_FLAGS_HOST_ONLY, weight_group);
            self.layer_bos.appendAssumeCapacity(xrt.XrtBuffer.init(w_bo));
        }

        self.initialized = true;
        log.info("Kernel {s} initialized: md={d} kd={d} nd={d} layers={d} instrs={d}", .{
            self.name, md, kd, nd, n_layers, self.instr_count,
        });
    }

    /// INT8 quantize and pack a weight matrix into the layer's weight BO.
    /// weights is [k * n] row-major floats (in PyTorch convention: [out_features, in_features]).
    /// Returns the quantization scale (for dequant on output).
    pub fn packWeight(self: *XclbinKernel, layer: u32, weights: []const f32, k: u32, n: u32) !f32 {
        if (layer >= self.n_layers) return error.LayerIndexOutOfBounds;

        // Find max absolute value
        var amax: f32 = 0.0;
        for (weights) |w| {
            if (std.math.isFinite(w)) {
                const a = @abs(w);
                if (a > amax) amax = a;
            }
        }
        if (amax < 1e-12) amax = 1.0;

        const scale = amax / 127.0;
        const inv_scale = 127.0 / amax;

        // Map the weight BO
        const bo = self.layer_bos.items[layer];
        const size = k * n;
        const mem = try bo.map(size);
        const dst = @as([*]i8, @ptrCast(@alignCast(mem.ptr)))[0..size];

        for (weights, 0..) |w, i| {
            var v = w;
            if (!std.math.isFinite(v)) v = 0.0;
            const q = @as(i32, @intFromFloat(@round(v * inv_scale)));
            const clamped = @max(-127, @min(127, q));
            dst[i] = @intCast(clamped);
        }

        // Sync to device
        try bo.sync(xrt.XCL_BO_SYNC_BO_TO_DEVICE, 0, size);

        return scale;
    }

    /// Run the kernel for a given layer:
    /// 1. INT8-quantize activations, upload to act_bo
    /// 2. Launch kernel on the NPU
    /// 3. Download and dequantize output
    /// activations: [am * ak] f32
    /// output: [am * an] f32
    pub fn run(
        self: *XclbinKernel,
        layer: u32,
        activations: []const f32,
        am: u32,
        ak: u32,
        ascale: f32,
        bscale: f32,
        output: []f32,
        an: u32,
    ) !void {
        if (!self.initialized) return error.KernelNotInitialized;
        if (layer >= self.n_layers) return error.LayerIndexOutOfBounds;

        // Quantize activations to INT8
        const ais = 1.0 / ascale;
        const act_size = am * self.kd;
        @memset(self.act_mapped[0..act_size], 0);

        for (0..am) |m| {
            for (0..ak) |k| {
                var v = activations[m * ak + k];
                if (!std.math.isFinite(v)) v = 0.0;
                const q = @as(i32, @intFromFloat(@round(v * ais)));
                const clamped = @max(-127, @min(127, q));
                self.act_mapped[m * self.kd + k] = @intCast(clamped);
            }
        }

        // Sync activation BO to device
        try self.bo_act.sync(xrt.XCL_BO_SYNC_BO_TO_DEVICE, 0, act_size);

        // Ensure weight BO is synced (it may have been modified)
        const w_bo = self.layer_bos.items[layer];
        try w_bo.sync(xrt.XCL_BO_SYNC_BO_TO_DEVICE, 0, self.kd * self.nd);

        // Launch kernel: run(3, instr_bo, instr_count, act_bo, weight_bo, out_bo)
        const run_handle = try self.kernel.run(
            self.bo_instr.handle.?,
            @intCast(self.instr_count),
            self.bo_act.handle.?,
            w_bo.handle.?,
            self.bo_out.handle.?,
        );
        defer xrt.XrtRun.init(run_handle).close();

        // Wait for completion
        try xrt.XrtRun.init(run_handle).wait();

        // Sync output BO from device
        const out_size = am * self.nd * 2;
        try self.bo_out.sync(xrt.XCL_BO_SYNC_BO_FROM_DEVICE, 0, out_size);

        // Dequantize: out[i] = Cm[i] * ascale * bscale
        const cs = ascale * bscale;
        for (0..am) |m| {
            for (0..an) |n| {
                const val = @as(f32, @floatFromInt(self.out_mapped[m * self.nd + n])) * cs;
                output[m * an + n] = if (std.math.isFinite(val)) val else 0.0;
            }
        }
    }

    /// Free all resources.
    pub fn deinit(self: *XclbinKernel) void {
        // Free layer weight BOs
        for (self.layer_bos.items) |*bo| {
            bo.free();
        }
        self.layer_bos.deinit(self.allocator);

        // Free output BO
        self.bo_out.free();
        // Free activation BO
        self.bo_act.free();
        // Free instruction BO
        self.bo_instr.free();

        // Close kernel
        self.kernel.close();

        // Destroy HW context
        if (self.hw_context) |ctx| {
            xrt.XrtHwContext.init(ctx).destroy();
        }

        self.initialized = false;
    }
};

/// Read a text file of instruction words (hex or decimal u32 values).
/// Each line can contain one value, or values can be space-separated.
fn readInstructionsFile(allocator: std.mem.Allocator, path: []const u8) ![]u32 {
    const compat = @import("compat.zig");
    const file = try compat.File.openAbsolute(path);
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 1024 * 1024); // 1MB max
    defer allocator.free(content);

    // Parse whitespace-separated integers
    var result = std.ArrayList(u32).empty;
    errdefer result.deinit(allocator);

    var it = std.mem.tokenizeAny(u8, content, " \t\r\n");
    while (it.next()) |token| {
        // Try hex first (0x or just hex digits), then decimal
        const trimmed = std.mem.trim(u8, token, &std.ascii.whitespace);
        if (trimmed.len == 0) continue;

        const val = if (std.mem.startsWith(u8, trimmed, "0x") or std.mem.startsWith(u8, trimmed, "0X"))
            std.fmt.parseInt(u32, trimmed[2..], 16) catch {
                return error.InvalidHexToken;
            }
        else
            std.fmt.parseInt(u32, trimmed, 10) catch {
                return error.InvalidHexToken;
            };

        try result.append(allocator, val);
    }

    return result.toOwnedSlice(allocator);
}
