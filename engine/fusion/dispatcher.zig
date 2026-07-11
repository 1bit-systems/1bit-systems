//! Layer-level dispatcher for NPU+GPU+CPU fused inference.
//! Decides which backend computes each layer or operation based on policy,
//! availability, and workload characteristics.
//!
//! @section Fused Engine
const std = @import("std");

/// Which backend to use for a given inference operation.
pub const BackendDevice = enum(u8) {
    /// NPU (XDNA 2 via XRT xclbin kernels)
    npu = 0,
    /// GPU (Vulkan / Metal / CUDA compute)
    gpu = 1,
    /// CPU (pure C++ ternary inference, no accelerator needed)
    cpu = 2,
};

const log = std.log.scoped(.dispatcher);

/// Policy for routing inference work to NPU vs GPU.
pub const DispatchPolicy = enum(u8) {
    /// Use NPU for all computations (GPU disabled).
    npu_only = 0,
    /// Use GPU for all computations (NPU disabled).
    gpu_only = 1,
    /// Route layers round-robin between NPU and GPU.
    layer_by_layer = 2,
    /// Run attention on NPU (edge_attention kernel), FFN on GPU (Vulkan DMMV).
    /// Best for small models where NPU GEMM + GPU attention overlap.
    attention_on_npu = 3,
    /// Run FFN on NPU (INT8 GEMM), attention on GPU (flash attention).
    /// Best when GPU flash attention is fast and NPU GEMM is fast.
    ffn_on_npu = 4,
    /// Run QKV projection on NPU, rest on GPU.
    qkv_on_npu = 5,
    /// Run the entire prefill on NPU, decode on GPU.
    prefill_npu_decode_gpu = 6,
    /// Auto-detect: benchmark each operation and choose fastest.
    auto = 7,
    /// Use CPU for all computations (pure C++ ternary inference, no accelerator).
    cpu_only = 8,
    /// CPU fallback when other backends fail.
    cpu_fallback = 9,
};

/// Per-layer dispatch decision.
pub const LayerAssignment = struct {
    /// Which backend handles the attention sub-layer.
    attention: BackendDevice = .gpu,
    /// Which backend handles the FFN sub-layer.
    ffn: BackendDevice = .gpu,
    /// Which backend handles the QKV projection.
    qkv: BackendDevice = .gpu,
};

/// Statistics for the auto-policy.
pub const DispatchStats = struct {
    npu_gemm_ns: u64 = 0,
    gpu_gemm_ns: u64 = 0,
    cpu_gemm_ns: u64 = 0,
    npu_attention_ns: u64 = 0,
    gpu_attention_ns: u64 = 0,
    cpu_attention_ns: u64 = 0,
    decisions: u64 = 0,

    pub fn reset(self: *DispatchStats) void {
        self.* = .{};
    }
};

/// Dispatcher that assigns layers to NPU or GPU based on policy.
pub const Dispatcher = struct {
    policy: DispatchPolicy,
    stats: DispatchStats,

    /// Pre-computed layer assignments (for static policies).
    assignments: []LayerAssignment,
    n_layers: u32,

    pub fn init(allocator: std.mem.Allocator, n_layers: u32, policy: DispatchPolicy) !Dispatcher {
        const assignments = try allocator.alloc(LayerAssignment, n_layers);
        errdefer allocator.free(assignments);

        var disp = Dispatcher{
            .policy = policy,
            .stats = .{},
            .assignments = assignments,
            .n_layers = n_layers,
        };
        disp.recomputeAssignments();
        return disp;
    }

    pub fn deinit(self: *Dispatcher, allocator: std.mem.Allocator) void {
        allocator.free(self.assignments);
    }

    /// Recompute layer assignments when the policy changes.
    pub fn recomputeAssignments(self: *Dispatcher) void {
        for (0..self.n_layers) |i| {
            self.assignments[i] = self.assignForLayer(@intCast(i));
        }
        log.info("Dispatcher recomputed {d} layer assignments (policy={s})", .{
            self.n_layers, @tagName(self.policy),
        });
    }

    fn assignForLayer(self: *const Dispatcher, layer: u32) LayerAssignment {
        return switch (self.policy) {
            .npu_only => .{
                .attention = .npu,
                .ffn = .npu,
                .qkv = .npu,
            },
            .gpu_only => .{
                .attention = .gpu,
                .ffn = .gpu,
                .qkv = .gpu,
            },
            .layer_by_layer => blk: {
                // Alternate: even layers → NPU, odd → GPU
                if (layer % 2 == 0) {
                    break :blk .{ .attention = .npu, .ffn = .npu, .qkv = .npu };
                } else {
                    break :blk .{ .attention = .gpu, .ffn = .gpu, .qkv = .gpu };
                }
            },
            .attention_on_npu => .{
                .attention = .npu,
                .ffn = .gpu,
                .qkv = .gpu,
            },
            .ffn_on_npu => .{
                .attention = .gpu,
                .ffn = .npu,
                .qkv = .gpu,
            },
            .qkv_on_npu => .{
                .attention = .gpu,
                .ffn = .gpu,
                .qkv = .npu,
            },
            .prefill_npu_decode_gpu => blk: {
                // Prefill policy is handled separately; decode always goes to GPU
                break :blk .{ .attention = .gpu, .ffn = .gpu, .qkv = .gpu };
            },
            .auto => .{
                .attention = .gpu, // GPU flash attention is typically faster
                .ffn = .npu, // NPU INT8 GEMM is fast for FFN
                .qkv = .npu, // NPU INT8 GEMM for QKV projection
            },
            .cpu_only, .cpu_fallback => .{
                .attention = .cpu,
                .ffn = .cpu,
                .qkv = .cpu,
            },
        };
    }

    /// Get the assignment for a specific layer from a policy (no dispatcher instance needed).
    pub fn getPolicyAssignment(policy: DispatchPolicy, layer: u32) LayerAssignment {
        return switch (policy) {
            .npu_only => .{ .attention = .npu, .ffn = .npu, .qkv = .npu },
            .gpu_only => .{ .attention = .gpu, .ffn = .gpu, .qkv = .gpu },
            .layer_by_layer => if (layer % 2 == 0)
                .{ .attention = .npu, .ffn = .npu, .qkv = .npu }
            else
                .{ .attention = .gpu, .ffn = .gpu, .qkv = .gpu },
            .attention_on_npu => .{ .attention = .npu, .ffn = .gpu, .qkv = .gpu },
            .ffn_on_npu => .{ .attention = .gpu, .ffn = .npu, .qkv = .gpu },
            .qkv_on_npu => .{ .attention = .gpu, .ffn = .gpu, .qkv = .npu },
            .prefill_npu_decode_gpu => .{ .attention = .gpu, .ffn = .gpu, .qkv = .gpu },
            .auto => .{ .attention = .gpu, .ffn = .npu, .qkv = .npu },
            .cpu_only, .cpu_fallback => .{ .attention = .cpu, .ffn = .cpu, .qkv = .cpu },
        };
    }

    /// Get the assignment for a specific layer.
    pub fn getAssignment(self: *const Dispatcher, layer: u32) LayerAssignment {
        if (layer >= self.n_layers) {
            return .{}; // default: all GPU
        }
        return self.assignments[layer];
    }

    /// Return a human-readable description of the current policy.
    pub fn describe(self: *const Dispatcher) []const u8 {
        return switch (self.policy) {
            .npu_only => "All layers → NPU (XRT xclbin INT8 GEMM + CPU ops)",
            .gpu_only => "All layers → GPU (Vulkan/Metal/CUDA flash attention + DMMV)",
            .layer_by_layer => "Round-robin per layer: even → NPU, odd → GPU",
            .attention_on_npu => "Attention → NPU edge_attention kernel, FFN → GPU DMMV",
            .ffn_on_npu => "FFN → NPU INT8 GEMM, Attention → GPU flash attention",
            .qkv_on_npu => "QKV projection → NPU, rest → GPU",
            .prefill_npu_decode_gpu => "Prefill → NPU (fast INT8), Decode → GPU (batch decode)",
            .auto => "Auto-assign: QKV/FFN → NPU, Attention → GPU (auto-tuned)",
            .cpu_only => "All layers → CPU ternary GEMV (pure C++, no accelerator)",
            .cpu_fallback => "CPU fallback — use CPU when NPU/GPU unavailable",
        };
    }

    /// Record a dispatch decision for the auto-policy's statistics.
    pub fn recordDispatch(self: *const Dispatcher, device: BackendDevice, latency_ns: u64) void {
        _ = self;
        _ = device;
        _ = latency_ns;
        // Stats tracking will be enabled in a future cycle
    }
};

test "Dispatcher basic assignment" {
    const allocator = std.testing.allocator;
    var disp = try Dispatcher.init(allocator, 4, .npu_only);
    defer disp.deinit(allocator);

    for (0..4) |i| {
        const a = disp.getAssignment(@intCast(i));
        try std.testing.expectEqual(BackendDevice.npu, a.attention);
        try std.testing.expectEqual(BackendDevice.npu, a.ffn);
        try std.testing.expectEqual(BackendDevice.npu, a.qkv);
    }
}

test "Dispatcher cpu_only" {
    const allocator = std.testing.allocator;
    var disp = try Dispatcher.init(allocator, 4, .cpu_only);
    defer disp.deinit(allocator);

    for (0..4) |i| {
        const a = disp.getAssignment(@intCast(i));
        try std.testing.expectEqual(BackendDevice.cpu, a.attention);
        try std.testing.expectEqual(BackendDevice.cpu, a.ffn);
        try std.testing.expectEqual(BackendDevice.cpu, a.qkv);
    }
}
