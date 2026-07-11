//! CPU Backend — pure CPU ternary inference module for the fused engine.
//!
//! Wraps cpu_layer.c (C ABI) and provides a CpuBackend struct that mirrors
//! the NPU/GPU backend interfaces so the dispatcher can route to CPU.
//!
//! ## Integration
//! In engine.zig, add `cpu_backend` to the imports and a CpuBackend field:
//! ```zig
//! const cpu_backend = @import("cpu_backend.zig");
//! // ...
//! cpu: ?CpuBackend = null,
//! cpu_available: bool = false,
//! ```
//!
//! In fused_execute.zig, add `.cpu` to the Backend enum and dispatch to
//! `ex.cpu.?.executeLayer(...)` when Backend == .cpu.
//!
//! @section Fused Engine

const std = @import("std");
const c = @cImport({
    @cInclude("cpu_layer.h");
});

const log = std.log.scoped(.cpu_backend);

/// CPU device handle — owns model weights and scratch buffers in host memory.
pub const CpuBackend = struct {
    allocator: std.mem.Allocator,

    // Model config
    hidden_dim: u32,
    inter_size: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    gqa_ratio: u32,
    vocab_size: u32,
    n_layers: u32,
    max_seq_len: u32,
    rms_norm_eps: f32,

    // Weights (host memory, fp32)
    emb_table: []f32,           // [vocab_size * hidden_dim]

    // Per-layer packed ternary weights (host memory)
    q_packed:    []u32, q_scales:    []f32,
    k_packed:    []u32, k_scales:    []f32,
    v_packed:    []u32, v_scales:    []f32,
    o_packed:    []u32, o_scales:    []f32,
    gate_packed: []u32, gate_scales: []f32,
    up_packed:   []u32, up_scales:   []f32,
    down_packed: []u32, down_scales: []f32,

    // Layer norms (fp32)
    in_norm:  [][]f32,   // [n_layers][hidden_dim]
    pa_norm:  [][]f32,   // [n_layers][hidden_dim]
    q_norm:   [][]f32,   // [n_layers][head_dim] or null
    k_norm:   [][]f32,   // [n_layers][head_dim] or null
    final_norm: []f32,   // [hidden_dim]

    // RoPE tables
    sin_table: []f32,    // [max_seq_len * head_dim]
    cos_table: []f32,    // [max_seq_len * head_dim]

    // KV cache
    k_cache: [][]f32,    // [n_layers][max_seq_len * n_kv_heads * head_dim]
    v_cache: [][]f32,    // [n_layers][max_seq_len * n_kv_heads * head_dim]

    // Scratch buffers
    scratch_qkv:  []f32,   // [n_heads*hd + 2*n_kv_heads*hd]
    scratch_attn: []f32,   // [n_heads*hd]
    scratch_ffn:  []f32,   // [2 * inter_size]
    scratch_act:  []f32,   // [inter_size]

    // LM head weights (may be same as emb_table if tied)
    lm_head_table: []f32,

    pub fn init(
        allocator: std.mem.Allocator,
        config: struct {
            hidden_dim: u32, inter_size: u32,
            n_heads: u32, n_kv_heads: u32, head_dim: u32,
            vocab_size: u32, n_layers: u32, max_seq_len: u32,
            rms_norm_eps: f32,
        },
    ) !CpuBackend {
        const H = config.hidden_dim;
        const IM = config.inter_size;
        const NH = config.n_heads;
        const NKV = config.n_kv_heads;
        const HD = config.head_dim;
        const NV = config.vocab_size;
        const L = config.n_layers;
        const MSL = config.max_seq_len;
        const GQA = if (NKV > 0) @divExact(NH, NKV) else @as(u32, 1);

        // Scratch buffers
        const qkv_sz  = NH * HD + 2 * NKV * HD;
        const ffn_sz  = 2 * IM;

        const scratch_qkv  = try allocator.alloc(f32, qkv_sz);
        errdefer allocator.free(scratch_qkv);
        const scratch_attn = try allocator.alloc(f32, NH * HD);
        errdefer allocator.free(scratch_attn);
        const scratch_ffn  = try allocator.alloc(f32, ffn_sz);
        errdefer allocator.free(scratch_ffn);
        const scratch_act  = try allocator.alloc(f32, IM);
        errdefer allocator.free(scratch_act);

        // Weights (zero-initialized — caller fills or loads)
        const w = @import("std").mem;

        return CpuBackend{
            .allocator = allocator,
            .hidden_dim = H, .inter_size = IM,
            .n_heads = NH, .n_kv_heads = NKV, .head_dim = HD,
            .gqa_ratio = GQA, .vocab_size = NV, .n_layers = L,
            .max_seq_len = MSL, .rms_norm_eps = config.rms_norm_eps,
            .emb_table = &.{},
            .q_packed = &.{}, .q_scales = &.{},
            .k_packed = &.{}, .k_scales = &.{},
            .v_packed = &.{}, .v_scales = &.{},
            .o_packed = &.{}, .o_scales = &.{},
            .gate_packed = &.{}, .gate_scales = &.{},
            .up_packed = &.{}, .up_scales = &.{},
            .down_packed = &.{}, .down_scales = &.{},
            .in_norm = &.{}, .pa_norm = &.{},
            .q_norm = &.{}, .k_norm = &.{},
            .final_norm = &.{},
            .sin_table = &.{}, .cos_table = &.{},
            .k_cache = &.{}, .v_cache = &.{},
            .scratch_qkv = scratch_qkv,
            .scratch_attn = scratch_attn,
            .scratch_ffn = scratch_ffn,
            .scratch_act = scratch_act,
            .lm_head_table = &.{},
        };
    }

    pub fn deinit(self: *CpuBackend) void {
        const aa = self.allocator;
        aa.free(self.scratch_act);
        aa.free(self.scratch_ffn);
        aa.free(self.scratch_attn);
        aa.free(self.scratch_qkv);
        // Weight cleanup would free all allocated slices here
        // In practice, weights are loaded once and live for the engine lifetime
    }

    /// Execute one transformer layer on CPU.
    pub fn executeLayer(
        self: *CpuBackend,
        layer: u32,
        hidden: []f32,
        pos: u32,
    ) !void {
        const H = self.hidden_dim;
        const IM = self.inter_size;
        const NH = self.n_heads;
        const NKV = self.n_kv_heads;
        const HD = self.head_dim;
        const GQA = self.gqa_ratio;
        const L = @as(u32, @intCast(layer));

        // Call the C function
        const rc = c.cpu_layer_forward_qwen3(
            hidden.ptr,
            self.scratch_qkv.ptr,
            self.scratch_attn.ptr,
            self.scratch_ffn.ptr,
            self.scratch_act.ptr,
            self.in_norm[L].ptr,
            if (self.q_norm.len > 0) self.q_norm[L].ptr else null,
            if (self.k_norm.len > 0) self.k_norm[L].ptr else null,
            self.pa_norm[L].ptr,
            null, // final_norm — applied at end of all layers
            // Packed weights
            self.getLayerWeights(L),
            // KV cache
            self.k_cache[L].ptr,
            self.v_cache[L].ptr,
            // Config
            @as(c_int, H),
            @as(c_int, IM),
            @as(c_int, NH),
            @as(c_int, NKV),
            @as(c_int, HD),
            @as(c_int, GQA),
            @as(c_int, pos),
            self.sin_table.ptr,
            self.cos_table.ptr,
            @as(c_float, self.rms_norm_eps),
        );

        if (rc != 0) {
            log.err("cpu_layer_forward_qwen3 returned {d}", .{rc});
            return error.CpuLayerError;
        }
    }

    /// Helper to gather all per-layer weight pointers into one struct for the C function
    fn getLayerWeights(self: *CpuBackend, layer: u32) struct {
        q_packed: [*c]const u32, q_scales: [*c]const f32,
        k_packed: [*c]const u32, k_scales: [*c]const f32,
        v_packed: [*c]const u32, v_scales: [*c]const f32,
        o_packed: [*c]const u32, o_scales: [*c]const f32,
        gate_packed: [*c]const u32, gate_scales: [*c]const f32,
        up_packed: [*c]const u32, up_scales: [*c]const f32,
        down_packed: [*c]const u32, down_scales: [*c]const f32,
    } {
        return .{
            .q_packed = self.q_packed.ptr, .q_scales = self.q_scales.ptr,
            .k_packed = self.k_packed.ptr, .k_scales = self.k_scales.ptr,
            .v_packed = self.v_packed.ptr, .v_scales = self.v_scales.ptr,
            .o_packed = self.o_packed.ptr, .o_scales = self.o_scales.ptr,
            .gate_packed = self.gate_packed.ptr, .gate_scales = self.gate_scales.ptr,
            .up_packed = self.up_packed.ptr, .up_scales = self.up_scales.ptr,
            .down_packed = self.down_packed.ptr, .down_scales = self.down_scales.ptr,
        };
    }

    /// Load weights from a flat binary dump (nnpkg format).
    /// Format TBD — for now, caller must fill arrays directly.
    pub fn loadWeights(self: *CpuBackend, _path: []const u8) !void {
        _ = self;
        _ = _path;
        log.info("Weight loading from file not yet implemented", .{});
        return error.NotImplemented;
    }

    /// Run LM head: compute logits, return argmax token.
    pub fn lmHead(self: *CpuBackend, hidden: []const f32) !u32 {
        const NV = self.vocab_size;
        const H = self.hidden_dim;
        const table = if (self.lm_head_table.len > 0) self.lm_head_table.ptr else self.emb_table.ptr;

        // Reuse scratch_act as logits buffer (it's IM-sized, may not hold vocab)
        // For now, allocate on stack if reasonable
        const max_stack = 4096;
        var small_logits: [max_stack]f32 = undefined;
        const logits = if (NV <= max_stack)
            &small_logits
        else
            @panic("vocab too large for stack — use allocator");

        c.cpu_lm_head(hidden.ptr, table, &logits, @as(c_int, NV), @as(c_int, H));

        const best = c.cpu_argmax(&logits, @as(c_int, NV));
        return @as(u32, @intCast(best));
    }

    /// Embed token: copy embedding row into output.
    pub fn embed(self: *CpuBackend, token: u32, output: []f32) void {
        c.cpu_embed(self.emb_table.ptr, @as(c_int, @intCast(token)), output.ptr, @as(c_int, @intCast(self.hidden_dim)));
    }

    /// Apply final RMSNorm.
    pub fn finalNorm(self: *CpuBackend, hidden: []f32) void {
        c.cpu_rmsnorm(
            hidden.ptr,
            self.final_norm.ptr,
            hidden.ptr,
            @as(c_int, @intCast(self.hidden_dim)),
            self.rms_norm_eps,
        );
    }
};

test "CpuBackend type compiles" {
    const allocator = std.testing.allocator;
    var cpu = try CpuBackend.init(allocator, .{
        .hidden_dim = 1536, .inter_size = 4096,
        .n_heads = 12, .n_kv_heads = 2, .head_dim = 128,
        .vocab_size = 100, .n_layers = 4, .max_seq_len = 256,
        .rms_norm_eps = 1e-6,
    });
    defer cpu.deinit();
    try std.testing.expectEqual(@as(u32, 1536), cpu.hidden_dim);
}

test "cpu_ternary_gemv C function" {
    // Test the raw C function directly
    const M = 4;
    const K = 32;
    var packed: [M * K / 16]u32 = undefined;
    var x: [K]f32 = undefined;
    var scales: [M]f32 = undefined;
    var y: [M]f32 = undefined;

    // Fill: all weights = +1 (code 0x1)
    for (&packed) |*w| w.* = 0x11111111;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i));
    for (&scales) |*s| s.* = 2.0;

    c.cpu_ternary_gemv(&packed, &x, &scales, &y, M, K);

    // Each row: sum(x) = 0+1+...+31 = 496. scale=2. y=992
    for (&y) |v| try std.testing.expectApproxEqAbs(@as(f32, 992.0), v, 0.01);
}

test "cpu_rmsnorm C function" {
    var x = [_]f32{ 3.0, 4.0 };
    var w = [_]f32{ 1.0, 1.0 };
    var y: [2]f32 = undefined;
    c.cpu_rmsnorm(&x, &w, &y, 2, 1e-6);

    // RMS = sqrt((9+16)/2) = sqrt(12.5) = 3.536
    // y = x / rms * w = [3/3.536, 4/3.536] = [0.848, 1.131]
    try std.testing.expectApproxEqAbs(@as(f32, 0.848), y[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.131), y[1], 0.01);
}

test "cpu_silu_glu C function" {
    var gate = [_]f32{ 1.0, 2.0, 0.0, -1.0 };
    var up = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    var out: [4]f32 = undefined;
    c.cpu_silu_glu(&gate, &up, &out, 4);

    // silu(1) ≈ 0.731, *10 = 7.31
    // silu(2) ≈ 1.761, *20 = 35.22
    // silu(0) = 0, *30 = 0
    // silu(-1) ≈ -0.269, *40 = -10.76
    try std.testing.expectApproxEqAbs(@as(f32, 7.31), out[0], 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 35.22), out[1], 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[2], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -10.76), out[3], 0.1);
}

test "cpu_argmax C function" {
    var vals = [_]f32{ 3.0, 1.0, 99.0, -5.0, 50.0 };
    const best = c.cpu_argmax(&vals, 5);
    try std.testing.expectEqual(@as(c_int, 2), best);
}

test "cpu_rope C function" {
    var x = [_]f32{ 1.0, 0.0, 1.0, 0.0 }; // 2 heads × 2 dim
    var sin_table = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0 };
    var cos_table = [_]f32{ 1.0, 1.0, 1.0, 1.0, 0.866, 1.0, 0.866, 1.0 };
    c.cpu_rope(&x, 0, 2, 2, &sin_table, &cos_table);

    // pos=0: cos=1, sin=0 → no change
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), x[1], 0.01);
}
