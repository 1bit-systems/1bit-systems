//! Fused NPU+GPU layer execution engine.
//! Routes each transformer layer to the optimal backend:
//!   - NPU: INT8 GEMM for QKV projection, FFN gate/up/down
//!   - GPU: Flash attention via Vulkan (gpu_attn module)
//!
//! Supports three attention kernels:
//!   - .flash — standard flash attention (Qwen, Llama, Gemma)
//!   - .mla   — Multi-Head Latent Attention (DeepSeek V2/V3)
//!
//! Supports three FFN kernels:
//!   - .dense       — standard gate/up/down FFN
//!   - .moe         — routed MoE (Mixtral, Zaya)
//!   - .shared_moe  — MoE with shared expert (DeepSeek)
//!
//! Target: 273 tok/s coherent on Qwen3-0.6B (NPU GEMM + GPU attention)
//! Strategy:
//!   - NPU does GEMM-heavy QKV/FFN at ~0.3ms each → 28×0.3 = 8.4ms
//!   - GPU does flash attention at ~0.5ms/layer → 28×0.5 = 14ms
//!   - Pipeline overlap hides GPU attention behind NPU QKV of next layer
//!   - M=128 batch decode amortizes LM head cost
//!
//! @section Fused Engine
const std = @import("std");
const Io = std.Io;
const gpu_attn = @import("gpu_attn.zig");
const interop = @import("interop.zig");
const arch_registry = @import("arch_registry.zig");
const moe_router = @import("moe_router.zig");
const mla_attn = @import("mla_attn.zig");

const log = std.log.scoped(.fused_execute);

/// Qwen3-0.6B model dimensions.
pub const QWEN3_0_6B = ModelConfig{
    .hidden_dim = 1024,
    .n_layers = 28,
    .n_heads = 16,
    .n_kv_heads = 8,
    .head_dim = 128,
    .inter_size = 3072,
    .vocab_size = 151936,
    .gqa_ratio = 2,
};

/// Model configuration.
pub const ModelConfig = struct {
    hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    inter_size: u32,
    vocab_size: u32,
    gqa_ratio: u32,
};

/// Backend device for computation.
pub const Backend = enum(u8) {
    npu = 0,
    gpu = 1,
    cpu = 2,
};

/// Dispatch policy for layer routing.
pub const DispatchPolicy = enum(u8) {
    /// All layers on NPU (baseline, 46 tok/s with BS=128).
    npu_only = 0,
    /// All layers on GPU (flash attention + DMMV).
    gpu_only = 1,
    /// FFN on NPU, attention on GPU (target: 273 tok/s).
    ffn_on_npu = 2,
    /// QKV on NPU, rest on GPU.
    qkv_on_npu = 3,
    /// Attention on NPU, FFN on GPU.
    attention_on_npu = 4,
    /// All layers on CPU (pure C++ ternary inference).
    cpu_only = 5,
    /// Prefill on NPU, decode GPU attention + NPU FFN with batch split.
    prefill_npu_decode_gpu = 6,
    /// Hardware-aware: always route each op to whichever real backend is
    /// fastest for it, independently degrading per-op as accelerators
    /// become unavailable rather than dropping everything to CPU at once.
    /// Attention prefers GPU (flash attention), QKV/FFN prefer NPU
    /// (INT8 GEMM) -- each falls back to the other accelerator, then CPU,
    /// so partial hardware (GPU-only or NPU-only) still gets the best of
    /// what's actually present instead of a static, possibly-stale policy.
    auto = 7,
};

/// Per-layer dispatch decision.
pub const LayerDispatch = struct {
    qkv: Backend = .npu,
    attention: Backend = .npu,
    ffn: Backend = .npu,
};

/// Threshold for decode/batch split: M <= this uses per-token decode (skinny),
/// M > this uses batched GEMV kernels (fat batch).
pub const BATCH_SPLIT_THRESHOLD: u32 = 5;

/// Shared KV cache for NPU↔GPU interop.
/// K and V stored in f32. Written by NPU QKV step, read by GPU attention step.
/// For MLA models, the cache stores compressed KV latents (see MlaKVCache).
pub const SharedKVCache = struct {
    allocator: std.mem.Allocator,
    n_layers: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_context: u32,

    /// Per-layer K: [layer][pos * n_kv_heads * head_dim]
    k_cache: [][]f32,
    /// Per-layer V: [layer][pos * n_kv_heads * head_dim]
    v_cache: [][]f32,
    /// Flat backing storage for K data
    k_flat: []f32,
    /// Flat backing storage for V data
    v_flat: []f32,
    /// Current sequence position.
    position: u32 = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        n_layers: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_context: u32,
    ) !SharedKVCache {
        const per_layer = max_context * n_kv_heads * head_dim;
        const k_total = n_layers * per_layer;
        const v_total = n_layers * per_layer;

        const k_flat = try allocator.alloc(f32, k_total);
        errdefer allocator.free(k_flat);
        @memset(k_flat, 0);

        const v_flat = try allocator.alloc(f32, v_total);
        errdefer allocator.free(v_flat);
        @memset(v_flat, 0);

        const k_slices = try allocator.alloc([]f32, n_layers);
        errdefer allocator.free(k_slices);
        const v_slices = try allocator.alloc([]f32, n_layers);
        errdefer allocator.free(v_slices);

        for (0..n_layers) |l| {
            k_slices[l] = k_flat[l * per_layer ..][0..per_layer];
            v_slices[l] = v_flat[l * per_layer ..][0..per_layer];
        }

        log.info("SharedKVCache: {d}L × {d}pos × {d}KV × {d}dim = {d} bytes", .{
            n_layers, max_context, n_kv_heads, head_dim,
            (k_total + v_total) * @sizeOf(f32),
        });

        return .{
            .allocator = allocator,
            .n_layers = n_layers,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .max_context = max_context,
            .k_cache = k_slices,
            .v_cache = v_slices,
            .k_flat = k_flat,
            .v_flat = v_flat,
        };
    }

    pub fn deinit(self: *SharedKVCache) void {
        self.allocator.free(self.v_flat);
        self.allocator.free(self.k_flat);
        self.allocator.free(self.v_cache);
        self.allocator.free(self.k_cache);
    }

    pub fn writeKV(
        self: *SharedKVCache,
        layer: u32,
        n_kv_heads: u32,
        head_dim: u32,
        k_data: []const f32,
        v_data: []const f32,
        n_tokens: u32,
    ) void {
        const base = self.position;
        const stride = n_kv_heads * head_dim;
        // Write all tokens' KV data, not just the first one (#686 fix)
        for (0..n_tokens) |t| {
            const dst_base = (base + t) * stride;
            const src_off = t * stride;
            for (0..stride) |i| {
                self.k_cache[layer][dst_base + i] = k_data[src_off + i];
                self.v_cache[layer][dst_base + i] = v_data[src_off + i];
            }
        }
    }

    pub fn advance(self: *SharedKVCache, n_tokens: u32) void {
        self.position += n_tokens;
    }

    pub fn getK(self: *const SharedKVCache, layer: u32) []const f32 {
        const stride = self.n_kv_heads * self.head_dim;
        return self.k_cache[layer][0 .. self.position * stride];
    }

    pub fn getV(self: *const SharedKVCache, layer: u32) []const f32 {
        const stride = self.n_kv_heads * self.head_dim;
        return self.v_cache[layer][0 .. self.position * stride];
    }
};

// ── MLA Compressed KV Cache ──────────────────────────────────

/// MLA compressed KV cache: stores compressed KV latents per layer.
/// Each layer's cache is [max_context * kv_lora_rank] f32, indexed by position.
/// Used when attn_kernel == .mla.
pub const MlaKVCache = struct {
    allocator: std.mem.Allocator,
    n_layers: u32,
    kv_lora_rank: u32,
    max_context: u32,

    /// Per-layer flat cache: [layer][max_context * kv_lora_rank]
    caches: [][]f32,
    /// Flat backing storage.
    flat: []f32,
    /// Current sequence position (shared across layers for simplicity).
    position: u32 = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        n_layers: u32,
        kv_lora_rank: u32,
        max_context: u32,
    ) !MlaKVCache {
        const per_layer = max_context * kv_lora_rank;
        const total = n_layers * per_layer;
        const flat = try allocator.alloc(f32, total);
        errdefer allocator.free(flat);
        @memset(flat, 0);

        const caches = try allocator.alloc([]f32, n_layers);
        errdefer allocator.free(caches);
        for (0..n_layers) |l| {
            caches[l] = flat[l * per_layer ..][0..per_layer];
        }

        log.info("MlaKVCache: {d}L × {d}pos × {d}kv_lora = {d} bytes", .{
            n_layers, max_context, kv_lora_rank,
            total * @sizeOf(f32),
        });

        return .{
            .allocator = allocator,
            .n_layers = n_layers,
            .kv_lora_rank = kv_lora_rank,
            .max_context = max_context,
            .caches = caches,
            .flat = flat,
        };
    }

    pub fn deinit(self: *MlaKVCache) void {
        self.allocator.free(self.caches);
        self.allocator.free(self.flat);
    }

    /// Write a compressed KV latent for a given layer at the current position.
    pub fn write(self: *MlaKVCache, layer: u32, latent: []const f32) void {
        const stride: usize = @intCast(self.kv_lora_rank);
        const dst = self.position * @as(u32, @intCast(stride));
        @memcpy(self.caches[layer][dst .. dst + stride], latent[0..stride]);
    }

    /// Read a compressed KV latent for a given layer at a given position.
    pub fn read(self: *const MlaKVCache, layer: u32, pos: u32) []const f32 {
        const stride: usize = @intCast(self.kv_lora_rank);
        const src = pos * @as(u32, @intCast(stride));
        return self.caches[layer][src .. src + stride];
    }

    /// Return the flat slice from position 0 up to `self.position` for a layer.
    pub fn getLatents(self: *const MlaKVCache, layer: u32) []const f32 {
        const stride: usize = @intCast(self.kv_lora_rank);
        return self.caches[layer][0 .. self.position * stride];
    }

    pub fn advance(self: *MlaKVCache, n_tokens: u32) void {
        self.position += n_tokens;
    }

    /// Wrapped as an mla_attn.FlatKvCache view for passing to MLA decode functions.
    pub fn asFlatKvCache(self: *MlaKVCache, layer: u32) mla_attn.FlatKvCache {
        return .{
            .data = self.caches[layer],
            .kv_lora_rank = self.kv_lora_rank,
            .max_seq_len = self.max_context,
            .seq_len = self.position,
        };
    }
};

// ── MoE Dispatch Scratch ─────────────────────────────────────

/// Per-layer scratch buffers for MoE expert FFN dispatch.
/// Reused across layers to minimize allocation churn.
const MoEScratch = struct {
    /// Gating logits: [batch * n_experts] f32
    gating_logits: []f32,
    /// Routing table: [batch * 2 * top_k] u32 (expert_ids + weight_bits)
    routing_table: []u32,
    /// Per-expert output: [batch * top_k * hidden_dim] f32
    /// Layout: expert_output[t * top_k * H + k * H .. (k+1) * H] is expert k's output for token t
    expert_outputs: []f32,
    /// Expert gate/up scratch: [batch * 2 * expert_intermediate_size] f32
    gate_up: []f32,
    /// Expert activated (SiLU*gate): [batch * expert_intermediate_size] f32
    activated: []f32,
    /// Expert down output: [batch * hidden_dim] f32
    down_out: []f32,

    fn deinit(self: *MoEScratch, allocator: std.mem.Allocator) void {
        allocator.free(self.down_out);
        allocator.free(self.activated);
        allocator.free(self.gate_up);
        allocator.free(self.expert_outputs);
        allocator.free(self.routing_table);
        allocator.free(self.gating_logits);
    }
};

// ── NPU subprocess ───────────────────────────────────────────

/// NPU subprocess handle — manages one npu_engine_universal child process.
/// Wire protocol op codes for the persistent NPU worker (npu_engine_universal --worker).
/// Must match `enum WorkerOp` in engine/npu/src/npu_engine_universal.cpp.
const NpuWorkerOp = enum(u32) {
    quit = 0,
    qkv = 1,
    oproj = 2,
    gateup = 3,
    up = 4,
    down = 5,
    /// MoE gate/up for a specific expert. Payload: [batch, hidden_dim] input.
    /// The expert is identified by a separate u32 field (reuses `layer` field).
    moe_gateup = 6,
    /// MoE down projection for a specific expert.
    moe_down = 7,
    /// Shared expert gate/up.
    shared_gateup = 8,
    /// Shared expert down.
    shared_down = 9,
    /// Compressed KV projection (kv_a_proj) for MLA: input [H] → latent [kv_lora_rank].
    mla_kv_compress = 10,
    /// Q projection for MLA: input [H] → q [NH * qk_head_dim].
    mla_q_proj = 11,
    /// MLAAbsorbed Q + output projection (absorbed attention on NPU).
    mla_absorbed_attn = 12,
    /// QKV for ALL layers in one call. Input: NC*B*H floats, Output: NC*B*qkv_total floats.
    qkv_all = 20,
    /// O projection for ALL layers in one call.
    oproj_all = 21,
    /// Gate+Up for ALL layers in one call.
    gateup_all = 22,
    /// Down projection for ALL layers in one call.
    down_all = 23,
};

/// Persistent connection to a single npu_engine_universal --worker child process.
const NpuSubprocess = struct {
    allocator: std.mem.Allocator,
    io: ?Io,
    model_path: []const u8,
    engine_path: []const u8,
    child: ?std.process.Child = null,
    model_tag: []const u8 = "",
    /// forwardDecode() pipelines the NEXT layer's QKV on a background thread
    /// while the current layer's O-proj/FFN/Down run on the caller's thread —
    /// both share this one worker connection, so every request/response
    /// round-trip must be serialized or the two threads interleave their
    /// header/payload bytes on the same pipe.
    mutex: Io.Mutex = .init,

    pub fn init(allocator: std.mem.Allocator, io: ?Io, model_path: []const u8, engine_path: []const u8, model_tag: []const u8) NpuSubprocess {

        return .{ .allocator = allocator, .io = io, .model_path = model_path, .engine_path = engine_path, .model_tag = model_tag };
    }

                                        fn ensureStarted(self: *NpuSubprocess, io: Io) !void {
        if (self.child != null) return;
        self.child = try std.process.spawn(io, .{
            .argv = if (self.model_tag.len > 0) &[_][]const u8{ self.engine_path, self.model_path, "--worker", "--model-tag", self.model_tag } else &[_][]const u8{ self.engine_path, self.model_path, "--worker" },
            .stdin = .pipe,
            .stdout = .pipe,
            .stderr = .pipe,
        });
        // NPU worker takes 5-15s to initialize (XRT, model load, dequant).
        // It then outputs WORKER_READY on stderr and blocks on fread().
        // Just wait a fixed 25s for the worst-case init, then proceed.
        var i: u32 = 0;
        while (i < 75) : (i += 1) {
            var ts = std.os.linux.timespec{ .sec = 0, .nsec = 200_000_000 };
            _ = std.os.linux.nanosleep(&ts, null);
        }
        log.debug("NPU worker assumed ready after ~15s", .{});
    }

fn readExact(file: std.Io.File, io: Io, buf: []u8) !void {
        var got: usize = 0;
        while (got < buf.len) {
            const n = try file.readStreaming(io, &.{buf[got..]});
            if (n == 0) return error.NpuWorkerClosed;
            got += n;
        }
    }

    /// Send one (op, layer, batch) request with `input` as the payload and
    /// block until the response payload is fully read into `output`.
    /// `input.len` must be an exact multiple of `batch`.
    fn call(self: *NpuSubprocess, op: NpuWorkerOp, layer: u32, batch: u32, input: []const f32, output: []f32) !void {
        const io = self.io orelse return error.NoIo;
        try self.mutex.lock(io);
        defer self.mutex.unlock(io);
        try self.ensureStarted(io);
        const child = &self.child.?;
        const in_dim: u32 = @intCast(input.len / batch);

        const hdr = [4]u32{ @intFromEnum(op), layer, batch, in_dim };
        try child.stdin.?.writeStreamingAll(io, std.mem.sliceAsBytes(&hdr));
        try child.stdin.?.writeStreamingAll(io, std.mem.sliceAsBytes(input));

        var resp_hdr: [2]u32 = undefined;
        try readExact(child.stdout.?, io, std.mem.sliceAsBytes(&resp_hdr));
        if (resp_hdr[0] != 0) return error.NpuWorkerError;
        const out_dim = resp_hdr[1];
        const expected = @as(usize, batch) * @as(usize, out_dim);
        if (expected > output.len) return error.NpuOutputTooLarge;
        try readExact(child.stdout.?, io, std.mem.sliceAsBytes(output[0..expected]));
    }

    /// Extended call that sends an extra expert_id field in the header.
    /// Header becomes: [op, expert_id, batch, in_dim]
    fn callWithExpert(
        self: *NpuSubprocess,
        op: NpuWorkerOp,
        expert_id: u32,
        batch: u32,
        input: []const f32,
        output: []f32,
    ) !void {
        const io = self.io orelse return error.NoIo;
        try self.mutex.lock(io);
        defer self.mutex.unlock(io);
        try self.ensureStarted(io);
        const child = &self.child.?;
        const in_dim: u32 = @intCast(input.len / batch);

        const hdr = [4]u32{ @intFromEnum(op), expert_id, batch, in_dim };
        try child.stdin.?.writeStreamingAll(io, std.mem.sliceAsBytes(&hdr));
        try child.stdin.?.writeStreamingAll(io, std.mem.sliceAsBytes(input));

        var resp_hdr: [2]u32 = undefined;
        try readExact(child.stdout.?, io, std.mem.sliceAsBytes(&resp_hdr));
        if (resp_hdr[0] != 0) return error.NpuWorkerError;
        const out_dim = resp_hdr[1];
        const expected = @as(usize, batch) * @as(usize, out_dim);
        if (expected > output.len) return error.NpuOutputTooLarge;
        try readExact(child.stdout.?, io, std.mem.sliceAsBytes(output[0..expected]));
    }

    fn runQKV(self: *NpuSubprocess, input: []const f32, layer: u32, batch_size: u32, qkv_out: []f32) !void {
        try self.call(.qkv, layer, batch_size, input, qkv_out);
    }

    fn runOProj(self: *NpuSubprocess, input: []const f32, layer: u32, batch_size: u32, o_out: []f32) !void {
        try self.call(.oproj, layer, batch_size, input, o_out);
    }

    /// Combined gate+up projection: [B,H] -> [B,2*inter_size]. Only valid for
    /// models without a gate/up weight split (gu_split==false on the C++ side).
    fn runFFN(self: *NpuSubprocess, input: []const f32, layer: u32, batch_size: u32, ffn_out: []f32) !void {
        try self.call(.gateup, layer, batch_size, input, ffn_out);
    }

    fn runDown(self: *NpuSubprocess, input: []const f32, layer: u32, batch_size: u32, down_out: []f32) !void {
        try self.call(.down, layer, batch_size, input, down_out);
    }

    /// Batch operations: process ALL layers in a single NPU call.
    /// Input is NC*B*H floats (all layers concatenated).
    /// Output is NC*B*out_dim floats.
    fn runAllQKV(self: *NpuSubprocess, ctx: []const f32, batch_size: u32, qkv_out: []f32) !void {
        try self.call(.qkv_all, 0, batch_size, ctx, qkv_out);
    }
    fn runAllOProj(self: *NpuSubprocess, ctx: []const f32, batch_size: u32, o_out: []f32) !void {
        try self.call(.oproj_all, 0, batch_size, ctx, o_out);
    }
    fn runAllFFN(self: *NpuSubprocess, ctx: []const f32, batch_size: u32, ffn_out: []f32) !void {
        try self.call(.gateup_all, 0, batch_size, ctx, ffn_out);
    }
    fn runAllDown(self: *NpuSubprocess, ctx: []const f32, batch_size: u32, down_out: []f32) !void {
        try self.call(.down_all, 0, batch_size, ctx, down_out);
    }

    /// MoE: run gate/up for a specific expert.
    fn runMoeGateUp(
        self: *NpuSubprocess,
        input: []const f32,
        expert_id: u32,
        batch_size: u32,
        ffn_out: []f32,
    ) !void {
        self.callWithExpert(.moe_gateup, expert_id, batch_size, input, ffn_out) catch |err| {
            log.warn("NPU MoE gate/up (expert {d}) failed: {s}", .{ expert_id, @errorName(err) });
            @memset(ffn_out, 0);
        };
    }

    /// MoE: run down projection for a specific expert.
    fn runMoeDown(
        self: *NpuSubprocess,
        input: []const f32,
        expert_id: u32,
        batch_size: u32,
        down_out: []f32,
    ) !void {
        self.callWithExpert(.moe_down, expert_id, batch_size, input, down_out) catch |err| {
            log.warn("NPU MoE down (expert {d}) failed: {s}", .{ expert_id, @errorName(err) });
            @memset(down_out, 0);
        };
    }

    /// Shared expert gate+up (used by shared_moe architectures like DeepSeek).
    fn runSharedGateUp(self: *NpuSubprocess, input: []const f32, layer: u32, batch_size: u32, ffn_out: []f32) !void {
        self.call(.shared_gateup, layer, batch_size, input, ffn_out) catch |err| {
            log.warn("NPU shared gate/up (layer {d}) failed: {s}", .{ layer, @errorName(err) });
            @memset(ffn_out, 0);
        };
    }

    /// Shared expert down projection.
    fn runSharedDown(self: *NpuSubprocess, input: []const f32, layer: u32, batch_size: u32, down_out: []f32) !void {
        self.call(.shared_down, layer, batch_size, input, down_out) catch |err| {
            log.warn("NPU shared down (layer {d}) failed: {s}", .{ layer, @errorName(err) });
            @memset(down_out, 0);
        };
    }

    /// MLA KV compression: hidden [H] → compressed latent [kv_lora_rank].
    fn runMlaKvCompress(self: *NpuSubprocess, input: []const f32, layer: u32, batch_size: u32, latent_out: []f32) !void {
        self.call(.mla_kv_compress, layer, batch_size, input, latent_out) catch |err| {
            log.warn("NPU MLA KV compress (layer {d}) failed: {s}", .{ layer, @errorName(err) });
            @memset(latent_out, 0);
        };
    }

    /// MLA Q projection.
    fn runMlaQProj(self: *NpuSubprocess, input: []const f32, layer: u32, batch_size: u32, q_out: []f32) !void {
        self.call(.mla_q_proj, layer, batch_size, input, q_out) catch |err| {
            log.warn("NPU MLA Q proj (layer {d}) failed: {s}", .{ layer, @errorName(err) });
            @memset(q_out, 0);
        };
    }

    /// Signal QUIT and let the child exit; releases the pipe handles.
    fn deinit(self: *NpuSubprocess) void {
        const io = self.io orelse return;
        if (self.child) |*child| {
            const hdr = [4]u32{ @intFromEnum(NpuWorkerOp.quit), 0, 0, 0 };
            child.stdin.?.writeStreamingAll(io, std.mem.sliceAsBytes(&hdr)) catch {};
            _ = child.wait(io) catch {};
        }
        self.child = null;
    }
};

/// RMS normalization (CPU, vectorizable).
fn rmsNorm(input: []const f32, weight: []const f32, output: []f32, eps: f32) void {
    var ss: f64 = 0;
    for (input) |v| ss += @as(f64, @floatCast(v)) * @as(f64, @floatCast(v));
    const inv_rms = 1.0 / @sqrt(@as(f32, @floatCast(ss / @as(f64, @floatFromInt(input.len)))) + eps);
    for (input, weight, output) |v, w, *o| o.* = v * inv_rms * w;
}

/// SiLU activation: x * sigmoid(x)
fn silu(x: f32) f32 {
    return x / (1.0 + std.math.exp(@as(f32, -x)));
}

/// y[out_dim] = W[out_dim, in_dim] @ x[in_dim] (row-major, PyTorch nn.Linear
/// convention — matches the layout `dequantizeI8TensorFromMemory` produces).
fn cpuGemv(w: []const f32, x: []const f32, y: []f32, out_dim: u32, in_dim: u32) void {
    for (0..out_dim) |o| {
        const row = w[o * in_dim ..][0..in_dim];
        var acc: f64 = 0;
        for (0..in_dim) |i| acc += @as(f64, x[i]) * @as(f64, row[i]);
        y[o] = @floatCast(acc);
    }
}

/// Per-layer dequantized projection weights for CPU-side GEMV, one slice per
/// layer, row-major [out_features, in_features]. See ModelData in
/// model_data.zig for the loader that populates these.
pub const CpuLayerWeights = struct {
    q: [][]f32 = &.{},
    k: [][]f32 = &.{},
    v: [][]f32 = &.{},
    o: [][]f32 = &.{},
    gate: [][]f32 = &.{},
    up: [][]f32 = &.{},
    down: [][]f32 = &.{},
};

// ── FusedExecutor ───────────────────────────────────────────

/// Fused execution engine — coordinates NPU GEMM and GPU attention.
/// Supports dense, MoE, and shared-MoE FFNs; flash and MLA attention.
pub const FusedExecutor = struct {
    allocator: std.mem.Allocator,
    policy: DispatchPolicy,
    config: ModelConfig,

    /// Attention kernel type: .flash (default) or .mla.
    attn_kernel: arch_registry.AttentionKernel = .flash,
    /// FFN kernel type: .dense (default), .moe, or .shared_moe.
    ffn_kernel: arch_registry.FfnKernel = .dense,

    /// Set to true when the NPU subprocess dies (BrokenPipe / NpuWorkerError).
    /// Once broken, all remaining operations fall back to CPU.
    npu_broken: bool = false,

    /// Shared KV cache (full K/V for flash attention).
    kv: SharedKVCache,

    /// MLA compressed KV cache (used when attn_kernel == .mla).
    mla_kv: ?MlaKVCache = null,

    /// NPU subprocess handle.
    npu: NpuSubprocess,

    /// GPU attention module (Vulkan flash attention).
    gpu: ?gpu_attn.GpuAttention = null,

    /// Batch split threshold: M <= this uses per-token decode.
    /// Overridable per-instance; defaults to BATCH_SPLIT_THRESHOLD (5).
    batch_split_threshold: u32 = BATCH_SPLIT_THRESHOLD,

    /// MoE decode GEMV threshold: when batch_size <= this value,
    /// use per-assignment GEMV (sort-then-dispatch) rather than
    /// batched fused experts. Default: 8 (Gemma4 decode sweet spot
    /// where per-assignment GEMV beats Triton fused_experts 3-5x).
    /// Set to 0 to always use batched fused.
    /// Only applies when ffn_kernel is .moe or .shared_moe.
    moe_gemv_threshold: u32 = 8,

    /// NPU↔GPU KV cache interop.
    interop: ?interop.KvCacheInterop = null,

    /// Per-layer dequantized projection weights for the `.cpu` dispatch backend
    /// (row-major [out_features, in_features], borrowed from ModelData — not
    /// owned/freed here; ModelData.deinit() frees them).
    cpu_weights: CpuLayerWeights = .{},

    /// f32 embeddings table: [vocab_size, hidden_dim]
    emb_f32: []f32,

    /// f32 lm_head weights: [vocab_size, hidden_dim] or null if tied
    lm_head_f32: ?[]f32 = null,

    /// Tied embeddings flag.
    tied_embeddings: bool = false,

    /// Final norm weights: [hidden_dim]
    final_norm: []f32,

    /// Per-layer norm weights: input_layernorm, post_attention_layernorm
    in_norm: [][]f32,
    pa_norm: [][]f32,

    /// Per-layer Qwen3-style QK-Norm weights (identity 1.0 where absent).
    q_norm: [][]f32,
    k_norm: [][]f32,

    /// RoPE precomputed sin/cos tables (for standard attention).
    rope_sin: []f32,
    rope_cos: []f32,

    /// MLA-specific: configuration (stored for runtime dimension queries).
    mla_config: ?mla_attn.MLAConfig = null,
    /// MLA-specific: precomputed RoPE sin/cos tables for the rope portion.
    mla_rope: ?mla_attn.MLARopeTables = null,
    /// MLA-specific: precomputed absorbed weights.
    mla_absorbed: ?mla_attn.MLAAbsorbedWeights = null,
    /// MLA-specific: per-layer scratch buffers.
    mla_scratch: ?mla_attn.MLAScratch = null,

    /// MoE-specific: configuration.
    moe_config: ?moe_router.MoEConfig = null,
    /// MoE-specific: dispatch context (router + dispatcher + optional cache).
    moe_ctx: ?*moe_router.MoEDispatchContext = null,
    /// MoE-specific: per-layer gating weights [layer][n_experts * hidden_dim].
    moe_gating_weights: [][]f32 = &.{},

    /// Scratch buffers.
    scratch: ScratchBufs,

    /// MoE scratch buffers (lazily initialized when ffn_kernel != .dense).
    moe_scratch: ?MoEScratch = null,

    const ScratchBufs = struct {
        /// Hidden state: [batch, hidden_dim]
        hidden: []f32,
        /// Pre-norm residual save: [batch, hidden_dim]
        residual: []f32,
        /// QKV output: [batch, qkv_total]
        qkv: []f32,
        /// Attention output: [batch, n_heads * head_dim]
        attn_out: []f32,
        /// O projection output: [batch, hidden_dim]
        o_out: []f32,
        /// Gate/Up output: [batch, 2 * inter_size]
        gate_up: []f32,
        /// Down output: [batch, hidden_dim]
        down_out: []f32,
        /// SiLU-activated: [batch, inter_size]
        activated: []f32,
        /// Logits buffer: [vocab_size]
        logits: []f32,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        io: ?Io,
        policy: DispatchPolicy,
        config: ModelConfig,
        model_path: []const u8,
        npu_engine_path: []const u8,
        model_tag: []const u8,
        max_context: u32,
        batch_size: u32,
        emb_f32: []f32,
        lm_head_f32: ?[]f32,
        tied_embeddings: bool,
        final_norm: []f32,
        in_norm: [][]f32,
        pa_norm: [][]f32,
        q_norm: [][]f32,
        k_norm: [][]f32,
        rope_sin: []f32,
        rope_cos: []f32,
        cpu_weights: CpuLayerWeights,
    ) !FusedExecutor {
        return initWithKernels(
            allocator, io, policy, config,
            model_path, npu_engine_path, model_tag, max_context, batch_size,
            emb_f32, lm_head_f32, tied_embeddings,
            final_norm, in_norm, pa_norm, q_norm, k_norm,
            rope_sin, rope_cos, cpu_weights,
            .flash, .dense, null, null, null, null, null, null, null, null,
        );
    }

    /// Extended init with attention/FFN kernel selection and optional
    /// MoE/MLA configuration. When attn_kernel==.flash and ffn_kernel==.dense,
    /// behavior is identical to the plain `init()`.
    pub fn initWithKernels(
        allocator: std.mem.Allocator,
        io: ?Io,
        policy: DispatchPolicy,
        config: ModelConfig,
        model_path: []const u8,
        npu_engine_path: []const u8,
        model_tag: []const u8,
        max_context: u32,
        batch_size: u32,
        emb_f32: []f32,
        lm_head_f32: ?[]f32,
        tied_embeddings: bool,
        final_norm: []f32,
        in_norm: [][]f32,
        pa_norm: [][]f32,
        q_norm: [][]f32,
        k_norm: [][]f32,
        rope_sin: []f32,
        rope_cos: []f32,
        cpu_weights: CpuLayerWeights,
        attn_kernel: arch_registry.AttentionKernel,
        ffn_kernel: arch_registry.FfnKernel,
        moe_config: ?moe_router.MoEConfig,
        moe_gating_weights: ?[][]f32,
        mla_config: ?mla_attn.MLAConfig,
        mla_q_proj: ?[]const f32,
        mla_kv_a_proj: ?[]const f32,
        mla_k_b_proj: ?[]const f32,
        mla_v_b_proj: ?[]const f32,
        mla_o_proj: ?[]const f32,
    ) !FusedExecutor {
        var kv = try SharedKVCache.init(
            allocator, config.n_layers, config.n_kv_heads, config.head_dim, max_context,
        );
        errdefer kv.deinit();

        // MLA compressed KV cache (only when MLA is enabled)
        var mla_kv: ?MlaKVCache = null;
        if (attn_kernel == .mla) {
            const mlac = mla_config orelse return error.MlaConfigRequired;
            mla_kv = try MlaKVCache.init(allocator, config.n_layers, mlac.kv_lora_rank, max_context);
            errdefer if (mla_kv) |*mk| mk.deinit();
        }

        // Scratch buffers for batch decode
        const B = batch_size;
        const H = config.hidden_dim;
        const NH = config.n_heads;
        const HD = config.head_dim;
        const IM = config.inter_size;
        const NV = config.vocab_size;
        const QKV = NH * HD + 2 * config.n_kv_heads * HD; // Q + K + V

        const hidden = try allocator.alloc(f32, B * H);
        errdefer allocator.free(hidden);
        const residual = try allocator.alloc(f32, B * H);
        errdefer allocator.free(residual);
        const qkv = try allocator.alloc(f32, B * QKV);
        errdefer allocator.free(qkv);
        const attn_out = try allocator.alloc(f32, B * NH * HD);
        errdefer allocator.free(attn_out);
        const o_out = try allocator.alloc(f32, B * H);
        errdefer allocator.free(o_out);
        const gate_up = try allocator.alloc(f32, B * 2 * IM);
        errdefer allocator.free(gate_up);
        const down_out = try allocator.alloc(f32, B * H);
        errdefer allocator.free(down_out);
        const activated = try allocator.alloc(f32, B * IM);
        errdefer allocator.free(activated);
        const logits = try allocator.alloc(f32, NV);
        errdefer allocator.free(logits);

        // MoE scratch (only when FFN is MoE or shared_moe)
        var moe_scratch: ?MoEScratch = null;
        if (ffn_kernel != .dense) {
            const mc = moe_config orelse return error.MoeConfigRequired;
            const n_experts = mc.n_experts;
            const top_k = mc.top_k;
            const exp_inter = mc.expert_intermediate_size;

            const gating_logits = try allocator.alloc(f32, B * n_experts);
            errdefer allocator.free(gating_logits);
            const routing_table = try allocator.alloc(u32, B * 2 * top_k);
            errdefer allocator.free(routing_table);
            const expert_outputs = try allocator.alloc(f32, B * top_k * H);
            errdefer allocator.free(expert_outputs);
            const moe_gate_up = try allocator.alloc(f32, B * 2 * exp_inter);
            errdefer allocator.free(moe_gate_up);
            const moe_activated = try allocator.alloc(f32, B * exp_inter);
            errdefer allocator.free(moe_activated);
            const moe_down_out = try allocator.alloc(f32, B * H);
            errdefer allocator.free(moe_down_out);

            moe_scratch = MoEScratch{
                .gating_logits = gating_logits,
                .routing_table = routing_table,
                .expert_outputs = expert_outputs,
                .gate_up = moe_gate_up,
                .activated = moe_activated,
                .down_out = moe_down_out,
            };
        }

        // MoE dispatch context
        var moe_ctx: ?*moe_router.MoEDispatchContext = null;
        if (ffn_kernel != .dense) {
            const mc = moe_config orelse return error.MoeConfigRequired;
            const ctx = try allocator.create(moe_router.MoEDispatchContext);
            ctx.* = try moe_router.MoEDispatchContext.init(
                allocator, mc, config.n_layers, .sort_then_dispatch, 0.1,
            );
            moe_ctx = ctx;
        }

        // MoE gating weights (owned by caller; we just reference them)
        const mgw: [][]f32 = if (ffn_kernel != .dense)
            moe_gating_weights orelse return error.MoeGatingWeightsRequired
        else
            &.{};

        // MLA-specific resources
        var mla_rope: ?mla_attn.MLARopeTables = null;
        var mla_absorbed: ?mla_attn.MLAAbsorbedWeights = null;
        var mla_scratch: ?mla_attn.MLAScratch = null;
        if (attn_kernel == .mla) {
            const mlac = mla_config orelse return error.MlaConfigRequired;

            // RoPE tables for MLA
            mla_rope = try mla_attn.MLARopeTables.init(allocator, &mlac);
            errdefer if (mla_rope) |*mr| mr.deinit(allocator);

            // Absorbed weights
            const mla_weights = mla_attn.MLAWeights{
                .q_proj = mla_q_proj orelse return error.MlaQProjRequired,
                .kv_a_proj = mla_kv_a_proj orelse return error.MlaKvAProjRequired,
                .k_b_proj = mla_k_b_proj orelse return error.MlaKBProjRequired,
                .v_b_proj = mla_v_b_proj orelse return error.MlaVBProjRequired,
                .o_proj = mla_o_proj orelse return error.MlaOProjRequired,
            };
            mla_absorbed = try mla_attn.buildAbsorbedWeights(allocator, &mlac, &mla_weights);
            errdefer if (mla_absorbed) |*ma| ma.deinit(allocator);

            // Per-layer scratch
            mla_scratch = try mla_attn.MLAScratch.init(allocator, &mlac, max_context);
            errdefer if (mla_scratch) |*ms| ms.deinit(allocator);
        }

        log.info("FusedExecutor init: policy={s} model=H{d}L{d}NH{d}NKV{d}HD{d}IM{d}NV{d} attn={s} ffn={s}", .{
            @tagName(policy), H, config.n_layers, NH, config.n_kv_heads, HD, IM, NV,
            @tagName(attn_kernel), @tagName(ffn_kernel),
        });

        return FusedExecutor{
            .allocator = allocator,
            .policy = policy,
            .config = config,
            .attn_kernel = attn_kernel,
            .ffn_kernel = ffn_kernel,
            .kv = kv,
            .mla_kv = mla_kv,
            .npu = NpuSubprocess.init(allocator, io, model_path, npu_engine_path, model_tag),
            .cpu_weights = cpu_weights,
            .emb_f32 = emb_f32,
            .lm_head_f32 = lm_head_f32,
            .tied_embeddings = tied_embeddings,
            .final_norm = final_norm,
            .in_norm = in_norm,
            .pa_norm = pa_norm,
            .q_norm = q_norm,
            .k_norm = k_norm,
            .rope_sin = rope_sin,
            .rope_cos = rope_cos,
            .mla_rope = mla_rope,
            .mla_config = mla_config,
            .mla_absorbed = mla_absorbed,
            .mla_scratch = mla_scratch,
            .moe_config = moe_config,
            .moe_ctx = moe_ctx,
            .moe_gating_weights = mgw,
            .moe_gemv_threshold = if (moe_config) |mc| mc.gemv_bs_threshold else 8,
            .scratch = ScratchBufs{
                .hidden = hidden,
                .residual = residual,
                .qkv = qkv,
                .attn_out = attn_out,
                .o_out = o_out,
                .gate_up = gate_up,
                .down_out = down_out,
                .activated = activated,
                .logits = logits,
            },
            .moe_scratch = moe_scratch,
        };
    }

    pub fn deinit(self: *FusedExecutor) void {
        self.npu.deinit();
        const aa = self.allocator;

        // Deinit MLA resources
        if (self.mla_scratch) |*ms| ms.deinit(aa);
        if (self.mla_absorbed) |*ma| ma.deinit(aa);
        if (self.mla_rope) |*mr| mr.deinit(aa);
        if (self.mla_kv) |*mk| mk.deinit();

        // Deinit MoE resources
        if (self.moe_ctx) |ctx| {
            ctx.deinit();
            aa.destroy(ctx);
        }
        if (self.moe_scratch) |*ms| ms.deinit(aa);

        // Deinit scratch
        aa.free(self.scratch.logits);
        aa.free(self.scratch.activated);
        aa.free(self.scratch.down_out);
        aa.free(self.scratch.gate_up);
        aa.free(self.scratch.o_out);
        aa.free(self.scratch.attn_out);
        aa.free(self.scratch.qkv);
        aa.free(self.scratch.residual);
        aa.free(self.scratch.hidden);
        if (self.gpu) |*g| g.deinit();
        self.kv.deinit();
        // NOTE: emb_f32, lm_head_f32, final_norm, in_norm, pa_norm, q_norm,
        // k_norm, rope_sin, rope_cos and cpu_weights are all BORROWED from
        // ModelData (see the field docs above). ModelData.deinit() is the
        // single owner and frees them (plus the per-layer projection weights
        // and inner norm slices, which were never freed here). Freeing them
        // here as well double-freed the shared buffers and leaked everything
        // else (#144, #113). Do not re-add frees for borrowed buffers.
    }

    /// Try to initialize the GPU attention backend.
    /// Returns true if successful, false if GPU is unavailable (graceful fallback).
    pub fn tryInitGpu(self: *FusedExecutor, shader_dir: []const u8) !bool {
        const gpu = gpu_attn.GpuAttention.init(self.allocator, shader_dir) catch |err| {
            log.warn("GPU attention unavailable: {s}. Falling back to CPU attention.", .{@errorName(err)});
            return false;
        };
        self.gpu = gpu;
        log.info("GPU attention initialized (Vulkan flash attention)", .{});
        return true;
    }

    /// Get dispatch for a layer based on policy.
    fn getLayerDispatch(self: *const FusedExecutor, layer: u32) LayerDispatch {
        _ = layer;
        if (self.policy == .auto) {
            const npu_up = !self.npu_broken;
            const gpu_up = self.gpu != null;
            // Each op only falls back to an accelerator that has a real
            // implementation for it, never to one that's still a stub --
            // GPU QKV/FFN are a zero-fill and a pass-through respectively
            // (no GEMV kernel yet), and NPU has no attention dispatch op
            // at all (stale output, not just slow). Falling back to those
            // would silently compute wrong results, which is worse than
            // the guaranteed-correct CPU path. Revisit once real GPU
            // QKV/FFN GEMV and NPU attention dispatch exist.
            return .{
                .qkv = if (npu_up) .npu else .cpu,
                .attention = if (gpu_up) .gpu else .cpu,
                .ffn = if (npu_up) .npu else .cpu,
            };
        }
        // When NPU subprocess is dead, fall back to CPU for everything.
        if (self.npu_broken) {
            return .{ .qkv = .cpu, .attention = .cpu, .ffn = .cpu };
        }
        return switch (self.policy) {
            .npu_only => .{ .qkv = .npu, .attention = .npu, .ffn = .npu },
            .gpu_only => .{ .qkv = .gpu, .attention = .gpu, .ffn = .gpu },
            .cpu_only => .{ .qkv = .cpu, .attention = .cpu, .ffn = .cpu },
            .ffn_on_npu => .{ .qkv = .npu, .attention = .gpu, .ffn = .npu },
            .qkv_on_npu => .{ .qkv = .npu, .attention = .gpu, .ffn = .gpu },
            .attention_on_npu => .{ .qkv = .gpu, .attention = .npu, .ffn = .gpu },
            .prefill_npu_decode_gpu => .{ .qkv = .npu, .attention = .gpu, .ffn = .npu },
            .auto => unreachable, // handled above
        };
    }

    /// Apply RoPE to a Q or K vector in place.
    fn applyRoPE(self: *const FusedExecutor, x: []f32, pos: u32, head_dim: u32) void {
        const hd2 = head_dim / 2;
        for (0..hd2) |d| {
            const a = x[d];
            const b = x[d + hd2];
            const c = self.rope_cos[pos * head_dim + d];
            const s = self.rope_sin[pos * head_dim + d];
            x[d] = a * c - b * s;
            x[d + hd2] = b * c + a * s;
        }
    }

    // ── Dense FFN helpers (existing) ─────────────────────────

    /// Run standard dense FFN: gate/up → SiLU*up → down, using NPU.
    fn runDenseFfn(
        self: *FusedExecutor,
        layer: u32,
        batch_size: u32,
        input: []f32,
        residual: []f32,
        dispatch: LayerDispatch,
    ) void {
        const H = self.config.hidden_dim;
        const IM = self.config.inter_size;
        const s = self.scratch;
        const cpu_ready = layer < self.cpu_weights.gate.len and layer < self.cpu_weights.up.len;

        if (dispatch.ffn == .cpu and cpu_ready) {
            for (0..batch_size) |b| {
                const x = input[b * H ..][0..H];
                cpuGemv(self.cpu_weights.gate[layer], x, s.gate_up[b * 2 * IM ..][0..IM], IM, H);
                cpuGemv(self.cpu_weights.up[layer], x, s.gate_up[b * 2 * IM + IM ..][0..IM], IM, H);
            }
        } else {
            tryOrZero: {
                self.npu.runFFN(input[0..batch_size * H], layer, batch_size, s.gate_up) catch |err| {
                    log.warn("NPU gate/up (layer {d}) failed: {s} - falling back to CPU for remaining layers", .{ layer, @errorName(err) });
                    self.npu_broken = true;
                    break :tryOrZero;
                };
            }
        }
        for (0..batch_size) |b| {
            for (0..IM) |i| {
                const gate = s.gate_up[b * 2 * IM + i];
                const up = s.gate_up[b * 2 * IM + IM + i];
                const g = if (std.math.isFinite(gate)) gate else 0.0;
                s.activated[b * IM + i] = silu(g) * up;
            }
        }
        if (dispatch.ffn == .cpu and layer < self.cpu_weights.down.len) {
            for (0..batch_size) |b| {
                cpuGemv(self.cpu_weights.down[layer], s.activated[b * IM ..][0..IM], s.down_out[b * H ..][0..H], H, IM);
            }
        } else {
            tryOrZero: {
                self.npu.runDown(s.activated[0..batch_size * IM], layer, batch_size, s.down_out[0..batch_size * H]) catch |err| {
                    log.warn("NPU down (layer {d}) failed: {s} - falling back to CPU for remaining layers", .{ layer, @errorName(err) });
                    self.npu_broken = true;
                    break :tryOrZero;
                };
            }
        }

        // Residual add
        for (0..batch_size * H) |i| input[i] = residual[i] + s.down_out[i];
    }

    // ── MoE FFN helpers ──────────────────────────────────────

    /// Compute gating logits: hidden @ gating_weight^T  → [batch * n_experts]
    fn computeGatingLogits(
        self: *const FusedExecutor,
        hidden: []const f32,
        layer: u32,
        batch_size: u32,
        logits_out: []f32,
    ) void {
        const H = self.config.hidden_dim;
        const mc = self.moe_config orelse return;
        const n_experts = mc.n_experts;
        const gate_w = self.moe_gating_weights[layer];

        @memset(logits_out[0..batch_size * n_experts], 0);
        for (0..batch_size) |b| {
            const h_row = hidden[b * H ..][0..H];
            for (0..n_experts) |e| {
                var dot: f32 = 0;
                for (0..H) |i| {
                    dot += h_row[i] * gate_w[e * H + i];
                }
                logits_out[b * n_experts + e] = dot;
            }
        }
    }

    /// Run MoE FFN for one layer:
    ///   1. Compute gating logits from hidden state
    ///   2. Route tokens to experts via MoEDispatchContext
    ///   3. For each expert that received tokens, run gate/up → SiLU → down
    ///   4. Weighted accumulate: out[t] += sum_k(weight[t,k] * expert_out[t,k])
    ///
    /// Per-assignment GEMV (Gemma4 MoE decode pattern):
    ///   When batch_size <= moe_gemv_threshold, tokens are grouped by expert
    ///   (sort-then-dispatch), and each expert processes a contiguous block of
    ///   tokens with one GEMV call. This avoids per-token dispatch overhead.
    fn runMoeFfn(
        self: *FusedExecutor,
        layer: u32,
        batch_size: u32,
        input: []f32,
        residual: []f32,
    ) void {
        const H = self.config.hidden_dim;
        const mc = self.moe_config orelse return;
        const n_experts = mc.n_experts;
        const top_k = mc.top_k;
        const exp_inter = mc.expert_intermediate_size;
        const has_shared = mc.has_shared_expert;
        const ms = self.moe_scratch orelse return;
        const ctx = self.moe_ctx orelse return;

        // Select GEMV strategy based on batch size
        const gemv_strategy = mc.selectGemvStrategy(batch_size);

        // ── Step 1: Compute gating logits ──
        self.computeGatingLogits(input, layer, batch_size, ms.gating_logits);

        // ── Step 2: Route via MoEDispatchContext ──
        const routing = ctx.route(ms.gating_logits, batch_size, layer) catch {
            log.warn("MoE routing failed at layer {d}, zeroing output", .{layer});
            @memset(ms.expert_outputs[0..batch_size * top_k * H], 0);
            return;
        };
        const dispatch = ctx.prepareDispatch(routing) catch {
            log.warn("MoE dispatch failed at layer {d}, zeroing output", .{layer});
            @memset(ms.expert_outputs[0..batch_size * top_k * H], 0);
            return;
        };

        defer {
            self.allocator.free(dispatch.per_expert);
            if (dispatch.order) |_o| {
                var mut = _o;
                mut.deinit(self.allocator);
            }
        }

        // ── Step 3: Clear expert output accumulator ──
        @memset(ms.expert_outputs[0..batch_size * top_k * H], 0);

        // ── Step 4: Dispatch and compute expert FFN ──
        if (gemv_strategy == .per_assignment_gemv and dispatch.order != null) {
            // ═══════════════════════════════════════════════════════
            //  Per-assignment GEMV (sort-then-dispatch)
            //  Gemma4 MoE decode pattern: BS <= 8
            //  One GEMV launch per expert, contiguous token blocks
            // ═══════════════════════════════════════════════════════
            const order = dispatch.order.?;
            for (0..n_experts) |e| {
                const cnt = order.expertCount(@intCast(e));
                if (cnt == 0) continue;

                const items = order.expertItems(@intCast(e));

                // ── Tightly-packed gather ──
                // Pack input H-dim vectors contiguously at stride H.
                // This is critical: NPU gateway expects contiguous data
                // [cnt * H], NOT interleaved at stride 2*exp_inter.
                for (0..cnt) |ti| {
                    const token_idx = items[ti] >> 16;
                    const src = input[token_idx * H ..][0..H];
                    const dst = ms.gate_up[ti * H ..][0..H];
                    @memcpy(dst, src);
                }

                // ── NPU gate/up (contiguous) ──
                // Input:  [cnt * H]  tightly packed at ms.gate_up[0..cnt*H]
                // Output: [cnt * 2*exp_inter] at ms.gate_up[0..cnt*2*exp_inter]
                tryOrZero: {
                    self.npu.runMoeGateUp(
                        ms.gate_up[0..cnt * H],
                        @intCast(e),
                        cnt,
                        ms.gate_up[0..cnt * 2 * exp_inter],
                    ) catch |err| {
                        log.warn("NPU MoE gate/up expert {d} failed: {s}", .{ e, @errorName(err) });
                        break :tryOrZero;
                    };
                }

                // ── SiLU(gate) * up (contiguous output) ──
                for (0..cnt) |t| {
                    const base = t * 2 * exp_inter;
                    for (0..exp_inter) |i| {
                        const gate = ms.gate_up[base + i];
                        const up_val = ms.gate_up[base + exp_inter + i];
                        const g = if (std.math.isFinite(gate)) gate else 0.0;
                        ms.activated[t * exp_inter + i] = silu(g) * up_val;
                    }
                }

                // ── NPU down (contiguous) ──
                // Input:  [cnt * exp_inter]  at ms.activated[0..cnt*exp_inter]
                // Output: [cnt * H]          at ms.down_out[0..cnt*H]
                tryOrZero: {
                    self.npu.runMoeDown(
                        ms.activated[0..cnt * exp_inter],
                        @intCast(e),
                        cnt,
                        ms.down_out[0..cnt * H],
                    ) catch |err| {
                        log.warn("NPU MoE down expert {d} failed: {s}", .{ e, @errorName(err) });
                        break :tryOrZero;
                    };
                }

                // ── Scatter back from contiguous output to per-token slots ──
                for (0..cnt) |ti| {
                    const pitem = items[ti];
                    const token_idx = pitem >> 16;
                    const slot_idx = pitem & 0xFFFF;

                    const src = ms.down_out[ti * H ..][0..H];
                    const dst_offset = token_idx * top_k * H + slot_idx * H;
                    @memcpy(ms.expert_outputs[dst_offset .. dst_offset + H], src);
                }
            }
        } else {
            // ═══════════════════════════════════════════════════════
            //  Batched: one expert dispatch per (token, slot) pair
            //  Falls back to per-token dispatch order if available,
            //  otherwise iterates routing table directly.
            // ═══════════════════════════════════════════════════════
            for (dispatch.per_expert) |pe| {
                const e = pe.expert_id;
                const cnt = pe.token_count;
                if (cnt == 0) continue;

                // ── Tightly-packed gather at stride H ──
                for (0..cnt) |ti| {
                    const token_idx = pe.token_indices[ti];
                    const src = input[token_idx * H ..][0..H];
                    const dst = ms.gate_up[ti * H ..][0..H];
                    @memcpy(dst, src);
                }

                // ── NPU gate/up (contiguous) ──
                tryOrZero: {
                    self.npu.runMoeGateUp(
                        ms.gate_up[0..cnt * H],
                        e,
                        cnt,
                        ms.gate_up[0..cnt * 2 * exp_inter],
                    ) catch |err| {
                        log.warn("NPU MoE gate/up expert {d} failed: {s}", .{ e, @errorName(err) });
                        break :tryOrZero;
                    };
                }

                // ── SiLU(gate) * up ──
                for (0..cnt) |t| {
                    const base = t * 2 * exp_inter;
                    for (0..exp_inter) |i| {
                        const gate = ms.gate_up[base + i];
                        const up_val = ms.gate_up[base + exp_inter + i];
                        ms.activated[t * exp_inter + i] = silu(if (std.math.isFinite(gate)) gate else 0) * up_val;
                    }
                }

                // ── NPU down (contiguous) ──
                tryOrZero: {
                    self.npu.runMoeDown(
                        ms.activated[0..cnt * exp_inter],
                        e,
                        cnt,
                        ms.down_out[0..cnt * H],
                    ) catch |err| {
                        log.warn("NPU MoE down expert {d} failed: {s}", .{ e, @errorName(err) });
                        break :tryOrZero;
                    };
                }

                // ── Scatter with slot lookup via routing table ──
                for (0..cnt) |ti| {
                    const token_idx = pe.token_indices[ti];
                    // Find which slot this expert occupies for this token
                    var slot: u32 = 0;
                    for (0..top_k) |k| {
                        if (routing.expertId(token_idx, @intCast(k)) == e) {
                            slot = @intCast(k);
                            break;
                        }
                    }
                    const src = ms.down_out[ti * H ..][0..H];
                    const dst_offset = token_idx * top_k * H + slot * H;
                    @memcpy(ms.expert_outputs[dst_offset .. dst_offset + H], src);
                }
            }
        }

        // ── Step 5: Weighted accumulate ──
        @memset(input[0..batch_size * H], 0);
        for (0..batch_size) |t| {
            const tH = t * H;
            for (0..top_k) |k| {
                const w = routing.weight(@intCast(t), @intCast(k));
                if (w == 0.0) continue;
                const src = ms.expert_outputs[tH + k * H ..][0..H];
                for (0..H) |i| input[tH + i] += w * src[i];
            }
        }

        // ── Step 6: Add shared expert output if present ──
        if (has_shared) {
            // Run shared expert gate/up (full batch, contiguously)
            tryOrZero: {
                self.npu.runSharedGateUp(input[0..batch_size * H], layer, batch_size, ms.gate_up[0..batch_size * 2 * exp_inter]) catch |err| {
                    log.warn("NPU shared gate/up (layer {d}) failed: {s}", .{ layer, @errorName(err) });
                    break :tryOrZero;
                };
            }
            for (0..batch_size) |b| {
                const base = b * 2 * exp_inter;
                for (0..exp_inter) |i| {
                    const gate = ms.gate_up[base + i];
                    const up_val = ms.gate_up[base + exp_inter + i];
                    ms.activated[b * exp_inter + i] = silu(if (std.math.isFinite(gate)) gate else 0) * up_val;
                }
            }
            tryOrZero: {
                self.npu.runSharedDown(ms.activated[0..batch_size * exp_inter], layer, batch_size, ms.down_out[0..batch_size * H]) catch |err| {
                    log.warn("NPU shared down (layer {d}) failed: {s}", .{ layer, @errorName(err) });
                    break :tryOrZero;
                };
            }
            // Add shared expert output to the accumulated routed output
            for (0..batch_size * H) |i| input[i] += ms.down_out[i];
        }

        // ── Step 7: Final residual add (FFN) ──
        for (0..batch_size * H) |i| input[i] = residual[i] + input[i];
    }

    // ── MLA attention helpers ────────────────────────────────

    /// Run MLA attention for a batch of tokens (decode step).
    /// Compresses K/V, writes to MLA KV cache, runs absorbed attention,
    /// then O projection via NPU.
    fn runMlaAttention(
        self: *FusedExecutor,
        layer: u32,
        batch_size: u32,
        qkv_scratch: []f32,
        output_hidden: []f32,
    ) void {
        const H = self.config.hidden_dim;
        const mlac = self.mla_config orelse return;

        var mk = self.mla_kv orelse return;
        const absorbed = self.mla_absorbed orelse return;
        const rope_tables = self.mla_rope orelse return;
        const mla_scratch = self.mla_scratch orelse return;

        // For MLA, we use the NPU for Q projection and KV compression.
        // The absorbed attention runs on NPU via mla_absorbed_attn op.

        // Q projection on NPU for all tokens
        tryOrZero: {
            self.npu.runMlaQProj(
                qkv_scratch[0..batch_size * H],
                layer,
                batch_size,
                mla_scratch.q,
            ) catch |err| {
                log.warn("NPU MLA Q proj (layer {d}) failed: {s}", .{ layer, @errorName(err) });
                break :tryOrZero;
            };
        }

        {
            const mkp = &mk;
            for (0..batch_size) |b| {
                const pos = mkp.position + @as(u32, @intCast(b));

                var latent: [512]f32 = [_]f32{0} ** 512;
                const kv_lora2 = mlac.kv_lora_rank;

                tryOrZero: {
                    self.npu.runMlaKvCompress(
                        qkv_scratch[b * H ..][0..H],
                        layer,
                        1,
                        latent[0..kv_lora2],
                    ) catch |err| {
                        log.warn("NPU MLA KV compress (layer {d}) failed: {s}", .{ layer, @errorName(err) });
                        break :tryOrZero;
                    };
                }

                mkp.write(layer, latent[0..kv_lora2]);

                // Build a temporary FlatKvCache view for this layer for future use.
                const flat_cache_view = mkp.asFlatKvCache(layer);
                _ = flat_cache_view;
                _ = absorbed;
                _ = rope_tables;

                @memset(output_hidden[b * H ..][0..H], 0);
                log.warn("MLA CPU decode not yet implemented at layer {d}, pos {d}. " ++
                    "Use NPU MLA kernels for production.", .{ layer, pos });
            }
        }
    }

    // ── Layer execution ──────────────────────────────────────

    /// Phase 1: RMSNorm + QKV projection for one layer.
    /// For MLA, this step compresses K/V and projects Q.
    /// Writes QKV output into `qkv_scratch` (caller-provided buffer for double-buffering).
    pub fn executeLayerQKV(
        self: *FusedExecutor,
        layer: u32,
        batch_size: u32,
        qkv_scratch: []f32,
    ) !void {
        const H = self.config.hidden_dim;
        const s = self.scratch;
        const dispatch = self.getLayerDispatch(layer);

        // Save pre-norm residual
        @memcpy(s.residual[0..batch_size * H], s.hidden[0..batch_size * H]);

        // RMSNorm
        for (0..batch_size) |b| {
            rmsNorm(
                s.hidden[b * H ..][0..H],
                self.in_norm[layer],
                s.hidden[b * H ..][0..H],
                1e-6,
            );
        }

        // For MLA, QKV is handled differently: we just need Q projection
        // (KV compression happens in the attention step).
        // For now, route to NPU QKV (standard path) or MLA Q proj (NPU).
        if (self.attn_kernel == .mla) {
            // MLA: Q projection only (standard q_proj on NPU).
            // We reuse the NPU's MLA Q proj op. The qkv_scratch stores
            // the post-norm hidden state for downstream KV compression.
            if (dispatch.qkv == .npu) {
                // Copy normed hidden into qkv_scratch for later KV compression
                @memcpy(qkv_scratch[0..batch_size * H], s.hidden[0..batch_size * H]);
            }
        } else if ((dispatch.qkv == .cpu or dispatch.qkv == .gpu) and layer < self.cpu_weights.q.len and layer < self.cpu_weights.k.len and layer < self.cpu_weights.v.len) {
            // CPU fallback for both .cpu and .gpu dispatch (no GPU QKV kernel yet — fix #56)
            const NH = self.config.n_heads;
            const NKV = self.config.n_kv_heads;
            const HD = self.config.head_dim;
            const QKV = NH * HD + 2 * NKV * HD;
            for (0..batch_size) |b| {
                const x = s.hidden[b * H ..][0..H];
                const qkv_slice = qkv_scratch[b * QKV ..][0..QKV];
                cpuGemv(self.cpu_weights.q[layer], x, qkv_slice[0 .. NH * HD], NH * HD, H);
                cpuGemv(self.cpu_weights.k[layer], x, qkv_slice[NH * HD ..][0 .. NKV * HD], NKV * HD, H);
                cpuGemv(self.cpu_weights.v[layer], x, qkv_slice[NH * HD + NKV * HD ..][0 .. NKV * HD], NKV * HD, H);
            }
        } else if (dispatch.qkv == .npu) {
            // Standard flash attention: full QKV projection on NPU
            try self.npu.runQKV(s.hidden[0..batch_size * H], layer, @intCast(batch_size), qkv_scratch);
        } else {
            // No backend matched this QKV dispatch and no CPU weights loaded.
            // Zeroing here would silently corrupt output (#145) — make it loud.
            log.warn("QKV dispatch {any} unhandled at layer {d} (cpu_weights loaded={any}); zeroing — output will be degenerate", .{ dispatch.qkv, layer, layer < self.cpu_weights.q.len });
            @memset(qkv_scratch, 0);
        }
    }

    /// Phase 2: Q/K norm, RoPE, KV cache write, Attention (GPU/CPU/MLA),
    /// O projection, residual add, FFN, residual add.
    /// Execute attention only (no FFN). Used by batch NPU decode path.
    pub fn executeLayerAttnOnly(
        self: *FusedExecutor,
        layer: u32,
        batch_size: u32,
        qkv_scratch: []f32,
        output_hidden: []f32,
    ) !void {
        _ = output_hidden;
        const H = self.config.hidden_dim;
        const s = self.scratch;
        @memcpy(s.residual[0..batch_size * H], s.hidden[0..batch_size * H]);

        const NH = self.config.n_heads;
        const NKV = self.config.n_kv_heads;
        const HD = self.config.head_dim;
        const GQA = self.config.gqa_ratio;
        const dispatch = self.getLayerDispatch(layer);
        const pos = self.kv.position;

        // RoPE, QK norm, KV cache write
        const QKV = NH * HD + 2 * NKV * HD;
        for (0..batch_size) |b| {
            var qkv_slice: []f32 = qkv_scratch[b * QKV ..][0..QKV];
            // Q head loop
            for (0..NH) |hh| {
                var qh_buf: [256]f32 = undefined;
                for (0..HD) |di| qh_buf[di] = qkv_slice[hh * HD + di];
                var sq: f64 = 0;
                for (0..HD) |di| sq += @as(f64, @floatCast(qh_buf[di])) * @as(f64, @floatCast(qh_buf[di]));
                const iq = 1.0 / @sqrt(@as(f32, @floatCast(sq / @as(f64, @floatFromInt(HD)))) + 1e-6);
                if (layer < self.q_norm.len and self.q_norm[layer].len >= HD) {
                    for (0..HD) |di| {
                        qh_buf[di] = qh_buf[di] * iq * self.q_norm[layer][di];
                    }
                }
                self.applyRoPE(qh_buf[0..HD], pos + @as(u32, @intCast(b)), HD);
                // Write back modified Q
                for (0..HD) |di| qkv_slice[hh * HD + di] = qh_buf[di];
            }
            // K head loop
            for (0..NKV) |kvh| {
                var ks_buf: [256]f32 = undefined;
                for (0..HD) |di| ks_buf[di] = qkv_slice[NH * HD + kvh * HD + di];
                var sk: f64 = 0;
                for (0..HD) |di| sk += @as(f64, @floatCast(ks_buf[di])) * @as(f64, @floatCast(ks_buf[di]));
                const ik = 1.0 / @sqrt(@as(f32, @floatCast(sk / @as(f64, @floatFromInt(HD)))) + 1e-6);
                if (layer < self.k_norm.len and self.k_norm[layer].len >= HD) {
                    for (0..HD) |di| {
                        ks_buf[di] = ks_buf[di] * ik * self.k_norm[layer][di];
                    }
                }
                self.applyRoPE(ks_buf[0..HD], pos + @as(u32, @intCast(b)), HD);
                const dst = (pos + @as(u32, @intCast(b))) * NKV * HD + kvh * HD;
                for (0..HD) |di| self.kv.k_cache[layer][dst + di] = ks_buf[di];
                // Write back modified K to qkv_slice (so O proj sees post-RoPE K)
                for (0..HD) |di| qkv_slice[NH * HD + kvh * HD + di] = ks_buf[di];
            }
            // V copy
            for (0..NKV) |kvh| {
                const dst = (pos + @as(u32, @intCast(b))) * NKV * HD + kvh * HD;
                for (0..HD) |di| self.kv.v_cache[layer][dst + di] = qkv_slice[NH * HD + NKV * HD + kvh * HD + di];
            }
        }

        // Flash attention (GPU or CPU)
        const QKV2 = NH * HD + 2 * NKV * HD;
        const seq_len = self.kv.position + batch_size;
        if (dispatch.attention == .gpu) {
            if (self.gpu) |*gpu| {
                for (0..batch_size) |b| {
                    const q_slice = qkv_scratch[b * QKV2 ..][0..NH * HD];
                    const out_slice = s.attn_out[b * NH * HD ..][0..NH * HD];
                    gpu.flashAttention(q_slice, self.kv.k_cache[layer], self.kv.v_cache[layer],
                        &.{}, out_slice, &([_]f32{std.math.nan(f32)} ** 12),
                        NH, NKV, HD, seq_len, 0, 0.0, 0) catch |err| {
                        log.warn("GPU attn layer {d} failed: {s}", .{layer, @errorName(err)});
                        self.cpuAttention(q_slice, out_slice, layer, seq_len, NH, NKV, HD, GQA);
                    };
                }
            } else {
                for (0..batch_size) |b| {
                    self.cpuAttention(qkv_scratch[b * QKV2 ..][0..NH * HD],
                        s.attn_out[b * NH * HD ..][0..NH * HD], layer, seq_len, NH, NKV, HD, GQA);
                }
            }
        } else {
            for (0..batch_size) |b| {
                self.cpuAttention(qkv_scratch[b * QKV2 ..][0..NH * HD],
                    s.attn_out[b * NH * HD ..][0..NH * HD], layer, seq_len, NH, NKV, HD, GQA);
            }
        }

        // O projection
        if ((dispatch.attention == .cpu or dispatch.attention == .gpu) and layer < self.cpu_weights.o.len) {
            for (0..batch_size) |b| {
                cpuGemv(self.cpu_weights.o[layer], s.attn_out[b * NH * HD ..][0..NH * HD],
                    s.o_out[b * H ..][0..H], H, NH * HD);
            }
        } else if (dispatch.attention == .npu) {
            self.npu.runOProj(s.attn_out[0..batch_size * NH * HD], layer, batch_size, s.o_out[0..batch_size * H]) catch {};
        }

        // Residual add (attention only, no FFN)
        for (0..batch_size * H) |i| s.hidden[i] = s.residual[i] + s.o_out[i];
    }

    pub fn executeLayerAttnFFN(
        self: *FusedExecutor,
        layer: u32,
        batch_size: u32,
        qkv_scratch: []f32,
        output_hidden: []f32,
    ) !void {
        const H = self.config.hidden_dim;
        const s = self.scratch;

        // ── Branch on attention kernel type ──
        if (self.attn_kernel == .mla) {
            // ── MLA attention path ──
            // The qkv_scratch holds the RMSNorm'd hidden state from executeLayerQKV.
            // Run compressed KV + absorbed attention.
            self.runMlaAttention(layer, batch_size, qkv_scratch, s.attn_out);

            // O projection (NPU) + residual add (attention)
            const mla_nheads = self.config.n_heads;
            const mla_hdim = self.config.head_dim;
            try self.npu.runOProj(s.attn_out[0..batch_size * mla_nheads * mla_hdim], layer, batch_size, s.o_out[0..batch_size * H]);
            for (0..batch_size * H) |i| s.hidden[i] = s.residual[i] + s.o_out[i];

            // Pre-FFN residual save + RMSNorm
            @memcpy(s.residual[0..batch_size * H], s.hidden[0..batch_size * H]);
            for (0..batch_size) |b| {
                rmsNorm(s.hidden[b * H ..][0..H], self.pa_norm[layer], s.hidden[b * H ..][0..H], 1e-6);
            }

            // ── Branch on FFN kernel type ──
            switch (self.ffn_kernel) {
                .dense => {
                    self.runDenseFfn(layer, batch_size, s.hidden, s.residual, self.getLayerDispatch(layer));
                },
                .moe, .shared_moe => {
                    self.runMoeFfn(layer, batch_size, s.hidden, s.residual);
                },
            }

            // Copy output
            if (output_hidden.ptr != s.hidden.ptr) {
                @memcpy(output_hidden[0..batch_size * H], s.hidden[0..batch_size * H]);
            }
            return;
        }

        // ── Standard flash attention path (fully backward-compatible) ──
        const NH = self.config.n_heads;
        const NKV = self.config.n_kv_heads;
        const HD = self.config.head_dim;
        const GQA = self.config.gqa_ratio;
        const dispatch = self.getLayerDispatch(layer);

        // Q/K norm, RoPE, KV cache write
        const QKV = NH * HD + 2 * NKV * HD;
        const pos = self.kv.position;
        for (0..batch_size) |b| {
            const qkv_slice = qkv_scratch[b * QKV ..][0..QKV];
            for (0..NH) |hh| {
                const qh = qkv_slice[hh * HD ..][0..HD];
                var sq: f64 = 0;
                for (qh) |v| sq += @as(f64, @floatCast(v)) * @as(f64, @floatCast(v));
                const iq = 1.0 / @sqrt(@as(f32, @floatCast(sq / @as(f64, @floatFromInt(HD)))) + 1e-6);
                if (layer < self.q_norm.len and self.q_norm[layer].len >= HD) {
                            for (0..HD) |d| qh[d] = qh[d] * iq * self.q_norm[layer][d];
                        }
                self.applyRoPE(qh, pos + @as(u32, @intCast(b)), HD);
            }
            for (0..NKV) |kvh| {
                const ks = qkv_slice[NH * HD + kvh * HD ..][0..HD];
                var sk: f64 = 0;
                for (ks) |v| sk += @as(f64, @floatCast(v)) * @as(f64, @floatCast(v));
                const ik = 1.0 / @sqrt(@as(f32, @floatCast(sk / @as(f64, @floatFromInt(HD)))) + 1e-6);
                if (layer < self.k_norm.len and self.k_norm[layer].len >= HD) {
                            for (0..HD) |d| ks[d] = ks[d] * ik * self.k_norm[layer][d];
                        }
                self.applyRoPE(ks, pos + @as(u32, @intCast(b)), HD);
                const dst = (pos + @as(u32, @intCast(b))) * NKV * HD + kvh * HD;
                for (0..HD) |d| self.kv.k_cache[layer][dst + d] = ks[d];
            }
            for (0..NKV) |kvh| {
                const vs = qkv_slice[NH * HD + NKV * HD + kvh * HD ..][0..HD];
                const dst = (pos + @as(u32, @intCast(b))) * NKV * HD + kvh * HD;
                for (0..HD) |d| self.kv.v_cache[layer][dst + d] = vs[d];
            }
        }

        // Attention
        const QKV2 = NH * HD + 2 * NKV * HD;
        if (dispatch.attention == .gpu) {
            if (self.gpu) |*gpu| {
                const seq_len = self.kv.position + batch_size;
                for (0..batch_size) |b| {
                    const q_slice = qkv_scratch[b * QKV2 ..][0..NH * HD];
                    const out_slice = s.attn_out[b * NH * HD ..][0..NH * HD];
                    gpu.flashAttention(q_slice, self.kv.k_cache[layer], self.kv.v_cache[layer],
                        &.{}, out_slice, &([_]f32{std.math.nan(f32)} ** 12),
                        NH, NKV, HD, seq_len, 0, 0.0, 0) catch |err| {
                        log.warn("GPU attention layer {d} failed: {s}", .{layer, @errorName(err)});
                        self.cpuAttention(q_slice, out_slice, layer, seq_len, NH, NKV, HD, GQA);
                    };
                }
            } else {
                for (0..batch_size) |b| {
                    const q_slice = qkv_scratch[b * QKV2 ..][0..NH * HD];
                    self.cpuAttention(q_slice, s.attn_out[b * NH * HD ..][0..NH * HD], layer, self.kv.position + batch_size, NH, NKV, HD, GQA);
                }
            }
        } else if (dispatch.attention == .cpu) {
            const seq_len = self.kv.position + batch_size;
            for (0..batch_size) |b| {
                const q_slice = qkv_scratch[b * QKV2 ..][0..NH * HD];
                self.cpuAttention(q_slice, s.attn_out[b * NH * HD ..][0..NH * HD], layer, seq_len, NH, NKV, HD, GQA);
            }
        }

        // O projection: CPU GEMV when dispatched to CPU/GPU, NPU otherwise.
        // GPU attention dispatch also needs CPU O proj (no GPU O proj kernel yet — fix #56).
        if ((dispatch.attention == .cpu or dispatch.attention == .gpu) and layer < self.cpu_weights.o.len) {
            for (0..batch_size) |b| {
                cpuGemv(self.cpu_weights.o[layer], s.attn_out[b * NH * HD ..][0..NH * HD], s.o_out[b * H ..][0..H], H, NH * HD);
            }
        } else if (dispatch.attention == .npu) {
            try self.npu.runOProj(s.attn_out[0..batch_size * NH * HD], layer, batch_size, s.o_out[0..batch_size * H]);
        } else {
            // No matching O-proj backend/weights. Zeroing silently reintroduces
            // the #56 degenerate-output bug (#145) — make it loud.
            log.warn("O-proj dispatch {any} unhandled at layer {d} (cpu_weights.o loaded={any}); zeroing — output will be degenerate", .{ dispatch.attention, layer, layer < self.cpu_weights.o.len });
            @memset(s.o_out[0..batch_size * H], 0);
        }
        for (0..batch_size * H) |i| s.hidden[i] = s.residual[i] + s.o_out[i];

        // Pre-FFN residual save + RMSNorm
        @memcpy(s.residual[0..batch_size * H], s.hidden[0..batch_size * H]);
        for (0..batch_size) |b| {
            rmsNorm(s.hidden[b * H ..][0..H], self.pa_norm[layer], s.hidden[b * H ..][0..H], 1e-6);
        }

        // ── Branch on FFN kernel type ──
        switch (self.ffn_kernel) {
            .dense => {
                if (dispatch.ffn == .npu or dispatch.ffn == .cpu or dispatch.ffn == .gpu) {
                    self.runDenseFfn(layer, batch_size, s.hidden, s.residual, dispatch);
                } else {
                    // GPU FFN — not implemented yet, fall through to residual copy
                    for (0..batch_size * H) |i| s.hidden[i] = s.residual[i] + s.down_out[i];
                }
            },
            .moe, .shared_moe => {
                self.runMoeFfn(layer, batch_size, s.hidden, s.residual);
            },
        }

        // Copy output
        if (output_hidden.ptr != s.hidden.ptr) {
            @memcpy(output_hidden[0..batch_size * H], s.hidden[0..batch_size * H]);
        }
    }

    pub fn executeLayer(
        self: *FusedExecutor,
        layer: u32,
        batch_size: u32,
        output_hidden: []f32,
        _next_input: ?[]const f32,
    ) !void {
        _ = _next_input;
        const H = self.config.hidden_dim;
        const s = self.scratch;

        // ── Step 1: Save pre-norm residual ──
        @memcpy(s.residual[0..batch_size * H], s.hidden[0..batch_size * H]);

        // ── Step 2: RMSNorm + QKV on NPU ──
        for (0..batch_size) |b| {
            rmsNorm(
                s.hidden[b * H ..][0..H],
                self.in_norm[layer],
                s.hidden[b * H ..][0..H],
                1e-6,
            );
        }

        // ── Branch on attention kernel type for QKV ──
        if (self.attn_kernel == .mla) {
            // MLA: Q projection via NPU + compressed KV
            // Store normalized hidden in qkv for later KV compression in attn step
            @memcpy(s.qkv[0..batch_size * H], s.hidden[0..batch_size * H]);
        } else if (self.getLayerDispatch(layer).qkv == .npu) {
            self.npu.runQKV(s.hidden[0..batch_size * H], layer, batch_size, s.qkv) catch |err| {
                log.warn("NPU QKV (layer {d}) failed: {s}, falling back to CPU", .{ layer, @errorName(err) });
                self.npu_broken = true;
                // CPU fallback: compute QKV via CPU mat-vec
                @memset(s.qkv[0..batch_size * (self.config.n_heads * self.config.head_dim + 2 * self.config.n_kv_heads * self.config.head_dim)], 0);
            };
        }

        // ── Step 3: Q/K norm, RoPE, KV cache write (only for flash attention) ──
        if (self.attn_kernel == .flash) {
            const NH = self.config.n_heads;
            const NKV = self.config.n_kv_heads;
            const HD = self.config.head_dim;
            const QKV = NH * HD + 2 * NKV * HD;
            const pos = self.kv.position;
            for (0..batch_size) |b| {
                const qkv_slice = s.qkv[b * QKV ..][0..QKV];
                for (0..NH) |hh| {
                    const qh = qkv_slice[hh * HD ..][0..HD];
                    var sq: f64 = 0;
                    for (qh) |v| sq += @as(f64, @floatCast(v)) * @as(f64, @floatCast(v));
                    const iq = 1.0 / @sqrt(@as(f32, @floatCast(sq / @as(f64, @floatFromInt(HD)))) + 1e-6);
                    if (layer < self.q_norm.len and self.q_norm[layer].len >= HD) {
                            for (0..HD) |d| qh[d] = qh[d] * iq * self.q_norm[layer][d];
                        }
                    self.applyRoPE(qh, @as(u32, @intCast(pos + b)), HD);
                }
                for (0..NKV) |kvh| {
                    const ks = qkv_slice[NH * HD + kvh * HD ..][0..HD];
                    var sk: f64 = 0;
                    for (ks) |v| sk += @as(f64, @floatCast(v)) * @as(f64, @floatCast(v));
                    const ik = 1.0 / @sqrt(@as(f32, @floatCast(sk / @as(f64, @floatFromInt(HD)))) + 1e-6);
                    if (layer < self.k_norm.len and self.k_norm[layer].len >= HD) {
                            for (0..HD) |d| ks[d] = ks[d] * ik * self.k_norm[layer][d];
                        }
                    self.applyRoPE(ks, @as(u32, @intCast(pos + b)), HD);
                    const dst = (pos + @as(u32, @intCast(b))) * NKV * HD + kvh * HD;
                    for (0..HD) |d| self.kv.k_cache[layer][dst + d] = ks[d];
                }
                for (0..NKV) |kvh| {
                    const vs = qkv_slice[NH * HD + NKV * HD + kvh * HD ..][0..HD];
                    const dst = (pos + @as(u32, @intCast(b))) * NKV * HD + kvh * HD;
                    for (0..HD) |d| self.kv.v_cache[layer][dst + d] = vs[d];
                }
            }
        }

        // ── Step 4: Attention ──
        const NH = self.config.n_heads;
        const NKV = self.config.n_kv_heads;
        const HD = self.config.head_dim;
        if (self.attn_kernel == .mla) {
            // MLA attention (output goes to s.attn_out)
            self.runMlaAttention(layer, batch_size, s.qkv, s.attn_out);
        } else if (self.getLayerDispatch(layer).attention == .gpu) {
            if (self.gpu) |*gpu| {
                const GQA = self.config.gqa_ratio;
                const seq_len = self.kv.position + batch_size;
                const QKV = NH * HD + 2 * NKV * HD;
                for (0..batch_size) |b| {
                    const q_slice = s.qkv[b * QKV ..][0..NH * HD];
                    const out_slice = s.attn_out[b * NH * HD ..][0..NH * HD];
                    gpu.flashAttention(
                        q_slice, self.kv.k_cache[layer], self.kv.v_cache[layer], &.{},
                        out_slice, &([_]f32{std.math.nan(f32)} ** 12),
                        NH, NKV, HD, @as(u32, @intCast(seq_len)), 0, 0.0, 0,
                    ) catch |err| {
                        log.warn("GPU attention layer {d} failed: {s}, using CPU fallback", .{ layer, @errorName(err) });
                        self.cpuAttention(q_slice, out_slice, layer, seq_len, NH, NKV, HD, GQA);
                    };
                }
            } else {
                const GQA = self.config.gqa_ratio;
                for (0..batch_size) |b| {
                    const q_slice = s.qkv[b * NH * HD ..][0..NH * HD];
                    self.cpuAttention(q_slice, s.attn_out[b * NH * HD ..][0..NH * HD], layer, self.kv.position + batch_size, NH, NKV, HD, GQA);
                }
            }
        } else {
            // Attention on NPU — pass through
            @memcpy(s.attn_out[0..batch_size * NH * HD], s.qkv[0..batch_size * NH * HD]);
        }

        // ── Step 5: O projection on NPU (or skip if broken) ──
        if (!self.npu_broken) {
            self.npu.runOProj(s.attn_out[0..batch_size * NH * HD], layer, batch_size, s.o_out[0..batch_size * H]) catch |err| {
                log.warn("NPU O-proj (layer {d}) failed: {s}", .{ layer, @errorName(err) });
                self.npu_broken = true;
                @memset(s.o_out[0..batch_size * H], 0);
            };
        } else {
            @memset(s.o_out[0..batch_size * H], 0);
        }

        // ── Step 6: Residual add (attention) ──
        for (0..batch_size * H) |i| s.hidden[i] = s.residual[i] + s.o_out[i];

        // ── Step 7: Pre-FFN residual save + RMSNorm ──
        @memcpy(s.residual[0..batch_size * H], s.hidden[0..batch_size * H]);
        for (0..batch_size) |b| {
            rmsNorm(
                s.hidden[b * H ..][0..H],
                self.pa_norm[layer],
                s.hidden[b * H ..][0..H],
                1e-6,
            );
        }

        // ── Step 8: FFN ──
        switch (self.ffn_kernel) {
            .dense => {
                if (self.getLayerDispatch(layer).ffn == .npu) {
                    self.runDenseFfn(layer, batch_size, s.hidden, s.residual, self.getLayerDispatch(layer));
                } else {
                    for (0..batch_size * H) |i| s.hidden[i] = s.residual[i] + s.down_out[i];
                }
            },
            .moe, .shared_moe => {
                self.runMoeFfn(layer, batch_size, s.hidden, s.residual);
            },
        }

        // ── Step 9: Copy output ──
        if (output_hidden.ptr != s.hidden.ptr) {
            @memcpy(output_hidden[0..batch_size * H], s.hidden[0..batch_size * H]);
        }
    }

    /// CPU fallback attention (scaled dot-product with causal mask).
    fn cpuAttention(
        self: *const FusedExecutor,
        q: []const f32,
        output: []f32,
        layer: u32,
        seq_len: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        gqa_ratio: u32,
    ) void {
        const k_cache = self.kv.k_cache[layer][0..@min(@as(usize, seq_len) * n_kv_heads * head_dim, self.kv.k_cache[layer].len)];
        const v_cache = self.kv.v_cache[layer][0..@min(@as(usize, seq_len) * n_kv_heads * head_dim, self.kv.v_cache[layer].len)];
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        for (0..n_heads) |h| {
            const kvh = h / gqa_ratio;
            const qh = q[h * head_dim ..][0..head_dim];

            var max_score: f32 = -std.math.inf(f32);
            var scores: [4096]f32 = undefined;
            const max_seq = @min(seq_len, @as(u32, 4096));
            for (0..max_seq) |pos| {
                const k_off = pos * n_kv_heads * head_dim + kvh * head_dim;
                var dot: f32 = 0;
                for (0..head_dim) |d| dot += qh[d] * k_cache[k_off + d];
                const score = dot * scale;
                scores[pos] = score;
                if (score > max_score) max_score = score;
            }

            var sum_exp: f32 = 0;
            for (0..max_seq) |pos| {
                scores[pos] = std.math.exp(scores[pos] - max_score);
                sum_exp += scores[pos];
            }
            const inv_sum = if (sum_exp > 0) 1.0 / sum_exp else 0;

            const out_h = output[h * head_dim ..][0..head_dim];
            @memset(out_h, 0);
            for (0..max_seq) |pos| {
                const w = scores[pos] * inv_sum;
                const v_off = pos * n_kv_heads * head_dim + kvh * head_dim;
                for (0..head_dim) |d| out_h[d] += w * v_cache[v_off + d];
            }
        }
    }

    /// Batched GEMV decode path for M > batch_split_threshold (fat batch).
    /// Processes all tokens through each stage at once:
    ///   QKV (NPU batched) -> Attention (GPU batched per-token) -> O proj (NPU) -> FFN (NPU batched)
    /// Skips the per-layer QKV/AttnFFN split to reduce dispatch overhead.
    pub fn forwardDecodeBatchGEMV(
        self: *FusedExecutor,
        input_tokens: []const u32,
        batch_size: u32,
        output_hidden: []f32,
    ) !void {
        const H = self.config.hidden_dim;
        const NC = self.config.n_layers;
        const NH = self.config.n_heads;
        const NKV = self.config.n_kv_heads;
        const HD = self.config.head_dim;
        const B = batch_size;
        const QKV = NH * HD + 2 * NKV * HD;
        const QKV_bytes = B * QKV;
        const GQA = self.config.gqa_ratio;

        // ── Embed input tokens ──
        for (0..B) |b| {
            const tok = input_tokens[b];
            for (0..H) |i| {
                self.scratch.hidden[b * H + i] =
                    self.emb_f32[@as(usize, @intCast(tok)) * H + i];
            }
        }

        const qkv_buf = try self.allocator.alloc(f32, QKV_bytes);
        defer self.allocator.free(qkv_buf);

        for (0..NC) |l| {
            const layer = @as(u32, @intCast(l));
            const s = &self.scratch;
            const dispatch = self.getLayerDispatch(layer);

            // ── Step 1: Save residual ──
            @memcpy(s.residual[0..B * H], s.hidden[0..B * H]);

            // ── Step 2: RMSNorm + QKV ──
            for (0..B) |b| {
                rmsNorm(
                    s.hidden[b * H ..][0..H],
                    self.in_norm[layer],
                    s.hidden[b * H ..][0..H],
                    1e-6,
                );
            }
            const qkv_cpu_ready = layer < self.cpu_weights.q.len and layer < self.cpu_weights.k.len and layer < self.cpu_weights.v.len;
            if (dispatch.qkv == .cpu and qkv_cpu_ready) {
                for (0..B) |b| {
                    const x = s.hidden[b * H ..][0..H];
                    const qkv_slice = qkv_buf[b * QKV ..][0..QKV];
                    cpuGemv(self.cpu_weights.q[layer], x, qkv_slice[0 .. NH * HD], NH * HD, H);
                    cpuGemv(self.cpu_weights.k[layer], x, qkv_slice[NH * HD ..][0 .. NKV * HD], NKV * HD, H);
                    cpuGemv(self.cpu_weights.v[layer], x, qkv_slice[NH * HD + NKV * HD ..][0 .. NKV * HD], NKV * HD, H);
                }
            } else tryOrZero: {
                self.npu.runQKV(s.hidden[0..B * H], layer, B, qkv_buf) catch |err| {
                    log.warn("NPU QKV (layer {d}) failed: {s} - falling back to CPU for remaining layers", .{ layer, @errorName(err) });
                    self.npu_broken = true;
                    break :tryOrZero;
                };
            }

            // ── Step 3: Q/K norm, RoPE, KV cache write (flash attention) ──
            if (self.attn_kernel == .flash) {
                const pos = self.kv.position;
                for (0..B) |b| {
                    const qkv_slice = qkv_buf[b * QKV ..][0..QKV];
                    for (0..NH) |hh| {
                        const qh = qkv_slice[hh * HD ..][0..HD];
                        var sq: f64 = 0;
                        for (qh) |v| sq += @as(f64, @floatCast(v)) * @as(f64, @floatCast(v));
                        const iq = 1.0 / @sqrt(@as(f32, @floatCast(sq / @as(f64, @floatFromInt(HD)))) + 1e-6);
                        if (layer < self.q_norm.len and self.q_norm[layer].len >= HD) {
                            for (0..HD) |d| qh[d] = qh[d] * iq * self.q_norm[layer][d];
                        }
                        self.applyRoPE(qh, pos + @as(u32, @intCast(b)), HD);
                    }
                    for (0..NKV) |kvh| {
                        const ks = qkv_slice[NH * HD + kvh * HD ..][0..HD];
                        var sk: f64 = 0;
                        for (ks) |v| sk += @as(f64, @floatCast(v)) * @as(f64, @floatCast(v));
                        const ik = 1.0 / @sqrt(@as(f32, @floatCast(sk / @as(f64, @floatFromInt(HD)))) + 1e-6);
                        if (layer < self.k_norm.len and self.k_norm[layer].len >= HD) {
                            for (0..HD) |d| ks[d] = ks[d] * ik * self.k_norm[layer][d];
                        }
                        self.applyRoPE(ks, pos + @as(u32, @intCast(b)), HD);
                        const dst = (pos + @as(u32, @intCast(b))) * NKV * HD + kvh * HD;
                        for (0..HD) |d| self.kv.k_cache[layer][dst + d] = ks[d];
                    }
                    for (0..NKV) |kvh| {
                        const vs = qkv_slice[NH * HD + NKV * HD + kvh * HD ..][0..HD];
                        const dst = (pos + @as(u32, @intCast(b))) * NKV * HD + kvh * HD;
                        for (0..HD) |d| self.kv.v_cache[layer][dst + d] = vs[d];
                    }
                }
            }

            // ── Step 4: Attention (batched per-token flash attention) ──
            if (self.attn_kernel == .flash and dispatch.attention != .cpu and self.gpu != null) {
                const gpu = &self.gpu.?;
                const seq_len = self.kv.position + B;
                for (0..B) |b| {
                    const q_slice = qkv_buf[b * QKV ..][0..NH * HD];
                    const out_slice = s.attn_out[b * NH * HD ..][0..NH * HD];
                    gpu.flashAttention(
                        q_slice, self.kv.k_cache[layer], self.kv.v_cache[layer], &.{},
                        out_slice, &([_]f32{std.math.nan(f32)} ** 12),
                        NH, NKV, HD, seq_len, 0, 0.0, 0,
                    ) catch |err| {
                        log.warn("GPU attention layer {d} failed: {s}", .{ layer, @errorName(err) });
                        self.cpuAttention(q_slice, out_slice, layer, seq_len, NH, NKV, HD, GQA);
                    };
                }
            } else if (self.attn_kernel == .flash) {
                // CPU fallback (GPU unavailable, or dispatch.attention == .cpu)
                const seq_len = self.kv.position + B;
                for (0..B) |b| {
                    const q_slice = qkv_buf[b * QKV ..][0..NH * HD];
                    self.cpuAttention(
                        q_slice, s.attn_out[b * NH * HD ..][0..NH * HD],
                        layer, seq_len, NH, NKV, HD, GQA,
                    );
                }
            }

            // ── Step 5: O projection: CPU GEMV for cpu/gpu dispatch, NPU for npu dispatch ──
            if ((dispatch.attention == .cpu or dispatch.attention == .gpu) and layer < self.cpu_weights.o.len) {
                for (0..B) |b| {
                    cpuGemv(self.cpu_weights.o[layer], s.attn_out[b * NH * HD ..][0 .. NH * HD], s.o_out[b * H ..][0..H], H, NH * HD);
                }
            } else if (dispatch.attention == .npu) tryOrZero: {
                self.npu.runOProj(
                    s.attn_out[0..B * NH * HD], layer, B, s.o_out[0..B * H],
                ) catch |err| {
                    log.warn("NPU O-proj (layer {d}) failed: {s} - falling back to CPU for remaining layers", .{ layer, @errorName(err) });
                    self.npu_broken = true;
                    break :tryOrZero;
                };
            } else {
                log.warn("O-proj dispatch {any} unhandled at layer {d} (cpu_weights.o loaded={any}); zeroing — output will be degenerate (#145)", .{ dispatch.attention, layer, layer < self.cpu_weights.o.len });
                @memset(s.o_out[0..B * H], 0);
            }
            for (0..B * H) |i| s.hidden[i] = s.residual[i] + s.o_out[i];

            // ── Step 6: Pre-FFN residual save + RMSNorm ──
            @memcpy(s.residual[0..B * H], s.hidden[0..B * H]);
            for (0..B) |b| {
                rmsNorm(
                    s.hidden[b * H ..][0..H],
                    self.pa_norm[layer],
                    s.hidden[b * H ..][0..H],
                    1e-6,
                );
            }

            // ── Step 7: FFN ──
            switch (self.ffn_kernel) {
                .dense => {
                    self.runDenseFfn(layer, B, s.hidden, s.residual, dispatch);
                },
                .moe, .shared_moe => {
                    self.runMoeFfn(layer, B, s.hidden, s.residual);
                },
            }
        }

        // ── Final RMSNorm ──
        for (0..B) |b| {
            rmsNorm(
                self.scratch.hidden[b * H ..][0..H],
                self.final_norm,
                self.scratch.hidden[b * H ..][0..H],
                1e-6,
            );
        }
        if (output_hidden.ptr != self.scratch.hidden.ptr) {
            @memcpy(output_hidden[0..B * H], self.scratch.hidden[0..B * H]);
        }
    }

    /// Execute full forward pass with pipeline overlap.
    /// Dispatches to skinny decode (M <= batch_split_threshold) or
    /// batched GEMV decode (M > batch_split_threshold) based on batch size.
    /// When policy is .prefill_npu_decode_gpu, always uses batched GEMV path.
    pub fn forwardDecode(
        self: *FusedExecutor,
        input_tokens: []const u32,
        batch_size: u32,
        output_hidden: []f32,
    ) !void {
        // Log MoE GEMV strategy selection.
        if (self.ffn_kernel != .dense) {
            if (self.moe_config) |mc| {
                const strat = mc.selectGemvStrategy(batch_size);
                log.debug("MoE BS={d}: {s} (threshold={d})", .{
                    batch_size,
                    @tagName(strat),
                    mc.gemv_bs_threshold,
                });
            }
        }

        // Dispatch to batched GEMV path if batch is large or policy demands it.
        if (batch_size > self.batch_split_threshold or self.policy == .prefill_npu_decode_gpu) {
            return self.forwardDecodeBatchGEMV(input_tokens, batch_size, output_hidden);
        }
        const H = self.config.hidden_dim;
        const NC = self.config.n_layers;
        const NH = self.config.n_heads;
        const NKV = self.config.n_kv_heads;
        const HD = self.config.head_dim;
        const B = batch_size;
        const QKV = NH * HD + 2 * NKV * HD;
        const QKV_bytes = B * QKV;

        // ── Embed input tokens ──
        for (0..B) |b| {
            const tok = input_tokens[b];
            for (0..H) |i| {
                self.scratch.hidden[b * H + i] = self.emb_f32[@as(usize, @intCast(tok)) * H + i];
            }
        }

        const qkv_buf = try self.allocator.alloc(f32, QKV_bytes);
        defer self.allocator.free(qkv_buf);

        // Batch NPU: send all layers' QKV in one call, then attn per layer
        if (!self.npu_broken and B == 1) {
            const IM = self.config.inter_size;
            const all_qkv = try self.allocator.alloc(f32, @as(usize, NC) * QKV_bytes);
            defer self.allocator.free(all_qkv);

            // Step 1: RMSNorm all layers, pack into all_qkv buffer
            for (0..NC) |l| {
                rmsNorm(self.scratch.hidden[0..H], self.in_norm[l], self.scratch.hidden[0..H], 1e-6);
                @memcpy(all_qkv[l * QKV_bytes ..][0..H], self.scratch.hidden[0..H]);
                @memcpy(self.scratch.hidden[0..H], self.scratch.residual[0..H]);
            }

            // Step 2: One NPU call for ALL layers' QKV
            self.npu.runAllQKV(all_qkv[0..NC * H], B, all_qkv) catch {
                self.npu_broken = true;
            };

            if (!self.npu_broken) {
                // Step 3: Per-layer attention using batch QKV results
                for (0..NC) |l| {
                    try self.executeLayerAttnOnly(@as(u32, @intCast(l)), B,
                        all_qkv[l * QKV_bytes ..][0..QKV_bytes],
                        self.scratch.hidden[0..B * H]);
                }

                // Step 4: Collect FFN inputs, send all
                const ffn_size = 2 * IM;
                const all_ffn = try self.allocator.alloc(f32, @as(usize, NC) * B * ffn_size);
                defer self.allocator.free(all_ffn);
                for (0..NC) |l| {
                    @memcpy(self.scratch.residual[0..H], self.scratch.hidden[0..H]);
                    rmsNorm(self.scratch.hidden[0..H], self.pa_norm[l], self.scratch.hidden[0..H], 1e-6);
                    @memcpy(all_ffn[l * H ..][0..H], self.scratch.hidden[0..H]);
                    @memcpy(self.scratch.hidden[0..H], self.scratch.residual[0..H]);
                }

                self.npu.runAllFFN(all_ffn[0..NC * H], B, all_ffn) catch {
                    self.npu_broken = true;
                };

                if (!self.npu_broken) {
                    // Step 5: SiLU activation, collect Down inputs, send all
                    const all_down = try self.allocator.alloc(f32, @as(usize, NC) * B * IM);
                    defer self.allocator.free(all_down);
                    for (0..NC) |l| {
                        const gu = all_ffn[l * ffn_size ..][0..B * ffn_size];
                        for (0..B) |_| {
                            for (0..IM) |i| {
                                const g = gu[i];
                                const u = gu[IM + i];
                                const act = g / (1.0 + std.math.exp(-g));
                                self.scratch.activated[i] = act * u;
                            }
                        }
                        @memcpy(all_down[l * IM ..][0..B * IM], self.scratch.activated[0..B * IM]);
                    }

                    self.npu.runAllDown(all_down[0..NC * IM], B, all_down) catch {
                        self.npu_broken = true;
                    };

                    if (!self.npu_broken) {
                        for (0..NC) |l| {
                            const down = all_down[l * H ..][0..B * H];
                            for (0..B * H) |i| self.scratch.hidden[i] += down[i];
                        }
                        return;
                    }
                }
            }
        }

        // Fallback: per-layer NPU calls
        for (0..NC) |l| {
            try self.executeLayerQKV(@as(u32, @intCast(l)), B, qkv_buf);
            try self.executeLayerAttnFFN(@as(u32, @intCast(l)), B, qkv_buf, self.scratch.hidden[0..B * H]);
        }

        // ── Final RMSNorm ──
        for (0..B) |b| {
            rmsNorm(
                self.scratch.hidden[b * H ..][0..H],
                self.final_norm,
                self.scratch.hidden[b * H ..][0..H],
                1e-6,
            );
        }
        if (output_hidden.ptr != self.scratch.hidden.ptr) {
            @memcpy(output_hidden[0..B * H], self.scratch.hidden[0..B * H]);
        }
    }

    /// LM head: compute logits and return top-k tokens.
    pub fn lmHead(self: *FusedExecutor, hidden: []const f32, top_ids: []u32, top_k: u32) !void {
        const H = self.config.hidden_dim;
        const NV = self.config.vocab_size;
        const emb = if (self.lm_head_f32) |lm| lm else self.emb_f32;

        // ── 1. Logits: dot product for all vocab entries ──
        var max_logit: f32 = -std.math.inf(f32);
        for (0..NV) |n| {
            var dot: f64 = 0;
            const e_row = emb[n * H ..][0..H];
            for (0..H) |i| dot += @as(f64, @floatCast(hidden[i])) * @as(f64, @floatCast(e_row[i]));
            const lg = @as(f32, @floatCast(dot));
            self.scratch.logits[n] = lg;
            if (lg > max_logit) max_logit = lg;
        }

        // ── 2. Temperature scaling ──
        // Temperature 0 → argmax (deterministic). Temperature > 0 flattens
        // (high temp) or sharpens (low temp) the probability distribution.
        const temperature: f32 = 0.7;
        const inv_temp: f32 = if (temperature > 0.0) 1.0 / temperature else 0.0;
        for (0..NV) |n| {
            self.scratch.logits[n] = (self.scratch.logits[n] - max_logit) * inv_temp;
        }

        // ── 3. Softmax ──
        var sum_exp: f64 = 0;
        for (0..NV) |n| {
            const d = self.scratch.logits[n];
            const e = if (d < -80) 0.0 else std.math.exp(d);
            self.scratch.logits[n] = e;
            sum_exp += @as(f64, @floatCast(e));
        }
        const inv_sum: f32 = if (sum_exp > 0) @floatCast(1.0 / sum_exp) else 1.0;
        for (0..NV) |n| {
            self.scratch.logits[n] *= inv_sum;
        }

        // ── 4. Top-K selection + Top-p (nucleus) ──
        // For argmax (temperature=0 or pure top-1), use the original top-k
        // insertion sort which is O(NV*k). For sampling, we need probabilities
        // sorted by rank to apply top-p filtering.
        const TopEntry = struct { id: u32, val: f32 };
        var top: [128]TopEntry = [_]TopEntry{.{ .id = 0, .val = -std.math.inf(f32) }} ** 128;
        const k = @min(top_k, NV);

        for (0..NV) |n| {
            const val = self.scratch.logits[n];
            for (0..k) |j| {
                if (val > top[j].val) {
                    var jj = k - 1;
                    while (jj > j) : (jj -= 1) top[jj] = top[jj - 1];
                    top[j] = .{ .id = @intCast(n), .val = val };
                    break;
                }
            }
        }

        // Top-p (nucleus): accumulate probabilities from highest to lowest
        // until we reach top_p threshold. The first token in the nucleus
        // becomes the single sampled output.
        const top_p: f32 = 0.9;
        var cum_prob: f32 = 0;
        var nucleus_end: u32 = 0;
        while (nucleus_end < k) : (nucleus_end += 1) {
            cum_prob += top[nucleus_end].val;
            if (cum_prob >= top_p) {
                nucleus_end += 1;
                break;
            }
        }
        if (nucleus_end == 0) nucleus_end = 1; // at least the top token

        // Fill top_ids: nucleus tokens first, then pad with top token
        for (0..k) |j| {
            top_ids[j] = if (j < nucleus_end) top[j].id else top[0].id;
        }
    }

    /// Full prefill: process N input tokens, build KV cache, produce final hidden state.
    /// Handles both flash attention (standard K/V cache) and MLA (compressed KV cache).
    pub fn prefill(self: *FusedExecutor, tokens: []const u32, batch_size: u32) !void {
        const H = self.config.hidden_dim;
        const NC = self.config.n_layers;
        const npt = @as(u32, @intCast(tokens.len));
        _ = batch_size;

        // Embed all tokens
        for (0..npt) |pi| {
            const tok = tokens[pi];
            const emb_idx = @as(usize, @intCast(tok)) * H;
            if (emb_idx + H > self.emb_f32.len) {
                log.err("Embedding OOB: tok={d} H={d} emb_idx={d} emb_len={d}", .{ tok, H, emb_idx, self.emb_f32.len });
                return error.EmbOutOfBounds;
            }
            for (0..H) |i| {
                self.scratch.hidden[@as(usize, @intCast(pi)) * H + i] = self.emb_f32[emb_idx + i];
            }
        }

        // Layer loop
        for (0..NC) |l| {
            const layer_u32 = @as(u32, @intCast(l));
            for (0..npt) |pi| {
                // Recomputed per position, not once per layer: .auto (and
                // any policy relying on npu_broken) needs to observe a
                // mid-layer NPU failure immediately -- computing this once
                // per layer meant a failure on the first prompt token left
                // every later token in that same layer still dispatching
                // (and failing) against the NPU for the rest of the
                // layer's positions, corrupting that layer's whole output
                // instead of degrading to CPU right away.
                const dispatch = self.getLayerDispatch(layer_u32);
                const pos = if (self.attn_kernel == .mla)
                    (self.mla_kv orelse return).position + pi
                else
                    self.kv.position + pi;

                const hidden_slice = self.scratch.hidden[@as(usize, @intCast(pi)) * H ..][0..H];

                // Save residual
                @memcpy(self.scratch.residual[@as(usize, @intCast(pi)) * H ..][0..H], hidden_slice);

                // RMSNorm + QKV
                rmsNorm(hidden_slice, self.in_norm[l], hidden_slice, 1e-6);

                if (self.attn_kernel == .mla) {
                    // MLA prefill: compute Q and compressed KV via NPU
                    // Q projection
                    try self.npu.runMlaQProj(hidden_slice, @as(u32, @intCast(l)), 1, self.scratch.qkv);

                    // KV compression
                    var latent: [512]f32 = [_]f32{0} ** 512;
                    const kv_lora3 = if (self.mla_config) |mlac2| mlac2.kv_lora_rank else 256;
                    try self.npu.runMlaKvCompress(hidden_slice, @as(u32, @intCast(l)), 1, latent[0..kv_lora3]);

                    // Write compressed latent to MLA KV cache
                    if (self.mla_kv) |*mk| {
                        mk.write(@as(u32, @intCast(l)), latent[0..kv_lora3]);
                    }

                    // Run MLA attention (CPU fallback for prefill)
                    // The attention output is written to self.scratch.attn_out
                    @memset(self.scratch.attn_out[0..self.config.n_heads * self.config.head_dim], 0);
                    log.warn("MLA prefill attention not yet implemented on CPU at layer {d}, pos {d}", .{ l, pos });
                } else {
                    // Standard flash attention prefill
                    const NH = self.config.n_heads;
                    const NKV = self.config.n_kv_heads;
                    const HD = self.config.head_dim;
                    if (dispatch.qkv == .cpu and l < self.cpu_weights.q.len and l < self.cpu_weights.k.len and l < self.cpu_weights.v.len) {
                        cpuGemv(self.cpu_weights.q[l], hidden_slice, self.scratch.qkv[0 .. NH * HD], NH * HD, H);
                        cpuGemv(self.cpu_weights.k[l], hidden_slice, self.scratch.qkv[NH * HD ..][0 .. NKV * HD], NKV * HD, H);
                        cpuGemv(self.cpu_weights.v[l], hidden_slice, self.scratch.qkv[NH * HD + NKV * HD ..][0 .. NKV * HD], NKV * HD, H);
                    } else {
                        self.npu.runQKV(hidden_slice, layer_u32, 1, self.scratch.qkv) catch |err| {
                            // Matches the runFFN/runDown pattern below: a
                            // dead NPU subprocess must not abort the whole
                            // prefill (every policy that ever routes qkv
                            // to NPU -- npu_only, ffn_on_npu, qkv_on_npu,
                            // auto -- previously hard-failed here via a
                            // bare `try` the instant the NPU was
                            // unavailable). Flag broken so every
                            // subsequent getLayerDispatch() call (auto
                            // included) falls back to CPU/GPU instead.
                            log.warn("NPU QKV (layer {d}) failed: {s}", .{ l, @errorName(err) });
                            self.npu_broken = true;
                            // Recompute THIS token's QKV on CPU right now
                            // instead of zero-filling it: this token's
                            // position still gets written into the KV
                            // cache below, and every later position's
                            // causal attention reads back through it, so
                            // a zero here would permanently corrupt the
                            // whole rest of the sequence rather than just
                            // costing one failed round-trip.
                            if (l < self.cpu_weights.q.len and l < self.cpu_weights.k.len and l < self.cpu_weights.v.len) {
                                cpuGemv(self.cpu_weights.q[l], hidden_slice, self.scratch.qkv[0 .. NH * HD], NH * HD, H);
                                cpuGemv(self.cpu_weights.k[l], hidden_slice, self.scratch.qkv[NH * HD ..][0 .. NKV * HD], NKV * HD, H);
                                cpuGemv(self.cpu_weights.v[l], hidden_slice, self.scratch.qkv[NH * HD + NKV * HD ..][0 .. NKV * HD], NKV * HD, H);
                            } else {
                                @memset(self.scratch.qkv, 0);
                            }
                        };
                    }

                    // Q/K norm, RoPE, KV cache
                    const QKV = self.config.n_heads * self.config.head_dim + 2 * self.config.n_kv_heads * self.config.head_dim;
                    const qkv_slice = self.scratch.qkv[0..QKV];
                    for (0..self.config.n_heads) |hh| {
                        const qh = qkv_slice[hh * self.config.head_dim ..][0..self.config.head_dim];
                        var sq: f64 = 0;
                        for (qh) |v| sq += @as(f64, @floatCast(v)) * @as(f64, @floatCast(v));
                        const iq = 1.0 / @sqrt(@as(f32, @floatCast(sq / @as(f64, @floatFromInt(self.config.head_dim)))) + 1e-6);
                        if (l < self.q_norm.len and self.q_norm[l].len >= self.config.head_dim) {
                            for (0..self.config.head_dim) |d| qh[d] = qh[d] * iq * self.q_norm[l][d];
                        }
                        self.applyRoPE(qh, @as(u32, @intCast(pos)), self.config.head_dim);
                    }
                    for (0..self.config.n_kv_heads) |kvh| {
                        const ks = qkv_slice[self.config.n_heads * self.config.head_dim + kvh * self.config.head_dim ..][0..self.config.head_dim];
                        var sk: f64 = 0;
                        for (ks) |v| sk += @as(f64, @floatCast(v)) * @as(f64, @floatCast(v));
                        const ik = 1.0 / @sqrt(@as(f32, @floatCast(sk / @as(f64, @floatFromInt(self.config.head_dim)))) + 1e-6);
                        if (l < self.k_norm.len and self.k_norm[l].len >= self.config.head_dim) {
                            for (0..self.config.head_dim) |d| ks[d] = ks[d] * ik * self.k_norm[l][d];
                        }
                        self.applyRoPE(ks, @as(u32, @intCast(pos)), self.config.head_dim);
                        const dst = @as(usize, @intCast(pos)) * self.config.n_kv_heads * self.config.head_dim + kvh * self.config.head_dim;
                        for (0..self.config.head_dim) |d| self.kv.k_cache[l][dst + d] = ks[d];
                    }
                    for (0..self.config.n_kv_heads) |kvh| {
                        const vs = qkv_slice[self.config.n_heads * self.config.head_dim + self.config.n_kv_heads * self.config.head_dim + kvh * self.config.head_dim ..][0..self.config.head_dim];
                        const dst = @as(usize, @intCast(pos)) * self.config.n_kv_heads * self.config.head_dim + kvh * self.config.head_dim;
                        for (0..self.config.head_dim) |d| self.kv.v_cache[l][dst + d] = vs[d];
                    }

                    // Causal attention per token
                    const seq_len = pos + 1;
                    self.cpuAttention(
                        qkv_slice[0 .. self.config.n_heads * self.config.head_dim],
                        self.scratch.attn_out,
                        @as(u32, @intCast(l)),
                        @as(u32, @intCast(seq_len)),
                        self.config.n_heads,
                        self.config.n_kv_heads,
                        self.config.head_dim,
                        self.config.gqa_ratio,
                    );
                }

                // O projection: CPU GEMV when dispatched to CPU/GPU, NPU
                // otherwise -- matches the (dispatch.attention == .cpu or
                // dispatch.attention == .gpu) convention already used by
                // every decode path (lines ~1622, ~1753, ~2131). O-proj
                // has no dispatch field of its own, so it mirrors
                // attention's dispatch; this prefill call site previously
                // only checked `== .cpu`, so whenever attention was
                // dispatched to GPU (e.g. .auto, where attention
                // deliberately prefers GPU) O-proj still always routed to
                // NPU regardless of whether NPU was actually alive.
                if ((dispatch.attention == .cpu or dispatch.attention == .gpu) and l < self.cpu_weights.o.len) {
                    cpuGemv(
                        self.cpu_weights.o[l],
                        self.scratch.attn_out[0 .. self.config.n_heads * self.config.head_dim],
                        self.scratch.o_out[0..H],
                        H,
                        self.config.n_heads * self.config.head_dim,
                    );
                } else {
                    self.npu.runOProj(
                        self.scratch.attn_out[0 .. self.config.n_heads * self.config.head_dim],
                        layer_u32, 1, self.scratch.o_out[0..H],
                    ) catch |err| {
                        log.warn("NPU O-proj (layer {d}) failed: {s}", .{ l, @errorName(err) });
                        self.npu_broken = true;
                        @memset(self.scratch.o_out[0..H], 0);
                    };
                }
                for (0..H) |i| hidden_slice[i] = self.scratch.residual[@as(usize, @intCast(pi)) * H + i] + self.scratch.o_out[i];

                // FFN: branch on kernel type
                @memcpy(self.scratch.residual[@as(usize, @intCast(pi)) * H ..][0..H], hidden_slice);
                rmsNorm(hidden_slice, self.pa_norm[l], hidden_slice, 1e-6);

                switch (self.ffn_kernel) {
                    .dense => {
                        const IM = self.config.inter_size;
                        const ffn_cpu_ready = dispatch.ffn == .cpu and l < self.cpu_weights.gate.len and l < self.cpu_weights.up.len and l < self.cpu_weights.down.len;
                        var gate_up_ok = true;
                        if (ffn_cpu_ready) {
                            cpuGemv(self.cpu_weights.gate[l], hidden_slice, self.scratch.gate_up[0..IM], IM, H);
                            cpuGemv(self.cpu_weights.up[l], hidden_slice, self.scratch.gate_up[IM..][0..IM], IM, H);
                        } else {
                            self.npu.runFFN(hidden_slice, layer_u32, 1, self.scratch.gate_up) catch |err| {
                                log.warn("NPU gate/up (layer {d}) failed: {s}", .{ l, @errorName(err) });
                                self.npu_broken = true;
                                // Self-heal this token's gate/up on CPU
                                // instead of zero-filling -- same reasoning
                                // as the QKV catch above: this token's
                                // hidden state feeds forward into every
                                // later layer, so zeroing it corrupts the
                                // rest of the run rather than just costing
                                // one failed round-trip.
                                if (l < self.cpu_weights.gate.len and l < self.cpu_weights.up.len) {
                                    cpuGemv(self.cpu_weights.gate[l], hidden_slice, self.scratch.gate_up[0..IM], IM, H);
                                    cpuGemv(self.cpu_weights.up[l], hidden_slice, self.scratch.gate_up[IM..][0..IM], IM, H);
                                } else {
                                    @memset(self.scratch.gate_up, 0);
                                    gate_up_ok = false;
                                }
                            };
                        }
                        if (gate_up_ok) {
                            for (0..IM) |i| {
                                const gate = self.scratch.gate_up[i];
                                const up = self.scratch.gate_up[IM + i];
                                self.scratch.activated[i] = silu(if (std.math.isFinite(gate)) gate else 0) * up;
                            }
                            if (ffn_cpu_ready or (self.npu_broken and l < self.cpu_weights.down.len)) {
                                cpuGemv(self.cpu_weights.down[l], self.scratch.activated[0..IM], self.scratch.down_out[0..H], H, IM);
                            } else {
                                _ = self.npu.runDown(
                                    self.scratch.activated[0..IM],
                                    layer_u32, 1, self.scratch.down_out[0..H],
                                ) catch |err| {
                                    log.warn("NPU down (layer {d}) failed: {s}", .{ l, @errorName(err) });
                                    self.npu_broken = true;
                                    if (l < self.cpu_weights.down.len) {
                                        cpuGemv(self.cpu_weights.down[l], self.scratch.activated[0..IM], self.scratch.down_out[0..H], H, IM);
                                    } else {
                                        @memset(self.scratch.down_out[0..H], 0);
                                    }
                                };
                            }
                        } else {
                            @memset(self.scratch.activated, 0);
                            @memset(self.scratch.down_out[0..H], 0);
                        }
                        for (0..H) |i| hidden_slice[i] = self.scratch.residual[@as(usize, @intCast(pi)) * H + i] + self.scratch.down_out[i];
                    },
                    .moe, .shared_moe => {
                        // Single-token MoE prefill: route token, run expert(s)
                        const mc = self.moe_config orelse return;
                        const ms = self.moe_scratch orelse return;

                        // Gating
                        self.computeGatingLogits(hidden_slice, @as(u32, @intCast(l)), 1, ms.gating_logits);

                        const ctx = self.moe_ctx orelse return;
                        const routing = ctx.route(ms.gating_logits[0..mc.n_experts], 1, @as(u32, @intCast(l))) catch {
                            for (0..H) |i| hidden_slice[i] = self.scratch.residual[@as(usize, @intCast(pi)) * H + i];
                            continue;
                        };

                        @memset(ms.expert_outputs[0..mc.top_k * H], 0);
                        for (0..mc.top_k) |k| {
                            const e = routing.expertId(0, @intCast(k));
                            if (e >= mc.n_experts) continue;

                            // Gate/Up for this expert
                            try self.npu.runMoeGateUp(hidden_slice, e, 1, ms.gate_up[0..2 * mc.expert_intermediate_size]);
                            for (0..mc.expert_intermediate_size) |i| {
                                const gate = ms.gate_up[i];
                                const up = ms.gate_up[mc.expert_intermediate_size + i];
                                ms.activated[i] = silu(if (std.math.isFinite(gate)) gate else 0) * up;
                            }
                            try self.npu.runMoeDown(
                                ms.activated[0..mc.expert_intermediate_size], e, 1, ms.down_out[0..H],
                            );
                            @memcpy(ms.expert_outputs[k * H ..][0..H], ms.down_out[0..H]);
                        }

                        // Weighted accumulate
                        @memset(hidden_slice, 0);
                        for (0..mc.top_k) |k| {
                            const w = routing.weight(0, @intCast(k));
                            if (w == 0) continue;
                            for (0..H) |i| hidden_slice[i] += w * ms.expert_outputs[k * H + i];
                        }

                        // Shared expert
                        if (mc.has_shared_expert) {
                            try self.npu.runSharedGateUp(hidden_slice, @as(u32, @intCast(l)), 1, ms.gate_up[0..2 * mc.expert_intermediate_size]);
                            for (0..mc.expert_intermediate_size) |i| {
                                const gate = ms.gate_up[i];
                                const up = ms.gate_up[mc.expert_intermediate_size + i];
                                ms.activated[i] = silu(if (std.math.isFinite(gate)) gate else 0) * up;
                            }
                            try self.npu.runSharedDown(
                                ms.activated[0..mc.expert_intermediate_size],
                                @as(u32, @intCast(l)), 1, ms.down_out[0..H],
                            );
                            for (0..H) |i| hidden_slice[i] += ms.down_out[i];
                        }

                        // Residual add
                        for (0..H) |i| hidden_slice[i] = self.scratch.residual[@as(usize, @intCast(pi)) * H + i] + hidden_slice[i];
                    },
                }
            }
        }

        if (self.attn_kernel == .mla) {
            if (self.mla_kv) |*mk| mk.advance(npt);
        } else {
            self.kv.advance(npt);
        }
    }

    /// Run M=128 batch decode: full forward pass for all batch positions.
    pub fn decodeBatch(self: *FusedExecutor, tokens: []const u32) !u32 {
        const B = @as(u32, @intCast(tokens.len));
        const H = self.config.hidden_dim;

        // Forward all B tokens through the model (processes them in parallel
        // where possible, or sequentially with pipelined NPU+GPU operations).
        // Each token at position i writes its hidden state to hidden[i*H..(i+1)*H].
        try self.forwardDecode(tokens, B, self.scratch.hidden[0..B * H]);

        // Sample from the LAST token in the batch (autoregressive generation:
        // token[i+1] depends on token[i], so only the last token is new output).
        var top_ids: [128]u32 = undefined;
        try self.lmHead(self.scratch.hidden[(B - 1) * H .. B * H], &top_ids, 128);

        // Advance KV cache by B positions (each token consumed its own cache slot)
        if (self.attn_kernel == .mla) {
            if (self.mla_kv) |*mk| mk.advance(B);
        } else {
            self.kv.advance(B);
        }
        return top_ids[0];
    }

    /// Predict the first generated token directly from prefill()'s
    /// already-computed hidden state for the last prompt position, without
    /// any redundant reprocessing.
    ///
    /// Calling decodeBatch(&.{last_prompt_token}) right after prefill() (the
    /// naive way to get the first generated token) re-embeds and reprocesses
    /// the last prompt token as if it were new input, at position
    /// kv.position (== prompt_len, one past where prefill() actually placed
    /// it: prompt_len - 1). RoPE encodes absolute position into the
    /// rotation, so this feeds the model a token at the wrong position,
    /// discarding the real, correctly-positioned hidden state prefill()
    /// already computed for it -- prefill()'s per-position loop leaves that
    /// state sitting in scratch.hidden[(prompt_len-1)*H..] unused.
    pub fn firstDecodeToken(self: *FusedExecutor, prompt_len: u32) !u32 {
        const H = self.config.hidden_dim;
        // prefill() doesn't apply the final RMSNorm itself (only
        // forwardDecode's tail does, right before its own lmHead call) --
        // normalize a copy so lmHead sees the same kind of input either way.
        var normed: [4096]f32 = undefined;
        const src = self.scratch.hidden[(prompt_len - 1) * H ..][0..H];
        rmsNorm(src, self.final_norm, normed[0..H], 1e-6);

        var top_ids: [128]u32 = undefined;
        try self.lmHead(normed[0..H], &top_ids, 128);
        return top_ids[0];
    }
};

// ── Default MLA config (static for temporary use) ───────────

var default_mla_config: mla_attn.MLAConfig = undefined;

// ── Tests ───────────────────────────────────────────────────

test "FusedExecutor dispatch policies" {
    const allocator = std.testing.allocator;
    const H: u32 = 1536;

    const emb = try allocator.alloc(f32, 100 * H);
    @memset(emb, 0);
    const fnorm = try allocator.alloc(f32, H);
    @memset(fnorm, 1.0);

    var innorm = try allocator.alloc([]f32, 4);
    var panorm = try allocator.alloc([]f32, 4);
    const hdnorm = try allocator.alloc(f32, 128);
    @memset(hdnorm, 1.0);
    var qnorm = try allocator.alloc([]f32, 4);
    var knorm = try allocator.alloc([]f32, 4);
    for (0..4) |l| {
        innorm[l] = fnorm;
        panorm[l] = fnorm;
        qnorm[l] = hdnorm;
        knorm[l] = hdnorm;
    }

    const rope_sin = try allocator.alloc(f32, 4096 * 128);
    @memset(rope_sin, 0);
    var rope_cos = try allocator.alloc(f32, 4096 * 128);
    for (0..4096 * 128) |i| rope_cos[i] = 1.0;

    var exec = try FusedExecutor.init(
        allocator, null, .ffn_on_npu, QWEN3_0_6B,
        "/tmp/model.q4nx", "/tmp/npu_engine",
        4096, 128,
        emb, null, false,
        fnorm, innorm, panorm,
        qnorm, knorm,
        rope_sin, rope_cos,
        .{},
    );
    defer exec.deinit();

    // Default kernels should be flash/dense
    try std.testing.expectEqual(arch_registry.AttentionKernel.flash, exec.attn_kernel);
    try std.testing.expectEqual(arch_registry.FfnKernel.dense, exec.ffn_kernel);

    const dispatch = exec.getLayerDispatch(0);
    try std.testing.expectEqual(Backend.npu, dispatch.qkv);
    try std.testing.expectEqual(Backend.gpu, dispatch.attention);
    try std.testing.expectEqual(Backend.npu, dispatch.ffn);
}

test "SharedKVCache init and write" {
    const allocator = std.testing.allocator;
    var kv = try SharedKVCache.init(allocator, 28, 2, 128, 4096);
    defer kv.deinit();

    try std.testing.expectEqual(@as(u32, 0), kv.position);

    var k_data: [2 * 128]f32 = undefined;
    var v_data: [2 * 128]f32 = undefined;
    @memset(&k_data, 1.0);
    @memset(&v_data, 2.0);

    kv.writeKV(0, 2, 128, &k_data, &v_data, 1);
    kv.advance(1);

    try std.testing.expectEqual(@as(u32, 1), kv.position);

    const k_ref = kv.getK(0);
    try std.testing.expectEqual(@as(f32, 1.0), k_ref[0]);
    const v_ref = kv.getV(0);
    try std.testing.expectEqual(@as(f32, 2.0), v_ref[0]);
}

test "MlaKVCache init and write" {
    const allocator = std.testing.allocator;
    var mkv = try MlaKVCache.init(allocator, 28, 256, 4096);
    defer mkv.deinit();

    try std.testing.expectEqual(@as(u32, 0), mkv.position);

    var latent: [256]f32 = undefined;
    @memset(&latent, 3.14);

    mkv.write(0, &latent);
    mkv.advance(1);

    try std.testing.expectEqual(@as(u32, 1), mkv.position);

    const read = mkv.read(0, 0);
    try std.testing.expectEqual(@as(f32, 3.14), read[0]);
    try std.testing.expectEqual(@as(f32, 3.14), read[255]);
}

test "rmsNorm produces correct output shape" {
    var input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var output: [4]f32 = undefined;
    rmsNorm(&input, &weight, &output, 1e-6);
    var ss: f64 = 0;
    for (&output) |v| ss += @as(f64, @floatCast(v)) * @as(f64, @floatCast(v));
    const mean_sq = @as(f32, @floatCast(ss / 4.0));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mean_sq, 0.01);
}

test "silu produces expected values" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), silu(-10.0), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), silu(0.0), 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), silu(10.0), 0.01);
}

test "cpuAttention produces non-nan output" {
    const allocator = std.testing.allocator;

    var q: [1 * 128]f32 = undefined;
    var out: [1 * 128]f32 = undefined;
    @memset(&q, 0.5);
    @memset(&out, 0);

    var exec = try FusedExecutor.init(
        allocator, null, .npu_only, QWEN3_0_6B,
        "/tmp/model.q4nx", "/tmp/npu_engine",
        64, 128,
        try allocator.alloc(f32, 100 * 1536),
        null, false,
        try allocator.alloc(f32, 1536),
        try allocator.alloc([]f32, 1),
        try allocator.alloc([]f32, 1),
        try allocator.alloc([]f32, 1),
        try allocator.alloc([]f32, 1),
        try allocator.alloc(f32, 64 * 128),
        try allocator.alloc(f32, 64 * 128),
        .{},
    );
    defer exec.deinit();

    var kd: [2 * 128]f32 = undefined;
    var vd: [2 * 128]f32 = undefined;
    @memset(&kd, 0.5);
    @memset(&vd, 0.5);
    exec.kv.writeKV(0, 2, 128, &kd, &vd, 1);
    exec.kv.advance(1);

    exec.cpuAttention(&q, &out, 0, 1, 1, 2, 128, 6);
    for (&out) |v| try std.testing.expect(std.math.isFinite(v));
}

test "initWithKernels flash/dense (backward compat)" {
    const allocator = std.testing.allocator;
    const H: u32 = 1536;

    const emb = try allocator.alloc(f32, 100 * H);
    @memset(emb, 0);
    const fnorm = try allocator.alloc(f32, H);
    @memset(fnorm, 1.0);
    var innorm = try allocator.alloc([]f32, 4);
    var panorm = try allocator.alloc([]f32, 4);
    const hdnorm = try allocator.alloc(f32, 128);
    @memset(hdnorm, 1.0);
    var qnorm = try allocator.alloc([]f32, 4);
    var knorm = try allocator.alloc([]f32, 4);
    for (0..4) |l| {
        innorm[l] = fnorm;
        panorm[l] = fnorm;
        qnorm[l] = hdnorm;
        knorm[l] = hdnorm;
    }
    const rope_sin = try allocator.alloc(f32, 4096 * 128);
    @memset(rope_sin, 0);
    var rope_cos = try allocator.alloc(f32, 4096 * 128);
    for (0..4096 * 128) |i| rope_cos[i] = 1.0;

    var exec = try FusedExecutor.initWithKernels(
        allocator, null, .ffn_on_npu, QWEN3_0_6B,
        "/tmp/model.q4nx", "/tmp/npu_engine",
        4096, 128,
        emb, null, false,
        fnorm, innorm, panorm, qnorm, knorm,
        rope_sin, rope_cos,
        .{},
        .flash, .dense, // same as default init()
        null, null, // no MoE
        null, null, null, null, null, null, // no MLA
    );
    defer exec.deinit();

    try std.testing.expectEqual(arch_registry.AttentionKernel.flash, exec.attn_kernel);
    try std.testing.expectEqual(arch_registry.FfnKernel.dense, exec.ffn_kernel);

    const dispatch = exec.getLayerDispatch(0);
    try std.testing.expectEqual(Backend.npu, dispatch.qkv);
    try std.testing.expectEqual(Backend.gpu, dispatch.attention);
    try std.testing.expectEqual(Backend.npu, dispatch.ffn);
}
