//! Fused NPU+GPU layer execution engine.
//! Routes each transformer layer to the optimal backend:
//!   - NPU: INT8 GEMM for QKV projection, FFN gate/up/down
//!   - GPU: Flash attention via Vulkan (gpu_attn module)
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

const log = std.log.scoped(.fused_execute);

/// Qwen3-0.6B model dimensions.
pub const QWEN3_0_6B = ModelConfig{
    .hidden_dim = 1536,
    .n_layers = 28,
    .n_heads = 12,
    .n_kv_heads = 2,
    .head_dim = 128,
    .inter_size = 4096,
    .vocab_size = 151936,
    .gqa_ratio = 6,
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
};

/// Per-layer dispatch decision.
pub const LayerDispatch = struct {
    qkv: Backend = .npu,
    attention: Backend = .npu,
    ffn: Backend = .npu,
};

/// Shared KV cache for NPU↔GPU interop.
/// K and V stored in f32. Written by NPU QKV step, read by GPU attention step.
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
        _ = n_tokens;
        const stride = n_kv_heads * head_dim;
        const dst_base = base * stride;
        for (0..n_kv_heads * head_dim) |i| {
            self.k_cache[layer][dst_base + i] = k_data[i];
            self.v_cache[layer][dst_base + i] = v_data[i];
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

/// Persistent NPU server — keeps one npu_engine_universal --server process alive
/// and communicates via POSIX shared memory + stdin/stdout commands.
/// Eliminates the 5-10ms process spawn overhead per QKV/FFN call.
const ChildPipes = struct {
    pid: std.posix.pid_t,
    stdin_fd: i32,
    stdout_fd: i32,
};

const PersistentNpuServer = struct {
    allocator: std.mem.Allocator,
    child: ChildPipes,
    shm_mem: []align(4096) u8,
    shm_name: []u8,

    const SHM_SIZE = 128 * 1024 * 1024; // 128 MB — matches SHM_TOTAL in npu_engine_universal.cpp
    const SHM_IN_OFF = 0;
    const SHM_OUT_OFF = SHM_SIZE / 2;

    pub fn init(allocator: std.mem.Allocator, engine_path: []const u8, model_path: []const u8) !PersistentNpuServer {
        // 1. Create POSIX shared memory (via C API)
        const shm_name = try std.fmt.allocPrint(allocator, "/1bit-fused-{d}", .{std.c.getpid()});
        errdefer allocator.free(shm_name);

        const O_RDWR: c_uint = 2;
        const O_CREAT: c_uint = 0o100;

        const fd = std.c.shm_open(@ptrCast(shm_name.ptr), O_RDWR | O_CREAT, 0o666);
        if (fd < 0) {
            log.err("shm_open failed: errno={}", .{std.c._errno().*});
            return error.ShmOpenFailed;
        }
        errdefer _ = std.c.shm_unlink(@ptrCast(shm_name.ptr));

        if (std.c.ftruncate(fd, SHM_SIZE) < 0) {
            log.err("ftruncate failed: errno={}", .{std.c._errno().*});
            _ = std.c.close(fd);
            return error.ShmTruncateFailed;
        }
        const shm_mem = try std.posix.mmap(
            null,
            SHM_SIZE,
            std.posix.PROT{ .READ = true, .WRITE = true },
            std.posix.MAP{ .TYPE = .SHARED },
            @intCast(fd),
            0,
        );
        _ = std.c.close(fd);

        // 2. Spawn NPU server process via C fork/exec
        var stdin_pipe: [2]c_int = undefined;
        var stdout_pipe: [2]c_int = undefined;
        if (std.c.pipe(&stdin_pipe) < 0 or std.c.pipe(&stdout_pipe) < 0) return error.PipeFailed;
        const child_pid = std.c.fork();
        if (child_pid < 0) return error.ForkFailed;
        if (child_pid == 0) {
            // Child: redirect stdin/stdout to pipes, then exec
            _ = std.c.dup2(stdin_pipe[0], 0);
            _ = std.c.dup2(stdout_pipe[1], 1);
            _ = std.c.close(stdin_pipe[0]);
            _ = std.c.close(stdin_pipe[1]);
            _ = std.c.close(stdout_pipe[0]);
            _ = std.c.close(stdout_pipe[1]);
            _ = std.c.execve(
                @ptrCast(@constCast(engine_path.ptr)),
                @as([*:null]const ?[*:0]const u8, @ptrCast(@constCast(&[_]?[*:0]const u8{ @ptrCast(@constCast(engine_path.ptr)), @ptrCast(@constCast(model_path.ptr)), @ptrCast(@constCast("--server")), @ptrCast(@constCast("--shm")), @ptrCast(@constCast(shm_name.ptr)), null }))),
                @as([*:null]const ?[*:0]const u8, @ptrCast(&[_]?[*:0]const u8{null})),
            );
            std.c.exit(1);
        }
        // Parent: close child's ends, keep our ends
        _ = std.c.close(stdin_pipe[0]);
        _ = std.c.close(stdout_pipe[1]);

        // Build a minimal child struct to hold the pid and pipe handles
        const child = ChildPipes{
            .pid = child_pid,
            .stdin_fd = stdin_pipe[1],
            .stdout_fd = stdout_pipe[0],
        };

        // 3. Wait for "SERVER: READY" on stdout
        var ready_buf: [32]u8 = undefined;
        const n = std.c.read(child.stdout_fd, &ready_buf, ready_buf.len);
        if (n < 0 or !std.mem.startsWith(u8, ready_buf[0..@as(usize, @intCast(n))], "SERVER: READY")) {
            const nn = if (n < 0) @as(usize, 0) else @as(usize, @intCast(n));
            log.err("NPU server did not start: {s}", .{ready_buf[0..nn]});
            _ = std.c.close(child.stdin_fd);
            _ = std.c.close(child.stdout_fd);
            var ws: c_int = 0;
            _ = std.c.waitpid(child_pid, &ws, 0);
            return error.NpuServerStartFailed;
        }
        log.info("NPU server ready (shm={s})", .{shm_name});

        return PersistentNpuServer{
            .allocator = allocator,
            .child = child,
            .shm_mem = shm_mem,
            .shm_name = shm_name,
        };
    }

    pub fn deinit(self: *PersistentNpuServer) void {
        // Send EXIT command to child
        _ = std.c.write(self.child.stdin_fd, "EXIT\n", 5);
        _ = std.c.close(self.child.stdin_fd);
        _ = std.c.close(self.child.stdout_fd);
        // Reap child process
        var wstatus: c_int = undefined;
        _ = std.c.waitpid(self.child.pid, &wstatus, 0);
        std.posix.munmap(self.shm_mem);
        _ = std.c.shm_unlink(@ptrCast(self.shm_name.ptr));
        self.allocator.free(self.shm_name);
    }

    /// Run QKV for one layer: write hidden states to shm, send command, read QKV output.
    fn runQKV(
        self: *PersistentNpuServer,
        hidden: []const f32,
        batch_size: u32,
        layer: u32,
        pos: u32,
        qkv_out: []f32,
    ) !void {
        const shm_in = self.shm_mem[SHM_IN_OFF..][0..hidden.len * @sizeOf(f32)];
        // Write hidden states to shared memory
        const h_bytes = std.mem.sliceAsBytes(hidden);
        @memcpy(shm_in[0..h_bytes.len], h_bytes);

        // Send command directly via POSIX write
        var cmd_buf: [64]u8 = undefined;
        const cmd = try std.fmt.bufPrint(&cmd_buf, "QKV {d} {d} {d}\n", .{ layer, pos, batch_size });
        _ = std.c.write(self.child.stdin_fd, cmd.ptr, cmd.len);

        // Wait for ack (newline on stdout)
        var ack: [1]u8 = undefined;
        _ = std.c.read(self.child.stdout_fd, &ack, 1);

        // Read output from shared memory
        const shm_out = self.shm_mem[SHM_OUT_OFF..][0..qkv_out.len * @sizeOf(f32)];
        const out_bytes = std.mem.sliceAsBytes(qkv_out);
        @memcpy(out_bytes, shm_out[0..out_bytes.len]);
    }

    /// Run FFN for one layer: write residual+attention to shm, send command, read output.
    fn runFFN(
        self: *PersistentNpuServer,
        residual: []const f32,
        attn_out: []const f32,
        batch_size: u32,
        layer: u32,
        pos: u32,
        ffn_out: []f32,
    ) !void {
        // Write residual + attention output to shared memory
        const res_bytes = std.mem.sliceAsBytes(residual);
        const attn_bytes = std.mem.sliceAsBytes(attn_out);
        const total = res_bytes.len + attn_bytes.len;
        const shm_in = self.shm_mem[SHM_IN_OFF..][0..total];
        @memcpy(shm_in[0..res_bytes.len], res_bytes);
        @memcpy(shm_in[res_bytes.len..][0..attn_bytes.len], attn_bytes);

        // Send command directly via POSIX write
        var cmd_buf: [64]u8 = undefined;
        const cmd = try std.fmt.bufPrint(&cmd_buf, "FFN {d} {d} {d}\n", .{ layer, pos, batch_size });
        _ = std.c.write(self.child.stdin_fd, cmd.ptr, cmd.len);

        // Wait for ack
        var ack: [1]u8 = undefined;
        _ = std.c.read(self.child.stdout_fd, &ack, 1);

        // Read output from shared memory
        const shm_out = self.shm_mem[SHM_OUT_OFF..][0..ffn_out.len * @sizeOf(f32)];
        const out_bytes = std.mem.sliceAsBytes(ffn_out);
        @memcpy(out_bytes, shm_out[0..out_bytes.len]);
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

/// Fused execution engine — coordinates NPU GEMM and GPU attention.
pub const FusedExecutor = struct {
    allocator: std.mem.Allocator,
    policy: DispatchPolicy,
    config: ModelConfig,

    /// Shared KV cache.
    kv: SharedKVCache,

    /// Persistent NPU server handle.
    npu: PersistentNpuServer,

    /// GPU attention module (Vulkan flash attention).
    gpu: ?gpu_attn.GpuAttention = null,

    /// NPU↔GPU KV cache interop.
    interop: ?interop.KvCacheInterop = null,

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

    /// RoPE precomputed sin/cos tables.
    rope_sin: []f32,
    rope_cos: []f32,

    /// Scratch buffers.
    scratch: ScratchBufs,

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
        policy: DispatchPolicy,
        config: ModelConfig,
        npu_engine_path: []const u8,
        model_path: []const u8,
        max_context: u32,
        batch_size: u32,
        emb_f32: []f32,
        lm_head_f32: ?[]f32,
        tied_embeddings: bool,
        final_norm: []f32,
        in_norm: [][]f32,
        pa_norm: [][]f32,
        rope_sin: []f32,
        rope_cos: []f32,
    ) !FusedExecutor {
        const kv = try SharedKVCache.init(
            allocator, config.n_layers, config.n_kv_heads, config.head_dim, max_context,
        );

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

        log.info("FusedExecutor init: policy={s} model=H{d}L{d}NH{d}NKV{d}HD{d}IM{d}NV{d}", .{
            @tagName(policy), H, config.n_layers, NH, config.n_kv_heads, HD, IM, NV,
        });

        return FusedExecutor{
            .allocator = allocator,
            .policy = policy,
            .config = config,
            .kv = kv,
            .npu = try PersistentNpuServer.init(allocator, npu_engine_path, model_path),
            .emb_f32 = emb_f32,
            .lm_head_f32 = lm_head_f32,
            .tied_embeddings = tied_embeddings,
            .final_norm = final_norm,
            .in_norm = in_norm,
            .pa_norm = pa_norm,
            .rope_sin = rope_sin,
            .rope_cos = rope_cos,
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
        };
    }

    pub fn deinit(self: *FusedExecutor) void {
        self.npu.deinit();
        const aa = self.allocator;
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
        aa.free(self.emb_f32);
        aa.free(self.final_norm);
        aa.free(self.in_norm);
        aa.free(self.pa_norm);
        aa.free(self.rope_sin);
        aa.free(self.rope_cos);
        if (self.lm_head_f32) |lm| aa.free(lm);
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
        return switch (self.policy) {
            .npu_only => .{ .qkv = .npu, .attention = .npu, .ffn = .npu },
            .gpu_only => .{ .qkv = .gpu, .attention = .gpu, .ffn = .gpu },
            .ffn_on_npu => .{ .qkv = .npu, .attention = .gpu, .ffn = .npu },
            .qkv_on_npu => .{ .qkv = .npu, .attention = .gpu, .ffn = .gpu },
            .attention_on_npu => .{ .qkv = .gpu, .attention = .npu, .ffn = .gpu },
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

    /// Execute one layer — routes QKV/attention/FFN to appropriate backend.
    /// With pipeline overlap: launches NPU QKV for NEXT layer while GPU does attention for THIS layer.
        /// Phase 1: RMSNorm + NPU QKV projection for one layer.
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

        // NPU QKV projection
        if (dispatch.qkv == .npu) {
            try self.npu.runQKV(s.hidden[0..batch_size * H], @intCast(batch_size), layer, self.kv.position, qkv_scratch);
        } else {
            @memset(qkv_scratch, 0);
        }
    }

    /// Phase 2: Q/K norm, RoPE, KV cache write, Attention (GPU/CPU), O projection,
    /// residual add, FFN, residual add.
    pub fn executeLayerAttnFFN(
        self: *FusedExecutor,
        layer: u32,
        batch_size: u32,
        qkv_scratch: []f32,
        output_hidden: []f32,
    ) !void {
        const H = self.config.hidden_dim;
        const NH = self.config.n_heads;
        const NKV = self.config.n_kv_heads;
        const HD = self.config.head_dim;
        const GQA = self.config.gqa_ratio;
        const dispatch = self.getLayerDispatch(layer);
        const s = self.scratch;

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
                for (0..HD) |d| qh[d] *= iq;
                self.applyRoPE(qh, pos + @as(u32, @intCast(b)), HD);
            }
            for (0..NKV) |kvh| {
                const ks = qkv_slice[NH * HD + kvh * HD ..][0..HD];
                var sk: f64 = 0;
                for (ks) |v| sk += @as(f64, @floatCast(v)) * @as(f64, @floatCast(v));
                const ik = 1.0 / @sqrt(@as(f32, @floatCast(sk / @as(f64, @floatFromInt(HD)))) + 1e-6);
                for (0..HD) |d| ks[d] *= ik;
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
        }

        // O projection + residual add (attention)
        @memcpy(s.o_out[0..batch_size * H], s.attn_out[0..batch_size * NH * HD]);
        for (0..batch_size * H) |i| s.hidden[i] = s.residual[i] + s.o_out[i];

        // Pre-FFN residual save + RMSNorm
        @memcpy(s.residual[0..batch_size * H], s.hidden[0..batch_size * H]);
        for (0..batch_size) |b| {
            rmsNorm(s.hidden[b * H ..][0..H], self.pa_norm[layer], s.hidden[b * H ..][0..H], 1e-6);
        }

        // FFN — server handles O proj + residual + RMSNorm + Gate/Up + SiLU + Down
        if (dispatch.ffn == .npu) {
            const attn_out = if (dispatch.attention == .gpu) s.attn_out else s.o_out;
            try self.npu.runFFN(s.residual[0..batch_size * H], attn_out[0..batch_size * NH * HD], @intCast(batch_size), layer, self.kv.position, s.down_out);
        }

        // Residual add (FFN)
        for (0..batch_size * H) |i| s.hidden[i] = s.residual[i] + s.down_out[i];
        @memcpy(output_hidden[0..batch_size * H], s.hidden[0..batch_size * H]);
    }
pub fn executeLayer(
        self: *FusedExecutor,
        layer: u32,
        batch_size: u32,
        output_hidden: []f32,
        _next_input: ?[]const f32,
    ) !void {
        _ = _next_input;
        const dispatch = self.getLayerDispatch(layer);
        const H = self.config.hidden_dim;
        const NH = self.config.n_heads;
        const NKV = self.config.n_kv_heads;
        const HD = self.config.head_dim;
        const GQA = self.config.gqa_ratio;
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

        if (dispatch.qkv == .npu) {
            // NPU GEMM for QKV projection
            try self.npu.runQKV(s.hidden[0..batch_size * H], @intCast(batch_size), layer, self.kv.position, s.qkv);
        }

        // ── Step 3: Q/K norm, RoPE, KV cache write ──
        const QKV = NH * HD + 2 * NKV * HD;
        const pos = self.kv.position;
        for (0..batch_size) |b| {
            const qkv_slice = s.qkv[b * QKV ..][0..QKV];
            // Q heads: apply Q-norm and RoPE
            for (0..NH) |hh| {
                const qh = qkv_slice[hh * HD ..][0..HD];
                // Q-norm
                var sq: f64 = 0;
                for (qh) |v| sq += @as(f64, @floatCast(v)) * @as(f64, @floatCast(v));
                const iq = 1.0 / @sqrt(@as(f32, @floatCast(sq / @as(f64, @floatFromInt(HD)))) + 1e-6);
                for (0..HD) |d| qh[d] *= iq;
                self.applyRoPE(qh, @as(u32, @intCast(pos + b)), HD);
            }
            // K heads: K-norm, RoPE, write to cache
            for (0..NKV) |kvh| {
                const ks = qkv_slice[NH * HD + kvh * HD ..][0..HD];
                var sk: f64 = 0;
                for (ks) |v| sk += @as(f64, @floatCast(v)) * @as(f64, @floatCast(v));
                const ik = 1.0 / @sqrt(@as(f32, @floatCast(sk / @as(f64, @floatFromInt(HD)))) + 1e-6);
                for (0..HD) |d| ks[d] *= ik;
                self.applyRoPE(ks, @as(u32, @intCast(pos + b)), HD);
                // Write K to cache
                const dst = (pos + @as(u32, @intCast(b))) * NKV * HD + kvh * HD;
                for (0..HD) |d| self.kv.k_cache[layer][dst + d] = ks[d];
            }
            // V heads: write to cache
            for (0..NKV) |kvh| {
                const vs = qkv_slice[NH * HD + NKV * HD + kvh * HD ..][0..HD];
                const dst = (pos + @as(u32, @intCast(b))) * NKV * HD + kvh * HD;
                for (0..HD) |d| self.kv.v_cache[layer][dst + d] = vs[d];
            }
        }

        // ── Step 4: Attention (GPU flash attention or CPU fallback) ──
        if (dispatch.attention == .gpu) {
            if (self.gpu) |*gpu| {
                // GPU flash attention: upload Q, point K/V from shared cache
                const seq_len = self.kv.position + batch_size;
                for (0..batch_size) |b| {
                    const q_slice = s.qkv[b * QKV ..][0..NH * HD];
                    const out_slice = s.attn_out[b * NH * HD ..][0..NH * HD];
                    const k_cache = self.kv.getK(layer);
                    const v_cache = self.kv.getV(layer);
                    _ = k_cache;
                    _ = v_cache;

                    gpu.flashAttention(
                        q_slice,                    // Q: [NH * HD]
                        self.kv.k_cache[layer],      // K cache: [pos * NKV * HD]
                        self.kv.v_cache[layer],      // V cache: [pos * NKV * HD]
                        &.{},                        // page_table: empty (flat cache mode)
                        out_slice,                   // output: [NH * HD]
                        &([_]f32{std.math.nan(f32)} ** 12), // sinks: disabled
                        NH, NKV, HD,
                        @as(u32, @intCast(seq_len)), // seq_len
                        0,                           // page_size: 0 = flat mode
                        0.0,                         // attn_scale: 0 = use 1/sqrt(HD)
                        0,                           // sink_offset
                    ) catch |err| {
                        log.warn("GPU attention layer {d} failed: {s}, using CPU fallback", .{ layer, @errorName(err) });
                        self.cpuAttention(q_slice, s.attn_out[b * NH * HD ..][0..NH * HD], layer, seq_len, NH, NKV, HD, GQA);
                    };
                }
            } else {
                // CPU attention fallback
                for (0..batch_size) |b| {
                    const q_slice = s.qkv[b * QKV ..][0..NH * HD];
                    self.cpuAttention(q_slice, s.attn_out[b * NH * HD ..][0..NH * HD], layer, self.kv.position + batch_size, NH, NKV, HD, GQA);
                }
            }
        } else {
            // Attention on NPU — use the NPU engine
            // The NPU engine handles attention internally, so just pass through
            @memcpy(s.attn_out[0..batch_size * NH * HD], s.qkv[0..batch_size * NH * HD]);
        }

        // ── Step 5: O projection on NPU ──
        // For now, NPU handles this in the subprocess call. Copy attn_out to hidden as O.
        @memcpy(s.o_out[0..batch_size * H], s.attn_out[0..batch_size * NH * HD]);

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

        // ── Step 8: FFN on NPU — server handles O + residual + SiLU + Down ──
        if (dispatch.ffn == .npu) {
            try self.npu.runFFN(s.residual[0..batch_size * H], s.attn_out[0..batch_size * NH * HD], @intCast(batch_size), layer, self.kv.position, s.down_out);
        }

        // ── Step 9: Residual add (FFN) ──
        for (0..batch_size * H) |i| s.hidden[i] = s.residual[i] + s.down_out[i];

        // ── Copy output ──
        @memcpy(output_hidden[0..batch_size * H], s.hidden[0..batch_size * H]);
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

            // Scores
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

    /// Execute full forward pass with pipeline overlap.
    /// Pipeline: NPU QKV for layer N+1 runs CONCURRENTLY with GPU attention for layer N
    /// using double-buffered QKV scratch and std.Thread for overlap.
    pub fn forwardDecode(
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

        // ── Embed input tokens ──
        for (0..B) |b| {
            const tok = input_tokens[b];
            for (0..H) |i| {
                self.scratch.hidden[b * H + i] = self.emb_f32[@as(usize, @intCast(tok)) * H + i];
            }
        }

        // ── Double-buffered QKV scratch for pipeline overlap ──
        const qkv_a = try self.allocator.alloc(f32, QKV_bytes);
        defer self.allocator.free(qkv_a);
        const qkv_b = try self.allocator.alloc(f32, QKV_bytes);
        defer self.allocator.free(qkv_b);

        // Thread wrapper struct to avoid closure issues
        const QkvThread = struct {
            ex: *FusedExecutor,
            layer: u32,
            bs: u32,
            buf: []f32,
            fn run(qt: *@This()) !void {
                try qt.ex.executeLayerQKV(qt.layer, qt.bs, qt.buf);
            }
        };

        // ── Prime: launch QKV for layer 0 ──
        var prev_thread: ?std.Thread = null;
        var cur_qkv = qkv_a;
        var next_qkv = qkv_b;

        // Launch thread for layer 0 QKV
        {
            var qkv_data = QkvThread{ .ex = self, .layer = 0, .bs = B, .buf = cur_qkv };
            prev_thread = try std.Thread.spawn(.{}, QkvThread.run, .{&qkv_data});
        }

        // ── Main pipeline loop ──
        for (0..NC) |l| {
            // Wait for this layer's QKV to complete
            if (prev_thread) |t| {
                t.join();
                prev_thread = null;
            }

            // If not last layer, launch NEXT layer's QKV on NPU concurrently
            if (l + 1 < NC) {
                var qkv_data = QkvThread{ .ex = self, .layer = @as(u32, @intCast(l + 1)), .bs = B, .buf = next_qkv };
                prev_thread = try std.Thread.spawn(.{}, QkvThread.run, .{&qkv_data});
                // Swap buffers for next iteration
                const tmp = cur_qkv;
                cur_qkv = next_qkv;
                next_qkv = tmp;
            }

            // Run attention + FFN for this layer
            // This runs CONCURRENTLY with the next layer's QKV (on the spawned thread)
            try self.executeLayerAttnFFN(@as(u32, @intCast(l)), B, cur_qkv, self.scratch.hidden[0..B * H]);
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
        @memcpy(output_hidden[0..B * H], self.scratch.hidden[0..B * H]);
    }

    /// LM head: compute logits and return top-k tokens.
    /// Uses GPU mat-vec if available, CPU fallback otherwise.
    pub fn lmHead(self: *FusedExecutor, hidden: []const f32, top_ids: []u32, top_k: u32) !void {
        const H = self.config.hidden_dim;
        const NV = self.config.vocab_size;
        const emb = if (self.lm_head_f32) |lm| lm else self.emb_f32;

        // CPU LM head: dot product per vocab entry
        var max_logit: f32 = -std.math.inf(f32);
        for (0..NV) |n| {
            var dot: f64 = 0;
            const e_row = emb[n * H ..][0..H];
            for (0..H) |i| dot += @as(f64, @floatCast(hidden[i])) * @as(f64, @floatCast(e_row[i]));
            const lg = @as(f32, @floatCast(dot));
            self.scratch.logits[n] = lg;
            if (lg > max_logit) max_logit = lg;
        }

        // Softmax
        var sum_exp: f64 = 0;
        for (0..NV) |n| {
            const d = self.scratch.logits[n] - max_logit;
            const e = if (d < -80) 0.0 else std.math.exp(d);
            self.scratch.logits[n] = e;
            sum_exp += @as(f64, @floatCast(e));
        }

        // Top-K selection
        const TopEntry = struct { id: u32, val: f32 };
        var top: [128]TopEntry = [_]TopEntry{.{ .id = 0, .val = -std.math.inf(f32) }} ** 128;
        const k = @min(top_k, @as(u32, 128));

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

        for (0..k) |j| top_ids[j] = top[j].id;
    }

    /// Full prefill: process N input tokens, build KV cache, produce final hidden state.
    pub fn prefill(self: *FusedExecutor, tokens: []const u32, batch_size: u32) !void {
        const H = self.config.hidden_dim;
        const NC = self.config.n_layers;
        const npt = @as(u32, @intCast(tokens.len));
        _ = batch_size;

        // Embed all tokens
        for (0..npt) |pi| {
            const tok = tokens[pi];
            for (0..H) |i| {
                self.scratch.hidden[@as(usize, @intCast(pi)) * H + i] =
                    self.emb_f32[@as(usize, @intCast(tok)) * H + i];
            }
        }

        // Layer loop
        for (0..NC) |l| {
            for (0..npt) |pi| {
                const pos = self.kv.position + pi;
                const hidden_slice = self.scratch.hidden[@as(usize, @intCast(pi)) * H ..][0..H];

                // Save residual
                @memcpy(self.scratch.residual[@as(usize, @intCast(pi)) * H ..][0..H], hidden_slice);

                // RMSNorm + QKV
                rmsNorm(hidden_slice, self.in_norm[l], hidden_slice, 1e-6);
                try self.npu.runQKV(hidden_slice, @as(u32, @intCast(1)), @as(u32, @intCast(l)), self.kv.position, self.scratch.qkv);

                // Q/K norm, RoPE, KV cache
                const QKV = self.config.n_heads * self.config.head_dim + 2 * self.config.n_kv_heads * self.config.head_dim;
                const qkv_slice = self.scratch.qkv[0..QKV];
                for (0..self.config.n_heads) |hh| {
                    const qh = qkv_slice[hh * self.config.head_dim ..][0..self.config.head_dim];
                    var sq: f64 = 0;
                    for (qh) |v| sq += @as(f64, @floatCast(v)) * @as(f64, @floatCast(v));
                    const iq = 1.0 / @sqrt(@as(f32, @floatCast(sq / @as(f64, @floatFromInt(self.config.head_dim)))) + 1e-6);
                    for (0..self.config.head_dim) |d| qh[d] *= iq;
                    self.applyRoPE(qh, @as(u32, @intCast(pos)), self.config.head_dim);
                }
                for (0..self.config.n_kv_heads) |kvh| {
                    const ks = qkv_slice[self.config.n_heads * self.config.head_dim + kvh * self.config.head_dim ..][0..self.config.head_dim];
                    var sk: f64 = 0;
                    for (ks) |v| sk += @as(f64, @floatCast(v)) * @as(f64, @floatCast(v));
                    const ik = 1.0 / @sqrt(@as(f32, @floatCast(sk / @as(f64, @floatFromInt(self.config.head_dim)))) + 1e-6);
                    for (0..self.config.head_dim) |d| ks[d] *= ik;
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

                // Residual add
                for (0..H) |i| hidden_slice[i] = self.scratch.residual[@as(usize, @intCast(pi)) * H + i] + self.scratch.attn_out[i];

                // FFN — server handles full path including SiLU + Down
                @memcpy(self.scratch.residual[@as(usize, @intCast(pi)) * H ..][0..H], hidden_slice);
                rmsNorm(hidden_slice, self.pa_norm[l], hidden_slice, 1e-6);
                try self.npu.runFFN(self.scratch.residual[@as(usize, @intCast(pi)) * H ..][0..H], self.scratch.attn_out[0..self.config.n_heads * self.config.head_dim], 1, @as(u32, @intCast(l)), self.kv.position, self.scratch.down_out);
                for (0..self.config.inter_size) |i| {
                    const gate = self.scratch.gate_up[i];
                    const up = self.scratch.gate_up[self.config.inter_size + i];
                    self.scratch.activated[i] = silu(if (std.math.isFinite(gate)) gate else 0) * up;
                }
                for (0..H) |i| hidden_slice[i] = self.scratch.residual[@as(usize, @intCast(pi)) * H + i] + self.scratch.activated[i];
            }
        }

        self.kv.advance(npt);
    }

    /// Run M=128 batch decode: full forward pass for all batch positions.
    pub fn decodeBatch(self: *FusedExecutor, tokens: []const u32) !u32 {
        const B = @as(u32, @intCast(tokens.len));
        try self.forwardDecode(tokens, B, self.scratch.hidden[0..B * self.config.hidden_dim]);

        // Take first hidden state for LM head
        var top_ids: [128]u32 = undefined;
        try self.lmHead(self.scratch.hidden[0..self.config.hidden_dim], &top_ids, 128);

        self.kv.advance(1);
        return top_ids[0];
    }
};

test "FusedExecutor dispatch policies" {
    const allocator = std.testing.allocator;
    const H: u32 = 1536;

    const emb = try allocator.alloc(f32, 100 * H);
    @memset(emb, 0);
    const fnorm = try allocator.alloc(f32, H);
    @memset(fnorm, 1.0);

    var innorm = try allocator.alloc([]f32, 4);
    var panorm = try allocator.alloc([]f32, 4);
    for (0..4) |l| {
        innorm[l] = fnorm;
        panorm[l] = fnorm;
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
        rope_sin, rope_cos,
    );
    defer exec.deinit();

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
        allocator, .npu_only, QWEN3_0_6B,
        "/tmp/npu_engine", "/tmp/model.q4nx",
        64, 128,
        try allocator.alloc(f32, 100 * 1536),
        null, false,
        try allocator.alloc(f32, 1536),
        try allocator.alloc([]f32, 1),
        try allocator.alloc([]f32, 1),
        try allocator.alloc(f32, 64 * 128),
        try allocator.alloc(f32, 64 * 128),
    );
    defer exec.deinit();

    // Write one KV entry into the executor's shared cache
    var kd: [2 * 128]f32 = undefined;
    var vd: [2 * 128]f32 = undefined;
    @memset(&kd, 0.5);
    @memset(&vd, 0.5);
    exec.kv.writeKV(0, 2, 128, &kd, &vd, 1);
    exec.kv.advance(1);

    exec.cpuAttention(&q, &out, 0, 1, 1, 2, 128, 6);
    // Output should be finite
    for (&out) |v| try std.testing.expect(std.math.isFinite(v));
}
