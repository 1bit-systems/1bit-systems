//! Unified FusedEngine — wraps NPU (XRT), GPU (Vulkan/Metal/CUDA), and CPU backends
//! behind a single inference interface. The dispatcher (dispatcher.zig) handles
//! per-layer routing, and the KV cache is shared via the unified scheduler.
//!
//! Architecture:
//!   ┌────────────────────────────────────────────┐
//!   │              FusedEngine                   │
//!   │  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
//!   │  │ NPU     │  │ GPU     │  │ CPU     │    │
//!   │  │ (XRT)   │  │(Vulkan) │  │(C++ TRG)│    │
//!   │  └────┬────┘  └────┬────┘  └────┬────┘    │
//!   │       └──────┬──────┘           │         │
//!   │     Shared KV Cache (scheduler/) │         │
//!   └────────────────────────────────────────────┘
//!
//! @section Fused Engine
const std = @import("std");

const sched_mod = @import("sched");
const npu_engine = @import("npu_engine");
const cpu_backend = @import("cpu_backend");

const dispatcher = @import("dispatcher.zig");
const memory = @import("memory.zig");
const vk_wrapper = @import("vk_wrapper");

pub const KvPagePool = sched_mod.KvPagePool;
pub const EvictionPolicy = sched_mod.EvictionPolicy;
pub const Scheduler = sched_mod.Scheduler;
pub const Request = sched_mod.Request;
pub const RequestState = sched_mod.RequestState;
pub const GenerationParams = sched_mod.GenerationParams;
pub const Dispatcher = dispatcher.Dispatcher;
pub const DispatchPolicy = dispatcher.DispatchPolicy;
pub const CrossBackendMemory = memory.CrossBackendMemory;
pub const SharedBuffer = memory.SharedBuffer;
pub const VkContext = vk_wrapper.VkContext;
pub const Connection = struct {
    stream: std.net.Stream,

    pub fn init(stream: std.net.Stream) Connection {
        return .{ .stream = stream };
    }

    pub fn close(self: *Connection) void {
        self.stream.close();
        self.* = undefined;
    }

    pub fn reader(self: *Connection) std.net.Stream.Reader {
        return self.stream.reader();
    }

    pub fn writer(self: *Connection) std.net.Stream.Writer {
        return self.stream.writer();
    }
};

pub const ServerConfig = struct {
    port: u16 = 8080,
    max_parallel: u32 = 4,
    total_kv_pages: u32 = 1024,
    dispatch_policy: DispatchPolicy = .auto,
    model_path: []const u8 = "",
    xclbin_dir: []const u8 = "",
    model_tag: []const u8 = "",
};

const log = std.log.scoped(.fused_engine);

/// Which backend to use for a given inference operation.
pub const BackendDevice = enum(u8) {
    /// NPU (XDNA 2 via XRT xclbin kernels)
    npu = 0,
    /// GPU (Vulkan / Metal / CUDA compute)
    gpu = 1,
    /// CPU (pure C++ ternary inference)
    cpu = 2,
};

/// Capabilities advertised by each backend.
pub const BackendCapabilities = struct {
    supports_int8_gemm: bool = false,
    supports_fp16_gemm: bool = false,
    supports_flash_attention: bool = false,
    supports_moe: bool = false,
    supports_ssm: bool = false,
    supports_radix_attention: bool = false,
    max_batch_tokens: u32 = 1,
};

/// Runtime state for one fused decode step.
pub const FusedDecodeState = struct {
    /// Which backend handled the most recent step.
    last_backend: BackendDevice = .gpu,
    /// Allocator for owned resources.
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) FusedDecodeState {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *FusedDecodeState) void {
        self.* = undefined;
    }
};

/// Top-level fused inference engine.
/// Owns the NPU device (XRT), the GPU device (Vulkan), the CPU backend,
/// and the shared KV cache.
pub const FusedEngine = struct {
    allocator: std.mem.Allocator,

    // NPU backend
    npu: ?*npu_engine.NpuEngine = null,
    npu_available: bool = false,

    // GPU backend (Vulkan)
    gpu: ?VkContext = null,
    gpu_available: bool = false,
    /// Vulkan device memory handle for imported KV cache dma-buf.
    gpu_kv_cache_memory: u64 = 0,

    // CPU backend (always available)
    cpu: cpu_backend.CpuBackend,
    cpu_available: bool = true,

    // Shared scheduler / KV cache
    scheduler: Scheduler,
    kv_page_pool: KvPagePool,
    cross_memory: CrossBackendMemory,

    // KV cache backing buffer (GTT dma-buf shared between GPU, CPU, NPU)
    kv_cache_buffer: ?SharedBuffer,
    /// Size of bytes per KV cache page (pre-calculated).
    kv_bytes_per_page: u32,

    // Dispatch policy (how to route layers)
    dispatch_policy: DispatchPolicy,

    // Model metadata
    n_layers: u32,
    hidden_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,

    /// Initialize the fused engine.
    /// If a backend fails to init, the engine proceeds with the remaining backends.
    pub fn init(
        allocator: std.mem.Allocator,
        model_path: []const u8,
        xclbin_dir: []const u8,
        model_tag: []const u8,
        max_parallel: u32,
        total_kv_pages: u32,
        policy: DispatchPolicy,
    ) !FusedEngine {
        var kv_page_pool = try KvPagePool.initWithEviction(
            allocator, total_kv_pages, 16, .h2o_attention_score,
        );
        const scheduler = try Scheduler.init(allocator, max_parallel, &kv_page_pool, 2);
        const cross_memory = try CrossBackendMemory.initWithProbe(allocator);

        var engine = FusedEngine{
            .allocator = allocator,
            .scheduler = scheduler,
            .kv_page_pool = kv_page_pool,
            .cross_memory = cross_memory,
            .kv_cache_buffer = null,
            .kv_bytes_per_page = 0,
            .gpu = null,
            .gpu_kv_cache_memory = 0,
            .dispatch_policy = policy,
            .n_layers = 0,
            .hidden_dim = 0,
            .n_heads = 0,
            .n_kv_heads = 0,
            .head_dim = 0,
            // CPU backend initialized below after model config is known
            .cpu = undefined,
        };

        // Try NPU backend
        engine.npu = engine.tryInitNpu(allocator, model_path, xclbin_dir, model_tag, max_parallel, total_kv_pages) catch |err| {
            log.warn("NPU backend unavailable: {s}. Fusing without NPU.", .{@errorName(err)});
            null;
        };
        engine.npu_available = engine.npu != null;
        if (engine.npu_available) {
            engine.n_layers = engine.npu.?.config.NC;
            engine.hidden_dim = engine.npu.?.config.H;
            engine.n_heads = engine.npu.?.config.NH;
            engine.n_kv_heads = engine.npu.?.config.NKV;
            engine.head_dim = engine.npu.?.config.HD;
        }

        // Try GPU backend (Vulkan) — init after model config is known.
        if (vk_wrapper.VkContext.init(allocator)) |vk_ctx| {
            engine.gpu = vk_ctx;
            engine.gpu_available = true;
            log.info("Vulkan GPU backend ready", .{});
        } else |err| {
            log.warn("Vulkan GPU backend unavailable: {s}", .{@errorName(err)});
            engine.gpu_available = false;
        }

        // CPU backend always available — init with defaults (weights loaded separately)
        engine.cpu = cpu_backend.CpuBackend.init(allocator, .{
            .hidden_dim = engine.hidden_dim,
            .inter_size = 0,
            .n_heads = engine.n_heads,
            .n_kv_heads = engine.n_kv_heads,
            .head_dim = engine.head_dim,
            .vocab_size = 0,
            .n_layers = engine.n_layers,
            .max_seq_len = 4096,
            .rms_norm_eps = 1e-6,
        }) catch |err| {
            log.warn("CPU backend init failed: {s}", .{@errorName(err)});
            engine.cpu_available = false;
            return err;
        };
        log.info("CPU backend ready (always available)", .{});

        // Allocate KV cache backing buffer as a shared GTT dma-buf.
        const n_kv_heads_actual = if (engine.npu_available) engine.npu.?.config.NKV else 8;
        const head_dim_actual = if (engine.npu_available) engine.npu.?.config.HD else 64;
        const n_layers_actual = if (engine.npu_available) engine.npu.?.config.NC else 28;
        const kv_slots_per_page: u32 = 16; // Must match KvPagePool page_size
        const kv_bytes_per_slot: u32 = 2 * n_kv_heads_actual * head_dim_actual * 2; // K+V * fp16
        engine.kv_bytes_per_page = kv_slots_per_page * kv_bytes_per_slot;
        const kv_cache_total_bytes = @as(u64, total_kv_pages) * @as(u64, engine.kv_bytes_per_page) * @as(u64, n_layers_actual);

        if (cross_memory.isAvailable()) {
            const buf = cross_memory.allocateShared(kv_cache_total_bytes);
            if (buf.valid) {
                engine.kv_cache_buffer = buf;
                log.info("KV cache buffer: {} pages × {}/page × {} layers = {} MB (GTT dma-buf)", .{
                    total_kv_pages, engine.kv_bytes_per_page, n_layers_actual,
                    @divFloor(kv_cache_total_bytes, 1024 * 1024),
                });
            } else {
                log.warn("KV cache GTT allocation failed — backends will use staging", .{});
            }
        } else {
            log.warn("dma-buf unavailable — KV cache uses staging copies", .{});
        }

        // Import KV cache dma-buf into Vulkan for GPU-side access.
        if (engine.gpu_available and engine.kv_cache_buffer) |*buf| {
            if (buf.valid and buf.dma_buf_fd >= 0) {
                const vk = engine.gpu.?;
                if (vk.importDmaBuf(buf.dma_buf_fd, buf.size)) |mem| {
                    engine.gpu_kv_cache_memory = mem;
                    log.info("KV cache dma-buf imported into Vulkan: VkDeviceMemory=0x{x}", .{mem});
                } else |err| {
                    log.warn("Vulkan dma-buf import failed: {s}", .{@errorName(err)});
                    engine.gpu_kv_cache_memory = 0;
                    engine.gpu_available = false;
                }
            } else {
                engine.gpu_available = false;
            }
        } else {
            engine.gpu_available = false;
        }

        log.info("FusedEngine ready: NPU={} GPU={} CPU={} policy={s}", .{
            engine.npu_available, engine.gpu_available, engine.cpu_available, @tagName(policy),
        });

        return engine;
    }

    fn tryInitNpu(
        self: *FusedEngine,
        allocator: std.mem.Allocator,
        model_path: []const u8,
        xclbin_dir: []const u8,
        model_tag: []const u8,
        max_parallel: u32,
        total_kv_pages: u32,
    ) !*npu_engine.NpuEngine {
        _ = self;
        const npu = try allocator.create(npu_engine.NpuEngine);
        errdefer allocator.destroy(npu);
        npu.* = try npu_engine.NpuEngine.init(
            allocator, model_path, xclbin_dir, model_tag, max_parallel, total_kv_pages,
        );
        return npu;
    }

    pub fn deinit(self: *FusedEngine) void {
        if (self.npu) |npu| {
            npu.deinit();
            self.allocator.destroy(npu);
        }
        // Clean up Vulkan GPU resources.
        if (self.gpu) |*vk| {
            if (self.gpu_kv_cache_memory != 0) {
                vk.freeMemory(self.gpu_kv_cache_memory);
            }
            vk.deinit();
        }
        self.cpu.deinit();
        self.scheduler.deinit();
        self.kv_page_pool.deinit();
        // Free KV cache backing buffer before cross_memory.
        if (self.kv_cache_buffer) |*buf| {
            self.cross_memory.freeShared(buf);
        }
        self.cross_memory.deinit();
    }

    /// Return capabilities of the active backends.
    pub fn capabilities(self: *const FusedEngine) BackendCapabilities {
        return .{
            .supports_int8_gemm = self.npu_available,
            .supports_fp16_gemm = self.cpu_available,
            .supports_flash_attention = self.gpu_available,
            .max_batch_tokens = if (self.npu_available) 4 else 1,
        };
    }

    /// Return the KV cache backing buffer for backend use.
    /// Returns null when dma-buf is unavailable (backends should use staging).
    pub fn getKvCacheBuffer(self: *const FusedEngine) ?*const SharedBuffer {
        if (self.kv_cache_buffer) |*buf| {
            if (buf.valid) return buf;
        }
        return null;
    }

    /// Return the Vulkan device memory handle for the imported KV cache.
    /// Returns 0 when not imported or unavailable.
    pub fn getGpuKvCacheMemory(self: *const FusedEngine) u64 {
        return if (self.gpu_available) self.gpu_kv_cache_memory else 0;
    }

    /// Return the Vulkan context for direct GPU operations.
    /// Returns null when Vulkan is unavailable.
    pub fn getVkContext(self: *FusedEngine) ?*VkContext {
        return if (self.gpu_available) &self.gpu.? else null;
    }

    /// Return the dma-buf fd for the KV cache backing buffer (for backend import).
    /// Returns -1 when dma-buf is unavailable.
    pub fn getKvCacheDmaBufFd(self: *const FusedEngine) i32 {
        if (self.kv_cache_buffer) |buf| {
            if (buf.valid) return buf.dma_buf_fd;
        }
        return -1;
    }

    /// Return the byte offset into the KV cache buffer for a given (layer, page_id, slot).
    pub fn kvCacheOffset(self: *const FusedEngine, layer: u32, page_id: u32, slot_in_page: u32) u64 {
        // Layout: [layer0][layer1]...[layerN] where each layer has [page0][page1]...[pageM]
        // Each page has kv_bytes_per_page bytes.
        return @as(u64, layer) * @as(u64, self.kv_page_pool.total_pages) * @as(u64, self.kv_bytes_per_page) +
               @as(u64, page_id) * @as(u64, self.kv_bytes_per_page) +
               @as(u64, slot_in_page) * @as(u64, self.kv_bytes_per_page / @as(u64, 16));
    }

    /// Select which backend handles a layer based on position (simple round-robin for testing).
    fn selectLayerBackend(self: *const FusedEngine, _position: u32) BackendDevice {
        _ = self;
        return if (_position % 2 == 0) .npu else .gpu;
    }

    /// Route attention computation to NPU (edge_attention kernel), FFN to GPU.
    fn selectAttentionNpu(self: *const FusedEngine, _position: u32) BackendDevice {
        _ = self;
        _ = _position;
        return .npu;
    }

    /// Route FFN computation to NPU (INT8 GEMM), attention to GPU.
    fn selectFfnNpu(self: *const FusedEngine, _position: u32) BackendDevice {
        _ = self;
        _ = _position;
        return .npu;
    }
};

test "FusedEngine type compiles" {
    try std.testing.expectEqual(@sizeOf(usize), @sizeOf(*FusedEngine));
}
