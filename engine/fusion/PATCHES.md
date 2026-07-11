// ── PATCHES for dispatcher.zig ─────────────────────────────────
// Add to BackendDevice enum:
//
//   /// CPU (pure C++ ternary inference, no GPU/NPU needed)
//   cpu = 2,
//
// Add to DispatchPolicy enum:
//
//   /// Use CPU for all computations (no GPU/NPU needed).
//   cpu_only = 8,
//   /// CPU fallback when other backends fail.
//   cpu_fallback = 9,
//
// Add to describe():
//
//   .cpu_only => "All layers → CPU ternary GEMV (pure C++, no accelerator)",
//   .cpu_fallback => "CPU fallback — use CPU when NPU/GPU unavailable",
//
// Add to assignForLayer() and getPolicyAssignment():
//
//   .cpu_only => .{ .attention = .cpu, .ffn = .cpu, .qkv = .cpu },
//   .cpu_fallback => .{ .attention = .cpu, .ffn = .cpu, .qkv = .cpu },

// ── PATCHES for engine.zig ─────────────────────────────────────
// Add import:
//
//   const cpu_backend = @import("cpu_backend");
//
// Add to BackendDevice enum:
//
//   /// CPU (pure C++ ternary inference)
//   cpu = 2,
//
// Add to FusedEngine fields:
//
//   // CPU backend
//   cpu: ?cpu_backend.CpuBackend = null,
//   cpu_available: bool = false,
//
// Add to init() before the log line:
//
//   // Try CPU backend (always available)
//   engine.cpu = try cpu_backend.CpuBackend.init(allocator, .{
//       .hidden_dim = engine.hidden_dim,
//       .inter_size = 0,  // filled from model
//       .n_heads = engine.n_heads,
//       .n_kv_heads = engine.n_kv_heads,
//       .head_dim = engine.head_dim,
//       .vocab_size = 0,
//       .n_layers = engine.n_layers,
//       .max_seq_len = 4096,
//       .rms_norm_eps = 1e-6,
//   });
//   engine.cpu_available = true;
//   log.info("CPU backend ready (always available)", .{});
//
// Update log line:
//
//   log.info("FusedEngine ready: NPU={} GPU={} CPU={} policy={s}", .{
//       engine.npu_available, engine.gpu_available, engine.cpu_available, @tagName(policy),
//   });
//
// Add to capabilities():
//
//   .supports_fp16_gemm = self.cpu_available,
//
// Add to deinit():
//
//   if (self.cpu) |*c| c.deinit();

// ── PATCHES for fused_execute.zig ──────────────────────────────
// Add to Backend enum:
//
//   cpu = 2,
//
// Add to getLayerDispatch():
//
//   .cpu_only => .{ .qkv = .cpu, .attention = .cpu, .ffn = .cpu },
//
// Add to main.rs or main.zig:
//
//   "cpu_only" => "CPU-only ternary inference (pure C++, no accelerator)",
//
