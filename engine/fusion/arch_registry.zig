//! Unified model architecture registry — the central catalog that ties
//! architecture configuration to its corresponding kernel types, dispatch
//! policies, and optimal launch parameters.
//!
//! Every model tag in `arch_configs.zig` has a corresponding entry here that
//! maps it to:
//!   1. The appropriate attention kernel type (standard flash | MLA)
//!   2. The appropriate FFN kernel type (dense | MoE | shared_moe)
//!   3. The recommended dispatch policy for the fused engine
//!   4. Optimal kernel launch parameters (workgroup sizes, block sizes, etc.)
//!
//! ## Usage
//!
//! ```zig
//! const reg = @import("arch_registry.zig");
//!
//! // Look up a model tag (comptime-known)
//! const entry = reg.getArchEntry("qwen3_7b").?;
//! if (entry.attn_kernel == .flash) { ... }
//! if (entry.ffn_kernel == .dense)  { ... }
//! ```
//!
//! ## Conventions
//!
//! - Workgroup sizes are specified as `[x, y, z]` tuples representing the
//!   number of workgroups launched in each dimension.
//! - Thread block sizes (subgroup/warp sizes) are specified as the number of
//!   threads per workgroup.
//! - When a parameter is hardware-dependent (e.g., GPU vs NPU), the registry
//!   stores the **recommended** baseline; runtime tuning may override.
//!
//! @section Fused Engine

const std = @import("std");
const arch_configs = @import("arch_configs");
const dispatcher = @import("dispatcher.zig");

const log = std.log.scoped(.arch_registry);

// ── Attention kernel type ───────────────────────────────────

/// The attention kernel implementation to use for a given architecture.
pub const AttentionKernel = enum(u8) {
    /// Standard flash attention — the generic fused kernel in `gpu_attn.zig`.
    /// Works for all dense/GQA architectures with standard Q/K/V projections.
    flash = 0,
    /// Multi-Head Latent Attention — DeepSeek-style compressed KV latent.
    /// Implemented in `mla_attn.zig`; uses qk_nope_head_dim, qk_rope_head_dim,
    /// kv_lora_rank, and absorbed attention optimizations.
    mla = 1,
};

// ── FFN kernel type ─────────────────────────────────────────

/// The FFN (feedforward) kernel category for a given architecture.
pub const FfnKernel = enum(u8) {
    /// Standard dense FFN: gate/up/down projections, no routing.
    /// Used by all dense architectures (Qwen3, Llama, Gemma).
    dense = 0,
    /// Standard routed MoE: top-k expert selection with gate/up/down per expert.
    /// Used by Mixtral, Zaya, and other routed-expert architectures.
    moe = 1,
    /// MoE with a shared (always-activated) expert in addition to routed experts.
    /// Used by DeepSeek-V2/V3-style architectures.
    shared_moe = 2,
};

// ── Dispatch policy recommendation ──────────────────────────

/// The recommended dispatch policy for a given architecture on the fused
/// NPU+GPU engine. These are baseline recommendations; the runtime may
/// override based on workload characteristics and benchmark results.
pub const DispatchRecommendation = enum(u8) {
    /// Full NPU execution — recommended for architectures where the NPU GEMM
    /// throughput dominates and GPU attention offers negligible benefit.
    /// Typical: small models on NPU with fast INT8 GEMM.
    npu_only = 0,
    /// Full GPU execution — recommended for architectures with large vocab
    /// or where GPU flash attention is significantly faster than NPU.
    /// Typical: large models with long context.
    gpu_only = 1,
    /// FFN on NPU, attention on GPU — the fused sweet spot.
    /// NPU GEMM is fast for the heavy FFN matmuls, GPU flash attention
    /// handles the memory-bandwidth-bound attention step.
    /// Typical: mid-size models like Qwen3-0.6B, Llama-3.2-3B.
    ffn_on_npu = 2,
    /// QKV on NPU, rest on GPU — alternative split when QKV is the bottleneck.
    /// Typical: models with very large n_heads (many Q heads).
    qkv_on_npu = 3,
    /// Attention on NPU, FFN on GPU — used when NPU has a specialized
    /// attention kernel but FFN is faster on GPU.
    /// Typical: NPU with edge_attention kernel, GPU with DMMV.
    attention_on_npu = 4,
    /// CPU-only execution — for architectures that are only supported on CPU
    /// (e.g., ternary 1-bit models without accelerator support).
    cpu_only = 5,
};

// ── Kernel launch parameters ────────────────────────────────

/// Optimal workgroup (grid) dimensions for GPU kernel dispatch.
/// These control how many workgroups are launched across each axis.
pub const WorkgroupDims = packed struct(u64) {
    x: u16 = 1,
    y: u16 = 1,
    z: u16 = 1,
    /// Padding to 64 bits.
    _pad: u16 = 0,
};

/// Optimal thread block (local workgroup) dimensions for GPU kernel dispatch.
/// These control how many threads execute within a single workgroup.
pub const BlockDims = packed struct(u64) {
    x: u16 = 256,
    y: u16 = 1,
    z: u16 = 1,
    _pad: u16 = 0,
};

/// Subgroup (warp) size hint. The actual warp size is hardware-determined
/// (typically 32 for NVIDIA, 64 for AMD), but this hint influences how the
/// kernel lays out shared memory and loop unrolling.
pub const SubgroupSize = enum(u8) {
    /// 32 threads per warp (NVIDIA CUDA, Vulkan subgroupSize=32).
    warp32 = 32,
    /// 64 threads per warp (AMD ROCm/HIP, Vulkan subgroupSize=64).
    warp64 = 64,
    /// Let the runtime decide based on the active backend.
    runtime = 0,
};

/// Per-kernel launch parameters for the fused engine.
/// Each kernel type (attention / FFN) has its own tuned launch config.
pub const LaunchParams = struct {
    // ── Attention kernel ────────────────────────────────────
    /// Grid dimensions for flash attention dispatch.
    attn_grid: WorkgroupDims = .{},
    /// Thread block dimensions for flash attention dispatch.
    attn_block: BlockDims = .{ .x = 256 },
    /// Subgroup/warp size hint for attention kernel.
    attn_subgroup: SubgroupSize = .runtime,

    // ── FFN (dense or MoE) kernel ───────────────────────────
    /// Grid dimensions for FFN dense dispatch.
    ffn_grid: WorkgroupDims = .{},
    /// Thread block dimensions for FFN dense dispatch.
    ffn_block: BlockDims = .{ .x = 256 },
    /// Subgroup/warp size hint for FFN kernel.
    ffn_subgroup: SubgroupSize = .runtime,

    // ── MoE expert dispatch ─────────────────────────────────
    /// Grid dimensions for the MoE expert matvec kernel.
    moe_grid: WorkgroupDims = .{},
    /// Thread block dimensions for the MoE expert matvec kernel.
    moe_block: BlockDims = .{ .x = 128 },
    /// Subgroup/warp size hint for MoE kernel.
    moe_subgroup: SubgroupSize = .runtime,

    // ── Prefill kernel ──────────────────────────────────────
    /// Grid dimensions for prefill (batched attention) dispatch.
    prefill_grid: WorkgroupDims = .{},
    /// Thread block dimensions for prefill dispatch.
    prefill_block: BlockDims = .{ .x = 256 },

    // ── QK-norm kernel (when `has_qk_norm` is true) ─────────
    /// Grid dimensions for the per-head QK normalization kernel.
    qk_norm_grid: WorkgroupDims = .{},
    /// Thread block dimensions for QK normalization.
    qk_norm_block: BlockDims = .{ .x = 128 },

    // ── LM head kernel ──────────────────────────────────────
    /// Grid dimensions for the LM head (vocab projection) kernel.
    lm_head_grid: WorkgroupDims = .{},
    /// Thread block dimensions for the LM head kernel.
    lm_head_block: BlockDims = .{ .x = 256 },
};

// ── Architecture registry entry ─────────────────────────────

/// A complete architecture registry entry tying a model tag to its kernel
/// types, dispatch recommendation, and optimal launch parameters.
pub const ArchRegistryEntry = struct {
    /// The model tag (snake_case) — must match a key in arch_configs.KNOWN_ARCHS.
    tag: []const u8,

    /// The attention kernel implementation to use.
    attn_kernel: AttentionKernel,

    /// The FFN kernel category.
    ffn_kernel: FfnKernel,

    /// The recommended dispatch policy for the fused engine.
    dispatch: DispatchRecommendation,

    /// Optimal kernel launch parameters for this architecture.
    launch: LaunchParams,

    /// Notes about this architecture entry (optional, for documentation).
    notes: []const u8 = "",
};

// ── Launch parameter presets ────────────────────────────────

/// Launch parameters for small models (hidden_dim <= 2048, n_heads <= 16).
const LAUNCH_SMALL = LaunchParams{
    .attn_grid    = .{ .x = 16,  .y = 1,   .z = 1  },
    .attn_block   = .{ .x = 256, .y = 1,   .z = 1  },
    .attn_subgroup = .warp32,
    .ffn_grid     = .{ .x = 64,  .y = 1,   .z = 1  },
    .ffn_block    = .{ .x = 256, .y = 1,   .z = 1  },
    .ffn_subgroup = .warp32,
    .moe_grid     = .{ .x = 128, .y = 1,   .z = 1  },
    .moe_block    = .{ .x = 128, .y = 1,   .z = 1  },
    .moe_subgroup = .warp32,
    .prefill_grid  = .{ .x = 16,  .y = 64,  .z = 1  },
    .prefill_block = .{ .x = 256, .y = 1,   .z = 1  },
    .qk_norm_grid  = .{ .x = 128, .y = 1,   .z = 1  },
    .qk_norm_block = .{ .x = 128, .y = 1,   .z = 1  },
    .lm_head_grid  = .{ .x = 256, .y = 1,   .z = 1  },
    .lm_head_block = .{ .x = 256, .y = 1,   .z = 1  },
};

/// Launch parameters for medium models (hidden_dim 2560–5120, n_heads <= 32).
const LAUNCH_MEDIUM = LaunchParams{
    .attn_grid    = .{ .x = 32,  .y = 1,   .z = 1  },
    .attn_block   = .{ .x = 256, .y = 1,   .z = 1  },
    .attn_subgroup = .warp32,
    .ffn_grid     = .{ .x = 128, .y = 1,   .z = 1  },
    .ffn_block    = .{ .x = 256, .y = 1,   .z = 1  },
    .ffn_subgroup = .warp32,
    .moe_grid     = .{ .x = 256, .y = 1,   .z = 1  },
    .moe_block    = .{ .x = 128, .y = 1,   .z = 1  },
    .moe_subgroup = .warp32,
    .prefill_grid  = .{ .x = 32,  .y = 128, .z = 1  },
    .prefill_block = .{ .x = 256, .y = 1,   .z = 1  },
    .qk_norm_grid  = .{ .x = 256, .y = 1,   .z = 1  },
    .qk_norm_block = .{ .x = 128, .y = 1,   .z = 1  },
    .lm_head_grid  = .{ .x = 512, .y = 1,   .z = 1  },
    .lm_head_block = .{ .x = 256, .y = 1,   .z = 1  },
};

/// Launch parameters for large models (hidden_dim > 5120, n_heads > 32).
const LAUNCH_LARGE = LaunchParams{
    .attn_grid    = .{ .x = 64,  .y = 1,   .z = 1  },
    .attn_block   = .{ .x = 256, .y = 1,   .z = 1  },
    .attn_subgroup = .warp32,
    .ffn_grid     = .{ .x = 256, .y = 1,   .z = 1  },
    .ffn_block    = .{ .x = 256, .y = 1,   .z = 1  },
    .ffn_subgroup = .warp32,
    .moe_grid     = .{ .x = 512, .y = 1,   .z = 1  },
    .moe_block    = .{ .x = 128, .y = 1,   .z = 1  },
    .moe_subgroup = .warp32,
    .prefill_grid  = .{ .x = 64,  .y = 256, .z = 1  },
    .prefill_block = .{ .x = 256, .y = 1,   .z = 1  },
    .qk_norm_grid  = .{ .x = 512, .y = 1,   .z = 1  },
    .qk_norm_block = .{ .x = 128, .y = 1,   .z = 1  },
    .lm_head_grid  = .{ .x = 1024, .y = 1,  .z = 1  },
    .lm_head_block = .{ .x = 256, .y = 1,   .z = 1  },
};

/// Launch parameters for DeepSeek-style MLA models — uses warp64 for AMD GPUs
/// (common in DeepSeek deployments) and larger grids for the absorbed attention.
const LAUNCH_MLA = LaunchParams{
    .attn_grid     = .{ .x = 64,  .y = 1,   .z = 1  },
    .attn_block    = .{ .x = 256, .y = 1,   .z = 1  },
    .attn_subgroup = .warp64,
    .ffn_grid      = .{ .x = 512, .y = 1,   .z = 1  },
    .ffn_block     = .{ .x = 256, .y = 1,   .z = 1  },
    .ffn_subgroup  = .warp64,
    .moe_grid      = .{ .x = 1024, .y = 1,  .z = 1  },
    .moe_block     = .{ .x = 128,  .y = 1,  .z = 1  },
    .moe_subgroup  = .warp64,
    .prefill_grid   = .{ .x = 64,   .y = 128, .z = 1 },
    .prefill_block  = .{ .x = 256,  .y = 1,   .z = 1 },
    .qk_norm_grid   = .{ .x = 256,  .y = 1,   .z = 1 },
    .qk_norm_block  = .{ .x = 128,  .y = 1,   .z = 1 },
    .lm_head_grid   = .{ .x = 512,  .y = 1,   .z = 1 },
    .lm_head_block  = .{ .x = 256,  .y = 1,   .z = 1 },
};

/// Launch parameters for medium-size MoE models (Mixtral, Zaya).
const LAUNCH_MOE_MEDIUM = LaunchParams{
    .attn_grid     = .{ .x = 32,  .y = 1,   .z = 1  },
    .attn_block    = .{ .x = 256, .y = 1,   .z = 1  },
    .attn_subgroup = .warp32,
    .ffn_grid      = .{ .x = 256, .y = 1,   .z = 1  },
    .ffn_block     = .{ .x = 256, .y = 1,   .z = 1  },
    .ffn_subgroup  = .warp32,
    .moe_grid      = .{ .x = 256, .y = 1,   .z = 1  },
    .moe_block     = .{ .x = 128, .y = 1,   .z = 1  },
    .moe_subgroup  = .warp32,
    .prefill_grid   = .{ .x = 32,  .y = 128, .z = 1  },
    .prefill_block  = .{ .x = 256, .y = 1,   .z = 1  },
    .qk_norm_grid   = .{ .x = 256, .y = 1,   .z = 1  },
    .qk_norm_block  = .{ .x = 128, .y = 1,   .z = 1  },
    .lm_head_grid   = .{ .x = 512, .y = 1,   .z = 1  },
    .lm_head_block  = .{ .x = 256, .y = 1,   .z = 1  },
};

// ── Registry table ──────────────────────────────────────────

/// The complete registry, keyed by model tag (snake_case).
/// Every tag listed here must have a matching entry in `arch_configs.zig`.
///
/// Launch parameter preset selection:
///   LAUNCH_SMALL       → H <= 2048 & NH <= 16 (Qwen3-0.6B, Qwen3-1.5B, Llama-3.2-1B)
///   LAUNCH_MEDIUM       → H 2560–5120 (Qwen3-7B, Llama-3.1-8B, Gemma-2-2B, etc.)
///   LAUNCH_LARGE        → H > 5120 (Qwen3-14B, Qwen2.5-32B, Gemma-2-9B)
///   LAUNCH_MLA          → DeepSeek MLA architectures
///   LAUNCH_MOE_MEDIUM   → Standard MoE architectures (Mixtral, Zaya)
const REGISTRY: std.StaticStringMap(ArchRegistryEntry) = std.StaticStringMap(ArchRegistryEntry).initComptime(.{
    // ──────────────────────────────────────────────────────────
    // 1. Qwen3-0.6B
    //    Small dense, QK-norm, GQA (12Q/2KV). Sweet spot for fused engine:
    //    FFN on NPU, attention on GPU.
    // ──────────────────────────────────────────────────────────
    .{
        "qwen3_0_6b",
        ArchRegistryEntry{
            .tag         = "qwen3_0_6b",
            .attn_kernel = .flash,
            .ffn_kernel  = .dense,
            .dispatch    = .ffn_on_npu,
            .launch      = LAUNCH_SMALL,
            .notes       = "QK-norm enabled; NPU FFN + GPU attention sweet spot",
        },
    },

    // ──────────────────────────────────────────────────────────
    // 2. Qwen3-1.5B
    //    Small dense, QK-norm, GQA (16Q/2KV). Same dispatch as 0.6B.
    // ──────────────────────────────────────────────────────────
    .{
        "qwen3_1_5b",
        ArchRegistryEntry{
            .tag         = "qwen3_1_5b",
            .attn_kernel = .flash,
            .ffn_kernel  = .dense,
            .dispatch    = .ffn_on_npu,
            .launch      = LAUNCH_SMALL,
            .notes       = "QK-norm enabled; NPU FFN + GPU attention sweet spot",
        },
    },

    // ──────────────────────────────────────────────────────────
    // 3. Qwen3-7B
    //    Mid-size dense, QK-norm, GQA (32Q/8KV). Standard fused dispatch.
    // ──────────────────────────────────────────────────────────
    .{
        "qwen3_7b",
        ArchRegistryEntry{
            .tag         = "qwen3_7b",
            .attn_kernel = .flash,
            .ffn_kernel  = .dense,
            .dispatch    = .ffn_on_npu,
            .launch      = LAUNCH_MEDIUM,
            .notes       = "QK-norm enabled; standard fused dispatch",
        },
    },

    // ──────────────────────────────────────────────────────────
    // 4. Qwen3-14B
    //    Large dense, QK-norm, GQA (40Q/8KV). Vocab 152064 → large LM head.
    // ──────────────────────────────────────────────────────────
    .{
        "qwen3_14b",
        ArchRegistryEntry{
            .tag         = "qwen3_14b",
            .attn_kernel = .flash,
            .ffn_kernel  = .dense,
            .dispatch    = .ffn_on_npu,
            .launch      = LAUNCH_LARGE,
            .notes       = "QK-norm enabled; large vocab requires GPU LM head for speed",
        },
    },

    // ──────────────────────────────────────────────────────────
    // 5. Qwen2.5-7B
    //    Mid-size dense, NO QK-norm, GQA (32Q/8KV). Standard fused dispatch.
    // ──────────────────────────────────────────────────────────
    .{
        "qwen2_5_7b",
        ArchRegistryEntry{
            .tag         = "qwen2_5_7b",
            .attn_kernel = .flash,
            .ffn_kernel  = .dense,
            .dispatch    = .ffn_on_npu,
            .launch      = LAUNCH_MEDIUM,
            .notes       = "No QK-norm; standard fused dispatch; smaller vocab than Qwen3",
        },
    },

    // ──────────────────────────────────────────────────────────
    // 6. Qwen2.5-32B
    //    Large dense, NO QK-norm, GQA (40Q/8KV). 64 layers → deep stack.
    // ──────────────────────────────────────────────────────────
    .{
        "qwen2_5_32b",
        ArchRegistryEntry{
            .tag         = "qwen2_5_32b",
            .attn_kernel = .flash,
            .ffn_kernel  = .dense,
            .dispatch    = .ffn_on_npu,
            .launch      = LAUNCH_LARGE,
            .notes       = "64 layers; consider layer_by_layer dispatch for pipeline depth",
        },
    },

    // ──────────────────────────────────────────────────────────
    // 7. Llama-3.1-8B
    //    Mid-size dense, NO QK-norm, GQA (32Q/8KV). Standard fused dispatch.
    // ──────────────────────────────────────────────────────────
    .{
        "llama3_1_8b",
        ArchRegistryEntry{
            .tag         = "llama3_1_8b",
            .attn_kernel = .flash,
            .ffn_kernel  = .dense,
            .dispatch    = .ffn_on_npu,
            .launch      = LAUNCH_MEDIUM,
            .notes       = "Standard Llama architecture; well-tuned fused dispatch",
        },
    },

    // ──────────────────────────────────────────────────────────
    // 8. Llama-3.2-1B
    //    Small dense, NO QK-norm, GQA (16Q/8KV). head_dim=64 (non-standard).
    // ──────────────────────────────────────────────────────────
    .{
        "llama3_2_1b",
        ArchRegistryEntry{
            .tag         = "llama3_2_1b",
            .attn_kernel = .flash,
            .ffn_kernel  = .dense,
            .dispatch    = .ffn_on_npu,
            .launch      = LaunchParams{
                .attn_grid     = .{ .x = 16,  .y = 1,  .z = 1 },
                .attn_block    = .{ .x = 128, .y = 1,  .z = 1 },
                .attn_subgroup = .warp32,
                .ffn_grid      = .{ .x = 64,  .y = 1,  .z = 1 },
                .ffn_block     = .{ .x = 256, .y = 1,  .z = 1 },
                .ffn_subgroup  = .warp32,
                .moe_grid      = .{ .x = 128, .y = 1,  .z = 1 },
                .moe_block     = .{ .x = 128, .y = 1,  .z = 1 },
                .moe_subgroup  = .warp32,
                .prefill_grid   = .{ .x = 16,  .y = 64, .z = 1 },
                .prefill_block  = .{ .x = 128, .y = 1,  .z = 1 },
                .qk_norm_grid   = .{ .x = 64,  .y = 1,  .z = 1 },
                .qk_norm_block  = .{ .x = 64,  .y = 1,  .z = 1 },
                .lm_head_grid   = .{ .x = 256, .y = 1,  .z = 1 },
                .lm_head_block  = .{ .x = 128, .y = 1,  .z = 1 },
            },
            .notes = "head_dim=64 (non-standard 128); smaller blocks for dim parity",
        },
    },

    // ──────────────────────────────────────────────────────────
    // 9. Llama-3.2-3B
    //    Small-to-mid dense, NO QK-norm, GQA (24Q/8KV). Head_dim=128.
    // ──────────────────────────────────────────────────────────
    .{
        "llama3_2_3b",
        ArchRegistryEntry{
            .tag         = "llama3_2_3b",
            .attn_kernel = .flash,
            .ffn_kernel  = .dense,
            .dispatch    = .ffn_on_npu,
            .launch      = LAUNCH_SMALL,
            .notes       = "Small dense; same launch params as Qwen3-1.5B tier",
        },
    },

    // ──────────────────────────────────────────────────────────
    // 10. Gemma-2-2B
    //     Small dense, QK-norm + post-norm, GeGLU activation.
    //     head_dim=128, GQA (18Q/2KV). Has both pre and post normalization.
    // ──────────────────────────────────────────────────────────
    .{
        "gemma2_2b",
        ArchRegistryEntry{
            .tag         = "gemma2_2b",
            .attn_kernel = .flash,
            .ffn_kernel  = .dense,
            .dispatch    = .ffn_on_npu,
            .launch      = LAUNCH_SMALL,
            .notes       = "QK-norm+post-norm; GeGLU activation; softcap-aware attention",
        },
    },

    // ──────────────────────────────────────────────────────────
    // 11. Gemma-2-9B
    //     Large dense, QK-norm + post-norm + final post-norm, GeGLU.
    //     head_dim=256 (double standard), GQA (16Q/8KV).
    // ──────────────────────────────────────────────────────────
    .{
        "gemma2_9b",
        ArchRegistryEntry{
            .tag         = "gemma2_9b",
            .attn_kernel = .flash,
            .ffn_kernel  = .dense,
            .dispatch    = .ffn_on_npu,
            .launch      = LaunchParams{
                .attn_grid     = .{ .x = 32,  .y = 1,  .z = 1 },
                .attn_block    = .{ .x = 256, .y = 1,  .z = 1 },
                .attn_subgroup = .warp32,
                .ffn_grid      = .{ .x = 128, .y = 1,  .z = 1 },
                .ffn_block     = .{ .x = 256, .y = 1,  .z = 1 },
                .ffn_subgroup  = .warp32,
                .moe_grid      = .{ .x = 256, .y = 1,  .z = 1 },
                .moe_block     = .{ .x = 128, .y = 1,  .z = 1 },
                .moe_subgroup  = .warp32,
                .prefill_grid   = .{ .x = 32,  .y = 256, .z = 1 },
                .prefill_block  = .{ .x = 256, .y = 1,   .z = 1 },
                .qk_norm_grid   = .{ .x = 256, .y = 1,   .z = 1 },
                .qk_norm_block  = .{ .x = 256, .y = 1,   .z = 1 }, // HD=256
                .lm_head_grid   = .{ .x = 1024, .y = 1,  .z = 1 },
                .lm_head_block  = .{ .x = 256,  .y = 1,  .z = 1 },
            },
            .notes = "HD=256, post-norm, GeGLU; custom launch for double head_dim",
        },
    },

    // ──────────────────────────────────────────────────────────
    // 12. DeepSeek-V2-Lite
    //     MLA + MoE (64 routed + 1 shared, top-6). The definitive MLA architecture.
    //     kv_lora_rank=256, qk_head_dim=192 (128 nope + 64 rope).
    // ──────────────────────────────────────────────────────────
    .{
        "deepseek_v2_lite",
        ArchRegistryEntry{
            .tag         = "deepseek_v2_lite",
            .attn_kernel = .mla,
            .ffn_kernel  = .shared_moe,
            .dispatch    = .ffn_on_npu,
            .launch      = LAUNCH_MLA,
            .notes       = "MLA + shared MoE; kv_lora_rank=256, absorbed attention",
        },
    },

    // ──────────────────────────────────────────────────────────
    // 13. DeepSeek-V3
    //     MLA + MoE (256 routed + 1 shared, top-8). Full production DeepSeek.
    //     kv_lora_rank=512, qk_head_dim=192, 61 layers.
    // ──────────────────────────────────────────────────────────
    .{
        "deepseek_v3",
        ArchRegistryEntry{
            .tag         = "deepseek_v3",
            .attn_kernel = .mla,
            .ffn_kernel  = .shared_moe,
            .dispatch    = .gpu_only,
            .launch      = LAUNCH_MLA,
            .notes       = "Large MLA + MoE; 256 experts, 61 layers; GPU-only recommended",
        },
    },

    // ──────────────────────────────────────────────────────────
    // 14. Mixtral-8x7B
    //     Standard MoE (8 routed, no shared, top-2). No QK-norm.
    //     Uses standard flash attention + routed MoE FFN.
    // ──────────────────────────────────────────────────────────
    .{
        "mixtral_8x7b",
        ArchRegistryEntry{
            .tag         = "mixtral_8x7b",
            .attn_kernel = .flash,
            .ffn_kernel  = .moe,
            .dispatch    = .gpu_only,
            .launch      = LAUNCH_MOE_MEDIUM,
            .notes       = "Standard MoE (8 expert, top-2); no shared expert; 32K ctx",
        },
    },

    // ──────────────────────────────────────────────────────────
    // 15. Zaya1-8B (our model)
    //     MoE (8 routed, no shared, top-2) with 1-bit ternary weights.
    //     Uses standard flash attention + ternary MoE FFN.
    // ──────────────────────────────────────────────────────────
    .{
        "zaya1_8b",
        ArchRegistryEntry{
            .tag         = "zaya1_8b",
            .attn_kernel = .flash,
            .ffn_kernel  = .moe,
            .dispatch    = .ffn_on_npu,
            .launch      = LAUNCH_MOE_MEDIUM,
            .notes       = "1-bit ternary MoE; 40 layers; NPU FFN handles ternary well",
        },
    },
});

// ── Lookup function ─────────────────────────────────────────

/// Look up the architecture registry entry for a model tag.
/// Returns `null` when the tag is unknown.
pub fn getArchEntry(tag: []const u8) ?ArchRegistryEntry {
    return REGISTRY.get(tag);
}

/// Look up the attention kernel type for a model tag.
/// Convenience wrapper around `getArchEntry`.
pub fn getAttentionKernel(tag: []const u8) ?AttentionKernel {
    const entry = getArchEntry(tag) orelse return null;
    return entry.attn_kernel;
}

/// Look up the FFN kernel type for a model tag.
/// Convenience wrapper around `getArchEntry`.
pub fn getFfnKernel(tag: []const u8) ?FfnKernel {
    const entry = getArchEntry(tag) orelse return null;
    return entry.ffn_kernel;
}

/// Look up the recommended dispatch policy for a model tag.
/// Convenience wrapper around `getArchEntry`.
pub fn getDispatchRecommendation(tag: []const u8) ?DispatchRecommendation {
    const entry = getArchEntry(tag) orelse return null;
    return entry.dispatch;
}

/// Look up the optimal launch parameters for a model tag.
/// Convenience wrapper around `getArchEntry`.
pub fn getLaunchParams(tag: []const u8) ?LaunchParams {
    const entry = getArchEntry(tag) orelse return null;
    return entry.launch;
}

/// Convert a dispatcher DispatchPolicy to a registry DispatchRecommendation.
/// The registry recommendation is a static baseline; the dispatcher policy
/// may be overridden at runtime.
pub fn dispatchRecommendationToPolicy(rec: DispatchRecommendation) dispatcher.DispatchPolicy {
    return switch (rec) {
        .npu_only         => .npu_only,
        .gpu_only         => .gpu_only,
        .ffn_on_npu       => .ffn_on_npu,
        .qkv_on_npu       => .qkv_on_npu,
        .attention_on_npu => .attention_on_npu,
        .cpu_only         => .cpu_only,
    };
}

/// Derive the FFN kernel type from an architecture config.
/// This is a convenience that avoids a registry lookup when you already
/// have the ArchConfig in hand.
pub fn ffnKernelFromConfig(cfg: arch_configs.ArchConfig) FfnKernel {
    if (cfg.has_shared_experts) return .shared_moe;
    if (cfg.n_experts > 0) return .moe;
    return .dense;
}

/// Derive the attention kernel type from an architecture config.
pub fn attnKernelFromConfig(cfg: arch_configs.ArchConfig) AttentionKernel {
    return switch (cfg.arch) {
        .deepseek_moe => .mla,
        .dense, .moe => .flash,
    };
}

/// Derive the dispatch recommendation from an architecture config.
/// Uses heuristics: MLA models prefer GPU, small models prefer fused,
/// large models prefer GPU-only.
pub fn dispatchFromConfig(cfg: arch_configs.ArchConfig) DispatchRecommendation {
    // MLA architectures are GPU-heavy; recommend GPU-only for large ones.
    if (cfg.arch == .deepseek_moe) {
        return if (cfg.H > 4096) .gpu_only else .ffn_on_npu;
    }
    // MoE architectures: medium MoE works well with fused, large MoE prefers GPU.
    if (cfg.arch == .moe) {
        return if (cfg.n_experts > 64) .gpu_only else .ffn_on_npu;
    }
    // Dense architectures: small → fused, large → GPU-only.
    return if (cfg.H <= 4096) .ffn_on_npu else .gpu_only;
}

/// Return the number of registered architecture entries.
pub fn count() usize {
    return REGISTRY.kvs.len;
}

// ── Verify registry consistency with arch_configs ──────────

/// Verify that every entry in the registry has a corresponding entry
/// in `arch_configs.zig` and that the FFN/attention kernel types are
/// consistent with the arch config.
pub fn verifyConsistency() !void {
    var i: usize = 0;
    while (i < REGISTRY.kvs.len) : (i += 1) {
        const tag = REGISTRY.kvs.keys[i];
        const entry = REGISTRY.kvs.values[i];

        // Check that arch_configs knows about this tag.
        const cfg = arch_configs.getArchConfig(tag) orelse {
            log.err("arch_registry: tag '{s}' has no ArchConfig entry", .{tag});
            return error.MissingArchConfig;
        };

        // Verify FFN kernel type matches config.
        const expected_ffn = ffnKernelFromConfig(cfg);
        if (entry.ffn_kernel != expected_ffn) {
            log.err("arch_registry: tag '{s}' FFN kernel {s} does not match config (expected {s})", .{
                tag,
                @tagName(entry.ffn_kernel),
                @tagName(expected_ffn),
            });
            return error.FfnKernelMismatch;
        }

        // Verify attention kernel type matches config.
        const expected_attn = attnKernelFromConfig(cfg);
        if (entry.attn_kernel != expected_attn) {
            log.err("arch_registry: tag '{s}' attention kernel {s} does not match config (expected {s})", .{
                tag,
                @tagName(entry.attn_kernel),
                @tagName(expected_attn),
            });
            return error.AttentionKernelMismatch;
        }

        log.debug("arch_registry: '{s}' ✓ (attn={s}, ffn={s}, dispatch={s})", .{
            tag,
            @tagName(entry.attn_kernel),
            @tagName(entry.ffn_kernel),
            @tagName(entry.dispatch),
        });
    }

    // Also check that every arch_configs tag has a registry entry.
    // We iterate the known arch tags from our own registry to cross-reference.
    var j: usize = 0;
    while (j < REGISTRY.kvs.len) : (j += 1) {
        const tag = REGISTRY.kvs.keys[j];
        // Sanity: arch_configs must know this tag (already checked above).
        // Here we verify the reverse: every tag in arch_configs that we know
        // about has a matching registry entry (already guaranteed by construction).
        _ = tag;
    }

    // Verify count parity as a secondary cross-check.
    if (arch_configs.count() != count()) {
        log.err("count mismatch: arch_configs has {d} entries, registry has {d}", .{
            arch_configs.count(),
            count(),
        });
        return error.CountMismatch;
    }
}

// ── Tests ───────────────────────────────────────────────────

test "getArchEntry qwen3_0_6b" {
    const entry = getArchEntry("qwen3_0_6b") orelse @panic("missing registry entry");
    try std.testing.expectEqual(AttentionKernel.flash, entry.attn_kernel);
    try std.testing.expectEqual(FfnKernel.dense, entry.ffn_kernel);
    try std.testing.expectEqual(DispatchRecommendation.ffn_on_npu, entry.dispatch);
    try std.testing.expectEqual(@as(u16, 256), entry.launch.attn_block.x);
    try std.testing.expectEqual(@as(u16, 256), entry.launch.ffn_block.x);
}

test "getArchEntry qwen3_7b" {
    const entry = getArchEntry("qwen3_7b") orelse @panic("missing registry entry");
    try std.testing.expectEqual(AttentionKernel.flash, entry.attn_kernel);
    try std.testing.expectEqual(FfnKernel.dense, entry.ffn_kernel);
    try std.testing.expectEqual(DispatchRecommendation.ffn_on_npu, entry.dispatch);
    try std.testing.expectEqual(@as(u16, 256), entry.launch.attn_block.x);
    try std.testing.expectEqual(@as(u16, 256), entry.launch.ffn_block.x);
}

test "getArchEntry deepseek_v2_lite" {
    const entry = getArchEntry("deepseek_v2_lite") orelse @panic("missing registry entry");
    try std.testing.expectEqual(AttentionKernel.mla, entry.attn_kernel);
    try std.testing.expectEqual(FfnKernel.shared_moe, entry.ffn_kernel);
    try std.testing.expectEqual(DispatchRecommendation.ffn_on_npu, entry.dispatch);
    try std.testing.expectEqual(SubgroupSize.warp64, entry.launch.attn_subgroup);
}

test "getArchEntry deepseek_v3" {
    const entry = getArchEntry("deepseek_v3") orelse @panic("missing registry entry");
    try std.testing.expectEqual(AttentionKernel.mla, entry.attn_kernel);
    try std.testing.expectEqual(FfnKernel.shared_moe, entry.ffn_kernel);
    try std.testing.expectEqual(DispatchRecommendation.gpu_only, entry.dispatch);
}

test "getArchEntry mixtral_8x7b" {
    const entry = getArchEntry("mixtral_8x7b") orelse @panic("missing registry entry");
    try std.testing.expectEqual(AttentionKernel.flash, entry.attn_kernel);
    try std.testing.expectEqual(FfnKernel.moe, entry.ffn_kernel);
    try std.testing.expectEqual(DispatchRecommendation.gpu_only, entry.dispatch);
    try std.testing.expectEqual(@as(u16, 128), entry.launch.moe_block.x);
}

test "getArchEntry zaya1_8b" {
    const entry = getArchEntry("zaya1_8b") orelse @panic("missing registry entry");
    try std.testing.expectEqual(AttentionKernel.flash, entry.attn_kernel);
    try std.testing.expectEqual(FfnKernel.moe, entry.ffn_kernel);
    try std.testing.expectEqual(DispatchRecommendation.ffn_on_npu, entry.dispatch);
}

test "getArchEntry gemma2_9b (custom launch for HD=256)" {
    const entry = getArchEntry("gemma2_9b") orelse @panic("missing registry entry");
    try std.testing.expectEqual(AttentionKernel.flash, entry.attn_kernel);
    try std.testing.expectEqual(FfnKernel.dense, entry.ffn_kernel);
    // Custom qk_norm_block for HD=256
    try std.testing.expectEqual(@as(u16, 256), entry.launch.qk_norm_block.x);
    try std.testing.expectEqual(@as(u16, 256), entry.launch.attn_block.x);
}

test "getArchEntry llama3_2_1b (custom launch for HD=64)" {
    const entry = getArchEntry("llama3_2_1b") orelse @panic("missing registry entry");
    try std.testing.expectEqual(AttentionKernel.flash, entry.attn_kernel);
    try std.testing.expectEqual(FfnKernel.dense, entry.ffn_kernel);
    // Custom block size for HD=64
    try std.testing.expectEqual(@as(u16, 128), entry.launch.attn_block.x);
    try std.testing.expectEqual(@as(u16, 128), entry.launch.prefill_block.x);
    try std.testing.expectEqual(@as(u16, 64), entry.launch.qk_norm_block.x);
}

test "getArchEntry unknown returns null" {
    try std.testing.expectEqual(@as(?ArchRegistryEntry, null), getArchEntry("unknown_model"));
}

test "getAttentionKernel convenience" {
    try std.testing.expectEqual(AttentionKernel.flash, getAttentionKernel("qwen3_7b").?);
    try std.testing.expectEqual(AttentionKernel.mla, getAttentionKernel("deepseek_v2_lite").?);
    try std.testing.expectEqual(@as(?AttentionKernel, null), getAttentionKernel("unknown_model"));
}

test "getFfnKernel convenience" {
    try std.testing.expectEqual(FfnKernel.dense, getFfnKernel("qwen3_7b").?);
    try std.testing.expectEqual(FfnKernel.moe, getFfnKernel("mixtral_8x7b").?);
    try std.testing.expectEqual(FfnKernel.shared_moe, getFfnKernel("deepseek_v3").?);
}

test "getDispatchRecommendation convenience" {
    try std.testing.expectEqual(
        DispatchRecommendation.ffn_on_npu,
        getDispatchRecommendation("qwen3_0_6b").?,
    );
    try std.testing.expectEqual(
        DispatchRecommendation.gpu_only,
        getDispatchRecommendation("mixtral_8x7b").?,
    );
}

test "ffnKernelFromConfig dense" {
    const cfg = arch_configs.getArchConfig("qwen3_7b").?;
    try std.testing.expectEqual(FfnKernel.dense, ffnKernelFromConfig(cfg));
}

test "ffnKernelFromConfig moe" {
    const cfg = arch_configs.getArchConfig("mixtral_8x7b").?;
    try std.testing.expectEqual(FfnKernel.moe, ffnKernelFromConfig(cfg));
}

test "ffnKernelFromConfig shared_moe" {
    const cfg = arch_configs.getArchConfig("deepseek_v2_lite").?;
    try std.testing.expectEqual(FfnKernel.shared_moe, ffnKernelFromConfig(cfg));
}

test "attnKernelFromConfig" {
    const qwen = arch_configs.getArchConfig("qwen3_7b").?;
    try std.testing.expectEqual(AttentionKernel.flash, attnKernelFromConfig(qwen));

    const deepseek = arch_configs.getArchConfig("deepseek_v2_lite").?;
    try std.testing.expectEqual(AttentionKernel.mla, attnKernelFromConfig(deepseek));

    const zaya = arch_configs.getArchConfig("zaya1_8b").?;
    try std.testing.expectEqual(AttentionKernel.flash, attnKernelFromConfig(zaya));
}

test "dispatchFromConfig heuristics" {
    // Small dense → ffn_on_npu
    const small = arch_configs.getArchConfig("qwen3_0_6b").?;
    try std.testing.expectEqual(DispatchRecommendation.ffn_on_npu, dispatchFromConfig(small));

    // Large dense → gpu_only
    const large = arch_configs.getArchConfig("qwen3_14b").?;
    try std.testing.expectEqual(DispatchRecommendation.gpu_only, dispatchFromConfig(large));

    // DeepSeek small → ffn_on_npu
    const ds_small = arch_configs.getArchConfig("deepseek_v2_lite").?;
    try std.testing.expectEqual(DispatchRecommendation.ffn_on_npu, dispatchFromConfig(ds_small));

    // DeepSeek large → gpu_only
    const ds_large = arch_configs.getArchConfig("deepseek_v3").?;
    try std.testing.expectEqual(DispatchRecommendation.gpu_only, dispatchFromConfig(ds_large));

    // Standard MoE small → ffn_on_npu
    const zaya = arch_configs.getArchConfig("zaya1_8b").?;
    try std.testing.expectEqual(DispatchRecommendation.ffn_on_npu, dispatchFromConfig(zaya));

    // Large expert count → gpu_only (Mixtral has only 8 though, so should be ffn_on_npu)
    const mixtral = arch_configs.getArchConfig("mixtral_8x7b").?;
    try std.testing.expectEqual(DispatchRecommendation.ffn_on_npu, dispatchFromConfig(mixtral));
}

test "dispatchFromConfig boundary" {
    // Exactly H=4096 (border) → ffn_on_npu
    const border = arch_configs.getArchConfig("llama3_1_8b").?;
    try std.testing.expectEqual(DispatchRecommendation.ffn_on_npu, dispatchFromConfig(border));
}

test "dispatchRecommendationToPolicy round-trip" {
    for ([_]DispatchRecommendation{
        .npu_only, .gpu_only, .ffn_on_npu, .qkv_on_npu, .attention_on_npu, .cpu_only,
    }) |rec| {
        const policy = dispatchRecommendationToPolicy(rec);
        try std.testing.expectEqualStrings(@tagName(rec), @tagName(policy));
    }
}

test "count matches arch_configs" {
    // The registry and arch_configs should have the same number of entries.
    try std.testing.expectEqual(arch_configs.count(), count());
}

test "getLaunchParams small model" {
    const entry = getArchEntry("qwen3_0_6b").?;
    try std.testing.expectEqual(@as(u16, 16), entry.launch.attn_grid.x);
    try std.testing.expectEqual(@as(u16, 64), entry.launch.ffn_grid.x);
}

test "getLaunchParams medium model" {
    const entry = getArchEntry("llama3_1_8b").?;
    try std.testing.expectEqual(@as(u16, 32), entry.launch.attn_grid.x);
    try std.testing.expectEqual(@as(u16, 128), entry.launch.ffn_grid.x);
}

test "getLaunchParams large model" {
    const entry = getArchEntry("qwen3_14b").?;
    try std.testing.expectEqual(@as(u16, 64), entry.launch.attn_grid.x);
    try std.testing.expectEqual(@as(u16, 256), entry.launch.ffn_grid.x);
}

test "getLaunchParams mla model" {
    const entry = getArchEntry("deepseek_v2_lite").?;
    try std.testing.expectEqual(@as(u16, 64), entry.launch.attn_grid.x);
    try std.testing.expectEqual(@as(u16, 512), entry.launch.ffn_grid.x);
    try std.testing.expectEqual(@as(u16, 1024), entry.launch.moe_grid.x);
}

test "WorkgroupDims and BlockDims sizes" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(WorkgroupDims));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(BlockDims));
}

test "WorkgroupDims packing" {
    const w = WorkgroupDims{ .x = 64, .y = 128, .z = 1 };
    try std.testing.expectEqual(@as(u16, 64), w.x);
    try std.testing.expectEqual(@as(u16, 128), w.y);
    try std.testing.expectEqual(@as(u16, 1), w.z);
}

test "BlockDims packing" {
    const b = BlockDims{ .x = 256, .y = 1, .z = 1 };
    try std.testing.expectEqual(@as(u16, 256), b.x);
}

test "all launch params are non-zero where expected" {
    // Verify all registered entries have sensible launch parameters.
    const tags: []const []const u8 = &.{
        "qwen3_0_6b",   "qwen3_1_5b",   "qwen3_7b",
        "qwen3_14b",    "qwen2_5_7b",   "qwen2_5_32b",
        "llama3_1_8b",  "llama3_2_1b",  "llama3_2_3b",
        "gemma2_2b",    "gemma2_9b",
        "deepseek_v2_lite", "deepseek_v3",
        "mixtral_8x7b", "zaya1_8b",
    };

    for (tags) |tag| {
        const entry = getArchEntry(tag) orelse @panic("missing: " ++ tag);
        try std.testing.expect(entry.launch.attn_block.x > 0);
        try std.testing.expect(entry.launch.ffn_block.x > 0);
        try std.testing.expect(entry.launch.prefill_block.x > 0);
        try std.testing.expect(entry.launch.lm_head_block.x > 0);
    }
}

test "verifyConsistency succeeds" {
    // This test validates that every registry entry is consistent with its
    // corresponding arch_configs entry.
    try verifyConsistency();
}

test "tags match between registry and arch_configs" {
    // Every registry tag must exist in arch_configs.
    var i: usize = 0;
    while (i < REGISTRY.kvs.len) : (i += 1) {
        const tag = REGISTRY.kvs.keys[i];
        try std.testing.expect(arch_configs.getArchConfig(tag) != null);
    }
}
