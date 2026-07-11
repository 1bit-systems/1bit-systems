//! Architecture configuration table for 1bit fused inference engine.
//!
//! Each entry describes a known model architecture with its full dimensions,
//! activation, normalization, position encoding, and MoE configuration.
//!
//! Add new architectures here as the codebase grows — the StaticStringMap
//! lookup provides fast comptime dispatch by model tag.
//!
//! @section Fused Engine

const std = @import("std");
const log = std.log.scoped(.arch_configs);

// ── Architecture type ─────────────────────────────────────────

/// High-level architecture family.
pub const ArchType = enum(u8) {
    /// Standard dense (every token visits every neuron).
    dense = 0,
    /// Mixture of Experts (routed experts, optional shared expert).
    moe = 1,
    /// DeepSeek-style MLA + MoE (Multi-Head Latent Attention + routed/shared experts).
    deepseek_moe = 2,
};

// ── Activation function ──────────────────────────────────────

/// Activation function used in the FFN sub-layer.
pub const Activation = enum(u8) {
    silu = 0,
    gelu = 1, // includes GeGLU variants
    relu = 2,
};

// ── Normalization type ───────────────────────────────────────

/// Normalization applied before (pre) and optionally after (post) sub-layers.
pub const NormType = enum(u8) {
    rmsnorm = 0,
    layernorm = 1,
};

// ── Position encoding ────────────────────────────────────────

/// Position encoding mechanism used in self-attention.
pub const PosEncoding = enum(u8) {
    rope = 0,
    alibi = 1,
    none = 2,
};

// ── Architecture configuration ───────────────────────────────

/// Full architecture specification beyond the base ModelConfig dimensions.
///
/// Field naming matches the fused engine convention:
///   H  = hidden_dim
///   NC = n_layers
///   NH = n_heads
///   NKV = n_kv_heads
///   HD = head_dim
///   IM = intermediate_size (FFN hidden)
///   NV = vocab_size
pub const ArchConfig = struct {
    // ── Base dimensions (mirrors ModelConfig fields) ──
    H: u32,
    NC: u32,
    NH: u32,
    NKV: u32,
    HD: u32,
    IM: u32,
    NV: u32,
    max_seq_len: u32,

    // ── Architecture type ──
    arch: ArchType,

    // ── Activation function ──
    activation: Activation,

    // ── Normalization type ──
    norm: NormType,

    // ── Position encoding ──
    pos_encoding: PosEncoding,

    // ── QK-normalization ──
    /// True when the model applies per-head normalization to Q and K
    /// before RoPE (Qwen3-style QK-Norm, Gemma-2 QK-Norm with softcap).
    has_qk_norm: bool,

    // ── Mixture of Experts ──
    /// True when the model has a shared (always-activated) expert in
    /// addition to the routed experts (DeepSeek-V2/V3 style).
    has_shared_experts: bool,

    /// Total number of routed experts in the MoE layer.
    /// Set to 0 for dense architectures.
    n_experts: u32,

    /// Number of top-K experts selected per token via router.
    /// Set to 0 for dense architectures.
    top_k: u32,

    /// Intermediate size of each routed expert.
    /// For dense models: 0 (use IM).
    /// For MoE models where all experts share the same FFN size as a
    /// dense MLP would: equals IM.
    /// For DeepSeek-style models with a shared expert and narrower
    /// routed experts: may differ from IM.
    expert_inter_size: u32,
};

// ── Known architecture configurations ────────────────────────

/// All supported model architectures, keyed by model tag (snake_case).
/// Add new entries here when adding support for a new model family.
const KNOWN_ARCHS = std.StaticStringMap(ArchConfig).initComptime(.{
    // ──────────────────────────────────────────────────────
    // 1. Qwen3-0.6B
    // ──────────────────────────────────────────────────────
    .{
        "qwen3_0_6b",
        ArchConfig{
            .H = 1536,
            .NC = 28,
            .NH = 12,
            .NKV = 2,
            .HD = 128,
            .IM = 4096,
            .NV = 151936,
            .max_seq_len = 4096,
            .arch = .dense,
            .activation = .silu,
            .norm = .rmsnorm,
            .pos_encoding = .rope,
            .has_qk_norm = true,
            .has_shared_experts = false,
            .n_experts = 0,
            .top_k = 0,
            .expert_inter_size = 0,
        },
    },

    // ──────────────────────────────────────────────────────
    // 2. Qwen3-1.5B
    // ──────────────────────────────────────────────────────
    .{
        "qwen3_1_5b",
        ArchConfig{
            .H = 2048,
            .NC = 28,
            .NH = 16,
            .NKV = 2,
            .HD = 128,
            .IM = 8192,
            .NV = 151936,
            .max_seq_len = 4096,
            .arch = .dense,
            .activation = .silu,
            .norm = .rmsnorm,
            .pos_encoding = .rope,
            .has_qk_norm = true,
            .has_shared_experts = false,
            .n_experts = 0,
            .top_k = 0,
            .expert_inter_size = 0,
        },
    },

    // ──────────────────────────────────────────────────────
    // 3. Qwen3-7B
    // ──────────────────────────────────────────────────────
    .{
        "qwen3_7b",
        ArchConfig{
            .H = 4096,
            .NC = 32,
            .NH = 32,
            .NKV = 8,
            .HD = 128,
            .IM = 16384,
            .NV = 151936,
            .max_seq_len = 8192,
            .arch = .dense,
            .activation = .silu,
            .norm = .rmsnorm,
            .pos_encoding = .rope,
            .has_qk_norm = true,
            .has_shared_experts = false,
            .n_experts = 0,
            .top_k = 0,
            .expert_inter_size = 0,
        },
    },

    // ──────────────────────────────────────────────────────
    // 4. Qwen3-14B
    // ──────────────────────────────────────────────────────
    .{
        "qwen3_14b",
        ArchConfig{
            .H = 5120,
            .NC = 40,
            .NH = 40,
            .NKV = 8,
            .HD = 128,
            .IM = 20480,
            .NV = 152064,
            .max_seq_len = 8192,
            .arch = .dense,
            .activation = .silu,
            .norm = .rmsnorm,
            .pos_encoding = .rope,
            .has_qk_norm = true,
            .has_shared_experts = false,
            .n_experts = 0,
            .top_k = 0,
            .expert_inter_size = 0,
        },
    },

    // ──────────────────────────────────────────────────────
    // 5. Qwen2.5-7B
    // ──────────────────────────────────────────────────────
    .{
        "qwen2_5_7b",
        ArchConfig{
            .H = 4096,
            .NC = 28,
            .NH = 32,
            .NKV = 8,
            .HD = 128,
            .IM = 11008,
            .NV = 152064,
            .max_seq_len = 4096,
            .arch = .dense,
            .activation = .silu,
            .norm = .rmsnorm,
            .pos_encoding = .rope,
            .has_qk_norm = false,
            .has_shared_experts = false,
            .n_experts = 0,
            .top_k = 0,
            .expert_inter_size = 0,
        },
    },

    // ──────────────────────────────────────────────────────
    // 6. Qwen2.5-32B
    // ──────────────────────────────────────────────────────
    .{
        "qwen2_5_32b",
        ArchConfig{
            .H = 5120,
            .NC = 64,
            .NH = 40,
            .NKV = 8,
            .HD = 128,
            .IM = 20480,
            .NV = 152064,
            .max_seq_len = 8192,
            .arch = .dense,
            .activation = .silu,
            .norm = .rmsnorm,
            .pos_encoding = .rope,
            .has_qk_norm = false,
            .has_shared_experts = false,
            .n_experts = 0,
            .top_k = 0,
            .expert_inter_size = 0,
        },
    },

    // ──────────────────────────────────────────────────────
    // 7. Llama-3.1-8B
    // ──────────────────────────────────────────────────────
    .{
        "llama3_1_8b",
        ArchConfig{
            .H = 4096,
            .NC = 32,
            .NH = 32,
            .NKV = 8,
            .HD = 128,
            .IM = 14336,
            .NV = 128256,
            .max_seq_len = 8192,
            .arch = .dense,
            .activation = .silu,
            .norm = .rmsnorm,
            .pos_encoding = .rope,
            .has_qk_norm = false,
            .has_shared_experts = false,
            .n_experts = 0,
            .top_k = 0,
            .expert_inter_size = 0,
        },
    },

    // ──────────────────────────────────────────────────────
    // 8. Llama-3.2-1B
    // ──────────────────────────────────────────────────────
    .{
        "llama3_2_1b",
        ArchConfig{
            .H = 2048,
            .NC = 16,
            .NH = 16,
            .NKV = 8,
            .HD = 64,
            .IM = 8192,
            .NV = 128256,
            .max_seq_len = 8192,
            .arch = .dense,
            .activation = .silu,
            .norm = .rmsnorm,
            .pos_encoding = .rope,
            .has_qk_norm = false,
            .has_shared_experts = false,
            .n_experts = 0,
            .top_k = 0,
            .expert_inter_size = 0,
        },
    },

    // ──────────────────────────────────────────────────────
    // 9. Llama-3.2-3B
    // ──────────────────────────────────────────────────────
    .{
        "llama3_2_3b",
        ArchConfig{
            .H = 3072,
            .NC = 28,
            .NH = 24,
            .NKV = 8,
            .HD = 128,
            .IM = 8192,
            .NV = 128256,
            .max_seq_len = 8192,
            .arch = .dense,
            .activation = .silu,
            .norm = .rmsnorm,
            .pos_encoding = .rope,
            .has_qk_norm = false,
            .has_shared_experts = false,
            .n_experts = 0,
            .top_k = 0,
            .expert_inter_size = 0,
        },
    },

    // ──────────────────────────────────────────────────────
    // 10. Gemma-2-2B
    // ──────────────────────────────────────────────────────
    .{
        "gemma2_2b",
        ArchConfig{
            .H = 2304,
            .NC = 26,
            .NH = 18,
            .NKV = 2,
            .HD = 128,
            .IM = 9216,
            .NV = 256128,
            .max_seq_len = 8192,
            .arch = .dense,
            .activation = .gelu, // GeGLU gated variant
            .norm = .rmsnorm, // pre-norm + post-norm
            .pos_encoding = .rope,
            .has_qk_norm = true,
            .has_shared_experts = false,
            .n_experts = 0,
            .top_k = 0,
            .expert_inter_size = 0,
        },
    },

    // ──────────────────────────────────────────────────────
    // 11. Gemma-2-9B
    // ──────────────────────────────────────────────────────
    .{
        "gemma2_9b",
        ArchConfig{
            .H = 3584,
            .NC = 42,
            .NH = 16,
            .NKV = 8,
            .HD = 256,
            .IM = 14336,
            .NV = 256128,
            .max_seq_len = 8192,
            .arch = .dense,
            .activation = .gelu, // GeGLU gated variant
            .norm = .rmsnorm, // pre-norm + post-norm + final post-norm
            .pos_encoding = .rope,
            .has_qk_norm = true,
            .has_shared_experts = false,
            .n_experts = 0,
            .top_k = 0,
            .expert_inter_size = 0,
        },
    },

    // ──────────────────────────────────────────────────────
    // 12. DeepSeek-V2-Lite
    //     MLA + MoE (64 routed, 1 shared, top-6)
    // ──────────────────────────────────────────────────────
    .{
        "deepseek_v2_lite",
        ArchConfig{
            .H = 2048,
            .NC = 27,
            .NH = 16,
            .NKV = 2,
            .HD = 128,
            .IM = 1536, // routed expert intermediate size
            .NV = 102400,
            .max_seq_len = 4096,
            .arch = .deepseek_moe, // MLA attention + routed/shared MoE
            .activation = .silu,
            .norm = .rmsnorm,
            .pos_encoding = .rope,
            .has_qk_norm = false, // no explicit QK-norm (MLA handles this)
            .has_shared_experts = true, // 1 shared expert
            .n_experts = 64,
            .top_k = 6,
            .expert_inter_size = 1536, // same as IM for routed experts
        },
    },

    // ──────────────────────────────────────────────────────
    // 13. DeepSeek-V3
    //     MLA + MoE (256 routed, 1 shared, top-8)
    // ──────────────────────────────────────────────────────
    .{
        "deepseek_v3",
        ArchConfig{
            .H = 7168,
            .NC = 61,
            .NH = 56,
            .NKV = 8,
            .HD = 128,
            .IM = 2048, // routed expert intermediate size
            .NV = 129280,
            .max_seq_len = 8192,
            .arch = .deepseek_moe, // MLA attention + routed/shared MoE
            .activation = .silu,
            .norm = .rmsnorm,
            .pos_encoding = .rope,
            .has_qk_norm = false, // no explicit QK-norm (MLA handles this)
            .has_shared_experts = true, // 1 shared expert
            .n_experts = 256,
            .top_k = 8,
            .expert_inter_size = 2048, // same as IM for routed experts
        },
    },

    // ──────────────────────────────────────────────────────
    // 14. Mixtral-8x7B
    //     Standard MoE (8 routed, no shared, top-2)
    // ──────────────────────────────────────────────────────
    .{
        "mixtral_8x7b",
        ArchConfig{
            .H = 4096,
            .NC = 32,
            .NH = 32,
            .NKV = 8,
            .HD = 128,
            .IM = 14336,
            .NV = 32000,
            .max_seq_len = 32768,
            .arch = .moe,
            .activation = .silu,
            .norm = .rmsnorm,
            .pos_encoding = .rope,
            .has_qk_norm = false,
            .has_shared_experts = false,
            .n_experts = 8,
            .top_k = 2,
            .expert_inter_size = 14336, // same as IM
        },
    },

    // ──────────────────────────────────────────────────────
    // 15. Zaya1-8B (our model)
    //     MoE (8 routed, no shared, top-2, 1-bit ternary)
    // ──────────────────────────────────────────────────────
    .{
        "zaya1_8b",
        ArchConfig{
            .H = 4096,
            .NC = 40,
            .NH = 32,
            .NKV = 8,
            .HD = 128,
            .IM = 14336,
            .NV = 151936,
            .max_seq_len = 4096,
            .arch = .moe,
            .activation = .silu,
            .norm = .rmsnorm,
            .pos_encoding = .rope,
            .has_qk_norm = false,
            .has_shared_experts = false,
            .n_experts = 8,
            .top_k = 2,
            .expert_inter_size = 14336, // same as IM
        },
    },
});

// ── Helper functions ─────────────────────────────────────────

/// Look up the architecture configuration for a model tag.
/// Returns null when the tag is unknown.
pub fn getArchConfig(tag: []const u8) ?ArchConfig {
    return KNOWN_ARCHS.get(tag);
}

/// Return the number of known architecture entries.
pub fn count() usize {
    return KNOWN_ARCHS.kvs.len;
}

/// Convenience: return true when the given tag describes an MoE architecture
/// (either standard MoE or DeepSeek-style MLA+MoE).
pub fn isMoe(tag: []const u8) bool {
    const cfg = getArchConfig(tag) orelse return false;
    return cfg.arch != .dense;
}

/// Convenience: return true when the given tag describes a DeepSeek-style
/// MLA + MoE architecture (requires MLA attention kernel).
pub fn isDeepSeekMoe(tag: []const u8) bool {
    const cfg = getArchConfig(tag) orelse return false;
    return cfg.arch == .deepseek_moe;
}

/// Convenience: return true when the model uses QK-normalization
/// (Qwen3-style or Gemma-2-style).
pub fn hasQkNorm(tag: []const u8) bool {
    const cfg = getArchConfig(tag) orelse return false;
    return cfg.has_qk_norm;
}

/// Convenience: return true when the model has a shared expert
/// (DeepSeek-V2/V3-style).
pub fn hasSharedExpert(tag: []const u8) bool {
    const cfg = getArchConfig(tag) orelse return false;
    return cfg.has_shared_experts;
}

// ── Tests ────────────────────────────────────────────────────

test "getArchConfig qwen3_0_6b" {
    const cfg = getArchConfig("qwen3_0_6b") orelse @panic("missing arch config");
    try std.testing.expectEqual(@as(u32, 1536), cfg.H);
    try std.testing.expectEqual(@as(u32, 28), cfg.NC);
    try std.testing.expectEqual(@as(u32, 12), cfg.NH);
    try std.testing.expectEqual(@as(u32, 2), cfg.NKV);
    try std.testing.expectEqual(@as(u32, 128), cfg.HD);
    try std.testing.expectEqual(@as(u32, 4096), cfg.IM);
    try std.testing.expectEqual(@as(u32, 151936), cfg.NV);
    try std.testing.expectEqual(@as(u32, 4096), cfg.max_seq_len);
    try std.testing.expectEqual(ArchType.dense, cfg.arch);
    try std.testing.expectEqual(Activation.silu, cfg.activation);
    try std.testing.expectEqual(NormType.rmsnorm, cfg.norm);
    try std.testing.expectEqual(PosEncoding.rope, cfg.pos_encoding);
    try std.testing.expectEqual(true, cfg.has_qk_norm);
    try std.testing.expectEqual(false, cfg.has_shared_experts);
}

test "getArchConfig deepseek_v2_lite" {
    const cfg = getArchConfig("deepseek_v2_lite") orelse @panic("missing arch config");
    try std.testing.expectEqual(@as(u32, 2048), cfg.H);
    try std.testing.expectEqual(@as(u32, 27), cfg.NC);
    try std.testing.expectEqual(@as(u32, 16), cfg.NH);
    try std.testing.expectEqual(@as(u32, 2), cfg.NKV);
    try std.testing.expectEqual(ArchType.deepseek_moe, cfg.arch);
    try std.testing.expectEqual(true, cfg.has_shared_experts);
    try std.testing.expectEqual(@as(u32, 64), cfg.n_experts);
    try std.testing.expectEqual(@as(u32, 6), cfg.top_k);
}

test "getArchConfig mixtral_8x7b" {
    const cfg = getArchConfig("mixtral_8x7b") orelse @panic("missing arch config");
    try std.testing.expectEqual(@as(u32, 4096), cfg.H);
    try std.testing.expectEqual(@as(u32, 32), cfg.NC);
    try std.testing.expectEqual(ArchType.moe, cfg.arch);
    try std.testing.expectEqual(false, cfg.has_shared_experts);
    try std.testing.expectEqual(@as(u32, 8), cfg.n_experts);
    try std.testing.expectEqual(@as(u32, 2), cfg.top_k);
    try std.testing.expectEqual(@as(u32, 14336), cfg.expert_inter_size);
}

test "getArchConfig zaya1_8b" {
    const cfg = getArchConfig("zaya1_8b") orelse @panic("missing arch config");
    try std.testing.expectEqual(@as(u32, 4096), cfg.H);
    try std.testing.expectEqual(@as(u32, 40), cfg.NC);
    try std.testing.expectEqual(@as(u32, 32), cfg.NH);
    try std.testing.expectEqual(@as(u32, 8), cfg.NKV);
    try std.testing.expectEqual(ArchType.moe, cfg.arch);
    try std.testing.expectEqual(@as(u32, 8), cfg.n_experts);
    try std.testing.expectEqual(@as(u32, 2), cfg.top_k);
}

test "getArchConfig gemma2_9b" {
    const cfg = getArchConfig("gemma2_9b") orelse @panic("missing arch config");
    try std.testing.expectEqual(@as(u32, 3584), cfg.H);
    try std.testing.expectEqual(@as(u32, 42), cfg.NC);
    try std.testing.expectEqual(@as(u32, 16), cfg.NH);
    try std.testing.expectEqual(@as(u32, 8), cfg.NKV);
    try std.testing.expectEqual(@as(u32, 256), cfg.HD);
    try std.testing.expectEqual(@as(u32, 14336), cfg.IM);
    try std.testing.expectEqual(@as(u32, 256128), cfg.NV);
    try std.testing.expectEqual(Activation.gelu, cfg.activation);
    try std.testing.expectEqual(true, cfg.has_qk_norm);
}

test "getArchConfig unknown returns null" {
    try std.testing.expectEqual(@as(?ArchConfig, null), getArchConfig("unknown_model"));
}

test "isMoe dense returns false" {
    try std.testing.expectEqual(false, isMoe("qwen3_0_6b"));
}

test "isMoe moe returns true" {
    try std.testing.expectEqual(true, isMoe("mixtral_8x7b"));
    try std.testing.expectEqual(true, isMoe("zaya1_8b"));
}

test "isMoe deepseek returns true" {
    try std.testing.expectEqual(true, isMoe("deepseek_v2_lite"));
    try std.testing.expectEqual(true, isMoe("deepseek_v3"));
}

test "isDeepSeekMoe" {
    try std.testing.expectEqual(true, isDeepSeekMoe("deepseek_v2_lite"));
    try std.testing.expectEqual(true, isDeepSeekMoe("deepseek_v3"));
    try std.testing.expectEqual(false, isDeepSeekMoe("mixtral_8x7b"));
    try std.testing.expectEqual(false, isDeepSeekMoe("qwen3_7b"));
}

test "hasQkNorm" {
    try std.testing.expectEqual(true, hasQkNorm("qwen3_7b"));
    try std.testing.expectEqual(true, hasQkNorm("gemma2_2b"));
    try std.testing.expectEqual(false, hasQkNorm("llama3_1_8b"));
}

test "hasSharedExpert" {
    try std.testing.expectEqual(true, hasSharedExpert("deepseek_v2_lite"));
    try std.testing.expectEqual(true, hasSharedExpert("deepseek_v3"));
    try std.testing.expectEqual(false, hasSharedExpert("mixtral_8x7b"));
    try std.testing.expectEqual(false, hasSharedExpert("qwen3_7b"));
}

test "count" {
    // Verify all 15 architectures are registered
    try std.testing.expectEqual(@as(usize, 15), count());
}

test "getArchConfig llama3_2_1b" {
    const cfg = getArchConfig("llama3_2_1b") orelse @panic("missing arch config");
    try std.testing.expectEqual(@as(u32, 2048), cfg.H);
    try std.testing.expectEqual(@as(u32, 16), cfg.NC);
    try std.testing.expectEqual(@as(u32, 16), cfg.NH);
    try std.testing.expectEqual(@as(u32, 8), cfg.NKV);
    try std.testing.expectEqual(@as(u32, 64), cfg.HD);
    try std.testing.expectEqual(@as(u32, 8192), cfg.IM);
    try std.testing.expectEqual(@as(u32, 128256), cfg.NV);
}

test "getArchConfig llama3_2_3b" {
    const cfg = getArchConfig("llama3_2_3b") orelse @panic("missing arch config");
    try std.testing.expectEqual(@as(u32, 3072), cfg.H);
    try std.testing.expectEqual(@as(u32, 28), cfg.NC);
    try std.testing.expectEqual(@as(u32, 24), cfg.NH);
    try std.testing.expectEqual(@as(u32, 128), cfg.HD);
}

test "all architectures have consistent fields" {
    // Verify every registered entry has non-zero essential dimensions
    const tags = [_][]const u8{
        "qwen3_0_6b",   "qwen3_1_5b",   "qwen3_7b",
        "qwen3_14b",    "qwen2_5_7b",   "qwen2_5_32b",
        "llama3_1_8b",  "llama3_2_1b",  "llama3_2_3b",
        "gemma2_2b",    "gemma2_9b",
        "deepseek_v2_lite", "deepseek_v3",
        "mixtral_8x7b", "zaya1_8b",
    };

    inline for (tags) |tag| {
        const cfg = getArchConfig(tag) orelse @panic("missing: " ++ tag);
        try std.testing.expect(cfg.H > 0);
        try std.testing.expect(cfg.NC > 0);
        try std.testing.expect(cfg.NH > 0);
        try std.testing.expect(cfg.NKV > 0);
        try std.testing.expect(cfg.HD > 0);
        try std.testing.expect(cfg.IM > 0);
        try std.testing.expect(cfg.NV > 0);
        try std.testing.expect(cfg.max_seq_len > 0);
    }
}

test "MoE entries have consistent expert fields" {
    const moe_tags = [_][]const u8{
        "deepseek_v2_lite", "deepseek_v3",
        "mixtral_8x7b",     "zaya1_8b",
    };

    inline for (moe_tags) |tag| {
        const cfg = getArchConfig(tag) orelse @panic("missing: " ++ tag);
        try std.testing.expect(cfg.n_experts > 0);
        try std.testing.expect(cfg.top_k > 0);
        try std.testing.expect(cfg.top_k <= cfg.n_experts);
        try std.testing.expect(cfg.expert_inter_size > 0);
    }
}

test "dense entries have zeroed MoE fields" {
    const dense_tags = [_][]const u8{
        "qwen3_0_6b",   "qwen3_1_5b",   "qwen3_7b",
        "qwen3_14b",    "qwen2_5_7b",   "qwen2_5_32b",
        "llama3_1_8b",  "llama3_2_1b",  "llama3_2_3b",
        "gemma2_2b",    "gemma2_9b",
    };

    inline for (dense_tags) |tag| {
        const cfg = getArchConfig(tag) orelse @panic("missing: " ++ tag);
        try std.testing.expectEqual(@as(u32, 0), cfg.n_experts);
        try std.testing.expectEqual(@as(u32, 0), cfg.top_k);
        try std.testing.expectEqual(@as(u32, 0), cfg.expert_inter_size);
        try std.testing.expectEqual(false, cfg.has_shared_experts);
    }
}
