//! Dynamic KV cache quantization scheme selection and dispatch.
//! @section Scheduler
//!
//! Different transformer layers have different KV cache precision requirements.
//! Early layers often benefit from higher precision (FP16), while deeper layers
//! can tolerate aggressive quantization (INT4, NF4) with negligible quality loss.
//! This module selects the optimal scheme per-layer and per-page, and provides
//! the metadata the GPU needs to dequantize on read.
//!
//! ## Scheme encoding
//!
//! Each page table entry stores the quantization scheme in the upper 4 bits:
//!
//!     bit [31:28] = scheme_id  (1 nybble, up to 16 schemes)
//!     bit [27:0]  = physical_page_id  (up to 256M pages)
//!
//! The shader extracts the scheme ID and dispatches the appropriate
//! dequantization micro-kernel.
//!
//! ## Supported schemes
//!
//! | ID | Name  | Storage | Scale | Group size | Quality |
//! |----|-------|---------|-------|------------|---------|
//! | 0  | F32   | 32-bit  | none  | 1 (per-elem) | Reference |
//! | 1  | Q8_0  | 8-bit   | FP16  | 32          | High    |
//! | 2  | Q4_0  | 4-bit   | FP16  | 32          | Medium  |
//! | 3  | NF4   | 4-bit   | FP32  | 64          | Good (QLoRA) |
//! | 4  | Q4_1  | 4-bit   | FP16 x2 (scale+min) | 32 | Medium+ |
//! | 5  | Q3_0  | 3-bit   | FP16  | 32          | Low     |
//! | 6  | Q2_0  | 2-bit   | FP16  | 32          | Lowest  |
//! | 7  | MXFP4 | 4-bit   | E8M0  | 32          | Good (AMD) |
const std = @import("std");

const log = std.log.scoped(.kv_quant);

/// Supported KV cache quantization schemes.
pub const KvQuantScheme = enum(u4) {
    /// Full precision FP32 (no quantization).
    f32 = 0,
    /// INT8 per-block (Q8_0): 8-bit symmetric, block size 32, FP16 scale.
    q8_0 = 1,
    /// INT4 symmetric (Q4_0): 4-bit, block size 32, FP16 scale.
    q4_0 = 2,
    /// NormalFloat4 (NF4): 4-bit, block size 64, FP32 scale.
    /// Matches QLoRA's NF4 datatype for optimal 4-bit distribution.
    nf4 = 3,
    /// INT4 asymmetric (Q4_1): 4-bit, block size 32, FP16 scale + min.
    q4_1 = 4,
    /// INT3 symmetric (Q3_0): 3-bit, block size 32, FP16 scale.
    q3_0 = 5,
    /// INT2 symmetric (Q2_0): 2-bit, block size 32, FP16 scale.
    q2_0 = 6,
    /// Microscaling FP4 (MXFP4): 4-bit mantissa, shared E8M0 scale,
    /// block size 32. Matches AMD CDNA3 MXFP4 format.
    mxfp4 = 7,

    /// Number of defined schemes.
    const COUNT = 8;

    /// Storage bytes per element for this scheme.
    /// Returns the effective storage cost per KV element (not including
    /// shared scale overhead, which is accounted in bytesPerPage).
    pub fn bytesPerElement(self: KvQuantScheme) u32 {
        return switch (self) {
            .f32 => 4,
            .q8_0 => 1, // 8-bit data
            .q4_0 => 1, // 4-bit → 0.5 bytes, ceil to 1
            .nf4 => 1, // 4-bit
            .q4_1 => 1, // 4-bit
            .q3_0 => 1, // 3-bit → 0.375 bytes, ceil to 1
            .q2_0 => 1, // 2-bit → 0.25 bytes, ceil to 1
            .mxfp4 => 1, // 4-bit
        };
    }

    /// Total storage bytes for a page of `page_size` elements with `head_dim` KV heads.
    pub fn bytesPerPage(self: KvQuantScheme, page_size: u32, n_kv_heads: u32, head_dim: u32) u32 {
        const elements = page_size * n_kv_heads * head_dim;
        return switch (self) {
            .f32 => elements * 4,
            .q8_0 => elements + (elements / 32) * 2, // data + FP16 scales
            .q4_0 => elements / 2 + (elements / 32) * 2,
            .nf4 => elements / 2 + (elements / 64) * 4,
            .q4_1 => elements / 2 + (elements / 32) * 4,
            .q3_0 => (elements * 3 + 7) / 8 + (elements / 32) * 2,
            .q2_0 => elements / 4 + (elements / 32) * 2,
            .mxfp4 => elements / 2 + (elements / 32) * 1, // data + E8M0 scale
        };
    }

    /// Encode this scheme into a page table entry.
    /// Upper 4 bits = scheme_id, lower 28 bits = physical_page_id.
    pub fn encode(self: KvQuantScheme, physical_page_id: u32) u32 {
        const scheme_id: u32 = @intCast(@intFromEnum(self));
        return (scheme_id << 28) | (physical_page_id & 0x0FFFFFFF);
    }

    /// Decode scheme from a page table entry.
    pub fn decodeScheme(entry: u32) KvQuantScheme {
        const scheme_id = (entry >> 28) & 0xF;
        return @enumFromInt(@as(u4, @intCast(scheme_id)));
    }

    /// Decode physical page ID from a page table entry.
    pub fn decodePageId(entry: u32) u32 {
        return entry & 0x0FFFFFFF;
    }
};

/// Per-layer quantization profile. Determines which scheme to use
/// for each layer's KV cache.
pub const LayerQuantProfile = struct {
    /// Scheme to use for this layer.
    scheme: KvQuantScheme,
    /// Reason for this choice (for logging).
    reason: []const u8,
};

/// Quantization profile selector that maps layer index → scheme.
/// Uses simple heuristics: early layers get higher precision.
pub const QuantProfileSelector = struct {
    /// Per-layer profiles, indexed by layer index.
    profiles: []LayerQuantProfile,
    /// Number of layers.
    n_layers: u32,

    /// Build a quantization profile for all layers.
    /// @param n_layers Total number of transformer layers.
    /// @param scheme_base Base scheme for the middle layers.
    /// @param early_precision Boost precision for first N layers.
    /// @param late_compression Use sparser scheme for final layers.
    pub fn init(allocator: std.mem.Allocator, n_layers: u32, scheme_base: KvQuantScheme, early_precision: u32, late_compression: u32) !QuantProfileSelector {
        var profiles = try allocator.alloc(LayerQuantProfile, n_layers);

        for (0..n_layers) |i| {
            var scheme = scheme_base;
            if (i < early_precision) {
                scheme = upgradeScheme(scheme_base);
            } else if (i >= n_layers - late_compression) {
                scheme = downgradeScheme(scheme_base);
            }

            profiles[i] = .{
                .scheme = scheme,
                .reason = if (i < early_precision) "early layer (higher precision)" else if (i >= n_layers - late_compression) "late layer (compression ok)" else "default",
            };
        }

        log.info("Dynamic KV quant: {d} layers, base={s}, early={d}, late={d}", .{
            n_layers, @tagName(scheme_base), early_precision, late_compression,
        });

        return QuantProfileSelector{
            .profiles = profiles,
            .n_layers = n_layers,
        };
    }

    /// Get the quantization scheme for a specific layer.
    pub fn schemeForLayer(self: *const QuantProfileSelector, layer_idx: u32) KvQuantScheme {
        if (layer_idx >= self.n_layers) return .f32;
        return self.profiles[layer_idx].scheme;
    }

    /// Release profile storage.
    pub fn deinit(self: *QuantProfileSelector, allocator: std.mem.Allocator) void {
        allocator.free(self.profiles);
    }

    /// Upgrade scheme: move one level toward higher precision.
    fn upgradeScheme(scheme: KvQuantScheme) KvQuantScheme {
        return switch (scheme) {
            .q2_0 => .q3_0,
            .q3_0 => .q4_0,
            .q4_0 => .nf4,
            .q4_1 => .q8_0,
            .nf4 => .q8_0,
            .mxfp4 => .q8_0,
            .q8_0 => .f32,
            .f32 => .f32,
        };
    }

    /// Downgrade scheme: move one level toward lower precision.
    fn downgradeScheme(scheme: KvQuantScheme) KvQuantScheme {
        return switch (scheme) {
            .f32 => .q8_0,
            .q8_0 => .q4_0,
            .nf4 => .q4_0,
            .q4_0 => .q3_0,
            .q4_1 => .q4_0,
            .mxfp4 => .q4_0,
            .q3_0 => .q2_0,
            .q2_0 => .q2_0,
        };
    }
};

// ── Tests ──────────────────────────────────────────────────────────────────

test "KvQuantScheme encode/decode roundtrip" {
    const schemes = [_]KvQuantScheme{ .f32, .q8_0, .q4_0, .nf4, .q4_1, .q3_0, .q2_0, .mxfp4 };
    for (schemes) |scheme| {
        const encoded = scheme.encode(42);
        const decoded_scheme = KvQuantScheme.decodeScheme(encoded);
        const decoded_page = KvQuantScheme.decodePageId(encoded);
        try std.testing.expectEqual(scheme, decoded_scheme);
        try std.testing.expectEqual(@as(u32, 42), decoded_page);
    }
}

test "KvQuantScheme encode preserves lower bits" {
    const scheme = KvQuantScheme.q4_0;
    const page_id: u32 = 0x00ABCDEF;
    const encoded = scheme.encode(page_id);
    try std.testing.expectEqual(page_id, KvQuantScheme.decodePageId(encoded));
    try std.testing.expectEqual(scheme, KvQuantScheme.decodeScheme(encoded));
}

test "KvQuantScheme bytes per element" {
    try std.testing.expectEqual(@as(u32, 4), KvQuantScheme.f32.bytesPerElement());
    try std.testing.expectEqual(@as(u32, 1), KvQuantScheme.q8_0.bytesPerElement());
    try std.testing.expectEqual(@as(u32, 1), KvQuantScheme.q4_0.bytesPerElement());
    try std.testing.expectEqual(@as(u32, 1), KvQuantScheme.nf4.bytesPerElement());
}

test "QuantProfileSelector builds correct profiles" {
    const allocator = std.testing.allocator;
    var selector = try QuantProfileSelector.init(allocator, 32, .q4_0, 4, 4);
    defer selector.deinit(allocator);

    // Early layers (0-3): upgraded from q4_0 → nf4
    for (0..4) |i| {
        try std.testing.expectEqual(KvQuantScheme.nf4, selector.schemeForLayer(@intCast(i)));
    }
    // Middle layers (4-27): base q4_0
    try std.testing.expectEqual(KvQuantScheme.q4_0, selector.schemeForLayer(10));
    // Late layers (28-31): downgraded from q4_0 → q3_0
    try std.testing.expectEqual(KvQuantScheme.q3_0, selector.schemeForLayer(30));
}

test "QuantProfileSelector all layers default" {
    const allocator = std.testing.allocator;
    var selector = try QuantProfileSelector.init(allocator, 8, .f32, 0, 0);
    defer selector.deinit(allocator);

    for (0..8) |i| {
        try std.testing.expectEqual(KvQuantScheme.f32, selector.schemeForLayer(@intCast(i)));
    }
}

test "KvQuantScheme encode upper bits untouched by page_id" {
    const scheme = KvQuantScheme.q8_0;
    // Very large page ID still fits in 28 bits
    const page_id: u32 = 0x0FFFFFFF;
    const encoded = scheme.encode(page_id);
    try std.testing.expectEqual(page_id, KvQuantScheme.decodePageId(encoded));
    try std.testing.expectEqual(scheme, KvQuantScheme.decodeScheme(encoded));
}
