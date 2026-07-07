//! Q4NX model data loader — extracts embeddings, norm weights, and RoPE tables
//! needed by the fused NPU+GPU execution engine (fused_execute.zig).
//!
//! Uses the existing model_ zig for header parsing and tensor index resolution.
//!
//! @section Fused Engine
const std = @import("std");
// Use the model_reader module (imported via build.zig)
const reader = @import("model_reader");

pub const ModelConfig = reader.ModelConfig;

/// Find the data_offsets[0] value for a given tensor key.





const log = std.log.scoped(.model_data);

/// All model data needed by the fused executor.
pub const ModelData = struct {
    config: ModelConfig,
    /// f32 embeddings: [vocab_size, hidden_dim]
    emb_f32: []f32,
    /// Final RMSNorm weights: [hidden_dim]
    final_norm: []f32,
    /// Per-layer input layernorm: [n_layers][hidden_dim]
    in_norm: [][]f32,
    /// Per-layer post-attention layernorm: [n_layers][hidden_dim]
    pa_norm: [][]f32,
    /// Precomputed RoPE sin: [max_context, head_dim]
    rope_sin: []f32,
    /// Precomputed RoPE cos: [max_context, head_dim]
    rope_cos: []f32,
    /// Separate LM head weights (null if tied with embeddings)
    lm_head_f32: ?[]f32,
    /// Whether embeddings are tied with lm_head
    tied_embeddings: bool,

    pub fn deinit(self: *ModelData, allocator: std.mem.Allocator) void {
        if (self.lm_head_f32) |lm| allocator.free(lm);
        allocator.free(self.rope_cos);
        allocator.free(self.rope_sin);
        for (self.pa_norm) |n| allocator.free(n);
        allocator.free(self.pa_norm);
        for (self.in_norm) |n| allocator.free(n);
        allocator.free(self.in_norm);
        allocator.free(self.final_norm);
        allocator.free(self.emb_f32);
    }
};

/// Load the full Q4NX model data from disk.
/// mmap's the file, parses the JSON header, extracts all tensors as f32.
pub fn loadModel(allocator: std.mem.Allocator, model_path: []const u8, model_tag: []const u8) !ModelData {
    const cfg = try  reader.parseQ4nxHeader(model_path, model_tag);
    if (!cfg.valid()) return error.InvalidModelConfig;

    log.info("Model: H={d} NC={d} NH={d} NKV={d} HD={d} IM={d} NV={d} GQA={d}", .{
        cfg.H, cfg.NC, cfg.NH, cfg.NKV, cfg.HD, cfg.IM, cfg.NV, cfg.GQA,
    });

    // ── Open and mmap model file ──
    const fd = try std.posix.open(model_path, .{ .RDONLY = true }, 0);
    errdefer std.posix.close(fd);

    var st: std.posix.Stat = undefined;
    _ = std.posix.fstat(fd, &st);
    const file_size = @as(usize, @intCast(st.size));

    const mapping = try std.posix.mmap(
        null,
        file_size,
        .{ .READ = true },
        .{ .TYPE = .PRIVATE },
        fd,
        0,
    );
    errdefer std.posix.munmap(mapping);
    std.posix.close(fd);

    if (mapping.len < 8) return error.InvalidModelFile;

    // Parse header size and JSON
    const hdr_size = std.mem.readInt(u64, mapping[0..8].*, .little);
    if (hdr_size == 0 or 8 + hdr_size > mapping.len) return error.InvalidModelHeader;

    const js = mapping[8 .. 8 + hdr_size];
    const data_start: usize = 8 + @as(usize, @intCast(hdr_size));
    _ = data_start;

    // Helper: get byte pointer to a tensor's data
    const tensorData = struct {
        fn get(tensor_offset: u64) []const u8 {
            const off = @as(usize, @intCast(tensor_offset));
            if (off >= mapping.len) return &.{};
            return mapping[off..];
        }
    }.get;

    const H = cfg.H;
    const NC = cfg.NC;
    const NV = cfg.NV;
    const MAX_CTX: u32 = 4096;

    // ── Check config.json for tie_word_embeddings ──
    var tied_embeddings = false;
    if (cfg.model_dir.len > 0) {
        const cfg_path = try std.fs.path.join(allocator, &[_][]const u8{ cfg.model_dir, "config.json" });
        defer allocator.free(cfg_path);
        if (std.fs.cwd().readFileAlloc(allocator, cfg_path, 1024 * 1024)) |cjs| {
            defer allocator.free(cjs);
            if (std.mem.indexOf(u8, cjs, "\"tie_word_embeddings\": true") != null or
                std.mem.indexOf(u8, cjs, "\"tie_word_embeddings\":true") != null)
            {
                tied_embeddings = true;
                log.info("tie_word_embeddings = true", .{});
            }
        } else |_| {}
    }

    // ── Load embeddings ──
    const emb_offset =  reader.findTensorInfo(js, "model.embed_tokens.weight") orelse return error.MissingEmbedTokens;
    log.info("Loading embeddings: {d} x {d}", .{ NV, H });
    const emb_raw = tensorData(emb_offset);
    const emb_f32 = try allocator.alloc(f32, @as(usize, NV) * H);
    errdefer allocator.free(emb_f32);

    const emb_hdr_len: usize = 0; // data starts at offset 0 within the tensor data
    const emb_bf16 = std.mem.bytesAsSlice(u16, emb_raw[emb_hdr_len..]);
    const emb_count = @min(emb_bf16.len, emb_f32.len);
    for (0..emb_count) |i| emb_f32[i] =  reader.bf16ToF32(emb_bf16[i]);
    for (emb_count..emb_f32.len) |i| emb_f32[i] = 0.0;

    // ── Load norm weights ──
    const in_norm_raw = try allocator.alloc([]f32, NC);
    errdefer allocator.free(in_norm_raw);
    const pa_norm_raw = try allocator.alloc([]f32, NC);
    errdefer allocator.free(pa_norm_raw);

    var bn_buf: [128]u8 = undefined;
    for (0..NC) |l| {
        // Input layernorm
        const in_key = try std.fmt.bufPrint(&bn_buf, "model.layers.{d}.input_layernorm.weight", .{l});
        const in_off =  reader.findTensorInfo(js, in_key) orelse return error.MissingLayerNorm;
        const in_slice = try allocator.alloc(f32, H);
        const in_raw = tensorData(in_off);
        const in_bf16 = std.mem.bytesAsSlice(u16, in_raw[0..H * 2]);
        for (0..H) |i| in_slice[i] =  reader.bf16ToF32(in_bf16[i]);
        in_norm_raw[l] = in_slice;

        // Post-attention layernorm
        const pa_key = try std.fmt.bufPrint(&bn_buf, "model.layers.{d}.post_attention_layernorm.weight", .{l});
        const pa_off =  reader.findTensorInfo(js, pa_key) orelse return error.MissingLayerNorm;
        const pa_slice = try allocator.alloc(f32, H);
        const pa_raw = tensorData(pa_off);
        const pa_bf16 = std.mem.bytesAsSlice(u16, pa_raw[0..H * 2]);
        for (0..H) |i| pa_slice[i] =  reader.bf16ToF32(pa_bf16[i]);
        pa_norm_raw[l] = pa_slice;
    }

    // ── Load final norm ──
    const fn_off =  reader.findTensorInfo(js, "model.norm.weight") orelse return error.MissingFinalNorm;
    const final_norm = try allocator.alloc(f32, H);
    const fn_raw = tensorData(fn_off);
    const fn_bf16 = std.mem.bytesAsSlice(u16, fn_raw[0..H * 2]);
    for (0..H) |i| final_norm[i] =  reader.bf16ToF32(fn_bf16[i]);

    // ── Load lm_head (if present and not tied) ──
    const lm_head_f32: ?[]f32 = null;
    if ( reader.keyExists(js, "lm_head.weight")) {
        const lm_off =  reader.findTensorInfo(js, "lm_head.weight") orelse 0;
        if (lm_off > 0 and !tied_embeddings) {
            const _lm_raw = tensorData(lm_off);
            _ = _lm_raw;
            const lm_tile_rows =  reader.findTileRows(js, "lm_head.weight") orelse 0;
            if (lm_tile_rows > 0) {
                // lm_head is I4/INT8 packed, not BF16
                // For now, fall back to using emb_f32 for LM head
                log.info("lm_head found (tile_rows={d}), using emb_f32 fallback", .{lm_tile_rows});
            }
        }
    }
    if (lm_head_f32 == null) {
        log.info("LM head: using emb_f32 (tied or no separate lm_head)", .{});
    }

    // ── Precompute RoPE tables ──
    log.info("Precomputing RoPE: theta={d:.0} max_ctx={d} head_dim={d}", .{ cfg.rope_theta, MAX_CTX, cfg.HD });
    const rope_sin = try allocator.alloc(f32, @as(usize, MAX_CTX) * cfg.HD);
    errdefer allocator.free(rope_sin);
    const rope_cos = try allocator.alloc(f32, @as(usize, MAX_CTX) * cfg.HD);
    errdefer allocator.free(rope_cos);

    const hd2 = cfg.HD / 2;
    for (0..MAX_CTX) |pos| {
        for (0..hd2) |d| {
            const freq = 1.0 / std.math.pow(f32, cfg.rope_theta, @as(f32, @floatFromInt(d)) / @as(f32, @floatFromInt(hd2)));
            const angle = @as(f32, @floatFromInt(pos)) * freq;
            rope_sin[pos * cfg.HD + d] = @sin(angle);
            rope_cos[pos * cfg.HD + d] = @cos(angle);
            // Mirror for the second half (used by some RoPE implementations)
            rope_sin[pos * cfg.HD + hd2 + d] = @sin(angle);
            rope_cos[pos * cfg.HD + hd2 + d] = @cos(angle);
        }
    }

    std.posix.munmap(mapping);

    return ModelData{
        .config = cfg,
        .emb_f32 = emb_f32,
        .final_norm = final_norm,
        .in_norm = in_norm_raw,
        .pa_norm = pa_norm_raw,
        .rope_sin = rope_sin,
        .rope_cos = rope_cos,
        .lm_head_f32 = lm_head_f32,
        .tied_embeddings = tied_embeddings,
    };
}

test "loadModel parses Qwen3-0.6B Q4NX" {
    const allocator = std.testing.allocator;
    const model_path = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";

    var data = try loadModel(allocator, model_path, "qwen3_0_6b");
    defer data.deinit(allocator);

    try std.testing.expect(data.config.valid());
    try std.testing.expectEqual(@as(u32, 1536), data.config.H);
    try std.testing.expectEqual(@as(u32, 28), data.config.NC);
    try std.testing.expectEqual(@as(u32, 12), data.config.NH);
    try std.testing.expectEqual(@as(u32, 2), data.config.NKV);
    try std.testing.expectEqual(@as(u32, 128), data.config.HD);
    try std.testing.expectEqual(@as(u32, 4096), data.config.IM);
    try std.testing.expect(data.emb_f32.len > 0);
    try std.testing.expect(data.final_norm.len == data.config.H);
    try std.testing.expect(data.in_norm.len == data.config.NC);
    try std.testing.expect(data.pa_norm.len == data.config.NC);
    try std.testing.expect(data.rope_sin.len == 4096 * data.config.HD);
    try std.testing.expect(data.rope_cos.len == 4096 * data.config.HD);
    // Verify some actual values
    try std.testing.expect(data.emb_f32[0] != 0.0);
    try std.testing.expect(data.final_norm[0] != 0.0);
    try std.testing.expect(data.in_norm[0][0] != 0.0);
}
