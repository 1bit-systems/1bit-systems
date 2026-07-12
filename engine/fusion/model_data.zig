//! Model data structures and loader for Q4NX-format quantized models.
//!
//! Q4NX format layout:
//!   1. [4 bytes] Magic: "Q4NX" (0x51344E58 as u32 BE)
//!   2. [4 bytes] Reserved (flags)
//!   3. [8 bytes] JSON header size (u64 LE)
//!   4. [n bytes] JSON header: { "tensor_name": { "dtype": "BF16"|"I8", "shape": [...], "data_offsets": [start, end] }, ... }
//!   5. [data]    Tensor payloads (relative to end of header)
//!
//! BF16 tensors: 2 bytes per element, stored as raw bfloat16 values.
//! I8/Q4NX tensors: 5120-byte tiled blocks containing scales, zero-points, and packed 4-bit data.
//!
//! After loading, all weights are in f32. Large I8 projection weights (Q, K, V, O, gate, up, down)
//! are not stored in ModelData — they are loaded directly by the NPU/CPU backends.
//! ModelData holds only: embeddings, norms, and RoPE tables.
//!
//! @section Fused Engine

const std = @import("std");
const log = std.log.scoped(.model_data);
const Io = std.Io;

/// Q4NX magic bytes as a 4-byte string literal.
const Q4NX_MAGIC = [4]u8{ 0x18, 0x87, 0x00, 0x00 };

// ── BF16 ↔ f32 conversion ─────────────────────────────────────

/// Convert a bfloat16 value (stored as u16) to f32.
fn bf16ToF32(v: u16) f32 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}

/// Convert an f32 value to bfloat16 (stored as u16), rounding to nearest even.
fn f32ToBf16(f: f32) u16 {
    const bits = @as(u32, @bitCast(f));
    const rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    return @as(u16, @intCast((bits +| rounding_bias) >> 16));
}

// ── Model configuration ──────────────────────────────────────

/// Transformer model configuration.
/// Short field names match the naming convention used across the fused engine.
pub const ModelConfig = struct {
    H: u32, // hidden_dim
    NC: u32, // n_layers
    NH: u32, // n_heads
    NKV: u32, // n_kv_heads
    HD: u32, // head_dim
    IM: u32, // intermediate_size (FFN hidden)
    NV: u32, // vocab_size
    max_seq_len: u32, // maximum supported sequence length
};

// ── Parsed tensor metadata from JSON header ──────────────────

/// Tensor dtype stored in the Q4NX file.
const TensorDtype = enum {
    bf16,
    i8_q4nx, // 4-bit block quantized in I8 tile format
    unknown,

    fn fromString(s: []const u8) TensorDtype {
        if (std.ascii.eqlIgnoreCase(s, "BF16")) return .bf16;
        if (std.ascii.eqlIgnoreCase(s, "I8") or std.ascii.eqlIgnoreCase(s, "I4")) return .i8_q4nx;
        return .unknown;
    }
};

// ── Default configs by model tag ─────────────────────────────

/// Known model configurations keyed by tag.
/// Add new models here as the codebase grows.
const KNOWN_MODELS = std.StaticStringMap(ModelConfig).initComptime(.{
    .{
        "qwen3_0_6b",
        ModelConfig{
            .H = 1024,
            .NC = 28,
            .NH = 16,
            .NKV = 8,
            .HD = 128,
            .IM = 3072,
            .NV = 151936,
            .max_seq_len = 4096,
        },
    },
    .{
        "qwen3_1_5b",
        ModelConfig{
            .H = 2048,
            .NC = 28,
            .NH = 16,
            .NKV = 2,
            .HD = 128,
            .IM = 8192,
            .NV = 151936,
            .max_seq_len = 4096,
        },
    },
    .{
        "qwen3_7b",
        ModelConfig{
            .H = 4096,
            .NC = 32,
            .NH = 32,
            .NKV = 8,
            .HD = 128,
            .IM = 16384,
            .NV = 151936,
            .max_seq_len = 8192,
        },
    },
});

/// Get the default configuration for a model tag.
/// Returns null for unknown tags — caller should derive from tensor shapes.
fn getDefaultConfig(tag: []const u8) ?ModelConfig {
    return KNOWN_MODELS.get(tag);
}

// ── Loaded model data ────────────────────────────────────────

/// All loaded model weights in f32, ready for inference.
///
/// Owns all slices. Call `deinit(allocator)` to free memory.
pub const ModelData = struct {
    config: ModelConfig,

    /// Token embeddings table: [NV * H] f32, row-major.
    emb_f32: []f32,

    /// Language model head weights: [NV * H] f32, row-major.
    /// If `tied_embeddings` is true, this slice is empty and the engine
    /// should use `emb_f32` for the LM head as well.
    lm_head_f32: []f32,

    /// Whether token embeddings and LM head share the same weight matrix.
    tied_embeddings: bool,

    /// Final RMS normalization weights: [H] f32.
    final_norm: []f32,

    /// Per-layer input RMS normalization weights: [NC][H] f32.
    in_norm: [][]f32,

    /// Per-layer post-attention RMS normalization weights: [NC][H] f32.
    pa_norm: [][]f32,

    /// Per-layer Qwen3-style QK-Norm weights (per-head RMSNorm applied to Q
    /// and K before RoPE): [NC][HD] f32. Empty inner slices if the tensor
    /// wasn't found in the file (older/non-Qwen3 architectures).
    q_norm: [][]f32,
    k_norm: [][]f32,

    /// Precomputed RoPE sin table: [max_seq_len * HD] f32.
    /// For position p and pair index i: sin(p * base^{-2i/HD}).
    rope_sin: []f32,

    /// Precomputed RoPE cos table: [max_seq_len * HD] f32.
    rope_cos: []f32,

    /// Per-layer dequantized projection weights, row-major [out_features, in_features]
    /// (PyTorch nn.Linear convention), for CPU-side GEMV. Empty slices (len 0) mean
    /// the tensor wasn't found in the file. Populated for every load regardless of
    /// dispatch policy — used by the `.cpu` backend in fused_execute.zig.
    q_weight: [][]f32, // [NC][NH*HD * H]
    k_weight: [][]f32, // [NC][NKV*HD * H]
    v_weight: [][]f32, // [NC][NKV*HD * H]
    o_weight: [][]f32, // [NC][H * NH*HD]
    gate_weight: [][]f32, // [NC][IM * H]
    up_weight: [][]f32, // [NC][IM * H]
    down_weight: [][]f32, // [NC][H * IM]

    /// Free all owned memory.
    pub fn deinit(self: *ModelData, allocator: std.mem.Allocator) void {
        log.debug("deinit: freeing model data", .{});

        // Embeddings
        allocator.free(self.emb_f32);

        // LM head (only free if separate from embeddings)
        if (!self.tied_embeddings and self.lm_head_f32.len > 0) {
            allocator.free(self.lm_head_f32);
        }

        // Final norm
        allocator.free(self.final_norm);

        // Per-layer input norms
        for (self.in_norm) |slice| {
            allocator.free(slice);
        }
        allocator.free(self.in_norm);

        // Per-layer post-attention norms
        for (self.pa_norm) |slice| {
            allocator.free(slice);
        }
        allocator.free(self.pa_norm);

        // Per-layer QK-norm weights
        for (self.q_norm) |slice| {
            allocator.free(slice);
        }
        allocator.free(self.q_norm);
        for (self.k_norm) |slice| {
            allocator.free(slice);
        }
        allocator.free(self.k_norm);

        // Per-layer projection weights
        inline for (.{ "q_weight", "k_weight", "v_weight", "o_weight", "gate_weight", "up_weight", "down_weight" }) |field| {
            for (@field(self, field)) |slice| {
                allocator.free(slice);
            }
            allocator.free(@field(self, field));
        }

        // RoPE tables
        allocator.free(self.rope_sin);
        allocator.free(self.rope_cos);

        self.* = undefined;
    }
};

// ── Q4NX I8 tile dequantization ─────────────────────────────

/// Dequantize one 5120-byte I8 tile into the output matrix.
///
/// Each tile is a 32×256 block of 4-bit quantized values.
/// Layout (per tile):
///   [0..512)     = 256 BF16 scales (uint16), indexed as [g*32 + lr]
///   [512..1024)  = 256 BF16 zero-points, indexed as [g*32 + lr]
///   [1024..5120) = 4096 bytes packed I4 data
///     For row lr (0..31):
///       lane = lr / 16, lr2 = lr % 16, bi = lr2 / 2, ns = lr % 2
///       packed_offset = lane * 2048 + col * 8 + bi
///       ns==0 → low nibble, ns==1 → high nibble
///     out[row][col] = nibble_value * scale + zero_point
fn dequantizeI8Block(block: *const [5120]u8, out: []f32, tr: u32, tc: u32, out_rows: u32, out_cols: u32) void {
    const scales: *const [256]u16 = @ptrCast(@alignCast(&block[0]));
    const zps: *const [256]u16 = @ptrCast(@alignCast(&block[512]));
    const packed_data = block[1024..];

    var lr: u32 = 0;
    while (lr < 32) : (lr += 1) {
        const row = tr * 32 + lr;
        if (row >= out_rows) continue;

        const lane = lr / 16;
        const lr2 = lr % 16;
        const bi = lr2 / 2;
        const ns = lr % 2;

        var g: u32 = 0;
        while (g < 8) : (g += 1) {
            const s_raw = bf16ToF32(scales[g * 32 + lr]);
            const z_raw = bf16ToF32(zps[g * 32 + lr]);
            // This Q4NX file carries widespread corrupt scale/zero-point bf16
            // values -- not just a handful of isolated NaN/Inf bit patterns,
            // but a real tail of finite-but-wild magnitudes at every scale
            // from ~10 up through ~1e38 (bf16 shares f32's exponent range).
            // Sampling real lm_head scales shows a tight, well-behaved
            // legitimate distribution (p50=0.008, p99=0.013 -- real
            // zero-points top out under 0.2) with *no* natural continuum
            // into larger values: it jumps straight from ~0.01 to
            // astronomical between p99 and p99.9. There is no legitimate
            // scale/zero-point anywhere near 1.0 in this model, let alone a
            // "moderately large but real" outlier tier -- a single corrupt
            // value (observed concretely: scale=-860 in one lm_head row,
            // comfortably under a naively-chosen 1000 threshold) still
            // dominates that row's entire dot product, wrecking argmax over
            // the whole vocabulary (or QKV/FFN, for per-layer projection
            // weights). Zero out just the affected 32-element group instead.
            const plausible_max: f32 = 1.0;
            const s_bad = !std.math.isFinite(s_raw) or @abs(s_raw) > plausible_max;
            const z_bad = !std.math.isFinite(z_raw) or @abs(z_raw) > plausible_max;
            if (s_bad or z_bad) {
                log.warn("Implausible dequant scale/zp at row={d} group={d} (scale={d}, zp={d}); zeroing group", .{ row, g, s_raw, z_raw });
            }
            const s: f32 = if (s_bad) 0 else s_raw;
            const z: f32 = if (z_bad) 0 else z_raw;

            var c: u32 = 0;
            while (c < 32) : (c += 1) {
                const col = tc * 256 + g * 32 + c;
                if (col >= out_cols) continue;

                const bv = packed_data[lane * 2048 + c * 8 + bi];
                const nibble: u32 = if (ns == 0) bv & 0x0F else (bv >> 4) & 0x0F;
                out[row * out_cols + col] = @as(f32, @floatFromInt(nibble)) * s + z;
            }
        }
    }
}

/// Dequantize a full I8/Q4NX tensor of shape [out_rows, out_cols] stored as
/// row-major 32×256 tiles (tr outer, tc inner — matches `quantize_i8_tiled`
/// in hf_to_q4nx.py) starting at `data_offset` within the file's tensor
/// payload region.
fn dequantizeI8TensorFromMemory(
    file_data: []const u8,
    hdr_size: u64,
    data_offset: u64,
    data_size: u64,
    out_rows: u32,
    out_cols: u32,
    allocator: std.mem.Allocator,
) ![]f32 {
    const file_offset: usize = 8 + @as(usize, @intCast(hdr_size)) + @as(usize, @intCast(data_offset));

    if (file_offset + @as(usize, @intCast(data_size)) > file_data.len) {
        log.err("I8 tensor data offset {d} + size {d} exceeds file size {d}", .{ file_offset, data_size, file_data.len });
        return error.TensorOutOfBounds;
    }

    const ntc = tileCols(out_cols);
    const ntr = (out_rows + 31) / 32;
    const expected_tiles = @as(usize, ntr) * @as(usize, ntc);
    const available_tiles = @as(usize, @intCast(data_size)) / 5120;
    const n_tiles = @min(expected_tiles, available_tiles);
    if (n_tiles < expected_tiles) {
        log.warn("I8 tensor has {d} tiles, expected {d}; remainder left as zero", .{ n_tiles, expected_tiles });
    }

    const numel = @as(usize, @intCast(out_rows)) * @as(usize, @intCast(out_cols));
    const result = try allocator.alloc(f32, numel);
    errdefer allocator.free(result);
    @memset(result, 0);

    const tile_data = file_data[file_offset..];

    var tile_idx: usize = 0;
    outer: for (0..ntr) |tr| {
        for (0..ntc) |tc| {
            if (tile_idx >= n_tiles) break :outer;
            const block_start = tile_idx * 5120;
            const block: *const [5120]u8 = @ptrCast(tile_data[block_start..][0..5120]);
            dequantizeI8Block(block, result, @intCast(tr), @intCast(tc), out_rows, out_cols);
            tile_idx += 1;
        }
    }

    return result;
}

// ── JSON header parsing ──────────────────────────────────────

/// Tensor descriptor extracted from the JSON header.
const TensorDesc = struct {
    name: []u8,
    dtype: []u8,
    shape: std.ArrayListUnmanaged(u32),
    data_offsets: std.ArrayListUnmanaged(u64),
};

/// Parse a JSON integer array like "[1, 2, 3]" into an ArrayList.
fn parseJsonInts(allocator: std.mem.Allocator, comptime T: type, s: []const u8) !std.ArrayListUnmanaged(T) {
    var result: std.ArrayListUnmanaged(T) = .empty;
    var i: usize = 0;

    while (i < s.len and (s[i] == ' ' or s[i] == '\t' or s[i] == '\n' or s[i] == '[')) i += 1;

    while (i < s.len and s[i] != ']') {
        while (i < s.len and (s[i] == ' ' or s[i] == '\t' or s[i] == '\n' or s[i] == ',')) i += 1;
        if (i >= s.len or s[i] == ']') break;

        const start = i;
        while (i < s.len and s[i] >= '0' and s[i] <= '9') i += 1;
        if (i > start) {
            const val = try std.fmt.parseInt(T, s[start..i], 10);
            try result.append(allocator, val);
        }
    }
    return result;
}

/// Simple JSON value scanner — jumps past a JSON value starting at `pos`,
/// returns the end position (one past the value).
fn skipJsonValue(s: []const u8, pos: usize) usize {
    if (pos >= s.len) return pos;
    var i = pos;
    switch (s[i]) {
        '"' => {
            i += 1;
            while (i < s.len) : (i += 1) {
                if (s[i] == '\\') {
                    i += 1;
                    continue;
                }
                if (s[i] == '"') {
                    i += 1;
                    break;
                }
            }
        },
        '{', '[' => {
            var depth: usize = 1;
            i += 1;
            while (i < s.len and depth > 0) : (i += 1) {
                if (s[i] == '"') {
                    i += 1;
                    while (i < s.len and !(s[i] == '"' and s[i - 1] != '\\')) i += 1;
                    continue;
                }
                if (s[i] == '{' or s[i] == '[') {
                    depth += 1;
                    continue;
                }
                if (s[i] == '}' or s[i] == ']') {
                    depth -= 1;
                    continue;
                }
            }
        },
        else => {
            while (i < s.len and s[i] != ',' and s[i] != '}' and s[i] != ']' and s[i] != ' ' and s[i] != '\t' and s[i] != '\n') i += 1;
        },
    }
    return i;
}

/// Extract a substring value for a given key from a JSON object string.
fn jsonFindValue(s: []const u8, key: []const u8) ?[]const u8 {
    var pos: usize = 0;

    while (pos < s.len) {
        const key_start = std.mem.indexOfScalarPos(u8, s, pos, '"') orelse return null;
        if (key_start == 0) return null;

        var pre = key_start;
        while (pre > 0 and (s[pre - 1] == ' ' or s[pre - 1] == '\t' or s[pre - 1] == '\n')) pre -= 1;
        if (pre == 0 or (s[pre - 1] != '{' and s[pre - 1] != ',')) {
            pos = key_start + 1;
            continue;
        }

        var ke = key_start + 1;
        while (ke < s.len and s[ke] != '"') : (ke += 1) {
            if (s[ke] == '\\') ke += 1;
        }
        if (ke >= s.len) return null;

        if (!std.mem.eql(u8, s[key_start + 1 .. ke], key)) {
            pos = ke + 1;
            continue;
        }

        const colon = std.mem.indexOfScalarPos(u8, s, ke + 1, ':') orelse return null;

        var vs = colon + 1;
        while (vs < s.len and (s[vs] == ' ' or s[vs] == '\t' or s[vs] == '\n')) vs += 1;
        if (vs >= s.len) return null;

        const ve = skipJsonValue(s, vs);
        return s[vs..ve];
    }
    return null;
}

/// Unquote a JSON string: "\"foo\"" -> "foo".
fn jsonUnquote(s: []const u8) []const u8 {
    if (s.len >= 2 and s[0] == '"' and s[s.len - 1] == '"') {
        return s[1 .. s.len - 1];
    }
    return s;
}

/// Parse the JSON header into a list of tensor descriptors.
fn parseJsonHeader(allocator: std.mem.Allocator, json_bytes: []const u8) !std.ArrayListUnmanaged(TensorDesc) {
    var tensors: std.ArrayListUnmanaged(TensorDesc) = .empty;
    errdefer {
        for (tensors.items) |*t| {
            allocator.free(t.name);
            allocator.free(t.dtype);
            t.shape.deinit(allocator);
            t.data_offsets.deinit(allocator);
        }
        tensors.deinit(allocator);
    }

    const s = json_bytes;
    var pos: usize = 0;

    while (pos < s.len and (s[pos] == ' ' or s[pos] == '\t' or s[pos] == '\n')) pos += 1;
    if (pos >= s.len or s[pos] != '{') {
        log.err("JSON header does not start with '{{'", .{});
        return error.InvalidJsonHeader;
    }
    pos += 1;

    while (pos < s.len) {
        while (pos < s.len and (s[pos] == ' ' or s[pos] == '\t' or s[pos] == '\n')) pos += 1;
        if (pos >= s.len or s[pos] == '}') break;

        if (s[pos] == ',') {
            pos += 1;
            continue;
        }

        if (s[pos] != '"') {
            pos += 1;
            continue;
        }
        var ke = pos + 1;
        while (ke < s.len and s[ke] != '"') : (ke += 1) {
            if (s[ke] == '\\') ke += 1;
        }
        if (ke >= s.len) break;
        const key = s[pos + 1 .. ke];
        pos = ke + 1;

        while (pos < s.len and s[pos] != ':') pos += 1;
        if (pos >= s.len) break;
        pos += 1;

        while (pos < s.len and (s[pos] == ' ' or s[pos] == '\t' or s[pos] == '\n')) pos += 1;
        if (pos >= s.len) break;

        if (s[pos] != '{') {
            const ve = skipJsonValue(s, pos);
            pos = ve;
            continue;
        }
        const vs = pos;
        const ve = skipJsonValue(s, vs);
        const val_sub = s[vs..ve];
        pos = ve;

        if (std.mem.indexOf(u8, val_sub, "data_offsets") == null) continue;

        const dtype_str = jsonUnquote(jsonFindValue(val_sub, "dtype") orelse continue);

        const shape_str = jsonFindValue(val_sub, "shape") orelse continue;
        var shape_list = try parseJsonInts(allocator, u32, shape_str);

        const offsets_str = jsonFindValue(val_sub, "data_offsets") orelse {
            shape_list.deinit(allocator);
            continue;
        };
        var offsets_list = try parseJsonInts(allocator, u64, offsets_str);
        if (offsets_list.items.len < 2) {
            shape_list.deinit(allocator);
            offsets_list.deinit(allocator);
            continue;
        }

        const name_copy = try allocator.dupe(u8, key);
        const dtype_copy = try allocator.dupe(u8, dtype_str);

        try tensors.append(allocator, .{
            .name = name_copy,
            .dtype = dtype_copy,
            .shape = shape_list,
            .data_offsets = offsets_list,
        });
    }

    log.info("Parsed {d} tensors from JSON header ({d} bytes)", .{ tensors.items.len, s.len });
    return tensors;
}

/// Find a tensor by name in the parsed list.
fn findTensor(tensors: []const TensorDesc, name: []const u8) ?usize {
    for (tensors, 0..) |t, i| {
        if (std.mem.eql(u8, t.name, name)) return i;
    }
    return null;
}

/// Count layers from tensor names by finding the max layer index.
fn countLayers(tensors: []const TensorDesc) u32 {
    var max_layer: u32 = 0;
    for (tensors) |t| {
        if (std.mem.startsWith(u8, t.name, "model.layers.")) {
            const rest = t.name["model.layers.".len..];
            const dot_pos = std.mem.indexOfScalar(u8, rest, '.') orelse continue;
            const layer_str = rest[0..dot_pos];
            const layer = std.fmt.parseInt(u32, layer_str, 10) catch continue;
            if (layer + 1 > max_layer) max_layer = layer + 1;
        }
    }
    return max_layer;
}

/// Derive the number of tile columns from hidden_dim.
fn tileCols(hidden_dim: u32) u32 {
    return (hidden_dim + 255) / 256;
}

/// Derive model configuration from parsed tensor shapes.
fn deriveConfig(tensors: []const TensorDesc, tag: []const u8) !ModelConfig {
    const defaults = getDefaultConfig(tag) orelse return error.UnknownModelTag;

    var H: u32 = defaults.H;
    var NV: u32 = defaults.NV;
    var NC: u32 = defaults.NC;
    var NH: u32 = defaults.NH;
    var NKV: u32 = defaults.NKV;
    var HD: u32 = defaults.HD;
    var IM: u32 = defaults.IM;
    const max_seq_len: u32 = defaults.max_seq_len;

    if (findTensor(tensors, "model.embed_tokens.weight")) |idx| {
        const t = tensors[idx];
        if (t.shape.items.len >= 2) {
            NV = t.shape.items[0];
            H = t.shape.items[1];
            log.info("deriveConfig: embedding shape NV={d} H={d}", .{ NV, H });
        }
    }
    log.info("deriveConfig: BEFORE q_proj H={d} NH={d} NKV={d} IM={d}", .{ H, NH, NKV, IM });

    const layer_count = countLayers(tensors);
    if (layer_count > 0) NC = layer_count;

    if (findTensor(tensors, "model.layers.0.self_attn.q_proj.weight")) |idx| {
        const t = tensors[idx];
        if (t.shape.items.len >= 1 and H > 0) {
            const n_blocks = t.shape.items[0];
            const ntc = tileCols(H);
            if (ntc > 0) {
                const ntr = n_blocks / ntc;
                const q_out = ntr * 32;
                const hd_guess: u32 = 128;
                NH = q_out / hd_guess;
                HD = hd_guess;
            }
        }
    }

    if (findTensor(tensors, "model.layers.0.self_attn.k_proj.weight")) |idx| {
        const t = tensors[idx];
        if (t.shape.items.len >= 1 and H > 0) {
            const n_blocks = t.shape.items[0];
            const ntc = tileCols(H);
            if (ntc > 0 and HD > 0) {
                const ntr = n_blocks / ntc;
                const k_out = ntr * 32;
                NKV = k_out / HD;
            }
        }
    }

    if (findTensor(tensors, "model.layers.0.mlp.gate_proj.weight")) |idx| {
        const t = tensors[idx];
        if (t.shape.items.len >= 1 and H > 0) {
            const n_blocks = t.shape.items[0];
            const ntc = tileCols(H);
            if (ntc > 0) {
                const ntr = n_blocks / ntc;
                IM = ntr * 32;
            }
        }
    }

    return ModelConfig{
        .H = H,
        .NC = NC,
        .NH = NH,
        .NKV = NKV,
        .HD = HD,
        .IM = IM,
        .NV = NV,
        .max_seq_len = max_seq_len,
    };
}

// ── RoPE table computation ───────────────────────────────────

/// Precompute sinusoidal RoPE tables for all positions up to max_seq_len.
fn computeRopeTables(allocator: std.mem.Allocator, head_dim: u32, max_seq_len: u32) !struct { sin_table: []f32, cos_table: []f32 } {
    const total = max_seq_len * head_dim;
    const sin_table = try allocator.alloc(f32, total);
    errdefer allocator.free(sin_table);
    const cos_table = try allocator.alloc(f32, total);
    errdefer allocator.free(cos_table);

    const base: f32 = 10000.0;
    const inv_scale: f32 = @as(f32, @floatFromInt(head_dim));

    // Layout must match applyRoPE's read pattern in fused_execute.zig, which
    // rotates pairs (x[d], x[d + head_dim/2]) HuggingFace "rotate_half" style
    // and reads a single cos/sin per frequency index d from
    // table[pos*head_dim + d] for d in 0..head_dim/2 -- i.e. the table's
    // first half must hold frequencies 0..head_dim/2-1 directly, in order.
    // (Previously stored as interleaved-duplicate pairs -- table[2i]=table[2i+1]=freq_i
    // -- which applyRoPE's flat per-index read silently reinterpreted as
    // freq_(d/2), scrambling every frequency past index 0. Harmless as long
    // as Q/K were always the zero vector upstream of this, which they were
    // until the #56 fix started running real per-layer compute.)
    var pos: u32 = 0;
    while (pos < max_seq_len) : (pos += 1) {
        const p = @as(f32, @floatFromInt(pos));
        var i: u32 = 0;
        while (i < head_dim / 2) : (i += 1) {
            const theta = p * std.math.pow(f32, base, -2.0 * @as(f32, @floatFromInt(i)) / inv_scale);
            const sin_val = @sin(theta);
            const cos_val = @cos(theta);
            const base_idx = pos * head_dim;
            sin_table[base_idx + i] = sin_val;
            cos_table[base_idx + i] = cos_val;
        }
    }

    log.info("RoPE tables: {d} positions × {d} dim = {d} values each", .{ max_seq_len, head_dim, total });
    return .{ .sin_table = sin_table, .cos_table = cos_table };
}

// ── Tensor loading helpers ───────────────────────────────────

/// Load a BF16 tensor from the in-memory file data and convert to f32.
fn loadBf16TensorFromMemory(file_data: []const u8, hdr_size: u64, data_offset: u64, data_size: u64, numel: usize, allocator: std.mem.Allocator) ![]f32 {
    const file_offset: usize = 8 + @as(usize, @intCast(hdr_size)) + @as(usize, @intCast(data_offset));

    if (file_offset + @as(usize, @intCast(data_size)) > file_data.len) {
        log.err("Tensor data offset {d} + size {d} exceeds file size {d}", .{ file_offset, data_size, file_data.len });
        return error.TensorOutOfBounds;
    }

    const bf16_bytes = file_data[file_offset .. file_offset + @as(usize, @intCast(data_size))];
    const bf16_values = std.mem.bytesAsSlice(u16, bf16_bytes);

    if (bf16_values.len != numel) {
        log.warn("Tensor element count mismatch: expected {d}, got {d}", .{ numel, bf16_values.len });
    }

    const result = try allocator.alloc(f32, numel);
    errdefer allocator.free(result);

    const count = @min(bf16_values.len, numel);
    for (bf16_values[0..count], 0..) |bf16, idx| {
        result[idx] = bf16ToF32(bf16);
    }
    if (count < numel) {
        @memset(result[count..], 0);
    }

    return result;
}

/// Load per-layer normalization weights (BF16) into a nested array.
fn loadLayerNormsFromMemory(
    file_data: []const u8,
    hdr_size: u64,
    tensors: []const TensorDesc,
    prefix: []const u8,
    n_layers: u32,
    hidden_dim: u32,
    allocator: std.mem.Allocator,
) ![][]f32 {
    const norms = try allocator.alloc([]f32, n_layers);
    errdefer {
        for (norms[0..]) |slice| allocator.free(slice);
        allocator.free(norms);
    }

    var layer: u32 = 0;
    while (layer < n_layers) : (layer += 1) {
        const tensor_name = try std.fmt.allocPrint(allocator, "model.layers.{d}.{s}.weight", .{ layer, prefix });
        defer allocator.free(tensor_name);

        if (findTensor(tensors, tensor_name)) |idx| {
            const t = tensors[idx];
            const dtype_str = t.dtype;
            const dtype = TensorDtype.fromString(dtype_str);
            const numel = @as(usize, @intCast(hidden_dim));
            const data_size = t.data_offsets.items[1] - t.data_offsets.items[0];
            const data_offset = t.data_offsets.items[0];

            norms[layer] = switch (dtype) {
                .bf16 => try loadBf16TensorFromMemory(file_data, hdr_size, data_offset, data_size, numel, allocator),
                .i8_q4nx => blk: {
                    log.warn("Layer norm '{s}' has I8 dtype, expected BF16", .{tensor_name});
                    const zeros = try allocator.alloc(f32, numel);
                    @memset(zeros, 1.0);
                    break :blk zeros;
                },
                .unknown => blk: {
                    log.warn("Layer norm '{s}' has unknown dtype '{s}'", .{ tensor_name, dtype_str });
                    const zeros = try allocator.alloc(f32, numel);
                    @memset(zeros, 1.0);
                    break :blk zeros;
                },
            };
        } else {
            log.warn("Layer norm tensor '{s}' not found; filling with ones", .{tensor_name});
            const fallback = try allocator.alloc(f32, @as(usize, @intCast(hidden_dim)));
            @memset(fallback, 1.0);
            norms[layer] = fallback;
        }
    }

    return norms;
}

/// Load and dequantize one per-layer I8/Q4NX projection weight across all layers,
/// row-major [out_features, in_features] (PyTorch nn.Linear convention). Used for
/// CPU-side GEMV (`.cpu` dispatch backend) — the NPU/GPU backends load these
/// tensors themselves and don't go through ModelData.
fn loadLayerProjWeightsFromMemory(
    file_data: []const u8,
    hdr_size: u64,
    tensors: []const TensorDesc,
    tensor_suffix: []const u8,
    n_layers: u32,
    out_features: u32,
    in_features: u32,
    allocator: std.mem.Allocator,
) ![][]f32 {
    const weights = try allocator.alloc([]f32, n_layers);
    errdefer {
        for (weights[0..]) |slice| allocator.free(slice);
        allocator.free(weights);
    }

    const numel = @as(usize, @intCast(out_features)) * @as(usize, @intCast(in_features));

    var layer: u32 = 0;
    while (layer < n_layers) : (layer += 1) {
        const tensor_name = try std.fmt.allocPrint(allocator, "model.layers.{d}.{s}.weight", .{ layer, tensor_suffix });
        defer allocator.free(tensor_name);

        if (findTensor(tensors, tensor_name)) |idx| {
            const t = tensors[idx];
            const dtype_str = t.dtype;
            const dtype = TensorDtype.fromString(dtype_str);
            const data_size = t.data_offsets.items[1] - t.data_offsets.items[0];
            const data_offset = t.data_offsets.items[0];

            weights[layer] = switch (dtype) {
                .i8_q4nx => try dequantizeI8TensorFromMemory(file_data, hdr_size, data_offset, data_size, out_features, in_features, allocator),
                .bf16 => try loadBf16TensorFromMemory(file_data, hdr_size, data_offset, data_size, numel, allocator),
                .unknown => blk: {
                    log.warn("Projection weight '{s}' has unknown dtype '{s}'; zeroing", .{ tensor_name, dtype_str });
                    const zeros = try allocator.alloc(f32, numel);
                    @memset(zeros, 0);
                    break :blk zeros;
                },
            };
        } else {
            log.warn("Projection weight tensor '{s}' not found; zeroing", .{tensor_name});
            const fallback = try allocator.alloc(f32, numel);
            @memset(fallback, 0);
            weights[layer] = fallback;
        }
    }

    return weights;
}

// ── Main loader ──────────────────────────────────────────────

/// Load a Q4NX model from file.
///
/// Parameters:
///   allocator - memory allocator (all loaded data is owned by this allocator)
///   io        - I/O abstraction (from process init, or a Threaded instance)
///   path      - filesystem path to the .q4nx model file
///   model_tag - model identifier (e.g., "qwen3_0_6b") for default config lookup
///
/// Returns a fully populated ModelData with all weights in f32.
pub fn loadModel(allocator: std.mem.Allocator, io: Io, path: []const u8, model_tag: []const u8) !ModelData {
    log.info("Loading model: {s} (tag: '{s}')", .{ path, model_tag });

    const file = try std.Io.Dir.openFileAbsolute(io, path, .{ .mode = .read_only });
    defer file.close(io);

    const file_len = try file.length(io);

    var read_buf: [8192]u8 = undefined;
    var file_reader = file.reader(io, &read_buf);
    const file_data = try file_reader.interface.readAlloc(allocator, file_len);
    errdefer allocator.free(file_data);

    // ── 1. Check magic ──
    if (file_data.len < 4) return error.InvalidMagic;
    if (!std.mem.eql(u8, file_data[0..4], &Q4NX_MAGIC)) {
        log.warn("Invalid magic: expected 'Q4NX', got '{any}'", .{file_data[0..4]});
        return error.InvalidMagic;
    }
    log.debug("Magic verified: Q4NX", .{});

    // ── 2. Read reserved flags (bytes 4-7) ──
    if (file_data.len >= 8) {
        const flags = std.mem.bytesAsSlice(u32, file_data[4..8])[0];
        if (flags != 0) {
            log.debug("Flags: 0x{X:0>8}", .{flags});
        }
    }

    // ── 3. JSON starts at byte 8 (no separate size field — find closing brace) ──
    const json_offset: usize = 8;
    var depth: u32 = 0;
    var hdr_end: usize = json_offset;
    while (hdr_end < file_data.len) : (hdr_end += 1) {
        const c = file_data[hdr_end];
        if (c == '{') depth += 1;
        if (c == '}') {
            depth -= 1;
            if (depth == 0) {
                hdr_end += 1;
                break;
            }
        }
    }
    if (depth != 0) return error.InvalidJsonHeader;
    const hdr_size = hdr_end - json_offset;
    log.info("Header size: {d} bytes", .{hdr_size});
    const hdr_json = file_data[json_offset..hdr_end];

    // ── 5. Parse JSON header into tensor descriptors ──
    var tensor_list = try parseJsonHeader(allocator, hdr_json);
    defer {
        for (tensor_list.items) |*t| {
            allocator.free(t.name);
            allocator.free(t.dtype);
            t.shape.deinit(allocator);
            t.data_offsets.deinit(allocator);
        }
        tensor_list.deinit(allocator);
    }

    const tensors = tensor_list.items;
    if (tensors.len == 0) {
        log.err("No tensors found in header", .{});
        return error.NoTensors;
    }

    // ── 6. Derive model configuration ──
    const config = try deriveConfig(tensors, model_tag);
    log.info("Config: H={d} NC={d} NH={d} NKV={d} HD={d} IM={d} NV={d} max_seq={d}", .{
        config.H, config.NC, config.NH, config.NKV, config.HD, config.IM, config.NV, config.max_seq_len,
    });

    const HD = config.HD;
    const max_seq_len = config.max_seq_len;

    // ── 7. Precompute RoPE tables ──
    const result = if (computeRopeTables(allocator, HD, max_seq_len)) |rope_tables| blk: {
        break :blk try loadModelFinish(allocator, file_data, hdr_size, config, tensors, rope_tables.sin_table, rope_tables.cos_table);
    } else |err| blk: {
        log.warn("RoPE table computation failed: {s}; using empty tables", .{@errorName(err)});
        const sin_empty = try allocator.alloc(f32, 0);
        const cos_empty = try allocator.alloc(f32, 0);
        break :blk try loadModelFinish(allocator, file_data, hdr_size, config, tensors, sin_empty, cos_empty);
    };

    // Free file data now that all tensors have been extracted
    allocator.free(file_data);

    return result;
}

/// Inner continuation of loadModel after RoPE tables are computed.
fn loadModelFinish(
    allocator: std.mem.Allocator,
    file_data: []const u8,
    hdr_size: u64,
    config: ModelConfig,
    tensors: []const TensorDesc,
    rope_sin: []f32,
    rope_cos: []f32,
) !ModelData {
    const H = config.H;
    const NV = config.NV;

    // ── 8. Load embeddings ──
    var emb_f32: []f32 = &.{};
    var lm_head_f32: []f32 = &.{};
    var tied_embeddings = false;

    // 8a. Token embeddings
    if (findTensor(tensors, "model.embed_tokens.weight")) |idx| {
        const t = tensors[idx];
        const dtype_str = t.dtype;
        const dtype = TensorDtype.fromString(dtype_str);
        const numel = @as(usize, @intCast(NV)) * @as(usize, @intCast(H));
        const data_size = t.data_offsets.items[1] - t.data_offsets.items[0];
        const data_offset = t.data_offsets.items[0];

        emb_f32 = switch (dtype) {
            .bf16 => try loadBf16TensorFromMemory(file_data, hdr_size, data_offset, data_size, numel, allocator),
            .i8_q4nx => blk: {
                log.info("Embeddings have I8 dtype; dequantizing {d} tiles", .{data_size / 5120});
                break :blk try dequantizeI8TensorFromMemory(file_data, hdr_size, data_offset, data_size, NV, H, allocator);
            },
            .unknown => blk: {
                log.warn("Embeddings have unknown dtype '{s}'", .{dtype_str});
                const zeros = try allocator.alloc(f32, numel);
                @memset(zeros, 0);
                break :blk zeros;
            },
        };
        log.info("Embeddings: {d}×{d} (dtype={s})", .{ NV, H, dtype_str });
    } else {
        log.warn("model.embed_tokens.weight not found; allocating zeros", .{});
        emb_f32 = try allocator.alloc(f32, @as(usize, @intCast(NV)) * @as(usize, @intCast(H)));
        @memset(emb_f32, 0);
    }

    // 8b. LM head (optional — may be tied with embeddings)
    if (findTensor(tensors, "lm_head.weight")) |idx| {
        const t = tensors[idx];
        const dtype_str = t.dtype;
        const dtype = TensorDtype.fromString(dtype_str);
        const numel = @as(usize, @intCast(NV)) * @as(usize, @intCast(H));
        const data_size = t.data_offsets.items[1] - t.data_offsets.items[0];
        const data_offset = t.data_offsets.items[0];

        lm_head_f32 = switch (dtype) {
            .bf16 => try loadBf16TensorFromMemory(file_data, hdr_size, data_offset, data_size, numel, allocator),
            .i8_q4nx => blk: {
                log.info("lm_head has I8 dtype; dequantizing {d} tiles", .{data_size / 5120});
                break :blk try dequantizeI8TensorFromMemory(file_data, hdr_size, data_offset, data_size, NV, H, allocator);
            },
            .unknown => blk: {
                log.warn("lm_head has unknown dtype '{s}'", .{dtype_str});
                const zeros = try allocator.alloc(f32, numel);
                @memset(zeros, 0);
                break :blk zeros;
            },
        };
        tied_embeddings = false;
        log.info("LM head: {d}×{d} (dtype={s}, separate from embeddings)", .{ NV, H, dtype_str });
    } else {
        lm_head_f32 = &.{};
        tied_embeddings = true;
        log.info("LM head: tied with embeddings", .{});
    }

    // ── 9. Load final norm ──
    var final_norm: []f32 = &.{};
    if (findTensor(tensors, "model.norm.weight")) |idx| {
        const t = tensors[idx];
        const dtype_str = t.dtype;
        const dtype = TensorDtype.fromString(dtype_str);
        const numel = @as(usize, @intCast(H));
        const data_size = t.data_offsets.items[1] - t.data_offsets.items[0];
        const data_offset = t.data_offsets.items[0];

        final_norm = switch (dtype) {
            .bf16 => try loadBf16TensorFromMemory(file_data, hdr_size, data_offset, data_size, numel, allocator),
            else => blk: {
                log.warn("final_norm has unexpected dtype '{s}'", .{dtype_str});
                const ones = try allocator.alloc(f32, numel);
                @memset(ones, 1.0);
                break :blk ones;
            },
        };
        log.info("Final norm: {d} floats", .{final_norm.len});
    } else {
        log.warn("model.norm.weight not found; filling with ones", .{});
        final_norm = try allocator.alloc(f32, @as(usize, @intCast(H)));
        @memset(final_norm, 1.0);
    }

    // ── 10. Load per-layer input norms ──
    log.info("Loading input norms for {d} layers...", .{config.NC});
    const in_norm = try loadLayerNormsFromMemory(file_data, hdr_size, tensors, "input_layernorm", config.NC, H, allocator);

    // ── 11. Load per-layer post-attention norms ──
    log.info("Loading post-attention norms for {d} layers...", .{config.NC});
    const pa_norm = try loadLayerNormsFromMemory(file_data, hdr_size, tensors, "post_attention_layernorm", config.NC, H, allocator);

    // ── 11b. Load per-layer QK-norm weights (Qwen3 architecture) ──
    const NH = config.NH;
    const NKV = config.NKV;
    const HD = config.HD;
    const IM = config.IM;
    log.info("Loading QK-norm weights for {d} layers...", .{config.NC});
    const q_norm = try loadLayerNormsFromMemory(file_data, hdr_size, tensors, "self_attn.q_norm", config.NC, HD, allocator);
    const k_norm = try loadLayerNormsFromMemory(file_data, hdr_size, tensors, "self_attn.k_norm", config.NC, HD, allocator);

    // ── 12. Load per-layer projection weights (dequantized f32, for CPU GEMV) ──
    log.info("Loading per-layer projection weights for {d} layers (CPU dequant)...", .{config.NC});
    const q_weight = try loadLayerProjWeightsFromMemory(file_data, hdr_size, tensors, "self_attn.q_proj", config.NC, NH * HD, H, allocator);
    const k_weight = try loadLayerProjWeightsFromMemory(file_data, hdr_size, tensors, "self_attn.k_proj", config.NC, NKV * HD, H, allocator);
    const v_weight = try loadLayerProjWeightsFromMemory(file_data, hdr_size, tensors, "self_attn.v_proj", config.NC, NKV * HD, H, allocator);
    const o_weight = try loadLayerProjWeightsFromMemory(file_data, hdr_size, tensors, "self_attn.o_proj", config.NC, H, NH * HD, allocator);
    const gate_weight = try loadLayerProjWeightsFromMemory(file_data, hdr_size, tensors, "mlp.gate_proj", config.NC, IM, H, allocator);
    const up_weight = try loadLayerProjWeightsFromMemory(file_data, hdr_size, tensors, "mlp.up_proj", config.NC, IM, H, allocator);
    const down_weight = try loadLayerProjWeightsFromMemory(file_data, hdr_size, tensors, "mlp.down_proj", config.NC, H, IM, allocator);

    log.info("Model load complete", .{});

    return ModelData{
        .config = config,
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
        .q_weight = q_weight,
        .k_weight = k_weight,
        .v_weight = v_weight,
        .o_weight = o_weight,
        .gate_weight = gate_weight,
        .up_weight = up_weight,
        .down_weight = down_weight,
    };
}

// ── Unit tests ───────────────────────────────────────────────

test "bf16ToF32 roundtrip" {
    const test_values = [_]f32{ 0.0, 1.0, -1.0, 3.140625, 0.5, -0.25, 42.0 };

    for (test_values) |orig| {
        const bf16 = f32ToBf16(orig);
        const roundtrip = bf16ToF32(bf16);
        if (@abs(orig) > 1e-10) {
            const rel_err = @abs(roundtrip - orig) / @abs(orig);
            try std.testing.expect(rel_err < 0.01);
        }
    }
}

test "bf16ToF32 known values" {
    try std.testing.expectApproxEqAbs(@as(f32, 3.125), bf16ToF32(0x4048), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), bf16ToF32(0x3F80), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), bf16ToF32(0x0000), 0.0001);
}

test "getDefaultConfig qwen3_0_6b" {
    const cfg = getDefaultConfig("qwen3_0_6b") orelse @panic("missing default config");
    try std.testing.expectEqual(@as(u32, 1024), cfg.H);
    try std.testing.expectEqual(@as(u32, 28), cfg.NC);
    try std.testing.expectEqual(@as(u32, 16), cfg.NH);
    try std.testing.expectEqual(@as(u32, 2), cfg.NKV);
    try std.testing.expectEqual(@as(u32, 128), cfg.HD);
    try std.testing.expectEqual(@as(u32, 4096), cfg.IM);
    try std.testing.expectEqual(@as(u32, 151936), cfg.NV);
    try std.testing.expectEqual(@as(u32, 4096), cfg.max_seq_len);
}

test "getDefaultConfig unknown returns null" {
    try std.testing.expectEqual(@as(?ModelConfig, null), getDefaultConfig("unknown_model"));
}

test "TensorDtype fromString" {
    try std.testing.expectEqual(TensorDtype.bf16, TensorDtype.fromString("BF16"));
    try std.testing.expectEqual(TensorDtype.bf16, TensorDtype.fromString("bf16"));
    try std.testing.expectEqual(TensorDtype.i8_q4nx, TensorDtype.fromString("I8"));
    try std.testing.expectEqual(TensorDtype.i8_q4nx, TensorDtype.fromString("I4"));
    try std.testing.expectEqual(TensorDtype.unknown, TensorDtype.fromString("FP32"));
    try std.testing.expectEqual(TensorDtype.unknown, TensorDtype.fromString(""));
}

test "jsonFindValue simple" {
    const s = "{\"key1\": 42, \"key2\": \"hello\", \"key3\": [1,2,3]}";
    const v1 = jsonFindValue(s, "key1") orelse return error.TestFailed;
    try std.testing.expectEqualStrings("42", v1);

    const v2 = jsonFindValue(s, "key2") orelse return error.TestFailed;
    try std.testing.expectEqualStrings("\"hello\"", v2);

    const v3 = jsonFindValue(s, "key3") orelse return error.TestFailed;
    try std.testing.expectEqualStrings("[1,2,3]", v3);

    try std.testing.expectEqual(@as(?[]const u8, null), jsonFindValue(s, "nonexistent"));
}

test "jsonFindValue nested object" {
    const s = "{\"outer\": {\"inner\": 99, \"data_offsets\": [0, 100]}, \"other\": 42}";
    const v = jsonFindValue(s, "outer") orelse return error.TestFailed;
    try std.testing.expect(v[0] == '{');
    const inner_v = jsonFindValue(v, "data_offsets") orelse return error.TestFailed;
    try std.testing.expectEqualStrings("[0, 100]", inner_v);
}

test "jsonUnquote" {
    try std.testing.expectEqualStrings("hello", jsonUnquote("\"hello\""));
    try std.testing.expectEqualStrings("", jsonUnquote("\"\""));
    try std.testing.expectEqualStrings("plain", jsonUnquote("plain"));
    try std.testing.expectEqualStrings("\"partial", jsonUnquote("\"partial"));
}

test "parseJsonInts u32" {
    var result = try parseJsonInts(std.testing.allocator, u32, "[1, 2, 3]");
    defer result.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 3), result.items.len);
    try std.testing.expectEqual(@as(u32, 1), result.items[0]);
    try std.testing.expectEqual(@as(u32, 2), result.items[1]);
    try std.testing.expectEqual(@as(u32, 3), result.items[2]);
}

test "parseJsonInts empty array" {
    var result = try parseJsonInts(std.testing.allocator, u32, "[]");
    defer result.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), result.items.len);
}

test "parseJsonInts single value" {
    var result = try parseJsonInts(std.testing.allocator, u32, "[42]");
    defer result.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 42), result.items[0]);
}

test "parseJsonInts u64" {
    var result = try parseJsonInts(std.testing.allocator, u64, "[0, 12800]");
    defer result.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 2), result.items.len);
    try std.testing.expectEqual(@as(u64, 0), result.items[0]);
    try std.testing.expectEqual(@as(u64, 12800), result.items[1]);
}

test "parseJsonHeader single tensor" {
    const json_str =
        \\{
        \\  "model.embed_tokens.weight": {
        \\    "dtype": "BF16",
        \\    "shape": [100, 64],
        \\    "data_offsets": [0, 12800]
        \\  }
        \\}
    ;
    var result = try parseJsonHeader(std.testing.allocator, json_str);
    defer {
        for (result.items) |*t| {
            std.testing.allocator.free(t.name);
            std.testing.allocator.free(t.dtype);
            t.shape.deinit(std.testing.allocator);
            t.data_offsets.deinit(std.testing.allocator);
        }
        result.deinit(std.testing.allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), result.items.len);
    try std.testing.expectEqualStrings("model.embed_tokens.weight", result.items[0].name);
    try std.testing.expectEqualStrings("BF16", result.items[0].dtype);
    try std.testing.expectEqual(@as(usize, 2), result.items[0].shape.items.len);
    try std.testing.expectEqual(@as(u32, 100), result.items[0].shape.items[0]);
    try std.testing.expectEqual(@as(u32, 64), result.items[0].shape.items[1]);
    try std.testing.expectEqual(@as(usize, 2), result.items[0].data_offsets.items.len);
    try std.testing.expectEqual(@as(u64, 0), result.items[0].data_offsets.items[0]);
    try std.testing.expectEqual(@as(u64, 12800), result.items[0].data_offsets.items[1]);
}

test "parseJsonHeader multiple tensors" {
    const json_str =
        \\{
        \\  "model.embed_tokens.weight": {
        \\    "dtype": "BF16",
        \\    "shape": [100, 64],
        \\    "data_offsets": [0, 12800]
        \\  },
        \\  "model.layers.0.input_layernorm.weight": {
        \\    "dtype": "BF16",
        \\    "shape": [64],
        \\    "data_offsets": [12800, 12928]
        \\  },
        \\  "model.norm.weight": {
        \\    "dtype": "BF16",
        \\    "shape": [64],
        \\    "data_offsets": [12928, 13056]
        \\  }
        \\}
    ;
    var result = try parseJsonHeader(std.testing.allocator, json_str);
    defer {
        for (result.items) |*t| {
            std.testing.allocator.free(t.name);
            std.testing.allocator.free(t.dtype);
            t.shape.deinit(std.testing.allocator);
            t.data_offsets.deinit(std.testing.allocator);
        }
        result.deinit(std.testing.allocator);
    }

    try std.testing.expectEqual(@as(usize, 3), result.items.len);
    try std.testing.expectEqualStrings("model.embed_tokens.weight", result.items[0].name);
    try std.testing.expectEqualStrings("model.layers.0.input_layernorm.weight", result.items[1].name);
    try std.testing.expectEqualStrings("model.norm.weight", result.items[2].name);
}

test "parseJsonHeader skips non-tensor keys" {
    const json_str =
        \\{
        \\  "metadata": {"format": "Q4NX", "version": 1},
        \\  "model.embed_tokens.weight": {
        \\    "dtype": "BF16",
        \\    "shape": [100, 64],
        \\    "data_offsets": [0, 12800]
        \\  }
        \\}
    ;
    var result = try parseJsonHeader(std.testing.allocator, json_str);
    defer {
        for (result.items) |*t| {
            std.testing.allocator.free(t.name);
            std.testing.allocator.free(t.dtype);
            t.shape.deinit(std.testing.allocator);
            t.data_offsets.deinit(std.testing.allocator);
        }
        result.deinit(std.testing.allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), result.items.len);
    try std.testing.expectEqualStrings("model.embed_tokens.weight", result.items[0].name);
}

test "countLayers" {
    const allocator = std.testing.allocator;
    var list: std.ArrayListUnmanaged(TensorDesc) = .empty;
    defer {
        for (list.items) |*t| {
            allocator.free(t.name);
            allocator.free(t.dtype);
            t.shape.deinit(allocator);
            t.data_offsets.deinit(allocator);
        }
        list.deinit(allocator);
    }

    const names = [_][]const u8{
        "model.layers.0.input_layernorm.weight",
        "model.layers.5.input_layernorm.weight",
        "model.layers.27.input_layernorm.weight",
        "model.embed_tokens.weight",
    };

    for (names) |n| {
        try list.append(allocator, .{
            .name = try allocator.dupe(u8, n),
            .dtype = try allocator.dupe(u8, "BF16"),
            .shape = .empty,
            .data_offsets = .empty,
        });
    }

    const count = countLayers(list.items);
    try std.testing.expectEqual(@as(u32, 28), count);
}

test "deriveConfig from qwen3_0_6b tensors" {
    const allocator = std.testing.allocator;
    var list: std.ArrayListUnmanaged(TensorDesc) = .empty;
    defer {
        for (list.items) |*t| {
            allocator.free(t.name);
            allocator.free(t.dtype);
            t.shape.deinit(allocator);
            t.data_offsets.deinit(allocator);
        }
        list.deinit(allocator);
    }

    {
        var shape: std.ArrayListUnmanaged(u32) = .empty;
        try shape.append(allocator, 151936);
        try shape.append(allocator, 1536);
        try list.append(allocator, .{
            .name = try allocator.dupe(u8, "model.embed_tokens.weight"),
            .dtype = try allocator.dupe(u8, "BF16"),
            .shape = shape,
            .data_offsets = .empty,
        });
    }

    {
        var shape: std.ArrayListUnmanaged(u32) = .empty;
        try shape.append(allocator, 288);
        try shape.append(allocator, 5120);
        try list.append(allocator, .{
            .name = try allocator.dupe(u8, "model.layers.0.self_attn.q_proj.weight"),
            .dtype = try allocator.dupe(u8, "I8"),
            .shape = shape,
            .data_offsets = .empty,
        });
    }

    {
        var shape: std.ArrayListUnmanaged(u32) = .empty;
        try shape.append(allocator, 48);
        try shape.append(allocator, 5120);
        try list.append(allocator, .{
            .name = try allocator.dupe(u8, "model.layers.0.self_attn.k_proj.weight"),
            .dtype = try allocator.dupe(u8, "I8"),
            .shape = shape,
            .data_offsets = .empty,
        });
    }

    {
        var shape: std.ArrayListUnmanaged(u32) = .empty;
        try shape.append(allocator, 768);
        try shape.append(allocator, 5120);
        try list.append(allocator, .{
            .name = try allocator.dupe(u8, "model.layers.0.mlp.gate_proj.weight"),
            .dtype = try allocator.dupe(u8, "I8"),
            .shape = shape,
            .data_offsets = .empty,
        });
    }

    {
        var shape: std.ArrayListUnmanaged(u32) = .empty;
        try shape.append(allocator, 1536);
        try list.append(allocator, .{
            .name = try allocator.dupe(u8, "model.layers.0.input_layernorm.weight"),
            .dtype = try allocator.dupe(u8, "BF16"),
            .shape = shape,
            .data_offsets = .empty,
        });
    }
    {
        var shape: std.ArrayListUnmanaged(u32) = .empty;
        try shape.append(allocator, 1536);
        try list.append(allocator, .{
            .name = try allocator.dupe(u8, "model.layers.27.input_layernorm.weight"),
            .dtype = try allocator.dupe(u8, "BF16"),
            .shape = shape,
            .data_offsets = .empty,
        });
    }

    const cfg = try deriveConfig(list.items, "qwen3_0_6b");
    try std.testing.expectEqual(@as(u32, 1024), cfg.H);
    try std.testing.expectEqual(@as(u32, 28), cfg.NC);
    try std.testing.expectEqual(@as(u32, 16), cfg.NH);
    try std.testing.expectEqual(@as(u32, 2), cfg.NKV);
    try std.testing.expectEqual(@as(u32, 128), cfg.HD);
    try std.testing.expectEqual(@as(u32, 4096), cfg.IM);
    try std.testing.expectEqual(@as(u32, 151936), cfg.NV);
    try std.testing.expectEqual(@as(u32, 4096), cfg.max_seq_len);
}

test "dequantizeI8Block basic" {
    var block: [5120]u8 = undefined;
    @memset(&block, 0);

    const bf16_one: u16 = 0x3F80;
    const scales_arr = [_]u16{bf16_one} ** 256;
    const scale_bytes = std.mem.sliceAsBytes(&scales_arr);
    @memcpy(block[0..512], scale_bytes);

    @memset(block[512..1024], 0);

    // 0x11 has both nibbles = 1
    @memset(block[1024..5120], 0x11);

    var output: [32 * 256]f32 = undefined;
    dequantizeI8Block(&block, &output, 0, 0, 32, 256);

    for (output) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), v, 0.001);
    }
}

test "dequantizeI8Block with scale" {
    var block: [5120]u8 = undefined;
    @memset(&block, 0);

    // 0.5 and 0.25 -- must stay under the plausible_max=1.0 sanitization
    // bound (real lm_head scales top out around p99=0.013; anything near
    // 1.0 or above is corruption, see dequantizeI8Block's comment) or this
    // test would get zeroed out by the sanitizer it isn't testing.
    const bf16_half: u16 = 0x3F00;
    const scales_arr = [_]u16{bf16_half} ** 256;
    const scale_bytes = std.mem.sliceAsBytes(&scales_arr);
    @memcpy(block[0..512], scale_bytes);

    const bf16_quarter: u16 = 0x3E80;
    const zp_arr = [_]u16{bf16_quarter} ** 256;
    const zp_bytes = std.mem.sliceAsBytes(&zp_arr);
    @memcpy(block[512..1024], zp_bytes);

    // 0x33 has both nibbles = 3
    @memset(block[1024..5120], 0x33);

    var output: [32 * 256]f32 = undefined;
    dequantizeI8Block(&block, &output, 0, 0, 32, 256);

    // 3 * 0.5 + 0.25 = 1.75
    for (output) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.75), v, 0.001);
    }
}

test "dequantizeI8Block sanitizes a NaN-pattern scale instead of propagating NaN" {
    var block: [5120]u8 = undefined;
    @memset(&block, 0);

    const bf16_one: u16 = 0x3F80;
    var scales_arr = [_]u16{bf16_one} ** 256;
    // Row lr=2, group g=0: real-world corrupt bf16 pattern observed in a
    // production model.q4nx lm_head tensor (exponent all-1s => NaN).
    scales_arr[0 * 32 + 2] = 0xFF9F;
    const scale_bytes = std.mem.sliceAsBytes(&scales_arr);
    @memcpy(block[0..512], scale_bytes);

    @memset(block[512..1024], 0);

    // 0x11 has both nibbles = 1
    @memset(block[1024..5120], 0x11);

    var output: [32 * 256]f32 = undefined;
    dequantizeI8Block(&block, &output, 0, 0, 32, 256);

    for (output) |v| {
        try std.testing.expect(std.math.isFinite(v));
    }
    // Row 2, group 0 (columns 0..31) had its scale zeroed out; every other
    // group/row is untouched and still dequantizes to 1.0.
    for (0..32) |c| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[2 * 256 + c], 0.001);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[3 * 256 + 0], 0.001);
}

test "dequantizeI8Block clamps a finite-but-implausibly-huge scale (overflow-to-Inf case)" {
    var block: [5120]u8 = undefined;
    @memset(&block, 0);

    const bf16_one: u16 = 0x3F80;
    var scales_arr = [_]u16{bf16_one} ** 256;
    // Row lr=5, group g=0: a finite bf16 value (~1.7e38, near bf16/f32 max) --
    // not NaN/Inf on its own, but `nibble * scale` overflows f32 to Inf. This
    // is the second corruption signature found in a real model.q4nx's
    // per-layer q_proj weights (isFinite() alone did not catch it).
    scales_arr[0 * 32 + 5] = 0x7F00;
    const scale_bytes = std.mem.sliceAsBytes(&scales_arr);
    @memcpy(block[0..512], scale_bytes);

    @memset(block[512..1024], 0);

    // 0x11 has both nibbles = 1
    @memset(block[1024..5120], 0x11);

    var output: [32 * 256]f32 = undefined;
    dequantizeI8Block(&block, &output, 0, 0, 32, 256);

    for (output) |v| {
        try std.testing.expect(std.math.isFinite(v));
    }
    for (0..32) |c| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[5 * 256 + c], 0.001);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[3 * 256 + 0], 0.001);
}

test "dequantizeI8TensorFromMemory reconstructs a multi-tile row-major tensor" {
    const allocator = std.testing.allocator;

    // 8-byte file preamble (magic + flags) + two 5120-byte tiles, laid out
    // row-major (tr=0,tc=0 then tr=0,tc=1), matching quantize_i8_tiled().
    var file_data: [8 + 2 * 5120]u8 = undefined;
    @memset(&file_data, 0);

    // Tile (tr=0, tc=0): scale=1, zp=0, nibble=1 everywhere -> dequantized value 1.0
    {
        const block = file_data[8 .. 8 + 5120];
        const scales_arr = [_]u16{0x3F80} ** 256;
        @memcpy(block[0..512], std.mem.sliceAsBytes(&scales_arr));
        @memset(block[512..1024], 0);
        @memset(block[1024..5120], 0x11);
    }
    // Tile (tr=0, tc=1): scale=1, zp=0, nibble=2 everywhere -> dequantized value 2.0
    {
        const block = file_data[8 + 5120 .. 8 + 2 * 5120];
        const scales_arr = [_]u16{0x3F80} ** 256;
        @memcpy(block[0..512], std.mem.sliceAsBytes(&scales_arr));
        @memset(block[512..1024], 0);
        @memset(block[1024..5120], 0x22);
    }

    const result = try dequantizeI8TensorFromMemory(&file_data, 0, 0, 2 * 5120, 32, 512, allocator);
    defer allocator.free(result);

    for (0..32) |r| {
        for (0..256) |c| {
            try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[r * 512 + c], 0.001);
        }
        for (256..512) |c| {
            try std.testing.expectApproxEqAbs(@as(f32, 2.0), result[r * 512 + c], 0.001);
        }
    }
}

test "tileCols" {
    try std.testing.expectEqual(@as(u32, 6), tileCols(1024));
    try std.testing.expectEqual(@as(u32, 1), tileCols(256));
    try std.testing.expectEqual(@as(u32, 1), tileCols(1));
    try std.testing.expectEqual(@as(u32, 2), tileCols(257));
}

test "ModelData deinit clears struct" {
    const allocator = std.testing.allocator;

    var md = ModelData{
        .config = ModelConfig{
            .H = 64, .NC = 2, .NH = 4, .NKV = 1, .HD = 16, .IM = 256, .NV = 1000, .max_seq_len = 512,
        },
        .emb_f32 = try allocator.alloc(f32, 64000),
        .lm_head_f32 = &.{},
        .tied_embeddings = true,
        .final_norm = try allocator.alloc(f32, 64),
        .in_norm = try allocator.alloc([]f32, 2),
        .pa_norm = try allocator.alloc([]f32, 2),
        .q_norm = &.{},
        .k_norm = &.{},
        .rope_sin = try allocator.alloc(f32, 8192),
        .rope_cos = try allocator.alloc(f32, 8192),
        .q_weight = &.{},
        .k_weight = &.{},
        .v_weight = &.{},
        .o_weight = &.{},
        .gate_weight = &.{},
        .up_weight = &.{},
        .down_weight = &.{},
    };
    md.in_norm[0] = try allocator.alloc(f32, 64);
    md.in_norm[1] = try allocator.alloc(f32, 64);
    md.pa_norm[0] = try allocator.alloc(f32, 64);
    md.pa_norm[1] = try allocator.alloc(f32, 64);

    md.deinit(allocator);
}

test "computeRopeTables basic" {
    const allocator = std.testing.allocator;
    const result = try computeRopeTables(allocator, 4, 2);
    defer allocator.free(result.sin_table);
    defer allocator.free(result.cos_table);

    try std.testing.expectEqual(@as(usize, 8), result.sin_table.len);
    try std.testing.expectEqual(@as(usize, 8), result.cos_table.len);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result.sin_table[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result.sin_table[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result.cos_table[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result.cos_table[1], 0.001);

    try std.testing.expectApproxEqAbs(@as(f32, 0.8415), result.sin_table[4], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5403), result.cos_table[4], 0.01);
}

test "computeRopeTables layout matches applyRoPE's flat per-frequency read pattern" {
    // Regression test for a bug where the table was written as interleaved-
    // duplicate pairs (table[2i] == table[2i+1] == freq_i) but applyRoPE in
    // fused_execute.zig reads it as a flat array (table[d] should equal
    // freq_d directly, for d in 0..head_dim/2). With head_dim=4 the previous
    // test's indices (0, 1, 4) all happened to land on frequency i=0 in both
    // the correct and the buggy layout, so it never caught this. Use
    // head_dim=8 (4 distinct frequencies) and check index 2 specifically,
    // which is where the two layouts first disagree: correct layout has
    // table[2] == freq_2; the old buggy layout had table[2] == freq_1.
    const allocator = std.testing.allocator;
    const head_dim: u32 = 8;
    const result = try computeRopeTables(allocator, head_dim, 2);
    defer allocator.free(result.sin_table);
    defer allocator.free(result.cos_table);

    const pos: f32 = 1.0;
    const base: f32 = 10000.0;

    var i: u32 = 0;
    while (i < head_dim / 2) : (i += 1) {
        const theta = pos * std.math.pow(f32, base, -2.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim)));
        const expected_sin = @sin(theta);
        const expected_cos = @cos(theta);
        // pos=1 block starts at index head_dim (=8) in the flat table.
        try std.testing.expectApproxEqAbs(expected_sin, result.sin_table[head_dim + i], 0.0001);
        try std.testing.expectApproxEqAbs(expected_cos, result.cos_table[head_dim + i], 0.0001);
    }
}

test "loadModel with synthetic small file" {
    const allocator = std.testing.allocator;

    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const tmp_path = "/tmp/model_data_test_small.q4nx";
    // File created by external setup script; not deleted here to allow re-runs

    var model = try loadModel(allocator, io, tmp_path, "qwen3_0_6b");
    defer model.deinit(allocator);

    // embed_tokens shape [100, 64] overrides NV and H
    try std.testing.expectEqual(@as(u32, 100), model.config.NV);
    try std.testing.expectEqual(@as(u32, 64), model.config.H);
    // Other config values come from qwen3_0_6b defaults
    // File has layers 0 and 1 → countLayers returns 2
    try std.testing.expectEqual(@as(u32, 2), model.config.NC);
    try std.testing.expectEqual(@as(u32, 12), model.config.NH);
    try std.testing.expectEqual(@as(u32, 2), model.config.NKV);
    try std.testing.expectEqual(@as(u32, 128), model.config.HD);
    try std.testing.expectEqual(@as(u32, 4096), model.config.IM);
    try std.testing.expectEqual(@as(u32, 4096), model.config.max_seq_len);

    try std.testing.expectEqual(@as(usize, 6400), model.emb_f32.len); // 100 * 64

    try std.testing.expectEqual(@as(usize, 6400), model.emb_f32.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), model.emb_f32[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), model.emb_f32[model.emb_f32.len - 1], 0.001);

    try std.testing.expectEqual(false, model.tied_embeddings);
    try std.testing.expectEqual(@as(usize, 6400), model.lm_head_f32.len);

    try std.testing.expectEqual(@as(usize, 64), model.final_norm.len);

    // NC=2 because file only has layers 0 and 1 norms
    try std.testing.expectEqual(@as(usize, 2), model.in_norm.len);
    try std.testing.expectEqual(@as(usize, 2), model.pa_norm.len);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), model.in_norm[0][0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), model.pa_norm[0][0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), model.in_norm[1][0], 0.001);

    try std.testing.expectEqual(@as(usize, 4096 * 128), model.rope_sin.len);
    try std.testing.expectEqual(@as(usize, 4096 * 128), model.rope_cos.len);
}

test "loadModel tied embeddings" {
    const allocator = std.testing.allocator;

    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const tmp_path = "/tmp/model_data_test_tied.q4nx";
    // File created by external setup script; not deleted here

    var model = try loadModel(allocator, io, tmp_path, "qwen3_0_6b");
    defer model.deinit(allocator);

    try std.testing.expectEqual(true, model.tied_embeddings);
    try std.testing.expectEqual(@as(usize, 0), model.lm_head_f32.len);
}

test "loadModel returns error on nonexistent file" {
    const allocator = std.testing.allocator;
    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    try std.testing.expectError(error.FileNotFound, loadModel(allocator, io, "/nonexistent/path/model.q4nx", "qwen3_0_6b"));
}

test "loadModel returns error on invalid magic" {
    const allocator = std.testing.allocator;

    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const tmp_path = "/tmp/model_data_test_invalid.q4nx";
    // File created by external setup script; not deleted here

    try std.testing.expectError(error.InvalidMagic, loadModel(allocator, io, tmp_path, "qwen3_0_6b"));
}
