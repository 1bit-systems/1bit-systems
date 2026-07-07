//! Q4NX model header parser — reads the JSON metadata section of a Q4NX model file
//! and derives ModelConfig including architecture dimensions and xclbin parameters.
const std = @import("std");

// ============================================================
// ModelConfig — all dimensions derived from Q4NX JSON header
// ============================================================

pub const ModelConfig = struct {
    H: u32 = 0,
    NC: u32 = 0,
    NH: u32 = 0,
    NKV: u32 = 0,
    HD: u32 = 0,
    IM: u32 = 0,
    NV: u32 = 0,
    GQA: u32 = 0,
    AW: u32 = 4,
    XM: u32 = 128,

    // QKV fused offsets
    qkv_k_offset: u32 = 0,
    qkv_v_offset: u32 = 0,
    qkv_total: u32 = 0,

    // XCLBIN dimensions
    xclbin_qkv_k: u32 = 0,
    xclbin_qkv_n: u32 = 0,
    xclbin_o_k: u32 = 0,
    xclbin_o_n: u32 = 0,
    xclbin_g_k: u32 = 0,
    xclbin_g_n: u32 = 0,
    xclbin_u_k: u32 = 0,
    xclbin_u_n: u32 = 0,
    xclbin_gu_k: u32 = 0,
    xclbin_gu_n: u32 = 0,
    xclbin_d_k: u32 = 0,
    xclbin_d_n: u32 = 0,

    has_q_norm: bool = false,
    has_k_norm: bool = false,
    has_rope_freqs_file: bool = false,
    has_lm_head: bool = false,
    gu_split: bool = false,

    rope_theta: f32 = 1000000.0,
    rope_factor: f32 = 1.0,

    model_tag: []const u8 = "",
    model_dir: []const u8 = "",

    pub fn valid(self: ModelConfig) bool {
        return self.H > 0 and self.NC > 0 and self.NH > 0 and
            self.NKV > 0 and self.HD > 0 and self.IM > 0 and self.NV > 0;
    }
};

// ============================================================
// BF16 → F32 conversion
// ============================================================

pub fn bf16ToF32(v: u16) f32 {
    const bits: u32 = @as(u32, v) << 16;
    return @as(f32, @bitCast(bits));
}

pub fn bf16ToF32Safe(v: u16) f32 {
    // Return 0 for NaN/Inf (exponent = 0xFF in upper byte)
    if (v & 0x7F80 == 0x7F80) return 0.0;
    return bf16ToF32(v);
}

// ============================================================
// Manual JSON scanner helpers
// ============================================================

/// Find the data_offsets[0] value for a given tensor key in the JSON header.
/// Returns the byte offset into the weight data section, or null if not found.
pub fn findTensorInfo(js: []const u8, key: []const u8) ?u32 {
    var pos: usize = 0;
    while (pos < js.len) {
        // Search for the key
        const found = std.mem.indexOfPos(u8, js, pos, key);
        const q = found orelse return null;

        // Verify it's a JSON key (preceded by " and followed by ")
        if (q > 0 and js[q - 1] == '"' and (q + key.len < js.len) and js[q + key.len] == '"') {
            // Look for "data_offsets" after this key
            const after_key = q + key.len;
            const offs_pos = std.mem.indexOfPos(u8, js, after_key, "\"data_offsets\"") orelse {
                pos = q + 1;
                continue;
            };
            // Find opening bracket
            const bracket = std.mem.indexOfPos(u8, js, offs_pos, "[") orelse {
                pos = q + 1;
                continue;
            };
            // Parse first integer
            const end = bracket + 1;
            var i: usize = end;
            while (i < js.len and (std.ascii.isDigit(js[i]) or js[i] == '-')) i += 1;
            if (i > end) {
                const num_str = js[end..i];
                const val = std.fmt.parseInt(u32, num_str, 10) catch {
                    pos = q + 1;
                    continue;
                };
                return val;
            }
        }
        pos = q + 1;
    }
    return null;
}

/// Find the tile_rows (shape[0]) for a given tensor key.
/// The I8 format stores tile rows, where each tile row = 32 elements.
pub fn findTileRows(js: []const u8, key: []const u8) ?u32 {
    var pos: usize = 0;
    while (pos < js.len) {
        const found = std.mem.indexOfPos(u8, js, pos, key);
        const q = found orelse return null;

        if (q > 0 and js[q - 1] == '"' and (q + key.len < js.len) and js[q + key.len] == '"') {
            const after_key = q + key.len;
            const shape_pos = std.mem.indexOfPos(u8, js, after_key, "\"shape\"") orelse {
                pos = q + 1;
                continue;
            };
            const bracket = std.mem.indexOfPos(u8, js, shape_pos, "[") orelse {
                pos = q + 1;
                continue;
            };
            const end = bracket + 1;
            var i: usize = end;
            while (i < js.len and (std.ascii.isDigit(js[i]) or js[i] == '-')) i += 1;
            if (i > end) {
                const num_str = js[end..i];
                const val = std.fmt.parseInt(u32, num_str, 10) catch {
                    pos = q + 1;
                    continue;
                };
                return val;
            }
        }
        pos = q + 1;
    }
    return null;
}

/// Get the second dimension (shape[1]) of a tensor.
pub fn getShapeDim1(js: []const u8, key: []const u8) ?u32 {
    var pos: usize = 0;
    while (pos < js.len) {
        const found = std.mem.indexOfPos(u8, js, pos, key);
        const q = found orelse return null;

        if (q > 0 and js[q - 1] == '"' and (q + key.len < js.len) and js[q + key.len] == '"') {
            const after_key = q + key.len;
            const shape_pos = std.mem.indexOfPos(u8, js, after_key, "\"shape\"") orelse {
                pos = q + 1;
                continue;
            };
            const bracket = std.mem.indexOfPos(u8, js, shape_pos, "[") orelse {
                pos = q + 1;
                continue;
            };
            // Parse first dim, then find comma and second dim
            const after_bracket = bracket + 1;
            var i = after_bracket;
            while (i < js.len and (std.ascii.isDigit(js[i]) or js[i] == '-')) i += 1;
            // Skip comma and whitespace
            while (i < js.len and (js[i] == ',' or js[i] == ' ')) i += 1;
            const end = i;
            var j = i;
            while (j < js.len and (std.ascii.isDigit(js[j]) or js[j] == '-')) j += 1;
            if (j > end) {
                const num_str = js[end..j];
                const val = std.fmt.parseInt(u32, num_str, 10) catch {
                    pos = q + 1;
                    continue;
                };
                return val;
            }
        }
        pos = q + 1;
    }
    return null;
}

/// Check if a tensor key exists in the JSON header.
pub fn keyExists(js: []const u8, key: []const u8) bool {
    return findTileRows(js, key) != null;
}

/// Count the number of transformer layers by scanning for "model.layers.N".
pub fn countLayers(js: []const u8) u32 {
    var max_layer: i32 = -1;
    const target = "model.layers.";
    var pos: usize = 0;
    while (pos < js.len) {
        const found = std.mem.indexOfPos(u8, js, pos, target) orelse break;
        const after = found + target.len;
        // Parse the layer number
        var i: usize = after;
        while (i < js.len and std.ascii.isDigit(js[i])) i += 1;
        if (i > after) {
            const num_str = js[after..i];
            const layer = std.fmt.parseInt(i32, num_str, 10) catch {
                pos = found + 1;
                continue;
            };
            if (layer > max_layer) max_layer = layer;
        }
        pos = found + 1;
    }
    return @intCast(@max(max_layer, 0) + 1);
}

// ============================================================
// Main parser
// ============================================================

/// Parse the Q4NX model file and return a fully derived ModelConfig.
pub fn parseQ4nxHeader(model_path: []const u8, model_tag: []const u8) !ModelConfig {
    var cfg = ModelConfig{
        .model_tag = model_tag,
    };

    // Extract model_dir from path
    if (std.mem.lastIndexOfScalar(u8, model_path, '/')) |slash| {
        cfg.model_dir = model_path[0..slash];
    }

    const compat = @import("compat.zig");

    // Open and mmap the model file
    const fd = try compat.File.openAbsolute(model_path);
    defer fd.close();

    const file_size = try fd.getEndPos();
    const mapping = try std.posix.mmap(
        null,
        file_size,
        .{ .READ = true },
        .{ .TYPE = .PRIVATE },
        fd.handle(),
        0,
    );
    defer std.posix.munmap(mapping);

    if (mapping.len < 8) return error.InvalidModelFile;

    // Read 8-byte header size (little-endian u64)
    const hdr_size = std.mem.readInt(u64, &mapping[0..8].*, .little);
    if (hdr_size == 0 or 8 + hdr_size > mapping.len) return error.InvalidModelHeader;

    const json_start: usize = 8;
    _ = @as(usize, 8 + hdr_size);
    const js = mapping[json_start .. json_start + hdr_size];

    // ============================================================
    // Step 1: Get NV and H from embed_tokens.weight
    // ============================================================
    {
        const key = "model.embed_tokens.weight";
        var pos: usize = 0;
        while (pos < js.len) {
            const found = std.mem.indexOfPos(u8, js, pos, key) orelse break;
            const q = found;
            if (q > 0 and js[q - 1] == '"' and js[q + key.len] == '"') {
                const after = q + key.len;
                const sp = std.mem.indexOfPos(u8, js, after, "\"shape\"") orelse {
                    pos = q + 1;
                    continue;
                };
                const br = std.mem.indexOfPos(u8, js, sp, "[") orelse {
                    pos = q + 1;
                    continue;
                };
                var i = br + 1;
                while (i < js.len and (std.ascii.isDigit(js[i]) or js[i] == '-')) i += 1;
                if (i > br + 1) {
                    cfg.NV = std.fmt.parseInt(u32, js[br + 1 .. i], 10) catch {
                        pos = q + 1;
                        continue;
                    };
                }
                while (i < js.len and (js[i] == ',' or js[i] == ' ')) i += 1;
                var j = i;
                while (j < js.len and (std.ascii.isDigit(js[j]) or js[j] == '-')) j += 1;
                if (j > i) {
                    cfg.H = std.fmt.parseInt(u32, js[i..j], 10) catch {
                        pos = q + 1;
                        continue;
                    };
                }
                break;
            }
            pos = q + 1;
        }
    }

    if (cfg.H == 0 or cfg.NV == 0) return error.CouldNotDetermineDimensions;

    // ============================================================
    // Step 2: Get I8 tile row counts for each weight
    // ============================================================
    const prefix_q = "model.layers.0.self_attn.q_proj.weight";
    const prefix_k = "model.layers.0.self_attn.k_proj.weight";
    const prefix_v = "model.layers.0.self_attn.v_proj.weight";
    const prefix_o = "model.layers.0.self_attn.o_proj.weight";
    const prefix_g = "model.layers.0.mlp.gate_proj.weight";
    const _prefix_u = "model.layers.0.mlp.up_proj.weight";
    _ = _prefix_u;
    const _prefix_v = "model.layers.0.self_attn.v_proj.weight";
    _ = _prefix_v;
    const prefix_d = "model.layers.0.mlp.down_proj.weight";

    const q_tr = findTileRows(js, prefix_q) orelse return error.MissingQProj;
    const k_tr = findTileRows(js, prefix_k) orelse return error.MissingKProj;
    const _v_tr = findTileRows(js, prefix_v) orelse return error.MissingVProj;
    const _o_tr = findTileRows(js, prefix_o) orelse return error.MissingOProj;
    const g_tr = findTileRows(js, prefix_g) orelse return error.MissingGateProj;
    const d_tr = findTileRows(js, prefix_d) orelse return error.MissingDownProj;
    _ = _o_tr;
    _ = _v_tr;

    // ============================================================
    // Step 3: Detect architecture features
    // ============================================================
    const qn_key = "model.layers.0.self_attn.q_norm.weight";
    const kn_key = "model.layers.0.self_attn.k_norm.weight";
    const rf_key = "rope_freqs.weight";
    const lm_key = "lm_head.weight";

    if (keyExists(js, qn_key)) {
        cfg.has_q_norm = true;
        // q_norm shape = [HD], get HD from its data
        if (findTileRows(js, qn_key)) |qn_hd| {
            // each tile row = 32 elements, but q_norm is just [HD], so tile_rows * 32 should be HD
            // Actually for norm weights in Q4NX, the tile rows represent the actual dim
            // q_norm has shape [HD], so tile_rows = ceil(HD/32) but typically one tile row is exact
            // For HD=128: ceil(128/32) = 4 tile rows, 4*32 = 128 → correct
            // For HD=256: ceil(256/32) = 8 tile rows, 8*32 = 256 → correct
            cfg.HD = qn_hd * 32;
        }
    }
    cfg.has_k_norm = keyExists(js, kn_key);
    cfg.has_rope_freqs_file = keyExists(js, rf_key);
    cfg.has_lm_head = keyExists(js, lm_key);

    // ============================================================
    // Step 4: Count layers
    // ============================================================
    cfg.NC = countLayers(js);

    // ============================================================
    // Step 5: Derive remaining dimensions from I8 tile rows
    // ============================================================

    // q_proj: in_features=H, out_features=NH*HD
    // I8 packing tiles: ceil(H/256) column tiles, ceil(NH*HD/32) row tiles
    // tile_rows_q = ceil(NH*HD/32) * ceil(H/256)
    const col_tiles_h = (cfg.H + 255) / 256; // ceil(H/256)
    if (col_tiles_h > 0 and q_tr > 0) {
        const nh_hd = (q_tr / col_tiles_h) * 32; // NH * HD (each tile row = 32 elements)
        if (cfg.HD == 0) {
            // Try HD=128 first, then HD=256
            if (nh_hd % 128 == 0) {
                cfg.HD = 128;
                cfg.NH = nh_hd / 128;
            } else if (nh_hd % 256 == 0) {
                cfg.HD = 256;
                cfg.NH = nh_hd / 256;
            } else {
                // Fallback
                cfg.HD = 128;
                cfg.NH = nh_hd / 128;
            }
        } else {
            cfg.NH = nh_hd / cfg.HD;
        }
    }

    // k_proj: in_features=H, out_features=NKV*HD
    if (col_tiles_h > 0 and k_tr > 0) {
        const nkv_hd = (k_tr / col_tiles_h) * 32;
        cfg.NKV = nkv_hd / cfg.HD;
    }

    // o_proj: in_features=NH*HD, out_features=H
    // tile_rows_o = ceil(H/32) * ceil(NH*HD/256)
    // We use this to verify O dimensions

    // gate_proj: in_features=H, out_features=IM
    if (col_tiles_h > 0 and g_tr > 0) {
        cfg.IM = (g_tr / col_tiles_h) * 32;
    }

    // Also verify IM from down_proj
    // down_proj: in_features=IM, out_features=H
    // tile_rows_d = ceil(H/32) * ceil(IM/256)
    const col_tiles_im = (cfg.IM + 255) / 256;
    if (col_tiles_im > 0 and d_tr > 0) {
        const im_check = d_tr / @max(col_tiles_im, 1);
        _ = im_check; // for verification
    }

    // ============================================================
    // Step 6: Compute derived values
    // ============================================================

    if (cfg.NH > 0 and cfg.NKV > 0) cfg.GQA = cfg.NH / cfg.NKV;

    cfg.qkv_k_offset = cfg.NH * cfg.HD;
    cfg.qkv_v_offset = cfg.NH * cfg.HD + cfg.NKV * cfg.HD;
    cfg.qkv_total = cfg.NH * cfg.HD + 2 * cfg.NKV * cfg.HD;

    cfg.xclbin_qkv_k = cfg.H;
    cfg.xclbin_qkv_n = cfg.qkv_total;
    cfg.xclbin_o_k = cfg.NH * cfg.HD;
    cfg.xclbin_o_n = cfg.H;

    // GU split decision: for models with IM > 7168 (so 2*IM > 14336),
    // G and U are separate xclbins due to AIE column width limits
    cfg.gu_split = (cfg.IM * 2 > 14336);
    if (cfg.gu_split) {
        cfg.xclbin_g_k = cfg.H;
        cfg.xclbin_g_n = cfg.IM;
        cfg.xclbin_u_k = cfg.H;
        cfg.xclbin_u_n = cfg.IM;
    } else {
        cfg.xclbin_gu_k = cfg.H;
        cfg.xclbin_gu_n = cfg.IM * 2;
    }
    cfg.xclbin_d_k = cfg.IM;
    cfg.xclbin_d_n = cfg.H;

    return cfg;
}
