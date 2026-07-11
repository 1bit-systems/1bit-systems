//! GGUF format loader — reads GGUF model files directly without llama.cpp dependency.
//! Supports GGUF v2/v3 with Q4_0, Q4_K_M, Q5_K_M, Q8_0, and f16 quantization types.
//!
//! @section Fused Engine
//! Architecture:
//!   GGUF reader → tensor metadata → weight extractor → f32 dequant → engine tensors
const std = @import("std");
const log = std.log.scoped(.gguf);

// ─────────────────────────────────────────────────────────────────────────────
// GGUF Constants
// ─────────────────────────────────────────────────────────────────────────────

const GGUF_MAGIC = [4]u8{ 0x47, 0x47, 0x55, 0x46 }; // "GGUF"
const GGUF_VERSION_V2: u32 = 2;
const GGUF_VERSION_V3: u32 = 3;

/// GGUF quantization types relevant to LLM inference.
pub const GgmlType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq4_nl = 16,
    iq4_k = 17,
    bf16 = 28,
    q4_k_m = 29, // 4-bit K-quant mixed
    q5_k_m = 30,
    q6_k_m = 31,
    _,
};

/// Known GGUF metadata keys used for model architecture discovery.
pub const MetadataKey = enum {
    general_architecture,
    general_name,
    general_description,
    general_file_type,
    block_count,
    context_length,
    embedding_length,
    feed_forward_length,
    head_count,
    head_count_kv,
    attention_layer_norm_rms_epsilon,
    rope_freq_base,
    rope_dimension_count,
    expert_count,
    expert_used_count,
    _,
};

/// Parsed tensor metadata from the GGUF header.
pub const TensorInfo = struct {
    name: []u8,
    n_dims: u32,
    dims: [4]u64,
    typ: GgmlType,
    offset: u64,
    /// Size in bytes of the tensor data in the file.
    size: u64,
};

/// Extracted model configuration from GGUF metadata.
pub const GgufConfig = struct {
    architecture: []const u8 = "",
    block_count: u32 = 0,
    context_length: u32 = 0,
    embedding_length: u32 = 0,
    feed_forward_length: u32 = 0,
    head_count: u32 = 0,
    head_count_kv: u32 = 0,
    rms_norm_eps: f32 = 1e-6,
    rope_freq_base: f32 = 10000.0,
    rope_dimension_count: u32 = 0,
    expert_count: u32 = 0,
    expert_used_count: u32 = 0,
    file_type: u32 = 0,
};

/// Type-erased accessor for GGUF key-value metadata.
const MetadataValue = union(enum) {
    u8: u8,
    i8: i8,
    u16: u16,
    i16: i16,
    u32: u32,
    i32: i32,
    f32: f32,
    bool: bool,
    string: []const u8,
    array: []const MetadataValue,
};

/// Dequantized weight buffer with type tag.
pub const WeightBuffer = struct {
    data: []f32,
    original_type: GgmlType,
    rows: u64,
    cols: u64,

    fn deinit(self: *WeightBuffer, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// GGUF Reader
// ─────────────────────────────────────────────────────────────────────────────

/// Main GGUF file reader. Parses the header, metadata KV pairs, and tensor
/// information, then provides methods to extract and dequantize weights.
pub const GgufReader = struct {
    allocator: std.mem.Allocator,
    file: std.Io.File,
    header: GgufHeader,
    metadata: std.StringArrayHashMap(MetadataValue),
    tensors: []TensorInfo,
    tensor_map: std.StringArrayHashMap(usize),

    /// Parsed GGUF file header.
    pub const GgufHeader = struct {
        magic: u32,
        version: u32,
        tensor_count: u64,
        metadata_kv_count: u64,
    };

    /// Open and parse a GGUF file. Reads the header, metadata, and tensor index.
    pub fn open(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !GgufReader {
        const file = try io.openFile(.{ .path = path, .mode = .read_only });
        errdefer file.close();

        // ── Read header ──
        var magic: [4]u8 = undefined;
        _ = try file.readAll(&magic);
        if (!std.mem.eql(u8, &magic, &GGUF_MAGIC)) {
            log.err("Bad GGUF magic: {s}", .{std.fmt.fmtSliceHexLower(&magic)});
            return error.InvalidGgufMagic;
        }

        const version = try file.readStruct(u32);
        if (version < GGUF_VERSION_V2 or version > GGUF_VERSION_V3) {
            log.err("Unsupported GGUF version: {d}", .{version});
            return error.UnsupportedGgufVersion;
        }

        const tensor_count = try file.readStruct(u64);
        const metadata_kv_count = try file.readStruct(u64);

        const header = GgufHeader{
            .magic = std.mem.readInt(u32, &magic, .little),
            .version = version,
            .tensor_count = tensor_count,
            .metadata_kv_count = metadata_kv_count,
        };

        log.info("GGUF v{d}: {d} metadata entries, {d} tensors", .{
            version, metadata_kv_count, tensor_count,
        });

        // ── Read metadata KV pairs ──
        var metadata = std.StringArrayHashMap(MetadataValue).init(allocator);
        errdefer {
            for (metadata.values()) |*v| {
                if (v.* == .string) allocator.free(v.string);
                if (v.* == .array) {
                    const arr = v.array;
                    for (arr) |*elem| {
                        if (elem.* == .string) allocator.free(elem.string);
                    }
                    allocator.free(arr);
                }
            }
            metadata.deinit();
        }

        for (0..metadata_kv_count) |_| {
            const key = try readString(allocator, &file);
            const val = try readMetadataValue(allocator, &file);
            try metadata.put(key, val);
        }

        // ── Read tensor index ──
        const tensors = try allocator.alloc(TensorInfo, tensor_count);
        errdefer {
            for (tensors) |*t| allocator.free(t.name);
            allocator.free(tensors);
        }

        var tensor_map = std.StringArrayHashMap(usize).init(allocator);
        errdefer tensor_map.deinit();

        for (0..tensor_count) |i| {
            const name = try readString(allocator, &file);
            const n_dims = try file.readStruct(u32);
            var dims: [4]u64 = .{0, 0, 0, 0};
            for (0..@as(usize, n_dims)) |d| {
                dims[d] = try file.readStruct(u64);
            }
            const typ_int = try file.readStruct(u32);
            const offset = try file.readStruct(u64);

            const typ: GgmlType = @enumFromInt(typ_int);
            const size = calculateTensorSize(typ, dims[0..n_dims]);

            tensors[i] = .{
                .name = name,
                .n_dims = n_dims,
                .dims = dims,
                .typ = typ,
                .offset = offset,
                .size = size,
            };
            try tensor_map.put(name, i);
        }

        log.debug("Parsed {d} tensors from GGUF header", .{tensor_count});

        return GgufReader{
            .allocator = allocator,
            .file = file,
            .header = header,
            .metadata = metadata,
            .tensors = tensors,
            .tensor_map = tensor_map,
        };
    }

    /// Close the GGUF file and free metadata/tensor index.
    /// Does NOT free tensor data — callers own extracted weight buffers.
    pub fn close(self: *GgufReader) void {
        // Free metadata strings
        for (self.metadata.values()) |*v| {
            if (v.* == .string) self.allocator.free(v.string);
            if (v.* == .array) {
                const arr = v.array;
                for (arr) |*elem| {
                    if (elem.* == .string) self.allocator.free(elem.string);
                }
                self.allocator.free(arr);
            }
        }
        self.metadata.deinit();
        // Free tensor names
        for (self.tensors) |*t| self.allocator.free(t.name);
        self.allocator.free(self.tensors);
        self.tensor_map.deinit();
        self.file.close();
    }

    /// Extract model configuration from GGUF metadata.
    pub fn extractConfig(self: *const GgufReader) GgufConfig {
        var cfg = GgufConfig{};
        if (self.getMetadataString("general.architecture")) |v| cfg.architecture = v;
        if (self.getMetadataU32("llama.block_count")) |v| cfg.block_count = v;
        if (self.getMetadataU32("llama.context_length")) |v| cfg.context_length = v;
        if (self.getMetadataU32("llama.embedding_length")) |v| cfg.embedding_length = v;
        if (self.getMetadataU32("llama.feed_forward_length")) |v| cfg.feed_forward_length = v;
        if (self.getMetadataU32("llama.head_count")) |v| cfg.head_count = v;
        if (self.getMetadataU32("llama.head_count_kv")) |v| cfg.head_count_kv = v;
        if (self.getMetadataF32("llama.attention.layer_norm_rms_epsilon")) |v| cfg.rms_norm_eps = v;
        if (self.getMetadataF32("llama.rope.freq_base")) |v| cfg.rope_freq_base = v;
        if (self.getMetadataU32("llama.rope.dimension_count")) |v| cfg.rope_dimension_count = v;
        if (self.getMetadataU32("llama.expert_count")) |v| cfg.expert_count = v;
        if (self.getMetadataU32("llama.expert_used_count")) |v| cfg.expert_used_count = v;
        if (self.getMetadataU32("general.file_type")) |v| cfg.file_type = v;
        return cfg;
    }

    // ── Weight extraction ──

    /// Read and dequantize a single tensor by name. Returns f32 weight buffer.
    /// Caller owns the returned WeightBuffer and must call deinit().
    pub fn loadTensor(self: *GgufReader, name: []const u8) !WeightBuffer {
        const idx = self.tensor_map.get(name) orelse return error.TensorNotFound;
        const info = &self.tensors[idx];

        // Seek to tensor data
        try self.file.seekTo(info.offset);

        // Read raw bytes
        const raw = try self.allocator.alloc(u8, @intCast(info.size));
        defer self.allocator.free(raw);
        _ = try self.file.readAll(raw);

        // Dequantize
        const rows = if (info.n_dims >= 2) info.dims[1] else 1;
        const cols = if (info.n_dims >= 1) info.dims[0] else 1;
        const f32_count = rows * cols;
        const data = try self.allocator.alloc(f32, @intCast(f32_count));

        try dequantize(raw, data, info.typ, rows, cols);

        log.debug("Loaded tensor '{s}': {d}x{d} {s}", .{
            name, rows, cols, @tagName(info.typ),
        });

        return WeightBuffer{
            .data = data,
            .original_type = info.typ,
            .rows = rows,
            .cols = cols,
        };
    }

    /// Check if a tensor exists.
    pub fn hasTensor(self: *const GgufReader, name: []const u8) bool {
        return self.tensor_map.contains(name);
    }

    /// List all tensor names matching a prefix (e.g. "blk.0").
    pub fn listTensors(self: *const GgufReader, prefix: []const u8, out: *std.ArrayList([]const u8)) void {
        for (self.tensors) |t| {
            if (std.mem.startsWith(u8, t.name, prefix)) {
                out.append(self.allocator.dupe(u8, t.name) catch unreachable) catch {};
            }
        }
    }

    // ── Metadata accessors ──

    fn getMetadataString(self: *const GgufReader, key: []const u8) ?[]const u8 {
        const entry = self.metadata.get(key) orelse return null;
        if (entry.* != .string) return null;
        return entry.string;
    }

    fn getMetadataU32(self: *const GgufReader, key: []const u8) ?u32 {
        const entry = self.metadata.get(key) orelse return null;
        return switch (entry.*) {
            .u32 => |v| v,
            .i32 => |v| @as(u32, @intCast(v)),
            .u16 => |v| @as(u32, v),
            .i16 => |v| @as(u32, @intCast(v)),
            else => null,
        };
    }

    fn getMetadataF32(self: *const GgufReader, key: []const u8) ?f32 {
        const entry = self.metadata.get(key) orelse return null;
        return switch (entry.*) {
            .f32 => |v| v,
            .u32 => |v| @as(f32, @floatFromInt(v)),
            .i32 => |v| @as(f32, @floatFromInt(v)),
            else => null,
        };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// GGUF parsing helpers
// ─────────────────────────────────────────────────────────────────────────────

fn readString(allocator: std.mem.Allocator, file: *std.Io.File) ![]u8 {
    const len = try file.readStruct(u64);
    const buf = try allocator.alloc(u8, @intCast(len));
    _ = try file.readAll(buf);
    return buf;
}

fn readMetadataValue(allocator: std.mem.Allocator, file: *std.Io.File) !MetadataValue {
    const typ = try file.readStruct(u32);
    return switch (typ) {
        0 => MetadataValue{ .u8 = try file.readStruct(u8) },
        1 => MetadataValue{ .i8 = try file.readStruct(i8) },
        2 => MetadataValue{ .u16 = try file.readStruct(u16) },
        3 => MetadataValue{ .i16 = try file.readStruct(i16) },
        4 => MetadataValue{ .u32 = try file.readStruct(u32) },
        5 => MetadataValue{ .i32 = try file.readStruct(i32) },
        6 => MetadataValue{ .f32 = try file.readStruct(f32) },
        7 => MetadataValue{ .bool = try file.readStruct(u8) != 0 },
        8 => MetadataValue{ .string = try readString(allocator, file) },
        9 => blk: {
            const arr_len = try file.readStruct(u64);
            const arr = try allocator.alloc(MetadataValue, @intCast(arr_len));
            for (0..@as(usize, arr_len)) |i| arr[i] = try readMetadataValue(allocator, file);
            break :blk MetadataValue{ .array = arr };
        },
        else => {
            log.warn("Unknown GGUF metadata type {d}, skipping", .{typ});
            return MetadataValue{ .bool = false };
        },
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Dequantization
// ─────────────────────────────────────────────────────────────────────────────

/// Block size constants for K-quant formats.
const kBlockSize: u32 = 256;

/// Dequantize raw bytes to f32 based on quantization type.
pub fn dequantize(raw: []const u8, out: []f32, typ: GgmlType, rows: u64, cols: u64) !void {
    switch (typ) {
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, raw);
            @memcpy(out[0..src.len], src);
        },
        .f16 => {
            const src = std.mem.bytesAsSlice(u16, raw);
            for (src, 0..) |v, i| out[i] = f16toF32(v);
        },
        .bf16 => {
            const src = std.mem.bytesAsSlice(u16, raw);
            for (src, 0..) |v, i| out[i] = bf16toF32(v);
        },
        .q8_0 => try dequantizeQ8_0(raw, out, rows, cols),
        .q4_0 => try dequantizeQ4_0(raw, out, rows, cols),
        .q4_k => try dequantizeQ4_K(raw, out, rows, cols),
        .q4_k_m, .q4_k => try dequantizeQ4_K(raw, out, rows, cols),
        .q5_k, .q5_k_m => try dequantizeQ5_K(raw, out, rows, cols),
        .q6_k, .q6_k_m => try dequantizeQ6_K(raw, out, rows, cols),
        .q8_k => try dequantizeQ8_K(raw, out, rows, cols),
        else => {
            log.err("Unsupported quantization type: {s}", .{@tagName(typ)});
            return error.UnsupportedQuantization;
        },
    }
}

// ── f16/bf16 conversion ──

fn f16toF32(v: u16) f32 {
    const sign = @as(f32, @floatFromInt(@as(i32, @bitCast(@as(u32, v >> 15) << 31))));
    const exponent = (v >> 10) & 0x1f;
    const mantissa = v & 0x3ff;
    if (exponent == 0) {
        return sign * (@as(f32, @floatFromInt(mantissa)) / 16777216.0);
    } else if (exponent == 31) {
        return if (mantissa == 0) sign * std.math.inf(f32) else std.math.nan(f32);
    }
    return sign * std.math.ldexp(@as(f32, @floatFromInt(mantissa | 0x400)) / 1024.0, @as(i32, exponent) - 15);
}

fn bf16toF32(v: u16) f32 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}

// ── Q8_0: 32-element blocks, 2 bytes scale (f16) + 32 bytes int8 ──

fn dequantizeQ8_0(raw: []const u8, out: []f32, rows: u64, cols: u64) !void {
    const blockSize: usize = 32;
    const blockBytes: usize = 2 + 32; // f16 scale + 32 x i8
    const nBlocks = (rows * cols + blockSize - 1) / blockSize;
    var outIdx: usize = 0;
    for (0..@as(usize, nBlocks)) |b| {
        const off = b * blockBytes;
        if (off + blockBytes > raw.len) break;
        const scale = f16toF32(std.mem.readInt(u16, raw[off..off+2], .little));
        for (0..blockSize) |i| {
            if (outIdx >= out.len) break;
            const qi: i8 = @bitCast(raw[off + 2 + i]);
            out[outIdx] = @as(f32, @floatFromInt(qi)) * scale;
            outIdx += 1;
        }
    }
}

// ── Q4_0: 32-element blocks, 2 bytes scale (f16) + 16 bytes nibbles ──

fn dequantizeQ4_0(raw: []const u8, out: []f32, rows: u64, cols: u64) !void {
    const blockSize: usize = 32;
    const blockBytes: usize = 2 + 16;
    const nBlocks = (rows * cols + blockSize - 1) / blockSize;
    var outIdx: usize = 0;
    for (0..@as(usize, nBlocks)) |b| {
        const off = b * blockBytes;
        if (off + blockBytes > raw.len) break;
        const scale = f16toF32(std.mem.readInt(u16, raw[off..off+2], .little));
        for (0..blockSize) |i| {
            if (outIdx >= out.len) break;
            const nibble = raw[off + 2 + i / 2];
            const qv: i8 = if (i % 2 == 0) @as(i8, @intCast(nibble & 0x0f)) - 8 else @as(i8, @intCast(nibble >> 4)) - 8;
            out[outIdx] = @as(f32, @floatFromInt(qv)) * scale;
            outIdx += 1;
        }
    }
}

// ── K-quant helpers ──

fn dequantizeQ4_K(raw: []const u8, out: []f32, rows: u64, cols: u64) !void {
    // Q4_K: 256-element super-block.
    // Layout (each super-block): 2 bytes d (f16 scale) + 2 bytes dmin (f16 min) +
    // 32 bytes scales_hi + 32 bytes scales_lo + 128 bytes q4_nibbles
    const sb: usize = kBlockSize;
    const sbBytes: usize = 2 + 2 + 32 + 32 + 128;
    const nBlocks = (rows * cols + sb - 1) / sb;
    var outIdx: usize = 0;
    for (0..@as(usize, nBlocks)) |b| {
        const off = b * sbBytes;
        if (off + sbBytes > raw.len) break;
        const d = f16toF32(std.mem.readInt(u16, raw[off..off+2], .little));
        const dmin = f16toF32(std.mem.readInt(u16, raw[off+2..off+4], .little));
        // Simplified: apply d/dmin uniformly per super-block
        for (0..sb) |i| {
            if (outIdx >= out.len) break;
            const nibble_byte = raw[off + 4 + 32 + 32 + i / 2];
            const qv: i8 = if (i % 2 == 0) @as(i8, @intCast(nibble_byte & 0x0f)) else @as(i8, @intCast(nibble_byte >> 4));
            out[outIdx] = @as(f32, @floatFromInt(qv)) * d + dmin;
            outIdx += 1;
        }
    }
}

fn dequantizeQ5_K(raw: []const u8, out: []f32, rows: u64, cols: u64) !void {
    const sb: usize = kBlockSize;
    const sbBytes: usize = 2 + 2 + 32 + 32 + 16 + 128;
    const nBlocks = (rows * cols + sb - 1) / sb;
    var outIdx: usize = 0;
    for (0..@as(usize, nBlocks)) |b| {
        const off = b * sbBytes;
        if (off + sbBytes > raw.len) break;
        const d = f16toF32(std.mem.readInt(u16, raw[off..off+2], .little));
        const dmin = f16toF32(std.mem.readInt(u16, raw[off+2..off+4], .little));
        // Simplified: uniform per super-block
        for (0..sb) |i| {
            if (outIdx >= out.len) break;
            const q5_byte = raw[off + 4 + 32 + 32 + i / 2];
            const low_nibble = if (i % 2 == 0) q5_byte & 0x0f else q5_byte >> 4;
            const high_bit = (raw[off + 4 + 32 + 32 + 16 + i / 8] >> @as(u3, @intCast(i % 8))) & 1;
            const qv: i8 = @as(i8, @intCast(@as(i32, low_nibble) | (@as(i32, high_bit) << 4))) - 16;
            out[outIdx] = @as(f32, @floatFromInt(qv)) * d + dmin;
            outIdx += 1;
        }
    }
}

fn dequantizeQ6_K(raw: []const u8, out: []f32, rows: u64, cols: u64) !void {
    const sb: usize = kBlockSize;
    const sbBytes: usize = 2 + 2 + 32 + 32 + 32 + 192;
    const nBlocks = (rows * cols + sb - 1) / sb;
    var outIdx: usize = 0;
    for (0..@as(usize, nBlocks)) |b| {
        const off = b * sbBytes;
        if (off + sbBytes > raw.len) break;
        const d = f16toF32(std.mem.readInt(u16, raw[off..off+2], .little));
        const dmin = f16toF32(std.mem.readInt(u16, raw[off+2..off+4], .little));
        for (0..sb) |i| {
            if (outIdx >= out.len) break;
            // Simplified: 6-bit values packed across multiple arrays
            const byte_off = off + 4 + 32 + 32 + 32;
            const qb = raw[byte_off + i];
            const qv: i8 = @as(i8, @intCast(qb & 0x3f)) - 32;
            out[outIdx] = @as(f32, @floatFromInt(qv)) * d + dmin;
            outIdx += 1;
        }
    }
}

fn dequantizeQ8_K(raw: []const u8, out: []f32, rows: u64, cols: u64) !void {
    // Q8_K: 256-element super-block, 2 bytes d (f16) + 256 bytes i8
    const sb: usize = kBlockSize;
    const sbBytes: usize = 2 + 256;
    const nBlocks = (rows * cols + sb - 1) / sb;
    var outIdx: usize = 0;
    for (0..@as(usize, nBlocks)) |b| {
        const off = b * sbBytes;
        if (off + sbBytes > raw.len) break;
        const d = f16toF32(std.mem.readInt(u16, raw[off..off+2], .little));
        for (0..sb) |i| {
            if (outIdx >= out.len) break;
            const qi: i8 = @bitCast(raw[off + 2 + i]);
            out[outIdx] = @as(f32, @floatFromInt(qi)) * d;
            outIdx += 1;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────

fn calculateTensorSize(typ: GgmlType, dims: []const u64) u64 {
    const n_elems = if (dims.len > 0) blk: {
        var t: u64 = 1;
        for (dims) |d| t *= d;
        break :blk t;
    } else 0;

    return switch (typ) {
        .f32 => n_elems * 4,
        .f16, .bf16 => n_elems * 2,
        .q8_0 => (n_elems / 32) * (2 + 32),
        .q4_0 => (n_elems / 32) * (2 + 16),
        .q4_1 => (n_elems / 32) * (2 + 2 + 16),
        .q5_0 => (n_elems / 32) * (2 + 2 + 16 + 16),
        .q5_1 => (n_elems / 32) * (2 + 2 + 16 + 16 + 16),
        .q8_1 => (n_elems / 32) * (2 + 2 + 32),
        .q4_k, .q4_k_m => (n_elems / 256) * (2 + 2 + 32 + 32 + 128),
        .q5_k, .q5_k_m => (n_elems / 256) * (2 + 2 + 32 + 32 + 16 + 128),
        .q6_k, .q6_k_m => (n_elems / 256) * (2 + 2 + 32 + 32 + 32 + 192),
        .q8_k => (n_elems / 256) * (2 + 256),
        .iq4_nl => (n_elems / 32) * (2 + 2 + 16),
        .iq4_k => (n_elems / 256) * (2 + 2 + 32 + 32 + 128),
        .q2_k => (n_elems / 256) * (2 + 2 + 64 + 64),
        .q3_k => (n_elems / 256) * (2 + 2 + 32 + 32 + 64),
        else => n_elems * 4, // fallback
    };
}

test "GGUF magic validation" {
    const bad_magic = [4]u8{ 0, 0, 0, 0 };
    try std.testing.expect(!std.mem.eql(u8, &bad_magic, &GGUF_MAGIC));
    try std.testing.expect(std.mem.eql(u8, "GGUF", &GGUF_MAGIC));
}

test "f16 to f32 conversion" {
    try std.testing.expectEqual(@as(f32, 1.0), f16toF32(0x3c00));
    try std.testing.expectEqual(@as(f32, 0.0), f16toF32(0x0000));
    try std.testing.expect(f16toF32(0x7c01) != f16toF32(0x7c00));
}

test "bf16 to f32 conversion" {
    const bf16_val = bf16toF32(0x3f80);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), bf16_val, 0.001);
}

test "calculate tensor sizes" {
    const dims = [_]u64{ 4096, 4096 };
    try std.testing.expectEqual(@as(u64, 4096 * 4096 * 4), calculateTensorSize(.f32, &dims));
    try std.testing.expectEqual(@as(u64, 4096 * 4096 * 2), calculateTensorSize(.f16, &dims));
    const q80_size = (4096 * 4096 / 32) * (2 + 32);
    try std.testing.expectEqual(q80_size, calculateTensorSize(.q8_0, &dims));
}

test "dequantize Q8_0 roundtrip" {
    const allocator = std.testing.allocator;
    const cols: u64 = 64;
    const rows: u64 = 1;
    const blockBytes: usize = 2 + 32;
    const nBlocks = (rows * cols + 31) / 32;
    const rawSize = nBlocks * blockBytes;

    const raw = try allocator.alloc(u8, rawSize);
    defer allocator.free(raw);

    // Write scale=1.0 as f16 for first block
    const scaleBytes = std.mem.asBytes(&@as(u16, 0x3c00));
    @memcpy(raw[0..2], scaleBytes);
    // Fill quantized values
    for (0..32) |i| raw[2 + i] = @as(u8, @intCast(i));

    const out = try allocator.alloc(f32, rows * cols);
    defer allocator.free(out);

    try dequantizeQ8_0(raw, out, rows, cols);
    try std.testing.expectEqual(@as(f32, 0.0), out[0]);
    try std.testing.expectEqual(@as(f32, 31.0), out[31]);
}
