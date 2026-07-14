// /home/bcloud/engine/fusion/tokenizer.zig
// Zig wrapper for the native C++ BPE tokenizer.
// Calls the C++ tokenizer directly via C ABI (no subprocess).
//
// Usage:
//   var tok = try tokenizer.Tokenizer.init(allocator, "path/to/tokenizer.json");
//   defer tok.deinit();
//   const ids = try tok.encode("Hello, world!");
//   const text = try tok.decode(&.{ 151644, 151645 });
//
// Build integration (build.zig):
//   root_mod.addCSourceFile(.{ .file = b.path("tokenize.cpp"),
//     .flags = &.{"-O3", "-march=native", "-std=c++17"} });
//   root_mod.linkLibCpp();
// ============================================================================

const std = @import("std");

// ── C ABI declarations (from tokenize.cpp) ──────────────────────────────────

const TokenizerHandle = opaque {};

extern fn tokenizer_load(json_path: [*:0]const u8) ?*TokenizerHandle;
extern fn tokenizer_encode(
    tok: *TokenizerHandle,
    text: [*:0]const u8,
    out_ids: [*]i32,
    max_ids: i32,
) i32;
extern fn tokenizer_decode(
    tok: *TokenizerHandle,
    ids: [*]const i32,
    n_ids: i32,
) ?[*:0]u8;
extern fn tokenizer_free(tok: *TokenizerHandle) void;

// ── Zig wrapper ─────────────────────────────────────────────────────────────

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    handle: *TokenizerHandle,

    pub fn init(allocator: std.mem.Allocator, json_path: []const u8) !Tokenizer {
        // Create null-terminated path
        const path_buf = try allocator.alloc(u8, json_path.len + 1);
        defer allocator.free(path_buf);
        @memcpy(path_buf[0..json_path.len], json_path);
        path_buf[json_path.len] = 0;

        const handle = tokenizer_load(@ptrCast(path_buf.ptr)) orelse {
            return error.TokenizerLoadFailed;
        };

        return Tokenizer{
            .allocator = allocator,
            .handle = handle,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        tokenizer_free(self.handle);
    }

    /// Encode text to token IDs. Caller owns the returned slice.
    pub fn encode(self: *Tokenizer, text: []const u8) ![]i32 {
        // Initial capacity: generous over-estimate
        const initial_cap: i32 = @intCast(@max(@as(usize, 256), text.len * 2));

        // Create null-terminated text
        const text_z = try self.allocator.alloc(u8, text.len + 1);
        defer self.allocator.free(text_z);
        @memcpy(text_z[0..text.len], text);
        text_z[text.len] = 0;

        // First attempt
        var ids = try self.allocator.alloc(i32, @intCast(initial_cap));
        errdefer self.allocator.free(ids);

        const n = tokenizer_encode(self.handle, @ptrCast(text_z.ptr), ids.ptr, initial_cap);
        if (n < 0) return error.TokenizeFailed;

        if (n > initial_cap) {
            // Need larger buffer — re-allocate with exact size
            self.allocator.free(ids);
            ids = try self.allocator.alloc(i32, @intCast(n));
            const n2 = tokenizer_encode(self.handle, @ptrCast(text_z.ptr), ids.ptr, n);
            if (n2 < 0) return error.TokenizeFailed;
            return ids[0..@intCast(n2)];
        }

        return ids[0..@intCast(n)];
    }

    /// Decode token IDs back to text. Caller owns the returned string.
    pub fn decode(self: *Tokenizer, ids: []const i32) ![]u8 {
        const result_ptr: [*:0]u8 = tokenizer_decode(
            self.handle,
            ids.ptr,
            @intCast(ids.len),
        ) orelse return error.TokenizeFailed;

        // result_ptr is null-terminated and malloc'd by C code
        var len: usize = 0;
        while (result_ptr[len] != 0) len += 1;

        const result = try self.allocator.alloc(u8, len);
        @memcpy(result, result_ptr[0..len]);
        std.c.free(result_ptr);
        return result;
    }
};

// ── Tests ───────────────────────────────────────────────────────────────────

const model_default = "tokenizer.json";  // override with -m/--model

test "tokenizer encode basic" {
    const alloc = std.testing.allocator;
    var tok = try Tokenizer.init(alloc, model_default);
    defer tok.deinit();

    const ids = try tok.encode("Hello, world!");
    defer alloc.free(ids);

    try std.testing.expect(ids.len == 4);
    try std.testing.expectEqual(@as(i32, 9707), ids[0]); // "Hello"
    try std.testing.expectEqual(@as(i32, 11), ids[1]); // ","
    try std.testing.expectEqual(@as(i32, 1879), ids[2]); // "Ġworld"
    try std.testing.expectEqual(@as(i32, 0), ids[3]); // "!"
}

test "tokenizer round-trip" {
    const alloc = std.testing.allocator;
    var tok = try Tokenizer.init(alloc, model_default);
    defer tok.deinit();

    const ids = try tok.encode("Hello, world!");
    defer alloc.free(ids);

    const text = try tok.decode(ids);
    defer alloc.free(text);

    try std.testing.expectEqualStrings("Hello, world!", text);
}

test "tokenizer special tokens" {
    const alloc = std.testing.allocator;
    var tok = try Tokenizer.init(alloc, model_default);
    defer tok.deinit();

    const ids = try tok.encode("<|im_start|>Hello<|im_end|>");
    defer alloc.free(ids);

    try std.testing.expect(ids.len >= 2);
    try std.testing.expectEqual(@as(i32, 151644), ids[0]); // <|im_start|>
    try std.testing.expectEqual(@as(i32, 151645), ids[ids.len - 1]); // <|im_end|>
}

test "tokenizer empty input" {
    const alloc = std.testing.allocator;
    var tok = try Tokenizer.init(alloc, model_default);
    defer tok.deinit();

    const ids = try tok.encode("");
    defer alloc.free(ids);
    try std.testing.expect(ids.len == 0);
}

test "tokenizer multi-space" {
    const alloc = std.testing.allocator;
    var tok = try Tokenizer.init(alloc, model_default);
    defer tok.deinit();

    const ids = try tok.encode("  world");
    defer alloc.free(ids);

    try std.testing.expectEqual(@as(i32, 220), ids[0]); // 'Ġ'
    try std.testing.expectEqual(@as(i32, 1879), ids[1]); // 'Ġworld'
}

test "tokenizer newline" {
    const alloc = std.testing.allocator;
    var tok = try Tokenizer.init(alloc, model_default);
    defer tok.deinit();

    const ids = try tok.encode("Hello\n\nworld");
    defer alloc.free(ids);

    try std.testing.expectEqual(@as(i32, 9707), ids[0]); // 'Hello'
    try std.testing.expectEqual(@as(i32, 271), ids[1]); // 'ĊĊ' (two newlines)
    try std.testing.expectEqual(@as(i32, 14615), ids[2]); // 'world'
}

test "tokenizer decode added tokens" {
    const alloc = std.testing.allocator;
    var tok = try Tokenizer.init(alloc, model_default);
    defer tok.deinit();

    const text = try tok.decode(&.{ 151644, 8948, 198, 2610, 151645 });
    defer alloc.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "<|im_start|>") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "system") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "<|im_end|>") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "You") != null);
}
