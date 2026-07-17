//! HTTP proxy client for forwarding inference requests to the FLM (Fast LLM) backend.
//! Connects via TCP and speaks HTTP/1.1 directly — no libcurl dependency.
//!
//! Usage:
//! ```zig
//! const proxy = try FlmProxy.init(allocator, .{
//!     .host = "127.0.0.1",
//!     .port = 8080,
//! });
//! defer proxy.deinit();
//!
//! const resp = try proxy.chatCompletions(
//!     \\{"model":"qwen3:0.6b","messages":[{"role":"user","content":"Hello"}],"stream":false}
//! );
//! defer resp.deinit();
//! std.debug.print("status={d} body={s}\n", .{ resp.status, resp.body });
//! ```
//!
//! @section Fused Engine
const std = @import("std");
const Io = std.Io;

const log = std.log.scoped(.flm_proxy);

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// All error conditions that the FLM proxy can produce.
pub const FlmProxyError = error{
    /// Unable to resolve hostname or establish TCP connection.
    ConnectionFailed,
    /// Connection was accepted but the remote peer closed it before the
    /// request could be sent or the response could be read.
    ConnectionClosed,
    /// The HTTP response from the FLM backend is malformed — missing
    /// status line, bad headers, invalid chunk encoding, etc.
    MalformedResponse,
    /// The server returned a Transfer-Encoding we cannot decode.
    UnsupportedTransferEncoding,
    /// The server sent a Content-Length that exceeds the internal buffer limit.
    ResponseTooLarge,
    /// Buffer or allocation error in the HTTP request builder.
    RequestTooLarge,
    /// The FLM backend returned a 5xx status code.
    UpstreamError,
    /// Threading or I/O interruption (e.g. timeout on a non-blocking socket).
    IoInterrupted,
};

/// Convenience alias for external use.
pub const Error = FlmProxyError;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the FLM backend connection.
pub const FlmConfig = struct {
    /// FLM backend hostname or IP address (IPv4 literal only — no DNS resolution yet).
    host: []const u8 = "127.0.0.1",
    /// FLM backend TCP port.
    port: u16 = 8080,
    /// Default model served by this proxy (sent in inference requests
    /// when the client does not specify one).
    default_model: []const u8 = "qwen3:0.6b",
    /// Maximum accepted response body size in bytes (0 = unlimited).
    max_body_bytes: usize = 128 * 1024 * 1024, // 128 MB
};

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

/// A complete HTTP response from the FLM backend.
pub const FlmResponse = struct {
    /// HTTP status code (200, 400, 500, etc.).
    status: u16,
    /// Response body as a slice of the allocator-owned buffer.
    /// Freed by calling `deinit()`.
    body: []const u8,
    /// The allocator that owns the body buffer.
    allocator: std.mem.Allocator,

    /// Release the response body memory.
    pub fn deinit(self: *FlmResponse) void {
        self.allocator.free(self.body);
        self.* = undefined;
    }
};

// ---------------------------------------------------------------------------
// Io handle — thread-local cached
// ----------------------------------------------------------------------------

/// Return a thread-local Io handle for synchronous blocking I/O.
/// Uses `init_single_threaded` which does not spawn any worker threads.
fn getIo() Io {
    var threaded = Io.Threaded.init_single_threaded;
    return threaded.io();
}

// ---------------------------------------------------------------------------
// Proxy
// ---------------------------------------------------------------------------

/// HTTP proxy client for the FLM inference backend.
///
/// Each request method opens a fresh TCP connection to the FLM backend,
/// sends the HTTP request, reads the full response, and returns it as an
/// `FlmResponse`. The caller is responsible for calling `deinit()` on every
/// response to free the body buffer.
pub const FlmProxy = struct {
    allocator: std.mem.Allocator,
    config: FlmConfig,

    /// Create a new FlmProxy.
    ///
    /// `config` is copied shallowly — the host and default_model slices are
    /// borrowed from the caller and must remain valid for the proxy's lifetime.
    pub fn init(allocator: std.mem.Allocator, config: FlmConfig) FlmProxy {
        log.info("proxy initialised → {s}:{d} (model={s})", .{
            config.host, config.port, config.default_model,
        });
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Free resources owned by the proxy.
    /// Currently a no-op since config strings are borrowed; kept for
    /// forward-compatibility.
    pub fn deinit(self: *FlmProxy) void {
        _ = self;
        log.debug("proxy deinitialised", .{});
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// POST `/v1/chat/completions` with the given JSON body.
    ///
    /// The body should be a valid JSON request per the OpenAI chat
    /// completions schema.  Returns the full HTTP response.
    pub fn chatCompletions(self: *FlmProxy, body: []const u8) !FlmResponse {
        log.debug("chat completions — {d} bytes", .{body.len});
        return self.request("POST", "/v1/chat/completions", body);
    }

    /// POST `/v1/completions` with the given JSON body.
    pub fn completions(self: *FlmProxy, body: []const u8) !FlmResponse {
        log.debug("completions — {d} bytes", .{body.len});
        return self.request("POST", "/v1/completions", body);
    }

    /// GET `/v1/models` — list available models from the FLM backend.
    pub fn listModels(self: *FlmProxy) !FlmResponse {
        log.debug("list models", .{});
        return self.request("GET", "/v1/models", "");
    }

    // ------------------------------------------------------------------
    // Internal — single request method
    // ------------------------------------------------------------------

    /// Open a TCP connection, send an HTTP request, read the full
    /// response (including chunked bodies), and return it.
    fn request(self: *FlmProxy, method: []const u8, path: []const u8, body: []const u8) !FlmResponse {
        const io = getIo();

        // 1. Connect
        const stream = self.connect(io) catch |err| {
            log.err("connection to {s}:{d} failed: {s}", .{
                self.config.host, self.config.port, @errorName(err),
            });
            return error.ConnectionFailed;
        };
        defer stream.close(io);

        // 2. Build & send request
        self.sendRequest(io, stream, method, path, body) catch |err| {
            log.err("failed to send request: {s}", .{@errorName(err)});
            return err;
        };

        // 3. Read response
        const resp = self.readResponse(io, stream) catch |err| {
            log.err("failed to read response: {s}", .{@errorName(err)});
            return err;
        };

        log.info("{s} {s} → {d} ({d} bytes body)", .{
            method, path, resp.status, resp.body.len,
        });
        return resp;
    }

    // ------------------------------------------------------------------
    // TCP connection
    // ------------------------------------------------------------------

    /// Resolve host and open a TCP stream to the FLM backend.
    fn connect(self: *FlmProxy, io: Io) !Io.net.Stream {
        const addr4 = Io.net.Ip4Address.parse(self.config.host, self.config.port) catch |err| {
            log.err("address parse failed for '{s}': {s}", .{
                self.config.host, @errorName(err),
            });
            return error.ConnectionFailed;
        };
        const addr = Io.net.IpAddress{ .ip4 = addr4 };

        const stream = Io.net.IpAddress.connect(&addr, io, .{
            .mode = .stream,
        }) catch |err| {
            log.err("TCP connect to {s}:{d} failed: {s}", .{
                self.config.host, self.config.port, @errorName(err),
            });
            return error.ConnectionFailed;
        };

        return stream;
    }

    // ------------------------------------------------------------------
    // HTTP request serialisation
    // ------------------------------------------------------------------

    /// Build an HTTP/1.1 request into a buffer and write it to the stream.
    fn sendRequest(
        self: *FlmProxy,
        io: Io,
        stream: Io.net.Stream,
        method: []const u8,
        path: []const u8,
        body: []const u8,
    ) !void {
        var write_buf: [8192]u8 = undefined;
        var writer = Io.net.Stream.writer(stream, io, &write_buf);

        // Request line
        try writer.interface.print("{s} {s} HTTP/1.1\r\n", .{ method, path });

        // Headers
        try writer.interface.print("Host: {s}:{d}\r\n", .{ self.config.host, self.config.port });
        try writer.interface.writeAll("User-Agent: 1bit-flm-proxy/0.1\r\n");
        try writer.interface.writeAll("Accept: application/json\r\n");
        try writer.interface.writeAll("Connection: close\r\n");

        if (body.len > 0) {
            try writer.interface.writeAll("Content-Type: application/json\r\n");
            try writer.interface.print("Content-Length: {d}\r\n", .{body.len});
        }

        // End of headers (blank line)
        try writer.interface.writeAll("\r\n");

        // Body
        if (body.len > 0) {
            try writer.interface.writeAll(body);
        }

        try writer.interface.flush();
    }

    // ------------------------------------------------------------------
    // HTTP response parsing
    // ------------------------------------------------------------------

    /// Read and parse the full HTTP/1.1 response from the stream.
    ///
    /// Covers:
    ///   - Content-Length response bodies
    ///   - chunked Transfer-Encoding bodies
    ///   - responses with no body (204, 304)
    ///   - no Content-Length / no chunked → read until EOF / close
    fn readResponse(self: *FlmProxy, io: Io, stream: Io.net.Stream) !FlmResponse {
        // Reserve a buffer for buffered line / chunk reading.
        // We use a single buffer for the entire response life-cycle.
        var read_buf: [8192]u8 = undefined;
        var reader = Io.net.Stream.reader(stream, io, &read_buf);

        // --- Status line ---
        const status_line_buf = try self.readLine(&reader);
        const status_str = std.mem.trimEnd(u8, status_line_buf, "\r\n");
        if (status_str.len < 12) {
            self.allocator.free(status_line_buf);
            return error.MalformedResponse;
        }

        const code_part = status_str[9..12]; // "HTTP/1.1 XXX..."
        const status = std.fmt.parseInt(u16, code_part, 10) catch {
            self.allocator.free(status_line_buf);
            return error.MalformedResponse;
        };
        self.allocator.free(status_line_buf);

        // --- Headers ---
        var content_length: ?usize = null;
        var chunked = false;
        var has_body = true;

        while (true) {
            const header_line = try self.readLine(&reader);
            const line = std.mem.trimEnd(u8, header_line, "\r\n");
            if (line.len == 0) {
                self.allocator.free(header_line);
                break; // end of headers
            }

            const colon = std.mem.indexOfScalar(u8, line, ':') orelse {
                self.allocator.free(header_line);
                continue;
            };
            const name_raw = line[0..colon];
            const value = std.mem.trim(u8, line[colon + 1 ..], " \t");

            // Compare header names case-insensitively using a small stack buffer.
            const is_content_length = blk: {
                if (name_raw.len != 14) break :blk false;
                var buf: [14]u8 = undefined;
                _ = std.ascii.lowerString(buf[0..], name_raw);
                break :blk std.mem.eql(u8, buf[0..], "content-length");
            };
            const is_transfer_encoding = blk: {
                if (name_raw.len != 17) break :blk false;
                var buf: [17]u8 = undefined;
                _ = std.ascii.lowerString(buf[0..], name_raw);
                break :blk std.mem.eql(u8, buf[0..], "transfer-encoding");
            };

            if (is_content_length) {
                content_length = std.fmt.parseInt(usize, value, 10) catch {
                    log.warn("malformed Content-Length header: '{s}'", .{value});
                    self.allocator.free(header_line);
                    continue;
                };
            } else if (is_transfer_encoding) {
                // Check for "chunked" in the value (case-insensitive)
                if (value.len >= 7) {
                    var val_buf: [32]u8 = undefined;
                    const val_len = @min(value.len, val_buf.len);
                    @memcpy(val_buf[0..val_len], value[0..val_len]);
                    _ = std.ascii.lowerString(val_buf[0..val_len], val_buf[0..val_len]);
                    if (std.mem.indexOf(u8, val_buf[0..val_len], "chunked") != null) {
                        chunked = true;
                    }
                }
            }

            self.allocator.free(header_line);
        }

        // Some status codes must not carry a body per RFC 7230 §3.3.
        if (status == 204 or status == 304) has_body = false;

        // --- Body ---
        const body = if (!has_body) blk: {
            break :blk try self.allocator.alloc(u8, 0);
        } else if (chunked) blk: {
            break :blk try self.readChunkedBody(&reader);
        } else if (content_length) |cl| blk: {
            if (self.config.max_body_bytes > 0 and cl > self.config.max_body_bytes) {
                log.err("response body too large: {d} > max {d}", .{
                    cl, self.config.max_body_bytes,
                });
                return error.ResponseTooLarge;
            }
            var buf = std.ArrayList(u8).empty;
            defer buf.deinit(self.allocator);
            reader.interface.appendExact(self.allocator, &buf, cl) catch |err| {
                return switch (err) {
                    error.EndOfStream, error.ReadFailed => error.ConnectionClosed,
                    else => error.MalformedResponse,
                };
            };
            break :blk try buf.toOwnedSlice(self.allocator);
        } else blk: {
            // No Content-Length and no chunked → read until EOF.
            break :blk try self.readUntilEof(&reader);
        };

        return FlmResponse{
            .status = status,
            .body = body,
            .allocator = self.allocator,
        };
    }

    // ------------------------------------------------------------------
    // Chunked transfer decoding
    // ------------------------------------------------------------------

    /// Decode an HTTP chunked Transfer-Encoding body.
    ///
    /// Format (RFC 7230 §4.1):
    ///   chunk-size   [ chunk-extension ] CRLF
    ///   chunk-data   CRLF
    ///   last-chunk   = 0 [ chunk-extension ] CRLF
    ///   trailer-part
    ///   CRLF
    fn readChunkedBody(self: *FlmProxy, reader: *Io.net.Stream.Reader) ![]u8 {
        var buf = std.ArrayList(u8).empty;
        errdefer buf.deinit(self.allocator);

        while (true) {
            // Read chunk-size line
            const chunk_line = try self.readLine(reader);
            const trimmed = std.mem.trimEnd(u8, chunk_line, "\r\n");

            // Ignore chunk-extensions after ';'
            const semicolon = std.mem.indexOfScalar(u8, trimmed, ';');
            const hex_str = if (semicolon) |s| trimmed[0..s] else trimmed;

            const chunk_size = std.fmt.parseInt(usize, hex_str, 16) catch {
                self.allocator.free(chunk_line);
                log.err("invalid chunk size: '{s}'", .{trimmed});
                return error.MalformedResponse;
            };
            self.allocator.free(chunk_line);

            if (chunk_size == 0) break; // last chunk

            if (self.config.max_body_bytes > 0 and buf.items.len + chunk_size > self.config.max_body_bytes) {
                return error.ResponseTooLarge;
            }

            // Read chunk data
            reader.interface.appendExact(self.allocator, &buf, chunk_size) catch {
                return error.ConnectionClosed;
            };

            // Consume trailing CRLF
            reader.interface.discardAll(2) catch {
                return error.MalformedResponse;
            };
        }

        // Consume trailer and final CRLF (RFC 7230 §4.1.2)
        while (true) {
            const trailer_line = try self.readLine(reader);
            const line = std.mem.trimEnd(u8, trailer_line, "\r\n");
            const is_empty = line.len == 0;
            self.allocator.free(trailer_line);
            if (is_empty) break;
        }

        return buf.toOwnedSlice(self.allocator);
    }

    // ------------------------------------------------------------------
    // EOF-bounded body reader
    // ------------------------------------------------------------------

    /// Read from the stream until EOF, bounded by max_body_bytes.
    fn readUntilEof(self: *FlmProxy, reader: *Io.net.Stream.Reader) ![]u8 {
        var buf = std.ArrayList(u8).empty;
        defer buf.deinit(self.allocator);

        const limit: Io.Limit = if (self.config.max_body_bytes > 0)
            Io.Limit.limited(self.config.max_body_bytes)
        else
            .nothing;

        reader.interface.appendRemaining(self.allocator, &buf, limit) catch |err| switch (err) {
            error.StreamTooLong => return error.ResponseTooLarge,
            error.ReadFailed => return error.ConnectionClosed,
            else => |e| return e,
        };

        return buf.toOwnedSlice(self.allocator);
    }

    // ------------------------------------------------------------------
    // Line reader helper
    // ------------------------------------------------------------------

    /// Read a single line (terminated by \n) from the stream, returned as an
    /// owned slice allocated with `self.allocator`.
    ///
    /// The caller owns the returned memory.
    fn readLine(self: *FlmProxy, reader: *Io.net.Stream.Reader) ![]u8 {
        var buf = std.ArrayList(u8).empty;
        errdefer buf.deinit(self.allocator);

        while (true) {
            const byte = reader.interface.peekArray(1) catch |err| {
                // If we already have data, return what we've got.
                if (buf.items.len > 0) return buf.toOwnedSlice(self.allocator);
                return switch (err) {
                    error.EndOfStream => error.ConnectionClosed,
                    else => error.MalformedResponse,
                };
            };
            const b = byte[0];
            reader.interface.toss(1);

            buf.append(self.allocator, b) catch return error.RequestTooLarge;
            if (b == '\n') break;

            // Safety: guard against unbounded header lines
            if (buf.items.len > 16384) {
                return error.MalformedResponse;
            }
        }

        return buf.toOwnedSlice(self.allocator);
    }
};

// =========================================================================
// Tests
// =========================================================================

const testing = std.testing;

test "FlmConfig default values" {
    const cfg = FlmConfig{};
    try testing.expectEqual(@as(u16, 8080), cfg.port);
    try testing.expectEqualStrings("127.0.0.1", cfg.host);
    try testing.expectEqualStrings("qwen3:0.6b", cfg.default_model);
}

test "FlmProxy init/deinit round-trip" {
    const allocator = testing.allocator;
    var proxy = FlmProxy.init(allocator, .{
        .host = "localhost",
        .port = 9090,
        .default_model = "test-model",
    });
    defer proxy.deinit();
    try testing.expectEqualStrings("localhost", proxy.config.host);
    try testing.expectEqual(@as(u16, 9090), proxy.config.port);
    try testing.expectEqualStrings("test-model", proxy.config.default_model);
}

test "FlmResponse deinit frees body" {
    const allocator = testing.allocator;
    const body = try allocator.alloc(u8, 10);
    @memset(body, 0);
    var resp = FlmResponse{
        .status = 200,
        .body = body,
        .allocator = allocator,
    };
    resp.deinit();
    // After deinit the struct is set to undefined — just verify no crash.
}

test "Connection to non-existent port returns ConnectionFailed" {
    const allocator = testing.allocator;
    var proxy = FlmProxy.init(allocator, .{ .host = "127.0.0.1", .port = 1 });
    defer proxy.deinit();

    const result = proxy.listModels();
    try testing.expect(result == error.ConnectionFailed);
}

test "FlmProxy all methods return ConnectionFailed on closed port" {
    const allocator = testing.allocator;
    var proxy = FlmProxy.init(allocator, .{
        .host = "127.0.0.1",
        .port = 9999,
    });
    defer proxy.deinit();

    try testing.expect(proxy.chatCompletions("{}") == error.ConnectionFailed);
    try testing.expect(proxy.completions("{}") == error.ConnectionFailed);
    try testing.expect(proxy.listModels() == error.ConnectionFailed);
}

test "FlmProxy error set shape" {
    try testing.expect(@typeInfo(FlmProxyError) == .error_set);
}

test "FlmConfig custom values" {
    const cfg = FlmConfig{
        .host = "10.0.0.1",
        .port = 9090,
        .default_model = "llama3:8b",
        .max_body_bytes = 1024,
    };
    try testing.expectEqualStrings("10.0.0.1", cfg.host);
    try testing.expectEqual(@as(u16, 9090), cfg.port);
    try testing.expectEqualStrings("llama3:8b", cfg.default_model);
    try testing.expectEqual(@as(usize, 1024), cfg.max_body_bytes);
}

test "FlmResponse owns body" {
    const allocator = testing.allocator;
    var resp = FlmResponse{
        .status = 200,
        .body = try allocator.dupe(u8, "OK"),
        .allocator = allocator,
    };
    defer resp.deinit();
    try testing.expectEqualStrings("OK", resp.body);
    try testing.expectEqual(@as(u16, 200), resp.status);
}

test "Ip4Address parse via Io.net" {
    const addr = try Io.net.Ip4Address.parse("192.168.1.1", 8080);
    try testing.expectEqual(@as(u16, 8080), addr.port);
    try testing.expectEqual(@as(u8, 192), addr.bytes[0]);
    try testing.expectEqual(@as(u8, 168), addr.bytes[1]);
    try testing.expectEqual(@as(u8, 1), addr.bytes[2]);
    try testing.expectEqual(@as(u8, 1), addr.bytes[3]);
}

test "FlmConfig max_body_bytes limits" {
    const cfg = FlmConfig{ .max_body_bytes = 0 };
    try testing.expectEqual(@as(usize, 0), cfg.max_body_bytes);
}

test "FlmResponse supports all status codes" {
    const allocator = testing.allocator;
    for ([_]u16{ 200, 201, 204, 304, 400, 404, 500, 503 }) |code| {
        var resp = FlmResponse{
            .status = code,
            .body = "",
            .allocator = allocator,
        };
        defer resp.deinit();
        try testing.expectEqual(code, resp.status);
    }
}
