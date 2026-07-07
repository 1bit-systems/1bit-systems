//! FLM (FastFlowLM) proxy backend for the fused engine.
//! Forwards inference requests to the running FLM daemon (port 52625)
//! and returns responses with dispatch policy metadata.
//!
//! This is the working inference path while we build true NPU+GPU fusion.
const std = @import("std");

const log = std.log.scoped(.flm_proxy);

/// FLM backend configuration.
pub const FlmConfig = struct {
    /// Host running FLM.
    host: []const u8 = "127.0.0.1",
    /// FLM API port.
    port: u16 = 52625,
    /// Default model ID to use when client doesn't specify one.
    default_model: []const u8 = "qwen3:0.6b",
    /// Connection timeout in ms.
    timeout_ms: u32 = 60000,
};

/// Response from FLM.
pub const FlmResponse = struct {
    status: u16,
    body: []u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *FlmResponse) void {
        self.allocator.free(self.body);
    }
};

/// FLM proxy client.
pub const FlmProxy = struct {
    config: FlmConfig,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: FlmConfig) FlmProxy {
        return .{ .config = config, .allocator = allocator };
    }

    /// Proxy a chat completion request to FLM.
    /// Takes the raw JSON body from the client and forwards it.
    pub fn chatCompletions(self: *FlmProxy, body: []const u8) !FlmResponse {
        const url = try std.fmt.allocPrint(self.allocator, "http://{s}:{d}/v1/chat/completions", .{
            self.config.host, self.config.port,
        });
        defer self.allocator.free(url);

        return self.doRequest(url, body);
    }

    /// Proxy a text completion request to FLM.
    pub fn completions(self: *FlmProxy, body: []const u8) !FlmResponse {
        const url = try std.fmt.allocPrint(self.allocator, "http://{s}:{d}/v1/completions", .{
            self.config.host, self.config.port,
        });
        defer self.allocator.free(url);

        return self.doRequest(url, body);
    }

    /// List models from FLM.
    pub fn listModels(self: *FlmProxy) !FlmResponse {
        const url = try std.fmt.allocPrint(self.allocator, "http://{s}:{d}/v1/models", .{
            self.config.host, self.config.port,
        });
        defer self.allocator.free(url);

        return self.doRequest(url, "");
    }

    /// Perform an HTTP POST request to the FLM API.
    fn doRequest(self: *FlmProxy, url: []const u8, body: []const u8) !FlmResponse {
        // Parse URL
        var url_parts = std.mem.splitScalar(u8, url, ':');
        _ = url_parts.next(); // skip http
        _ = url_parts.next(); // skip //
        const host = self.config.host;
        const port_str = url_parts.next() orelse return error.InvalidUrl;
        const path_start = std.mem.indexOfScalar(u8, port_str, '/') orelse return error.InvalidUrl;
        const port = std.fmt.parseInt(u16, port_str[0..path_start], 10) catch return error.InvalidUrl;
        const path = port_str[path_start..];

        // Resolve hostname
        const host_entries = try std.posix.getaddrinfo(
            host,
            port,
            null,
            .{ .family = std.posix.AF.INET, .sock = std.posix.SOCK.STREAM, .protocol = std.posix.IPPROTO.TCP },
        );
        defer std.posix.freeaddrinfo(host_entries);
        const addr = host_entries.?.addr;
        const sock_addr = @as(*const std.posix.sockaddr, @ptrCast(&addr));

        // Connect socket
        const fd = try std.posix.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, std.posix.IPPROTO.TCP);
        errdefer std.posix.close(fd);

        // Set timeout
        const tv = std.posix.timeval{
            .tv_sec = @intCast(self.config.timeout_ms / 1000),
            .tv_usec = @intCast((self.config.timeout_ms % 1000) * 1000),
        };
        std.posix.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&tv)) catch {};
        std.posix.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.SNDTIMEO, std.mem.asBytes(&tv)) catch {};

        try std.posix.connect(fd, sock_addr, @sizeOf(@TypeOf(addr)));

        // Build HTTP request
        var req_buf = std.ArrayList(u8).init(self.allocator);
        defer req_buf.deinit();

        const len_str = try std.fmt.allocPrint(self.allocator, "{d}", .{body.len});
        defer self.allocator.free(len_str);

        try req_buf.writer().print(
            "POST {s} HTTP/1.1\r\nHost: {s}:{d}\r\nContent-Type: application/json\r\nContent-Length: {s}\r\nConnection: close\r\n\r\n{s}",
            .{ path, host, port, len_str, body },
        );

        // Send request
        const written = try std.posix.write(fd, req_buf.items);
        _ = written;

        // Read response
        var resp_buf = std.ArrayList(u8).init(self.allocator);
        errdefer resp_buf.deinit();

        var buf: [4096]u8 = undefined;
        while (true) {
            const n = std.posix.read(fd, &buf) catch |err| {
                if (err == error.ConnectionResetByPeer or err == error.ConnectionTimedOut or err == error.WouldBlock) {
                    if (resp_buf.items.len == 0) return err;
                    break;
                }
                return err;
            };
            if (n == 0) break;
            try resp_buf.appendSlice(buf[0..n]);
        }

        // Parse response status
        const resp_text = resp_buf.items;
        const status_end = std.mem.indexOfScalar(u8, resp_text, '\r') orelse return error.InvalidResponse;
        const status_line = resp_text[0..status_end];
        var status_parts = std.mem.splitScalar(u8, status_line, ' ');
        _ = status_parts.next(); // HTTP/1.1
        const status_code_str = status_parts.next() orelse return error.InvalidResponse;
        const status_code = std.fmt.parseInt(u16, status_code_str, 10) catch return error.InvalidResponse;

        // Find body (after \r\n\r\n)
        const header_end = std.mem.indexOf(u8, resp_text, "\r\n\r\n") orelse return error.InvalidResponse;
        const body_start = header_end + 4;
        const resp_body = try self.allocator.dupe(u8, resp_text[body_start..]);

        return FlmResponse{
            .status = status_code,
            .body = resp_body,
            .allocator = self.allocator,
        };
    }
};

test "FlmProxy type compiles" {
    const allocator = std.testing.allocator;
    const proxy = FlmProxy.init(allocator, .{});
    _ = proxy;
    try std.testing.expect(true);
}
