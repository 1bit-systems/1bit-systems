//! Unified HTTP server for NPU+GPU fused inference.
//! Proxies inference requests to FLM (working NPU backend at 82 tok/s),
//! adding dispatch policy metadata and support for fused:// routing.
//!
//! When true NPU+GPU layer fusion is ready, the proxy path is replaced
//! with direct FusedEngine.prefill()/decodeStep() calls.
//!
//! @section Fused Engine
const std = @import("std");

const eng = @import("engine.zig");
const dispatcher = @import("dispatcher.zig");
const flm_proxy = @import("flm_proxy.zig");

pub const FusedEngine = eng.FusedEngine;
pub const FlmProxy = flm_proxy.FlmProxy;
pub const FlmConfig = flm_proxy.FlmConfig;
pub const DispatchPolicy = dispatcher.DispatchPolicy;

const log = std.log.scoped(.fusion_server);

/// Parsed HTTP request.
const HttpRequest = struct {
    method: []const u8,
    path: []const u8,
    body: []const u8,
    allocator: std.mem.Allocator,

    fn deinit(self: *HttpRequest) void {
        self.allocator.free(self.body);
    }
};

/// Parse an HTTP request from a connection.
fn parseRequest(conn: *eng.Connection, allocator: std.mem.Allocator) !HttpRequest {
    const reader = conn.reader();

    // Read request line
    const request_line = try reader.readUntilDelimiterAlloc(allocator, '\n', 4096);
    defer allocator.free(request_line);
    const req = std.mem.trim(u8, request_line, "\r\n ");

    var parts = std.mem.splitScalar(u8, req, ' ');
    const method = try allocator.dupe(u8, parts.next() orelse return error.BadRequest);
    const path = try allocator.dupe(u8, parts.next() orelse return error.BadRequest);

    // Read headers
    var content_length: usize = 0;
    while (true) {
        const header_line = try reader.readUntilDelimiterAlloc(allocator, '\n', 4096);
        defer allocator.free(header_line);
        const trimmed = std.mem.trim(u8, header_line, "\r\n ");
        if (trimmed.len == 0) break;

        // Parse Content-Length
        if (std.ascii.indexOfIgnoreCase(trimmed, "content-length:")) |_| {
            const colon = std.mem.indexOfScalar(u8, trimmed, ':') orelse continue;
            const value_str = std.mem.trim(u8, trimmed[colon + 1 ..], " ");
            content_length = std.fmt.parseInt(usize, value_str, 10) catch 0;
        }
    }

    // Read body
    const body = if (content_length > 0 and content_length < 65536)
        try reader.readNoEofAlloc(allocator, content_length)
    else
        try allocator.alloc(u8, 0);

    return HttpRequest{
        .method = method,
        .path = path,
        .body = body,
        .allocator = allocator,
    };
}

/// Format an HTTP response.
fn formatResponse(allocator: std.mem.Allocator, status: u16, content_type: []const u8, body: []const u8) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    const status_text = switch (status) {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        else => "Unknown",
    };
    try buf.writer().print("HTTP/1.1 {d} {s}\r\n", .{ status, status_text });
    try buf.writer().print("Content-Type: {s}\r\n", .{content_type});
    try buf.writer().print("Content-Length: {d}\r\n", .{body.len});
    try buf.writer().print("Access-Control-Allow-Origin: *\r\n", .{});
    try buf.writer().print("Connection: close\r\n", .{});
    try buf.writer().print("\r\n", .{});
    try buf.writer().writeAll(body);
    return buf.items;
}

/// Tag a response with dispatch policy metadata.
fn tagWithPolicy(allocator: std.mem.Allocator, body: []const u8, policy: DispatchPolicy, device: []const u8) ![]u8 {
    // Insert x-dispatch-policy and x-device into the JSON response
    // Simple: prepend a header-like prefix. But for JSON, we add fields.
    // Since we're proxying, just return the body as-is and let the metadata
    // come through the x- headers. For now, no JSON transformation needed.
    _ = allocator;
    _ = policy;
    _ = device;
    return body;
}

/// Handle one HTTP request.
pub fn handleRequest(
    conn: *eng.Connection,
    proxy: *FlmProxy,
    policy: DispatchPolicy,
    allocator: std.mem.Allocator,
) !void {
    const req = parseRequest(conn, allocator) catch |err| {
        log.warn("Failed to parse request: {s}", .{@errorName(err)});
        const resp = try formatResponse(allocator, 400, "application/json", "{\"error\":\"bad request\"}");
        defer allocator.free(resp);
        try conn.writer().writeAll(resp);
        return;
    };
    defer req.deinit();
    defer allocator.free(req.method);
    defer allocator.free(req.path);

    const path = req.path;
    // Route
    if (std.mem.eql(u8, path, "/health")) {
        const resp = try formatResponse(allocator, 200, "application/json",
            "{\"status\":\"ok\",\"backend\":\"npu-flm\",\"dispatch_policy\":\"" ++ @tagName(policy) ++ "\"}");
        defer allocator.free(resp);
        try conn.writer().writeAll(resp);

    } else if (std.mem.eql(u8, path, "/v1/chat/completions")) {
        try handleChatCompletions(conn, proxy, req.body, policy, allocator);

    } else if (std.mem.eql(u8, path, "/v1/completions")) {
        try handleCompletions(conn, proxy, req.body, policy, allocator);

    } else if (std.mem.eql(u8, path, "/v1/models")) {
        try handleModels(conn, proxy, policy, allocator);

    } else {
        const resp = try formatResponse(allocator, 404, "application/json", "{\"error\":\"not found\"}");
        defer allocator.free(resp);
        try conn.writer().writeAll(resp);
    }
}

fn handleChatCompletions(
    conn: *eng.Connection,
    proxy: *FlmProxy,
    body: []const u8,
    policy: DispatchPolicy,
    allocator: std.mem.Allocator,
) !void {
    // Parse the request to extract model name for routing
    var model_name: []const u8 = "qwen3:0.6b";
    if (body.len > 0) {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{ .ignore_unknown_fields = true }) catch null;
        if (parsed) |p| {
            defer p.deinit();
            if (p.value.object.get("model")) |m| {
                model_name = m.string;
            }
        }
    }

    // Determine if we should use FLM or another backend
    const use_flm = !std.mem.startsWith(u8, model_name, "fused://");

    if (use_flm) {
        // Proxy to FLM
        const start = std.time.milliTimestamp();
        const flm_resp = proxy.chatCompletions(body) catch |err| {
            log.err("FLM proxy failed: {s}", .{@errorName(err)});
            const resp = try formatResponse(allocator, 502, "application/json",
                "{\"error\":\"backend unavailable\",\"detail\":\"" ++ @errorName(err) ++ "\"}");
            defer allocator.free(resp);
            try conn.writer().writeAll(resp);
            return;
        };
        defer flm_resp.deinit();
        const elapsed = std.time.milliTimestamp() - start;

        // Tag response with dispatch policy via x-headers
        // For proper JSON: add fields to FLM's response
        // Simple approach: return FLM's response with policy in a wrapper
        const tagged = try tagWithPolicy(allocator, flm_resp.body, policy, "npu-flm");
        defer allocator.free(tagged);

        // Rewrite body to include dispatch metadata
        // Find the closing } and insert metadata
        var resp_body: []const u8 = flm_resp.body;
        if (std.mem.lastIndexOfScalar(u8, resp_body, '}')) |close_brace| {
            const meta = try std.fmt.allocPrint(allocator,
                ",\"x_dispatch_policy\":\"{s}\",\"x_device\":\"npu-flm\",\"x_backend_ms\":{d}",
                .{ @tagName(policy), elapsed },
            );
            defer allocator.free(meta);
            // Insert before closing brace
            const new_body = try std.mem.concat(allocator, u8, &[_][]const u8{
                resp_body[0..close_brace],
                meta,
                "}",
            });
            defer allocator.free(new_body);
            resp_body = new_body;
        }

        const resp = try formatResponse(allocator, flm_resp.status, "application/json", resp_body);
        defer allocator.free(resp);
        try conn.writer().writeAll(resp);
        log.info("Chat completion proxied to FLM: {d}ms, policy={s}", .{ elapsed, @tagName(policy) });
    } else {
        // Fused mode: route through FusedEngine (future)
        const resp = try formatResponse(allocator, 200, "application/json",
            "{\"id\":\"chatcmpl-fused\",\"object\":\"chat.completion\",\"choices\":[" ++
            "{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"Fused engine active. " ++
            "Dispatch policy: " ++ @tagName(policy) ++ "\"},\"finish_reason\":\"stop\"}]," ++
            "\"x_dispatch_policy\":\"" ++ @tagName(policy) ++ "\",\"x_device\":\"fusion\"}");
        defer allocator.free(resp);
        try conn.writer().writeAll(resp);
    }
}

fn handleCompletions(
    conn: *eng.Connection,
    proxy: *FlmProxy,
    body: []const u8,
    policy: DispatchPolicy,
    allocator: std.mem.Allocator,
) !void {
    _ = policy;
    const flm_resp = proxy.completions(body) catch |err| {
        log.err("FLM proxy failed: {s}", .{@errorName(err)});
        const resp = try formatResponse(allocator, 502, "application/json",
            "{\"error\":\"backend unavailable\",\"detail\":\"" ++ @errorName(err) ++ "\"}");
        defer allocator.free(resp);
        try conn.writer().writeAll(resp);
        return;
    };
    defer flm_resp.deinit();

    const resp = try formatResponse(allocator, flm_resp.status, "application/json", flm_resp.body);
    defer allocator.free(resp);
    try conn.writer().writeAll(resp);
}

fn handleModels(
    conn: *eng.Connection,
    proxy: *FlmProxy,
    policy: DispatchPolicy,
    allocator: std.mem.Allocator,
) !void {
    _ = policy;
    const flm_resp = proxy.listModels() catch |err| {
        log.err("FLM proxy failed: {s}", .{@errorName(err)});
        const resp = try formatResponse(allocator, 502, "application/json",
            "{\"error\":\"backend unavailable\",\"detail\":\"" ++ @errorName(err) ++ "\"}");
        defer allocator.free(resp);
        try conn.writer().writeAll(resp);
        return;
    };
    defer flm_resp.deinit();

    const resp = try formatResponse(allocator, flm_resp.status, "application/json", flm_resp.body);
    defer allocator.free(resp);
    try conn.writer().writeAll(resp);
}

/// Run the fused engine HTTP server.
pub fn runServer(config: eng.ServerConfig, allocator: std.mem.Allocator) !void {
    // Initialize FLM proxy
    const flm_config = FlmConfig{
        .host = "127.0.0.1",
        .port = 52625,
        .default_model = "qwen3:0.6b",
    };
    var proxy = FlmProxy.init(allocator, flm_config);

    const policy = config.dispatch_policy;

    log.info("Fused server starting on port {d} (policy: {s}, FLM proxy: {s}:{d})", .{
        config.port, @tagName(policy), flm_config.host, flm_config.port,
    });

    // Bind and listen using std.net
    const address = try std.net.Address.parseIp("127.0.0.1", config.port);
    var listener = try address.listen(.{
        .reuse_address = true,
    });
    defer listener.deinit();

    log.info("Fused server listening on 0.0.0.0:{d}", .{config.port});
    log.info("Dispatch policy: {s} — see dispatcher.zig for description", .{@tagName(policy)});

    // Accept loop
    while (true) {
        var client = listener.accept() catch |err| {
            log.warn("Accept failed: {s}", .{@errorName(err)});
            continue;
        };

        var conn = eng.Connection.init(client.stream);
        handleRequest(&conn, &proxy, policy, allocator) catch |err| {
            log.warn("Request failed: {s}", .{@errorName(err)});
        };
        conn.close();
    }
}
