//! HTTP server for the fused NPU+GPU inference engine.
//! Exposes REST API endpoints for model inference, proxying to the
//! FLM (Fast Language Model) backend for standard paths and routing
//! fused:// prefixed requests to the local FusedEngine.
//!
//! Architecture:
//!   ┌──────────┐    /v1/chat/completions (fused://*)   ┌─────────────┐
//!   │  Client  │ ────────────────────────────────────▶ │             │
//!   │          │    /v1/chat/completions (other)        │  FusedEngine│
//!   │          │ ────────────────────────────────────▶ │  (NPU+GPU)  │
//!   │          │    /v1/completions, /v1/models         │             │
//!   │          │ ────────────────────────────────────▶ │ ────┐       │
//!   │          │                                        │     │      │
//!   │          │    ┌──────────────┐   TCP 52625        │  FlmProxy  │
//!   │          │    │ FLM Backend  │ ◀───────────────── │ ◀───┘       │
//!   │          │    │ (upstream)   │                    └─────────────┘
//!   └──────────┘    └──────────────┘
//!
//! @section Fused Engine

const std = @import("std");
const engine = @import("engine.zig");
const dispatcher = @import("dispatcher.zig");
const flm_proxy = @import("flm_proxy.zig");

const FusedEngine = engine.FusedEngine;
const Connection = engine.Connection;
const ServerConfig = engine.ServerConfig;
const DispatchPolicy = dispatcher.DispatchPolicy;
const FlmProxy = flm_proxy.FlmProxy;

const log = std.log.scoped(.fusion_server);

/// Maximum request body size we'll accept (16 MB).
const MAX_BODY_SIZE: usize = 16 * 1024 * 1024;

/// Maximum header line length.
const MAX_HEADER_LINE: usize = 8192;

/// A parsed HTTP/1.1 request.
pub const ParsedRequest = struct {
    method: []const u8,
    path: []const u8,
    headers: std.StringArrayHashMap([]const u8),
    body: []const u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ParsedRequest) void {
        self.headers.deinit();
    }
};

/// HTTP response status code.
pub const StatusCode = enum(u16) {
    ok = 200,
    created = 201,
    no_content = 204,
    bad_request = 400,
    unauthorized = 401,
    not_found = 404,
    method_not_allowed = 405,
    request_too_large = 413,
    internal_server_error = 500,
    bad_gateway = 502,
    service_unavailable = 503,
    gateway_timeout = 504,

    pub fn reasonPhrase(self: StatusCode) []const u8 {
        return switch (self) {
            .ok => "OK",
            .created => "Created",
            .no_content => "No Content",
            .bad_request => "Bad Request",
            .unauthorized => "Unauthorized",
            .not_found => "Not Found",
            .method_not_allowed => "Method Not Allowed",
            .request_too_large => "Request Entity Too Large",
            .internal_server_error => "Internal Server Error",
            .bad_gateway => "Bad Gateway",
            .service_unavailable => "Service Unavailable",
            .gateway_timeout => "Gateway Timeout",
        };
    }
};

/// Parse an HTTP/1.1 request from the reader.
/// Caller owns the returned ParsedRequest (must call deinit).
pub fn parseRequest(reader: anytype, allocator: std.mem.Allocator) !ParsedRequest {
    // Read request line
    const request_line = (try reader.readUntilDelimiterOrEofAlloc(allocator, '\n', MAX_HEADER_LINE)) orelse {
        return error.ConnectionClosed;
    };
    defer allocator.free(request_line);
    const trimmed_line = std.mem.trimRight(u8, request_line, "\r\n");

    // Parse "METHOD /path HTTP/1.1"
    var parts = std.mem.splitScalar(u8, trimmed_line, ' ');
    const method = parts.next() orelse return error.InvalidRequest;
    const path = parts.next() orelse return error.InvalidRequest;
    const version = parts.next() orelse return error.InvalidRequest;

    // Validate HTTP version
    if (!std.mem.eql(u8, version, "HTTP/1.1") and !std.mem.eql(u8, version, "HTTP/1.0")) {
        return error.UnsupportedHttpVersion;
    }

    // Parse headers
    var headers = std.StringArrayHashMap([]const u8).init(allocator);
    errdefer headers.deinit();

    var content_length: usize = 0;

    while (true) {
        const header_line = reader.readUntilDelimiterAlloc(allocator, '\n', MAX_HEADER_LINE) catch |err| {
            if (err == error.EndOfStream) break;
            return err;
        };
        defer allocator.free(header_line);

        const trimmed_header = std.mem.trimRight(u8, header_line, "\r\n");

        // End of headers
        if (trimmed_header.len == 0) break;

        // Parse "Key: Value"
        if (std.mem.indexOfScalar(u8, trimmed_header, ':')) |colon_pos| {
            const key = std.mem.trim(u8, trimmed_header[0..colon_pos], " ");
            const value = std.mem.trim(u8, trimmed_header[colon_pos + 1 ..], " ");

            // Store lowercased version for case-insensitive lookup
            const key_lower = try allocator.alloc(u8, key.len);
            for (key, 0..) |c, i| {
                key_lower[i] = std.ascii.toLower(c);
            }
            const value_copy = try allocator.alloc(u8, value.len);
            @memcpy(value_copy, value);

            try headers.put(key_lower, value_copy);

            // Track Content-Length
            if (std.mem.eql(u8, key_lower, "content-length")) {
                content_length = std.fmt.parseInt(usize, value, 10) catch {
                    log.warn("invalid Content-Length: {s}", .{value});
                    return error.InvalidContentLength;
                };
                if (content_length > MAX_BODY_SIZE) {
                    log.warn("request body too large: {d} bytes (max {d})", .{ content_length, MAX_BODY_SIZE });
                    return error.RequestTooLarge;
                }
            }
        }
    }

    // 100-continue is noted but not handled in this parse-only phase
    // The caller is responsible for sending the 100 Continue response if needed

    // Read body
    const body = if (content_length > 0) blk: {
        const buf = try allocator.alloc(u8, content_length);
        const n = try reader.readAll(buf);
        break :blk buf[0..n];
    } else if (std.mem.eql(u8, method, "POST") or std.mem.eql(u8, method, "PUT") or
        std.mem.eql(u8, method, "PATCH"))
    {
        // Try to read chunked or remaining body (best effort)
        // For chunked transfer encoding we'd need more parsing — for now, empty body
        const buf = try allocator.alloc(u8, 0);
        break :blk buf;
    } else blk: {
        const buf = try allocator.alloc(u8, 0);
        break :blk buf;
    };

    return ParsedRequest{
        .method = try allocator.dupe(u8, method),
        .path = try allocator.dupe(u8, path),
        .headers = headers,
        .body = body,
        .allocator = allocator,
    };
}

/// Build an HTTP/1.1 response string.
/// Returns a single contiguous buffer that the caller must free.
pub fn formatResponse(
    allocator: std.mem.Allocator,
    status: StatusCode,
    content_type: []const u8,
    body: []const u8,
    extra_headers: []const []const u8,
) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    errdefer buf.deinit();

    const writer = buf.writer();

    // Status line
    try writer.print("HTTP/1.1 {d} {s}\r\n", .{
        @intFromEnum(status),
        status.reasonPhrase(),
    });

    // Standard headers
    try writer.print("Content-Type: {s}\r\n", .{content_type});
    try writer.print("Content-Length: {d}\r\n", .{body.len});
    try writer.writeAll("Connection: keep-alive\r\n");
    try writer.writeAll("Access-Control-Allow-Origin: *\r\n");
    try writer.writeAll("Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n");
    try writer.writeAll("Access-Control-Allow-Headers: Content-Type, Authorization\r\n");
    try writer.writeAll("Server: 1bit-fusion-server/0.1\r\n");

    // Extra headers (e.g., X-* metadata)
    for (extra_headers) |h| {
        try writer.writeAll(h);
        try writer.writeAll("\r\n");
    }

    // Blank line separating headers from body
    try writer.writeAll("\r\n");

    // Body
    if (body.len > 0) {
        try writer.writeAll(body);
    }

    return buf.items;
}

/// Route and handle a parsed HTTP request, writing the response to the writer.
pub fn handleRequest(
    req: *const ParsedRequest,
    writer: anytype,
    engine_instance: *?FusedEngine,
    proxy: *FlmProxy,
    config: *const ServerConfig,
    allocator: std.mem.Allocator,
) !void {
    // Handle OPTIONS preflight globally
    if (std.mem.eql(u8, req.method, "OPTIONS")) {
        const resp = try formatResponse(allocator, .no_content, "text/plain", "", &.{});
        defer allocator.free(resp);
        try writer.writeAll(resp);
        return;
    }

    // Route based on path
    if (std.mem.eql(u8, req.path, "/health")) {
        try handleHealth(req, writer, config, allocator);
    } else if (std.mem.eql(u8, req.path, "/v1/chat/completions")) {
        try handleChatCompletions(req, writer, engine_instance, proxy, config, allocator);
    } else if (std.mem.eql(u8, req.path, "/v1/completions")) {
        try handleCompletions(req, writer, proxy, allocator);
    } else if (std.mem.eql(u8, req.path, "/v1/models")) {
        try handleModels(req, writer, proxy, allocator);
    } else {
        // 404
        const body = try std.fmt.allocPrint(allocator,
            \\{{"error":"not_found","message":"path not found: {s}"}}
        , .{req.path});
        defer allocator.free(body);

        const resp = try formatResponse(allocator, .not_found, "application/json", body, &.{});
        defer allocator.free(resp);
        try writer.writeAll(resp);
    }
}

/// Handle GET /health — returns server status and dispatch policy.
fn handleHealth(
    req: *const ParsedRequest,
    writer: anytype,
    config: *const ServerConfig,
    allocator: std.mem.Allocator,
) !void {
    _ = req; // unused, health is always a GET

    const policy_str = @tagName(config.dispatch_policy);

    const body = try std.fmt.allocPrint(allocator,
        \\{{"status":"ok","backend":"fusion","policy":"{s}","port":{d},"max_parallel":{d},"total_kv_pages":{d},"model_tag":"{s}"}}
    , .{
        policy_str,
        config.port,
        config.max_parallel,
        config.total_kv_pages,
        config.model_tag,
    });
    defer allocator.free(body);

    const resp = try formatResponse(allocator, .ok, "application/json", body, &.{});
    defer allocator.free(resp);
    try writer.writeAll(resp);

    log.info("health check: policy={s}", .{policy_str});
}

/// Handle POST /v1/chat/completions.
/// If model field starts with "fused://", route to the local FusedEngine.
/// Otherwise proxy to the upstream FLM backend.
fn handleChatCompletions(
    req: *const ParsedRequest,
    writer: anytype,
    engine_instance: *?FusedEngine,
    proxy: *FlmProxy,
    config: *const ServerConfig,
    allocator: std.mem.Allocator,
) !void {
    if (!std.mem.eql(u8, req.method, "POST")) {
        const body = "{\"error\":\"method_not_allowed\",\"message\":\"Use POST\"}";
        const resp = try formatResponse(allocator, .method_not_allowed, "application/json", body, &.{});
        defer allocator.free(resp);
        try writer.writeAll(resp);
        return;
    }

    if (req.body.len == 0) {
        const body = "{\"error\":\"bad_request\",\"message\":\"empty request body\"}";
        const resp = try formatResponse(allocator, .bad_request, "application/json", body, &.{});
        defer allocator.free(resp);
        try writer.writeAll(resp);
        return;
    }

    // Extract the "model" field from the JSON body (simple scan — no full JSON parser)
    const model_name = extractJsonStringField(req.body, "model") orelse blk: {
        // Default model if not specified
        break :blk "default";
    };

    // Check if this is a fused:// model
    if (std.mem.startsWith(u8, model_name, "fused://")) {
        try handleFusedChatCompletions(req, writer, engine_instance, config, model_name, allocator);
    } else {
        // Proxy to FLM backend
        try proxyAndTagResponse(req, writer, proxy, "/v1/chat/completions", req.body, config.dispatch_policy, allocator);
    }
}

/// Handle a fused:// model chat completion request.
/// For now, returns a stub response with dispatch policy metadata.
fn handleFusedChatCompletions(
    req: *const ParsedRequest,
    writer: anytype,
    engine_instance: *?FusedEngine,
    config: *const ServerConfig,
    model_name: []const u8,
    allocator: std.mem.Allocator,
) !void {
    _ = req;
    _ = model_name;

    const policy_str = @tagName(config.dispatch_policy);

    if (engine_instance.*) |_| {
        // Future: call FusedEngine.generate() here
        // For now, return a stub response indicating the fused engine is live
        const body = try std.fmt.allocPrint(allocator,
            \\{{"id":"fused-stub-0000000000","object":"chat.completion","created":{d},"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":"Hello from the fused NPU+GPU engine! Dispatch policy: {s}. FusedEngine inference is active but returning a stub response in this build."}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}},"x_dispatch_policy":"{s}"}}
        , .{
            std.time.timestamp(),
            model_name,
            policy_str,
            policy_str,
        });
        defer allocator.free(body);

        const extra = try std.fmt.allocPrint(allocator, "X-Dispatch-Policy: {s}", .{policy_str});
        defer allocator.free(extra);

        const resp = try formatResponse(allocator, .ok, "application/json", body, &.{extra});
        defer allocator.free(resp);
        try writer.writeAll(resp);

        log.info("fused chat completion (stub): model={s}, policy={s}", .{ model_name, policy_str });
    } else {
        // FusedEngine not initialized
        const body = try std.fmt.allocPrint(allocator,
            \\{{"error":"service_unavailable","message":"FusedEngine not initialized","model":"{s}"}}
        , .{model_name});
        defer allocator.free(body);

        const resp = try formatResponse(allocator, .service_unavailable, "application/json", body, &.{});
        defer allocator.free(resp);
        try writer.writeAll(resp);

        log.warn("fused chat completion rejected: FusedEngine not initialized for model={s}", .{model_name});
    }
}

/// Handle POST /v1/completions — proxy to FLM backend.
fn handleCompletions(
    req: *const ParsedRequest,
    writer: anytype,
    proxy: *FlmProxy,
    allocator: std.mem.Allocator,
) !void {
    if (!std.mem.eql(u8, req.method, "POST")) {
        const body = "{\"error\":\"method_not_allowed\",\"message\":\"Use POST\"}";
        const resp = try formatResponse(allocator, .method_not_allowed, "application/json", body, &.{});
        defer allocator.free(resp);
        try writer.writeAll(resp);
        return;
    }

    try proxyAndTagResponse(req, writer, proxy, "/v1/completions", req.body, .npu_only, allocator);
}

/// Handle GET /v1/models — proxy to FLM backend.
fn handleModels(
    req: *const ParsedRequest,
    writer: anytype,
    proxy: *FlmProxy,
    allocator: std.mem.Allocator,
) !void {
    if (!std.mem.eql(u8, req.method, "GET")) {
        const body = "{\"error\":\"method_not_allowed\",\"message\":\"Use GET\"}";
        const resp = try formatResponse(allocator, .method_not_allowed, "application/json", body, &.{});
        defer allocator.free(resp);
        try writer.writeAll(resp);
        return;
    }

    // Proxy the models request to FLM backend
    const upstream_body = ""; // GET request has no body
    try proxyAndTagResponse(req, writer, proxy, "/v1/models", upstream_body, .npu_only, allocator);
}

/// Proxy a request to the FLM backend and tag the response with dispatch metadata.
fn proxyAndTagResponse(
    req: *const ParsedRequest,
    writer: anytype,
    proxy: *FlmProxy,
    upstream_path: []const u8,
    upstream_body: []const u8,
    policy: DispatchPolicy,
    allocator: std.mem.Allocator,
) !void {
    _ = req;

    // Forward the request to the upstream FLM backend
    const upstream_resp = proxy.request(upstream_path, upstream_body) catch |err| {
        log.err("FLM proxy request failed: path={s} error={s}", .{ upstream_path, @errorName(err) });
        const body = try std.fmt.allocPrint(allocator,
            \\{{"error":"upstream_error","message":"FLM backend unavailable: {s}"}}
        , .{@errorName(err)});
        defer allocator.free(body);
        const resp = try formatResponse(allocator, .bad_gateway, "application/json", body, &.{});
        defer allocator.free(resp);
        try writer.writeAll(resp);
        return;
    };
    defer allocator.free(upstream_resp);

    const policy_str = @tagName(policy);

    // Tag with X-Dispatch-Policy header
    const extra = try std.fmt.allocPrint(allocator, "X-Dispatch-Policy: {s}", .{policy_str});
    defer allocator.free(extra);

    // For chat completions, also inject dispatch policy into the response JSON body
    // We append it as an x_dispatch_policy field
    // This is done by modifying the JSON (simple approach: inject before closing brace)
    const tagged_body = if (std.mem.startsWith(u8, upstream_path, "/v1/chat/completions") or
        std.mem.startsWith(u8, upstream_path, "/v1/completions"))
    blk: {
        // Inject x_dispatch_policy field into the JSON response
        if (upstream_resp.len > 0 and upstream_resp[upstream_resp.len - 1] == '}') {
            const tag = try std.fmt.allocPrint(allocator, ",\"x_dispatch_policy\":\"{s}\"", .{policy_str});
            defer allocator.free(tag);
            const modified = try std.mem.concat(allocator, u8, &[_][]const u8{
                upstream_resp[0 .. upstream_resp.len - 1],
                tag,
                "}",
            });
            break :blk modified;
        }
        break :blk try allocator.dupe(u8, upstream_resp);
    } else blk: {
        break :blk try allocator.dupe(u8, upstream_resp);
    };
    defer allocator.free(tagged_body);

    // Determine Content-Type from upstream response (assume JSON)
    const resp = try formatResponse(allocator, .ok, "application/json", tagged_body, &.{extra});
    defer allocator.free(resp);
    try writer.writeAll(resp);

    log.debug("proxied {s}: {d} bytes, policy={s}", .{ upstream_path, upstream_resp.len, policy_str });
}

/// Extract a string field value from a JSON object.
/// Performs a simple scan for `"fieldName":"value"` — no nested object support.
/// Returns null if the field is not found. The returned slice is a view into `json`.
fn extractJsonStringField(json: []const u8, field_name: []const u8) ?[]const u8 {
    // Search for `"fieldName":"` in the JSON
    var search_start: usize = 0;
    while (search_start < json.len) {
        // Find the field name
        const field_start = std.mem.indexOfPos(u8, json, search_start, field_name) orelse return null;

        // Check that it's preceded by a quote
        if (field_start == 0 or json[field_start - 1] != '"') {
            search_start = field_start + 1;
            continue;
        }

        // Look for `: "` or `:"` after the field name
        const after_field = field_start + field_name.len;
        if (after_field >= json.len) return null;

        // Skip whitespace and colon
        var pos = after_field;
        while (pos < json.len and (json[pos] == ' ' or json[pos] == ':' or json[pos] == '\t')) {
            pos += 1;
        }

        // Expect opening quote
        if (pos >= json.len or json[pos] != '"') {
            search_start = after_field;
            continue;
        }
        pos += 1; // skip opening quote

        // Read until closing quote
        const value_start = pos;
        while (pos < json.len and json[pos] != '"') {
            // Handle escaped quotes
            if (json[pos] == '\\') {
                pos += 2; // skip escaped character
                continue;
            }
            pos += 1;
        }

        if (pos >= json.len) return null;

        return json[value_start..pos];
    }

    return null;
}

/// Run the fusion server.
/// Initializes the FLM proxy, binds a TCP listener, and accepts connections
/// in a loop, handling each request.
pub fn runServer(
    allocator: std.mem.Allocator,
    config: ServerConfig,
) !void {
    // Initialize FLM proxy pointing at the upstream FLM backend
    var proxy = FlmProxy.init(allocator, "127.0.0.1", 52625) catch |err| {
        log.err("failed to initialize FLM proxy: {s}", .{@errorName(err)});
        return error.FlmProxyInitFailed;
    };
    defer proxy.deinit();

    log.info("FLM proxy configured: 127.0.0.1:52625", .{});

    // Initialize FusedEngine (best-effort — may fail gracefully)
    var engine_instance: ?FusedEngine = null;
    if (config.model_path.len > 0) {
        engine_instance = FusedEngine.init(
            allocator,
            config.model_path,
            config.xclbin_dir,
            config.model_tag,
            config.max_parallel,
            config.total_kv_pages,
            config.dispatch_policy,
        ) catch |err| {
            log.warn("FusedEngine init failed (server will proxy only): {s}", .{@errorName(err)});
            engine_instance = null;
        };
    } else {
        log.info("FusedEngine not initialized (no model path) — server will proxy to FLM only", .{});
    }
    defer if (engine_instance) |*e| e.deinit();

    // Bind TCP listener
    const address = std.net.Address.initIp4(.{ 0, 0, 0, 0 }, config.port);
    const listen_backlog: u32 = if (config.max_parallel > 32) 128 else config.max_parallel * 2;

    var server = std.net.StreamServer.init(.{
        .reuse_address = true,
        .reuse_port = false,
        .kernel_backlog = listen_backlog,
    });

    try server.listen(address);
    defer server.deinit();

    log.info("fusion server listening on http://0.0.0.0:{d} (policy={s}, max_parallel={d})", .{
        config.port,
        @tagName(config.dispatch_policy),
        config.max_parallel,
    });

    // Accept loop
    var conn_id: u64 = 0;
    while (true) {
        const conn = server.accept() catch |err| {
            log.err("accept failed: {s}", .{@errorName(err)});
            std.time.sleep(100 * std.time.ns_per_ms);
            continue;
        };
        conn_id += 1;

        const local_conn_id = conn_id;
        log.debug("connection #{d} accepted from {s}", .{ local_conn_id, conn.address });

        // Handle the connection
        handleConnection(
            allocator,
            conn,
            &engine_instance,
            &proxy,
            &config,
            local_conn_id,
        ) catch |err| {
            log.warn("connection #{d} handler error: {s}", .{ local_conn_id, @errorName(err) });
        };
    }
}

/// Handle a single TCP connection — read requests, process them, and send responses.
/// Supports HTTP keep-alive for multiple requests per connection.
fn handleConnection(
    allocator: std.mem.Allocator,
    conn: std.net.Server.Connection,
    engine_instance: *?FusedEngine,
    proxy: *FlmProxy,
    config: *const ServerConfig,
    conn_id: u64,
) !void {
    var connection = Connection.init(conn.stream);
    defer connection.close();

    const reader = connection.reader();
    const writer = connection.writer();

    // Connection-level keep-alive state
    var keep_alive = true;
    var keep_alive_count: u32 = 0;
    const max_keep_alive: u32 = 100;

    while (keep_alive and keep_alive_count < max_keep_alive) {
        keep_alive_count += 1;

        // Use a per-request arena for efficient memory management
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const req_allocator = arena.allocator();

        // Read timeout via non-blocking peek? For now, use blocking reads.
        // A production server would use epoll/kqueue with timeouts.

        // Parse the HTTP request
        var req = parseRequest(reader, req_allocator) catch |err| {
            switch (err) {
                error.ConnectionClosed, error.EndOfStream => {
                    log.debug("connection #{d} closed by client after {d} requests", .{ conn_id, keep_alive_count });
                    return;
                },
                error.RequestTooLarge => {
                    const resp = try formatResponse(
                        req_allocator,
                        .request_too_large,
                        "application/json",
                        "{\"error\":\"request_too_large\",\"message\":\"Request body exceeds maximum size\"}",
                        &.{},
                    );
                    defer req_allocator.free(resp);
                    writer.writeAll(resp) catch {};
                    return;
                },
                error.InvalidRequest, error.InvalidContentLength, error.UnsupportedHttpVersion => {
                    const resp = try formatResponse(
                        req_allocator,
                        .bad_request,
                        "application/json",
                        "{\"error\":\"bad_request\",\"message\":\"Invalid HTTP request\"}",
                        &.{},
                    );
                    defer req_allocator.free(resp);
                    writer.writeAll(resp) catch {};
                    return;
                },
                else => {
                    log.err("connection #{d} parse error: {s}", .{ conn_id, @errorName(err) });
                    const resp = try formatResponse(
                        req_allocator,
                        .bad_request,
                        "application/json",
                        "{\"error\":\"bad_request\",\"message\":\"Failed to parse request\"}",
                        &.{},
                    );
                    defer req_allocator.free(resp);
                    writer.writeAll(resp) catch {};
                    return;
                },
            }
        };

        // Determine if we should keep the connection alive
        keep_alive = isKeepAlive(&req) and keep_alive_count < max_keep_alive;

        // Handle the request
        handleRequest(&req, writer, engine_instance, proxy, config, req_allocator) catch |err| {
            log.err("connection #{d} request handler error: {s}", .{ conn_id, @errorName(err) });
            // Try to send a 500 error
            const resp = formatResponse(
                req_allocator,
                .internal_server_error,
                "application/json",
                "{\"error\":\"internal_error\",\"message\":\"Server error processing request\"}",
                &.{},
            ) catch {
                return;
            };
            defer req_allocator.free(resp);
            writer.writeAll(resp) catch {};
            return;
        };

        log.debug("connection #{d} request #{d}: {s} {s}", .{
            conn_id, keep_alive_count, req.method, req.path,
        });
    }
}

/// Check if a connection should be kept alive based on the request headers.
fn isKeepAlive(req: *const ParsedRequest) bool {
    // Check Connection header
    if (req.headers.get("connection")) |conn_hdr| {
        if (std.ascii.eqlIgnoreCase(conn_hdr, "close")) {
            return false;
        }
        if (std.ascii.eqlIgnoreCase(conn_hdr, "keep-alive")) {
            return true;
        }
    }
    // HTTP/1.1 defaults to keep-alive; HTTP/1.0 defaults to close
    return true;
}

test "parseRequest - simple GET" {
    const allocator = std.testing.allocator;

    // Build a minimal HTTP request as a stream
    const request_text = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";

    // Use a FixedBufferStream as a reader
    var fbs = std.io.fixedBufferStream(request_text);
    const reader = fbs.reader();

    var req = try parseRequest(reader, allocator);
    defer req.deinit();

    try std.testing.expectEqualStrings("GET", req.method);
    try std.testing.expectEqualStrings("/health", req.path);

    const host = req.headers.get("host") orelse return error.TestFailed;
    try std.testing.expectEqualStrings("localhost", host);
}

test "parseRequest - POST with body" {
    const allocator = std.testing.allocator;

    const body = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    const request_text = try std.fmt.allocPrint(
        allocator,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n{s}",
        .{ body.len, body },
    );
    defer allocator.free(request_text);

    var fbs = std.io.fixedBufferStream(request_text);
    const reader = fbs.reader();

    var req = try parseRequest(reader, allocator);
    defer req.deinit();

    try std.testing.expectEqualStrings("POST", req.method);
    try std.testing.expectEqualStrings("/v1/chat/completions", req.path);

    const content_type = req.headers.get("content-type") orelse return error.TestFailed;
    try std.testing.expectEqualStrings("application/json", content_type);

    try std.testing.expectEqualStrings(body, req.body);
}

test "formatResponse - minimal" {
    const allocator = std.testing.allocator;

    const resp = try formatResponse(allocator, .ok, "application/json", "{\"status\":\"ok\"}", &.{});
    defer allocator.free(resp);

    try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
    try std.testing.expect(std.mem.indexOf(u8, resp, "Content-Type: application/json") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp, "Content-Length: 15") != null);
    try std.testing.expect(std.mem.endsWith(u8, resp, "{\"status\":\"ok\"}"));
}

test "formatResponse - with extra headers" {
    const allocator = std.testing.allocator;

    const extra = [_][]const u8{"X-Dispatch-Policy: ffn_on_npu"};
    const resp = try formatResponse(allocator, .ok, "text/plain", "hello", &extra);
    defer allocator.free(resp);

    try std.testing.expect(std.mem.indexOf(u8, resp, "X-Dispatch-Policy: ffn_on_npu") != null);
}

test "extractJsonStringField - basic" {
    const json = "{\"model\":\"fused://qwen3-0.6b\",\"messages\":[]}";

    const model = extractJsonStringField(json, "model") orelse return error.TestFailed;
    try std.testing.expectEqualStrings("fused://qwen3-0.6b", model);
}

test "extractJsonStringField - not found" {
    const json = "{\"model\":\"test\"}";
    try std.testing.expect(extractJsonStringField(json, "nonexistent") == null);
}

test "extractJsonStringField - with whitespace" {
    const json = "{\"model\" : \"gpt-4\" }";
    const model = extractJsonStringField(json, "model") orelse return error.TestFailed;
    try std.testing.expectEqualStrings("gpt-4", model);
}
