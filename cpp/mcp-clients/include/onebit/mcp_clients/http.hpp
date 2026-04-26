#pragma once

// Streamable-HTTP transport for MCP 2025-06-18. POSTs JSON-RPC frames
// at a single endpoint and parses a single-JSON response (we do not
// handle the SSE-stream form — covers all currently relevant remote MCP
// servers: GitHub, Semgrep, DeepWiki, Linear, Sentry).
//
// The transport is pluggable via the HttpTransport interface so unit
// tests can mock at the request boundary without opening a port. The
// production transport is HttplibTransport (cpp-httplib backed).

#include "onebit/mcp_clients/error.hpp"
#include "onebit/mcp_clients/protocol.hpp"

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::mcp_clients {

struct HttpRequest {
    std::string                        endpoint;
    std::string                        body;
    std::map<std::string, std::string> headers;
};

struct HttpResponse {
    int         status = 0;
    std::string body;
};

// Pluggable transport. Production code uses HttplibTransport. Unit
// tests use a fake that records the request and returns a canned reply.
class HttpTransport {
public:
    virtual ~HttpTransport() = default;
    virtual HttpResponse post(const HttpRequest& req) = 0;
};

// cpp-httplib backed transport. Parses scheme/host/port/path on each
// call; cheap enough for our caller-side use. Defined out-of-line so
// the public header doesn't drag httplib.h in.
class HttplibTransport : public HttpTransport {
public:
    HttpResponse post(const HttpRequest& req) override;
};

class HttpClient {
public:
    explicit HttpClient(std::string endpoint,
                        std::shared_ptr<HttpTransport> transport = nullptr);

    // Add a header applied to every outgoing request — e.g.
    // "Authorization: Bearer <pat>". Returns *this so callers can chain.
    // Throws McpError(Protocol) on invalid header name/value.
    HttpClient& add_header(std::string_view name, std::string_view value);

    [[nodiscard]] const std::string& endpoint() const noexcept { return endpoint_; }
    [[nodiscard]] const std::map<std::string, std::string>& headers() const noexcept
    {
        return headers_;
    }

    // JSON-RPC round trip. Returns the result payload, or throws.
    json round_trip(std::string_view method, const std::optional<json>& params);

    json initialize(std::string_view client_name, std::string_view client_version);
    [[nodiscard]] std::vector<Tool> list_tools();
    ToolCallResult call_tool(std::string_view name, json arguments);

    // Test seam — replace the underlying transport.
    void set_transport(std::shared_ptr<HttpTransport> transport);

private:
    std::uint64_t next_id();

    std::string                        endpoint_;
    std::shared_ptr<HttpTransport>     transport_;
    std::map<std::string, std::string> headers_;
    std::atomic<std::uint64_t>         next_id_{1};
};

// Validates an HTTP header name (RFC 7230 token chars). Used by add_header
// and exposed for unit tests.
[[nodiscard]] bool is_valid_header_name(std::string_view name) noexcept;
[[nodiscard]] bool is_valid_header_value(std::string_view value) noexcept;

} // namespace onebit::mcp_clients
