#include "onebit/mcp_clients/http.hpp"

#include "onebit/mcp/jsonrpc.hpp"

#include <httplib.h>

#include <cctype>
#include <cstdlib>
#include <sstream>
#include <utility>

namespace onebit::mcp_clients {

namespace jr = onebit::mcp::jsonrpc;

namespace {

// RFC 7230 token: 1*tchar
// tchar = "!" / "#" / "$" / "%" / "&" / "'" / "*"
//       / "+" / "-" / "." / "^" / "_" / "`" / "|" / "~"
//       / DIGIT / ALPHA
constexpr bool is_tchar(unsigned char c) noexcept
{
    if (std::isalnum(c)) return true;
    switch (c) {
        case '!': case '#': case '$': case '%': case '&': case '\'':
        case '*': case '+': case '-': case '.': case '^': case '_':
        case '`': case '|': case '~':
            return true;
        default:
            return false;
    }
}

struct ParsedUrl {
    std::string scheme;
    std::string host;
    int         port = -1;
    std::string path;  // includes leading slash
};

// Extremely small URL parser sufficient for `http(s)://host[:port]/path...`.
ParsedUrl parse_url(const std::string& url)
{
    ParsedUrl u;
    auto scheme_end = url.find("://");
    if (scheme_end == std::string::npos) {
        throw McpError::protocol("invalid endpoint url (no scheme): " + url);
    }
    u.scheme = url.substr(0, scheme_end);
    auto rest_start = scheme_end + 3;
    auto path_start = url.find('/', rest_start);
    std::string authority = (path_start == std::string::npos)
                                ? url.substr(rest_start)
                                : url.substr(rest_start, path_start - rest_start);
    u.path = (path_start == std::string::npos) ? "/" : url.substr(path_start);

    auto colon = authority.find(':');
    if (colon == std::string::npos) {
        u.host = authority;
        u.port = (u.scheme == "https") ? 443 : 80;
    } else {
        u.host = authority.substr(0, colon);
        try {
            u.port = std::stoi(authority.substr(colon + 1));
        } catch (...) {
            throw McpError::protocol("invalid port in endpoint url: " + url);
        }
    }
    return u;
}

} // namespace

bool is_valid_header_name(std::string_view name) noexcept
{
    if (name.empty()) return false;
    for (unsigned char c : name) {
        if (!is_tchar(c)) return false;
    }
    return true;
}

bool is_valid_header_value(std::string_view value) noexcept
{
    for (unsigned char c : value) {
        // Reject CR/LF explicitly to prevent header injection.
        if (c == '\r' || c == '\n' || c == '\0') return false;
    }
    return true;
}

HttpClient::HttpClient(std::string endpoint, std::shared_ptr<HttpTransport> transport)
    : endpoint_(std::move(endpoint)),
      transport_(transport ? std::move(transport)
                           : std::make_shared<HttplibTransport>())
{}

HttpClient& HttpClient::add_header(std::string_view name, std::string_view value)
{
    if (!is_valid_header_name(name)) {
        throw McpError::protocol(std::string("invalid header name: ") + std::string(name));
    }
    if (!is_valid_header_value(value)) {
        throw McpError::protocol(std::string("invalid header value: ") + std::string(value));
    }
    headers_[std::string(name)] = std::string(value);
    return *this;
}

void HttpClient::set_transport(std::shared_ptr<HttpTransport> transport)
{
    transport_ = std::move(transport);
}

std::uint64_t HttpClient::next_id()
{
    return next_id_.fetch_add(1, std::memory_order_relaxed);
}

json HttpClient::round_trip(std::string_view method, const std::optional<json>& params)
{
    const auto id  = next_id();
    json        req = build_request(id, method, params);

    HttpRequest hreq;
    hreq.endpoint = endpoint_;
    hreq.body     = req.dump();
    hreq.headers  = headers_;
    hreq.headers["Accept"]       = "application/json, text/event-stream";
    hreq.headers["Content-Type"] = "application/json";

    HttpResponse hresp = transport_->post(hreq);
    if (hresp.status < 200 || hresp.status >= 300) {
        std::ostringstream oss;
        oss << "http status " << hresp.status;
        throw McpError::protocol(oss.str());
    }

    json body;
    try {
        body = json::parse(hresp.body);
    } catch (const json::parse_error& e) {
        throw McpError::json(std::string("response parse: ") + e.what());
    }

    auto parsed = jr::parse_response(body);
    if (parsed.has_error()) {
        throw McpError::rpc(*parsed.error_code,
                            parsed.error_message.value_or(std::string{}));
    }
    if (!parsed.result) {
        throw McpError::protocol("empty result");
    }
    return *parsed.result;
}

json HttpClient::initialize(std::string_view client_name, std::string_view client_version)
{
    return round_trip("initialize", initialize_params(client_name, client_version));
}

std::vector<Tool> HttpClient::list_tools()
{
    json result = round_trip("tools/list", json::object());
    std::vector<Tool> out;
    if (result.contains("tools") && result.at("tools").is_array()) {
        for (const auto& v : result.at("tools")) {
            out.push_back(v.get<Tool>());
        }
    }
    return out;
}

ToolCallResult HttpClient::call_tool(std::string_view name, json arguments)
{
    json p = {{"name", std::string(name)}, {"arguments", std::move(arguments)}};
    json result = round_trip("tools/call", p);
    return result.get<ToolCallResult>();
}

// --- HttplibTransport ----------------------------------------------------

HttpResponse HttplibTransport::post(const HttpRequest& req)
{
    ParsedUrl u = parse_url(req.endpoint);

    httplib::Headers hh;
    for (const auto& [k, v] : req.headers) {
        hh.emplace(k, v);
    }
    if (hh.find("Content-Type") == hh.end()) {
        hh.emplace("Content-Type", "application/json");
    }
    const std::string content_type = hh.find("Content-Type")->second;

    auto do_post = [&](auto& cli) {
        cli.set_connection_timeout(10);
        cli.set_read_timeout(30);
        return cli.Post(u.path.c_str(), hh, req.body, content_type.c_str());
    };

    httplib::Result res;
    if (u.scheme == "http") {
        httplib::Client cli(u.host, u.port);
        res = do_post(cli);
    } else if (u.scheme == "https") {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        httplib::SSLClient cli(u.host, u.port);
        res = do_post(cli);
#else
        throw McpError::http("https endpoint requires cpp-httplib OpenSSL support; "
                             "rebuild with -DCPPHTTPLIB_OPENSSL_SUPPORT, or use a mock transport");
#endif
    } else {
        throw McpError::protocol("unsupported scheme: " + u.scheme);
    }

    if (!res) {
        throw McpError::http("transport error reaching " + req.endpoint);
    }
    HttpResponse out;
    out.status = res->status;
    out.body   = res->body;
    return out;
}

} // namespace onebit::mcp_clients
