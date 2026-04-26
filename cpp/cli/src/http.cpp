#include "onebit/cli/http.hpp"

#include <httplib.h>

#include <regex>
#include <string>

namespace onebit::cli {

namespace {

struct ParsedUrl {
    bool        is_https = false;
    std::string host;
    int         port     = 80;
    std::string path     = "/";
};

[[nodiscard]] std::expected<ParsedUrl, Error> parse_url(std::string_view url)
{
    // Pragmatic split — cpp-httplib's Client expects scheme+host, and a
    // path. Avoid pulling in a full URL parser.
    const std::string s(url);
    const std::regex re(R"(^(https?)://([^/:]+)(?::(\d+))?(.*)$)");
    std::smatch m;
    if (!std::regex_match(s, m, re)) {
        return std::unexpected(Error::invalid("malformed URL: " + s));
    }
    ParsedUrl p;
    p.is_https = (m[1].str() == "https");
    p.host     = m[2].str();
    p.port     = m[3].matched ? std::stoi(m[3].str()) : (p.is_https ? 443 : 80);
    p.path     = m[4].matched && !m[4].str().empty() ? m[4].str() : "/";
    return p;
}

class DefaultHttp final : public HttpClient {
public:
    std::expected<HttpResponse, Error>
    get(std::string_view url, std::uint32_t timeout_ms) override
    {
        auto parsed = parse_url(url);
        if (!parsed) return std::unexpected(parsed.error());

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        if (parsed->is_https) {
            httplib::SSLClient cli(parsed->host.c_str(), parsed->port);
            cli.set_connection_timeout(0, static_cast<long>(timeout_ms) * 1000);
            cli.set_read_timeout(0, static_cast<long>(timeout_ms) * 1000);
            cli.enable_server_certificate_verification(false);
            auto res = cli.Get(parsed->path.c_str());
            if (!res) {
                return std::unexpected(Error::network(
                    "GET " + std::string(url) + " failed"));
            }
            return HttpResponse{res->status, res->body};
        }
#else
        if (parsed->is_https) {
            return std::unexpected(Error::network(
                "https requested but cpp-httplib built without OPENSSL"));
        }
#endif
        httplib::Client cli(parsed->host.c_str(), parsed->port);
        cli.set_connection_timeout(0, static_cast<long>(timeout_ms) * 1000);
        cli.set_read_timeout(0, static_cast<long>(timeout_ms) * 1000);
        auto res = cli.Get(parsed->path.c_str());
        if (!res) {
            return std::unexpected(Error::network(
                "GET " + std::string(url) + " failed"));
        }
        return HttpResponse{res->status, res->body};
    }

    std::expected<HttpResponse, Error>
    post_json(std::string_view url, std::string_view body,
              std::uint32_t timeout_ms) override
    {
        auto parsed = parse_url(url);
        if (!parsed) return std::unexpected(parsed.error());
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        if (parsed->is_https) {
            httplib::SSLClient cli(parsed->host.c_str(), parsed->port);
            cli.set_connection_timeout(0, static_cast<long>(timeout_ms) * 1000);
            cli.set_read_timeout(0, static_cast<long>(timeout_ms) * 1000);
            cli.enable_server_certificate_verification(false);
            auto res = cli.Post(parsed->path.c_str(),
                                std::string(body),
                                "application/json");
            if (!res) {
                return std::unexpected(Error::network(
                    "POST " + std::string(url) + " failed"));
            }
            return HttpResponse{res->status, res->body};
        }
#else
        if (parsed->is_https) {
            return std::unexpected(Error::network(
                "https requested but cpp-httplib built without OPENSSL"));
        }
#endif
        httplib::Client cli(parsed->host.c_str(), parsed->port);
        cli.set_connection_timeout(0, static_cast<long>(timeout_ms) * 1000);
        cli.set_read_timeout(0, static_cast<long>(timeout_ms) * 1000);
        auto res = cli.Post(parsed->path.c_str(),
                            std::string(body),
                            "application/json");
        if (!res) {
            return std::unexpected(Error::network(
                "POST " + std::string(url) + " failed"));
        }
        return HttpResponse{res->status, res->body};
    }
};

}  // namespace

HttpClient& default_http_client()
{
    static DefaultHttp g;
    return g;
}

bool healthcheck(HttpClient& http, std::string_view url, std::uint32_t timeout_ms)
{
    if (url.empty()) return true;
    auto res = http.get(url, timeout_ms);
    if (!res) return false;
    return res->status >= 200 && res->status < 300;
}

}  // namespace onebit::cli
