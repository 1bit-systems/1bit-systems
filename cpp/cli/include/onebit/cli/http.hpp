#pragma once

// Thin wrapper around cpp-httplib for the few HTTP probes the CLI does.
// Production callers use http_get/http_post; tests use a fake by passing
// an `HttpClient` reference instead.

#include "onebit/cli/error.hpp"

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

namespace onebit::cli {

struct HttpResponse {
    int         status = 0;
    std::string body;
};

class HttpClient {
public:
    HttpClient() = default;
    HttpClient(const HttpClient&) = delete;
    HttpClient& operator=(const HttpClient&) = delete;
    HttpClient(HttpClient&&) noexcept = default;
    HttpClient& operator=(HttpClient&&) noexcept = default;
    virtual ~HttpClient() = default;

    [[nodiscard]] virtual std::expected<HttpResponse, Error>
    get(std::string_view url, std::uint32_t timeout_ms) = 0;

    [[nodiscard]] virtual std::expected<HttpResponse, Error>
    post_json(std::string_view url, std::string_view body, std::uint32_t timeout_ms) = 0;
};

[[nodiscard]] HttpClient& default_http_client();

// 2xx healthcheck. `timeout_ms` ≥ 1000.
[[nodiscard]] bool healthcheck(HttpClient& http,
                               std::string_view url,
                               std::uint32_t timeout_ms = 3000);

}  // namespace onebit::cli
