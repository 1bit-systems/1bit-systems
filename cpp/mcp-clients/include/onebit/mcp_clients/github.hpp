#pragma once

// GitHub MCP wrapper. Auth = PAT via Authorization: Bearer <token>.
// Fed from $GITHUB_TOKEN — same var our gh + cargo + systemd units use.

#include "onebit/mcp_clients/http.hpp"

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace onebit::mcp_clients {

inline constexpr std::string_view GITHUB_DEFAULT_ENDPOINT =
    "https://api.githubcopilot.com/mcp/";

class GitHub {
public:
    static GitHub with_token(std::string_view endpoint, std::string_view token);

    // Read $GITHUB_TOKEN; returns nullopt when env is missing/empty.
    static std::optional<GitHub> from_env();

    [[nodiscard]] HttpClient&       inner()       noexcept { return *http_; }
    [[nodiscard]] const HttpClient& inner() const noexcept { return *http_; }

private:
    explicit GitHub(std::shared_ptr<HttpClient> c) : http_(std::move(c)) {}
    std::shared_ptr<HttpClient> http_;
};

} // namespace onebit::mcp_clients
