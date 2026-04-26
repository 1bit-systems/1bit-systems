#pragma once

// Semgrep MCP wrapper. No auth. Used by Warden (CVG static-analysis
// gate) and Magistrate (PR policy scan) over the hosted endpoint.

#include "onebit/mcp_clients/http.hpp"

#include <memory>
#include <string_view>

namespace onebit::mcp_clients {

inline constexpr std::string_view SEMGREP_DEFAULT_ENDPOINT =
    "https://mcp.semgrep.ai/mcp";

class Semgrep {
public:
    Semgrep();
    explicit Semgrep(std::string_view endpoint);

    [[nodiscard]] HttpClient&       inner()       noexcept { return *http_; }
    [[nodiscard]] const HttpClient& inner() const noexcept { return *http_; }

private:
    std::shared_ptr<HttpClient> http_;
};

} // namespace onebit::mcp_clients
