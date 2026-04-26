#include "onebit/mcp_clients/github.hpp"

#include <cstdlib>
#include <string>

namespace onebit::mcp_clients {

GitHub GitHub::with_token(std::string_view endpoint, std::string_view token)
{
    auto http = std::make_shared<HttpClient>(std::string(endpoint));
    http->add_header("Authorization", std::string("Bearer ") + std::string(token));
    return GitHub(std::move(http));
}

std::optional<GitHub> GitHub::from_env()
{
    const char* tok = std::getenv("GITHUB_TOKEN");
    if (!tok || *tok == '\0') return std::nullopt;
    try {
        return GitHub::with_token(GITHUB_DEFAULT_ENDPOINT, tok);
    } catch (const McpError&) {
        return std::nullopt;
    }
}

} // namespace onebit::mcp_clients
