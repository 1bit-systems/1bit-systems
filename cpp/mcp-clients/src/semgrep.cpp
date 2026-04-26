#include "onebit/mcp_clients/semgrep.hpp"

namespace onebit::mcp_clients {

Semgrep::Semgrep()
    : http_(std::make_shared<HttpClient>(std::string(SEMGREP_DEFAULT_ENDPOINT)))
{}

Semgrep::Semgrep(std::string_view endpoint)
    : http_(std::make_shared<HttpClient>(std::string(endpoint)))
{}

} // namespace onebit::mcp_clients
