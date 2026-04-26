#include "onebit/agent/error.hpp"

#include <fmt/format.h>

namespace onebit::agent {

std::string AgentError::what() const
{
    return std::visit(
        [](const auto& e) -> std::string {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<T, ErrorConfig>) {
                return fmt::format("config: {}", e.msg);
            } else if constexpr (std::is_same_v<T, ErrorSqlite>) {
                return fmt::format("sqlite (rc={}): {}", e.rc, e.msg);
            } else if constexpr (std::is_same_v<T, ErrorBrain>) {
                if (e.http_status != 0) {
                    return fmt::format("brain (http {}): {}", e.http_status, e.msg);
                }
                return fmt::format("brain: {}", e.msg);
            } else if constexpr (std::is_same_v<T, ErrorAdapter>) {
                return fmt::format("adapter: {}", e.msg);
            } else if constexpr (std::is_same_v<T, ErrorAdapterTimeout>) {
                return "adapter: recv timeout";
            } else if constexpr (std::is_same_v<T, ErrorAdapterClosed>) {
                return "adapter: closed";
            } else if constexpr (std::is_same_v<T, ErrorTool>) {
                return fmt::format("tool {}: {}", e.name, e.msg);
            } else if constexpr (std::is_same_v<T, ErrorLoop>) {
                return fmt::format("loop: {}", e.msg);
            } else {
                return "agent: unknown error";
            }
        },
        v_);
}

} // namespace onebit::agent
