#include "onebit/cli/package.hpp"

#include <cstdlib>
#include <string>

namespace onebit::cli {

bool is_placeholder_seed(std::string_view raw) noexcept
{
    return raw == "$USER" || raw == "$HOME";
}

std::string expand_placeholder(std::string_view raw)
{
    if (raw == "$USER") {
        if (const char* v = std::getenv("USER"); v != nullptr) {
            return v;
        }
        return {};
    }
    if (raw == "$HOME") {
        if (const char* v = std::getenv("HOME"); v != nullptr) {
            return v;
        }
        return {};
    }
    return std::string(raw);
}

}  // namespace onebit::cli
