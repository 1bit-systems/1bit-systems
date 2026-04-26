#include "onebit/cli/paths.hpp"

#include <cstdlib>
#include <string>

namespace onebit::cli {

std::string env_or(std::string_view key, std::string_view fallback)
{
    const std::string k(key);
    if (const char* v = std::getenv(k.c_str()); v != nullptr && *v != '\0') {
        return v;
    }
    return std::string(fallback);
}

std::filesystem::path home_dir()
{
    if (const char* h = std::getenv("HOME"); h != nullptr && *h != '\0') {
        return std::filesystem::path(h);
    }
    return std::filesystem::path("/tmp");
}

std::filesystem::path xdg_config_home()
{
    if (const char* x = std::getenv("XDG_CONFIG_HOME"); x != nullptr && *x != '\0') {
        return std::filesystem::path(x);
    }
    return home_dir() / ".config";
}

std::filesystem::path xdg_data_home()
{
    if (const char* x = std::getenv("XDG_DATA_HOME"); x != nullptr && *x != '\0') {
        return std::filesystem::path(x);
    }
    return home_dir() / ".local" / "share";
}

}  // namespace onebit::cli
