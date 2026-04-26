#pragma once

#include "onebit/stream/handlers.hpp"

#include <filesystem>
#include <string>

namespace onebit::stream {

struct ServerConfig {
    std::string                      bind{"127.0.0.1"};
    int                              port{8150};
    std::filesystem::path            catalog_dir;
    AuthConfig                       auth{};
};

[[nodiscard]] std::filesystem::path default_catalog_dir();

// Bind + serve until httplib::Server::stop() is called externally. Returns
// 0 on clean shutdown, 1 on bind failure.
int run_server(const ServerConfig& cfg);

} // namespace onebit::stream
