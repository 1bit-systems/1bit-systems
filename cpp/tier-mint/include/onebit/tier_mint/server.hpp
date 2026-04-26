#pragma once

#include "onebit/tier_mint/state.hpp"

#include <string>

namespace onebit::tier_mint {

struct ServerConfig {
    std::string bind{"127.0.0.1"};
    int         port{8151};
    Config      cfg;
};

[[nodiscard]] int run_server(const ServerConfig& sc);

} // namespace onebit::tier_mint
