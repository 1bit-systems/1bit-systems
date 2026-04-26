#pragma once

// HTTP server wiring on top of cpp-httplib. Defaults match the Rust
// crate exactly: bind 127.0.0.1, port 8190, lemond at http://127.0.0.1:8180.

#include "onebit/landing/router.hpp"

#include <string>

namespace onebit::landing {

struct Config {
    std::string bind{"127.0.0.1"};
    int         port{8190};
    std::string lemond_url{"http://127.0.0.1:8180"};
};

// Bind + listen, blocking until SIGINT / Server::stop(). Returns 0 on
// clean shutdown, non-zero on bind failure.
int run_server(const Config& cfg);

} // namespace onebit::landing
