#include "onebit/tier_mint/server.hpp"

#include "onebit/tier_mint/routes.hpp"

#include <cstdio>
#include <httplib.h>

namespace onebit::tier_mint {

int run_server(const ServerConfig& sc)
{
    AppState state{sc.cfg};

    httplib::Server server;
    build_router(server, state);

    std::fprintf(stderr, "1bit-tier-mint listening on %s:%d\n",
                 sc.bind.c_str(), sc.port);
    if (!server.listen(sc.bind.c_str(), sc.port)) {
        std::fprintf(stderr, "1bit-tier-mint: bind %s:%d failed\n",
                     sc.bind.c_str(), sc.port);
        return 1;
    }
    return 0;
}

} // namespace onebit::tier_mint
