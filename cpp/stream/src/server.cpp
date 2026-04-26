#include "onebit/stream/server.hpp"

#include <cstdio>
#include <cstdlib>
#include <httplib.h>

namespace onebit::stream {

namespace fs = std::filesystem;

fs::path default_catalog_dir()
{
    if (const auto* xdg = std::getenv("XDG_DATA_HOME"); xdg != nullptr && xdg[0] != '\0') {
        return fs::path{xdg} / "1bit" / "catalogs";
    }
    if (const auto* home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
        return fs::path{home} / ".local" / "share" / "1bit" / "catalogs";
    }
    return fs::path{"/tmp/1bit/catalogs"};
}

int run_server(const ServerConfig& cfg)
{
    AppState state{cfg.catalog_dir, cfg.auth};
    auto rep = state.reindex();
    std::fprintf(stderr,
                 "1bit-stream initial reindex: loaded=%zu errors=%zu dir=%s\n",
                 rep.loaded, rep.errors.size(),
                 cfg.catalog_dir.string().c_str());

    httplib::Server server;
    build(server, state);

    std::fprintf(stderr,
                 "1bit-stream listening on %s:%d\n", cfg.bind.c_str(), cfg.port);
    if (!server.listen(cfg.bind.c_str(), cfg.port)) {
        std::fprintf(stderr,
                     "1bit-stream: bind %s:%d failed\n",
                     cfg.bind.c_str(), cfg.port);
        return 1;
    }
    return 0;
}

} // namespace onebit::stream
