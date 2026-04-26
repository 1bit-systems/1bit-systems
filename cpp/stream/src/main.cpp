// 1bit-stream binary — hosts .1bl catalogs over HTTP.
//
// Env vars (overrideable on the CLI):
//   HALO_STREAM_LISTEN       — bind host:port (default 127.0.0.1:8150)
//   HALO_STREAM_CATALOG_DIR  — directory scanned for *.1bl
//   HALO_STREAM_JWT_SECRET   — HS256 secret for the lossless gate
//   HALO_STREAM_ADMIN_BEARER — required header for POST /internal/*

#include "onebit/stream/server.hpp"

#include <CLI/CLI.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

[[nodiscard]] std::pair<std::string, int> parse_listen(std::string_view s, std::string& err)
{
    const auto colon = s.find(':');
    if (colon == std::string_view::npos) {
        err = "expected host:port";
        return {{}, 0};
    }
    std::string host{s.substr(0, colon)};
    int         port = 0;
    try {
        port = std::stoi(std::string{s.substr(colon + 1)});
    } catch (...) {
        err = "port not a number";
        return {{}, 0};
    }
    if (host.empty() || port <= 0 || port > 65535) {
        err = "host:port out of range";
        return {{}, 0};
    }
    return {std::move(host), port};
}

[[nodiscard]] std::string env_or(const char* name, std::string_view fb)
{
    const auto* v = std::getenv(name);
    if (v == nullptr) {
        return std::string{fb};
    }
    return std::string{v};
}

} // namespace

int main(int argc, char** argv)
{
    CLI::App app{"1bit-stream — .1bl catalog HTTP server"};
    app.set_version_flag("--version", std::string{"0.1.0"});

    std::string listen_arg = env_or("HALO_STREAM_LISTEN", "127.0.0.1:8150");
    std::string catalog_dir_arg = env_or("HALO_STREAM_CATALOG_DIR", "");
    app.add_option("--listen", listen_arg, "Bind host:port");
    app.add_option("--catalog-dir", catalog_dir_arg,
                   "Directory scanned for *.1bl");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    std::string err;
    auto        hp = parse_listen(listen_arg, err);
    if (!err.empty()) {
        std::fprintf(stderr, "bad --listen: %s\n", err.c_str());
        return 2;
    }

    onebit::stream::ServerConfig cfg;
    cfg.bind        = hp.first;
    cfg.port        = hp.second;
    cfg.catalog_dir = catalog_dir_arg.empty()
                          ? onebit::stream::default_catalog_dir()
                          : std::filesystem::path{catalog_dir_arg};
    cfg.auth        = onebit::stream::AuthConfig::from_env();

    std::error_code ec;
    std::filesystem::create_directories(cfg.catalog_dir, ec);

    return onebit::stream::run_server(cfg);
}
