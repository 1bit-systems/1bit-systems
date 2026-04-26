// 1bit-landing — marketing landing page + live status probe on :8190.
//
// Defaults match crates/1bit-landing/src/main.rs exactly:
//   --bind        127.0.0.1
//   --port        8190
//   --lemond-url  http://127.0.0.1:8180

#include "onebit/landing/server.hpp"

#include <CLI/CLI.hpp>

#include <cstdio>

int main(int argc, char** argv)
{
    onebit::landing::Config cfg;

    CLI::App app{"1bit-landing — marketing page on :8190"};
    app.add_option("--bind",       cfg.bind,
                   "bind address (default 127.0.0.1)")
        ->capture_default_str();
    app.add_option("--port",       cfg.port,
                   "TCP port (default 8190)")
        ->capture_default_str();
    app.add_option("--lemond-url", cfg.lemond_url,
                   "base URL for lemond probes (default http://127.0.0.1:8180)")
        ->capture_default_str();

    CLI11_PARSE(app, argc, argv);

    return onebit::landing::run_server(cfg);
}
