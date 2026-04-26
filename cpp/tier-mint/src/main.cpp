// 1bit-tier-mint entrypoint. Listens on HALO_TIER_LISTEN
// (default 127.0.0.1:8151).

#include "onebit/tier_mint/server.hpp"

#include <CLI/CLI.hpp>

#include <cstdio>
#include <cstdlib>

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
    return {std::move(host), port};
}

} // namespace

int main(int argc, char** argv)
{
    CLI::App app{"1bit-tier-mint — BTCPay -> Premium JWT issuance"};
    app.set_version_flag("--version", std::string{"0.1.0"});

    std::string listen_arg;
    if (const auto* env = std::getenv("HALO_TIER_LISTEN"); env != nullptr) {
        listen_arg = env;
    } else {
        listen_arg = "127.0.0.1:8151";
    }
    app.add_option("--listen", listen_arg, "Bind host:port");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    auto cfg = onebit::tier_mint::Config::from_env();
    if (!cfg) {
        std::fprintf(stderr, "config: %s\n", cfg.error().c_str());
        return 2;
    }

    std::string err;
    auto        hp = parse_listen(listen_arg, err);
    if (!err.empty()) {
        std::fprintf(stderr, "bad --listen: %s\n", err.c_str());
        return 2;
    }

    onebit::tier_mint::ServerConfig sc;
    sc.bind = hp.first;
    sc.port = hp.second;
    sc.cfg  = std::move(*cfg);
    return onebit::tier_mint::run_server(sc);
}
