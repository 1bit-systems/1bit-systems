// 1bit-agent — daemon entry point.
//
// Usage: halo-agent --config /etc/1bit-agent/halo-helpdesk.toml
//
// systemd contract:
//   * SIGTERM -> graceful drain (AgentLoop::stop). Returns 0.
//   * fatal config / sqlite open / brain init failure -> returns 1.
//
// Threads: one main thread + one adapter recv thread (owned by the
// adapter impl). No threads in the brain path.

#include "onebit/agent/brain.hpp"
#include "onebit/agent/config.hpp"
#include "onebit/agent/factories.hpp"
#include "onebit/agent/loop.hpp"
#include "onebit/agent/memory.hpp"

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include <atomic>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <iostream>

namespace {

std::atomic<onebit::agent::AgentLoop*> g_loop{nullptr};

extern "C" void on_sigterm(int /*signum*/) noexcept
{
    if (auto* l = g_loop.load(); l != nullptr) l->stop();
}

void install_signal_handlers()
{
    struct sigaction sa{};
    sa.sa_handler = &on_sigterm;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT,  &sa, nullptr);
}

} // namespace

int main(int argc, char** argv)
{
    using namespace onebit::agent;

    CLI::App app{"1bit-agent — autonomous agent daemon"};
    std::filesystem::path config_path;
    app.add_option("-c,--config", config_path, "TOML config path")
        ->required()
        ->check(CLI::ExistingFile);
    bool print_version = false;
    app.add_flag("--version", print_version, "print sqlite + agent versions and exit");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    if (print_version) {
        std::cout << "1bit-agent (sqlite " << Memory::sqlite_version() << ")\n";
        return 0;
    }

    auto cfg = load_config(config_path);
    if (!cfg) {
        std::fprintf(stderr, "config error: %s\n", cfg.error().what().c_str());
        return 1;
    }

    auto memory = Memory::open(cfg->memory.sqlite_path);
    if (!memory) {
        std::fprintf(stderr, "memory error: %s\n", memory.error().what().c_str());
        return 1;
    }
    if (cfg->memory.keep_messages > 0) {
        // Best-effort trim on boot; don't fail the daemon if it errors.
        (void)memory->trim_messages(cfg->memory.keep_messages);
    }

    Brain brain(cfg->agent.brain_url);

    auto adapter = make_adapter(*cfg);
    if (!adapter) {
        std::fprintf(stderr, "adapter error: %s\n", adapter.error().what().c_str());
        return 1;
    }
    auto tools = make_tool_registry(*cfg);
    if (!tools) {
        std::fprintf(stderr, "tools error: %s\n", tools.error().what().c_str());
        return 1;
    }

    AgentLoop loop(*cfg, adapter->get(), &brain, &*memory, tools->get());
    g_loop.store(&loop);
    install_signal_handlers();

    std::fprintf(stderr,
                 "1bit-agent: name=%s adapter=%s db=%s sqlite=%s\n",
                 cfg->agent.name.c_str(),
                 cfg->adapter.kind.c_str(),
                 cfg->memory.sqlite_path.string().c_str(),
                 Memory::sqlite_version().c_str());

    auto rc = loop.run_forever();
    g_loop.store(nullptr);

    // Stop the adapter even if the loop bailed (run_forever bails on
    // hard error before the shutdown branch runs).
    (*adapter)->stop();

    if (!rc) {
        std::fprintf(stderr, "fatal: %s\n", rc.error().what().c_str());
        return 1;
    }
    return 0;
}
