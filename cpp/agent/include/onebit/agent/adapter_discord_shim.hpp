// adapter_discord_shim — IAdapter that spawns scripts/discord-shim.ts as
// a child process and uses its line-delimited JSON wire to send and
// receive Discord traffic. Workaround for the post-IDENTIFY TCP RST that
// Cloudflare's gateway frontend serves to our hand-rolled OpenSSL/RFC6455
// path in adapter_discord.cpp. The shim runs under bun (TS) so this
// stays Rule-A clean (no Python in the runtime).
#pragma once

#include "onebit/agent/adapter.hpp"
#include "onebit/agent/error.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::agent {

struct DiscordShimAdapterConfig {
    std::string  token;        // mandatory; passed to shim via env DISCORD_TOKEN
    std::uint32_t intents = 0; // 0 → shim's default 37377
    // DM allowlist (user ids). Empty = drop all DMs. Guild messages
    // are gated by the shim itself (only @-mentions surface).
    std::vector<std::string> dm_allowlist;
    // Path to the shim script. Defaults to the one shipped in the repo.
    std::string  shim_path;
};

class DiscordShimAdapter final : public IAdapter {
public:
    explicit DiscordShimAdapter(DiscordShimAdapterConfig cfg);
    ~DiscordShimAdapter() override;

    DiscordShimAdapter(const DiscordShimAdapter&)            = delete;
    DiscordShimAdapter& operator=(const DiscordShimAdapter&) = delete;

    [[nodiscard]] std::expected<void, AgentError> start() override;
    [[nodiscard]] std::expected<IncomingMessage, AgentError>
        recv(std::chrono::milliseconds timeout) override;
    [[nodiscard]] std::expected<void, AgentError>
        send(const std::string& channel, std::string_view text) override;
    void stop() noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace onebit::agent
