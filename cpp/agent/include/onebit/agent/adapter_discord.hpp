// SPDX-License-Identifier: Apache-2.0
//
// 1bit-agent — Discord IAdapter implementation.
//
// Standalone, autonomous: no Claude / MCP-stdio dependency. Connects
// directly to Discord's WebSocket gateway (see discord_ws.hpp), pulls
// MESSAGE_CREATE events, gates them against a DM allowlist + guild
// @mention rule, dispatches to the agent loop via IAdapter::recv().
//
// Authentication & access control:
//   - Token from $DISCORD_BOT_TOKEN (wins) OR Config.adapter.token.
//   - DM allowlist from ~/.claude/channels/discord/access.json (compat
//     with the bun MCP plugin's format) OR a fallback list passed via
//     the constructor.
//   - Unallowed DMs are silently dropped (NO error response, NO log
//     beyond a debug-level breadcrumb — we don't echo back anything
//     that would let a probe confirm the bot is alive).
//
// Reconnect policy: exponential backoff on TLS / handshake / protocol
// errors, capped at 60 s. SessionInvalid forces a fresh IDENTIFY.

#pragma once

#include "onebit/agent/adapter.hpp"
#include "onebit/agent/error.hpp"
#include "onebit/agent/event.hpp"

#include <chrono>
#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::agent {

struct DiscordAdapterConfig {
    std::string  token;                      // bot token; never logged
    std::uint32_t intents = 0;                // 0 → discord::kIntentDefault
    // DM-only allowlist of user_ids that may DM the bot. Guild messages
    // are gated on @mention of the bot's own user, never on this list.
    std::vector<std::string> dm_allowlist;
    // Optional bot user id — if supplied, MESSAGE_CREATE in guild
    // channels is filtered to messages that @mention this id. If empty,
    // the adapter learns its own id from the READY payload.
    std::string  bot_user_id;
};

// Loads access.json from ~/.claude/channels/discord/access.json (or
// the supplied path) into a flat user_id list. Accepts two on-disk
// shapes for compat:
//   1. {"version":1,"users":[{"id":"...","name":"..."}, ...]}
//   2. ["user_id_1","user_id_2", ...]
// Missing file → empty vector + no error (caller falls back to TOML).
// Malformed file → ErrorAdapter with the parse message.
[[nodiscard]] std::expected<std::vector<std::string>, AgentError>
load_access_json(const std::string& path);

// Default path: $HOME/.claude/channels/discord/access.json.
[[nodiscard]] std::string default_access_json_path();

// Token resolution: env wins. Returns empty string if neither source
// has a token (caller should treat as fatal config error).
[[nodiscard]] std::string
resolve_token(std::string_view from_env_var,
              std::string_view from_config);

class DiscordAdapter final : public IAdapter {
public:
    explicit DiscordAdapter(DiscordAdapterConfig cfg);
    ~DiscordAdapter() override;

    DiscordAdapter(const DiscordAdapter&)            = delete;
    DiscordAdapter& operator=(const DiscordAdapter&) = delete;
    DiscordAdapter(DiscordAdapter&&) noexcept;
    DiscordAdapter& operator=(DiscordAdapter&&) noexcept;

    [[nodiscard]] std::expected<void, AgentError> start() override;

    [[nodiscard]] std::expected<IncomingMessage, AgentError>
    recv(std::chrono::milliseconds timeout) override;

    [[nodiscard]] std::expected<void, AgentError>
    send(const std::string& channel, std::string_view text) override;

    void stop() noexcept override;

    // ------------------------------------------------------------------
    // Test-only seams (exposed for doctest; not part of the IAdapter
    // contract).
    // ------------------------------------------------------------------

    // True iff `user_id` is on the DM allowlist. O(N), N tiny.
    [[nodiscard]] bool dm_allowed(std::string_view user_id) const noexcept;

    // True iff `text` mentions our bot user id (Discord encodes it as
    // "<@<id>>" or "<@!<id>>"). When bot_user_id_ is empty (READY not
    // yet seen) returns false — guild messages are dropped until then.
    [[nodiscard]] bool mentions_self(std::string_view text) const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace onebit::agent
