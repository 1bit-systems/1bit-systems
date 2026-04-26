// SPDX-License-Identifier: Apache-2.0
//
// 1bit-agent — Telegram Bot adapter.
//
// Long-poll Bot API over HTTPS — no WebSocket gateway, no Discord-style
// shard handshake. The dispatch model:
//
//   start()  — POST getMe; verify token + connectivity; cache bot identity.
//   recv()   — long-poll getUpdates?offset=<N>&timeout=30; transform to
//              IncomingMessage. update_id+1 advanced on each successful
//              parse so Telegram acks the batch.
//   send()   — POST sendMessage to api.telegram.org/bot<TOKEN>/sendMessage.
//   stop()   — flag-flip; the next recv() drains and returns ErrorAdapterClosed.
//
// Reconnect: any getUpdates failure -> exponential backoff capped at 60s.
// After 5 consecutive failures we keep retrying but log loudly so an
// operator can see the Telegram outage in journalctl.
//
// Allowlist: a chat_allowlist passed at construction time pre-filters
// inbound updates. Empty allowlist = deliver all (test-friendly).
// The production wire-up (sibling loop) loads the allowlist from
// ~/.claude/channels/telegram/access.json before constructing us.
//
// HTTP: cpp-httplib + OpenSSL (CPPHTTPLIB_OPENSSL_SUPPORT). The HTTP
// surface is abstracted behind ITelegramHttpClient so tests inject a
// fake without ever hitting api.telegram.org.

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
#include <unordered_set>
#include <vector>

namespace onebit::agent {

// ---------------------------------------------------------------------
// HTTP transport seam. Production binds to cpp-httplib::SSLClient via
// HttplibTelegramHttpClient (built into the .cpp). Tests inject a fake
// implementation and never touch the network.
// ---------------------------------------------------------------------
struct TelegramHttpResponse {
    int         status = 0;     // 0 means "no response" (DNS, TCP, SSL handshake)
    std::string body;
};

class ITelegramHttpClient {
public:
    virtual ~ITelegramHttpClient() = default;

    // GET https://api.telegram.org/bot<token>/<method>?<query>
    // `query` already includes the '?' if non-empty, or is "" for none.
    // Caller passes a per-request deadline; long-poll is ~32s, getMe ~10s.
    [[nodiscard]] virtual TelegramHttpResponse
    get(std::string_view path, std::chrono::seconds timeout) = 0;

    // POST https://api.telegram.org/bot<token>/<method> with
    // Content-Type: application/json and the given body.
    [[nodiscard]] virtual TelegramHttpResponse
    post_json(std::string_view path,
              std::string_view body,
              std::chrono::seconds timeout) = 0;
};

// Default transport — cpp-httplib SSLClient. Pinned to api.telegram.org.
[[nodiscard]] std::unique_ptr<ITelegramHttpClient>
make_default_telegram_http_client(std::string_view bot_token);

// ---------------------------------------------------------------------
// Configuration knobs. Keep these in the header so callers (TOML loader
// + tests) can synthesize them without touching the .cpp.
// ---------------------------------------------------------------------
struct TelegramAdapterConfig {
    // Bot token, e.g. "123456789:AAH...". Never logged. Caller resolves
    // ${ENV:TELEGRAM_BOT_TOKEN} before passing in. Must be non-empty.
    std::string token;

    // Numeric Telegram chat IDs allowed to deliver inbound messages.
    // For DMs the chat id == sender user id. Group chat ids are the
    // negative supergroup ids (-100... prefix). Empty set = deliver all
    // (test/dev convenience; production loop sets this from access.json).
    std::unordered_set<std::string> chat_allowlist;

    // Long-poll deadline passed to getUpdates?timeout=<N>. Telegram caps
    // at 50s; we default to 30 to match the skill server. Lowered in
    // tests so we don't wedge the suite.
    std::chrono::seconds long_poll_timeout{30};

    // Backoff schedule on getUpdates failure. backoff_base_ms doubles up
    // to backoff_cap_ms after each consecutive failure.
    std::chrono::milliseconds backoff_base{500};
    std::chrono::milliseconds backoff_cap{60000};

    // After this many consecutive getUpdates failures we log at error
    // level on every retry. Loop never gives up — Telegram outages are
    // recoverable and dropping inbound mid-outage causes silent message
    // loss.
    std::size_t loud_failure_threshold = 5;
};

// ---------------------------------------------------------------------
// TelegramAdapter — pImpl. Header is dependency-light so the dispatch
// loop can include it without dragging cpp-httplib + OpenSSL into every
// translation unit.
// ---------------------------------------------------------------------
class TelegramAdapter final : public IAdapter {
public:
    // Production constructor. Token + allowlist; default HTTP client.
    explicit TelegramAdapter(TelegramAdapterConfig cfg);

    // Test/DI constructor. Caller supplies the HTTP transport. The fake
    // is the only way tests get coverage without hitting api.telegram.org.
    TelegramAdapter(TelegramAdapterConfig cfg,
                    std::unique_ptr<ITelegramHttpClient> http);

    ~TelegramAdapter() override;

    TelegramAdapter(const TelegramAdapter&)            = delete;
    TelegramAdapter& operator=(const TelegramAdapter&) = delete;
    TelegramAdapter(TelegramAdapter&&) noexcept;
    TelegramAdapter& operator=(TelegramAdapter&&) noexcept;

    // IAdapter
    [[nodiscard]] std::expected<void, AgentError> start() override;
    void                                          stop() noexcept override;
    [[nodiscard]] std::expected<IncomingMessage, AgentError>
    recv(std::chrono::milliseconds timeout) override;
    [[nodiscard]] std::expected<void, AgentError>
    send(const std::string& channel, std::string_view text) override;

    // Bot identity from getMe; populated after start(). Empty before.
    [[nodiscard]] std::string bot_username() const;

    // Test introspection. Stable but not part of the IAdapter contract.
    [[nodiscard]] std::int64_t  next_offset() const noexcept;
    [[nodiscard]] std::size_t   pending_buffer_size() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

// ---------------------------------------------------------------------
// Free helpers — unit-tested standalone so the test suite can exercise
// parsing without a live HTTP transport.
// ---------------------------------------------------------------------

// Parses one Telegram getUpdates response body into a vector of
// IncomingMessage plus the highest update_id seen. Drops updates that
// carry no `message` field (edited_message, callback_query, etc).
//
// On `{"ok": false}` returns AgentError::adapter("...") with the
// description from `description`. Malformed JSON → AgentError::adapter.
struct GetUpdatesParse {
    std::vector<IncomingMessage> messages;
    std::int64_t                 highest_update_id = -1;
};
[[nodiscard]] std::expected<GetUpdatesParse, AgentError>
parse_get_updates(std::string_view json_body);

// Builds the JSON body for sendMessage. parse_mode is currently empty
// (plain text). Caller chooses chat_id encoding (string for either
// numeric DM ids or "@channelusername"; we pass through unchanged).
[[nodiscard]] std::string
build_send_message_body(std::string_view chat_id, std::string_view text);

// Validates a getMe response body and extracts username. Required-field
// failures return AgentError::adapter; this is what start() uses to
// decide whether the token is good.
[[nodiscard]] std::expected<std::string, AgentError>
parse_get_me(std::string_view json_body);

// True iff `chat_id` should be delivered. Empty allowlist = always true.
// Centralized so tests can exercise it without spinning up an adapter.
[[nodiscard]] bool
chat_allowed(const std::unordered_set<std::string>& allowlist,
             std::string_view                       chat_id) noexcept;

} // namespace onebit::agent
