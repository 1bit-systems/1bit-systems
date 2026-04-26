// SPDX-License-Identifier: Apache-2.0
//
// 1bit-agent — Telegram Bot adapter implementation.

#include "onebit/agent/adapter_telegram.hpp"

#include <nlohmann/json.hpp>

// CPPHTTPLIB_OPENSSL_SUPPORT is set on the onebit_agent_lib target so all
// adapter TUs see the same SSL surface. Don't redefine here.
#include <httplib.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace onebit::agent {

namespace {

using nlohmann::json;
using clock_type = std::chrono::steady_clock;

// api.telegram.org has been the canonical hostname since 2015; we don't
// support self-hosted Bot API forks. The full base URL is
// https://api.telegram.org/bot<TOKEN>/<METHOD>.
constexpr const char* kTelegramHost = "api.telegram.org";
constexpr int         kTelegramPort = 443;

// Chat types we deliver. "private", "group", "supergroup", "channel".
// Channel posts (no `from`) are not delivered today — adapter focuses
// on conversational surfaces only. Loop reading channel posts can come
// later behind a config knob.
[[nodiscard]] bool is_deliverable_chat_type(std::string_view t) noexcept
{
    return t == "private" || t == "group" || t == "supergroup";
}

// nlohmann::json's `value(key, default)` with non-string default needs a
// matching template arg; small helper avoids the verbose ternaries.
[[nodiscard]] std::string json_string_or_empty(const json& obj, std::string_view key)
{
    if (!obj.contains(key) || !obj.at(key).is_string()) return {};
    return obj.at(key).get<std::string>();
}

// Telegram numeric ids fit in int64. nlohmann::json stores them as
// either signed or unsigned; coerce both. `to_string` for outbound.
[[nodiscard]] std::string json_id_to_string(const json& v)
{
    if (v.is_number_integer())  return std::to_string(v.get<std::int64_t>());
    if (v.is_number_unsigned()) return std::to_string(v.get<std::uint64_t>());
    if (v.is_string())          return v.get<std::string>();
    return {};
}

} // namespace

// ---------------------------------------------------------------------
// HttplibTelegramHttpClient — production transport.
// ---------------------------------------------------------------------
namespace {

class HttplibTelegramHttpClient final : public ITelegramHttpClient {
public:
    explicit HttplibTelegramHttpClient(std::string_view bot_token)
        : token_(bot_token)
    {
        cli_.set_keep_alive(true);
        cli_.set_follow_location(false);
        // Default httplib timeouts are too tight for long-poll. We override
        // per-call inside get()/post_json().
    }

    TelegramHttpResponse
    get(std::string_view path, std::chrono::seconds timeout) override
    {
        std::lock_guard lk(mu_);
        cli_.set_connection_timeout(10);
        cli_.set_read_timeout(static_cast<long>(timeout.count()) + 5);

        const std::string full = "/bot" + token_ + std::string(path);
        auto res = cli_.Get(full.c_str());
        if (!res) return TelegramHttpResponse{0, {}};
        return TelegramHttpResponse{res->status, std::move(res->body)};
    }

    TelegramHttpResponse
    post_json(std::string_view path,
              std::string_view body,
              std::chrono::seconds timeout) override
    {
        std::lock_guard lk(mu_);
        cli_.set_connection_timeout(10);
        cli_.set_read_timeout(static_cast<long>(timeout.count()) + 5);

        const std::string full = "/bot" + token_ + std::string(path);
        auto res = cli_.Post(full.c_str(),
                             std::string(body),
                             "application/json");
        if (!res) return TelegramHttpResponse{0, {}};
        return TelegramHttpResponse{res->status, std::move(res->body)};
    }

private:
    // SSLClient is not thread-safe; serialise concurrent get/post against
    // it. Telegram getUpdates is a single in-flight call by design (only
    // one consumer per token), so the lock is uncontended in practice.
    std::mutex            mu_;
    std::string           token_;
    httplib::SSLClient    cli_{kTelegramHost, kTelegramPort};
};

} // namespace

std::unique_ptr<ITelegramHttpClient>
make_default_telegram_http_client(std::string_view bot_token)
{
    return std::make_unique<HttplibTelegramHttpClient>(bot_token);
}

// ---------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------

bool chat_allowed(const std::unordered_set<std::string>& allowlist,
                  std::string_view                       chat_id) noexcept
{
    if (allowlist.empty()) return true;
    // unordered_set<string> doesn't accept string_view lookup pre-C++23
    // heterogeneous lookup unless transparent hash is wired. Safer + cheap
    // here: one allocation per lookup. Adapter recv path is bounded by
    // Telegram's ~30s long-poll, so this is not hot.
    return allowlist.find(std::string(chat_id)) != allowlist.end();
}

std::expected<std::string, AgentError>
parse_get_me(std::string_view json_body)
{
    json j;
    try {
        j = json::parse(json_body);
    } catch (const json::parse_error& e) {
        return std::unexpected(AgentError::adapter(
            std::string("getMe: malformed JSON: ") + e.what()));
    }
    if (!j.is_object() || !j.value("ok", false)) {
        std::string desc = j.is_object() ? j.value("description", "") : "";
        return std::unexpected(AgentError::adapter(
            "getMe: ok=false: " + (desc.empty() ? std::string("(no description)") : desc)));
    }
    if (!j.contains("result") || !j.at("result").is_object()) {
        return std::unexpected(AgentError::adapter("getMe: missing result"));
    }
    const auto& r = j.at("result");
    if (!r.value("is_bot", false)) {
        return std::unexpected(AgentError::adapter(
            "getMe: token does not identify a bot"));
    }
    std::string username = json_string_or_empty(r, "username");
    if (username.empty()) {
        return std::unexpected(AgentError::adapter(
            "getMe: result.username missing"));
    }
    return username;
}

std::expected<GetUpdatesParse, AgentError>
parse_get_updates(std::string_view json_body)
{
    json j;
    try {
        j = json::parse(json_body);
    } catch (const json::parse_error& e) {
        return std::unexpected(AgentError::adapter(
            std::string("getUpdates: malformed JSON: ") + e.what()));
    }
    if (!j.is_object() || !j.value("ok", false)) {
        std::string desc = j.is_object() ? j.value("description", "") : "";
        return std::unexpected(AgentError::adapter(
            "getUpdates: ok=false: " + (desc.empty() ? std::string("(no description)") : desc)));
    }
    if (!j.contains("result") || !j.at("result").is_array()) {
        return std::unexpected(AgentError::adapter("getUpdates: missing result array"));
    }

    GetUpdatesParse out;
    out.messages.reserve(j.at("result").size());

    for (const auto& upd : j.at("result")) {
        if (!upd.is_object()) continue;
        if (upd.contains("update_id") && upd.at("update_id").is_number()) {
            const auto uid = upd.at("update_id").get<std::int64_t>();
            if (uid > out.highest_update_id) out.highest_update_id = uid;
        }
        if (!upd.contains("message") || !upd.at("message").is_object()) {
            // edited_message, callback_query, channel_post — silently
            // skipped today. Their update_id still bumps the offset above
            // so we don't re-receive them.
            continue;
        }
        const auto& m = upd.at("message");
        if (!m.contains("chat") || !m.at("chat").is_object()) continue;
        const auto& c = m.at("chat");
        const std::string chat_type = json_string_or_empty(c, "type");
        if (!is_deliverable_chat_type(chat_type)) continue;

        IncomingMessage im;
        im.channel = json_id_to_string(c.value("id", json{}));
        if (m.contains("from") && m.at("from").is_object()) {
            const auto& f = m.at("from");
            im.user_id   = json_id_to_string(f.value("id", json{}));
            im.user_name = json_string_or_empty(f, "username");
            if (im.user_name.empty()) im.user_name = json_string_or_empty(f, "first_name");
        }
        im.text = json_string_or_empty(m, "text");
        if (im.text.empty()) im.text = json_string_or_empty(m, "caption");

        // Attachments: photos array (largest variant), document, voice,
        // video. We capture file_id as the URL field (caller resolves
        // via getFile). No download here — keeps adapter side-effect-free.
        auto add_attachment = [&](const json& src, std::string_view default_mime) {
            if (!src.is_object()) return;
            Attachment a;
            a.url       = json_string_or_empty(src, "file_id");
            a.mime_type = json_string_or_empty(src, "mime_type");
            if (a.mime_type.empty()) a.mime_type = std::string(default_mime);
            if (src.contains("file_size") && src.at("file_size").is_number_unsigned()) {
                a.bytes = src.at("file_size").get<std::uint64_t>();
            }
            if (!a.url.empty()) im.attachments.push_back(std::move(a));
        };

        if (m.contains("photo") && m.at("photo").is_array() && !m.at("photo").empty()) {
            // Telegram returns multiple sizes; take the last (largest).
            add_attachment(m.at("photo").back(), "image/jpeg");
        }
        add_attachment(m.value("document", json{}), "application/octet-stream");
        add_attachment(m.value("voice",    json{}), "audio/ogg");
        add_attachment(m.value("video",    json{}), "video/mp4");
        add_attachment(m.value("audio",    json{}), "audio/mpeg");

        out.messages.push_back(std::move(im));
    }

    return out;
}

std::string
build_send_message_body(std::string_view chat_id, std::string_view text)
{
    // chat_id may be numeric ("123") or "@channelname"; Telegram accepts
    // both as JSON strings. We always send a string for simplicity.
    json body = {
        {"chat_id", std::string(chat_id)},
        {"text",    std::string(text)},
        // parse_mode left empty per spec — plain text. Caller stays
        // responsible for any future MarkdownV2 escaping when we wire
        // formatted output.
    };
    return body.dump();
}

// ---------------------------------------------------------------------
// TelegramAdapter::Impl
// ---------------------------------------------------------------------

struct TelegramAdapter::Impl {
    TelegramAdapterConfig                 cfg;
    std::unique_ptr<ITelegramHttpClient>  http;

    // Atomic so stop() can wake recv() without a mutex dance. recv()
    // checks running_ at every long-poll boundary.
    std::atomic<bool>                     running{false};
    std::atomic<bool>                     started{false};

    // update_id high-water mark. Telegram acks a batch when we send
    // offset = max_seen + 1.
    std::atomic<std::int64_t>             next_offset{0};

    // Drained one IncomingMessage at a time across recv() calls.
    mutable std::mutex                    buf_mu;
    std::deque<IncomingMessage>           pending;

    // Backoff tracking. Reset to 0 on a successful getUpdates round.
    std::size_t                           consecutive_failures = 0;

    // Cached bot identity from getMe.
    std::string                           bot_username;

    Impl(TelegramAdapterConfig c, std::unique_ptr<ITelegramHttpClient> h)
        : cfg(std::move(c)), http(std::move(h))
    {
        if (!http) {
            http = make_default_telegram_http_client(cfg.token);
        }
    }

    // Sleep with early-out if stop() flips running_. Used during backoff
    // so we don't hold a stop() up by 60s.
    void interruptible_sleep(std::chrono::milliseconds total)
    {
        constexpr auto step = std::chrono::milliseconds(100);
        auto remaining = total;
        while (remaining > std::chrono::milliseconds(0) && running.load(std::memory_order_acquire)) {
            const auto slice = std::min(step, remaining);
            std::this_thread::sleep_for(slice);
            remaining -= slice;
        }
    }

    // Compute backoff for the current consecutive_failures count. n=1
    // returns base; doubles each time, capped at backoff_cap.
    [[nodiscard]] std::chrono::milliseconds current_backoff() const
    {
        if (consecutive_failures == 0) return std::chrono::milliseconds(0);
        // Avoid shifting > 30 (1<<30 ms ~= 12 days) — saturate well before.
        const auto shift = std::min<std::size_t>(consecutive_failures - 1, 20);
        const auto scaled = std::chrono::duration_cast<std::chrono::milliseconds>(
            cfg.backoff_base * (1ULL << shift));
        return std::min(scaled, cfg.backoff_cap);
    }

    void log_failure(std::string_view what, std::string_view detail)
    {
        if (consecutive_failures >= cfg.loud_failure_threshold) {
            spdlog::error("telegram adapter: {} failed (#{}): {}",
                          what, consecutive_failures, detail);
        } else {
            spdlog::debug("telegram adapter: {} failed (#{}): {}",
                          what, consecutive_failures, detail);
        }
    }

    // One getUpdates round-trip + parse + filter + enqueue. Returns
    // true on success (offset advanced, possibly 0 messages). The
    // long-poll deadline is the smaller of cfg.long_poll_timeout and
    // the caller's recv() budget — important so a 100 ms recv() during
    // shutdown doesn't wedge for 30 s.
    [[nodiscard]] std::expected<void, AgentError>
    poll_once(std::chrono::seconds budget)
    {
        const auto    offs    = next_offset.load(std::memory_order_acquire);
        const auto    timeout = std::min(cfg.long_poll_timeout, budget);
        std::string   path = "/getUpdates?offset=" + std::to_string(offs)
                             + "&timeout=" + std::to_string(timeout.count());

        TelegramHttpResponse resp = http->get(path, timeout + std::chrono::seconds(2));
        if (resp.status == 0) {
            return std::unexpected(AgentError::adapter(
                "getUpdates: transport error (no response)"));
        }
        if (resp.status == 401 || resp.status == 403) {
            // Token revoked / banned. Permanent — but per spec we still
            // back off and retry (operator may fix the token).
            return std::unexpected(AgentError::adapter(
                "getUpdates: HTTP " + std::to_string(resp.status)
                + " (token rejected)"));
        }
        if (resp.status == 409) {
            // Another consumer holds the long-poll slot. Operationally
            // common after a crash; backoff lets the stale poller's TCP
            // session timeout kick in.
            return std::unexpected(AgentError::adapter(
                "getUpdates: HTTP 409 Conflict (another consumer)"));
        }
        if (resp.status < 200 || resp.status >= 300) {
            return std::unexpected(AgentError::adapter(
                "getUpdates: HTTP " + std::to_string(resp.status)));
        }

        auto parsed = parse_get_updates(resp.body);
        if (!parsed) return std::unexpected(parsed.error());

        if (parsed->highest_update_id >= 0) {
            next_offset.store(parsed->highest_update_id + 1,
                              std::memory_order_release);
        }

        // Filter and enqueue.
        std::lock_guard lk(buf_mu);
        for (auto& m : parsed->messages) {
            if (!chat_allowed(cfg.chat_allowlist, m.channel)) {
                spdlog::debug("telegram adapter: dropped non-allowlisted chat={}", m.channel);
                continue;
            }
            pending.push_back(std::move(m));
        }
        return {};
    }
};

// ---------------------------------------------------------------------
// TelegramAdapter
// ---------------------------------------------------------------------

TelegramAdapter::TelegramAdapter(TelegramAdapterConfig cfg)
    : p_(std::make_unique<Impl>(std::move(cfg), nullptr))
{
    if (p_->cfg.token.empty()) {
        // Don't throw — start() surfaces the same error via expected.
        // Defer the diagnosis to start() so construction is total.
    }
}

TelegramAdapter::TelegramAdapter(TelegramAdapterConfig cfg,
                                 std::unique_ptr<ITelegramHttpClient> http)
    : p_(std::make_unique<Impl>(std::move(cfg), std::move(http)))
{}

TelegramAdapter::~TelegramAdapter()                                    = default;
TelegramAdapter::TelegramAdapter(TelegramAdapter&&) noexcept           = default;
TelegramAdapter& TelegramAdapter::operator=(TelegramAdapter&&) noexcept = default;

std::expected<void, AgentError> TelegramAdapter::start()
{
    if (p_->started.load(std::memory_order_acquire)) return {};
    if (p_->cfg.token.empty()) {
        return std::unexpected(AgentError::adapter(
            "telegram: empty bot token (set TELEGRAM_BOT_TOKEN or config.adapter.token)"));
    }

    auto resp = p_->http->get("/getMe", std::chrono::seconds(10));
    if (resp.status == 0) {
        return std::unexpected(AgentError::adapter(
            "getMe: transport error (no response)"));
    }
    if (resp.status < 200 || resp.status >= 300) {
        // Surface 401/404 cleanly so operators see "bad token" not
        // "transport error".
        return std::unexpected(AgentError::adapter(
            "getMe: HTTP " + std::to_string(resp.status)
            + (resp.status == 401 || resp.status == 404
                   ? " (likely invalid token)"
                   : "")));
    }
    auto username = parse_get_me(resp.body);
    if (!username) return std::unexpected(username.error());

    p_->bot_username = *username;
    p_->running.store(true, std::memory_order_release);
    p_->started.store(true, std::memory_order_release);
    spdlog::info("telegram adapter: started as @{}", p_->bot_username);
    return {};
}

void TelegramAdapter::stop() noexcept
{
    p_->running.store(false, std::memory_order_release);
}

std::expected<IncomingMessage, AgentError>
TelegramAdapter::recv(std::chrono::milliseconds timeout)
{
    // Fast path: anything buffered from the previous poll? Drain one.
    {
        std::lock_guard lk(p_->buf_mu);
        if (!p_->pending.empty()) {
            IncomingMessage m = std::move(p_->pending.front());
            p_->pending.pop_front();
            return m;
        }
    }

    if (!p_->running.load(std::memory_order_acquire)) {
        return std::unexpected(AgentError::adapter_closed());
    }

    // Backoff if previous poll(s) failed. Capped by caller's budget so
    // a short recv() during shutdown doesn't get pinned by a 60s wait.
    if (p_->consecutive_failures > 0) {
        const auto wait = std::min<std::chrono::milliseconds>(
            p_->current_backoff(), timeout);
        p_->interruptible_sleep(wait);
        if (!p_->running.load(std::memory_order_acquire)) {
            return std::unexpected(AgentError::adapter_closed());
        }
    }

    // Telegram's long-poll uses whole seconds; floor here. Round up to
    // 1s minimum on any positive budget so we don't ping the API in a
    // tight loop. Zero-budget recv (loop polling its stop flag) skips
    // HTTP entirely.
    if (timeout <= std::chrono::milliseconds(0)) {
        return std::unexpected(AgentError::adapter_timeout());
    }
    const auto budget_s = std::max(
        std::chrono::seconds(1),
        std::chrono::duration_cast<std::chrono::seconds>(timeout));

    auto poll = p_->poll_once(budget_s);
    if (!poll) {
        ++p_->consecutive_failures;
        p_->log_failure("getUpdates", poll.error().what());
        // Convert to recoverable timeout; loop will call recv() again,
        // backoff applies above. We never propagate the underlying
        // ErrorAdapter to the dispatch loop because that's reserved for
        // truly fatal misuse (e.g. config error).
        return std::unexpected(AgentError::adapter_timeout());
    }
    p_->consecutive_failures = 0;

    // Try the buffer once more — poll_once may have enqueued.
    std::lock_guard lk(p_->buf_mu);
    if (p_->pending.empty()) {
        return std::unexpected(AgentError::adapter_timeout());
    }
    IncomingMessage m = std::move(p_->pending.front());
    p_->pending.pop_front();
    return m;
}

std::expected<void, AgentError>
TelegramAdapter::send(const std::string& channel, std::string_view text)
{
    if (!p_->started.load(std::memory_order_acquire)) {
        return std::unexpected(AgentError::adapter(
            "telegram: send() before start()"));
    }
    if (channel.empty()) {
        return std::unexpected(AgentError::adapter(
            "telegram: send() with empty chat_id"));
    }
    // Mirror the inbound gate — never send to a chat we wouldn't accept.
    // This covers the case where the loop stitches together a chat_id
    // from elsewhere; defence in depth.
    if (!chat_allowed(p_->cfg.chat_allowlist, channel)) {
        return std::unexpected(AgentError::adapter(
            "telegram: send() to non-allowlisted chat: " + channel));
    }

    const std::string body = build_send_message_body(channel, text);
    auto resp = p_->http->post_json("/sendMessage", body, std::chrono::seconds(20));
    if (resp.status == 0) {
        return std::unexpected(AgentError::adapter(
            "sendMessage: transport error (no response)"));
    }
    if (resp.status < 200 || resp.status >= 300) {
        // Surface Telegram's `description` if we can — they put rate-limit
        // hints there ("retry after N").
        std::string detail;
        try {
            auto j = json::parse(resp.body);
            if (j.is_object()) detail = j.value("description", "");
        } catch (...) { /* ignore */ }
        return std::unexpected(AgentError::adapter(
            "sendMessage: HTTP " + std::to_string(resp.status)
            + (detail.empty() ? std::string{} : (": " + detail))));
    }
    return {};
}

std::string TelegramAdapter::bot_username() const
{
    return p_->bot_username;
}

std::int64_t TelegramAdapter::next_offset() const noexcept
{
    return p_->next_offset.load(std::memory_order_acquire);
}

std::size_t TelegramAdapter::pending_buffer_size() const noexcept
{
    std::lock_guard lk(p_->buf_mu);
    return p_->pending.size();
}

} // namespace onebit::agent
