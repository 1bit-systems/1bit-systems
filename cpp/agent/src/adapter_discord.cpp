// SPDX-License-Identifier: Apache-2.0
//
// 1bit-agent — Discord IAdapter implementation.

// CPPHTTPLIB_OPENSSL_SUPPORT comes from httplib::httplib's INTERFACE
// compile definitions (see cpp/cmake/deps.cmake). httplib::SSLClient
// is unavailable without it.
#include "onebit/agent/adapter_discord.hpp"
#include "onebit/agent/discord_ws.hpp"
#include "onebit/log.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace onebit::agent {

namespace {

constexpr const char* kEnvToken     = "DISCORD_BOT_TOKEN";
constexpr const char* kDiscordHost  = "discord.com";
constexpr std::uint16_t kDiscordPort = 443;
constexpr std::chrono::seconds kReconnectMin{1};
constexpr std::chrono::seconds kReconnectMax{60};

// Read $HOME safely. Returns empty on absence.
[[nodiscard]] std::string home_dir()
{
    if (const char* h = std::getenv("HOME"); h != nullptr) {
        return std::string{h};
    }
    return {};
}

// Convert GatewayError to AgentError. Timeouts are not fatal; the
// loop maps adapter_timeout() to "no message yet, try again".
[[nodiscard]] AgentError gw_to_agent(const discord::GatewayError& e)
{
    using K = discord::GatewayError::Kind;
    switch (e.kind) {
        case K::Timeout: return AgentError::adapter_timeout();
        case K::Closed:  return AgentError::adapter_closed();
        default:
            return AgentError::adapter(
                std::string{"discord: "} + e.message);
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Token + access.json helpers (pure)
// ---------------------------------------------------------------------------

std::string resolve_token(std::string_view from_env_var,
                          std::string_view from_config)
{
    if (!from_env_var.empty()) {
        const std::string var{from_env_var};
        if (const char* v = std::getenv(var.c_str()); v != nullptr && *v != '\0') {
            return std::string{v};
        }
    }
    return std::string{from_config};
}

std::string default_access_json_path()
{
    auto h = home_dir();
    if (h.empty()) return {};
    return h + "/.claude/channels/discord/access.json";
}

std::expected<std::vector<std::string>, AgentError>
load_access_json(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        // Missing is OK — caller falls back to TOML config.
        return std::vector<std::string>{};
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string body = ss.str();
    if (body.empty()) {
        return std::vector<std::string>{};
    }
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(body);
    } catch (const nlohmann::json::parse_error& e) {
        return std::unexpected(AgentError::adapter(
            std::string{"access.json parse: "} + e.what()));
    }
    std::vector<std::string> out;
    // Form 1: {"version":1, "users":[{"id":"...","name":"..."}, ...]}
    if (j.is_object() && j.contains("users") && j["users"].is_array()) {
        for (const auto& u : j["users"]) {
            if (u.is_object() && u.contains("id") && u["id"].is_string()) {
                out.push_back(u["id"].get<std::string>());
            } else if (u.is_string()) {
                out.push_back(u.get<std::string>());
            }
        }
        return out;
    }
    // Form 2: ["user_id_1","user_id_2",...]
    if (j.is_array()) {
        for (const auto& s : j) {
            if (s.is_string()) out.push_back(s.get<std::string>());
        }
        return out;
    }
    return std::unexpected(AgentError::adapter(
        "access.json: expected object with 'users' or array of ids"));
}

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct DiscordAdapter::Impl {
    DiscordAdapterConfig                cfg;
    std::unique_ptr<discord::DiscordGateway> gw;
    std::atomic<bool>                   stop_flag{false};
    std::atomic<bool>                   started{false};
    // Background reader thread: pulls events, gates them, pushes
    // IncomingMessage onto the queue. recv() blocks on the queue.
    std::jthread                        reader;
    std::mutex                          q_mu;
    std::condition_variable             q_cv;
    std::vector<IncomingMessage>        queue;
    // Cached at READY; learns our own bot user id so guild
    // mention-filtering works without a config setting.
    std::string                         self_id_cached;
    mutable std::mutex                  self_id_mu;

    explicit Impl(DiscordAdapterConfig c) : cfg(std::move(c)) {}

    void push(IncomingMessage m)
    {
        {
            std::scoped_lock lk(q_mu);
            queue.push_back(std::move(m));
        }
        q_cv.notify_one();
    }

    [[nodiscard]] std::string self_id() const
    {
        std::scoped_lock lk(self_id_mu);
        if (!cfg.bot_user_id.empty()) return cfg.bot_user_id;
        return self_id_cached;
    }

    void set_self_id(std::string id)
    {
        std::scoped_lock lk(self_id_mu);
        self_id_cached = std::move(id);
    }

    [[nodiscard]] bool is_dm_event(const nlohmann::json& d) const
    {
        // Discord DM messages have guild_id == null/missing.
        return !d.contains("guild_id") || d["guild_id"].is_null();
    }

    void reader_loop(std::stop_token tok)
    {
        std::chrono::seconds backoff{kReconnectMin};
        while (!tok.stop_requested() &&
               !stop_flag.load(std::memory_order_acquire)) {
            // (Re)connect.
            if (auto r = gw->connect(); !r) {
                onebit::log::eprintln(
                    "[discord] connect failed: {}; backoff {}s",
                    r.error().message, backoff.count());
                std::this_thread::sleep_for(backoff);
                backoff = std::min<std::chrono::seconds>(
                    backoff * 2, kReconnectMax);
                continue;
            }
            backoff = kReconnectMin; // success → reset

            // Drain events.
            while (!tok.stop_requested() &&
                   !stop_flag.load(std::memory_order_acquire)) {
                auto ev = gw->recv_event(std::chrono::seconds(60));
                if (!ev) {
                    using K = discord::GatewayError::Kind;
                    if (ev.error().kind == K::Timeout) continue;
                    if (ev.error().kind == K::SessionInvalid) {
                        // Drop resume state and reconnect with fresh
                        // IDENTIFY. discord_ws cleared resume already.
                        break;
                    }
                    onebit::log::eprintln(
                        "[discord] recv_event error: {}", ev.error().message);
                    break; // outer loop reconnects
                }
                handle_event(*ev);
            }
        }
        // Wake any blocked recv() so the loop sees adapter_closed.
        q_cv.notify_all();
    }

    void handle_event(const discord::GatewayEvent& ev)
    {
        if (ev.type == "READY") {
            // Cache our own bot user id; guild gating uses it.
            if (ev.data.contains("user") && ev.data["user"].is_object() &&
                ev.data["user"].contains("id") &&
                ev.data["user"]["id"].is_string()) {
                set_self_id(ev.data["user"]["id"].get<std::string>());
            }
            return;
        }
        if (ev.type != "MESSAGE_CREATE") return;
        const auto& d = ev.data;
        if (!d.is_object()) return;

        // Skip bot/system messages and our own.
        if (d.contains("author") && d["author"].is_object()) {
            if (d["author"].contains("bot") &&
                d["author"]["bot"].is_boolean() &&
                d["author"]["bot"].get<bool>()) {
                return;
            }
        } else {
            return;
        }

        IncomingMessage msg;
        msg.channel  = d.value("channel_id", std::string{});
        msg.text     = d.value("content", std::string{});
        msg.user_id  = d["author"].value("id", std::string{});
        msg.user_name = d["author"].value("username", std::string{});
        if (msg.channel.empty() || msg.user_id.empty()) return;
        // Echo guard: never reply to ourselves.
        if (const auto sid = self_id(); !sid.empty() && msg.user_id == sid) {
            return;
        }

        if (d.contains("attachments") && d["attachments"].is_array()) {
            for (const auto& a : d["attachments"]) {
                if (!a.is_object()) continue;
                Attachment at;
                at.url = a.value("url", std::string{});
                at.mime_type = a.value("content_type", std::string{});
                if (a.contains("size") && a["size"].is_number_unsigned()) {
                    at.bytes = a["size"].get<std::uint64_t>();
                }
                if (!at.url.empty()) msg.attachments.push_back(std::move(at));
            }
        }

        if (is_dm_event(d)) {
            // DM gating: silent drop if user not on allowlist.
            const auto& allow = cfg.dm_allowlist;
            const bool ok = std::any_of(
                allow.begin(), allow.end(),
                [&](const std::string& u) { return u == msg.user_id; });
            if (!ok) return;
        } else {
            // Guild: require an explicit @mention of self.
            const auto sid = self_id();
            if (sid.empty()) return;
            // Discord encodes mentions as <@id> or <@!id>.
            const std::string a = "<@" + sid + ">";
            const std::string b = "<@!" + sid + ">";
            if (msg.text.find(a) == std::string::npos &&
                msg.text.find(b) == std::string::npos) {
                return;
            }
        }
        push(std::move(msg));
    }
};

DiscordAdapter::DiscordAdapter(DiscordAdapterConfig cfg)
    : p_(std::make_unique<Impl>(std::move(cfg)))
{
    if (p_->cfg.token.empty()) {
        // Caller is expected to have run resolve_token() already; we
        // intentionally do not log the token. start() will fail.
    }
}

DiscordAdapter::~DiscordAdapter() = default;
DiscordAdapter::DiscordAdapter(DiscordAdapter&&) noexcept            = default;
DiscordAdapter& DiscordAdapter::operator=(DiscordAdapter&&) noexcept = default;

std::expected<void, AgentError> DiscordAdapter::start()
{
    if (p_->cfg.token.empty()) {
        return std::unexpected(AgentError::adapter(
            "discord: no token in env or config"));
    }
    if (p_->started.exchange(true)) return {};

    discord::GatewayConfig gcfg;
    gcfg.token   = p_->cfg.token;
    gcfg.intents = p_->cfg.intents == 0 ? discord::kIntentDefault
                                        : p_->cfg.intents;
    p_->gw = std::make_unique<discord::DiscordGateway>(std::move(gcfg));

    p_->reader = std::jthread(
        [impl = p_.get()](std::stop_token tok) { impl->reader_loop(tok); });
    return {};
}

std::expected<IncomingMessage, AgentError>
DiscordAdapter::recv(std::chrono::milliseconds timeout)
{
    std::unique_lock lk(p_->q_mu);
    if (!p_->q_cv.wait_for(lk, timeout, [&] {
            return !p_->queue.empty() ||
                   p_->stop_flag.load(std::memory_order_acquire);
        })) {
        return std::unexpected(AgentError::adapter_timeout());
    }
    if (p_->queue.empty()) {
        return std::unexpected(AgentError::adapter_closed());
    }
    IncomingMessage m = std::move(p_->queue.front());
    p_->queue.erase(p_->queue.begin());
    return m;
}

std::expected<void, AgentError>
DiscordAdapter::send(const std::string& channel, std::string_view text)
{
    if (channel.empty()) {
        return std::unexpected(AgentError::adapter("discord: empty channel"));
    }
    httplib::SSLClient cli(kDiscordHost, kDiscordPort);
    cli.set_connection_timeout(15, 0);
    cli.set_read_timeout(30, 0);
    cli.set_follow_location(false);
    cli.enable_server_certificate_verification(true);

    httplib::Headers headers{
        {"Authorization", std::string{"Bot "} + p_->cfg.token},
        {"User-Agent",    "halo-agent/1.0 (+https://1bit.systems)"},
    };
    nlohmann::json body;
    body["content"] = std::string{text};
    const std::string body_s = body.dump();

    const std::string path = "/api/v10/channels/" + channel + "/messages";
    auto res = cli.Post(path.c_str(), headers, body_s, "application/json");
    if (!res) {
        return std::unexpected(AgentError::adapter(
            "discord: POST messages failed (network)"));
    }
    if (res->status < 200 || res->status >= 300) {
        return std::unexpected(AgentError::adapter(
            "discord: POST messages HTTP " +
            std::to_string(res->status)));
    }
    return {};
}

void DiscordAdapter::stop() noexcept
{
    if (!p_) return;
    p_->stop_flag.store(true, std::memory_order_release);
    if (p_->gw) p_->gw->stop();
    if (p_->reader.joinable()) {
        p_->reader.request_stop();
        // Don't join here; destructor handles it (jthread joins on
        // destruction). stop() must be safe to call from signal
        // handlers off-thread.
    }
    p_->q_cv.notify_all();
}

bool DiscordAdapter::dm_allowed(std::string_view user_id) const noexcept
{
    const auto& allow = p_->cfg.dm_allowlist;
    return std::any_of(allow.begin(), allow.end(),
                       [&](const std::string& u) { return u == user_id; });
}

bool DiscordAdapter::mentions_self(std::string_view text) const noexcept
{
    const auto sid = p_->self_id();
    if (sid.empty()) return false;
    const std::string a = "<@" + sid + ">";
    const std::string b = "<@!" + sid + ">";
    const std::string t{text};
    return t.find(a) != std::string::npos ||
           t.find(b) != std::string::npos;
}

} // namespace onebit::agent
