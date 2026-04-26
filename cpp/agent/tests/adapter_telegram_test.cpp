// SPDX-License-Identifier: Apache-2.0
//
// Telegram adapter tests. The FakeTelegramHttpClient intercepts every
// HTTP call so no test ever talks to api.telegram.org. Tests cover:
//
//   1. parse_get_me — happy path + token-rejected error
//   2. parse_get_updates — multi-update payload + filtering of edited_message
//   3. start() validates token via getMe; failure surfaces as adapter error
//   4. recv() drains updates one at a time and advances next_offset
//   5. send() refuses non-allowlisted chats; honors allowlist
//   6. recv() backoff on consecutive failures + loud-failure threshold
//   7. chat_allowed bypass on empty allowlist
//   8. build_send_message_body shape

// (main lives in test_main.cpp)
#include <doctest/doctest.h>

#include "onebit/agent/adapter_telegram.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace onebit::agent;
using nlohmann::json;

namespace {

// One scripted HTTP turn. Every test wires a deque of these into the
// fake; calls fail loudly if the deque underflows, so an unexpected
// extra HTTP call is caught immediately.
struct ScriptedTurn {
    std::string          expected_method;  // "GET" or "POST"
    std::string          expected_path;    // e.g. "/getMe", "/getUpdates?offset=0&timeout=30"
    TelegramHttpResponse response;
};

class FakeTelegramHttpClient final : public ITelegramHttpClient {
public:
    std::deque<ScriptedTurn> script;
    std::vector<std::string> seen_paths;
    std::vector<std::string> seen_post_bodies;

    TelegramHttpResponse
    get(std::string_view path, std::chrono::seconds /*timeout*/) override
    {
        INFO("GET " << std::string(path));
        REQUIRE(!script.empty());
        auto turn = std::move(script.front());
        script.pop_front();
        CHECK(turn.expected_method == "GET");
        CHECK(turn.expected_path == std::string(path));
        seen_paths.emplace_back(path);
        return turn.response;
    }

    TelegramHttpResponse
    post_json(std::string_view path,
              std::string_view body,
              std::chrono::seconds /*timeout*/) override
    {
        INFO("POST " << std::string(path));
        REQUIRE(!script.empty());
        auto turn = std::move(script.front());
        script.pop_front();
        CHECK(turn.expected_method == "POST");
        CHECK(turn.expected_path == std::string(path));
        seen_paths.emplace_back(path);
        seen_post_bodies.emplace_back(body);
        return turn.response;
    }
};

// One getMe success body matching Bot API docs.
const std::string kGetMeOk = R"({
    "ok": true,
    "result": {
        "id": 123456789,
        "is_bot": true,
        "first_name": "Halo Helpdesk",
        "username": "halo_helpdesk_bot",
        "can_join_groups": true,
        "can_read_all_group_messages": false,
        "supports_inline_queries": false
    }
})";

// Two-update getUpdates payload covering DM text + group photo + an
// edited_message that should be skipped.
const std::string kGetUpdatesPayload = R"({
    "ok": true,
    "result": [
        {
            "update_id": 100,
            "message": {
                "message_id": 1,
                "date": 1700000000,
                "chat": { "id": 412587349, "type": "private" },
                "from": { "id": 412587349, "is_bot": false, "username": "alice" },
                "text": "hello"
            }
        },
        {
            "update_id": 101,
            "message": {
                "message_id": 2,
                "date": 1700000001,
                "chat": { "id": -1001654782309, "type": "supergroup", "title": "halo dev" },
                "from": { "id": 628194073, "is_bot": false, "username": "bob" },
                "caption": "look at this",
                "photo": [
                    { "file_id": "AgADBAAD...small", "width": 90,  "height": 90,  "file_size": 1024 },
                    { "file_id": "AgADBAAD...large", "width": 1280,"height": 720, "file_size": 65536 }
                ]
            }
        },
        {
            "update_id": 102,
            "edited_message": {
                "message_id": 1,
                "date": 1700000002,
                "chat": { "id": 412587349, "type": "private" },
                "from": { "id": 412587349, "is_bot": false, "username": "alice" },
                "text": "hello edited"
            }
        }
    ]
})";

const std::string kEmptyUpdates = R"({"ok": true, "result": []})";
const std::string kSendMessageOk = R"({"ok": true, "result": {"message_id": 42}})";

TelegramAdapterConfig fast_config()
{
    TelegramAdapterConfig c;
    c.token             = "test:TOKEN";
    c.long_poll_timeout = std::chrono::seconds(1);
    c.backoff_base      = std::chrono::milliseconds(1);
    c.backoff_cap       = std::chrono::milliseconds(2);
    c.loud_failure_threshold = 3;
    return c;
}

} // namespace

TEST_CASE("parse_get_me extracts username on ok=true")
{
    auto r = parse_get_me(kGetMeOk);
    REQUIRE(r.has_value());
    CHECK(*r == "halo_helpdesk_bot");
}

TEST_CASE("parse_get_me surfaces ok=false description")
{
    const std::string body = R"({"ok": false, "description": "Unauthorized"})";
    auto r = parse_get_me(body);
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().what().find("Unauthorized") != std::string::npos);
}

TEST_CASE("parse_get_me rejects malformed JSON")
{
    auto r = parse_get_me("not json");
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().what().find("malformed") != std::string::npos);
}

TEST_CASE("parse_get_updates yields messages, skips edited_message, advances offset")
{
    auto r = parse_get_updates(kGetUpdatesPayload);
    REQUIRE(r.has_value());
    CHECK(r->highest_update_id == 102);   // even the edited_message bumps offset
    REQUIRE(r->messages.size() == 2);

    // DM text
    CHECK(r->messages[0].channel   == "412587349");
    CHECK(r->messages[0].user_id   == "412587349");
    CHECK(r->messages[0].user_name == "alice");
    CHECK(r->messages[0].text      == "hello");
    CHECK(r->messages[0].attachments.empty());

    // Group photo — caption + largest photo size as attachment
    CHECK(r->messages[1].channel == "-1001654782309");
    CHECK(r->messages[1].text    == "look at this");
    REQUIRE(r->messages[1].attachments.size() == 1);
    CHECK(r->messages[1].attachments[0].url       == "AgADBAAD...large");
    CHECK(r->messages[1].attachments[0].mime_type == "image/jpeg");
    CHECK(r->messages[1].attachments[0].bytes     == 65536u);
}

TEST_CASE("parse_get_updates rejects ok=false")
{
    auto r = parse_get_updates(R"({"ok": false, "description": "rate limited"})");
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().what().find("rate limited") != std::string::npos);
}

TEST_CASE("chat_allowed: empty allowlist accepts everything")
{
    std::unordered_set<std::string> empty;
    CHECK(chat_allowed(empty, "anything"));
    CHECK(chat_allowed(empty, ""));
}

TEST_CASE("chat_allowed: populated allowlist gates")
{
    std::unordered_set<std::string> allow{"412587349", "-1001654782309"};
    CHECK(chat_allowed(allow, "412587349"));
    CHECK(chat_allowed(allow, "-1001654782309"));
    CHECK_FALSE(chat_allowed(allow, "999"));
}

TEST_CASE("build_send_message_body shape")
{
    auto s = build_send_message_body("123", "hi");
    auto j = json::parse(s);
    CHECK(j["chat_id"] == "123");
    CHECK(j["text"]    == "hi");
    // parse_mode intentionally absent until we wire MarkdownV2 escaping.
    CHECK_FALSE(j.contains("parse_mode"));
}

TEST_CASE("start() validates token via getMe and caches username")
{
    auto fake = std::make_unique<FakeTelegramHttpClient>();
    fake->script.push_back({"GET", "/getMe", {200, kGetMeOk}});
    auto* fake_raw = fake.get();

    TelegramAdapter ad(fast_config(), std::move(fake));
    auto r = ad.start();
    REQUIRE(r.has_value());
    CHECK(ad.bot_username() == "halo_helpdesk_bot");
    CHECK(fake_raw->script.empty());

    // Idempotent.
    auto r2 = ad.start();
    CHECK(r2.has_value());
}

TEST_CASE("start() rejects empty token before touching network")
{
    auto fake = std::make_unique<FakeTelegramHttpClient>();
    // No script entries — any HTTP call would underflow the fake.
    TelegramAdapterConfig cfg = fast_config();
    cfg.token = "";
    TelegramAdapter ad(std::move(cfg), std::move(fake));
    auto r = ad.start();
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().what().find("empty bot token") != std::string::npos);
}

TEST_CASE("start() surfaces HTTP 401 as adapter error")
{
    auto fake = std::make_unique<FakeTelegramHttpClient>();
    fake->script.push_back({"GET", "/getMe", {401, R"({"ok":false,"description":"Unauthorized"})"}});
    TelegramAdapter ad(fast_config(), std::move(fake));
    auto r = ad.start();
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().what().find("401") != std::string::npos);
}

TEST_CASE("recv() drains buffered updates one at a time and advances offset")
{
    auto fake = std::make_unique<FakeTelegramHttpClient>();
    fake->script.push_back({"GET", "/getMe", {200, kGetMeOk}});
    fake->script.push_back({"GET", "/getUpdates?offset=0&timeout=1", {200, kGetUpdatesPayload}});
    auto* fake_raw = fake.get();

    TelegramAdapter ad(fast_config(), std::move(fake));
    REQUIRE(ad.start().has_value());

    // First recv() polls + drains the first message.
    auto m1 = ad.recv(std::chrono::milliseconds(2000));
    REQUIRE(m1.has_value());
    CHECK(m1->channel == "412587349");
    CHECK(m1->text    == "hello");
    CHECK(ad.next_offset() == 103);   // 102 + 1
    CHECK(ad.pending_buffer_size() == 1);

    // Second recv() takes from buffer — no HTTP.
    auto m2 = ad.recv(std::chrono::milliseconds(2000));
    REQUIRE(m2.has_value());
    CHECK(m2->channel == "-1001654782309");
    CHECK(ad.pending_buffer_size() == 0);

    CHECK(fake_raw->script.empty());
}

TEST_CASE("recv() filters by allowlist, drops non-allowlisted DM")
{
    auto fake = std::make_unique<FakeTelegramHttpClient>();
    fake->script.push_back({"GET", "/getMe", {200, kGetMeOk}});
    fake->script.push_back({"GET", "/getUpdates?offset=0&timeout=1", {200, kGetUpdatesPayload}});
    fake->script.push_back({"GET", "/getUpdates?offset=103&timeout=1", {200, kEmptyUpdates}});

    TelegramAdapterConfig cfg = fast_config();
    cfg.chat_allowlist = {"-1001654782309"};   // only the supergroup
    TelegramAdapter ad(std::move(cfg), std::move(fake));
    REQUIRE(ad.start().has_value());

    auto m1 = ad.recv(std::chrono::milliseconds(2000));
    REQUIRE(m1.has_value());
    CHECK(m1->channel == "-1001654782309");

    // Buffer is empty; next recv() polls and gets nothing.
    auto m2 = ad.recv(std::chrono::milliseconds(2000));
    REQUIRE_FALSE(m2.has_value());
    CHECK(m2.error().is_timeout());
}

TEST_CASE("recv() backs off on transport failure, surfaces timeout")
{
    auto fake = std::make_unique<FakeTelegramHttpClient>();
    fake->script.push_back({"GET", "/getMe", {200, kGetMeOk}});
    fake->script.push_back({"GET", "/getUpdates?offset=0&timeout=1", {0, ""}});
    fake->script.push_back({"GET", "/getUpdates?offset=0&timeout=1", {500, "{}"}});
    fake->script.push_back({"GET", "/getUpdates?offset=0&timeout=1", {200, kEmptyUpdates}});

    TelegramAdapter ad(fast_config(), std::move(fake));
    REQUIRE(ad.start().has_value());

    // Three failed/empty rounds = three timeouts. consecutive_failures
    // resets on the third (200 + empty result is success).
    auto a = ad.recv(std::chrono::milliseconds(1500)); CHECK(a.error().is_timeout());
    auto b = ad.recv(std::chrono::milliseconds(1500)); CHECK(b.error().is_timeout());
    auto c = ad.recv(std::chrono::milliseconds(1500)); CHECK(c.error().is_timeout());

    // next_offset stayed 0 since no update_id was acked.
    CHECK(ad.next_offset() == 0);
}

TEST_CASE("recv() returns ErrorAdapterClosed after stop()")
{
    auto fake = std::make_unique<FakeTelegramHttpClient>();
    fake->script.push_back({"GET", "/getMe", {200, kGetMeOk}});
    TelegramAdapter ad(fast_config(), std::move(fake));
    REQUIRE(ad.start().has_value());
    ad.stop();
    auto r = ad.recv(std::chrono::milliseconds(1000));
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().is_closed());
}

TEST_CASE("send() POSTs sendMessage with chat_id + text")
{
    auto fake = std::make_unique<FakeTelegramHttpClient>();
    fake->script.push_back({"GET", "/getMe", {200, kGetMeOk}});
    fake->script.push_back({"POST", "/sendMessage", {200, kSendMessageOk}});
    auto* fake_raw = fake.get();

    TelegramAdapter ad(fast_config(), std::move(fake));
    REQUIRE(ad.start().has_value());

    auto r = ad.send(std::string("412587349"), std::string_view("hello back"));
    REQUIRE(r.has_value());

    REQUIRE(fake_raw->seen_post_bodies.size() == 1);
    auto j = json::parse(fake_raw->seen_post_bodies[0]);
    CHECK(j["chat_id"] == "412587349");
    CHECK(j["text"]    == "hello back");
}

TEST_CASE("send() refuses non-allowlisted chat")
{
    auto fake = std::make_unique<FakeTelegramHttpClient>();
    fake->script.push_back({"GET", "/getMe", {200, kGetMeOk}});
    // No POST scripted — any send would underflow the fake.
    TelegramAdapterConfig cfg = fast_config();
    cfg.chat_allowlist = {"412587349"};
    TelegramAdapter ad(std::move(cfg), std::move(fake));
    REQUIRE(ad.start().has_value());

    auto r = ad.send(std::string("999999"), std::string_view("shouldn't go through"));
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().what().find("non-allowlisted") != std::string::npos);
}

TEST_CASE("send() before start() returns error")
{
    auto fake = std::make_unique<FakeTelegramHttpClient>();
    TelegramAdapter ad(fast_config(), std::move(fake));
    auto r = ad.send(std::string("123"), std::string_view("no"));
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().what().find("before start()") != std::string::npos);
}

TEST_CASE("send() surfaces Telegram description on non-2xx")
{
    auto fake = std::make_unique<FakeTelegramHttpClient>();
    fake->script.push_back({"GET", "/getMe", {200, kGetMeOk}});
    fake->script.push_back({"POST", "/sendMessage",
        {429, R"({"ok":false,"error_code":429,"description":"Too Many Requests: retry after 5"})"}});

    TelegramAdapter ad(fast_config(), std::move(fake));
    REQUIRE(ad.start().has_value());

    auto r = ad.send(std::string("any-chat"), std::string_view("spam"));
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().what().find("429") != std::string::npos);
    CHECK(r.error().what().find("retry after 5") != std::string::npos);
}
