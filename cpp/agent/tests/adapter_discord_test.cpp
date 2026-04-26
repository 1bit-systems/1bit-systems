// SPDX-License-Identifier: Apache-2.0
//
// 1bit-agent — Discord adapter doctest suite.
//
// Hermetic: no live network. We exercise the pure framers + JSON
// serializers + access.json loader + mention parser in isolation.
// Live gateway connect() / recv_event() / send() coverage is left to
// an integration harness gated on $DISCORD_BOT_TOKEN.

// (main lives in test_main.cpp)
#include <doctest/doctest.h>

#include "onebit/agent/adapter_discord.hpp"
#include "onebit/agent/discord_ws.hpp"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace onebit::agent;
using namespace onebit::agent::discord;

namespace {

std::string make_tmp_path(std::string_view tag)
{
    const auto dir = std::filesystem::temp_directory_path();
    return (dir / (std::string{"adapter_discord_"} + std::string{tag} +
                   ".json")).string();
}

void write_file(const std::string& path, std::string_view body)
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(body.data(), static_cast<std::streamsize>(body.size()));
}

} // namespace

// ---------------------------------------------------------------------------
// 1) Frame builder: small payload, length<126 path.
// ---------------------------------------------------------------------------
TEST_CASE("build_client_frame masks payload and sets fin bit")
{
    const std::string payload = "hello";
    const std::uint8_t mask[4] = {0xAA, 0xBB, 0xCC, 0xDD};
    const auto frame = build_client_frame(
        WsOpcode::Text,
        reinterpret_cast<const std::uint8_t*>(payload.data()),
        payload.size(), mask);

    // FIN | Text opcode = 0x81
    CHECK(frame[0] == 0x81U);
    // MASK bit set + length = 5 → 0x80 | 5 = 0x85
    CHECK(frame[1] == 0x85U);
    // Mask key bytes 2..5
    CHECK(frame[2] == 0xAA);
    CHECK(frame[3] == 0xBB);
    CHECK(frame[4] == 0xCC);
    CHECK(frame[5] == 0xDD);
    // Masked payload
    CHECK(frame.size() == 6 + payload.size());
    for (std::size_t i = 0; i < payload.size(); ++i) {
        const std::uint8_t expected =
            static_cast<std::uint8_t>(payload[i]) ^ mask[i & 3];
        CHECK(frame[6 + i] == expected);
    }
}

TEST_CASE("build_client_frame uses 16-bit ext length for medium payload")
{
    std::vector<std::uint8_t> payload(200, 0x41);
    const std::uint8_t mask[4] = {0x01, 0x02, 0x03, 0x04};
    const auto frame = build_client_frame(
        WsOpcode::Text, payload.data(), payload.size(), mask);
    // Header should be 2 + 2 (16-bit len) + 4 (mask) = 8 bytes.
    CHECK(frame[0] == 0x81U);
    CHECK(frame[1] == (0x80U | 126U));
    CHECK(frame[2] == 0x00);
    CHECK(frame[3] == 0xC8); // 200
    CHECK(frame.size() == 8 + payload.size());
}

// ---------------------------------------------------------------------------
// 2) Frame header parser: server → client (must NOT be masked).
// ---------------------------------------------------------------------------
TEST_CASE("parse_frame_header rejects masked server frames")
{
    const std::uint8_t hdr[2] = {0x81U, 0x80U | 5U}; // FIN|Text, MASK|len=5
    const auto r = parse_frame_header(hdr, sizeof(hdr));
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().kind == GatewayError::Kind::Protocol);
}

TEST_CASE("parse_frame_header accepts text frame and reports payload length")
{
    const std::uint8_t hdr[2] = {0x81U, 5U};
    const auto r = parse_frame_header(hdr, sizeof(hdr));
    REQUIRE(r.has_value());
    CHECK(r->fin);
    CHECK(r->opcode == WsOpcode::Text);
    CHECK(r->payload_len == 5U);
    CHECK(r->header_size == 2U);
}

TEST_CASE("parse_frame_header rejects non-minimal 16-bit length")
{
    // len marker 126 with actual length 100 (≤125) is a spec violation.
    const std::uint8_t hdr[4] = {0x81U, 126U, 0x00, 0x64};
    const auto r = parse_frame_header(hdr, sizeof(hdr));
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().kind == GatewayError::Kind::Protocol);
}

TEST_CASE("parse_frame_header rejects RSV bits")
{
    const std::uint8_t hdr[2] = {0x81U | 0x40U, 0x05U}; // RSV1 set
    const auto r = parse_frame_header(hdr, sizeof(hdr));
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().kind == GatewayError::Kind::Protocol);
}

// ---------------------------------------------------------------------------
// 3) Identify / Heartbeat / Resume JSON serializers.
// ---------------------------------------------------------------------------
TEST_CASE("serialize_identify carries token, intents, and properties")
{
    GatewayConfig cfg;
    cfg.token   = "TEST_TOKEN_NOT_REAL";
    cfg.intents = kIntentDefault;
    const std::string s = serialize_identify(cfg);
    auto j = nlohmann::json::parse(s);
    CHECK(j["op"].get<int>() == static_cast<int>(GatewayOp::Identify));
    CHECK(j["d"]["token"].get<std::string>() == "TEST_TOKEN_NOT_REAL");
    CHECK(j["d"]["intents"].get<std::uint32_t>() == kIntentDefault);
    CHECK(j["d"]["properties"]["os"].get<std::string>() == "linux");
    CHECK(j["d"]["properties"]["browser"].get<std::string>() == "halo-agent");
    CHECK(j["d"]["properties"]["device"].get<std::string>() == "halo-agent");
    // Privileged-intent bits sanity check.
    const auto bits = j["d"]["intents"].get<std::uint32_t>();
    CHECK((bits & kIntentMessageContent) != 0);
    CHECK((bits & kIntentDirectMessages) != 0);
    CHECK((bits & kIntentGuildMessages) != 0);
    CHECK((bits & kIntentGuilds) != 0);
}

TEST_CASE("serialize_heartbeat emits null d when no sequence yet")
{
    const std::string s0 = serialize_heartbeat(-1);
    auto j0 = nlohmann::json::parse(s0);
    CHECK(j0["op"].get<int>() == static_cast<int>(GatewayOp::Heartbeat));
    CHECK(j0["d"].is_null());

    const std::string s1 = serialize_heartbeat(42);
    auto j1 = nlohmann::json::parse(s1);
    CHECK(j1["d"].get<std::int64_t>() == 42);
}

TEST_CASE("serialize_resume carries token, session_id, and seq")
{
    const std::string s = serialize_resume("TKN", "SID", 7);
    auto j = nlohmann::json::parse(s);
    CHECK(j["op"].get<int>() == static_cast<int>(GatewayOp::Resume));
    CHECK(j["d"]["token"].get<std::string>() == "TKN");
    CHECK(j["d"]["session_id"].get<std::string>() == "SID");
    CHECK(j["d"]["seq"].get<std::int64_t>() == 7);
}

// ---------------------------------------------------------------------------
// 4) Inbound gateway frame parser.
// ---------------------------------------------------------------------------
TEST_CASE("parse_gateway_frame extracts op, sequence, type, data")
{
    const std::string raw = R"({"op":0,"s":12,"t":"MESSAGE_CREATE",)"
                            R"("d":{"channel_id":"C","content":"hi"}})";
    const auto r = parse_gateway_frame(raw);
    REQUIRE(r.has_value());
    CHECK(r->op == GatewayOp::Dispatch);
    CHECK(r->sequence == 12);
    CHECK(r->type == "MESSAGE_CREATE");
    CHECK(r->data["channel_id"].get<std::string>() == "C");
    CHECK(r->data["content"].get<std::string>() == "hi");
}

TEST_CASE("parse_gateway_frame surfaces Hello with heartbeat_interval")
{
    const std::string raw =
        R"({"op":10,"d":{"heartbeat_interval":41250},"s":null,"t":null})";
    const auto r = parse_gateway_frame(raw);
    REQUIRE(r.has_value());
    CHECK(r->op == GatewayOp::Hello);
    CHECK(r->data["heartbeat_interval"].get<int>() == 41250);
    CHECK(r->sequence == -1);
}

TEST_CASE("parse_gateway_frame rejects malformed JSON")
{
    const auto r = parse_gateway_frame("not json");
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().kind == GatewayError::Kind::Json);
}

// ---------------------------------------------------------------------------
// 5) Token resolution: env wins over config.
// ---------------------------------------------------------------------------
TEST_CASE("resolve_token prefers env over config")
{
    const char* kVar = "ONEBIT_TEST_DISCORD_TOKEN";
    ::setenv(kVar, "FROM_ENV", 1);
    CHECK(resolve_token(kVar, "FROM_CFG") == "FROM_ENV");
    ::unsetenv(kVar);
    CHECK(resolve_token(kVar, "FROM_CFG") == "FROM_CFG");
}

TEST_CASE("resolve_token returns empty when neither source set")
{
    const char* kVar = "ONEBIT_TEST_DISCORD_TOKEN_ABSENT";
    ::unsetenv(kVar);
    CHECK(resolve_token(kVar, "").empty());
}

// ---------------------------------------------------------------------------
// 6) access.json loader — both shapes, plus missing-file path.
// ---------------------------------------------------------------------------
TEST_CASE("load_access_json: missing file returns empty without error")
{
    const auto r = load_access_json("/nonexistent/path/access.json");
    REQUIRE(r.has_value());
    CHECK(r->empty());
}

TEST_CASE("load_access_json: object form with users array")
{
    const auto path = make_tmp_path("obj");
    write_file(path,
        R"({"version":1,"users":[)"
        R"({"id":"111","name":"alice"},)"
        R"({"id":"222","name":"bob"}]})");
    const auto r = load_access_json(path);
    REQUIRE(r.has_value());
    CHECK(r->size() == 2);
    CHECK((*r)[0] == "111");
    CHECK((*r)[1] == "222");
    std::filesystem::remove(path);
}

TEST_CASE("load_access_json: array-of-strings form")
{
    const auto path = make_tmp_path("arr");
    write_file(path, R"(["u1","u2","u3"])");
    const auto r = load_access_json(path);
    REQUIRE(r.has_value());
    CHECK(r->size() == 3);
    CHECK((*r)[0] == "u1");
    CHECK((*r)[2] == "u3");
    std::filesystem::remove(path);
}

TEST_CASE("load_access_json: malformed JSON surfaces ErrorAdapter")
{
    const auto path = make_tmp_path("bad");
    write_file(path, "not json {");
    const auto r = load_access_json(path);
    REQUIRE_FALSE(r.has_value());
    // .what() should mention "access.json"
    CHECK(r.error().what().find("access.json") != std::string::npos);
    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// 7) DiscordAdapter mention + DM-allowlist surface.
// ---------------------------------------------------------------------------
TEST_CASE("DiscordAdapter dm_allowed honours allowlist exactly")
{
    DiscordAdapterConfig cfg;
    cfg.token        = "T";
    cfg.dm_allowlist = {"111", "222"};
    DiscordAdapter a(std::move(cfg));
    CHECK(a.dm_allowed("111"));
    CHECK(a.dm_allowed("222"));
    CHECK_FALSE(a.dm_allowed("333"));
    CHECK_FALSE(a.dm_allowed(""));
}

TEST_CASE("DiscordAdapter mentions_self matches both <@id> and <@!id>")
{
    DiscordAdapterConfig cfg;
    cfg.token       = "T";
    cfg.bot_user_id = "999";
    DiscordAdapter a(std::move(cfg));
    CHECK(a.mentions_self("hey <@999> ping"));
    CHECK(a.mentions_self("yo <@!999> ping"));
    CHECK_FALSE(a.mentions_self("nobody home"));
    CHECK_FALSE(a.mentions_self("close <@9999>"));
}

TEST_CASE("DiscordAdapter mentions_self false when self id unknown")
{
    DiscordAdapterConfig cfg;
    cfg.token = "T"; // bot_user_id intentionally empty
    DiscordAdapter a(std::move(cfg));
    CHECK_FALSE(a.mentions_self("<@anything>"));
}

// ---------------------------------------------------------------------------
// 8) Identify intents bits (sanity — the privileged-intent rule is
//    central to the Discord dev-portal toggle).
// ---------------------------------------------------------------------------
TEST_CASE("intents bitmask matches Discord docs")
{
    CHECK(kIntentGuilds         == (1U << 0));
    CHECK(kIntentGuildMessages  == (1U << 9));
    CHECK(kIntentDirectMessages == (1U << 12));
    CHECK(kIntentMessageContent == (1U << 15));
    CHECK(kIntentDefault == (kIntentGuilds | kIntentGuildMessages |
                             kIntentDirectMessages | kIntentMessageContent));
}
