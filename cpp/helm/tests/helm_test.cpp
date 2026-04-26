#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/helm/app_model.hpp"
#include "onebit/helm/bearer.hpp"
#include "onebit/helm/conv_log.hpp"
#include "onebit/helm/conversation.hpp"
#include "onebit/helm/models.hpp"
#include "onebit/helm/session.hpp"
#include "onebit/helm/stream.hpp"
#include "onebit/helm/telemetry.hpp"
#include "onebit/helm/tray.hpp"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>

using namespace onebit::helm;

namespace {

std::filesystem::path scratch(const char* leaf)
{
    auto base = std::filesystem::temp_directory_path() / "onebit-helm-test";
    std::error_code ec;
    std::filesystem::create_directories(base, ec);
    return base / leaf;
}

char* test_env(const char* k)
{
    // Match std::getenv signature exactly so &test_env converts to EnvLookup.
    // const_cast is safe here — the literals live in static storage and are
    // not written to by the callee.
    if (std::string_view(k) == "HALO_HELM_URL")     return const_cast<char*>("http://test:9");
    if (std::string_view(k) == "HALO_HELM_MODEL")   return const_cast<char*>("tester");
    if (std::string_view(k) == "HALO_GAIA_URL")     return const_cast<char*>("http://legacy:8");
    return nullptr;
}

} // namespace

// ---- conversation -----------------------------------------------

TEST_CASE("Role round-trips through openai strings")
{
    CHECK(role_to_string(Role::System)    == "system");
    CHECK(role_to_string(Role::User)      == "user");
    CHECK(role_to_string(Role::Assistant) == "assistant");
    CHECK(role_from_string("user")        == Role::User);
    CHECK(role_from_string("assistant")   == Role::Assistant);
}

TEST_CASE("Conversation::to_openai_messages shape")
{
    Conversation c;
    c.push_system("sys");
    c.push_user("hi");
    c.push_assistant("hello");
    auto msgs = c.to_openai_messages();
    REQUIRE(msgs.is_array());
    REQUIRE(msgs.size() == 3);
    CHECK(msgs[0]["role"] == "system");
    CHECK(msgs[0]["content"] == "sys");
    CHECK(msgs[1]["role"] == "user");
    CHECK(msgs[2]["role"] == "assistant");
}

// ---- session ----------------------------------------------------

TEST_CASE("SessionConfig serde round-trip")
{
    SessionConfig c("http://127.0.0.1:8080", "bitnet-3b");
    c.bearer        = "tok";
    c.system_prompt = "You are Halo.";
    auto j = to_json(c);
    auto back = from_json(j);
    CHECK(back.server_url    == c.server_url);
    CHECK(back.default_model == c.default_model);
    CHECK(back.bearer        == c.bearer);
    CHECK(back.system_prompt == c.system_prompt);
}

TEST_CASE("SessionConfig defaults optional fields")
{
    auto src = R"({"server_url":"http://x","default_model":"m"})";
    auto j   = nlohmann::json::parse(src);
    auto c   = from_json(j);
    CHECK_FALSE(c.bearer.has_value());
    CHECK_FALSE(c.system_prompt.has_value());
}

// ---- stream parser ----------------------------------------------

TEST_CASE("parse_sse_line: well-formed delta")
{
    auto e = parse_sse_line(
        R"(data: {"choices":[{"delta":{"content":"Hi"}}]})");
    REQUIRE(is_delta(e));
    CHECK(std::get<SseDelta>(e).content == "Hi");
}

TEST_CASE("parse_sse_line: DONE sentinel + no-space variant")
{
    CHECK(is_done(parse_sse_line("data: [DONE]")));
    CHECK(is_done(parse_sse_line("data:[DONE]")));
}

TEST_CASE("parse_sse_line: ignores keepalives, role-only opener, malformed")
{
    CHECK_FALSE(is_delta(parse_sse_line("")));
    CHECK_FALSE(is_delta(parse_sse_line(": keepalive")));
    CHECK_FALSE(is_delta(parse_sse_line("event: message")));
    CHECK_FALSE(is_delta(parse_sse_line(
        R"(data: {"choices":[{"delta":{"role":"assistant"}}]})")));
    CHECK_FALSE(is_delta(parse_sse_line("data: {not json")));
}

TEST_CASE("parse_sse_line: trailing CR is stripped")
{
    CHECK(is_done(parse_sse_line("data: [DONE]\r")));
}

// ---- models -----------------------------------------------------

TEST_CASE("parse_models: standard openai shape")
{
    auto body = R"({
        "object": "list",
        "data": [
            { "id": "1bit-monster-2b", "object": "model", "owned_by": "halo", "created": 0 },
            { "id": "qwen3-4b-ternary", "object": "model", "owned_by": "halo", "created": 1 }
        ]
    })";
    auto rc = parse_models(body);
    REQUIRE(rc.has_value());
    REQUIRE(rc->size() == 2);
    CHECK((*rc)[0].id == "1bit-monster-2b");
    CHECK((*rc)[0].owned_by == "halo");
    CHECK((*rc)[1].id == "qwen3-4b-ternary");
}

TEST_CASE("parse_models: minimal shape without owned_by")
{
    auto rc = parse_models(R"({ "data": [ { "id": "only-id" } ] })");
    REQUIRE(rc.has_value());
    REQUIRE(rc->size() == 1);
    CHECK((*rc)[0].id == "only-id");
    CHECK((*rc)[0].owned_by.empty());
}

TEST_CASE("parse_models: empty + non-json")
{
    CHECK_FALSE(parse_models("").has_value());
    CHECK_FALSE(parse_models("not json").has_value());
    auto rc = parse_models(R"({"data":[]})");
    REQUIRE(rc.has_value());
    CHECK(rc->empty());
}

// ---- telemetry --------------------------------------------------

TEST_CASE("parse_stats: full shape")
{
    auto blob = R"({
        "loaded_model": "1bit-monster-2b",
        "tok_s_decode": 83.4,
        "gpu_temp_c": 54.0,
        "gpu_util_pct": 27,
        "npu_up": false,
        "shadow_burn_exact_pct": 92.1,
        "services": [
            { "name": "1bit-halo-lemonade", "active": true  },
            { "name": "strix-landing",      "active": true  },
            { "name": "1bit-halo-bitnet",   "active": false }
        ],
        "stale": false
    })";
    auto s = parse_stats(blob);
    REQUIRE(s.has_value());
    CHECK(s->loaded_model == "1bit-monster-2b");
    CHECK(s->tok_s_decode == doctest::Approx(83.4F));
    CHECK(s->gpu_util_pct == 27);
    CHECK_FALSE(s->npu_up);
    REQUIRE(s->services.size() == 3);
    CHECK(s->services[0].active);
    CHECK_FALSE(s->services[2].active);
}

TEST_CASE("parse_stats: tolerates missing fields and rejects garbage")
{
    auto s = parse_stats(R"({ "loaded_model": "m", "tok_s_decode": 1.0 })");
    REQUIRE(s.has_value());
    CHECK(s->loaded_model == "m");
    CHECK_FALSE(s->npu_up);
    CHECK(s->services.empty());

    CHECK_FALSE(parse_stats("not json").has_value());
    CHECK_FALSE(parse_stats("").has_value());
}

TEST_CASE("extract_landing_payload: data prefix + DONE handling")
{
    CHECK(*extract_landing_payload("data: {\"a\":1}") == "{\"a\":1}");
    CHECK(*extract_landing_payload("data:{\"a\":1}")  == "{\"a\":1}");
    CHECK(*extract_landing_payload("data: {\"a\":1}\r") == "{\"a\":1}");
    CHECK_FALSE(extract_landing_payload("").has_value());
    CHECK_FALSE(extract_landing_payload(": keepalive").has_value());
    CHECK_FALSE(extract_landing_payload("event: message").has_value());
    CHECK_FALSE(extract_landing_payload("data: [DONE]").has_value());
}

// ---- conv_log ---------------------------------------------------

TEST_CASE("conv_log: roundtrip write-then-read preserves turns")
{
    auto root = scratch("convlog-roundtrip");
    std::error_code ec;
    std::filesystem::remove_all(root, ec);

    Conversation c;
    c.push_user("what is 2+2?");
    c.push_assistant("4");
    c.push_user("thanks");

    auto rc = write_session(root, c);
    REQUIRE(rc.has_value());
    REQUIRE(std::filesystem::exists(*rc));
    CHECK(rc->extension() == ".jsonl");

    auto entries = read_session(*rc);
    REQUIRE(entries.has_value());
    REQUIRE(entries->size() == 3);
    CHECK((*entries)[0].role    == "user");
    CHECK((*entries)[0].content == "what is 2+2?");
    CHECK((*entries)[1].role    == "assistant");
    CHECK((*entries)[2].content == "thanks");

    std::filesystem::remove_all(root, ec);
}

TEST_CASE("conv_log: write creates root if missing + skips malformed")
{
    auto root = scratch("convlog-create") / "nested" / "deep";
    std::error_code ec;
    std::filesystem::remove_all(root.parent_path().parent_path(), ec);

    Conversation empty;
    auto rc = write_session(root, empty);
    REQUIRE(rc.has_value());
    REQUIRE(std::filesystem::exists(*rc));

    auto path = scratch("convlog-mixed.jsonl");
    {
        std::ofstream f(path, std::ios::trunc);
        f << R"({"role":"user","content":"hi","ts":1})" << "\n";
        f << "not json" << "\n";
        f << R"({"role":"assistant","content":"hello","ts":2})" << "\n";
    }
    auto out = read_session(path);
    REQUIRE(out.has_value());
    CHECK(out->size() == 2);
    CHECK((*out)[0].role == "user");
    CHECK((*out)[1].role == "assistant");
    std::filesystem::remove(path, ec);
}

// ---- bearer (file-only path; keyring path needs live dbus) ------

TEST_CASE("bearer: file fallback round-trip + 0600 perms")
{
    auto root = scratch("bearer-rt");
    std::error_code ec;
    std::filesystem::remove_all(root, ec);

    auto b = Bearer::with_file_only(root);
    CHECK(b.backend() == BearerBackend::None);
    CHECK_FALSE(b.get().has_value());

    auto rc = b.store("sk-test-123");
    REQUIRE(rc.has_value());
    CHECK(*rc == BearerBackend::XdgFile);
    CHECK(b.backend() == BearerBackend::XdgFile);
    REQUIRE(b.get().has_value());
    CHECK(*b.get() == "sk-test-123");

#if defined(__unix__)
    auto fp = b.fallback_path();
    auto perms = std::filesystem::status(fp).permissions();
    using P = std::filesystem::perms;
    CHECK((perms & (P::group_read | P::group_write |
                    P::others_read | P::others_write))
          == P::none);
#endif

    auto b2 = Bearer::with_file_only(root);
    b2.load();
    CHECK(b2.backend() == BearerBackend::XdgFile);
    REQUIRE(b2.get().has_value());
    CHECK(*b2.get() == "sk-test-123");

    auto cl = b2.clear();
    REQUIRE(cl.has_value());
    CHECK(b2.backend() == BearerBackend::None);
    CHECK_FALSE(b2.get().has_value());

    std::filesystem::remove_all(root, ec);
}

TEST_CASE("bearer: empty / whitespace token rejected")
{
    auto root = scratch("bearer-empty");
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    auto b = Bearer::with_file_only(root);
    CHECK_FALSE(b.store("").has_value());
    CHECK_FALSE(b.store("   \n\t").has_value());
    CHECK(b.backend() == BearerBackend::None);
    std::filesystem::remove_all(root, ec);
}

TEST_CASE("bearer backend label strings are stable")
{
    CHECK(bearer_backend_label(BearerBackend::Keyring)
          == "system keyring");
    CHECK(bearer_backend_label(BearerBackend::XdgFile)
          == "~/.config/1bit-helm/bearer.txt (0600)");
    CHECK(bearer_backend_label(BearerBackend::None) == "unset");
}

// ---- app_model --------------------------------------------------

TEST_CASE("AppModel: defaults are sane on construction")
{
    auto m = make_app_model(SessionConfig("http://127.0.0.1:8200",
                                          "1bit-monster-2b"));
    CHECK(m.cfg.server_url    == "http://127.0.0.1:8200");
    CHECK(m.gateway_url       == "http://127.0.0.1:8200");
    CHECK(m.landing_url       == "http://127.0.0.1:8190");
    CHECK(m.cfg.default_model == "1bit-monster-2b");
    CHECK(m.chat_conv.empty());
    CHECK(m.chat_input.empty());
    CHECK_FALSE(m.chat_streaming.has_value());
    CHECK(m.models.empty());
    CHECK_FALSE(m.last_error.has_value());
}

TEST_CASE("Pane: default + label table")
{
    CHECK(pane_label(Pane::Status)   == "Status");
    CHECK(pane_label(Pane::Chat)     == "Chat");
    CHECK(pane_label(Pane::Models)   == "Models");
    CHECK(pane_label(Pane::Settings) == "Settings");
    CHECK(*pane_from_string("Chat")  == Pane::Chat);
    CHECK_FALSE(pane_from_string("nope").has_value());
    CHECK(std::size(PANES_ALL) == 4);
    CHECK(PANES_ALL[0] == Pane::Status);
    CHECK(PANES_ALL[3] == Pane::Settings);
}

TEST_CASE("env_any: prefers earlier keys, returns nullopt when all empty")
{
    auto v = env_any(test_env, {"HALO_HELM_URL", "HALO_GAIA_URL"});
    REQUIRE(v.has_value());
    CHECK(*v == "http://test:9");
    auto miss = env_any(test_env, {"DOES_NOT_EXIST"});
    CHECK_FALSE(miss.has_value());
    auto fallback = env_any(test_env, {"DOES_NOT_EXIST", "HALO_GAIA_URL"});
    REQUIRE(fallback.has_value());
    CHECK(*fallback == "http://legacy:8");
}

TEST_CASE("build_chat_body: stream:true + system prompt + turns")
{
    SessionConfig cfg("http://x", "m");
    cfg.system_prompt = "be brief";
    Conversation conv;
    conv.push_user("q");
    auto body = build_chat_body(cfg, conv);
    CHECK(body["stream"] == true);
    CHECK(body["model"]  == "m");
    REQUIRE(body["messages"].is_array());
    REQUIRE(body["messages"].size() == 2);
    CHECK(body["messages"][0]["role"]    == "system");
    CHECK(body["messages"][0]["content"] == "be brief");
    CHECK(body["messages"][1]["role"]    == "user");
}

TEST_CASE("apply_ui_msg: chat delta + done lands an assistant turn")
{
    auto m = make_app_model(SessionConfig("http://x", "m"));
    m.chat_streaming = std::string{};
    apply_ui_msg(m, UiChatDelta{"Hel"});
    apply_ui_msg(m, UiChatDelta{"lo"});
    REQUIRE(m.chat_streaming.has_value());
    CHECK(*m.chat_streaming == "Hello");
    apply_ui_msg(m, UiChatDone{});
    CHECK_FALSE(m.chat_streaming.has_value());
    REQUIRE(m.chat_conv.turns().size() == 1);
    CHECK(m.chat_conv.turns().front().role == Role::Assistant);
    CHECK(m.chat_conv.turns().front().content == "Hello");
}

TEST_CASE("apply_ui_msg: telemetry connect/disconnect flips state")
{
    auto m = make_app_model(SessionConfig("http://x", "m"));
    LiveStats s;
    s.loaded_model = "m";
    s.tok_s_decode = 10.0F;
    apply_ui_msg(m, UiTelemetrySnapshot{s});
    CHECK(m.live_connected);
    CHECK(m.live.loaded_model == "m");

    apply_ui_msg(m, UiTelemetryDisconnect{"x"});
    CHECK_FALSE(m.live_connected);
    REQUIRE(m.live_last_error.has_value());
    CHECK(*m.live_last_error == "x");
}

TEST_CASE("flush_conversation: empty conv is a no-op, populated writes a file")
{
    auto m = make_app_model(SessionConfig("http://x", "m"));
    auto root = scratch("flush-conv");
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    m.log_root = root;

    auto rc_empty = flush_conversation(m);
    REQUIRE(rc_empty.has_value());
    CHECK_FALSE(rc_empty->has_value());

    m.chat_conv.push_user("hi");
    m.chat_conv.push_assistant("hello");
    auto rc = flush_conversation(m);
    REQUIRE(rc.has_value());
    REQUIRE(rc->has_value());
    CHECK(std::filesystem::exists(**rc));

    auto entries = read_session(**rc);
    REQUIRE(entries.has_value());
    CHECK(entries->size() == 2);

    std::filesystem::remove_all(root, ec);
}

// ---- tray pure logic -------------------------------------------

TEST_CASE("tray: SERVICES set is stable and de-duplicated")
{
    using namespace onebit::helm::tray;
    CHECK(SERVICES.size() == 2);
    CHECK(SERVICES[0] == "1bit-halo-bitnet");
    CHECK(SERVICES[1] == "strix-server");
    CHECK(SERVICES[0] != SERVICES[1]);
}

TEST_CASE("tray: status-line formatter")
{
    using namespace onebit::helm::tray;
    std::vector<ServiceStatus> rows = {
        {"1bit-halo-bitnet", ServiceState::Active},
        {"strix-server",     ServiceState::Active},
    };
    CHECK(build_status_line(rows)
          == "1bit-halo-bitnet: active · strix-server: active");
    CHECK(build_status_line({}) == "no services");

    CHECK(parse_service_state("active\n")  == ServiceState::Active);
    CHECK(parse_service_state("inactive")  == ServiceState::Inactive);
    CHECK(parse_service_state("failed")    == ServiceState::Inactive);
    CHECK(parse_service_state("???")       == ServiceState::Unknown);
}

TEST_CASE("tray: actions enum + labels are non-empty in declared order")
{
    using namespace onebit::helm::tray;
    CHECK(ACTIONS_ALL.size() == 6);
    CHECK(ACTIONS_ALL.front() == Action::Status);
    CHECK(ACTIONS_ALL.back()  == Action::Quit);
    for (auto a : ACTIONS_ALL) {
        CHECK_FALSE(action_label(a).empty());
    }
    CHECK(action_label(Action::OpenSite) == "Open 1bit.systems");
    CHECK(REFRESH_INTERVAL_MS == 3000);
    CHECK(ICON_PNG_PLACEHOLDER_B64.back() == '=');
}
