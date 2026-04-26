#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/halo_ralph/ralph.hpp"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

using onebit::halo_ralph::format_test_failure_feedback;
using onebit::halo_ralph::Message;
using onebit::halo_ralph::parse_base_url;
using onebit::halo_ralph::parse_first_choice_content;
using onebit::halo_ralph::run_test_cmd;
using onebit::halo_ralph::serialize_chat_request;
using onebit::halo_ralph::TestCmdResult;

TEST_CASE("chat_request_shape_matches_openai")
{
    std::vector<Message> msgs = {{"user", "ping"}};
    auto wire = serialize_chat_request("1bit-halo-v2", msgs, 0.3f, false);
    auto v    = nlohmann::json::parse(wire);
    CHECK(v["model"]                 == "1bit-halo-v2");
    CHECK(v["stream"]                == false);
    CHECK(v["messages"][0]["role"]   == "user");
    CHECK(v["messages"][0]["content"] == "ping");
    CHECK(std::abs(v["temperature"].get<double>() - 0.3) < 1e-6);
}

TEST_CASE("parse_first_choice_content_happy_path")
{
    auto body = R"({
        "choices": [
            {"message": {"role": "assistant", "content": "hello world"}}
        ]
    })";
    auto c = parse_first_choice_content(body);
    REQUIRE(c.has_value());
    CHECK(*c == "hello world");
}

TEST_CASE("parse_first_choice_content_handles_empty_choices")
{
    auto c = parse_first_choice_content(R"({"choices": []})");
    REQUIRE(c.has_value());
    CHECK(c->empty());
}

TEST_CASE("parse_first_choice_content_rejects_garbage_returns_nullopt")
{
    auto c = parse_first_choice_content("not json {{{");
    CHECK_FALSE(c.has_value());
}

TEST_CASE("parse_base_url_default_lemond")
{
    auto u = parse_base_url("http://localhost:8180/v1");
    REQUIRE(u.has_value());
    CHECK(u->is_https == false);
    CHECK(u->host     == "localhost");
    CHECK(u->port     == 8180);
    CHECK(u->base_path == "/v1");
}

TEST_CASE("parse_base_url_https_default_port")
{
    auto u = parse_base_url("https://api.openai.com/v1");
    REQUIRE(u.has_value());
    CHECK(u->is_https);
    CHECK(u->host == "api.openai.com");
    CHECK(u->port == 443);
    CHECK(u->base_path == "/v1");
}

TEST_CASE("parse_base_url_strips_trailing_slash")
{
    auto u = parse_base_url("http://h:1234/v1/");
    REQUIRE(u.has_value());
    CHECK(u->base_path == "/v1");
}

TEST_CASE("parse_base_url_rejects_missing_scheme")
{
    CHECK_FALSE(parse_base_url("localhost:8180").has_value());
}

TEST_CASE("test_cmd_zero_exit_round_trip")
{
    auto r = run_test_cmd("printf 'ok'");
    REQUIRE(r.has_value());
    CHECK(r->exit_code == 0);
    CHECK(r->stdout_text == "ok");
}

TEST_CASE("test_cmd_nonzero_exit_captures_stderr")
{
    auto r = run_test_cmd("printf 'oops' >&2; exit 7");
    REQUIRE(r.has_value());
    CHECK(r->exit_code == 7);
    CHECK(r->stderr_text == "oops");
}

TEST_CASE("format_test_failure_feedback_shape")
{
    TestCmdResult r;
    r.exit_code   = 7;
    r.stdout_text = "out";
    r.stderr_text = "err";
    auto msg = format_test_failure_feedback("cargo test", r);
    CHECK(msg.find("Previous action failed") != std::string::npos);
    CHECK(msg.find("`cargo test`")           != std::string::npos);
    CHECK(msg.find("exited with code 7")     != std::string::npos);
    CHECK(msg.find("STDOUT:\nout")           != std::string::npos);
    CHECK(msg.find("STDERR:\nerr")           != std::string::npos);
}
