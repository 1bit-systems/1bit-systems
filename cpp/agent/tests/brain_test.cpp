// (main lives in test_main.cpp)
#include <doctest/doctest.h>

#include "onebit/agent/brain.hpp"

using namespace onebit::agent;
using json = nlohmann::json;

TEST_CASE("build_request_body: basic fields land at the right keys")
{
    BrainRequest req;
    req.model       = "halo-1.58b";
    req.temperature = 0.3;
    req.stream      = true;
    ChatMessage sys; sys.role = "system"; sys.content = "be terse";
    ChatMessage usr; usr.role = "user";   usr.content = "hi";
    req.history = {sys, usr};

    auto body = build_request_body(req);
    CHECK_EQ(body["model"], "halo-1.58b");
    CHECK(body["temperature"].get<double>() == doctest::Approx(0.3));
    CHECK(body["stream"].get<bool>());
    REQUIRE(body["messages"].is_array());
    CHECK_EQ(body["messages"].size(), 2u);
    CHECK_EQ(body["messages"][0]["role"], "system");
    CHECK_EQ(body["messages"][1]["role"], "user");
    CHECK_FALSE(body.contains("tools"));
}

TEST_CASE("build_request_body: tools[] is forwarded verbatim")
{
    BrainRequest req;
    req.tools.push_back(json{
        {"type", "function"},
        {"function", {{"name", "repo_search"}, {"description", "x"}}},
    });
    auto body = build_request_body(req);
    REQUIRE(body.contains("tools"));
    CHECK_EQ(body["tools"][0]["function"]["name"], "repo_search");
}

TEST_CASE("build_request_body: assistant tool_calls round-trip into OpenAI shape")
{
    BrainRequest req;
    ChatMessage asst;
    asst.role = "assistant";
    ToolCall tc;
    tc.id        = "call_1";
    tc.name      = "repo_search";
    tc.args_json = json{{"query", "RoPE"}};
    asst.tool_calls = {tc};
    req.history = {asst};

    auto body = build_request_body(req);
    const auto& msg = body["messages"][0];
    REQUIRE(msg.contains("tool_calls"));
    CHECK_EQ(msg["tool_calls"][0]["id"], "call_1");
    CHECK_EQ(msg["tool_calls"][0]["function"]["name"], "repo_search");
    // arguments must be a JSON-encoded string per OpenAI spec.
    CHECK(msg["tool_calls"][0]["function"]["arguments"].is_string());
}

TEST_CASE("parse_response_body: plain text response")
{
    constexpr const char* kBody = R"({
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop"
        }]
    })";
    auto reply = parse_response_body(kBody);
    REQUIRE(reply.has_value());
    CHECK_EQ(reply->text, "hello");
    CHECK(reply->tool_calls.empty());
}

TEST_CASE("parse_response_body: tool_calls extracted with parsed args")
{
    constexpr const char* kBody = R"({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_a",
                    "type": "function",
                    "function": {
                        "name": "url_fetch",
                        "arguments": "{\"url\":\"https://1bit.systems\"}"
                    }
                }]
            }
        }]
    })";
    auto reply = parse_response_body(kBody);
    REQUIRE(reply.has_value());
    REQUIRE_EQ(reply->tool_calls.size(), 1u);
    CHECK_EQ(reply->tool_calls[0].id, "call_a");
    CHECK_EQ(reply->tool_calls[0].name, "url_fetch");
    CHECK_EQ(reply->tool_calls[0].args_json["url"], "https://1bit.systems");
}

TEST_CASE("parse_response_body: missing choices is a brain error")
{
    auto reply = parse_response_body("{}");
    REQUIRE_FALSE(reply.has_value());
    CHECK(reply.error().what().find("choices") != std::string::npos);
}

TEST_CASE("apply_sse_line: text deltas concatenate; [DONE] flips done")
{
    BrainReply acc;
    auto a = apply_sse_line(R"(data: {"choices":[{"delta":{"content":"hel"}}]})", acc);
    CHECK_EQ(a.text_delta, "hel");
    CHECK_FALSE(a.done);
    auto b = apply_sse_line(R"(data: {"choices":[{"delta":{"content":"lo"}}]})", acc);
    CHECK_EQ(b.text_delta, "lo");
    CHECK_EQ(acc.text, "hello");
    auto d = apply_sse_line("data: [DONE]", acc);
    CHECK(d.done);
}

TEST_CASE("apply_sse_line: tool_call deltas accumulate by index and finalize as JSON")
{
    BrainReply acc;
    apply_sse_line(R"(data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"repo_search","arguments":"{\"q\":"}}]}}]})", acc);
    auto step = apply_sse_line(R"(data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"rope\"}"}}]}}]})", acc);
    CHECK(step.had_tool_delta);
    REQUIRE_EQ(acc.tool_calls.size(), 1u);
    CHECK_EQ(acc.tool_calls[0].id, "c1");
    CHECK_EQ(acc.tool_calls[0].name, "repo_search");
    // args_json is still a streamed string until finalize_tool_call_args
    // runs in complete_streaming. apply_sse_line's contract is "merge",
    // not "parse"; we assert the streamed string is the assembled JSON.
    CHECK(acc.tool_calls[0].args_json.is_string());
    CHECK_EQ(acc.tool_calls[0].args_json.get<std::string>(),
             "{\"q\":\"rope\"}");
}

TEST_CASE("apply_sse_line: comment lines and non-data lines are ignored")
{
    BrainReply acc;
    auto step = apply_sse_line(":heartbeat", acc);
    CHECK(step.text_delta.empty());
    CHECK_FALSE(step.done);
    step = apply_sse_line("event: ping", acc);
    CHECK(step.text_delta.empty());
}
