// (main lives in test_main.cpp)
#include <doctest/doctest.h>

#include "onebit/agent/loop.hpp"

#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>

using namespace onebit::agent;
using json = nlohmann::json;

namespace {

// --- Mock adapter ------------------------------------------------------
class MockAdapter final : public IAdapter {
public:
    std::expected<void, AgentError> start() override { ++start_calls; return {}; }

    void enqueue(IncomingMessage m)
    {
        std::lock_guard lk(m_);
        in_.push_back(std::move(m));
    }

    void close()
    {
        std::lock_guard lk(m_);
        closed_ = true;
    }

    std::expected<IncomingMessage, AgentError>
    recv(std::chrono::milliseconds /*timeout*/) override
    {
        std::lock_guard lk(m_);
        if (!in_.empty()) {
            auto msg = std::move(in_.front());
            in_.pop_front();
            return msg;
        }
        if (closed_) return std::unexpected(AgentError::adapter_closed());
        return std::unexpected(AgentError::adapter_timeout());
    }

    std::expected<void, AgentError>
    send(const std::string& /*channel*/, std::string_view text) override
    {
        std::lock_guard lk(m_);
        sent.emplace_back(text);
        return {};
    }

    void stop() noexcept override {}

    std::vector<std::string>     sent;
    int                          start_calls = 0;

private:
    std::mutex                   m_;
    std::deque<IncomingMessage>  in_;
    bool                         closed_ = false;
};

// --- Mock brain --------------------------------------------------------
class MockBrain final : public IBrain {
public:
    // Each scripted reply is consumed in order. If we run out, return
    // an error so tests notice over-calling.
    void script(BrainReply r) { replies.push_back(std::move(r)); }

    std::expected<BrainReply, AgentError>
    chat(const BrainRequest& req, const StreamCallback& /*on_chunk*/) override
    {
        last_request = req;
        ++calls;
        if (replies.empty()) {
            return std::unexpected(AgentError::brain("script exhausted"));
        }
        auto r = std::move(replies.front());
        replies.pop_front();
        return r;
    }

    std::deque<BrainReply> replies;
    BrainRequest           last_request;
    int                    calls = 0;
};

// --- Mock tool registry ------------------------------------------------
class MockTools final : public IToolRegistry {
public:
    [[nodiscard]] std::vector<json>
    list_tools_openai_format() const override { return tools_; }

    void add_tool(json schema) { tools_.push_back(std::move(schema)); }
    void script_result(ToolResult r) { results_.push_back(std::move(r)); }
    void script_error(std::string name, std::string msg)
    {
        errors_.push_back({std::move(name), std::move(msg)});
    }

    std::expected<ToolResult, AgentError>
    call(const ToolCall& c) override
    {
        ++calls;
        last_call = c;
        if (!errors_.empty()) {
            auto e = std::move(errors_.front());
            errors_.pop_front();
            return std::unexpected(AgentError::tool(e.first, e.second));
        }
        if (results_.empty()) {
            return std::unexpected(AgentError::tool(c.name, "no scripted result"));
        }
        auto r = std::move(results_.front());
        results_.pop_front();
        return r;
    }

    int       calls = 0;
    ToolCall  last_call{};

private:
    std::vector<json> tools_;
    std::deque<ToolResult> results_;
    std::deque<std::pair<std::string, std::string>> errors_;
};

// --- Boilerplate -------------------------------------------------------
struct Harness {
    Config       cfg;
    Memory       memory;
    MockAdapter  adapter;
    MockBrain    brain;
    MockTools    tools;
    AgentLoop    loop;

    Harness()
        : memory(make_memory()),
          loop(make_cfg(), &adapter, &brain, &memory, &tools)
    {}

    static Memory make_memory()
    {
        auto m = Memory::open(":memory:");
        REQUIRE(m.has_value());
        return std::move(*m);
    }

    static Config make_cfg()
    {
        Config c;
        c.agent.brain_url      = "http://stub";
        c.agent.system_prompt  = "be terse";
        c.agent.max_history    = 16;
        c.agent.max_tool_iters = 3;
        c.agent.stream         = false;
        c.adapter.kind         = "stdin";
        c.memory.sqlite_path   = ":memory:";
        return c;
    }
};

IncomingMessage user_msg(std::string text)
{
    IncomingMessage m;
    m.channel   = "chan-1";
    m.user_id   = "alice";
    m.user_name = "Alice";
    m.text      = std::move(text);
    return m;
}

} // namespace

TEST_CASE("AgentLoop: text-only reply path -> persisted + sent")
{
    Harness h;
    BrainReply r; r.text = "hi back";
    h.brain.script(std::move(r));

    auto rc = h.loop.handle_for_tests(user_msg("hi"));
    REQUIRE(rc.has_value());

    REQUIRE_EQ(h.adapter.sent.size(), 1u);
    CHECK_EQ(h.adapter.sent[0], "hi back");
    CHECK_EQ(h.loop.stats().messages_in, 1);
    CHECK_EQ(h.loop.stats().messages_out, 1);
    CHECK_EQ(h.loop.stats().brain_calls, 1);
    CHECK_EQ(h.loop.stats().tool_calls, 0);

    // Two persisted messages: user then assistant.
    auto rows = h.memory.recent_messages("chan-1", 10);
    REQUIRE(rows.has_value());
    REQUIRE_EQ(rows->size(), 2u);
    CHECK_EQ((*rows)[0].role, "user");
    CHECK_EQ((*rows)[1].role, "assistant");
    CHECK_EQ((*rows)[1].content, "hi back");
}

TEST_CASE("AgentLoop: tool round -> tool message inserted, brain re-invoked, final text wins")
{
    Harness h;
    BrainReply r1;
    ToolCall tc; tc.id = "c1"; tc.name = "repo_search";
    tc.args_json = json{{"q", "rope"}};
    r1.tool_calls = {tc};
    h.brain.script(std::move(r1));

    BrainReply r2; r2.text = "ok found it";
    h.brain.script(std::move(r2));

    h.tools.script_result({/*success=*/true, "rope.cpp:42"});

    auto rc = h.loop.handle_for_tests(user_msg("find rope"));
    REQUIRE(rc.has_value());

    CHECK_EQ(h.brain.calls, 2);
    CHECK_EQ(h.tools.calls, 1);
    CHECK_EQ(h.tools.last_call.name, "repo_search");
    REQUIRE_EQ(h.adapter.sent.size(), 1u);
    CHECK_EQ(h.adapter.sent[0], "ok found it");

    auto rows = h.memory.recent_messages("chan-1", 10);
    REQUIRE(rows.has_value());
    // user, assistant(with tool_calls), tool, assistant(final)
    REQUIRE_EQ(rows->size(), 4u);
    CHECK_EQ((*rows)[0].role, "user");
    CHECK_EQ((*rows)[1].role, "assistant");
    CHECK_FALSE((*rows)[1].tool_calls_json.empty());
    CHECK_EQ((*rows)[2].role, "tool");
    CHECK_EQ((*rows)[2].user_id, "c1");      // tool_call_id round-tripped
    CHECK_EQ((*rows)[3].role, "assistant");
    CHECK_EQ((*rows)[3].content, "ok found it");
}

TEST_CASE("AgentLoop: tool error is fed back as content, loop continues")
{
    Harness h;
    BrainReply r1;
    ToolCall tc; tc.id = "c1"; tc.name = "broken_tool";
    r1.tool_calls = {tc};
    h.brain.script(std::move(r1));

    BrainReply r2; r2.text = "i tried, sorry";
    h.brain.script(std::move(r2));

    h.tools.script_error("broken_tool", "boom");

    auto rc = h.loop.handle_for_tests(user_msg("call the broken one"));
    REQUIRE(rc.has_value());
    REQUIRE_EQ(h.adapter.sent.size(), 1u);
    CHECK_EQ(h.adapter.sent[0], "i tried, sorry");
    auto rows = h.memory.recent_messages("chan-1", 10);
    REQUIRE(rows.has_value());
    // tool message body should contain the error JSON
    bool saw_error_row = false;
    for (const auto& row : *rows) {
        if (row.role == "tool" && row.content.find("boom") != std::string::npos) {
            saw_error_row = true;
        }
    }
    CHECK(saw_error_row);
}

TEST_CASE("AgentLoop: max_tool_iters cap surfaces a giveup message")
{
    Harness h;
    // Force infinite tool round.
    for (int i = 0; i < 10; ++i) {
        BrainReply r;
        ToolCall tc; tc.id = "c" + std::to_string(i); tc.name = "echo";
        r.tool_calls = {tc};
        h.brain.script(std::move(r));
        h.tools.script_result({true, "x"});
    }

    auto rc = h.loop.handle_for_tests(user_msg("loop forever"));
    REQUIRE(rc.has_value());
    REQUIRE_EQ(h.adapter.sent.size(), 1u);
    CHECK(h.adapter.sent[0].find("max_tool_iters") != std::string::npos);
    // brain.calls = max_tool_iters + 1 (zero-indexed loop bound)
    CHECK_EQ(h.brain.calls, h.loop.config().agent.max_tool_iters + 1);
}

TEST_CASE("AgentLoop::pump_once: timeout is non-fatal and increments counter")
{
    Harness h;
    auto rc = h.loop.pump_once(std::chrono::milliseconds{1});
    REQUIRE(rc.has_value());
    CHECK_EQ(h.loop.stats().adapter_timeouts, 1);
    CHECK_EQ(h.loop.stats().messages_in, 0);
}

TEST_CASE("AgentLoop::pump_once: closed adapter surfaces as expected error")
{
    Harness h;
    h.adapter.close();
    auto rc = h.loop.pump_once(std::chrono::milliseconds{1});
    REQUIRE_FALSE(rc.has_value());
    CHECK(rc.error().is_closed());
}

TEST_CASE("AgentLoop: history window includes system prompt + prior turns")
{
    Harness h;
    BrainReply r1; r1.text = "first";
    h.brain.script(std::move(r1));
    auto rc1 = h.loop.handle_for_tests(user_msg("first turn"));
    REQUIRE(rc1.has_value());

    BrainReply r2; r2.text = "second";
    h.brain.script(std::move(r2));
    auto rc2 = h.loop.handle_for_tests(user_msg("second turn"));
    REQUIRE(rc2.has_value());

    // last_request snapshot is from the second turn.
    const auto& hist = h.brain.last_request.history;
    REQUIRE_GE(hist.size(), 4u);
    CHECK_EQ(hist[0].role, "system");
    CHECK_EQ(hist[0].content, "be terse");
    // chronological: user1, asst1, user2 must appear in order
    bool saw_first = false;
    for (const auto& m : hist) {
        if (m.role == "user" && m.content == "first turn") saw_first = true;
        if (m.role == "user" && m.content == "second turn") {
            CHECK(saw_first); // second strictly after first
        }
    }
}

TEST_CASE("AgentLoop::run_forever: drains queued messages then exits cleanly on close")
{
    Harness h;
    h.adapter.enqueue(user_msg("ping1"));
    h.adapter.enqueue(user_msg("ping2"));

    BrainReply a; a.text = "pong1"; h.brain.script(std::move(a));
    BrainReply b; b.text = "pong2"; h.brain.script(std::move(b));

    std::thread driver([&] {
        // Give the loop time to pick up both queued items, then close.
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
        h.adapter.close();
    });

    auto rc = h.loop.run_forever();
    driver.join();

    REQUIRE(rc.has_value());
    CHECK_EQ(h.adapter.start_calls, 1);
    CHECK_EQ(h.loop.stats().messages_in, 2);
    CHECK_EQ(h.loop.stats().messages_out, 2);
}
