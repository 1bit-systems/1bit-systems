#include "onebit/agent/loop.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cassert>
#include <chrono>
#include <utility>

namespace onebit::agent {

using json = nlohmann::json;

namespace {

std::int64_t now_unix_seconds()
{
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// Serializes a tool_calls vector for the messages.tool_calls_json
// column. Empty vector -> empty string (cheaper to query for "no
// tools" than to parse "[]" everywhere).
std::string serialize_tool_calls(const std::vector<ToolCall>& tcs)
{
    if (tcs.empty()) return {};
    json arr = json::array();
    for (const auto& tc : tcs) {
        arr.push_back({
            {"id",   tc.id},
            {"name", tc.name},
            {"args", tc.args_json},
        });
    }
    return arr.dump();
}

std::vector<ToolCall> deserialize_tool_calls(std::string_view s)
{
    std::vector<ToolCall> out;
    if (s.empty()) return out;
    try {
        auto v = json::parse(s);
        if (!v.is_array()) return out;
        for (const auto& el : v) {
            ToolCall c;
            if (el.contains("id") && el["id"].is_string()) {
                c.id = el["id"].get<std::string>();
            }
            if (el.contains("name") && el["name"].is_string()) {
                c.name = el["name"].get<std::string>();
            }
            if (el.contains("args")) c.args_json = el["args"];
            out.push_back(std::move(c));
        }
    } catch (const json::parse_error&) {
        // ignore malformed history rows
    }
    return out;
}

// Builds the OpenAI-compat history window from the system prompt +
// recent stored messages. Newest `max_history` rows are taken in
// chronological order (recent_messages already returns chronological).
std::vector<ChatMessage>
build_history(const Config&                     cfg,
              const std::vector<StoredMessage>& rows)
{
    std::vector<ChatMessage> out;
    out.reserve(rows.size() + 1);
    if (!cfg.agent.system_prompt.empty()) {
        ChatMessage sys;
        sys.role    = "system";
        sys.content = cfg.agent.system_prompt;
        out.push_back(std::move(sys));
    }
    for (const auto& r : rows) {
        ChatMessage m;
        m.role    = r.role;
        m.content = r.content;
        if (r.role == "assistant") {
            m.tool_calls = deserialize_tool_calls(r.tool_calls_json);
        }
        // tool messages need the tool_call_id round-tripped via the
        // user_id column; the loop stamps that on append.
        if (r.role == "tool") m.tool_call_id = r.user_id;
        out.push_back(std::move(m));
    }
    return out;
}

} // namespace

// ---------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------

struct AgentLoop::Impl {
    Config         cfg;
    IAdapter*      adapter = nullptr;
    IBrain*        brain   = nullptr;
    Memory*        memory  = nullptr;
    IToolRegistry* tools   = nullptr;

    std::atomic<bool> stop_flag{false};
    LoopStats         stats{};

    // ---- one full turn ------------------------------------------------
    std::expected<void, AgentError>
    handle_message(const IncomingMessage& msg)
    {
        ++stats.messages_in;

        // 1. persist the user turn.
        if (auto rc = memory->append_message(
                msg.channel, msg.user_id, "user", msg.text, "",
                now_unix_seconds());
            !rc) {
            ++stats.hard_errors;
            return std::unexpected(rc.error());
        }

        // 2. iterate brain<->tool until we either get plain text or
        //    blow the iter cap.
        for (int iter = 0; iter <= cfg.agent.max_tool_iters; ++iter) {
            auto rows = memory->recent_messages(msg.channel, cfg.agent.max_history);
            if (!rows) {
                ++stats.hard_errors;
                return std::unexpected(rows.error());
            }
            BrainRequest req;
            req.history     = build_history(cfg, *rows);
            req.tools       = tools->list_tools_openai_format();
            req.model       = cfg.agent.model;
            req.temperature = cfg.agent.temperature;
            req.stream      = cfg.agent.stream;
            req.timeout     = std::chrono::milliseconds{cfg.agent.request_timeout_ms};

            ++stats.brain_calls;
            auto reply = brain->chat(req, {});
            if (!reply) {
                ++stats.hard_errors;
                return std::unexpected(reply.error());
            }

            if (reply->tool_calls.empty()) {
                // 3a. final answer; persist + send.
                if (auto rc = memory->append_message(
                        msg.channel, /*user_id=*/std::string_view{},
                        "assistant", reply->text, "", now_unix_seconds());
                    !rc) {
                    ++stats.hard_errors;
                    return std::unexpected(rc.error());
                }
                if (auto rc = adapter->send(msg.channel, reply->text); !rc) {
                    ++stats.hard_errors;
                    return std::unexpected(rc.error());
                }
                ++stats.messages_out;
                return {};
            }

            // 3b. tool round. Persist the assistant's tool-call
            //     announcement so the next iteration sees it in
            //     history (OpenAI contract: tool messages must be
            //     preceded by an assistant message with tool_calls).
            auto tc_json = serialize_tool_calls(reply->tool_calls);
            if (auto rc = memory->append_message(
                    msg.channel, /*user_id=*/std::string_view{},
                    "assistant", reply->text, tc_json,
                    now_unix_seconds());
                !rc) {
                ++stats.hard_errors;
                return std::unexpected(rc.error());
            }

            // Execute each tool call in order.
            for (const auto& tc : reply->tool_calls) {
                ++stats.tool_calls;
                auto result = tools->call(tc);
                std::string body;
                if (!result) {
                    // Tool error gets fed back as content rather than
                    // bailing the whole turn — the brain may apologize
                    // and try a different tool. Hard infra failures
                    // (e.g. registry corruption) still propagate.
                    body = json{
                        {"error", result.error().what()},
                    }.dump();
                } else {
                    body = json{
                        {"success", result->success},
                        {"content", result->content},
                    }.dump();
                }
                // user_id column carries tool_call_id for role=tool.
                if (auto rc = memory->append_message(
                        msg.channel, /*tool_call_id=*/tc.id,
                        "tool", body, "",
                        now_unix_seconds());
                    !rc) {
                    ++stats.hard_errors;
                    return std::unexpected(rc.error());
                }
            }
            // loop: re-prompt brain with tool results in history
        }

        // Cap blown. Bail with a dignified message rather than burning
        // tokens forever. We surface a synthetic assistant turn so the
        // user sees something instead of silence.
        const std::string giveup =
            "(stopped: max_tool_iters reached without a final answer.)";
        if (auto rc = memory->append_message(
                msg.channel, /*user_id=*/std::string_view{},
                "assistant", giveup, "", now_unix_seconds());
            !rc) {
            ++stats.hard_errors;
            return std::unexpected(rc.error());
        }
        if (auto rc = adapter->send(msg.channel, giveup); !rc) {
            ++stats.hard_errors;
            return std::unexpected(rc.error());
        }
        ++stats.messages_out;
        return {};
    }
};

// ---------------------------------------------------------------------
// AgentLoop public surface
// ---------------------------------------------------------------------

AgentLoop::AgentLoop(Config         cfg,
                     IAdapter*      adapter,
                     IBrain*        brain,
                     Memory*        memory,
                     IToolRegistry* tools)
    : p_(std::make_unique<Impl>())
{
    assert(adapter != nullptr && "AgentLoop: adapter must be non-null");
    assert(brain   != nullptr && "AgentLoop: brain must be non-null");
    assert(memory  != nullptr && "AgentLoop: memory must be non-null");
    assert(tools   != nullptr && "AgentLoop: tools must be non-null");
    p_->cfg     = std::move(cfg);
    p_->adapter = adapter;
    p_->brain   = brain;
    p_->memory  = memory;
    p_->tools   = tools;
}

AgentLoop::~AgentLoop()                                    = default;
AgentLoop::AgentLoop(AgentLoop&&) noexcept                 = default;
AgentLoop& AgentLoop::operator=(AgentLoop&&) noexcept      = default;

const Config&    AgentLoop::config() const noexcept { return p_->cfg; }
const LoopStats& AgentLoop::stats()  const noexcept { return p_->stats; }

void AgentLoop::stop() noexcept       { p_->stop_flag.store(true); }
bool AgentLoop::stopping() const noexcept { return p_->stop_flag.load(); }

std::expected<void, AgentError>
AgentLoop::pump_once(std::chrono::milliseconds timeout)
{
    auto msg = p_->adapter->recv(timeout);
    if (!msg) {
        if (msg.error().is_timeout()) {
            ++p_->stats.adapter_timeouts;
            return {};
        }
        if (msg.error().is_closed()) {
            return std::unexpected(msg.error());
        }
        ++p_->stats.hard_errors;
        return std::unexpected(msg.error());
    }
    return p_->handle_message(*msg);
}

std::expected<void, AgentError>
AgentLoop::run_forever()
{
    if (auto rc = p_->adapter->start(); !rc) {
        return std::unexpected(rc.error());
    }
    using namespace std::chrono_literals;
    while (!p_->stop_flag.load()) {
        auto rc = pump_once(250ms);
        if (!rc) {
            if (rc.error().is_closed()) break;
            // Hard error — bail. Caller decides whether to restart.
            return std::unexpected(rc.error());
        }
    }
    return {};
}

std::expected<void, AgentError>
AgentLoop::handle_for_tests(const IncomingMessage& msg)
{
    return p_->handle_message(msg);
}

} // namespace onebit::agent
