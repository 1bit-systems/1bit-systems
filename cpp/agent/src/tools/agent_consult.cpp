// agent_consult — cross-route a question to a peer specialty agent.
//
// Both ends of the pair speak the same OpenAI /v1/chat/completions
// dialect to the same lemond on :8180. The only differences between
// peers are `model` (different recipe / weights) and `system_prompt`
// (different specialty framing). Calling this tool issues a one-shot
// completion against the peer's recipe and returns its reply text as a
// tool result, which the asking agent then folds into its final answer.
//
// Cycle prevention: if the brain emits agent_consult with
// `peer_name == self`, refuse — that's a loop. max_tool_iters in the
// caller's config caps per-turn depth on top of that.

#include "onebit/agent/tools/registry.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <string>
#include <string_view>

namespace onebit::agent::tools {

namespace {

constexpr std::size_t kMaxQuestionBytes = 4 * 1024;
constexpr std::size_t kMaxReplyBytes    = 8 * 1024;
constexpr int         kRequestTimeoutMs = 30 * 1000;

[[nodiscard]] nlohmann::json make_schema(std::string_view peer_name)
{
    std::string desc = "Consult the peer specialty agent ";
    if (!peer_name.empty()) {
        desc += "(\"";
        desc += peer_name;
        desc += "\") ";
    }
    desc +=
        "for a one-shot answer that you fold into your own reply. Use ";
    desc +=
        "when the question crosses your specialty boundary. The peer ";
    desc +=
        "cannot consult you back (cycle guard). Returns the peer's ";
    desc += "reply text verbatim.";

    return nlohmann::json{
        {"type", "function"},
        {"function", {
            {"name", "agent_consult"},
            {"description", desc},
            {"parameters", {
                {"type", "object"},
                {"properties", {
                    {"question", {
                        {"type", "string"},
                        {"description",
                         "The question to send to the peer agent. Be "
                         "self-contained — the peer has no access to "
                         "your conversation history."},
                    }},
                }},
                {"required", nlohmann::json::array({"question"})},
            }},
        }},
    };
}

// Split a brain_url like "http://127.0.0.1:8180/v1" into ("http://127.0.0.1:8180", "/v1").
struct UrlSplit { std::string host_and_scheme; std::string path; };

[[nodiscard]] UrlSplit split_brain_url(std::string_view url)
{
    UrlSplit s;
    auto scheme_end = url.find("://");
    if (scheme_end == std::string_view::npos) {
        s.host_and_scheme = std::string(url);
        s.path            = "/v1";
        return s;
    }
    auto path_pos = url.find('/', scheme_end + 3);
    if (path_pos == std::string_view::npos) {
        s.host_and_scheme = std::string(url);
        s.path            = "/v1";
    } else {
        s.host_and_scheme = std::string(url.substr(0, path_pos));
        s.path            = std::string(url.substr(path_pos));
    }
    if (s.path.empty()) s.path = "/v1";
    return s;
}

} // namespace

ToolDef make_agent_consult(std::string peer_name,
                           std::string peer_brain_url,
                           std::string peer_model,
                           std::string self_name)
{
    ToolDef d;
    d.name   = "agent_consult";
    d.schema = make_schema(peer_name);
    d.invoke = [peer_name      = std::move(peer_name),
                peer_brain_url = std::move(peer_brain_url),
                peer_model     = std::move(peer_model),
                self_name      = std::move(self_name)](
                   const nlohmann::json& args)
        -> std::expected<ToolResult, AgentError>
    {
        if (peer_brain_url.empty() || peer_model.empty()) {
            return ToolResult{false,
                "agent_consult: not configured "
                "(missing [tools.agent_consult] peer_brain_url / peer_model)"};
        }
        if (!peer_name.empty() && !self_name.empty() &&
            peer_name == self_name)
        {
            return ToolResult{false,
                "agent_consult: refused — self-target loop"};
        }
        const std::string question = args.at("question").get<std::string>();
        if (question.empty()) {
            return ToolResult{false, "bad args: question is empty"};
        }
        if (question.size() > kMaxQuestionBytes) {
            return ToolResult{false, "bad args: question exceeds 4 KiB cap"};
        }

        const auto split = split_brain_url(peer_brain_url);

        nlohmann::json body;
        body["model"]       = peer_model;
        body["temperature"] = 0.2;
        body["stream"]      = false;
        body["messages"]    = nlohmann::json::array({
            nlohmann::json{
                {"role", "system"},
                {"content",
                 "You are responding to a peer agent that asked you a "
                 "single one-shot question. Reply with the answer only. "
                 "No greetings, no follow-up offers."}},
            nlohmann::json{
                {"role", "user"},
                {"content", question}},
        });

        httplib::Client cli(split.host_and_scheme);
        cli.set_connection_timeout(0, 5 * 1000 * 1000); // 5 s
        cli.set_read_timeout(kRequestTimeoutMs / 1000, 0);
        cli.set_write_timeout(5, 0);

        const std::string completions_path = split.path + "/chat/completions";
        auto res = cli.Post(completions_path, body.dump(), "application/json");
        if (!res) {
            return ToolResult{false,
                "agent_consult: POST " + completions_path + " failed: " +
                httplib::to_string(res.error())};
        }
        if (res->status < 200 || res->status >= 300) {
            return ToolResult{false,
                "agent_consult: peer returned non-2xx (" +
                std::to_string(res->status) + "): " + res->body};
        }

        std::string reply;
        try {
            auto j = nlohmann::json::parse(res->body);
            reply = j.at("choices").at(0).at("message").at("content")
                       .get<std::string>();
        } catch (const std::exception& e) {
            return ToolResult{false,
                "agent_consult: malformed peer reply: " + std::string(e.what())};
        }
        if (reply.size() > kMaxReplyBytes) {
            reply.resize(kMaxReplyBytes);
            reply += "\n\n…(truncated at 8 KiB cap)";
        }

        std::string content;
        content.reserve(reply.size() + 64);
        content += "[peer ";
        content += peer_name.empty() ? "(unnamed)" : peer_name;
        content += " replied via ";
        content += peer_model;
        content += "]\n\n";
        content += reply;

        return ToolResult{true, std::move(content)};
    };
    return d;
}

} // namespace onebit::agent::tools
