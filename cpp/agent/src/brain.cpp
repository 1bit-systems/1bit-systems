#include "onebit/agent/brain.hpp"

// httplib pulled in only inside the .cpp so users of brain.hpp don't
// transitively pay the compile cost. cpp-httplib is the same dep helm
// uses (cpp/cmake/deps.cmake).
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace onebit::agent {

using json = nlohmann::json;

namespace {

// --- URL split ----------------------------------------------------------
// httplib::Client wants ("http://host:port"). We accept everything
// including trailing path segments (lemond's base URL is just :8180/).
struct ParsedUrl {
    std::string scheme_host;     // "http://127.0.0.1:8180"
    std::string base_path;       // "" or e.g. "/lemonade" if user prefixed
};

ParsedUrl split_url(std::string_view url)
{
    ParsedUrl p;
    auto scheme_end = url.find("://");
    if (scheme_end == std::string_view::npos) {
        // bare "host:port" -> assume http
        auto path_start = url.find('/');
        if (path_start == std::string_view::npos) {
            p.scheme_host = "http://" + std::string(url);
        } else {
            p.scheme_host = "http://" + std::string(url.substr(0, path_start));
            p.base_path   = std::string(url.substr(path_start));
        }
        return p;
    }
    auto after_scheme = scheme_end + 3;
    auto path_start   = url.find('/', after_scheme);
    if (path_start == std::string_view::npos) {
        p.scheme_host = std::string(url);
    } else {
        p.scheme_host = std::string(url.substr(0, path_start));
        p.base_path   = std::string(url.substr(path_start));
    }
    while (!p.base_path.empty() && p.base_path.back() == '/') {
        p.base_path.pop_back();
    }
    // Tolerate users who set brain_url to ".../v1": OpenAI-compat
    // base path is the bare host, so we strip the suffix here rather
    // than emit "/v1/v1/chat/completions" later.
    constexpr std::string_view kV1 = "/v1";
    if (p.base_path.size() >= kV1.size()
        && std::string_view(p.base_path).substr(p.base_path.size() - kV1.size()) == kV1) {
        p.base_path.resize(p.base_path.size() - kV1.size());
    }
    return p;
}

// --- ChatMessage -> JSON ------------------------------------------------
json chat_msg_to_json(const ChatMessage& m)
{
    json out;
    out["role"] = m.role;
    if (!m.content.empty() || m.tool_calls.empty()) {
        out["content"] = m.content;
    }
    if (!m.tool_call_id.empty()) {
        out["tool_call_id"] = m.tool_call_id;
    }
    if (!m.tool_calls.empty()) {
        json arr = json::array();
        for (const auto& tc : m.tool_calls) {
            arr.push_back({
                {"id",   tc.id},
                {"type", "function"},
                {"function", {
                    {"name",      tc.name},
                    {"arguments", tc.args_json.dump()},
                }},
            });
        }
        out["tool_calls"] = std::move(arr);
    }
    return out;
}

// Inverse: pull tool_calls[] off an OpenAI choice/message.
std::vector<ToolCall>
tool_calls_from_json(const json& msg)
{
    std::vector<ToolCall> out;
    if (!msg.contains("tool_calls") || !msg["tool_calls"].is_array()) {
        return out;
    }
    for (const auto& tc : msg["tool_calls"]) {
        ToolCall c;
        if (tc.contains("id") && tc["id"].is_string()) {
            c.id = tc["id"].get<std::string>();
        }
        if (tc.contains("function") && tc["function"].is_object()) {
            const auto& fn = tc["function"];
            if (fn.contains("name") && fn["name"].is_string()) {
                c.name = fn["name"].get<std::string>();
            }
            if (fn.contains("arguments")) {
                if (fn["arguments"].is_string()) {
                    auto raw = fn["arguments"].get<std::string>();
                    try {
                        c.args_json = json::parse(raw);
                    } catch (const json::parse_error&) {
                        c.args_json = raw; // keep as string if not parseable
                    }
                } else {
                    c.args_json = fn["arguments"];
                }
            }
        }
        out.push_back(std::move(c));
    }
    return out;
}

// --- SSE delta merge for tool_calls ------------------------------------
// OpenAI streams tool_calls as an array of {index, id, function:{name,
// arguments (partial)}} fragments. Merge each delta-tc into acc by
// index. We don't bother to round-trip non-function tools.
void merge_tool_call_delta(std::vector<ToolCall>& acc, const json& delta_tcs)
{
    if (!delta_tcs.is_array()) return;
    for (const auto& dt : delta_tcs) {
        std::size_t idx = 0;
        if (dt.contains("index") && dt["index"].is_number_integer()) {
            auto i = dt["index"].get<std::int64_t>();
            if (i < 0) continue;
            idx = static_cast<std::size_t>(i);
        }
        if (acc.size() <= idx) acc.resize(idx + 1);
        auto& slot = acc[idx];
        if (dt.contains("id") && dt["id"].is_string()) {
            slot.id = dt["id"].get<std::string>();
        }
        if (dt.contains("function") && dt["function"].is_object()) {
            const auto& fn = dt["function"];
            if (fn.contains("name") && fn["name"].is_string()) {
                slot.name = fn["name"].get<std::string>();
            }
            if (fn.contains("arguments") && fn["arguments"].is_string()) {
                // Append partial arguments string into a scratch buffer
                // we stash as a raw string in args_json. We re-parse at
                // the end (apply_sse_line callers do this; or
                // complete_streaming below does it).
                if (slot.args_json.is_null() || !slot.args_json.is_string()) {
                    slot.args_json = std::string{};
                }
                slot.args_json = slot.args_json.get<std::string>()
                               + fn["arguments"].get<std::string>();
            }
        }
    }
}

// Final pass: any tool_call whose args_json is still a string (because
// it was streamed in fragments) gets re-parsed to JSON. Best-effort —
// if it doesn't parse we leave the string in place for the registry to
// validate.
void finalize_tool_call_args(std::vector<ToolCall>& tcs)
{
    for (auto& tc : tcs) {
        if (tc.args_json.is_string()) {
            auto raw = tc.args_json.get<std::string>();
            try {
                tc.args_json = json::parse(raw);
            } catch (const json::parse_error&) {
                // leave as string
            }
        }
    }
}

} // namespace

// ---- free helpers (header-declared, unit tested) -----------------------

json build_request_body(const BrainRequest& req)
{
    json body;
    body["model"]        = req.model;
    body["temperature"]  = req.temperature;
    body["stream"]       = req.stream;
    body["messages"]     = json::array();
    for (const auto& m : req.history) {
        body["messages"].push_back(chat_msg_to_json(m));
    }
    if (!req.tools.empty()) {
        body["tools"] = req.tools;
    }
    return body;
}

std::expected<BrainReply, AgentError>
parse_response_body(std::string_view json_body)
{
    json v;
    try {
        v = json::parse(json_body);
    } catch (const json::parse_error& e) {
        return std::unexpected(AgentError::brain(
            std::string("response not JSON: ") + e.what()));
    }
    if (!v.contains("choices") || !v["choices"].is_array()
        || v["choices"].empty()) {
        return std::unexpected(AgentError::brain("response missing choices[]"));
    }
    const auto& choice = v["choices"][0];
    if (!choice.contains("message") || !choice["message"].is_object()) {
        return std::unexpected(AgentError::brain("response missing choices[0].message"));
    }
    const auto& msg = choice["message"];
    BrainReply r;
    if (msg.contains("content") && msg["content"].is_string()) {
        r.text = msg["content"].get<std::string>();
    }
    r.tool_calls = tool_calls_from_json(msg);
    return r;
}

SseStep apply_sse_line(std::string_view line, BrainReply& acc)
{
    SseStep step;
    while (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    if (line.empty() || line.front() == ':') return step;
    constexpr std::string_view kPrefix = "data:";
    if (line.size() < kPrefix.size()
        || line.substr(0, kPrefix.size()) != kPrefix) {
        return step;
    }
    auto payload = line.substr(kPrefix.size());
    while (!payload.empty() && payload.front() == ' ') payload.remove_prefix(1);
    if (payload == "[DONE]") {
        step.done = true;
        return step;
    }
    json v;
    try {
        v = json::parse(payload);
    } catch (const json::parse_error&) {
        return step;
    }
    if (!v.contains("choices") || !v["choices"].is_array() || v["choices"].empty()) {
        return step;
    }
    const auto& choice = v["choices"][0];
    if (!choice.contains("delta") || !choice["delta"].is_object()) {
        return step;
    }
    const auto& delta = choice["delta"];
    if (delta.contains("content") && delta["content"].is_string()) {
        auto s = delta["content"].get<std::string>();
        if (!s.empty()) {
            acc.text.append(s);
            step.text_delta = std::move(s);
        }
    }
    if (delta.contains("tool_calls")) {
        merge_tool_call_delta(acc.tool_calls, delta["tool_calls"]);
        step.had_tool_delta = true;
    }
    return step;
}

// ---- Brain pImpl --------------------------------------------------------

struct Brain::Impl {
    std::string base_url;
    ParsedUrl   parsed;

    explicit Impl(std::string url) : base_url(std::move(url))
    {
        parsed = split_url(base_url);
    }

    std::string completions_path() const
    {
        return parsed.base_path + "/v1/chat/completions";
    }

    httplib::Client make_client(std::chrono::milliseconds timeout) const
    {
        httplib::Client cli(parsed.scheme_host);
        cli.set_connection_timeout(std::chrono::duration_cast<std::chrono::seconds>(timeout));
        cli.set_read_timeout(std::chrono::duration_cast<std::chrono::seconds>(timeout));
        cli.set_write_timeout(std::chrono::duration_cast<std::chrono::seconds>(timeout));
        return cli;
    }
};

Brain::Brain(std::string base_url) : p_(std::make_unique<Impl>(std::move(base_url))) {}
Brain::~Brain()                                = default;
Brain::Brain(Brain&&) noexcept                 = default;
Brain& Brain::operator=(Brain&&) noexcept      = default;

const std::string& Brain::base_url() const noexcept { return p_->base_url; }

std::expected<BrainReply, AgentError>
Brain::complete(const BrainRequest& req)
{
    auto body = build_request_body(req);
    body["stream"] = false;
    auto cli = p_->make_client(req.timeout);
    auto res = cli.Post(p_->completions_path().c_str(),
                        body.dump(), "application/json");
    if (!res) {
        return std::unexpected(AgentError::brain(
            "POST " + p_->completions_path() + " failed: "
            + httplib::to_string(res.error())));
    }
    if (res->status < 200 || res->status >= 300) {
        return std::unexpected(AgentError::brain(
            "non-2xx body: " + res->body, res->status));
    }
    return parse_response_body(res->body);
}

std::expected<BrainReply, AgentError>
Brain::complete_streaming(const BrainRequest& req, const StreamCallback& on_chunk)
{
    auto body = build_request_body(req);
    body["stream"] = true;

    BrainReply acc;
    std::string buffer; // SSE line accumulator
    bool        done   = false;
    std::string callback_error;

    auto cli = p_->make_client(req.timeout);

    // NOTE: cpp-httplib in this tree exposes no Post overload that takes a
    // ContentReceiver, so we cannot truly stream the response. Issue the
    // POST as a normal request, then walk the buffered body line-by-line
    // and replay the same on_chunk callbacks. Functionally identical for
    // every caller — only end-to-first-token latency is worse.
    httplib::Headers headers{{"Accept", "text/event-stream"}};
    auto res = cli.Post(
        p_->completions_path(), headers, body.dump(), "application/json");

    if (!res) {
        return std::unexpected(AgentError::brain(
            "POST " + p_->completions_path() + " (stream) failed: "
            + httplib::to_string(res.error())));
    }
    if (res->status < 200 || res->status >= 300) {
        return std::unexpected(AgentError::brain(
            "non-2xx stream body: " + res->body, res->status));
    }

    buffer = res->body;
    while (!done) {
        auto nl = buffer.find('\n');
        if (nl == std::string::npos) break;
        std::string line = buffer.substr(0, nl);
        buffer.erase(0, nl + 1);
        auto step = apply_sse_line(line, acc);
        if (step.done) { done = true; break; }
        if (!step.text_delta.empty() && on_chunk) {
            try {
                on_chunk(step.text_delta);
            } catch (const std::exception& e) {
                callback_error = e.what();
                break;
            } catch (...) {
                callback_error = "stream callback threw non-std exception";
                break;
            }
        }
    }
    if (!callback_error.empty()) {
        return std::unexpected(AgentError::brain(
            "stream callback failed: " + callback_error));
    }
    finalize_tool_call_args(acc.tool_calls);
    return acc;
}

std::expected<BrainReply, AgentError>
Brain::chat(const BrainRequest& req, const StreamCallback& on_chunk)
{
    if (req.stream) return complete_streaming(req, on_chunk);
    return complete(req);
}

} // namespace onebit::agent
