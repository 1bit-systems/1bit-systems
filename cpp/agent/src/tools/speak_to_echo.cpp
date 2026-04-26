// speak_to_echo — opt-in voice mirror via cpp/echo TTS.
//
// The agent's text reply still goes back through the primary adapter
// (Discord/Telegram). This tool only mirrors the reply to the echo
// daemon's TTS endpoint when the brain explicitly asks for voice (or
// auto_speak is on in config). echo returns a media URL; we surface
// that to the user so they can play/queue it.

#include "onebit/agent/tools/registry.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <string>
#include <string_view>

namespace onebit::agent::tools {

namespace {

constexpr std::size_t kMaxTextBytes = 8 * 1024;
constexpr int         kTtsTimeoutMs = 30 * 1000;

[[nodiscard]] nlohmann::json make_schema(bool auto_speak)
{
    std::string desc =
        "Mirror this reply through the echo TTS daemon. Returns a media "
        "URL that the user can play.";
    if (auto_speak) {
        desc += " auto_speak is ON: agent should call this for every "
                "user-facing reply.";
    } else {
        desc += " Default OFF: only call when the user explicitly asks "
                "for voice (e.g. \"voice:\", \"speak this\", \"read out loud\").";
    }
    return nlohmann::json{
        {"type", "function"},
        {"function", {
            {"name", "speak_to_echo"},
            {"description", desc},
            {"parameters", {
                {"type", "object"},
                {"properties", {
                    {"text", {
                        {"type", "string"},
                        {"description", "Plain-text reply to synthesize."},
                    }},
                    {"voice", {
                        {"type", "string"},
                        {"description",
                         "Optional voice id (echo's default applies if omitted)."},
                    }},
                }},
                {"required", nlohmann::json::array({"text"})},
            }},
        }},
    };
}

struct UrlSplit { std::string host_and_scheme; std::string path; };

[[nodiscard]] UrlSplit split_url(std::string_view url)
{
    UrlSplit s;
    auto scheme_end = url.find("://");
    if (scheme_end == std::string_view::npos) {
        s.host_and_scheme = std::string(url);
        s.path            = "/";
        return s;
    }
    auto path_pos = url.find('/', scheme_end + 3);
    if (path_pos == std::string_view::npos) {
        s.host_and_scheme = std::string(url);
        s.path            = "/";
    } else {
        s.host_and_scheme = std::string(url.substr(0, path_pos));
        s.path            = std::string(url.substr(path_pos));
    }
    return s;
}

} // namespace

ToolDef make_speak_to_echo(std::string echo_url, bool auto_speak)
{
    ToolDef d;
    d.name   = "speak_to_echo";
    d.schema = make_schema(auto_speak);
    d.invoke = [echo_url = std::move(echo_url)](
                   const nlohmann::json& args)
        -> std::expected<ToolResult, AgentError>
    {
        if (echo_url.empty()) {
            return ToolResult{false,
                "speak_to_echo: not configured "
                "(missing [tools.speak_to_echo] echo_url)"};
        }
        const std::string text = args.at("text").get<std::string>();
        if (text.empty()) {
            return ToolResult{false, "bad args: text is empty"};
        }
        if (text.size() > kMaxTextBytes) {
            return ToolResult{false, "bad args: text exceeds 8 KiB cap"};
        }
        const std::string voice =
            args.value("voice", std::string{});

        const auto split = split_url(echo_url);

        nlohmann::json body;
        body["text"] = text;
        if (!voice.empty()) body["voice"] = voice;

        httplib::Client cli(split.host_and_scheme);
        cli.set_connection_timeout(0, 3 * 1000 * 1000); // 3 s
        cli.set_read_timeout(kTtsTimeoutMs / 1000, 0);
        cli.set_write_timeout(5, 0);

        auto res = cli.Post(split.path, body.dump(), "application/json");
        if (!res) {
            return ToolResult{false,
                "speak_to_echo: POST " + split.path + " failed: " +
                httplib::to_string(res.error())};
        }
        if (res->status < 200 || res->status >= 300) {
            return ToolResult{false,
                "speak_to_echo: echo returned non-2xx (" +
                std::to_string(res->status) + "): " + res->body};
        }

        std::string media_url;
        try {
            auto j = nlohmann::json::parse(res->body);
            if (j.contains("url")) {
                media_url = j.at("url").get<std::string>();
            } else if (j.contains("audio_url")) {
                media_url = j.at("audio_url").get<std::string>();
            }
        } catch (...) {
            // echo may also reply with a plain Location header or text.
        }
        if (media_url.empty()) {
            // Surface raw body as fallback so the brain can still inform
            // the user that voice was synthesized.
            return ToolResult{true,
                "voice synthesized; echo replied: " + res->body};
        }
        return ToolResult{true, "voice ready: " + media_url};
    };
    return d;
}

} // namespace onebit::agent::tools
