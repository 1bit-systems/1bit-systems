#include "routing.h"

#include <cctype>
#include <httplib.h>
#include <map>

using json = nlohmann::json;

namespace jarvis {

namespace {

// Lowercase, strip everything but alphanumerics. unified_server's
// discovered model ids are inconsistent ("ZR1 1.5B", "zamba2-1.2b-
// instruct-v2-q4_0", "blackmamba-2.8b") and depend on which weight files
// happen to be present on disk at any given moment — hardcoding an exact
// match is fragile. Normalized substring matching against the live
// discovery list is resilient to spacing/casing/hyphen drift instead.
std::string normalize(const std::string& s) {
    std::string out;
    for (char c : s) if (std::isalnum((unsigned char)c)) out += (char)std::tolower((unsigned char)c);
    return out;
}

// Resolves a routing-table hint (e.g. "ZR1-1.5B") to whatever
// unified_server actually calls that model right now, via a live query.
// Falls back to the hint unchanged if unified_server is unreachable or
// nothing matches — same honest fallback as resolve_model() itself.
std::string resolve_unified_model_name(const std::string& hint) {
    try {
        httplib::Client cli(unified_server_url());
        cli.set_connection_timeout(3);
        cli.set_read_timeout(5);
        auto res = cli.Get("/v1/models");
        if (!res || res->status != 200) return hint;
        json models = json::parse(res->body);
        if (!models.contains("data")) return hint;

        std::string norm_hint = normalize(hint);
        for (auto& m : models["data"]) {
            std::string id = m.value("id", "");
            std::string norm_id = normalize(id);
            if (norm_id.find(norm_hint) != std::string::npos || norm_hint.find(norm_id) != std::string::npos)
                return id;
        }
    } catch (...) {
        // fall through
    }
    return hint;
}

} // namespace

std::string unified_server_url() {
    if (const char* env = getenv("UNIFIED_URL")) return env;
    return "http://127.0.0.1:8088";
}

std::string ollama_url() {
    if (const char* env = getenv("OLLAMA_URL")) return env;
    return "http://127.0.0.1:11434";
}

namespace {

// model_id -> Route. Matches models/catalog/README.md's real catalog
// entries for the "unified" targets (the original's literal Python
// strings for these had drifted from what unified_server actually
// discovers by the time this was ported).
const std::map<std::string, Route>& routing_table() {
    static const std::map<std::string, Route> table = {
        {"qwen3:0.6b", {RouteBackend::UnifiedServer, "qwen3:0.6b"}},
        {"qwen3:1.7b", {RouteBackend::UnifiedServer, "qwen3:1.7b"}},
        {"qwen3:4b", {RouteBackend::UnifiedServer, "qwen3:4b"}},
        {"bonsai:1.7b", {RouteBackend::UnifiedServer, "bonsai:1.7b"}},
        {"gemma4:e2b", {RouteBackend::UnifiedServer, "gemma4:e2b"}},
        {"phi4-mini:4b", {RouteBackend::UnifiedServer, "phi4-mini:4b"}},

        {"qwen3vl:4b", {RouteBackend::Vision, "qwen3vl-it:4b"}},
        {"qwen3-vl:4b", {RouteBackend::Vision, "qwen3vl-it:4b"}},

        {"qwen3.5:9b", {RouteBackend::Ollama, "qwen3.5:9b"}},
        {"llama3.1:8b", {RouteBackend::Ollama, "llama3.1:8b"}},
        {"deepseek-r1:8b", {RouteBackend::Ollama, "deepseek-r1:8b"}},
        {"qwen2.5:7b", {RouteBackend::Ollama, "qwen2.5:7b"}},
        {"mistral:7b", {RouteBackend::Ollama, "mistral:7b"}},
        {"gpt-oss:20b", {RouteBackend::Ollama, "gpt-oss:20b"}},
        {"llama3.2-vision", {RouteBackend::Ollama, "llama3.2-vision"}},

        {"zr1:1.5b", {RouteBackend::UnifiedServer, "ZR1-1.5B"}},
        {"zamba2:1.2b", {RouteBackend::UnifiedServer, "Zamba2-1.2B-Instruct-v2"}},
        {"zamba2:2.7b", {RouteBackend::UnifiedServer, "Zamba2-2.7B-Instruct-v2"}},
        {"zamba2:7b", {RouteBackend::UnifiedServer, "Zamba2-7B-Instruct-v2"}},
        {"zaya1:8b", {RouteBackend::UnifiedServer, "ZAYA1-8B"}},
        // blackmamba-1.5b/2.8b intentionally excluded: their GGUF
        // conversion lacks a usable tokenizer (issue #590, still open as
        // of this port) — unified_server falls back to raw token-ID
        // passthrough for them, not fit for a chat-routed model.
    };
    return table;
}

} // namespace

std::vector<std::string> model_ids() {
    std::vector<std::string> ids;
    for (auto& [id, route] : routing_table()) ids.push_back(id);
    return ids;
}

Route resolve_model(const std::string& model_id) {
    const auto& table = routing_table();
    auto it = table.find(model_id);
    if (it != table.end()) return it->second;

    // Unknown id: check whether Ollama has it installed.
    try {
        httplib::Client cli(ollama_url());
        cli.set_connection_timeout(5);
        cli.set_read_timeout(5);
        auto res = cli.Get("/api/tags");
        if (res && res->status == 200) {
            json tags = json::parse(res->body);
            if (tags.contains("models")) {
                for (auto& m : tags["models"]) {
                    std::string name = m.value("name", "");
                    if (!name.empty() && name.find(model_id) != std::string::npos) {
                        return {RouteBackend::Ollama, model_id};
                    }
                }
            }
        }
    } catch (...) {
        // fall through
    }

    // Fall back to unified_server, passed through unchanged — its own
    // model discovery decides whether it actually exists.
    return {RouteBackend::UnifiedServer, model_id};
}

static json build_chat_request(const std::string& model, const json& messages, int max_tokens, float temperature) {
    json body;
    body["model"] = model;
    body["messages"] = messages;
    body["max_tokens"] = max_tokens;
    body["temperature"] = temperature;
    body["stream"] = false;
    return body;
}

nlohmann::json unified_chat(const std::string& model, const json& messages, int max_tokens, float temperature) {
    try {
        std::string resolved_model = resolve_unified_model_name(model);
        httplib::Client cli(unified_server_url());
        cli.set_connection_timeout(5);
        cli.set_read_timeout(180); // cold model-switch on a 7B+ model can exceed a minute
        json body = build_chat_request(resolved_model, messages, max_tokens, temperature);
        auto res = cli.Post("/v1/chat/completions", body.dump(), "application/json");
        if (!res) return {{"error", "unified: connection failed"}};
        if (res->status != 200) return {{"error", "unified: HTTP " + std::to_string(res->status)}};
        return json::parse(res->body);
    } catch (const std::exception& e) {
        return {{"error", std::string("unified: ") + e.what()}};
    }
}

nlohmann::json ollama_chat(const std::string& model, const json& messages, int max_tokens, float temperature) {
    try {
        httplib::Client cli(ollama_url());
        cli.set_connection_timeout(5);
        cli.set_read_timeout(300);
        json body;
        body["model"] = model;
        body["messages"] = messages;
        body["stream"] = false;
        body["options"] = {{"num_predict", max_tokens}, {"temperature", temperature}};
        auto res = cli.Post("/api/chat", body.dump(), "application/json");
        if (!res) return {{"error", "GPU: connection failed"}};
        if (res->status != 200) return {{"error", "GPU: HTTP " + std::to_string(res->status)}};
        json parsed = json::parse(res->body);
        std::string content = parsed.contains("message") ? parsed["message"].value("content", "") : "";
        return {{"response", content}};
    } catch (const std::exception& e) {
        return {{"error", std::string("GPU: ") + e.what()}};
    }
}

} // namespace jarvis
