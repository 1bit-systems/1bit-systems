#include "tools.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <map>

#include "routing.h"

using json = nlohmann::json;

namespace jarvis {

const char* SYSTEM_PROMPT_TOOLS =
    "You have access to exactly four tools. To use one, respond with ONLY a "
    "single line of the form:\n"
    "TOOL_CALL: {\"name\": \"<tool_name>\", \"arguments\": {...}}\n\n"
    "Available tools:\n"
    "- search_knowledge(query): search the local knowledge base.\n"
    "- get_time(): current UTC time.\n"
    "- list_models(): list available model ids.\n"
    "- add_note(title, content): save a note to the knowledge base (requires write permission).\n\n"
    "For anything else — math, reasoning, writing, general knowledge — answer directly, "
    "do not emit a TOOL_CALL.";

std::optional<ToolCall> parse_tool_call(const std::string& content) {
    static const std::string marker = "TOOL_CALL:";
    size_t marker_pos = content.find(marker);
    if (marker_pos == std::string::npos) return std::nullopt;

    size_t brace_start = content.find('{', marker_pos);
    if (brace_start == std::string::npos) return std::nullopt;

    // Brace-depth match to find the corresponding closing brace, tolerant of
    // braces inside string literals within the JSON.
    int depth = 0;
    bool in_string = false;
    bool escape = false;
    size_t i = brace_start;
    size_t brace_end = std::string::npos;
    for (; i < content.size(); i++) {
        char c = content[i];
        if (escape) { escape = false; continue; }
        if (c == '\\') { escape = true; continue; }
        if (c == '"') { in_string = !in_string; continue; }
        if (in_string) continue;
        if (c == '{') depth++;
        else if (c == '}') {
            depth--;
            if (depth == 0) { brace_end = i; break; }
        }
    }
    if (brace_end == std::string::npos) return std::nullopt;

    std::string blob = content.substr(brace_start, brace_end - brace_start + 1);
    json parsed;
    try {
        parsed = json::parse(blob);
    } catch (...) {
        return std::nullopt;
    }
    if (!parsed.contains("name") || !parsed["name"].is_string()) return std::nullopt;

    ToolCall call;
    call.name = parsed["name"].get<std::string>();
    call.arguments = parsed.value("arguments", json::object());
    return call;
}

namespace {

json tool_search_knowledge(KnowledgeBase& kb, const json& args) {
    std::string query = args.value("query", "");
    auto results = kb.search(query, 5);
    json arr = json::array();
    for (auto& r : results) arr.push_back({{"path", r.path}, {"title", r.title}, {"score", r.score}, {"snippet", r.snippet}});
    return {{"results", arr}};
}

json tool_get_time(KnowledgeBase&, const json&) {
    return {{"time", iso8601_now()}};
}

json tool_list_models(KnowledgeBase&, const json&) {
    return {{"models", model_ids()}};
}

json tool_add_note(KnowledgeBase& kb, const json& args) {
    std::string title = args.value("title", "note");
    std::string content = args.value("content", "");
    std::string path = kb.add_document(title, content);
    return {{"saved", path}};
}

struct ToolSpec {
    ToolClass cls;
    json (*fn)(KnowledgeBase&, const json&);
};

const std::map<std::string, ToolSpec>& tool_registry() {
    static const std::map<std::string, ToolSpec> reg = {
        {"search_knowledge", {ToolClass::Safe, tool_search_knowledge}},
        {"get_time", {ToolClass::Safe, tool_get_time}},
        {"list_models", {ToolClass::Safe, tool_list_models}},
        {"add_note", {ToolClass::Sensitive, tool_add_note}},
    };
    return reg;
}

void audit(KnowledgeBase& kb, const std::string& tool, const json& arguments, bool allowed,
           const std::string& result_summary) {
    std::filesystem::path log_path = kb.root() / "tools" / "audit.log";
    std::ofstream f(log_path, std::ios::binary | std::ios::app);
    if (!f) return;
    json entry = {
        {"ts", iso8601_now()},
        {"tool", tool},
        {"arguments", arguments},
        {"allowed", allowed},
        {"result_summary", result_summary.substr(0, 200)},
    };
    f << entry.dump() << "\n";
}

} // namespace

ToolRunResult run_tool(KnowledgeBase& kb, const std::string& name, const json& arguments, bool allow_write) {
    const auto& reg = tool_registry();
    auto it = reg.find(name);
    if (it == reg.end()) {
        json err = {{"error", "unknown tool: " + name}};
        audit(kb, name, arguments, false, err.dump());
        return {err, false};
    }

    if (it->second.cls == ToolClass::Sensitive && !allow_write) {
        json err = {{"error", "write permission denied for tool: " + name}};
        audit(kb, name, arguments, false, err.dump());
        return {err, false};
    }

    try {
        json result = it->second.fn(kb, arguments);
        audit(kb, name, arguments, true, result.dump());
        return {result, true};
    } catch (const std::exception& e) {
        json err = {{"error", std::string(e.what())}};
        audit(kb, name, arguments, true, err.dump());
        return {err, true};
    }
}

std::string format_tool_followup(const json& result, bool allowed) {
    if (!allowed) {
        std::string reason = result.value("error", "denied");
        return "The tool call was not permitted (" + reason +
               "). Do not retry it — answer from your own knowledge instead, "
               "and mention that the action requires write permission if relevant.";
    }
    return "The tool returned this result:\n" + result.dump() +
           "\nUse this result to answer the original question.";
}

} // namespace jarvis
