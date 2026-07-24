// tools.h — tool-calling loop + permission gate.
// C++ port of jarvis/tools.py (recovered from git history at c252174aa~1).
#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <string>

#include "rag.h"

namespace jarvis {

enum class ToolClass { Safe, Sensitive };

struct ToolCall {
    std::string name;
    nlohmann::json arguments;
};

// System prompt instructing the model on the TOOL_CALL: {...} protocol and
// the exact four available tools. Prepended when tool use is enabled and no
// existing system message already mentions TOOL_CALL.
extern const char* SYSTEM_PROMPT_TOOLS;

// Detects a "TOOL_CALL: {...}" directive via brace-depth matching (not a
// greedy regex-to-last-brace — must tolerate a model emitting more than one
// TOOL_CALL line, matching only the first well-formed one). Returns
// nullopt if no call is found or the JSON is malformed / missing "name".
std::optional<ToolCall> parse_tool_call(const std::string& content);

// Runs a tool subject to the permission gate: "sensitive"-class tools
// (currently just add_note) require allow_write=true, else the call is
// denied. Every invocation — allowed, denied, or throwing — is appended to
// <kb root>/tools/audit.log as one JSON line. Returns the tool's JSON
// result (or an {"error": ...} object) and whether it was allowed to run.
struct ToolRunResult {
    nlohmann::json result;
    bool allowed;
};
ToolRunResult run_tool(KnowledgeBase& kb, const std::string& name,
                        const nlohmann::json& arguments, bool allow_write);

// Formats the tool result (or denial reason) as a follow-up user message
// telling the model how to use it in its final answer.
std::string format_tool_followup(const nlohmann::json& result, bool allowed);

} // namespace jarvis
