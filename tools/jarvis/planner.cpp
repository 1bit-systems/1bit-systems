#include "planner.h"

#include <regex>
#include <sstream>

#include "routing.h"
#include "tools.h"

using json = nlohmann::json;

namespace jarvis {
namespace {

const char* PLANNER_MODEL = "qwen3:0.6b";
const char* SYNTH_MODEL = "qwen3.6:35b";

const char* PLAN_PROMPT_TMPL =
    "Break the following request into 2-5 concrete subtasks needed to fully "
    "answer it. Reply with ONLY a JSON list of strings, nothing else.\n\n"
    "Request: {request}";

// (regex, model_id) — checked in order, first match wins.
const std::vector<std::pair<std::regex, std::string>>& subtask_hints() {
    static const std::vector<std::pair<std::regex, std::string>> hints = {
        {std::regex(R"(\b(image|photo|picture|diagram|screenshot)\b)", std::regex::icase), "qwen3vl:4b"},
        {std::regex(R"(\b(reason|analy[sz]e|compare|plan|strategy|deep|complex)\b)", std::regex::icase), SYNTH_MODEL},
    };
    return hints;
}

std::string pick_model(const std::string& step) {
    for (auto& [re, model] : subtask_hints()) {
        if (std::regex_search(step, re)) return model;
    }
    return PLANNER_MODEL;
}

// Single-turn chat via the routed backend. Returns the reply text, or
// "[error: <msg>]" on failure — matches the original's error sentinel so
// callers can surface a visible failure rather than silently continuing.
std::string chat_one(const std::string& model_id, const json& messages, int max_tokens = 256) {
    Route route = resolve_model(model_id);
    if (route.backend == RouteBackend::Ollama) {
        json result = ollama_chat(route.target_model, messages, max_tokens);
        if (result.contains("error")) return "[error: " + result["error"].get<std::string>() + "]";
        return result.value("response", "");
    }
    json result = unified_chat(route.target_model, messages, max_tokens);
    if (result.contains("error")) return "[error: " + result["error"].get<std::string>() + "]";
    if (result.contains("choices") && !result["choices"].empty())
        return result["choices"][0]["message"].value("content", "");
    return "[error: malformed response]";
}

std::vector<std::string> make_plan(const std::string& request) {
    std::string prompt = std::regex_replace(std::string(PLAN_PROMPT_TMPL), std::regex(R"(\{request\})"), request);
    json messages = json::array({{{"role", "user"}, {"content", prompt}}});
    std::string reply = chat_one(PLANNER_MODEL, messages, 200);

    std::smatch m;
    static const std::regex bracket_re(R"(\[[\s\S]*\])");
    if (std::regex_search(reply, m, bracket_re)) {
        try {
            json arr = json::parse(m.str());
            std::vector<std::string> steps;
            for (auto& item : arr) {
                if (item.is_string() && !item.get<std::string>().empty()) steps.push_back(item.get<std::string>());
                if (steps.size() >= 5) break;
            }
            if (!steps.empty()) return steps;
        } catch (...) {
            // fall through to single-step fallback
        }
    }
    return {request};
}

json run_step(KnowledgeBase& kb, const std::string& step, bool allow_write) {
    std::string model = pick_model(step);
    json messages = json::array({
        {{"role", "system"}, {"content", SYSTEM_PROMPT_TOOLS}},
        {{"role", "user"}, {"content", step}},
    });
    std::string reply = chat_one(model, messages);

    json step_result = {{"step", step}, {"model", model}};

    auto call = parse_tool_call(reply);
    if (call) {
        step_result["tool_call"] = {{"name", call->name}, {"arguments", call->arguments}};
        ToolRunResult tr = run_tool(kb, call->name, call->arguments, allow_write);
        step_result["tool_allowed"] = tr.allowed;

        messages.push_back({{"role", "assistant"}, {"content", reply}});
        messages.push_back({{"role", "user"}, {"content", format_tool_followup(tr.result, tr.allowed)}});
        std::string final_reply = chat_one(model, messages);
        step_result["output"] = final_reply;
        step_result["_tool_result"] = tr.result;
    } else {
        step_result["output"] = reply;
    }
    return step_result;
}

std::string format_step_for_synthesis(const json& step_result) {
    std::ostringstream out;
    out << "- " << step_result.value("step", "") << ": " << step_result.value("output", "");
    if (step_result.contains("tool_call") && step_result.value("tool_allowed", false)) {
        out << " (tool " << step_result["tool_call"].value("name", "")
            << " actually returned: " << step_result.value("_tool_result", json::object()).dump() << ")";
    }
    return out.str();
}

} // namespace

json run_plan(KnowledgeBase& kb, const std::string& request, bool allow_write) {
    std::vector<std::string> plan = make_plan(request);

    json steps = json::array();
    std::ostringstream synthesis_input;
    for (auto& step : plan) {
        json step_result = run_step(kb, step, allow_write);
        synthesis_input << format_step_for_synthesis(step_result) << "\n";
        step_result.erase("_tool_result"); // internal only, not part of the public shape
        steps.push_back(step_result);
    }

    std::ostringstream synth_prompt;
    synth_prompt << "Original request: " << request << "\n\nSubtask results:\n"
                 << synthesis_input.str() << "\nGive one concise final answer to the original request.";
    json synth_messages = json::array({{{"role", "user"}, {"content", synth_prompt.str()}}});
    std::string answer = chat_one(SYNTH_MODEL, synth_messages, 400);

    json plan_arr = json::array();
    for (auto& s : plan) plan_arr.push_back(s);

    return {{"plan", plan_arr}, {"steps", steps}, {"answer", answer}};
}

} // namespace jarvis
