// planner.h — multi-step task planning.
// C++ port of jarvis/planner.py (recovered from git history at
// c252174aa~1): decompose a request into 2-5 subtasks, route each to the
// model that best fits it, run each (with its own tool-call sub-loop),
// then synthesize one final answer grounded on the raw subtask/tool
// results rather than each subtask model's own paraphrase.
#pragma once

#include <nlohmann/json.hpp>
#include <string>

#include "rag.h"

namespace jarvis {

// Returns {"plan": [...steps...], "steps": [...per-step detail...], "answer": "..."}.
nlohmann::json run_plan(KnowledgeBase& kb, const std::string& request, bool allow_write);

} // namespace jarvis
