// routing.h — model routing table + backend HTTP clients.
// C++ port of jarvis/routing.py (recovered from git history at
// c252174aa~1), simplified: the original's separate NPU/FLM bridge
// (127.0.0.1:52625) no longer exists in this project (FLM was fully
// removed, see model_router.h) — every 1bit-catalog model now routes
// through unified_server's own auto-backend-selecting /v1/chat/completions
// instead. Models not in the catalog still fall through to Ollama, same as
// the original.
#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace jarvis {

// Overridable via env UNIFIED_URL / OLLAMA_URL; defaults match the
// original's hardcoded values (and this project's own default ports).
std::string unified_server_url();
std::string ollama_url();

enum class RouteBackend { UnifiedServer, Vision, Ollama };

struct Route {
    RouteBackend backend;
    std::string target_model; // model name/id to send to the backend
};

// All model ids known to the static routing table (used by the list_models
// tool and the /v1/models route).
std::vector<std::string> model_ids();

// Resolves a model id to a Route. For unknown ids: queries Ollama's
// /api/tags for a substring match; on any failure or no match, falls back
// to UnifiedServer with the id passed through unchanged (unified_server's
// own model discovery decides whether it actually exists).
Route resolve_model(const std::string& model_id);

// POST {unified_server}/v1/chat/completions. Returns the parsed JSON
// response, or {"error": "unified: <reason>"} on failure.
nlohmann::json unified_chat(const std::string& model, const nlohmann::json& messages,
                             int max_tokens = 256, float temperature = 0.7f);

// POST {ollama}/api/chat (native chat endpoint — preserves full message
// history, unlike /api/generate). Returns {"response": <content>} or
// {"error": "GPU: <reason>"}.
nlohmann::json ollama_chat(const std::string& model, const nlohmann::json& messages,
                            int max_tokens = 256, float temperature = 0.7f);

} // namespace jarvis
