// token_router.h — Central routing engine: detects backends, selects models, dispatches inference.
// Part of the unified zaya_server binary. No external deps.
#pragma once
#include "backend.h"
#include <vector>
#include <string>
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <cmath>

enum class RouteStrategy {
    AUTO,
    CASCADE,
    SPEC_DECODE,
    CONTENT,
    PASSTHROUGH,
};

struct TokenRouter {
    std::vector<InferenceBackend*> backends;
    InferenceBackend* primary = nullptr;
    std::vector<ModelConfig> loaded_models;
    RouteStrategy strategy = RouteStrategy::AUTO;

    bool init() {
        fprintf(stderr, "\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557\n");
        fprintf(stderr, "\u2551  1bit TokenRouter — Multi-Backend       \u2551\n");
        fprintf(stderr, "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d\n\n");
        fprintf(stderr, "Detecting backends...\n");
        backends = detect_backends();
        if (backends.empty()) { fprintf(stderr, "  FATAL: No backends available!\n"); return false; }
        fprintf(stderr, "\nAvailable backends:\n");
        for (auto* b : backends) {
            fprintf(stderr, "  %-12s %-30s ~%.0f tok/s  %s\n",
                b->name(), b->is_available() ? "ready" : "unavailable",
                b->estimated_tok_s(), b->is_coherent() ? "[coherent]" : "[raw]");
        }
        primary = select_best_backend();
        if (primary) fprintf(stderr, "\n  Primary: %s (%.0f tok/s)\n\n", primary->name(), primary->estimated_tok_s());
        return true;
    }

    bool load_model(const ModelConfig& cfg) {
        if (!primary || !primary->is_available()) {
            for (auto* b : backends) { if (b->is_available()) { primary = b; break; } }
        }
        if (!primary) { fprintf(stderr, "  No backend available!\n"); return false; }
        fprintf(stderr, "Loading %s on %s backend...\n", cfg.model_name.c_str(), primary->name());
        if (!primary->load_model(cfg)) {
            for (auto* b : backends) {
                if (b != primary && b->is_available()) {
                    fprintf(stderr, "  Falling back to %s...\n", b->name());
                    if (b->load_model(cfg)) { primary = b; loaded_models.push_back(cfg); return true; }
                }
            }
            return false;
        }
        loaded_models.push_back(cfg);
        return true;
    }

    InferenceResult infer(const std::vector<int>& prompt_tokens, int max_tokens, RouteStrategy strat = RouteStrategy::AUTO) {
        InferenceResult result;
        if (!primary) { result.text = "[no backend]"; return result; }
        RouteStrategy use_strat = (strat == RouteStrategy::AUTO) ? strategy : strat;
        auto t0 = std::chrono::high_resolution_clock::now();
        primary->reset_state();
        std::vector<int> out_tokens;
        int last_token = prompt_tokens.empty() ? 2 : prompt_tokens.back();

        switch (use_strat) {
        case RouteStrategy::PASSTHROUGH:
        case RouteStrategy::AUTO:
            for (int i = 0; i < max_tokens; i++) {
                int next = primary->forward(last_token, i);
                out_tokens.push_back(next);
                last_token = next;
                if (next == 106) break;
            }
            break;

        case RouteStrategy::CASCADE: {
            InferenceBackend* fallback = nullptr;
            for (auto* b : backends) { if (b != primary && b->is_available() && b->is_coherent()) { fallback = b; break; } }
            bool on_primary = true;
            InferenceBackend* current = primary;
            for (int i = 0; i < max_tokens; i++) {
                int next = current->forward(on_primary ? last_token : (out_tokens.empty() ? last_token : out_tokens.back()), i);
                out_tokens.push_back(next);
                if (on_primary && fallback && out_tokens.size() >= 4) {
                    bool repeating = true;
                    for (size_t j = out_tokens.size() - 3; j < out_tokens.size(); j++)
                        if (out_tokens[j] != out_tokens.back()) { repeating = false; break; }
                    if (repeating) {
                        fprintf(stderr, "  [cascade] switching to fallback\n");
                        on_primary = false; current = fallback;
                        fallback->reset_state();
                        for (size_t j = 0; j < out_tokens.size() - 1; j++) fallback->forward(out_tokens[j], (int)j);
                    }
                }
                last_token = next;
                if (next == 106) break;
            }
            break;
        }

        case RouteStrategy::SPEC_DECODE: {
            InferenceBackend* drafter = primary;
            InferenceBackend* verifier = nullptr;
            for (auto* b : backends) { if (b != drafter && b->is_available()) { verifier = b; break; } }
            if (!verifier) {
                for (int i = 0; i < max_tokens; i++) {
                    int next = drafter->forward(last_token, i);
                    out_tokens.push_back(next); last_token = next;
                    if (next == 106) break;
                }
            } else {
                int n_draft = 4, generated = 0;
                while (generated < max_tokens) {
                    std::vector<int> drafts;
                    int draft_last = last_token;
                    for (int d = 0; d < n_draft && generated + d < max_tokens; d++) {
                        int next = drafter->forward(draft_last, generated + d);
                        drafts.push_back(next); draft_last = next;
                        if (next == 106) break;
                    }
                    if (!drafts.empty()) {
                        int verified = verifier->forward(last_token, generated);
                        if (verified == drafts[0]) {
                            out_tokens.push_back(drafts[0]); last_token = drafts[0]; generated++;
                            for (size_t d = 1; d < drafts.size() && generated < max_tokens; d++) {
                                int v = verifier->forward(drafts[d-1], generated);
                                if (v == drafts[d]) { out_tokens.push_back(drafts[d]); last_token = drafts[d]; generated++; }
                                else { out_tokens.push_back(v); last_token = v; generated++; break; }
                            }
                        } else { out_tokens.push_back(verified); last_token = verified; generated++; }
                        if (last_token == 106) break;
                    } else break;
                }
            }
            break;
        }

        case RouteStrategy::CONTENT:
            for (int i = 0; i < max_tokens; i++) {
                int next = primary->forward(last_token, i);
                out_tokens.push_back(next); last_token = next;
                if (next == 106) break;
            }
            break;
        }

        float ms = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
        result.tokens = out_tokens; result.gen_ms = ms;
        result.tok_s = ms > 0 ? out_tokens.size() / (ms / 1000.0f) : 0;
        return result;
    }
};

inline bool should_use_large_model(const std::string& user_message) {
    static const std::vector<std::string> gpu_keywords = {
        "code", "explain", "analyze", "write", "implement", "debug",
        "refactor", "function", "algorithm", "bug", "error", "review",
        "optimize", "design", "architecture", "test",
    };
    if (user_message.size() > 800) return true;
    std::string lower = user_message;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    for (const auto& kw : gpu_keywords)
        if (lower.find(kw) != std::string::npos) return true;
    return false;
}
