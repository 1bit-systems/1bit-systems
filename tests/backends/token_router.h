// token_router.h — Central routing engine: detects backends, selects models, dispatches inference.
// Part of the unified zaya_server binary. No external deps.
#pragma once
#include "backend.h"
#include "parallel_moe.h"
#include <vector>
#include <string>
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <cmath>

// ─── Routing strategy ───────────────────────────────────────────────
enum class RouteStrategy {
    AUTO,           // Pick fastest available backend
    CASCADE,        // Per-token: stream from fast, fall back on low confidence
    SPEC_DECODE,    // NPU drafts → GPU verifies
    CONTENT,        // Keyword-based: small model vs large model
    PARALLEL_MOE,   // GPU attention + NPU experts pipelined across layers
    PASSTHROUGH,    // Fixed single backend
};

// ─── TokenRouter: one engine to rule them all ───────────────────────
struct TokenRouter {
    std::vector<InferenceBackend*> backends;
    InferenceBackend* primary = nullptr;
    InferenceBackend* gpu_backend = nullptr;
    InferenceBackend* npu_backend = nullptr;
    std::vector<ModelConfig> loaded_models;
    RouteStrategy strategy = RouteStrategy::AUTO;
    MoePipeline moe_pipeline_;

    // ── Initialize: detect hardware, load backends ─────────────────
    bool init() {
        fprintf(stderr, "\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557\n");
        fprintf(stderr, "\u2551  1bit TokenRouter — Multi-Backend       \u2551\n");
        fprintf(stderr, "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d\n\n");
        fprintf(stderr, "Detecting backends...\n");

        backends = detect_backends();
        if (backends.empty()) {
            fprintf(stderr, "  FATAL: No backends available!\n");
            return false;
        }

        fprintf(stderr, "\nAvailable backends:\n");
        for (auto* b : backends) {
            // estimated_tok_s() is a prior used for backend selection, not a
            // measurement (issue #231). Label it so the startup banner never
            // reads as a validated throughput figure.
            fprintf(stderr, "  %-12s %-30s ~%.0f tok/s (est.)  %s\n",
                b->name(), b->is_available() ? "ready" : "unavailable",
                b->estimated_tok_s(),
                b->is_coherent() ? "[coherent]" : "[raw]");
        }

        primary = select_best_backend(&backends);
        if (primary) {
            fprintf(stderr, "\n  Primary: %s (~%.0f tok/s est.)\n\n",
                primary->name(), primary->estimated_tok_s());
        }

        // ── Detect GPU and NPU for parallel MoE + hybrid ──────────
        for (auto* b : backends) {
            if (b->is_available()) {
                if (b->type() == BackendType::HIP_GPU || b->type() == BackendType::VULKAN || b->type() == BackendType::ZINC_GPU) {
                    if (!gpu_backend) gpu_backend = b;
                }
                if (b->type() == BackendType::NPU_XRT || b->type() == BackendType::NPU_FLM) {
                    if (!npu_backend) npu_backend = b;
                }
            }
        }
        if (gpu_backend && npu_backend) {
            fprintf(stderr, "  GPU+NPU Hybrid (Zero-Copy DMA): active\n");
            fprintf(stderr, "    GPU: %s (attention + dense layers)\n", gpu_backend->name());
            fprintf(stderr, "    NPU: %s (expert FFNs + offload)\n", npu_backend->name());
            fprintf(stderr, "    Memory: unified LPDDR5X — zero-copy DMA between GPU/NPU\n");
            fprintf(stderr, "    Estimated speedup: %.1fx (pipeline overlap + zero-copy)\n\n",
                    moe_pipeline_.estimated_speedup());
        }
        return true;
    }

    // ── Load a model onto the best available backend ───────────────
    bool load_model(const ModelConfig& cfg) {
        if (!primary || !primary->is_available()) {
            // Fall back to first available
            for (auto* b : backends) {
                if (b->is_available()) { primary = b; break; }
            }
        }
        if (!primary) {
            fprintf(stderr, "  No backend available for model loading!\n");
            return false;
        }

        fprintf(stderr, "Loading %s (H=%d L=%d V=%d) on %s backend...\n",
            cfg.model_name.c_str(), cfg.hidden_size, cfg.num_layers, cfg.vocab_size, primary->name());

        bool loaded = false;
        try {
            loaded = primary->load_model(cfg);
        } catch (std::exception& e) {
            fprintf(stderr, "  %s: exception: %s — trying next backend\n", primary->name(), e.what());
            loaded = false;
        }

        if (!loaded) {
            // Try next backend
            for (auto* b : backends) {
                if (b != primary && b->is_available()) {
                    fprintf(stderr, "  Falling back to %s...\n", b->name());
                    if (b->load_model(cfg)) {
                        primary = b;
                        loaded_models.push_back(cfg);
                        return true;
                    }
                }
            }
            fprintf(stderr, "  All backends failed to load model!\n");
            return false;
        }

        loaded_models.push_back(cfg);

        // ── Initialize MoE pipeline if GPU+NPU are available ─────
        if (gpu_backend && npu_backend &&
            gpu_backend->is_available() && npu_backend->is_available()) {
            if (cfg.num_experts > 1) {
                fprintf(stderr, "  Initializing GPU+NPU MoE pipeline (%d experts)...\n",
                        cfg.num_experts);
                moe_pipeline_.init(gpu_backend, npu_backend, cfg);
            }
        }

        return true;
    }

    // ── Run inference with routing strategy ────────────────────────
    InferenceResult infer(const std::vector<int>& prompt_tokens, int max_tokens,
                          RouteStrategy strat = RouteStrategy::AUTO)
    {
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
                float best_tok_s = 0;
                for (auto* b : backends) {
                    if (b != primary && b->is_available() && b->is_coherent()
                        && b->estimated_tok_s() > best_tok_s) {
                        fallback = b;
                        best_tok_s = b->estimated_tok_s();
                    }
                }
                bool on_primary = true;
                InferenceBackend* current = primary;
                for (int i = 0; i < max_tokens; i++) {
                    int next = current->forward(on_primary ? last_token : (out_tokens.empty() ? last_token : out_tokens.back()), i);
                    out_tokens.push_back(next);
                    if (on_primary && fallback && out_tokens.size() >= 6) {
                        // Detect any 3+ repeated token (AAA...) or alternating pattern (ABAB...)
                        bool repeating = false;
                        size_t n = out_tokens.size();
                        // Check AAA pattern (last 3 identical)
                        if (out_tokens[n-1] == out_tokens[n-2] && out_tokens[n-2] == out_tokens[n-3])
                            repeating = true;
                        // Check ABAB pattern (last 4 alternating)
                        if (!repeating && n >= 4 &&
                            out_tokens[n-1] == out_tokens[n-3] &&
                            out_tokens[n-2] == out_tokens[n-4])
                            repeating = true;
                        if (repeating) {
                            fprintf(stderr, "  [cascade] switching to fallback\n");
                            on_primary = false; current = fallback;
                            fallback->reset_state();
                            for (size_t j = 0; j < out_tokens.size() - 1; j++)
                                fallback->forward(out_tokens[j], (int)j);
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
                for (auto* b : backends) {
                    if (b != drafter && b->is_available()) { verifier = b; break; }
                }
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
                                    if (v == drafts[d]) {
                                        out_tokens.push_back(drafts[d]); last_token = drafts[d]; generated++;
                                    } else {
                                        out_tokens.push_back(v); last_token = v; generated++; break;
                                    }
                                }
                            } else {
                                out_tokens.push_back(verified); last_token = verified; generated++;
                            }
                            if (last_token == 106) break;
                        } else break;
                    }
                }
                break;
            }

            case RouteStrategy::CONTENT:
                for (int i = 0; i < max_tokens; i++) {
                    int next = primary->forward(last_token, i);
                    out_tokens.push_back(next);
                    last_token = next;
                    if (next == 106) break;
                }
                break;

            case RouteStrategy::PARALLEL_MOE:
                if (moe_pipeline_.enabled_) {
                    fprintf(stderr, "  [parallel-moe] GPU+NPU pipelined inference\n");
                    result = moe_pipeline_.infer_pipelined(prompt_tokens, max_tokens);
                    float ms = std::chrono::duration<float, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                    result.gen_ms = ms;
                    result.tok_s = ms > 0 ? result.tokens.size() / (ms / 1000.0f) : 0;
                    return result;
                }
                fprintf(stderr, "  [parallel-moe] pipeline not available, falling back\n");
                for (int i = 0; i < max_tokens; i++) {
                    int next = primary->forward(last_token, i);
                    out_tokens.push_back(next);
                    last_token = next;
                    if (next == 106) break;
                }
                break;
        }

        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        result.tokens = out_tokens;
        result.gen_ms = ms;
        result.tok_s = ms > 0 ? out_tokens.size() / (ms / 1000.0f) : 0;
        return result;
    }
};

// ─── Content-based model selection ──────────────────────────────────
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
