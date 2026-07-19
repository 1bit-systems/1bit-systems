// strategy_engine.cpp — Per-token routing strategy implementations
//
// Each strategy's route() call is a pure function — no side effects,
// no locks, no I/O. The agent watchdog modifies agent state via atomics;
// route() reads atomics and is lock-free.

#include "strategy_engine.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>

// ══════════════════════════════════════════════════════════════════════
//  CascadeConfig
// ══════════════════════════════════════════════════════════════════════

RoutingDecision CascadeConfig::route(const TokenContext& ctx) const noexcept {
    if (agent) {
        agent->total_tokens_routed.fetch_add(1, std::memory_order_relaxed);
    }

    // After N tokens, always use large backend
    if (ctx.position >= min_context_for_large) {
        return RoutingDecision{large_backend, false, "", 0};
    }

    // Determine threshold: use agent override if set, otherwise config default
    double threshold = confidence_threshold;
    if (agent) {
        // acquire: ensures visibility of watchdog writes (fixes #364)
        double override = agent->cascade_threshold_override.load(std::memory_order_acquire);
        if (override > -900.0) {  // sentinel check
            threshold = override;
        }
    }

    // Check log-prob against threshold
    if (ctx.log_prob > -1e9 && ctx.log_prob < threshold) {
        if (agent) {
            agent->cascade_count.fetch_add(1, std::memory_order_relaxed);
        }
        return RoutingDecision{large_backend, false, "", 0};
    }

    // Default: fast small backend
    return RoutingDecision{small_backend, false, "", 0};
}

// ══════════════════════════════════════════════════════════════════════
//  PerformanceConfig
// ══════════════════════════════════════════════════════════════════════

std::string PerformanceConfig::best_backend_for_model(const std::string& model) const {
    if (live_table.empty()) {
        return default_backend;
    }

    // Case-insensitive substring match against performance table
    std::string lower;
    lower.resize(model.size());
    std::transform(model.begin(), model.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    for (const auto& route : live_table) {
        if (lower.find(route.model_pattern) != std::string::npos) {
            return route.backend;
        }
    }

    return default_backend;
}

RoutingDecision PerformanceConfig::route(const TokenContext& ctx) const noexcept {
    (void)ctx;
    return RoutingDecision{default_backend, false, "", 0};
}

// ══════════════════════════════════════════════════════════════════════
//  SpecDecodeConfig
// ══════════════════════════════════════════════════════════════════════

RoutingDecision SpecDecodeConfig::route(const TokenContext& ctx) const noexcept {
    // For very short sequences, spec decode overhead isn't worth it
    if (ctx.position < 4) {
        return RoutingDecision{draft_backend, false, "", 0};
    }

    // Use dynamic n_draft from agent if available, else config default
    int nd = n_draft;
    if (agent) {
        // acquire: ensures visibility of watchdog writes (fixes #364)
        int dynamic = agent->dynamic_n_draft.load(std::memory_order_acquire);
        if (dynamic > 0) nd = dynamic;
    }

    return RoutingDecision{
        target_backend,
        true,                    // speculative
        draft_backend,
        nd
    };
}

// ══════════════════════════════════════════════════════════════════════
//  ContentRouterConfig
// ══════════════════════════════════════════════════════════════════════

bool ContentRouterConfig::has_gpu_keywords(const std::string& text) const noexcept {
    if (text.empty()) return false;

    std::string lower;
    lower.resize(text.size());
    std::transform(text.begin(), text.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    for (const auto& kw : gpu_keywords) {
        if (lower.find(kw) != std::string::npos) {
            return true;
        }
    }
    return false;
}

RoutingDecision ContentRouterConfig::route(const TokenContext& ctx) const noexcept {
    // After threshold, always use large backend
    if (ctx.position >= max_small_tokens) {
        return RoutingDecision{large_backend, false, "", 0};
    }

    // Check message for GPU keywords
    if (has_gpu_keywords(ctx.message_text)) {
        return RoutingDecision{large_backend, false, "", 0};
    }

    // Long input → GPU
    if (ctx.message_text.size() > 800) {
        return RoutingDecision{large_backend, false, "", 0};
    }

    // Default: fast small backend
    return RoutingDecision{small_backend, false, "", 0};
}

// ══════════════════════════════════════════════════════════════════════
//  AdaptiveConfig — The "true agent" strategy
// ══════════════════════════════════════════════════════════════════════

RoutingDecision AdaptiveConfig::route(const TokenContext& ctx) const noexcept {
    double npu_share = 0.7;
    bool npu_off = false;
    bool gpu_off = false;
    double threshold = -2.5;

    if (agent) {
        // acquire: ensures visibility of watchdog writes (fixes #364)
        npu_share = agent->npu_load_share.load(std::memory_order_acquire);
        npu_off = agent->npu_disabled.load(std::memory_order_acquire);
        gpu_off = agent->gpu_disabled.load(std::memory_order_acquire);
        threshold = agent->adaptive_cascade_threshold.load(std::memory_order_acquire);
    }

    // ── 1. Check if any backend is disabled by watchdog ──
    if (npu_off && gpu_off) {
        return RoutingDecision{fallback_backend, false, "", 0};
    }
    if (npu_off) {
        return RoutingDecision{secondary_backend, false, "", 0};
    }
    if (gpu_off) {
        return RoutingDecision{primary_backend, false, "", 0};
    }

    // ── 2. Cascade: check confidence with dynamic threshold ──
    if (ctx.log_prob > -1e9 && ctx.log_prob < threshold) {
        return RoutingDecision{secondary_backend, false, "", 0};
    }

    // ── 3. Load-balanced routing ──
    if (agent && monitor) {
        auto* npu_metrics = monitor->for_backend(primary_backend);
        auto* gpu_metrics = monitor->for_backend(secondary_backend);
        if (npu_metrics && gpu_metrics) {
            double npu_tps = npu_metrics->tokens_per_second.load(std::memory_order_relaxed);
            double gpu_tps = gpu_metrics->tokens_per_second.load(std::memory_order_relaxed);

            // Adjust share based on relative throughput
            if (gpu_tps > npu_tps * 1.5) npu_share = 0.3;
            else if (npu_tps > gpu_tps * 1.5) npu_share = 0.8;

            // Use position mod for deterministic load balancing
            double pos_norm = (ctx.position % 100) / 100.0;
            if (pos_norm < npu_share) {
                return RoutingDecision{primary_backend, false, "", 0};
            }
            return RoutingDecision{secondary_backend, false, "", 0};
        }
    }

    return RoutingDecision{primary_backend, false, "", 0};
}

// ══════════════════════════════════════════════════════════════════════
//  StrategyEngine
// ══════════════════════════════════════════════════════════════════════

StrategyEngine::StrategyEngine()
    : strategy_(PassthroughConfig{"default"})
    , strategy_name_("passthrough")
    , agent_(std::make_shared<AgentState>())
{}

bool StrategyEngine::init(const std::string& strategy_name,
                          BackendManager& mgr,
                          const std::vector<ModelRoute>& perf_table) {
    strategy_name_ = strategy_name;

    // Discover available backends by ID pattern
    std::string npu_id, gpu_id, cpu_id;
    for (auto& b : mgr.backends()) {
        if (!b.available) continue;
        if (b.id.find("npu") != std::string::npos) npu_id = b.id;
        else if (b.id.find("hip") != std::string::npos) gpu_id = b.id;
        else if (b.id.find("vulkan") != std::string::npos && gpu_id.empty()) gpu_id = b.id;
        else if (b.id.find("cpu") != std::string::npos) cpu_id = b.id;
    }

    // Fallback: use first available
    if (npu_id.empty()) {
        for (auto& b : mgr.backends()) {
            if (b.available) { npu_id = b.id; break; }
        }
    }
    if (gpu_id.empty()) gpu_id = npu_id;
    if (cpu_id.empty()) cpu_id = npu_id;

    // Reset agent state for fresh start
    agent_ = std::make_shared<AgentState>();

    // ── Build the requested strategy ──
    if (strategy_name == "cascade") {
        CascadeConfig cfg;
        cfg.small_backend = npu_id;
        cfg.large_backend = gpu_id;
        cfg.confidence_threshold = -2.5;
        cfg.min_context_for_large = 50;
        cfg.agent = agent_;
        strategy_ = std::move(cfg);
    }
    else if (strategy_name == "performance") {
        PerformanceConfig cfg;
        cfg.default_backend = perf_table.empty() ? npu_id : perf_table[0].backend;
        cfg.live_table = perf_table;
        strategy_ = std::move(cfg);
        mgr.set_strategy(SelectionStrategy::FASTEST);
    }
    else if (strategy_name == "spec_decode") {
        SpecDecodeConfig cfg;
        cfg.draft_backend = npu_id;
        cfg.target_backend = gpu_id;
        cfg.n_draft = 4;
        cfg.acceptance_threshold = 0.8;
        cfg.agent = agent_;
        strategy_ = std::move(cfg);
    }
    else if (strategy_name == "content") {
        ContentRouterConfig cfg;
        cfg.small_backend = npu_id;
        cfg.large_backend = gpu_id;
        cfg.gpu_keywords = {
            "code", "explain", "analyze", "debug", "refactor",
            "implement", "function", "algorithm", "write", "fix",
            "design", "architecture", "test", "review", "optimize"
        };
        cfg.max_small_tokens = 2000;
        strategy_ = std::move(cfg);
    }
    else if (strategy_name == "adaptive") {
        AdaptiveConfig cfg;
        cfg.primary_backend = npu_id;
        cfg.secondary_backend = gpu_id;
        cfg.fallback_backend = cpu_id;
        cfg.agent = agent_;
        cfg.monitor = mgr.monitor_stats();
        strategy_ = std::move(cfg);
    }
    else { // passthrough (default)
        strategy_ = PassthroughConfig{npu_id};
    }

    return true;
}

RoutingDecision StrategyEngine::route(const TokenContext& ctx) const noexcept {
    return std::visit([&](const auto& s) {
        return s.route(ctx);
    }, strategy_);
}

const char* StrategyEngine::name() const noexcept {
    return strategy_name_.c_str();
}

std::string StrategyEngine::state_json() const {
    std::string json;
    json += "{\"strategy\":\"" + strategy_name_ + "\"";

    std::visit([&](const auto& s) {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, CascadeConfig>) {
            json += ",\"small_backend\":\"" + s.small_backend + "\"";
            json += ",\"large_backend\":\"" + s.large_backend + "\"";
            json += ",\"confidence_threshold\":" + std::to_string(s.confidence_threshold);
            if (s.agent) {
                double rate = s.agent->total_tokens_routed.load() > 0
                    ? (double)s.agent->cascade_count.load() / s.agent->total_tokens_routed.load()
                    : 0.0;
                json += ",\"cascade_rate\":" + std::to_string(rate);
            }
        }
        else if constexpr (std::is_same_v<T, AdaptiveConfig>) {
            json += ",\"primary\":\"" + s.primary_backend + "\"";
            json += ",\"secondary\":\"" + s.secondary_backend + "\"";
            json += ",\"fallback\":\"" + s.fallback_backend + "\"";
            if (s.agent) {
                json += ",\"npu_disabled\":" +
                    std::string(s.agent->npu_disabled.load() ? "true" : "false");
                json += ",\"gpu_disabled\":" +
                    std::string(s.agent->gpu_disabled.load() ? "true" : "false");
                json += ",\"cascade_threshold\":" +
                    std::to_string(s.agent->adaptive_cascade_threshold.load());
                json += ",\"npu_load_share\":" +
                    std::to_string(s.agent->npu_load_share.load());
            }
        }
        else if constexpr (std::is_same_v<T, SpecDecodeConfig>) {
            json += ",\"draft_backend\":\"" + s.draft_backend + "\"";
            json += ",\"target_backend\":\"" + s.target_backend + "\"";
            if (s.agent) {
                json += ",\"acceptance_rate\":" +
                    std::to_string(s.agent->acceptance_rate.load());
                json += ",\"n_draft\":" +
                    std::to_string(s.agent->dynamic_n_draft.load());
            }
        }
        else if constexpr (std::is_same_v<T, PerformanceConfig>) {
            json += ",\"default_backend\":\"" + s.default_backend + "\"";
            json += ",\"table_size\":" + std::to_string(s.live_table.size());
        }
        else if constexpr (std::is_same_v<T, ContentRouterConfig>) {
            json += ",\"small_backend\":\"" + s.small_backend + "\"";
            json += ",\"large_backend\":\"" + s.large_backend + "\"";
            json += ",\"keywords\":" + std::to_string(s.gpu_keywords.size());
        }
        else if constexpr (std::is_same_v<T, PassthroughConfig>) {
            json += ",\"backend\":\"" + s.backend + "\"";
        }
    }, strategy_);

    json += "}";
    return json;
}

// ══════════════════════════════════════════════════════════════════════
//  Helpers
// ══════════════════════════════════════════════════════════════════════

std::vector<ModelRoute> build_performance_table(BackendManager& mgr,
                                                  const std::string& model_pattern) {
    std::vector<ModelRoute> table;

    for (auto& b : mgr.backends()) {
        if (!b.available || !b.functional) continue;
        double speed = b.score > 0 ? (1000.0 / b.score) : 0.0; // ms→tok/s
        table.push_back({model_pattern, b.id, speed});
    }

    // Sort by speed descending
    std::sort(table.begin(), table.end(),
              [](const ModelRoute& a, const ModelRoute& b) {
                  return a.speed_tok_s > b.speed_tok_s;
              });

    return table;
}

const char* backend_tier_name(const std::string& backend_id) {
    if (backend_id.find("npu") != std::string::npos) return "accelerator";
    if (backend_id.find("hip") != std::string::npos) return "gpu";
    if (backend_id.find("vulkan") != std::string::npos) return "gpu";
    if (backend_id.find("cpu") != std::string::npos) return "cpu";
    return "unknown";
}
