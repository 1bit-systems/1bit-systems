// strategy_engine.cpp — Per-token routing strategy implementations
//
// Each strategy's route() call is a pure function — no side effects,
// no locks, no I/O. The agent watchdog modifies strategy state from
// a background thread; route() reads atomics and is lock-free.

#include "strategy_engine.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>

// ══════════════════════════════════════════════════════════════════════
//  CascadeConfig
// ══════════════════════════════════════════════════════════════════════

RoutingDecision CascadeConfig::route(const TokenContext& ctx) const noexcept {
    total_tokens_routed.fetch_add(1, std::memory_order_relaxed);

    // After N tokens, always use large backend
    if (ctx.position >= min_context_for_large) {
        return RoutingDecision{large_backend, false, "", 0};
    }

    // Use threshold_override if set (by watchdog), else config default
    double threshold = threshold_override.load(std::memory_order_relaxed);
    if (threshold < -900.0) {  // sentinel: no override
        threshold = confidence_threshold;
    }

    // Check log-prob against threshold
    if (ctx.log_prob > -1e9 && ctx.log_prob < threshold) {
        cascade_count.fetch_add(1, std::memory_order_relaxed);
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
    // Performance routing is stateless per-token — just return default
    // In practice, the BackendManager is set to FASTEST strategy and
    // picks the best backend automatically.
    //
    // But this gives us an explicit decision for the agent to log/adapt.
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

    // Use dynamic n_draft if available, else config default
    int nd = dynamic_n_draft.load(std::memory_order_relaxed);
    if (nd <= 0) nd = n_draft;

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

    // Case-insensitive keyword search
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
    // ── 1. Check if any backend is disabled by watchdog ──
    if (npu_disabled.load(std::memory_order_relaxed) &&
        gpu_disabled.load(std::memory_order_relaxed)) {
        // Both disabled → CPU fallback
        return RoutingDecision{fallback_backend, false, "", 0};
    }

    if (npu_disabled.load(std::memory_order_relaxed)) {
        return RoutingDecision{secondary_backend, false, "", 0};
    }

    if (gpu_disabled.load(std::memory_order_relaxed)) {
        return RoutingDecision{primary_backend, false, "", 0};
    }

    // ── 2. Cascade: check confidence with dynamic threshold ──
    double threshold = cascade_threshold.load(std::memory_order_relaxed);
    if (ctx.log_prob > -1e9 && ctx.log_prob < threshold) {
        return RoutingDecision{secondary_backend, false, "", 0};
    }

    // ── 3. Load-balanced routing ──
    // If both backends are healthy, split load by npu_load_share
    if (monitor) {
        auto* npu_metrics = monitor->for_backend(primary_backend);
        auto* gpu_metrics = monitor->for_backend(secondary_backend);
        if (npu_metrics && gpu_metrics) {
            double npu_tps = npu_metrics->tokens_per_second.load(std::memory_order_relaxed);
            double gpu_tps = gpu_metrics->tokens_per_second.load(std::memory_order_relaxed);
            double share = npu_load_share.load(std::memory_order_relaxed);

            // If GPU is significantly faster, bias toward it
            if (gpu_tps > npu_tps * 1.5) share = 0.3;
            // If NPU is significantly faster, bias toward it
            if (npu_tps > gpu_tps * 1.5) share = 0.8;

            // Use position mod to approximate load balancing
            // (deterministic per-position, no RNG needed)
            double pos_norm = (ctx.position % 100) / 100.0;
            if (pos_norm < share) {
                return RoutingDecision{primary_backend, false, "", 0};
            }
            return RoutingDecision{secondary_backend, false, "", 0};
        }
    }

    // Default: primary
    return RoutingDecision{primary_backend, false, "", 0};
}

// ══════════════════════════════════════════════════════════════════════
//  StrategyEngine
// ══════════════════════════════════════════════════════════════════════

StrategyEngine::StrategyEngine()
    : strategy_(PassthroughConfig{"default"})
    , strategy_name_("passthrough")
{}

bool StrategyEngine::init(const std::string& strategy_name,
                          BackendManager& mgr,
                          const std::vector<ModelRoute>& perf_table) {
    strategy_name_ = strategy_name;

    // Discover available backends
    std::string npu_id, gpu_id, cpu_id;
    for (auto& b : mgr.backends()) {
        if (!b.available) continue;
        if (b.id.find("npu") != std::string::npos) npu_id = b.id;
        else if (b.id.find("hip") != std::string::npos) gpu_id = b.id;
        else if (b.id.find("vulkan") != std::string::npos && gpu_id.empty()) gpu_id = b.id;
        else if (b.id.find("cpu") != std::string::npos) cpu_id = b.id;
    }

    // Fallback defaults
    if (npu_id.empty() && !mgr.backends().empty()) {
        // Pick first available
        for (auto& b : mgr.backends()) {
            if (b.available) { npu_id = b.id; break; }
        }
    }
    if (gpu_id.empty()) gpu_id = npu_id;
    if (cpu_id.empty()) cpu_id = npu_id;

    // ── Build the requested strategy ──
    if (strategy_name == "cascade") {
        strategy_ = CascadeConfig{
            npu_id, gpu_id, -2.5, 50
        };
    }
    else if (strategy_name == "performance") {
        std::string default_b = perf_table.empty() ? npu_id : perf_table[0].backend;
        strategy_ = PerformanceConfig{default_b, perf_table};
        // Also tell the BackendManager
        mgr.set_strategy(SelectionStrategy::FASTEST);
    }
    else if (strategy_name == "spec_decode") {
        strategy_ = SpecDecodeConfig{
            npu_id, gpu_id, 4, 0.8
        };
    }
    else if (strategy_name == "content") {
        strategy_ = ContentRouterConfig{
            npu_id, gpu_id,
            {"code", "explain", "analyze", "debug", "refactor",
             "implement", "function", "algorithm", "write", "fix",
             "design", "architecture", "test", "review", "optimize"},
            2000
        };
    }
    else if (strategy_name == "adaptive") {
        AdaptiveConfig acfg;
        acfg.primary_backend = npu_id;
        acfg.secondary_backend = gpu_id;
        acfg.fallback_backend = cpu_id;
        acfg.monitor = mgr.monitor_stats();
        strategy_ = std::move(acfg);
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
            json += ",\"cascade_rate\":" +
                std::to_string(s.total_tokens_routed.load() > 0
                    ? (double)s.cascade_count.load() / s.total_tokens_routed.load()
                    : 0.0);
        }
        else if constexpr (std::is_same_v<T, AdaptiveConfig>) {
            json += ",\"primary\":\"" + s.primary_backend + "\"";
            json += ",\"secondary\":\"" + s.secondary_backend + "\"";
            json += ",\"fallback\":\"" + s.fallback_backend + "\"";
            json += ",\"npu_disabled\":" +
                std::string(s.npu_disabled.load() ? "true" : "false");
            json += ",\"gpu_disabled\":" +
                std::string(s.gpu_disabled.load() ? "true" : "false");
            json += ",\"cascade_threshold\":" +
                std::to_string(s.cascade_threshold.load());
        }
        else if constexpr (std::is_same_v<T, SpecDecodeConfig>) {
            json += ",\"draft_backend\":\"" + s.draft_backend + "\"";
            json += ",\"target_backend\":\"" + s.target_backend + "\"";
            json += ",\"acceptance_rate\":" +
                std::to_string(s.acceptance_rate.load());
            json += ",\"n_draft\":" +
                std::to_string(s.dynamic_n_draft.load());
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
        ModelRoute route{model_pattern, b.id, speed};
        table.push_back(std::move(route));
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
