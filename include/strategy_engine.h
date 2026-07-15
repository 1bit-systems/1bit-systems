// strategy_engine.h — Per-token routing strategy engine
//
// Zero-overhead std::variant dispatch for all routing strategies.
// Each strategy is a plain struct with a route() call. The engine
// selects the active strategy at boot and adapts it at runtime via
// the agent watchdog.
//
// Strategies:
//   Cascade       — NPU first, switch to GPU on low log-prob confidence
//   Performance   — Model→backend lookup from live benchmark table
//   SpecDecode    — NPU drafts N tokens, GPU verifies in one pass
//   ContentRouter — Keyword-based request-level routing
//   Adaptive      — Monitor-driven: adjusts cascade threshold, load-balances
//   Passthrough   — Fixed single backend (no decision)
//
// The "true agent" is the Adaptive wrapper + AgentWatchdog thread:
//   Watchdog observes BackendMonitor metrics at runtime, and modifies
//   the active strategy's parameters (thresholds, backends, weights)
//   to respond to thermal throttling, latency degradation, failure spikes.

#pragma once

#include "backend_manager.h"
#include "backend_monitor.h"
#include <string>
#include <vector>
#include <variant>
#include <atomic>
#include <chrono>

// ── Per-token routing context ─────────────────────────────────────────
// Lightweight: no heap, no copies — passed by const& to route().

struct TokenContext {
    int token_id;                  // last generated token ID (-1 = first token)
    double log_prob;               // log-probability of last token (-inf = unknown)
    double entropy;                // entropy of last token (-1 = unknown)
    size_t position;               // token position in this response (0-based)
    size_t total_prompt_tokens;    // length of prompt
    size_t max_tokens;             // max generation length
    const std::string& message_text; // user message text (for content routing)
};

// ── Routing decision ──────────────────────────────────────────────────

struct RoutingDecision {
    std::string backend;           // which backend to use
    bool speculative = false;      // speculative decode mode
    std::string draft_backend;     // for speculative decode
    int n_draft = 0;               // number of draft tokens
};

// ── Performance table entry (from live benchmarks) ────────────────────

struct ModelRoute {
    std::string model_pattern;     // e.g. "qwen3", "bonsai"
    std::string backend;           // e.g. "npu_xrt", "hip_gpu"
    double speed_tok_s;            // measured decode speed
};

// ── Strategy configs (each is a std::variant alternative) ──────────────

struct CascadeConfig {
    // Config (set at boot)
    std::string small_backend;     // NPU
    std::string large_backend;     // GPU
    double confidence_threshold = -2.5; // log-prob below this → route to large
    size_t min_context_for_large = 50;  // always use large after N tokens

    // Live adaptation state (modified by watchdog)
    std::atomic<double> threshold_override{-999.0}; // -999 = use default
    std::atomic<size_t> cascade_count{0};
    std::atomic<size_t> total_tokens_routed{0};

    RoutingDecision route(const TokenContext& ctx) const noexcept;
};

struct PerformanceConfig {
    // Config (set at boot from live benchmarks)
    std::string default_backend;
    std::vector<ModelRoute> live_table;  // populated by microbenchmark

    // Live adaptation state
    std::atomic<bool> force_reprofile{false};

    RoutingDecision route(const TokenContext& ctx) const noexcept;
    std::string best_backend_for_model(const std::string& model) const;
};

struct SpecDecodeConfig {
    // Config
    std::string draft_backend;
    std::string target_backend;
    int n_draft = 4;
    double acceptance_threshold = 0.8;

    // Live adaptation state
    std::atomic<double> acceptance_rate{0.85}; // EMA of acceptance rate
    std::atomic<int> dynamic_n_draft{4};

    RoutingDecision route(const TokenContext& ctx) const noexcept;
};

struct ContentRouterConfig {
    // Config
    std::string small_backend;
    std::string large_backend;
    std::vector<std::string> gpu_keywords;
    size_t max_small_tokens = 2000;  // tokens before forced switch to large

    RoutingDecision route(const TokenContext& ctx) const noexcept;

private:
    bool has_gpu_keywords(const std::string& text) const noexcept;
};

struct AdaptiveConfig {
    // The adaptive strategy wraps another strategy and applies
    // monitor-driven adjustments. This is the "true agent" entry point.
    //
    // At route() time, it checks BackendMonitor live metrics and
    // adjusts behavior:
    //   - GPU temp > 85°C → shift to NPU more aggressively
    //   - NPU failure rate > 10% → stop routing to NPU
    //   - GPU latency P95 > 2× baseline → cascade earlier
    //   - Load balance: keep both busy if both healthy

    std::string primary_backend;   // preferred
    std::string secondary_backend; // fallback
    std::string fallback_backend;  // CPU / always-available

    // Dynamic thresholds (modified by watchdog)
    std::atomic<double> cascade_threshold{-2.5};
    std::atomic<double> npu_load_share{0.7};  // 0-1: fraction to send to NPU
    std::atomic<bool> npu_disabled{false};
    std::atomic<bool> gpu_disabled{false};

    // Reference to monitor for live adaptation
    BackendMonitor* monitor = nullptr;

    RoutingDecision route(const TokenContext& ctx) const noexcept;
};

// ── Strategy variant (zero-overhead dispatch) ─────────────────────────

using StrategyVariant = std::variant<
    CascadeConfig,
    PerformanceConfig,
    SpecDecodeConfig,
    ContentRouterConfig,
    AdaptiveConfig,
    struct PassthroughConfig  // forward-declared below
>;

struct PassthroughConfig {
    std::string backend;

    RoutingDecision route(const TokenContext& ctx) const noexcept {
        return RoutingDecision{backend, false, "", 0};
    }
};

// ── Strategy engine — wraps the variant with runtime management ───────

class StrategyEngine {
public:
    StrategyEngine();

    /// Initialize from config — builds the appropriate variant
    /// based on strategy name ("cascade", "performance", "spec_decode",
    /// "content", "adaptive", "passthrough").
    bool init(const std::string& strategy_name,
              BackendManager& mgr,
              const std::vector<ModelRoute>& perf_table = {});

    /// Route the next token — the main dispatch call.
    /// Zero-overhead: std::visit monomorphizes at compile time.
    RoutingDecision route(const TokenContext& ctx) const noexcept;

    /// Get the active strategy name (for observability)
    const char* name() const noexcept;

    /// Get a mutable pointer to the variant for watchdog updates.
    /// The watchdog can visit() and modify strategy params.
    StrategyVariant* mutable_variant() noexcept { return &strategy_; }
    const StrategyVariant& variant() const noexcept { return strategy_; }

    /// Serialize current strategy state for metrics endpoint
    std::string state_json() const;

private:
    StrategyVariant strategy_;
    std::string strategy_name_;
};

// ── Convenience: build a PerformanceConfig from BackendManager benchmarks ──

/// Build a live performance table by benchmarking all available backends.
/// Returns the table and the fastest backend ID.
std::vector<ModelRoute> build_performance_table(BackendManager& mgr,
                                                 const std::string& model_pattern);

/// Get backend tier name for a backend ID (for observability)
const char* backend_tier_name(const std::string& backend_id);
