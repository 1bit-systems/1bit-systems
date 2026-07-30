// dynamic_router.h — Per-token dynamic router for GPU ↔ NPU inference.
//
// Instead of selecting ONE backend for an entire model session, the router
// picks the fastest engine for EACH token. This enables:
//
//   • GPU primary + NPU backfill  (GPU does ~80%, NPU picks up slack)
//   • NPU primary + GPU backfill  (NPU does ~80%, GPU picks up slack)
//   • Fastest per-token           (measure + pick dynamically)
//   • Pure GPU or pure NPU        (when one is optimal for the model)
//
// Performance monitoring: running avg of last N token latencies per backend.
// Backends are initialized once, both loaded with the same model.
//
// Integration:
//   DynamicRouter router;
//   router.add_backend("gpu", gpu_backend, Strategy::GPU_PRIMARY);
//   router.add_backend("npu", npu_backend, Strategy::GPU_PRIMARY);
//   int next = router.generate(token_id);  // auto-routes to fastest
//
// Architecture:
//   ┌─────────────────────────────────────────────┐
//   │              DynamicRouter                   │
//   │  ┌────────────┐  ┌────────────┐             │
//   │  │ GPU Engine │  │ NPU Engine │             │
//   │  │ (HIP,      │  │ (FLM/xclb,│             │
//   │  │  ~50tok/s) │  │  ~67tok/s)│             │
//   │  └─────┬──────┘  └─────┬──────┘             │
//   │        │               │                    │
//   │  ┌─────┴───────────────┴──────┐             │
//   │  │   Scheduler per-strategy:  │             │
//   │  │   • FASTEST → benchmark    │             │
//   │  │   • GPU_BACKFILL → 80/20  │             │
//   │  │   • NPU_BACKFILL → 20/80  │             │
//   │  │   • GPU_ONLY / NPU_ONLY   │             │
//   │  └───────────────────────────┘             │
//   └─────────────────────────────────────────────┘
//
// Models sit in a unified memory pool (SharedBO) — NPU-owned, GPU-imported.

#pragma once
#include "backend.h"
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <memory>
#include <chrono>

class DynamicRouter {
public:
    // Backend selection strategies
    enum class Strategy {
        FASTEST,        // Always pick the backend with lowest avg latency
        GPU_BACKFILL,   // GPU primary (~80%), NPU backfill
        NPU_BACKFILL,   // NPU primary (~80%), GPU backfill
        GPU_ONLY,       // GPU exclusively
        NPU_ONLY        // NPU exclusively
    };

    // Performance snapshot
    struct BackendStats {
        std::string id;
        double avg_ms_per_tok = 0;    // running average
        double tok_s = 0;              // tokens per second
        long long total_tokens = 0;
        long long failed_tokens = 0;
        bool alive = true;
    };

    DynamicRouter() = default;
    ~DynamicRouter();

    // ── Lifecycle ──

    /// Register a backend. Backend must already be init()'d.
    /// The router does NOT own the backend — caller manages lifetime.
    void add_backend(const std::string& id, std::shared_ptr<Backend> backend, Strategy strategy);

    /// Remove a backend by id.
    void remove_backend(const std::string& id);

    /// Reset all backends for new sequence.
    bool reset_all();

    /// Set routing strategy.
    void set_strategy(Strategy s);
    Strategy strategy() const;

    // ── Inference ──

    /// Generate one token — routes to the best backend per strategy.
    int generate(int token_id);

    /// Forward + lm_head via router.
    bool forward(int token_id, float* hidden_out);
    bool lm_head(const float* hidden, float* logits, int* argmax);

    // ── Stats ──

    /// Get current stats for all backends.
    std::vector<BackendStats> stats() const;

    /// Print stats to stdout.
    void report() const;

private:
    struct BackendEntry {
        std::string id;
        std::shared_ptr<Backend> backend;
        BackendStats stats;
        std::vector<double> recent_latencies;
    };

    std::vector<BackendEntry> entries_;
    Strategy strategy_ = Strategy::FASTEST;
    mutable std::mutex mtx_;
    int round_robin_counter_ = 0;

    // Sliding window config
    static constexpr int WINDOW_SIZE = 10;

    BackendEntry* pick_backend();
    void record_latency(BackendEntry* entry, double ms, bool success);
    double window_avg(const BackendEntry* entry) const;
};

// Strategy name helpers
const char* strategy_name(DynamicRouter::Strategy s);
DynamicRouter::Strategy strategy_from_string(const std::string& s);
