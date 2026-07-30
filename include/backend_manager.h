// backend_manager.h — Backend Manager: auto-detect, auto-select, auto-fallback
// Windows ML-style orchestrator for the Zaya inference engine.
// Mirrors the WinML execution provider model:
//   - Auto-detects all available backends (NPU, GPU, CPU)
//   - Selects the fastest available at init
//   - Monitors health and performance at runtime
//   - Transparent failover on error (NPU→GPU→CPU)
//   - Plugin loading via shared library (ORT EP equivalent)

#pragma once
#include "backend.h"
#include "pilot.h"
#include "dynamic_router.h"
#include "backend_monitor.h"
#include <string>
#include <vector>
#include <functional>
#include <mutex>
#include <memory>
#include <algorithm>

// ── Backend priority / tier ──
enum class BackendTier : uint8_t {
    T1_ACCELERATOR = 0,  // NPU (low-power, always-on)
    T2_GPU         = 1,  // GPU (high-throughput)
    T3_CPU         = 2,  // CPU (always available fallback)
};

inline const char* tier_name(BackendTier t) {
    switch(t) {
        case BackendTier::T1_ACCELERATOR: return "NPU/Accelerator";
        case BackendTier::T2_GPU:         return "GPU";
        case BackendTier::T3_CPU:         return "CPU";
        default: return "unknown";
    }
}

// ── Backend info (like an ORT EP descriptor) ──
struct BackendInfo {
    std::string id;            // unique ID: "npu_xrt", "hip_gpu", "vulkan", "cpu_avx512"
    BackendType type;
    BackendTier tier;
    std::string description;   // "AMD XDNA NPU via XRT", "AMD Radeon 8060S via HIP"
    int priority;              // higher = preferred (auto-assigned by tier)
    float score;               // benchmark score (ms/token, 0 = untested)
    bool available;            // detected at startup
    bool functional;           // passed health check
    bool auto_selectable = true; // false = never tried by init()'s auto-loop or
                                  // select_best() (see docs/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md);
                                  // true manual opt-in needs on-demand lazy init, not yet built
    uint64_t total_inferences; // lifetime counter
    uint64_t failed_inferences;
    double cumulative_ms;      // total inference time in ms

    std::shared_ptr<Backend> instance;  // shared_ptr enables lock-free inference (fixes #96)
    void* plugin_handle;       // dlopen handle if loaded as plugin

    // Guards actual compute calls (generate/reset/benchmark/init) on `instance`.
    // generate() intentionally releases BackendManager::mtx_ during inference
    // (fixes #96) so slow calls don't block bookkeeping, but that means mtx_
    // alone can't stop AgentWatchdog's health_check()/benchmark_all() from
    // calling reset()/benchmark()/init() on the same live instance concurrently
    // with an in-flight generate() — a data race on the backend's internal
    // state (e.g. HIPBackend's ZayaState/pos) that crashed the agent-watchdog
    // thread inside zaya_init on 2026-07-24. shared_ptr so BackendInfo stays
    // movable/copyable in the vector.
    std::shared_ptr<std::mutex> compute_mtx = std::make_shared<std::mutex>();
};

// ── Fallback policy ──
enum class FallbackPolicy : uint8_t {
    NONE       = 0,  // No fallback — fail-fast
    SEQUENTIAL = 1,  // Try primary, then secondary, then tertiary (default)
    BEST_EFFORT = 2, // Try all in parallel, use whichever finishes first
};

// ── Selection strategy ──
enum class SelectionStrategy : uint8_t {
    FASTEST  = 0,  // Pick the backend with best benchmark score (default)
    LOWEST_POWER = 1,  // NPU > GPU > CPU regardless of speed
    MANUAL   = 2,   // User-specified override
    ROUND_ROBIN = 3, // Cycle through available backends
};

// ── Plugin descriptor (for dlopen-based backend loading) ──
struct BackendPlugin {
    std::string path;          // full path to .so
    std::string id;            // plugin identifier
    std::string version;       // semver
    bool loaded = false;

    // Functions expected in the plugin
    // extern "C" Backend* create_backend();
    // extern "C" int get_backend_type();
    // extern "C" const char* get_backend_id();
};

// ── BackendManager: the central orchestrator ──
// Like Windows ML — it queries available EPs, selects the best one,
// monitors health, and handles failover.
class BackendManager {
public:
    BackendManager();
    ~BackendManager();

    // ── System init ──
    /// Scan all available backends (hardware probe, no allocation yet)
    void discover();
    /// Initialize the selected backend(s) and load weights
    bool init(const ModelConfig& cfg, const std::string& weights_dir);
    /// Same, but try backend ids in `preferred_ids` order first (falling back to
    /// normal priority order for anything not in the list, or if all preferred
    /// ids fail). Empty preferred_ids behaves exactly like the two-arg overload.
    bool init(const ModelConfig& cfg, const std::string& weights_dir,
              const std::vector<std::string>& preferred_ids);
    /// Destroy all backends
    void destroy();

    // ── Selection ──
    /// Choose which backend to use for inference
    void set_strategy(SelectionStrategy s);
    SelectionStrategy strategy() const;
    /// Manually force a specific backend
    bool select_backend(const std::string& id);
    bool select_backend(BackendType type);
    /// Get the currently active backend
    Backend* active_backend();
    const BackendInfo* active_info() const;
    /// List all discovered backends
    const std::vector<BackendInfo>& backends() const { return backends_; }

    // ── Inference ──
    /// Run one token — returns token ID, with automatic failover
    int generate(int token_id);
    /// Forward pass with hidden state output
    bool forward(int token_id, float* hidden_out);
    /// LM head
    bool lm_head(const float* hidden, float* logits, int* argmax);
    /// Reset state
    bool reset();

    // ── Failover ──
    void set_fallback_policy(FallbackPolicy p);
    FallbackPolicy fallback_policy() const;

    // ── Health ──
    /// Probe the active backend; return false if unhealthy
    bool health_check();
    /// Check all backends; trigger failover if active is unhealthy
    void monitor();

    /// Access the pilot prefetch engine (for tuning/reporting).
    const Pilot& pilot() const { return pilot_; }
    Pilot& pilot() { return pilot_; }
    bool pilot_active() const { return pilot_active_; }
    std::vector<DynamicRouter::BackendStats> router_stats() const { return router_.stats(); }
    DynamicRouter& router() { return router_; }

    /// Re-evaluate active backend against strategy and current scores.
    /// For FASTEST: switches to the backend with best (lowest) benchmark score.
    /// For ROUND_ROBIN: cycles to the next available backend.
    /// For LOWEST_POWER: switches to the highest-priority available backend.
    /// Returns true if the active backend changed.
    bool re_evaluate();

    // ── Benchmarking ──
    /// Run benchmark on all backends, updating their scores
    void benchmark_all(int tokens = 10);
    /// Record a benchmark score for a single backend (e.g. a caller that
    /// benchmarked only the active backend directly via its Backend instance,
    /// bypassing benchmark_all() — see unified_server.cpp Phase 4).
    void set_score(const std::string& id, float ms);
    /// Get the best backend for a given tier
    const BackendInfo* best_for_tier(BackendTier tier) const;

    // ── Plugins ──
    /// Load a backend from a shared library
    bool load_plugin(const std::string& so_path);
    /// Load all plugins from a directory
    int load_plugins(const std::string& directory);
    const std::vector<BackendPlugin>& plugins() const { return plugins_; }

    // ── Monitoring / Stats ──
    const BackendMonitor* monitor_stats() const { return &monitor_; }
    BackendMonitor* monitor_stats() { return &monitor_; }
    std::string report() const;  // human-readable summary

private:
    // Internal backend lifecycle (GPU/NPU via dlsym, CPU linked in)
    Backend* create_instance_rt(const BackendInfo& info);
    void destroy_instance(BackendInfo& info);
    // Shared by both init() overloads: try backends_[order[i]] in sequence.
    // Caller must hold mtx_.
    bool init_in_order(const ModelConfig& cfg, const std::string& weights_dir,
                        const std::vector<size_t>& order);

    // Try next backend in fallback chain
    bool failover();
    // Select backend by priority (highest available + functional)
    bool select_best();
    // Rank all backends by score + tier
    void rank_backends();

    std::vector<BackendInfo> backends_;
    std::vector<BackendPlugin> plugins_;
    size_t active_idx_ = 0;       // index into backends_
    size_t fallback_idx_ = 1;     // next to try on failure
    ModelConfig cfg_;
    std::string weights_dir_;

    SelectionStrategy strategy_ = SelectionStrategy::FASTEST;
    FallbackPolicy fallback_policy_ = FallbackPolicy::SEQUENTIAL;

    BackendMonitor monitor_;      // live performance tracking
    DynamicRouter router_;        // per-token GPU ↔ NPU router
    Pilot pilot_;
    bool pilot_active_ = false;
    bool initialized_ = false;
    mutable std::mutex mtx_;
};

// ── Convenience: global singleton ──
BackendManager& backend_manager();
