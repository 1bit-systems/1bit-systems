// agent_watchdog.h — The "true agent" background watcher
//
// Runs as a background thread inside the unified server. Every ~10 seconds
// it reads BackendMonitor metrics and adapts strategy parameters:
//
//   Thermal:    GPU temp > 85°C → deprioritize GPU, shift load to NPU/CPU
//   Latency:    P95 > 2× baseline → mark backend degraded
//   Failures:   Error rate > 10% → disable backend, re-route
//   Throughput: Rebalance load share between NPU and GPU
//   Stability:  Re-profile all backends if aggregate error < 1% for 5 min
//
// The watchdog modifies StrategyVariant fields through std::visit + atomics.
// All route() calls are lock-free — they read atomics set by the watchdog.

#pragma once

#include "strategy_engine.h"
#include "backend_manager.h"
#include <thread>
#include <atomic>
#include <chrono>

// ── Thermal monitoring source ──
// Reads GPU temperature from sysfs. Returns -1 if unavailable.
// Paths: /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input
double read_gpu_temp_c();

// ── Agent Watchdog ──

class AgentWatchdog {
public:
    /// Create but do NOT start the watchdog thread yet.
    AgentWatchdog(StrategyEngine& engine, BackendManager& mgr);

    /// Start the background watchdog thread.
    void start();

    /// Signal the watchdog to stop and join.
    void stop();

    /// Check if the watchdog is running.
    bool running() const { return running_.load(); }

    /// Get the time since last re-profile (in seconds).
    double seconds_since_reprofile() const;

private:
    /// The main watchdog loop (runs on background thread).
    void run();

    StrategyEngine& engine_;
    BackendManager& mgr_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_{false};

    // Re-profile tracking
    std::chrono::steady_clock::time_point last_reprofile_;
    std::mutex stats_mtx_;

    // Baseline latencies (set after first 100 inferences)
    std::atomic<bool> baseline_established_{false};
    std::atomic<double> npu_baseline_p50_{0.0};
    std::atomic<double> gpu_baseline_p50_{0.0};
};
