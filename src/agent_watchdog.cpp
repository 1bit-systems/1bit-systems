// agent_watchdog.cpp — The "true agent" background watcher implementation
//
// This is the closed-loop that makes 1bit.systems an agent rather than
// just a router. It observes real hardware conditions and adapts the
// strategy in real-time — no human in the loop.

#include "agent_watchdog.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <algorithm>

// ══════════════════════════════════════════════════════════════════════
//  Thermal monitoring
// ══════════════════════════════════════════════════════════════════════

double read_gpu_temp_c() {
    // Try common sysfs paths for AMD GPUs
    static const char* paths[] = {
        "/sys/class/drm/card1/device/hwmon/hwmon4/temp1_input",
        "/sys/class/drm/card0/device/hwmon/hwmon3/temp1_input",
        "/sys/class/drm/card1/device/hwmon/hwmon3/temp1_input",
        "/sys/class/drm/card0/device/hwmon/hwmon2/temp1_input",
        "/sys/class/drm/card0/device/hwmon/hwmon1/temp1_input",
        "/sys/class/drm/card1/device/hwmon/hwmon1/temp1_input",
        "/sys/class/drm/card0/device/hwmon/hwmon0/temp1_input",
    };

    for (auto path : paths) {
        std::ifstream f(path);
        if (f.is_open()) {
            int millideg = 0;
            f >> millideg;
            if (f.good() && millideg > 0) {
                return millideg / 1000.0;
            }
        }
    }

    // Try AMD-specific ROCm sysfs path
    std::ifstream f("/sys/kernel/debug/dri/0/amdgpu_pm_info");
    if (f.is_open()) {
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("temperature") != std::string::npos ||
                line.find("temp") != std::string::npos) {
                // Parse "XX.X°C" or "XX C" patterns
                size_t pos = line.find_first_of("0123456789");
                if (pos != std::string::npos) {
                    double t = std::atof(line.c_str() + pos);
                    if (t > 0 && t < 200) return t;
                }
            }
        }
    }

    return -1.0;  // not available
}

// ══════════════════════════════════════════════════════════════════════
//  AgentWatchdog
// ══════════════════════════════════════════════════════════════════════

AgentWatchdog::AgentWatchdog(StrategyEngine& engine, BackendManager& mgr)
    : engine_(engine)
    , mgr_(mgr)
    , last_reprofile_(std::chrono::steady_clock::now())
{}

void AgentWatchdog::start() {
    if (running_.exchange(true)) return;  // already running
    stop_ = false;
    thread_ = std::thread(&AgentWatchdog::run, this);

    // Set thread name for observability
#if defined(__linux__)
    pthread_setname_np(thread_.native_handle(), "agent-watchdog");
#endif

    printf("  🧠 Agent watchdog started (interval: 10s)\n");
}

void AgentWatchdog::stop() {
    stop_ = true;
    if (thread_.joinable()) {
        thread_.join();
    }
    running_ = false;
    printf("  🧠 Agent watchdog stopped\n");
}

double AgentWatchdog::seconds_since_reprofile() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - last_reprofile_).count();
}

void AgentWatchdog::run() {
    int cycle = 0;

    while (!stop_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        cycle++;

        // ── Get current strategy variant ──
        StrategyVariant* var = engine_.mutable_variant();
        if (!var) continue;

        // ── Get monitor ──
        BackendMonitor* monitor = mgr_.monitor_stats();
        if (!monitor) continue;

        // ── 1. Thermal check ──
        double gpu_temp = read_gpu_temp_c();
        if (gpu_temp > 0) {
            printf("  🌡️  GPU temp: %.1f°C\n", gpu_temp);
        }

        // ── 2. Health check all backends ──
        mgr_.health_check();

        // ── 3. Gather per-backend metrics ──
        double npu_tps = 0, gpu_tps = 0, npu_p95 = 0, gpu_p95 = 0;
        double npu_fail_rate = 0, gpu_fail_rate = 0;
        uint64_t npu_infs = 0, gpu_infs = 0, npu_fails = 0, gpu_fails = 0;

        for (auto* pm : monitor->all_metrics()) {
            bool is_npu = pm->backend_id.find("npu") != std::string::npos;
            bool is_gpu = pm->backend_id.find("hip") != std::string::npos ||
                          pm->backend_id.find("vulkan") != std::string::npos;

            double tps = pm->tokens_per_second.load(std::memory_order_relaxed);
            double p95 = pm->recent_ms.p95();
            uint64_t infs = pm->inferences.load(std::memory_order_relaxed);
            uint64_t fails = pm->failures.load(std::memory_order_relaxed);

            if (is_npu) {
                npu_tps = tps; npu_p95 = p95;
                npu_infs = infs; npu_fails = fails;
                npu_fail_rate = infs > 0 ? (double)fails / infs : 0;
            }
            if (is_gpu) {
                gpu_tps = tps; gpu_p95 = p95;
                gpu_infs = infs; gpu_fails = fails;
                gpu_fail_rate = infs > 0 ? (double)fails / infs : 0;
            }
        }

        // ── 4. Apply adaptations based on strategy type ──
        std::visit([&](auto&& s) -> void {
            using T = std::decay_t<decltype(s)>;

            // ── Cascade strategy adaptation ──
            if constexpr (std::is_same_v<T, CascadeConfig>) {
                // If GPU is thermal throttling, make cascade less aggressive
                // (keep more tokens on NPU)
                if (gpu_temp > 85.0 && gpu_temp > 0) {
                    double new_threshold = s.confidence_threshold - 1.0;
                    s.threshold_override.store(new_threshold, std::memory_order_relaxed);
                    printf("  🔥 GPU at %.1f°C — cascade threshold adjusted to %.1f (less GPU)\n",
                           gpu_temp, new_threshold);
                }
                // If GPU temp normalizes, restore default
                else if (gpu_temp < 70.0 && gpu_temp > 0) {
                    s.threshold_override.store(-999.0, std::memory_order_relaxed); // reset
                    printf("  ✅ GPU temp normal (%.1f°C) — cascade threshold restored\n",
                           gpu_temp);
                }

                // If NPU failure rate is high, bypass cascade and use GPU
                if (npu_fail_rate > 0.10 && npu_infs > 20) {
                    printf("  ⚠️  NPU failure rate %.1f%% — forcing GPU for all tokens\n",
                           npu_fail_rate * 100.0);
                    // Flip to GPU-only by setting threshold to +inf
                    s.threshold_override.store(999.0, std::memory_order_relaxed);
                } else if (npu_fail_rate < 0.02 && npu_infs > 50) {
                    // NPU recovered — let the temp override stand but clear fail override
                    if (gpu_temp < 70.0 || gpu_temp < 0) {
                        s.threshold_override.store(-999.0, std::memory_order_relaxed);
                    }
                }
            }

            // ── Adaptive strategy ──
            if constexpr (std::is_same_v<T, AdaptiveConfig>) {
                // Thermal throttling
                if (gpu_temp > 85.0 && gpu_temp > 0) {
                    // Shift more load to NPU
                    s.npu_load_share.store(0.9, std::memory_order_relaxed);
                    printf("  🔥 GPU at %.1f°C — shifting load to NPU (90%%)\n", gpu_temp);
                } else if (gpu_temp < 70.0 && gpu_temp > 0) {
                    // Restore balanced load
                    s.npu_load_share.store(0.6, std::memory_order_relaxed);
                    printf("  ✅ GPU temp normal — restoring balanced load (60/40)\n");
                }

                // Failure spike → disable backend
                if (npu_fail_rate > 0.10 && npu_infs > 20) {
                    if (!s.npu_disabled.load()) {
                        printf("  ⛔ NPU failure rate %.1f%% — DISABLING NPU\n",
                               npu_fail_rate * 100.0);
                        s.npu_disabled.store(true, std::memory_order_relaxed);
                    }
                }
                if (gpu_fail_rate > 0.10 && gpu_infs > 20) {
                    if (!s.gpu_disabled.load()) {
                        printf("  ⛔ GPU failure rate %.1f%% — DISABLING GPU\n",
                               gpu_fail_rate * 100.0);
                        s.gpu_disabled.store(true, std::memory_order_relaxed);
                    }
                }

                // Re-enable if failure rate drops and enough samples
                if (s.npu_disabled.load() && npu_infs > 50 && npu_fail_rate < 0.02) {
                    printf("  ✅ NPU recovered — re-enabling\n");
                    s.npu_disabled.store(false, std::memory_order_relaxed);
                }
                if (s.gpu_disabled.load() && gpu_infs > 50 && gpu_fail_rate < 0.02) {
                    printf("  ✅ GPU recovered — re-enabling\n");
                    s.gpu_disabled.store(false, std::memory_order_relaxed);
                }

                // Load balancing by throughput
                if (npu_tps > 0 && gpu_tps > 0) {
                    double ratio = gpu_tps / npu_tps;
                    if (ratio > 1.5) {
                        // GPU is significantly faster → send more to GPU
                        s.npu_load_share.store(0.3, std::memory_order_relaxed);
                    } else if (ratio < 0.67) {
                        // NPU is significantly faster → send more to NPU
                        s.npu_load_share.store(0.8, std::memory_order_relaxed);
                    }
                    // else: roughly equal, keep current balance
                }

                // Latency degradation check
                if (npu_p95 > 0 && baseline_established_.load()) {
                    double npu_baseline = npu_baseline_p50_.load();
                    if (npu_baseline > 0 && npu_p95 > npu_baseline * 3.0) {
                        printf("  ⚠️  NPU P95 (%.1fms) > 3× baseline (%.1fms) — deprioritizing\n",
                               npu_p95, npu_baseline);
                        s.npu_load_share.store(0.2, std::memory_order_relaxed);
                    }
                }
                if (gpu_p95 > 0 && baseline_established_.load()) {
                    double gpu_baseline = gpu_baseline_p50_.load();
                    if (gpu_baseline > 0 && gpu_p95 > gpu_baseline * 3.0) {
                        printf("  ⚠️  GPU P95 (%.1fms) > 3× baseline (%.1fms) — deprioritizing\n",
                               gpu_p95, gpu_baseline);
                        s.npu_load_share.store(0.9, std::memory_order_relaxed);
                    }
                }

                // Cascade threshold adaptation: if GPU is slow/temp-throttled,
                // make cascade more aggressive (NPU keeps more tokens)
                if (gpu_temp > 80.0 || gpu_p95 > 15.0) {
                    s.cascade_threshold.store(-1.0, std::memory_order_relaxed);
                } else {
                    s.cascade_threshold.store(-2.5, std::memory_order_relaxed);
                }
            }

            // ── Speculative decode strategy adaptation ──
            if constexpr (std::is_same_v<T, SpecDecodeConfig>) {
                // Adjust n_draft based on acceptance rate
                double rate = s.acceptance_rate.load(std::memory_order_relaxed);
                if (rate > 0.9 && s.dynamic_n_draft.load() < 8) {
                    s.dynamic_n_draft.fetch_add(1, std::memory_order_relaxed);
                    printf("  📈 Spec decode acceptance %.0f%% — increasing n_draft to %d\n",
                           rate * 100.0, s.dynamic_n_draft.load());
                } else if (rate < 0.6 && s.dynamic_n_draft.load() > 2) {
                    s.dynamic_n_draft.fetch_sub(1, std::memory_order_relaxed);
                    printf("  📉 Spec decode acceptance %.0f%% — decreasing n_draft to %d\n",
                           rate * 100.0, s.dynamic_n_draft.load());
                }
            }
        }, *var);

        // ── 5. Record baseline after first 100 inferences ──
        if (!baseline_established_.load()) {
            uint64_t total = 0;
            for (auto* pm : monitor->all_metrics()) {
                total += pm->inferences.load(std::memory_order_relaxed);
            }
            if (total >= 100) {
                for (auto* pm : monitor->all_metrics()) {
                    bool is_npu = pm->backend_id.find("npu") != std::string::npos;
                    bool is_gpu = pm->backend_id.find("hip") != std::string::npos ||
                                  pm->backend_id.find("vulkan") != std::string::npos;
                    if (is_npu) npu_baseline_p50_.store(pm->recent_ms.p50());
                    if (is_gpu) gpu_baseline_p50_.store(pm->recent_ms.p50());
                }
                baseline_established_ = true;
                printf("  📊 Baseline established — NPU P50=%.1fms GPU P50=%.1fms\n",
                       npu_baseline_p50_.load(), gpu_baseline_p50_.load());
            }
        }

        // ── 6. Re-profile if stable for 5 minutes ──
        double stable_seconds = seconds_since_reprofile();
        if (stable_seconds > 300.0) {  // 5 minutes
            // Check all backends are healthy and aggregate error is low
            bool all_healthy = true;
            for (auto* pm : monitor->all_metrics()) {
                if (!pm->healthy.load(std::memory_order_relaxed)) {
                    all_healthy = false;
                    break;
                }
            }

            if (all_healthy) {
                printf("  📊 Re-profiling all backends (stable for %.0fs)...\n", stable_seconds);
                mgr_.benchmark_all(10);
                last_reprofile_ = std::chrono::steady_clock::now();

                // Update performance table if needed
                auto table = build_performance_table(mgr_, "zaya");
                std::visit([&](auto&& s) {
                    using T = std::decay_t<decltype(s)>;
                    if constexpr (std::is_same_v<T, PerformanceConfig>) {
                        s.live_table = table;
                        printf("  ✅ Performance table updated (%zu entries)\n", table.size());
                    }
                }, *var);
            }
        }

        // ── 7. Periodic status line ──
        if (cycle % 3 == 0) {  // every 30 seconds
            printf("  📊 [watchdog] strategy=%s npu=%.0f tok/s gpu=%.0f tok/s "
                   "npu_fail=%.1f%% gpu_fail=%.1f%%",
                   engine_.name(),
                   npu_tps, gpu_tps,
                   npu_fail_rate * 100.0, gpu_fail_rate * 100.0);
            if (gpu_temp > 0) printf(" gpu_temp=%.1f°C", gpu_temp);
            printf("\n");
        }
    }  // end while
}
