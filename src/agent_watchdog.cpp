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

    std::ifstream f("/sys/kernel/debug/dri/0/amdgpu_pm_info");
    if (f.is_open()) {
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("temperature") != std::string::npos ||
                line.find("temp") != std::string::npos) {
                size_t pos = line.find_first_of("0123456789");
                if (pos != std::string::npos) {
                    double t = std::atof(line.c_str() + pos);
                    if (t > 0 && t < 200) return t;
                }
            }
        }
    }

    return -1.0;
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
    if (running_.exchange(true)) return;
    stop_ = false;
    thread_ = std::thread(&AgentWatchdog::run, this);

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

        // ── Get shared agent state for lock-free adaptation ──
        auto agent = engine_.agent_state();
        if (!agent) continue;

        BackendMonitor* monitor = mgr_.monitor_stats();
        if (!monitor) continue;

        // ── 1. Thermal check ──
        double gpu_temp = read_gpu_temp_c();
        if (gpu_temp > 0 && (cycle % 3 == 0)) {
            printf("  🌡️  GPU temp: %.1f°C\n", gpu_temp);
        }

        // ── 2. Health check all backends ──
        mgr_.health_check();

        // ── 3. Gather per-backend metrics ──
        double npu_tps = 0, gpu_tps = 0, npu_p95 = 0, gpu_p95 = 0;
        double npu_fail_rate = 0, gpu_fail_rate = 0;
        uint64_t npu_infs = 0, gpu_infs = 0;

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
                npu_infs = infs;
                npu_fail_rate = infs > 0 ? (double)fails / infs : 0;
            }
            if (is_gpu) {
                gpu_tps = tps; gpu_p95 = p95;
                gpu_infs = infs;
                gpu_fail_rate = infs > 0 ? (double)fails / infs : 0;
            }
        }

        // ── 4. Apply adaptations via AgentState (lock-free atomics) ──

        // ── Thermal: GPU temp > 85°C → more load on NPU ──
        if (gpu_temp > 85.0 && gpu_temp > 0) {
            // release: ensures HTTP handler threads observe these writes (fixes #364)
            agent->npu_load_share.store(0.9, std::memory_order_release);
            agent->adaptive_cascade_threshold.store(-1.0, std::memory_order_release);
            agent->cascade_threshold_override.store(-1.0, std::memory_order_release);
            if (cycle % 2 == 0)
                printf("  🔥 GPU at %.1f°C — shifted load to NPU (90%%)\n", gpu_temp);
        }
        // ── GPU temp normal → restore balance ──
        else if (gpu_temp < 70.0 && gpu_temp > 0) {
            // release: ensures HTTP handler threads observe these writes (fixes #364)
            agent->npu_load_share.store(0.6, std::memory_order_release);
            agent->adaptive_cascade_threshold.store(-2.5, std::memory_order_release);
            agent->cascade_threshold_override.store(-999.0, std::memory_order_release); // reset
            if (cycle % 3 == 0)
                printf("  ✅ GPU temp normal (%.1f°C) — load balance restored\n", gpu_temp);
        }

        // ── Failure rate → disable backend ──
        if (npu_fail_rate > 0.10 && npu_infs > 20) {
            if (!agent->npu_disabled.load()) {
                printf("  ⛔ NPU failure rate %.1f%% — DISABLING NPU\n", npu_fail_rate * 100.0);
                agent->npu_disabled.store(true, std::memory_order_release);
            }
        }
        if (gpu_fail_rate > 0.10 && gpu_infs > 20) {
            if (!agent->gpu_disabled.load()) {
                printf("  ⛔ GPU failure rate %.1f%% — DISABLING GPU\n", gpu_fail_rate * 100.0);
                agent->gpu_disabled.store(true, std::memory_order_release);
            }
        }

        // ── Re-enable if recovered ──
        if (agent->npu_disabled.load() && npu_infs > 50 && npu_fail_rate < 0.02) {
            printf("  ✅ NPU recovered — re-enabling\n");
            agent->npu_disabled.store(false, std::memory_order_release);
        }
        if (agent->gpu_disabled.load() && gpu_infs > 50 && gpu_fail_rate < 0.02) {
            printf("  ✅ GPU recovered — re-enabling\n");
            agent->gpu_disabled.store(false, std::memory_order_release);
        }

        // ── Throughput-based load balancing ──
        if (npu_tps > 0 && gpu_tps > 0) {
            double ratio = gpu_tps / npu_tps;
            if (ratio > 1.5) {
                agent->npu_load_share.store(0.3, std::memory_order_release);
            } else if (ratio < 0.67) {
                agent->npu_load_share.store(0.8, std::memory_order_release);
            }
        }

        // ── Latency degradation ──
        if (npu_p95 > 0 && baseline_established_.load()) {
            double baseline = npu_baseline_p50_.load();
            if (baseline > 0 && npu_p95 > baseline * 3.0) {
                printf("  ⚠️  NPU P95 (%.1fms) > 3× baseline (%.1fms)\n", npu_p95, baseline);
                agent->npu_load_share.store(0.2, std::memory_order_release);
            }
        }
        if (gpu_p95 > 0 && baseline_established_.load()) {
            double baseline = gpu_baseline_p50_.load();
            if (baseline > 0 && gpu_p95 > baseline * 3.0) {
                printf("  ⚠️  GPU P95 (%.1fms) > 3× baseline (%.1fms)\n", gpu_p95, baseline);
                agent->npu_load_share.store(0.9, std::memory_order_release);
            }
        }

        // ── Speculative decode: adjust n_draft by acceptance rate ──
        double accept_rate = agent->acceptance_rate.load(std::memory_order_acquire);
        if (accept_rate > 0.9) {
            int nd = agent->dynamic_n_draft.load();
            if (nd < 8) {
                agent->dynamic_n_draft.store(nd + 1, std::memory_order_release);
                printf("  📈 Spec decode acceptance %.0f%% — n_draft increased to %d\n",
                       accept_rate * 100.0, nd + 1);
            }
        } else if (accept_rate < 0.6) {
            int nd = agent->dynamic_n_draft.load();
            if (nd > 2) {
                agent->dynamic_n_draft.store(nd - 1, std::memory_order_release);
                printf("  📉 Spec decode acceptance %.0f%% — n_draft decreased to %d\n",
                       accept_rate * 100.0, nd - 1);
            }
        }

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
        if (stable_seconds > 300.0) {
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
            }
        }

        // ── 7. Periodic status ──
        if (cycle % 3 == 0) {
            printf("  📊 [watchdog] strategy=%s npu=%.0f tok/s gpu=%.0f tok/s "
                   "npu_fail=%.1f%% gpu_fail=%.1f%%",
                   engine_.name(), npu_tps, gpu_tps,
                   npu_fail_rate * 100.0, gpu_fail_rate * 100.0);
            if (gpu_temp > 0) printf(" gpu_temp=%.1f°C", gpu_temp);
            printf("\n");
        }
    }
}
