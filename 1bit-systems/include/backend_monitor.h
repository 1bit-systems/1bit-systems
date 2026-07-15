// backend_monitor.h — Real-time performance monitoring for all backends
// Equivalent to Windows Task Manager's NPU tab + WPA's Neural Processing profile.
// Tracks per-call timing, utilization estimates, throughput, and error rates.

#pragma once
#include <cstdint>
#include <chrono>
#include <string>
#include <vector>
#include <atomic>
#include <array>
#include <algorithm>
#include <mutex>

// ── Metric window (ring buffer) ──
// Stores the last N inference latencies for statistical analysis.
template<size_t N = 256>
struct MetricWindow {
    std::array<float, N> samples{};
    size_t head = 0;
    size_t count = 0;

    void push(float ms) {
        samples[head] = ms;
        head = (head + 1) % N;
        if (count < N) count++;
    }

    float p50() const { return percentile(0.50f); }
    float p95() const { return percentile(0.95f); }
    float p99() const { return percentile(0.99f); }

    float avg() const {
        if (count == 0) return 0;
        float sum = 0;
        for (size_t i = 0; i < count; i++) sum += samples[i];
        return sum / count;
    }

    float min() const {
        if (count == 0) return 0;
        float m = samples[0];
        for (size_t i = 1; i < count; i++)
            if (samples[i] < m) m = samples[i];
        return m;
    }

    float max() const {
        if (count == 0) return 0;
        float m = samples[0];
        for (size_t i = 1; i < count; i++)
            if (samples[i] > m) m = samples[i];
        return m;
    }

private:
    float percentile(float p) const {
        if (count == 0) return 0;
        // Copy to sorted buffer (only on the last N, not the full window)
        std::vector<float> sorted(samples.begin(), samples.begin() + count);
        std::sort(sorted.begin(), sorted.end());
        size_t idx = std::min(size_t(p * (count - 1)), count - 1);
        return sorted[idx];
    }
};

// ── Per-backend metrics ──
struct PerBackendMetrics {
    std::string backend_id;

    // Lifetime counters (atomic: safe from multiple threads)
    std::atomic<uint64_t> inferences{0};
    std::atomic<uint64_t> failures{0};
    std::atomic<uint64_t> fallbacks{0};

    // Timing (most recent and windowed)
    std::atomic<float> last_ms{0};
    // MetricWindow is NOT thread-safe; protected by mutex in BackendMonitor
    MetricWindow<256> recent_ms;

    // Throughput
    std::atomic<double> tokens_per_second{0};  // smoothed

    // Utilization estimates (0.0 – 1.0, best-effort)
    std::atomic<float> estimated_utilization{0};

    // Health
    std::atomic<bool> healthy{true};
    std::atomic<uint64_t> last_health_check{0};
    std::string last_error;

    // Tracking state (internal) — only accessed under BackendMonitor lock
    std::chrono::steady_clock::time_point window_start;
    uint64_t window_inferences;

    PerBackendMetrics() { reset_window(); }

    void reset_window() {
        window_start = std::chrono::steady_clock::now();
        window_inferences = 0;
    }

    void record_inference(float ms, bool ok) {
        inferences++;
        if (!ok) failures++;
        last_ms.store(ms);
        recent_ms.push(ms);

        // Update throughput every 10 inferences
        window_inferences++;
        if (window_inferences >= 10) {
            auto now = std::chrono::steady_clock::now();
            double elapsed_s = std::chrono::duration<double>(now - window_start).count();
            if (elapsed_s > 0) {
                double tps = window_inferences / elapsed_s;
                // Exponential moving average
                tokens_per_second.store(0.3 * tps + 0.7 * tokens_per_second.load());
            }
            reset_window();
        }

        // Simple utilization heuristic: if we're doing inference back-to-back
        // with no idle, utilization is high. This is a proxy — real utilization
        // requires kernel-level counters (like WPR's NeuralProcessing profile).
        estimated_utilization.store(
            0.9f * estimated_utilization.load() + 0.1f * (ok ? 1.0f : 0.0f));
    }

    void record_failure(const std::string& error) {
        failures++;
        last_error = error;
        healthy = false;
    }

    void record_fallback() {
        fallbacks++;
    }

    std::string summary() const {
        char buf[256];
        snprintf(buf, sizeof(buf),
            "[%s] %lu infs, %lu fails, %.1f ms/tok avg, %.1f tok/s, util=%.0f%%%s",
            backend_id.c_str(),
            (unsigned long)inferences.load(),
            (unsigned long)failures.load(),
            (double)recent_ms.avg(),
            tokens_per_second.load(),
            estimated_utilization.load() * 100.0f,
            healthy.load() ? "" : " ⚠️ UNHEALTHY");
        return buf;
    }
};

// ── Aggregate monitor ──
// Collects metrics across all backends, tracks system-wide stats.
class BackendMonitor {
public:
    BackendMonitor();

    /// Get or create per-backend metrics
    PerBackendMetrics* for_backend(const std::string& id);

    /// Record an inference on a given backend
    void record(const std::string& backend_id, float ms, bool ok);

    /// Record a failure
    void record_failure(const std::string& backend_id, const std::string& error);

    /// Record a fallover event
    void record_fallback(const std::string& from_backend, const std::string& to_backend);

    /// System-wide totals
    uint64_t total_inferences() const;
    uint64_t total_failures() const;
    uint64_t total_fallbacks() const;

    /// Get all metric sets
    const std::vector<PerBackendMetrics*>& all_metrics() const { return metrics_; }

    /// Generate a full monitoring report
    std::string full_report() const;

    /// Live dashboard string (compact, single-line)
    std::string dashboard_line() const;

private:
    PerBackendMetrics* find_or_create(const std::string& id);
    std::vector<PerBackendMetrics*> metrics_;
    mutable std::mutex mtx_;
};
