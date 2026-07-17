// backend_monitor.cpp — Performance monitoring for all backends
// Windows Task Manager NPU tab equivalent — live metrics, latency histograms,
// throughput tracking, and health status.

#include "backend_monitor.h"
#include <cstdio>
#include <algorithm>

BackendMonitor::BackendMonitor() {}

PerBackendMetrics* BackendMonitor::for_backend(const std::string& id) {
    return find_or_create(id);
}

void BackendMonitor::record(const std::string& backend_id, float ms, bool ok) {
    auto* pm = find_or_create(backend_id);
    if (pm) pm->record_inference(ms, ok);
}

void BackendMonitor::record_failure(const std::string& backend_id, const std::string& error) {
    auto* pm = find_or_create(backend_id);
    if (pm) pm->record_failure(error);
}

void BackendMonitor::record_fallback(const std::string& from_backend, const std::string& to_backend) {
    auto* pm_from = find_or_create(from_backend);
    auto* pm_to = find_or_create(to_backend);
    if (pm_from) pm_from->record_fallback();
    if (pm_to) pm_to->record_fallback();
}

uint64_t BackendMonitor::total_inferences() const {
    uint64_t total = 0;
    for (auto* pm : metrics_) total += pm->inferences.load();
    return total;
}

uint64_t BackendMonitor::total_failures() const {
    uint64_t total = 0;
    for (auto* pm : metrics_) total += pm->failures.load();
    return total;
}

uint64_t BackendMonitor::total_fallbacks() const {
    uint64_t total = 0;
    for (auto* pm : metrics_) total += pm->fallbacks.load();
    return total;
}

std::string BackendMonitor::full_report() const {
    std::string r;
    r += "Per-Backend Metrics:\n";
    if (metrics_.empty()) {
        r += "  (no data yet)\n";
        return r;
    }

    for (auto* pm : metrics_) {
        char buf[512];
        auto avg = pm->recent_ms.avg();
        auto p95 = pm->recent_ms.p95();
        auto p99 = pm->recent_ms.p99();
        auto mn = pm->recent_ms.min();
        auto mx = pm->recent_ms.max();
        snprintf(buf, sizeof(buf),
            "  %s:\n"
            "    Inferences:  %lu\n"
            "    Failures:    %lu (%.1f%%)\n"
            "    Fallbacks:   %lu\n"
            "    Latency:     avg=%.2f  p50=%.2f  p95=%.2f  p99=%.2f  min=%.2f  max=%.2f ms\n"
            "    Throughput:  %.1f tok/s\n"
            "    Utilization: %.0f%%\n"
            "    Health:      %s%s\n",
            pm->backend_id.c_str(),
            (unsigned long)pm->inferences.load(),
            (unsigned long)pm->failures.load(),
            pm->inferences.load() > 0
                ? 100.0 * pm->failures.load() / pm->inferences.load() : 0.0,
            (unsigned long)pm->fallbacks.load(),
            avg, pm->recent_ms.p50(), p95, p99, mn, mx,
            pm->tokens_per_second.load(),
            pm->estimated_utilization.load() * 100.0,
            pm->healthy.load() ? "OK" : "UNHEALTHY",
            pm->last_error.empty() ? "" : (" — " + pm->last_error).c_str());
        r += buf;
    }

    // Totals
    char buf[128];
    snprintf(buf, sizeof(buf),
        "\n  System total: %lu inferences, %lu failures, %lu fallbacks\n",
        (unsigned long)total_inferences(),
        (unsigned long)total_failures(),
        (unsigned long)total_fallbacks());
    r += buf;
    return r;
}

std::string BackendMonitor::dashboard_line() const {
    std::string r;
    for (auto* pm : metrics_) {
        char buf[128];
        auto avg = pm->recent_ms.avg();
        auto tps = pm->tokens_per_second.load();
        snprintf(buf, sizeof(buf), "%s:%.1fms/%.1ftok/s%s ",
                 pm->backend_id.c_str(), avg > 0 ? avg : 0.0, tps,
                 pm->healthy.load() ? "" : "⚠");
        r += buf;
    }
    return r;
}

PerBackendMetrics* BackendMonitor::find_or_create(const std::string& id) {
    // First, check existing
    for (auto* pm : metrics_) {
        if (pm->backend_id == id) return pm;
    }

    // Create new
    std::lock_guard<std::mutex> lock(mtx_);
    // Double-check after acquiring lock
    for (auto* pm : metrics_) {
        if (pm->backend_id == id) return pm;
    }
    auto* pm = new PerBackendMetrics();
    pm->backend_id = id;
    metrics_.push_back(pm);
    return pm;
}
