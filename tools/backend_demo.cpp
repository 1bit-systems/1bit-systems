// backend_demo.cpp — Backend Manager Live Dashboard Demo
// Windows Task Manager NPU tab equivalent for the Zaya engine.
// Shows backend discovery, selection, inference with failover, and live metrics.
//
// Build: cmake --build . --target backend_demo -j8
// Run:   ./build/backend_demo

#include "backend_manager.h"
#include "backend_plugin.h"
#include "backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>
#include <signal.h>
#include <unistd.h>

static volatile bool keep_running = true;
void handle_sigint(int) { keep_running = false; }

// ── Live dashboard (redraws every second) ──
void dashboard_thread(BackendManager* mgr) {
    int tick = 0;
    while (keep_running) {
        printf("\033[2J\033[H");  // clear screen, home cursor

        printf("╔══════════════════════════════════════════════════╗\n");
        printf("║   Zaya Backend Manager — Live Dashboard         ║\n");
        printf("║   (Windows Task Manager NPU tab equivalent)     ║\n");
        printf("╚══════════════════════════════════════════════════╝\n\n");

        // Active backend
        auto* active = mgr->active_info();
        if (active) {
            printf("  Active:  %s (%s)\n", active->id.c_str(), active->description.c_str());
            printf("  Tier:    %s\n", tier_name(active->tier));
            printf("  Status:  %s\n", active->functional ? "✅ Functional" : "❌ Failed");
            printf("  Score:   %.1f ms/tok\n", active->score);
            printf("  Infs:    %lu total, %lu failed\n",
                   (unsigned long)active->total_inferences,
                   (unsigned long)active->failed_inferences);
        }

        // All backends
        printf("\n  ── All Backends ──\n");
        for (auto& b : mgr->backends()) {
            const char* status = "✗";
            if (b.available && b.functional) status = "✓";
            else if (b.available && !b.functional) status = "⚠";
            printf("  %s %-15s [%s] %.0f ms/tok  %lu infs\n",
                   status, b.id.c_str(),
                   b.available ? "avail" : "offline",
                   b.score, (unsigned long)b.total_inferences);
        }

        // Monitor stats
        auto* monitor = mgr->monitor_stats();
        printf("\n  ── Metrics ──\n");
        for (auto* pm : monitor->all_metrics()) {
            printf("  %s\n", pm->summary().c_str());
        }
        printf("\n  System: %lu total inferences, %lu failures, %lu fallbacks\n",
               (unsigned long)monitor->total_inferences(),
               (unsigned long)monitor->total_failures(),
               (unsigned long)monitor->total_fallbacks());

        printf("\n  ── Dashboard line ──\n");
        printf("  %s\n", monitor->dashboard_line().c_str());

        printf("\n  T+%ds | Press Ctrl+C to exit\n", tick);
        fflush(stdout);

        // Wait, checking every 100ms for interrupt
        for (int i = 0; i < 10 && keep_running; i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        tick++;
    }
}

// ── Simulate inference workload ──
void inference_worker(BackendManager* mgr, int num_tokens, int delay_ms) {
    printf("  Worker: generating %d tokens with %dms delay...\n", num_tokens, delay_ms);

    for (int i = 0; i < num_tokens && keep_running; i++) {
        // Simulate: just call generate (which internally handles failover)
        // In real usage, you'd pass actual token IDs from a prompt.
        // Here we just send token 0 and show the mechanism.
        int result = mgr->generate(0);

        if (result < 0) {
            printf("\n  ❌ Token %d: ALL BACKENDS FAILED\n", i);
            break;
        }

        if ((i + 1) % 5 == 0) {
            printf("  ✅ Tokens %d/%d\n", i + 1, num_tokens);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }

    printf("  Worker: done (%d tokens generated)\n", num_tokens);
    keep_running = false;
}

int main(int argc, char** argv) {
    signal(SIGINT, handle_sigint);

    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║   Zaya Backend Manager Demo              ║\n");
    printf("║   Windows ML-style backend orchestrator  ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    int num_tokens = 50;
    int delay_ms = 50;

    // Parse args
    if (argc > 1) num_tokens = atoi(argv[1]);
    if (argc > 2) delay_ms = atoi(argv[2]);

    // ── Create BackendManager ──
    auto& mgr = backend_manager();

    // ── Phase 1: Discover ──
    printf("Phase 1: Hardware Discovery\n");
    printf("─────────────────────────────\n");
    mgr.discover();

    // ── Phase 2: Strategy configuration ──
    printf("Phase 2: Configure Strategy\n");
    printf("────────────────────────────\n");
    mgr.set_strategy(SelectionStrategy::FASTEST);
    mgr.set_fallback_policy(FallbackPolicy::SEQUENTIAL);

    // ── Phase 3: Init ──
    printf("\nPhase 3: Initialize Backend\n");
    printf("─────────────────────────────\n");

    // In a real deployment, this would load actual weights.
    // For the demo, we check if any backend can init.
    ModelConfig cfg;
    const char* home = getenv("HOME");
    std::string weights_dir = (home && home[0]) ? std::string(home) + "/.local/share/1bit-systems/weights" : "/tmp/zaya_weights";

    bool inited = mgr.init(cfg, weights_dir);
    if (!inited) {
        printf("\n⚠  No backend initialized (expected without weights).\n");
        printf("   Running in demo/display mode.\n");
    }

    // ── Phase 4: Benchmark ──
    if (inited) {
        printf("\nPhase 4: Benchmark\n");
        printf("───────────────────\n");
        mgr.benchmark_all(5);
    }

    // ── Phase 5: Live dashboard ──
    printf("\nPhase 5: Live Dashboard (Task Manager NPU tab)\n");
    printf("───────────────────────────────────────────────\n");

    // Start dashboard thread
    std::thread dash(dashboard_thread, &mgr);

    // Run inference workload in main thread
    if (inited) {
        inference_worker(&mgr, num_tokens, delay_ms);
    } else {
        // Still show the dashboard for display
        std::this_thread::sleep_for(std::chrono::seconds(5));
        keep_running = false;
    }

    dash.join();

    // ── Phase 6: Report ──
    printf("\n\nPhase 6: Final Report\n");
    printf("──────────────────────\n");
    printf("%s\n", mgr.report().c_str());

    // ── Plugin discovery ──
    printf("\nPlugin Scan:\n");
    printf("─────────────\n");
    auto plugins = load_backend_plugins();
    if (plugins.empty()) {
        printf("  No backend plugins found (expected — none deployed yet)\n");
    } else {
        printf("  Found %zu plugin(s):\n", plugins.size());
        for (auto* p : plugins) {
            printf("    %s v%s — %s\n", p->id().c_str(), p->version().c_str(), p->description().c_str());
            delete p;
        }
    }

    mgr.destroy();
    printf("\nDone.\n");
    return 0;
}
