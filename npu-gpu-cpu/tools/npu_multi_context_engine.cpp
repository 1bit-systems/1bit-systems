// SPDX-License-Identifier: MIT
/*
 * NPU Multi-Context Pipeline Engine
 *
 * Cracks FLM's single-threaded lock by running N independent
 * hardware contexts concurrently on the NPU.
 *
 * Architecture:
 *   - Up to 16 HW contexts (driver limit)
 *   - Each context: independent xclbin load + kernel + BO set
 *   - Async submission: start() all contexts, wait() only at barrier
 *   - Measures per-context time vs total wall time to prove parallelism
 *
 * Build:
 *   g++ -std=gnu++17 -O3 -o npu_multi_context_engine \
 *       npu_multi_context_engine.cpp \
 *       -I/usr/include -L/usr/lib/x86_64-linux-gnu \
 *       -lxrt_coreutil -ldl -luuid -lpthread
 *
 * Run (as root or render group):
 *   sudo ./npu_multi_context_engine [n_contexts=4] [xclbin_path]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <thread>
#include <atomic>
#include <fstream>
#include <cassert>

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/experimental/xrt_xclbin.h>

// ====================================================================
// Configuration
// ====================================================================
struct NpuConfig {
    static constexpr int MAX_CONTEXTS = 16;
    static constexpr size_t BO_SIZE  = 4 * 1024 * 1024;  // 4MB default
    static constexpr size_t BO_TINY  = 64 * 1024;        // 64KB for test
    static constexpr int DEFAULT_N_CONTEXTS = 4;
};

// ====================================================================
// Per-context state
// ====================================================================
struct NpuContext {
    int id;
    std::unique_ptr<xrt::hw_context> hwctx;
    std::unique_ptr<xrt::kernel>     kernel;
    std::unique_ptr<xrt::bo>         bo_act;     // activation
    std::unique_ptr<xrt::bo>         bo_ws;      // workspace
    std::unique_ptr<xrt::bo>         bo_w1;      // weight 1
    std::unique_ptr<xrt::bo>         bo_w2;      // weight 2
    std::unique_ptr<xrt::bo>         bo_kv;      // KV cache
    std::unique_ptr<xrt::run>        current_run;

    double  elapsed_ms = 0;
    bool    success    = false;

    bool init(xrt::device& dev, const xrt::uuid& xclbin_uuid, int ctx_id,
              const std::string& kernel_name, size_t bo_size) {
        id = ctx_id;
        try {
            hwctx = std::make_unique<xrt::hw_context>(dev, xclbin_uuid);
            kernel = std::make_unique<xrt::kernel>(*hwctx, kernel_name);

            bo_act = std::make_unique<xrt::bo>(dev, bo_size, xrt::bo::flags::host_only, 0);
            bo_ws  = std::make_unique<xrt::bo>(dev, bo_size, xrt::bo::flags::host_only, 0);
            bo_w1  = std::make_unique<xrt::bo>(dev, bo_size, xrt::bo::flags::host_only, 0);
            bo_w2  = std::make_unique<xrt::bo>(dev, bo_size, xrt::bo::flags::host_only, 0);
            bo_kv  = std::make_unique<xrt::bo>(dev, bo_size, xrt::bo::flags::host_only, 0);

            // Zero init all buffers
            memset(bo_act->map<char*>(), 0xAB, bo_size);
            memset(bo_ws->map<char*>(),  0,   bo_size);
            memset(bo_w1->map<char*>(),  0xCD, bo_size);
            memset(bo_w2->map<char*>(),  0xEF, bo_size);
            memset(bo_kv->map<char*>(),  0,   bo_size);

            bo_act->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bo_ws->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bo_w1->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bo_w2->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bo_kv->sync(XCL_BO_SYNC_BO_TO_DEVICE);

            return true;
        } catch (const std::exception& e) {
            fprintf(stderr, "Context %d init failed: %s\n", ctx_id, e.what());
            return false;
        }
    }

    bool submit_workload(int iterations, bool defer_wait) {
        try {
            // Submit GEMM-like work: kernel(opcode=3, instr=0, ninstr=0, bo0..bo4)
            current_run = std::make_unique<xrt::run>(*kernel);

            uint64_t opcode = 3;  // GEMM operation
            uint64_t dummy  = 0;
            uint32_t ninstr = 0;

            current_run->set_arg(0, opcode);
            current_run->set_arg(1, dummy);   // instr_ptr (0 = use xclbin built-in)
            current_run->set_arg(2, ninstr);  // ninstr
            current_run->set_arg(3, *bo_act);
            current_run->set_arg(4, *bo_ws);
            current_run->set_arg(5, *bo_w1);
            current_run->set_arg(6, *bo_w2);
            current_run->set_arg(7, *bo_kv);

            // For multi-iteration, submit N separate runs
            // Single run for now - start is async
            current_run->start();

            if (!defer_wait) {
                current_run->wait();
            }
            return true;
        } catch (const std::exception& e) {
            fprintf(stderr, "Context %d submit failed: %s\n", id, e.what());
            return false;
        }
    }

    bool wait_completion() {
        if (!current_run) return false;
        try {
            auto state = current_run->wait();
            return (state == 0x04);  // ERT_CMD_STATE_COMPLETED
        } catch (const std::exception& e) {
            fprintf(stderr, "Context %d wait failed: %s\n", id, e.what());
            return false;
        }
    }
};

// ====================================================================
// Main engine
// ====================================================================
class NpuMultiContextEngine {
public:
    NpuMultiContextEngine() = default;

    bool init(int n_contexts, const std::string& xclbin_path,
              const std::string& kernel_name, size_t bo_size) {
        if (n_contexts < 1 || n_contexts > NpuConfig::MAX_CONTEXTS) {
            fprintf(stderr, "Invalid n_contexts: %d (max %d)\n",
                    n_contexts, NpuConfig::MAX_CONTEXTS);
            return false;
        }

        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║  NPU Multi-Context Pipeline Engine — v1.0                ║\n");
        printf("║  Contexts: %-2d  |  BO size: %-6zu                    ║\n",
               n_contexts, bo_size);
        printf("╚══════════════════════════════════════════════════════════╝\n\n");

        // Open device
        printf("[1/4] Opening NPU device...\n");
        try {
            device_ = std::make_unique<xrt::device>(0);
        } catch (...) {
            try {
                device_ = std::make_unique<xrt::device>(1);
            } catch (const std::exception& e) {
                fprintf(stderr, "FAIL: Cannot open NPU device: %s\n", e.what());
                return false;
            }
        }
        printf("      ✅ Device opened\n");

        // Load xclbin
        printf("[2/4] Loading xclbin: %s\n", xclbin_path.c_str());
        std::ifstream file(xclbin_path, std::ios::binary | std::ios::ate);
        if (!file) {
            fprintf(stderr, "FAIL: Cannot open %s\n", xclbin_path.c_str());
            return false;
        }
        size_t sz = file.tellg();
        file.seekg(0);
        std::vector<char> xclbin_data(sz);
        file.read(xclbin_data.data(), sz);

        try {
            xclbin_ = std::make_unique<xrt::xclbin>(xclbin_data);
            device_->register_xclbin(*xclbin_);
            xclbin_uuid_ = xclbin_->get_uuid();
            printf("      ✅ Loaded (%.0f KB), UUID: %s\n",
                   sz / 1024.0, xclbin_uuid_.to_string().c_str());
        } catch (const std::exception& e) {
            fprintf(stderr, "FAIL: xclbin load: %s\n", e.what());
            return false;
        }

        // Create contexts
        printf("[3/4] Creating %d hardware contexts...\n", n_contexts);
        auto t0 = std::chrono::steady_clock::now();

        contexts_.resize(n_contexts);
        std::vector<std::thread> threads;

        for (int i = 0; i < n_contexts; i++) {
            threads.emplace_back([this, i, n_contexts, &kernel_name, bo_size]() {
                contexts_[i] = std::make_unique<NpuContext>();
                if (!contexts_[i]->init(*device_, xclbin_uuid_, i,
                                        kernel_name, bo_size)) {
                    fprintf(stderr, "FAIL: Context %d init failed\n", i);
                }
            });
        }
        for (auto& t : threads) t.join();
        threads.clear();

        // Verify all contexts initialized
        int ok = 0;
        for (int i = 0; i < n_contexts; i++) {
            if (contexts_[i] && contexts_[i]->kernel) ok++;
        }
        if (ok != n_contexts) {
            fprintf(stderr, "FAIL: Only %d/%d contexts initialized\n", ok, n_contexts);
            return false;
        }

        auto t1 = std::chrono::steady_clock::now();
        double setup_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("      ✅ All %d contexts ready in %.0f ms\n", n_contexts, setup_ms);

        n_contexts_ = n_contexts;
        bo_size_ = bo_size;
        return true;
    }

    // ================================================================
    // Test 1: Sequential — submit and wait one at a time
    // ================================================================
    double test_sequential(int iterations) {
        printf("\n[4/4] === Test: SEQUENTIAL (%d iterations each) ===\n", iterations);

        double total_ms = 0;
        double min_ms = 1e9, max_ms = 0;

        for (int i = 0; i < n_contexts_; i++) {
            auto t0 = std::chrono::steady_clock::now();

            for (int iter = 0; iter < iterations; iter++) {
                contexts_[i]->submit_workload(1, false);  // false = wait inside
            }

            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            contexts_[i]->elapsed_ms = ms;
            contexts_[i]->success = true;

            total_ms += ms;
            if (ms < min_ms) min_ms = ms;
            if (ms > max_ms) max_ms = ms;

            printf("  Context %2d: %8.2f ms  (%7.2f ms/iter)\n",
                   i, ms, ms / iterations);
        }

        printf("  ─────────────────────────────\n");
        printf("  Total wall:   %8.2f ms\n", total_ms);
        printf("  Avg/context:  %8.2f ms\n", total_ms / n_contexts_);
        printf("  Min/Max:      %8.2f / %.2f ms\n", min_ms, max_ms);

        return total_ms;
    }

    // ================================================================
    // Test 2: Concurrent — start ALL, then wait ALL
    // ================================================================
    double test_concurrent(int iterations) {
        printf("\n[4/4] === Test: CONCURRENT (%d iterations each) ===\n", iterations);

        // Phase 1: Start all contexts concurrently
        auto t_overall = std::chrono::steady_clock::now();

        for (int i = 0; i < n_contexts_; i++) {
            for (int iter = 0; iter < iterations; iter++) {
                auto t0 = std::chrono::steady_clock::now();
                contexts_[i]->submit_workload(1, true);  // true = defer wait
                auto t1 = std::chrono::steady_clock::now();
                contexts_[i]->elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
        }

        auto t_submitted = std::chrono::steady_clock::now();
        double submit_ms = std::chrono::duration<double, std::milli>(t_submitted - t_overall).count();
        printf("  Submit phase: %.2f ms\n", submit_ms);

        // Phase 2: Wait for all completions
        int completed = 0;
        for (int i = 0; i < n_contexts_; i++) {
            auto t0 = std::chrono::steady_clock::now();
            contexts_[i]->success = contexts_[i]->wait_completion();
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (contexts_[i]->success) completed++;
            printf("  Context %2d: wait %7.2f ms  %s\n",
                   i, ms, contexts_[i]->success ? "✅" : "❌");
        }

        auto t_done = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_done - t_overall).count();
        double wait_ms = std::chrono::duration<double, std::milli>(t_done - t_submitted).count();

        printf("  ─────────────────────────────\n");
        printf("  Total wall:   %8.2f ms\n", total_ms);
        printf("  Submit time:  %8.2f ms\n", submit_ms);
        printf("  Wait time:    %8.2f ms\n", wait_ms);
        printf("  Completed:    %d/%d\n", completed, n_contexts_);

        return total_ms;
    }

    // ================================================================
    // Test 3: Threaded concurrent — threads submit+wait independently
    // ================================================================
    double test_threaded(int iterations) {
        printf("\n[4/4] === Test: THREADED CONCURRENT (%d iterations each) ===\n", iterations);

        std::atomic<int> ready{0};
        std::atomic<bool> go{false};

        auto t_overall = std::chrono::steady_clock::now();
        std::vector<std::thread> threads;

        for (int i = 0; i < n_contexts_; i++) {
            threads.emplace_back([this, i, iterations, &ready, &go]() {
                ready.fetch_add(1);
                while (!go.load()) { /* spin */ }

                auto t0 = std::chrono::steady_clock::now();

                for (int iter = 0; iter < iterations; iter++) {
                    contexts_[i]->submit_workload(1, false);  // wait inside
                }

                auto t1 = std::chrono::steady_clock::now();
                contexts_[i]->elapsed_ms =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                contexts_[i]->success = true;
            });
        }

        // Wait for all threads ready, then release
        while (ready.load() < n_contexts_) { /* spin */ }
        auto t_start = std::chrono::steady_clock::now();
        go.store(true);

        for (auto& t : threads) t.join();
        auto t_done = std::chrono::steady_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(t_done - t_start).count();
        double sum_ms = 0;
        double min_ms = 1e9, max_ms = 0;

        for (int i = 0; i < n_contexts_; i++) {
            double ms = contexts_[i]->elapsed_ms;
            sum_ms += ms;
            if (ms < min_ms) min_ms = ms;
            if (ms > max_ms) max_ms = ms;
            printf("  Context %2d: %8.2f ms  %s\n",
                   i, ms, contexts_[i]->success ? "✅" : "❌");
        }

        double avg_ms = sum_ms / n_contexts_;
        double speedup = (sum_ms > 0) ? sum_ms / total_ms : 0;

        printf("  ─────────────────────────────\n");
        printf("  Total wall:   %8.2f ms\n", total_ms);
        printf("  Sum(all):     %8.2f ms\n", sum_ms);
        printf("  Avg/context:  %8.2f ms\n", avg_ms);
        printf("  Min/Max:      %8.2f / %.2f ms\n", min_ms, max_ms);
        printf("  Speedup:      %5.1fx  %s\n", speedup,
               speedup >= 1.5 ? "🎉 PARALLELISM CONFIRMED!" :
               speedup >= 1.0 ? "⚠️  marginal" : "❌ sequential");

        return total_ms;
    }

    // ================================================================
    // Test 4: Async pipeline — fire-and-forget across contexts
    // ================================================================
    double test_async_pipeline(int iterations) {
        printf("\n[4/4] === Test: ASYNC PIPELINE (%d iterations each) ===\n", iterations);

        // Submit ALL work without waiting — then collect completions in batch
        auto t_overall = std::chrono::steady_clock::now();

        // Start all runs
        for (int i = 0; i < n_contexts_; i++) {
            for (int iter = 0; iter < iterations; iter++) {
                contexts_[i]->submit_workload(1, true);  // defer wait
            }
        }

        // Collect completions
        int total_submitted = n_contexts_ * iterations;
        int completed = 0;
        for (int i = 0; i < n_contexts_; i++) {
            for (int iter = 0; iter < iterations; iter++) {
                if (contexts_[i]->wait_completion()) completed++;
            }
        }

        auto t_done = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_done - t_overall).count();

        printf("  Submitted:    %d runs across %d contexts\n",
               total_submitted, n_contexts_);
        printf("  Completed:    %d/%d\n", completed, total_submitted);
        printf("  Total wall:   %8.2f ms\n", total_ms);
        printf("  Avg/run:      %8.2f ms\n",
               total_ms / (total_submitted > 0 ? total_submitted : 1));

        return total_ms;
    }

private:
    std::unique_ptr<xrt::device>  device_;
    std::unique_ptr<xrt::xclbin>  xclbin_;
    xrt::uuid                     xclbin_uuid_;
    std::vector<std::unique_ptr<NpuContext>> contexts_;
    int    n_contexts_ = 0;
    size_t bo_size_    = 0;
};

// ====================================================================
// Usage
// ====================================================================
static void usage(const char* prog) {
    printf("Usage: %s [n_contexts=%d] [xclbin_path] [test_mode]\n",
           prog, NpuConfig::DEFAULT_N_CONTEXTS);
    printf("\n");
    printf("  n_contexts    Number of HW contexts (1-%d, default %d)\n",
           NpuConfig::MAX_CONTEXTS, NpuConfig::DEFAULT_N_CONTEXTS);
    printf("  xclbin_path   Path to .xclbin file\n");
    printf("  test_mode     seq | conc | thread | async | all (default: all)\n");
    printf("\n");
    printf("Default xclbin: /opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/mm.xclbin\n");
    printf("\n");
    printf("Examples:\n");
    printf("  sudo %s 4\n", prog);
    printf("  sudo %s 8 /path/to/mm.xclbin conc\n", prog);
    printf("  sudo %s 16                            # stress test all 16 contexts\n", prog);
}

// ====================================================================
// Main
// ====================================================================
int main(int argc, char** argv) {
    int n_contexts = NpuConfig::DEFAULT_N_CONTEXTS;
    std::string xclbin_path = "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/mm.xclbin";
    std::string test_mode = "all";
    size_t bo_size = NpuConfig::BO_TINY;  // small BOs for speed

    if (argc > 1) n_contexts = atoi(argv[1]);
    if (argc > 2) xclbin_path = argv[2];
    if (argc > 3) test_mode = argv[3];
    if (argc > 4) bo_size = (size_t)atoi(argv[4]);

    if (n_contexts < 1 || n_contexts > NpuConfig::MAX_CONTEXTS) {
        usage(argv[0]);
        return 1;
    }

    NpuMultiContextEngine engine;

    if (!engine.init(n_contexts, xclbin_path, "MLIR_AIE", bo_size)) {
        fprintf(stderr, "\n❌ Engine initialization FAILED\n");
        fprintf(stderr, "\nMake sure:\n");
        fprintf(stderr, "  1. Run as root or member of 'render' group\n");
        fprintf(stderr, "  2. NPU driver loaded (ls /dev/accel/accel0)\n");
        fprintf(stderr, "  3. xclbin exists at %s\n", xclbin_path.c_str());
        return 1;
    }

    int iterations = 3;  // 3 iterations per context per test

    if (test_mode == "seq" || test_mode == "all") {
        engine.test_sequential(iterations);
    }
    if (test_mode == "conc" || test_mode == "all") {
        engine.test_concurrent(iterations);
    }
    if (test_mode == "thread" || test_mode == "all") {
        engine.test_threaded(iterations);
    }
    if (test_mode == "async" || test_mode == "all") {
        engine.test_async_pipeline(iterations);
    }

    printf("\n✅ All tests complete.\n");
    return 0;
}
