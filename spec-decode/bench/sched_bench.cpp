// Context Scheduler Benchmark — measures throughput of request admission,
// KV page allocation, H2O eviction, and batch collection
#include "scheduler/context_scheduler.h"
#include <cstdio>
#include <chrono>
#include <random>

using namespace specdecode::sched;

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

int main() {
    printf("═══ Context Scheduler Benchmark ═══\n\n");

    std::mt19937 rng(42);

    KVCacheConfig kv_cfg;
    kv_cfg.total_pages = 4096;
    kv_cfg.page_size = 16;
    kv_cfg.pages_per_request = 8;
    kv_cfg.enable_h2o_eviction = true;

    const int N_REQUESTS = 1000;
    const int MAX_PARALLEL = 8;

    // Generate random prompts
    std::vector<std::vector<int32_t>> prompts(N_REQUESTS);
    for (int i = 0; i < N_REQUESTS; i++) {
        int len = 10 + (rng() % 1000);  // 10-1009 tokens
        for (int j = 0; j < len; j++)
            prompts[i].push_back(rng() % 151936);
    }

    // 1. Request admission throughput
    printf("── Request Admission ──\n");
    {
        Scheduler sched(kv_cfg, MAX_PARALLEL);
        GenerationParams params;

        double start = now_ms();
        for (auto& p : prompts) {
            sched.enqueue(p, params);
        }
        double elapsed = now_ms() - start;

        printf("  Enqueue %d requests: %.3f ms (%.0f req/s)\n",
               N_REQUESTS, elapsed, N_REQUESTS / (elapsed / 1000.0));
    }

    // 2. Admission + KV allocation
    printf("\n── Admission + KV Allocation ──\n");
    {
        Scheduler sched(kv_cfg, MAX_PARALLEL);
        GenerationParams params;

        // Enqueue all
        for (auto& p : prompts)
            sched.enqueue(p, params);

        double start = now_ms();
        int admitted = 0;
        while (sched.pending_count() > 0 && sched.has_free_slot()) {
            if (sched.admit_next()) admitted++;
            else break;
        }
        double elapsed = now_ms() - start;

        auto& pool = sched.kv_pool();
        printf("  Admitted %d requests: %.3f ms (%.0f req/s)\n",
               admitted, elapsed, admitted / (elapsed / 1000.0));
        printf("  KV pages: %d used, %d free\n", pool.used_pages(), pool.free_pages());
    }

    // 3. Continuous batch cycling (simulate decode loop)
    printf("\n── Continuous Batch Cycling ──\n");
    {
        Scheduler sched(kv_cfg, MAX_PARALLEL);
        GenerationParams params;
        params.max_new_tokens = 256;

        double total_admit_ms = 0;
        double total_release_ms = 0;
        double total_collect_ms = 0;
        double total_token_add_ms = 0;
        int n_cycles = 0;

        // Initial admission
        for (int i = 0; i < 100; i++)
            sched.enqueue(prompts[i], params);
        sched.admit_all();

        // Simulate decode loop
        int request_idx = 100;
        for (int cycle = 0; cycle < 1000; cycle++) {
            // Collect decode batch
            double t0 = now_ms();
            auto decode = sched.collect_decode_batch();
            total_collect_ms += now_ms() - t0;

            // Add tokens to each request
            t0 = now_ms();
            for (auto* req : decode.requests) {
                sched.add_token(req->id, rng() % 151936);
                if (req->finished()) {
                    sched.release(req->id);
                }
            }
            total_token_add_ms += now_ms() - t0;

            // Admit new requests to fill freed slots
            t0 = now_ms();
            while (sched.has_free_slot() && request_idx < N_REQUESTS && sched.pending_count() > 0) {
                sched.admit_next();
            }
            total_admit_ms += now_ms() - t0;

            // Enqueue more as we go
            if (sched.pending_count() == 0 && request_idx < N_REQUESTS) {
                int batch = std::min(10, N_REQUESTS - request_idx);
                for (int i = 0; i < batch; i++)
                    sched.enqueue(prompts[request_idx++], params);
            }

            n_cycles++;
        }

        printf("  Cycles: %d\n", n_cycles);
        printf("  Batch collect:  %.3f ms total (%.3f avg)\n",
               total_collect_ms, total_collect_ms / n_cycles);
        printf("  Token add+release: %.3f ms total (%.3f avg)\n",
               total_token_add_ms, total_token_add_ms / n_cycles);
        printf("  Admission: %.3f ms total (%.3f avg)\n",
               total_admit_ms, total_admit_ms / n_cycles);
    }

    // 4. H2O eviction stress test
    printf("\n── H2O Eviction Stress ──\n");
    {
        KVCacheConfig stress_kv;
        stress_kv.total_pages = 128;
        stress_kv.page_size = 16;
        stress_kv.pages_per_request = 2;
        stress_kv.enable_h2o_eviction = true;

        Scheduler sched(stress_kv, MAX_PARALLEL);
        GenerationParams params;

        // Saturate KV cache
        for (int i = 0; i < 200; i++) {
            std::vector<int32_t> p(20, i);
            sched.enqueue(p, params);
        }

        double start = now_ms();
        int total_admitted = 0;
        while (sched.pending_count() > 0 && total_admitted < 200) {
            while (sched.has_free_slot() && sched.pending_count() > 0) {
                if (sched.admit_next()) total_admitted++;
                else break;
            }
            // Simulate decode: process one step per active request
            auto active = sched.collect_active();
            for (auto* req : active.prefill) {
                sched.begin_decoding(req->id);
            }
            for (auto* req : active.decode) {
                if (req->generated_tokens.size() < 10) {
                    sched.add_token(req->id, rng() % 1000);
                } else {
                    sched.release(req->id);
                }
            }
        }
        double elapsed = now_ms() - start;

        printf("  Admitted %d requests under KV pressure: %.3f ms\n",
               total_admitted, elapsed);
        printf("  Final KV state: %d used / %d total\n",
               sched.kv_pool().used_pages(), sched.kv_pool().total_pages());
    }

    printf("\n=== Benchmark Complete ===\n");
    return 0;
}
