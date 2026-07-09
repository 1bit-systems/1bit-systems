// Test suite for Context Scheduler
#include "scheduler/context_scheduler.h"
#include <cstdio>
#include <cassert>

using namespace specdecode::sched;

int test_request_lifecycle() {
    printf("=== Request Lifecycle ===\n");

    KVCacheConfig kv_cfg;
    kv_cfg.total_pages = 64;
    kv_cfg.page_size = 16;
    kv_cfg.pages_per_request = 4;
    kv_cfg.enable_h2o_eviction = true;

    Scheduler sched(kv_cfg, 4);

    // Enqueue a request
    std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    GenerationParams params;
    uint64_t id = sched.enqueue(prompt, params);

    assert(id == 1);
    assert(sched.pending_count() == 1);
    assert(sched.active_count() == 0);
    assert(sched.free_slots() == 4);

    // Admit it
    auto* req = sched.admit_next();
    assert(req != nullptr);
    assert(req->id == id);
    assert(req->phase == RequestPhase::kPrefilling);
    assert(req->kv_start_slot >= 0);
    assert(sched.pending_count() == 0);
    assert(sched.active_count() == 1);

    // Transition to decode
    sched.begin_decoding(id);
    req = sched.find_request(id);
    assert(req != nullptr);
    assert(req->phase == RequestPhase::kDecoding);

    // Add tokens
    sched.add_token(id, 100);
    sched.add_token(id, 200);
    req = sched.find_request(id);
    assert(req != nullptr);
    assert(req->generated_tokens.size() == 2);
    assert(req->seq_len == 2);

    // Release
    sched.release(id);
    assert(sched.active_count() == 0);

    printf("  PASS\n");
    return 0;
}

int test_priority_scheduling() {
    printf("=== Priority Scheduling ===\n");

    KVCacheConfig kv_cfg;
    kv_cfg.total_pages = 64;
    Scheduler sched(kv_cfg, 2);

    GenerationParams low_prio, high_prio;
    low_prio.priority = RequestPriority::kLow;
    high_prio.priority = RequestPriority::kHigh;

    // Enqueue low priority first, then high
    std::vector<int32_t> p1 = {1, 2, 3};
    std::vector<int32_t> p2 = {4, 5, 6};
    uint64_t low_id = sched.enqueue(p1, low_prio);
    uint64_t high_id = sched.enqueue(p2, high_prio);

    // Should admit high priority first
    auto* first = sched.admit_next();
    assert(first != nullptr);
    assert(first->params.priority == RequestPriority::kHigh);
    printf("  First admitted: priority=%d (expected High=%d)\n",
           (int)first->params.priority, (int)RequestPriority::kHigh);

    auto* second = sched.admit_next();
    assert(second != nullptr);
    assert(second->params.priority == RequestPriority::kLow);
    printf("  Second admitted: priority=%d (expected Low=%d)\n",
           (int)second->params.priority, (int)RequestPriority::kLow);

    printf("  PASS\n");
    return 0;
}

int test_kv_page_pool() {
    printf("=== KV Page Pool ===\n");

    KVCacheConfig kv_cfg;
    kv_cfg.total_pages = 16;
    kv_cfg.enable_h2o_eviction = true;

    KvPagePool pool(kv_cfg);

    assert(pool.total_pages() == 16);
    assert(pool.free_pages() == 15);  // One page is zero page
    assert(pool.used_pages() == 0);

    // Allocate a page
    int32_t p1 = pool.allocate(1);
    assert(p1 >= 0 && p1 < 15);
    assert(pool.free_pages() == 14);
    assert(pool.is_valid(p1));

    // Record some scores
    pool.record_score(p1, 0.5f);
    pool.record_score(p1, 0.3f);

    auto& page = pool.get_page(p1);
    assert(page.cumulative_attention_score == 0.8f);
    assert(page.score_count == 2);
    printf("  Page %d avg score: %.4f\n", p1, page.avg_attention_score());

    // Free owner
    pool.free_owner(1);
    assert(pool.free_pages() == 15);

    printf("  PASS\n");
    return 0;
}

int test_h2o_eviction() {
    printf("=== H2O Eviction ===\n");

    KVCacheConfig kv_cfg;
    kv_cfg.total_pages = 8;  // 7 usable + 1 zero
    kv_cfg.enable_h2o_eviction = true;

    KvPagePool pool(kv_cfg);

    // Allocate all pages
    std::vector<int32_t> pages;
    for (int i = 0; i < 7; i++) {
        int32_t p = pool.allocate(1);
        assert(p >= 0);
        pages.push_back(p);
        printf("  Allocated page %d\n", p);
    }

    // All should be used
    assert(pool.free_pages() == 0);
    assert(pool.used_pages() == 7);

    // Record high scores on some, low on others
    for (size_t i = 0; i < pages.size(); i++) {
        for (int j = 0; j < 5; j++) {
            pool.record_score(pages[i], (float)(i == 0 ? 0.001f : 0.5f + i * 0.1f));
        }
    }

    // H2O should evict the lowest-scoring page (page 0 with 0.001 score)
    // when we try to allocate one more
    int32_t new_page = pool.allocate(2);
    assert(new_page >= 0);  // Should succeed via eviction
    printf("  Evicted low-score page, allocated %d\n", new_page);

    // The zero page should not be valid
    assert(!pool.is_valid(pool.zero_page_id()));

    printf("  PASS\n");
    return 0;
}

int test_batch_collection() {
    printf("=== Batch Collection ===\n");

    KVCacheConfig kv_cfg;
    kv_cfg.total_pages = 64;
    Scheduler sched(kv_cfg, 8);

    // Add multiple requests
    GenerationParams params;
    std::vector<int32_t> p1 = {1, 2, 3};
    std::vector<int32_t> p2 = {4, 5, 6, 7, 8};
    auto id1 = sched.enqueue(p1, params);
    auto id2 = sched.enqueue(p2, params);

    sched.admit_all();
    assert(sched.active_count() == 2);

    // Both are prefill phase
    auto phases = sched.collect_active();
    assert(phases.prefill.size() == 2);
    assert(phases.decode.size() == 0);

    // Transition one to decode
    sched.begin_decoding(id1);
    phases = sched.collect_active();
    assert(phases.prefill.size() == 1);
    assert(phases.decode.size() == 1);

    // Verify batch sizes
    assert(sched.decode_batch_size() == 1);

    // Collect prefill batch
    auto prefill_batch = sched.collect_prefill_batch();
    assert(prefill_batch.requests.size() == 1);
    printf("  Prefill batch: %zu requests, %d total tokens\n",
           prefill_batch.requests.size(), prefill_batch.total_tokens);

    // Collect decode batch
    auto decode_batch = sched.collect_decode_batch();
    assert(decode_batch.batch_size == 1);

    printf("  PASS\n");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_request_lifecycle();
    failures += test_priority_scheduling();
    failures += test_kv_page_pool();
    failures += test_h2o_eviction();
    failures += test_batch_collection();

    printf("\n=== Context Scheduler Tests: %d failures ===\n", failures);
    return failures;
}
