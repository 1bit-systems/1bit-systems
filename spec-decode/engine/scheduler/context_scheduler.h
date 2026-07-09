#pragma once
// Context Scheduler — Continuous Batching + KV Cache Management + H2O Eviction
//
// Manages concurrent inference requests through:
//   1. Request queuing with priority levels
//   2. KV cache page pool with H2O (Heavy-Hitter Oracle) eviction
//   3. Prefill/decode phase scheduling
//   4. Slot-based concurrent request tracking
//
// Design ported from engine/fusion/sched/scheduler.zig with C++23 enhancements.

#include <cstdint>
#include <vector>
#include <deque>
#include <queue>
#include <span>
#include <memory>
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>
#include <numeric>

namespace specdecode::sched {

// ─── Forward Declarations ───────────────────────────────────────────────────

enum class RequestPriority : uint8_t {
    kLow = 0,
    kNormal = 1,
    kHigh = 2,
    kCritical = 3,
};

enum class RequestPhase : uint8_t {
    kPending,       // Not yet started
    kPrefilling,    // Processing prompt
    kDecoding,      // Generating tokens
    kComplete,      // Finished (EOS or max tokens)
    kCancelled,
};

// ─── Generation Parameters ──────────────────────────────────────────────────

struct GenerationParams {
    int32_t max_new_tokens = 1024;
    int32_t max_context = 4096;
    float temperature = 0.0f;
    float top_p = 1.0f;
    int32_t top_k = -1;
    int32_t eos_token_id = 151645;
    int32_t pad_token_id = 151643;
    bool use_spec_decode = true;
    int32_t spec_block_size = 7;       // Draft tokens per speculative step
    RequestPriority priority = RequestPriority::kNormal;
};

// ─── Request ────────────────────────────────────────────────────────────────

struct Request {
    uint64_t id = 0;
    RequestPhase phase = RequestPhase::kPending;
    GenerationParams params;

    // Token data
    std::vector<int32_t> prompt_tokens;
    std::vector<int32_t> generated_tokens;

    // KV cache tracking
    int32_t kv_start_slot = -1;         // First logical KV slot
    int32_t num_kv_slots = 0;           // Number of slots allocated
    int32_t seq_len = 0;                // Current sequence length (prompt + generated)

    // Timing
    std::chrono::steady_clock::time_point arrival_time;
    std::chrono::steady_clock::time_point start_time;

    // Stats
    int32_t num_prefill_tokens = 0;
    int32_t num_decode_tokens = 0;
    int32_t num_accepted_draft = 0;
    int32_t num_rejected_draft = 0;

    bool finished() const noexcept {
        return phase == RequestPhase::kComplete ||
               phase == RequestPhase::kCancelled;
    }

    bool is_pending() const noexcept { return phase == RequestPhase::kPending; }
    bool is_prefilling() const noexcept { return phase == RequestPhase::kPrefilling; }
    bool is_decoding() const noexcept { return phase == RequestPhase::kDecoding; }

    int32_t total_tokens() const noexcept {
        return (int32_t)(prompt_tokens.size() + generated_tokens.size());
    }

    float acceptance_rate() const noexcept {
        auto total = num_accepted_draft + num_rejected_draft;
        return total > 0 ? (float)num_accepted_draft / total : 0.0f;
    }
};

// ─── KV Cache Configuration ─────────────────────────────────────────────────

struct KVCacheConfig {
    int32_t num_layers = 28;
    int32_t num_kv_heads = 8;
    int32_t head_dim = 128;
    int32_t page_size = 16;          // Tokens per page
    int32_t total_pages = 1024;      // Total KV cache pages
    int32_t pages_per_request = 16;  // Default pages allocated per request
    bool enable_h2o_eviction = true;
    float h2o_score_threshold = 0.01f;
};

// ─── KV Page ────────────────────────────────────────────────────────────────

struct KvPage {
    int32_t page_id = -1;
    uint64_t owner_id = 0;
    int32_t token_start = 0;
    int32_t token_count = 0;

    // H2O eviction metadata
    float cumulative_attention_score = 0.0f;
    int32_t score_count = 0;
    float avg_attention_score() const noexcept {
        return score_count > 0 ? cumulative_attention_score / (float)score_count : 0.0f;
    }

    bool evicted = false;
    uint64_t last_access_seq = 0;
};

// ─── KV Page Pool ───────────────────────────────────────────────────────────

class KvPagePool {
public:
    KvPagePool(const KVCacheConfig& cfg)
        : cfg_(cfg) {
        pages_.resize(cfg.total_pages);

        // Reserve page 0 as zero page (for H2O eviction)
        zero_page_id_ = cfg.total_pages - 1;
        for (int32_t i = 0; i < cfg.total_pages - 1; i++) {
            free_list_.push_back(i);
            pages_[i].page_id = i;
        }
        pages_[zero_page_id_].page_id = zero_page_id_;
        pages_[zero_page_id_].evicted = true;  // Zero page is always "evicted"
    }

    // Allocate a page for a request. Returns page_id or -1 on failure.
    int32_t allocate(uint64_t owner_id) {
        if (!free_list_.empty()) {
            int32_t pid = free_list_.back();
            free_list_.pop_back();
            auto& page = pages_[pid];
            page.owner_id = owner_id;
            page.token_start = 0;
            page.token_count = 0;
            page.evicted = false;
            page.cumulative_attention_score = 0.0f;
            page.score_count = 0;
            page.last_access_seq = access_seq_++;
            return pid;
        }

        // No free pages — try H2O eviction
        if (cfg_.enable_h2o_eviction) {
            return evict_lowest_score();
        }

        return -1; // OOM
    }

    // Allocate multiple pages at once
    std::vector<int32_t> allocate_batch(uint64_t owner_id, int32_t count) {
        std::vector<int32_t> result;
        result.reserve(count);
        for (int32_t i = 0; i < count; i++) {
            int32_t pid = allocate(owner_id);
            if (pid < 0) break;
            result.push_back(pid);
        }
        return result;
    }

    // Free all pages owned by a request
    void free_owner(uint64_t owner_id) {
        for (auto& page : pages_) {
            if (page.owner_id == owner_id && !page.evicted && page.page_id != zero_page_id_) {
                page.owner_id = 0;
                page.evicted = false;
                page.cumulative_attention_score = 0.0f;
                page.score_count = 0;
                free_list_.push_back(page.page_id);
            }
        }
    }

    // Record attention score for a page (for H2O scoring)
    void record_score(int32_t page_id, float score) {
        if (page_id >= 0 && page_id < (int32_t)pages_.size() && page_id != zero_page_id_) {
            auto& page = pages_[page_id];
            page.cumulative_attention_score += std::abs(score);
            page.score_count++;
            page.last_access_seq = access_seq_++;
        }
    }

    // Get page info
    const KvPage& get_page(int32_t page_id) const {
        return pages_[page_id];
    }

    // Check if page is valid (not evicted, not zero page)
    bool is_valid(int32_t page_id) const {
        return page_id >= 0 && page_id < (int32_t)pages_.size() &&
               !pages_[page_id].evicted && page_id != zero_page_id_;
    }

    int32_t total_pages() const noexcept { return cfg_.total_pages; }
    int32_t free_pages() const noexcept { return (int32_t)free_list_.size(); }
    int32_t used_pages() const noexcept { return cfg_.total_pages - 1 - free_pages(); }
    int32_t zero_page_id() const noexcept { return zero_page_id_; }

    // Reset pool
    void reset() {
        free_list_.clear();
        for (int32_t i = 0; i < cfg_.total_pages - 1; i++) {
            free_list_.push_back(i);
            pages_[i] = KvPage{};
            pages_[i].page_id = i;
        }
    }

private:
    // Evict the page with the lowest average attention score
    int32_t evict_lowest_score() {
        int32_t worst_page = -1;
        float worst_score = std::numeric_limits<float>::max();
        uint64_t oldest_access = std::numeric_limits<uint64_t>::max();

        for (auto& page : pages_) {
            if (page.evicted || page.page_id == zero_page_id_ || page.owner_id == 0)
                continue;

            float avg = page.avg_attention_score();
            // Prefer evicting pages with low scores; tiebreak by age (LRU within score band)
            if (avg < worst_score ||
                (std::abs(avg - worst_score) < cfg_.h2o_score_threshold &&
                 page.last_access_seq < oldest_access)) {
                worst_score = avg;
                oldest_access = page.last_access_seq;
                worst_page = page.page_id;
            }
        }

        if (worst_page >= 0) {
            auto& page = pages_[worst_page];
            page.evicted = true;
            page.owner_id = 0;
            page.cumulative_attention_score = 0.0f;
            page.score_count = 0;
            return worst_page;
        }

        return -1;
    }

    KVCacheConfig cfg_;
    std::vector<KvPage> pages_;
    std::vector<int32_t> free_list_;
    int32_t zero_page_id_ = 0;
    uint64_t access_seq_ = 1;
};

// ─── Scheduler ──────────────────────────────────────────────────────────────

class Scheduler {
public:
    Scheduler(const KVCacheConfig& kv_cfg, int32_t max_parallel = 8)
        : kv_pool_(std::make_unique<KvPagePool>(kv_cfg)),
          kv_cfg_(kv_cfg),
          slots_(max_parallel),
          max_parallel_(max_parallel) {}

    // Enqueue a new request (returns request ID)
    uint64_t enqueue(
        std::span<const int32_t> prompt_tokens,
        const GenerationParams& params
    ) {
        uint64_t id = next_id_++;
        auto req = std::make_unique<Request>();
        req->id = id;
        req->params = params;
        req->prompt_tokens.assign(prompt_tokens.begin(), prompt_tokens.end());
        req->phase = RequestPhase::kPending;
        req->arrival_time = std::chrono::steady_clock::now();

        pending_queue_.push_back(std::move(req));
        num_pending_++;
        return id;
    }

    // Admit next pending request to a free slot
    // Returns the admitted request, or nullptr if no free slots or no pending
    Request* admit_next() {
        if (num_pending_ == 0) return nullptr;

        int32_t slot = find_free_slot();
        if (slot < 0) return nullptr;

        // Find highest priority pending request
        auto best_it = pending_queue_.end();
        RequestPriority best_prio = RequestPriority::kLow;

        for (auto it = pending_queue_.begin(); it != pending_queue_.end(); ++it) {
            if ((*it)->params.priority > best_prio ||
                ((*it)->params.priority == best_prio && best_it == pending_queue_.end())) {
                best_prio = (*it)->params.priority;
                best_it = it;
            }
        }

        if (best_it == pending_queue_.end()) return nullptr;

        auto req = std::move(*best_it);
        pending_queue_.erase(best_it);
        num_pending_--;

        // Allocate KV pages
        int32_t needed_pages = (int32_t)req->prompt_tokens.size() / kv_cfg_.page_size + 1;
        needed_pages = std::max(needed_pages, kv_cfg_.pages_per_request);

        auto pages = kv_pool_->allocate_batch(req->id, needed_pages);
        req->kv_start_slot = pages.empty() ? -1 : pages[0];
        req->num_kv_slots = (int32_t)pages.size();
        req->phase = RequestPhase::kPrefilling;
        req->start_time = std::chrono::steady_clock::now();

        slots_[slot] = std::move(req);
        num_active_++;
        return slots_[slot].get();
    }

    // Transition request from prefill to decode phase
    void begin_decoding(uint64_t request_id) {
        auto* req = find_request(request_id);
        if (req && req->phase == RequestPhase::kPrefilling) {
            req->phase = RequestPhase::kDecoding;
        }
    }

    // Add a generated token to a request
    void add_token(uint64_t request_id, int32_t token) {
        auto* req = find_request(request_id);
        if (!req) return;

        req->generated_tokens.push_back(token);
        req->seq_len++;

        if (token == req->params.eos_token_id ||
            (int32_t)req->generated_tokens.size() >= req->params.max_new_tokens) {
            req->phase = RequestPhase::kComplete;
        }
    }

    // Release a request (free its resources)
    void release(uint64_t request_id) {
        for (auto& slot : slots_) {
            if (slot && slot->id == request_id) {
                kv_pool_->free_owner(request_id);
                slot.reset();
                num_active_--;
                return;
            }
        }
    }

    // Collect active requests by phase
    struct PhaseCollection {
        std::vector<Request*> prefill;
        std::vector<Request*> decode;
        std::vector<Request*> complete;
    };

    PhaseCollection collect_active() const {
        PhaseCollection result;
        for (auto& slot : slots_) {
            if (!slot) continue;
            switch (slot->phase) {
                case RequestPhase::kPrefilling:
                    result.prefill.push_back(slot.get());
                    break;
                case RequestPhase::kDecoding:
                    result.decode.push_back(slot.get());
                    break;
                case RequestPhase::kComplete:
                    result.complete.push_back(slot.get());
                    break;
                default:
                    break;
            }
        }
        return result;
    }

    // Collect pending requests sorted by priority
    std::vector<Request*> collect_pending() const {
        std::vector<Request*> result;
        for (auto& r : pending_queue_)
            result.push_back(r.get());
        std::sort(result.begin(), result.end(), [](const Request* a, const Request* b) {
            if (a->params.priority != b->params.priority)
                return a->params.priority > b->params.priority;
            return a->arrival_time < b->arrival_time;
        });
        return result;
    }

    // Find a request by ID
    Request* find_request(uint64_t id) {
        for (auto& slot : slots_) {
            if (slot && slot->id == id) return slot.get();
        }
        return nullptr;
    }

    const Request* find_request(uint64_t id) const {
        for (auto& slot : slots_) {
            if (slot && slot->id == id) return slot.get();
        }
        return nullptr;
    }

    // Stats
    int32_t active_count() const noexcept { return num_active_; }
    int32_t pending_count() const noexcept { return num_pending_; }
    int32_t max_parallel() const noexcept { return max_parallel_; }
    int32_t free_slots() const noexcept { return max_parallel_ - num_active_; }
    bool has_free_slot() const noexcept { return num_active_ < max_parallel_; }

    KvPagePool& kv_pool() { return *kv_pool_; }
    const KvPagePool& kv_pool() const { return *kv_pool_; }

    // Get decode batch size (number of requests currently decoding)
    int32_t decode_batch_size() const {
        int32_t count = 0;
        for (auto& slot : slots_) {
            if (slot && slot->phase == RequestPhase::kDecoding) count++;
        }
        return count;
    }

    // Admit all pending requests that fit
    int32_t admit_all() {
        int32_t admitted = 0;
        while (has_free_slot() && num_pending_ > 0) {
            if (admit_next()) admitted++;
            else break;
        }
        return admitted;
    }

    // Prefill batch: collect all prefill requests with their prompts
    struct PrefillBatch {
        std::vector<Request*> requests;
        std::vector<int32_t> token_counts;
        int32_t total_tokens = 0;
    };

    PrefillBatch collect_prefill_batch() const {
        PrefillBatch batch;
        for (auto& slot : slots_) {
            if (slot && slot->phase == RequestPhase::kPrefilling) {
                batch.requests.push_back(slot.get());
                batch.token_counts.push_back((int32_t)slot->prompt_tokens.size());
                batch.total_tokens += (int32_t)slot->prompt_tokens.size();
            }
        }
        return batch;
    }

    // Decode batch: collect all decoding requests
    struct DecodeBatch {
        std::vector<Request*> requests;
        int32_t batch_size = 0;
    };

    DecodeBatch collect_decode_batch() const {
        DecodeBatch batch;
        for (auto& slot : slots_) {
            if (slot && slot->phase == RequestPhase::kDecoding) {
                batch.requests.push_back(slot.get());
                batch.batch_size++;
            }
        }
        return batch;
    }

    // Reset scheduler
    void reset() {
        for (auto& slot : slots_) slot.reset();
        for (auto& slot : slots_) slot.reset();
        pending_queue_.clear();
        kv_pool_->reset();
        num_active_ = 0;
        num_pending_ = 0;
        next_id_ = 1;
    }

private:
    int32_t find_free_slot() const {
        for (int32_t i = 0; i < max_parallel_; i++) {
            if (!slots_[i]) return i;
        }
        return -1;
    }

    // Priority-sorted pending queue (actually a deque, sorted by priority on admit)
    std::deque<std::unique_ptr<Request>> pending_queue_;
    std::vector<std::unique_ptr<Request>> slots_;
    std::unique_ptr<KvPagePool> kv_pool_;

    KVCacheConfig kv_cfg_;
    int32_t max_parallel_;
    int32_t num_active_ = 0;
    int32_t num_pending_ = 0;
    uint64_t next_id_ = 1;
};

} // namespace specdecode::sched
