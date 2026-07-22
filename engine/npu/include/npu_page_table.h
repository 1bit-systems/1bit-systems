#pragma once
/// Bridge between the unified KvPagePool scheduler and NPU XRT buffer objects.
/// Translates logical KV page IDs into byte offsets within NPU XRT buffer objects
/// for attention read/write.
///
/// Layout within a single token slot:
///   [K_head0 (hd*4), K_head1 (hd*4), ..., K_headN (hd*4),
///    V_head0 (hd*4), V_head1 (hd*4), ..., V_headN (hd*4)]
/// Ported from engine/npu/src/npu_page_table.zig
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>
#include "xrt_wrapper.h"

class NpuPageTable {
public:
    NpuPageTable(uint32_t page_size_tokens, uint32_t n_kv_heads, uint32_t n_layers, uint32_t head_dim)
        : page_size_tokens_(page_size_tokens)
        , n_kv_heads_(n_kv_heads)
        , n_layers_(n_layers)
        , head_dim_(head_dim)
        , total_pages_(0)
        , bo_size_(0)
        , mapped_(nullptr) {}

    ~NpuPageTable() { deinit(); }

    NpuPageTable(const NpuPageTable&) = delete;
    NpuPageTable& operator=(const NpuPageTable&) = delete;
    NpuPageTable(NpuPageTable&&) = delete;
    NpuPageTable& operator=(NpuPageTable&&) = delete;

    void allocKVBO(XrtDevice& device, uint32_t total_pages) {
        total_pages_ = total_pages;

        uint64_t total_bytes = bytesPerLayer() * n_layers_;

        printf("KV BO: %u layers * %u pages * %u tokens * %u heads * %u dim * 4 * 2 (K+V) = %lu MB\n",
               n_layers_, total_pages, page_size_tokens_, n_kv_heads_, head_dim_,
               (unsigned long)(total_bytes / (1024 * 1024)));

        kv_bo_ = XrtBuffer(device.allocBO(total_bytes, XRT_BO_FLAGS_HOST_ONLY, 3));
        bo_size_ = total_bytes;

        mapped_ = kv_bo_.map(total_bytes);
        std::memset(mapped_, 0, total_bytes);
        kv_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, total_bytes);
    }

    /// Byte offset for a specific K or V head within the BO.
    uint64_t offsetKV(uint32_t layer, uint32_t page_id, uint32_t token_in_page, uint32_t head, bool is_v) const {
        uint64_t head_bytes = head_dim_ * 4;
        uint64_t k_heads_bytes = n_kv_heads_ * head_bytes;
        uint64_t token_bytes = k_heads_bytes * 2;  // K + V

        uint64_t off = bytesPerLayer() * layer;
        off += (uint64_t)page_id * bytesPerPage();
        off += (uint64_t)token_in_page * token_bytes;
        off += (uint64_t)head * head_bytes;
        if (is_v) off += k_heads_bytes;
        return off;
    }

    /// Write one token's K at [n_kv_heads x head_dim].
    void writeK(uint32_t layer, uint32_t page_id, uint32_t token_in_page, const float* k_data) {
        uint64_t head_bytes = head_dim_ * 4;
        for (uint32_t kvh = 0; kvh < n_kv_heads_; ++kvh) {
            uint64_t off = offsetKV(layer, page_id, token_in_page, kvh, false);
            std::memcpy(&mapped_[off], &k_data[kvh * head_dim_], head_bytes);
        }
    }

    /// Write one token's V at [n_kv_heads x head_dim].
    void writeV(uint32_t layer, uint32_t page_id, uint32_t token_in_page, const float* v_data) {
        uint64_t head_bytes = head_dim_ * 4;
        for (uint32_t kvh = 0; kvh < n_kv_heads_; ++kvh) {
            uint64_t off = offsetKV(layer, page_id, token_in_page, kvh, true);
            std::memcpy(&mapped_[off], &v_data[kvh * head_dim_], head_bytes);
        }
    }

    /// Write both K and V in one call.
    void writeKV(uint32_t layer, uint32_t page_id, uint32_t token_in_page,
                 const float* k_data, const float* v_data) {
        writeK(layer, page_id, token_in_page, k_data);
        writeV(layer, page_id, token_in_page, v_data);
    }

    /// Read all KV data from pages into flat arrays.
    /// k_out/v_out: [n_tokens x n_kv_heads x head_dim].
    void readAllKV(uint32_t layer, const uint32_t* page_ids, uint32_t num_page_ids,
                   uint32_t n_tokens, float* k_out, float* v_out) const {
        uint32_t token_idx = 0;
        for (uint32_t pi = 0; pi < num_page_ids && token_idx < n_tokens; ++pi) {
            uint32_t page_id = page_ids[pi];
            for (uint32_t t_in_page = 0; t_in_page < page_size_tokens_ && token_idx < n_tokens; ++t_in_page) {
                for (uint32_t kvh = 0; kvh < n_kv_heads_; ++kvh) {
                    // Read K
                    uint64_t k_off = offsetKV(layer, page_id, t_in_page, kvh, false);
                    float* k_dst = &k_out[token_idx * n_kv_heads_ * head_dim_ + kvh * head_dim_];
                    std::memcpy(k_dst, &mapped_[k_off], head_dim_ * 4);

                    // Read V
                    uint64_t v_off = offsetKV(layer, page_id, t_in_page, kvh, true);
                    float* v_dst = &v_out[token_idx * n_kv_heads_ * head_dim_ + kvh * head_dim_];
                    std::memcpy(v_dst, &mapped_[v_off], head_dim_ * 4);
                }
                token_idx++;
            }
        }
    }

    /// Zero-fill a page (used after H2O eviction).
    void zeroFillPage(uint32_t layer, uint32_t page_id) {
        uint64_t off = bytesPerLayer() * layer + (uint64_t)page_id * bytesPerPage();
        uint64_t end = off + bytesPerPage();
        if (end <= bo_size_) {
            std::memset(&mapped_[off], 0, bytesPerPage());
        }
    }

    void deinit() {
        kv_bo_.free();
        mapped_ = nullptr;
        bo_size_ = 0;
    }

    // Accessors
    XrtBuffer& kvBO() { return kv_bo_; }
    const XrtBuffer& kvBO() const { return kv_bo_; }

private:
    uint64_t bytesPerToken() const {
        return (uint64_t)n_kv_heads_ * head_dim_ * 4 * 2; // K + V
    }
    uint64_t bytesPerPage() const {
        return (uint64_t)page_size_tokens_ * bytesPerToken();
    }
    uint64_t bytesPerLayer() const {
        return (uint64_t)total_pages_ * bytesPerPage();
    }

    XrtBuffer kv_bo_;
    uint8_t* mapped_;
    uint64_t bo_size_;
    uint32_t page_size_tokens_;
    uint32_t n_kv_heads_;
    uint32_t n_layers_;
    uint32_t head_dim_;
    uint32_t total_pages_;
};

// ─── PageMapping — tracks which pages a request owns ─────────────

struct PageMapping {
    std::vector<uint32_t> page_ids;
    uint32_t token_count = 0;
    uint32_t capacity = 0;

    static constexpr uint32_t TOKENS_PER_PAGE = 16;

    PageMapping() = default;

    explicit PageMapping(uint32_t max_pages)
        : page_ids(max_pages, 0), capacity(max_pages) {}

    uint32_t addPage(uint32_t page_id) {
        if (pageCount() >= capacity) throw std::runtime_error("PageMapping full");
        uint32_t idx = pageCount();
        page_ids[idx] = page_id;
        return idx;
    }

    uint32_t pageCount() const {
        return std::min((token_count + TOKENS_PER_PAGE - 1) / TOKENS_PER_PAGE, capacity);
    }

    uint32_t currentPage() const {
        uint32_t pc = pageCount();
        return (pc == 0) ? 0 : page_ids[pc - 1];
    }

    uint32_t currentTokenInPage() const {
        if (token_count == 0) return 0;
        return (token_count - 1) % TOKENS_PER_PAGE;
    }

    void recordToken() { token_count++; }

    void reset() {
        token_count = 0;
        std::fill(page_ids.begin(), page_ids.end(), 0);
    }
};
