// zaya_engine.h — Zaya inference engine declarations
//
// Architecture is now runtime-configurable via ZayaConfig. The engine accepts
// any model dimensions at init time; kernel launch grids and buffer allocations
// are sized dynamically from the config. The simple helper kernels (rmsnorm,
// matmul, silu_mul, etc.) already take size parameters — only the launch grids
// needed parameterization.
//
// Default config matches Zaya1-8B (H=2048, L=40, NQ=8, NKV=2, ...) for
// backward compatibility.
//
// The implementation lives in zaya_engine.cpp (compiled as HIP).
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <vector>
#include <string>

// ── Runtime configuration for Zaya engine ──
// This is the single source of truth for model dimensions. Pass to zaya_init().
// Paged KV cache: positions are tracked in pages of KV_PAGE_SIZE tokens.
// Each layer has a page table; pages are lazily allocated on first write.
// Peak memory is bounded by n_layers * kv_pool_pages * pagesize, not by
// n_layers * max_seq_len (which can be 18+ GB for 64K context).
static constexpr int KV_PAGE_SIZE = 16;       // tokens per page
static constexpr int KV_DEFAULT_PAGES = 256;   // default pool = 4096 tokens

struct ZayaConfig {
    int h = 2048;
    int n_layers = 40;
    int nq = 8;
    int nkv = 2;
    int hd = 128;
    int qd = 1024;    // nq * hd
    int kd = 256;     // nkv * hd
    int qkv = 1280;   // qd + kd
    int vocab = 262272;
    int n_exp = 16;
    int n_exp_t = 17; // n_exp + 1 (MOD skip token)
    int n_ff = 2048;
    int rtr_h = 256;
    int max_seq_len = 4096;     // model's max context length
    int kv_pool_pages = 256;    // actual KV pool size (pages)

    // Default: Zaya1-8B
    static ZayaConfig zaya1_8b() {
        ZayaConfig c;
        c.h = 2048; c.n_layers = 40; c.nq = 8; c.nkv = 2; c.hd = 128;
        c.qd = 1024; c.kd = 256; c.qkv = 1280;
        c.vocab = 262272; c.n_exp = 16; c.n_exp_t = 17;
        c.n_ff = 2048; c.rtr_h = 256;
        c.max_seq_len = 4096;
        c.kv_pool_pages = 256;
        return c;
    }

    // Build from ModelConfig
    static ZayaConfig from_model(int hidden, int n_layers, int n_heads,
                                  int n_kv_heads, int head_dim, int vocab,
                                  int n_exp = 16, int n_ff = 0, int rtr_h = 256,
                                  int max_seq = 4096) {
        ZayaConfig c;
        c.h = hidden;
        c.n_layers = n_layers;
        c.nq = n_heads;
        c.nkv = n_kv_heads;
        c.hd = head_dim;
        c.qd = c.nq * c.hd;
        c.kd = c.nkv * c.hd;
        c.qkv = c.qd + c.kd;
        c.vocab = vocab;
        c.n_exp = n_exp;
        c.n_exp_t = n_exp + 1;
        c.n_ff = (n_ff > 0) ? n_ff : hidden;
        c.rtr_h = rtr_h;
        c.max_seq_len = max_seq;
        int max_pages = (max_seq + KV_PAGE_SIZE - 1) / KV_PAGE_SIZE;
        c.kv_pool_pages = std::min(KV_DEFAULT_PAGES, max_pages);
        if (c.kv_pool_pages < 1) c.kv_pool_pages = 1;
        return c;
    }
};

// ── Per-layer weights ──
struct LayerW {
    __half *nw, *wq, *wk, *wv1, *wv2, *wo, *pan;
    float *cdw, *cdb, *cgw, *cgb, *ks;
    float *pahss, *pahsb, *parss, *parsb;
    float *gdw, *gdb, *rfn, *rf1, *rf1b, *rf2, *rf2b, *rout, *bb;
    __half *gu, *dn;
    float *pmhss, *pmhsb, *pmrss, *pmrsb;
};

// ── Engine state (C++ POD; extern "C" functions below use opaque pointer) ──
struct ZayaState {
    __half *d_hs = nullptr, *d_ao = nullptr, *d_tmp = nullptr;
    __half *d_fnw = nullptr, *d_lm_out = nullptr, *d_embed = nullptr;
    __half *d_conv = nullptr, *d_phs = nullptr, *d_lm_vocab = nullptr;
    // Device-side embeddings: eliminates per-token H2D copy (#5)
    __half *d_ibias = nullptr, *d_iscale = nullptr;
    // HIP graph capture (#2): record forward pass once, replay per token
    hipGraphExec_t graph_exec = nullptr;
    hipGraph_t graph = nullptr;
    bool graph_captured = false;
    int* d_token_id = nullptr;  // GPU-resident token_id for graph replay
    // Paged KV cache: flat pooled allocation [n_layers, kv_pool_pages, PAGE_SIZE, NKV, HD]
    // Page table tracks which pages are allocated per layer.
    __half *d_kcache = nullptr, *d_vcache = nullptr;
    // Linear KV cache: contiguous [n_layers, max_seq, NKV, HD] (#3) — set use_linear_kv=false for paged
    bool use_linear_kv = true;
    __half *d_vrec = nullptr;                         // [n_layers, KD/2] prev token's v_proj_delayed
    __half *d_qout = nullptr, *d_kout = nullptr, *d_vout = nullptr; // per-step prepared Q,K,V
    int *d_skip_flag = nullptr;  // fixup_skip_expert flag (now GPU-side, no sync needed #1)
    float *d_prev_rs = nullptr;
    int pos = 0, max_seq = 4096;
    // Paging: page table per layer, page_size tokens per page.
    // page_alloc[layer][page] = true if that page has been written to.
    int kv_pool_pages = 0;     // number of pages allocated in d_kcache/d_vcache
    int page_size = KV_PAGE_SIZE;
    int n_kv_pages = 0;        // total pages for max_seq
    std::vector<std::vector<bool>> page_alloc;  // [layer][page] — is allocated
    std::vector<std::vector<int>>  page_map;    // [layer][page] → pool_page_idx, or -1
    std::vector<std::vector<int>>  page_lru;    // [layer][pool_page] = last used position
    std::vector<int>               page_next_evict; // [layer] next pool page to evict (round-robin)
    // Scratch buffer for gathering non-contiguous KV pages before attention.
    __half *d_k_gather = nullptr, *d_v_gather = nullptr;
    int gather_seq_len = 0;    // current gathered length
    // Device-side page map for gather kernel (host-updated on each alloc)
    int *d_page_map = nullptr;
    int *d_gather_seq_len = nullptr; // GPU-side counter for gather skip optimization
    int *d_argmax_idx = nullptr;
    float *d_argmax_val = nullptr;
    int *d_expert_idx = nullptr; float *d_expert_wt = nullptr;
    int *d_sorted_ids = nullptr, *d_expert_counts = nullptr, *d_expert_offsets = nullptr;
    hipStream_t st = nullptr;
    std::vector<LayerW> lw;
    std::vector<bool> has_eda;
    std::vector<float> eda_scale;
    std::vector<float> embed, ibias, iscale;
};

// ── C ABI functions ──
#ifdef __cplusplus
extern "C" {
#endif

/// Initialize the Zaya engine for a specific model architecture.
/// Passing nullptr for cfg uses the default Zaya1-8B config.
ZayaState* zaya_init(const char* weights_dir, const ZayaConfig* cfg = nullptr);
int   zaya_apply_lora(ZayaState* s, const char* lora_path);
void zaya_forward(ZayaState* s, int token_id, float* logits_out);
int  zaya_forward_greedy(ZayaState* s, int token_id);
void zaya_forward_batch(ZayaState* s, const int* token_ids, float* logits_out, int B);
void zaya_reset(ZayaState* s);
void zaya_destroy(ZayaState* s);

#ifdef __cplusplus
}
#endif
