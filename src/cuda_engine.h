// cuda_engine.h — CUDA inference engine declarations
//
// Mirrors zaya_engine.h for NVIDIA GPUs. Provides the same ZayaConfig
// interface but backed by CUDA kernels and cuBLAS/cuDNN where available.
// On Apple Silicon with no NVIDIA GPU, this file is unused; see backend_metal.mm.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <string>

// ── Runtime configuration for CUDA engine ──
// Identical semantics to ZayaConfig in zaya_engine.h.
static constexpr int CUDA_KV_PAGE_SIZE = 16;
static constexpr int CUDA_KV_DEFAULT_PAGES = 256;

struct CudaConfig {
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
    int n_exp_t = 17;
    int n_ff = 2048;
    int rtr_h = 256;
    int max_seq_len = 4096;
    int kv_pool_pages = CUDA_KV_DEFAULT_PAGES;

    static CudaConfig cuda_default() {
        CudaConfig c;
        c.h = 2048; c.n_layers = 40; c.nq = 8; c.nkv = 2; c.hd = 128;
        c.qd = 1024; c.kd = 256; c.qkv = 1280;
        c.vocab = 262272; c.n_exp = 16; c.n_exp_t = 17;
        c.n_ff = 2048; c.rtr_h = 256;
        c.max_seq_len = 4096;
        c.kv_pool_pages = CUDA_KV_DEFAULT_PAGES;
        return c;
    }

    static CudaConfig from_model(int hidden, int n_layers, int n_heads,
                                  int n_kv_heads, int head_dim, int vocab,
                                  int n_exp = 16, int n_ff = 0, int rtr_h = 256,
                                  int max_seq = 4096) {
        CudaConfig c;
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
        int max_pages = (max_seq + CUDA_KV_PAGE_SIZE - 1) / CUDA_KV_PAGE_SIZE;
        c.kv_pool_pages = std::min(CUDA_KV_DEFAULT_PAGES, max_pages);
        if (c.kv_pool_pages < 1) c.kv_pool_pages = 1;
        return c;
    }
};

// ── CUDA engine state ──
struct CudaState {
    half *d_hs = nullptr, *d_ao = nullptr, *d_tmp = nullptr;
    half *d_fnw = nullptr, *d_lm_out = nullptr, *d_embed = nullptr;
    half *d_conv = nullptr, *d_phs = nullptr, *d_lm_vocab = nullptr;
    half *d_ibias = nullptr, *d_iscale = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaGraph_t graph = nullptr;
    bool graph_captured = false;
    int* d_token_id = nullptr;
    half *d_kcache = nullptr, *d_vcache = nullptr;
    bool use_linear_kv = true;
    half *d_vrec = nullptr;
    half *d_qout = nullptr, *d_kout = nullptr, *d_vout = nullptr;
    int *d_skip_flag = nullptr;
    float *d_prev_rs = nullptr;
    int pos = 0, max_seq = 4096;
    int kv_pool_pages = 0;
    int page_size = CUDA_KV_PAGE_SIZE;
    int n_kv_pages = 0;
    std::vector<std::vector<bool>> page_alloc;
    std::vector<std::vector<int>>  page_map;
    std::vector<std::vector<int>>  page_lru;
    std::vector<int>               page_next_evict;
    half *d_k_gather = nullptr, *d_v_gather = nullptr;
    int gather_seq_len = 0;
    int *d_page_map = nullptr;
    int *d_gather_seq_len = nullptr;
    int *d_argmax_idx = nullptr;
    float *d_argmax_val = nullptr;
    int *d_expert_idx = nullptr; float *d_expert_wt = nullptr;
    int *d_sorted_ids = nullptr, *d_expert_counts = nullptr, *d_expert_offsets = nullptr;
    cudaStream_t st = nullptr;
    bool initialized = false;
    std::vector<float> embed, ibias, iscale;

    // Per-layer weight device pointers (owned, freed in cuda_destroy)
    std::vector<half*> layer_wq, layer_wk, layer_wv, layer_wo;
    std::vector<half*> layer_w1, layer_w2, layer_w3;
    std::vector<half*> layer_rms_a, layer_rms_f;

    // We reuse the same algorithmic structure as ZayaState but with CUDA types.
    // For simplicity, the per-layer weight structures and kernel dispatch
    // are implemented in cuda_engine.cu.
};

// ── C ABI functions ──
#ifdef __cplusplus
extern "C" {
#endif

/// Initialize the CUDA engine for a specific model architecture.
CudaState* cuda_init(const char* weights_dir, const CudaConfig* cfg = nullptr);

/// Forward + argmax in one fused call (mimics zaya_forward_greedy).
int  cuda_forward_greedy(CudaState* s, int token_id);

/// Reset KV cache and position counter.
void cuda_reset(CudaState* s);

/// Destroy engine and free all GPU resources.
void cuda_destroy(CudaState* s);

#ifdef __cplusplus
}
#endif
