// cuda_engine.cu — CUDA inference engine implementation
//
// Mirrors the Zaya HIP engine (zaya_engine.cpp + zaya_engine.h) using CUDA.
// Uses the same weight format and directory structure. Falls back to generic
// CPU-style matmul when cuBLAS is unavailable.
//
// CUDA kernel style note: we use the same single-kernel-per-operation pattern
// as the HIP version for consistency, not the fused-mega-kernel approach.

#include "cuda_engine.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <new>
#include <cublas_v2.h>

// ── Error checking macros (mirror hip_check.h) ──
#ifndef CUDA_CHECK
#define CUDA_CHECK(e) do {                                                      \
    cudaError_t _s_ = (e);                                                      \
    if (_s_ != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA Error %s:%d: %s (code %d)\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_s_), (int)_s_);         \
        throw std::runtime_error(std::string("CUDA error at ") + __FILE__ +     \
                                 ":" + std::to_string(__LINE__) + ": " +        \
                                 cudaGetErrorString(_s_));                      \
    }                                                                           \
} while(0)
#endif

#ifndef CUDA_OK_V
#define CUDA_OK_V(e) do {                                                       \
    cudaError_t _s_ = (e);                                                      \
    if (_s_ != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA Error %s:%d: %s (code %d)\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_s_), (int)_s_);         \
        return;                                                                 \
    }                                                                           \
} while(0)
#endif

#ifndef CUDA_OK_R
#define CUDA_OK_R(e, retval) do {                                               \
    cudaError_t _s_ = (e);                                                      \
    if (_s_ != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA Error %s:%d: %s (code %d)\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_s_), (int)_s_);         \
        return (retval);                                                        \
    }                                                                           \
} while(0)
#endif

// ── cuBLAS handle (lazily initialized) ──
static cublasHandle_t g_cublas = nullptr;
static cublasHandle_t get_cublas() {
    if (!g_cublas) {
        cublasCreate(&g_cublas);
    }
    return g_cublas;
}

// ── Runtime config ──
static constexpr float RMD_EPS = 1e-5f;
static constexpr int BLK = 256;
static thread_local CudaConfig eng;

// ── CUDA kernels ──

// RMS normalization kernel
__global__ void rmsnorm_k(half* x, const half* w, int n) {
    __shared__ float r[32];
    int tx = threadIdx.x, wid = tx / 32, l = tx % 32;
    float ss = 0;
    for (int i = tx; i < n; i += blockDim.x)
        ss += (float)x[i] * (float)x[i];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
    if (l == 0) r[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        ss = (l < (256/32)) ? r[l] : 0;
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
        if (l == 0) r[0] = ss;
    }
    __syncthreads();
    float iv = rsqrtf(r[0] / n + RMD_EPS);
    for (int i = tx; i < n; i += blockDim.x)
        x[i] = __float2half((float)x[i] * iv * (float)w[i]);
}

// Element-wise copy
__global__ void copy_k(half* d, const half* s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d[i] = s[i];
}

// Matrix-vector multiply (GEMV): out[M] = in[K] @ wt[K, M]
__global__ void gemv_k(half* out, const half* in, const half* wt, int M, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    float s = 0;
    for (int k = 0; k < K; k++)
        s += (float)in[k] * (float)wt[(size_t)k * M + i];
    out[i] = __float2half(s);
}

// SiLU(x) * y elementwise
__global__ void silu_mul_k(half* out, const half* g, const half* u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = (float)g[i];
    out[i] = __float2half((v / (1.0f + expf(-v))) * (float)u[i]);
}

// Residual + hidden-state scale/blend
__global__ void residual_scale_k(half* out, const half* res,
                                  const float* hs_s, const float* hs_b,
                                  const float* res_s, const float* res_b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half((float)out[i] * hs_s[i] + hs_b[i] +
                          (float)res[i] * res_s[i] + res_b[i]);
}

// Embedding lookup kernel
__global__ void embed_lookup_k(half* out, const half* embed, const half* ibias,
                                const half* iscale, const int* d_token_id, int h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h) return;
    int token_id = *d_token_id;
    float raw = (float)embed[(size_t)token_id * h + i];
    out[i] = __float2half((raw + (float)ibias[i]) * (float)iscale[i]);
}

// Argmax kernel
__global__ void argmax_k(const float* logits, int* idx, float* val, int n) {
    __shared__ int s_idx[32];
    __shared__ float s_val[32];
    int tx = threadIdx.x, wid = tx / 32, l = tx % 32;
    float best = -1e38f;
    int best_i = 0;
    for (int i = tx; i < n; i += blockDim.x) {
        if (logits[i] > best) { best = logits[i]; best_i = i; }
    }
    // warp reduce
    for (int o = 16; o > 0; o >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, best, o);
        int other_i = __shfl_xor_sync(0xffffffff, best_i, o);
        if (other > best) { best = other; best_i = other_i; }
    }
    if (l == 0) { s_val[wid] = best; s_idx[wid] = best_i; }
    __syncthreads();
    if (wid == 0) {
        if (l < (256 / 32)) { best = s_val[l]; best_i = s_idx[l]; }
        else { best = -1e38f; best_i = 0; }
        for (int o = 16; o > 0; o >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, best, o);
            int other_i = __shfl_xor_sync(0xffffffff, best_i, o);
            if (other > best) { best = other; best_i = other_i; }
        }
        if (l == 0) { *idx = best_i; *val = best; }
    }
}

// Simple flash-attention-style KV attention kernel (single head, small context)
__global__ void attention_k(half* out, const half* Q, const half* K, const half* V,
                            int n_kv_heads, int head_dim, int seq_len, float scale) {
    int h = blockIdx.x;     // query head
    int tx = threadIdx.x;
    if (h >= n_kv_heads) return;

    const half* q_head = Q + (size_t)h * head_dim;
    const half* k_head = K + (size_t)h * head_dim;
    const half* v_head = V + (size_t)h * head_dim;
    half* o_head = out + (size_t)h * head_dim;

    // Online softmax: compute attention scores and weighted sum in one pass
    float score_max = -1e38f;
    float exp_sum = 0.0f;
    float acc = 0.0f;

    for (int t = 0; t < seq_len; t++) {
        // dot product Q @ K_t
        float s = 0;
        for (int i = tx; i < head_dim; i += blockDim.x) {
            s += (float)q_head[i] * (float)k_head[(size_t)t * n_kv_heads * head_dim + i];
        }
        // warp reduce sum
        for (int o = 16; o > 0; o >>= 1)
            s += __shfl_xor_sync(0xffffffff, s, o);
        s *= scale;

        // online softmax
        float new_max = max(score_max, s);
        float old_exp_sum = exp_sum;
        float e = expf(s - new_max);
        float e_old_scale = expf(score_max - new_max);
        exp_sum = old_exp_sum * e_old_scale + e;
        score_max = new_max;

        // accumulate weighted value
        float v = (float)v_head[(size_t)t * n_kv_heads * head_dim + tx];
        acc = acc * e_old_scale + v * e;
    }

    __syncthreads();
    // final divide by exp_sum for the responsible thread
    if (tx == 0) {
        o_head[tx] = __float2half(acc / exp_sum);
    }
    for (int i = tx + 1; i < head_dim; i += blockDim.x) {
        // This simplified kernel handles head_dim <= 128 and seq_len <= 1024
        // For larger sequences, use the flash-decoding approach
        o_head[i] = __float2half((float)o_head[i] / exp_sum);
    }
}

// ── Weight loading helpers ──
static std::vector<float> load_bin(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "Missing: %s\n", p.c_str()); return {}; }
    size_t n = f.tellg() / sizeof(float); f.seekg(0);
    std::vector<float> d(n); f.read((char*)d.data(), n * sizeof(float)); return d;
}

static const std::string g_cuda_weights_dir = []() -> std::string {
    const char* env = getenv("CUDA_WEIGHTS_DIR");
    if (env && env[0]) return env;
    const char* xdg = getenv("XDG_DATA_HOME");
    if (xdg && xdg[0]) return std::string(xdg) + "/1bit-systems/weights/";
    const char* home = getenv("HOME");
    if (home && home[0]) return std::string(home) + "/.local/share/1bit-systems/weights/";
    return "/tmp/cuda_weights/";
}();

#define W(N) load_bin(g_cuda_weights_dir + N)

static void upf16(const std::vector<float>& s, half* d, int n, cudaStream_t h = 0) {
    std::vector<half> b(n); for (int i = 0; i < n; i++) b[i] = __float2half(s[i]);
    CUDA_OK_V(cudaMemcpyAsync(d, b.data(), n * 2, cudaMemcpyHostToDevice, h));
}

// ── Launch config helper ──
static void launch_gemv(half* out, const half* in, const half* wt, int M, int K, cudaStream_t st) {
    // Use cuBLAS GEMV when available for better performance
    cublasHandle_t cublas = get_cublas();
    if (cublas) {
        // Weight `wt` is stored row-major [K][M]: wt[k*M + m] = weight for output m from input k.
        // In cuBLAS column-major, this is an M×K matrix A where A[m][k] = wt[k*M + m] = wt[m + k*M].
        // So we need CUBLAS_OP_N: out[M×1] = A[M×K] @ in[K×1].
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                     M, 1, K, &alpha,
                     wt, CUDA_R_16F, M,    // A is M×K col-major, lda=M
                     in, CUDA_R_16F, K,    // x is K×1
                     &beta,
                     out, CUDA_R_16F, M,   // y is M×1
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);
    } else {
        // Fallback: naive kernel
        int block = 256;
        int grid = (M + block - 1) / block;
        gemv_k<<<grid, block, 0, st>>>(out, in, wt, M, K);
    }
}

// ── Engine implementation ──

extern "C" {

CudaState* cuda_init(const char* weights_dir, const CudaConfig* cfg) {
    if (cfg) {
        eng = *cfg;
    } else {
        eng = CudaConfig::cuda_default();
    }

    CudaState* s = new (std::nothrow) CudaState();
    if (!s) {
        fprintf(stderr, "cuda_init: failed to allocate CudaState (OOM)\n");
        return nullptr;
    }
    CUDA_OK_R(cudaStreamCreate(&s->st), nullptr);

    s->embed = W("model_embed_tokens_weight.bin");
    auto fnorm = W("model_norm_weight.bin");
    s->iscale = W("model_input_hidden_states_scale.bin");
    s->ibias = W("model_input_hidden_states_bias.bin");

    if (s->embed.empty() || fnorm.empty() || s->iscale.empty() || s->ibias.empty()) {
        fprintf(stderr, "cuda_init: failed to load initial weight files — aborting\n");
        cuda_destroy(s);
        return nullptr;
    }

    int ndev = 0;
    CUDA_OK_R(cudaGetDeviceCount(&ndev), nullptr);
    if (ndev < 1) {
        fprintf(stderr, "cuda_init: No CUDA-capable GPU found (device count=%d).\n", ndev);
        cuda_destroy(s);
        return nullptr;
    }

    // Dimension validation
    size_t expected_embed = (size_t)eng.vocab * eng.h;
    if (s->embed.size() != expected_embed) {
        fprintf(stderr, "cuda_init: embed size %zu != expected %zu (H=%d, vocab=%d)\n",
                s->embed.size(), expected_embed, eng.h, eng.vocab);
        cuda_destroy(s);
        return nullptr;
    }

    // Allocate GPU buffers
    auto alloc_f16 = [&](auto& p, int n) -> bool {
        cudaError_t e = cudaMalloc(&p, (size_t)n * 2);
        if (e != cudaSuccess) { fprintf(stderr, "CUDA OOM: %s\n", cudaGetErrorString(e)); return false; }
        return true;
    };
    auto alloc_f32 = [&](auto& p, int n) -> bool {
        cudaError_t e = cudaMalloc(&p, (size_t)n * 4);
        if (e != cudaSuccess) { fprintf(stderr, "CUDA OOM: %s\n", cudaGetErrorString(e)); return false; }
        return true;
    };
    #define ALLOC_OR_FAIL(s_, alloc_fn, ptr, n) do { if (!alloc_fn(ptr, n)) { cuda_destroy(s_); return nullptr; } } while(0)

    ALLOC_OR_FAIL(s, alloc_f16, s->d_hs, eng.h);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_ao, eng.h);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_tmp, std::max(eng.h, 2 * eng.n_ff));
    ALLOC_OR_FAIL(s, alloc_f16, s->d_fnw, eng.h);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_lm_out, 4096);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_lm_vocab, eng.vocab);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_argmax_idx, 1);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_argmax_val, 1);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_sorted_ids, 8);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_expert_counts, 17);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_expert_offsets, 17);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_embed, eng.vocab * eng.h);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_ibias, eng.h);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_iscale, eng.h);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_token_id, 1);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_conv, eng.n_layers * 2 * eng.qkv);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_phs, eng.n_layers * eng.h);

    // KV cache
    s->max_seq = eng.max_seq_len > 0 ? eng.max_seq_len : 4096;
    int kv_elems = eng.n_layers * s->max_seq * eng.nkv * eng.hd;
    ALLOC_OR_FAIL(s, alloc_f16, s->d_kcache, kv_elems);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_vcache, kv_elems);
    fprintf(stderr, "  KV cache: linear contiguous %d tok x %d layers = %.1f MB\n",
            s->max_seq, eng.n_layers, (double)kv_elems * 2 / (1024 * 1024));

    ALLOC_OR_FAIL(s, alloc_f16, s->d_vrec, eng.n_layers * (eng.kd / 2));
    ALLOC_OR_FAIL(s, alloc_f16, s->d_qout, eng.qd);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_kout, eng.kd);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_vout, eng.kd);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_skip_flag, 1);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_prev_rs, eng.n_layers * eng.rtr_h);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_expert_idx, 1);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_expert_wt, 1);
    #undef ALLOC_OR_FAIL

    // Upload embeddings
    upf16(s->embed, s->d_embed, eng.vocab * eng.h, s->st);
    std::vector<half> h_ibias(eng.h), h_iscale(eng.h);
    for (int i = 0; i < eng.h; i++) {
        h_ibias[i] = __float2half(s->ibias[i]);
        h_iscale[i] = __float2half(s->iscale[i]);
    }
    CUDA_OK_R(cudaMemcpy(s->d_ibias, h_ibias.data(), eng.h * 2, cudaMemcpyHostToDevice), nullptr);
    CUDA_OK_R(cudaMemcpy(s->d_iscale, h_iscale.data(), eng.h * 2, cudaMemcpyHostToDevice), nullptr);

    // Upload norm weight
    upf16(fnorm, s->d_fnw, eng.h, s->st);

    // Upload per-layer weights and store device pointers in state vectors
    for (int il = 0; il < eng.n_layers; il++) {
        std::string p = g_cuda_weights_dir + "model_layers_" + std::to_string(il) + "_";

        auto wq = load_bin(p + "self_attn_q_proj.weight.bin");
        auto wk = load_bin(p + "self_attn_k_proj.weight.bin");
        auto wv = load_bin(p + "self_attn_v_proj.weight.bin");
        auto wo = load_bin(p + "self_attn_o_proj.weight.bin");
        auto w1 = load_bin(p + "mlp_gate_proj.weight.bin");
        auto w2 = load_bin(p + "mlp_down_proj.weight.bin");
        auto w3 = load_bin(p + "mlp_up_proj.weight.bin");
        auto rms_a = load_bin(p + "input_layernorm.weight.bin");
        auto rms_f = load_bin(p + "post_attention_layernorm.weight.bin");

        auto upload = [&](const std::vector<float>& src, half*& dst) {
            if (src.empty()) { dst = nullptr; return; }
            cudaMalloc(&dst, src.size() * sizeof(half));
            upf16(src, dst, src.size(), s->st);
        };

        half *d_wq = nullptr, *d_wk = nullptr, *d_wv = nullptr, *d_wo = nullptr;
        half *d_w1 = nullptr, *d_w2 = nullptr, *d_w3 = nullptr;
        half *d_rms_a = nullptr, *d_rms_f = nullptr;

        upload(wq, d_wq); upload(wk, d_wk); upload(wv, d_wv); upload(wo, d_wo);
        upload(w1, d_w1); upload(w2, d_w2); upload(w3, d_w3);
        upload(rms_a, d_rms_a); upload(rms_f, d_rms_f);

        s->layer_wq.push_back(d_wq);
        s->layer_wk.push_back(d_wk);
        s->layer_wv.push_back(d_wv);
        s->layer_wo.push_back(d_wo);
        s->layer_w1.push_back(d_w1);
        s->layer_w2.push_back(d_w2);
        s->layer_w3.push_back(d_w3);
        s->layer_rms_a.push_back(d_rms_a);
        s->layer_rms_f.push_back(d_rms_f);

        fprintf(stderr, "  Layer %d weights loaded (Q:%s K:%s V:%s O:%s GATE:%s DOWN:%s UP:%s)\n",
                il,
                d_wq ? "OK" : "--", d_wk ? "OK" : "--", d_wv ? "OK" : "--",
                d_wo ? "OK" : "--", d_w1 ? "OK" : "--", d_w2 ? "OK" : "--", d_w3 ? "OK" : "--");
    }

    cudaStreamSynchronize(s->st);
    s->pos = 0;
    s->initialized = true;
    fprintf(stderr, "CUDA engine initialized: H=%d L=%d NQ=%d NKV=%d V=%d\n",
            eng.h, eng.n_layers, eng.nq, eng.nkv, eng.vocab);
    return s;
}

void cuda_reset(CudaState* s) {
    if (!s) return;
    s->pos = 0;
    // Zero out KV cache (optional, depends on desired semantics)
    // For performance, we just reset position counter
}

// ── Per-layer inline helper: get weight pointer, check not null ──
#define LW(vec, il, label) \
    auto* w_##label = s->vec.size() > (size_t)il ? s->vec[il] : nullptr; \
    if (!w_##label) { fprintf(stderr, "CUDA: layer %d missing " #label " weight\n", il); return; }
#define LW_R(label) auto* w_##label = s->vec.size() > (size_t)il ? s->vec[il] : nullptr; \
    if (!w_##label) { fprintf(stderr, "CUDA: layer %d missing " #label " weight\n", il); return -1; }

void cuda_forward(CudaState* s, int token_id, float* logits_out) {
    if (!s || !s->initialized) return;
    
    // 1. Embedding lookup
    CUDA_OK_V(cudaMemcpyAsync(s->d_token_id, &token_id, sizeof(int), cudaMemcpyHostToDevice, s->st));
    embed_lookup_k<<<1, 256, 0, s->st>>>(s->d_hs, s->d_embed, s->d_ibias, s->d_iscale, s->d_token_id, eng.h);
    
    int pos = s->pos;
    
    // 2. Process each layer with correct per-layer weights
    for (int il = 0; il < eng.n_layers; il++) {
        half *d_hs = s->d_hs;
        half *d_ao = s->d_ao;
        half *d_tmp = s->d_tmp;
        
        // Load per-layer weight pointers from state vectors
        LW(layer_rms_a, il, rms_a);
        LW(layer_wq, il, wq);
        LW(layer_wk, il, wk);
        LW(layer_wv, il, wv);
        LW(layer_wo, il, wo);
        
        // --- Self-attention ---
        // RMS norm on hidden state
        rmsnorm_k<<<1, 256, 0, s->st>>>(d_tmp, w_rms_a, eng.h);
        
        // Q projection: out[qd] = hs[h] @ wq[h, qd]
        launch_gemv(s->d_qout, d_tmp, w_wq, eng.qd, eng.h, s->st);
        // K projection: out[kd] = hs[h] @ wk[h, kd]
        launch_gemv(s->d_kout, d_tmp, w_wk, eng.kd, eng.h, s->st);
        // V projection: out[kd] = hs[h] @ wv[h, kd]
        launch_gemv(s->d_vout, d_tmp, w_wv, eng.kd, eng.h, s->st);
        
        // Store KV in cache
        size_t kv_offset = (size_t)il * s->max_seq * eng.nkv * eng.hd;
        half* k_dst = s->d_kcache + kv_offset + (size_t)pos * eng.nkv * eng.hd;
        half* v_dst = s->d_vcache + kv_offset + (size_t)pos * eng.nkv * eng.hd;
        CUDA_OK_V(cudaMemcpyAsync(k_dst, s->d_kout, eng.kd * 2, cudaMemcpyDeviceToDevice, s->st));
        CUDA_OK_V(cudaMemcpyAsync(v_dst, s->d_vout, eng.kd * 2, cudaMemcpyDeviceToDevice, s->st));
        
        // Attention
        int seq_len = pos + 1;
        float scale = 1.0f / sqrtf((float)eng.hd);
        attention_k<<<eng.nq, 256, 0, s->st>>>(
            d_ao, s->d_qout, s->d_kcache + kv_offset, s->d_vcache + kv_offset,
            eng.nkv, eng.hd, seq_len, scale);
        
        // Output projection: hs[h] = ao[qd] @ wo[qd, h]
        launch_gemv(d_hs, d_ao, w_wo, eng.h, eng.qd, s->st);
        
        // --- FFN ---
        LW(layer_rms_f, il, rms_f);
        LW(layer_w1, il, w1);
        LW(layer_w2, il, w2);
        LW(layer_w3, il, w3);
        
        // RMS norm
        rmsnorm_k<<<1, 256, 0, s->st>>>(d_tmp, w_rms_f, eng.h);
        
        // Gate projection: gate[n_ff] = tmp[h] @ w1[h, n_ff]
        launch_gemv(d_tmp, d_tmp, w_w1, eng.n_ff, eng.h, s->st);
        // Up projection: up[n_ff] = hs[h] @ w3[h, n_ff]
        launch_gemv(s->d_ao, d_tmp, w_w3, eng.n_ff, eng.h, s->st);
        // SiLU(gate) * up
        silu_mul_k<<<(eng.n_ff + 255) / 256, 256, 0, s->st>>>(d_tmp, d_tmp, s->d_ao, eng.n_ff);
        // Down projection: hs[h] = result[n_ff] @ w2[n_ff, h]
        launch_gemv(d_hs, d_tmp, w_w2, eng.h, eng.n_ff, s->st);
    }
    
    // 3. Final RMS norm
    rmsnorm_k<<<1, 256, 0, s->st>>>(s->d_hs, s->d_fnw, eng.h);
    
    // 4. LM head: logits[vocab] = hs[h] @ embed[V, h]
    launch_gemv(s->d_lm_vocab, s->d_hs, s->d_embed, eng.vocab, eng.h, s->st);
    
    // 5. Copy logits to host if requested
    if (logits_out) {
        CUDA_OK_V(cudaMemcpyAsync(logits_out, s->d_lm_vocab, eng.vocab * 4, cudaMemcpyDeviceToHost, s->st));
    }
    
    s->pos = pos + 1;
}

int cuda_forward_greedy(CudaState* s, int token_id) {
    if (!s || !s->initialized) return -1;
    
    // 1. Embedding lookup
    CUDA_OK_R(cudaMemcpyAsync(s->d_token_id, &token_id, sizeof(int), cudaMemcpyHostToDevice, s->st), -1);
    embed_lookup_k<<<1, 256, 0, s->st>>>(s->d_hs, s->d_embed, s->d_ibias, s->d_iscale, s->d_token_id, eng.h);
    
    int pos = s->pos;
    
    // 2. Process each layer with correct per-layer weights
    for (int il = 0; il < eng.n_layers; il++) {
        half *d_hs = s->d_hs;
        half *d_ao = s->d_ao;
        half *d_tmp = s->d_tmp;
        
        // Load per-layer weight pointers from state vectors
        LW_R(layer_rms_a, rms_a);
        LW_R(layer_wq, wq);
        LW_R(layer_wk, wk);
        LW_R(layer_wv, wv);
        LW_R(layer_wo, wo);
        
        // --- Self-attention ---
        // RMS norm
        rmsnorm_k<<<1, 256, 0, s->st>>>(d_tmp, w_rms_a, eng.h);
        
        // Q projection
        launch_gemv(s->d_qout, d_tmp, w_wq, eng.qd, eng.h, s->st);
        // K projection
        launch_gemv(s->d_kout, d_tmp, w_wk, eng.kd, eng.h, s->st);
        // V projection
        launch_gemv(s->d_vout, d_tmp, w_wv, eng.kd, eng.h, s->st);
        
        // Store KV in cache
        size_t kv_offset = (size_t)il * (size_t)s->max_seq * eng.nkv * eng.hd;
        half* k_dst = s->d_kcache + kv_offset + (size_t)pos * eng.nkv * eng.hd;
        half* v_dst = s->d_vcache + kv_offset + (size_t)pos * eng.nkv * eng.hd;
        cudaMemcpyAsync(k_dst, s->d_kout, eng.kd * 2, cudaMemcpyDeviceToDevice, s->st);
        cudaMemcpyAsync(v_dst, s->d_vout, eng.kd * 2, cudaMemcpyDeviceToDevice, s->st);
        
        // Attention
        int seq_len = pos + 1;
        float scale = 1.0f / sqrtf((float)eng.hd);
        attention_k<<<eng.nq, 256, 0, s->st>>>(
            d_ao, s->d_qout, s->d_kcache + kv_offset, s->d_vcache + kv_offset,
            eng.nkv, eng.hd, seq_len, scale);
        
        // Output projection
        launch_gemv(d_hs, d_ao, w_wo, eng.h, eng.qd, s->st);
        
        // --- FFN ---
        LW_R(layer_rms_f, rms_f);
        LW_R(layer_w1, w1);
        LW_R(layer_w2, w2);
        LW_R(layer_w3, w3);
        
        // RMS norm
        rmsnorm_k<<<1, 256, 0, s->st>>>(d_tmp, w_rms_f, eng.h);
        
        // Gate
        launch_gemv(d_tmp, d_tmp, w_w1, eng.n_ff, eng.h, s->st);
        // Up
        launch_gemv(s->d_ao, d_tmp, w_w3, eng.n_ff, eng.h, s->st);
        // SiLU(gate) * up
        silu_mul_k<<<(eng.n_ff + 255) / 256, 256, 0, s->st>>>(d_tmp, d_tmp, s->d_ao, eng.n_ff);
        // Down
        launch_gemv(d_hs, d_tmp, w_w2, eng.h, eng.n_ff, s->st);
    }
    
    // 3. Final norm
    rmsnorm_k<<<1, 256, 0, s->st>>>(s->d_hs, s->d_fnw, eng.h);
    
    // 4. LM head: logits[vocab] = hs[h] @ embed[V, h]
    launch_gemv(s->d_lm_vocab, s->d_hs, s->d_embed, eng.vocab, eng.h, s->st);
    
    // 5. Argmax on GPU
    argmax_k<<<1, 256, 0, s->st>>>(s->d_lm_vocab, s->d_argmax_idx, s->d_argmax_val, eng.vocab);
    
    int next_token;
    CUDA_OK_R(cudaMemcpy(&next_token, s->d_argmax_idx, sizeof(int), cudaMemcpyDeviceToHost), -1);
    
    s->pos = pos + 1;
    return next_token;
}

void cuda_destroy(CudaState* s) {
    if (!s) return;
    #define CUDA_FREE(p) do { if (p) { cudaFree(p); p = nullptr; } } while(0)
    CUDA_FREE(s->d_hs);
    CUDA_FREE(s->d_ao);
    CUDA_FREE(s->d_tmp);
    CUDA_FREE(s->d_fnw);
    CUDA_FREE(s->d_lm_out);
    CUDA_FREE(s->d_embed);
    CUDA_FREE(s->d_conv);
    CUDA_FREE(s->d_phs);
    CUDA_FREE(s->d_lm_vocab);
    CUDA_FREE(s->d_ibias);
    CUDA_FREE(s->d_iscale);
    CUDA_FREE(s->d_token_id);
    CUDA_FREE(s->d_kcache);
    CUDA_FREE(s->d_vcache);
    CUDA_FREE(s->d_vrec);
    CUDA_FREE(s->d_qout);
    CUDA_FREE(s->d_kout);
    CUDA_FREE(s->d_vout);
    CUDA_FREE(s->d_skip_flag);
    CUDA_FREE(s->d_prev_rs);
    CUDA_FREE(s->d_k_gather);
    CUDA_FREE(s->d_v_gather);
    CUDA_FREE(s->d_page_map);
    CUDA_FREE(s->d_gather_seq_len);
    CUDA_FREE(s->d_argmax_idx);
    CUDA_FREE(s->d_argmax_val);
    CUDA_FREE(s->d_expert_idx);
    CUDA_FREE(s->d_expert_wt);
    CUDA_FREE(s->d_sorted_ids);
    CUDA_FREE(s->d_expert_counts);
    CUDA_FREE(s->d_expert_offsets);
    // Free per-layer weight pointers
    auto free_vec = [](auto& vec) {
        for (auto* p : vec) { if (p) cudaFree(p); }
        vec.clear();
    };
    free_vec(s->layer_wq); free_vec(s->layer_wk); free_vec(s->layer_wv); free_vec(s->layer_wo);
    free_vec(s->layer_w1); free_vec(s->layer_w2); free_vec(s->layer_w3);
    free_vec(s->layer_rms_a); free_vec(s->layer_rms_f);
    #undef CUDA_FREE
    if (s->graph_exec) { cudaGraphExecDestroy(s->graph_exec); s->graph_exec = nullptr; }
    if (s->graph) { cudaGraphDestroy(s->graph); s->graph = nullptr; }
    if (s->st) { cudaStreamDestroy(s->st); s->st = nullptr; }
    delete s;
    #undef LW
    #undef LW_R
}

} // extern "C"
