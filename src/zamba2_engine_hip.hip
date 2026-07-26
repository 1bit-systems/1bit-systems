// zamba2_engine_hip.hip — GPU-accelerated Zamba2 forward pass
#include "zamba2_hybrid_gpu.hip"
//
// Uses the Mamba2 selective scan HIP kernels for GPU inference.
// Replaces the CPU reference implementation with GPU kernel launches.
//
// For each layer:
//   1. Upload input to GPU (or keep on GPU)
//   2. Launch Mamba2 conv1d kernel
//   3. Launch Mamba2 selective scan kernel
//   4. Launch attention/FFN kernels for hybrid layers
//   5. Download output logits (only last token for generation)

#include "zamba2_engine.h"
#include "mamba2_kernels.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#ifndef HIP_CHECK
#define HIP_CHECK(e) do { hipError_t _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP Error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_s)); abort(); } } while(0)
#endif

// Local RMS norm kernel — avoids link dependency on laguna_ops.hip's device stub.
__global__ void zamba2_rms_norm_kernel(float* __restrict__ out, const float* __restrict__ x,
                                        const float* __restrict__ w, int N, float eps) {
    __shared__ float s_ss[256];
    int tid = threadIdx.x;
    float l = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) l += x[i] * x[i];
    s_ss[tid] = l;
    __syncthreads();
    if (tid == 0) {
        double ss = 0.0;
        for (int i = 0; i < blockDim.x; i++) ss += s_ss[i];
        float ir = 1.0f / sqrtf((float)(ss / N) + eps);
        for (int i = 0; i < N; i++) out[i] = x[i] * ir * w[i];
    }
}

// ── HIP state for Zamba2 ──
struct Zamba2HIPState {
    // Device pointers for weights
    float *d_embed = nullptr;          // [vocab, d_model]
    float *d_final_norm = nullptr;     // [d_model]

    // Per-layer device weight pointers
    struct LayerDevWeights {
        float *in_proj_w = nullptr;    // [d_in_proj, d_model]
        float *conv1d_w = nullptr;     // [d_conv, conv_dim]
        float *conv1d_b = nullptr;     // [conv_dim]
        float *dt_bias = nullptr;      // [n_head]
        float *A_log = nullptr;        // [n_head]
        float *D = nullptr;            // [n_head]
        float *norm_w = nullptr;       // [d_inner/n_group, n_group]
        float *out_proj_w = nullptr;   // [d_model, d_inner]
        float *input_norm_w = nullptr; // [d_model]
        // Hybrid only:
        float *linear_w = nullptr;     // [d_model, d_model]
        float *attn_q = nullptr;       // [n_heads * hd, d_model]
        float *attn_k = nullptr;       // [n_kv * hd, d_model]
        float *attn_v = nullptr;       // [n_kv * hd, d_model]
        float *attn_o = nullptr;       // [d_model, n_heads * hd]
        float *ffn_gate = nullptr;     // [d_ff, d_model]
        float *ffn_up = nullptr;       // [d_ff, d_model]
        float *ffn_down = nullptr;     // [d_model, d_ff]
        float *post_attn_norm = nullptr; // [d_model]
        float *ffn_norm = nullptr;     // [d_model]
        bool is_hybrid = false;
    };
    std::vector<LayerDevWeights> layer_weights;

    // GPU state buffers
    float *d_conv_state = nullptr;
    float *d_ssm_state = nullptr;
    float *d_k_cache = nullptr;
    float *d_v_cache = nullptr;

    // Working buffers
    float *d_hidden = nullptr;
    float *d_tmp = nullptr;
    float *d_logits = nullptr;

    // Config snapshot
    Zamba2Config cfg;
    int conv_dim = 0;
    int n_hybrid = 0;

    hipStream_t stream = nullptr;

    ~Zamba2HIPState() { destroy(); }

    void destroy() {
        auto safe_free = [](auto*& p) { if (p) { hipFree(p); p = nullptr; } };
        safe_free(d_embed); safe_free(d_final_norm);
        safe_free(d_conv_state); safe_free(d_ssm_state);
        safe_free(d_k_cache); safe_free(d_v_cache);
        safe_free(d_hidden); safe_free(d_tmp); safe_free(d_logits);
        for (auto& lw : layer_weights) {
            safe_free(lw.in_proj_w); safe_free(lw.conv1d_w);
            safe_free(lw.conv1d_b); safe_free(lw.dt_bias);
            safe_free(lw.A_log); safe_free(lw.D);
            safe_free(lw.norm_w); safe_free(lw.out_proj_w);
            safe_free(lw.input_norm_w); safe_free(lw.linear_w);
            safe_free(lw.attn_q); safe_free(lw.attn_k);
            safe_free(lw.attn_v); safe_free(lw.attn_o);
            safe_free(lw.ffn_gate); safe_free(lw.ffn_up);
            safe_free(lw.ffn_down); safe_free(lw.post_attn_norm);
            safe_free(lw.ffn_norm);
        }
        layer_weights.clear();
        if (stream) { hipStreamDestroy(stream); stream = nullptr; }
    }
};

// ── Upload weights from Zamba2Model to GPU ──
extern "C" Zamba2HIPState* zamba2_hip_init(Zamba2Model& model) {
    Zamba2HIPState* s = new Zamba2HIPState();
    auto& cfg = model.cfg;
    s->cfg = cfg;
    s->conv_dim = cfg.d_inner + 2 * cfg.n_group * cfg.d_state;

    hipStreamCreate(&s->stream);

    auto upload = [&](const std::vector<float>& src, float** dst) {
        if (src.empty()) { *dst = nullptr; return; }
        hipMalloc(dst, src.size() * sizeof(float));
        hipMemcpy(*dst, src.data(), src.size() * sizeof(float), hipMemcpyHostToDevice);
    };

    // Embedding + final norm
    upload(model.embed_w, &s->d_embed);
    upload(model.final_norm_w, &s->d_final_norm);

    // Layer weights
    int n_layers = cfg.n_layers;
    s->layer_weights.resize(n_layers);

    for (int l = 0; l < n_layers; ++l) {
        auto& lw = s->layer_weights[l];
        bool is_hyb = model.hybrid_layers.find(l) != model.hybrid_layers.end();
        lw.is_hybrid = is_hyb;

        if (is_hyb) {
            auto& hl = model.hybrid_layers[l];
            upload(hl.input_norm_w, &lw.input_norm_w);
            upload(hl.mamba.in_proj_w, &lw.in_proj_w);
            upload(hl.mamba.conv1d_w, &lw.conv1d_w);
            upload(hl.mamba.conv1d_b, &lw.conv1d_b);
            upload(hl.mamba.dt_bias, &lw.dt_bias);
            upload(hl.mamba.A_log, &lw.A_log);
            upload(hl.mamba.D, &lw.D);
            upload(hl.mamba.norm_w, &lw.norm_w);
            upload(hl.mamba.out_proj_w, &lw.out_proj_w);
            upload(hl.linear_w, &lw.linear_w);
            upload(hl.shared_transformer_q, &lw.attn_q);
            upload(hl.shared_transformer_k, &lw.attn_k);
            upload(hl.shared_transformer_v, &lw.attn_v);
            upload(hl.shared_transformer_o, &lw.attn_o);
            upload(hl.shared_transformer_gate, &lw.ffn_gate);
            upload(hl.shared_transformer_up, &lw.ffn_up);
            upload(hl.shared_transformer_down, &lw.ffn_down);
            upload(hl.shared_transformer_pre_ff_norm, &lw.post_attn_norm);
            upload(hl.shared_transformer_ffn_norm, &lw.ffn_norm);
        } else {
            auto& ml = model.mamba_layers[l];
            upload(ml.input_norm_w, &lw.input_norm_w);
            upload(ml.in_proj_w, &lw.in_proj_w);
            upload(ml.conv1d_w, &lw.conv1d_w);
            upload(ml.conv1d_b, &lw.conv1d_b);
            upload(ml.dt_bias, &lw.dt_bias);
            upload(ml.A_log, &lw.A_log);
            upload(ml.D, &lw.D);
            upload(ml.norm_w, &lw.norm_w);
            upload(ml.out_proj_w, &lw.out_proj_w);
        }
    }

    // Allocate state buffers
    s->n_hybrid = (int)model.hybrid_layers.size();
    int max_seq = cfg.max_seq_len;
    int n_kv = cfg.n_kv_heads;
    int hd = cfg.attn_head_dim;
    int d_model = cfg.d_model;
    int conv_dim = s->conv_dim;
    int d_inner = cfg.d_inner;
    int d_state = cfg.d_state;

    hipMalloc(&s->d_conv_state, (size_t)n_layers * (cfg.d_conv - 1) * conv_dim * sizeof(float));
    hipMemset(s->d_conv_state, 0, (size_t)n_layers * (cfg.d_conv - 1) * conv_dim * sizeof(float));
    hipMalloc(&s->d_ssm_state, (size_t)n_layers * d_state * d_inner * sizeof(float));
    hipMemset(s->d_ssm_state, 0, (size_t)n_layers * d_state * d_inner * sizeof(float));
    hipMalloc(&s->d_k_cache, (size_t)s->n_hybrid * max_seq * n_kv * hd * sizeof(float));
    hipMemset(s->d_k_cache, 0, (size_t)s->n_hybrid * max_seq * n_kv * hd * sizeof(float));
    hipMalloc(&s->d_v_cache, (size_t)s->n_hybrid * max_seq * n_kv * hd * sizeof(float));
    hipMemset(s->d_v_cache, 0, (size_t)s->n_hybrid * max_seq * n_kv * hd * sizeof(float));

    // Working buffers
    hipMalloc(&s->d_hidden, (size_t)d_model * sizeof(float));
    hipMalloc(&s->d_tmp, (size_t)std::max(d_model, d_inner) * 4 * sizeof(float));  // multiple temps
    hipMalloc(&s->d_logits, (size_t)cfg.vocab_size * sizeof(float));

    fprintf(stderr, "[zamba2-hip] GPU buffers allocated: conv=%zu ssm=%zu kv=%zu\n",
            (size_t)n_layers * (cfg.d_conv-1) * conv_dim * sizeof(float),
            (size_t)n_layers * d_state * d_inner * sizeof(float),
            (size_t)s->n_hybrid * max_seq * n_kv * hd * 2 * sizeof(float));

    return s;
}

// ── HIP: forward one token ──
// Launches GPU kernels for each layer. Designed for autoregressive decode.
// Hybrid layers (attention + FFN) fall back to CPU since GPU kernels are not
// yet implemented for the shared transformer portion.
extern "C" void zamba2_hip_forward(Zamba2HIPState* s, Zamba2Model& model, int token_id,
                         float* logits_out, int pos) {
    auto& cfg = s->cfg;
    int d_model = cfg.d_model;
    int vocab = cfg.vocab_size;
    int n_layers = cfg.n_layers;
    int conv_dim = s->conv_dim;
    int d_inner = cfg.d_inner;
    int d_state = cfg.d_state;
    int n_head = cfg.n_head;
    int n_group = cfg.n_group;
    int head_dim = cfg.head_dim;

    hipStream_t st = s->stream;

    // ── Embedding lookup ──
    hipMemcpyDtoDAsync(s->d_hidden, s->d_embed + (size_t)token_id * d_model,
                       d_model * sizeof(float), st);

    // ── Layer loop ──
    int hyb_idx = 0;
    int max_seq = cfg.max_seq_len;
    int n_attn_heads = cfg.n_attn_heads;
    int n_kv_heads = cfg.n_kv_heads;
    int attn_head_dim = cfg.attn_head_dim;
    for (int l = 0; l < n_layers; ++l) {
        auto& lw = s->layer_weights[l];

        // Input RMS norm
        int threads = 256;
        int blocks = (d_model + threads - 1) / threads;
        hipLaunchKernelGGL(zamba2_rms_norm_kernel, dim3(blocks), dim3(threads), 0, st,
            s->d_hidden, s->d_tmp, lw.input_norm_w, d_model, cfg.rms_norm_eps);
        HIP_CHECK(hipGetLastError());
        hipMemcpyDtoDAsync(s->d_hidden, s->d_tmp, d_model * sizeof(float), st);

        float* conv_state = s->d_conv_state + (size_t)l * (cfg.d_conv - 1) * conv_dim;
        float* ssm_state  = s->d_ssm_state + (size_t)l * d_state * d_inner;

        // ── Mamba2 GPU forward (single-token decode) ──
        mamba2_gpu_decode_block(
            s->d_hidden, lw.in_proj_w, lw.conv1d_w, lw.conv1d_b,
            lw.dt_bias, lw.A_log, lw.D, lw.out_proj_w,
            conv_state, ssm_state,
            s->d_hidden, s->d_tmp,
            d_model, d_inner, d_state, cfg.d_conv,
            n_head, n_group, head_dim, conv_dim, st);

        // ── Hybrid layer: attention + FFN (CPU fallback) ──
        // ── Hybrid layer: GPU attention + FFN (zero CPU round-trips) ──
        if (lw.is_hybrid) {
            auto hit = model.hybrid_layers.find(l);
            if (hit == model.hybrid_layers.end()) continue;
            const HybridLayerWeights& hl = hit->second;

            int d_ff_layer = (int)hl.shared_transformer_up.size() / d_model;
            float* k_cache_gpu = s->d_k_cache + (size_t)hyb_idx * 2 * max_seq * n_kv_heads * attn_head_dim;
            float* v_cache_gpu = s->d_v_cache + (size_t)hyb_idx * 2 * max_seq * n_kv_heads * attn_head_dim;

            // Get the mamba sub-weights from the hybrid layer
            const float* mamba_inp_norm = nullptr;  // not needed separately - hybrid uses own norm

            zamba2_hybrid_forward_gpu(
                lw.in_proj_w, lw.conv1d_w, lw.conv1d_b,
                lw.dt_bias, lw.A_log, lw.D, lw.out_proj_w,
                conv_state, ssm_state,
                lw.input_norm_w,
                lw.linear_w,
                lw.attn_q, lw.attn_k, lw.attn_v, lw.attn_o,
                lw.post_attn_norm, lw.ffn_norm,
                lw.ffn_gate, lw.ffn_up, lw.ffn_down,
                k_cache_gpu, v_cache_gpu,
                s->d_hidden, s->d_tmp + d_model * 2,
                d_model, d_inner, d_state, cfg.d_conv,
                n_head, n_group, head_dim, conv_dim,
                n_attn_heads, n_kv_heads, attn_head_dim,
                d_ff_layer, pos, max_seq, st);

            hyb_idx++;
        }
    }

    // ── Final norm ──
    int threads = 256;
    int blocks = (d_model + threads - 1) / threads;
    hipLaunchKernelGGL(zamba2_rms_norm_kernel, dim3(blocks), dim3(threads), 0, st,
        s->d_hidden, s->d_tmp, s->d_final_norm, d_model, cfg.rms_norm_eps);
    HIP_CHECK(hipGetLastError());
    hipMemcpyDtoDAsync(s->d_hidden, s->d_tmp, d_model * sizeof(float), st);

    // ── LM head on GPU (tiled GEMV on tied embedding) ──
    // logits[V] = embed[V, d_model] @ hidden[d_model]
    int gemv_blocks = (vocab + 2 - 1) / 2;
    hipLaunchKernelGGL(mamba2_tiled_gemv_kernel,
        dim3(gemv_blocks), dim3(256), 0, st,
        s->d_embed, s->d_hidden, s->d_logits, vocab, d_model);

    // Async D2H copy of logits
    hipMemcpyAsync(logits_out, s->d_logits, vocab * sizeof(float),
                   hipMemcpyDeviceToHost, st);
    hipStreamSynchronize(st);
}

extern "C" void zamba2_hip_destroy(Zamba2HIPState* s) {
    if (s) { s->destroy(); }
}
