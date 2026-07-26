// zamba2_engine_hip.hip — GPU-accelerated Zamba2 forward pass
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
static Zamba2HIPState* zamba2_hip_init(Zamba2Model& model) {
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

    hipMalloc(&s->d_conv_state, n_layers * (cfg.d_conv - 1) * conv_dim * sizeof(float));
    hipMemset(s->d_conv_state, 0, n_layers * (cfg.d_conv - 1) * conv_dim * sizeof(float));
    hipMalloc(&s->d_ssm_state, n_layers * d_state * d_inner * sizeof(float));
    hipMemset(s->d_ssm_state, 0, n_layers * d_state * d_inner * sizeof(float));
    hipMalloc(&s->d_k_cache, s->n_hybrid * max_seq * n_kv * hd * sizeof(float));
    hipMemset(s->d_k_cache, 0, s->n_hybrid * max_seq * n_kv * hd * sizeof(float));
    hipMalloc(&s->d_v_cache, s->n_hybrid * max_seq * n_kv * hd * sizeof(float));
    hipMemset(s->d_v_cache, 0, s->n_hybrid * max_seq * n_kv * hd * sizeof(float));

    // Working buffers
    hipMalloc(&s->d_hidden, d_model * sizeof(float));
    hipMalloc(&s->d_tmp, std::max(d_model, d_inner) * 4 * sizeof(float));  // multiple temps
    hipMalloc(&s->d_logits, cfg.vocab_size * sizeof(float));

    fprintf(stderr, "[zamba2-hip] GPU buffers allocated: conv=%zu ssm=%zu kv=%zu\n",
            n_layers * (cfg.d_conv-1) * conv_dim * sizeof(float),
            n_layers * d_state * d_inner * sizeof(float),
            s->n_hybrid * max_seq * n_kv * hd * 2 * sizeof(float));

    return s;
}

// ── HIP: forward one token ──
// Launches GPU kernels for each layer. Designed for autoregressive decode.
// NOTE: Requires Zamba2Model host reference for hybrid layer CPU fallback
// and LM head computation (embed_w). The model ref is passed separately
// from the HIP state because the state holds only GPU device pointers.
void zamba2_hip_forward(Zamba2HIPState* s, Zamba2Model& model, int token_id,
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
    // Copy token_id-th row of embedding to d_hidden
    hipMemcpyDtoDAsync(s->d_hidden, s->d_embed + (size_t)token_id * d_model,
                       d_model * sizeof(float), st);

    // ── Layer loop ──
    int hyb_counter = 0;
    for (int l = 0; l < n_layers; ++l) {
        auto& lw = s->layer_weights[l];

        // Input RMS norm: d_hidden = rms_norm(d_hidden) * input_norm_w
        // Using a simple HIP kernel for RMS norm
        int threads = 256;
        int blocks = (d_model + threads - 1) / threads;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(rms_norm_kernel), dim3(blocks), dim3(threads), 0, st,
            s->d_hidden, s->d_tmp, lw.input_norm_w, d_model, cfg.rms_norm_eps);
        HIP_CHECK(hipGetLastError());
        // Copy back: d_hidden = d_tmp (normalized)
        hipMemcpyDtoDAsync(s->d_hidden, s->d_tmp, d_model * sizeof(float), st);

        // ── Mamba2 block ──
        // in_proj: [d_in_proj] = W @ hidden
        int d_in_proj = d_inner + conv_dim + n_head;
        // Simple matmul via HIP: we use gemm for now, or unrolled for small batches
        // For single-token decode, we use a fused kernel approach
        // Launch mamba2 fused layer kernel (defined in mamba2_kernels.hip)

        float* conv_state = s->d_conv_state + (size_t)l * (cfg.d_conv - 1) * conv_dim;
        float* ssm_state  = s->d_ssm_state + (size_t)l * d_state * d_inner;

        // ── Mamba2 GPU forward (single-token decode) ──
        mamba2_gpu_decode_block(
            s->d_hidden,                   // x_in
            lw.in_proj_w,                   // in_proj_w
            lw.conv1d_w,                    // conv1d_w
            lw.conv1d_b,                    // conv1d_b
            lw.dt_bias,                     // dt_bias
            lw.A_log,                       // A_log
            lw.D,                           // D
            lw.out_proj_w,                  // out_proj_w
            conv_state,                     // conv_state (updated)
            ssm_state,                      // ssm_state (updated)
            s->d_hidden,                    // y_out (write back to hidden)
            s->d_tmp,                       // workspace (must be large enough)
            d_model, d_inner, d_state, cfg.d_conv,
            n_head, n_group, head_dim, conv_dim,
            st);

        // ── Hybrid layer: attention + FFN (after Mamba2 block) ──
        // NOTE: Full GPU implementation not yet built (file not in CMakeLists.txt).
        // This path requires host-side weight access from HybridLayerWeights for
        // the shared transformer (attention, RoPE, FFN). The Zamba2Model& ref is
        // now passed to zamba2_hip_forward.
        // See issue #954 for tracking the full GPU kernel implementation.
        if (lw.is_hybrid) {
            if (!hybrid_warned) {
                fprintf(stderr, "[zamba2-hip] WARNING: hybrid layer running on CPU "
                                "(GPU kernel not yet implemented, issue #954)\n");
                hybrid_warned = true;
            }

            // Look up this layer's host-side weights from the model
            auto hit = model.hybrid_layers.find(l);
            if (hit == model.hybrid_layers.end()) continue;
            const HybridLayerWeights& hl = hit->second;

            // Sync Mamba2 GPU results to host for CPU hybrid computation
            hipStreamSynchronize(st);
            std::vector<float> hidden_cpu(d_model);
            hipMemcpy(hidden_cpu.data(), s->d_hidden, d_model * sizeof(float),
                      hipMemcpyDeviceToHost);

            int max_seq = cfg.max_seq_len;
            int n_heads = cfg.n_attn_heads;
            int n_kv = cfg.n_kv_heads;
            int hd = cfg.attn_head_dim;

            // Read GPU KV cache to host
            float* k_cache_gpu = s->d_k_cache + hyb_idx * 2 * max_seq * n_kv * hd;
            float* v_cache_gpu = s->d_v_cache + hyb_idx * 2 * max_seq * n_kv * hd;
            std::vector<float> k_cpu(max_seq * n_kv * hd);
            std::vector<float> v_cpu(max_seq * n_kv * hd);
            hipMemcpy(k_cpu.data(), k_cache_gpu, max_seq * n_kv * hd * sizeof(float),
                      hipMemcpyDeviceToHost);
            hipMemcpy(v_cpu.data(), v_cache_gpu, max_seq * n_kv * hd * sizeof(float),
                      hipMemcpyDeviceToHost);

            // CPU computation using host-side weights from model.hybrid_layers
            int d_m = d_model;
            // Linear projection: projected = linear_w @ mamba_out
            std::vector<float> projected(d_m, 0.0f);
            for (int i = 0; i < d_m; ++i)
                for (int j = 0; j < d_m; ++j)
                    projected[i] += hl.linear_w[i * d_m + j] * hidden_cpu[j];
            // Mamba decoder residual: projected += hidden_cpu
            for (int i = 0; i < d_m; ++i) projected[i] += hidden_cpu[i];

            // QKV projections
            std::vector<float> q(n_heads * hd, 0.0f), k(n_kv * hd, 0.0f), v(n_kv * hd, 0.0f);
            for (int h = 0; h < n_heads; ++h)
                for (int d = 0; d < hd; ++d)
                    for (int j = 0; j < d_m; ++j)
                        q[h * hd + d] += hl.shared_transformer_q[(h * hd + d) * d_m + j] * projected[j];
            for (int h = 0; h < n_kv; ++h)
                for (int d = 0; d < hd; ++d)
                    for (int j = 0; j < d_m; ++j) {
                        k[h * hd + d] += hl.shared_transformer_k[(h * hd + d) * d_m + j] * projected[j];
                        v[h * hd + d] += hl.shared_transformer_v[(h * hd + d) * d_m + j] * projected[j];
                    }

            apply_rope(q.data(), k.data(), pos, hd, n_heads, n_kv, cfg.rope_theta);

            // Store to KV cache
            for (int h = 0; h < n_kv; ++h)
                for (int d = 0; d < hd; ++d) {
                    k_cpu[pos * n_kv * hd + h * hd + d] = k[h * hd + d];
                    v_cpu[pos * n_kv * hd + h * hd + d] = v[h * hd + d];
                }

            // Attention
            std::vector<float> attn_out(d_m);
            attention_forward(q.data(), k_cpu.data(), v_cpu.data(), attn_out.data(),
                             pos, max_seq, n_heads, n_kv, hd, d_m);

            // O projection + residual
            std::vector<float> attn_proj(d_m, 0.0f);
            for (int i = 0; i < d_m; ++i)
                for (int j = 0; j < d_m; ++j)
                    attn_proj[i] += hl.shared_transformer_o[i * d_m + j] * attn_out[j];
            for (int i = 0; i < d_m; ++i) projected[i] += attn_proj[i];

            // FFN: norm -> gate/up -> SiLU -> down -> residual
            std::vector<float> ff_normed(d_m);
            rms_norm(projected.data(), ff_normed.data(),
                     hl.shared_transformer_ffn_norm.data(), d_m, cfg.rms_norm_eps);

            int d_ff = (int)hl.shared_transformer_up.size() / d_m;
            std::vector<float> gate(d_ff, 0.0f), up(d_ff, 0.0f), act(d_ff);
            for (int i = 0; i < d_ff; ++i)
                for (int j = 0; j < d_m; ++j)
                    gate[i] += hl.shared_transformer_gate[i * d_m + j] * ff_normed[j];
            for (int i = 0; i < d_ff; ++i)
                for (int j = 0; j < d_m; ++j)
                    up[i] += hl.shared_transformer_up[i * d_m + j] * ff_normed[j];
            for (int i = 0; i < d_ff; ++i)
                act[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];

            std::vector<float> ff_out(d_m, 0.0f);
            for (int i = 0; i < d_m; ++i)
                for (int j = 0; j < d_ff; ++j)
                    ff_out[i] += hl.shared_transformer_down[i * d_ff + j] * act[j];

            // Final residual: hidden = projected + ff_out
            for (int i = 0; i < d_m; ++i) hidden_cpu[i] = projected[i] + ff_out[i];

            // Copy results back to GPU
            hipMemcpy(s->d_hidden, hidden_cpu.data(), d_m * sizeof(float),
                      hipMemcpyHostToDevice);
            hipMemcpy(k_cache_gpu, k_cpu.data(), max_seq * n_kv * hd * sizeof(float),
                      hipMemcpyHostToDevice);
            hipMemcpy(v_cache_gpu, v_cpu.data(), max_seq * n_kv * hd * sizeof(float),
                      hipMemcpyHostToDevice);
            hyb_idx++;
        }
    }

    // ── Final norm ──
    int threads = 256;
    int blocks = (d_model + threads - 1) / threads;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rms_norm_kernel), dim3(blocks), dim3(threads), 0, st,
        s->d_hidden, s->d_tmp, s->d_final_norm, d_model, cfg.rms_norm_eps);
    HIP_CHECK(hipGetLastError());
    hipMemcpyDtoDAsync(s->d_hidden, s->d_tmp, d_model * sizeof(float), st);

    // ── LM head: logits = embed @ hidden ──
    // Simple GEMV: vocab x d_model @ d_model -> vocab
    // For now, do on CPU
    hipStreamSynchronize(st);

    std::vector<float> hidden_cpu(d_model);
    hipMemcpy(hidden_cpu.data(), s->d_hidden, d_model * sizeof(float), hipMemcpyDeviceToHost);

    for (int v = 0; v < vocab; ++v) {
        float sum = 0.0f;
        for (int j = 0; j < d_model; ++j) {
            sum += model.embed_w[v * d_model + j] * hidden_cpu[j];
        }
        logits_out[v] = sum;
    }
}

// ── HIP kernel: RMS Norm ──
// __global__ void rms_norm_kernel(float* x, float* y, const float* w, int n, float eps)
// Defined in mamba2_kernels.hip
