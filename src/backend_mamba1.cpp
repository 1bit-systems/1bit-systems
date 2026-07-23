// backend_mamba1.cpp — Mamba1 GPU inference backend
// Uses mamba1_engine.hip kernels for GPU-accelerated Mamba1 SSM layers
// on AMD Strix Halo (gfx1151). Handles both pure Mamba1 (Zamba-7B-v1)
// and Mamba1+MoE (BlackMamba) architectures.
//
// BlackMamba alternates two layer types:
//   "r" (regular): x = x + mamba1_ssm(rmsnorm(x))
//   "8" (MoE):     x = x + top1_moe_swiglu(rmsnorm(x))
//
// This backend:
//   1. Loads weights from GGUF (via gguf_reader.h)
//   2. Uploads per-layer SSM weights to GPU
//   3. For MoE layers, loads expert FFN weights and router
//   4. Runs per-layer dispatch: mamba1_inner kernel for SSM layers,
//      expert GEMV + SiLU + GEMV for MoE layers
//   5. Returns logits via tied embedding LM head

#include "backend.h"
#include "gguf_reader.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <unordered_map>

#define HIP_CHECK(e) do { auto _s=(e); if(_s!=hipSuccess) { \
    fprintf(stderr,"[mamba1] HIP Error %d at %s:%d\n",_s,__FILE__,__LINE__); \
    abort(); }} while(0)

// ── Kernel launch wrappers (defined in mamba1_engine.hip) ──
extern "C" {
    void launch_rms_ker(const float* x, float* y, const float* w, int n, float eps, hipStream_t stream);
    void launch_fp32_gemv(const float* W, const float* x, float* y, int M, int K, hipStream_t stream);
    void launch_add_residual_k(float* y, const float* x, int n, hipStream_t stream);
    void launch_silu_gate_k(float* y, const float* fused, int hidden, hipStream_t stream);
    void launch_scale_add_residual_k(float* y, const float* x, float scale, int n, hipStream_t stream);

    int mamba1_forward(
        float* hidden,
        const float* in_proj_w, const float* norm_w,
        const float* conv1d_w, const float* conv1d_b, int d_conv,
        const float* x_proj_w,
        const float* dt_w, const float* dt_b,
        const float* A_log, const float* D,
        const float* out_proj_w,
        float* conv_state, float* ssm_state,
        float* normed, float* inproj, float* gated,
        int d_model, int d_inner, int d_state, int dt_rank,
        hipStream_t stream);
}

// ── Per-layer Mamba1 SSM weights (host copy) ──
struct Mamba1LayerHost {
    std::vector<float> input_norm_w;     // [d_model]
    std::vector<float> in_proj;          // [2*d_inner, d_model]
    std::vector<float> conv1d_w;         // [d_conv, d_inner]
    std::vector<float> conv1d_b;         // [d_inner]
    std::vector<float> x_proj;           // [dt_rank+2*d_state, d_inner]
    std::vector<float> dt_w;             // [d_inner, dt_rank]
    std::vector<float> dt_b;             // [d_inner]
    std::vector<float> A_log;            // [d_state, d_inner]
    std::vector<float> D;                // [d_inner]
    std::vector<float> out_proj;         // [d_model, d_inner]
};

// ── Per-layer MoE FFN weights (host copy) ──
struct MoeLayerHost {
    std::vector<float> input_norm_w;     // [d_model] — RMS norm
    std::vector<float> router_w;         // [n_experts, d_model]
    std::vector<float> router_b;         // [n_experts] — optional
    std::vector<std::vector<float>> fc1; // [n_experts][2*hidden, d_model] — fused gate+up
    std::vector<std::vector<float>> fc2; // [n_experts][d_model, hidden]
    int n_experts = 0;
    int hidden = 0;                      // intermediate_size (ffn_hidden)
};

// ── Per-layer GPU state ──
struct Mamba1LayerGPU {
    float *d_in_proj = nullptr, *d_conv1d_w = nullptr, *d_conv1d_b = nullptr;
    float *d_x_proj = nullptr, *d_dt_w = nullptr, *d_dt_b = nullptr;
    float *d_A_log = nullptr, *d_D = nullptr, *d_out_proj = nullptr, *d_norm_w = nullptr;
    float *d_conv_state = nullptr, *d_ssm_state = nullptr;
};

struct MoeLayerGPU {
    float *d_norm_w = nullptr;
    float *d_router_w = nullptr, *d_router_b = nullptr;
    std::vector<float*> d_fc1, d_fc2;
};

// ── Mamba1 Backend ──
struct Mamba1Backend : Backend {
    // Model config
    int d_model = 0, d_inner = 0, d_state = 0, d_conv = 0, dt_rank = 0;
    int vocab_size = 0, n_layers = 0;
    bool is_moe = false;
    std::vector<bool> layer_is_mamba;  // true=SSM, false=MoE

    // Host weights
    std::vector<float> embed;           // [vocab, d_model]
    std::vector<float> final_norm_w;    // [d_model]

    // Per-layer SSM weights (Mamba1 layers)
    std::vector<Mamba1LayerHost> mamba_layer_host;

    // Per-layer MoE weights
    std::vector<MoeLayerHost> moe_layer_host;

    // GPU pointers
    Mamba1LayerGPU* mamba_layer_gpu = nullptr;
    MoeLayerGPU* moe_layer_gpu = nullptr;
    float *d_embed = nullptr, *d_final_norm_w = nullptr;
    float *d_hidden = nullptr, *d_normed = nullptr, *d_inproj = nullptr;
    float *d_gated = nullptr, *d_logits = nullptr, *d_moe_scratch = nullptr;
    hipStream_t stream = nullptr;

    // State
    int pos = 0;
    std::vector<float> logits_buf;

    Mamba1Backend() {
        type = BackendType::HIP_GPU;
        name = "Mamba1 GPU (Strix Halo)";
    }

    ~Mamba1Backend() override { destroy(); }

    // ── HIP-safe upload helpers ──
    float* upload_f32(const std::vector<float>& src) {
        if (src.empty()) return nullptr;
        float* d = nullptr;
        HIP_CHECK(hipMalloc(&d, src.size() * sizeof(float)));
        HIP_CHECK(hipMemcpyAsync(d, src.data(), src.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        return d;
    }

    template<typename T>
    static void safe_free(T*& p) { if (p) { (void)hipFree(p); p = nullptr; } }

    // ── Load Mamba1 SSM layer from GGUF ──
    bool load_mamba_layer(GgufReader& r, int layer_idx, Mamba1LayerHost& ml) {
        auto p = [&](const std::string& name) -> std::string {
            return "blk." + std::to_string(layer_idx) + "." + name;
        };

        if (!r.get_tensor_f32(p("attn_norm.weight"), ml.input_norm_w)) return false;
        if (!r.get_tensor_f32(p("ssm_in.weight"), ml.in_proj)) return false;

        // Derive d_inner from in_proj size: in_proj is [2*d_inner, d_model]
        if (d_model == 0) d_model = (int)ml.in_proj.size() / (2 * d_inner);
        int expected_inner = (int)ml.in_proj.size() / d_model;
        // If d_inner not set yet, derive it
        if (d_inner == 0) d_inner = expected_inner / 2;

        r.get_tensor_f32(p("ssm_conv1d.weight"), ml.conv1d_w);
        r.get_tensor_f32(p("ssm_conv1d.bias"), ml.conv1d_b);
        r.get_tensor_f32(p("ssm_x.weight"), ml.x_proj);

        // dt_rank from x_proj shape
        if (ml.x_proj.size() > 0 && d_inner > 0) {
            int xp_per_inner = (int)(ml.x_proj.size() / d_inner);
            dt_rank = xp_per_inner - 2 * d_state;
            if (dt_rank <= 0) dt_rank = d_inner / 16;  // fallback guess
        }

        r.get_tensor_f32(p("ssm_dt.weight"), ml.dt_w);
        r.get_tensor_f32(p("ssm_dt.bias"), ml.dt_b);
        r.get_tensor_f32(p("ssm_a"), ml.A_log);
        r.get_tensor_f32(p("ssm_d"), ml.D);
        r.get_tensor_f32(p("ssm_out.weight"), ml.out_proj);

        return true;
    }

    // ── Load MoE FFN layer from GGUF ──
    bool load_moe_layer(GgufReader& r, int layer_idx, MoeLayerHost& el) {
        auto p = [&](const std::string& name) -> std::string {
            return "blk." + std::to_string(layer_idx) + "." + name;
        };

        if (!r.get_tensor_f32(p("attn_norm.weight"), el.input_norm_w)) return false;
        r.get_tensor_f32(p("ffn_gate.weight"), el.router_w);
        r.get_tensor_f32(p("ffn_gate.bias"), el.router_b);

        int n_exp = (int)(el.router_w.size() / d_model);
        if (n_exp <= 0) n_exp = 16;  // BlackMamba default

        el.n_experts = n_exp;
        el.fc1.resize(n_exp);
        el.fc2.resize(n_exp);

        for (int x = 0; x < n_exp; x++) {
            char ep[96];
            snprintf(ep, sizeof(ep), "blk.%d.ffn_expert.%d.weight_1", layer_idx, x);
            r.get_tensor_f32(ep, el.fc1[x]);
            snprintf(ep, sizeof(ep), "blk.%d.ffn_expert.%d.weight_2", layer_idx, x);
            r.get_tensor_f32(ep, el.fc2[x]);
        }

        // Derive hidden from first expert's fc1 size
        if (!el.fc1.empty() && !el.fc1[0].empty()) {
            el.hidden = (int)(el.fc1[0].size() / d_model) / 2;  // fused gate+up
        }

        return true;
    }

    // ── Init ──
    bool init(const ModelConfig& cfg, const std::string& weights_dir) override {
        this->cfg = cfg;
        d_model = cfg.hidden_size;
        n_layers = cfg.num_layers;
        vocab_size = cfg.vocab_size;

        std::string model_path = cfg.model_path;
        fprintf(stderr, "[mamba1] Initializing Mamba1 GPU backend: %s\n", model_path.c_str());

        // ── Open GGUF ──
        GgufReader r;
        if (!r.open(model_path)) {
            fprintf(stderr, "[mamba1] Failed to open GGUF: %s\n", model_path.c_str());
            return false;
        }

        // Read SSM config from GGUF metadata
        r.get_u32("mamba.ssm.state_size", (uint32_t&)d_state);
        r.get_u32("mamba.ssm.conv_kernel", (uint32_t&)d_conv);
        if (d_state == 0) d_state = 16;   // fallback: Zamba-7B-v1 default
        if (d_conv == 0) d_conv = 4;      // fallback

        // ── Load embedding and final norm ──
        std::vector<float> embed_tmp;
        size_t n;
        if (!r.get_tensor_f32("token_embd.weight", embed_tmp, &n)) {
            fprintf(stderr, "[mamba1] Missing token_embd.weight\n");
            return false;
        }
        // GGUF stores [d_model, vocab]; transpose to [vocab, d_model]
        int actual_vocab = (int)n / d_model;
        embed.resize((size_t)actual_vocab * d_model);
        for (int i = 0; i < actual_vocab; ++i)
            for (int j = 0; j < d_model; ++j)
                embed[(size_t)i * d_model + j] = embed_tmp[(size_t)j * actual_vocab + i];
        vocab_size = actual_vocab;

        r.get_tensor_f32("output_norm.weight", final_norm_w);

        // ── Detect layer types ──
        layer_is_mamba.resize(n_layers);
        mamba_layer_host.resize(n_layers);
        moe_layer_host.resize(n_layers);
        is_moe = false;

        for (int l = 0; l < n_layers; l++) {
            auto p = [&](const std::string& name) -> std::string {
                return "blk." + std::to_string(l) + "." + name;
            };
            if (r.has_tensor(p("ssm_in.weight"))) {
                layer_is_mamba[l] = true;
                if (!load_mamba_layer(r, l, mamba_layer_host[l])) {
                    fprintf(stderr, "[mamba1] Failed to load SSM layer %d\n", l);
                    return false;
                }
            } else if (r.has_tensor(p("ffn_gate.weight"))) {
                layer_is_mamba[l] = false;
                is_moe = true;
                if (!load_moe_layer(r, l, moe_layer_host[l])) {
                    fprintf(stderr, "[mamba1] Failed to load MoE layer %d\n", l);
                    return false;
                }
            } else {
                fprintf(stderr, "[mamba1] Layer %d: unknown type (no ssm_in or ffn_gate)\n", l);
                return false;
            }
        }

        // ── Allocate GPU memory ──
        HIP_CHECK(hipStreamCreate(&stream));

        // Embeddings + final norm
        d_embed = upload_f32(embed);
        d_final_norm_w = upload_f32(final_norm_w);

        // Per-layer SSM GPU state
        mamba_layer_gpu = new Mamba1LayerGPU[n_layers];
        moe_layer_gpu = new MoeLayerGPU[n_layers];

        for (int l = 0; l < n_layers; l++) {
            if (layer_is_mamba[l]) {
                auto& ml = mamba_layer_host[l];
                auto& gl = mamba_layer_gpu[l];
                gl.d_in_proj   = upload_f32(ml.in_proj);
                gl.d_conv1d_w  = upload_f32(ml.conv1d_w);
                gl.d_conv1d_b  = upload_f32(ml.conv1d_b);
                gl.d_x_proj    = upload_f32(ml.x_proj);
                gl.d_dt_w      = upload_f32(ml.dt_w);
                gl.d_dt_b      = upload_f32(ml.dt_b);
                gl.d_A_log     = upload_f32(ml.A_log);
                gl.d_D         = upload_f32(ml.D);
                gl.d_out_proj  = upload_f32(ml.out_proj);
                gl.d_norm_w    = upload_f32(ml.input_norm_w);

                // Allocate SSM state (persistent across tokens)
                HIP_CHECK(hipMalloc(&gl.d_conv_state, (size_t)(d_conv - 1) * d_inner * sizeof(float)));
                HIP_CHECK(hipMalloc(&gl.d_ssm_state, (size_t)d_state * d_inner * sizeof(float)));
                HIP_CHECK(hipMemsetAsync(gl.d_conv_state, 0, (size_t)(d_conv - 1) * d_inner * sizeof(float), stream));
                HIP_CHECK(hipMemsetAsync(gl.d_ssm_state, 0, (size_t)d_state * d_inner * sizeof(float), stream));
            } else {
                auto& el = moe_layer_host[l];
                auto& gl = moe_layer_gpu[l];
                gl.d_norm_w = upload_f32(el.input_norm_w);
                gl.d_router_w = upload_f32(el.router_w);
                gl.d_router_b = upload_f32(el.router_b);
                gl.d_fc1.resize(el.n_experts);
                gl.d_fc2.resize(el.n_experts);
                for (int x = 0; x < el.n_experts; x++) {
                    gl.d_fc1[x] = upload_f32(el.fc1[x]);
                    gl.d_fc2[x] = upload_f32(el.fc2[x]);
                }
            }
        }

        // Scratch buffers
        int max_hidden_moe = 0;
        for (int l = 0; l < n_layers; l++) {
            if (!layer_is_mamba[l]) {
                max_hidden_moe = std::max(max_hidden_moe, moe_layer_host[l].hidden);
            }
        }
        HIP_CHECK(hipMalloc(&d_hidden, (size_t)d_model * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_normed, (size_t)d_model * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_inproj, (size_t)2 * d_inner * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_gated, (size_t)d_inner * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_logits, (size_t)vocab_size * sizeof(float)));
        // MoE scratch: router logits + fc1 output + activated
        int moe_scratch = std::max(64, 2 * max_hidden_moe);  // n_experts ≤ 64, fc1 = 2*hidden
        HIP_CHECK(hipMalloc(&d_moe_scratch, (size_t)moe_scratch * sizeof(float)));
        HIP_CHECK(hipMemsetAsync(d_hidden, 0, (size_t)d_model * sizeof(float), stream));

        // Host logits buffer
        logits_buf.resize(vocab_size, 0.0f);

        HIP_CHECK(hipStreamSynchronize(stream));
        int n_ssm = 0, n_moe = 0;
        for (int l = 0; l < n_layers; l++) {
            if (layer_is_mamba[l]) n_ssm++; else n_moe++;
        }
        fprintf(stderr, "[mamba1] Loaded %d layers (%d SSM + %d MoE) on GPU.\n",
                n_layers, n_ssm, n_moe);

        initialized = true;
        return true;
    }

    bool reset() override {
        if (!initialized) return false;
        pos = 0;
        // Reset SSM states
        for (int l = 0; l < n_layers; l++) {
            if (layer_is_mamba[l]) {
                auto& gl = mamba_layer_gpu[l];
                if (gl.d_conv_state) {
                    HIP_CHECK(hipMemsetAsync(gl.d_conv_state, 0,
                        (size_t)(d_conv - 1) * d_inner * sizeof(float), stream));
                }
                if (gl.d_ssm_state) {
                    HIP_CHECK(hipMemsetAsync(gl.d_ssm_state, 0,
                        (size_t)d_state * d_inner * sizeof(float), stream));
                }
            }
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        return true;
    }

    bool forward(int token_id, float* hidden_out) override {
        if (!initialized) return false;

        // 1. Embedding lookup
        HIP_CHECK(hipMemcpyAsync(d_hidden, &embed[(size_t)token_id * d_model],
                    (size_t)d_model * sizeof(float), hipMemcpyHostToDevice, stream));

        // 2. Per-layer loop
        for (int l = 0; l < n_layers; l++) {
            if (layer_is_mamba[l]) {
                // ── Mamba1 SSM layer (fully GPU) ──
                auto& gl = mamba_layer_gpu[l];
                mamba1_forward(
                    d_hidden,
                    gl.d_in_proj, gl.d_norm_w,
                    gl.d_conv1d_w, gl.d_conv1d_b, d_conv,
                    gl.d_x_proj,
                    gl.d_dt_w, gl.d_dt_b,
                    gl.d_A_log, gl.d_D,
                    gl.d_out_proj,
                    gl.d_conv_state, gl.d_ssm_state,
                    d_normed, d_inproj, d_gated,
                    d_model, d_inner, d_state, dt_rank,
                    stream);

            } else {
                // ── MoE FFN layer (GPU GEMVs + host argmax) ──
                auto& gl = moe_layer_gpu[l];
                auto& el = moe_layer_host[l];

                // 2a. RMS norm: d_normed = rmsnorm(d_hidden)
                launch_rms_ker(d_hidden, d_normed, gl.d_norm_w, d_model, 1e-5f, stream);

                // 2b. Router GEMV: d_moe_scratch[0..n_experts] = W_router @ normed
                launch_fp32_gemv(gl.d_router_w, d_normed, d_moe_scratch, el.n_experts, d_model, stream);

                // 2c. Download router logits, find top-1 expert (host-side,
                //     since n_experts ≤ 64 — cheap enough)
                std::vector<float> router_logits(el.n_experts);
                HIP_CHECK(hipMemcpyAsync(router_logits.data(), d_moe_scratch,
                            (size_t)el.n_experts * sizeof(float),
                            hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipStreamSynchronize(stream));

                int top1 = 0;
                float top1_val = -1e30f;
                for (int x = 0; x < el.n_experts; x++) {
                    float v = el.router_b.empty() ? router_logits[x] : router_logits[x] + el.router_b[x];
                    if (v > top1_val) { top1_val = v; top1 = x; }
                }
                float gate_w = 1.0f / (1.0f + std::expf(-top1_val));

                // 2d. Expert FC1 GEMV: d_moe_scratch[2*hidden] = W_fc1 @ normed
                int hidden = el.hidden;
                launch_fp32_gemv(gl.d_fc1[top1], d_normed, d_moe_scratch, 2 * hidden, d_model, stream);

                // 2e. SiLU gate: d_inproj[hidden] = silu(gate) * up
                launch_silu_gate_k(d_inproj, d_moe_scratch, hidden, stream);

                // 2f. Expert FC2 GEMV: d_normed[d_model] = W_fc2 @ activated
                launch_fp32_gemv(gl.d_fc2[top1], d_inproj, d_normed, d_model, hidden, stream);

                // 2g. Scale by gate weight + residual: d_hidden += gate_w * d_normed
                launch_scale_add_residual_k(d_hidden, d_normed, gate_w, d_model, stream);
            }
        }

        // 3. Download hidden state (not logits — lm_head does that)
        HIP_CHECK(hipMemcpyAsync(hidden_out, d_hidden, (size_t)d_model * sizeof(float),
                    hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        pos++;
        return true;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        if (!hidden || !logits) return false;

        // Upload hidden state to GPU
        HIP_CHECK(hipMemcpyAsync(d_hidden, hidden, (size_t)d_model * sizeof(float),
                    hipMemcpyHostToDevice, stream));

        // Final RMS norm: d_normed = rmsnorm(d_hidden)
        launch_rms_ker(d_hidden, d_normed, d_final_norm_w, d_model, 1e-5f, stream);

        // LM head GEMV: d_logits[v] = embed[v,:] @ d_normed
        // One block per vocab row, 256 threads reducing across d_model
        launch_fp32_gemv(d_embed, d_normed, d_logits, vocab_size, d_model, stream);

        // Download logits
        HIP_CHECK(hipMemcpyAsync(logits, d_logits, (size_t)vocab_size * sizeof(float),
                    hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        // Argmax
        if (argmax) {
            *argmax = 0;
            float best = logits[0];
            for (int v = 1; v < vocab_size; v++) {
                if (logits[v] > best) { best = logits[v]; *argmax = v; }
            }
        }
        return true;
    }

    int generate(int token_id) override {
        std::vector<float> hidden(d_model);
        if (!forward(token_id, hidden.data())) return -1;
        if (!lm_head(hidden.data(), logits_buf.data(), nullptr)) return -1;
        int best = 0;
        for (int v = 1; v < vocab_size; v++)
            if (logits_buf[v] > logits_buf[best]) best = v;
        return best;
    }

    float benchmark(int tokens = 10) override {
        if (!initialized) return 0;
        reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        int tok = 1;
        for (int i = 0; i < tokens; i++) {
            std::vector<float> hidden(d_model);
            forward(tok, hidden.data());
            lm_head(hidden.data(), logits_buf.data(), &tok);
        }
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return ms / tokens;
    }

    void destroy() override {
        HIP_CHECK(hipStreamSynchronize(stream));
        if (mamba_layer_gpu) {
            for (int l = 0; l < n_layers; l++) {
                auto& gl = mamba_layer_gpu[l];
                safe_free(gl.d_in_proj);
                safe_free(gl.d_conv1d_w);
                safe_free(gl.d_conv1d_b);
                safe_free(gl.d_x_proj);
                safe_free(gl.d_dt_w);
                safe_free(gl.d_dt_b);
                safe_free(gl.d_A_log);
                safe_free(gl.d_D);
                safe_free(gl.d_out_proj);
                safe_free(gl.d_norm_w);
                safe_free(gl.d_conv_state);
                safe_free(gl.d_ssm_state);
            }
            delete[] mamba_layer_gpu;
            mamba_layer_gpu = nullptr;
        }
        if (moe_layer_gpu) {
            for (int l = 0; l < n_layers; l++) {
                auto& gl = moe_layer_gpu[l];
                safe_free(gl.d_norm_w);
                safe_free(gl.d_router_w);
                safe_free(gl.d_router_b);
                for (auto p : gl.d_fc1) safe_free(p);
                for (auto p : gl.d_fc2) safe_free(p);
            }
            delete[] moe_layer_gpu;
            moe_layer_gpu = nullptr;
        }
        safe_free(d_embed);
        safe_free(d_final_norm_w);
        safe_free(d_hidden);
        safe_free(d_normed);
        safe_free(d_inproj);
        safe_free(d_gated);
        safe_free(d_logits);
        safe_free(d_moe_scratch);
        if (stream) { (void)hipStreamDestroy(stream); stream = nullptr; }
        initialized = false;
    }
};

// ── Factory ──
extern "C" Backend* create_mamba1_backend() {
    return new Mamba1Backend();
}
