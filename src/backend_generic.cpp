// backend_generic.cpp — Universal CPU inference backend
// Reads ModelConfig from any discovered GGUF/H1B/BIN model and runs inference.
// Supports Llama, Mistral, Qwen2, Qwen3, Gemma, Phi architectures with:
//   RMSNorm / LayerNorm, RoPE (partial/full), GQA/MHA, SiLU/SwiGLU/GeGLU, KV cache,
//   optional QKV bias, optional per-head Q/K-norm (Qwen3), MoE FFN routing
//   (top-k softmax gating over "_exps" stacked expert tensors — Qwen2-MoE/
//   Qwen3-MoE/Mixtral convention; no shared-expert or expert-bias support).

#include "backend.h"
#include <sys/stat.h>
#include <dirent.h>
#include "model_discovery.h"
#include "rocm_cpp/tokenizer.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>

// ── Generic CPU Backend ──────────────────────────────────────────────────────
struct GenericBackend : Backend {
    ModelConfig cfg;
    std::vector<float> embed, final_norm, output_weight;
    std::vector<std::vector<float>> layer_w;  // flat per-layer weights
    std::vector<std::vector<float>> k_cache, v_cache; // KV cache [n_layers][max_seq * n_kv * hd]
    int pos = 0;
    std::vector<float> logits_buf;

    // Per-layer weight indices. bq/bk/bv are optional QKV biases (Qwen2 and
    // some other architectures use biased attention projections, unlike
    // Llama) — SIZE_MAX means "not present in this model", distinct from a
    // legitimate index 0 into flat_weights.
    // q_norm/k_norm: optional per-head QK RMSNorm (Qwen3 and others), applied
    // to each head's head_dim-sized slice with a shared [head_dim] weight,
    // right after QKV bias and before RoPE.
    // moe_*: present only on MoE layers (cfg.n_experts > 0). w1/w2/w3 are
    // unused for such layers — the FFN routes through the expert tensors
    // instead. moe_gate_exps/moe_up_exps/moe_down_exps are flat [n_expert *
    // n_ff * hidden]-sized blocks (GGUF's stacked "_exps" 3D tensors); expert
    // e's slice starts at index e * n_ff * hidden and is row-major [n_ff,
    // hidden] — i.e. the exact same layout matmul() already expects for the
    // dense case, just offset per-expert.
    struct LayerW {
        size_t wq, wk, wv, wo, rms_attn, rms_ffn;
        size_t w1 = SIZE_MAX, w2 = SIZE_MAX, w3 = SIZE_MAX;
        size_t bq = SIZE_MAX, bk = SIZE_MAX, bv = SIZE_MAX;
        size_t q_norm = SIZE_MAX, k_norm = SIZE_MAX;
        size_t moe_gate_inp = SIZE_MAX, moe_gate_exps = SIZE_MAX, moe_up_exps = SIZE_MAX, moe_down_exps = SIZE_MAX;
    };
    std::vector<LayerW> layers;

    GenericBackend() { type = BackendType::GENERIC; name = "Generic CPU (GGUF)"; }

    void load_weights(const std::string& base) {
        // Weights stored as flat float vectors: model_layers_N_name.bin
        // Read by the existing W() macro pattern
        auto W = [&](const std::string& name) -> std::vector<float> {
            std::string path = base + "/" + name;
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f) return {};
            size_t n = f.tellg() / sizeof(float); f.seekg(0);
            std::vector<float> d(n); f.read((char*)d.data(), n * sizeof(float));
            return d;
        };
        embed = W("model_embed_tokens_weight.bin");
        final_norm = W("model_norm_weight.bin");

        int L = cfg.n_layers;
        layers.resize(L);
        for (int i = 0; i < L; i++) {
            std::string p = "model_layers_" + std::to_string(i) + "_";
            // LayerW order: wq, wk, wv, wo, rms_attn, rms_ffn, w1, w2, w3
            layers[i] = {
                push(W(p + "self_attn_q_proj.weight")),          // wq
                push(W(p + "self_attn_k_proj.weight")),          // wk
                push(W(p + "self_attn_v_proj.weight")),          // wv
                push(W(p + "self_attn_o_proj.weight")),          // wo
                push(W(p + "input_layernorm.weight")),            // rms_attn
                push(W(p + "post_attention_layernorm.weight")),   // rms_ffn
                push(W(p + "mlp_gate_proj.weight")),              // w1
                push(W(p + "mlp_up_proj.weight")),                // w2
                push(W(p + "mlp_down_proj.weight")),              // w3
            };
        }
    }

    size_t push(std::vector<float>&& v) {
        size_t idx = flat_weights.size();
        flat_weights.insert(flat_weights.end(), v.begin(), v.end());
        return idx;
    }
    std::vector<float> flat_weights;
    float* w(size_t idx) { return flat_weights.data() + idx; }

    bool init(const ModelConfig& model_cfg, const std::string& weights_dir) override {
        cfg = model_cfg;
        printf("Generic: initializing %s (%d layers, %d hidden, %d heads)\n",
               cfg.model_name.c_str(), cfg.n_layers, cfg.hidden, cfg.n_heads);

        // Try loading weights from a GGUF file first
        bool loaded = false;
        if (!cfg.model_path.empty()) {
            std::string gguf_path = cfg.model_path;
            // If model_path is a directory, look for a .gguf inside
            struct stat st;
            if (stat(gguf_path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
                // Find first .gguf in the directory
                DIR* d = opendir(gguf_path.c_str());
                if (d) {
                    struct dirent* e;
                    while ((e = readdir(d)) != nullptr) {
                        std::string n(e->d_name);
                        if (n.size() > 5 && n.substr(n.size()-5) == ".gguf") {
                            gguf_path = gguf_path + "/" + n;
                            break;
                        }
                    }
                    closedir(d);
                }
            }
            printf("Generic: trying GGUF path: %s\n", gguf_path.c_str());
            loaded = load_gguf(gguf_path);
        }
        if (!loaded) {
            // Fall back: old .bin format
            load_weights(weights_dir);
            loaded = !embed.empty();
        }
        if (!loaded) {
            fprintf(stderr, "Generic: could not load weights from %s\n", weights_dir.c_str());
            return false;
        }
        logits_buf.resize(cfg.vocab);
        k_cache.resize(cfg.n_layers);
        v_cache.resize(cfg.n_layers);
        for (auto& k : k_cache) k.resize(cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim);
        for (auto& v : v_cache) v.resize(cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim);
        initialized = true;
        return true;
    }

    bool load_gguf(const std::string& path) {
        ModelConfig hdr_cfg;
        if (!read_gguf_header(path, hdr_cfg)) return false;
        fprintf(stderr, "load_gguf: %s, %d layers, %d hidden\n", hdr_cfg.model_name.c_str(), hdr_cfg.n_layers, hdr_cfg.hidden);
        
        int H = hdr_cfg.hidden_size, L = hdr_cfg.n_layers, NH = hdr_cfg.n_heads;
        int NKV = hdr_cfg.n_kv_heads, HD = hdr_cfg.head_dim;
        int FF = hdr_cfg.intermediate_size;
        [[maybe_unused]] int V = hdr_cfg.vocab_size;
        
        auto load = [&](const std::string& name, std::vector<float>& dst, size_t expected) -> bool {
            std::vector<float> buf;
            size_t n = 0;
            if (!read_gguf_tensor(path, name, buf, &n)) return false;
            if (n != expected) { fprintf(stderr, "  %s: expected %zu, got %zu\n", name.c_str(), expected, n); return false; }
            dst = std::move(buf);
            return true;
        };
        
        // Embedding
        int real_vocab = read_gguf_vocab(path);
        if (real_vocab > 0) cfg.vocab = cfg.vocab_size = real_vocab;
        load("token_embd.weight", embed, (size_t)real_vocab * H);
        
        // Final norm
        load("output_norm.weight", final_norm, H);

        // LM head — optional. Many GGUF exports omit it entirely when the
        // source model ties embeddings (tie_word_embeddings: true); when
        // present, it's a genuinely different matrix from token_embd.weight
        // and using the tied embedding instead silently produces wrong
        // logits (issue #319). output_weight stays empty when absent, and
        // forward() falls back to the tied embedding in that case.
        load("output.weight", output_weight, (size_t)real_vocab * H);

        
        // Per-layer weights
        layers.resize(L);
        flat_weights.clear();
        
        auto load_tensor = [&](const std::string& name, size_t expected) -> size_t {
            std::vector<float> buf;
            size_t n = 0;
            if (!read_gguf_tensor(path, name, buf, &n)) return 0;
            if (n != expected) { fprintf(stderr, "  %s: expected %zu, got %zu\n", name.c_str(), expected, n); return 0; }
            size_t idx = flat_weights.size();
            flat_weights.insert(flat_weights.end(), buf.begin(), buf.end());
            return idx;
        };
        // Like load_tensor, but returns SIZE_MAX (not 0) when the tensor
        // simply isn't present — for genuinely optional tensors (QKV bias),
        // where "absent" and "present at index 0" must stay distinguishable.
        auto load_tensor_optional = [&](const std::string& name, size_t expected) -> size_t {
            std::vector<float> buf;
            size_t n = 0;
            if (!read_gguf_tensor(path, name, buf, &n)) return SIZE_MAX;
            if (n != expected) { fprintf(stderr, "  %s: expected %zu, got %zu\n", name.c_str(), expected, n); return SIZE_MAX; }
            size_t idx = flat_weights.size();
            flat_weights.insert(flat_weights.end(), buf.begin(), buf.end());
            return idx;
        };

        int NE = cfg.n_experts;
        for (int i = 0; i < L; i++) {
            std::string p = "blk." + std::to_string(i) + ".";
            LayerW lw;
            lw.rms_attn = load_tensor(p + "attn_norm.weight", H);
            lw.rms_ffn  = load_tensor(p + "ffn_norm.weight", H);
            lw.wq = load_tensor(p + "attn_q.weight", NH*HD*H);
            lw.wk = load_tensor(p + "attn_k.weight", NKV*HD*H);
            lw.wv = load_tensor(p + "attn_v.weight", NKV*HD*H);
            lw.wo = load_tensor(p + "attn_output.weight", H*NH*HD);
            if (NE > 0) {
                // MoE layer: no dense ffn_gate/up/down — route through the
                // stacked per-expert "_exps" tensors instead.
                lw.moe_gate_inp  = load_tensor(p + "ffn_gate_inp.weight", (size_t)NE*H);
                lw.moe_gate_exps = load_tensor(p + "ffn_gate_exps.weight", (size_t)NE*FF*H);
                lw.moe_up_exps   = load_tensor(p + "ffn_up_exps.weight", (size_t)NE*FF*H);
                lw.moe_down_exps = load_tensor(p + "ffn_down_exps.weight", (size_t)NE*H*FF);
            } else {
                lw.w1 = load_tensor(p + "ffn_gate.weight", FF*H);
                lw.w2 = load_tensor(p + "ffn_up.weight", FF*H);
                lw.w3 = load_tensor(p + "ffn_down.weight", H*FF);
            }
            lw.bq = load_tensor_optional(p + "attn_q.bias", NH*HD);
            lw.bk = load_tensor_optional(p + "attn_k.bias", NKV*HD);
            lw.bv = load_tensor_optional(p + "attn_v.bias", NKV*HD);
            lw.q_norm = load_tensor_optional(p + "attn_q_norm.weight", HD);
            lw.k_norm = load_tensor_optional(p + "attn_k_norm.weight", HD);
            layers[i] = lw;
        }
        {
            int with_bias = 0, with_qknorm = 0;
            for (auto& lw : layers) {
                if (lw.bq != SIZE_MAX) with_bias++;
                if (lw.q_norm != SIZE_MAX) with_qknorm++;
            }
            if (with_bias > 0) printf("Generic: %d/%d layers have biased QKV projections\n", with_bias, L);
            if (with_qknorm > 0) printf("Generic: %d/%d layers have Q/K-norm\n", with_qknorm, L);
            if (NE > 0) printf("Generic: MoE model — %d experts, %d used per token\n", NE, cfg.num_experts_top);
        }
        
        printf("Generic: loaded %zu layers, embed=%zu, final_norm=%zu, lm_head=%s\n",
               layers.size(), embed.size(), final_norm.size(),
               output_weight.empty() ? "tied" : "untied");
        return !embed.empty() && layers.size() == (size_t)L;
    }

    size_t push_vec(float* data, size_t n) {
        if (!data) return 0;
        size_t idx = flat_weights.size();
        flat_weights.insert(flat_weights.end(), data, data + n);
        return idx;
    }

    bool reset() override {
        pos = 0;
        for (auto& k : k_cache) std::fill(k.begin(), k.end(), 0.0f);
        for (auto& v : v_cache) std::fill(v.begin(), v.end(), 0.0f);
        return true;
    }

    static void rmsnorm(float* o, const float* x, const float* w, int n, float eps) {
        float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
        float r = 1.0f / sqrtf(ss / n + eps);
        for (int i = 0; i < n; i++) o[i] = x[i] * r * w[i];
    }

    // NeoX-style (half-split) RoPE — the convention GGUF/llama.cpp-family
    // models (Llama, Qwen, Mistral, ...) actually use: pairs element i with
    // i+rot_dim/2, not adjacent elements (i, i+1). Cross-checked against
    // ZINC's shaders (src/shaders/rope_fused.comp and siblings, all
    // independently confirm half_rot = rope_dim/2 pairing) since ZINC is
    // independently verified to produce coherent output on these same
    // models — this file's previous adjacent-pair version was the GPT-J
    // convention, wrong for this model family, and produced incoherent
    // (real-vocabulary but semantically scrambled) output as a result.
    static void rope(float* q, float* k, int pos, int n_heads, int n_kv, int hd, int rot_dim, float theta) {
        int half = rot_dim / 2;
        for (int h = 0; h < n_heads; h++) {
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / powf(theta, (2.0f * i) / (float)rot_dim);
                float t = pos * freq;
                float cosv = cosf(t), sinv = sinf(t);
                int i0 = h * hd + i, i1 = h * hd + i + half;
                float q0 = q[i0], q1 = q[i1];
                q[i0] = q0 * cosv - q1 * sinv;
                q[i1] = q0 * sinv + q1 * cosv;
            }
        }
        for (int h = 0; h < n_kv; h++) {
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / powf(theta, (2.0f * i) / (float)rot_dim);
                float t = pos * freq;
                float cosv = cosf(t), sinv = sinf(t);
                int i0 = h * hd + i, i1 = h * hd + i + half;
                float k0 = k[i0], k1 = k[i1];
                k[i0] = k0 * cosv - k1 * sinv;
                k[i1] = k0 * sinv + k1 * cosv;
            }
        }
    }

    static void matmul(float* out, const float* in, const float* w, int M, int K) {
        for (int i = 0; i < M; i++) {
            float s = 0;
            for (int j = 0; j < K; j++) s += in[j] * w[i * (size_t)K + j];
            out[i] = s;
        }
    }

    static void silu(float* out, const float* gate, const float* up, int n) {
        for (int i = 0; i < n; i++) {
            float g = gate[i];
            out[i] = (g / (1.0f + expf(-g))) * up[i];
        }
    }

    // GELU activation (tanh approximation): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    static void gelu(float* out, const float* x, int n) {
        const float c = 0.7978845608f; // sqrt(2/pi)
        for (int i = 0; i < n; i++) {
            float v = x[i];
            float x3 = v * v * v;
            float inner = c * (v + 0.044715f * x3);
            out[i] = 0.5f * v * (1.0f + tanhf(inner));
        }
    }

    // GeGLU: gelu(gate) * up — used by Gemma
    static void geglu(float* out, const float* gate, const float* up, int n) {
        gelu(out, gate, n);
        for (int i = 0; i < n; i++) out[i] *= up[i];
    }

    // Squared ReLU GLU: relu(gate)^2 * up — used by Phi
    static void squared_relu_glu(float* out, const float* gate, const float* up, int n) {
        for (int i = 0; i < n; i++) {
            float g = gate[i];
            float r = g > 0.0f ? g : 0.0f;
            out[i] = r * r * up[i];
        }
    }

    // Architecture-specific FFN activation dispatch
    static void ffn_activate(float* out, const float* gate, const float* up, int n, rcpp_arch_t arch) {
        switch (arch) {
            case RCPP_ARCH_GEMMA:
                // GeGLU: gelu(gate) * up
                geglu(out, gate, up, n);
                break;
            case RCPP_ARCH_PHI:
                // Squared ReLU GLU: relu(gate)^2 * up
                squared_relu_glu(out, gate, up, n);
                break;
            default:
                // SwiGLU: silu(gate) * up — Llama, Mistral, Qwen2, Qwen3, BitNet, fallback
                silu(out, gate, up, n);
                break;
        }
    }

    static void softmax(float* x, int n) {
        float mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
        float sum = 0; for (int i = 0; i < n; i++) sum += expf(x[i] - mx);
        float inv = 1.0f / (sum + 1e-10f);
        for (int i = 0; i < n; i++) x[i] = expf(x[i] - mx) * inv;
    }

    int generate(int token_id) override {
        if (!initialized) return -1;
        return forward(token_id);
    }

    bool forward(int token_id, float* hidden_out) override {
        int tok = forward(token_id);
        if (hidden_out) *hidden_out = 0.0f;
        return tok >= 0;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        return false;  // not implemented — use generate() instead
    }

    int forward(int token) {
        if (token < 0 || token >= (int)cfg.vocab) {
            fprintf(stderr, "[generic] token_id=%d out of range [0,%d)\n", token, (int)cfg.vocab);
            return -1;
        }
        std::vector<float> x0(cfg.hidden);
        for (int i = 0; i < cfg.hidden; i++) x0[i] = embed[token * (size_t)cfg.hidden + i];
        return forward_embed(x0.data());
    }

    // Same transformer body as forward(int), but takes a precomputed
    // embedding vector directly instead of doing a token_embd lookup —
    // the splice point for injecting vision embeddings (mm.2 output) at
    // image-placeholder positions instead of a text token's row.
     int forward_embed(const float* x_in) override {
        // Bounds-check KV cache position before writing (fixes OOB/overflow)
        if (pos >= cfg.max_seq_len) {
            fprintf(stderr, "[generic] KV cache overflow: pos=%d >= max_seq_len=%d\n", pos, cfg.max_seq_len);
            return -1;
        }
        int H = cfg.hidden, NH = cfg.n_heads, NKV = cfg.n_kv_heads, HD = cfg.head_dim;
        int GQA = NH / NKV, FF = cfg.intermediate_size, V = cfg.vocab;
        float eps = cfg.rms_norm_eps, theta = cfg.rope_theta;
        int rot_dim = cfg.head_dim;  // full RoPE by default

        std::vector<float> x(H), x2(H), q(NH*HD), k(NKV*HD), v(NKV*HD), scores(HD);
        std::vector<float> att(NH*HD);
        std::vector<float> gate_up(FF*2);

        for (int i = 0; i < H; i++) x[i] = x_in[i];

        for (int il = 0; il < cfg.n_layers; il++) {
            auto& l = layers[il];
            int kv_begin = pos * NKV * HD;

            // RMSNorm → QKV
            rmsnorm(x2.data(), x.data(), w(l.rms_attn), H, eps);
            matmul(q.data(), x2.data(), w(l.wq), NH*HD, H);
            matmul(k.data(), x2.data(), w(l.wk), NKV*HD, H);
            matmul(v.data(), x2.data(), w(l.wv), NKV*HD, H);

            // Optional QKV bias (Qwen2 and others use biased attention
            // projections; absent for architectures like Llama).
            if (l.bq != SIZE_MAX) { float* b = w(l.bq); for (int i = 0; i < NH*HD; i++) q[i] += b[i]; }
            if (l.bk != SIZE_MAX) { float* b = w(l.bk); for (int i = 0; i < NKV*HD; i++) k[i] += b[i]; }
            if (l.bv != SIZE_MAX) { float* b = w(l.bv); for (int i = 0; i < NKV*HD; i++) v[i] += b[i]; }

            // Optional per-head QK-norm (Qwen3 and others): RMSNorm applied
            // independently to each head's head_dim-sized slice with a
            // shared [head_dim] weight. Must happen before RoPE.
            if (l.q_norm != SIZE_MAX) {
                float* qn = w(l.q_norm);
                for (int h = 0; h < NH; h++) rmsnorm(&q[h*HD], &q[h*HD], qn, HD, eps);
            }
            if (l.k_norm != SIZE_MAX) {
                float* kn = w(l.k_norm);
                for (int h = 0; h < NKV; h++) rmsnorm(&k[h*HD], &k[h*HD], kn, HD, eps);
            }

            // RoPE
            rope(q.data(), k.data(), pos, NH, NKV, HD, rot_dim, theta);

            // KV cache
            memcpy(&k_cache[il][kv_begin], k.data(), NKV * HD * sizeof(float));
            memcpy(&v_cache[il][kv_begin], v.data(), NKV * HD * sizeof(float));

            // Attention: GQA
            std::fill(att.begin(), att.end(), 0.0f);
            for (int h = 0; h < NH; h++) {
                int kv_h = h / GQA;
                float* Q = &q[h * HD];
                // Score over all past positions
                for (int t = 0; t <= pos; t++) {
                    float* K = &k_cache[il][t * NKV * HD + kv_h * HD];
                    float s = 0;
                    for (int d = 0; d < HD; d++) s += Q[d] * K[d];
                    scores[t] = s / sqrtf((float)HD);
                }
                softmax(scores.data(), pos + 1);
                // Weighted sum of V
                for (int d = 0; d < HD; d++) {
                    float sum = 0;
                    for (int t = 0; t <= pos; t++) {
                        float* V = &v_cache[il][t * NKV * HD + kv_h * HD];
                        sum += scores[t] * V[d];
                    }
                    att[h * HD + d] = sum;
                }
            }

            // O proj
            matmul(x2.data(), att.data(), w(l.wo), H, NH*HD);
            // Residual
            for (int i = 0; i < H; i++) x[i] += x2[i];

            // FFN: RMSNorm → gate/up → activation (arch-specific) → down → residual (dense), or
            // RMSNorm → router top-k → per-expert gate/up/activation/down,
            // weighted sum → residual (MoE).
            rmsnorm(x2.data(), x.data(), w(l.rms_ffn), H, eps);
            if (l.moe_gate_inp != SIZE_MAX) {
                int NE = cfg.n_experts, NEU = cfg.num_experts_top;
                std::vector<float> router_probs(NE);
                matmul(router_probs.data(), x2.data(), w(l.moe_gate_inp), NE, H);
                softmax(router_probs.data(), NE);

                // Top-k expert selection by router probability (descending),
                // then renormalize just the selected weights to sum to 1 —
                // matches llama.cpp's build_moe_ffn with norm_w=true (the
                // convention used by Qwen2-MoE/Qwen3-MoE/Mixtral).
                std::vector<int> idx(NE);
                for (int e = 0; e < NE; e++) idx[e] = e;
                std::partial_sort(idx.begin(), idx.begin() + NEU, idx.end(),
                                   [&](int a, int b) { return router_probs[a] > router_probs[b]; });
                float wsum = 0.0f;
                for (int t = 0; t < NEU; t++) wsum += router_probs[idx[t]];
                if (wsum < 6.103515625e-5f) wsum = 6.103515625e-5f; // match ggml's clamp

                std::vector<float> ffn_acc(H, 0.0f);
                std::vector<float> gate_buf(FF), up_buf(FF), silu_buf(FF), down_buf(H);
                for (int t = 0; t < NEU; t++) {
                    int e = idx[t];
                    float we = router_probs[e] / wsum;
                    float* wg = w(l.moe_gate_exps) + (size_t)e * FF * H;
                    float* wu = w(l.moe_up_exps)   + (size_t)e * FF * H;
                    float* wd = w(l.moe_down_exps) + (size_t)e * H * FF;
                    matmul(gate_buf.data(), x2.data(), wg, FF, H);
                    matmul(up_buf.data(), x2.data(), wu, FF, H);
                    ffn_activate(silu_buf.data(), gate_buf.data(), up_buf.data(), FF, cfg.arch);
                    matmul(down_buf.data(), silu_buf.data(), wd, H, FF);
                    for (int i = 0; i < H; i++) ffn_acc[i] += we * down_buf[i];
                }
                for (int i = 0; i < H; i++) x[i] += ffn_acc[i];
            } else {
                matmul(gate_up.data(), x2.data(), w(l.w1), FF, H);
                matmul(&gate_up[FF], x2.data(), w(l.w2), FF, H);
                std::vector<float> silu_buf(FF);
                ffn_activate(silu_buf.data(), gate_up.data(), &gate_up[FF], FF, cfg.arch);
                matmul(x2.data(), silu_buf.data(), w(l.w3), H, FF);
                for (int i = 0; i < H; i++) x[i] += x2[i];
            }
        }

        // Final RMSNorm
        rmsnorm(x2.data(), x.data(), final_norm.data(), H, eps);

        // LM head — untied output.weight when the model has one, else tied embedding.
        const float* lm_head = output_weight.empty() ? embed.data() : output_weight.data();
        matmul(logits_buf.data(), x2.data(), lm_head, V, H);

        pos++;

        // Argmax
        int best = 0; float bestv = logits_buf[0];
        for (int i = 1; i < V; i++) {
            if (logits_buf[i] > bestv) { bestv = logits_buf[i]; best = i; }
        }
        return best;
    }

    void destroy() override { initialized = false; }

    ~GenericBackend() override { destroy(); }

    float benchmark(int tokens = 10) override {
        if (!initialized) return 0;
        reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        int tok = 100;
        for (int i = 0; i < tokens; i++) tok = forward(tok);
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return ms / tokens;
    }
};

Backend* create_generic_backend() { return new GenericBackend(); }
