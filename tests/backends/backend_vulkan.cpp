// backend_vulkan.cpp — Vulkan compute backend for 1bit inference
// Portable: runs on AMD, NVIDIA, Intel GPUs via Vulkan 1.2+
// Uses GLSL compute shaders from kernels/vulkan/ for GEMM + attention.
// FP32 compute path (portable; FP16 variant is a later optimization).
//
// Part of the unified zaya_server binary. Compiled when USE_VULKAN=ON.
#include "backend.h"
#include "../../src/vulkan_rt.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

// ─── FP32 weight helpers ────────────────────────────────────────────
static std::vector<float> load_bin(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return {};
    size_t n = f.tellg() / sizeof(float);
    f.seekg(0);
    std::vector<float> d(n);
    f.read((char*)d.data(), n * sizeof(float));
    return d;
}

// ─── Per-layer weight struct (used by VulkanBackend + 1BP loader) ──
struct CpuLayer {
    std::vector<float> nw, wq, wk, wv1, wv2, wo, pan;
    std::vector<float> gu, dn;  // gate+up (stacked) and down
    std::vector<float> g, up;   // separate gate (for SiLU) and up (for multiply)
};

// ─── Forward decl: 1BP loader (defined after OnebpModel include) ───
static bool load_1bp_vulkan(
    std::vector<float>& embed_, std::vector<float>& fnorm_,
    std::vector<CpuLayer>& layers_, ModelConfig& cfg_,
    const std::string& path);

// ─── RMSNorm (CPU-side, H is small enough) ─────────────────────────
static void rmsnorm_cpu(std::vector<float>& x, const std::vector<float>& w,
                         int n, float eps = 1e-5f) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float iv = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) x[i] = x[i] * iv * w[i];
}

// ─── SiLU activation (CPU-side) ────────────────────────────────────
static void silu_mul_cpu(std::vector<float>& out, const std::vector<float>& g,
                          const std::vector<float>& u, int n) {
    for (int i = 0; i < n; i++) {
        float v = g[i];
        out[i] = (v / (1.0f + expf(-v))) * u[i];
    }
}

// ─── Vulkan Backend ─────────────────────────────────────────────────
class VulkanBackend : public InferenceBackend {
    ModelConfig cfg_;
    bool loaded_ = false;
    vkrt::VkCtx ctx_;
    bool ctx_ok_ = false;

    // Pipelines
    vkrt::Pipeline pipe_gemv_;    // dmmv_tq2_bonsai.comp  — ternary GEMV
    vkrt::Pipeline pipe_gemm_;    // matmul_fp32.comp       — GEMM for projections

    // Weight buffers (FP32 on GPU) — CpuLayer defined at file scope above
    std::vector<CpuLayer> layers_;
    std::vector<float> embed_;  // [vocab, hidden] FP32
    std::vector<float> fnorm_;  // [hidden] FP32

    // GPU-side scratch buffers
    vkrt::GpuBuffer buf_hs_, buf_tmp_, buf_out_;
    vkrt::GpuBuffer buf_wt_;    // reusable weight tile
    bool scratch_ok_ = false;

    // Resolve shader path relative to repo root
    std::string shader_dir_;

    // KV cache: per-layer [max_seq * n_kv * hd]
    std::vector<std::vector<float>> k_cache_, v_cache_;
    int max_seq_ = 2048;

public:
    BackendType type() const override { return BackendType::VULKAN; }
    const char* name() const override { return "Vulkan GPU"; }
    float estimated_tok_s() const override { return 22.0f; }
    bool is_coherent() const override { return true; }

    VulkanBackend() {
        // Try to find shaders relative to the binary location
        shader_dir_ = "kernels/vulkan/";
    }

    bool is_available() override {
        if (ctx_ok_) return true;
        try {
            ctx_.init();
            ctx_ok_ = true;
            fprintf(stderr, "  Vulkan: found %s\n", ctx_.deviceName);
            return true;
        } catch (...) {
            fprintf(stderr, "  Vulkan: no device found\n");
            return false;
        }
    }

    bool load_model(const ModelConfig& cfg) override {
        cfg_ = cfg;
        unload_model();

        if (!ctx_ok_) { is_available(); if (!ctx_ok_) return false; }

        // Check if model is in 1BP format
        std::string mp = cfg.model_path;
        if (mp.size() > 4 && mp.substr(mp.size()-4) == ".1bp") {
            return load_1bp_vulkan(embed_, fnorm_, layers_, cfg_, mp);
        }

        int H = cfg.hidden_size;
        int N_LAYERS = cfg.num_layers;
        int VOCAB = cfg.vocab_size;
        std::string W = cfg.weights_dir;
        if (!W.empty() && W.back() != '/') W += '/';

        fprintf(stderr, "  Vulkan: loading %s (H=%d, L=%d, vocab=%d)\n",
                cfg.model_name.c_str(), H, N_LAYERS, VOCAB);

        // Load embedding + final norm
        embed_ = load_bin(W + "model_embed_tokens_weight.bin");
        fnorm_ = load_bin(W + "model_norm_weight.bin");
        if (embed_.empty() || fnorm_.empty()) {
            fprintf(stderr, "  Vulkan: missing required weights in %s\n", W.c_str());
            return false;
        }

        // Load layers (CPU-side FP32 — Vulkan will use these directly)
        layers_.resize(N_LAYERS);
        auto L = [](int i) { return "model_layers_" + std::to_string(i); };

        for (int il = 0; il < N_LAYERS; il++) {
            auto& l = layers_[il];
            std::string lp = L(il) + "_";
            l.nw   = load_bin(W + lp + "input_layernorm_weight.bin");
            l.wq   = load_bin(W + lp + "self_attn_qkv_proj_q_proj_weight.bin");
            l.wk   = load_bin(W + lp + "self_attn_qkv_proj_k_proj_weight.bin");
            l.wv1  = load_bin(W + lp + "self_attn_qkv_proj_v_proj_current_weight.bin");
            l.wv2  = load_bin(W + lp + "self_attn_qkv_proj_v_proj_delayed_weight.bin");
            l.wo   = load_bin(W + lp + "self_attn_o_proj_weight.bin");
            l.pan  = load_bin(W + lp + "post_attention_layernorm_weight.bin");
            l.gu   = load_bin(W + lp + "mlp_experts_gate_up_proj.bin");
            l.dn   = load_bin(W + lp + "mlp_experts_down_proj.bin");
            if (!l.nw.empty() && il == 0) {
                fprintf(stderr, "  Vulkan: layer 0 nw=%zu wq=%zu\n", l.nw.size(), l.wq.size());
            }
        }

        // Allocate scratch buffers (sized for one row of GEMM output)
        size_t max_dim = std::max({(size_t)H, (size_t)(cfg.num_heads * cfg.head_dim),
                                    (size_t)(cfg.num_kv_heads * cfg.head_dim),
                                    (size_t)(cfg.intermediate_size * 2), (size_t)VOCAB});
        buf_hs_.create(ctx_.dev, ctx_.memProps, H * sizeof(float),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        buf_tmp_.create(ctx_.dev, ctx_.memProps, max_dim * sizeof(float),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        buf_out_.create(ctx_.dev, ctx_.memProps, max_dim * sizeof(float),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        buf_wt_.create(ctx_.dev, ctx_.memProps, max_dim * H * sizeof(float),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        scratch_ok_ = true;

        // Create pipelines
        auto spv_gemm = shader_dir_ + "matmul_fp32.comp";
        auto spv_gemv = shader_dir_ + "dmmv_tq2_bonsai.comp";

        // Use matmul_fp32 for all GEMM ops
        pipe_gemm_.create(ctx_, spv_gemm.c_str(), 3, 20);  // 3 bindings, 20B push constants
        // Use dmmv_tq2_bonsai for GEMV (1 row × K)
        try {
            pipe_gemv_.create(ctx_, spv_gemv.c_str(), 3, 24);  // 3 bindings, 24B push constants
        } catch (...) {
            fprintf(stderr, "  Vulkan: dmmv_tq2_bonsai.comp not found, using matmul for GEMV too\n");
        }

        loaded_ = true;
        return true;
    }

    void unload_model() override {
        if (scratch_ok_) {
            buf_hs_.destroy();
            buf_tmp_.destroy();
            buf_out_.destroy();
            buf_wt_.destroy();
            scratch_ok_ = false;
        }
        if (pipe_gemm_.pipeline != VK_NULL_HANDLE) pipe_gemm_.destroy(ctx_.dev);
        if (pipe_gemv_.pipeline != VK_NULL_HANDLE) pipe_gemv_.destroy(ctx_.dev);
        layers_.clear();
        embed_.clear();
        fnorm_.clear();
        k_cache_.clear();
        v_cache_.clear();
        loaded_ = false;
    }

    void reset_state() override {
        k_cache_.clear();
        v_cache_.clear();
    }

    // ─── Vulkan GEMM: out[M] = in[K] @ wt[M×K]^T ──────────────────
    void vk_gemm(vkrt::Pipeline& pipe, vkrt::GpuBuffer& buf_in, vkrt::GpuBuffer& buf_wt,
                 vkrt::GpuBuffer& buf_out, int M, int K, bool accum = false) {
        uint32_t pc[5] = {(uint32_t)M, (uint32_t)K, 0u, 0u, (uint32_t)(accum ? 1u : 0u)};

        // For GEMV (M=1), use pipe_gemv_ if available
        vkrt::Pipeline* p = &pipe;
        if (M <= 2 && pipe_gemv_.pipeline != VK_NULL_HANDLE) {
            p = &pipe_gemv_;
        }

        vkrt::GpuBuffer* bufs[3] = {&buf_wt, &buf_in, &buf_out};
        auto ds = createDescriptorSet(ctx_, *p, bufs, 3);

        uint32_t gx = (M + 1) / 2;  // 2 rows per workgroup for GEMV
        if (M > 2) gx = (M + 63) / 64;  // 64 rows per workgroup for GEMM

        dispatchOnce(ctx_, *p, ds, gx, 1, 1, pc);
    }

    // ─── CPU-assisted Vulkan forward pass ──────────────────────────
    // rotr = rotary position embedding (simplified: cos/sin precomputed)
    static void apply_rope(float* q, float* k, int nq, int nkv, int hd, int pos) {
        for (int h = 0; h < nq; h++) {
            float* qh = q + h * hd;
            for (int i = 0; i < hd; i += 2) {
                float theta = (float)pos / powf(10000.0f, (float)i / (float)hd);
                float cos_t = cosf(theta);
                float sin_t = sinf(theta);
                float q0 = qh[i], q1 = qh[i+1];
                qh[i]   = q0 * cos_t - q1 * sin_t;
                qh[i+1] = q0 * sin_t + q1 * cos_t;
            }
        }
        for (int h = 0; h < nkv; h++) {
            float* kh = k + h * hd;
            for (int i = 0; i < hd; i += 2) {
                float theta = (float)pos / powf(10000.0f, (float)i / (float)hd);
                float cos_t = cosf(theta);
                float sin_t = sinf(theta);
                float k0 = kh[i], k1 = kh[i+1];
                kh[i]   = k0 * cos_t - k1 * sin_t;
                kh[i+1] = k0 * sin_t + k1 * cos_t;
            }
        }
    }

    // ─── Scaled dot-product attention (CPU, with KV cache) ─────────
    static void attend_cpu_cached(
        const float* q, const float* k_cache, const float* v_cache,
        float* out, int nq, int nkv, int hd, int seq_len
    ) {
        // Multi-head attention with GQA: nq query heads, nkv KV heads
        // k_cache/v_cache: [nkv * hd * seq_len] — K,V for all past positions
        float scale = 1.0f / sqrtf((float)hd);
        for (int h = 0; h < nq; h++) {
            int kv_h = h * nkv / nq;
            const float* qh = q + h * hd;
            float* oh = out + h * hd;

            // Compute scores against all cached positions
            std::vector<float> scores(seq_len);
            float max_score = -1e30f;
            for (int t = 0; t < seq_len; t++) {
                const float* kh = k_cache + (kv_h * seq_len + t) * hd;
                float s = 0;
                for (int i = 0; i < hd; i++) s += qh[i] * kh[i];
                s *= scale;
                scores[t] = s;
                if (s > max_score) max_score = s;
            }

            // Softmax over sequence
            float sum = 0;
            for (int t = 0; t < seq_len; t++) {
                scores[t] = expf(scores[t] - max_score);
                sum += scores[t];
            }
            float inv_sum = 1.0f / (sum + 1e-10f);

            // Weighted sum of V
            memset(oh, 0, hd * sizeof(float));
            for (int t = 0; t < seq_len; t++) {
                float a = scores[t] * inv_sum;
                const float* vh = v_cache + (kv_h * seq_len + t) * hd;
                for (int i = 0; i < hd; i++) oh[i] += a * vh[i];
            }
        }
    }

    int forward(int token_id, int pos) override {
        if (!loaded_) return 0;
        int H = cfg_.hidden_size;
        int NQ = cfg_.num_heads, NKV = cfg_.num_kv_heads, HD = cfg_.head_dim;
        int QD = NQ * HD, KD = NKV * HD;
        int N_LAYERS = cfg_.num_layers, VOCAB = cfg_.vocab_size;
        int N_FF = cfg_.intermediate_size;

        // Grow KV cache if needed
        if ((int)k_cache_.size() < N_LAYERS) {
            k_cache_.resize(N_LAYERS);
            v_cache_.resize(N_LAYERS);
        }
        int cache_stride = NKV * HD;

        // Embedding lookup
        std::vector<float> hidden(H);
        if (!embed_.empty() && embed_.size() >= (size_t)(token_id + 1) * H) {
            for (int i = 0; i < H; i++)
                hidden[i] = embed_[(size_t)token_id * H + i];
        }

        for (int il = 0; il < N_LAYERS; il++) {
            auto& l = layers_[il];
            std::vector<float> residual = hidden;

            // ── Pre-attention RMSNorm ──
            if (!l.nw.empty()) rmsnorm_cpu(hidden, l.nw, H, cfg_.rms_norm_eps);

            // ── Q, K, V projections via Vulkan GEMM ──
            std::vector<float> q(QD), k(KD), v(KD);

            if (l.wq.size() >= (size_t)QD * H) {
                buf_wt_.upload(l.wq.data());
                buf_hs_.upload(hidden.data());
                vk_gemm(pipe_gemm_, buf_hs_, buf_wt_, buf_out_, QD, H);
                buf_out_.download(q.data());
            }
            if (l.wk.size() >= (size_t)KD * H) {
                buf_wt_.upload(l.wk.data());
                buf_hs_.upload(hidden.data());
                vk_gemm(pipe_gemm_, buf_hs_, buf_wt_, buf_out_, KD, H);
                buf_out_.download(k.data());
            }
            if (l.wv1.size() >= (size_t)KD * H) {
                buf_wt_.upload(l.wv1.data());
                buf_hs_.upload(hidden.data());
                vk_gemm(pipe_gemm_, buf_hs_, buf_wt_, buf_out_, KD, H);
                buf_out_.download(v.data());
            }

            // ── RoPE ──
            apply_rope(q.data(), k.data(), NQ, NKV, HD, pos);

            // ── Store K,V in cache ──
            auto& kc = k_cache_[il];
            auto& vc = v_cache_[il];
            int seq = pos + 1;
            if ((int)kc.size() < seq * cache_stride) {
                kc.resize(seq * cache_stride);
                vc.resize(seq * cache_stride);
            }
            for (int h = 0; h < NKV; h++) {
                for (int i = 0; i < HD; i++) {
                    kc[(h * seq + pos) * HD + i] = k[h * HD + i];
                    vc[(h * seq + pos) * HD + i] = v[h * HD + i];
                }
            }

            // ── Attention with full KV cache ──
            std::vector<float> attn_out(QD);
            attend_cpu_cached(q.data(), kc.data(), vc.data(), attn_out.data(),
                              NQ, NKV, HD, seq);

            // ── Output projection Wo ──
            if (!l.wo.empty() && l.wo.size() >= (size_t)H * QD) {
                buf_wt_.upload(l.wo.data());
                buf_hs_.upload(attn_out.data());
                vk_gemm(pipe_gemm_, buf_hs_, buf_wt_, buf_out_, H, QD);
                buf_out_.download(hidden.data());
            } else {
                for (int i = 0; i < H && i < QD; i++) hidden[i] = attn_out[i];
                for (int i = QD; i < H; i++) hidden[i] = 0;
            }

            // ── Residual ──
            for (int i = 0; i < H; i++) hidden[i] += residual[i];
            std::vector<float> attn_residual = hidden;
            if (!l.pan.empty()) rmsnorm_cpu(hidden, l.pan, H, cfg_.rms_norm_eps);

            // ── FFN with SiLU ──
            if (!l.g.empty() && !l.up.empty() && l.g.size() >= (size_t)N_FF * H) {
                buf_wt_.upload(l.g.data());
                buf_hs_.upload(hidden.data());
                vk_gemm(pipe_gemm_, buf_hs_, buf_wt_, buf_out_, N_FF, H);
                std::vector<float> gate(N_FF);
                buf_out_.download(gate.data());

                buf_wt_.upload(l.up.data());
                buf_hs_.upload(hidden.data());
                vk_gemm(pipe_gemm_, buf_hs_, buf_wt_, buf_out_, N_FF, H);
                std::vector<float> up(N_FF);
                buf_out_.download(up.data());

                std::vector<float> ffn_act(N_FF);
                for (int i = 0; i < N_FF; i++) {
                    float g = gate[i];
                    float silu = g / (1.0f + expf(-g));
                    ffn_act[i] = silu * up[i];
                }

                if (!l.dn.empty() && l.dn.size() >= (size_t)H * N_FF) {
                    buf_wt_.upload(l.dn.data());
                    buf_tmp_.upload(ffn_act.data());
                    vk_gemm(pipe_gemm_, buf_tmp_, buf_wt_, buf_out_, H, N_FF);
                    buf_out_.download(hidden.data());
                }
            } else if (!l.gu.empty() && l.gu.size() >= (size_t)(2 * N_FF) * H) {
                int gus = (int)(l.gu.size() / H);
                buf_wt_.upload(l.gu.data());
                buf_hs_.upload(hidden.data());
                vk_gemm(pipe_gemm_, buf_hs_, buf_wt_, buf_out_, gus, H);
                std::vector<float> gate_up(gus);
                buf_out_.download(gate_up.data());

                int nf = gus / 2;
                std::vector<float> ffn_act(nf);
                for (int i = 0; i < nf; i++) {
                    float g = gate_up[i];
                    float silu = g / (1.0f + expf(-g));
                    ffn_act[i] = silu * gate_up[i + nf];
                }

                if (!l.dn.empty() && l.dn.size() >= (size_t)H * nf) {
                    buf_wt_.upload(l.dn.data());
                    buf_tmp_.upload(ffn_act.data());
                    vk_gemm(pipe_gemm_, buf_tmp_, buf_wt_, buf_out_, H, nf);
                    buf_out_.download(hidden.data());
                }
            }

            // ── FFN residual ──
            for (int i = 0; i < H; i++) hidden[i] += attn_residual[i];
        }

        // ── Final RMSNorm ──
        if (!fnorm_.empty()) rmsnorm_cpu(hidden, fnorm_, H, cfg_.rms_norm_eps);

        // ── lm_head: argmax via chunked Vulkan GEMM ──
        float best_val = -1e30f;
        int best_idx = 0;

        if (embed_.size() >= (size_t)VOCAB * H) {
            const int CHUNK = 4096;
            for (int vstart = 0; vstart < VOCAB; vstart += CHUNK) {
                int vend = std::min(vstart + CHUNK, VOCAB);
                int chunk_size = vend - vstart;

                buf_wt_.upload(embed_.data() + (size_t)vstart * H);
                buf_hs_.upload(hidden.data());
                vk_gemm(pipe_gemm_, buf_hs_, buf_wt_, buf_out_, chunk_size, H);

                std::vector<float> scores(chunk_size);
                buf_out_.download(scores.data());

                for (int i = 0; i < chunk_size; i++) {
                    if (scores[i] > best_val) {
                        best_val = scores[i];
                        best_idx = vstart + i;
                    }
                }
            }
        } else {
            for (int v = 0; v < VOCAB; v++) {
                float dot = 0;
                const float* emb_row = embed_.data() + (size_t)v * H;
                for (int i = 0; i < H; i++) dot += hidden[i] * emb_row[i];
                if (dot > best_val) { best_val = dot; best_idx = v; }
            }
        }

        return best_idx;
    }
};

// ─── 1BP model weights loader (pulled in as unity build) ──────────
// Note: onebp_loader.cpp defines its own OnebpModel class with open() and
// get_tensor_f32(). Do NOT also include onebp_loader.h (struct version).
#include "../../engine/npu/src/onebp_loader.cpp"

// ─── Load weights from a 1BP file into a VulkanBackend ────────────
// Defined here (after OnebpModel is available) so VulkanBackend can call it.
// Forward-declared inside the class as load_1bp_vulkan().
static bool load_1bp_vulkan(
    std::vector<float>& embed_, std::vector<float>& fnorm_,
    std::vector<CpuLayer>& layers_, ModelConfig& cfg_,
    const std::string& path)
{
    fprintf(stderr, "  Vulkan: loading 1BP: %s\n", path.c_str());
    OnebpModel mdl;
    if (!mdl.open(path.c_str())) {
        fprintf(stderr, "  Vulkan: failed to open 1BP\n");
        return false;
    }
    auto& h = mdl.header();
    int H = h.hidden_size;
    int N_LAYERS = h.num_layers;
    int NH = h.num_attention_heads;
    int NKV = h.num_kv_heads ? h.num_kv_heads : NH;
    int HD = h.head_dim ? h.head_dim : 128;
    int IM = h.intermediate_size;
    int VOCAB = h.vocab_size;

    cfg_.hidden_size = cfg_.hidden = H;
    cfg_.num_layers = cfg_.n_layers = N_LAYERS;
    cfg_.num_heads = cfg_.n_heads = cfg_.num_attention_heads = NH;
    cfg_.num_kv_heads = cfg_.n_kv_heads = NKV;
    cfg_.head_dim = HD;
    cfg_.intermediate_size = cfg_.n_ff = IM;
    cfg_.vocab_size = cfg_.vocab = VOCAB;

    auto ld = [&](const char* n, std::vector<float>& v) {
        return mdl.get_tensor_f32(n, v);
    };
    ld("token_embd.weight", embed_);
    if (!ld("output_norm.weight", fnorm_))
        ld("token_embd_norm.weight", fnorm_);
    if (embed_.empty()) {
        fprintf(stderr, "  Vulkan: 1BP missing token_embd.weight\n");
        return false;
    }

    layers_.resize(N_LAYERS);
    char buf[128];
    for (int l = 0; l < N_LAYERS; l++) {
        auto& cl = layers_[l];
        auto ldW = [&](const char* bk, std::vector<float>& dst) {
            snprintf(buf, sizeof(buf), "blk.%d.%s", l, bk);
            std::vector<float> w;
            if (!mdl.get_tensor_f32(buf, w)) {
                snprintf(buf, sizeof(buf), "model.layers.%d.%s", l, bk);
                mdl.get_tensor_f32(buf, w);
            }
            dst = std::move(w);
        };
        ldW("attn_q.weight", cl.wq);
        ldW("attn_k.weight", cl.wk);
        ldW("attn_v.weight", cl.wv1);
        ldW("attn_output.weight", cl.wo);
        ldW("ffn_gate.weight", cl.g);
        ldW("ffn_up.weight", cl.up);
        ldW("ffn_down.weight", cl.dn);
        // Stack gate + up into gu for single GEMM (layout: [gate | up])
        if (!cl.g.empty() && !cl.up.empty() && cl.g.size() == cl.up.size()) {
            cl.gu.resize(cl.g.size() + cl.up.size());
            memcpy(cl.gu.data(), cl.g.data(), cl.g.size() * sizeof(float));
            memcpy(cl.gu.data() + cl.g.size(), cl.up.data(), cl.up.size() * sizeof(float));
        }
        ldW("attn_norm.weight", cl.nw);
        ldW("ffn_norm.weight", cl.pan);
    }
    fprintf(stderr, "  Vulkan: 1BP loaded — %d layers, %.0fM params\n",
            N_LAYERS, (double)embed_.size() / 1e6);
    return true;
}

// ─── Factory ────────────────────────────────────────────────────────
std::vector<InferenceBackend*> detect_backends_vulkan() {
    std::vector<InferenceBackend*> backends;
    static VulkanBackend vk;
    backends.push_back(&vk);
    return backends;
}
