/**
 * zamba2_npu.cpp — Zamba2 NPU+CPU Hybrid Inference
 *
 * Phase 1: SSM layers on CPU via mamba2_cpu_forward(), attention layers on NPU.
 * Phase 2+ targets: NPU GEMM for in_proj/out_proj, custom AIE SSM kernel.
 *
 * Build:
 *   g++ -std=c++23 -O3 -march=native -o zamba2_npu \
 *       tools/zamba2_npu.cpp \
 *       -I include -I src -I engine/npu/include \
 *       -lpthread -fopenmp
 *
 * Run:
 *   ./zamba2_npu models/Zamba2-2.7B-Instruct-v2.1bp
 *
 * Architecture (Zamba2-2.7B):
 *   54 layers: 45 Mamba2 SSM + 9 hybrid attention (every 6th: 6,12,18,24,30,36,42,47,51)
 *
 * Mamba2 SSM layer:  in_proj → conv1d → selective_scan → group_norm → gate → out_proj
 * Hybrid attn layer: QKV → RoPE → attn → O → residual → FFN(GU+D)
 */

#include "mamba2_kernels.h"
#include "onebp_loader.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <thread>
#include <omp.h>

// ─── Zamba2-2.7B config ─────────────────────────────────────────────
static constexpr int H = 2560;           // d_model
static constexpr int D_INNER = 5120;     // d_inner (expansion_factor * d_model)
static constexpr int D_STATE = 64;       // SSM state size
static constexpr int D_CONV = 4;         // conv1d kernel size
static constexpr int N_HEADS = 80;       // SSM heads (d_inner / head_dim)
static constexpr int HEAD_DIM = 64;      // SSM head dimension
static constexpr int N_GROUP = 1;        // SSM groups (shared B/C)
static constexpr int CONV_DIM = D_INNER + 2 * N_GROUP * D_STATE; // 5248
static constexpr int D_IN_PROJ = D_INNER + CONV_DIM + N_HEADS;   // 10384 + 80
static constexpr int D_OUT_PROJ = D_INNER;

// Attention config
static constexpr int N_ATTN_HEADS = 32;
static constexpr int N_KV_HEADS = 32;
static constexpr int ATTN_HD = 80;       // attention head_dim
static constexpr int QKV_N = N_ATTN_HEADS * ATTN_HD + 2 * N_KV_HEADS * ATTN_HD; // 7680

// Model structure
static constexpr int N_LAYERS = 54;
// Indices of hybrid attention layers
static constexpr int HYBRID_LAYERS[] = {6, 12, 18, 24, 30, 36, 42, 47, 51};
static constexpr int N_HYBRID = 9;

// Tokenizer (minimal: BOS=151644, EOS=151645, just for smoke test)
static constexpr int BOS_ID = 151644;
static constexpr int EOS_ID = 151645;
static constexpr int VOCAB = 262272;  // Zamba2 vocab

// ─── Silu activation ────────────────────────────────────────────────
static inline float silu(float x) {
    return x / (1.0f + std::expf(-x));
}

// ─── RMS norm ───────────────────────────────────────────────────────
static void rms_norm(float* x, const float* w, int n, float eps = 1e-6f) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float ir = 1.0f / std::sqrt((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) x[i] = x[i] * ir * w[i];
}

// ─── RoPE (for attention layers) ────────────────────────────────────
static std::vector<float> rope_cache;
static void build_rope(int max_pos, int hd, float theta = 500000.0f) {
    rope_cache.resize(max_pos * hd);
    for (int p = 0; p < max_pos; p++)
        for (int d = 0; d < hd / 2; d++) {
            float f = 1.0f / std::pow(theta, (float)d / (hd / 2));
            rope_cache[p * hd + d] = std::cos(p * f);
            rope_cache[p * hd + hd / 2 + d] = std::sin(p * f);
        }
}
static void apply_rope(float* x, int hd, int pos) {
    for (int d = 0; d < hd / 2; d++) {
        float a = x[d], b = x[d + hd / 2];
        float c = rope_cache[pos * hd + d];
        float s = rope_cache[pos * hd + hd / 2 + d];
        x[d] = a * c - b * s;
        x[d + hd / 2] = b * c + a * s;
    }
}

// ─── Attention (CPU, for non-NPU path) ──────────────────────────────
static void attention_cpu(
    const float* q, const float* k_cache, const float* v_cache,
    float* out, int seq_len, int nh, int nkv, int hd, int gqa, int pos)
{
    const int n_groups = nh / gqa;
    std::vector<float> scores(seq_len);
    for (int h = 0; h < n_groups; h++) {
        for (int g = 0; g < gqa; g++) {
            int hi = h * gqa + g;
            const float* qh = q + hi * hd;
            float* oh = out + hi * hd;
            // Scores
            float max_score = -1e30f;
            for (int t = 0; t < seq_len; t++) {
                float s = 0;
                const float* kh = k_cache + (size_t)t * nkv * hd + h * hd;
                for (int d = 0; d < hd; d++) s += qh[d] * kh[d];
                scores[t] = s * (1.0f / std::sqrt((float)hd));
                if (scores[t] > max_score) max_score = scores[t];
            }
            // Softmax
            float sum = 0;
            for (int t = 0; t < seq_len; t++) {
                scores[t] = std::exp(scores[t] - max_score);
                sum += scores[t];
            }
            float inv_sum = 1.0f / sum;
            // Weighted sum
            for (int d = 0; d < hd; d++) oh[d] = 0;
            for (int t = 0; t < seq_len; t++) {
                const float* vh = v_cache + (size_t)t * nkv * hd + h * hd;
                float w = scores[t] * inv_sum;
                for (int d = 0; d < hd; d++) oh[d] += w * vh[d];
            }
        }
    }
}

// ─── Zamba2 Model Weights (loaded from 1BP) ────────────────────────
struct Zamba2Weights {
    // Embedding + output
    std::vector<float> embed;     // [VOCAB, H]
    std::vector<float> lm_head;   // [VOCAB, H]
    std::vector<float> final_norm;// [H]

    // Per-layer SSM weights (45 layers)
    struct SSMLayer {
        std::vector<float> in_proj_w;   // [D_IN_PROJ, H]
        std::vector<float> conv1d_w;    // [D_CONV, CONV_DIM]
        std::vector<float> conv1d_b;    // [CONV_DIM]
        std::vector<float> dt_bias;     // [N_HEADS]
        std::vector<float> A_log;       // [N_HEADS]
        std::vector<float> D;           // [N_HEADS]
        std::vector<float> norm_w;      // [D_INNER / N_GROUP, N_GROUP] = [5120, 1]
        std::vector<float> out_proj_w;  // [H, D_INNER]
    };
    SSMLayer ssm_layers[N_LAYERS]; // all layers, attn layers leave some empty

    // Per-layer attention weights (9 layers)
    struct AttnLayer {
        std::vector<float> qkv_w;     // [QKV_N, H]
        std::vector<float> o_w;       // [H, N_ATTN_HEADS*ATTN_HD]
        std::vector<float> gate_w;    // [D_INNER, H]  — FFN gate (G)
        std::vector<float> up_w;      // [D_INNER, H]  — FFN up (U)
        std::vector<float> down_w;    // [H, D_INNER]  — FFN down (D)
        std::vector<float> attn_norm_w; // [H] — pre-attention RMS norm
        std::vector<float> ffn_norm_w;  // [H] — pre-FFN RMS norm
        float q_norm_w[ATTN_HD];       // Q norm (per dimension)
        float k_norm_w[ATTN_HD];       // K norm
    };
    AttnLayer attn_layers[N_LAYERS]; // only hybrid layers are filled

    bool load_from_1bp(const char* path) {
        OnebpModel m;
        if (!m.load(path)) {
            fprintf(stderr, "Failed to load 1BP model: %s\n", path);
            return false;
        }
        fprintf(stderr, "Loading Zamba2 from %s (%d tensors)\n", path, m.header.tensor_count);
        
        // Index tensors by name
        auto get_tensor = [&](const std::string& name) -> const float* {
            for (auto& t : m.tensors) {
                if (t.name == name) {
                    return (const float*)m.tensor_data(t);
                }
            }
            fprintf(stderr, "  WARNING: tensor '%s' not found\n", name.c_str());
            return nullptr;
        };

        // Load embedding
        if (auto* p = get_tensor("token_embd.weight")) {
            embed.assign(p, p + VOCAB * H);
            fprintf(stderr, "  embed: %d x %d\n", VOCAB, H);
        }

        // Load final norm
        if (auto* p = get_tensor("output_norm.weight")) {
            final_norm.assign(p, p + H);
        } else if (auto* p = get_tensor("token_embd_norm.weight")) {
            final_norm.assign(p, p + H);
        }

        // Load LM head (tied or separate)
        if (auto* p = get_tensor("output.weight")) {
            lm_head.assign(p, p + VOCAB * H);
        } else {
            lm_head = embed; // tied embeddings
        }

        // Per-layer weights
        for (int l = 0; l < N_LAYERS; l++) {
            bool is_hybrid = false;
            for (int h : HYBRID_LAYERS) if (h == l) { is_hybrid = true; break; }

            if (is_hybrid) {
                // Attention layer
                auto& a = attn_layers[l];
                auto& s = ssm_layers[l];  // SSM weights also present for hybrid layers

                auto load = [&](std::vector<float>& v, const std::string& name, int n) {
                    auto* p = get_tensor(name);
                    if (p) { v.assign(p, p + n); return true; }
                    fprintf(stderr, "  L%d: missing %s (%d)\n", l, name.c_str(), n);
                    return false;
                };

                load(s.in_proj_w,  "blk." + std::to_string(l) + ".ssm_in_proj.weight", D_IN_PROJ * H);
                load(s.conv1d_w,   "blk." + std::to_string(l) + ".conv1d.weight", D_CONV * CONV_DIM);
                load(s.conv1d_b,   "blk." + std::to_string(l) + ".conv1d.bias", CONV_DIM);
                load(s.dt_bias,    "blk." + std::to_string(l) + ".dt_bias.weight", N_HEADS);
                load(s.A_log,      "blk." + std::to_string(l) + ".A_log.weight", N_HEADS);
                load(s.D,          "blk." + std::to_string(l) + ".D.weight", N_HEADS);
                load(s.norm_w,     "blk." + std::to_string(l) + ".ssm_norm.weight", D_INNER);
                load(s.out_proj_w, "blk." + std::to_string(l) + ".ssm_out_proj.weight", H * D_INNER);

                load(a.qkv_w,      "blk." + std::to_string(l) + ".attn_qkv.weight", QKV_N * H);
                load(a.o_w,        "blk." + std::to_string(l) + ".attn_o.weight", H * N_ATTN_HEADS * ATTN_HD);
                load(a.attn_norm_w,"blk." + std::to_string(l) + ".attn_norm.weight", H);
                load(a.ffn_norm_w, "blk." + std::to_string(l) + ".ffn_norm.weight", H);
                load(a.gate_w,     "blk." + std::to_string(l) + ".ffn_gate.weight", D_INNER * H);
                load(a.up_w,       "blk." + std::to_string(l) + ".ffn_up.weight", D_INNER * H);
                load(a.down_w,     "blk." + std::to_string(l) + ".ffn_down.weight", H * D_INNER);

                fprintf(stderr, "  L%d: hybrid (SSM+attn) loaded\n", l);
            } else {
                // Pure SSM layer
                auto& s = ssm_layers[l];
                auto load = [&](std::vector<float>& v, const std::string& base, int n) {
                    std::string name = "blk." + std::to_string(l) + "." + base;
                    auto* p = get_tensor(name);
                    if (p) { v.assign(p, p + n); return true; }
                    fprintf(stderr, "  L%d: missing %s (%d)\n", l, name.c_str(), n);
                    return false;
                };

                load(s.in_proj_w,  "ssm_in_proj.weight", D_IN_PROJ * H);
                load(s.conv1d_w,   "conv1d.weight", D_CONV * CONV_DIM);
                load(s.conv1d_b,   "conv1d.bias", CONV_DIM);
                load(s.dt_bias,    "dt_bias.weight", N_HEADS);
                load(s.A_log,      "A_log.weight", N_HEADS);
                load(s.D,          "D.weight", N_HEADS);
                load(s.norm_w,     "ssm_norm.weight", D_INNER);
                load(s.out_proj_w, "ssm_out_proj.weight", H * D_INNER);
            }
        }

        fprintf(stderr, "Model loaded: %d layers\n", N_LAYERS);
        return true;
    }
};

// ─── GEMM helper (CPU) ──────────────────────────────────────────────
static void gemm_cpu(float* out, const float* w, const float* in,
                     int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += w[n * K + k] * in[m * K + k];
            out[m * N + n] = sum;
        }
}

// ─── Forward pass through one SSM layer ─────────────────────────────
static void forward_ssm_layer(
    float* h,                           // [H] — input/output
    const Zamba2Weights::SSMLayer& w,
    float* conv_state,                  // [(D_CONV-1) * CONV_DIM]
    float* ssm_state,                   // [N_HEADS * D_STATE]
    const Mamba2Config& cfg,
    int pos)                            // token position (for conv state)
{
    (void)pos;
    mamba2_cpu_forward(
        h,                      // input
        w.in_proj_w.data(),     // in_proj weights
        w.conv1d_w.data(),      // conv1d weights
        w.conv1d_b.data(),      // conv1d bias
        w.dt_bias.data(),       // dt bias
        w.A_log.data(),         // A_log
        w.D.data(),             // D
        w.norm_w.data(),        // norm weights (reshaped for group norm)
        w.out_proj_w.data(),    // out_proj weights
        conv_state,             // conv state (updated in-place)
        ssm_state,              // SSM state (updated in-place)
        h,                      // output (same buffer as input)
        cfg                     // model config
    );
}

// ─── Forward pass through one hybrid attention layer ────────────────
static void forward_hybrid_layer(
    float* h,                           // [H] — input/output
    const Zamba2Weights::AttnLayer& w_attn,
    const Zamba2Weights::SSMLayer& w_ssm,
    float* kv_cache_k,                  // [max_seq, NKV, ATTN_HD]
    float* kv_cache_v,                  // [max_seq, NKV, ATTN_HD]
    float* conv_state,
    float* ssm_state,
    const Mamba2Config& cfg,
    int pos, int seq_len)
{
    // ── SSM path (hybrid layers have both SSM and attn) ──
    // In Zamba2, hybrid layers run BOTH SSM and attention:
    // output = SSM(h) + attention(h)
    
    float h_copy[H];
    memcpy(h_copy, h, H * sizeof(float));
    
    // Run SSM on copy
    forward_ssm_layer(h_copy, w_ssm, conv_state, ssm_state, cfg, pos);
    
    // ── Attention path ──
    // QKV projection
    float qkv_buf[QKV_N];
    gemm_cpu(qkv_buf, w_attn.qkv_w.data(), h, 1, QKV_N, H);
    
    // Split Q, K, V
    int q_dim = N_ATTN_HEADS * ATTN_HD; // 2560
    float* q = qkv_buf;
    float* k = qkv_buf + q_dim;          // NKV * ATTN_HD
    float* v = qkv_buf + q_dim + N_KV_HEADS * ATTN_HD;
    
    // Apply Q/K norm + RoPE
    for (int hh = 0; hh < N_ATTN_HEADS; hh++) {
        float* qh = q + hh * ATTN_HD;
        double sq = 0;
        for (int d = 0; d < ATTN_HD; d++) sq += qh[d] * qh[d];
        float iq = 1.0f / std::sqrt(sq / ATTN_HD + 1e-6f);
        for (int d = 0; d < ATTN_HD; d++) qh[d] *= iq * w_attn.q_norm_w[d];
        apply_rope(qh, ATTN_HD, pos);
    }
    for (int kvh = 0; kvh < N_KV_HEADS; kvh++) {
        float* kh = k + kvh * ATTN_HD;
        double sk = 0;
        for (int d = 0; d < ATTN_HD; d++) sk += kh[d] * kh[d];
        float ik = 1.0f / std::sqrt(sk / ATTN_HD + 1e-6f);
        for (int d = 0; d < ATTN_HD; d++) kh[d] *= ik * w_attn.k_norm_w[d];
        apply_rope(kh, ATTN_HD, pos);
        // Store in KV cache
        memcpy(kv_cache_k + (size_t)pos * N_KV_HEADS * ATTN_HD + kvh * ATTN_HD, kh, ATTN_HD * 4);
        memcpy(kv_cache_v + (size_t)pos * N_KV_HEADS * ATTN_HD + kvh * ATTN_HD, v + kvh * ATTN_HD, ATTN_HD * 4);
    }
    
    // Attention (CPU)
    float attn_out[N_ATTN_HEADS * ATTN_HD];
    attention_cpu(q, kv_cache_k, kv_cache_v, attn_out, seq_len,
                  N_ATTN_HEADS, N_KV_HEADS, ATTN_HD, N_ATTN_HEADS / N_KV_HEADS, pos);
    
    // O projection
    float o_out[H];
    gemm_cpu(o_out, w_attn.o_w.data(), attn_out, 1, H, N_ATTN_HEADS * ATTN_HD);
    
    // Residual add with SSM output
    for (int i = 0; i < H; i++) h[i] = h_copy[i] + o_out[i];
    
    // ── FFN (post attention) ──
    float ffn_in[H];
    memcpy(ffn_in, h, H * 4);
    rms_norm(ffn_in, w_attn.ffn_norm_w.data(), H);
    
    float gate_buf[D_INNER], up_buf[D_INNER];
    gemm_cpu(gate_buf, w_attn.gate_w.data(), ffn_in, 1, D_INNER, H);
    gemm_cpu(up_buf, w_attn.up_w.data(), ffn_in, 1, D_INNER, H);
    
    // SiLU gate
    for (int i = 0; i < D_INNER; i++) up_buf[i] = silu(gate_buf[i]) * up_buf[i];
    
    // Down projection
    float down_out[H];
    gemm_cpu(down_out, w_attn.down_w.data(), up_buf, 1, H, D_INNER);
    
    // Residual add
    for (int i = 0; i < H; i++) h[i] += down_out[i];
}

// ─── Main ───────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.1bp> [tokens]\n", argv[0]);
        return 1;
    }
    
    int n_tokens = (argc > 2) ? atoi(argv[2]) : 32;
    
    fprintf(stderr, "╔════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Zamba2 NPU+CPU Hybrid Inference       ║\n");
    fprintf(stderr, "╚════════════════════════════════════════╝\n\n");
    
    // Load model
    Zamba2Weights weights;
    if (!weights.load_from_1bp(argv[1])) return 1;
    
    // Mamba2 config
    Mamba2Config cfg;
    cfg.d_model = H;
    cfg.d_state = D_STATE;
    cfg.d_conv = D_CONV;
    cfg.d_inner = D_INNER;
    cfg.n_head = N_HEADS;
    cfg.n_group = N_GROUP;
    cfg.head_dim = HEAD_DIM;
    
    // KV cache for attention layers
    const int max_seq = 4096;
    std::vector<float> kv_cache_k((size_t)max_seq * N_KV_HEADS * ATTN_HD, 0);
    std::vector<float> kv_cache_v((size_t)max_seq * N_KV_HEADS * ATTN_HD, 0);
    
    // Per-layer conv and SSM state
    std::vector<std::vector<float>> conv_states(N_LAYERS);
    std::vector<std::vector<float>> ssm_states(N_LAYERS);
    for (int l = 0; l < N_LAYERS; l++) {
        conv_states[l].resize((D_CONV - 1) * CONV_DIM, 0);
        ssm_states[l].resize(N_HEADS * D_STATE, 0);
    }
    
    // RoPE cache
    build_rope(max_seq, ATTN_HD);
    
    // Embedding lookup (first token = BOS)
    std::vector<float> h(H, 0);
    memcpy(h.data(), weights.embed.data() + (size_t)BOS_ID * H, H * 4);
    
    // Benchmark
    auto t_start = std::chrono::steady_clock::now();
    int tokens_generated = 0;
    
    for (int pos = 0; pos < n_tokens; pos++) {
        auto t_layer_start = std::chrono::steady_clock::now();
        
        // Save input embedding for this position
        if (pos > 0) {
            memcpy(h.data(), weights.embed.data() + (size_t)0 * H, H * 4);  // placeholder
        }
        
        // Per-layer forward
        for (int l = 0; l < N_LAYERS; l++) {
            bool is_hybrid = false;
            for (int hh : HYBRID_LAYERS) if (hh == l) { is_hybrid = true; break; }
            
            auto t_op = std::chrono::steady_clock::now();
            
            if (is_hybrid) {
                // Hybrid: SSM + attention
                forward_hybrid_layer(
                    h.data(), weights.attn_layers[l], weights.ssm_layers[l],
                    kv_cache_k.data(), kv_cache_v.data(),
                    conv_states[l].data(), ssm_states[l].data(),
                    cfg, pos, pos + 1
                );
                fprintf(stderr, "  L%d [hybrid]  %.2fms\n", l,
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t_op).count());
            } else {
                // Pure SSM
                forward_ssm_layer(
                    h.data(), weights.ssm_layers[l],
                    conv_states[l].data(), ssm_states[l].data(),
                    cfg, pos
                );
                fprintf(stderr, "  L%d [ssm]    %.2fms\n", l,
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t_op).count());
            }
        }
        
        // Final norm
        rms_norm(h.data(), weights.final_norm.data(), H);
        
        // LM head (greedy: pick highest logit)
        float max_logit = -1e30f;
        int max_id = 0;
        for (int v = 0; v < VOCAB; v++) {
            float logit = 0;
            for (int i = 0; i < H; i++)
                logit += weights.lm_head[(size_t)v * H + i] * h[i];
            if (logit > max_logit) { max_logit = logit; max_id = v; }
        }
        
        tokens_generated++;
        fprintf(stderr, "  Token %d: id=%d (%.2fms total)\n", pos, max_id,
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_layer_start).count());
        
        // Prepare next token embedding
        if (max_id == EOS_ID) break;
        memcpy(h.data(), weights.embed.data() + (size_t)max_id * H, H * 4);
    }
    
    auto t_end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    fprintf(stderr, "\n╔════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Results                               ║\n");
    fprintf(stderr, "╚════════════════════════════════════════╝\n");
    fprintf(stderr, "  Tokens: %d\n", tokens_generated);
    fprintf(stderr, "  Time:   %.0f ms\n", ms);
    fprintf(stderr, "  Speed:  %.1f tok/s\n", tokens_generated / (ms / 1000.0f));
    fprintf(stderr, "  Layers: %d (%d SSM + %d hybrid)\n", N_LAYERS, N_LAYERS - N_HYBRID, N_HYBRID);
    fprintf(stderr, "  SSM layers: %.0f%% of compute\n",
        (float)(N_LAYERS - N_HYBRID) / N_LAYERS * 100);
    
    return 0;
}
