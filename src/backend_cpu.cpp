// backend_cpu.cpp — CPU scalar backend for Zaya1-8B
// Reference implementation. No GPU, no NPU, no special hardware.
// Uses the same weight format as GPU backends.

#include "backend.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ── Architecture constants ──
static constexpr int H = 2048;
static constexpr int NQ = 8;
static constexpr int NKV = 2;
static constexpr int HD = 128;
static constexpr int QD = NQ*HD;
static constexpr int KD = NKV*HD;
static constexpr int QKV = QD+KD;
static constexpr int GQA = 4;
static constexpr int NROT = 64;
static constexpr int N_EXP = 16;
static constexpr int N_FF = 2048;
static constexpr int RTR_H = 256;
static constexpr int VOCAB = 262272;
static constexpr int N_LAYERS = 40;
static constexpr float RMD_EPS = 1e-5f;
static constexpr float ROPE_BASE = 5000000.0f;

// ── SIMD / threading support ──
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

// ── Math helpers ──
static float gelu(float x) {
    float t = tanhf(0.79788456f * (x + 0.044715f * x * x * x));
    return 0.5f * x * (1.0f + t);
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

static void rmsnorm(float* x, const float* w, int n) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float r = 1.0f / sqrtf((float)(ss / n) + RMD_EPS);
    for (int i = 0; i < n; i++) x[i] = x[i] * r * w[i];
}

// ── Optimized matmul: out[M] = in[K] @ wt[M×K]^T ──
// Uses AVX-512 when available, OpenMP threading, cache blocking
#if defined(__AVX512F__)
static void matmul_t(float* out, const float* in, const float* wt, int M, int K) {
    // Cache-blocked: process K in chunks of 16 for vectorization
    constexpr int BK = 16; // 16 floats = 1 AVX-512 register
    #pragma omp parallel for if(M > 64)
    for (int i = 0; i < M; i++) {
        __m512 vacc = _mm512_setzero_ps();
        int k;
        for (k = 0; k + BK <= K; k += BK) {
            __m512 act = _mm512_loadu_ps(in + k);
            __m512 w = _mm512_loadu_ps(wt + i * (size_t)K + k);
            vacc = _mm512_fmadd_ps(act, w, vacc);
        }
        float s = _mm512_reduce_add_ps(vacc);
        for (; k < K; k++) s += in[k] * wt[i * (size_t)K + k];
        out[i] = s;
    }
}
#elif defined(__AVX2__)
static void matmul_t(float* out, const float* in, const float* wt, int M, int K) {
    #pragma omp parallel for if(M > 64)
    for (int i = 0; i < M; i++) {
        __m256 vacc = _mm256_setzero_ps();
        int k;
        for (k = 0; k + 8 <= K; k += 8) {
            __m256 act = _mm256_loadu_ps(in + k);
            __m256 w = _mm256_loadu_ps(wt + i * (size_t)K + k);
            vacc = _mm256_fmadd_ps(act, w, vacc);
        }
        float s = _mm256_reduce_ps(vacc); // horizontal add
        s += _mm256_cvtss_f32(_mm256_permute2f128_ps(vacc, vacc, 1));
        // Actually just reduce manually
        alignas(32) float tmp[8]; _mm256_store_ps(tmp, vacc);
        s += tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
        for (; k < K; k++) s += in[k] * wt[i * (size_t)K + k];
        out[i] = s;
    }
}
#else
// Scalar fallback
static void matmul_t(float* out, const float* in, const float* wt, int M, int K) {
    for (int i = 0; i < M; i++) {
        float s = 0;
        for (int k = 0; k < K; k++) s += in[k] * wt[i * (size_t)K + k];
        out[i] = s;
    }
}
#endif

// ── Batched matmul for lm_head: out[V×1] = in[H] @ wt[V×H]^T ──
// Parallelized over vocab with OpenMP, SIMD inner loop
static void matmul_lmhead(float* out, const float* in, const float* wt, int V, int H_) {
    #pragma omp parallel for schedule(static, 64) if(V > 1024)
    for (int v = 0; v < V; v++) {
        const float* row = wt + v * (size_t)H_;
        #if defined(__AVX512F__)
        __m512 vacc = _mm512_setzero_ps();
        int h;
        for (h = 0; h + 16 <= H_; h += 16) {
            __m512 a = _mm512_loadu_ps(in + h);
            __m512 w = _mm512_loadu_ps(row + h);
            vacc = _mm512_fmadd_ps(a, w, vacc);
        }
        alignas(64) float tmp[16];
        _mm512_store_ps(tmp, vacc);
        float s = 0; for (int i = 0; i < 16; i++) s += tmp[i];
        for (; h < H_; h++) s += in[h] * row[h];
        out[v] = s;
        #elif defined(__AVX2__)
        __m256 vacc0 = _mm256_setzero_ps(), vacc1 = _mm256_setzero_ps();
        int h;
        for (h = 0; h + 16 <= H_; h += 16) {
            vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(in + h), _mm256_loadu_ps(row + h), vacc0);
            vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(in + h + 8), _mm256_loadu_ps(row + h + 8), vacc1);
        }
        alignas(32) float t0[8], t1[8];
        _mm256_store_ps(t0, vacc0); _mm256_store_ps(t1, vacc1);
        float s = 0; for (int i = 0; i < 8; i++) s += t0[i] + t1[i];
        for (; h < H_; h++) s += in[h] * row[h];
        out[v] = s;
        #else
        float s = 0;
        for (int h = 0; h < H_; h++) s += in[h] * row[h];
        out[v] = s;
        #endif
    }
}

static float cosim(const float* a, const float* b, int n) {
    float d = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return d / (sqrtf(na) * sqrtf(nb) + 1e-12f);
}

// ── CCA Attention (CPU, single token) ──
static void cca_attn_cpu(float* h, const float* wq_in, const float* wk_in,
    const float* wv1_in, const float* wv2_in, const float* wo_in,
    const float* cdw, const float* cdb, const float* cgw, const float* cgb,
    const float* ks, const float* nw,
    float* prev_hs, float* conv_state, int pos) {

    float buf[H], q[QD], k[KD], v[KD];
    float sqk[QKV], dw0[QKV], dw1[QKV];

    // 1. RMSNorm
    memcpy(buf, h, H * sizeof(float));
    rmsnorm(buf, nw, H);

    // 2/3. Q, K projections
    matmul_t(q, buf, wq_in, QD, H);
    matmul_t(k, buf, wk_in, KD, H);

    // raw qk concat
    memcpy(sqk, q, QD * sizeof(float));
    memcpy(sqk + QD, k, KD * sizeof(float));

    // 4. V projections (current + delayed)
    matmul_t(v, buf, wv1_in, KD/2, H);
    matmul_t(&v[KD/2], prev_hs, wv2_in, KD/2, H);

    // 5. conv_qk: depthwise
    for (int c = 0; c < QKV; c++) {
        dw0[c] = cdw[c*2] * conv_state[c] + cdw[c*2+1] * conv_state[QKV+c] + cdb[c];
        dw1[c] = cdw[c*2] * conv_state[QKV+c] + cdw[c*2+1] * sqk[c] + cdb[c];
    }
    // grouped
    for (int oc = 0; oc < QKV; oc++) {
        int g = oc / 128, base = g * 128;
        float s = cgb[oc];
        for (int j = 0; j < 128; j++)
            s += cgw[oc*256 + j*2] * dw0[base+j] + cgw[oc*256 + j*2 + 1] * dw1[base+j];
        sqk[oc] = s;
    }

    // 6. qk_means blend
    for (int hh = 0; hh < NQ; hh++) {
        int kv = hh / GQA;
        for (int d = 0; d < HD; d++)
            sqk[hh*HD+d] = sqk[hh*HD+d] + 0.5f * q[hh*HD+d] + 0.5f * k[kv*HD+d];
    }
    for (int khv = 0; khv < NKV; khv++) {
        for (int d = 0; d < HD; d++) {
            float sm = 0;
            for (int g = 0; g < GQA; g++) sm += q[(khv*GQA+g)*HD+d];
            sqk[QD + khv*HD + d] = sqk[QD + khv*HD + d] + 0.5f * sm / GQA + 0.5f * k[khv*HD+d];
        }
    }

    // 7. L2 normalize to sqrt(HD); K scaled by per-head temp
    float shd = sqrtf((float)HD);
    for (int hh = 0; hh < NQ; hh++) {
        float s = 0; for (int d = 0; d < HD; d++) s += sqk[hh*HD+d] * sqk[hh*HD+d];
        float iv = shd / (sqrtf(s) + 1e-12f);
        for (int d = 0; d < HD; d++) sqk[hh*HD+d] *= iv;
    }
    for (int khv = 0; khv < NKV; khv++) {
        float s = 0; for (int d = 0; d < HD; d++) s += sqk[QD+khv*HD+d] * sqk[QD+khv*HD+d];
        float iv = shd * ks[khv] / (sqrtf(s) + 1e-12f);
        for (int d = 0; d < HD; d++) sqk[QD+khv*HD+d] *= iv;
    }

    // 8. Partial RoPE (first NROT dims)
    for (int hh = 0; hh < NQ; hh++) {
        float* base = sqk + hh * HD;
        for (int d = 0; d < NROT/2; d++) {
            float theta = pos * powf(ROPE_BASE, -2.0f * d / (float)NROT);
            float c = cosf(theta), s = sinf(theta);
            float xv = base[d], xw = base[d + NROT/2];
            base[d] = xv * c - xw * s;
            base[d + NROT/2] = xv * s + xw * c;
        }
    }
    for (int khv = 0; khv < NKV; khv++) {
        float* base = sqk + QD + khv * HD;
        for (int d = 0; d < NROT/2; d++) {
            float theta = pos * powf(ROPE_BASE, -2.0f * d / (float)NROT);
            float c = cosf(theta), s = sinf(theta);
            float xv = base[d], xw = base[d + NROT/2];
            base[d] = xv * c - xw * s;
            base[d + NROT/2] = xv * s + xw * c;
        }
    }

    // 9. Attention (single token: softmax over 1 key = 1, output = V)
    float attn_out[H];
    for (int hh = 0; hh < NQ; hh++) {
        int kv = hh / GQA;
        for (int d = 0; d < HD; d++) attn_out[hh*HD+d] = v[kv*HD+d];
    }

    // 10. Output projection
    float final_out[H];
    matmul_t(final_out, attn_out, wo_in, H, QD);
    memcpy(h, final_out, H * sizeof(float));

    // Update state
    memcpy(prev_hs, h, H * sizeof(float));
    memcpy(conv_state, conv_state + QKV, QKV * sizeof(float));
    memcpy(conv_state + QKV, sqk, QKV * sizeof(float));
}

// ── EDA Router + MoE (CPU) ──
static void eda_router_moe_cpu(float* h, const float* gate_down_w, const float* gate_down_b,
    const float* router_norm_w, const float* rf1, const float* rf1b,
    const float* rf2, const float* rf2b, const float* rout, const float* bb,
    const float* gu, const float* dn, float* prev_rs, float* moe_out) {

    float rs[RTR_H], tmp[RTR_H], scores[17];
    int top_i[2]; float top_v[2];

    // gate_down
    matmul_t(rs, h, gate_down_w, RTR_H, H);
    for (int i = 0; i < RTR_H; i++) rs[i] += gate_down_b[i];

    // EDA carry
    for (int i = 0; i < RTR_H; i++) rs[i] += prev_rs[i];

    // Save for next token
    memcpy(prev_rs, rs, RTR_H * sizeof(float));

    // RMSNorm
    double ss = 0; for (int i = 0; i < RTR_H; i++) ss += rs[i] * rs[i];
    float inv = 1.0f / sqrtf((float)(ss / RTR_H) + 1e-5f);
    for (int i = 0; i < RTR_H; i++) rs[i] = rs[i] * inv * router_norm_w[i];

    // fc1 + GELU
    matmul_t(tmp, rs, rf1, RTR_H, RTR_H);
    for (int i = 0; i < RTR_H; i++) tmp[i] += rf1b[i];
    for (int i = 0; i < RTR_H; i++) tmp[i] = gelu(tmp[i]);

    // fc2 + GELU
    matmul_t(rs, tmp, rf2, RTR_H, RTR_H);
    for (int i = 0; i < RTR_H; i++) rs[i] += rf2b[i];
    for (int i = 0; i < RTR_H; i++) rs[i] = gelu(rs[i]);

    // out_proj + balancing biases
    float probs[17];
    for (int i = 0; i < 17; i++) {
        float s = 0; for (int j = 0; j < RTR_H; j++) s += rs[j] * rout[i*RTR_H+j];
        probs[i] = s + bb[i];
    }

    // Softmax
    float mv = probs[0]; for (int i = 1; i < 17; i++) if (probs[i] > mv) mv = probs[i];
    float sum = 0; for (int i = 0; i < 17; i++) { probs[i] = expf(probs[i] - mv); sum += probs[i]; }
    for (int i = 0; i < 17; i++) probs[i] /= sum;

    // Top-1 (for CPU simplicity, use top-1 instead of top-2)
    int best = 0; for (int i = 1; i < 17; i++) if (probs[i] > probs[best]) best = i;
    float best_w = probs[best];

    // MoE expert FFN
    memset(moe_out, 0, H * sizeof(float));
    if (best < N_EXP && gu && dn) {
        // gate_up
        float gate_up[2*N_FF];
        matmul_t(gate_up, h, &gu[(size_t)best*2*N_FF*H], 2*N_FF, H);

        // SiLU(gate) * up
        float moe_buf[N_FF];
        for (int i = 0; i < N_FF; i++)
            moe_buf[i] = silu(gate_up[i]) * gate_up[N_FF+i];

        // down
        matmul_t(moe_out, moe_buf, &dn[(size_t)best*H*N_FF], H, N_FF);

        // Scale by expert weight
        for (int i = 0; i < H; i++) moe_out[i] *= best_w;
    } else {
        // MOD skip: output = input
        memcpy(moe_out, h, H * sizeof(float));
    }
}

// ── CPU Backend implementation ──
struct CPUBackend : Backend {
    // Weights (always in memory)
    float* embed = nullptr;
    float* fnorm = nullptr;
    float* iscale = nullptr;
    float* ibias = nullptr;
    float* lm_head_w = nullptr;

    // Per-layer weights (small ones in memory, big ones mmap'd)
    struct LayerW {
        // Small weights (< 1 MB each, kept in memory)
        float nw[H], wq[QD*H], wk[KD*H], wv1[(KD/2)*H], wv2[(KD/2)*H], wo[H*QD], pan[H];
        float cdw[QKV*2], cdb[QKV], cgw[QKV*128*2], cgb[QKV], ks[NKV];
        float pahss[H], pahsb[H], parss[H], parsb[H];
        float gdw[RTR_H*H], gdb[RTR_H], rfn[RTR_H], rf1[RTR_H*RTR_H], rf1b[RTR_H];
        float rf2[RTR_H*RTR_H], rf2b[RTR_H], rout[17*RTR_H], bb[17];
        // Big weights (> 100 MB, mmap'd — only resident when accessed)
        float *gu_mmap = nullptr, *dn_mmap = nullptr;
        size_t gu_size = 0, dn_size = 0;
        float pmhss[H], pmhsb[H], pmrss[H], pmrsb[H];
    };
    LayerW* lw = nullptr;
    std::string wd;

    // mmap helpers
    static float* mmap_file(const std::string& path, size_t* out_size) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) { fprintf(stderr,"mmap: can't open %s\n",path.c_str()); *out_size = 0; return nullptr; }
        struct stat st;
        fstat(fd, &st);
        *out_size = st.st_size;
        float* mapped = (float*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (mapped == MAP_FAILED) {
            fprintf(stderr,"mmap: failed %s (%zu bytes): ",path.c_str(),st.st_size);
            perror("");
            *out_size = 0; return nullptr;
        }
        return mapped;
    }
    static void munmap_file(float* ptr, size_t sz) {
        if (ptr) munmap(ptr, sz);
    }

    // State (per-sequence, cleared on reset)
    float hs[H];
    float prev_hs[H];
    float conv_state[2*QKV];
    float prev_rs[RTR_H];
    float moe_out[H];
    int pos = 0;

    bool moe_weights_available_ = false;
    std::string weights_dir;

    CPUBackend() { type = BackendType::CPU_SCALAR; name = "CPU (scalar)"; }

    ~CPUBackend() override { destroy(); }

    bool init(const ModelConfig& cfg, const std::string& wd) override {
        this->cfg = cfg;
        this->wd = wd;

        // Validate that ModelConfig dimensions match hardcoded constants.
        // The CPU backend uses compile-time sizes; if config differs, all
        // pointer arithmetic would go out of bounds -> SIGSEGV.
        if (cfg.hidden_size != H || cfg.num_layers != N_LAYERS ||
            cfg.vocab_size > VOCAB || cfg.head_dim != HD) {
            fprintf(stderr, "CPU: model config mismatch - need H=%d, L=%d, V<=%d, HD=%d "
                            "but got H=%d, L=%d, V=%d, HD=%d\n",
                    H, N_LAYERS, VOCAB, HD,
                    cfg.hidden_size, cfg.num_layers, cfg.vocab_size, cfg.head_dim);
            return false;
        }

        auto W = [&](const std::string& name) -> std::vector<float> {
            std::ifstream f(wd + "/" + name, std::ios::binary | std::ios::ate);
            if (!f) { fprintf(stderr,"CPU: missing %s\n", (wd+"/"+name).c_str()); return {}; }
            size_t n = f.tellg() / sizeof(float); f.seekg(0);
            std::vector<float> d(n); f.read((char*)d.data(), n * sizeof(float)); return d;
        };
        auto L = [](int i) { return std::to_string(i); };
        auto load = [&](float* dst, const std::string& name) {
            auto v = W(name); if (!v.empty()) memcpy(dst, v.data(), v.size()*4);
        };

        // Base weights
        embed    = new float[VOCAB * H];
        fnorm    = new float[H];
        iscale   = new float[H];
        ibias    = new float[H];
        lm_head_w = new float[VOCAB * H];

        load(embed,    "model_embed_tokens_weight.bin");
        load(fnorm,    "model_norm_weight.bin");
        load(iscale,   "model_input_hidden_states_scale.bin");
        load(ibias,    "model_input_hidden_states_bias.bin");
        // lm_head uses tied embeddings
        {
            auto lm_tmp = W("model_lm_head_weight.bin");
            if (lm_tmp.size() >= (size_t)VOCAB*H) {
                memcpy(lm_head_w, lm_tmp.data(), (size_t)VOCAB*H*4);
            } else {
                printf("CPU: lm_head using tied embeddings\n");
                memcpy(lm_head_w, embed, (size_t)VOCAB*H*4);
            }
        }

        // Layer weights — small ones in memory, MoE experts mmap'd
        printf("CPU: Loading %d layers...\n", N_LAYERS);
        lw = new LayerW[N_LAYERS];
        for (int il = 0; il < N_LAYERS; il++) {
            auto& l = lw[il];
            std::string B = "model_layers_" + L(il) + "_";
            load(l.nw,  B + "input_layernorm_weight.bin");
            load(l.wq,  B + "self_attn_qkv_proj_q_proj_weight.bin");
            load(l.wk,  B + "self_attn_qkv_proj_k_proj_weight.bin");
            load(l.wv1, B + "self_attn_qkv_proj_v_proj_current_weight.bin");
            load(l.wv2, B + "self_attn_qkv_proj_v_proj_delayed_weight.bin");
            load(l.wo,  B + "self_attn_o_proj_weight.bin");
            load(l.pan, B + "post_attention_layernorm_weight.bin");
            load(l.cdw, B + "self_attn_qkv_proj_conv_qk_depthwise_weight.bin");
            load(l.cdb, B + "self_attn_qkv_proj_conv_qk_depthwise_bias.bin");
            load(l.cgw, B + "self_attn_qkv_proj_conv_qk_grouped_weight.bin");
            load(l.cgb, B + "self_attn_qkv_proj_conv_qk_grouped_bias.bin");
            load(l.ks,  B + "self_attn_qk_norm_temp.bin");
            load(l.pahss, B + "post_attention_residual_scale_hidden_states_scale.bin");
            load(l.pahsb, B + "post_attention_residual_scale_hidden_states_bias.bin");
            load(l.parss, B + "post_attention_residual_scale_residual_scale.bin");
            load(l.parsb, B + "post_attention_residual_scale_residual_bias.bin");
            load(l.gdw,  B + "mlp_gate_down_proj_weight.bin");
            load(l.gdb,  B + "mlp_gate_down_proj_bias.bin");
            load(l.rfn,  B + "mlp_gate_router_mlp_norm_weight.bin");
            load(l.rf1,  B + "mlp_gate_router_mlp_fc1_weight.bin");
            load(l.rf1b, B + "mlp_gate_router_mlp_fc1_bias.bin");
            load(l.rf2,  B + "mlp_gate_router_mlp_fc2_weight.bin");
            load(l.rf2b, B + "mlp_gate_router_mlp_fc2_bias.bin");
            load(l.rout, B + "mlp_gate_router_mlp_out_proj_weight.bin");
            load(l.bb,   B + "mlp_gate_balancing_biases.bin");
            // MoE expert weights — mmap'd
            l.gu_mmap = mmap_file(wd + "/" + B + "mlp_experts_gate_up_proj.bin", &l.gu_size);
            l.dn_mmap = mmap_file(wd + "/" + B + "mlp_experts_down_proj.bin", &l.dn_size);
            if (!l.gu_mmap) fprintf(stderr,"WARN: mmap gate_up layer %d failed\n",il);
            if (!l.dn_mmap) fprintf(stderr,"WARN: mmap down layer %d failed\n",il);
            if (l.gu_mmap && l.dn_mmap) moe_weights_available_ = true;
            load(l.pmhss, B + "post_mlp_residual_scale_hidden_states_scale.bin");
            load(l.pmhsb, B + "post_mlp_residual_scale_hidden_states_bias.bin");
            load(l.pmrss, B + "post_mlp_residual_scale_residual_scale.bin");
            load(l.pmrsb, B + "post_mlp_residual_scale_residual_bias.bin");
        }
        if (!moe_weights_available_) {
            printf("CPU: FATAL - no MoE expert weights loaded (missing weight files?)\n");
            initialized = false;
            return false;
        }
        printf("CPU: %d layers loaded\n", N_LAYERS);
        initialized = true;
        return true;
    }

    bool reset() override {
        memset(hs, 0, sizeof(hs));
        memset(prev_hs, 0, sizeof(prev_hs));
        memset(conv_state, 0, sizeof(conv_state));
        memset(prev_rs, 0, sizeof(prev_rs));
        pos = 0;
        return true;
    }

    bool forward(int token_id, float* hidden_out) override {
        // Embed
        for (int i = 0; i < H; i++)
            hs[i] = (embed[token_id * (size_t)H + i] + ibias[i]) * iscale[i];

        for (int il = 0; il < N_LAYERS; il++) {
            auto& l = lw[il];

            // CCA Attention
            cca_attn_cpu(hs, l.wq, l.wk, l.wv1, l.wv2, l.wo,
                l.cdw, l.cdb, l.cgw, l.cgb, l.ks, l.nw,
                prev_hs, conv_state, il);

            // Post-attention residual scale
            float attn_out[H];
            memcpy(attn_out, hs, H * sizeof(float));
            memcpy(hs, attn_out, H * sizeof(float));  // residual
            for (int i = 0; i < H; i++)
                hs[i] = attn_out[i] * l.pahss[i] + l.pahsb[i] + hs[i] * l.parss[i] + l.parsb[i];

            // Post-attention RMSNorm
            rmsnorm(hs, l.pan, H);

            // EDA Router + MoE
            eda_router_moe_cpu(hs, l.gdw, l.gdb, l.rfn, l.rf1, l.rf1b,
                l.rf2, l.rf2b, l.rout, l.bb,
                l.gu_mmap, l.dn_mmap, prev_rs, moe_out);

            // Post-MLP residual scale
            for (int i = 0; i < H; i++)
                hs[i] = moe_out[i] * l.pmhss[i] + l.pmhsb[i] + hs[i] * l.pmrss[i] + l.pmrsb[i];
        }

        memcpy(hidden_out, hs, H * sizeof(float));
        pos++;
        return true;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        // Final RMSNorm
        float tmp[H];
        memcpy(tmp, hidden, H * sizeof(float));
        rmsnorm(tmp, fnorm, H);

        // Compute logits with SIMD batched matmul
        matmul_lmhead(logits, tmp, embed, VOCAB, H);

        if (argmax) {
            *argmax = 0;
            for (int v = 1; v < VOCAB; v++) if (logits[v] > logits[*argmax]) *argmax = v;
        }
        return true;
    }

    int generate(int token_id) override {
        float hidden[H];
        if (!forward(token_id, hidden)) return -1;
        float* logits = new float[VOCAB];
        int result;
        lm_head(hidden, logits, &result);
        delete[] logits;
        return result;
    }

    float benchmark(int tokens = 10) override {
        if (!initialized || !embed || !lw) {
            fprintf(stderr, "CPU: benchmark called but not initialized — skipping\n");
            return 0.0f;
        }
        reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        int tok = 100;
        for (int i = 0; i < tokens; i++) {
            tok = generate(tok);
            if (tok < 0) break;
        }
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return ms / tokens;
    }

    void destroy() override {
        delete[] embed; delete[] fnorm; delete[] iscale; delete[] ibias; delete[] lm_head_w;
        if (lw) {
            for (int il = 0; il < N_LAYERS; il++) {
                munmap_file(lw[il].gu_mmap, lw[il].gu_size);
                munmap_file(lw[il].dn_mmap, lw[il].dn_size);
            }
            delete[] lw;
        }
        initialized = false;
    }
};

Backend* create_cpu_backend() { return new CPUBackend(); }
