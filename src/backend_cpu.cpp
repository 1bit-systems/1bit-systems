// backend_cpu.cpp — CPU scalar reference backend for Zaya architecture
// Reference implementation. No GPU, no NPU, no special hardware.
// Uses the same weight format as GPU backends.
//
// NOTE: The CCA attention and MoE router use compile-time stack-allocated
// arrays and shared-memory patterns that require fixed dimensions. For
// arbitrary architectures, use backend_generic.cpp instead.
// The validation gate in init() rejects unsupported configs.

#include "backend.h"
#include "colibri_kernels.h"
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

// ── Optional: use colibri int4 quantized kernels for big matmuls ──
// Quantizes weight matrices at load time (8x smaller memory) and uses int4
// SIMD matmul at inference. ~2x speedup on CPU, huge memory savings.
// Toggle USE_COLIBRI_Q4=0 to restore original f32 behavior.
#ifndef USE_COLIBRI_Q4
#define USE_COLIBRI_Q4 1
#endif

// ── Architecture constants (Zaya1-8B reference dimensions) ──
// These remain constexpr for stack-allocated array sizes in CCA/MoE.
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
#if defined(__AVX512F__) || defined(__AVX2__)
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
        alignas(32) float tmp[8]; _mm256_store_ps(tmp, vacc);
        float s = 0; for (int j = 0; j < 8; j++) s += tmp[j];
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
// Kept for reference — not currently used
[[maybe_unused]] static void matmul_lmhead(float* out, const float* in, const float* wt, int V, int H_) {
    (void)out; (void)in; (void)wt; (void)V; (void)H_;
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

[[maybe_unused]] static float cosim(const float* a, const float* b, int n) {
    float d = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return d / (sqrtf(na) * sqrtf(nb) + 1e-12f);
}

// ── CCA Attention (CPU, single token) ──
// When q4 pointers are non-null, uses int4 quantized matmuls for Q/K/V/O.
static void cca_attn_cpu(float* h, const float* wq_in, const float* wk_in,
    const float* wv1_in, const float* wv2_in, const float* wo_in,
    const float* cdw, const float* cdb, const float* cgw, const float* cgb,
    const float* ks, const float* nw,
    float* prev_hs, float* conv_state, int pos
#if USE_COLIBRI_Q4
    , const uint8_t* q4_wq, const float* q4s_wq,
    const uint8_t* q4_wk, const float* q4s_wk,
    const uint8_t* q4_wv1, const float* q4s_wv1,
    const uint8_t* q4_wv2, const float* q4s_wv2,
    const uint8_t* q4_wo, const float* q4s_wo
#endif
) {

    float buf[H], q[QD], k[KD], v[KD];
    float sqk[QKV], dw0[QKV], dw1[QKV];

    // 1. RMSNorm
    memcpy(buf, h, H * sizeof(float));
    rmsnorm(buf, nw, H);

    // 2/3. Q, K projections (f32 or int4)
#if USE_COLIBRI_Q4
    if (q4_wq) {
        colibri_matmul_q4(q, buf, q4_wq, q4s_wq, 1, H, QD);
    } else
#endif
    matmul_t(q, buf, wq_in, QD, H);

#if USE_COLIBRI_Q4
    if (q4_wk) {
        colibri_matmul_q4(k, buf, q4_wk, q4s_wk, 1, H, KD);
    } else
#endif
    matmul_t(k, buf, wk_in, KD, H);

    // raw qk concat
    memcpy(sqk, q, QD * sizeof(float));
    memcpy(sqk + QD, k, KD * sizeof(float));

    // 4. V projections (current + delayed)
#if USE_COLIBRI_Q4
    if (q4_wv1) {
        colibri_matmul_q4(v, buf, q4_wv1, q4s_wv1, 1, H, KD/2);
    } else
#endif
    matmul_t(v, buf, wv1_in, KD/2, H);

#if USE_COLIBRI_Q4
    if (q4_wv2) {
        colibri_matmul_q4(&v[KD/2], prev_hs, q4_wv2, q4s_wv2, 1, H, KD/2);
    } else
#endif
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

    // 10. Output projection (f32 or int4)
    float final_out[H];
#if USE_COLIBRI_Q4
    if (q4_wo) {
        colibri_matmul_q4(final_out, attn_out, q4_wo, q4s_wo, 1, QD, H);
    } else
#endif
    matmul_t(final_out, attn_out, wo_in, H, QD);
    memcpy(h, final_out, H * sizeof(float));

    // Update state
    memcpy(prev_hs, h, H * sizeof(float));
    memcpy(conv_state, conv_state + QKV, QKV * sizeof(float));
    memcpy(conv_state + QKV, sqk, QKV * sizeof(float));
}

// ── EDA Router + MoE (CPU) ──
// When q4 pointers are non-null, uses int4 quantized matmuls.
static void eda_router_moe_cpu(float* h, const float* gate_down_w, const float* gate_down_b,
    const float* router_norm_w, const float* rf1, const float* rf1b,
    const float* rf2, const float* rf2b, const float* rout, const float* bb,
    const float* gu, const float* dn, float* prev_rs, float* moe_out
#if USE_COLIBRI_Q4
    , const uint8_t* q4_gdw, const float* q4s_gdw,
    const uint8_t* q4_rf1, const float* q4s_rf1,
    const uint8_t* q4_rf2, const float* q4s_rf2,
    const uint8_t* q4_rout, const float* q4s_rout,
    const uint8_t* q4_gu, const float* q4s_gu,
    const uint8_t* q4_dn, const float* q4s_dn
#endif
) {

    float rs[RTR_H], tmp[RTR_H];

    // gate_down (f32 or int4)
#if USE_COLIBRI_Q4
    if (q4_gdw) {
        colibri_matmul_q4(rs, h, q4_gdw, q4s_gdw, 1, H, RTR_H);
    } else
#endif
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

    // fc1 + GELU (f32 or int4)
#if USE_COLIBRI_Q4
    if (q4_rf1) {
        colibri_matmul_q4(tmp, rs, q4_rf1, q4s_rf1, 1, RTR_H, RTR_H);
    } else
#endif
    matmul_t(tmp, rs, rf1, RTR_H, RTR_H);
    for (int i = 0; i < RTR_H; i++) tmp[i] += rf1b[i];
    for (int i = 0; i < RTR_H; i++) tmp[i] = gelu(tmp[i]);

    // fc2 + GELU
#if USE_COLIBRI_Q4
    if (q4_rf2) {
        colibri_matmul_q4(rs, tmp, q4_rf2, q4s_rf2, 1, RTR_H, RTR_H);
    } else
#endif
    matmul_t(rs, tmp, rf2, RTR_H, RTR_H);
    for (int i = 0; i < RTR_H; i++) rs[i] += rf2b[i];
    for (int i = 0; i < RTR_H; i++) rs[i] = gelu(rs[i]);

    // out_proj + balancing biases (f32 or int4)
    float probs[17];
#if USE_COLIBRI_Q4
    if (q4_rout && q4s_rout) {
        colibri_matmul_q4(probs, rs, q4_rout, q4s_rout, 1, RTR_H, 17);
    } else
#endif
    {
        for (int i = 0; i < 17; i++) {
            float s = 0; for (int j = 0; j < RTR_H; j++) s += rs[j] * rout[i*RTR_H+j];
            probs[i] = s;
        }
    }
    for (int i = 0; i < 17; i++) probs[i] += bb[i];

    // Softmax
    float mv = probs[0]; for (int i = 1; i < 17; i++) if (probs[i] > mv) mv = probs[i];
    float sum = 0; for (int i = 0; i < 17; i++) { probs[i] = expf(probs[i] - mv); sum += probs[i]; }
    for (int i = 0; i < 17; i++) probs[i] /= sum;

    // Top-1 (for CPU simplicity, use top-1 instead of top-2)
    int best = 0; for (int i = 1; i < 17; i++) if (probs[i] > probs[best]) best = i;
    float best_w = probs[best];

    // MoE expert FFN
    memset(moe_out, 0, H * sizeof(float));
    if (best < N_EXP && (gu || q4_gu)) {
        // gate_up (f32 or int4)
        float gate_up[2*N_FF];
#if USE_COLIBRI_Q4
        if (q4_gu && q4s_gu) {
            const uint8_t* gu_row = q4_gu + (size_t)best * 2 * N_FF * ((H + 1) / 2);
            const float*   gs_row = q4s_gu + (size_t)best * 2 * N_FF;
            colibri_matmul_q4(gate_up, h, gu_row, gs_row, 1, H, 2*N_FF);
        } else
#endif
        { matmul_t(gate_up, h, &gu[(size_t)best*2*N_FF*H], 2*N_FF, H); }

        // SiLU(gate) * up
        float moe_buf[N_FF];
        for (int i = 0; i < N_FF; i++)
            moe_buf[i] = silu(gate_up[i]) * gate_up[N_FF+i];

        // down (f32 or int4)
#if USE_COLIBRI_Q4
        if (q4_dn && q4s_dn) {
            const uint8_t* dn_row = q4_dn + (size_t)best * H * ((N_FF + 1) / 2);
            const float*   ds_row = q4s_dn + (size_t)best * H;
            colibri_matmul_q4(moe_out, moe_buf, dn_row, ds_row, 1, N_FF, H);
        } else
#endif
        { matmul_t(moe_out, moe_buf, &dn[(size_t)best*H*N_FF], H, N_FF); }

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

#if USE_COLIBRI_Q4
    // Quantized embedding / lm_head (enormous: 262k×2048 = 2.1 GB f32 → 268 MB q4)
    std::vector<uint8_t> q4_embed;
    std::vector<float>   q4s_embed;
    std::vector<uint8_t> q4_lm_head;
    std::vector<float>   q4s_lm_head;
    // Scratch buffer for on-the-fly activation quantization (IDOT decode path)
    std::vector<int8_t> idot_scratch;
#endif

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

#if USE_COLIBRI_Q4
        // Quantized (int4) versions of the big weight matrices
        // Storage: q4_buf[O * ((I+1)/2)] uint8_t + scale[O] float
        // Memory: 0.5 bytes/param vs 4.0 for f32 — 8x compression
        std::vector<uint8_t> q4_wq, q4_wk, q4_wv1, q4_wv2, q4_wo;
        std::vector<float>   q4s_wq, q4s_wk, q4s_wv1, q4s_wv2, q4s_wo;
        std::vector<uint8_t> q4_gdw;
        std::vector<float>   q4s_gdw;
        std::vector<uint8_t> q4_rf1, q4_rf2, q4_rout;
        std::vector<float>   q4s_rf1, q4s_rf2, q4s_rout;
        // MoE expert weights: quantized copies (loaded from mmap)
        std::vector<uint8_t> q4_gu, q4_dn;
        std::vector<float>   q4s_gu, q4s_dn;
        bool experts_q4 = false;
#endif
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

    // LM head logits cache (avoids 1 MB heap alloc per token - fix #740)
    std::vector<float> logits_cache;

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

#if USE_COLIBRI_Q4
        // Quantize embedding/lm_head to int4 (8x compression: 2.1 GB → 268 MB)
        {
            printf("CPU: quantizing embedding table (%d×%d = %.1fM params)...\n",
                   VOCAB, H, (double)VOCAB*H/1e6);
            q4_embed.resize((size_t)VOCAB * ((H + 1) / 2));
            q4s_embed.resize(VOCAB);
            colibri_quantize_matrix_q4(embed, q4_embed.data(), q4s_embed.data(), VOCAB, H);
            
            bool tied = (lm_head_w == embed);
            if (!tied) {
                q4_lm_head.resize((size_t)VOCAB * ((H + 1) / 2));
                q4s_lm_head.resize(VOCAB);
                colibri_quantize_matrix_q4(lm_head_w, q4_lm_head.data(), q4s_lm_head.data(), VOCAB, H);
            } else {
                // Share embed's quantized copy for lm_head
                q4_lm_head.clear(); q4s_lm_head.clear();
            }
            // Scratch for IDOT: H bytes per row
            idot_scratch.resize(H);
            printf("CPU: embedding quantized: f32=%.1f MB → q4=%.1f MB\n",
                   (double)VOCAB*H*4/1e6,
                   (double)(VOCAB * ((H+1)/2) + VOCAB*4)/1e6);
        }
#endif

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

#if USE_COLIBRI_Q4
            // Quantize this layer's dense weights to int4
            auto q4_alloc = [](int O, int I) -> size_t { return (size_t)O * ((size_t)(I + 1) / 2); };
            l.q4_wq .resize(q4_alloc(QD, H)); l.q4s_wq .resize(QD);
            l.q4_wk .resize(q4_alloc(KD, H)); l.q4s_wk .resize(KD);
            l.q4_wv1.resize(q4_alloc(KD/2, H)); l.q4s_wv1.resize(KD/2);
            l.q4_wv2.resize(q4_alloc(KD/2, H)); l.q4s_wv2.resize(KD/2);
            l.q4_wo .resize(q4_alloc(H, QD)); l.q4s_wo .resize(H);
            l.q4_gdw.resize(q4_alloc(RTR_H, H)); l.q4s_gdw.resize(RTR_H);
            l.q4_rf1 .resize(q4_alloc(RTR_H, RTR_H)); l.q4s_rf1 .resize(RTR_H);
            l.q4_rf2 .resize(q4_alloc(RTR_H, RTR_H)); l.q4s_rf2 .resize(RTR_H);
            l.q4_rout.resize(q4_alloc(17, RTR_H)); l.q4s_rout.resize(17);
            colibri_quantize_matrix_q4(l.wq,  l.q4_wq.data(),  l.q4s_wq.data(),  QD,   H);
            colibri_quantize_matrix_q4(l.wk,  l.q4_wk.data(),  l.q4s_wk.data(),  KD,   H);
            colibri_quantize_matrix_q4(l.wv1, l.q4_wv1.data(), l.q4s_wv1.data(), KD/2, H);
            colibri_quantize_matrix_q4(l.wv2, l.q4_wv2.data(), l.q4s_wv2.data(), KD/2, H);
            colibri_quantize_matrix_q4(l.wo,  l.q4_wo.data(),  l.q4s_wo.data(),  H,    QD);
            colibri_quantize_matrix_q4(l.gdw, l.q4_gdw.data(), l.q4s_gdw.data(), RTR_H, H);
            colibri_quantize_matrix_q4(l.rf1, l.q4_rf1.data(), l.q4s_rf1.data(), RTR_H, RTR_H);
            colibri_quantize_matrix_q4(l.rf2, l.q4_rf2.data(), l.q4s_rf2.data(), RTR_H, RTR_H);
            colibri_quantize_matrix_q4(l.rout,l.q4_rout.data(),l.q4s_rout.data(),17,    RTR_H);

            // Quantize MoE expert weights (from mmap'd f32)
            if (l.gu_mmap && l.dn_mmap) {
                size_t gu_rows = (size_t)N_EXP * 2 * N_FF;  // rows of [H] each
                size_t dn_rows = (size_t)N_EXP * H;           // rows of [N_FF] each
                l.q4_gu.resize((size_t)gu_rows * ((H + 1) / 2));
                l.q4s_gu.resize(gu_rows);
                l.q4_dn.resize((size_t)dn_rows * ((N_FF + 1) / 2));
                l.q4s_dn.resize(dn_rows);
                // Quantize row by row from the mmap'd float data
                for (size_t r = 0; r < gu_rows; r++) {
                    l.q4s_gu[r] = colibri_quantize_q4(
                        l.gu_mmap + r * H,
                        l.q4_gu.data() + r * ((H + 1) / 2), H);
                }
                for (size_t r = 0; r < dn_rows; r++) {
                    l.q4s_dn[r] = colibri_quantize_q4(
                        l.dn_mmap + r * N_FF,
                        l.q4_dn.data() + r * ((N_FF + 1) / 2), N_FF);
                }
                l.experts_q4 = true;
                // Free the mmap'd f32 copies — we won't need them anymore
                // (the quantized copies in memory are smaller anyway)
                // We keep the mmap'd originals for now in case of fallback
            }
#endif
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
        if (!embed || !fnorm) {
            printf("CPU: FATAL - missing critical weight files (embed/fnorm)\n");
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

    // CPU: all weights already resident in RAM — preload is a cache hint
    bool preload_layer(int layer) override {
        if (layer < 0 || layer >= N_LAYERS || !lw) return false;
        auto& l = lw[layer];
        // Touch the first cache line of each big weight matrix
        // to pull it into L2/L3 before the layer actually runs.
        // The weights are already in DRAM (loaded at init), so this
        // is purely a prefetch hint — not strictly necessary on CPU
        // but demonstrates the pattern for GPU/NPU backends.
        __builtin_prefetch(l.wq, 0, 3);    // Q proj
        __builtin_prefetch(l.wk, 0, 3);    // K proj
        __builtin_prefetch(l.wo, 0, 3);    // O proj
        __builtin_prefetch(l.gdw, 0, 3);   // gate_down
        return true;
    }

    bool forward(int token_id, float* hidden_out) override {
        // Embed
        for (int i = 0; i < H; i++)
            hs[i] = (embed[token_id * (size_t)H + i] + ibias[i]) * iscale[i];

        for (int il = 0; il < N_LAYERS; il++) {
            auto& l = lw[il];

            // Pilot: notify layer start (for prefetch + timing)
            if (pilot_) {
                pilot_->on_layer_start(il);
                pilot_->prefetch_next(il);
            }

            // Save pre-attention hidden state for residual connection
            // (cca_attn_cpu modifies hs in-place — issue #352)
            float pre_attn_hs[H];
            memcpy(pre_attn_hs, hs, H * sizeof(float));

            // CCA Attention (with optional int4 quantized weights)
            cca_attn_cpu(hs, l.wq, l.wk, l.wv1, l.wv2, l.wo,
                l.cdw, l.cdb, l.cgw, l.cgb, l.ks, l.nw,
                prev_hs, conv_state, il
#if USE_COLIBRI_Q4
                , l.q4_wq.data(), l.q4s_wq.data(),
                l.q4_wk.data(), l.q4s_wk.data(),
                l.q4_wv1.data(), l.q4s_wv1.data(),
                l.q4_wv2.data(), l.q4s_wv2.data(),
                l.q4_wo.data(), l.q4s_wo.data()
#endif
            );

            // Post-attention residual scale
            //   output = attn_out * pahss + pahsb + pre_attn_hs * parss + parsb
            float attn_out[H];
            memcpy(attn_out, hs, H * sizeof(float));
            for (int i = 0; i < H; i++)
                hs[i] = attn_out[i] * l.pahss[i] + l.pahsb[i] + pre_attn_hs[i] * l.parss[i] + l.parsb[i];

            // Post-attention RMSNorm
            rmsnorm(hs, l.pan, H);

            // EDA Router + MoE (with optional int4 quantized weights)
            eda_router_moe_cpu(hs, l.gdw, l.gdb, l.rfn, l.rf1, l.rf1b,
                l.rf2, l.rf2b, l.rout, l.bb,
                l.gu_mmap, l.dn_mmap, prev_rs, moe_out
#if USE_COLIBRI_Q4
                , l.q4_gdw.data(), l.q4s_gdw.data(),
                l.q4_rf1.data(), l.q4s_rf1.data(),
                l.q4_rf2.data(), l.q4s_rf2.data(),
                l.q4_rout.data(), l.q4s_rout.data(),
                l.experts_q4 ? l.q4_gu.data() : nullptr,
                l.experts_q4 ? l.q4s_gu.data() : nullptr,
                l.experts_q4 ? l.q4_dn.data() : nullptr,
                l.experts_q4 ? l.q4s_dn.data() : nullptr
#endif
            );

            // Post-MLP residual scale
            for (int i = 0; i < H; i++)
                hs[i] = moe_out[i] * l.pmhss[i] + l.pmhsb[i] + hs[i] * l.pmrss[i] + l.pmrsb[i];

            // Pilot: notify layer done
            if (pilot_) pilot_->on_layer_done(il);
        }

        // Pilot: end of token
        if (pilot_) pilot_->on_token_done();

        memcpy(hidden_out, hs, H * sizeof(float));
        pos++;
        return true;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        // Final RMSNorm
        float tmp[H];
        memcpy(tmp, hidden, H * sizeof(float));
        rmsnorm(tmp, fnorm, H);

        // Compute logits — use int4 quantized matmul for the massive V×H projection
#if USE_COLIBRI_Q4
        bool tied = q4_lm_head.empty();  // tied to embed?
        const uint8_t* w = tied ? q4_embed.data() : q4_lm_head.data();
        const float* s = tied ? q4s_embed.data() : q4s_lm_head.data();
        colibri_matmul_q4(logits, tmp, w, s, 1, H, VOCAB);
#else
        matmul_lmhead(logits, tmp, embed, VOCAB, H);
#endif

        if (argmax) {
            *argmax = 0;
            for (int v = 1; v < VOCAB; v++) if (logits[v] > logits[*argmax]) *argmax = v;
        }
        return true;
    }

    int generate(int token_id) override {
        float hidden[H];
        if (!forward(token_id, hidden)) return -1;
        if (logits_cache.empty()) logits_cache.resize(VOCAB);
        int result;
        lm_head(hidden, logits_cache.data(), &result);
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
