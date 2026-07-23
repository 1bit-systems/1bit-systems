// cpu_layer.cpp — CPU ternary inference backend implementation
// C ABI functions callable from Zig via @cImport.
//
// Build: g++ -O3 -march=native -fPIC -std=c++17 -shared -o libcpu_trg.so cpu_layer.cpp
// Or link as object: g++ -O3 -march=native -std=c++17 -c cpu_layer.cpp

#include "cpu_layer.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>

// ═══════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════

// Decode one packed uint32 → 16 float signs in { -1, 0, +1 }
static inline void decode_ternary_word(uint32_t word, float* signs_out) {
    for (int v = 0; v < 16; v++) {
        uint32_t bits = (word >> (v * 2)) & 0x3;
        signs_out[v] = (float)((bits == 1) ? 1 : (bits == 2) ? -1 : 0);
    }
}

// ═══════════════════════════════════════════════════════════════════
// CPU ternary GEMV — scalar + SIMD (AVX2 / AVX-512)
// ═══════════════════════════════════════════════════════════════════

// Forward declare the SIMD variants
#if defined(__AVX512F__)
    #include <immintrin.h>
    static void gemv_avx512(const uint32_t* rw, const float* x, double* acc, int pk) {
        __m512 vacc = _mm512_setzero_ps();
        int u = 0;
        // Process 2 uint32 (32 vals) per AVX-512 iteration
        for (; u + 2 <= pk; u += 2) {
            uint32_t w0 = rw[u], w1 = rw[u+1];
            // Process 16 vals from w0, then 16 vals from w1
            for (int wi = 0; wi < 2; wi++) {
                uint32_t word = (wi == 0) ? w0 : w1;
                // Broadcast word to all lanes
                __m512i w_vec = _mm512_set1_epi32((int)word);
                // Per-lane shift: lane i extracts bits [2i:2i+1]
                // Use set_epi32 for the shift amounts (0,2,4,...,30)
                __m512i shifts = _mm512_set_epi32(30,28,26,24,22,20,18,16,
                                                    14,12,10,8,6,4,2,0);
                __m512i shifted = _mm512_srlv_epi32(w_vec, shifts);
                __m512i bits = _mm512_and_si512(shifted, _mm512_set1_epi32(3));
                // sign = (bits==1) - (bits==2)
                __mmask16 is_pos = _mm512_cmpeq_epi32_mask(bits, _mm512_set1_epi32(1));
                __mmask16 is_neg = _mm512_cmpeq_epi32_mask(bits, _mm512_set1_epi32(2));
                // Load activations for this uint32's 16 vals
                __m512 act = _mm512_loadu_ps(x + (u+wi)*16);
                // pos_act = act for pos lanes, 0 otherwise
                __m512 pos_act = _mm512_maskz_mov_ps(is_pos, act);
                __m512 neg_act = _mm512_maskz_mov_ps(is_neg, act);
                vacc = _mm512_add_ps(vacc, pos_act);
                vacc = _mm512_sub_ps(vacc, neg_act);
            }
        }
        // Remainder
        for (; u < pk; u++) {
            uint32_t word = rw[u];
            for (int v = 0; v < 16; v++) {
                uint32_t bits = (word >> (v*2)) & 3;
                float sign = (float)(bits == 1) - (float)(bits == 2);
                vacc = _mm512_add_ps(vacc, _mm512_set1_ps(sign * x[u*16+v]));
            }
        }
        *acc = (double)_mm512_reduce_add_ps(vacc);
    }
#elif defined(__AVX2__)
    #include <immintrin.h>
    static void gemv_avx2(const uint32_t* rw, const float* x, double* acc, int pk) {
        __m256 vacc = _mm256_setzero_ps();
        int u = 0;
        // Process 1 uint32 (16 vals) per AVX2 iteration, split as 2×8
        for (; u < pk; u++) {
            uint32_t word = rw[u];
            // Process 8 vals at a time, 2 iterations per uint32
            for (int sub = 0; sub < 2; sub++) {
                int bit_off = sub * 16;  // sub=0: bits 0-15, sub=1: bits 16-31
                uint16_t code8 = (uint16_t)((word >> bit_off) & 0xFFFF);
                // Extract 8 × 2-bit codes
                __m256i ci = _mm256_set_epi32(
                    (code8 >> 14) & 3,  // lane 7
                    (code8 >> 12) & 3,  // lane 6
                    (code8 >> 10) & 3,  // lane 5
                    (code8 >> 8)  & 3,  // lane 4
                    (code8 >> 6)  & 3,  // lane 3
                    (code8 >> 4)  & 3,  // lane 2
                    (code8 >> 2)  & 3,  // lane 1
                    (code8 >> 0)  & 3); // lane 0
                // sign = (bits==1) - (bits==2)
                __m256i is_pos = _mm256_cmpeq_epi32(ci, _mm256_set1_epi32(1));
                __m256i is_neg = _mm256_cmpeq_epi32(ci, _mm256_set1_epi32(2));
                // Convert mask ints to floats. cmpeq yields -1 for true.
                // sign = neg_f - pos_f  (because -1 - (-1) = 0, 0 - (-1) = 1, -1-0=-1)
                // Actually: is_pos = -1 for true. cvtepi32 gives -1.0f for true.
                // is_neg = -1 for true. cvtepi32 gives -1.0f for true.
                // We want: val = 1 for pos, -1 for neg, 0 otherwise.
                // val = -pos_f - neg_f? No.
                // pos_f = -1.0 for pos. neg_f = -1.0 for neg.
                // val = neg_f - pos_f:
                //   pos: pos_f=-1, neg_f=0 → 0 - (-1) = 1 ✓
                //   neg: pos_f=0, neg_f=-1 → -1 - 0 = -1 ✓
                //   none: 0 - 0 = 0 ✓
                __m256 pos_f = _mm256_cvtepi32_ps(is_pos);  // -1.0 or 0.0
                __m256 neg_f = _mm256_cvtepi32_ps(is_neg);  // -1.0 or 0.0
                __m256 sign_f = _mm256_sub_ps(neg_f, pos_f);
                // Load activations
                __m256 act = _mm256_loadu_ps(x + u*16 + sub*8);
                vacc = _mm256_fmadd_ps(sign_f, act, vacc);
            }
        }
        // Horizontal sum
        __m128 hi = _mm256_extractf128_ps(vacc, 1);
        __m128 lo = _mm256_castps256_ps128(vacc);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        float result;
        _mm_store_ss(&result, sum);
        *acc = (double)result;
    }
#endif

void cpu_ternary_gemv(
    const uint32_t* packed,
    const float*    x,
    const float*    scales,
    float*          y,
    int M, int K)
{
    const int pk = K / 16;

    #if defined(__AVX512F__)
    // AVX-512 path — 16 vals/iteration
    for (int row = 0; row < M; row++) {
        double acc = 0.0;
        gemv_avx512(packed + row * pk, x, &acc, pk);
        y[row] = (float)(acc * (double)scales[row]);
    }
    #elif defined(__AVX2__)
    // AVX2 path — 8 vals/iteration
    for (int row = 0; row < M; row++) {
        double acc = 0.0;
        gemv_avx2(packed + row * pk, x, &acc, pk);
        y[row] = (float)(acc * (double)scales[row]);
    }
    #else
    // Scalar fallback
    for (int row = 0; row < M; row++) {
        double acc = 0.0;
        const uint32_t* rw = packed + row * pk;
        for (int u = 0; u < pk; u++) {
            uint32_t word = rw[u];
            for (int v = 0; v < 16; v++) {
                uint32_t bits = (word >> (v * 2)) & 0x3;
                float sign;
                if (bits == 1) sign = 1.0f;
                else if (bits == 2) sign = -1.0f;
                else sign = 0.0f;
                acc += (double)sign * (double)x[u * 16 + v];
            }
        }
        y[row] = (float)(acc * (double)scales[row]);
    }
    #endif
}

// ── Per-block scale GEMV (block = 256 ternary = 16 uint32s) ──────
void cpu_ternary_gemv_block(
    const uint32_t* packed,
    const float*    x,
    const float*    scales,
    float*          y,
    int M, int K, int n_blocks)
{
    const int blk_pk = 16;  // 256 ternary / 16 vals per uint32 = 16 uint32s
    const int pk = K / 16;

    for (int row = 0; row < M; row++) {
        double total = 0.0;
        const uint32_t* rw = packed + row * pk;
        const float* sc = scales + row * n_blocks;

        for (int b = 0; b < n_blocks; b++) {
            double blk_acc = 0.0;
            for (int u = 0; u < blk_pk; u++) {
                uint32_t word = rw[b * blk_pk + u];
                for (int v = 0; v < 16; v++) {
                    uint32_t bits = (word >> (v * 2)) & 0x3;
                    float sign;
                    if (bits == 1) sign = 1.0f;
                    else if (bits == 2) sign = -1.0f;
                    else sign = 0.0f;
                    blk_acc += (double)sign * (double)x[(b * blk_pk + u) * 16 + v];
                }
            }
            total += blk_acc * (double)sc[b];
        }
        y[row] = (float)total;
    }
}

// ═══════════════════════════════════════════════════════════════════
// RMSNorm
// ═══════════════════════════════════════════════════════════════════
void cpu_rmsnorm(
    const float* x,
    const float* weight,
    float*       y,
    int N, float eps)
{
    double ss = 0.0;
    for (int i = 0; i < N; i++) {
        ss += (double)x[i] * (double)x[i];
    }
    float rms = 1.0f / std::sqrt((float)(ss / (double)N) + eps);
    for (int i = 0; i < N; i++) {
        y[i] = x[i] * rms * weight[i];
    }
}

// ═══════════════════════════════════════════════════════════════════
// RoPE (in-place)
// ═══════════════════════════════════════════════════════════════════
void cpu_rope(
    float*       x,
    int          pos,
    int          n_heads,
    int          head_dim,
    const float* sin_table,
    const float* cos_table)
{
    const int hd2 = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float* head = x + h * head_dim;
        const float* sin_row = sin_table + pos * head_dim;
        const float* cos_row = cos_table + pos * head_dim;
        for (int d = 0; d < hd2; d++) {
            float a = head[d];
            float b = head[d + hd2];
            float c = cos_row[d];
            float s = sin_row[d];
            head[d]      = a * c - b * s;
            head[d + hd2] = b * c + a * s;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// SiLU: x / (1 + exp(-x))
// ═══════════════════════════════════════════════════════════════════
static inline float silu_f(float x) {
    // Clamp to avoid exp overflow
    if (x < -80.0f) return 0.0f;
    if (x >  80.0f) return x;
    return x / (1.0f + std::exp(-x));
}

void cpu_silu_glu(
    const float* gate,
    const float* up,
    float*       out,
    int N)
{
    for (int i = 0; i < N; i++) {
        out[i] = silu_f(gate[i]) * up[i];
    }
}

// ═══════════════════════════════════════════════════════════════════
// ReLU² GLU (BitNet style)
// ═══════════════════════════════════════════════════════════════════
void cpu_relu2_glu(
    const float* gate,
    const float* up,
    float*       out,
    int N)
{
    for (int i = 0; i < N; i++) {
        float r = (gate[i] > 0.0f) ? gate[i] : 0.0f;
        out[i] = r * r * up[i];
    }
}

// ═══════════════════════════════════════════════════════════════════
// Causal attention
// ═══════════════════════════════════════════════════════════════════
void cpu_attention(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    float*       output,
    int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int gqa_ratio)
{
    const float scale = 1.0f / std::sqrt((float)head_dim);

    for (int h = 0; h < n_heads; h++) {
        int kvh = h / gqa_ratio;
        const float* qh = q + h * head_dim;
        float* out_h = output + h * head_dim;

        // Compute scores (causal: only attend to pos ≤ current)
        int max_pos = seq_len;
        const int MAX_SEQ = 4096; // stack limit — should match model max_seq_len
        float scores_stack[4096];

        float max_score = -1e30f;
        for (int pos = 0; pos < max_pos && pos < MAX_SEQ; pos++) {
            const float* k_row = k_cache + pos * n_kv_heads * head_dim + kvh * head_dim;
            double dot = 0.0;
            for (int d = 0; d < head_dim; d++) {
                dot += (double)qh[d] * (double)k_row[d];
            }
            float s = (float)(dot * (double)scale);
            scores_stack[pos] = s;
            if (s > max_score) max_score = s;
        }

        // Softmax
        double sum_exp = 0.0;
        for (int pos = 0; pos < max_pos && pos < MAX_SEQ; pos++) {
            float e = std::exp(scores_stack[pos] - max_score);
            scores_stack[pos] = e;
            sum_exp += (double)e;
        }
        float inv_sum = (sum_exp > 0.0) ? (float)(1.0 / sum_exp) : 0.0f;

        // Weighted sum of V
        std::memset(out_h, 0, head_dim * sizeof(float));
        for (int pos = 0; pos < max_pos && pos < MAX_SEQ; pos++) {
            float w = scores_stack[pos] * inv_sum;
            const float* v_row = v_cache + pos * n_kv_heads * head_dim + kvh * head_dim;
            for (int d = 0; d < head_dim; d++) {
                out_h[d] += w * v_row[d];
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Full transformer layer (Qwen3)
// ═══════════════════════════════════════════════════════════════════
int cpu_layer_forward_qwen3(
    float*       hidden,
    float*       scratch_qkv,
    float*       scratch_attn,
    float*       scratch_ffn,
    float*       scratch_act,
    const float* in_norm_weight,
    const float* attn_q_norm,
    const float* attn_k_norm,
    const float* pa_norm_weight,
    const float* final_norm_weight,
    const uint32_t* q_packed,  const float* q_scales,
    const uint32_t* k_packed,  const float* k_scales,
    const uint32_t* v_packed,  const float* v_scales,
    const uint32_t* o_packed,  const float* o_scales,
    const uint32_t* gate_packed, const float* gate_scales,
    const uint32_t* up_packed,   const float* up_scales,
    const uint32_t* down_packed, const float* down_scales,
    float* k_cache,
    float* v_cache,
    int hidden_dim, int inter_size,
    int n_heads, int n_kv_heads, int head_dim, int gqa_ratio,
    int pos,
    const float* sin_table,
    const float* cos_table,
    float rms_norm_eps)
{
    // ── Save residual ──
    float residual[4096]; // max hidden_dim
    std::memcpy(residual, hidden, hidden_dim * sizeof(float));

    // ── RMSNorm + QKV ternary GEMV ──
    cpu_rmsnorm(hidden, in_norm_weight, hidden, hidden_dim, rms_norm_eps);

    int qkv_dim = n_heads * head_dim + 2 * n_kv_heads * head_dim;
    cpu_ternary_gemv(q_packed, hidden, q_scales, scratch_qkv, n_heads * head_dim, hidden_dim);
    cpu_ternary_gemv(k_packed, hidden, k_scales, scratch_qkv + n_heads * head_dim, n_kv_heads * head_dim, hidden_dim);
    cpu_ternary_gemv(v_packed, hidden, v_scales, scratch_qkv + n_heads * head_dim + n_kv_heads * head_dim, n_kv_heads * head_dim, hidden_dim);

    // ── Q/K norm + RoPE ──
    if (attn_q_norm) {
        for (int h = 0; h < n_heads; h++) {
            cpu_rmsnorm(scratch_qkv + h * head_dim, attn_q_norm,
                        scratch_qkv + h * head_dim, head_dim, rms_norm_eps);
        }
    }
    cpu_rope(scratch_qkv, pos, n_heads, head_dim, sin_table, cos_table);

    if (attn_k_norm) {
        for (int h = 0; h < n_kv_heads; h++) {
            float* ks = scratch_qkv + n_heads * head_dim + h * head_dim;
            cpu_rmsnorm(ks, attn_k_norm, ks, head_dim, rms_norm_eps);
        }
    }
    cpu_rope(scratch_qkv + n_heads * head_dim, pos, n_kv_heads, head_dim, sin_table, cos_table);

    // ── Write KV cache ──
    for (int h = 0; h < n_kv_heads; h++) {
        std::memcpy(k_cache + pos * n_kv_heads * head_dim + h * head_dim,
                    scratch_qkv + n_heads * head_dim + h * head_dim,
                    head_dim * sizeof(float));
        std::memcpy(v_cache + pos * n_kv_heads * head_dim + h * head_dim,
                    scratch_qkv + n_heads * head_dim + n_kv_heads * head_dim + h * head_dim,
                    head_dim * sizeof(float));
    }

    // ── Attention ──
    cpu_attention(scratch_qkv, k_cache, v_cache, scratch_attn,
                  n_heads, n_kv_heads, head_dim, pos + 1, gqa_ratio);

    // ── O projection + residual ──
    cpu_ternary_gemv(o_packed, scratch_attn, o_scales, hidden, hidden_dim, n_heads * head_dim);
    for (int i = 0; i < hidden_dim; i++) {
        hidden[i] = residual[i] + hidden[i];
    }

    // ── Pre-FFN residual save + RMSNorm ──
    std::memcpy(residual, hidden, hidden_dim * sizeof(float));
    cpu_rmsnorm(hidden, pa_norm_weight, hidden, hidden_dim, rms_norm_eps);

    // ── FFN: gate/up ──
    cpu_ternary_gemv(gate_packed, hidden, gate_scales, scratch_ffn, inter_size, hidden_dim);
    cpu_ternary_gemv(up_packed,   hidden, up_scales,   scratch_ffn + inter_size, inter_size, hidden_dim);

    // ── SiLU GLU + down ──
    cpu_silu_glu(scratch_ffn, scratch_ffn + inter_size, scratch_act, inter_size);
    cpu_ternary_gemv(down_packed, scratch_act, down_scales, hidden, hidden_dim, inter_size);

    // ── Residual add (FFN) ──
    for (int i = 0; i < hidden_dim; i++) {
        hidden[i] = residual[i] + hidden[i];
    }

    // ── Final norm (only for last layer) ──
    if (final_norm_weight) {
        cpu_rmsnorm(hidden, final_norm_weight, hidden, hidden_dim, rms_norm_eps);
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════
// LM head
// ═══════════════════════════════════════════════════════════════════
void cpu_lm_head(
    const float* hidden,
    const float* embedding,
    float*       logits,
    int vocab_size, int hidden_dim)
{
    for (int n = 0; n < vocab_size; n++) {
        double dot = 0.0;
        const float* row = embedding + n * hidden_dim;
        for (int i = 0; i < hidden_dim; i++) {
            dot += (double)hidden[i] * (double)row[i];
        }
        logits[n] = (float)dot;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Embedding lookup
// ═══════════════════════════════════════════════════════════════════
void cpu_embed(
    const float* table,
    int token,
    float*       out,
    int hidden_dim)
{
    std::memcpy(out, table + token * hidden_dim, hidden_dim * sizeof(float));
}

// ═══════════════════════════════════════════════════════════════════
// Argmax
// ═══════════════════════════════════════════════════════════════════
int cpu_argmax(const float* values, int N) {
    int best = 0;
    float best_val = values[0];
    for (int i = 1; i < N; i++) {
        if (values[i] > best_val) {
            best_val = values[i];
            best = i;
        }
    }
    return best;
}

// ═══════════════════════════════════════════════════════════════════
// Softmax (in-place)
// ═══════════════════════════════════════════════════════════════════
void cpu_softmax(float* values, int N) {
    float max_val = -1e30f;
    for (int i = 0; i < N; i++) {
        if (values[i] > max_val) max_val = values[i];
    }
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        float e = std::exp(values[i] - max_val);
        values[i] = e;
        sum += (double)e;
    }
    float inv = (sum > 0.0) ? (float)(1.0 / sum) : 0.0f;
    for (int i = 0; i < N; i++) {
        values[i] *= inv;
    }
}
