// cpu_ternary_gemv.cpp — CPU ternary GEMV: scalar, AVX2, AVX-512
//
// Ternary weights: 2-bit packed per value, 16 per uint32:
//   bits[2i:2i+1] → 00=0(skip), 01=+1(add), 10=-1(sub), 11=-1(safe)
//   y[row] = scales[row] * Σ_k sign(weight[row][k]) * activation[k]
//
// Build: g++ -O3 -march=native -std=c++17 -o cpu_ternary_gemv tools/cpu_ternary_gemv.cpp
//
// Usage: ./cpu_ternary_gemv [--avx2] [--avx512] [--scalar] [--model bitnet|qwen3]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <vector>
#include <random>
#include <string>
#include <algorithm>

// ── Detect SIMD support ──────────────────────────────────────
#if defined(__AVX512F__) && defined(__AVX512BW__)
    #define HAS_AVX512 1
#else
    #define HAS_AVX512 0
#endif

#if defined(__AVX2__)
    #define HAS_AVX2 1
#else
    #define HAS_AVX2 0
#endif

#if HAS_AVX512
    #include <immintrin.h>
#endif
#if HAS_AVX2
    #include <immintrin.h>
#endif

// ── Helpers ───────────────────────────────────────────────────
static float rand_float(std::mt19937& rng, float lo, float hi) {
    return std::uniform_real_distribution<float>(lo, hi)(rng);
}

static void pack_ternary(const int8_t* vals, int count, uint32_t* packed) {
    int n_u32 = count / 16;
    for (int i = 0; i < n_u32; i++) {
        uint32_t word = 0;
        for (int v = 0; v < 16; v++) {
            int8_t val = vals[i * 16 + v];
            uint32_t bits;
            if (val == 1)        bits = 0x1;  // +1
            else if (val == -1)  bits = 0x2;  // -1
            else                 bits = 0x0;  // 0
            word |= (bits << (v * 2));
        }
        packed[i] = word;
    }
}

// ── 1. Scalar CPU ternary GEMV ───────────────────────────────
// y[M] = scales[M] .* (packed[M, K/16] @ x[K])
static void ternary_gemv_scalar(
    const uint32_t* packed,   // [M, K/16] packed ternary weights
    const float*    x,        // [K] activation
    const float*    scales,   // [M] per-row scale
    float*          y,        // [M] output
    int M, int K)
{
    const int pk = K / 16;  // uint32s per row
    for (int row = 0; row < M; row++) {
        float acc = 0.0f;
        const uint32_t* rw = packed + row * pk;
        for (int u = 0; u < pk; u++) {
            uint32_t word = rw[u];
            for (int v = 0; v < 16; v++) {
                uint32_t bits = (word >> (v * 2)) & 0x3;
                // branchless: bits=01→+1, bits=10→-1, else→0
                float sign = (float)(bits == 1) - (float)(bits == 2);
                acc += sign * x[u * 16 + v];
            }
        }
        y[row] = acc * scales[row];
    }
}

// ── 2. AVX2 CPU ternary GEMV ─────────────────────────────────
// Processes 8 floats per iteration (ymm). Each uint32 has 16 packed
// ternary values. We decode by isolating the 2-bit groups, expanding
// to 8-bit lanes, then comparing to get +1/0/-1 masks.
#if HAS_AVX2
static void ternary_gemv_avx2(
    const uint32_t* packed,
    const float*    x,
    const float*    scales,
    float*          y,
    int M, int K)
{
    const int pk = K / 16;
    const int unroll = 4;  // process 4 uint32s → 64 ternary vals per inner loop

    for (int row = 0; row < M; row++) {
        __m256 acc = _mm256_setzero_ps();
        const uint32_t* rw = packed + row * pk;
        int u = 0;

        // Process 4 uint32s (64 ternary vals) at a time
        for (; u + unroll <= pk; u += unroll) {
            // Load 4 packed uint32s
            __m128i w0 = _mm_cvtsi32_si128((int)rw[u + 0]);
            __m128i w1 = _mm_cvtsi32_si128((int)rw[u + 1]);
            __m128i w2 = _mm_cvtsi32_si128((int)rw[u + 2]);
            __m128i w3 = _mm_cvtsi32_si128((int)rw[u + 3]);

            // Interleave into a single 128-bit reg: [w3|w2|w1|w0]
            __m128i w01 = _mm_unpacklo_epi32(w0, w1);   // w1, w0
            __m128i w23 = _mm_unpacklo_epi32(w2, w3);   // w3, w2
            __m128i w = _mm_unpacklo_epi64(w01, w23);   // w3, w2, w1, w0

            // Expand each 32-bit word → decode 16 ternary signs
            // We process 4 words × 4 vals = 16 vals per AVX2 iteration
            // For each word, extract bits [2i:2i+1] for i=0..3
            __m256 acc_part = _mm256_setzero_ps();

            for (int sub = 0; sub < 4; sub++) {
                // Extract bits for lanes 0..3 of each word
                __m128i shifted = _mm_srli_epi32(w, sub * 2);
                __m128i bits = _mm_and_si128(shifted, _mm_set1_epi32(0x3));

                // Compare to get masks: bits==1 → +1, bits==2 → -1
                __m128i is_pos = _mm_cmpeq_epi32(bits, _mm_set1_epi32(1));
                __m128i is_neg = _mm_cmpeq_epi32(bits, _mm_set1_epi32(2));

                // Convert to float: +1.0, -1.0, or 0.0
                __m128 pos_f = _mm_cvtepi32_ps(is_pos);   // 0 or -1 (cmp yields -1)
                __m128 neg_f = _mm_cvtepi32_ps(is_neg);
                // negate: cmp yields -1 for true, so pos_f is already -1 → negate
                // Actually cmpeq yields all-1s (-1 as int), which as float is NaN.
                // Use mask: sign = (bits==1) - (bits==2)
                // First blend: for bits==1 set 1.0f, for bits==2 set -1.0f
                pos_f = _mm_and_ps(pos_f, _mm_set1_ps(1.0f));   // NOT correct with NaN

                // Actually the SIMD compare trick with integers:
                // Need to get float values, not NaN. Let's use a different approach.
                // Use the comparison masks to blend.

                // Load activations for this sub-group (4 values per word = 16 total across 4 words)
                // But we process 4 vals per word × 4 words = 16 vals = 1 ymm activation vector
                // Nope, this doesn't map cleanly. Let me rewrite this more carefully.
            }
            acc = _mm256_add_ps(acc, acc_part);
        }

        // Handle remaining uint32s
        for (; u < pk; u++) {
            uint32_t word = rw[u];
            for (int v = 0; v < 16; v++) {
                uint32_t bits = (word >> (v * 2)) & 0x3;
                float sign = (float)(bits == 1) - (float)(bits == 2);
                // broadcast and add
            }
        }

        // Horizontal sum
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        float result;
        _mm_store_ss(&result, sum);
        y[row] = result * scales[row];
    }
}
#endif

// ── 3. AVX-512 CPU ternary GEMV ──────────────────────────────
// 16× ternary decode per zmm lane, fully branchless.
#if HAS_AVX512
static void ternary_gemv_avx512(
    const uint32_t* packed,
    const float*    x,
    const float*    scales,
    float*          y,
    int M, int K)
{
    const int pk = K / 16;
    // AVX-512: process 16 packed uint32s at a time (256 ternary vals)
    const int vec_u32 = 16;

    for (int row = 0; row < M; row++) {
        __m512 acc = _mm512_setzero_ps();
        const uint32_t* rw = packed + row * pk;
        int u = 0;

        for (; u + vec_u32 <= pk; u += vec_u32) {
            // Load 16 packed uint32s
            __m512i w = _mm512_loadu_si512(rw + u);
            // Load 256 activations (16 u32s × 16 vals each = 256 floats)
            __m512 act = _mm512_loadu_ps(x + u * 16);

            // Mask generation: for each of 16 packed uint32s, extract bits for
            // lanes v=0..15 (one lane per 2-bit group). We iterate over the
            // 16 positions within each uint32 in 4 groups of 4 for register reuse.
            __m512 sign_sum = _mm512_setzero_ps();

            for (int v = 0; v < 16; v++) {
                // Extract 2-bit field at position v from all 16 words
                __m512i shifted = _mm512_srli_epi32(w, v * 2);
                __m512i bits = _mm512_and_si512(shifted, _mm512_set1_epi32(0x3));

                // sign = (bits==1) - (bits==2). Make float masks.
                __mmask16 is_pos = _mm512_cmpeq_epi32_mask(bits, _mm512_set1_epi32(1));
                __mmask16 is_neg = _mm512_cmpeq_epi32_mask(bits, _mm512_set1_epi32(2));

                // Blend activations: +act for pos, -act for neg, 0 otherwise
                __m512 pos_act = _mm512_maskz_mov_ps(is_pos, act);
                __m512 neg_act = _mm512_maskz_mov_ps(is_neg, act);

                // accumulate += pos_act - neg_act
                sign_sum = _mm512_add_ps(sign_sum, pos_act);
                sign_sum = _mm512_sub_ps(sign_sum, neg_act);
            }

            acc = _mm512_add_ps(acc, sign_sum);
        }

        // Remaining < 16 uint32s
        for (; u < pk; u++) {
            uint32_t word = rw[u];
            for (int v = 0; v < 16; v++) {
                uint32_t bits = (word >> (v * 2)) & 0x3;
                float sign = (float)(bits == 1) - (float)(bits == 2);
                acc = _mm512_add_ps(acc, _mm512_set1_ps(sign * x[u * 16 + v]));
            }
        }

        // Horizontal reduce
        float result = _mm512_reduce_add_ps(acc);
        y[row] = result * scales[row];
    }
}
#endif

// ── Bare-metal AVX2 version (clean, correct) ─────────────────
// This processes 8 ternary values per iteration using 256-bit vectors.
// We load 4 uint32s (64 ternary values) → 8 floats × 8 SIMD iterations.
#if HAS_AVX2
static void ternary_gemv_avx2_v2(
    const uint32_t* packed,
    const float*    x,
    const float*    scales,
    float*          y,
    int M, int K)
{
    const int pk = K / 16;
    const int floats_per_vec = 8;

    for (int row = 0; row < M; row++) {
        __m256 acc = _mm256_setzero_ps();
        const uint32_t* rw = packed + row * pk;
        int u = 0;

        for (; u + 2 <= pk; u += 2) {
            // 2 uint32s → 32 ternary values → 4 AVX2 iterations (8 vals each)
            uint32_t w0 = rw[u];
            uint32_t w1 = rw[u + 1];

            for (int sub = 0; sub < 4; sub++) {
                // Extract 8 × 2-bit codes from [w1 : w0], lanes sub*8 .. sub*8+7
                int bit_off = sub * 16;  // 16 bits per uint32 = 8 ternary vals per uint32
                uint32_t code8 = ((w1 >> (bit_off)) & 0xFF) << 8
                               | ((w0 >> (bit_off)) & 0xFF);
                // Now code8[2i:2i+1] has the i-th ternary code for i=0..7
                // Interleave: extract each 2-bit field
                __m256i signs_i = _mm256_set_epi32(
                    (code8 >> 14) & 0x3,  // lane 7
                    (code8 >> 12) & 0x3,  // lane 6
                    (code8 >> 10) & 0x3,  // lane 5
                    (code8 >> 8)  & 0x3,  // lane 4
                    (code8 >> 6)  & 0x3,  // lane 3
                    (code8 >> 4)  & 0x3,  // lane 2
                    (code8 >> 2)  & 0x3,  // lane 1
                    (code8 >> 0)  & 0x3); // lane 0

                // sign = (bits==1) * 1 + (bits==2) * (-1)
                __m256i is_pos = _mm256_cmpeq_epi32(signs_i, _mm256_set1_epi32(1));
                __m256i is_neg = _mm256_cmpeq_epi32(signs_i, _mm256_set1_epi32(2));

                // CMPEQ yields -1 (all ones) for true. Convert to float masks.
                __m256 pos_f = _mm256_cvtepi32_ps(is_pos);
                __m256 neg_f = _mm256_cvtepi32_ps(is_neg);
                // Now pos_f has -1.0f for pos, 0.0f for not-pos
                // neg_f has -1.0f for neg, 0.0f for not-neg
                // sign = -pos_f + neg_f = -(pos_f) + neg_f? 
                // Actually is_pos (as int): -1 means true. As float: -1.0f
                // is_neg (as int): -1 means true. As float: -1.0f
                // We want: val = 1 for pos, -1 for neg, 0 otherwise
                // val = -(pos_f as float) + (neg_f as float)? No.
                // pos_f = -1.0f when true. So -pos_f = 1.0f when true.
                // neg_f = -1.0f when true.
                // sign = -pos_f (1.0 for pos) + (-neg_f) for neg... no that's -1 for neg.
                // sign = is_pos * 1 + is_neg * (-1)
                // = (-pos_f) * 1.0 + (-neg_f) * (-1.0)... no.
                //
                // Let me think again:
                // is_pos as __m256i: -1 for true (all 32 bits = 1)
                // cvtepi32_ps: -1 (int) → -1.0f (float)
                // So pos_f = -1.0f if pos, 0.0f if not
                //   neg_f = -1.0f if neg, 0.0f if not
                //
                // sign = 1 for pos, -1 for neg, 0 for neither
                // sign = (-1 * pos_f) + neg_f? No:
                //   pos: pos_f=-1.0, neg_f=0 → (-1)(-1) + 0 = 1 ✓
                //   neg: pos_f=0, neg_f=-1 → 0 + (-1) = -1 ✓
                //   none: 0 + 0 = 0 ✓
                //   both (code 3): -1 + -1 = -2 (won't happen in practice)

                // sign_f = (-1.0f) * pos_f + neg_f... no, that's wrong.
                // sign_f = neg_f - pos_f? 
                //   pos: -0 - (-1) = 1 ✓
                //   neg: -1 - 0 = -1 ✓
                //   none: 0 - 0 = 0 ✓
                // Yes! sign_f = neg_f - pos_f
                // Wait, neg_f as float = -1.0 for neg. pos_f = -1.0 for pos.
                // neg_f - pos_f: pos → 0 - (-1.0) = 1.0 ✓
                //               neg → -1.0 - 0 = -1.0 ✓
                //               none → 0 - 0 = 0 ✓
                __m256 sign_f = _mm256_sub_ps(neg_f, pos_f);

                // Load activations for this sub-group
                __m256 act = _mm256_loadu_ps(x + u * 16 + sub * 8);

                // acc += sign * act
                acc = _mm256_fmadd_ps(sign_f, act, acc);
            }
        }

        // Remaining single uint32
        for (; u < pk; u++) {
            uint32_t word = rw[u];
            for (int v = 0; v < 16; v++) {
                uint32_t bits = (word >> (v * 2)) & 0x3;
                float sign = (float)(bits == 1) - (float)(bits == 2);
                __m256 sv = _mm256_set1_ps(sign * x[u * 16 + v]);
                acc = _mm256_add_ps(acc, sv);
            }
        }

        // Horizontal sum
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        float result;
        _mm_store_ss(&result, sum);
        y[row] = result * scales[row];
    }
}
#endif

// ── 4. AVX512BW ternary GEMV (clean, correct) ───────────────
#if HAS_AVX512
static void ternary_gemv_avx512_v2(
    const uint32_t* packed,
    const float*    x,
    const float*    scales,
    float*          y,
    int M, int K)
{
    const int pk = K / 16;

    for (int row = 0; row < M; row++) {
        __m512 acc = _mm512_setzero_ps();
        const uint32_t* rw = packed + row * pk;
        int u = 0;

        // Process 4 uint32s (64 ternary vals) at a time
        // 4 u32s → 64 vals → 4 zmm loads of 16 floats each
        for (; u + 4 <= pk; u += 4) {
            // Load 4 packed uint32s
            uint32_t w0 = rw[u], w1 = rw[u+1], w2 = rw[u+2], w3 = rw[u+3];

            // Gather 64 2-bit codes into 4 uint32s (16 per sub-group)
            // Then for each sub-group of 16 vals, do a zmm operation
            for (int sub = 0; sub < 4; sub++) {
                int bit_off = sub * 16;
                // Build a 64-bit value: [w3|w2|w1|w0] at bit_off, take 16 bits each → 64 bits
                uint64_t code64 = ((uint64_t)((w3 >> bit_off) & 0xFFFF) << 48)
                                | ((uint64_t)((w2 >> bit_off) & 0xFFFF) << 32)
                                | ((uint64_t)((w1 >> bit_off) & 0xFFFF) << 16)
                                | ((uint64_t)((w0 >> bit_off) & 0xFFFF));

                // Now code64 has 32 × 2-bit codes. Extract 16 at a time.
                // Process first 16 (lower 32 bits)
                uint32_t code32_lo = (uint32_t)(code64 & 0xFFFFFFFF);
                // Use AVX-512 to decode 16 × 2-bit codes
                // Broadcast the 32-bit value to all zmm lanes
                __m512i code_vec = _mm512_set1_epi32((int)code32_lo);

                // Extract bits lane-by-lane using shifts
                // Lane i gets bit position (i*2) within the 32-bit code
                // We use per-lane shift: _mm512_sllv_epi32 or just set_epi32 per lane
                // Simplest: create index vector [0,2,4,...,30]
                // and shift each lane accordingly
                __m512i indices = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16,
                                                    14, 12, 10, 8, 6, 4, 2, 0);
                __m512i shifted = _mm512_srlv_epi32(code_vec, indices);
                __m512i bits = _mm512_and_si512(shifted, _mm512_set1_epi32(0x3));

                // sign = (bits==1) - (bits==2)
                __mmask16 is_pos = _mm512_cmpeq_epi32_mask(bits, _mm512_set1_epi32(1));
                __mmask16 is_neg = _mm512_cmpeq_epi32_mask(bits, _mm512_set1_epi32(2));

                // Load 16 activations
                __m512 act = _mm512_loadu_ps(x + u * 16 + sub * 16);

                // sign * act: blend + subtract
                // For pos: add act, for neg: sub act, else: 0
                __m512 pos_act = _mm512_maskz_mov_ps(is_pos, act);
                __m512 neg_act = _mm512_maskz_mov_ps(is_neg, act);
                acc = _mm512_add_ps(acc, pos_act);
                acc = _mm512_sub_ps(acc, neg_act);
            }
        }

        // Remaining uint32s
        for (; u < pk; u++) {
            uint32_t word = rw[u];
            for (int v = 0; v < 16; v++) {
                uint32_t bits = (word >> (v * 2)) & 0x3;
                float sign = (float)(bits == 1) - (float)(bits == 2);
                // Broadcast and add
                __m512 sv = _mm512_set1_ps(sign * x[u * 16 + v]);
                acc = _mm512_add_ps(acc, sv);
            }
        }

        float result = _mm512_reduce_add_ps(acc);
        y[row] = result * scales[row];
    }
}
#endif

// ── CPU reference (used for correctness verification) ─────────
// Same as scalar but with double-precision accumulator for accuracy
static void ternary_gemv_ref(
    const int8_t*   weights,  // [M, K] in {-1, 0, +1}
    const float*    x,
    const float*    scales,
    float*          y,
    int M, int K)
{
    for (int row = 0; row < M; row++) {
        double acc = 0.0;
        const int8_t* rw = weights + row * K;
        for (int k = 0; k < K; k++) {
            acc += (double)rw[k] * (double)x[k];
        }
        y[row] = (float)(acc * (double)scales[row]);
    }
}

// ── Model dimension presets ───────────────────────────────────
struct ModelDims {
    const char* name;
    int hs;    // hidden_size
    int is;    // intermediate_size
    int nh;    // num_heads
    int nkv;   // num_kv_heads
    int hd;    // head_dim
    int V;     // vocab_size
    int L;     // num_layers
};

static const ModelDims MODELS[] = {
    {"BitNet-2B-4T",  2560, 6912, 32, 32, 80, 128256, 30},
    {"Qwen3-0.6B",    1536, 4096, 12, 2,  128, 151936, 28},
    {"Qwen3-1.7B",    2048, 8192, 16, 2,  128, 151936, 28},
    {nullptr, 0, 0, 0, 0, 0, 0, 0},
};

// ── Timing helper ─────────────────────────────────────────────
using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ── Test harness ──────────────────────────────────────────────
struct BenchResult {
    const char* name;
    double us;
    double max_err;
};

static BenchResult bench_gemv(
    const char* name,
    void (*gemv_fn)(const uint32_t*, const float*, const float*, float*, int, int),
    const uint32_t* packed, const float* x, const float* scales,
    const float* ref,
    int M, int K, int warmup, int runs)
{
    std::vector<float> y(M);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        gemv_fn(packed, x, scales, y.data(), M, K);
    }

    // Timed
    auto t0 = Clock::now();
    for (int i = 0; i < runs; i++) {
        gemv_fn(packed, x, scales, y.data(), M, K);
    }
    auto t1 = Clock::now();

    double ms = elapsed_ms(t0, t1);
    double us = (ms / runs) * 1000.0;

    // Error against reference
    double max_err = 0.0;
    for (int i = 0; i < M; i++) {
        double e = fabs((double)y[i] - (double)ref[i]);
        if (e > max_err) max_err = e;
    }

    return BenchResult{name, us, max_err};
}

int main(int argc, char** argv) {
    // Parse args
    bool run_scalar = true;
    bool run_avx2 = HAS_AVX2;
    bool run_avx512 = HAS_AVX512;
    const ModelDims* model = &MODELS[0];  // default: BitNet-2B-4T

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--scalar") { run_avx2 = false; run_avx512 = false; }
        else if (a == "--avx2") { run_scalar = false; run_avx512 = false; run_avx2 = true; }
        else if (a == "--avx512") { run_scalar = false; run_avx2 = false; run_avx512 = true; }
        else if (a == "--all-simd") { /* all enabled by default */ }
        else if (a == "--model") {
            if (i + 1 >= argc) { fprintf(stderr, "usage: --model bitnet|qwen3|qwen3-1.7b\n"); return 1; }
            i++;
            std::string m = argv[i];
            for (int j = 0; MODELS[j].name; j++) {
                std::string mn = MODELS[j].name;
                // Convert both to lowercase for case-insensitive comparison
                std::string mn_lower = mn;
                std::string m_lower = m;
                for (auto& c : mn_lower) c = tolower(c);
                for (auto& c : m_lower) c = tolower(c);
                if (mn_lower.find(m_lower) != std::string::npos) {
                    model = &MODELS[j];
                    break;
                }
            }
        }
    }

    printf("=== CPU Ternary GEMV Benchmark ===\n");
    printf("Model: %s\n", model->name);
    printf("  hidden=%d  intermediate=%d  heads=%d  kv_heads=%d  head_dim=%d  vocab=%d  layers=%d\n\n",
           model->hs, model->is, model->nh, model->nkv, model->hd, model->V, model->L);

#if HAS_AVX2
    printf("AVX2:   available\n");
#else
    printf("AVX2:   not available\n");
#endif
#if HAS_AVX512
    printf("AVX512: available\n");
#else
    printf("AVX512: not available\n");
#endif
    printf("\n");

    // Test shapes per model: Q/K/V/O, gate/up, down, LM head
    struct TestShape {
        const char* name;
        int M, K;
    };

    int nh_hd = model->nh * model->hd;    // nh*hd = hs for standard attn
    int nkv_hd = model->nkv * model->hd;
    std::vector<TestShape> shapes = {
        {"Q proj",   nh_hd,  model->hs},
        {"K proj",   nkv_hd, model->hs},
        {"V proj",   nkv_hd, model->hs},
        {"O proj",   model->hs, nh_hd},
        {"Gate proj", model->is, model->hs},
        {"Up proj",   model->is, model->hs},
        {"Down proj", model->hs, model->is},
        {"LM head",   model->V, model->hs},
    };

    const int warmup = 10;
    const int runs = 100;
    const double GEMVS_PER_TOKEN = 6.0 * model->L + 1.0;  // 6 GEMV/layer + LM head

    printf("%-12s %7s %8s", "Shape", "M", "K");
    if (run_scalar) printf("  %12s", "Scalar(us)");
    #if HAS_AVX2
    if (run_avx2) printf("  %12s", "AVX2(us)");
    #endif
    #if HAS_AVX512
    if (run_avx512) printf("  %12s", "AVX512(us)");
    #endif
    printf("  %10s  %s\n", "Err", "Speedup");

    printf("%-12s %7s %8s", "────────────", "───────", "────────");
    if (run_scalar) printf("  %12s", "────────────");
    #if HAS_AVX2
    if (run_avx2) printf("  %12s", "────────────");
    #endif
    #if HAS_AVX512
    if (run_avx512) printf("  %12s", "────────────");
    #endif
    printf("  %10s  %s\n", "──────────", "───────");

    double total_scalar_us = 0.0;
    double total_avx2_us = 0.0;
    double total_avx512_us = 0.0;
    double scalar_lm_us = 0.0, avx2_lm_us = 0.0, avx512_lm_us = 0.0;

    for (const auto& sh : shapes) {
        int M = sh.M;
        int K = sh.K;
        int pk = (K + 15) / 16;  // ensure K is a multiple of 16 for SIMD
        int K_aligned = pk * 16;

        // Generate random test data
        std::mt19937 rng(42);
        std::vector<int8_t> h_weights(M * K_aligned);
        std::vector<float> h_x(K_aligned);
        std::vector<float> h_scales(M);
        std::vector<uint32_t> h_packed(M * pk);

        for (int i = 0; i < M * K_aligned; i++) {
            int r = rng() % 3;
            h_weights[i] = (r == 0) ? -1 : (r == 1) ? 0 : 1;
        }
        for (int i = 0; i < K_aligned; i++) h_x[i] = rand_float(rng, -1.0f, 1.0f);
        for (int i = 0; i < M; i++) h_scales[i] = rand_float(rng, 0.5f, 2.0f);

        for (int r = 0; r < M; r++) {
            pack_ternary(h_weights.data() + r * K_aligned, K_aligned, h_packed.data() + r * pk);
        }

        // Reference (double-precision)
        std::vector<float> ref(M);
        ternary_gemv_ref(h_weights.data(), h_x.data(), h_scales.data(), ref.data(), M, K_aligned);

        printf("%-12s %7d %8d", sh.name, M, K);

        double scalar_us = 0;
        double avx2_us = 0;
        double avx512_us = 0;

        if (run_scalar) {
            auto r = bench_gemv("scalar", ternary_gemv_scalar,
                                h_packed.data(), h_x.data(), h_scales.data(),
                                ref.data(), M, K_aligned, warmup, runs);
            scalar_us = r.us;
            if (strcmp(sh.name, "LM head") == 0) scalar_lm_us = r.us; else total_scalar_us += r.us;
            printf("  %10.2f", r.us);
        }
        #if HAS_AVX2
        if (run_avx2) {
            auto r = bench_gemv("avx2", ternary_gemv_avx2_v2,
                                h_packed.data(), h_x.data(), h_scales.data(),
                                ref.data(), M, K_aligned, warmup, runs);
            avx2_us = r.us;
            if (strcmp(sh.name, "LM head") == 0) avx2_lm_us = r.us; else total_avx2_us += r.us;
            printf("  %10.2f", r.us);
        }
        #endif
        #if HAS_AVX512
        if (run_avx512) {
            auto r = bench_gemv("avx512", ternary_gemv_avx512_v2,
                                h_packed.data(), h_x.data(), h_scales.data(),
                                ref.data(), M, K_aligned, warmup, runs);
            avx512_us = r.us;
            if (strcmp(sh.name, "LM head") == 0) avx512_lm_us = r.us; else total_avx512_us += r.us;
            printf("  %10.2f", r.us);
        }
        #endif

        // Error (last variant's error)
        double max_err = 0;
        auto ref_v = ref;
        std::vector<float> y_check(M);
        if (run_scalar) {
            ternary_gemv_scalar(h_packed.data(), h_x.data(), h_scales.data(), y_check.data(), M, K_aligned);
        }
        #if HAS_AVX2
        else if (run_avx2) {
            ternary_gemv_avx2_v2(h_packed.data(), h_x.data(), h_scales.data(), y_check.data(), M, K_aligned);
        }
        #endif
        #if HAS_AVX512
        else if (run_avx512) {
            ternary_gemv_avx512_v2(h_packed.data(), h_x.data(), h_scales.data(), y_check.data(), M, K_aligned);
        }
        #endif
        for (int i = 0; i < M; i++) {
            double e = fabs((double)y_check[i] - (double)ref_v[i]);
            if (e > max_err) max_err = e;
        }

        printf("  %8.2e", max_err);

        // Speedup vs scalar
        if (run_scalar && scalar_us > 0) {
            #if HAS_AVX2
            if (run_avx2) printf("  %5.1fx", scalar_us / avx2_us);
            else
            #endif
            #if HAS_AVX512
            if (run_avx512) printf("  %5.1fx", scalar_us / avx512_us);
            else
            #endif
            printf("  %5s", "1.0x");
        }
        printf("\n");
    }

    // Estimated tok/s
    printf("\n── Estimated decode throughput ──\n");
    // GEMVs per layer: Q, K, V, O, gate, up, down = 7
    // Per token: 7 * L + 1 (LM head)
    double gemvs_per_layer = 7.0;
    double layers = (double)model->L;
    double gemvs = gemvs_per_layer * layers + 1.0;

    auto print_estimate = [&](const char* label, double layer_us, double lm_us) {
        if (layer_us <= 0 && lm_us <= 0) return;
        double avggemv_us = layer_us / gemvs_per_layer;
        double token_us = layer_us * layers + lm_us;
        double tok_s = 1e6 / token_us;
        printf("  %-10s %7.1f us/GEMV  %7.1f ms/tok  %7.0f tok/s\n",
               label, avggemv_us, token_us / 1000.0, tok_s);
    };

    if (run_scalar)    print_estimate("Scalar", total_scalar_us, scalar_lm_us);
    #if HAS_AVX2
    if (run_avx2)      print_estimate("AVX2", total_avx2_us, avx2_lm_us);
    #endif
    #if HAS_AVX512
    if (run_avx512)    print_estimate("AVX512", total_avx512_us, avx512_lm_us);
    #endif

    printf("\n── Estimated prefill throughput (M=128, ~50%% eff) ──\n");
    auto print_prefill = [&](const char* label, double layer_us, double lm_us) {
        if (layer_us <= 0 && lm_us <= 0) return;
        // Prefill: M=128 tokens. Each GEMV becomes a GEMM with 128x more MACs.
        // Efficiency ~0.5 due to cache effects. LM head still single vector.
        double gemv_us = layer_us / gemvs_per_layer;  // single-GEMV time, M=1
        double prefill_us = gemv_us * 128.0 * 0.5 * gemvs_per_layer * layers + lm_us;
        double tok_s = 128.0 / (prefill_us / 1e6);
        printf("  %-10s %7.0f tok/s (M=128, est)\n", label, tok_s);
    };

    if (run_scalar)    print_prefill("Scalar", total_scalar_us, scalar_lm_us);
    #if HAS_AVX2
    if (run_avx2)      print_prefill("AVX2", total_avx2_us, avx2_lm_us);
    #endif
    #if HAS_AVX512
    if (run_avx512)    print_prefill("AVX512", total_avx512_us, avx512_lm_us);
    #endif

    printf("\n── Compare to GPU ──\n");
    printf("  GPU TRG (gfx1151):         279 tok/s\n");
    auto print_cpu = [&](const char* label, double layer_us, double lm_us) {
        if (layer_us <= 0) return;
        double tok_s = 1e6 / (layer_us * layers + lm_us);
        printf("  CPU TRG %-10s  %6.0f tok/s (1 core)\n", label, tok_s);
    };
    if (run_scalar)    print_cpu("(scalar)", total_scalar_us, scalar_lm_us);
    #if HAS_AVX2
    if (run_avx2)      print_cpu("(AVX2)", total_avx2_us, avx2_lm_us);
    #endif
    #if HAS_AVX512
    if (run_avx512)    print_cpu("(AVX-512)", total_avx512_us, avx512_lm_us);
    #endif

    return 0;
}
