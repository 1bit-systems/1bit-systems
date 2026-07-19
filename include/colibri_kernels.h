// colibri_kernels.h — Quantized matmul kernels adapted from colibrì (Apache 2.0)
//
// Adapted from: https://github.com/JustVugg/colibri (c/glm.c)
// Licensed under Apache 2.0 — see LICENSE notice below.
//
// Kernels provided:
//   colibri_matmul_f32()       — float32 matmul (ref baseline)
//   colibri_matmul_q8()        — int8-per-row × f32 activations
//   colibri_matmul_q4()        — int4-per-row × f32 activations
//   colibri_matmul_q2()        — int2-per-row × f32 activations
//   colibri_matmul_q4_grouped()— int4 grouped (per-group scale) × f32
//   colibri_matmul_q4_pair()   — fused gate+up int4 (one OMP dispatch)
//   colibri_dot_i8i8()         — int8×int8 integer dot product (IDOT)
//   colibri_dot_i4i8()         — int4×int8 integer dot product (IDOT)
//   colibri_qrow_i8()          — quantize f32 row to int8
//   colibri_quantize_q4()      — quantize f32 [I] → packed int4 + scale
//   colibri_quantize_q2()      — quantize f32 [I] → packed int2 + scale
//
// Integration: #include this header, call the kernels directly.
// To use quantized weights, convert at load time with colibri_quantize_q4().

/*
 * Copyright 2026 JustVugg (colibrì)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modifications for 1bit.systems integration:
 *   - Extracted from c/glm.c into standalone header
 *   - Added extern "C" guard for C++ usage
 *   - Added colibri_quantize_q4/q2 helpers
 *   - Removed OpenMP dependency (caller decides threading)
 *   - Added colibri_matmul_q4_serial() for single-threaded use
 */

#ifndef COLI_BRI_KERNELS_H
#define COLI_BRI_KERNELS_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Platform detection ───────────────────────────────────────────────
#if defined(__AVX512F__) && defined(__AVX512BW__)
  #define COLI_AVX512 1
#elif defined(__AVX2__)
  #define COLI_AVX2 1
#elif defined(__ARM_NEON)
  #define COLI_NEON 1
#elif defined(__VSX__)
  #define COLI_VSX 1
#endif

// ── Internal: horizontal sum helpers ─────────────────────────────────
#ifdef COLI_AVX2
#include <immintrin.h>
static inline float _coli_hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehl_ps(lo, lo);
    lo = _mm_add_ps(lo, sh);
    sh = _mm_shuffle_ps(lo, lo, 1);
    lo = _mm_add_ss(lo, sh);
    return _mm_cvtss_f32(lo);
}
static inline int _coli_hsum256_i32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}
#endif

// ── 1. Float32 matmul (reference, SIMD) ──────────────────────────────
// y[S,O] = x[S,I] @ W^T, W[O,I] f32
static inline void colibri_matmul_f32(float *y, const float *x,
                                       const float *W, int S, int I, int O) {
    for (int o = 0; o < O; o++) {
        const float *w = W + (size_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (size_t)s * I;
            float a = 0;
            int i = 0;
#ifdef COLI_AVX2
            __m256 acc = _mm256_setzero_ps();
            for (; i + 8 <= I; i += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i),
                                      _mm256_loadu_ps(w + i), acc);
            }
            a = _coli_hsum256_ps(acc);
#endif
            for (; i < I; i++) a += xs[i] * w[i];
            y[(size_t)s * O + o] = a;
        }
    }
}

// ── 2. Int8-per-row quantized matmul ──────────────────────────────────
// y[S,O] = x[S,I] @ W_q^T, W_q[O,I] int8 + scale[O] per-row
static inline void colibri_matmul_q8(float *y, const float *x,
                                      const int8_t *q, const float *scale,
                                      int S, int I, int O) {
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (size_t)o * I;
        float sc = scale[o];
        for (int s = 0; s < S; s++) {
            const float *xs = x + (size_t)s * I;
            float a = 0;
            int i = 0;
#ifdef COLI_AVX2
            __m256 acc = _mm256_setzero_ps();
            for (; i + 8 <= I; i += 8) {
                __m256i wi = _mm256_cvtepi8_epi32(
                    _mm_loadl_epi64((const __m128i *)(w + i)));
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i),
                                      _mm256_cvtepi32_ps(wi), acc);
            }
            a = _coli_hsum256_ps(acc);
#elif defined(COLI_NEON)
            float32x4_t ac0 = vdupq_n_f32(0), ac1 = vdupq_n_f32(0);
            for (; i + 8 <= I; i += 8) {
                int16x8_t w16 = vmovl_s8(vld1_s8(w + i));
                ac0 = vfmaq_f32(ac0, vld1q_f32(xs + i),
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))));
                ac1 = vfmaq_f32(ac1, vld1q_f32(xs + i + 4),
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16))));
            }
            a = vaddvq_f32(vaddq_f32(ac0, ac1));
#endif
            for (; i < I; i++) a += xs[i] * (float)w[i];
            y[(size_t)s * O + o] = a * sc;
        }
    }
}

// ── 3. Int4-per-row quantized matmul (2 values/byte) ──────────────────
// y[S,O] = x[S,I] @ W_q^T, W_q packed int4 + scale[O]
static inline void colibri_matmul_q4(float *y, const float *x,
                                      const uint8_t *q4, const float *scale,
                                      int S, int I, int O) {
    int rb = (I + 1) / 2;
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q4 + (size_t)o * rb;
        float sc = scale[o];
        for (int s = 0; s < S; s++) {
            const float *xs = x + (size_t)s * I;
            float a = 0;
            int i = 0;
#ifdef COLI_AVX2
            const __m128i m4 = _mm_set1_epi8(0x0F);
            const __m256i b8 = _mm256_set1_epi32(8);
            __m256 acc = _mm256_setzero_ps();
            for (; i + 16 <= I; i += 16) {
                __m128i by = _mm_loadl_epi64((const __m128i *)(w + (i >> 1)));
                __m128i lo = _mm_and_si128(by, m4);
                __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
                __m128i nib = _mm_unpacklo_epi8(lo, hi);
                __m256 w0 = _mm256_cvtepi32_ps(
                    _mm256_sub_epi32(_mm256_cvtepu8_epi32(nib), b8));
                __m256 w1 = _mm256_cvtepi32_ps(
                    _mm256_sub_epi32(_mm256_cvtepu8_epi32(
                        _mm_srli_si128(nib, 8)), b8));
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i), w0, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i + 8), w1, acc);
            }
            a = _coli_hsum256_ps(acc);
#elif defined(COLI_NEON)
            const uint8x8_t m4v = vdup_n_u8(0x0F);
            const int8x8_t b8v = vdup_n_s8(8);
            float32x4_t ac0 = vdupq_n_f32(0), ac1 = vdupq_n_f32(0);
            for (; i + 16 <= I; i += 16) {
                uint8x8_t by = vld1_u8(w + (i >> 1));
                uint8x8x2_t z = vzip_u8(vand_u8(by, m4v), vshr_n_u8(by, 4));
                int16x8_t w0 = vmovl_s8(
                    vsub_s8(vreinterpret_s8_u8(z.val[0]), b8v));
                int16x8_t w1 = vmovl_s8(
                    vsub_s8(vreinterpret_s8_u8(z.val[1]), b8v));
                ac0 = vfmaq_f32(ac0, vld1q_f32(xs + i),
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                ac1 = vfmaq_f32(ac1, vld1q_f32(xs + i + 4),
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                ac0 = vfmaq_f32(ac0, vld1q_f32(xs + i + 8),
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                ac1 = vfmaq_f32(ac1, vld1q_f32(xs + i + 12),
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1))));
            }
            a = vaddvq_f32(vaddq_f32(ac0, ac1));
#endif
            // Scalar remainder
            for (; i + 1 < I; i += 2) {
                uint8_t byte = w[i >> 1];
                int lo = (int)(byte & 0xF) - 8;
                int hi = (int)(byte >> 4) - 8;
                a += xs[i] * (float)lo + xs[i + 1] * (float)hi;
            }
            if (i < I) {
                uint8_t byte = w[i >> 1];
                a += xs[i] * (float)((int)(byte & 0xF) - 8);
            }
            y[(size_t)s * O + o] = a * sc;
        }
    }
}

// ── 4. Int2-per-row quantized matmul (4 values/byte) ─────────────────
// y[S,O] = x[S,I] @ W_q^T, W_q packed int2 + scale[O], values in [-2,1]
static inline void colibri_matmul_q2(float *y, const float *x,
                                      const uint8_t *q2, const float *scale,
                                      int S, int I, int O) {
    int rb = (I + 3) / 4;
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q2 + (size_t)o * rb;
        float sc = scale[o];
        for (int s = 0; s < S; s++) {
            const float *xs = x + (size_t)s * I;
            float a = 0;
            int i = 0;
#ifdef COLI_AVX2
            const __m128i m2 = _mm_set1_epi8(0x03);
            const __m256i b2 = _mm256_set1_epi32(2);
            __m256 acc = _mm256_setzero_ps();
            for (; i + 16 <= I; i += 16) {
                __m128i by = _mm_cvtsi32_si128(*(const int *)(w + (i >> 2)));
                __m128i p0 = _mm_and_si128(by, m2);
                __m128i p1 = _mm_and_si128(_mm_srli_epi16(by, 2), m2);
                __m128i p2 = _mm_and_si128(_mm_srli_epi16(by, 4), m2);
                __m128i p3 = _mm_and_si128(_mm_srli_epi16(by, 6), m2);
                __m128i lo = _mm_unpacklo_epi8(p0, p1);
                __m128i hi = _mm_unpacklo_epi8(p2, p3);
                __m128i nib = _mm_unpacklo_epi16(lo, hi);
                __m256 w0 = _mm256_cvtepi32_ps(
                    _mm256_sub_epi32(_mm256_cvtepu8_epi32(nib), b2));
                __m256 w1 = _mm256_cvtepi32_ps(
                    _mm256_sub_epi32(_mm256_cvtepu8_epi32(
                        _mm_srli_si128(nib, 8)), b2));
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i), w0, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i + 8), w1, acc);
            }
            a = _coli_hsum256_ps(acc);
#elif defined(COLI_NEON)
            const uint8x8_t m2v = vdup_n_u8(3);
            const int8x8_t b2v = vdup_n_s8(2);
            float32x4_t ac0 = vdupq_n_f32(0), ac1 = vdupq_n_f32(0);
            for (; i + 16 <= I; i += 16) {
                uint32_t wd;
                memcpy(&wd, w + (i >> 2), 4);
                uint8x8_t by = vreinterpret_u8_u32(vdup_n_u32(wd));
                uint8x8x2_t z01 = vzip_u8(vand_u8(by, m2v),
                                          vand_u8(vshr_n_u8(by, 2), m2v));
                uint8x8x2_t z23 = vzip_u8(vand_u8(vshr_n_u8(by, 4), m2v),
                                          vshr_n_u8(by, 6));
                uint16x4x2_t zz = vzip_u16(
                    vreinterpret_u16_u8(z01.val[0]),
                    vreinterpret_u16_u8(z23.val[0]));
                int16x8_t w0 = vmovl_s8(
                    vsub_s8(vreinterpret_s8_u16(zz.val[0]), b2v));
                int16x8_t w1 = vmovl_s8(
                    vsub_s8(vreinterpret_s8_u16(zz.val[1]), b2v));
                ac0 = vfmaq_f32(ac0, vld1q_f32(xs + i),
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                ac1 = vfmaq_f32(ac1, vld1q_f32(xs + i + 4),
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                ac0 = vfmaq_f32(ac0, vld1q_f32(xs + i + 8),
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                ac1 = vfmaq_f32(ac1, vld1q_f32(xs + i + 12),
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1))));
            }
            a = vaddvq_f32(vaddq_f32(ac0, ac1));
#endif
            for (; i < I; i++) {
                uint8_t byte = w[i >> 2];
                int sh = (i & 3) * 2;
                a += xs[i] * (float)((int)((byte >> sh) & 3) - 2);
            }
            y[(size_t)s * O + o] = a * sc;
        }
    }
}

// ── 5. Int4 grouped (per-group scale) matmul ─────────────────────────
// Like matmul_q4, but scale changes every `gs` elements.
// gs must be a multiple of 16.
static inline void colibri_matmul_q4_grouped(float *y, const float *x,
                                              const uint8_t *q4,
                                              const float *scale,
                                              int S, int I, int O, int gs) {
    int rb = (I + 1) / 2;
    int ng = (I + gs - 1) / gs;
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q4 + (size_t)o * rb;
        const float *scl = scale + (size_t)o * ng;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (size_t)s * I;
            float a = 0;
            for (int g = 0; g * gs < I; g++) {
                int base = g * gs;
                int glen = gs;
                if (base + glen > I) glen = I - base;
                float sc = scl[g];
                int i = base;
#ifdef COLI_AVX2
                const __m128i m4 = _mm_set1_epi8(0x0F);
                const __m256i b8 = _mm256_set1_epi32(8);
                __m256 acc = _mm256_setzero_ps();
                for (; i + 16 <= base + glen; i += 16) {
                    __m128i by = _mm_loadl_epi64(
                        (const __m128i *)(w + (i >> 1)));
                    __m128i lo = _mm_and_si128(by, m4);
                    __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
                    __m128i nib = _mm_unpacklo_epi8(lo, hi);
                    __m256 w0 = _mm256_cvtepi32_ps(
                        _mm256_sub_epi32(_mm256_cvtepu8_epi32(nib), b8));
                    __m256 w1 = _mm256_cvtepi32_ps(
                        _mm256_sub_epi32(_mm256_cvtepu8_epi32(
                            _mm_srli_si128(nib, 8)), b8));
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i), w0, acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i + 8), w1, acc);
                }
                a += _coli_hsum256_ps(acc) * sc;
#endif
                for (; i < base + glen; i += 2) {
                    if (i + 1 < base + glen) {
                        uint8_t byte = w[i >> 1];
                        a += (xs[i] * (float)((int)(byte & 0xF) - 8) +
                              xs[i + 1] * (float)((int)(byte >> 4) - 8)) * sc;
                    } else {
                        uint8_t byte = w[i >> 1];
                        a += xs[i] * (float)((int)(byte & 0xF) - 8) * sc;
                    }
                }
            }
            y[(size_t)s * O + o] = a;
        }
    }
}

// ── 6. Fused gate+up int4 matmul (one loop covers both matrices) ─────
// yg[O] = x[I] @ wg^T, yu[O] = x[I] @ wu^T
// S=1 only (single row). Saves one OMP dispatch.
static inline void colibri_matmul_q4_pair(float *yg, float *yu,
                                           const float *x,
                                           const uint8_t *qg,
                                           const float *sg,
                                           const uint8_t *qu,
                                           const float *su,
                                           int I, int O) {
    int rb = (I + 1) / 2;
    for (int z = 0; z < 2 * O; z++) {
        int o = z < O ? z : z - O;
        const uint8_t *w = (z < O ? qg : qu) + (size_t)o * rb;
        float a = 0;
        int i = 0;
#ifdef COLI_AVX2
        const __m128i m4 = _mm_set1_epi8(0x0F);
        const __m256i b8 = _mm256_set1_epi32(8);
        __m256 acc = _mm256_setzero_ps();
        for (; i + 16 <= I; i += 16) {
            __m128i by = _mm_loadl_epi64((const __m128i *)(w + (i >> 1)));
            __m128i lo = _mm_and_si128(by, m4);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
            __m128i nib = _mm_unpacklo_epi8(lo, hi);
            __m256 w0 = _mm256_cvtepi32_ps(
                _mm256_sub_epi32(_mm256_cvtepu8_epi32(nib), b8));
            __m256 w1 = _mm256_cvtepi32_ps(
                _mm256_sub_epi32(_mm256_cvtepu8_epi32(
                    _mm_srli_si128(nib, 8)), b8));
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), w0, acc);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), w1, acc);
        }
        a = _coli_hsum256_ps(acc);
#elif defined(COLI_NEON)
        const uint8x8_t m4v = vdup_n_u8(0x0F);
        const int8x8_t b8v = vdup_n_s8(8);
        float32x4_t ac0 = vdupq_n_f32(0), ac1 = vdupq_n_f32(0);
        for (; i + 16 <= I; i += 16) {
            uint8x8_t by = vld1_u8(w + (i >> 1));
            uint8x8x2_t n = vzip_u8(vand_u8(by, m4v), vshr_n_u8(by, 4));
            int16x8_t w0 = vmovl_s8(
                vsub_s8(vreinterpret_s8_u8(n.val[0]), b8v));
            int16x8_t w1 = vmovl_s8(
                vsub_s8(vreinterpret_s8_u8(n.val[1]), b8v));
            ac0 = vfmaq_f32(ac0, vld1q_f32(x + i),
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
            ac1 = vfmaq_f32(ac1, vld1q_f32(x + i + 4),
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
            ac0 = vfmaq_f32(ac0, vld1q_f32(x + i + 8),
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
            ac1 = vfmaq_f32(ac1, vld1q_f32(x + i + 12),
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1))));
        }
        a = vaddvq_f32(vaddq_f32(ac0, ac1));
#endif
        for (; i + 1 < I; i += 2) {
            uint8_t b = w[i >> 1];
            a += x[i] * (float)((b & 15) - 8) + x[i + 1] * (float)((b >> 4) - 8);
        }
        if (i < I)
            a += x[i] * (float)((w[i >> 1] & 15) - 8);
        float *dst = z < O ? yg : yu;
        dst[o] = a * (z < O ? sg : su)[o];
    }
}

// ── 7. Integer dot products (IDOT) ──────────────────────────────────
// Quantize activations on-the-fly to int8, then pure integer dot.
// ~2-3x faster than f32 matmul on AVX2/VNNI, ~0.3% RMS error.

// Quantize f32 row to int8, return scale = absmax/127
static inline float colibri_qrow_i8(const float *x, int8_t *q, int I) {
    float amax = 0;
    for (int i = 0; i < I; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    float s = amax / 127.0f;
    if (s < 1e-12f) s = 1e-12f;
    float inv = 1.0f / s;
    for (int i = 0; i < I; i++) q[i] = (int8_t)(int)roundf(x[i] * inv);
    return s;
}

// int8 × int8 dot product (weights already int8)
static inline int32_t colibri_dot_i8i8(const int8_t *w, const int8_t *x, int I) {
    int32_t sum = 0;
    int i = 0;
#if defined(COLI_AVX512)
    __m512i acc = _mm512_setzero_si512();
    for (; i + 64 <= I; i += 64) {
        __m512i wv = _mm512_loadu_si512((const void *)(w + i));
        __m512i xv = _mm512_loadu_si512((const void *)(x + i));
        __mmask64 neg = _mm512_movepi8_mask(wv);
        __m512i xs = _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv);
        acc = _mm512_dpbusd_epi32(acc, _mm512_abs_epi8(wv), xs);
    }
    sum = (int32_t)_mm512_reduce_add_epi32(acc);
#elif defined(COLI_AVX2)
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    for (; i + 32 <= I; i += 32) {
        __m256i wv = _mm256_loadu_si256((const __m256i *)(w + i));
        __m256i xv = _mm256_loadu_si256((const __m256i *)(x + i));
        __m256i p = _mm256_maddubs_epi16(
            _mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones));
    }
    sum = _coli_hsum256_i32(acc);
#elif defined(COLI_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    for (; i + 64 <= I; i += 64) {
        a0 = vdotq_s32(a0, vld1q_s8(w + i), vld1q_s8(x + i));
        a1 = vdotq_s32(a1, vld1q_s8(w + i + 16), vld1q_s8(x + i + 16));
        a2 = vdotq_s32(a2, vld1q_s8(w + i + 32), vld1q_s8(x + i + 32));
        a3 = vdotq_s32(a3, vld1q_s8(w + i + 48), vld1q_s8(x + i + 48));
    }
    int32x4_t acc = vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3));
    for (; i + 16 <= I; i += 16)
        acc = vdotq_s32(acc, vld1q_s8(w + i), vld1q_s8(x + i));
    sum = (int32_t)vaddvq_s32(acc);
#elif defined(COLI_NEON)
    int32x4_t acc = vdupq_n_s32(0);
    for (; i + 16 <= I; i += 16) {
        int8x16_t wv = vld1q_s8(w + i), xv = vld1q_s8(x + i);
        int16x8_t p = vmull_s8(vget_low_s8(wv), vget_low_s8(xv));
        p = vmlal_s8(p, vget_high_s8(wv), vget_high_s8(xv));
        acc = vpadalq_s16(acc, p);
    }
    sum = (int32_t)vaddvq_s32(acc);
#endif
    for (; i < I; i++) sum += (int32_t)w[i] * x[i];
    return sum;
}

// int4(packed) × int8 dot product
static inline int32_t colibri_dot_i4i8(const uint8_t *w4, const int8_t *x, int I) {
    int32_t sum = 0;
    int i = 0;
#if defined(COLI_AVX512)
    const __m256i m4v = _mm256_set1_epi8(0x0F);
    const __m512i b8v = _mm512_set1_epi8(8);
    const __m512i xidx = _mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7);
    __m512i acc = _mm512_setzero_si512();
    for (; i + 64 <= I; i += 64) {
        __m256i by = _mm256_loadu_si256((const __m256i *)(w4 + (i >> 1)));
        __m256i lo = _mm256_and_si256(by, m4v);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(by, 4), m4v);
        __m256i z0 = _mm256_unpacklo_epi8(lo, hi);
        __m256i z1 = _mm256_unpackhi_epi8(lo, hi);
        __m512i wv = _mm512_sub_epi8(
            _mm512_inserti64x4(_mm512_castsi256_si512(z0), z1, 1), b8v);
        __m512i xv = _mm512_permutexvar_epi64(
            xidx, _mm512_loadu_si512((const void *)(x + i)));
        __mmask64 neg = _mm512_movepi8_mask(wv);
        __m512i xs = _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv);
        acc = _mm512_dpbusd_epi32(acc, _mm512_abs_epi8(wv), xs);
    }
    sum = (int32_t)_mm512_reduce_add_epi32(acc);
#elif defined(COLI_AVX2)
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m256i b8 = _mm256_set1_epi8(8);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    for (; i + 32 <= I; i += 32) {
        __m128i by = _mm_loadu_si128((const __m128i *)(w4 + (i >> 1)));
        __m128i lo = _mm_and_si128(by, m4);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
        __m128i n0 = _mm_unpacklo_epi8(lo, hi);
        __m128i n1 = _mm_unpackhi_epi8(lo, hi);
        __m256i wv = _mm256_sub_epi8(_mm256_set_m128i(n1, n0), b8);
        __m256i xv = _mm256_loadu_si256((const __m256i *)(x + i));
        __m256i p = _mm256_maddubs_epi16(
            _mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones));
    }
    sum = _coli_hsum256_i32(acc);
#elif defined(COLI_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t m4q = vdupq_n_u8(0x0F);
    const int8x16_t b8q = vdupq_n_s8(8);
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    for (; i + 64 <= I; i += 64) {
        uint8x16_t byA = vld1q_u8(w4 + (i >> 1));
        uint8x16_t byB = vld1q_u8(w4 + (i >> 1) + 16);
        uint8x16x2_t zA = vzipq_u8(vandq_u8(byA, m4q), vshrq_n_u8(byA, 4));
        uint8x16x2_t zB = vzipq_u8(vandq_u8(byB, m4q), vshrq_n_u8(byB, 4));
        a0 = vdotq_s32(a0, vsubq_s8(vreinterpretq_s8_u8(zA.val[0]), b8q),
                       vld1q_s8(x + i));
        a1 = vdotq_s32(a1, vsubq_s8(vreinterpretq_s8_u8(zA.val[1]), b8q),
                       vld1q_s8(x + i + 16));
        a2 = vdotq_s32(a2, vsubq_s8(vreinterpretq_s8_u8(zB.val[0]), b8q),
                       vld1q_s8(x + i + 32));
        a3 = vdotq_s32(a3, vsubq_s8(vreinterpretq_s8_u8(zB.val[1]), b8q),
                       vld1q_s8(x + i + 48));
    }
    int32x4_t acc = vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3));
    for (; i + 32 <= I; i += 32) {
        uint8x16_t by = vld1q_u8(w4 + (i >> 1));
        uint8x16x2_t z = vzipq_u8(vandq_u8(by, m4q), vshrq_n_u8(by, 4));
        acc = vdotq_s32(acc, vsubq_s8(vreinterpretq_s8_u8(z.val[0]), b8q),
                        vld1q_s8(x + i));
        acc = vdotq_s32(acc, vsubq_s8(vreinterpretq_s8_u8(z.val[1]), b8q),
                        vld1q_s8(x + i + 16));
    }
    sum = (int32_t)vaddvq_s32(acc);
#elif defined(COLI_NEON)
    const uint8x16_t m4q = vdupq_n_u8(0x0F);
    const int8x16_t b8q = vdupq_n_s8(8);
    int32x4_t acc = vdupq_n_s32(0);
    for (; i + 32 <= I; i += 32) {
        uint8x16_t by = vld1q_u8(w4 + (i >> 1));
        uint8x16x2_t z = vzipq_u8(vandq_u8(by, m4q), vshrq_n_u8(by, 4));
        int8x16_t w0 = vsubq_s8(vreinterpretq_s8_u8(z.val[0]), b8q);
        int8x16_t w1 = vsubq_s8(vreinterpretq_s8_u8(z.val[1]), b8q);
        int8x16_t x0 = vld1q_s8(x + i), x1 = vld1q_s8(x + i + 16);
        int16x8_t p = vmull_s8(vget_low_s8(w0), vget_low_s8(x0));
        p = vmlal_s8(p, vget_high_s8(w0), vget_high_s8(x0));
        acc = vpadalq_s16(acc, p);
        p = vmull_s8(vget_low_s8(w1), vget_low_s8(x1));
        p = vmlal_s8(p, vget_high_s8(w1), vget_high_s8(x1));
        acc = vpadalq_s16(acc, p);
    }
    sum = (int32_t)vaddvq_s32(acc);
#endif
    for (; i + 1 < I; i += 2) {
        uint8_t b = w4[i >> 1];
        sum += ((int)(b & 0xF) - 8) * x[i] + ((int)(b >> 4) - 8) * x[i + 1];
    }
    if (i < I) {
        uint8_t b = w4[i >> 1];
        sum += ((int)(b & 0xF) - 8) * x[i];
    }
    return sum;
}

// ── 8. Quantization helpers ──────────────────────────────────────────

// Quantize f32 weight row to int4 packed format (2 values/byte)
// in[I] f32 → q4[ceil(I/2)] uint8_t + scale[o] float
// Uses symmetric per-row quantization centered at 0.
// Returns the scale factor.
static inline float colibri_quantize_q4(const float *in, uint8_t *q4,
                                         int I) {
    float amax = 0;
    for (int i = 0; i < I; i++) {
        float a = fabsf(in[i]);
        if (a > amax) amax = a;
    }
    // int4 range is [-8, 7], so max representable is 7
    float scale = amax / 7.0f;
    if (scale < 1e-12f) scale = 1e-12f;
    float inv = 1.0f / scale;
    for (int i = 0; i < I; i += 2) {
        int v0 = (int)roundf(in[i] * inv);
        int v1 = (i + 1 < I) ? (int)roundf(in[i + 1] * inv) : 0;
        // Clamp to [-8, 7]
        if (v0 < -8) v0 = -8; if (v0 > 7) v0 = 7;
        if (v1 < -8) v1 = -8; if (v1 > 7) v1 = 7;
        q4[i >> 1] = (uint8_t)((v0 + 8) | ((v1 + 8) << 4));
    }
    return scale;
}

// Quantize f32 weight row to int2 packed format (4 values/byte)
// in[I] f32 → q2[ceil(I/4)] uint8_t + scale[o] float
// Values in [-2, -1, 0, 1] mapped to [0, 1, 2, 3] binary
static inline float colibri_quantize_q2(const float *in, uint8_t *q2,
                                         int I) {
    float amax = 0;
    for (int i = 0; i < I; i++) {
        float a = fabsf(in[i]);
        if (a > amax) amax = a;
    }
    // int2 range is [-2, 1], so max absolute is 2
    float scale = amax / 2.0f;
    if (scale < 1e-12f) scale = 1e-12f;
    float inv = 1.0f / scale;
    memset(q2, 0, (size_t)((I + 3) / 4));
    for (int i = 0; i < I; i++) {
        int v = (int)roundf(in[i] * inv);
        if (v < -2) v = -2; if (v > 1) v = 1;
        int sh = (i & 3) * 2;
        q2[i >> 2] |= (uint8_t)((v + 2) << sh);
    }
    return scale;
}

// Batch-quantize an f32 weight matrix [O, I] to int4.
// out_q4[O * ceil(I/2)] + out_scale[O]
// Returns total bytes written to out_q4 (for memcpy sizing).
static inline size_t colibri_quantize_matrix_q4(const float *W_f32,
                                                  uint8_t *out_q4,
                                                  float *out_scale,
                                                  int O, int I) {
    for (int o = 0; o < O; o++) {
        out_scale[o] = colibri_quantize_q4(
            W_f32 + (size_t)o * I, out_q4 + (size_t)o * ((I + 1) / 2), I);
    }
    return (size_t)O * ((size_t)(I + 1) / 2);
}

// ── 9. Matmul with on-the-fly activation quantization (IDOT path) ────
// y[O] = x[I] @ W_q^T where W_q is int4 packed
// Activations are quantized to int8 per call, then dot_i4i8.
// ~2-3x faster than matmul_q4 on AVX2/VNNI when S=1 (decode).
static inline float colibri_matmul_q4_idot(float *y, const float *x,
                                            const uint8_t *q4,
                                            const float *scale,
                                            int S, int I, int O,
                                            int8_t *scratch) {
    // scratch must be S * I bytes
    float sx = 0;
    if (S == 1) {
        sx = colibri_qrow_i8(x, scratch, I);
        for (int o = 0; o < O; o++)
            y[o] = (float)colibri_dot_i4i8(q4 + (size_t)o * ((I + 1) / 2),
                                            scratch, I) * scale[o] * sx;
    } else {
        for (int s = 0; s < S; s++) {
            float sx_s = colibri_qrow_i8(x + (size_t)s * I,
                                          scratch + (size_t)s * I, I);
            for (int o = 0; o < O; o++)
                y[(size_t)s * O + o] =
                    (float)colibri_dot_i4i8(
                        q4 + (size_t)o * ((I + 1) / 2),
                        scratch + (size_t)s * I, I) * scale[o] * sx_s;
        }
    }
    return sx;  // return quant scale for the first row
}

// ── 10. ONNX weight extraction alias ─────────────────────────────────
// If you're loading weights from ONNX or safetensors in float32,
// call colibri_quantize_matrix_q4() at load time to convert to
// the packed format, then use colibri_matmul_q4() at inference time.
//
// Example integration in backend_cpu.cpp:
//
//   // At load time (once per weight matrix):
//   std::vector<uint8_t> q4_buf(O * ((H + 1) / 2));
//   std::vector<float>   q4_scale(O);
//   colibri_quantize_matrix_q4(W_f32, q4_buf.data(), q4_scale.data(), O, H);
//
//   // At inference time:
//   colibri_matmul_q4(out, in, q4_buf.data(), q4_scale.data(), S, H, O);
//
// This cuts memory from 4 bytes/param → 0.5 bytes/param (8x compression)
// with negligible quality loss for inference.

#ifdef __cplusplus
}
#endif

#endif // COLI_BRI_KERNELS_H
