// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
// Sherry — see LICENSE-SHERRY.md and SHERRY-FILES.txt at the repo root.
// Commercial use requires a separate license.
//
// bench_sherry.cpp — Sherry 1.25-bit ternary GEMV microbench
//
// Generates a 3:4-sparse ternary weight matrix (every group of 4 has exactly
// one zero), packs it in BOTH halo v2 (2 bpw) and Sherry v3 (1.25 bpw),
// runs both kernels, compares results for correctness, times both.
//
// Purpose: validate that Sherry delivers the 37.5% bandwidth reduction on
// gfx1151 before investing in the offline requantizer.
//
// Build is added to CMakeLists; run as:  build/bench_sherry [M K iters]

#include "rocm_cpp/ck_gemm.h"
#include "rocm_cpp/sherry.h"  // clean-room reference launcher (regression tripwire)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <cmath>
#include <cstdint>
#include <random>

#define HC(cmd) do { hipError_t e = cmd; if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %d at %s:%d: %s\n", e, __FILE__, __LINE__, \
            hipGetErrorString(e)); std::exit(1); } } while(0)

// ── Halo v2 packer — identical to what ternary_gemv_phase5_halo expects ─────
// Code: 0 -> -1, 1 -> 0, 2 -> +1. 16 weights per u32, K-contiguous.
static void pack_halo_v2(uint32_t* out, const int8_t* vals, int K) {
    const int u32s = K / 16;
    for (int u = 0; u < u32s; ++u) {
        uint32_t word = 0;
        for (int v = 0; v < 16; ++v) {
            int8_t wv = vals[u * 16 + v];
            uint32_t bits = (wv == -1) ? 0x0 : (wv == 1) ? 0x2 : 0x1;
            word |= (bits << (v * 2));
        }
        out[u] = word;
    }
}

// ── Sherry v3 packer: 5 bits per group of 4 weights.  ───────────────────────
// One group of 4 MUST have exactly one zero (3:4 sparsity). Row stride:
// K * 5 / 32 bytes. K must be a multiple of 32.
static void pack_sherry_v3(uint8_t* out, const int8_t* vals, int K) {
    const int bits_per_row = K * 5 / 4;
    const int bytes_per_row = bits_per_row / 8;
    std::memset(out, 0, bytes_per_row);
    const int groups = K / 4;
    for (int g = 0; g < groups; ++g) {
        const int8_t* gv = vals + g * 4;
        int zero_pos = -1;
        uint32_t signs_field = 0;
        int sign_idx = 0;
        for (int p = 0; p < 4; ++p) {
            if (gv[p] == 0) {
                if (zero_pos >= 0) {
                    fprintf(stderr, "pack_sherry_v3: group %d has >1 zero\n", g);
                    std::exit(2);
                }
                zero_pos = p;
            }
        }
        if (zero_pos < 0) {
            fprintf(stderr, "pack_sherry_v3: group %d has no zero\n", g);
            std::exit(2);
        }
        for (int p = 0; p < 4; ++p) {
            if (p == zero_pos) continue;
            int bit = (gv[p] == 1) ? 1 : 0;
            signs_field |= ((uint32_t)bit << sign_idx);
            ++sign_idx;
        }
        uint32_t code = ((uint32_t)zero_pos << 3) | signs_field;
        int bit_pos = 5 * g;
        int byte = bit_pos >> 3;
        int shift = bit_pos & 7;
        out[byte]     |= (uint8_t)(code << shift);
        if (shift + 5 > 8) out[byte + 1] |= (uint8_t)(code >> (8 - shift));
    }
}

// ── TQ1 v4 packer (base-3, 1.6 bpw) — for cross-kernel sanity check ────────
static void pack_tq1_v4(uint8_t* out, const int8_t* vals, int K) {
    const int row_bytes = K / 5;  // K must be multiple of 5
    for (int i = 0; i < row_bytes; ++i) {
        int base = i * 5;
        int byte = 0, mul = 1;
        for (int d = 0; d < 5; ++d) {
            int k = base + d;
            int digit = (k < K) ? (vals[k] + 1) : 1;  // pad = 0 (digit=1)
            byte += digit * mul;
            mul *= 3;
        }
        out[i] = (uint8_t)byte;
    }
}

// ── Deterministic 3:4-sparse weight generator ──────────────────────────────
// For each group of 4: pick one index uniformly as zero, others ±1 uniformly.
static void gen_ternary_3_of_4(int8_t* out, int K, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pos(0, 3);
    std::uniform_int_distribution<int> sign(0, 1);
    const int groups = K / 4;
    for (int g = 0; g < groups; ++g) {
        int zp = pos(rng);
        for (int p = 0; p < 4; ++p) {
            out[g * 4 + p] = (p == zp) ? 0 : (sign(rng) ? 1 : -1);
        }
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? std::atoi(argv[1]) : 6912;
    int K = (argc > 2) ? std::atoi(argv[2]) : 2560;
    int iters = (argc > 3) ? std::atoi(argv[3]) : 256;
    if (K % 32 != 0) { fprintf(stderr, "K must be multiple of 32\n"); return 1; }

    printf("[bench_sherry] M=%d K=%d iters=%d\n", M, K, iters);

    // Generate 3:4-sparse weights identical across both packers.
    std::vector<int8_t> weights(M * K);
    for (int r = 0; r < M; ++r) gen_ternary_3_of_4(&weights[r * K], K, 0xC0FFEE ^ (uint32_t)r);

    // Pack all three formats.
    const int halo_u32_per_row = K / 16;
    std::vector<uint32_t> halo_packed(M * halo_u32_per_row);
    const int sherry_bytes_per_row = K * 5 / 32;
    std::vector<uint8_t>  sherry_packed(M * sherry_bytes_per_row);
    const int tq1_bytes_per_row    = K / 5;  // K must be multiple of 20 for kernel alignment
    std::vector<uint8_t>  tq1_packed(M * tq1_bytes_per_row);
    for (int r = 0; r < M; ++r) {
        pack_halo_v2(&halo_packed[r * halo_u32_per_row], &weights[r * K], K);
        pack_sherry_v3(&sherry_packed[r * sherry_bytes_per_row], &weights[r * K], K);
        pack_tq1_v4(&tq1_packed[r * tq1_bytes_per_row], &weights[r * K], K);
    }

    // Activations, scales, outputs.
    std::vector<int8_t> x_i8(K);
    {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> d(-127, 127);
        for (int k = 0; k < K; ++k) x_i8[k] = (int8_t)d(rng);
    }
    // FP16 activations for the clean-room sherry_ref launcher (no x_scale —
    // the kernel takes fp16 acts directly). Mirror the int8 distribution
    // numerically so the timed work is comparable, even though sherry_ref's
    // arithmetic differs (fp16 mul-add vs sdot4 int8 reduce).
    std::vector<__half> x_fp16(K);
    for (int k = 0; k < K; ++k) x_fp16[k] = __float2half((float)x_i8[k] / 127.0f);
    std::vector<float> scales(M, 1.0f);
    std::vector<__half> y_halo(M), y_sherry(M);
    const float x_scale = 1.0f / 127.0f;

    // Upload.
    void *d_halo, *d_sherry, *d_tq1, *d_x, *d_x_fp16, *d_scales,
         *d_y_halo, *d_y_sherry, *d_y_tq1, *d_y_sherry_ref;
    HC(hipMalloc(&d_halo, halo_packed.size() * 4));
    HC(hipMalloc(&d_sherry, sherry_packed.size()));
    HC(hipMalloc(&d_tq1, tq1_packed.size()));
    HC(hipMalloc(&d_x, K));
    HC(hipMalloc(&d_x_fp16, K * 2));
    HC(hipMalloc(&d_scales, M * 4));
    HC(hipMalloc(&d_y_halo, M * 2));
    HC(hipMalloc(&d_y_sherry, M * 2));
    HC(hipMalloc(&d_y_tq1, M * 2));
    HC(hipMalloc(&d_y_sherry_ref, M * 2));
    HC(hipMemcpy(d_halo, halo_packed.data(), halo_packed.size() * 4, hipMemcpyHostToDevice));
    HC(hipMemcpy(d_sherry, sherry_packed.data(), sherry_packed.size(), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_tq1, tq1_packed.data(), tq1_packed.size(), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_x, x_i8.data(), K, hipMemcpyHostToDevice));
    HC(hipMemcpy(d_x_fp16, x_fp16.data(), K * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(d_scales, scales.data(), M * 4, hipMemcpyHostToDevice));
    std::vector<__half> y_tq1(M);

    // Warm + correctness (halo is ground truth).
    rcpp_ternary_gemv_halo_f16    (d_halo,   d_x, x_scale, d_scales, d_y_halo,   M, K, nullptr);
    rcpp_ternary_gemv_sherry_f16  (d_sherry, d_x, x_scale, d_scales, d_y_sherry, M, K, nullptr);
    rcpp_ternary_gemv_tq1_halo_f16(d_tq1,    d_x, x_scale, d_scales, d_y_tq1,    M, K, nullptr);
    // sherry_ref is the clean-room fp16-in/fp16-out launcher exposed by
    // include/rocm_cpp/sherry.h. We don't correctness-check its output here
    // (different scaling convention — pure signed-sum, no x_scale, no
    // row-scale fold); the differential test in tests/test_sherry_gemv.cpp
    // already validates it bit-for-bit against the scalar reference. Bench
    // it solely to flag perf regressions if anyone wires it into prod.
    sherry_ternary_gemv_launch(
        static_cast<const uint8_t*>(d_sherry),
        static_cast<const uint16_t*>(d_x_fp16),
        static_cast<uint16_t*>(d_y_sherry_ref),
        M, K, /*stream=*/nullptr);
    HC(hipDeviceSynchronize());
    HC(hipMemcpy(y_halo.data(),   d_y_halo,   M * 2, hipMemcpyDeviceToHost));
    HC(hipMemcpy(y_sherry.data(), d_y_sherry, M * 2, hipMemcpyDeviceToHost));
    HC(hipMemcpy(y_tq1.data(),    d_y_tq1,    M * 2, hipMemcpyDeviceToHost));

    // halo-vs-sherry check (requires 3:4-sparse weights, set earlier).
    int max_diff_idx = -1;
    float max_diff = 0.0f;
    for (int m = 0; m < M; ++m) {
        float a = (float)y_halo[m], b = (float)y_sherry[m];
        float d = std::fabs(a - b);
        if (d > max_diff) { max_diff = d; max_diff_idx = m; }
    }
    printf("[bench_sherry] max |halo - sherry| = %g at row %d (halo=%g sherry=%g)\n",
           max_diff, max_diff_idx,
           max_diff_idx >= 0 ? (float)y_halo[max_diff_idx] : 0.0f,
           max_diff_idx >= 0 ? (float)y_sherry[max_diff_idx] : 0.0f);

    // halo-vs-tq1 check — TQ1 is lossless for ternary, so diff should be 0.
    max_diff = 0.0f; max_diff_idx = -1;
    int bad_rows = 0;
    for (int m = 0; m < M; ++m) {
        float a = (float)y_halo[m], b = (float)y_tq1[m];
        float d = std::fabs(a - b);
        if (d > 0.5f) ++bad_rows;
        if (d > max_diff) { max_diff = d; max_diff_idx = m; }
    }
    printf("[bench_sherry] max |halo - tq1|    = %g at row %d (halo=%g tq1=%g), bad_rows=%d/%d\n",
           max_diff, max_diff_idx,
           max_diff_idx >= 0 ? (float)y_halo[max_diff_idx] : 0.0f,
           max_diff_idx >= 0 ? (float)y_tq1[max_diff_idx] : 0.0f,
           bad_rows, M);

    // Time halo.
    HC(hipDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        rcpp_ternary_gemv_halo_f16(d_halo, d_x, x_scale, d_scales, d_y_halo, M, K, nullptr);
    HC(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double halo_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Time sherry.
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        rcpp_ternary_gemv_sherry_f16(d_sherry, d_x, x_scale, d_scales, d_y_sherry, M, K, nullptr);
    HC(hipDeviceSynchronize());
    auto t3 = std::chrono::high_resolution_clock::now();
    double sherry_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Time tq1.
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        rcpp_ternary_gemv_tq1_halo_f16(d_tq1, d_x, x_scale, d_scales, d_y_tq1, M, K, nullptr);
    HC(hipDeviceSynchronize());
    auto t5 = std::chrono::high_resolution_clock::now();
    double tq1_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

    // Time sherry_ref (clean-room launcher). Same packed weights as `sherry`,
    // fp16 acts instead of int8 + post-scale. Expected ~2-3x slower than the
    // fast `sherry` row — that gap is the regression tripwire: if `sherry_ref`
    // ever lands within 10% of `sherry`, someone has either tuned the
    // clean-room kernel up to par (great, retire the i8 path) or the i8
    // kernel got slower (bad, investigate). If `sherry_ref` ever ends up
    // FASTER than `sherry`, the i8 path has rotted — page the architect.
    auto t6 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        sherry_ternary_gemv_launch(
            static_cast<const uint8_t*>(d_sherry),
            static_cast<const uint16_t*>(d_x_fp16),
            static_cast<uint16_t*>(d_y_sherry_ref),
            M, K, /*stream=*/nullptr);
    HC(hipDeviceSynchronize());
    auto t7 = std::chrono::high_resolution_clock::now();
    double sherry_ref_ms = std::chrono::duration<double, std::milli>(t7 - t6).count();

    double halo_us       = halo_ms       * 1000.0 / iters;
    double sherry_us     = sherry_ms     * 1000.0 / iters;
    double tq1_us        = tq1_ms        * 1000.0 / iters;
    double sherry_ref_us = sherry_ref_ms * 1000.0 / iters;
    double halo_bytes   = (double)M * K / 4.0;          // 2 bpw
    double sherry_bytes = (double)M * K * 5.0 / 32.0;   // 1.25 bpw
    double tq1_bytes    = (double)M * K / 5.0;          // 1.6 bpw
    double halo_gbs       = halo_bytes   / (halo_us       * 1e-6) / 1e9;
    double sherry_gbs     = sherry_bytes / (sherry_us     * 1e-6) / 1e9;
    double tq1_gbs        = tq1_bytes    / (tq1_us        * 1e-6) / 1e9;
    double sherry_ref_gbs = sherry_bytes / (sherry_ref_us * 1e-6) / 1e9;

    printf("[bench_sherry] halo       : %.2f µs/call  %.1f GB/s   (%.2f MB/row reads)\n",
           halo_us, halo_gbs, halo_bytes / 1e6);
    printf("[bench_sherry] sherry     : %.2f µs/call  %.1f GB/s   (%.2f MB/row reads)\n",
           sherry_us, sherry_gbs, sherry_bytes / 1e6);
    printf("[bench_sherry] sherry_ref : %.2f µs/call  %.1f GB/s   (%.2f MB/row reads)  [clean-room, offline-only]\n",
           sherry_ref_us, sherry_ref_gbs, sherry_bytes / 1e6);
    printf("[bench_sherry] tq1        : %.2f µs/call  %.1f GB/s   (%.2f MB/row reads)\n",
           tq1_us, tq1_gbs, tq1_bytes / 1e6);
    printf("[bench_sherry] sherry vs halo:        %.2fx speedup  %.1f%% bytes-reduction\n",
           halo_us / sherry_us, (1.0 - sherry_bytes / halo_bytes) * 100.0);
    printf("[bench_sherry] tq1    vs halo:        %.2fx speedup  %.1f%% bytes-reduction\n",
           halo_us / tq1_us,    (1.0 - tq1_bytes    / halo_bytes) * 100.0);
    printf("[bench_sherry] sherry vs sherry_ref:  %.2fx faster (regression tripwire — alert if <1.5x)\n",
           sherry_ref_us / sherry_us);

    hipFree(d_halo); hipFree(d_sherry); hipFree(d_tq1); hipFree(d_x); hipFree(d_x_fp16); hipFree(d_scales);
    hipFree(d_y_halo); hipFree(d_y_sherry); hipFree(d_y_tq1); hipFree(d_y_sherry_ref);
    return 0;
}
