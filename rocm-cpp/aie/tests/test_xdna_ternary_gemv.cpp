// test_xdna_ternary_gemv.cpp — bit-exact test for the v0 ternary GEMV.
//
// Layout:
//   1. Generate random ternary trits ∈ {-1, 0, +1} for the [M,K] weight
//      matrix and pack them into [M, K/4] uint8 using the same encoding the
//      kernel decodes (LSB-first 2-bit nibbles, 00/01/10 = 0/+1/-1).
//   2. Generate random i8 activations.
//   3. Run on NPU.
//   4. Compute scalar reference on CPU using the same trit decode.
//   5. Compare i32 path bit-exact (we hijack the wrapper's scale path with
//      scales=1.0, x_scale=1.0, then cast back through the same fp16 round
//      to compare). Also compute reference y in fp32 with realistic scales
//      and check fp16 output matches within 1 ULP.
//
// Skips with exit 0 if the xclbin isn't present (matches the spec's CI
// behavior on a clean clone).
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (c) 2026 Daniel <d1r7yman@gmail.com>

#include "rocm_cpp/xdna_ternary_gemv.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr int kM = 64;
constexpr int kK = 64;
constexpr int kKpack = kK / 4;

constexpr const char* kXclbinPath =
    "/home/bcloud/repos/rocm-cpp/aie/build/final_ternary_gemv_64x64.xclbin";

// Encode a single trit into a 2-bit nibble matching the kernel-side decode.
// Caller MUST only pass values in {-1, 0, +1}. Returns:
//   0  -> 00
//  +1  -> 01
//  -1  -> 10
inline uint8_t encode_trit(int t) {
    if (t == 0)  return 0b00;
    if (t == 1)  return 0b01;
    if (t == -1) return 0b10;
    std::abort();  // out-of-range trit; tests must never hit this
}

inline int decode_trit_host(uint8_t byte, int i) {
    uint8_t nibble = (byte >> (i * 2)) & 0x3u;
    int lo = nibble & 0x1;
    int hi = (nibble >> 1) & 0x1;
    return lo - hi;
}

// Pack [M, K] of trits (int8 in {-1,0,+1}) into [M, K/4] uint8.
void pack_ternary(const std::vector<int8_t>& trits,
                  std::vector<uint8_t>& packed,
                  int M, int K) {
    const int Kp = K / 4;
    packed.assign(static_cast<size_t>(M) * Kp, 0);
    for (int m = 0; m < M; ++m) {
        for (int kp = 0; kp < Kp; ++kp) {
            uint8_t byte = 0;
            for (int i = 0; i < 4; ++i) {
                const int k = kp * 4 + i;
                byte |= static_cast<uint8_t>(
                    encode_trit(trits[m * K + k])) << (i * 2);
            }
            packed[m * Kp + kp] = byte;
        }
    }
}

// Scalar reference: trits + i8 -> i32 dot. Mirrors the kernel exactly,
// using the host-side decode_trit so the encode/decode pair is exercised.
void cpu_ternary_gemv_i32(const std::vector<uint8_t>& A_packed,
                          const std::vector<int8_t>& b,
                          std::vector<int32_t>& c,
                          int M, int K) {
    const int Kp = K / 4;
    c.assign(M, 0);
    for (int m = 0; m < M; ++m) {
        int32_t acc = 0;
        for (int kp = 0; kp < Kp; ++kp) {
            uint8_t byte = A_packed[m * Kp + kp];
            for (int i = 0; i < 4; ++i) {
                int t = decode_trit_host(byte, i);
                acc += t * static_cast<int32_t>(b[kp * 4 + i]);
            }
        }
        c[m] = acc;
    }
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    if (!fs::exists(kXclbinPath)) {
        std::cout << "// SKIPPED test_xdna_ternary_gemv — xclbin not present at "
                  << kXclbinPath
                  << "\n// Build it via:\n"
                  << "//   bash /home/bcloud/repos/rocm-cpp/aie/scripts/"
                     "build_ternary_xclbin.sh 64 64\n";
        return 0;
    }

    try {
        rocm_cpp::XdnaTernaryGemv gemv(kXclbinPath);
        if (!gemv.is_loaded()) {
            std::cerr << "FAIL: gemv not loaded after construction\n";
            return 1;
        }
        std::cout << "Loaded xclbin. Kernel name: " << gemv.kernel_name()
                  << "  M=" << gemv.M() << " K=" << gemv.K() << "\n";

        std::mt19937 rng(2026);
        std::uniform_int_distribution<int> trit_dist(-1, 1);
        std::uniform_int_distribution<int> act_dist(-127, 127);
        std::uniform_real_distribution<float> scale_dist(0.5f, 2.0f);

        // 1. Generate trits.
        std::vector<int8_t> trits(static_cast<size_t>(kM) * kK);
        for (auto& t : trits) t = static_cast<int8_t>(trit_dist(rng));

        // 2. Pack.
        std::vector<uint8_t> A_packed;
        pack_ternary(trits, A_packed, kM, kK);

        // 3. Activations.
        std::vector<int8_t> x_i8(kK);
        for (auto& v : x_i8) v = static_cast<int8_t>(act_dist(rng));

        // 4. Scales.
        std::vector<float> scales(kM);
        for (auto& s : scales) s = scale_dist(rng);
        const float x_scale = 1.0f / 127.0f;

        std::vector<_Float16> y_npu(kM, _Float16{0});
        std::vector<_Float16> y_ref(kM, _Float16{0});

        const auto t0 = std::chrono::steady_clock::now();
        gemv.run(A_packed.data(), x_i8.data(), scales.data(), x_scale,
                 y_npu.data(), kM, kK);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "NPU run latency (single GEMV, includes BO sync): "
                  << ms << " ms\n";

        // 5. Scalar reference for the i32 path.
        std::vector<int32_t> acc_ref;
        cpu_ternary_gemv_i32(A_packed, x_i8, acc_ref, kM, kK);

        // Reconstruct the NPU's i32 accumulator from y_npu by inverting the
        // (scales, x_scale, fp16 cast) chain — but that's lossy. Better:
        // compute the reference fp16 the same way the wrapper does and
        // compare bit-exact.
        for (int m = 0; m < kM; ++m) {
            float v = static_cast<float>(acc_ref[m]) * scales[m] * x_scale;
            y_ref[m] = static_cast<_Float16>(v);
        }

        // Bit-exact i32 check by re-running with scales=1, x_scale=1 so the
        // y_out path becomes (i32)acc -> fp16 with no quant loss for our
        // small dot-product range (|acc| <= K * 127 = 8128 < 65504 fp16 max).
        std::vector<float> ones(kM, 1.0f);
        std::vector<_Float16> y_i32_check(kM);
        gemv.run(A_packed.data(), x_i8.data(), ones.data(), 1.0f,
                 y_i32_check.data(), kM, kK);

        long long mismatches_i32 = 0;
        int64_t max_abs_i32 = 0;
        for (int m = 0; m < kM; ++m) {
            int32_t got = static_cast<int32_t>(static_cast<float>(y_i32_check[m]));
            int32_t want = acc_ref[m];
            int64_t d = std::abs(static_cast<int64_t>(got) - want);
            if (d != 0) ++mismatches_i32;
            if (d > max_abs_i32) max_abs_i32 = d;
        }
        std::cout << "i32-path  max |Δ| = " << max_abs_i32
                  << "  mismatches = " << mismatches_i32 << "/" << kM << "\n";

        // fp16 path: should match bit-for-bit because both sides use the
        // identical (acc * scales * x_scale) formula and identical fp16 cast.
        long long mismatches_fp16 = 0;
        for (int m = 0; m < kM; ++m) {
            uint16_t a, b;
            std::memcpy(&a, &y_npu[m], 2);
            std::memcpy(&b, &y_ref[m], 2);
            if (a != b) ++mismatches_fp16;
        }
        std::cout << "fp16 path mismatches = " << mismatches_fp16 << "/" << kM
                  << "\n";

        if (max_abs_i32 == 0 && mismatches_fp16 == 0) {
            std::cout << "PASS\n";
            return 0;
        }
        // Print first few mismatches to aid kernel-bug diagnosis.
        if (max_abs_i32 != 0) {
            int printed = 0;
            for (int m = 0; m < kM && printed < 8; ++m) {
                int32_t got = static_cast<int32_t>(static_cast<float>(y_i32_check[m]));
                int32_t want = acc_ref[m];
                if (got != want) {
                    std::cout << "  row " << m << ": NPU=" << got
                              << "  REF=" << want
                              << "  diff=" << (got - want) << "\n";
                    ++printed;
                }
            }
        }
        std::cout << "FAIL\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "FAIL with exception: " << e.what() << "\n";
        return 2;
    }
}
