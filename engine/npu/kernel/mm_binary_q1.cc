//===- mm_binary_q1.cc -------------------------------------------*- C++ -*-===//
//
// Phase 4: Q1_0 (1-bit binary) AIE microkernel — XDNA 2 (NPU2)
//
// Q1_0 packs 1 bit per value: 128 sign bits + 1 fp16 scale per 128-element group.
// bit = 0 → -scale, bit = 1 → +scale
// Storage: ceil(K/128) * 18 bytes per column (vs 16 for TQ2, 13 for TQ1)
//
// Licensed under Apache 2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

constexpr int M_TILE = 32;
constexpr int K_TILE = 64;
constexpr int N_TILE = 128;

// Q1_0: 128 elements per block, 16 bytes sign bits + 2 bytes fp16 scale = 18 bytes
constexpr int Q1_BLOCK_K = 128;
constexpr int Q1_BLOCK_BYTES = 18;
constexpr int Q1_SIGN_BYTES = 16;
constexpr int Q1_BYTES_PER_COL = (K_TILE + Q1_BLOCK_K - 1) / Q1_BLOCK_K * Q1_BLOCK_BYTES; // = 18 for K=64

extern "C" {

static int g_counter = 0;

void binary_q1_gemv(bfloat16 *pA, uint8_t *pB,
                     bfloat16 *pS, bfloat16 *pC) {
    event0();
    pC += g_counter * M_TILE * N_TILE;
    if (g_counter == 3) g_counter = 0; else g_counter++;

    // Decode 1-bit weights to int8 {-1, +1} × scale
    alignas(32) int8_t w_dec[N_TILE * K_TILE];
    for (int n = 0; n < N_TILE; n++) {
        auto *src = pB + n * Q1_BYTES_PER_COL;
        auto *dst = w_dec + n * K_TILE;
        // One 128-bit block (K=64 < 128, so one block with 64 valid bits)
        // fp16 scale at offset Q1_SIGN_BYTES
        // We process the first 64 bits
        uint16_t scale_bits;
        __builtin_memcpy(&scale_bits, src + Q1_SIGN_BYTES, 2);
        float scale = [](uint16_t h) {
            uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
            float sign = s ? -1.0f : 1.0f;
            if (e == 0) return sign * (float)m * 5.9604644775390625e-08f;
            if (e == 31) return m ? 0.0f : sign * 1.0f / 0.0f;
            return sign * (1.0f + (float)m / 1024.0f) * (float)(1 << (e - 15));
        }(scale_bits);
        // Load 8 bytes of sign bits (64 bits for K=64)
        uint64_t bits;
        __builtin_memcpy(&bits, src, 8);
        for (int i = 0; i < K_TILE; i++) {
            dst[i] = (bits & ((uint64_t)1 << i)) ? (int8_t)(scale) : (int8_t)(-scale);
        }
    }

    // Accumulate
    for (int m = 0; m < M_TILE; m++) {
        for (int n = 0; n < N_TILE; n++) {
            float sum = 0.0f;
            auto *w = w_dec + n * K_TILE;
            auto *a = pA + m * K_TILE;
            for (int k = 0; k < K_TILE; k++) {
                sum += (float)w[k] * (float)a[k];
            }
            pC[m * N_TILE + n] += (bfloat16)sum;
        }
    }

    event1();
}

void zero_kernel_q1(bfloat16 *cOut) {
    constexpr int N = M_TILE * N_TILE;
    constexpr int r = 512 / 16;
    auto zeros = aie::zeros<bfloat16, r>();
    for (int i = 0; i < N; i += r)
        aie::store_v(cOut + i, zeros);
}
}
