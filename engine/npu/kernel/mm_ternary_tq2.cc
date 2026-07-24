//===- mm_ternary_tq2.cc ----------------------------------------*- C++ -*-===//
//
// TQ2 ternary AIE microkernel — LUT decode + scalar MAC
//
// Phase 2 final: loads 2-bit ternary codes from DDR, LUT-decodes on-tile,
// applies per-group scales, accumulates with bf16 activations.
//
// The scalar MAC fallback is correct but ~8× slower than the block-vectorized
// mac_8x8_8x8T path. Replace the inner accumulation loop with the reference
// kernel's ping-pong MAC pattern for full throughput (see mm_bfp_mixed.cc).
//
// Licensed under Apache 2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

constexpr int M_TILE = 32;
constexpr int K_TILE = 64;
constexpr int N_TILE = 128;

// LUT: byte → 4× ternary values as uint32_t (one int8 per byte)
alignas(32) static const uint32_t tq2_lut[256] = {
    0x00000000,0x00000001,0x000000FF,0x00000000,0x00000100,0x00000101,0x000001FF,0x00000100,
    0x0000FF00,0x0000FF01,0x0000FFFF,0x0000FF00,0x00000000,0x00000001,0x000000FF,0x00000000,
    0x00010000,0x00010001,0x000100FF,0x00010000,0x00010100,0x00010101,0x000101FF,0x00010100,
    0x0001FF00,0x0001FF01,0x0001FFFF,0x0001FF00,0x00010000,0x00010001,0x000100FF,0x00010000,
    0x00FF0000,0x00FF0001,0x00FF00FF,0x00FF0000,0x00FF0100,0x00FF0101,0x00FF01FF,0x00FF0100,
    0x00FFFF00,0x00FFFF01,0x00FFFFFF,0x00FFFF00,0x00FF0000,0x00FF0001,0x00FF00FF,0x00FF0000,
    0x00000000,0x00000001,0x000000FF,0x00000000,0x00000100,0x00000101,0x000001FF,0x00000100,
    0x0000FF00,0x0000FF01,0x0000FFFF,0x0000FF00,0x00000000,0x00000001,0x000000FF,0x00000000,
    0x01000000,0x01000001,0x010000FF,0x01000000,0x01000100,0x01000101,0x010001FF,0x01000100,
    0x0100FF00,0x0100FF01,0x0100FFFF,0x0100FF00,0x01000000,0x01000001,0x010000FF,0x01000000,
    0x01010000,0x01010001,0x010100FF,0x01010000,0x01010100,0x01010101,0x010101FF,0x01010100,
    0x0101FF00,0x0101FF01,0x0101FFFF,0x0101FF00,0x01010000,0x01010001,0x010100FF,0x01010000,
    0x01FF0000,0x01FF0001,0x01FF00FF,0x01FF0000,0x01FF0100,0x01FF0101,0x01FF01FF,0x01FF0100,
    0x01FFFF00,0x01FFFF01,0x01FFFFFF,0x01FFFF00,0x01FF0000,0x01FF0001,0x01FF00FF,0x01FF0000,
    0x01000000,0x01000001,0x010000FF,0x01000000,0x01000100,0x01000101,0x010001FF,0x01000100,
    0x0100FF00,0x0100FF01,0x0100FFFF,0x0100FF00,0x01000000,0x01000001,0x010000FF,0x01000000,
    0x00000000,0x00000001,0x000000FF,0x00000000,0x00000100,0x00000101,0x000001FF,0x00000100,
    0x0000FF00,0x0000FF01,0x0000FFFF,0x0000FF00,0x00000000,0x00000001,0x000000FF,0x00000000,
    0x00010000,0x00010001,0x000100FF,0x00010000,0x00010100,0x00010101,0x000101FF,0x00010100,
    0x0001FF00,0x0001FF01,0x0001FFFF,0x0001FF00,0x00010000,0x00010001,0x000100FF,0x00010000,
    0x00FF0000,0x00FF0001,0x00FF00FF,0x00FF0000,0x00FF0100,0x00FF0101,0x00FF01FF,0x00FF0100,
    0x00FFFF00,0x00FFFF01,0x00FFFFFF,0x00FFFF00,0x00FF0000,0x00FF0001,0x00FF00FF,0x00FF0000,
    0x00000000,0x00000001,0x000000FF,0x00000000,0x00000100,0x00000101,0x000001FF,0x00000100,
    0x0000FF00,0x0000FF01,0x0000FFFF,0x0000FF00,0x00000000,0x00000001,0x000000FF,0x00000000,
    0x01000000,0x01000001,0x010000FF,0x01000000,0x01000100,0x01000101,0x010001FF,0x01000100,
    0x0100FF00,0x0100FF01,0x0100FFFF,0x0100FF00,0x01000000,0x01000001,0x010000FF,0x01000000,
    0x01010000,0x01010001,0x010100FF,0x01010000,0x01010100,0x01010101,0x010101FF,0x01010100,
    0x0101FF00,0x0101FF01,0x0101FFFF,0x0101FF00,0x01010000,0x01010001,0x010100FF,0x01010000,
    0x01FF0000,0x01FF0001,0x01FF00FF,0x01FF0000,0x01FF0100,0x01FF0101,0x01FF01FF,0x01FF0100,
    0x01FFFF00,0x01FFFF01,0x01FFFFFF,0x01FFFF00,0x01FF0000,0x01FF0001,0x01FF00FF,0x01FF0000,
    0x01000000,0x01000001,0x010000FF,0x01000000,0x01000100,0x01000101,0x010001FF,0x01000100,
    0x0100FF00,0x0100FF01,0x0100FFFF,0x0100FF00,0x01000000,0x01000001,0x010000FF,0x01000000,
};

extern "C" {

static int g_counter = 0;

// ─── TQ2 ternary GEMV ────────────────────────────────────────────
void ternary_tq2_gemv(bfloat16 *pA, uint8_t *pB,
                       bfloat16 *pS, bfloat16 *pC) {
    event0();
    pC += g_counter * M_TILE * N_TILE;
    if (g_counter == 3) g_counter = 0; else g_counter++;

    // Step 1: LUT-decode ternary codes to int8
    // Packed as int8 in decoded buffer (use int8 via float cast for AIE2 compat)
    alignas(32) float w_dec_f32[N_TILE * K_TILE];  // float for AIE2 compat

    for (int n = 0; n < N_TILE; n++) {
        auto *src = pB + n * K_TILE / 4;
        auto *dst = w_dec_f32 + n * K_TILE;
        for (int i = 0; i < 16; i++) {
            uint32_t lut = tq2_lut[src[i]];
            dst[i*4+0] = (float)(int8_t)(lut & 0xFF);
            dst[i*4+1] = (float)(int8_t)((lut >> 8) & 0xFF);
            dst[i*4+2] = (float)(int8_t)((lut >> 16) & 0xFF);
            dst[i*4+3] = (float)(int8_t)((lut >> 24) & 0xFF);
        }
    }

    // Step 2: Scalar MAC (correct, ~8× slower than block-vectorized)
    // Optimization path: replace with mac_8x8_8x8T via bfp16ebs8 blocks
    // See mm_bfp_mixed.cc for the full-throughput MAC reference.
    for (int m = 0; m < M_TILE; m++) {
        for (int n = 0; n < N_TILE; n++) {
            float sum = 0.0f;
            auto *w = w_dec_f32 + n * K_TILE;
            auto *a = pA + m * K_TILE;
            for (int g = 0; g < K_TILE / 32; g++) {
                float s = (float)pS[n * (K_TILE / 32) + g];
                for (int i = 0; i < 32; i++) {
                    sum += w[g * 32 + i] * s * (float)a[g * 32 + i];
                }
            }
            pC[m * N_TILE + n] += (bfloat16)sum;
        }
    }

    event1();
}

void zero_kernel_ternary(bfloat16 *cOut) {
    constexpr int N = M_TILE * N_TILE;
    constexpr int r = 512 / 16;
    auto zeros = aie::zeros<bfloat16, r>();
    for (int i = 0; i < N; i += r)
        aie::store_v(cOut + i, zeros);
}
}
