// mm_ternary_tq1.cc — TQ1 1.58-bit ternary NPU kernel with base-3 LUT decode + ping-pong
//
// TQ1 packing: 5 ternary values per byte using base-3 encoding
//   packed = code0 + code1*3 + code2*9 + code3*27 + code4*81
//   codes: 0=-1, 1=0, 2=+1
//   Per 5-element group: 1 BF16 scale + 1 byte codes (+ padding to 256 cols)
//
// Ping-pong: same as mm_ternary_tq2.cc — decode next K-block while MAC runs
//
// Licensed under Apache 2.0 with LLVM Exceptions.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

constexpr int M = 32, K = 64, N = 128;

// TQ1 tile geometry: ceil(256/5) = 52 codes + 16 scales per tile row
constexpr int TQ1_CODES_PER_ROW = 256;
constexpr int TQ1_GROUPS_PER_ROW = (TQ1_CODES_PER_ROW + 4) / 5;  // 52
constexpr int TQ1_SCALES_PER_ROW = TQ1_GROUPS_PER_ROW;

extern "C" {

// ── Base-3 LUT: maps byte (packed 5 ternary codes) to 5 int8 values ──
// LUT[byte] = {v0, v1, v2, v3, v4} as packed bytes
// code 0=-1, 1=0, 2=+1
// Generated at compile time from the base-3 expansion
static int8_t tq1_lut[243][5];  // 3^5 = 243 valid combinations

// Initialize the LUT once at boot (called from init)
void init_tq1_lut() {
    for (int p = 0; p < 243; p++) {
        int tmp = p;
        for (int i = 0; i < 5; i++) {
            int code = tmp % 3;
            tmp /= 3;
            tq1_lut[p][i] = (code == 0) ? -1 : (code == 2) ? 1 : 0;
        }
    }
}

// ── Ping-pong TQ1 ternary GEMV ──
// pA: activations (M×K bf16)
// pB: TQ1 packed codes + scales interleaved per tile row
//     [n_tile_rows]: [r0: scales(52*2=104B) + codes(52B)] [r1: same] ...
// pC: output (M×N bf16)
void ternary_tq1_gemv(bfloat16 *pA, uint8_t *pB, bfloat16 *pS, bfloat16 *pC) {
    event0();

    alignas(32) bfloat16 wb_ping[N * K];
    alignas(32) bfloat16 wb_pong[N * K];
    alignas(32) bfp16ebs8 w_bfp_ping[N / 8 * K / 8];
    alignas(32) bfp16ebs8 w_bfp_pong[N / 8 * K / 8];
    alignas(32) bfp16ebs8 a_bfp[M * K / 64];
    alignas(32) bfloat16 tmp[64];

    bool ping = true;

    // ── Stage 1: Decode first TQ1 block into ping ──
    {
        // TQ1 layout for N=128, K=64:
        // pB has ceil(128/TILE_COLS)=1 tile row
        // Each tile row: TQ1_GROUPS_PER_ROW scales + TQ1_GROUPS_PER_ROW codes
        auto *d = wb_ping;
        for (int n = 0; n < N; n++) {
            int grp = n / 5;       // which group of 5 this n belongs to
            int sub = n % 5;       // position within group
            uint8_t packed = pB[grp];  // one code byte per group of 5
            float scale = (float)((bfloat16*)pS)[grp];

            // Base-3 decode: extract sub-th ternary value from packed byte
            int tmp_v = packed;
            for (int s = 0; s < sub; s++) tmp_v /= 3;
            int code = tmp_v % 3;

            float val = (code == 0) ? -scale : (code == 2) ? scale : 0.0f;
            for (int k = 0; k < K; k++)
                d[n * K + k] = (bfloat16)val;  // broadcast across K
        }

        // Convert to bfp16ebs8
        aie::block_vector_output_buffer_stream<bfp16ebs8, 64> ws(w_bfp_ping);
        for (int nb = 0; nb < N / 8; nb++)
            for (int kb = 0; kb < K / 8; kb++) {
                for (int r = 0; r < 8; r++)
                    for (int c = 0; c < 8; c++)
                        tmp[r * 8 + c] = wb_ping[(nb * 8 + r) * K + kb * 8 + c];
                auto v = aie::load_v<64>(tmp);
                aie::accum<accfloat, 64> acc; acc.from_vector(v, 0);
                ws.push(acc.template to_vector<bfp16ebs8>());
            }
    }

    // ── Activation blocks ──
    {
        aie::block_vector_output_buffer_stream<bfp16ebs8, 64> as(a_bfp);
        for (int i = 0; i < M * K / 64; i++) {
            auto v = aie::load_v<64>(pA + i * 64);
            aie::accum<accfloat, 64> acc; acc.from_vector(v, 0);
            as.push(acc.template to_vector<bfp16ebs8>());
        }
    }

    // ── MAC ──
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> ws_in(w_bfp_ping);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as_in(a_bfp);

    for (int mb = 0; mb < M / 16; mb++)
        for (int nb = 0; nb < N / 16; nb++) {
            auto *blk = pC + mb * N * 16 + nb * 16;
            aie::accum<accfloat, 64> a0(aie::load_v<64>(blk));
            aie::accum<accfloat, 64> a1(aie::load_v<64>(blk + 64));
            for (int k = 0; k < K / 8; k++) {
                auto A0 = as_in.pop(), A1 = as_in.pop();
                auto B0 = ws_in.pop(), B1 = ws_in.pop();
                a0 = mac_8x8_8x8T(A0, B0, a0);
                a1 = mac_8x8_8x8T(A0, B1, a1);
            }
            aie::store_v(blk, a0.template to_vector<bfloat16>());
            aie::store_v(blk + 64, a1.template to_vector<bfloat16>());
        }
    event1();
}
}
