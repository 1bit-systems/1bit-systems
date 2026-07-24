// mm_ternary_tq2.cc — TQ2 ternary — LUT decode + block-vectorized mac_8x8_8x8T
//
// Decodes ternary to bf16 in L1, feeds through block_vector streams for
// bfp16ebs8 conversion, then mac_8x8_8x8T for full vector throughput.
//
// The 8×8 weight block is assembled into a contiguous temp buffer, loaded
// as a single vector, converted to bfp16ebs8 via accfloat, then pushed
// to the output stream. The input stream's pop() returns the native type
// mac_8x8_8x8T needs.
//
// Licensed under Apache 2.0 with LLVM Exceptions.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

constexpr int M = 32, K = 64, N = 128;

extern "C" {
static int g = 0;

void ternary_tq2_gemv(bfloat16 *pA, uint8_t *pB, bfloat16 *pS, bfloat16 *pC) {
    event0();
    pC += g * M * N; if (g == 3) g = 0; else g++;

    // LUT-decode ternary → bf16
    alignas(32) bfloat16 w[M * N]; // transposed: w[row*n + col] for contiguous 8×8 blocks
    // Actually store in [n*K + k] layout, use temp buffer for 8×8 assembly
    
    // Decode ternary codes to bf16 in [n*K + k] layout
    alignas(32) bfloat16 wb[N * K];
    static const bfloat16 cv[4] = {(bfloat16)-1,(bfloat16)0,(bfloat16)1,(bfloat16)0};
    for (int n = 0; n < N; n++) {
        auto *s = pB + n * K / 4;
        auto *d = wb + n * K;
        for (int i = 0; i < 16; i++) {
            uint8_t b = s[i];
            d[i*4+0] = cv[b & 3]; d[i*4+1] = cv[(b>>2)&3];
            d[i*4+2] = cv[(b>>4)&3]; d[i*4+3] = cv[(b>>6)&3];
        }
    }
    // Apply scales
    for (int n = 0; n < N; n++)
        for (int grp = 0; grp < 2; grp++)
            for (int i = 0; i < 32; i++)
                wb[n * K + grp * 32 + i] = wb[n * K + grp * 32 + i] * pS[n * 2 + grp];

    // Convert to bfp16ebs8 blocks: assemble 8×8 tiles into contiguous buffer
    alignas(32) bfp16ebs8 w_bfp[N / 8 * K / 8];
    alignas(32) bfp16ebs8 a_bfp[M * K / 64];

    // Weight blocks: copy 8 strided rows into a flat temp buffer, then convert
    {
        aie::block_vector_output_buffer_stream<bfp16ebs8, 64> ws(w_bfp);
        alignas(32) bfloat16 tmp[64]; // 8×8 contiguous buffer
        for (int nb = 0; nb < N / 8; nb++) {
            for (int kb = 0; kb < K / 8; kb++) {
                // Copy 8 rows × 8 cols from strided layout to contiguous
                for (int r = 0; r < 8; r++)
                    for (int c = 0; c < 8; c++)
                        tmp[r * 8 + c] = wb[(nb * 8 + r) * K + kb * 8 + c];
                // Load as 64-element bf16 vector (contiguous → one load_v<64>)
                auto v = aie::load_v<64>(tmp);
                aie::accum<accfloat, 64> acc; acc.from_vector(v, 0);
                ws.push(acc.template to_vector<bfp16ebs8>());
            }
        }
    }
    // Activation blocks: contiguous in memory, load directly
    {
        aie::block_vector_output_buffer_stream<bfp16ebs8, 64> as(a_bfp);
        for (int i = 0; i < M * K / 64; i++) {
            auto v = aie::load_v<64>(pA + i * 64);
            aie::accum<accfloat, 64> acc; acc.from_vector(v, 0);
            as.push(acc.template to_vector<bfp16ebs8>());
        }
    }

    // mac_8x8_8x8T via stream pop()
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> ws_in(w_bfp);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as_in(a_bfp);

    for (int mb = 0; mb < M / 16; mb++) {
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
    }
    event1();
}
void zero_kernel_ternary(bfloat16 *cOut) {
    auto z = aie::zeros<bfloat16, 32>();
    for (int i = 0; i < M * N; i += 32) aie::store_v(cOut + i, z);
}
}
