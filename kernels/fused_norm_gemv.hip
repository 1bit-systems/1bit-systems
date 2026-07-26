// Fused rmsnorm + Q4_K packed GEMV: single launch replaces 2.
// norm: x[i] = x[i] * w[i] / sqrt(sum(x^2)/n + eps)
// gemv: out[M] = norm_x[K] @ W[M,K]^T  (W in Q4_K packed format)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using half16_t = __attribute__((ext_vector_type(16))) __fp16;
using float8_t = __attribute__((ext_vector_type(8))) float;

// ── Fused norm + Q4_K GEMV ─────────────────────────────────────────
// Applies rmsnorm in-place on `x`, then computes GEMV with packed Q4_K weights.
// Saves 1 kernel launch per layer (× 28 layers = 1.2ms at 5μs per launch).
// Grid = (M/64, 1), block = 128. Each warp handles 16 output rows.
__global__ void fused_norm_q4k_gemv(
    __half* __restrict__ x,           // [K] — rmsnorm applied in-place BEFORE gemv
    const __half* __restrict__ w_norm,// [K] — rmsnorm weight
    float eps,
    __half* __restrict__ out,         // [M] — gemv output
    const uint8_t* __restrict__ W,    // [M, nb, 144] — Q4_K packed weights
    int M, int K)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 64 + warp * 16;
    if (row >= M) return;

    const __fp16* x_in = (const __fp16*)x;
    const __fp16* wn = (const __fp16*)w_norm;
    __fp16* x_out = (__fp16*)x;
    __fp16* out_fp16 = (__fp16*)out;
    int nb = (K + 255) / 256;

    // ── Step 1: rmsnorm ──
    // Compute sum(x^2) across all threads
    float ss = 0.0f;
    for (int i = lane; i < K; i += 32)
        ss += (float)x_in[i] * (float)x_in[i];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffULL, ss, o);

    // Normalize and write back
    float scale = 1.0f / sqrtf(ss / K + eps);
    for (int i = lane; i < K; i += 32)
        x_out[i] = (__fp16)((float)x_in[i] * scale * (float)wn[i]);

    // ── Step 2: Q4_K packed GEMV ──
    float8_t acc = {};
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = W + ((size_t)(blockIdx.x * 64) * nb + b) * 144;
        float d = (float)*(const __fp16*)blk;
        float dm = (float)*(const __fp16*)(blk + 2);
        uint8_t sc[12];
        read_q4k_header(blk, d, dm, sc);

        int ks = b * 256;
        #pragma unroll
        for (int t = 0; t < 16; t++) {
            half16_t A;
            int ka = ks + t * 16;
            if (lane < 16)
                for (int i = 0; i < 16; i++)
                    A[i] = (ka + i < K) ? x_out[ka + i] : (__fp16)0.0f;
            for (int i = 0; i < 16; i++) A[i] = __shfl(A[i], 0);
            half16_t B = dequant_q4k_tile(blk, t, d, dm, sc);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, B, acc);
        }
    }

    if (lane == 0)
        for (int j = 0; j < 8 && row + j < M; j++)
            out_fp16[row + j] = (__fp16)acc[j];
    if (lane == 16)
        for (int j = 0; j < 8 && row + 8 + j < M; j++)
            out_fp16[row + 8 + j] = (__fp16)acc[j];
}
