// zaya_fused_qkv.hip — Fused RMSNorm + Q/K/V projection (#4)
//
// Replaces 5 kernel launches per attention layer with 1.
// Note: parameters use names that avoid conflicting with macros from
// other included .hip files (H, qd, kd are all common macro names).
//
// Block distribution for Zaya1-8B: blocks handle Q(0..63), K(64..79),
// V1(80..87), V2(88..95) = 96 blocks @ 128 threads each.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define FQ_M   16
#define FQ_THREADS 128
#define FQ_RMS_BLOCK 256

__launch_bounds__(128) __global__ void fused_rmsnorm_qkv_kernel(
    __half* __restrict__ hs,
    const __half* __restrict__ nw,
    __half* __restrict__ tmp,
    const __half* __restrict__ wq,
    const __half* __restrict__ wk,
    const __half* __restrict__ wv1,
    const __half* __restrict__ wv2,
    int Hdim, int qdim, int kdim)
{
    int bid = blockIdx.x;
    int tx = threadIdx.x;
    int hv2 = kdim / 2;

    // Phase 1: Block 0 computes RMSNorm
    if (bid == 0) {
        __shared__ float r[8];
        int wid = tx / 32;
        int lane = tx % 32;
        float ss = 0;
        for (int i = tx; i < Hdim; i += FQ_RMS_BLOCK) {
            float v = (float)hs[i];
            ss += v * v;
        }
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o);
        if (lane == 0) r[wid] = ss;
        __syncthreads();
        if (wid == 0) {
            ss = (lane < 8) ? r[lane] : 0;
            for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o);
            if (lane == 0) r[0] = ss;
        }
        __syncthreads();
        float iv = 1.0f / sqrtf(r[0] / Hdim + 1e-5f);
        for (int i = tx; i < Hdim; i += FQ_RMS_BLOCK) {
            hs[i] = __float2half((float)hs[i] * iv * (float)nw[i]);
        }
    }
    __syncthreads();

    // Phase 2: Determine which projection this block computes
    int qb = (qdim + FQ_M - 1) / FQ_M;
    int kb = (kdim + FQ_M - 1) / FQ_M;
    int v1b = (hv2 + FQ_M - 1) / FQ_M;
    int v2b = (hv2 + FQ_M - 1) / FQ_M;

    int out_M, out_off;
    const __half* wt;

    if (bid < qb) {
        out_M = qdim; out_off = 0; wt = wq;
    } else if (bid < qb + kb) {
        out_M = kdim; out_off = qdim; wt = wk; bid -= qb;
    } else if (bid < qb + kb + v1b) {
        out_M = hv2; out_off = qdim + kdim; wt = wv1; bid -= qb + kb;
    } else {
        bid -= qb + kb + v1b;
        if (bid >= v2b) return;
        out_M = hv2; out_off = qdim + kdim + hv2; wt = wv2;
    }

    int row = bid * FQ_M;
    if (row >= out_M) return;

    float acc[FQ_M] = {0.0f};
    for (int k = tx; k < Hdim; k += FQ_THREADS) {
        float v_in = (float)hs[k];
        for (int j = 0; j < FQ_M; j++)
            acc[j] = __fmaf_rn(v_in, (float)wt[(row + j) * (size_t)Hdim + k], acc[j]);
    }

    for (int j = 0; j < FQ_M; j++)
        for (int o = 16; o > 0; o >>= 1)
            acc[j] += __shfl_xor(acc[j], o);

    __shared__ float s_acc[FQ_M * 4];
    int warp = tx / 32;
    int lane = tx % 32;
    if (lane == 0)
        for (int j = 0; j < FQ_M; j++)
            s_acc[warp * FQ_M + j] = acc[j];
    __syncthreads();

    if (warp == 0 && lane < FQ_THREADS / 32) {
        for (int j = 0; j < FQ_M; j++)
            acc[j] = s_acc[lane * FQ_M + j];
        for (int o = 16; o > 0; o >>= 1)
            for (int j = 0; j < FQ_M; j++)
                acc[j] += __shfl_xor(acc[j], o);
        if (lane == 0)
            for (int j = 0; j < FQ_M; j++)
                if (row + j < out_M)
                    tmp[out_off + row + j] = __float2half(acc[j]);
    }
}
