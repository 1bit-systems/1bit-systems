// zaya_fused_qkv.hip — Fused RMSNorm + Q/K/V projection (#4)
// Replaces 5 kernel launches per attention layer with 1.
// Parameters use Hdim/qdim/kdim to avoid macro conflicts.
//
// NOTE: RMSNorm and QKV projection are SEPARATE kernel launches to avoid
// cross-block data races. Block 0 writing hs[] for RMSNorm while other blocks
// read hs[] for QKV projection would be unsynchronized across blocks — GPU
// thread blocks can execute in any order, and __syncthreads() only syncs within
// a single block. See issue #870.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define FQ_M   16
#define FQ_THREADS 128
#define FQ_RMS_BLOCK 256

// ── Step 1: RMSNorm kernel (separate launch — not fused with QKV) ──
// Writes normalized values back to hs[].
__launch_bounds__(256) __global__ void rmsnorm_fused_kernel(
    __half* __restrict__ hs,
    const __half* __restrict__ nw,
    int Hdim)
{
    int tx = threadIdx.x;
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

// ── Step 2: Fused QKV projection kernel (no RMSNorm — reads pre-normalized hs) ──
// Projects hs[Hdim] into Q[qdim], K[kdim], V1[kdim/2], V2[kdim/2] in tmp buffer.
// Must be launched AFTER rmsnorm_fused_kernel completes.
__launch_bounds__(128) __global__ void fused_qkv_kernel(
    const __half* __restrict__ hs,      // pre-normalized hidden state (read-only)
    __half* __restrict__ tmp,           // [qdim + kdim + kdim] output buffer
    const __half* __restrict__ wq,
    const __half* __restrict__ wk,
    const __half* __restrict__ wv1,
    const __half* __restrict__ wv2,
    int Hdim, int qdim, int kdim)
{
    int bid = blockIdx.x;
    int tx = threadIdx.x;
    int hv2 = kdim / 2;

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
