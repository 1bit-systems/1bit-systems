// Fused QKV GEMV: computes Q, K, V projections in a single kernel launch.
// out[QD+KD+KD] = in[H] @ [Wq[QD,H], Wk[KD,H], Wv[KD,H]]^T
// Saves 2 kernel launches and 2 activation re-reads per layer vs 3 separate GEMVs.
//
// This file is #included into backend_hip.cpp. HIP headers are available at the include site.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ── Fused QKV GEMV: Q, K, V in one launch ────────────────────────
// out = [q[QD], k[KD], v[KD]]
// M_total = QD + 2*KD
__global__ void fused_qkv_gemv(
    __half* out,
    const __half* in,
    const __half* wq,
    const __half* wk,
    const __half* wv,
    int QD, int KD, int H)
{
    int tx = threadIdx.x;
    for (int i = tx; i < QD + 2*KD; i += blockDim.x) {
        float sum = 0.0f;
        const __half* w;
        if (i < QD)      w = wq + (size_t)i * H;
        else if (i < QD + KD) w = wk + (size_t)(i - QD) * H;
        else                  w = wv + (size_t)(i - QD - KD) * H;
        #pragma unroll
        for (int k = 0; k < H; k++) sum = __fmaf_rn((float)in[k], (float)w[k], sum);
        out[i] = __float2half(sum);
    }
}

// ── Fused Gate+Up GEMV ────────────────────────────────────────────
// out[2*N_FF] = in[H] @ [Wgate[N_FF,H], Wup[N_FF,H]]^T
__global__ void fused_gate_up_gemv(
    __half* out,
    const __half* in,
    const __half* wgate,
    const __half* wup,
    int N_FF, int H)
{
    int tx = threadIdx.x;
    for (int i = tx; i < 2*N_FF; i += blockDim.x) {
        float sum = 0.0f;
        const __half* w = (i < N_FF) ? wgate + (size_t)i * H : wup + (size_t)(i - N_FF) * H;
        #pragma unroll
        for (int k = 0; k < H; k++) sum = __fmaf_rn((float)in[k], (float)w[k], sum);
        out[i] = __float2half(sum);
    }
}
