// laguna_fused.hip — Fused QKV + GateUp matmuls
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>

constexpr int TR=32, TC=256, GS=32, NG=TC/GS, RB=NG*4+TC/2, TB=TR*RB;

__device__ inline float bf16_f32(uint16_t bf) {
    return __uint_as_float((uint32_t)bf << 16);
}

// Fused QKV: grid dims = (ntcq + ntck + ntcv, n_tr). Each block handles one matrix tile.
__global__ void fused_qkv(
    const float* __restrict__ x,
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ wk,
    const uint8_t* __restrict__ wv,
    float* __restrict__ y,
    int K, int Mq, int Mk, int Mv,
    int ntcq, int ntck, int ntcv
) {
    int trr = blockIdx.y;
    int bx = blockIdx.x;
    int r = threadIdx.x;
    int rs = trr * TR, re = min(rs + TR, K);
    if (r >= (re - rs)) return;
    
    // Determine which matrix and local tile column
    const uint8_t* w;
    int M, off, ntc;
    if (bx < ntcq) { w = wq; M = Mq; off = 0; ntc = ntcq; }
    else if (bx < ntcq + ntck) { w = wk; M = Mk; off = Mq; ntc = ntck; }
    else { w = wv; M = Mv; off = Mq + Mk; ntc = ntcv; }
    int tcc = bx - (bx < ntcq ? 0 : (bx < ntcq+ntck ? ntcq : ntcq+ntck));
    
    float xr = x[rs + r];
    if (xr == 0.0f) return;
    
    int cs = tcc * TC, ce = min(cs + TC, M);
    const uint8_t* tp = w + (((size_t)trr * ntc + tcc) * TB);
    const uint8_t* rp = tp + r * RB;
    
    for (int g = 0; g < NG; g++) {
        float sc = bf16_f32(*(const uint16_t*)(rp + g*2));
        float zp = bf16_f32(*(const uint16_t*)(rp + NG*2 + g*2));
        const uint8_t* pk = rp + NG*4;
        for (int i = 0; i < GS; i++) {
            int bi = g*GS/2 + i/2;
            int nl = (i&1) ? (pk[bi]>>4) : (pk[bi]&0x0F);
            float val = (float)nl * sc + zp;
            if (val != val || val > 1e10f || val < -1e10f) val = 0;
            int col = g*GS + i;
            if (col < (ce - cs)) atomicAdd(&y[off + cs + col], xr * val);
        }
    }
}

// Fused Gate+Up
__global__ void fused_gate_up(
    const float* __restrict__ x,
    const uint8_t* __restrict__ wg,
    const uint8_t* __restrict__ wu,
    float* __restrict__ y,
    int K, int FF, int ntc_g, int ntc_u
) {
    int trr = blockIdx.y, bx = blockIdx.x, r = threadIdx.x;
    int rs = trr * TR, re = min(rs + TR, K);
    if (r >= (re - rs)) return;
    
    bool is_up = (bx >= ntc_g);
    const uint8_t* w = is_up ? wu : wg;
    int tcc = is_up ? (bx - ntc_g) : bx;
    int off = is_up ? FF : 0;
    int ntc = is_up ? ntc_u : ntc_g;
    int M = FF;
    
    float xr = x[rs + r];
    if (xr == 0.0f) return;
    
    int cs = tcc * TC, ce = min(cs + TC, M);
    const uint8_t* tp = w + (((size_t)trr * ntc + tcc) * TB);
    const uint8_t* rp = tp + r * RB;
    for (int g = 0; g < NG; g++) {
        float sc = bf16_f32(*(const uint16_t*)(rp + g*2));
        float zp = bf16_f32(*(const uint16_t*)(rp + NG*2 + g*2));
        const uint8_t* pk = rp + NG*4;
        for (int i = 0; i < GS; i++) {
            int bi = g*GS/2 + i/2;
            int nl = (i&1) ? (pk[bi]>>4) : (pk[bi]&0x0F);
            float val = (float)nl * sc + zp;
            if (val != val || val > 1e10f || val < -1e10f) val = 0;
            int col = g*GS + i;
            if (col < (ce - cs)) atomicAdd(&y[off + cs + col], xr * val);
        }
    }
}

extern "C" void launch_fused_qkv(
    const float* x, const uint8_t* wq, const uint8_t* wk, const uint8_t* wv,
    float* y, int K, int Mq, int Mk, int Mv, int ntcq, int ntck, int ntcv, void* stream
) {
    int n_tr = (K+TR-1)/TR;
    dim3 grid(ntcq+ntck+ntcv, n_tr);
    hipLaunchKernelGGL(fused_qkv, grid, TR, 0, (hipStream_t)stream,
        x, wq, wk, wv, y, K, Mq, Mk, Mv, ntcq, ntck, ntcv);
}

extern "C" void launch_fused_gate_up(
    const float* x, const uint8_t* wg, const uint8_t* wu,
    float* y, int K, int FF, int ntc_g, int ntc_u, void* stream
) {
    int n_tr = (K+TR-1)/TR;
    dim3 grid(ntc_g+ntc_u, n_tr);
    hipLaunchKernelGGL(fused_gate_up, grid, TR, 0, (hipStream_t)stream,
        x, wg, wu, y, K, FF, ntc_g, ntc_u);
}
