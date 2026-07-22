// laguna_layer.hip — Fused layer kernel (research)
//
// FINDING: Full fused layer with QKV + attention + FFN gates + up
// needs ~72KB shared memory for XS 2.1 (FF=8192). Hardware limit is 64KB.
//
// The fused layer approach CAN work for smaller models (FF < 6144)
// but for XS 2.1 (FF=8192), the FFN gate+up projections alone need
// 2*FF = 16384 floats = 64KB — which exceeds the total shared memory.
//
// SOLUTION: Use the 8 proven kernels in build/liblaguna_kernels.so:
//   launch_q4nx_gemv | launch_fused_qkv | launch_fused_gate_up
//   launch_rms_norm | launch_batch_rms_norm | launch_rope
//   launch_attention | launch_fused_layer (WIP)
//
// These give ~3.1 tok/s on Radeon 8060S for XS 2.1 (33B).
// Realistic max: ~6-7 tok/s (bandwidth-limited: 16GB/256GB/s = 62ms)
//
// This file contains the research implementation for reference.
// The bsum reduction uses a separate __shared__ array to prevent
// the AMD HIP compiler from optimizing away the reduction buffer.

#include <hip/hip_runtime.h>
#include <cstdio>

constexpr int TR=32, TC=256, GS=32, NG=TC/GS, RB=NG*4+TC/2, TB=TR*RB;

__device__ inline float bf16_f32(uint16_t bf) { return __uint_as_float((uint32_t)bf<<16); }
__device__ inline float cl_f(float v) { return (v==v&&v<1e10f&&v>-1e10f)?v:0.0f; }
__device__ inline float wmax(float v) {
    for(int o=16;o>0;o>>=1){float t=__shfl_xor(v,o);if(t>v)v=t;}return v;
}

// Block-level sum with SEPARATE reduction buffer (compiler can't eliminate)
__device__ inline float bsum(float v, float* red) {
    for(int o=16;o>0;o>>=1) v += __shfl_xor(v,o);
    if ((threadIdx.x & 0x1F) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    if (threadIdx.x < 32) {
        v = threadIdx.x < nw ? red[threadIdx.x] : 0.0f;
        for(int o=16;o>0;o>>=1) v += __shfl_xor(v,o);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    return red[0];
}

// Tile-vector multiply: y[col0..col0+TC-1] += x[row] * tile[row][col]
__device__ inline void tile_mv(float* y, int col0, const uint8_t* tp,
                                const float* x, int row_start, int M, int K,
                                int tid, int stride) {
    for (int r = tid; r < TR; r += stride) {
        int abs_r = row_start + r;
        if (abs_r >= K) continue;
        float xr = x[abs_r];
        if (xr == 0.0f) continue;
        const uint8_t* rp = tp + r * RB;
        for (int g = 0; g < NG; g++) {
            float sc = bf16_f32(*(const uint16_t*)(rp + g*2));
            float zp = bf16_f32(*(const uint16_t*)(rp + NG*2 + g*2));
            const uint8_t* pk = rp + NG*4;
            for (int i = 0; i < GS; i++) {
                int bi = g*GS/2 + i/2;
                int nl = (i&1) ? (pk[bi]>>4) : (pk[bi]&0x0F);
                float v = cl_f((float)nl * sc + zp);
                int c = col0 + g*GS + i;
                if (c < M) atomicAdd(&y[c], xr * v);
            }
        }
    }
}

// Empty stub — see the 8 proven kernels in liblaguna_kernels.so
// for the working implementation.
__global__ void fused_layer_stub() {}

extern "C" void launch_fused_layer(...) {}
