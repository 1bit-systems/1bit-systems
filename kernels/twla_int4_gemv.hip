// TWLA-style ternary × INT4 GEMV (W1.58A4)
// Drop-in improvement over ternary_gemv_phase5_dot4:
//   INT4 activations instead of INT8 → 2× memory bandwidth
//   16 ternary + 1 FP8 scale per block → 9 bytes vs 8+4=12 bytes
//
// API matches existing launcher pattern for easy integration.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>

#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define LDS_TILE 2048  // INT8 activations (dequantized from INT4)
#define TWLA_BLOCK_K 32
#define TWLA_BLOCK_BYTES 9  // 4B (packed u32) + 1B (FP8 scale) + 4B (reserved)

__device__ inline float fp8e4m3_to_f32(uint8_t b) {
    if (b == 0) return 0.0f;
    uint32_t s = (uint32_t)(b >> 7) << 31;
    int e = (b >> 3) & 0xF;
    uint32_t m = (uint32_t)(b & 7) << 20;
    uint32_t f = s | ((e + 120) << 23) | m;
    float r; __builtin_memcpy(&r, &f, 4); return r;
}

// Pack 2 INT4 values per byte
__device__ inline int8_t lo_nibble(uint8_t b) { int8_t v = b & 0xF; return v >= 8 ? v - 16 : v; }
__device__ inline int8_t hi_nibble(uint8_t b) { int8_t v = b >> 4; return v >= 8 ? v - 16 : v; }

__attribute__((amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))
__global__ void twla_int4_gemv_kernel(
    const uint32_t* __restrict__ w,     // [M, K/16] packed ternary
    const uint8_t*  __restrict__ x_i4,  // [K/2]   packed INT4
    const uint8_t*  __restrict__ s,     // [M, K/32] FP8 scales
    float  x_scale,
    float* __restrict__ y,              // [M]
    int M, int K)
{
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int pk = K / 16;
    int nb = (K + TWLA_BLOCK_K - 1) / TWLA_BLOCK_K;
    const uint32_t* rw = w + (size_t)row * pk;
    const uint8_t*  rs = s + (size_t)row * nb;

    __shared__ int8_t xs[LDS_TILE];  // INT8 dequantized from INT4
    float acc = 0.0f;

    for (int kb = 0; kb < K; kb += LDS_TILE) {
        if (kb > 0) __syncthreads();
        // Load INT4 → expand to INT8 in LDS
        int nt = LDS_TILE / 2 / BLOCK_SIZE + 1;
        for (int i = 0; i < nt; i++) {
            int bi = kb/2 + tid + i*BLOCK_SIZE;
            if (bi < K/2) {
                uint8_t b = x_i4[bi];
                xs[(tid + i*BLOCK_SIZE)*2]     = lo_nibble(b);
                xs[(tid + i*BLOCK_SIZE)*2 + 1] = hi_nibble(b);
            }
        }
        __syncthreads();
        int b0 = kb / TWLA_BLOCK_K;
        int bn = (LDS_TILE + TWLA_BLOCK_K - 1) / TWLA_BLOCK_K;
        for (int b = tid; b < bn; b += BLOCK_SIZE) {
            int gb = b0 + b;
            if (gb >= nb) break;
            uint32_t w0 = (gb*TWLA_BLOCK_K/16 < pk) ?
                rw[gb*TWLA_BLOCK_K/16] : 0;
            uint32_t w1 = (gb*TWLA_BLOCK_K/16+1 < pk) ?
                rw[gb*TWLA_BLOCK_K/16+1] : 0;
            float sc = fp8e4m3_to_f32(rs[gb]);
            int ba = 0, xo = b * TWLA_BLOCK_K;
            #pragma unroll
            for (int v = 0; v < 16; v++) {
                int b0 = (w0 >> (v*2)) & 3;
                int b1 = (w1 >> (v*2)) & 3;
                int s0 = (b0==1)-(b0==2), s1 = (b1==1)-(b1==2);
                ba += s0 * (int)xs[xo+v] + s1 * (int)xs[xo+v+16];
            }
            acc += (float)ba * sc;
        }
    }
    acc *= x_scale;
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor(acc, o);
    __shared__ float wr[4];
    if (lane == 0) wr[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (lane < 4) ? wr[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor(acc, o);
        if (lane == 0) y[row] = acc;
    }
}

extern "C" void twla_int4_gemv_launch(
    const void* w, const void* x_i4, const void* s,
    float x_scale, void* y, int M, int K, void* stream)
{
    hipLaunchKernelGGL(twla_int4_gemv_kernel,
        dim3(M,1,1), dim3(BLOCK_SIZE,1,1), 0, (hipStream_t)stream,
        (const uint32_t*)w, (const uint8_t*)x_i4,
        (const uint8_t*)s, x_scale, (float*)y, M, K);
}
