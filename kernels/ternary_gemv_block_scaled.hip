// Block-Scaled Ternary GEMV — multiplication-free ternary with per-block FP8 scale.
// 16 ternary values per block, one FP8 E4M3 scale per block.
// No multiply on hot path (conditional add/subtract), one scale FMA per 16 elements.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>

#define BLOCK_SIZE  128
#define WARP_SIZE   32
#define NUM_WARPS   (BLOCK_SIZE / WARP_SIZE)
#define BST_BLOCK_K   16
#define BST_BLOCK_BYTES 5
#define LDS_TILE_I8   2048

__device__ __forceinline__ float fp8e4m3_to_f32(uint8_t fp8) {
    // Uses precomputed FP32 subnormal table from include/block_scaled_ternary.h
    if (fp8 == 0xFF) { uint32_t b = 0x7FC00000u; float r; __builtin_memcpy(&r,&b,4); return r; }
    uint32_t s=(fp8>>7)&1,e=(fp8>>3)&0xF,m=fp8&0x7;
    uint32_t b;
    if(e==0){
        static const uint32_t tbl[8]={0,0x3B000000u,0x3B800000u,0x3BC00000u,0x3C000000u,0x3C200000u,0x3C400000u,0x3C600000u};
        b = tbl[m] | (s<<31);
    } else b = (s<<31)|((e+120)<<23)|(m<<20);
    float r; __builtin_memcpy(&r,&b,4); return r;
}

__attribute__((amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))
__global__ void ternary_gemv_block_scaled_kernel(
    const uint8_t* __restrict__ packed, int8_t* __restrict__ x_i8,
    float x_scale, float* __restrict__ y, int M, int K)
{
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x, lane = tid & (WARP_SIZE-1), warp_id = tid / WARP_SIZE;
    // Ceiling division — handles K not multiple of 16
    int blocks_per_row = (K + BST_BLOCK_K - 1) / BST_BLOCK_K;
    const uint8_t* row_w = packed + (size_t)row * blocks_per_row * BST_BLOCK_BYTES;
    __shared__ int8_t xs[LDS_TILE_I8];
    float acc = 0.0f;
    int kb_max = ((K + LDS_TILE_I8 - 1) / LDS_TILE_I8) * LDS_TILE_I8;
    for (int kb = 0; kb < kb_max; kb += LDS_TILE_I8) {
        __syncthreads();
        // Load activations with zero-padding at tail
        int ks = kb + tid * 16;
        for (int b = 0; b < 16; ++b) {
            int k = ks + b;
            xs[tid * 16 + b] = (k < K) ? x_i8[k] : (int8_t)0;
        }
        __syncthreads();
        int tb = (LDS_TILE_I8 + BST_BLOCK_K - 1) / BST_BLOCK_K;
        for (int b = tid; b < tb; b += BLOCK_SIZE) {
            int gb = kb / BST_BLOCK_K + b;
            if (gb >= blocks_per_row) break;
            const uint8_t* bp = row_w + (size_t)gb * BST_BLOCK_BYTES;
            uint32_t w; uint8_t sb;
            __builtin_memcpy(&w, bp, 4);
            sb = bp[4];
            float bs = x_scale * fp8e4m3_to_f32(sb);
            int32_t ba = 0;
            int xo = b * BST_BLOCK_K;
            int ve = (xo + BST_BLOCK_K <= K) ? BST_BLOCK_K : (K - xo);
            #pragma unroll
            for (int v = 0; v < BST_BLOCK_K; v++) {
                int xv = (v < ve) ? (int)xs[xo + v] : 0;
                uint32_t bits = (w >> (v * 2)) & 3;
                int sign = (int)(bits == 1) - (int)(bits == 2);
                ba += sign * xv;
            }
            acc += (float)ba * bs;
        }
    }
    #pragma unroll
    for (int o = WARP_SIZE/2; o > 0; o >>= 1) acc += __shfl_xor(acc, o);
    __shared__ float wr[NUM_WARPS];
    if (lane == 0) wr[warp_id] = acc;
    __syncthreads();
    if (warp_id == 0) {
        acc = (lane < NUM_WARPS) ? wr[lane] : 0.0f;
        #pragma unroll
        for (int o = WARP_SIZE/2; o > 0; o >>= 1) acc += __shfl_xor(acc, o);
        if (lane == 0) y[row] = acc;
    }
}

extern "C" void ternary_gemv_block_scaled_launch(
    const void* packed, const void* x_i8, float x_scale,
    void* y, int M, int K, void* stream)
{
    hipLaunchKernelGGL(ternary_gemv_block_scaled_kernel,
        dim3(M,1,1), dim3(BLOCK_SIZE,1,1), 0, (hipStream_t)stream,
        (const uint8_t*)packed, (int8_t*)x_i8, x_scale, (float*)y, M, K);
}
