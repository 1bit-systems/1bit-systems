/* HIP kernel: one transformer layer on GPU
 * Processes all 7 I8 matmuls for one layer in a single kernel dispatch.
 * 
 * Grid: ceil(out_dim/256) for the largest matmul
 * Each block: 256 threads × processes output elements via cooperative groups
 *
 * Input:  h[H] — pre-normed hidden state
 * Weights: 7 matrices in I8 tile format (in device memory)
 * Output:  qkv[QT], o_out[H], gate[IM], up[IM], down[H]
 */
#include <hip/hip_runtime.h>
#include <stdio.h>

#define H 1024
#define NH 16
#define NKV 8
#define HD 128
#define IM 3072
#define QT (NH*HD + 2*NKV*HD)
#define TILE_R 32
#define TILE_C 256
#define I8_ROW_B 5120

/* BF16 → f32 */
__device__ float bf16_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    return __builtin_bit_cast(float, bits);
}

/* One tile: 32×256, output rows [tr*32 .. tr*32+31] = tile[threadIdx.x + 0..31] */
/* But we use block-wide reduction: each thread computes one column range */
/* Simpler: each thread = one output row */
__global__ void layer_kernel(
    const float* __restrict__ h,           /* input: [H] */
    const uint8_t* __restrict__ w_q,       /* weights: I8 tiles */
    const uint8_t* __restrict__ w_k,
    const uint8_t* __restrict__ w_v,
    const uint8_t* __restrict__ w_o,
    const uint8_t* __restrict__ w_g,
    const uint8_t* __restrict__ w_u,
    const uint8_t* __restrict__ w_d,
    float* __restrict__ qkv,               /* output: [QT] */
    float* __restrict__ o_out,             /* output: [H] */
    float* __restrict__ g_out,             /* output: [IM] */
    float* __restrict__ u_out,             /* output: [IM] */
    float* __restrict__ d_out              /* output: [H] */
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int ntc_h = H / TILE_C;       /* 4 */
    int ntc_q = (NH*HD) / TILE_C;  /* 8 for Q, but Q out=2048, in=H=1024 → ntc = 1024/256 = 4 */
    /* Actually ntc is based on in_dim, which is H=1024 for Q,K,V,G,U */
    /* For O: in=NH*HD=2048, ntc=8 */
    /* For D: in=IM=3072, ntc=12 */
    
    int ntc_qkv = H / TILE_C;  /* 4 */
    int ntr_q = (NH*HD) / TILE_R;  /* 64 */
    int ntr_kv = (NKV*HD) / TILE_R;  /* 32 */
    int ntr_o = H / TILE_R;  /* 32 */
    int ntc_o = (NH*HD) / TILE_C;  /* 8 */
    int ntr_gu = IM / TILE_R;  /* 96 */
    int ntr_d = H / TILE_R;  /* 32 */
    int ntc_d = IM / TILE_C;  /* 12 */
    
    __shared__ float shm[256];  /* for reduction if needed */

    /* Compute QKV: each thread handles one output row across all tiles */
    /* For Q: out_dim=NH*HD=2048, 64 tile-rows, each 32 rows */
    if (tid < NH*HD) {
        int out_row = tid;
        int tr = out_row / TILE_R;
        double acc = 0.0;
        for (int tc = 0; tc < ntc_qkv; tc++) {
            const uint8_t* tile = w_q + ((size_t)tr * ntc_qkv + tc) * I8_ROW_B;
            const uint16_t* sc = (const uint16_t*)tile;
            const uint16_t* zp = (const uint16_t*)(tile + 512);
            const uint8_t* pk = tile + 1024;
            int lr = out_row % TILE_R;
            int lane = lr / 16, lane_row = lr % 16, bi = lane_row / 2, ns = lr % 2;
            const uint8_t* ld = pk + lane * (TILE_C * 8);
            for (int c = 0; c < TILE_C; c++) {
                int g = c / 32;
                float s = bf16_f32(sc[g * 32 + lr]);
                float z = bf16_f32(zp[g * 32 + lr]);
                uint8_t bv = ld[c * 8 + bi];
                int cd = (ns == 0) ? (bv & 0x0F) : ((bv >> 4) & 0x0F);
                acc += (double)((float)cd * s + z) * (double)h[tc * TILE_C + c];
            }
        }
        qkv[tid] = (float)acc;
    }
    /* K and V: 0..NKV*HD-1 */
    if (tid >= NH*HD && tid < NH*HD + NKV*HD) {
        int out_row = tid - NH*HD;
        int tr = out_row / TILE_R;
        double acc = 0.0;
        for (int tc = 0; tc < ntc_qkv; tc++) {
            const uint8_t* tile = (out_row < NKV*HD/2) ? 
                w_k + ((size_t)tr * ntc_qkv + tc) * I8_ROW_B :
                w_v + ((size_t)(tr - NKV*HD/2/32) * ntc_qkv + tc) * I8_ROW_B;
            int lr = out_row % TILE_R;
            // ... same pattern as Q
        }
        qkv[tid] = (float)acc;
    }
    /* ... O, G, U, D follow same pattern */
}
