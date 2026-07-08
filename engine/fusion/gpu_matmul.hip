/* HIP kernel: I8 tile dequant + matmul on Radeon 8060S (55 TFLOPS)
 * Each workgroup processes 32 output rows from one tile (32×256).
 * Grid: (ntr * ntc) workgroups, each with 32 work items.
 */
#include <hip/hip_runtime.h>
#include <stdio.h>

#define TILE_R 32
#define TILE_C 256
#define I8_ROW_B 5120

/* Convert BF16 stored as uint16 to float32 */
__device__ float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    return __builtin_bit_cast(float, bits);
}

/* One workgroup processes one 32×256 tile.
 * Each work item (thread) processes one output row.
 * wt: pointer to I8 tile data (5120 bytes)
 * inp: input vector (256 floats)
 * out: output vector (written to out[tile_row*32 + thread_id])
 */
__global__ void i8_tile_mul_kernel(
    const uint8_t* __restrict__ wt,
    const float*   __restrict__ inp,
    float*         __restrict__ out,
    int tile_row,      /* which tile row (for output offset) */
    int tile_col,      /* which tile col (for input offset) */
    int ntc,           /* tiles per col */
    int out_stride     /* output stride (total output dim) */
) {
    int lr = threadIdx.x;  /* 0..31, which row in this tile */
    if (lr >= TILE_R) return;

    /* Tile data layout: [256 BF16 scales][256 BF16 zps][4096 packed I4] */
    const uint16_t* sc = (const uint16_t*)wt;
    const uint16_t* zp = (const uint16_t*)(wt + 512);
    const uint8_t*  pk = wt + 1024;

    int lane = lr / 16;
    int lane_row = lr % 16;
    int bi = lane_row / 2;
    int ns = lr % 2;
    const uint8_t* lane_data = pk + lane * (TILE_C * 8);

    double acc = 0.0;
    int inp_base = tile_col * TILE_C;

    #pragma unroll 8
    for (int g = 0; g < 8; g++) {
        float s = bf16_to_f32(sc[g * 32 + lr]);
        float z = bf16_to_f32(zp[g * 32 + lr]);
        #pragma unroll 32
        for (int c = 0; c < 32; c++) {
            int col = g * 32 + c;
            uint8_t bv = lane_data[col * 8 + bi];
            int cd = (ns == 0) ? (bv & 0x0F) : ((bv >> 4) & 0x0F);
            float w = (float)cd * s + z;
            acc += (double)w * (double)inp[inp_base + col];
        }
    }

    out[tile_row * TILE_R + lr] += (float)acc;
}

/* Host-side matmul: dispatch tiles for one weight matrix */
extern "C" void gpu_i8_mm(
    const uint8_t* d_wt,    /* device weight pointer */
    const float*   d_inp,   /* device input */
    float*         d_out,   /* device output */
    int out_dim,            /* output dimension */
    int in_dim,             /* input dimension */
    hipStream_t stream
) {
    int ntc = in_dim / TILE_C; if (ntc < 1) ntc = 1;
    int ntr = out_dim / TILE_R; if (ntr < 1) ntr = 1;

    dim3 grid(ntr * ntc);
    dim3 block(TILE_R);

    for (int tr = 0; tr < ntr; tr++) {
        for (int tc = 0; tc < ntc; tc++) {
            const uint8_t* tile = d_wt + ((size_t)tr * ntc + tc) * I8_ROW_B;
            hipLaunchKernelGGL(i8_tile_mul_kernel, dim3(1), block, 0, stream,
                tile, d_inp, d_out, tr, tc, ntc, out_dim);
        }
    }
}
