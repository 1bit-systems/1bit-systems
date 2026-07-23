// Q4_K packed GEMV with WMMA — 4 independent warps, minimal registers.
// Header/scales broadcast from lane 0; nibbles loaded on-the-fly.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using half16_t = __attribute__((ext_vector_type(16))) __fp16;
using float8_t = __attribute__((ext_vector_type(8))) float;

// ── Header read: returns d, dm, and full scales[12] ───────────────
// All 32 lanes get the header from lane 0 via __shfl.
__device__ __forceinline__ void read_q4k_header(
    const uint8_t* blk,
    float& d, float& dm, uint8_t sc[12])
{
    if (threadIdx.x % 32 == 0) {
        d  = (float)*(const __fp16*)(blk);
        dm = (float)*(const __fp16*)(blk + 2);
        for (int i = 0; i < 12; i++) sc[i] = blk[4 + i];
    }
    d  = __shfl(d, 0);
    dm = __shfl(dm, 0);
    for (int i = 0; i < 12; i++) sc[i] = (uint8_t)__shfl((int)sc[i], 0);
}

// ── On-the-fly dequant of one tile (16 elements) ──────────────────
// Reads nibbles from blk + 16, returns half16_t vector.
__device__ __forceinline__ half16_t dequant_q4k_tile(
    const uint8_t* blk, int tile, float d, float dm, const uint8_t sc[12])
{
    int group = tile >> 1;
    uint8_t sv, mv;
    if (group < 4) { sv = sc[group] & 63; mv = sc[group + 4] & 63; }
    else { sv = (sc[group + 4] & 0xF) | ((sc[group - 4] >> 6) << 4);
           mv = (sc[group + 4] >> 4) | ((sc[group] >> 6) << 4); }
    float sf = d * (float)sv, mf = dm * (float)mv;
    const uint8_t* qs = blk + 16 + tile * 8;
    half16_t r;
    #pragma unroll
    for (int i = 0; i < 16; i++)
        r[i] = (__fp16)((float)(int)((qs[i/2] >> ((i&1)<<2)) & 0xF) * sf - mf);
    return r;
}

// ── Q4_K WMMA GEMV: 4 warps, 64 rows/block ───────────────────────
// Grid = (M/64, 1), block = 128. No shared memory, no sync.
// Each warp loads the Q4_K block independently (L1 cache hit after warp 0).
__global__ void q4k_gemv_wmma(
    __half* __restrict__ out_,
    const __half* __restrict__ in_,
    const uint8_t* __restrict__ W,
    int M, int K)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 64 + warp * 16;
    if (row >= M) return;

    const __fp16* in = (const __fp16*)in_;
    __fp16* out = (__fp16*)out_;
    int nb = (K + 255) / 256;
    float8_t acc = {};

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = W + ((size_t)(blockIdx.x * 64) * nb + b) * 144;

        // Read header: all lanes get d, dm, scales (broadcast from lane 0)
        float d, dm; uint8_t sc[12];
        read_q4k_header(blk, d, dm, sc);

        int ks = b * 256;
        #pragma unroll
        for (int t = 0; t < 16; t++) {
            half16_t A;
            int ka = ks + t * 16;
            if (lane < 16)
                for (int i = 0; i < 16; i++)
                    A[i] = (ka + i < K) ? in[ka + i] : (__fp16)0.0f;
            for (int i = 0; i < 16; i++) A[i] = __shfl(A[i], 0);
            half16_t B = dequant_q4k_tile(blk, t, d, dm, sc);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, B, acc);
        }
    }

    if (lane == 0)
        for (int j = 0; j < 8 && row + j < M; j++)
            out[row + j] = (__fp16)acc[j];
    if (lane == 16)
        for (int j = 0; j < 8 && row + 8 + j < M; j++)
            out[row + 8 + j] = (__fp16)acc[j];
}
