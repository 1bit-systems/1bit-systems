// Phase 5.1 — INT4 activations (halo-encoded ternary B).
//
// Takes packed INT4 A: [K/2] bytes, 2 signed 4-bit values per byte
// (little-endian within byte: low nibble = element 2i, high = 2i+1).
// Halves A memory traffic vs Phase 5 INT8 A. On the LM head (128256x2560)
// where A is streamed through LDS the same number of times per block but
// from half as many bytes, this gives meaningful speedup.
//
// Target shape: M*rows_per_block blocks, 128 threads each, 8 rows per block
// (matches the winning Phase 5 dot4 topology).
//
// B encoding matches ternary_gemv_phase5_halo (0->-1, 1->0, 2->+1).
//
// INT4 encoding for A: signed 4-bit, two's complement. Values in [-8, +7].
// User pre-quantizes FP16 activations with a per-tensor scale.

#include <hip/hip_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define ROWS_PER_BLOCK 8
#define ROWS_PER_WARP  2
#define VALS_PER_U32 16
#define LDS_TILE_VALS 2048   // INT4 values per tile (= 1024 bytes of packed A)

static __device__ __forceinline__ uint32_t pack4(int8_t a, int8_t b, int8_t c, int8_t d) {
    return ((uint32_t)(uint8_t)a)
         | ((uint32_t)(uint8_t)b << 8)
         | ((uint32_t)(uint8_t)c << 16)
         | ((uint32_t)(uint8_t)d << 24);
}

// Halo encoding: sign = (bits == 2) - (bits == 0)
static __device__ __forceinline__ uint32_t unpack_ternary_halo_4(uint32_t word, int base_v) {
    const uint32_t b0 = (word >> ((base_v + 0) * 2)) & 0x3;
    const uint32_t b1 = (word >> ((base_v + 1) * 2)) & 0x3;
    const uint32_t b2 = (word >> ((base_v + 2) * 2)) & 0x3;
    const uint32_t b3 = (word >> ((base_v + 3) * 2)) & 0x3;
    const int8_t s0 = (int8_t)((int)(b0 == 2) - (int)(b0 == 0));
    const int8_t s1 = (int8_t)((int)(b1 == 2) - (int)(b1 == 0));
    const int8_t s2 = (int8_t)((int)(b2 == 2) - (int)(b2 == 0));
    const int8_t s3 = (int8_t)((int)(b3 == 2) - (int)(b3 == 0));
    return pack4(s0, s1, s2, s3);
}

// Sign-extend 4-bit to 8-bit: n[3:0] -> signed 8-bit value in [-8, +7].
static __device__ __forceinline__ int8_t s4_to_s8(uint32_t nib) {
    // nib[3] = sign bit; when set, fill upper 4 bits with 1s.
    const int8_t raw = (int8_t)(nib & 0xF);
    return (int8_t)((raw << 4) >> 4);   // arithmetic shift sign-extends
}

// Given 4 uint32s packing 16 INT4 A-values, unpack to 4 uint32s of packed INT8.
// Input layout: 1 uint32 holds 8 INT4 values (4 bytes * 2 nibbles).
static __device__ __forceinline__ void unpack_16_int4_to_4_int8x4(
    const uint8_t* src_bytes,    // 8 bytes = 16 INT4 values
    uint32_t out[4])             // 4 uint32s, 4 INT8 each = 16 values
{
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        // Each group of 4 INT4 lives in 2 bytes
        const uint8_t b0 = src_bytes[g * 2 + 0];
        const uint8_t b1 = src_bytes[g * 2 + 1];
        const int8_t v0 = s4_to_s8(b0 & 0xF);
        const int8_t v1 = s4_to_s8(b0 >> 4);
        const int8_t v2 = s4_to_s8(b1 & 0xF);
        const int8_t v3 = s4_to_s8(b1 >> 4);
        out[g] = pack4(v0, v1, v2, v3);
    }
}

__attribute__((amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))
__global__ void ternary_gemv_phase5_i4a_kernel(
    const uint32_t* __restrict__ packed,       // [M, K/16] weights (halo uint8 reinterpreted)
    const uint8_t*  __restrict__ x_i4_packed,  // [K/2] A bytes
    float                       x_scale,
    const float*    __restrict__ scales,
    float*          __restrict__ y,
    int M, int K)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int wave = tid >> 5;

    const int row_base = blockIdx.x * ROWS_PER_BLOCK;
    const int warp_row_base = row_base + wave * ROWS_PER_WARP;

    const int packed_k = K / VALS_PER_U32;

    // LDS holds decoded INT8 values (after 4-bit unpack) so compute path is
    // identical to Phase 5 dot4. 2048 int8 = 2 KiB.
    __shared__ int8_t x_shared[LDS_TILE_VALS];

    int32_t acc[ROWS_PER_WARP] = {};

    for (int k_base = 0; k_base < K; k_base += LDS_TILE_VALS) {
        __syncthreads();

        // 128 threads cooperatively decode 2048 INT4 values (= 1024 bytes)
        // into LDS INT8 buffer. Each thread handles 16 INT4 values = 8 bytes
        // from global.
        const int val_start  = k_base + tid * 16;
        const int byte_start = val_start / 2;
        if (val_start + 16 <= K) {
            // 8 contiguous bytes -> 16 INT4 -> 16 INT8 in LDS
            uint8_t src[8];
            const uint64_t* src64 = reinterpret_cast<const uint64_t*>(x_i4_packed + byte_start);
            const uint64_t blob = *src64;
            #pragma unroll
            for (int i = 0; i < 8; ++i) src[i] = (blob >> (i * 8)) & 0xFF;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint8_t b = src[i];
                x_shared[tid * 16 + i * 2 + 0] = s4_to_s8(b & 0xF);
                x_shared[tid * 16 + i * 2 + 1] = s4_to_s8(b >> 4);
            }
        } else {
            // Tail
            #pragma unroll
            for (int e = 0; e < 16; ++e) {
                int k = val_start + e;
                int8_t v = 0;
                if (k < K) {
                    const uint8_t byte = x_i4_packed[k / 2];
                    v = s4_to_s8((k & 1) ? (byte >> 4) : (byte & 0xF));
                }
                x_shared[tid * 16 + e] = v;
            }
        }
        __syncthreads();

        const int tile_u32_start = k_base / VALS_PER_U32;
        const int tile_u32_count = LDS_TILE_VALS / VALS_PER_U32;    // 128 u32s

        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const int my_row = warp_row_base + r;
            if (my_row >= M) continue;
            const uint32_t* row_weights = packed + (size_t)my_row * packed_k;

            for (int u = lane; u < tile_u32_count; u += WARP_SIZE) {
                int global_u32 = tile_u32_start + u;
                if (global_u32 >= packed_k) break;
                const uint32_t word = row_weights[global_u32];
                const int x_offset  = u * VALS_PER_U32;

                const uint32_t* xpack = reinterpret_cast<const uint32_t*>(&x_shared[x_offset]);
                const uint32_t xw0 = xpack[0];
                const uint32_t xw1 = xpack[1];
                const uint32_t xw2 = xpack[2];
                const uint32_t xw3 = xpack[3];

                const uint32_t tw0 = unpack_ternary_halo_4(word, 0);
                const uint32_t tw1 = unpack_ternary_halo_4(word, 4);
                const uint32_t tw2 = unpack_ternary_halo_4(word, 8);
                const uint32_t tw3 = unpack_ternary_halo_4(word, 12);

                int32_t local = 0;
                local = __builtin_amdgcn_sudot4(true, tw0, true, xw0, local, false);
                local = __builtin_amdgcn_sudot4(true, tw1, true, xw1, local, false);
                local = __builtin_amdgcn_sudot4(true, tw2, true, xw2, local, false);
                local = __builtin_amdgcn_sudot4(true, tw3, true, xw3, local, false);
                acc[r] += local;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int32_t v = acc[r];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v += __shfl_xor(v, offset);
        }
        const int my_row = warp_row_base + r;
        if (lane == 0 && my_row < M) {
            y[my_row] = (float)v * x_scale * scales[my_row];
        }
    }
}

extern "C" void ternary_gemv_phase5_i4a_launch(
    const void* packed, const void* x_i4_packed, float x_scale,
    const void* scales, void* y,
    int M, int K, void* stream)
{
    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    hipLaunchKernelGGL(ternary_gemv_phase5_i4a_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const uint32_t*>(packed),
                       static_cast<const uint8_t*>(x_i4_packed),
                       x_scale,
                       static_cast<const float*>(scales),
                       static_cast<float*>(y),
                       M, K);
}
