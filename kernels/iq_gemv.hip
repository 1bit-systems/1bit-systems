// iq_gemv.hip — IQ1_S/IQ1_M importance-quantized GEMV kernel
//
// Full port of llama.cpp ggml-quants.c dequantize_row_iq1_s to GPU.
// Block structure (from llama.cpp ggml-common.h):
//
//   IQ1_S (1.5625 bpw): 256 elements, 206 bytes (50 actual + padding)
//     struct { ggml_half d; uint8_t qs[32]; uint16_t qh[8]; }  // 2+32+16=50
//     d:    superblock fp16 scale
//     qs:   grid index low 8 bits (4 per sub-block, 8 sub-blocks)
//     qh:   per 32-element sub-block:
//             bits 0-11: high 3 bits for 4 LUT indices
//             bits 12-14: sub-block scale multiplier (0-7)
//             bit 15: delta sign
//
//   IQ1_M (1.75 bpw): 256 elements, 230 bytes
//     struct { uint8_t qs[32]; uint8_t qh[32]; uint8_t scales[8]; } // 72 bytes main
//     + d (fp16 scale) packed into scales via iq1m_scale_t
//     More complex multi-level scaling
//
// Dequant: for each 32-element sub-block (8 per block):
//   dl = d * (2 * ((qh >> 12) & 7) + 1)        // scale
//   delta = (qh & 0x8000) ? -0.125f : 0.125f    // grid shift
//   for 4 indices:
//     idx = qs[l] | ((qh >> (3*l)) & 7) << 8     // 11-bit LUT index
//     grid = iq1s_grid[idx]                       // 8 int8 values
//     output = dl * (grid[j] + delta) for j in 0..7

#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip_check.h"

#define IQ_BLOCK_SIZE 256
#define IQ1_S_BYTES 206
#define IQ1_S_CORE_BYTES 50   // actual block struct size: d(2) + qs(32) + qh(16)
#define IQ1_M_BYTES 230
#define IQ1_S_SUBBLOCKS 8     // 256/32

#include "iq1s_grid_data.h"

__constant__ int8_t iq1s_grid_dev[IQ1S_GRID_SIZE][8];
__constant__ float iq1s_delta_dev = 0.125f;

extern "C" void iq_upload_grid(hipStream_t stream) {
    HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(iq1s_grid_dev), iq1s_grid_host_unpacked,
                           sizeof(iq1s_grid_host_unpacked), 0,
                           hipMemcpyHostToDevice, stream));
}

// ─── IQ1_S: dequantize one 256-element block to float ───────────
// Ported from llama.cpp ggml-quants.c dequantize_row_iq1_s
// Block data: first IQ1_S_CORE_BYTES (50) of the IQ1_S_BYTES (206) block
__device__ void dequant_iq1_s_block(const uint8_t* block, float* out) {
    // Read superblock scale d (fp16 at offset 0)
    __half d_h;
    memcpy(&d_h, block, 2);
    float d = (float)d_h;

    // qs at offset 2 (32 bytes)
    const uint8_t* qs = block + 2;
    // qh at offset 34 (8 × uint16_t = 16 bytes)
    const uint16_t* qh = (const uint16_t*)(block + 34);

    for (int ib = 0; ib < IQ1_S_SUBBLOCKS; ib++) {
        // Sub-block scale: d * (2*scale_mult + 1)
        float dl = d * (float)(2 * ((qh[ib] >> 12) & 7) + 1);
        // Grid shift (delta)
        float delta = (qh[ib] & 0x8000) ? -iq1s_delta_dev : iq1s_delta_dev;

        // 4 LUT lookups, each producing 8 values = 32 values
        for (int l = 0; l < 4; l++) {
            // LUT index: qs[l] (low 8 bits) | ((qh >> 3*l) & 7) << 8 (high 3 bits)
            uint16_t idx = (uint16_t)(qs[l] | (((qh[ib] >> (3 * l)) & 7) << 8));
            if (idx >= IQ1S_GRID_SIZE) idx = 0;

            for (int j = 0; j < 8; j++) {
                out[ib * 32 + l * 8 + j] = dl * ((float)iq1s_grid_dev[idx][j] + delta);
            }
        }
        qs += 4;
    }
}

// ─── IQ1_M dequant: 256-element block → float32 ─────────────────
// Ported from llama.cpp ggml-quants.c dequantize_row_iq1_m
// Block layout (block_iq1_m): qs[32] + qh[16] + scales[8] = 56 bytes core
// within the 230-byte GGUF block. scales stores 4 uint16_t with 6-bit
// sub-block scales + 4-bit superblock scale packed in upper bits.
// Decode IQ1_M superblock scale from 4 uint16_t scale entries
// scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)
// where sc = (const uint16_t*)scales (8 bytes = 4 uint16_t)
static inline __device__ uint16_t iq1m_combine_scales(const uint16_t* sc) {
    return (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                      ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
}

__device__ void dequant_iq1_m_block(const uint8_t* block, float* out) {
    // IQ1_M layout within 230-byte GGUF block:
    //   qs[32] at offset 0
    //   qh[16] at offset 32
    //   scales[8] at offset 48
    //   remaining 174 bytes: padding/metadata (ignored)
    const uint8_t* qs = block;
    const uint8_t* qh = block + 32;
    const uint8_t* scales = block + 48;

    // Decode superblock scale from packed u16 (read scales as uint16_t[4])
    uint16_t scale_u16 = iq1m_combine_scales((const uint16_t*)scales);
    // Convert fp16 bits to float — iq1m_scale_t stores as ggml_half
    // The packing: scale_u16 is the raw fp16 bit pattern
    __half d_h;
    memcpy(&d_h, &scale_u16, 2);
    float d = (float)d_h;

    // Scales are also used as uint16_t[4] for sub-block scale reads
    const uint16_t* sc = (const uint16_t*)scales;
    // Zero out the low 4 bits of each uint16_t (they're used for superblock
    // scale packing, not sub-block scales)
    // Wait — the sub-block scales are in bits 0-5 and 6-11 of each sc[i].
    // The superblock scale uses bits 12-15. So reading sc[ib/2] and shifting
    // gives the correct sub-block scale without masking the superblock bits.

    int out_idx = 0;
    for (int ib = 0; ib < IQ1_S_SUBBLOCKS; ib++) {
        // Two scale values per sub-block
        int sc_idx = ib / 2;
        int sc_shift = (ib % 2) * 6;
        float dl1 = d * (float)(2 * ((sc[sc_idx] >> (sc_shift + 0)) & 7) + 1);
        float dl2 = d * (float)(2 * ((sc[sc_idx] >> (sc_shift + 3)) & 7) + 1);

        // 4 LUT indices
        uint16_t idx[4];
        idx[0] = (uint16_t)(qs[0] | ((qh[0] << 8) & 0x700));
        idx[1] = (uint16_t)(qs[1] | ((qh[0] << 4) & 0x700));
        idx[2] = (uint16_t)(qs[2] | ((qh[1] << 8) & 0x700));
        idx[3] = (uint16_t)(qs[3] | ((qh[1] << 4) & 0x700));

        // Delta signs (bit 3 and bit 7 of qh[0] and qh[1])
        float delta[4];
        delta[0] = (qh[0] & 0x08) ? -iq1s_delta_dev : iq1s_delta_dev;
        delta[1] = (qh[0] & 0x80) ? -iq1s_delta_dev : iq1s_delta_dev;
        delta[2] = (qh[1] & 0x08) ? -iq1s_delta_dev : iq1s_delta_dev;
        delta[3] = (qh[1] & 0x80) ? -iq1s_delta_dev : iq1s_delta_dev;

        // First 2 LUT lookups use dl1
        for (int l = 0; l < 2; l++) {
            uint16_t lut_idx = (idx[l] < IQ1S_GRID_SIZE) ? idx[l] : 0;
            for (int j = 0; j < 8; j++) {
                out[out_idx++] = dl1 * ((float)iq1s_grid_dev[lut_idx][j] + delta[l]);
            }
        }
        // Last 2 LUT lookups use dl2
        for (int l = 2; l < 4; l++) {
            uint16_t lut_idx = (idx[l] < IQ1S_GRID_SIZE) ? idx[l] : 0;
            for (int j = 0; j < 8; j++) {
                out[out_idx++] = dl2 * ((float)iq1s_grid_dev[lut_idx][j] + delta[l]);
            }
        }

        qs += 4;
        qh += 2;
    }
}

// ─── IQ dequant+GEMV kernel ─────────────────────────────────────
// Each thread processes one row (output neuron).
// K=256*n_blocks per row, dequantized on the fly, accumulated with INT8 activations.
__global__ void iq_dequant_gemv_kernel(
    const uint8_t* __restrict__ packed,
    const int8_t*  __restrict__ x_i8,
    float          x_scale,
    float*         __restrict__ y_f32,
    int            M,
    int            K,
    int            dtype)    // 0=IQ1_S, 1=IQ1_M
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    int block_bytes = (dtype == 0) ? IQ1_S_BYTES : IQ1_M_BYTES;
    int n_blocks = (K + IQ_BLOCK_SIZE - 1) / IQ_BLOCK_SIZE;
    int row_stride = n_blocks * block_bytes;

    float sum = 0.0f;
    float vals[IQ_BLOCK_SIZE];  // 256 floats per block

    for (int b = 0; b < n_blocks; b++) {
        int block_start = b * IQ_BLOCK_SIZE;
        const uint8_t* block = packed + (size_t)row * row_stride + b * block_bytes;

        // Dequantize 256 elements
        if (dtype == 0) {
            dequant_iq1_s_block(block, vals);
        } else {
            dequant_iq1_m_block(block, vals);
        }

        int cols_here = min(IQ_BLOCK_SIZE, K - block_start);
        for (int i = 0; i < cols_here; i++) {
            sum += vals[i] * (float)x_i8[block_start + i];
        }
    }

    sum *= x_scale;
    y_f32[row] = sum;
}

// ─── Launch wrapper ──────────────────────────────────────────────
extern "C" void iq_dequant_gemv_launch(
    const void* packed, const void* x_i8, float x_scale,
    void* y, int M, int K, int dtype, void* stream_void)
{
    if (M <= 0 || K <= 0) return;
    hipStream_t stream = stream_void ? *(hipStream_t*)stream_void : hipStreamPerThread;
    dim3 block(256), grid((M + 255) / 256);
    hipLaunchKernelGGL(iq_dequant_gemv_kernel,
        grid, block, 0, stream,
        (const uint8_t*)packed, (const int8_t*)x_i8, x_scale,
        (float*)y, M, K, dtype);
}

extern "C" void iq_init(hipStream_t stream) { iq_upload_grid(stream); }
