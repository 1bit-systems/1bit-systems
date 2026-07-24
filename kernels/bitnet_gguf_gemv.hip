// bitnet_gguf_gemv.hip — GEMV kernels for llama.cpp native ternary block formats
//
// Supports:
//   GGML_TYPE_TQ1_0 (34): 1.6875 bpw — 256-element blocks, 54 bytes
//   GGML_TYPE_TQ2_0 (35): 2.0625 bpw — 256-element blocks, 66 bytes
//
// Block structure (from llama.cpp ggml-common.h):
//
// TQ2_0 (simpler):
//   struct { uint8_t qs[256/4]; ggml_half d; }
//   qs: 64 bytes of 2-bit codes (4 per byte, LSB-first)
//   d:  fp16 scale (shared per 256-element block)
//   codes: 0=-d, 1=0, 2=+d, 3=0
//
// TQ1_0 (base-3, more complex):
//   struct { uint8_t qs[(256-4*4)/5]; uint8_t qh[256/64]; ggml_half d; }
//   qs: 48 bytes of base-3 packed codes (5 per byte) — first 240 elements
//   qh: 4 bytes, each stores 4 2-bit codes — elements 240..255 (16 elements)
//   d:  fp16 scale (shared per 256-element block)

#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define BT_BLOCK_K 256

// ─── TQ2_0: 2-bit symmetric ternary, 66 bytes/block ──────────────
#define BT_TQ2_0_BYTES 66
#define BT_TQ2_0_QS_BYTES 64   // 256 * 2 / 8

__global__ void bitnet_gemv_tq2_0_kernel(
    const uint8_t* __restrict__ packed,   // [M][n_blocks * BT_TQ2_0_BYTES]
    const int8_t*  __restrict__ x_i8,
    float          x_scale,
    const __half*  __restrict__ row_scales,
    __half*        __restrict__ y_f16,
    int            M,
    int            K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    int n_blocks = (K + BT_BLOCK_K - 1) / BT_BLOCK_K;
    int row_stride = n_blocks * BT_TQ2_0_BYTES;

    float sum = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        int block_start = b * BT_BLOCK_K;
        int block_end = min(block_start + BT_BLOCK_K, K);

        const uint8_t* block = packed + (size_t)row * row_stride + b * BT_TQ2_0_BYTES;

        // Read shared scale (fp16 at end of block)
        __half scale_h;
        memcpy(&scale_h, block + BT_TQ2_0_QS_BYTES, 2);
        float scale = (float)scale_h;

        // Read 64 bytes of 2-bit codes
        // Each byte: 4 codes (LSB-first)
        for (int i = 0; i < block_end - block_start; i++) {
            int col = block_start + i;
            int byte_i = i >> 2;    // i / 4
            int bit_off = (i & 3) << 1;  // (i % 4) * 2
            uint8_t code = (block[byte_i] >> bit_off) & 3;

            float w;
            if (code == 0)      w = -scale;
            else if (code == 2) w =  scale;
            else                w = 0.0f;  // 1 or 3 → 0

            sum += w * (float)x_i8[col];
        }
    }

    sum *= x_scale;
    if (row_scales) sum *= (float)row_scales[row];
    y_f16[row] = (__half)sum;
}

// ─── TQ1_0: base-3 ternary, 54 bytes/block ───────────────────────
#define BT_TQ1_0_QS_BYTES 48
#define BT_TQ1_0_QH_BYTES 4
#define BT_TQ1_0_BYTES 54   // 48 + 4 + 2
// base-3 decode LUT: byte → 5 ternary values
// Precomputed: for each byte 0..242, decode 5 codes
// (We only need bytes < 243, but handle the full range)
__constant__ int8_t bt_tq1_lut[243][5];

static const int bt_pow3[5] = {1, 3, 9, 27, 81};

// Host-side init for the LUT
extern "C" void bt_tq1_init_lut() {
    int8_t lut[243][5];
    for (int b = 0; b < 243; b++) {
        int tmp = b;
        for (int i = 0; i < 5; i++) {
            int code = tmp % 3;
            tmp /= 3;
            // code: 0=-1, 1=0, 2=+1
            lut[b][i] = (code == 0) ? -1 : (code == 2) ? 1 : 0;
        }
    }
    hipMemcpyToSymbol(HIP_SYMBOL(bt_tq1_lut), lut, sizeof(lut));
}

__global__ void bitnet_gemv_tq1_0_kernel(
    const uint8_t* __restrict__ packed,
    const int8_t*  __restrict__ x_i8,
    float          x_scale,
    const __half*  __restrict__ row_scales,
    __half*        __restrict__ y_f16,
    int            M,
    int            K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    int n_blocks = (K + BT_BLOCK_K - 1) / BT_BLOCK_K;
    int row_stride = n_blocks * BT_TQ1_0_BYTES;

    float sum = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        int block_start = b * BT_BLOCK_K;
        int block_end = min(block_start + BT_BLOCK_K, K);

        const uint8_t* block = packed + (size_t)row * row_stride + b * BT_TQ1_0_BYTES;

        // Read shared scale (fp16 at end of block)
        __half scale_h;
        memcpy(&scale_h, block + BT_TQ1_0_QS_BYTES + BT_TQ1_0_QH_BYTES, 2);
        float scale = (float)scale_h;

        // Part 1: qs[48] — 5 base-3 values per byte, first 240 elements
        for (int byte_i = 0; byte_i < 48 && block_start + byte_i * 5 < block_end; byte_i++) {
            uint8_t packed_byte = block[byte_i];
            int base = byte_i * 5;
            if (packed_byte < 243) {
                // Use LUT for fast decode
                for (int j = 0; j < 5; j++) {
                    int col = block_start + base + j;
                    if (col >= K) break;
                    float w = (float)bt_tq1_lut[packed_byte][j] * scale;
                    sum += w * (float)x_i8[col];
                }
            } else {
                // Slow path for bytes ≥ 243 (shouldn't happen in valid data)
                int tmp = packed_byte;
                for (int j = 0; j < 5; j++) {
                    int col = block_start + base + j;
                    if (col >= K) break;
                    uint8_t code = tmp % 3;
                    tmp /= 3;
                    float w = (code == 0) ? -scale : (code == 2) ? scale : 0.0f;
                    sum += w * (float)x_i8[col];
                }
            }
        }

        // Part 2: qh[4] — remaining 16 elements (240..255), 2-bit codes, 4 per byte
        const uint8_t* qh = block + BT_TQ1_0_QS_BYTES;
        for (int i = 0; i < BT_BLOCK_K - 240; i++) {
            int col = block_start + 240 + i;
            if (col >= K) break;

            int byte_i = i >> 2;      // i / 4
            int bit_off = (i & 3) << 1;  // (i % 4) * 2
            uint8_t code = (qh[byte_i] >> bit_off) & 3;

            float w;
            if (code == 0)      w = -scale;
            else if (code == 2) w =  scale;
            else                w = 0.0f;

            sum += w * (float)x_i8[col];
        }
    }

    sum *= x_scale;
    if (row_scales) sum *= (float)row_scales[row];
    y_f16[row] = (__half)sum;
}

// ─── Launch wrappers ──────────────────────────────────────────────

extern "C" void bitnet_gemv_tq2_0_f16_launch(
    const void* packed, const void* x_i8, float x_scale,
    const void* row_scales, void* y_f16, int M, int K, void* stream_void)
{
    if (M <= 0 || K <= 0) return;
    hipStream_t stream = stream_void ? *(hipStream_t*)stream_void : hipStreamPerThread;
    dim3 block(64), grid((M + 63) / 64);
    hipLaunchKernelGGL(bitnet_gemv_tq2_0_kernel,
        grid, block, 0, stream,
        (const uint8_t*)packed, (const int8_t*)x_i8, x_scale,
        (const __half*)row_scales, (__half*)y_f16, M, K);
}

extern "C" void bitnet_gemv_tq1_0_f16_launch(
    const void* packed, const void* x_i8, float x_scale,
    const void* row_scales, void* y_f16, int M, int K, void* stream_void)
{
    if (M <= 0 || K <= 0) return;
    hipStream_t stream = stream_void ? *(hipStream_t*)stream_void : hipStreamPerThread;
    dim3 block(64), grid((M + 63) / 64);
    hipLaunchKernelGGL(bitnet_gemv_tq1_0_kernel,
        grid, block, 0, stream,
        (const uint8_t*)packed, (const int8_t*)x_i8, x_scale,
        (const __half*)row_scales, (__half*)y_f16, M, K);
}
