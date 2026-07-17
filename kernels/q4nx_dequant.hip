// q4nx_dequant.hip — GPU-accelerated Q4NX I4_CUSTOM dequantization
// Dequantizes packed INT4 groups to FP16 on GPU
//
// I4_CUSTOM format: groups of 8 values
//   Each group = [scale_bf16(2B)][zp_bf16(2B)][4 bytes I4 data]
//   I4 data: 2 nibbles per byte, little-endian: byte = (hi_nibble << 4) | lo_nibble
//
// Grid: (total_groups / 256, 1, 1)  ×  Block: 256 threads
// Each thread handles one group = 8 values

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ── BF16 → float32 ──
__device__ inline float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f; memcpy(&f, &bits, sizeof(f));
    return f;
}

// ── Dequantize I4_CUSTOM on GPU ──
// in:  packed I4 data, groups * 8 bytes
// out: fp16 output, groups * 8 * sizeof(__half) bytes
// groups_per_row: number of groups per row (cols/8)
// out_stride: stride between rows in output (in elements, not bytes)
__global__ void dequant_i4_custom_kernel(
    const uint8_t* __restrict__ in,
    __half* __restrict__ out,
    int total_groups,
    int groups_per_row,
    int out_stride)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total_groups) return;

    const uint8_t* grp = in + (size_t)g * 8;

    // Read scale and zero-point as BF16
    uint16_t s_raw, z_raw;
    memcpy(&s_raw, grp, 2);
    memcpy(&z_raw, grp + 2, 2);
    float scale = bf16_to_f32(s_raw);
    float zp = bf16_to_f32(z_raw);
    if (!isfinite(scale) || !isfinite(zp)) { scale = 1.0f; zp = 0.0f; }

    // Compute row/col within the logical matrix
    int row = g / groups_per_row;
    int col_in_group = (g % groups_per_row) * 8;

    // Dequantize 8 values
    uint8_t byte0 = grp[4], byte1 = grp[5], byte2 = grp[6], byte3 = grp[7];

    int nibbles[8] = {
        byte0 & 0x0F, (byte0 >> 4) & 0x0F,
        byte1 & 0x0F, (byte1 >> 4) & 0x0F,
        byte2 & 0x0F, (byte2 >> 4) & 0x0F,
        byte3 & 0x0F, (byte3 >> 4) & 0x0F
    };

    for (int j = 0; j < 8; j++) {
        int v = nibbles[j];
        if (v >= 8) v -= 16;  // sign-extend
        float fval = (float)v * scale + zp;
        int col = col_in_group + j;
        out[(size_t)row * out_stride + col] = __float2half(fval);
    }
}

// ── Dequantize FLM Q4NX I8 tile on GPU ──
// FLM Q4NX format: 5120-byte tiles, 32×256, groups of 32
// Per tile: 256 BF16 scales (512B) + 256 BF16 zero-points (512B) + 4096B I4 data
__global__ void dequant_i8_tile_kernel(
    const uint8_t* __restrict__ tiles,
    __half* __restrict__ out,
    int n_tiles,
    int n_tile_cols,
    int out_rows,
    int out_cols)
{
    int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_idx >= n_tiles) return;
    if (tile_idx >= n_tiles) return;

    const uint8_t* tile = tiles + (size_t)tile_idx * 5120;
    int tile_row = tile_idx / n_tile_cols;
    int tile_col = tile_idx % n_tile_cols;

    const uint16_t* scales = (const uint16_t*)tile;
    const uint16_t* zps    = (const uint16_t*)(tile + 512);
    const uint8_t* packed  = tile + 1024;

    for (int lr = 0; lr < 32; lr++) {
        int global_row = tile_row * 32 + lr;
        if (global_row >= out_rows) continue;

        int lane = lr / 16;
        int lane_row = lr % 16;
        int byte_idx = lane_row / 2;
        int ns = lr % 2;

        for (int col = 0; col < 256; col++) {
            int group = col / 32;
            float scale = bf16_to_f32(scales[group * 32 + lr]);
            float zp = bf16_to_f32(zps[group * 32 + lr]);

            uint8_t byte_val = packed[lane * 2048 + col * 8 + byte_idx];
            int nibble = (ns == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
            if (nibble >= 8) nibble -= 16;

            int global_col = tile_col * 256 + col;
            if (global_col >= out_cols) continue;
            out[(size_t)global_row * out_cols + global_col] = __float2half((float)nibble * scale + zp);
        }
    }
}
