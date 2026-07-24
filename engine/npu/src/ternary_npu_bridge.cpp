// ternary_npu_bridge.cpp — TQ2/TQ1/BST ternary → INT8 NPU adapter
//
// Packs ternary weights into INT8 format for the existing NPU xclbin GEMM kernels.
// This enables immediate NPU inference for ternary models: the same xclbins that
// run INT8 models also run ternary models — we just pre-quantize ternary weights
// to INT8 before uploading to the NPU weight BOs.
//
// Memory cost: 8x larger than native TQ2 (INT8 vs 2-bit). For true NPU ternary
// (2-bit xclbins), see docs/npu-ternary-roadmap.md.

#include "ternary_npu_bridge.h"
#include "block_scaled_ternary.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

// ─── helpers ─────────────────────────────────────────────────────
static inline float bf16_to_f32(uint16_t bf) {
    uint32_t b = (uint32_t)bf << 16;
    float f; memcpy(&f, &b, 4); return f;
}

static void quantize_to_int8(const float* f32, int8_t* i8, int n, float& out_scale) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(f32[i]);
        if (a > amax) amax = a;
    }
    if (amax < 1e-12f) amax = 1.0f;
    out_scale = amax / 127.0f;
    float inv = 127.0f / amax;
    for (int i = 0; i < n; i++) {
        int q = (int)roundf(f32[i] * inv);
        if (q > 127) q = 127;
        else if (q < -127) q = -127;
        i8[i] = (int8_t)q;
    }
}

// ─── TQ2 → INT8 ──────────────────────────────────────────────────
//
// TQ2 tile: 32×256, per-tile-row: [grps×2 bf16 scales][tc/4 bytes 2-bit codes]
//   codes: 0=-scale, 1=0, 2=+scale (LSB-first, 4 codes/byte)

TernaryNpuPackResult pack_tq2_to_npu_int8(
    const uint8_t* tq2_data, int rows, int cols,
    int tile_rows, int tile_cols, int group_size)
{
    int ntr = (rows + tile_rows - 1) / tile_rows;
    int ntc = (cols + tile_cols - 1) / tile_cols;
    int grps = tile_cols / group_size;
    int tile_scales_bytes = tile_rows * grps * 2;  // bf16
    int tile_codes_bytes  = tile_rows * tile_cols / 4;
    int tile_bytes = tile_scales_bytes + tile_codes_bytes;

    std::vector<float> f32((size_t)rows * cols, 0.0f);

    for (int tr = 0; tr < ntr; tr++) {
        for (int tc = 0; tc < ntc; tc++) {
            const uint8_t* tile = tq2_data + (size_t)(tr * ntc + tc) * tile_bytes;
            const uint16_t* scales = (const uint16_t*)tile;
            const uint8_t* codes   = tile + tile_scales_bytes;

            for (int rr = 0; rr < tile_rows && (tr * tile_rows + rr) < rows; rr++) {
                int ar = tr * tile_rows + rr;
                for (int g = 0; g < grps; g++) {
                    float s = bf16_to_f32(scales[rr * grps + g]);
                    for (int i = 0; i < group_size; i += 4) {
                        int byte_idx = rr * (tile_cols / 4) + (g * group_size + i) / 4;
                        uint8_t b = codes[byte_idx];
                        for (int j = 0; j < 4; j++) {
                            int ac = tc * tile_cols + g * group_size + i + j;
                            if (ac >= cols) break;
                            uint8_t code = (b >> (j * 2)) & 3;
                            float v;
                            if (code == 0)      v = -s;
                            else if (code == 2) v =  s;
                            else                v = 0.0f;  // 1 or 3 → 0
                            f32[(size_t)ar * cols + ac] = v;
                        }
                    }
                }
            }
        }
    }

    float scale;
    int8_t* out = (int8_t*)malloc((size_t)rows * cols);
    quantize_to_int8(f32.data(), out, rows * cols, scale);

    printf("[ternary_npu_bridge] TQ2 → INT8: %dx%d amax=%g scale=%g\n",
           rows, cols,
           *std::max_element(f32.begin(), f32.end(),
               [](float a, float b){ return fabsf(a) < fabsf(b); }),
           scale);

    return {out, scale, rows, cols};
}

// ─── TQ1 (1.58-bit base-3) → INT8 ────────────────────────────────
//
// TQ1 packs 5 ternary values per byte using base-3 encoding:
//   packed = code0 + code1*3 + code2*9 + code3*27 + code4*81
//   where code: 0=-1, 1=0, 2=+1
// Per 5-element group: 1 bf16 scale + 1 byte codes

TernaryNpuPackResult pack_tq1_to_npu_int8(
    const uint8_t* tq1_data, int rows, int cols,
    int tile_rows, int tile_cols)
{
    int grps = (tile_cols + 4) / 5;  // groups of 5 per tile row
    int tile_scales_bytes = tile_rows * grps * 2;
    int tile_codes_bytes  = tile_rows * grps;
    int tile_bytes = tile_scales_bytes + tile_codes_bytes;

    int ntr = (rows + tile_rows - 1) / tile_rows;
    int ntc = (cols + tile_cols - 1) / tile_cols;

    std::vector<float> f32((size_t)rows * cols, 0.0f);

    for (int tr = 0; tr < ntr; tr++) {
        for (int tc = 0; tc < ntc; tc++) {
            const uint8_t* tile = tq1_data + (size_t)(tr * ntc + tc) * tile_bytes;
            const uint16_t* scales = (const uint16_t*)tile;
            const uint8_t* codes   = tile + tile_scales_bytes;

            for (int rr = 0; rr < tile_rows && (tr * tile_rows + rr) < rows; rr++) {
                int ar = tr * tile_rows + rr;
                for (int g = 0; g < grps; g++) {
                    float s = bf16_to_f32(scales[rr * grps + g]);
                    uint8_t packed = codes[rr * grps + g];
                    uint8_t tmp = packed;
                    for (int i = 0; i < 5; i++) {
                        int ac = tc * tile_cols + g * 5 + i;
                        if (ac >= cols) break;
                        uint8_t code = tmp % 3;
                        tmp /= 3;
                        float v;
                        if (code == 0)      v = -s;   // -1
                        else if (code == 2) v =  s;   // +1
                        else                v = 0.0f; // code 1 = 0
                        f32[(size_t)ar * cols + ac] = v;
                    }
                }
            }
        }
    }

    float scale;
    int8_t* out = (int8_t*)malloc((size_t)rows * cols);
    quantize_to_int8(f32.data(), out, rows * cols, scale);

    printf("[ternary_npu_bridge] TQ1 (1.58-bit) → INT8: %dx%d scale=%g\n",
           rows, cols, scale);

    return {out, scale, rows, cols};
}

// ─── BST (block-scaled ternary) → INT8 ────────────────────────────
//
// BST: 5 bytes per 16-element block
//   [4 bytes: 2-bit packed codes (LSB-first)][1 byte: FP8 E4M3 scale]
//   codes: 0=-scale, 1=+scale, 2=-scale... (see block_scaled_ternary.h)
//
// Wait — BST uses a DIFFERENT code mapping than TQ2!
// From block_scaled_ternary.h ternary_unpack_16():
//   bits: 0→0, 1→+1, 2→-1, 3→0
// NOT the same as TQ2 where 0=-scale, 1=0, 2=+scale.

TernaryNpuPackResult pack_bst_to_npu_int8(
    const uint8_t* bst_data, int rows, int cols)
{
    const int BK = BST_BLOCK_K;    // 16
    const int BB = BST_BLOCK_BYTES; // 5
    int n_blocks = (cols + BK - 1) / BK;

    std::vector<float> f32((size_t)rows * cols, 0.0f);

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < n_blocks; b++) {
            const uint8_t* block = bst_data + (size_t)(r * n_blocks + b) * BB;

            // FP8 E4M3 scale (see block_scaled_ternary.h fp8e4m3_to_fp32)
            uint8_t fp8 = block[4];
            float scale;
            if (fp8 == 0xFF) {
                uint32_t nan = 0x7FC00000u;
                memcpy(&scale, &nan, 4);
            } else {
                uint32_t s = (fp8 >> 7) & 1, e = (fp8 >> 3) & 0xF, m = fp8 & 0x7;
                uint32_t bits;
                if (e == 0) {
                    static const uint32_t FP8_SUBNORM[8] = {
                        0, 0x3B000000u, 0x3B800000u, 0x3BC00000u,
                        0x3C000000u, 0x3C200000u, 0x3C400000u, 0x3C600000u
                    };
                    bits = FP8_SUBNORM[m] | (s << 31);
                } else {
                    bits = (s << 31) | ((e + 120) << 23) | (m << 20);
                }
                memcpy(&scale, &bits, 4);
            }

            uint32_t packed;
            memcpy(&packed, block, 4);

            for (int i = 0; i < BK; i++) {
                int ac = b * BK + i;
                if (ac >= cols) break;
                uint32_t bits = (packed >> (i * 2)) & 0x3;
                float v;
                if (bits == 1)      v =  scale;   // +1
                else if (bits == 2) v = -scale;   // -1
                else                v = 0.0f;     // 0 or 3
                f32[(size_t)r * cols + ac] = v;
            }
        }
    }

    float scale;
    int8_t* out = (int8_t*)malloc((size_t)rows * cols);
    quantize_to_int8(f32.data(), out, rows * cols, scale);

    printf("[ternary_npu_bridge] BST → INT8: %dx%d scale=%g\n",
           rows, cols, scale);

    return {out, scale, rows, cols};
}

// ─── Free ────────────────────────────────────────────────────────

void free_ternary_npu_pack(TernaryNpuPackResult* result) {
    if (result && result->weights) {
        free(result->weights);
        result->weights = nullptr;
    }
}
