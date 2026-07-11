/**
 * Q4NX INT4 dequantization — torch2aie chunk format.
 *
 * Each I8 row (5120 bytes) = ONE tile of [32 BF16 rows × 256 BF16 cols].
 * Tiles are arranged row-major in a grid covering the full weight matrix.
 *
 * Per I8 row (5120 bytes):
 *   [0..511]:   256 BF16 scales. For group g=0..7, row r=0..31: scales[g*32+r]
 *   [512..1023]: 256 BF16 zero_points. Same layout.
 *   [1024..5119]: 4096 bytes packed INT4:
 *     Lane 0 (rows 0-15): bytes 1024-3071
 *     Lane 1 (rows 16-31): bytes 3072-5119
 *     Within lane: for col 0..255, byte_idx 0..7: lane_base + col*8 + byte_idx
 *     nibbles: lo = row(byte_idx*2), hi = row(byte_idx*2+1)
 *
 * Tile grid: I8 rows row-major.
 *   n_tile_cols = in_features / 256 (usually 4 for hidden=1024)
 *   tile_row = I8_row / n_tile_cols
 *   tile_col = I8_row % n_tile_cols
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TILE_ROWS 32
#define TILE_COLS 256

static inline float bf16_to_float(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f; memcpy(&f, &bits, sizeof(f)); return f;
}

/**
 * Dequantize an I8 tensor (torch2aie Q4NX chunk format) to float.
 * Output: [out_rows, out_cols] row-major float array (caller must free).
 * out_rows = n_tile_rows * 32, out_cols = n_tile_cols * 256
 */
// Forward declaration for the wrapper
float* dequant_i8_to_float_ex(const uint8_t* data, int i8_rows, int in_features,
                              int* out_rows, int* out_cols);

float* dequant_i8_to_float(const uint8_t* data, int i8_rows,
                           int* out_rows, int* out_cols) {
    return dequant_i8_to_float_ex(data, i8_rows, 1024, out_rows, out_cols);
}

/**
 * Extended version with explicit in_features (hidden_dim).
 * For Q4NX format: n_tile_cols = in_features / TILE_COLS.
 */
float* dequant_i8_to_float_ex(const uint8_t* data, int i8_rows, int in_features,
                              int* out_rows, int* out_cols) {
    // Determine tile grid: rows first, columns second
    int n_tile_cols, n_tile_rows;

    n_tile_cols = in_features / TILE_COLS;
    n_tile_rows = i8_rows / n_tile_cols;

    *out_rows = n_tile_rows * TILE_ROWS;
    *out_cols = n_tile_cols * TILE_COLS;

    float* out = (float*)calloc((*out_rows) * (*out_cols), sizeof(float));
    if (!out) return NULL;

    for (int ir = 0; ir < i8_rows; ir++) {
        const uint8_t* rd = data + ir * 5120;
        int tile_row = ir / n_tile_cols;
        int tile_col = ir % n_tile_cols;

        const uint16_t* scales = (const uint16_t*)(rd);
        const uint16_t* zeros  = (const uint16_t*)(rd + 512);
        const uint8_t* packed  = rd + 1024;

        for (int lr = 0; lr < TILE_ROWS; lr++) {
            int lane = lr / 16;
            int lane_row = lr % 16;
            int byte_idx = lane_row / 2;
            int nibble_sel = lr % 2;

            const uint8_t* lane_data = packed + lane * (TILE_COLS * 8);

            for (int col = 0; col < TILE_COLS; col++) {
                int group = col / 32;
                float scale = bf16_to_float(scales[group * 32 + lr]);
                float zp = bf16_to_float(zeros[group * 32 + lr]);
                if (!isfinite(scale) || fabsf(scale) > 100.0f) scale = 0.0f;
                if (!isfinite(zp) || fabsf(zp) > 100.0f) zp = 0.0f;

                uint8_t byte_val = lane_data[col * 8 + byte_idx];
                int8_t val;
                if (nibble_sel == 0) val = (int8_t)(byte_val & 0x0F);
                else                 val = (int8_t)((byte_val >> 4) & 0x0F);
                if (val >= 8) val -= 16;

                out[(tile_row * TILE_ROWS + lr) * (*out_cols) +
                    (tile_col * TILE_COLS + col)] = (float)val * scale + zp;
            }
        }
    }
    return out;
}
