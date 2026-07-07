#pragma once
/**
 * i4_loader.h — Load raw I4 data from Q4NX format directly into XRT BOs.
 *
 * Skips the broken dequant_i8_to_float entirely. The Q4NX I8 format stores
 * weights as I4 nibbles in 5120-byte rows ([32 tile rows] x [256 cols]).
 * This function:
 *   1. Reads the 4096-byte packed section of each I8 row
 *   2. Expands nibbles to signed INT8 (-8..7)
 *   3. Arranges them in the BO in [in_features, out_features] order
 *
 * The caller must provide:
 *   - BO pointer for the weight buffer (size = K * N bytes)
 *   - Q4NX file data pointer and tensor offset
 *   - I8 tensor dimensions (i8_rows, in_features, i8_data_offset)
 *   - The output feature offset within the fused QKV BO
 *
 * Bscale should be set to max_i4_range / 127 = 7.0f/127.0f = 0.055118f
 * since I4 values span [-8, 7].
 */

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>

// Q4NX I8 row constants
static constexpr int I4_TILE_ROWS = 32;
static constexpr int I4_TILE_COLS = 256;
static constexpr int I4_ROW_BYTES = 5120;
static constexpr int I4_PACKED_BYTES = 4096;   // bytes per I8 row (I4 nibbles)
static constexpr int I4_HEADER_BYTES = 1024;    // reserved for future format info

/**
 * Extract a signed I4 value from the packed section of an I8 row.
 *
 * @param packed  Pointer to the 4096-byte packed section
 * @param lr      Tile-local row (0..31)
 * @param col     Tile-local column (0..255)
 * @return Signed I4 value (-8..7)
 */
static inline int8_t extract_i4(const uint8_t* packed, int lr, int col) {
    int lane = lr / 16;           // 2 lanes per I8 row
    int lane_row = lr % 16;        // row within lane (0..15)
    int byte_idx = lane_row / 2;   // which byte within this column's 8-byte slot
    int nibble_sel = lane_row % 2; // 0=lo nibble, 1=hi nibble
    
    // Layout: [lane 0: 256 cols x 8 bytes][lane 1: 256 cols x 8 bytes]
    const uint8_t* lane_data = packed + lane * (I4_TILE_COLS * 8);
    uint8_t byte_val = lane_data[col * 8 + byte_idx];
    
    int8_t val;
    if (nibble_sel == 0)
        val = (int8_t)(byte_val & 0x0F);
    else
        val = (int8_t)((byte_val >> 4) & 0x0F);
    
    // Sign-extend 4-bit to 8-bit
    if (val >= 8) val -= 16;
    return val;
}

/**
 * Load raw I4 data from a Q4NX I8 tensor into a section of the weight BO.
 *
 * @param bo_ptr       Pointer to the XRT BO weight buffer
 * @param bo_stride    ND value: number of output feature elements per input row
 * @param out_offset   Output feature offset within the BO (e.g., 0 for Q, 2048 for K)
 * @param q4nx_data    Pointer to the mmap'd Q4NX file data
 * @param tensor_off   Byte offset of the I8 tensor within q4nx_data
 * @param i8_rows      Number of I8 rows in this tensor
 * @param in_features  Model hidden dimension (H=1024)
 * @param out_features Number of output features (e.g., NH*HD=2048 for Q)
 */
static void load_i4_to_bo(int8_t* bo_ptr, int bo_stride, int out_offset,
                           const uint8_t* q4nx_data, uint64_t tensor_off,
                           int i8_rows, int in_features, int out_features) {
    int n_tile_cols = in_features / I4_TILE_COLS;
    int n_tile_rows = i8_rows / n_tile_cols;
    
    // Validate
    int expected_out = n_tile_rows * I4_TILE_ROWS;
    if (expected_out != out_features) {
        fprintf(stderr, "[I4] WARNING: expected %d output features, "
                "tile grid gives %d (i8_rows=%d, in=%d)\n",
                out_features, expected_out, i8_rows, in_features);
    }
    
    int rows_checked = 0;
    for (int in = 0; in < in_features; in++) {
        int tc = in / I4_TILE_COLS;        // tile column
        int col = in % I4_TILE_COLS;        // position within tile
        
        for (int out = 0; out < out_features; out++) {
            int tr = out / I4_TILE_ROWS;     // tile row
            int lr = out % I4_TILE_ROWS;     // position within tile
            
            // I8 row index for this tile
            int ir = tr * n_tile_cols + tc;
            
            // Read packed section of this I8 row
            const uint8_t* i8_row = q4nx_data + tensor_off + (uint64_t)ir * I4_ROW_BYTES;
            const uint8_t* packed = i8_row + I4_HEADER_BYTES;
            
            int8_t i4_val = extract_i4(packed, lr, col);
            
            // BO position: [in_feature][out_feature]
            // Bm[in * bo_stride + out]
            bo_ptr[in * bo_stride + out_offset + out] = i4_val;
            rows_checked++;
        }
    }
}

/**
 * Compute global I4 scale for Bcol.
 * For raw I4 values in [-8, 7], amax = 7 (actually 8 since we use abs).
 * Returns: amax / 127.0f
 */
static inline float i4_bscale() {
    return 8.0f / 127.0f;  // max |I4| = 8
}
