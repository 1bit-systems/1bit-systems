#pragma once
// ternary_npu_bridge.h — TQ2/TQ1 ternary → INT8 NPU adapter
//
// Converts ternary weight formats (TQ2, TQ1, block-scaled ternary)
// to INT8 format compatible with existing NPU xclbin kernels.
// This enables immediate NPU inference for any ternary model without
// requiring new AIE kernel designs or xclbin bitstreams.
//
// For the real NPU ternary kernel path (2-bit xclbins), see:
//   docs/npu-ternary-roadmap.md

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

struct TernaryNpuPackResult {
    int8_t* weights;       // INT8 packed weights [rows * cols]
    float   dequant_scale; // scale to multiply NPU INT8 output by
    int     rows;          // k (input dim)
    int     cols;          // n (output dim)
};

// Pack TQ2 (2-bit symmetric ternary, 1BP format tiles) to INT8 NPU format.
// tq2_data: raw 1BP TQ2 tile data pointer
// rows, cols: logical weight matrix dimensions (k, n)
// tile_rows, tile_cols, group_size: tile geometry (default 32, 256, 32)
TernaryNpuPackResult pack_tq2_to_npu_int8(
    const uint8_t* tq2_data,
    int rows, int cols,
    int tile_rows, int tile_cols, int group_size
);

// Pack TQ1 (1.58-bit base-3 ternary) to INT8 NPU format.
// tq1_data: raw 1BP TQ1 packed data
// rows, cols: logical weight matrix dimensions
// tile_rows, tile_cols: tile geometry
TernaryNpuPackResult pack_tq1_to_npu_int8(
    const uint8_t* tq1_data,
    int rows, int cols,
    int tile_rows, int tile_cols
);

// Pack block-scaled ternary (BST) to INT8 NPU format.
// bst_data: BST packed blocks (5 bytes per 16-element block)
// rows, cols: logical weight matrix dimensions
TernaryNpuPackResult pack_bst_to_npu_int8(
    const uint8_t* bst_data,
    int rows, int cols
);

// Free the INT8 weight buffer from any pack function.
void free_ternary_npu_pack(TernaryNpuPackResult* result);

#ifdef __cplusplus
}
#endif
