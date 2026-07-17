#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

// ── C-callable Q4NX GPU dequant launcher ──
// Launches a GPU kernel to dequantize I4_CUSTOM packed data to FP16.
//
// Parameters:
//   d_in        - device pointer to packed I4 data (groups * 8 bytes)
//   d_out       - device pointer for FP16 output (rows * cols * 2 bytes)
//   total_groups - total number of I4 groups (rows * cols / 8)
//   groups_per_row - number of groups per row (cols / 8)
//   out_stride  - stride between rows in output (in elements, usually = cols)
//   stream      - HIP stream for async launch
//
// Returns 0 on success, nonzero on error.
extern "C" int launch_dequant_i4_custom(
    const uint8_t* d_in, __half* d_out,
    int total_groups, int groups_per_row, int out_stride,
    hipStream_t stream);
