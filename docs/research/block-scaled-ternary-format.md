# Block-Scaled Ternary Format — 16-element blocks with FP8 shared scale.
#
# Motivated by NVFP4/MXFP4 research (arXiv:2509.23202, 0xsero deep-dive):
# per-block scaling at 16-element granularity provides ~0.3-0.5 perplexity
# improvement over per-row or per-tensor scaling, with only ~7% storage
# overhead (5 bytes per 16 values = 2.5 b/elem vs 2 b/elem raw ternary).
#
# Layout per block (5 bytes):
#   [3:0]  packed16       : 16 ternary values, 2 bits each, packed uint32 (LE)
#   [4]    block_scale    : FP8 E4M3 (1+4+3, ±max FP8 value)
#
# Ternary encoding (2 bits per value):
#   00  ->  0
#   01  -> +1
#   10  -> -1
#   11  ->  0  (reserved)
#
# For a weight matrix [rows, cols]:
#   blocks_per_row = (cols + 15) / 16
#   Total storage = rows * blocks_per_row * 5 bytes
#
# Dequant: out[i] = ternary_decode(packed[bid][i]) * block_scale[bid]
#
# This file is a README. The C++ implementation is in include/block_scaled_ternary.h
# and the GPU kernel is in kernels/ternary_gemv_block_scaled.hip.
# The Python rotation preprocessor is in tools/mr_gptq_rotate.py.
