# Q4NX Weight Format Specification

Reverse-engineered from `libq4_npu_eXpress.so` (GCC 13.3.0, Ubuntu 24.04.1)

## Overview

Q4NX is a proprietary 4-bit quantized weight format used by FastFlowLM to store model weights for NPU inference on AMD XDNA2. It wraps the HuggingFace SafeTensors format with a custom 4-bit block quantization layout optimized for the NPU's DMA engine.

## File Structure

A Q4NX model is stored in a directory containing:
```
/model.q4nx          — Quantized weight file (Q4NX format)
/model.safetensors   — Optional original safetensors (for re-quantization)
```

## Format Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `group_size` | 32 | Number of elements in one quantization group |
| `group_size_bytes` | 40960 | Bytes per group (0xa000 = 40960 — includes quantized data + scales) |
| `row_chunk_size` | 32 | Rows per chunk (NPU DMA granularity) |
| `col_chunk_size` | 256 | Columns per chunk |
| `weight_per_chunk` | 8192 | Bytes per chunk (0x2000) |
| `num_groups_per_chunk_row` | 8 | Groups along one chunk row |
| `num_groups_per_row_parallel` | 2 | Parallel groups per row |
| `block_size` | 5120 | NPU memory block size in bytes (0x1400) |

## Quantization Scheme

### Group-level quantization

Weights are quantized in groups of 32 elements (`group_size = 32`). Each group stores:

```
Group layout (32 elements, 4-bit each = 16 bytes):
┌──────────────────────────────────────┐
│  4-bit quantized indices (16 bytes)  │  ← 32 indices packed into 16 bytes
│  Scale (bfloat16, 2 bytes)           │  ← per-group scale factor
│  Zero point (bfloat16, 2 bytes)      │  ← per-group zero point
└──────────────────────────────────────┘
Total: 20 bytes per 32 elements = 0.625 bytes/element (5 bits per value)
```

### Dequantization formula

```
dequantized_value = (q4_value * scale) + zero_point
```

Where `q4_value` is a 4-bit unsigned integer (0-15), `scale` and `zero_point` are bfloat16.

### Reordering for NPU

The `_q4nx_reorder()` function transforms the logical weight layout into an NPU-optimized memory layout:

```
Signature: _q4nx_reorder(bytes&, buffer<uint32_t>&, buffer<bfloat16_t>&, buffer<int>&, int)

Parameters:
  - bytes& quantized_data       — input quantized weight bytes
  - buffer<uint32_t>& qdata      — output uint32 aligned buffer (quantized indices)
  - buffer<bfloat16_t>& scales   — output scale factors
  - buffer<int>& zero_points     — output zero points (can be zero-filled)
  - int orig_size               — original dimension size
```

**Reordering algorithm** (reverse-engineered from disassembly):

```
1. Compute num_chunks = ceil(orig_size / 32)    // 32 = row_chunk_size
2. Compute padded_size = round_up(chunk_row, 32) // align to 32-element boundary
3. For each chunk in num_chunks:
   a. Allocate padded buffers:
      - qdata.resize(padded_size * orig_size / 8)     // 4-bit packing: /8 bytes
      - scales.resize(padded_size * num_chunks)
      - zero_points.resize(padded_size * num_chunks)
   b. Zero-initialize:
      - qdata entries for the chunk row
      - bf16 scales = 0 for the chunk row
      - int zero_points = 0 for the chunk row
   c. Copy original data into chunked layout:
      For each group in chunk:
        - qdata[group_offset] = packed 4-bit indices
        - scales[scale_offset] = bf16 scale
        - zero_points[zp_offset] = (zero point, stored as int)
```

The reordering converts from a flat `[rows, cols]` layout into a tiled `[chunks, 32, cols]` layout where each 32-row tile is contiguous in NPU memory.

## Weight Metadata JSON

Each Q4NX model stores metadata about its quantization parameters. Keys found in the binary:

```json
{
    "shape":           [rows, cols],
    "dtype":           "Q4NX",
    "data_offsets":    [start_byte, end_byte],
    "group_size":      32,
    "row_chunk_size":  32,
    "col_chunk_size":  256,
    "row_parallel_size": 2,
    "scale_type":      "bf16",
    "zero_point_type": "bf16",
    "_scale":          <scale values>,
    "__metadata__": {
        "shape":       "model architecture identifier"
    }
}
```

## LLama-specific Conversion

`_convert_llama()` processes each weight tensor from the safetensors file through `_convert_to_q4nx()`. It iterates through:

| Weight Name | Tensor Shape | Description |
|-------------|-------------|-------------|
| `model.embed_tokens.weight` | [vocab_size, dim] | Token embedding |
| `model.layers.N.input_layernorm.weight` | [dim] | Pre-attention RMS norm |
| `model.layers.N.self_attn.q_proj.weight` | [dim, dim] | Query projection |
| `model.layers.N.self_attn.k_proj.weight` | [dim, dim] | Key projection |
| `model.layers.N.self_attn.v_proj.weight` | [dim, dim] | Value projection |
| `model.layers.N.self_attn.o_proj.weight` | [dim, dim] | Output projection |
| `model.layers.N.post_attention_layernorm.weight` | [dim] | Post-attention RMS norm |
| `model.layers.N.mlp.gate_proj.weight` | [dim, ffn_dim] | Gate projection (SwiGLU) |
| `model.layers.N.mlp.up_proj.weight` | [dim, ffn_dim] | Up projection |
| `model.layers.N.mlp.down_proj.weight` | [ffn_dim, dim] | Down projection |
| `model.norm.weight` | [dim] | Final RMS norm |
| `lm_head.weight` | [vocab_size, dim] | LM head (tied with embed often) |

## Conversion Pipeline

```
SafeTensors file
    │
    ▼
SafeTensors::_open_file()       — reads .safetensors header
SafeTensors::get_metadata()     — parses JSON metadata (shape, dtype, offsets)
    │
    ▼
Q4NX::_grap_metadata()          — extracts quantization parameters
    │
    ▼
Q4NX::_convert_to_q4nx()       — per-tensor conversion:
    1. Load raw float/bf16 weights
    2. Quantize to 4-bit per group of 32
    3. Compute bf16 scale + zero point per group
    4. Pack 4-bit indices (2 per byte, little-endian)
    │
    ▼
Q4NX::_q4nx_reorder()          — reorder for NPU DMA access pattern
    │
    ▼
Q4NX::convert_model()          — write /model.q4nx output file
```

## Reordering Deep-Dive (from disassembly)

The `_q4nx_reorder` function at address `0xa4900` (~547 instructions):

```
Input:  bytes&         — raw quantized bytes (4-bit packed)
        buffer<uint>&  — output qdata (will be resized)
        buffer<bf16>&  — output scales (will be resized)
        buffer<int>&   — output zero_points (will be resized)
        int orig_size  — original dimension being chunked

Steps:
1.  num_chunks = (orig_size + 31) >> 5       // ceil division by 32
2.  padded_chunk_size = ((chunk_row + 31) >> 5) << 5  // round up to 32
3.  For each chunk (P):
    a. Output array sizes: 
       qdata_size = padded_chunk_size * orig_size * 4 / 32  // 4-bit, 2 per byte
       scales_size = padded_chunk_size * num_chunks
       zp_size = padded_chunk_size * num_chunks
    b. Zero-fill all output arrays for this chunk
    c. Copy and interleave data from original buffer into chunk layout
4.  Final result is a tile-laid-out buffer where each 32×col_chunk 
    block is contiguous for efficient NPU DMA
```

## NPU Block Size Interaction

The `get_block_size()` returns `0x1400` = 5120 bytes. This is the NPU's memory transaction granularity. The weight layout ensures each DMA transfer aligns to this block size by padding chunk rows to 32-element boundaries.

The `get_weight_per_chunk()` returns `0x2000` = 8192 bytes, which is the total data (quantized + metadata) per chunk row.

## Key Assertion

Found in binary: `"a_block_size % vetrical_block_interleave_byte_size == 0"` — the block size must be evenly divisible by the vertical interleave size for correct NPU memory access.

Also found: `"total_size == q4nx_weight.size()"` — size validation during conversion.

## Dequantization Functions (for reference implementation)

Two overloads found:
1. `q4nx_dequantize<float>(bytes& input, bytes& output, int row)`
2. `q4nx_dequantize<bfloat16_t>(bytes& input, bytes& output, int row)`

These reverse the quantization to recover the original bf16 weights for CPU-side fallback.
