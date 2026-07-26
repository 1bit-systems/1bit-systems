# FastFlowLM (FLM) Reverse Engineering Analysis

## Overview

This directory contains the complete reverse-engineering extraction of **FastFlowLM (FLM)** v0.9.24 — AMD's commercial C++ LLM inference engine for XDNA2 NPUs. All 22 proprietary `.so` libraries have been fully disassembled (5.8M lines), their 1,390 exported symbols documented, and every optimization secret extracted.

## What Was Extracted

| Document | Content |
|----------|---------|
| `FLM_SECRETS.md` | **Complete secrets** — Q4NX format, NPU ISA, XRT architecture, inference pipeline, MHA variants, per-model extras, build toolchain |
| `NPU_ISA.md` | **NPU instruction set** — All 5 opcodes (0x00/0x01/0x03/0x80/0x81), exact bitfield encodings, RTP register map (0x1000-0x1f0a0), tile layout, DMA 3D addressing |
| `Q4NX_FORMAT.md` | **Weight format spec** — 4-bit block quant with bf16 scales, 32-element groups, tile reordering algorithm (32×256 tiles), NPU DMA block alignment |
| `API_SURFACE.md` | **All 1,390 exported symbols** — Demangled C++ signatures for every function in all 22 `.so` files |
| `q4nx_converter/` | **Python converter** — Full Q4NX conversion for all model architectures |

## Architecture Secrets

### NPU Communication Stack
```
Model Code → npu_sequence → npu_app (compiles to ELF) → XRT → xrt::bo/xrt::run → AIE Array
```

### 5 NPU Instruction Types
| Opcode | Name | Size | Purpose |
|--------|------|------|---------|
| `0x00` | XAIE_IO_WRITE | 6 words | Register write / Queue push |
| `0x01` | XAIE_IO_BLOCKWRITE | 12 words | N-dimensional DMA transfer |
| `0x03` | XAIE_IO_MASKWRITE | 7 words | Issue sync token |
| `0x80` | XAIE_IO_CUSTOM_OP_TCT | variable | Wait for DMA completion |
| `0x81` | XAIE_IO_CUSTOM_OP_DDR_PATCH | 10 words | DDR address patching |

### Q4NX Weight Format
- **Group size**: 32 elements
- **Packing**: 4-bit indices (2 per byte, little-endian)
- **Scale**: bf16 per group
- **Zero point**: bf16 per group (usually 0)
- **Tile**: 32 rows × 256 cols = 5120 bytes per NPU block
- **Overhead**: 20 bytes per 32 elements = 0.625 bytes/element

### GEMM on NPU (from libgemm.so)
- K-dimension blocked into 32-element chunks
- N-dimension tiled into 128-element columns
- 8 SHIM tiles (IT0-IT7) handle DDR DMA
- 32 compute tiles (CT00-CT37) do the actual matmul
- RTP registers at 0x1000-0x1010 configure M/K/N/activation/bias per tile
- Kick-off register at 0x1f0a0 starts execution

### MHA Attention (from libmha.so)
6 variants decoded:
| Variant | Head Dim | Quant | Use Case |
|---------|----------|-------|----------|
| `d64_q4` | 64 | Q4 | Small heads |
| `d128_q2` | 128 | Q2 | Compressed medium |
| `d128_q3` | 128 | Q3 | Balanced medium |
| `d128_q4` | 128 | Q4 | Standard medium |
| `d256_q2` | 256 | Q2 | Compressed large |
| `d256_q4` | 256 | Q4 | Standard large |

### Checkpoint/Restore
- `bytes::sync_from_device()` — KV cache NPU → host
- `bytes::sync_to_device()` — KV cache host → NPU
- Returns context length at checkpoint time for prefix restoration
- Uses `xrt::bo::sync()` for DMA

### Build Environment
- GCC 13.3.0 (Ubuntu 24.04.1), C++20
- XRT 2.x + AIEBU + nlohmann/json 3.12.0

## Files

- `FLM_SECRETS.md` — Comprehensive secrets document (13.6 KB)
- `NPU_ISA.md` — NPU instruction set architecture (10.7 KB)
- `Q4NX_FORMAT.md` — Weight format specification (8.2 KB)
- `API_SURFACE.md` — All 1,390 exported symbols (115 KB)
- `q4nx_converter/` — Tools for converting models to/from Q4NX format
- `../engine/npu/src/gemm_npu_instructions.cpp` — Open-source GEMM/MHA instruction generator
- `../engine/npu/src/checkpoint_restore.cpp` — KV cache checkpoint/restore implementation
