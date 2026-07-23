# NPU AIE Kernels — Qwen3-0.6B Support

## Overview

The NPU engine uses 5 AIE (AI Engine) kernels compiled for the XDNA 2 array on
Strix Halo. Each kernel runs on the AIE compute tiles and processes one layer
of the transformer decode pipeline.

## Kernel Pipeline

```
                    ┌──────────────────────────┐
                    │  qwen3_decode_kernels     │
                    │  (main16 Q4NX GEMM)       │
                    │  QKV, O, G, U, D proj    │
                    └──────┬───────────┬────────┘
                           │           │
                    ┌──────▼──┐  ┌─────▼──────────┐
                    │postproc │  │  full_vector    │
                    │_qkv     │  │  _station       │
                    │RoPE +   │  │  residual add   │
                    │KV cache │  │  + RMSNorm      │
                    └────┬────┘  └───────┬─────────┘
                         │              │
                    ┌────▼────┐   ┌──────▼─────────┐
                    │edge_attn│   │   swiglu        │
                    │QK softmax│  │   SiLU(Gate*Up) │
                    │×V accum │   │   + Down proj   │
                    └─────────┘   └────────────────┘
```

## Model Dimension Constants

### Qwen3-0.6B (defined in `qwen3_constants_06b.h`)
| Parameter | Value | Description |
|-----------|-------|-------------|
| H         | 1024  | Hidden size |
| NH        | 16    | Num Q heads |
| NKV       | 8     | Num K/V heads |
| HD        | 128   | Head dimension |
| IM        | 3072  | Intermediate size (FFN) |
| NV        | 151936| Vocab size |
| NC        | 28    | Num layers |
| AW        | 4     | Attention workers (32-tile grid: 4 cols × 2 rows) |
| WQH       | 4     | Q heads per worker = NH/AW |

### Key differences from 8B
| Parameter | 0.6B | 8B | Reason |
|-----------|------|----|--------|
| kQBodyRecords | 4 | 8 | Q output = NH×HD = 2048, 2048/512 = 4 |
| kOBodyRecords | 2 | 8 | O output = H = 1024, 1024/512 = 2 |
| kKvBodyRecords | 2 | 2 | K/V output = NKV×HD = 1024, same |
| kUpGateReplays | 12 | 48 | IM=3072, 6 blocks each for UP+GATE |
| kDownBodyRecords | 2 | 8 | Down output = H = 1024 |
| kQChunksPerRecord | 4 | 16 | Chunk size = 256, Q in dim = 1024 |
| kOChunksPerRecord | 8 | 16 | O in dim = 2048 (NH×HD) |
| kDownChunksPerRecord | 12 | 48 | Down in dim = 3072 (IM) |
| kHeads (edge_attn) | 4 | 8 | WQH=NH/AW=16/4=4 |

## Kernel Source Files

| Kernel | Source | Status | Notes |
|--------|--------|--------|-------|
| main16 Q4NX GEMM | `qwen3_decode_kernels_06b.cc` | ✅ Done | Includes `qwen3_constants_06b.h` |
| Edge attention | `edge_attention.cc` | ✅ Done | Switchable via `MODEL_QWEN3_8B` / default=0.6B |
| Post-process QKV | `postprocess_qkv_06b.cc` | ✅ Done | RoPE + KV cache prep. Q=2048, K=V=1024 |
| Full vector station | `full_vector_station_06b.cc` | ✅ Done | Residual add + RMSNorm, H=1024 |
| SwiGLU | `swiglu_06b.cc` | ✅ Done | IM-independent LUT-based |

## Building

Kernels require AMD's AIE compiler toolchain (xchesscc/aiecc):

```bash
# Set up the MLIR/AIE toolchain
export TORCH2AIE_ROOT=/path/to/torch2aie

# Build all 5 kernels
bash engine/npu/kernel/build_06b_kernels.sh
```

For the full fused-layer xclbin, see the MLIR generator at:
`~/torch2aie/examples/qwen3-decode-layer/cases/`

## MLIR Generation for 0.6B

The fused xclbin is generated from an MLIR design produced by
`full_layer_engine_generate.py`. To generate a 0.6B design:

1. Create `qwen3_06b_decode_layer_runner` (copied from `qwen3_8b` case)
2. Override constants to use `qwen3_constants_06b.h` values
3. Adjust `WEIGHT_PATCH_BD_IDS` for 5 spans instead of 8 (see docs/MLIR-GENERATOR-BLOCKER.md)
4. Update `link_with` paths to `*_06b.o` kernel objects

See `docs/MLIR-GENERATOR-BLOCKER.md` for the full blocker analysis.
