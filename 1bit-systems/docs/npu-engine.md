# NPU Inference Engine

The NPU engine (`engine/npu/src/npu_engine_universal.cpp`) runs inference on the AMD XDNA 2 NPU via XRT. It uses compiled AI Engine (AIE) xclbin programs for GEMM operations and falls back to CPU for attention.

## Prerequisites

- AMD Strix Halo (Ryzen AI MAX+ 395) or compatible XDNA 2 NPU
- Linux kernel with `amdxdna` driver loaded (check `lsmod | grep amdxdna`)
- XRT runtime (`libxrt_core`, `libxrt_coreutil`)
- NPU firmware (`npu_7.sbin`) at `/lib/firmware/amdnpu/17f0_11/`

## Xclbin Files

The NPU engine requires compiled xclbin and instruction files at `engine/npu/xclbins/`:

```
engine/npu/xclbins/
├── final_i8_QKV_qwen3_0_6b.xclbin   # QKV projection
├── final_i8_O_qwen3_0_6b.xclbin     # O projection  
├── final_i8_GU_qwen3_0_6b.xclbin    # Gate+Up projection
├── final_i8_D_qwen3_0_6b.xclbin     # Down projection
├── insts_i8_QKV_qwen3_0_6b.txt      # Instructions for QKV
├── insts_i8_O_qwen3_0_6b.txt        # Instructions for O
├── insts_i8_GU_qwen3_0_6b.txt       # Instructions for GU
└── insts_i8_D_qwen3_0_6b.txt        # Instructions for D
```

### Setup

Use the setup script to symlink FLM-compiled xclbins:

```bash
# Point at your fastflowlm-build checkout
export NPU_XCLBIN_DIR=/path/to/fastflowlm-build/src/xclbins
./scripts/setup_npu_xclbins.sh
```

Or set the environment variable at runtime:
```bash
export NPU_XCLBIN_DIR=/path/to/fastflowlm-build/src/xclbins
```

## Running

```bash
# Build
cd engine/npu && mkdir -p build
g++ -std=c++17 -O3 -fopenmp -o build/npu_engine src/npu_engine_universal.cpp \
    -lxrt_core -lxrt_coreutil -luuid -lpthread \
    -I/usr/include -L/usr/lib/x86_64-linux-gnu

# Run (model.q4nx + decode tokens + optional input token file)
./build/npu_engine /path/to/model.q4nx 10
```

## Architecture

The engine works in three phases:

1. **Init**: Load Q4NX model → Dequantize weights → Pack to INT8 → Upload to NPU
2. **Prefill**: Process prompt tokens sequentially through NPU GEMMs + CPU attention
3. **Decode**: Autoregressive token generation with NPU GEMMs + CPU attention

Each token goes through:
- QKV projection (NPU via `cq` context)
- QK norm + RoPE + KV cache update (CPU)
- Causal attention (CPU via `attn_omp` — OpenMP parallelized)
- O projection (NPU via `co` context)
- Gate+Up projection (NPU via `cg` context)
- SiLU activation (CPU)
- Down projection (NPU via `cd` context)

## Performance

| Model | Prefill | Decode | Notes |
|-------|---------|--------|-------|
| Qwen3-0.6B | ~180s (9 tok) | ~2s/tok | CPU attention bottleneck |

The CPU attention is the current bottleneck. An NPU attention xclbin would significantly improve performance.

## Worker Mode

The engine supports a subprocess protocol for the Zig fused executor (`engine/fusion/main.zig`):

```bash
./npu_engine model.q4nx --worker
```

Protocol: stdin/stdout with 4-byte header `[op, layer, batch, in_dim]` followed by float input data. Supported operations: 1=QKV, 2=O, 3=Gate+Up, 4=Up, 5=Down.
