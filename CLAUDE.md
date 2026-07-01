# CLAUDE.md — 1bit.systems

## Project

1-bit/ternary inference engine for AMD Strix Halo NPU (XDNA 2).
Currently INT8 inference on Qwen3-0.6B at 243 ms/tok. Roadmap to BitNet b1.58.

## Architecture

- `engine/src/npu_engine_i8.cpp` — Main inference engine (145 lines C++23)
- `engine/src/dequant_q4nx.c` — Q4NX weight dequantizer (C)
- `engine/xclbins/n1_core_i8_v2.py` — INT8 MLIR generator (K-interleave fixed)
- `engine/kernel/edge_attention.cc` — NPU attention kernel (Chess C++)
- `docs/` — Architecture docs, build guide, roadmap

## Key facts

- Pure C++ runtime. Zero Python at runtime. Zero Rust.
- 4 INT8 GEMM contexts + 4 attention contexts alive simultaneously
- Per-layer weight BOs pre-loaded at startup — no weight memcpy during inference
- Q4NX weights dequantized once at startup (4.5s), then INT8 xclbins take over
- NPU2 supports multiple concurrent hw_contexts (tested: 8 alive at once)
- Chess compiler license required for xclbin builds (free from AMD Ryzen AI EA)

## Build

```bash
# One-time xclbin build
cd engine/xclbins && source torch2aie/scripts/env.sh
python3 n1_core_i8_v2.py -M 128 -K $K -N $N > design.mlir
aiecc --aietools=$AIETOOLS_DIR --aie-generate-xclbin design.mlir

# Engine build
g++ -std=c++23 -O3 -o npu_engine engine/src/npu_engine_i8.cpp \
    engine/build/dequant_q4nx.o \
    -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl
```

## References

- `/home/bcloud/npu-sandbox/` — NPU sandbox with all experiments
- `/home/bcloud/torch2aie/` — AMD toolchain (xclbin compilation)
- `/home/bcloud/Desktop/HANDOFF-NPU-OPTIMIZATION.md` — Full project history
- `/home/bcloud/npu-gpu-cpu/` — Git history with INT8 engine commits
