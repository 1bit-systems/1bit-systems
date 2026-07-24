# INT8 Inference Engine

Pure C++ inference engine for Qwen3-0.6B on AMD Strix Halo NPU.

## Files

| File | Purpose |
|------|---------|
| `src/npu_engine_i8.cpp` | Main engine — 4-live INT8 contexts, NPU attention |
| `src/dequant_q4nx.cpp` | Q4NX weight dequantizer (C99) |
| `xclbins/n1_core_i8_v2.py` | INT8 MLIR generator with K-interleaving fix |
| `kernel/edge_attention.cc` | NPU attention kernel (Chess C++) |
| `build/dequant_q4nx.o` | Compiled dequantizer (committed, zero-dependency) |

## Build

```bash
# One-time: compile dequantizer
gcc -c -O3 -o build/dequant_q4nx.o src/dequant_q4nx.cpp

# Compile engine (requires XRT headers + libs)
g++ -std=c++23 -O3 -o build/npu_engine \
    src/npu_engine_i8.cpp build/dequant_q4nx.o \
    -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl

# Run
./build/npu_engine
```

## Architecture

4 INT8 GEMM xclbins + 4 NPU attention xclbins, all alive simultaneously.
Per-layer weight BOs pre-loaded at startup. Zero Python at runtime.
