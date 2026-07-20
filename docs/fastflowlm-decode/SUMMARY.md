# FastFlowLM Decode Report

## What was decoded

| Layer | Production Artifact | Rebuilt Source |
|-------|-------------------|----------------|
| CLI + Server | flm 87.8MB binary | Open-source GitHub (17.5MB) |
| NPU sequence gen | 22 proprietary .so files | libnpu_engine_universal.so |
| FPCA bitstreams | 209 xclbin files | 63 rebuilt from AIE generators |
| Toolchain | AMD Xilinx IP | aiecc + Peano/LLVM-AIE (licensed) |

## Build pipeline
Python AIE kernel generator → MLIR → aiecc + Peano → .xclbin

## Benchmarks (Strix Halo XDNA2 NPU)
- qwen3:0.6b:     98 tok/s decode, 517ms TTFT
- qwen3:1.7b:     41 tok/s decode, 630ms TTFT  
- phi4-mini-it:4b: 21 tok/s decode, 899ms TTFT

## Key insight
The .so files were NPU instruction SEQUENCE GENERATORS, not compute kernels.
The actual computation lives in .xclbin FPGA bitstreams.
Both are now fully rebuildable from source.
