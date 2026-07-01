# CLAUDE.md — 1bit.systems

Two inference engines, one chip. NPU (C++) + GPU (Zig). Zero Python.
Contact: admin@1bit.systems

## Engine: NPU (`engine/npu/`)

C++23 INT8 inference on XDNA 2 NPU. 4-live contexts, 246 ms/tok.

- `engine/npu/src/npu_engine_i8.cpp` — Main loop (155 lines)
- `engine/npu/src/dequant_q4nx.c` — Q4NX dequantizer
- `engine/npu/kernel/edge_attention.cc` — NPU attention (Chess C++)
- `engine/npu/xclbins/n1_core_i8_v2.py` — INT8 MLIR generator

Build: `g++ -std=c++23 -O3 -o npu_engine engine/npu/src/npu_engine_i8.cpp engine/npu/build/dequant_q4nx.o -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl`

## Engine: GPU (`engine/gpu/`)

Zig inference on Vulkan/CUDA/Metal. GGUF native. Compute shaders.

- `engine/gpu/src/vulkan/forward.zig` — Vulkan prefill + decode
- `engine/gpu/src/cuda/` — CUDA backend
- `engine/gpu/src/metal/` — Metal backend (Apple Silicon)
- `engine/gpu/src/shaders/` — GLSL compute shaders (SPIR-V)

Build: `zig build -Doptimize=ReleaseFast`

## Key facts

- Both engines share the same brand, same domain, same chip targets
- NPU engine uses INT8 xclbins + XRT runtime
- GPU engine uses Vulkan 1.3 compute shaders + SPIR-V
- Both are pure native — no Python, no Rust, no runtime interpreters
- Journey doc at docs/journey.md (full audit trail)

## References

- `/home/bcloud/npu-sandbox/` — NPU experiments
- `/home/bcloud/torch2aie/` — AMD toolchain
- `/home/bcloud/zinc/` — Original GPU engine source
