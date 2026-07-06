# Engines

**NPU** — C++23, XDNA 2 INT8 xclbins, M=32 batch, 97 tok/s  
→ [`engine/npu/`](../engine/npu/)

**GPU (Vulkan ⭐)** — Zig, GLSL compute shaders, flash attention, DMMV  
**GPU (ROCm)** — Zig/C, HIP Driver API, `-Dbackend=rocm`  
**GPU (CUDA)** — Zig/C, NVIDIA NVRTC, `-Dbackend=cuda`  
**GPU (Metal)** — Zig, Apple Silicon  
→ [`engine/gpu/`](../engine/gpu/)

**Fused** — Zig, NPU+GPU per-layer dispatch, shared KV cache, 8 policies  
→ [`engine/fusion/`](../engine/fusion/)

**Architecture:**
```
engine/
├── npu/       C++23 — NPU
├── gpu/       Zig — GPU (Vulkan / ROCm / CUDA / Metal)
└── fusion/    Zig — NPU+GPU
```
