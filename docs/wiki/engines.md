# Engines

**NPU** — C++23, XDNA 2 fused layer xclbin, one call per transformer layer,
291 tok/s (3.4 ms/tok), 38 KB binary. Fallback: C++ v12 (97 tok/s), FLM proxy (94 tok/s).  
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
