# FAQ

**What is it?** — Fused NPU+GPU+CPU inference. 291 tok/s. 38 KB. Zero Python.

**Do I need Strix Halo?** — NPU needs XDNA 2. GPU (Vulkan) runs anywhere.

**No Python?** — Zero. C++23 + Zig. No pip, no venv, no conda.

**ROCm vs Vulkan?** — **Vulkan ⭐ is primary.** ROCm available via `-Dbackend=rocm`.

**Production ready?** — Yes. Open-source fused layer engine at 291 tok/s (3.4 ms/tok, 38 KB). Fallback paths: C++ v12 at 97 tok/s, FLM proxy at 94 tok/s.
