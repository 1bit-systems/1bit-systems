# FAQ

**What is it?** — Fused NPU+GPU+CPU inference. 74 KB. Zero Python.

**Do I need Strix Halo?** — NPU needs XDNA 2. GPU (Vulkan) runs anywhere.

**No Python?** — Zero. C++23 + Zig. No pip, no venv, no conda.

**ROCm vs Vulkan?** — **Vulkan ⭐ is primary.** ROCm available via `-Dbackend=rocm`.

**Production ready?** — Yes. FLM daemon (94 tok/s) + open C++ engine (97 tok/s).
