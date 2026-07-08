# FAQ

**What is it?** — Fused NPU+GPU+CPU inference. 291 tok/s. 30 KB. Zero Python.

**Do I need Strix Halo?** — NPU needs XDNA 2. GPU (Vulkan) runs anywhere.

**No Python?** — Zero. C++23 + Zig. No pip, no venv, no conda.

**ROCm vs Vulkan?** — **Vulkan ⭐ is primary.** ROCm available via `-Dbackend=rocm`.

**Production ready?** — Yes. FLM proxy at 94 tok/s, cascade token router on :13306 routes NPU↔GPU by confidence. Spec-decode binary at 437 KB. Rust token router at 9.7 MB.
