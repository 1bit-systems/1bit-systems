---
okf_version: "0.1"
links: []
backlinks: []
---

# structures

- [Zaya Model Architecture](zaya_architecture.md) — Hybrid CCA-attention + Mixture-of-Experts with EDA router. Supported in zaya-llama.cpp with custom ROCm kernels.  [Structure]

> Unified KV Cache (H2O Eviction) was retired along with `engine/gpu/src/scheduler/`
> in commit `cd232a091` — archived, clearly marked retired, at
> [`.kb/archive/structures/kv_cache.md`](../archive/structures/kv_cache.md).
> The current NPU target (`spec-decode/engine/npu_target_model.h`) uses a
> simple per-layer KV cache with none of that eviction/sharing machinery.
