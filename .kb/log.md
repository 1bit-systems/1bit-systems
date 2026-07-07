# Update Log

## 2026-07-07

Commit `cd232a091` deleted `engine/npu/`, `engine/gpu/`, `engine/fusion/`,
`engine/video/`, `engine/lora/`, and `npu-gpu-cpu/` (superseded by the
`spec-decode/` stack). Updated `.kb/` to match:

- Archived (moved to `archive/`, headers marked retired): `engines/fused_engine.md`,
  `engines/npu_engine.md`, `engines/unified_daemon.md`,
  `references/dispatch_policies.md`, `structures/kv_cache.md`.
- Rewrote `references/performance_benchmarks.md` in place to drop the
  retired/non-coherent "NPU fused"/"NPU C++ v12" figures and reflect current
  ground truth from `docs/wiki/performance.md` / `1bit-site/benchmarks.json`
  (NPU FLM 94 tok/s validated production; DSpark spec-decode marked
  disproven/experimental).
- Left `references/gpu_benchmarks.md`, `engines/rocm_backend.md`, and
  `structures/zaya_architecture.md` untouched — they don't reference deleted
  paths and their numbers still check out against the ground truth.
- Updated `index.md` files (root, `engines/`, `references/`, `structures/`)
  to drop dead links and point to `archive/`.
- Regenerated `viz.html` from the updated bundle via `generate_viz.py`.

## 2026-07-06
