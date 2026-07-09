# Archived Documentation

These docs are **historical** — they describe work on the now-retired
standalone NPU/GPU/fusion engine stack, or status/handoff notes for completed
work. They are kept for the audit trail but are **not current**.

## Why they were archived

Commit `cd232a091` retired the standalone `engine/npu/` (legacy C++ NPU
runtime), `engine/gpu/` (Zig GPU backend), and `engine/fusion/` (Zig NPU+GPU
dispatcher) — all superseded by the `spec-decode/` stack. The docs below
reference those retired code paths as if they were active, or record the status
of investigations that have since concluded.

## For current information, see

- **[docs/wiki/performance.md](../wiki/performance.md)** — source of truth for
  per-engine status and benchmark numbers.
- **[docs/wiki/engines.md](../wiki/engines.md)** — current engine overview.
- **[site/benchmarks.json](../../site/benchmarks.json)** — machine-readable
  benchmark numbers.
- **[docs/STATUS.md](../STATUS.md)** — current engine status summary.
- **[docs/journey.md](../journey.md)** — full chronological audit trail
  (including the entries that produced the docs below).

## Contents

### NPU engine correctness & blocker investigations (retired `engine/npu/`)
- `GEMM-KERNEL-CORRECTNESS-CONFIRMED.md` — INT8 GEMM correctness root-cause (Jul 5).
- `NPU-ENGINE-CORRECTNESS-STATUS.md` — end-to-end correctness analysis (Jul 5).
- `NPU-QKV-CACHE-WEIGHTS-BROKEN.md` — QKV weight-cache corruption (Jul 5, FIXED).
- `V12-CORRECTNESS-BLOCKER.md` — v12 engine coherence audit (Jul 5).
- `NPU-DEEP-TRACE.md` — per-sublayer trace of `npu_engine_cb` (Jul 5).
- `XCLBIN-WEIGHT-LAYOUT-BUG.md` — Q4NX tile-blocked weight layout bug.

### Fused-xclbin / MLIR work (retired fused layer)
- `FUSED-INTEGRATION-BLOCKER.md` — fused xclbin integration blocker (Jul 2).
- `FUSED-XCLBIN-PORT-PLAN.md` — fused xclbin port plan for Qwen3-0.6B.
- `WEIGHT-STREAM-BLOCKER.md` — fused xclbin final status (Jul 2).
- `MLIR-GENERATOR-BLOCKER.md` — MLIR generator port blocker (Jul 2).
- `NPU-ATTENTION-STATUS.md` — NPU attention kernel status (Jul 2).

### Tooling / flags reference (retired engines)
- `NPU-LOCK-DEVICE-AND-FLAGS.md` — flags & device-lock reference for retired engines (Jul 6).

### Handoff & optimization logs
- `HANDOFF-NPU-OPTIMIZATION.md` — large running handoff log for NPU optimization.

### Plans / scope for retired code paths
- `npu-full-unlock-plan.md` — perf-extraction plan for the retired standalone NPU engine.
- `npu-gpu-direct-wiring.md` — NPU+GPU zero-copy wiring (both engines retired).
- `mlx-npu-backend-scope.md` — MLX IRON backend scope (premise reused retired `engine/npu/` files).
- `flm-integration.md` — early FLM integration against retired `engine/npu/` v3 (self-labeled historical).

---

*Archived during the 2026-07-08 stale-doc cleanup. Numbers and code paths in
these files should not be treated as current.*
