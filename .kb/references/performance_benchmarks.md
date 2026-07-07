---
type: Reference
title: Performance Benchmarks (July 2026)
description: Source-of-truth benchmark results — NPU FLM, GPU Vulkan/ROCm, and spec-decode/DSpark — cross-checked against docs/wiki/performance.md and 1bit-site/benchmarks.json.
tags: [benchmarks, performance, npu, gpu, rocm, spec-decode]
resource: https://github.com/1bit-systems/1bit/blob/main/docs/wiki/performance.md
timestamp: 2026-07-07T00:00:00Z
---

# Overview

The single source of truth for benchmark numbers is
[`docs/wiki/performance.md`](https://github.com/1bit-systems/1bit/blob/main/docs/wiki/performance.md),
mirrored machine-readable at `1bit-site/benchmarks.json`. This file summarizes
it for agents — **if these ever disagree, the wiki page and benchmarks.json
win.**

Status labels (from `1bit-site/benchmarks.json._legend`):

| Label | Meaning |
|-------|---------|
| `validated` | measured on-device, coherent output |
| `measured` | throughput measured via a third-party tool (llama.cpp) |
| `projected` | base engine × spec-decode acceptance — NOT an end-to-end measurement |
| `disproven` | a projection contradicted by end-to-end measurement |
| `raw` | kernel runs at this speed but engine output is not yet coherent |
| `reported` | reported, not independently re-measured |

**Only `validated`/`measured` numbers should be quoted as production.**

## Engine Head-to-Head (Qwen3-0.6B unless noted)

| Engine | tok/s | Status | Notes |
|--------|-------|--------|-------|
| NPU FLM (production) | **94** | validated | Served via FLM daemon; `spec-decode/engine/spec_proxy.cpp` proxies speculative-decode requests in front of it |
| GPU ternary (Vulkan) | **279** | validated | Bonsai-1.7B Q2_0 (1.58-bit) |
| GPU 1-bit (llama.cpp) | **381** | measured | Qwen2-0.5B IQ1_S — see [GPU 1-Bit Benchmarks](/references/gpu_benchmarks.md) |
| GPU ZINC (Vulkan) | **22** | validated | Bonsai-1.7B F16 |
| ROCm (HIP) | **113** | reported | Bonsai-1.7B TQ2 — see [ROCm Backend](/engines/rocm_backend.md) |
| DSpark spec-decode | **0.1–0.2** | disproven | 0% draft acceptance measured end-to-end 2026-07-07; the earlier "~572 tok/s" figure was a projection from a different model (Qwen3-4B) and does not hold on the NPU INT8 target. Experimental, not production. |

## Retired engines (deleted 2026-07-06/07)

`engine/fusion/`, `engine/npu/`, and `npu-gpu-cpu/` were removed in commit
`cd232a091`, superseded by the `spec-decode/` stack. Their reported numbers —
"NPU fused" (291 tok/s) and "NPU C++ v12" (97 tok/s) — were already flagged
`raw` (kernel throughput only, **output not yet coherent**) before deletion,
so they were never valid production figures even when the code existed. See
the archived, clearly-marked docs if you need historical context:

- [\[RETIRED\] NPU Inference Engine (Fused Layer)](/archive/engines/npu_engine.md)
- [\[RETIRED\] Fused Engine (NPU+GPU Hybrid)](/archive/engines/fused_engine.md)
- [\[RETIRED\] Unified Serving Daemon](/archive/engines/unified_daemon.md)

## Full Benchmark Source

See [`docs/wiki/performance.md`](https://github.com/1bit-systems/1bit/blob/main/docs/wiki/performance.md)
for the complete live-verified tables: full 1-bit model benchmarks, ROCm
kernel benchmarks, the DSpark acceptance-profile breakdown, H2O KV-cache
eviction numbers (historical — no current engine implements H2O eviction),
and Wave32 GPU optimization notes.

## Citations

[1] [Performance wiki](https://github.com/1bit-systems/1bit/blob/main/docs/wiki/performance.md)
[2] [benchmarks.json](https://github.com/1bit-systems/1bit/blob/main/1bit-site/benchmarks.json)
[3] [GPU 1-Bit Benchmarks](/references/gpu_benchmarks.md)
