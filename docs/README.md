# 1bit.systems — Documentation

Model-agnostic **NPU + GPU + CPU** inference engine for AMD Strix Halo.
Pure C++23, zero Python at runtime, MIT.

## Start here

| Doc | What it covers |
|-----|----------------|
| [guides/getting-started.md](guides/getting-started.md) | Install, build, first inference |
| [guides/building.md](guides/building.md) | Full build guide (CMake, ROCm/TheRock, targets) |
| [guides/architecture.md](guides/architecture.md) | Engine internals: loaders, backends, routing |
| [journey.md](journey.md) | **The hero story** — reverse-engineering the XDNA 2 NPU in 4 days |
| [guides/roadmap.md](guides/roadmap.md) | Where we're headed |

## Guides & platform

- [guides/windows.md](guides/windows.md) — Windows notes
- [guides/Lemonade-Compat.md](guides/Lemonade-Compat.md) — Lemonade compatibility
- [guides/launch.md](guides/launch.md) — launch/serving

## Research & technical deep-dives

- [research/npu-engine.md](research/npu-engine.md) · [research/npu-dynamic-instr.md](research/npu-dynamic-instr.md) — Native XDNA 2 NPU engine
- [research/npu-ternary-roadmap.md](research/npu-ternary-roadmap.md) — Ternary/binary NPU kernel roadmap
- [research/fastflowlm-decode/SUMMARY.md](research/fastflowlm-decode/SUMMARY.md) — FastFlowLM reverse-engineering report (historical — fully replaced)
- [research/fastflowlm-analysis/](research/fastflowlm-analysis/) — Raw binary analysis, xclbin captures, instruction traces
- [research/block-scaled-ternary-format.md](research/block-scaled-ternary-format.md) — 1BP / TQ2 ternary storage
- [research/kernel-analysis.md](research/kernel-analysis.md) · [research/hybrid-w4a8-router.md](research/hybrid-w4a8-router.md) — Kernels & routing
- [research/vision-module.md](research/vision-module.md) — Vision-language support
- [research/flux-feasibility.md](research/flux-feasibility.md) — Flux image-model feasibility notes
- [research/npu/AMD-XDNA-40COLUMN-UNLOCK.md](research/npu/AMD-XDNA-40COLUMN-UNLOCK.md) — NPU column-unlock deep dive

## Validation & status

- [research/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md](research/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md)
- [research/bosgame-m5-full-validation.md](research/bosgame-m5-full-validation.md) · [research/bosgame-m5-iommu-validation.md](research/bosgame-m5-iommu-validation.md)

## Wiki

Deeper operational docs live in [`wiki/`](wiki/): [performance](wiki/performance.md) (**single
source of truth for benchmark numbers**), [Installation](wiki/Installation.md),
[npu-architecture](wiki/npu-architecture.md), [boot-configuration](wiki/boot-configuration.md),
[Network-Topology](wiki/Network-Topology.md).

## Business & marketing

- [business/business-plan.md](business/business-plan.md) — Business plan and economics
- [business/marketing/](business/marketing/) — Launch write-ups and social posts

## Archive

Historical status notes, resolved blockers, handoffs, and superseded plans are kept
in [`archive/`](archive/) for provenance — not current guidance.
