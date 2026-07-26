# 1bit.systems — Documentation

Model-agnostic **NPU + GPU + CPU** inference engine for AMD Strix Halo.
Pure C++23, zero Python at runtime, MIT.

## Start here

| Doc | What it covers |
|-----|----------------|
| [getting-started.md](getting-started.md) | Install, build, first inference |
| [building.md](building.md) | Full build guide (CMake, ROCm/TheRock, targets) |
| [architecture.md](architecture.md) | Engine internals: loaders, backends, routing |
| [journey.md](journey.md) | **The hero story** — reverse-engineering the XDNA 2 NPU in 4 days |
| [roadmap.md](roadmap.md) | Where we're headed |

## Reference

| Doc | Topic |
|-----|-------|
| [npu-engine.md](npu-engine.md) · [npu-dynamic-instr.md](npu-dynamic-instr.md) | Native XDNA 2 NPU engine |
| [fastflowlm-decode/SUMMARY.md](fastflowlm-decode/SUMMARY.md) | FastFlowLM reverse-engineering report (historical — fully replaced) |
| [block-scaled-ternary-format.md](block-scaled-ternary-format.md) | 1BP / TQ2 ternary storage |
| [kernel-analysis.md](kernel-analysis.md) · [hybrid-w4a8-router.md](hybrid-w4a8-router.md) | Kernels & routing |
| [vision-module.md](vision-module.md) | Vision-language support |
| [npu/AMD-XDNA-40COLUMN-UNLOCK.md](npu/AMD-XDNA-40COLUMN-UNLOCK.md) | NPU column-unlock deep dive |

## Guides & platform

- [windows.md](windows.md) — Windows notes
- [Lemonade-Compat.md](Lemonade-Compat.md) — Lemonade compatibility
- [launch.md](launch.md) — launch/serving

## Validation & status

- [GEMM-KERNEL-CORRECTNESS-CONFIRMED.md](GEMM-KERNEL-CORRECTNESS-CONFIRMED.md)
- [bosgame-m5-full-validation.md](bosgame-m5-full-validation.md) · [bosgame-m5-iommu-validation.md](bosgame-m5-iommu-validation.md)

## Wiki

Deeper operational docs live in [`wiki/`](wiki/): [performance](wiki/performance.md),
[Installation](wiki/Installation.md), [npu-architecture](wiki/npu-architecture.md),
[boot-configuration](wiki/boot-configuration.md), [Network-Topology](wiki/Network-Topology.md).

## Marketing / posts

Launch write-ups and social posts are in [`marketing/`](marketing/).

## Archive

Historical status notes, resolved blockers, handoffs, and superseded plans are kept
in [`archive/`](archive/) for provenance — not current guidance.
