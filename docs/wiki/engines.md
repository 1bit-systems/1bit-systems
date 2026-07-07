# Engines

> The standalone `engine/npu/`, `engine/gpu/` (Zig), and `engine/fusion/` (Zig
> NPU+GPU dispatcher) directories referenced by earlier versions of this page
> were retired (commit `cd232a091`) — superseded by the `spec-decode/` stack.
> See [docs/wiki/performance.md](performance.md) (source of truth) for current,
> honest per-engine status.

**NPU (production)** — FLM proxy (AMD's proprietary FastFlowLM runtime),
94 tok/s, ✅ measured, coherent.

**NPU (raw, experimental)** — the retired `engine/npu/` C++ stack hit 291 tok/s
(fused layer) and 97 tok/s (v12) raw throughput, but output was never coherent
text. `spec-decode/` is the current development direction for a faster
*coherent* NPU path.

**GPU (Vulkan ⭐ / ZINC)** — Vulkan compute shaders, flash attention, DMMV.
22 tok/s (Bonsai-1.7B F16) up to 381 tok/s (1-bit quant via llama.cpp), all
✅ measured/coherent or measured via third-party tool. See performance.md for
the full breakdown by model/quant.

**GPU (ROCm)** — HIP backend, 113 tok/s reported (not independently
re-measured).

**Spec-decode (DSpark)** — experimental draft-engine for the NPU target.
Current active development direction. The earlier "~572 tok/s" projection was
disproven by end-to-end measurement (2026-07-07): 0.1–0.2 tok/s at 0% draft
acceptance. See `spec-decode/` and `docs/wiki/performance.md`.

**Architecture (current):**
```
spec-decode/
├── draft/     C++ — DSpark draft engine
└── engine/    C++ — spec-decode orchestrator
```
