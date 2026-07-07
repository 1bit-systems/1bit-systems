# Engine Status

**The standalone `engine/npu/`, `engine/gpu/`, and `engine/fusion/` implementations
described in earlier versions of this doc were retired** (commit `cd232a091`) —
superseded by the `spec-decode/` stack. This page now reflects current status per
the source of truth, [docs/wiki/performance.md](wiki/performance.md).

## Production: NPU FLM (94 tok/s) ✅

AMD's proprietary FastFlowLM runtime, used as a proxy. Measured, coherent output.
This is the only NPU path currently in production.

- **Status**: ✅ measured, coherent — production

## GPU ZINC / Ternary (Vulkan) ✅

Vulkan compute shaders. Coherent output verified.

- **GPU ZINC** (Bonsai-1.7B-F16): 22 tok/s, ✅ measured, coherent
- **GPU ternary** (Bonsai-1.7B Q2_0, 1.58-bit): 279 tok/s, ✅ measured, coherent
- **GPU 1-bit** (llama.cpp, Qwen2-0.5B IQ1_S): 381 tok/s, measured (third-party tool)

## NPU fused layer (291 tok/s) ⚙️ raw — not production

One xclbin call per transformer layer (QKV→attention→O→GU→SiLU→D on NPU).
Raw kernel throughput only — **output is not yet coherent**. Retired along with
the rest of `engine/npu/` in commit `cd232a091`; historical detail in
`docs/FUSED-INTEGRATION-BLOCKER.md` and `docs/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md`.

## C++ v12 (97 tok/s) ⚙️ raw — not production

Standalone INT8 GEMM engine. Raw throughput only — output not yet coherent.
Retired along with the rest of `engine/npu/` in commit `cd232a091`.

## DSpark spec-decode — experimental, disproven projection

The current active development direction (`spec-decode/`). The earlier "~572
tok/s" projection was disproven by end-to-end measurement on 2026-07-07: 0.1–0.2
tok/s at 0% draft acceptance. See `docs/wiki/performance.md` for full detail.

---

*Last updated: 2026-07-07, to match `docs/wiki/performance.md` (source of truth).*
