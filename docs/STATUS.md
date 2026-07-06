# Engine Status

## Production: Fused Layer Engine (291 tok/s, 3.4 ms/tok) ✅

The fused layer engine runs the full transformer layer in one xclbin call
(QKV→attention→O→GU→SiLU→D on NPU, no CPU attention). Achieves 291 tok/s —
3× the previous v12 engine at 97 tok/s. Engine binary: 38 KB.

- **Binary**: `engine/npu/src/npu_engine_fused.cpp` → 38 KB
- **Models**: Qwen3-0.6B
- **Status**: Verified, production stable

## C++ v12 Engine (97 tok/s) ✅ — Fallback (coherent)

The standalone INT8 GEMM engine is now a fallback path. Coherence fixed July 5
(AIE micro-tiling root cause resolved). Runs at 97 tok/s on Qwen3-0.6B.

- **Binary**: `engine/npu/src/npu_engine_universal.cpp` → 117 KB
- **Status**: Coherent, fallback path

## FLM Proxy (94 tok/s) — Fallback v2

AMD's proprietary runtime. Used as third fallback when fused and v12 are
unavailable.

- **Port**: 52625
- **Models**: Qwen3-0.6B (turbo)
- **Status**: Legacy fallback

## GPU ZINC Engine (22 tok/s) ✅

Vulkan compute shaders via ZINC. Verified coherent output on Bonsai-1.7B-F16.

---

*Last updated: July 6, 2026*
