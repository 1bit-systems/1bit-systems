# Performance

**Source of truth:** [`engine/npu/BENCHMARKS.md`](../engine/npu/BENCHMARKS.md) (July 4, 2026 refresh)

| Engine | Speed | Model |
|--------|-------|-------|
| NPU FLM | **94 tok/s** | Qwen3-0.6B |
| NPU v12 (C++) | **97 tok/s** | Qwen3-0.6B |
| GPU Ternary (Vulkan) | **279 tok/s** | Q2_0 |
| GPU ROCm (HIP) | **113 tok/s** | Bonsai TQ2 |
| GPU ZINC (Vulkan) | **22 tok/s** | Bonsai-1.7B |

**73+ models** across 6 backends → [`docs/models.md`](../models.md)  
**Fused dispatch** (8 policies) → [`engine/fusion/`](../engine/fusion/)
