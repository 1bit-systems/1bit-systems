# engines

- [Fused Engine (NPU+GPU Hybrid)](fused_engine.md) — Unified NPU+GPU hybrid inference engine dispatching per-layer to XRT (INT8 GEMM) or Vulkan (flash attention) through a single API.  [Engine]
- [NPU Inference Engine (Fused Layer)](npu_engine.md) — Production inference engine on AMD NPU via XRT. 291 tok/s Qwen3-0.6B, 3.4 ms/tok decode (fused layer, 38 KB binary).  [Engine]
- [ROCm Custom Kernel Backend](rocm_backend.md) — Custom HIP kernels for 1-bit and ternary inference on AMD Strix Halo (gfx1151), folded into zaya-llama.cpp as ggml-rocm backend.  [Engine]
- [Unified Serving Daemon](unified_daemon.md) — Single 38 KB binary serving all models via a HTTP API (OpenAI-compatible), proxying through fused NPU layer (291 tok/s) or C++ v12 (97 tok/s) / FLM fallback.  [Engine]
