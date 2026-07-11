# Changelog

## v1.0.0 (2026-07-01) — Initial Release

### NPU Engine
- INT8 inference on AMD Strix Halo XDNA 2 NPU
- 4-live GEMM contexts + 4 attention contexts, zero swapping
- 244 ms/tok (4.1 tok/s) on Qwen3-0.6B with diverse token output
- Batched prefill: 20 ms/tok (13.5× faster than single-token)
- Raw INT8 GEMM: 55.7 TFLOPS D projection (exceeds 50 TOPS rating)
- Pure C++23, single binary, zero Python at runtime

### GPU Engine (ZINC)
- Vulkan 1.3 compute shaders on Radeon 8060S (gfx1151)
- 27 µs/tok decode on Qwen3.5-9B Q4_K
- CUDA + Metal backends included
- Zig build system, single `zinc` binary

### 1-bit GPU
- Bonsai-1.7B IQ1_S: 281 tok/s, 385 MB
- llama.cpp Q2_0 validation patched for Strix Halo
- Models verified on-disk at /home/bcloud/models/bonsai-1.7b/

### Infrastructure
- Landing page at https://1bit.systems (Cloudflare Pages)
- Live NPU dashboard at https://1bit.systems/live.html
- PR-Agent: Qodo + OpenCode GLM-5.2 auto-review
- Debian packaging template
- Full audit trail: docs/journey.md (1,184+ lines, 15 updates)

### Key Fixes
- INT8 K-interleaving (dataReuse on ObjectFifo DMA)
- BFP16 precision collapse (switched to INT8 quantization)
- NPU2 multi-context support (8+ simultaneous hw_contexts)
- NaN accumulation (safe softmax + 4-layer error containment)
- Chess license activation (31.4 TFLOPS config2 verified)
