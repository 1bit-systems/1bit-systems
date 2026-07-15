# Roadmap

## Phase 1: One Binary to Rule Them All

The **282 KB pure C++ binary** (zaya_server.cpp) auto-detects model architecture and loads weights from multiple formats. Backend matrix measured on **AMD Strix Halo** (Ryzen AI Max+ 395):

| Engine | Backend | tok/s | Status |
|--------|---------|:-----:|--------|
| Q1 GEMV (native) | ROCm HIP | **417** | this build |
| GPU ternary (ZINC) | Vulkan GLSL | **318** | bench_zinc_vulkan.sh |
| NPU v12 | XDNA 2 xclbin | **69** | validated |
| GPU ROCm HIP (kernels) | ROCm HIP | **64** | bench_rocm_hip.sh |
| NPU FLM (production) | XDNA 2 xclbin | **57** | validated |
| C++ all-5 (auto-detect) | Q4NX header parse | **42** | raw |
| GPU Zaya | ROCm HIP | **10.6** | validated |
| NPU fused | XDNA 2 xclbin | -- | broken (issue #56) |
| DSpark (spec-decode) | Speculative draft | **0.8** | unresolved |

> Correction 2026-07-15: The old 373-383 tok/s claim for GPU 1-bit measured prompt eval (prefill) speed, not decode. Real decode is 229 tok/s via llama.cpp ROCm. Our native Q1 GEMV kernel (417 tok/s) is the correct headline number.

## Phase 2: The Token Router

Framework for intelligent multi-backend routing exists but is not yet production-ready:
- Fused engine (engine/fusion/) produces degenerate output -- NPU worker stub returns zeros (issue #56)
- Daemon routes by model size but has known bugs in failover paths
- Per-layer NPU/GPU dispatch is scaffolded but not verified on real hardware

## Phase 3: Universal Model Support

Supports select architectures (Qwen, Llama, Mistral, DeepSeek, BitNet b1.58) and quantization formats (INT8, Q4NX, Q8_0, GGUF, ternary 1.58-bit). Coverage varies by backend.

## Phase 4: What's Next

| Focus | Priority | Notes |
|-------|----------|-------|
| Fix NPU fused engine | Critical | Issue #56 -- produces token 0 on every policy |
| Expand GPU backends | High | CUDA, improved ROCm coverage |
| More model formats | Medium | ONNX, AWQ, GPTQ, Safetensors direct load |
| Windows support | Medium | Via AMD XDNA 2 driver, MSVC build |
| Speculative decoding | Medium | Draft-verify pipeline (issue #115) |
| NPU attention dispatch | Low | High-context via XDNA 2 |
| Package registries | Low | Homebrew, Chocolatey, PyPI, npm |