# Roadmap

> Full detailed roadmap: [docs/roadmap.md](docs/roadmap.md)

## ✅ Phase 1: One Binary to Rule Them All

The **207 KB pure C++ binary** (`zaya_server.cpp`) auto-detects model architecture from headers and dispatches to the fastest available backend — NPU fused, GPU ternary, ROCm HIP, Vulkan, or CPU fallback. No configuration files. No Python. No Rust.

| Engine | Backend | tok/s | Status |
|--------|---------|:-----:|--------|
| GPU 1-bit (llama.cpp ROCm) 🏆 | ROCm HIP | **383** | ✅ |
| NPU fused | XDNA 2 xclbin (32 tiles) | **291** | ✅ |
| GPU ternary | Vulkan GLSL | **307** | ✅ |
| C++ all-5 (auto-detect) | Q4NX header parse | **28** | ✅ |
| Zaya Server (pure C++) | ROCm HIP kernels | **10.6** | ✅ |

## ✅ Phase 2: The Token Router

Intelligent multi-backend routing dispatches every token to the highest-performance backend available, in priority order: NPU fused → GPU ternary → ROCm HIP → CPU. Backends are profiled at startup; per-layer decisions adapt to context length and dispatch overhead.

## ✅ Phase 3: Universal Model Support

Supports all major architectures (Qwen, Llama, Mistral, DeepSeek, BitNet b1.58) and quantization formats (INT8, Q4NX, Q8_0, GGUF, ternary 1.58-bit). Model format is auto-detected from the file header — zero configuration.

## 🔮 Phase 4: What's Next

| Focus | Priority | Notes |
|-------|----------|-------|
| **Expanded GPU backends** | High | CUDA, Vulkan SPIR-V, improved ROCm coverage |
| **More model formats** | Medium | ONNX, AWQ, GPTQ, Safetensors direct load |
| **Windows support** | Medium | Via AMD XDNA 2 driver, MSVC build |
| **Speculative decoding** | Medium | Draft-verify pipeline targeting <50 ms/tok |
| **NPU attention dispatch** | Low | High-context (>32 tokens) via XDNA 2 |
| **Package registries** | Low | Homebrew, Chocolatey, PyPI, npm |

---

*"One Binary to rule them all, and in the hardware bind them."*
