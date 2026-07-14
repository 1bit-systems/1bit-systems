# Roadmap

## ✅ Phase 1: One Binary to Rule Them All

The **282 KB pure C++ binary** (`zaya_server.cpp`) auto-detects model architecture and loads weights from multiple formats. Backend matrix measured on **AMD Strix Halo** (Ryzen AI Max+ 395):

| Engine | Backend | tok/s | Status |
|--------|---------|:-----:|--------|
| GPU 1-bit (llama.cpp ROCm) 🏆 | ROCm HIP | **383** | ✅ measured |
| GPU ternary | Vulkan GLSL | **307** | ✅ validated |
| NPU FLM (production) | XDNA 2 xclbin | **94** | ✅ validated |
| GPU ROCm HIP (kernels) | ROCm HIP | **113** | 📋 reported |
| NPU v12 | XDNA 2 xclbin | **69** | ⚙️ raw (see notes) |
| C++ all-5 (auto-detect) | Q4NX header parse | **42** | ⚙️ raw (see notes) |
| NPU fused | XDNA 2 xclbin | **291** | ❌ broken (issue #56) |
| GPU Zaya | ROCm HIP | **10.6** | ✅ validated |
| DSpark (spec-decode) | Speculative draft | **0.8** | 🔶 unresolved |

## 🚧 Phase 2: The Token Router

Framework for intelligent multi-backend routing exists but is not yet production-ready:
- **Fused engine** (`engine/fusion/`) produces degenerate output — NPU worker stub returns zeros for many operations (issue #56)
- **Daemon** (`daemon/npu-gpu-cpud.py`) routes by model size but has known bugs in failover paths
- Per-layer NPU/GPU dispatch is scaffolded but not verified on real hardware

## 🚧 Phase 3: Universal Model Support

Supports select architectures (Qwen, Llama, Mistral, DeepSeek, BitNet b1.58) and quantization formats (INT8, Q4NX, Q8_0, GGUF, ternary 1.58-bit). Coverage varies by backend — not all formats work on all backends.

## 🔮 Phase 4: What's Next

| Focus | Priority | Notes |
|-------|----------|-------|
| **Fix NPU fused engine** | Critical | Issue #56 — currently produces token 0 on every policy |
| **Expand GPU backends** | High | CUDA, improved ROCm coverage |
| **More model formats** | Medium | ONNX, AWQ, GPTQ, Safetensors direct load |
| **Windows support** | Medium | Via AMD XDNA 2 driver, MSVC build |
| **Speculative decoding** | Medium | Draft-verify pipeline (issue #115 — draft model now wired) |
| **NPU attention dispatch** | Low | High-context (>32 tokens) via XDNA 2 |
| **Package registries** | Low | Homebrew, Chocolatey, PyPI, npm |

---

*"One Binary to rule them all, and in the hardware bind them."*
