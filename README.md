> **I reverse-engineered AMD's proprietary NPU stack in 4 days.**
> One person. A free Chess license. A C++ compiler. **401 KB (single binary).**
> Today: a DSpark NPU+GPU+CPU inference engine at **572 tok/s** (5.90x spec-decode).
> No Python. No Docker. No vendor lock. Your hardware. [MIT licensed](LICENSE).

<div align="center">

<img src="site/assets/brand-lockup.svg" alt="1bit.systems" width="460">

**Generate video. Create images. Synthesize audio. Run LLMs.**  
401 KB single binary. All engines merged. Zero Python. Your hardware.

[![279 tok/s ternary](https://img.shields.io/badge/279%20tok%2Fs-ternary-00ff00.svg)](docs/wiki/performance.md)
[![572 tok/s DSpark spec-decode](https://img.shields.io/badge/291%20tok%2Fs-fused%20layer-12a0ed.svg)](docs/wiki/performance.md)
[![401 KB single binary](https://img.shields.io/badge/binary-401kb-00ff88.svg)](engine/npu/src/npu_engine_fused.cpp)
[![Zero Python](https://img.shields.io/badge/deps-0-00ff00.svg)](docs/wiki/install.md)
[![113 tok/s ROCm](https://img.shields.io/badge/113%20tok%2Fs-ROCm-ff0000.svg)](docs/wiki/performance.md)
[![Vulkan ⭐](https://img.shields.io/badge/Vulkan-%E2%AD%90%20primary-b3802c.svg)](docs/wiki/engines.md)
[![MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/discord-1bit.systems-00ff88.svg?logo=discord&logoColor=white)](https://discord.gg/dSyV646eBs)
[![clones today](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/bong-water-water-bong/1bit-systems/main/site/daily-clones.json&cacheSeconds=3600)](https://github.com/bong-water-water-bong/1bit-systems/graphs/traffic)
[![clones:stars](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/bong-water-water-bong/1bit-systems/main/site/clones-stars.json)](https://github.com/bong-water-water-bong/1bit-systems/graphs/traffic)
<br>
<sub>deb · snap · docker · AUR · homebrew · ollama</sub>

</div>

---

## Quick Start

```bash
curl -sL https://1bit.systems/npu-install.sh | bash  # 30 seconds
1bit pull qwen3-0.6b
1bit chat                                              # auto-detect NPU → GPU → CPU
```

OpenAI-compatible API:
```bash
1bit serve &
curl localhost:8081/v1/chat/completions \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"Hello!"}]}'
```

Multi-modal:
```bash
video-lora generate --model wan --prompt "cinematic dolly zoom"
video-lora generate --model flux --prompt "portrait, soft lighting"
video-lora generate --model stable-audio --prompt "rain on window"
```

## Performance Highlights

| Engine | Hardware | Speed | Model |
|--------|----------|-------|-------|
| **DSpark** (production) 🏆 | XDNA 2 · 32 tiles | **572 tok/s** | Qwen3-0.6B + 5-layer draft |
| **NPU v12** (fallback) | XDNA 2 · 32 tiles | **97 tok/s** | Qwen3-0.6B (standalone) |
| **GPU ZINC** (Vulkan ⭐) | Radeon 8060S | **22 tok/s** | Bonsai-1.7B |
| **Ternary** (Vulkan) | Radeon 8060S | **279 tok/s** | Q2_0 |
| **ROCm** (HIP) | Radeon 8060S | **113 tok/s** | Bonsai TQ2 |
| **Zaya** (AMD-native) | Radeon 8060S | **~18 tok/s** | Zaya 1.8B |

**73+ models** across 6 backends · **22 multi-modal** (video, image, audio)  
**55.7 TFLOPS** INT8 GEMM · **24× speedup** (244→10 ms/tok)

## The Unlock

AMD shipped Strix Halo with a 50 TOPS NPU but locked INT8 behind proprietary runtimes.
**4 days. 401 KB (single binary). 279 tok/s ternary. Open source.**
The silicon was never the bottleneck. The business model was.

## Architecture

```
engine/
├── npu/       C++23 — NPU (XDNA 2, INT8 xclbins)
├── gpu/       Zig — GPU (Vulkan ⭐ / ROCm / CUDA / Metal)
└── dspark/    C++ — DSpark spec-decode (5.90x), 8 policies
```

One unified KV cache (H2O eviction, RadixAttention). One serving API.
Every model auto-detected. Fused Engine dispatches per-layer to the fastest backend.

## More

| Topic | Link |
|-------|------|
| 🚀 **Install & Build** | [docs/wiki/install.md](docs/wiki/install.md) |
| 📊 **Benchmarks & Models** | [docs/wiki/performance.md](docs/wiki/performance.md) |
| ⚙️ **Engine Docs** | [docs/wiki/engines.md](docs/wiki/engines.md) |
| 🌐 **Ecosystem & Clients** | [docs/wiki/ecosystem.md](docs/wiki/ecosystem.md) |
| ❓ **FAQ** | [docs/wiki/faq.md](docs/wiki/faq.md) |
| 📖 **Full Wiki** | [docs/wiki/landing.md](docs/wiki/landing.md) |
| 📜 **Journey (audit trail)** | [docs/journey.md](docs/journey.md) |
| 🤝 **Contributing** | [CONTRIBUTING.md](CONTRIBUTING.md) |

## License

MIT — see [LICENSE](LICENSE).  
*Built on Strix Halo. NPU + GPU + CPU. One chip. One binary. Every model.*  
*admin@1bit.systems*
