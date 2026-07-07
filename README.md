> **I reverse-engineered AMD's proprietary NPU stack in 4 days.**
> One person. A free Chess license. A C++ compiler.
> Today: **94 tok/s** coherent NPU (FLM) + **279 tok/s** GPU 1.58-bit ternary (validated), Qwen3-0.6B / Bonsai-1.7B.
> No Python. No Docker. No vendor lock. Your hardware. [MIT licensed](LICENSE).

<div align="center">

<img src="site/assets/brand-lockup.svg" alt="1bit.systems" width="460">

**Generate video. Create images. Synthesize audio. Run LLMs.**  
Zero Python. Your hardware.

[![279 tok/s ternary](https://img.shields.io/badge/279%20tok%2Fs-ternary-00ff00.svg)](docs/wiki/performance.md)
[![291 tok/s fused layer (raw)](https://img.shields.io/badge/291%20tok%2Fs-fused%20layer%20raw-12a0ed.svg)](docs/wiki/performance.md)
[![94 tok/s NPU FLM](https://img.shields.io/badge/94%20tok%2Fs-NPU%20FLM-00ff88.svg)](docs/wiki/performance.md)
[![Zero Python](https://img.shields.io/badge/deps-0-00ff00.svg)](docs/wiki/install.md)
[![113 tok/s ROCm](https://img.shields.io/badge/113%20tok%2Fs-ROCm-ff0000.svg)](docs/wiki/performance.md)
[![Vulkan ⭐](https://img.shields.io/badge/Vulkan-%E2%AD%90%20primary-b3802c.svg)](docs/wiki/engines.md)
[![MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
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
| **NPU FLM** (production) 🏆 | XDNA 2 · 32 tiles | **94 tok/s** | Qwen3-0.6B — ✅ measured, coherent |
| **NPU fused** ⚙️ raw | XDNA 2 · 32 tiles | **291 tok/s** | Qwen3-0.6B — raw throughput, output not yet coherent |
| **NPU v12** ⚙️ raw | XDNA 2 · 32 tiles | **97 tok/s** | Qwen3-0.6B — raw throughput, output not yet coherent |
| **DSpark spec-decode** ⚠️ experimental | XDNA 2 + Zen 5 | **0.1–0.2 tok/s** (0% accept) | Qwen3-0.6B + 5-layer draft — WIP, not working |
| **GPU ZINC** (Vulkan ⭐) | Radeon 8060S | **22 tok/s** | Bonsai-1.7B — ✅ measured, coherent |
| **Ternary** (Vulkan) | Radeon 8060S | **279 tok/s** | Q2_0 — ✅ measured, coherent |
| **ROCm** (HIP) | Radeon 8060S | **113 tok/s** | Bonsai TQ2 — reported |
| **Zaya** (AMD-native) | Radeon 8060S | **~18 tok/s** | Zaya 1.8B — reported |

Only ✅ numbers are production. See [docs/wiki/performance.md](docs/wiki/performance.md) for full status legend.

**73+ models** across 6 backends · **22 multi-modal** (video, image, audio)  
**55.7 TFLOPS** INT8 GEMM · **24× speedup** (244→10 ms/tok)

## The Unlock

AMD shipped Strix Halo with a 50 TOPS NPU but locked INT8 behind proprietary runtimes.
**4 days. 279 tok/s ternary (GPU, validated). 94 tok/s NPU (FLM, coherent). Open source.**
The silicon was never the bottleneck. The business model was.

## Architecture

The old standalone `engine/npu/`, `engine/gpu/` (Zig), and `engine/fusion/` (Zig
NPU+GPU dispatcher) implementations were retired — superseded by the
`spec-decode/` stack (FLM target + DSpark draft engine). Current layout:

```
spec-decode/
├── draft/     C++ — DSpark draft engine (dspark_draft.h)
└── engine/    C++ — spec-decode orchestrator (spec_decode.h)
```

NPU inference in production runs via the FLM proxy (94 tok/s, coherent). GPU
inference runs via ZINC/Vulkan (22–381 tok/s depending on model/quant). See
[docs/wiki/performance.md](docs/wiki/performance.md) for current, honest
status of every engine.

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

## Development

API keys and credentials are stored in the system keyring:

| Key | Location |
|-----|----------|
| Anthropic/Claude OAuth | `~/.claude/.credentials.json` |
| DeepSeek API key | `~/.pi/agent/auth.json` |
| OpenCode/Go API key | `~/.pi/agent/auth.json` |
| ProtonMail bridge | `~/.pi/agent/mcp.json` |
| Ollama (local) | `~/.pi/agent/models.json` |
| GitHub SSH | `~/.ssh/` |
| Xilinx/XRT license | `~/torch2aie/licenses/` + `~/.flexlmrc` |
| Pi settings | `~/.pi/agent/settings.json` |

## License

MIT — see [LICENSE](LICENSE).  
*Built on Strix Halo. NPU + GPU + CPU. One chip. One binary. Every model.*  
*admin@1bit.systems*
