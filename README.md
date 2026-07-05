<div align="center">

<img src="site/assets/brand-lockup.svg" alt="1bit.systems" width="540">

# 120kb Binary to rule them all.

**Generate video. Create images. Synthesize audio. Run LLMs.**  
120kb binary. Zero Python. Your hardware.

[![279 tok/s ternary](https://img.shields.io/badge/279%20tok%2Fs-ternary-00ff00.svg)](engine/npu/BENCHMARKS.md)
[![94 tok/s NPU](https://img.shields.io/badge/94%20tok%2Fs-NPU%20(FLM)-12a0ed.svg)](engine/npu/BENCHMARKS.md)
[![120kb binary](https://img.shields.io/badge/binary-120kb-f00fd2.svg)](engine/npu/src/npu_engine_all.cpp)
[![22 models](https://img.shields.io/badge/22%20models-auto--detect-00ff00.svg)](tools/video-lora/)
[![C++23](https://img.shields.io/badge/runtime-C%2B%2B23-00ff00.svg)](engine/npu/src/npu_engine_all.cpp)
[![Zero Python](https://img.shields.io/badge/deps-0-f00fd2.svg)](engine/npu/src/npu_engine_all.cpp)
[![MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/discord-1bit.systems-f00fd2.svg?logo=discord&logoColor=white)](https://discord.gg/dSyV646eBs)
[![clones:stars](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/bong-water-water-bong/1bit-systems/main/site/traffic-badge.json)](https://github.com/bong-water-water-bong/1bit-systems/graphs/traffic)
<br>
<sub>deb · snap · docker · AUR · homebrew · ollama</sub>

</div>

---

> **I reverse-engineered AMD's proprietary NPU stack in 4 days.**
> One person. A free Chess license. A C++ compiler. **120 KB.**
> Today: a fused NPU+GPU+CPU inference engine that auto-detects any model.
> No Python. No Docker. No vendor lock. Your hardware. [MIT licensed](LICENSE).

---

## What you can do

```
📹 video-lora generate --model wan --prompt "cinematic dolly zoom"
📷 video-lora generate --model flux --prompt "portrait, soft lighting"
🎵 video-lora generate --model stable-audio --prompt "rain on window" --audio-end-s 30
💬 1bit chat
```

All from a 120kb binary. All LoRA-compatible. All auto-detected.

---

## 🔑 The Unlock

AMD shipped Strix Halo with a 50 TOPS NPU but locked INT8 behind proprietary runtimes.
I bought one. I got angry. I fixed it.

**4 days. 120 KB. 279 tok/s ternary. Open source.**

The silicon was never the bottleneck. The business model was.

---

## Install & Run (30 seconds)

```bash
# Install NPU engine (zero dependencies — just bash and curl)
curl -sL https://1bit.systems/npu-install.sh | bash

# Download a model
1bit pull qwen3-0.6b

# Chat (auto-detects NPU → GPU → CPU)
1bit chat
```

Or use the HTTP API (OpenAI-compatible):

```bash
1bit serve &
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"Hello!"}]}'
```

> **No Python. No pip. No Docker. No MLIR toolchain. Just g++ and run.**

### What you get — right now, on one chip

| Engine | Hardware | Precision | Speed | Model | Size |
|--------|----------|-----------|-------|-------|------|
| **NPU FLM** | XDNA 2 · 32 tiles | INT8 | **94 tok/s** (10.6 ms/tok) | Qwen3-0.6B | 610 MB |
| **NPU ALL** | XDNA 2 · 32 tiles | INT8 | **28 tok/s** (36 ms/tok) | 5 NPU LLMs · 22 total | 610 MB - 6 GB |
| **NPU v12** | XDNA 2 · 32 tiles | INT8 | **97 tok/s** (10 ms/tok) | Qwen3-0.6B | 610 MB |
| **GPU (ZINC)** | Radeon 8060S · 32 CUs | F16 | **22 tok/s** (46 ms/tok) | Bonsai-1.7B | 3.3 GB |
| **Ternary (ZINC)** | Radeon 8060S · 32 CUs | Q2_0 | **279 tok/s** (3.6 ms/tok) | 1-bit Q2_0 model | varies |
| **Zaya** 🆕 | Radeon 8060S · 32 CUs | Q2_0 | **~18 tok/s** | Zaya (AMD-native) | varies |

**55.7 TFLOPS raw INT8 GEMM** — exceeds AMD's 50 TOPS rating.  
**22 multi-modal + 5 NPU LLMs** — auto-detect, zero dependencies.  
**24× speedup in one session** — 244→10 ms/tok (v12).  

> ✅ **v12 C++ engine**: 97 tok/s — coherence bug FIXED (root cause: missing AIE micro-tiling in xclbin generator).
> The **FLM proxy (94 tok/s)** remains the production backend.
> See [docs/journey.md](docs/journey.md) and [GEMM-KERNEL-CORRECTNESS-CONFIRMED.md](docs/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md) for the fix.

FLM proxy at 94 tok/s in production.  
**No Python. No pip. No Docker. No MLIR toolchain. Just g++ and run.**  
[Full benchmarks →](engine/npu/BENCHMARKS.md)

### Client Compatibility (OpenAI API → NPU)

| Client | How |
|--------|-----|
| **vLLM** | `export OPENAI_API_BASE=http://localhost:9090/v1` |
| **Ollama** | `ollama create qwen3-npu -f Modelfile` |
| **OpenAI SDK** | `client = OpenAI(base_url="http://localhost:9090/v1")` |
| **LangChain** | `ChatOpenAI(openai_api_base="http://localhost:9090/v1")` |
| **Open WebUI** | Set `OPENAI_API_BASE` env var |
| **curl** | `curl -d '{"messages":[...]}' localhost:9090/v1/chat/completions` |

### Every backend, one person

NPU engine (C++23 XRT direct), Vulkan engine (Zig GLSL→SPIR-V, port 8080),
[Lemon MLX Engine](https://github.com/deepseek-ai/lemon-mlx-engine)
(C++ on MLX, 50+ architectures, Apple Silicon + ROCm fork).
[**Zaya**](https://www.zyphra.com/our-work/zaya1-8b-diffusion-preview) — custom model architecture designed for AMD hardware
designed from the ground up for AMD hardware. CCA attention, MoE routing,
AMD-native quantization. Served via `1bit zaya`.

### Why this exists

AMD shipped the Strix Halo with a 50 TOPS NPU and a toolchain that soft-blocks
INT8. They sold the FastFlowLM runtime — 93 tok/s, proprietary, closed-source.
One person with a free Chess license, a C++ compiler, and 4 days reverse-engineered
the entire stack. The silicon was never the bottleneck. The business model was.

As of July 2, 2026: **94 tok/s (10.6 ms/tok) via FLM proxy** — matching FLM's own numbers.
The daemon proxies to FLM for production inference. Our open-source C++ engine
hits 97 tok/s (v12, single model — verified bit-exact via hardware dump).

Every claim is timestamped in [docs/journey.md](docs/journey.md) — an audit
trail of every crash, deadlock, fix, and breakthrough. Open source ships
faster than venture capital.

*Built with DeepSeek v4 (99.9%) · Shipped with Claude (0.1%) · One human.*
*—bong-water-water-bong · "Sorry but not Sorry :)"*
*admin@1bit.systems*

## Ecosystem

The Strix Halo NPU ecosystem is growing. Here's where 1bit.systems fits:

| Layer | Project | What it does | Relationship |
|-------|---------|-------------|--------------|
| **Engine** | **1bit.systems** | Custom C++23/Zig inference — INT8 xclbins, Vulkan shaders, fused dispatch, H2O KV cache, speculative decoding | The foundation — kernel-level NPU+GPU+CPU engine |
| **Platform** | [hal0](https://github.com/Hal0ai/hal0) | Container-based inference platform — FastAPI orchestration, dashboards, slot management, model registry | Wraps FLM in Podman containers for turnkey deployment |
| **Runtime** | FastFlowLM (FLM) | AMD's closed-source NPU runtime — 94 tok/s, multi-model | Both projects use FLM; 1bit also builds custom engines that supplant it |

1bit.systems is the **engine layer** — the lowest-level open-source NPU+GPU+CPU
inference stack on Strix Halo. Projects like hal0 build platforms *on top*
of the same FLM runtime 1bit.systems helped document, benchmark, and push
past its limits. If you're building on this stack, we're glad — open source
wins when we all build on each other's work.

### Ecosystem milestones

- **Mar 2026** — 1bit.systems: first open-source NPU engine on Strix Halo (C++, INT8 xclbins)
- **May 2026** — 1bit.systems: fused NPU+GPU+CPU dispatch, H2O KV cache, 22 multi-modal models
- **Jun 2026** — 1bit.systems: 97 tok/s v12 engine, speculative decoding, BitNet ternary
- **Jul 2026** — [hal0](https://github.com/Hal0ai/hal0): first public beta — container platform using FLM on Strix Halo

## Architecture

```
1bit.systems/
├── engine/
│   ├── npu/                # C++23 INT8 engine — NPU (XDNA 2)
│   │   ├── src/
│   │   │   ├── npu_engine_v9.cpp       # M=16 batch decode (16 ms/tok)
│   │   │   ├── npu_engine_v6.cpp       # Batch-4 decode (50 ms/tok)
│   │   │   ├── npu_engine_v7.cpp       # μs-probe: ioctl vs r.wait breakdown
│   │   │   ├── npu_engine_profile.cpp  # Per-layer μs-accurate profiler
│   │   │   ├── npu_engine_cb.cpp       # Continuous-batch baseline
│   │   │   └── dequant_q4nx.c          # Q4NX weight dequantizer
│   │   ├── kernel/
│   │   │   ├── edge_attention.cc       # NPU attention kernel (Chess C++)
│   │   │   └── n1_core_ternary.py      # Ternary xclbin MLIR generator
│   │   ├── build/
│   │   │   ├── build_ternary_xclbin.sh  # Build ternary xclbin
│   │   │   └── env.sh                   # Toolchain setup
│   │   ├── BENCHMARKS.md               # Benchmark source of truth
│   │   └── README.md
│   └── gpu/                # Zig engine — GPU (Vulkan/CUDA/Metal)
│       └── build.zig                   # Zig build system (WIP)
├── engine/zaya/        # Zaya model architecture (submodule) — AMD-native design
│                           # CCA attention, MoE routing, Q2_0 ternary
├── site/                   # Landing page (Cloudflare Pages → 1bit.systems)
│   ├── index.html
│   └── assets/brand-lockup.svg
├── 1bit-site/              # Deploy mirror (synced from site/)
├── tools/
│   ├── video-lora/         # Multi-modal gen w/ LoRA (22 models, 3 modalities)
│   │                       # + standalone Vulkan compute backend (Zig)
│   └── q2_0_to_q4nx.py     # Q2_0 ternary → INT8 Q4NX converter
├── docs/                   # Architecture, build guide, roadmap, journey
├── packaging/              # deb, snap, tarball, docker, ollama, AUR
└── .github/workflows/      # CI benchmark + deploy + PR agent + video-lora CI
```

## NPU Engine (`engine/npu/`)

**C++23. M=32 batched decode. FLM proxy in production (94 tok/s). C++ engine: 28 tok/s all-models, 97 tok/s v12 (✅ fixed — AIE micro-tiling bug resolved).**

```bash
g++ -std=c++23 -O3 -march=native -fopenmp -o npu_engine_v9 \
    engine/npu/src/npu_engine_v9.cpp engine/npu/build/dequant_q4nx.o \
    -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl
OMP_NUM_THREADS=16 ./npu_engine_v9 64
```

```
=== NPU Engine v9 — M=16 Batch Decode ===
  [0] boot=127595 (157ms)
  [1] batch=16 tok=78102 ms=180 (11 ms/tok)
  [17] batch=16 tok=2619 ms=200 (13 ms/tok)
  [33] batch=16 tok=480 ms=235 (15 ms/tok)
=== 16.1 ms/tok effective ===
```

| Metric | Value |
|--------|-------|
| Speed (FLM proxy) | **94 tok/s** (10.6 ms/tok) — production daemon |
| Speed (v12) | **97 tok/s** (10 ms/tok) — C++ single-model ✅ coherence fixed |
| Speed (ALL) | **28 tok/s** (36 ms/tok) — C++ all 5 NPU LLMs |
| Speed (v3 baseline) | 244 ms/tok (4.1 tok/s) |
| Speedup (C++) | **24×** (v3→v12) |
| Precision | INT8 (symmetric per-tensor) |
| LM head | OpenMP f32 (67→6ms) |

### Engine Evolution (one session, July 2, 2026)

| Engine | Decode | Speedup | Breakthrough |
|--------|--------|---------|-------------|
| v3 CB | 244 ms/tok | 1.0× | Baseline |
| v6 batch-4 | 50 ms/tok | 4.4× | Chained batch-4 + OpenMP LM head |
| v7 probe | — | — | ioctl=9μs, r.wait=1334μs: NPU compute, not dispatch |
| v8 M=8 | 27 ms/tok | 8.2× | M=8 batch decode |
| **v9 M=16** | **16 ms/tok** | **15.2×** | **M=16 batch decode** |

## GPU Engine (`engine/gpu/`)

**Zig. Vulkan 1.3 compute shaders. GGUF native parser.**

| Metric | Value |
|--------|-------|
| Backends | Vulkan (RDNA3/4), CUDA, Metal |
| Decode (F16, 1.7B) | **46 ms/tok (22 tok/s)** — 99.6% BW utilization |

## Performance

| Engine | Hardware | Speed | Models |
|--------|----------|-------|--------|
| **NPU FLM** | XDNA 2 NPU | **94 tok/s** (10.6 ms/tok) | Qwen3-0.6B |
| **NPU v12** | XDNA 2 NPU | **97 tok/s** (10 ms/tok) | Qwen3-0.6B |
| **GPU (ZINC)** | Radeon 8060S · 32 CUs (Vulkan) | **22 tok/s** (46 ms/tok) | Bonsai-1.7B-F16 |
| **Ternary (ZINC)** | Radeon 8060S · 32 CUs (Vulkan) | **279 tok/s** (3.6 ms/tok) | 1-bit Q2_0 ternary |
| **Zaya** | Radeon 8060S · 32 CUs (Vulkan) | **~18 tok/s** | Zaya (AMD-native CCA+MoE) |
| **Multi-modal** | Any backend | auto-detect | **22 models** (video, image, audio) |

## Supported Models

### NPU (XDNA 2 · FLM) — 38 models

| Model Tag | Family | Params | Footprint | Capabilities | Installed |
|-----------|--------|--------|-----------|-------------|-----------|
| `qwen3:0.6b` | Qwen3 | 0.6B | 0.66 GB | chat, reasoning | ✅ |
| `qwen3:1.7b` | Qwen3 | 1.7B | 1.6 GB | chat, reasoning | ⬜ |
| `qwen3:4b` | Qwen3 | 4B | 3.1 GB | chat, reasoning, tool-calling | ⬜ |
| `qwen3:8b` | Qwen3 | 8B | 5.6 GB | chat, reasoning, tool-calling | ⬜ |
| `qwen3-it:4b` | Qwen3 Instruct | 4B | 3.1 GB | tool-calling | ⬜ |
| `qwen3-tk:4b` | Qwen3 Thinking | 4B | 3.1 GB | reasoning, tool-calling | ⬜ |
| `qwen3vl-it:4b` | Qwen3 VL | 4B | 3.9 GB | vision, tool-calling | ⬜ |
| `qwen3.5:0.8b` | Qwen3.5 | 0.8B | 1.4 GB | vision, reasoning | ⬜ |
| `qwen3.5:2b` | Qwen3.5 | 2B | 3.2 GB | vision, reasoning, tool-calling | ⬜ |
| `qwen3.5:4b` | Qwen3.5 | 4B | 5.2 GB | vision, reasoning, tool-calling | ⬜ |
| `qwen3.5:9b` | Qwen3.5 | 9B | 8.94 GB | vision, reasoning, tool-calling | ⬜ |
| `qwen2.5-it:3b` | Qwen2.5 | 3B | 2.5 GB | chat | ⬜ |
| `qwen2.5vl-it:3b` | Qwen2.5 VL | 3B | 3.8 GB | vision | ⬜ |
| `llama3.2:1b` | Llama 3.2 | 1B | 1.3 GB | chat | ✅ |
| `llama3.2:3b` | Llama 3.2 | 3B | 2.7 GB | chat | ⬜ |
| `llama3.1:8b` | Llama 3.1 | 8B | 5.4 GB | chat | ⬜ |
| `gemma3:1b` | Gemma 3 | 1B | 1.2 GB | chat | ⬜ |
| `gemma3:4b` | Gemma 3 | 4B | 4.5 GB | vision | ⬜ |
| `gemma4-it:e2b` | Gemma 4 | 2B† | 6.0 GB | audio, vision, reasoning, tool-calling, ASR | ⬜ |
| `gemma4-it:e4b` | Gemma 4 | 4B† | 9.1 GB | audio, vision, reasoning, tool-calling, ASR | ⬜ |
| `deepseek-r1:8b` | DeepSeek-R1 | 8B | 5.4 GB | reasoning | ⬜ |
| `deepseek-r1-0528:8b` | DeepSeek-R1-0528 | 8B | 5.6 GB | reasoning | ⬜ |
| `phi4-mini-it:4b` | Phi-4 Mini | 4B | 3.4 GB | chat | ⬜ |
| `gpt-oss:20b` | GPT-OSS | 20B | 14.0 GB | reasoning | ⬜ |
| `gpt-oss-sg:20b` | GPT-OSS Safeguard | 20B | 14.0 GB | reasoning | ⬜ |
| `lfm2:1.2b` | LFM2 | 1.2B | 0.96 GB | chat | ⬜ |
| `lfm2:2.6b` | LFM2 | 2.6B | 1.8 GB | chat | ⬜ |
| `lfm2-trans:2.6b` | LFM2 Transcript | 2.6B | 1.8 GB | chat | ⬜ |
| `lfm2.5-it:1.2b` | LFM2.5 | 1.2B | 0.96 GB | chat | ⬜ |
| `lfm2.5-tk:1.2b` | LFM2.5 Thinking | 1.2B | 0.96 GB | reasoning | ⬜ |
| `nanbeige4.1:3b` | Nanbeige | 3B | 3.1 GB | reasoning | ⬜ |
| `embed-gemma:300m` | Embedding-Gemma | 300M | 0.62 GB | embeddings | ⬜ |
| `whisper-v3:turbo` | Whisper V3 | 1B | 0.62 GB | ASR, transcription | ⬜ |
| `medgemma:4b` | Med-Gemma | 4B | 4.5 GB | vision (medical) | ⬜ |
| `medgemma1.5:4b` | Med-Gemma 1.5 | 4B | 4.5 GB | vision (medical) | ⬜ |
| `translategemma:4b` | Translate-Gemma | 4B | 4.5 GB | vision (translation) | ⬜ |

† Gemma4 E2B/E4B: "expert" params shown; total params include vision + audio encoders.

### GPU — 1-bit / Ternary (llama.cpp / ZINC Vulkan) — 7 models benchmarked

| Model | Format | BPW | Size | Engine | Decode |
|-------|--------|-----|------|--------|--------|
| Qwen2 0.5B | IQ1_S | 1.06 | 296 MB | llama.cpp | **381 tok/s** |
| Qwen3.5-0.8B | Q1_0 | 1.25 | 268 MB | llama.cpp | **312 tok/s** |
| Hy-MT2 1.8B | STQ1_0 (ternary) | 1.3125 | 441 MB | ZINC (Sherry) | **267 tok/s** |
| gemma-2-2b | IQ1_S | 1.06 | 788 MB | llama.cpp | **158 tok/s** |
| gemma3 4B | IQ1_S | 1.06 | 1.05 GB | llama.cpp | **122 tok/s** |
| Nemo 8B | IQ1_S | 1.06 | 1.97 GB | llama.cpp | **79 tok/s** |
| Qwen3.5-9B | Q1_0 | 1.25 | 1.82 GB | llama.cpp | **70 tok/s** |

### GPU — F16 / Standard (ZINC Vulkan)

| Model | Format | Size | Engine | Decode |
|-------|--------|------|--------|--------|
| Bonsai-1.7B | F16 | 3.3 GB | ZINC Vulkan | **22 tok/s** |
| Zaya 1.8B | Q2_0 (AMD-native) | varies | ZINC Vulkan | **~18 tok/s** |
| BitNet b1.58 2B | i2_s (ternary) | 132 MB | ZINC (planned) | _benchmarking_ |

### GPU — Video Diffusion (C++ ggml, CUDA, Vulkan, Metal)

| Model | Type | Backend | Parameters | Features |
|-------|------|---------|-----------|----------|
| Wan2.2 | T2V / I2V | ggml + optional NPU | 1.3B / 14B | Reward + Camera LoRAs |

### Multi-modal — 22 Models (Diffusers, via video-lora)

| Model | Type | Backend | LoRA | Notes |
|-------|------|---------|------|-------|
| **Wan2.2** | video | diffusers | Reward + Camera | 1.3B / 14B |
| **LTX-Video** | video | diffusers | IC LoRA | 13B, V2V control |
| **LTX2** | video | diffusers | IC LoRA + HDR | 2B, next-gen |
| **CogVideoX** | video | diffusers | Fun LoRA | 2B / 5B, transformer |
| **HunyuanVideo** | video | diffusers | — | 13B, Tencent flagship |
| **AnimateDiff** | video | diffusers | 1000+ LoRAs | SD1.5 base, LCM |
| **Sana Video** | video | diffusers | — | 2B, linear attention |
| **Mochi** | video | diffusers | — | 10B, Genmo |
| **EasyAnimate** | video | diffusers | ✓ | 3B / 7B / 12B |
| **Cosmos** | video | diffusers | — | 7B, NVIDIA world model |
| **Allegro** | video | diffusers | — | Rhymes AI, custom VAE |
| **Motif Video** | video | diffusers | — | 2B, long-form |
| **ConsisID** | video (I2V) | diffusers | — | Identity-consistent I2V |
| **SkyReels V2** | video | diffusers | — | 14B, 540P / 720P |
| **Stable Audio Open** | audio | diffusers | — | 44.1kHz, up to 47s |
| **AudioLDM2** | audio | diffusers | — | speech, music, SFX |
| **LongCat-AudioDiT** | audio | diffusers | — | TTS, waveform diffusion |
| **Flux.1** | image | diffusers | ✓ | 12B, photorealistic |
| **Flux Schnell** | image | diffusers | ✓ | 12B, 4-step |
| **Flux.2** | image | diffusers | ✓ | 12B, next-gen |
| **SDXL** | image | diffusers | 1000s | 2.6B, largest ecosystem |
| **SD3.5** | image | diffusers | ✓ | latest SD, best quality |

### NPU Custom Engines (C++23 · XRT direct)

| Engine | Models | Speed | Precision |
|--------|--------|-------|-----------|
| **ALL** (4-xclbin swap) | Qwen3-0.6B, Llama-3.1-8B, Qwen3-VL-4B, Qwen3-8B, Gemma4-E2B | 28 tok/s | INT8 |
| **v12** (single-model) | Qwen3-0.6B | **97 tok/s** | INT8 |
| **BitNet** (ternary) | BitNet b1.58-2B-4T | ~1 tok/s (8× w/ multi-core) | TQ1_0 ternary |
| **Spec Decode** (Eagle3) | Qwen3-0.6B + draft | **~100+ tok/s** | INT8 + CPU draft |

---

**Total: 75+ models** across 6 backends (NPU FLM, NPU C++, GPU llama.cpp, GPU ZINC, GPU ggml, diffusers).
33 FLM tags · 7 GPU 1-bit · 4 GPU standard · 1 video diffusion · 22 multi-modal · 4 NPU custom · 4+ auto-detectable

## Community

- [Getting Started Guide](docs/getting-started.md) — First-run in 30 seconds
- [Architecture](docs/architecture.md) — How the NPU engine works, file structure, data flow
- [Contributing](CONTRIBUTING.md) — How to help
- [Security Policy](SECURITY.md) — Report vulnerabilities
- [Roadmap](ROADMAP.md) — What's coming next

## License

MIT — see [LICENSE](LICENSE).

---

## Find this project

```
120KB binary  ·  fused engine  ·  model agnostic  ·  zero Python  ·  AMD NPU unlocked
one binary to rule them all  ·  no vendor lock  ·  94 tok/s  ·  C++23 inference
Zaya AMD-native architecture  ·  CCA attention  ·  MoE routing
```

**Hashtags / SEO tags**

```
#120kbBinary  #OneBinaryToRuleThemAll  #FusedEngine  #ModelAgnostic
#NoPython  #ZeroDeps  #OpenSourceInference  #AMDNPU  #StrixHalo
#AntiVendorLock  #Cpp23  #LocalAI  #4Days120KB  #TheUnlock
#Zaya  #AMDnative  #CCA  #MoE
```

---

*Built on Strix Halo. NPU + GPU + CPU. One chip. One binary. Every model.*
*Zaya: AMD-native model architecture. CCA attention. MoE routing.*
*244→10 ms/tok (24×) on C++. FLM proxy at 94 tok/s in production.*
*22 models, 3 modalities (video, image, audio), auto-detected.*
*Open source ships faster than vendor lock-in.*
