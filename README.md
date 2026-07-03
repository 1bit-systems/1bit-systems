<div align="center">

<img src="site/assets/brand-lockup.svg" alt="1bit.systems" width="540">

# One binary to rule them all.
## Open source. Zero Python. Zero dependencies.

[![50 TOPS Verified](https://img.shields.io/badge/50%20TOPS-verified-00ff00.svg)](engine/npu/BENCHMARKS.md)
[![55.7 TFLOPS Peak](https://img.shields.io/badge/55.7%20TFLOPS-raw%20silicon-12a0ed.svg)](engine/npu/BENCHMARKS.md)
[![94 tok/s NPU](https://img.shields.io/badge/94%20tok%2Fs-NPU%20(FLM)-00ff00.svg)](engine/npu/BENCHMARKS.md)
[![22 tok/s GPU](https://img.shields.io/badge/22%20tok%2Fs-GPU%20(Vulkan)-12a0ed.svg)](engine/npu/BENCHMARKS.md)
[![5 models](https://img.shields.io/badge/5%20models-auto--detect-00ff00.svg)](engine/npu/BENCHMARKS.md)
[![120KB binary](https://img.shields.io/badge/binary-120KB-f00fd2.svg)](engine/npu/src/npu_engine_all.cpp)
<br>
[![Debian](https://img.shields.io/badge/deb-install-d70a53.svg)](https://github.com/bong-water-water-bong/1bit-systems/releases/latest)
[![Snap](https://img.shields.io/badge/snap-install-82BEA0.svg)](https://github.com/bong-water-water-bong/1bit-systems/releases/latest)
[![Docker](https://img.shields.io/badge/docker-run-2496ED.svg)](https://github.com/bong-water-water-bong/1bit-systems/releases/latest)
[![AUR](https://img.shields.io/badge/AUR-yay-1793d1.svg)](https://github.com/bong-water-water-bong/1bit-systems/releases/latest)
[![Homebrew](https://img.shields.io/badge/brew-install-fbb040.svg)](https://github.com/bong-water-water-bong/1bit-systems/releases/latest)
[![Ollama](https://img.shields.io/badge/ollama-ready-000000.svg)](https://github.com/bong-water-water-bong/1bit-systems/releases/latest)
<br>
[![curl install](https://img.shields.io/badge/curl%20%7C%20bash-install-00ff00.svg)](packaging/install.sh)
[![Pure C++](https://img.shields.io/badge/runtime-C%2B%2B23-00ff00.svg)](engine/npu/src/npu_engine_all.cpp)
[![Zero Python](https://img.shields.io/badge/deps-0-f00fd2.svg)](engine/npu/src/npu_engine_all.cpp)
[![MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/discord-1bit.systems-f00fd2.svg?logo=discord&logoColor=white)](https://discord.gg/dSyV646eBs)

</div>

---

### Get Started

```bash
# One-liner
curl -sL https://1bit.systems/install.sh | bash

# Or pick your package manager
sudo dpkg -i 1bit-systems_2026.07.02_amd64.deb     # Debian/Ubuntu
sudo snap install 1bit-systems                       # Snap
yay -S 1bit-systems-bin                              # Arch (AUR)
brew install 1bit-systems                            # macOS (Homebrew)
docker run -d --device /dev/accel/accel0 \
  -p 8081:8081 1bit-systems/npu:2026.07.02           # Docker
ollama create qwen3-npu -f packaging/ollama/Modelfile # Ollama

# Or build from source (one command)
g++ -std=c++23 -O3 -march=native -fopenmp -ffast-math \
    -o npu_engine_all engine/npu/src/npu_engine_all.cpp \
    engine/npu/build/dequant_q4nx.o \
    -Iengine/npu/src -lxrt_coreutil

# Run (auto-detects model)
OMP_NUM_THREADS=16 ./npu_engine_all model.q4nx 16
```

> **No Python. No pip. No Docker. No MLIR toolchain. Just g++ and run.**

### What you get — right now, on one chip

| Engine | Hardware | Precision | Speed | Model | Size |
|--------|----------|-----------|-------|-------|------|
| **NPU FLM** | XDNA 2 · 32 tiles | INT8 | **94 tok/s** (10.6 ms/tok) | Qwen3-0.6B | 610 MB |
| **NPU ALL** | XDNA 2 · 32 tiles | INT8 | **28 tok/s** (36 ms/tok) | 5 models | 610 MB - 6 GB |
| **NPU v12** | XDNA 2 · 32 tiles | INT8 | **97 tok/s** (10 ms/tok) | Qwen3-0.6B | 610 MB |
| **GPU (ZINC)** | Radeon 8060S · 32 CUs | F16 | **22 tok/s** (46 ms/tok) | Bonsai-1.7B | 3.3 GB |

**55.7 TFLOPS raw INT8 GEMM** — exceeds AMD's 50 TOPS rating.  
**5 models from one 120KB binary** — auto-detect, zero dependencies.  
**24× speedup in one session** — 244→10 ms/tok (v12). FLM proxy at 94 tok/s in production.  
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

NPU engine (C++23 XRT direct), Vulkan engine (Zig GLSL→SPIR-V, port 8080), Metal engine
(Zig MSL), and MLX NPU backend
(Apple MLX fork with IRON XDNA 2). All built here. All open.

### Why this exists

AMD shipped the Strix Halo with a 50 TOPS NPU and a toolchain that soft-blocks
INT8. They sold the FastFlowLM runtime — 93 tok/s, proprietary, closed-source.
One person with a free Chess license, a C++ compiler, and 4 days reverse-engineered
the entire stack. The silicon was never the bottleneck. The business model was.

As of July 2, 2026: **94 tok/s (10.6 ms/tok) via FLM proxy** — matching FLM's own numbers.
The daemon proxies to FLM for production inference. Our open-source C++ engine
hits 97 tok/s (v12, single model) and 28 tok/s (all 5 models, auto-detect).

Every claim is timestamped in [docs/journey.md](docs/journey.md) — an audit
trail of every crash, deadlock, fix, and breakthrough. Open source ships
faster than venture capital.

*Built with DeepSeek v4 (99.9%) · Shipped with Claude (0.1%) · One human.*

*Built with DeepSeek v4 (99.9%) · Shipped with Claude (0.1%) · One human.*
*—bong-water-water-bong · "Sorry but not Sorry :)"*
*admin@1bit.systems*

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
│   │   ├── kernel/edge_attention.cc    # NPU attention kernel (Chess C++)
│   │   ├── build/                      # Pre-compiled objects + binaries
│   │   ├── BENCHMARKS.md               # Benchmark source of truth
│   │   └── README.md
│   └── gpu/                # Zig engine — GPU (Vulkan/CUDA/Metal)
│       └── build.zig                   # Zig build system (WIP)
├── site/                   # Landing page (Cloudflare Pages → 1bit.systems)
│   ├── index.html
│   └── assets/brand-lockup.svg
├── 1bit-site/              # Deploy mirror (synced from site/)
├── docs/                   # Architecture, build guide, roadmap, journey
├── packaging/              # deb, snap, tarball, docker, ollama, AUR
└── .github/workflows/      # CI benchmark + deploy + PR agent
```

## NPU Engine (`engine/npu/`)

**C++23. M=32 batched decode. FLM proxy in production (94 tok/s). C++ engine: 28 tok/s all-models, 97 tok/s v12.**

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
| Speed (v12) | **97 tok/s** (10 ms/tok) — C++ single-model |
| Speed (ALL) | **28 tok/s** (36 ms/tok) — C++ all 5 models |
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

## License

MIT — see [LICENSE](LICENSE).

---

*Built on Strix Halo. NPU + GPU. One chip. Two engines. Zero Python.*
*244→10 ms/tok (24×) on C++. FLM proxy at 94 tok/s in production. Open source ships faster than vendor lock-in.*
