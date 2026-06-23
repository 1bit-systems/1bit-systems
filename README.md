<div align="center">

<img src="1bit-site/assets/brand-lockup.svg" alt="1bit.systems" width="540">

# Local 1-bit inference, wired for Strix Halo.

### One runtime layer: Lemonade SDK.

**[→ Project Wiki](docs/wiki/README.md)** — architecture, decisions, gotchas, and agent onboarding.

`1bit.systems` is a pure 1-bit inference engine for AMD Strix Halo (gfx1151). The runtime is [Lemonade SDK](https://github.com/lemonade-sdk/lemonade) with a native `BitNetServer` backend that wraps [rocm-cpp](https://github.com/bong-water-water-bong/rocm-cpp) — custom HIP C++ kernels delivering 4.9–7.2× faster decode than rocBLAS FP16 at 1/4 the memory.

[![CI](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![Site](https://img.shields.io/badge/site-1bit.systems-12a0ed.svg)](https://1bit.systems)
[![Discord](https://img.shields.io/badge/discord-1bit.systems-f00fd2.svg?logo=discord&logoColor=white)](https://discord.gg/dSyV646eBs)
[![Endpoint](https://img.shields.io/badge/endpoint-:13305%2Fv1-00ff00.svg)](#connect-apps)
[![Strix Halo](https://img.shields.io/badge/strix%20halo-gfx1151%20%2B%20XDNA%202-12a0ed.svg)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max-series.html)

</div>

---

## Architecture

```
lemonade-sdk/lemonade  →  1bit-lemonade (fork)
│
└── lemond (:13305)
     ├── BitNetServer  ←  bitnet_decode --server
     │    └── librocm_cpp.so  (ternary HIP, 4.9–7.2× rocBLAS)
     ├── LlamaCppServer  (fallback for non-1bit models)
     └── FastFlowLMServer  (NPU lane, optional)
```

No proxy. No shell-script orchestration. No toolbox containers. One `lemond` process manages backend subprocesses, model LRU eviction, and the full OpenAI-compatible API surface (including Ollama and Anthropic compatibility).

## Install

### Prerequisites

- AMD Strix Halo (Ryzen AI MAX+ 395, gfx1151)
- Arch Linux or CachyOS with ROCm 7.x (TheRock or system)
- GCC 14+, CMake, Ninja

### Quick install

```bash
# 1. Build and install rocm-cpp (the 1-bit kernels)
git clone https://github.com/bong-water-water-bong/rocm-cpp
cd rocm-cpp
cmake -B build -G Ninja && ninja -C build

# 2. Build and install 1bit-lemonade (Lemonade SDK + BitNet backend)
git clone https://github.com/bong-water-water-bong/1bit-lemonade
cd 1bit-lemonade
./setup.sh
cmake --build --preset default

# 3. Export a BitNet model to .h1b format (from HuggingFace safetensors)
cd ../rocm-cpp
./tools/export_bitnet.py --model 1bitSUPER/bitnet_b1_58-2B-4T --out ~/models/bitnet-b1.58-2b.h1b

# 4. Run
cd ../1bit-lemonade
./build/default/bin/lemond &
./build/default/bin/lemonade run BitNet-b1.58-2B-4T
```

### One-liner test

```bash
curl http://127.0.0.1:13305/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"BitNet-b1.58-2B-4T","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":5}'
```

## Connect Apps

Point any OpenAI-compatible client at Lemonade:

| App type | Base URL |
|---|---|
| OpenAI SDK (Python, Node, Go, etc.) | `http://127.0.0.1:13305/v1` |
| Open WebUI, AnythingLLM, Dify, n8n | `http://127.0.0.1:13305/v1` |
| Continue.dev, Aider, Cline | `http://127.0.0.1:13305/v1` |
| Ollama clients | `http://127.0.0.1:13305` |

API key: any non-empty string (e.g. `local-no-auth`).

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:13305/v1", api_key="local-no-auth")
print(client.chat.completions.create(
    model="BitNet-b1.58-2B-4T",
    messages=[{"role":"user","content":"Say stack OK in five words."}],
    max_tokens=20,
).choices[0].message.content)
```

## Verified Benchmarks

On the reference Strix Halo box (Ryzen AI MAX+ 395, 128 GB unified):

| Model | Prompt tok/s | Gen tok/s | Notes |
|---|---|---|---|
| BitNet-2B-4T | — | **82** | End-to-end, bit-matching PyTorch reference |
| Bonsai 1.7B IQ1_S | 5,001 | 231 | 231 MB on disk |
| Bonsai 4B IQ1_S | 2,125 | 126 | 540 MB on disk |
| Bonsai 8B IQ1_S | 1,325 | 96 | 1.07 GB on disk |
| Gianni BitNet 3B TQ2_0 | 1,796 | 76 | Ternary quantized |

**Decode GEMV:** 4.9–7.2× faster than rocBLAS FP16 at 1/16 the B memory.
**Prefill GEMM:** 30.15 TFlops on BitNet FFN shapes, beating rocBLAS FP16 at 1/4 the B memory.

## Repos

| Repo | Role |
|---|---|
| [1bit-lemonade](https://github.com/bong-water-water-bong/1bit-lemonade) | Fork of Lemonade SDK with `BitNetServer` backend |
| [rocm-cpp](https://github.com/bong-water-water-bong/rocm-cpp) | Native ROCm C++ HIP kernels for 1-bit/ternary inference |
| [1bit-systems](https://github.com/bong-water-water-bong/1bit-systems) | Website, docs, benchmarks (this repo) |

## License

MIT. See `LICENSE`.

rocm-cpp kernels: MIT (see `rocm-cpp/LICENSE`). Sherry-specific sparse ternary kernels: PolyForm Noncommercial (see `rocm-cpp/LICENSE-SHERRY.md`).
