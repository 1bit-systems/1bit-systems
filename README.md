<div align="center">

<img src="site/assets/brand-lockup.svg" alt="1bit" width="540">

# 1-bit inference, wired for Strix Halo.

### Pure Rust server. Custom HIP C++ kernels. One command.

`1bit` is the unified monorepo for 1-bit/ternary inference on AMD Strix Halo (gfx1151). It merges the Rust HTTP server (formerly `1bit-engine`) and the HIP C++ kernels (formerly `rocm-cpp`) into a single repository with a single build.

[![CI](https://github.com/bong-water-water-bong/1bit/actions/workflows/ci.yml/badge.svg)](https://github.com/bong-water-water-bong/1bit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![Site](https://img.shields.io/badge/site-1bit.systems-12a0ed.svg)](https://1bit.systems)
[![Rust 1.96+](https://img.shields.io/badge/rust-1.96%2B-orange.svg)](https://rustup.rs)
[![ROCm 7.2.4](https://img.shields.io/badge/rocm-7.2.4-blue.svg)](https://rocm.docs.amd.com)
[![Zaya](https://img.shields.io/badge/model-zaya-f00fd2.svg)](https://huggingface.co/zaya)
[![Strix Halo](https://img.shields.io/badge/strix%20halo-gfx1151%20%2B%20XDNA%202-12a0ed.svg)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max-series.html)

</div>

---

## Architecture

```
1bit/
├── rust/          axum HTTP server → spawns bitnet_decode
├── src/           HIP C++ kernels → ternary GEMV/GEMM
├── include/       C API headers → prefill_tuner.h, ck_gemm.h
├── kernels/       HIP implementations → phase5 decode, sherry, kv-cache
├── engine/
│   ├── npu/       C++23 INT8 engine — NPU (XDNA 2)
│   └── gpu/       Zig engine — GPU (Vulkan/CUDA/Metal), sibling to npu/
├── site/          1bit.systems website → static HTML
│   └── assets/brand-lockup.svg
├── 1bit-site/     Deploy mirror (synced from site/)
├── tools/
│   ├── video-lora/  Video gen w/ LoRA (Wan2.2, LTX-Video, AnimateDiff, CogVideoX)
│   │                + standalone Vulkan compute backend (Zig)
│   └── ...          Benchmarks, sweeps, model exporter
├── docs/          Architecture, build guide, roadmap, journey
├── packaging/     deb, snap, tarball, docker, ollama, AUR
├── benchmarks/    Historical benchmark data
└── .github/workflows/  CI benchmark + deploy + PR agent + video-lora CI
```

**One repo, one build, one binary.** `cmake` builds the kernels. `cargo` builds the server. `install.sh` does both.

## Performance

### Bonsai-1.7B (TQ2 Ternary) — Radeon 8060S

| Metric | Value |
|---|---|
| Decode | **113 tok/s** (8.8 ms/tok) |
| Prefill (1 tok) | 98 tok/s (10.2 ms) |
| Weight read per token | 436 MB (TQ2) |
| Prefill GEMM 2560×6912×2560 | **28.4 TFlops** (4h variant) |
| Ternary GEMM small M=16 | **51,779 tok/s** |
| Bonsai TQ2 GEMV 6912×6912 | **48.5 GB/s** (~3,824 tok/s) |
| Sherry GEMV 6912×6912 | **19.3 GB/s** (~2,588 tok/s) |

See [full benchmark data](benchmarks/RESULTS-stack-2026-04-28.md). *Strix Halo Radeon 8060S (gfx1151), 128 GB LPDDR5X.*

## One-Command Install

```bash
curl -fsSL https://raw.githubusercontent.com/bong-water-water-bong/1bit/main/install.sh | bash
```

# After install, set up the environment:
source ~/1bit/env.sh

Installs Rust toolchain, ROCm build deps, clones and builds kernels + server. Supports Ubuntu 24.04, Arch, CachyoS, Fedora.

```bash
# After install, set up the environment and run:
source ~/.cargo/env
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HSA_ENABLE_SDMA=0
export LD_LIBRARY_PATH=~/1bit/build:$LD_LIBRARY_PATH
~/1bit/rust/target/release/onebit --model model.h1b --port 13305 --tune-prefill --fp16-weights
```

## Connect Apps

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:13305/v1", api_key="any")
print(client.chat.completions.create(
    model="bitnet",
    messages=[{"role":"user","content":"Hello"}],
    max_tokens=20,
).choices[0].message.content)
```

Any OpenAI-compatible client works — Open WebUI, AnythingLLM, Continue, Aider, Cline, n8n, Dify.

## Build from Source

```bash
git clone https://github.com/bong-water-water-bong/1bit
cd 1bit

# Kernels (requires ROCm 7.2.4 — see [releases](https://github.com/ROCm/ROCm/releases))
cmake -B build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1151
ninja -C build rocm_cpp bitnet_decode bench_prefill_variants

# Server (requires Rust 1.96+)
cd rust && cargo build --release && cargo test --release  # 7/7 pass

source env.sh
```

## History

This monorepo merges four previously separate repositories (`rocm-cpp`, `1bit-engine`, `1bit-systems`, `1bit-lemonade`) into one. All development now happens here.

## Documentation

- [Kernel architecture](site/docs/architecture.html) — RDNA 3.5, ternary theory, packing
- [Benchmark report](benchmarks/RESULTS-stack-2026-04-28.md) — Full data
- [Kernel tuning guide](include/rocm_cpp/prefill_tuner.h) — Auto-tuner API
- [Install guide](install.sh) — One-command setup

## License

MIT. See `LICENSE`.

Sherry-specific kernels (3:4 N:M sparse ternary GEMV, tq1 packer): PolyForm Noncommercial 1.0.0. See `LICENSE-SHERRY.md`.
