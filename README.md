<div align="center">

<img src="site/assets/brand-lockup.svg" alt="1bit.systems" width="540">

# One Binary to rule them all.

### Pure C++ inference server · 282 KB · auto-detects every model

One server binary (`zaya_server`) runs every supported model across its backends — NPU fused, GPU ternary, ROCm HIP, Vulkan, CPU — auto-detecting the architecture from the model header.

[![CI](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![Site](https://img.shields.io/badge/site-1bit.systems-12a0ed.svg)](https://1bit.systems)
[![ROCm 7.2.4](https://img.shields.io/badge/rocm-7.2.4-blue.svg)](https://rocm.docs.amd.com)
[![Binary](https://img.shields.io/badge/binary-282%20KB-f00fd2.svg)](tests/zaya_server.cpp)
[![Strix Halo](https://img.shields.io/badge/strix%20halo-gfx1151%20%2B%20XDNA%202-12a0ed.svg)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max-series.html)
[![NPU fused](https://img.shields.io/badge/npu%20fused-broken-ff0000.svg)](https://github.com/bong-water-water-bong/1bit-systems/issues/56)
[![NPU↔GPU zero-copy](https://img.shields.io/endpoint?url=https://1bit.systems/zero-copy-badge.json&color=00ff00)](engine/fusion/zero_copy/test_parallel_real.hip)

</div>

---

## Hardware-Validated Performance

Measured on **AMD Strix Halo** (Ryzen AI Max+ 395) — 32 XDNA 2 NPU tiles + Radeon 8060S (gfx1151) GPU, 128 GB unified LPDDR5X memory.

| Engine | Backend | Hardware | tok/s | Status |
|--------|---------|----------|:-----:|--------|
| **GPU 1-bit** (llama.cpp ROCm) 🏆 | ROCm HIP | Radeon 8060S | **373** | ✅ measured |
| **NPU FLM** (production) | XDNA 2 xclbin | XDNA 2 · 32 tiles | **57** | ✅ validated |
| **GPU ternary** (Vulkan ZINC) | Vulkan GLSL | Radeon 8060S | **369** | ✅ validated |
| **GPU ZINC** (Vulkan) | Vulkan GLSL | Radeon 8060S | **22** | ✅ validated |
| **NPU v12** | XDNA 2 xclbin | XDNA 2 · 32 tiles | **69** | ⚙️ raw (see note) |
| **GPU ROCm HIP** (kernels) | ROCm HIP | Radeon 8060S | **65** | ✅ validated |
| **C++ all-5** (auto-detect) | Q4NX header parse | XDNA 2 · 32 tiles | **42** | ⚙️ raw (see note) |
| **NPU fused** | XDNA 2 xclbin | XDNA 2 · 32 tiles | **291** | ❌ broken (see note) |
| **GPU Zaya** (ROCm HIP) | HIP kernels | Radeon 8060S | **10.6** | ✅ validated |
| **DSpark** (spec-decode) | Speculative draft | XDNA 2 · 32 tiles | **0.8** | 🔶 unresolved (see note) |

> **❓ unsourced (GPU ZINC F16 / NPU fused 291 / DSpark 0.8):** GPU ZINC (22 tok/s) is historical not re-measured. NPU fused (291) and DSpark (0.8) remain blocked as documented in `benchmarks/latest.json` `_unverified`. All other rows in this table are from real, reproducible measurements on this hardware.

> **GPU 1-bit (373 tok/s tern), GPU ternary/ZINC Vulkan (369 tok/s), NPU FLM (57 tok/s), and GPU ROCm HIP (65 tok/s) are now ✅ validated** — see `tools/bench_gpu_1bit.sh`, `tools/bench_zinc_vulkan.sh`, `tools/bench_npu_flm.sh`, and `tools/bench_rocm_hip.sh` for the reproducible `validate_claims.py`-wired benchmarks that verify them daily.

> **Zero-copy NPU↔GPU substrate PROVEN** (`engine/fusion/zero_copy/test_zero_copy` passes on hardware: GPU reads CPU writes, CPU reads GPU writes — zero memcpy, zero IO_PAGE_FAULTs, 3/3 runs). This is the foundation for a real fused-layer engine.

> **NPU v12:** Re-measured 2026-07-12 after a 2026-07-11 correctness fix (RoPE convention, prefill causal mask, dynamic quant scale) that the fix's own commit admits was never validated against real hardware output before merging. Default OpenMP settings gave 6-8 tok/s (thread wake/sleep overhead dominating many small parallel regions); with OMP_NUM_THREADS=16 OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=cores, measured 49-70 tok/s depending on run length (69 tok/s typical, reproducible across 5 clean runs). An earlier pass this same day mistakenly re-tested a stale pre-fix binary and reported 110 tok/s -- wrong; confirmed the mistake by diffing binary hashes and rebuilding fresh from current source. Open issue: ~1/3-1/2 of runs hang at the boot-to-decode transition (a separate, pre-existing bug, not caused by this tuning). Old 97 tok/s figure was measured 2026-07-02, nine days before the correctness fix, on since-changed code.

> **C++ all-5 (auto-detect):** Re-measured 2026-07-12. Found and fixed a missing closing brace (issue #52) that made per-batch completion code -- including the loop-advance counter -- run once per layer (28x) instead of once per batch, so a requested N-token run actually executed ~28N steps, sampling from partially-computed hidden states most of the time and averaging in ever-slower later steps. Old 28 tok/s figure predates this fix. Re-measured on fixed code: 32-43 tok/s depending on run length, 42 typical at 64 tokens. Separate open issue: free() invalid size crash on exit at 128+ tokens, unrelated to this fix (happens after the measurement completes and prints).

> **NPU fused:** PR #42 (2026-07-12) fixed the tokenizer stub and decode-loop feedback (verified via a 5-token smoke test), but running the actual fused-engine CLI end-to-end (engine/fusion/main.zig) on 2026-07-12 reproducibly generates all-zero tokens and then hangs, even with a clean/uncontended NPU. The 291 tok/s figure predates this and is not currently reproducible on a working run.

> **DSpark (spec-decode):** Earlier ~572 tok/s projection and the 2026-07-07 '0% acceptance, disproven' conclusion were both wrong. Two real bugs found and fixed 2026-07-11: (1) a checkpoint-path wiring bug that made the benchmark silently run untrained, (2) a global_batch_size (512) larger than the whole dataset (360 examples), preventing real gradient steps. Ran an actual 420-step training pass after fixing both (and a broken torch/ROCm venv) -- loss dropped 26.5 to ~7.5, confirming real learning now happens for the first time. Re-measured: still 0.8 tok/s / 0% acceptance -- perplexity ~1800 means 343 examples/420 steps is too little data for a from-scratch draft head, not a new bug. Bugs are fixed and validated; nonzero acceptance needs substantially more training data/steps.

*Table auto-generated from [`site/benchmarks.json`](site/benchmarks.json) — last updated 2026-07-12*

See [full benchmark data](benchmarks/RESULTS-stack-2026-04-28.md).

---

## The Token Router

There is no single tiered auto-fallback router that ranks all backends by throughput and picks the fastest available one. Three real, different routing mechanisms exist across two repos, each doing something more specific:

```
┌─ Request ─────────────────────────────────────────────┐
│  POST /v1/chat/completions {"model":"zaya", ...}      │
└──────────────────────┬────────────────────────────────┘
                       ▼
              ┌────────┴────────┬──────────────────┐
              ▼                 ▼                  ▼
      cascade (per-token) spec_decode (draft+verify) content (keyword)
```

| Strategy | What it does | Where |
|---|---|---|
| **cascade** | Streams from the NPU, watches per-token log-probs mid-stream, switches to the GPU when confidence drops below a threshold, switches back when it recovers — one SSE stream, backend chosen per token, not per request | [`token-router`](https://github.com/bong-water-water-bong/token-router)`/src/cascade.rs` (separate repo) |
| **spec_decode** | NPU generates draft tokens, GPU verifies them in a batch (standard speculative decoding) | `tools/token_router.cpp` (this repo) |
| **content** | Routes the *whole request* to a small NPU model or a large GPU model based on keywords in the prompt (`code`, `explain`, `debug`, etc.) | `unified-router.py` (this repo) |
| **passthrough** | No routing — single fixed backend | `token-router`'s default strategy |

None of these is a static "rank backends by tok/s, always prefer the fastest one, fall back on unavailability" design — if you want that specific behavior, it doesn't exist yet. (Resolves #51 — a 2026-07-12 audit found the previously-documented 4-tier NPU→GPU→ROCm→CPU priority diagram didn't match any of the above; see the issue for the search that ruled it out.)

---

## Architecture

```
1bit/
├── tests/                  Pure C++ inference server
│   └── zaya_server.cpp    ← THE ONE BINARY — 282 KB
├── src/                    HIP C++ kernels → ternary GEMV/GEMM
├── include/                C API headers
├── kernels/                HIP implementations → zaya, bonsai, sherry, kv-cache
├── engine/
│   ├── npu/                C++23 INT8 engine — NPU (XDNA 2)
│   └── gpu/                Zig engine — GPU (Vulkan/CUDA/Metal)
├── site/                   1bit.systems website
│   ├── assets/             Brand assets, favicon
│   ├── blog/               Engineering blog
│   └── docs/               Architecture & deployment docs
├── tools/                  Model converters, benchmarks, video-lora
├── docs/                   Architecture, build guide, roadmap
├── packaging/              deb, snap, tarball, Docker
└── benchmarks/             Historical benchmark data
```

**The C++ server and HIP kernels build with one `cmake` invocation** (producing `zaya_server`, the `rocm_cpp` kernel library, and ~30 `test_*`/`bench_*` tools). The optional Zig (`engine/`), Rust (`rust/`, `npu-infer/rust/`), and TypeScript (`package.json`) components each have their own build systems and are not required to run the server.

---

## Quick Start

### Build zaya_server

```bash
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems
cmake -B build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build --target zaya_server -j8
```

### Run

```bash
./build/zaya_server                           # Starts on port 8088
# Optional flags:
#   --port 13305                              # Custom port
#   --model /path/to/model.h1b                # Model path
#   --tune-prefill                            # Enable prefill autotuning
#   --fp16-weights                            # Use FP16 weight cache
```

### OpenAI-compatible API

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8088/v1",
    api_key="any"  # required by spec, not checked
)

response = client.chat.completions.create(
    model="zaya",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=20,
)
print(response.choices[0].message.content)
```

Any OpenAI-compatible client works — Open WebUI, AnythingLLM, Continue, Aider, Cline, n8n, Dify.

---

## How It Works

### One Binary, All Models

`zaya_server.cpp` (19 KB of source, 282 KB compiled binary) reads the **Q4NX header** of any supported model at startup, auto-detects architecture dimensions (layers, heads, hidden size), allocates the correct buffers, and dispatches to the right backend. No config files. No model registry.

> **5 managed models in the catalog · 6 backends.** The header knows what it is; the binary figures out the rest. (The loader auto-detects additional GGUF architecture families beyond the 5 managed entries.)
>
> — [Read the full blog post →](site/blog/one-binary-to-rule-them-all.html)

### Runtime dependencies

The inference server `zaya_server` is C++17 compiled with HIP. It links the HIP/HSA runtime and cpp-httplib's transitive dependencies (OpenSSL, brotli, zlib) — run `ldd build/zaya_server` for the full list. It is **not** "HIP and libc only", and it has **no Python and no Rust runtime dependency**.

The wider repo is polyglot; those parts are optional and not required to build or run `zaya_server`:

- **Rust** — `rust/` is an OpenAI-compatible HTTP proxy in front of `bitnet_decode`; `npu-infer/rust/` is an FFI bridge.
- **Python** — model conversion, benchmarking, and training (`tools/`, `spec-decode/`, `engine/lora/`, `tools/video-lora/`).
- **TypeScript** — the `1bit` terminal coding agent (`src/`, `package.json`), based on [pi](https://github.com/earendil-works/pi-coding-agent).

### Verified Model Families

| Model | Size | Hidden | Layers | NPU speed |
|-------|------|--------|:------:|:---------:|
| Qwen3-0.6B | 0.6B | 1,536 | 28 | **69 tok/s** |
| Qwen3-8B | 8B | 4,096 | 32 | 215 ms/tok |
| Qwen3-VL-4B | 4B | 2,560 | 36 | 141 ms/tok |
| Llama-3.1-8B | 8B | 4,096 | 32 | 185 ms/tok |
| Gemma4-E2B | 2B | 2,304 | 26 | 117 ms/tok |

### 6 Backends

| Backend | API | Speed (0.6B) |
|---------|-----|:------------:|
| NPU v12 | XDNA 2 xclbin | **69 tok/s** |
| NPU fused | XDNA 2 xclbin | broken as of 2026-07-12 |
| ROCm HIP | HIP kernels | 113 tok/s |
| Vulkan | GLSL compute | 22 tok/s |
| CUDA | llama.cpp | ~30 tok/s |
| Metal | llama.cpp | ~15 tok/s |

---

## Documentation

| Resource | Description |
|----------|-------------|
| [1bit.systems](https://1bit.systems) | Full website with benchmarks, install, and blog |
| [One Binary Blog Post](site/blog/one-binary-to-rule-them-all.html) | How auto-detection works |
| [Kernel Architecture](site/docs/architecture.html) | RDNA 3.5, ternary theory, packing |
| [Build Guide](docs/) | Full build instructions |
| [Benchmark Data](benchmarks/) | Historical benchmark reports |
| [Packaging](packaging/) | deb, snap, tarball, Docker |

---

## License

MIT. See `LICENSE`.

Sherry-specific kernels (3:4 N:M sparse ternary GEMV, TQ1 packer): PolyForm Noncommercial 1.0.0. See `LICENSE-SHERRY.md`.

## Acknowledgements

The `1bit` terminal coding agent (`src/`, `package.json`) is based on [pi](https://github.com/earendil-works/pi-coding-agent) (MIT, © Earendil Works). "NPU-native" describes the inference engines in `engine/` and `tests/zaya_server.cpp`, **not** the coding-agent front-end, which is a cloud-LLM client like any other pi/OpenAI-compatible agent.
