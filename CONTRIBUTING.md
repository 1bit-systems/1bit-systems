# Contributing to 1bit.systems

**One Binary to rule them all.** A pure C++ LLM inference server that auto-detects every supported model architecture and dispatches tokens to the fastest available backend — NPU fused, GPU HIP, Vulkan, or CPU — from a single 282 KB binary. No Python at runtime. No Rust at runtime. Zero configuration files.

This guide covers how to build, test, and contribute to the project.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Quick Start](#quick-start)
- [Build System](#build-system)
  - [Prerequisites](#prerequisites)
  - [Build zaya_server](#build-zaya_server)
  - [Build NPU Engine](#build-npu-engine)
  - [Build Tests & Benchmarks](#build-tests--benchmarks)
- [Codebase Map](#codebase-map)
- [What Needs Help](#what-needs-help)
- [Contribution Guidelines](#contribution-guidelines)
  - [Pull Request Checklist](#pull-request-checklist)
  - [Conventional Commit Prefixes](#conventional-commit-prefixes)
  - [Code Review](#code-review)
- [Design Principles](#design-principles)
- [Testing & Performance](#testing--performance)
- [Reporting Issues](#reporting-issues)
- [Code of Conduct](#code-of-conduct)

---

## Project Overview

```
                     ┌──────────────────────────────────────┐
                     │       zaya_server  (282 KB)          │
                     │     Pure C++ · No Python · No Rust   │
                     └─────────────┬────────────────────────┘
                                   │
                     ┌─────────────▼────────────────────────┐
                     │        Token Router                  │
                     │  Profiles backends · Selects fastest │
                     └────┬──────────┬──────────┬───────────┘
                          │          │          │
                    ┌─────▼───┐ ┌───▼────┐ ┌───▼────┐
                    │ NPU     │ │ GPU    │ │ CPU    │
                    │ XDNA 2  │ │ ROCm   │ │ OpenMP │
                    │ C++23   │ │ HIP    │ │ Q4NX   │
                    │ 291 t/s │ │ 113 t/s│ │ ~5 t/s │
                    └─────────┘ └────────┘ └────────┘
```

### Key facts

| Metric | Value |
|--------|-------|
| Binary | `zaya_server` — **282 KB** |
| Language | **C++23** (NPU engine), **C++20 with HIP** (GPU kernels), C++20 (server) |
| Build system | **CMake** 3.21+ with **Ninja** |
| GPU compiler | **HIP** via AMD ROCm **7.2.4** (TheRock `/ amdclang++`) |
| Target GPU | **gfx1151** — AMD Radeon 8060S (Strix Halo) |
| Target NPU | **XDNA 2** — 32 tiles, INT8, via C++23 engine + XRT 2.21+ |
| CPU fallback | Any x86-64 with OpenMP |
| Auto-detect | Reads Q4NX model header — no config files |
| Linux kernel | Ubuntu 24.04 LTS or later / kernel 7.0.0+ (6.18.22-lts recommended for stability) |
| License | MIT (Sherry-specific kernels: PolyForm Noncommercial 1.0.0) |
| ⚠️ Known issue | Kernel 6.19.x has an amdgpu OPTC CRTC hang under sustained NPU+GPU load on Strix Halo (gfx1151). Use 7.0.0+ or 6.18.22-lts. See issue #1. |

---

## Quick Start

```bash
# Fork & clone
git clone https://github.com/YOUR_USERNAME/1bit-systems
cd 1bit-systems

# Build zaya_server (the one binary)
cmake -B build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build --target zaya_server -j$(nproc)

# Run (auto-detects model from default path)
./build/zaya_server

# With custom model path
./build/zaya_server --model /path/to/model.h1b
```

See [docs/building.md](docs/building.md) for full prerequisites and [docs/getting-started.md](docs/getting-started.md) for first-run instructions.

---

## Build System

### Prerequisites

**Hardware (required for GPU/NPU paths):**
- AMD Strix Halo (Ryzen AI Max+ 395) — Radeon 8060S (gfx1151) + XDNA 2 NPU
- 128 GB unified LPDDR5X memory (recommended for large models)

**Software:**
| Dependency | Version | Purpose |
|------------|---------|---------|
| Ubuntu | 24.04 LTS or later (kernel 7.0.0+) | Host OS |
| CMake | ≥ 3.21 | Build system |
| Ninja | latest | Fast builds |
| ROCm | **TheRock 7.15.0a** | HIP compiler (pip) |
| GCC | ≥ 13 | C++20 host compiler |
| AMD XRT | ≥ 2.21 | NPU runtime (`libxrt_coreutil`) |
| Git LFS | latest | Model file storage |

**TheRock 7.15.0a installation (pip):**
```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  rocm[devel,libraries]
export THEROCK_PIP_ROOT="$HOME/.cache/pip/therock"
```

The project auto-discovers the HIP compiler in this order (see CMakeLists.txt):
1. TheRock pip SDK (`$HOME/.cache/pip/therock`)
2. Lemonade cache (`$HOME/.cache/lemonade/bin/therock`)
3. Legacy TheRock path (`$HOME/therock/build/dist/rocm`)
4. System ROCm (`/opt/rocm`)

Set `THEROCK_PIP_ROOT` to pin to a specific TheRock installation.

### Build zaya_server

```bash
# Configure
cmake -B build -G Ninja \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release

# Build the one binary
cmake --build build --target zaya_server -j$(nproc)

# Verify size
ls -lh build/zaya_server
# Expected: ~282 KB (see site/numbers.json for the current auto-tracked value)
```

### Build NPU Engine

> Requires Strix Halo with XDNA 2 NPU, XRT 2.21+, and the torch2aie toolchain.

```bash
# Build xclbins (one-time per projection shape)
cd engine/npu
./build_xclbins.sh

# Build the engine
cd ../..
g++ -std=c++23 -O3 -o build/npu_engine \
  engine/npu/src/npu_engine_i8.cpp \
  engine/npu/build/dequant_q4nx.o \
  -I/opt/xilinx/xrt/include \
  -L/opt/xilinx/xrt/lib64 \
  -lxrt_coreutil -luuid -lm -ldl
```

See [docs/building.md](docs/building.md#step-3-build-int8-xclbins-one-time) for detailed xclbin generation steps.

### Build Tests & Benchmarks

```bash
# Build all tests
cmake --build build -j$(nproc)

# Run specific tests
./build/test_bonsai_gemv
./build/test_ternary_gemm_smallm
./build/test_backend
./build/test_zaya_moe_gemv

# Run benchmarks
./build/bench_bonsai_full
./build/bench_fused_tq2_1024
```

---

## Codebase Map

```
1bit-systems/
├── tests/
│   └── zaya_server.cpp       ← THE ONE BINARY — 282 KB, 19 KB source
├── src/                       HIP C++ kernels (ternary GEMV/GEMM, prefill, KV cache)
│   ├── bonsai_*.hip           Bonsai 1.58-bit ternary kernels
│   ├── sherry_*.hip           Sherry 3:4 N:M sparse ternary kernels
│   ├── kv_cache_attn*.hip     KV cache attention kernels
│   ├── backend_*.cpp          Backend abstraction (HIP, NPU, CPU, Vulkan)
│   └── zaya_engine.cpp        Token router and engine dispatch
├── kernels/                   Additional HIP implementations
│   ├── ternary_gemv*.hip      TQ2 ternary GEMV variants
│   ├── bonsai_q1_*.hip        Q1.58-bit GEMM kernels
│   └── hadamard_rotate*.hip   Hadamard rotation kernels
├── include/                   C API headers
│   ├── backend_*.h            Backend interface declarations
│   └── rocm_cpp/              ROCm CMake helpers
├── engine/
│   ├── npu/                   C++23 NPU engine — XDNA 2 INT8
│   │   ├── src/               Engine implementations (i8, v2–v13, batch, spec, cb)
│   │   ├── kernel/            NPU kernel sources
│   │   ├── xclbins/           Generated xclbins per projection shape
│   │   └── build/             Build artifacts & compiled xclbins
│   └── gpu/                   GPU engine (Zig)
├── packaging/                 Distribution packages
│   ├── deb/                   Debian packaging
│   ├── snap/                  Snapcraft packaging
│   ├── appimage/              AppImage packaging
│   ├── docker/                Docker image
│   ├── aur/                   Arch AUR scripts
│   └── homebrew/              Homebrew formula
├── .github/workflows/         CI/CD pipelines
├── docs/                      Architecture, build guide, roadmap
├── site/                      1bit.systems website (Cloudflare Pages)
├── tools/                     Model converters, benchmark scripts
└── benchmarks/                Historical benchmark data
```

---

## What Needs Help

| Area | Language | How to Contribute |
|------|----------|-------------------|
| **NPU Engine** | **C++23** | INT8 quantization, context scheduling, attention kernels, new model topology support, xclbin generation |
| **GPU HIP Kernels** | **C++20 + HIP** | Ternary GEMV/GEMM optimizations, WMMA intrinsic tuning, new kernel variants, prefill performance |
| **Token Router** | **C++20** | Backend profiling, dynamic dispatch heuristics, fallback logic, multi-backend coherence |
| **Server** | **C++20** | HTTP API, OpenAI-compatible endpoints, streaming, systemd integration, model auto-detection |
| **CPU Backend** | **C++20 + OpenMP** | Q4NX dequantization, OpenMP thread tuning, fallback performance |
| **Vulkan Backend** | **Zig + GLSL** | Compute shaders, SPIR-V toolchain, Vulkan memory management |
| **Packaging** | **Bash, YAML** | Snap Store, AUR, Homebrew, Docker Hub, AppImage, CI/CD release automation |
| **CI/CD** | **YAML, Python** | GitHub Actions workflows, cross-arch testing, benchmark regression bots |
| **Testing** | **C++20** | Benchmark regression suite, hardware-in-the-loop validation, fuzz testing |
| **Documentation** | **Markdown** | Model compatibility charts, debugging guides, architecture deep-dives |
| **Website** | **HTML/CSS/JS** | Cloudflare Pages, live benchmarks dashboard, blog posts |

---

## Contribution Guidelines

### Pull Request Checklist

1. **One change per PR** — keep it focused and atomic
2. **Conventional commit prefix** — see [table below](#conventional-commit-prefixes)
3. **Builds clean** — `cmake --build build --target zaya_server -j$(nproc)` without warnings
4. **Existing tests pass** — run relevant `test_*` binaries from `build/`
5. **Add tests** for new kernels, backends, or features
6. **Benchmark data** — for performance changes, include before/after `ms/tok` or `tok/s` deltas
7. **Size awareness** — for new features, document the binary size impact (`ls -lh build/zaya_server`)
8. **No Python or Rust dependencies at runtime** — the binary must run with zero interpreters

### Conventional Commit Prefixes

| Prefix | Scope |
|--------|-------|
| `[npu]` | NPU engine — XDNA 2 xclbins, C++23 engine, INT8 kernels |
| `[hip]` | GPU HIP kernels — ternary GEMV/GEMM, WMMA, KV cache |
| `[gpu]` | GPU engine — Vulkan, Zig runtime, GLSL shaders |
| `[router]` | Token router — backend dispatch, profiling, fallback |
| `[server]` | HTTP server — API, streaming, model loading |
| `[cpu]` | CPU backend — Q4NX, OpenMP |
| `[packaging]` | deb, snap, AppImage, AUR, Homebrew, Docker |
| `[ci]` | GitHub Actions, workflows, CI infrastructure |
| `[docs]` | Documentation, README, guides |
| `[site]` | 1bit.systems website |
| `[tools]` | Model converters, benchmark scripts |

**Examples:**
```
[npu] Add INT8 attention kernel for Qwen3-0.6B context prefill
[hip] Optimize ternary GEMV with WMMA f16 accumulators — 279 → 291 tok/s
[server] Add --tune-prefill flag for autotuning at startup
[packaging] Add snap confinement policy for NPU device access
[docs] Add model compatibility matrix for verified architectures
```

### Code Review

- Tag **`@bong-water-water-bong`** for review if time-sensitive
- PRs without test coverage for new kernels/backends will not be merged
- Performance PRs **must** include before/after benchmark numbers
- Binary size changes **must** be justified in the PR description

---

## Design Principles

These principles guide every contribution. Please respect them.

### Zero Python at Runtime

The server binary must run without Python, pip, or any interpreter. Python may be used in `tools/` and `scripts/` for model conversion, benchmarking, and CI, but **never** in the hot path or as a runtime dependency.

### Zero Rust at Runtime

No Rust runtime, Rust build tools, or Rust dynamic libraries may be required to build or run `zaya_server`. (Rust code exists in the `rust/` directory for auxiliary tooling; it must never be a dependency of the server.)

### One Binary, All Models

`zaya_server` reads the Q4NX header of any supported model at startup, auto-detects architecture dimensions (layers, heads, hidden size), allocates correct buffers, and dispatches to the right backend. **No configuration files. No model registry.** Every new model topology should work without recompilation.

### Binary Size Budget

Every new feature should justify its binary size cost. The server is 19 KB of C++ source; it originally compiled to a 207 KB static binary and is currently 282 KB (auto-tracked in `site/numbers.json` — see `tools/gen_numbers.py`). That 36% growth was never individually justified in a PR — it accumulated silently because nothing re-measured the binary size until an audit caught it. Size regressions require strong justification. When adding code, ask: *"Does this belong in the one binary, or can it live in the NPU engine or a tool?"*

### CMake-Driven Build

All build logic lives in `CMakeLists.txt`. The build must work with:
```bash
cmake -B build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build --target zaya_server -j$(nproc)
```
No Makefiles, no shell scripts, no Python wrappers for the core build.

### Auto-Detect, Don't Configure

The binary probes hardware at startup:
1. **NPU availability** → XRT device enumeration
2. **GPU capability** → HIP device properties + kernel profiling
3. **CPU fallback** → OpenMP thread count

No `--backend` flags. No config files. The token router profiles each backend and selects the fastest path per-layer, per-context.

### Target Hardware: Strix Halo (gfx1151 + XDNA 2)

All HIP kernels target **gfx1151** (AMD Radeon 8060S). The NPU engine targets **XDNA 2** with 32 tiles via XRT 2.21+. Code paths for other architectures are welcome as long as they don't regress Strix Halo performance.

---

## Testing & Performance

### Running Tests

```bash
# Build everything
cmake --build build -j$(nproc)

# Kernel correctness
./build/test_bonsai_gemv
./build/test_sherry_gemv
./build/test_ternary_gemm_smallm
./build/test_prim_and_attn

# Backend integration
./build/test_backend

# End-to-end
./build/test_bonsai_e2e
./build/zaya_e2e

# NPU engine (Strix Halo only)
./build/npu_engine
```

### Benchmarking

```bash
# Full benchmark suite
./build/bench_bonsai_full

# Targeted benchmarks
./build/bench_bonsai_gemv_only
./build/bench_fused_tq2_1024
./build/bench_bonsai_q1_1024
./build/bench_wmma_vs_tq2

# Cross-backend comparison
./build/zaya_infer
```

For performance PRs, report results in this format:

```
[hip] WMMA accumulation — Qwen3-0.6B, gfx1151, 32 tiles
Before: 279 tok/s (baseline wmma_peak_probe.hip)
After:  291 tok/s (+4.3%)
Binary delta: +1.2 KB (207.0 → 208.2 KB)
```

### CI Pipeline

The project uses GitHub Actions (`.github/workflows/ci.yml`):

| Job | What it checks |
|-----|----------------|
| `cpp` | CMake configure + build, ROCm availability, ShellCheck, host tests |
| `version` | Version consistency across manifest files + Q4NX round-trip test |
| `rust` | Rust proxy build + test |
| `ts` | TypeScript CLI type-check |
| `fusion` | Fused engine build + smoke test |

PRs must pass the `cpp` job before merge.

---

## Reporting Issues

### Bug Reports

Include in your bug report:

- **Hardware**: CPU model, NPU generation (XDNA 2), GPU (gfx1151)
- **ROCm version**: `cat /opt/rocm/share/doc/rocm-version/version`
- **XRT version**: `dpkg -l libxrt-dev | tail -1`
- **Kernel version**: `uname -a`
- **Command**: Exact command used, with all flags
- **Error output**: Full terminal output, not a summary
- **Model**: Model name, size, quantization format, source
- **Binary size**: `ls -lh build/zaya_server` (only if size-related)

### Feature Requests

Describe:
- **Use case**: What problem are you solving?
- **Why this backend**: Why does it need NPU-native / GPU / CPU execution?
- **Performance target**: Expected tok/s or latency
- **Binary size budget**: Estimated cost in KB

### Model Requests

Include:
- Model name and family (e.g., Qwen3-8B, Llama-3.1-8B)
- Architecture type (dense, MoE, etc.)
- Quantization format (Q4NX, Q8_0, etc.)
- Hidden dimensions, layer count, head count
- Where to obtain the model
- Whether it has been verified on Strix Halo

---

## Code of Conduct

Be excellent to each other. This is a solo-developed open-source project built for the AMD AI community. All contributions — code, docs, testing, discussions — are welcome and valued.

- **Be respectful**: Everyone is learning. Assume good faith.
- **Be constructive**: Critique code, not people. Offer alternatives.
- **Be patient**: This is a small project with one primary maintainer. Reviews take time.
- **Be size-conscious**: a small binary is a feature. Every byte counts (current: 282 KB, tracked in `site/numbers.json`).

This project is MIT-licensed. Sherry-specific kernels in `src/sherry_*.hip` are PolyForm Noncommercial 1.0.0 (see [LICENSE-SHERRY.md](LICENSE-SHERRY.md)).

---

## Questions?

- Open a [GitHub Discussion](https://github.com/bong-water-water-bong/1bit-systems/discussions)
- Read the [architecture docs](docs/architecture.md)
- See the [roadmap](ROADMAP.md)
