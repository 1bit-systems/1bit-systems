# Contributing to 1bit.systems

Thanks for your interest! This is a solo-developed project that runs LLM inference on AMD XDNA 2 NPUs from a single ~437 KB binary. Contributions are welcome — here's how to help.

## Quick Start for Contributors

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/1bit-systems
cd 1bit-systems

# Build the spec-decode engine (requires Strix Halo + XRT 2.21+)
make -C spec-decode

# Build the fused NPU+GPU engine
make -C engine

# Build all packages
make -C packaging package-deb
```

Prerequisites: [docs/building.md](docs/building.md)  ·  First run: [docs/getting-started.md](docs/getting-started.md)  ·  Current benchmarks: [BENCHMARKS.md](BENCHMARKS.md)

> **Numbers are sourced from `site/numbers.json`.** After changing binary sizes or performance numbers, run `bash scripts/update-binary-sizes.sh` then `bash scripts/parse-benchmarks.sh` and commit the updated files.

## What Needs Help

| Area | How to Contribute |
|------|-------------------|
| **Spec-Decode Engine** (C++23) | INT8 quantization, context scheduling, attention kernels, new model support, fused NPU+GPU pipeline |
| **GPU Engine** (Zig / C++) | ROCm HIP kernels, WMMA matmul, GGUF loader, HTTP server (`flm serve`), engine_peak optimizer |
| **Fused NPU+GPU** (C++) | Coherent NPU+GPU fused pipeline — ORT-style multi-device scheduling, DMA sync, attention offload |
| **Packaging** | Publish to Snap Store, AUR, Homebrew, Docker Hub; CI/CD release automation |
| **Docs** | Tutorials, model compatibility charts, debugging guides |
| **Testing** | Expand the CI test suite (`ci.yml`), add hardware-in-the-loop benchmarks (`bench_1bit_models.py`, `verify_70plus_models.py`), write unit tests for the spec-decode engine and server endpoints, improve badge-generation scripts (`scripts/parse-benchmarks.sh`, `scripts/measure-binary.sh`) |
| **Site** | Cloudflare Pages, dashboard, landing page improvements |

## Pull Request Guidelines

1. **Conventional commits** — prefix your PR title: `[npu]`, `[gpu]`, `[packaging]`, `[docs]`, `[ci]`, `[site]`, `[spec-decode]`
2. **One change per PR** — keep it focused
3. **Include ms/tok delta** — for performance changes, include before/after benchmark numbers
4. **Tag `@bong-water-water-bong`** for review if time-sensitive
5. **Pre-commit** — always run `/verify` before committing engine changes (see [CLAUDE.md](CLAUDE.md))
6. **Update numbers.json** — if you change binary sizes or performance numbers, run `bash scripts/update-binary-sizes.sh` and `bash scripts/parse-benchmarks.sh` to keep the single source of truth in sync

## NPU Engine Design Principles

- **Zero Python at runtime** — the binary must run without Python, pip, or any interpreter
- **Small binary target** — every new feature should justify its binary size cost (currently ~437 KB decode, ~85 KB fused layer)
- **Single binary, many models** — auto-detect, no recompilation per model
- **XRT direct** — no MLIR toolchain dependency at build time; link directly against `libxrt_coreutil`

## Reporting Issues

- **Bug**: include hardware (CPU, NPU generations), XRT version, exact command, full error output
- **Feature request**: describe the use case and why it needs NPU-native execution
- **Model request**: include model name, architecture, quantization format, and size

## Code of Conduct

Be excellent to each other. This project is MIT-licensed and built for the open-source community.

## Questions?

Open a [Discussion](https://github.com/bong-water-water-bong/1bit-systems/discussions).
