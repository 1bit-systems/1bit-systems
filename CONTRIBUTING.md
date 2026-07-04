# Contributing to 1bit.systems

Thanks for your interest! This is a solo-developed project that runs LLM inference on AMD XDNA 2 NPUs from a single 120 KB binary. Contributions are welcome — here's how to help.

## Quick Start for Contributors

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/1bit-systems
cd 1bit-systems

# Build the NPU engine (requires Strix Halo + XRT 2.21+)
make -C packaging npu

# Build all packages
make -C packaging package-deb
```

See [docs/building.md](docs/building.md) for prerequisites and [docs/getting-started.md](docs/getting-started.md) for first-run.

## What Needs Help

| Area | How to Contribute |
|------|-------------------|
| **NPU Engine** (C++23) | INT8 quantization, context scheduling, attention kernels, new model support |
| **GPU Engine** (Zig) | Vulkan/CUDA/Metal shaders, GGUF loader, scheduler, HTTP server |
| **Packaging** | Publish to Snap Store, AUR, Homebrew, Docker Hub; CI/CD release automation |
| **Docs** | Tutorials, model compatibility charts, debugging guides |
| **Testing** | Benchmark regression tests, CI test suite, hardware-in-the-loop validation |
| **Site** | Cloudflare Pages, dashboard, landing page improvements |

## Pull Request Guidelines

1. **Conventional commits** — prefix your PR title: `[npu]`, `[gpu]`, `[packaging]`, `[docs]`, `[ci]`, `[site]`
2. **One change per PR** — keep it focused
3. **Include ms/tok delta** — for performance changes, include before/after benchmark numbers
4. **Tag `@bong-water-water-bong`** for review if time-sensitive
5. **Pre-commit** — always run `/verify` before committing engine changes (see [CLAUDE.md](CLAUDE.md))

## NPU Engine Design Principles

- **Zero Python at runtime** — the binary must run without Python, pip, or any interpreter
- **120 KB target** — every new feature should justify its binary size cost
- **Single binary, many models** — auto-detect, no recompilation per model
- **XRT direct** — no MLIR toolchain dependency at build time; link directly against `libxrt_coreutil`

## Reporting Issues

- **Bug**: include hardware (CPU, NPU generations), XRT version, exact command, full error output
- **Feature request**: describe the use case and why it needs NPU-native execution
- **Model request**: include model name, architecture, quantization format, and size

## Code of Conduct

Be excellent to each other. This project is MIT-licensed and built for the open-source community.

## Questions?

Open a [Discussion](https://github.com/bong-water-water-bong/1bit-systems/discussions) or join the [Discord](https://discord.gg/dSyV646eBs).
