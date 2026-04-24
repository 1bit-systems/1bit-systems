<div align="center">

# 1bit systems

### the 1-bit monster — local AI inference, bare metal, no python at runtime

[![CI](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Rust 1.88](https://img.shields.io/badge/rust-1.88-orange.svg)](./Cargo.toml)
[![Platform: Strix Halo gfx1151](https://img.shields.io/badge/platform-Strix%20Halo%20gfx1151-red.svg)](./docs/wiki/Why-Strix-Halo.md)
[![Website: 1bit.systems](https://img.shields.io/badge/web-1bit.systems-7c3aed.svg)](https://1bit.systems)
[![Wiki](https://img.shields.io/badge/wiki-deep%20dives-9333ea.svg)](https://github.com/bong-water-water-bong/1bit-systems/wiki)
[![AUR](https://img.shields.io/badge/AUR-1bit--systems--bin-1793d1.svg)](https://aur.archlinux.org/packages/1bit-systems-bin)

*"I know kung fu."*

</div>

---

## what is this

Full local AI stack for AMD Strix Halo. One mini-PC, one binary per service, a 1.58-bit ternary LLM answering over OpenAI-compatible HTTP. Hand-written HIP kernels on `gfx1151`, Rust for everything above — router, HTTP, agents, MCP, desktop, package manager. **rule A: no python at runtime.**

## numbers

Strix Halo reference (Radeon 8060S, gfx1151, 128 GB LPDDR5X):

| metric | value |
|---|---|
| decode @ 64-tok | **80.8 tok/s** |
| decode @ 1024-tok | 68.23 tok/s |
| ternary GEMV | 92% of LPDDR5 peak |
| split-KV FD attention | 10.25× @ L=2048 · 11.98× @ L=8192 |
| PPL wikitext-103 | 9.16 (chunk-1024) · 11.98 (pass-4095) |
| WMMA FP16 peak | 50.17 TFLOPS |
| perf/W | 0.904 tok/J (Strix) · 0.439 tok/J (9070 XT) |

Raw methodology + cross-arch numbers → [Benchmarks wiki](./docs/wiki/Benchmarks.md).

## install

```sh
# 1. AppImage (any distro, glibc ≥ 2.35, ROCm 7.x on host)
curl -fsSLO https://github.com/bong-water-water-bong/1bit-systems/releases/download/v0.1.0/1bit-systems-0.1.0-x86_64.AppImage
echo "8a964f89bdef68ed914c04fcf23092ac642c424bb70f74dc10e8558e93b94036  1bit-systems-0.1.0-x86_64.AppImage" | sha256sum -c -
chmod +x 1bit-systems-0.1.0-x86_64.AppImage && ln -sfn "$PWD/1bit-systems-0.1.0-x86_64.AppImage" ~/.local/bin/1bit

# 2. AUR (Arch family)
yay -S 1bit-systems-bin

# 3. source
cargo build --release --workspace && 1bit install core
```

One AppImage, 20 symlinked binary names, dispatched via `$ARGV0`. Full walkthrough → [Installation wiki](./docs/wiki/Installation.md).

## status

| lane | state |
|---|---|
| LLM · TTS · STT · image | ✅ shipping on `:8180 / :8095 / :8190 / :1234` |
| NPU toolchain (IRON + MLIR-AIE + Peano + libxrt, npu5) | ✅ axpy 160/160 on Arch |
| NPU serve-path (BitNet-1.58 end-to-end) | ❌ kernel authoring in flight |
| public launch | 🔒 ship-gate closed until NPU demo trips it |

## read more

- [Architecture-Deep](./docs/wiki/Architecture-Deep.md) — stack diagram, crate map, feature gates
- [Benchmarks](./docs/wiki/Benchmarks.md) — raw JSON, cross-arch (9070 XT / gfx1201), methodology
- [Why-Strix-Halo](./docs/wiki/Why-Strix-Halo.md) — hardware rationale + supported floors
- [NPU-Kernel-Design](./docs/wiki/NPU-Kernel-Design.md) · [NPU-Unlock-20260423](./docs/wiki/NPU-Unlock-20260423.md) — AIE2P path
- [Training-Runs](./docs/wiki/Training-Runs.md) — absmean QAT, Sparse-BitNet, BitNet v2 Hadamard
- [Eight-Models-Roadmap](./docs/wiki/Eight-Models-Roadmap.md) — what's next

Five house rules (A–E: no python at runtime, C++20 for kernels, no hipBLAS, Rust 1.88 edition 2024, ORT+VitisAI primary NPU lane) — see [CLAUDE.md](./CLAUDE.md) and [CONTRIBUTING.md](./CONTRIBUTING.md).

### supported targets

Shipping target is Strix Halo `gfx1151` (plus the wider RDNA3 / RDNA3.5 / RDNA4 Wave32-WMMA arches covered by the `rocm-cpp` fat-binary). The Apple-Silicon MLX backend (`--features mlx-apple`, via the `1bit-mlx` crate wrapping `bitnet-mlx-rs`) is a **dev-only path; not a supported deployment target.** It exists for Mac-side development ergonomics — quick iteration on tokenizer, chat-template, and sampler without a strixhalo box in reach. Nothing in the release artefacts runs on MLX.

> *every piece snaps in and snaps out. zero telemetry. zero cloud. bring your own APU.*

## license

MIT. See [LICENSE](./LICENSE). Model weights follow their own upstream licenses (Microsoft MIT for BitNet b1.58-2B-4T).

---

<div align="center">

[**1bit.systems**](https://1bit.systems) · [@bong-water-water-bong](https://github.com/bong-water-water-bong)

</div>
