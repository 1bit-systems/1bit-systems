<div align="center">

<img src="1bit-site/assets/logo.png" alt="1bit-systems" width="120" height="120">

# 1bit-systems

### **the unified 1-bit engine for AMD Strix Halo &mdash; the 2-bit killer**

*One OpenAI-compatible endpoint. iGPU + NPU behind one API. 1-bit is the floor; sub-2-bit MoE is what you'll keep open all day.*

[![CI](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-00d4ff.svg)](LICENSE)
[![AUR: 1bit-systems-git](https://img.shields.io/badge/AUR-1bit--systems--git-1793D1.svg?logo=archlinux&logoColor=white)](packaging/aur/1bit-systems-git)
[![AUR: 1bit-systems-bin](https://img.shields.io/badge/AUR-1bit--systems--bin-1793D1.svg?logo=archlinux&logoColor=white)](packaging/aur/1bit-systems-bin)
[![Site](https://img.shields.io/badge/site-1bit.systems-00d4ff.svg)](https://1bit.systems)
[![Discord](https://img.shields.io/badge/discord-1bit.systems-5865F2.svg?logo=discord&logoColor=white)](https://discord.gg/dSyV646eBs)
[![Built on Lemonade](https://img.shields.io/badge/built%20on-Lemonade-yellow.svg)](https://lemonade-server.ai)
[![Built on FastFlowLM](https://img.shields.io/badge/built%20on-FastFlowLM-orange.svg)](https://github.com/FastFlowLM/FastFlowLM)
[![GitHub last commit](https://img.shields.io/github/last-commit/bong-water-water-bong/1bit-systems)](https://github.com/bong-water-water-bong/1bit-systems/commits/main)
[![Hardware: Strix Halo](https://img.shields.io/badge/hardware-Strix%20Halo%20%7C%20gfx1151%20%7C%20XDNA%202-red.svg)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max-series.html)

</div>

---

## Status — green on every lane

| Lane | Backend | State |
|---|---|:-:|
| **iGPU (ROCm, gfx1151)** | `llamacpp:rocm` `b1231` | ✅ |
| **iGPU (Vulkan)** | `llamacpp:vulkan` `b8668` | ✅ |
| **NPU (XDNA 2)** | `flm:npu` `v0.9.39` | ✅ |
| Image gen (ROCm) | `sd-cpp:rocm` | ✅ |
| TTS / STT | `kokoro:cpu`, `whispercpp:vulkan` | ✅ |
| Web UI | `open-webui` on `:3000` | ✅ |
| Agent UI | `amd-gaia` on `:5005` | ✅ |

**Unified.** All seven services above hang off **one OpenAI-compatible endpoint at `:13305`**. The same request shape routes to iGPU (ROCm or Vulkan) or NPU (FastFlowLM) based on the model name — Lemonade's native recipe routing, no proxy, no shim. Send the same `POST /v1/chat/completions` and switch which silicon answers by changing `model:`.

---

## Install (Arch / CachyOS)

> On Ubuntu, considering Distrobox, or trying to choose between AMD's closed Ryzen AI EP
> and the open Lemonade + FLM stack? See [`docs/INSTALL.md`](docs/INSTALL.md) for the
> both-paths writeup.

### Step 0 &mdash; prerequisites

A clean CachyOS / Arch base ships *most* of these, but a minimal install can be missing some. Run once before cloning:

```sh
sudo pacman -Syu                                  # fully update first
sudo pacman -S --needed git base-devel paru curl  # git for clone, base-devel for AUR builds,
                                                  # paru for the AUR lemonade-server package,
                                                  # curl is used by install.sh for the model pull
```

You'll also need:
- A user with **`sudo` privileges** (the installer writes `/etc/security/limits.d/99-1bit-systems.conf` and patches Lemonade's manifest under `/usr/share/`).
- A **Strix Halo** APU (`gfx1151` iGPU + XDNA 2 NPU). Other Ryzen AI gens (XDNA 1 — 7000 / 8000 / 200-series) will install but the `flm:npu` lane won't light up.
- A modern kernel with **in-tree `amdxdna`** &mdash; Linux 7.0+ ships it. CachyOS 7.0.x and Arch's mainline `linux` kernel are good. For kernels older than 7.0 (e.g. 6.x), install `paru -S amdxdna-dkms` first.
- ~70&nbsp;GB free on `/home` for the default model pulls. More if you plan to run the 35B IQ2_XXS daily driver alongside the Bonsai floor.

### Step 1 &mdash; clone + run

```sh
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems
./install.sh                 # actually install
./install.sh --dry-run       # preview every action without mutating
```

Installs `lemonade-server`, `fastflowlm`, `xrt`, `xrt-plugin-amdxdna`, `rocm-hip-sdk`. Writes memlock limits. Auto-patches Lemonade's `flm:npu` version pin to match your installed FastFlowLM (the silent `update_required` gotcha — see [docs/reddit-npu-gate.md](docs/reddit-npu-gate.md) for the receipts). Pulls the default 1-bit model (`Bonsai-1.7B-IQ1_S`, 385 MB). Installs the `1bit` CLI to `/usr/local/bin/`.

After first install: re-login or reboot once so memlock limits apply on the NPU lane.

#### Wall-time expectations

| Phase | Fresh box (100 Mbps) | Notes |
|---|---:|---|
| Step 0 prereqs (`pacman -Syu` + 4 small pkgs) | 1&ndash;3 min | depends how out-of-date the system is |
| `git clone` | ~5 s | tiny repo |
| `pacman -S` deps inside `install.sh` | **5&ndash;10 min** | dominated by ROCm: `composable-kernel` (1.6 GB) + `rocm-llvm` (1 GB) + `rccl` 424 MB + `rocblas` 323 MB + smaller &asymp; **~4 GB** total download + extract |
| `paru -S lemonade-server` (AUR build) | 1&ndash;2 min | Go build + dep resolution |
| memlock conf + `flm:npu` pin patch + CLI install | < 1 s | trivial |
| `lemond` start + health probe | ~3 s | |
| Bonsai-1.7B-IQ1_S model pull (385 MB) | ~5 s @ 75 MB/s | |
| **Total &mdash; fresh install** | **~10&ndash;15 min** | bottleneck is ~4 GB of ROCm packages from CachyOS repos |
| **Idempotent re-run** | **< 5 s** | every step detects "already installed" and skips |

Faster on gigabit (~5&ndash;8 min). The AUR build of `lemonade-server` is the only step that can't overlap with the model pull.

## Run

```sh
1bit up                     # starts lemond on :13305 + opens the web UI
1bit status                 # quick health on lemond + flm + memlock
1bit pull qwen3:0.6b        # auto-routes: NPU via flm, iGPU via lemonade
1bit bench                  # full 1-bit / ternary pile bench
1bit down                   # stop lemond
```

## Default models — benchmarked head-to-head

Captured **2026-04-28** on the production strixhalo box (Ryzen AI MAX+ 395 / `gfx1151`, 124 GB LPDDR5x), single-shot via OpenAI-compat curls against `:13305`. Full methodology in [`benchmarks/`](benchmarks).

### The floor — 1-bit Bonsai

[full sweep →](benchmarks/RESULTS-stack-2026-04-28.md)

| Model | Quant | Disk | Decode (steady) | Prefill |
|---|---|---:|---:|---:|
| `lilyanatia/Bonsai-1.7B-IQ1_S` | IQ1_S (~1.6 bpw) | **385 MB** | **255 — 281 tok/s** | 853 — 2220 tok/s |

The brand floor. Demonstrates that sub-2-bit on `gfx1151` is real, fast, and shipping today.

### The daily driver — sub-2-bit MoE

[full bench →](benchmarks/RESULTS-qwen3.5-35b-quant-2026-04-28.md)

| Model | Quant | Bits/wt | Disk | Decode (steady) |
|---|---|---:|---:|---:|
| `unsloth/Qwen3.5-35B-A3B-Q4_K_XL` (baseline) | Q4_K_XL | ~4.5 | 22 GB | 54 tok/s |
| `Manojb/Qwen3.5-35B-A3B-UD-IQ2_XXS` | **IQ2_XXS** | **~2.0** | **11.5 GB** | **73 tok/s** |
| **Δ sub-2-bit / baseline** | — | — | **0.52×** | **+35 %** |

Half the disk, **+35 % decode** on the same 35B MoE. This is the model `gaia chat` and `open-webui` default to.

### NPU lane — `flm:npu`

| Model | Decode (steady) | Prefill | TTFT |
|---|---:|---:|---:|
| `qwen3-0.6b-FLM` | 95 tok/s | 76 tok/s | 460 ms |
| `qwen3-1.7b-FLM` | 42 tok/s | 54 tok/s | 577 ms |
| `deepseek-r1-8b-FLM` | 11 tok/s | 15 tok/s | 1430 ms |

NPU isn't the throughput champion against the iGPU on small models — its value is **offload** (free up the iGPU for bigger ROCm models) and **per-request lane selection** behind the same OpenAI endpoint.

---

## Hardware

### Primary target — Strix Halo APU
- **AMD Ryzen AI MAX+ 395** APU (Strix Halo)
- iGPU: `gfx1151` Radeon 8060S, ROCm 7.2 + Vulkan
- NPU: XDNA 2 / AIE2P, FW 1.1.2.65, in-tree `amdxdna` (kernel 7.0+)
- 124 GB LPDDR5x unified memory, ~256 GB/s
- Linux 7.0+ kernel with in-tree `amdxdna` (CachyOS / Arch tested)

### Also works — discrete-GPU boxes
- **Radeon RX 9070 XT** (`gfx1201`, RDNA 4) on Ryzen 7 9800X3D verified 2026-04-28. `install.sh`'s `has_xdna_npu()` autodetect skips the FLM/XRT NPU lane on dGPU boxes; the iGPU/dGPU lane (`llamacpp:rocm` / `llamacpp:vulkan`) lights up.
  - Bonsai-1.7B-IQ1_S decodes at **~413 tok/s** on the 9070 XT (vs ~281 on the 8060S iGPU at the same model)
  - Qwen3.5-35B-A3B-IQ2_XXS holds **~131 tok/s** decode on long output (vs ~73 on the iGPU) — a 35B-class MoE running on a single 16 GB consumer card
  - Full sweep + side-by-side: [`benchmarks/RESULTS-9070xt-2026-04-28.md`](benchmarks/RESULTS-9070xt-2026-04-28.md)
- Other RDNA 3 / RDNA 4 ROCm-supported AMD GPUs should work the same way; the install script doesn't pin to a specific `gfx`.

### Future slot — drop-in NPU on dGPU boxes
- **Axelera Metis-B (16 GB / 214 TOPS)** is the natural NPU add-in for the ryzen box once the 16 GB variant ships. Would give a non-Strix-Halo machine a discrete-NPU lane equivalent to (or beyond) the iGPU box's XDNA 2 — closing the lane matrix on dGPU systems.

## AMD GAIA integration

[AMD GAIA](https://amd-gaia.ai) is AMD's local desktop UI for AI agents on Ryzen AI hardware. Install via `uv tool install --with 'amd-gaia[ui]' amd-gaia`, point it at `http://127.0.0.1:13305/v1`, done. `1bit up` will launch GAIA automatically if it's on `$PATH`. Default agent model on this box is `Qwen3.5-35B-A3B-IQ2_XXS` via the `LEMONADE_MODEL` env var.

## Layout

```
.
├── install.sh                      # bootstrap (idempotent, --dry-run supported)
├── scripts/1bit                    # control-plane CLI
├── benchmarks/                     # bench scripts + RESULTS-*.md
├── 1bit-site/                      # 1bit.systems landing page (Cloudflare Pages)
└── docs/
    ├── model-priority.md           # 1-bit > 2-bit > rest selection policy
    └── reddit-npu-gate.md          # the strikethrough Reddit post / receipts
```

## What lives outside this repo

- **`lemond` (Lemonade Server)** — installed via `paru` from AUR, serves on `:13305`. Backends cached at `~/.cache/lemonade/bin/`.
- **`flm` (FastFlowLM)** — installed via `pacman` from `cachyos-extra-znver4`, drives the NPU lane.
- **ROCm 7.2.x** — installed via `pacman` (`rocm-hip-sdk`).
- **XRT + `amdxdna`** — installed via `pacman`.
- **Models** — pulled from HuggingFace (`lilyanatia/*`, `Manojb/*`, `unsloth/*`, `gianni-cor/*`, etc.) into Lemonade's cache and registered in `~/.cache/lemonade/user_models.json`. See [`docs/model-priority.md`](docs/model-priority.md) for the tiering policy.

## Receipts

- [`docs/reddit-npu-gate.md`](docs/reddit-npu-gate.md) — the strikethrough Reddit post on AMD's Linux NPU LLM gate, three corrections kept visible because the failure mode is more interesting than the original argument.
- [`benchmarks/RESULTS-stack-2026-04-28.md`](benchmarks/RESULTS-stack-2026-04-28.md) — unified-stack bench, both lanes through one endpoint.
- [`benchmarks/RESULTS-qwen3.5-35b-quant-2026-04-28.md`](benchmarks/RESULTS-qwen3.5-35b-quant-2026-04-28.md) — sub-2-bit vs Q4 head-to-head on the same 35B MoE.

## License

MIT. See [`LICENSE`](LICENSE).
