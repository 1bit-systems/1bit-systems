<div align="center">

# 1bit systems

### ternary inference for the rest of us

[![Latest Release](https://img.shields.io/github/v/release/bong-water-water-bong/1bit-systems?include_prereleases)](https://github.com/bong-water-water-bong/1bit-systems/releases/latest)
[![CI](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml)
[![GitHub downloads](https://img.shields.io/github/downloads/bong-water-water-bong/1bit-systems/total.svg)](https://github.com/bong-water-water-bong/1bit-systems/releases)
[![GitHub issues](https://img.shields.io/github/issues/bong-water-water-bong/1bit-systems)](https://github.com/bong-water-water-bong/1bit-systems/issues)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](./CONTRIBUTING.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Star History](https://img.shields.io/badge/Star%20History-View-brightgreen)](https://star-history.com/#bong-water-water-bong/1bit-systems)
[![AUR](https://img.shields.io/badge/AUR-1bit--systems--bin-1793d1.svg)](https://aur.archlinux.org/packages/1bit-systems-bin)

<br>

<img src="./1bit-site/assets/logo.svg" alt="1bit systems" width="220" />

<br>

### [Install](https://1bit.systems/install) · [Docs](https://1bit.systems/docs) · [Site](https://1bit.systems) · [Discord](https://discord.gg/1bit)

</div>

---

<sub><em>"Whoa."</em></sub>

You bought a Strix Halo because the spec sheet read like science fiction — 128 GB of unified LPDDR5x, Radeon 8060S, an XDNA2 NPU welded onto the die. Then you booted Linux and discovered that the cloud-AI ecosystem still thinks "local" means a 4090 and a 1500W PSU. We built this for the other crowd. The mini-PC-on-the-desk crowd. The closet-server crowd. The "I want a chat endpoint that doesn't phone home" crowd. 1bit systems is a full ternary inference stack tuned for one machine — `gfx1151` plus its NPU — written in C++ where it has to be fast and Rust where it has to be careful. No Python at runtime. No Docker on the serving path. No telemetry, ever.

<sub><em>"There is no spoon."</em></sub>

## comes in three flavors

* **`lemond`** — the canonical local AI server. C++ HTTP front door. OpenAI / Ollama / Anthropic API surfaces on port `:8180`. Dispatches per-recipe to wrapped backends including the in-process `rocm-cpp` Engine. Forked from `lemonade-sdk/lemonade` and patched in-house — every wedge stays here.
* **`1bit-services`** — the apps tower above lemond. Operator CLI, desktop helm, landing page, voice loop, MCP bridge, power profile control, retrieval pipeline, watchdog. All Rust. All bare metal.
* **`halo-arcade`** — a vanilla-JS canvas-game cabinet that ships in the same release. Because every good rig deserves a coin slot.

## built by

A two-person crew on a Strix Halo box, plus the kindness of strangers who write good open-source kernels. We use AMD hardware. We are not affiliated with AMD. Anything that looks like a partnership is just us reading their docs at 2am.

## getting started

1. **Install** — pick your medicine on [1bit.systems/install](https://1bit.systems/install). CachyOS and Arch are first-class. The AppImage works on any glibc ≥ 2.35 + ROCm 7.x host.
2. **Get models** — `1bit pull bonsai-1.7b` grabs the ternary weights. We bundle eight `.h1b` ternary models on day one. Bring your own GGUF if you swing that way.
3. **Run** — `1bit run bonsai-1.7b` opens a chat. `lemond` is already serving OpenAI-compat on `:8180`.
4. **Mobile (someday)** — there is no app yet. The plan is a Rust core wrapped in uniffi-rs for iOS + Android. We will tell you when it boots.
5. **Connect** — point any OpenAI-compatible client at `http://localhost:8180/v1`. Open WebUI, Claude Code, Continue, your weekend Bun script. They all just work.

```sh
curl -fsSL https://1bit.systems/install.sh | sh
1bit install core
1bit run bonsai-1.7b
```

<sub><em>"I know kung fu."</em></sub>

## apps + integrations

Native first, then everything else.

| native (we ship it) | description |
|---|---|
| [`lemond`](https://github.com/bong-water-water-bong/lemonade) | C++ HTTP server. Forked from lemonade-sdk. OpenAI / Ollama / Anthropic surfaces. |
| [`1bit-helm`](./crates/1bit-helm) | egui desktop client. Plasma SNI tray icon. Start / stop / status. |
| [`1bit-landing`](./crates/1bit-landing) | live `/metrics` probe + landing page on `:8190`. |
| [`1bit-voice`](./crates/1bit-voice) | sentence-boundary streaming voice loop (LLM SSE → TTS chunks). |
| [`1bit-echo`](./crates/1bit-echo) | browser WebSocket gateway over `1bit-voice`. |
| [`1bit-mcp`](./crates/1bit-mcp) | stdio JSON-RPC MCP bridge for Claude Code and friends. |
| [`1bit-power`](./crates/1bit-power) | `1bit power` — RyzenAdj wrapper, profile control. |
| [`halo-arcade`](./browser) | vanilla JS canvas games. The good kind. |

| third-party (it just works) | how |
|---|---|
| [Open WebUI](https://docs.openwebui.com/) | point at `http://localhost:8180/v1`. |
| [Claude Code](https://claude.com/claude-code) | Anthropic-compat surface. |
| [Continue](https://continue.dev/) | OpenAI-compat. |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | image gen on `:1234`, native HIP for SDXL. |
| [whisper.cpp](https://github.com/ggerganov/whisper.cpp) | STT on `:8190`. |
| [kokoro](https://github.com/olokobayusuf/kokoro.cpp) | TTS on `:8095`. |

## supported platforms

| platform | state |
|---|---|
| ![CachyOS](https://img.shields.io/badge/CachyOS-canonical-008B8B?logoColor=white) | first-class. We dev here. |
| ![Arch Linux](https://img.shields.io/badge/Arch%20Linux-supported-1793D1?logo=arch-linux&logoColor=white) | AUR `1bit-systems-bin`. |
| ![Fedora](https://img.shields.io/badge/Fedora-AppImage-294172?logo=fedora&logoColor=white) | AppImage path. ROCm 7.x on host. |
| ![Debian / Ubuntu](https://img.shields.io/badge/Debian%2FUbuntu-AppImage-A81D33?logo=debian&logoColor=white) | AppImage. .deb someday. |
| ![NixOS](https://img.shields.io/badge/NixOS-flake-5277C3?logo=nixos&logoColor=white) | `flake.nix` in tree. Untested by us. |
| ![macOS](https://img.shields.io/badge/macOS-dev--only-999999?logo=apple&logoColor=white) | MLX feature gate; dev-only path, not a deploy target. |
| ![Windows](https://img.shields.io/badge/Windows-not%20yet-0078D6?logo=windows&logoColor=white) | use `lemond` upstream until we port. |

## CLI

```sh
# chat with a ternary model
1bit run bonsai-1.7b

# list everything we know how to pull
1bit list

# get models
1bit pull halo-1bit-2b

# launch a connected app from the catalog
1bit launch claude

# stack health
1bit status
1bit doctor
1bit logs lemond
```

```sh
# multi-modality, dispatched by lemond's recipe registry
1bit run kokoro-v1            # TTS
1bit run whisper-large-v3     # STT
1bit run sdxl-turbo           # image gen
```

## hardware

The shipping target is a single SKU: **AMD Strix Halo, Ryzen AI MAX+ Pro 395, Radeon 8060S iGPU (`gfx1151`), XDNA2 NPU, 128 GB LPDDR5x.** That is the closet machine. Everything in this repo is tuned around its bandwidth, its kernels, its NPU control packets, its thermal envelope.

The fat-binary build covers eight Wave32-WMMA AMD arches in one ship — `gfx1151` plus the rest of RDNA3 / RDNA3.5 / RDNA4. RX 9070 XT (`gfx1201`) on a Ryzen host is the sibling target; same kernels, more bandwidth.

NPU path: we author AIE2P kernels in C++ via `Xilinx/llvm-aie` (Peano), dispatch through `libxrt`, and use IRON / MLIR-AIE at compile time. AMD's VitisAI EP is the primary lane when it lands on Linux STX-H; until then, the custom-kernel lane carries the load.

<sub><em>"Where we're going, we don't need racks."</em></sub>

## honest numbers

Strix Halo reference, single-stream decode, real wall-clock:

| metric | value |
|---|---|
| Bonsai 1.7B ternary, decode @ 64-tok | **104 tok/s** |
| Halo 1bit 2B baseline, decode @ 64-tok | 88 tok/s |
| ternary GEMV memory utilization | 92% of LPDDR5x peak |
| split-KV FD attention vs naive | 6.78× @ L=2048 |
| NPU ternary GEMV @ 512×512 | 0.27 ms |
| NPU i8 matmul @ 512×512 | 0.93 ms |
| ternary `.h1b` models bundled | 8 |
| supported AMD arches in one fat binary | gfx1151 + 7 (RDNA3/3.5/4) |

We are not going to put a benchmark grid here that we can't reproduce on demand. Numbers above are reproducible from `benchmarks/` against checked-in recipes. Anything else is in the [Benchmarks wiki](./docs/wiki/Benchmarks.md) with raw JSON and methodology. If a number isn't in this table, we either haven't measured it yet or don't trust the measurement.

## status, honestly

| lane | state |
|---|---|
| LLM · TTS · STT · image | shipping on `:8180 / :8095 / :8190 / :1234` |
| NPU toolchain (IRON + MLIR-AIE + Peano + libxrt, npu5) | axpy 160/160 green on Arch |
| NPU serve path (BitNet-1.58 end-to-end) | kernel authoring in flight |
| Sherry 1.25-bit retrain | mid-run, weights not landed |
| Reddit / public launch | ship-gated until the NPU lane goes live |
| Wan 2.2 video lane | upstream-blocked on sd.cpp 5D ggml |

If you came here from a Reddit post — there isn't one yet. We are not announcing until the NPU demo trips the gate.

## the rules of the house

- **Rule A.** No Python at runtime. Scripts on a dev box are fine. A systemd unit serving HTTP is not.
- **Rule B.** Kernels are C++20 only. They live in `rocm-cpp/`. Don't reimplement them in Rust.
- **Rule C.** hipBLAS is banned in the runtime path. Native Tensile kernels only.
- **Rule D.** Rust 1.88, edition 2024. Pinned in workspace `Cargo.toml`.
- **Rule E.** NPU stack is ORT C++ with the VitisAI EP as the primary lane; Peano + libxrt + aie-rt is the custom-kernel lane.

The full long-form lives in [`CLAUDE.md`](./CLAUDE.md) and [`CONTRIBUTING.md`](./CONTRIBUTING.md). They are short on purpose.

## connect a client

The server speaks OpenAI-compat. Anything that takes a `base_url` works.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8180/v1",
    api_key="not-used-but-required",
)

resp = client.chat.completions.create(
    model="bonsai-1.7b",
    messages=[{"role": "user", "content": "hello, ternary world"}],
)
print(resp.choices[0].message.content)
```

Pick your language on the [Clients wiki](./docs/wiki/Clients.md). Rust, Go, C++, Node, Ruby, PHP, Java, C#. They all dial the same port.

## standing on shoulders

We forked, patched, and bundled work from a lot of people. They didn't ask for our patches and we don't push them upstream — our improvements stay in our forks, theirs flow into ours. Asymmetric, friendly, no relationship overhead.

- [`lemonade-sdk/lemonade`](https://github.com/lemonade-sdk/lemonade) — C++ server skeleton. We forked `lemond` from here.
- [`ggml/llama.cpp`](https://github.com/ggml-org/llama.cpp) — kernel idioms.
- [`ggml/whisper.cpp`](https://github.com/ggerganov/whisper.cpp) · [`ggml/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp)
- [`olokobayusuf/kokoro.cpp`](https://github.com/olokobayusuf/kokoro.cpp) — TTS.
- [`Xilinx/mlir-aie`](https://github.com/Xilinx/mlir-aie) · [`Xilinx/llvm-aie`](https://github.com/Xilinx/llvm-aie) · [`Xilinx/aie-rt`](https://github.com/Xilinx/aie-rt) — NPU toolchain.
- [Microsoft BitNet](https://github.com/microsoft/BitNet) — original 1.58-bit reference.

## read more

- [Architecture-Deep](./docs/wiki/Architecture-Deep.md) — pillars, crate map, feature gates.
- [Benchmarks](./docs/wiki/Benchmarks.md) — raw JSON, cross-arch (9070 XT / gfx1201), methodology.
- [Why-Strix-Halo](./docs/wiki/Why-Strix-Halo.md) — hardware rationale, supported floors.
- [NPU-Kernel-Design](./docs/wiki/NPU-Kernel-Design.md) · [NPU-Unlock-20260423](./docs/wiki/NPU-Unlock-20260423.md) — AIE2P path.
- [Training-Runs](./docs/wiki/Training-Runs.md) — absmean QAT, Sparse-BitNet, BitNet v2 Hadamard.
- [Eight-Models-Roadmap](./docs/wiki/Eight-Models-Roadmap.md) — what's next on weights.

## license + footer

MIT. See [LICENSE](./LICENSE). Model weights follow upstream licenses (Microsoft MIT for BitNet b1.58-2B-4T, etc.).

We don't transfer anything off your box without you asking. When you `1bit pull`, we go to Hugging Face. That's it. No analytics, no crash reporters, no "anonymous usage statistics."

---

<div align="center">

[**1bit.systems**](https://1bit.systems) · [@bong-water-water-bong](https://github.com/bong-water-water-bong)

<sub><em>no LLMs were harmed making this. one almost was.</em></sub>

</div>
