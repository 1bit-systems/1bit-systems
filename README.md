<div align="center">

# 1bit systems

### the 1-bit monster — local AI inference, bare metal, no python at runtime

**rocm c++ · ternary weights · fused HIP kernels · wave32 wmma · rust orchestration · zero telemetry · zero cloud**

*stamped by the architect*

[![CI](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Rust 1.86](https://img.shields.io/badge/rust-1.86-orange.svg)](./Cargo.toml)
[![Platform: Strix Halo gfx1151](https://img.shields.io/badge/platform-Strix%20Halo%20gfx1151-red.svg)](./docs/wiki/Why-Strix-Halo.md)
[![Website: 1bit.systems](https://img.shields.io/badge/web-1bit.systems-7c3aed.svg)](https://1bit.systems)

*"I know kung fu."* — the whole stack (router · HTTP · agents · MCP · desktop · package manager) in one Cargo workspace, sitting on hand-written HIP kernels, eating LPDDR5 at 92% of theoretical peak.

</div>

---

## what is this

1bit systems is the **install path for the 1-bit monster** — a full local AI stack that runs entirely in native code on AMD Strix Halo. no python at runtime. no cloud. no telemetry. no subscriptions.

one mini-PC in a closet, one binary per service, a 1.58-bit ternary LLM answering questions over OpenAI-compatible HTTP. no containers. no dial-home. **bring your own APU.**

the kernels are hand-written HIP targeting **gfx1151** (Strix Halo iGPU). everything above them — router, HTTP server, agent bus, MCP bridge, desktop client, package manager — is **Rust**. `systemd` supervises it. Caddy fronts it. when `1bit status` prints a full column of green dots, that's your household AI.

*"I know kung fu."*

## numbers that matter

no fudging, no "measured on a different box," no asterisks the size of Nebraska.

**Strix Halo reference (Radeon 8060S, gfx1151, 128 GB LPDDR5X):**

| metric | value | note |
|---|---|---|
| **decode @ 64-tok** | 66.8 tok/s | halo v2 — 2B BitNet 1.58, greedy |
| **ternary GEMV bandwidth** | 92% of LPDDR5 peak | rocprof on gfx1151 |
| **WMMA FP16 peak** | 50.2 TFLOPS | register-resident probe, 42% of theoretical |
| **split-KV FD attention** | 10.25× @ L=2048, 11.98× @ L=8192 | vs single-block, 1-ULP equiv |
| **PPL on wikitext-103** | 11.98 (4095-tok single-pass) · 9.16 (chunk-1024) | gen-1 baseline 9.1607 |
| **shadow-burnin byte-exact** | 96.67% | gen-1 ↔ gen-2 argmax parity |
| **power draw, perf/W** | 73 W, **0.90 tok/s/W** | silent-closet territory |

**RX 9070 XT cross-arch (Navi 48, gfx1201, GDDR6, ryzen host):** second target is bit-exact on the production path.

| metric | value | note |
|---|---|---|
| **decode @ 64-tok** | 78.6 tok/s | +18% over gfx1151 |
| **WMMA FP16 peak** | 138 TFLOPS | 81% of theoretical (RDNA4) |
| **WMMA INT8 peak** | 288 TOPS | 2.1× FP16 (RDNA4 widened INT8 pipe) |
| **split-KV FD @ L=8192** | **22.02×** | GDDR6 streams KV |
| **PPL on wikitext-103** | **11.9758** (identical to Strix) | bit-exact cross-arch |
| **power draw, perf/W** | 176 W, 0.44 tok/s/W | absolute winner, efficiency loser |

**Rule A:** 0 Python on the serving path. Period.

raw methodology: [benchmarks wiki](./docs/wiki/Benchmarks.md) · [peak projection](./docs/wiki/Peak-Performance-Projection.md) · [raw JSON outputs](./benchmarks/data/) (WMMA peak · cross-arch PPL · attention sweep · shadow-burnin · power + long-ctx — all unprocessed, re-runnable)

## the stack

```
┌──────────────────────────────────────────────────────────┐
│  any OpenAI-compatible client                             │
│  curl · DSPy · Open WebUI · LibreChat · Claude Code MCP   │
└────────────────────────────┬─────────────────────────────┘
                             │ HTTPS (Caddy, :443)
┌────────────────────────────▼─────────────────────────────┐
│  1bit-server (axum, Rust)              :8180 /v2/*       │
│  1bit-router → 1bit-core → 1bit-hip                      │
├──────────────────────────────────────────────────────────┤
│  17 rust specialists (agent bus + MCP bridge)            │
├──────────────────────────────────────────────────────────┤
│  rocm-cpp (HIP kernels, gfx1151, zero hipBLAS)           │
│  ternary GEMV · RMSNorm · RoPE · split-KV FD attention   │
├──────────────────────────────────────────────────────────┤
│  whisper.cpp (STT) · kokoro (TTS) · sd.cpp (image)       │
├──────────────────────────────────────────────────────────┤
│  ROCm 7.13 · gfx1151 · wave32 WMMA                       │
├──────────────────────────────────────────────────────────┤
│  CachyOS · systemd · btrfs                               │
└──────────────────────────────────────────────────────────┘
```

> *every layer is someone else's lego block if they want it. take the whole monster or take one piece.*

## what you get

### the engine
- **1bit-server** — OpenAI-compatible HTTP on `:8180/v2`. chat completions, models list, bearer auth optional, SSE streaming.
- **bitnet_decode** — the C++ gen-1 server on `:8080/v1`, held for parity while gen-2 bakes in.
- **HIP kernels** — fused ternary MatMul, RMSNorm, SiLU, RoPE, split-KV Flash-Decoding attention. wave32 WMMA. no CK, no hipBLAS at runtime.
- **`.h1b` + GGUF loaders** — ternary weights, memory-mapped, zero-copy. IQ2_S via GGUF for gen-1 compat.

### the agents
- **17 Rust specialists** on an async registry. each one job, each one trait impl. message bus with tamper-evident journal.
- **MCP bridge** — tokio stdio JSON-RPC 2.0. tool calls from any MCP client land on the registry.
- **consent-verification gate** — warden enforces policy/intent/consent/bounds. structural, not advisory.
- **hash-chained audit log** — every inbound and outbound message SHA-256 chained, genesis-seeded per session.

### the model + training
- **halo-1bit** — absmean quantization, QAT with STE, distillation from bf16 teachers. See the training subtree for the pipeline.
- **Sparse-BitNet** — 3:4 N:M sparsity on top of 1.58-bit, targeting 1.25 effective bits per weight. Run 4 on an H200 pod.
- **BitNet v2** — Hadamard-native W1.58 A4. planned next.

## quickstart

*buckle up.*

```sh
# build the whole workspace
cargo build --release --workspace

# install + supervise the services (reads packages.toml, drives systemd)
1bit install core
1bit status

# talk to the model over OpenAI-compat HTTP
curl -s http://127.0.0.1:8180/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "halo-1bit-2b",
    "messages": [{"role": "user", "content": "why ternary?"}]
  }'
```

skip the package manager, *go straight to the matrix*:

```sh
cargo run --release -p 1bit-server --features real-backend
```

(that's the red pill. `--features real-backend` wires HIP, loads the weights, and throws you onto gfx1151 with no safety harness. default features give you a `NullBackend` echo server — the blue pill — which is what CI runs.)

full install + first-boot walkthrough: [installation guide (wiki)](./docs/wiki/Repo-Layout.md) · [3-minute demo script](./DEMO.md)

### install via AppImage *(single file, no build)*

prebuilt single-file bundle of the user-facing Rust binaries. ROCm itself is **not** bundled — `1bit` itself runs, but `1bit install core` still needs a host `librocm_cpp.so` from the source bootstrap.

```sh
# download → verify → symlink to ~/.local/bin/1bit
curl -fsSLO https://1bit.systems/dl/1bit-systems-0.1.0-x86_64.AppImage
echo "cb24880525750d73bb9bc75c5db993e8aa7e83837165c80a442526c67226f6eb  1bit-systems-0.1.0-x86_64.AppImage" | sha256sum -c -
chmod +x 1bit-systems-0.1.0-x86_64.AppImage
ln -sfn "$PWD/1bit-systems-0.1.0-x86_64.AppImage" ~/.local/bin/1bit
1bit --version
```

the scripted path — `INSTALL_MODE=appimage ./install.sh` — does the same three steps and verifies against [1bit-site/releases.json](./1bit-site/releases.json). works on any x86_64 Linux with glibc ≥ 2.31 (Ubuntu 20.04, Debian 11, CachyOS).

symlink the AppImage to any of the 20 bundled binary names — `1bit-helm`, `1bit-halo-helm-tray`, `1bit-watch-discord`, `1bit-mcp-patreon`, `1bit-voice`, … — and it dispatches via `$ARGV0`. run `<AppImage> --list` to see them all.

## lego blocks

pick what you want. drop the rest.

| crate | role | status |
|---|---|---|
| `1bit-cli` | unified ops CLI: `status / logs / restart / doctor / update / install / chat / bench / ppl / say / version` | shipped |
| `1bit-core` | `.h1b` + `.htok` parsers, GGUF loader (IQ2_S), sampler, chat template | shipped |
| `1bit-router` | backend dispatcher (`HipBackend` / `MlxBackend`), format sniffing | shipped |
| `1bit-server` | axum HTTP, OpenAI-compat + `/ppl` + `/metrics` | shipped |
| `1bit-agents` | 17-specialist async registry, `TypedSpecialist` + JsonSchema | shipped |
| `1bit-mcp` | MCP bridge, stdio JSON-RPC | shipped |
| `1bit-landing` | marketing page + live `/metrics` probe on `:8190` | shipped |
| `1bit-lemonade` | OpenAI + Lemonade-SDK compat gateway on `:8200` | shipped |
| `1bit-helm` | egui/eframe desktop client (formerly halo-gaia) | shipped |
| `1bit-hip` | FFI → `rocm-cpp` ternary GEMV + Flash-Decoding attention | shipped |
| `1bit-mlx` | FFI → `bitnet-mlx-rs` (Apple Silicon, feature-gated) | shipped |

all eleven compile under default features with **zero** ROCm deps; `link-rocm` / `real-backend` / `mlx-apple` are the opt-in feature gates. CI builds all three variants.

## philosophy

> every piece snaps in and snaps out. no hard dependencies. no vendor lock-in. no cloud tethers.

python shipped the LLM era. C++ and Rust own the next one. python at training time is fine; python at runtime is a liability on hardware you own. **1bit systems has zero python at runtime.**

the AI industry wants you renting someone else's computer. we think you should own the whole stack — the hardware, the models, the weights, the pipeline. when you control your own software, you control your own destiny.

*"they get the kingdom. they forge their own keys."*

## privacy

**zero telemetry. zero tracking. zero data collection.** nothing phones home. your data stays on your machine.

paid API providers are supported through the router with your own keys — but that's your choice, not our default. local-first means local-first.

*"there is no cloud. there is only zuul."*

## clients

*show me the money.* `1bit-server` speaks plain OpenAI chat-completions on `:8180`, so off-the-shelf clients work out of the box. point them at `http://strixhalo.local:8180/v1` — or, through Caddy, `https://strixhalo.local/v2/...` with the halo bearer token.

### DSPy (Stanford)

```python
import dspy

lm = dspy.LM(
    "openai/halo-bitnet-1.58",
    api_base="http://strixhalo.local:8180/v1",
    api_key="halo-local",
    model_type="chat",
    cache=False,
)
dspy.configure(lm=lm)

qa = dspy.ChainOfThought("question -> answer")
print(qa(question="Why is ternary BitNet memory-bound on Strix Halo?").answer)
```

`1bit-mcp` tools are directly consumable via `dspy.Tool.from_mcp_tool(...)` — no shim needed. Rule A is untouched: python runs on the caller, `1bit-server` stays Rust.

### Open WebUI / LibreChat

any OpenAI-compat chat UI works. in Open WebUI:

```
Settings → Connections → Add OpenAI API
  Base URL: https://strixhalo.local/v2
  API Key:  sk-halo-<your-token>
  Model:    halo-1bit-2b
```

full RAG, multi-conversation, document chat, and MCP tools — Linux/macOS/Windows, zero server-side shim. LibreChat works identically through its `librechat.yaml` `endpoints.custom` block. both are the blessed desktop clients until the native `1bit-helm` hits feature parity.

## roadmap

- **near-term** — Sherry 1.25-bit weight packing (bytes-read reduction), BitNet v2 Hadamard activation quant (W1.58A4), Medusa speculative decoding heads. see [BitNet v2 Hadamard plan](./docs/wiki/BitNet-v2-Hadamard-Plan.md), [Medusa integration plan](./docs/wiki/Medusa-Integration-Plan.md), [Sherry default decision](./docs/wiki/Sherry-Default-Decision.md).
- **medium** — voice loop end-to-end (whisper.cpp STT streaming + Kokoro TTS), full sd.cpp image-gen wiring, `1bit-helm` as the first-class desktop client.
- **longer** — XDNA 2 NPU lane via ORT C++ + VitisAI EP for prefill. see [why no NPU yet](./docs/wiki/Why-No-NPU-Yet.md) and [NPU kernel design](./docs/wiki/NPU-Kernel-Design.md) for the current state of the Linux NPU stack.
- **distribution (post-launch)** — three install lanes: source build (canonical), **AppImage** (single-file, system-ROCm required), and **Flatpak** on Flathub (sandboxed). model weights download-on-first-run, cached under `.halo/models/`.

## docs

| doc | what it covers |
|---|---|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | data flow, crate map, feature gates, systemd layout, cutover plan |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | how to help, code style, testing |
| [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) | — |
| [SECURITY.md](./SECURITY.md) | responsible disclosure |
| [CHANGELOG.md](./CHANGELOG.md) | — |
| [CLAUDE.md](./CLAUDE.md) | conventions for AI agents (the hard rules A/B/C/D/E) |
| [DEMO.md](./DEMO.md) | 3-minute cold-open walkthrough |
| [CUTOVER.md](./CUTOVER.md) | gen-1 → gen-2 traffic flip criteria |
| [docs/wiki/Home.md](./docs/wiki/Home.md) | one page per architectural decision, plus FAQ and integration guides |

## how to help

we're gonna need a bigger box. contributions welcome from anyone running a Strix Halo (or any AMD APU — the CPU aggregator lane has open seats).

- **file an issue** with a reproducible case and `1bit doctor` output.
- **send a patch** — one logical change per commit, Conventional Commits.
- **run the benchmark** on your hardware. `1bit bench` output against a clean install is gold for the perf table.
- **test client compatibility** — if you wire Sorana, Aicono, TabNeuron, Hermes Agent, or anything else against `1bit-server`, document the config delta in an issue.

see [CONTRIBUTING.md](./CONTRIBUTING.md) and [CLAUDE.md](./CLAUDE.md) before sending code.

## acknowledgements

- **[Microsoft Research](https://github.com/microsoft/BitNet)** — the b1.58-2B-4T ternary weights we run and the [paper](https://arxiv.org/abs/2402.17764) that started this.
- **[Light Heart Labs](https://lightheartlabs.io/)** — their [DreamServer](https://github.com/Light-Heart-Labs/DreamServer) is a reference for local-AI-first architecture, and **[@Lightheartdevs](https://github.com/Lightheartdevs)** is a collaborator on this repo.
- **upstream projects** — AMD ROCm, `stable-diffusion.cpp`, `whisper.cpp`, `kokoro.cpp`, axum, tokio, MLX, the Rust ecosystem. none of this ships without them.

## license

MIT. see [LICENSE](./LICENSE). ternary model weights follow their own upstream licenses (Microsoft MIT for BitNet b1.58-2B-4T).

---

<div align="center">

**website:** [**1bit.systems**](https://1bit.systems) · **status:** pre-public launch; private until the XDNA 2 NPU lane lands. we'll be back. · **handle:** [@bong-water-water-bong](https://github.com/bong-water-water-bong)

*"the 1-bit monster is already here. it just had to learn to count."* — **stamped by the architect**

</div>
