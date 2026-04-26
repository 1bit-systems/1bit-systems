# 1bit systems architecture

This document is the long-form companion to the [README](./README.md). It describes how the three runtime pillars fit together, what each crate in this workspace is responsible for, how features gate the backends, and how the service is laid out on the box.

> **Status — post-cutover (v0.1.0, 2026-04-24).** The shadow burn-in phase closed before v0.1.0. Gen-2 Rust `1bit-server` on `:8180` is the production LLM path on Strix Halo. Gen-1 C++ `bitnet_decode` (`:8080`) is retired — its systemd unit is disabled and `:8080` does not bind in production. This document previously described a live gen-1/gen-2 split; that split no longer exists and the sections below have been updated to match the shipping reality. See the [Cutover history](#cutover-history-retained-for-reference) section at the end for the original burn-in plan.

## The three pillars

The stack has three pillars that run at inference time — kernels, Rust caller, agents/services. Upstream projects we borrow ideas from (Lemonade, MLX, composable_kernel, etc.) are **references**, not runtime pillars. "Pillar" means "something the serving process actually executes."

| # | pillar | repo | language | role |
|---|---|---|---|---|
| 1 | AMD HIP kernels | `rocm-cpp/` subtree (folded 2026-04-20) | C++20 / HIP | ternary GEMV, split-KV FD attention, RMSNorm, RoPE, SiLU, KV cache |
| 2 | Rust caller | this workspace (`1bit-halo-workspace`) | Rust 1.88, edition 2024 | CLI, HTTP (`1bit-server` on `:8180`), router, loader, tokenizer, chat template |
| 3 | Agents + services | this workspace (`1bit-agents`, `1bit-mcp`, `1bit-lemonade`, `1bit-landing`, voice sidecars) | Rust | specialist pool, MCP bridge, model gateway, landing page, STT/TTS |

Rule A: all three pillars are bare-metal C++ / Rust with zero Python at runtime. Pillar 1 is a compiled shared object (`librocm_cpp.so`) linked into pillar 2 via the `1bit-hip` FFI crate.

### Dev-only side paths (not pillars, not supported deployment targets)

- **MLX (Apple Silicon)** — `bong-water-water-bong/bitnet-mlx-rs` (fork of `leizerowicz/bitnet-mlx.rs`), wrapped by our `1bit-mlx` crate and gated behind `--features mlx-apple`. **Dev-only path; not a supported deployment target.** MLX exists for Mac-side development ergonomics (editor integration, quick iteration on tokenizer / chat-template / sampler without a strixhalo box in reach). Shipping target is Strix Halo gfx1151 plus the wider RDNA3 / RDNA3.5 / RDNA4 Wave32-WMMA arches covered by the `rocm-cpp` fat-binary. Nothing in the release artefacts runs on MLX.

### Upstream references (not runtime, not pillars)

- **Lemonade / Lemonade-SDK** (Python, `lemonade-sdk/lemonade`) — caller-side reference for OpenAI-compat surface patterns and the earlier Gaia UI. We fork/mirror it for tracking but do not import or invoke it from the serving path. Our own Rust `1bit-lemonade` crate (pillar 3) is a separate implementation of an OpenAI-compat model gateway, not a port of the Python Lemonade runtime. See `docs/wiki/Lemonade-Compat.md` and `docs/wiki/AMD-GAIA-Integration.md` for the tracking notes.
- **composable_kernel** — HIP kernel idioms we read; not linked.
- **TheRock ROCm** — system dependency; not vendored.

## Data flow

```
                   HTTP client (OpenAI-compat)
                              |
                              v
                   +----------+----------+
                   |    Caddy (TLS)      |
                   |  /v1/*  ->  :8180   |   <- 1bit-server (Rust, production)
                   +----------+----------+
                              |
                              v
                   +----------+----------+
                   |   1bit-server       |   axum, tokio, OpenAI-compat
                   +----------+----------+
                              |
                              v
                   +----------+----------+
                   |   lemond       |   dispatches per Backend trait
                   +----------+----------+
                              |
                    feature:  |
                    real-     |
                    backend   v
                   +----------+----------+
                   |   HipBackend        |
                   |   (1bit-hip)        |
                   +----------+----------+
                              | extern "C"
                              v
                   +----------+----------+
                   |   rocm-cpp          |
                   |   (HIP kernels,     |
                   |    gfx1151 +        |
                   |    RDNA3/3.5/4)     |
                   +---------------------+

1bit-core: tokenizer, sampler, chat template, loader (backend-agnostic)
1bit-agents + 1bit-mcp: sit alongside 1bit-server, not in the hot path
1bit-cli: operates on systemd units, reads packages.toml

Dev-only alt path (not shipped): --features mlx-apple routes through
MlxBackend → 1bit-mlx → bitnet-mlx-rs on aarch64-apple-darwin.
```

## Crate responsibilities

- **`1bit-cli`** — `halo` binary. `status / logs / restart / doctor / update / install / version`. Reads `packages.toml`, drives systemd units, queries the local HTTP server for liveness.
- **`1bit-core`** — GGUF loader, BPE tokenizer, sampler, chat-template renderer. Backend-agnostic. Shared by both `HipBackend` and `MlxBackend`.
- **`lemond`** — defines the `Backend` trait and picks an implementation at build time via cargo features.
- **`1bit-server`** — axum app. OpenAI-compat `/v1/chat/completions`, `/v1/completions`, `/v1/models`. Streams SSE. Binds `:8180` (loopback); fronted by Caddy on `:443`.
- **`1bit-agents`** — tokio-based agent bus for the specialist pool. Sits next to the server, not in the inference hot path.
- **`1bit-mcp`** — stdio JSON-RPC MCP bridge. Lets external Claude/MCP clients drive the local model.
- **`1bit-hip`** — `extern "C"` bindings to `rocm-cpp`. Builds against a system `librocm_cpp.so`. This is where pillar 1 meets pillar 2.
- **`1bit-mlx`** — Rust wrapper around the `bitnet-mlx-rs` fork. Only compiled on `aarch64-apple-darwin` targets with `--features mlx-apple`.

## Feature-gate matrix

| feature | target | backend | intended use |
|---|---|---|---|
| default (no extra features) | any | `NullBackend` (echoes request) | unit tests, CI on machines without a GPU |
| `real-backend` | x86_64-linux, gfx1151 | `HipBackend` → `rocm-cpp` | Strix Halo production |
| `mlx-apple` | aarch64-darwin (M-series) | `MlxBackend` → `bitnet-mlx-rs` | dev on Apple Silicon laptops |
| `real-backend` + `mlx-apple` | — | rejected at compile time | only one backend at a time |

The router picks the backend at build time, not runtime, so a given `1bit-server` binary only carries one inference path. CI builds all three variants.

## Systemd layout

User-scope units under `~/.config/systemd/user/`, installed by `1bit install <component>`.

| unit | binary | port | notes |
|---|---|---|---|
| `strix-server.service` | `1bit-server` | 8180 | `--features real-backend`; production LLM |
| `strix-lemonade.service` | `1bit-lemonade` | 8200 | OpenAI-compat model gateway |
| `strix-landing.service` | `1bit-landing` | 8190 | live status + landing page |
| `1bit-agents.service` | `1bit-agents` | internal | tokio bus, no HTTP |
| `1bit-mcp.service` | `1bit-mcp` | stdio | socket-activated |
| `1bit-halo-whisper.service` | halo-whisper | 8181 | STT (sliger B580 Vulkan in current topology) |
| `1bit-halo-kokoro.service` | halo-kokoro | 8182 | TTS (sliger B580 Vulkan in current topology) |
| `1bit-halo-sd.service` | sd.cpp | 8081 | SDXL image sidecar |
| `caddy.service` | Caddy | 443 | TLS front door, bearer check |
| `1bit-halo-bitnet.service` | gen-1 `bitnet_decode` | 8080 | **retired** — unit is disabled; retained for archive only |

`1bit status` polls the active set and reports a single rollup. `1bit logs <unit>` wraps `journalctl -u`.

## Chat template

The chat template is the single source of truth in `1bit-core::chat_template`. One turn renders as:

```
User: <msg><|eot_id|>Assistant:
```

Multi-turn concatenates rendered turns with no extra separators. A parity test in `1bit-core` renders a known conversation and byte-compares against a fixture captured from the retired gen-1 path, keeping wire format stable across any future backend swap.

## Cutover history (retained for reference)

This section documents the gen-1 → gen-2 cutover that happened before v0.1.0. Both gates were satisfied and the cutover landed; `:8080` is no longer used in production.

- **PPL parity on wikitext.** gen-2 hit **9.1805** vs gen-1 baseline **9.1607** — delta +0.02, within the ±0.05 tolerance. PASS.
- **Shadow-traffic burn-in.** Sustained `/v1` (gen-1) vs `/v2` (gen-2) argmax comparison ran through 2026-04-21 → 04-23. Byte-exact agreement reached **96.66%** across 1500+ rounds after the special-token fix; divergences attributable to sampler nondeterminism, not semantic drift. PASS.
- **Cutover.** v0.1.0 (2026-04-24) promoted `1bit-server` to own `/v1/*` on `:8180`; gen-1 systemd unit disabled. Caddy no longer splits paths.

The historical Caddyfile split, burn-in tap, and `/v2/*` shadow surface have been removed from production config. The `benchmarks/shadow-burnin.sh` harness is retained so a future backend swap (e.g. an NPU-backed path) can re-use the same comparator.
