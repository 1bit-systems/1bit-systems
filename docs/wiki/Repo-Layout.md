---
phase: implementation
owner: cartograph
---

# Repo layout — 1bit-systems monorepo

Canonical URL: `https://github.com/bong-water-water-bong/1bit-systems` (renamed from `halo-ai-rs` on 2026-04-20).

## Top level

```
1bit-systems/
├── Cargo.toml              Rust workspace manifest (~26 crates)
├── crates/
│   ├── 1bit-server/        OpenAI-compat HTTP (Rust, axum) — binds :8180
│   ├── 1bit-router/        backend dispatcher — iGPU / NPU / CPU
│   ├── 1bit-core/          model + tokenizer parsers (mmap, zero deps)
│   ├── 1bit-cli/           operator CLI — `1bit status / doctor / update / …`
│   ├── 1bit-hip/           FFI into rocm-cpp HIP kernels
│   ├── 1bit-mlx/           Apple Silicon backend (feature-gated)
│   ├── 1bit-onnx/          ORT C++ backend shim for the NPU lane
│   ├── 1bit-cpu/           CPU lane (aggregator path)
│   ├── 1bit-voice/         sentence-boundary TTS streaming orchestrator
│   ├── 1bit-echo/          WebSocket voice server (Opus 20 ms frames)
│   ├── 1bit-whisper/       STT via whisper.cpp FFI (feature-gated)
│   ├── 1bit-kokoro/        TTS via kokoro.cpp (stub-by-default)
│   ├── 1bit-agents/        17 typed specialists + discord/github watchers + dialectic memory
│   ├── 1bit-agents-legacy/ gen-1 agents, held for parity
│   ├── 1bit-mcp/           stdio MCP bridge
│   ├── 1bit-mcp-clients/   in-tree MCP client helpers
│   ├── 1bit-mcp-discord/   discord-surface MCP tools
│   ├── 1bit-mcp-linuxgsm/  game-server MCP tools
│   ├── 1bit-mcp-patreon/   patreon watch → Quartermaster → echo
│   ├── 1bit-lemonade/      OpenAI-compat gateway on :8200 (Hermes + AMD GAIA interop)
│   ├── 1bit-landing/       LAN dashboard on :8190 (live SSE telemetry)
│   ├── 1bit-helm/          native egui desktop client + SNI tray icon
│   ├── 1bit-power/         halo-power profile controller
│   ├── 1bit-pkg/           packages.toml driver (install/stop/start)
│   ├── 1bit-retrieval/     BM25 wiki full-text index
│   └── 1bit-halo-ralph/    gen-1 bridge
├── rocm-cpp/               [SUBTREE, post-fold] HIP kernels, C++20, gfx1151 + 1201
├── docs/wiki/              spec pages, one per decision — phase-tagged
├── strixhalo/              dotfiles: systemd units, Caddy config, bin/, fish shell
├── 1bit-site/              static website for 1bit.systems (CF Pages target)
├── packages.toml           pkg manifest consumed by `1bit install`
├── install.sh              fresh-box bootstrap
├── CLAUDE.md               hard rules + conventions for AI agents
├── README.md               public-facing repo README
└── CONTRIBUTING.md         Light Heart Labs attribution + contrib flow
```

## Crate naming — two-layer

Cargo package names can't start with a digit. So:

| Layer | Convention | Example |
|---|---|---|
| On-disk directory | `crates/1bit-<word>/` | `crates/1bit-server/` |
| Cargo package name | `onebit-<word>` | `[package] name = "onebit-server"` |
| Rust module path | `onebit_<word>` | `use onebit_server::X;` |
| Executable binary | `1bit-<word>` (or `1bit`) | `/usr/local/bin/1bit-mcp` |

`[lib] name = "onebit_foo"` is set explicitly in each crate's Cargo.toml to force the valid Rust identifier — Cargo would otherwise auto-derive `1bit_foo` which isn't a valid identifier.

Binary names CAN start with a digit (filesystem paths, not Rust idents), so `[[bin]] name = "1bit-mcp"` is fine.

## rocm-cpp subtree (post-fold)

HIP kernels live under `rocm-cpp/` as a git subtree. `1bit-hip`'s `build.rs` resolves `librocm_cpp.so` from:

1. `$ROCM_CPP_LIB_DIR` — explicit env override (CI / packaging)
2. `<workspace>/rocm-cpp/build` — canonical in-tree path
3. `$HOME/repos/rocm-cpp/build` — legacy-clone fallback
4. `/usr/local/lib`, `/usr/lib` — system install

Grace week: a parallel `bong-water-water-bong/rocm-cpp` standalone repo mirrors the subtree so external clones with `git submodule` pins keep working. Archived after one week without drift.

## Brand

Single brand across every surface (2026-04-20 lock-in, supersedes the earlier brand-vs-engineering split):

- **Brand:** `1bit systems` — hero, Reddit, Discord, marketing copy, repo, docs, README.
- **Domain:** `1bit.systems` — URL bar, `install.sh`, `wrangler.toml`. Always with dot.

Don't hyphenate. Don't spell out "one-bit". Don't write "1 bit" with a space. It is always "1bit systems" as the brand and "1bit.systems" as the address. Any earlier tagline variants are retired — don't reintroduce in new copy.

## Preserved halo-* references

Some names deliberately stay `halo-*` for historical / platform reasons:

| Name | Why preserved |
|---|---|
| `halo-ai-rs` git history | `git log` references land | redirect from renamed repo for 90 days |
| `halo-server-real` binary path | systemd ExecStart expects it; new `onebit-server` binary copied over this path (cosmetic only) |
| `halo-bitnet.service`, `halo-sd.service`, `halo-whisper.service`, `halo-kokoro.service`, `halo-agent.service`, `halo-anvil.timer`, etc. | gen-1 legacy systemd units (C++ bitnet_decode stack, still the `/v1/*` side of shadow-burnin parity gate). Don't rename until gen-1 retires. |
| `halo-agents::*` in some Rust comments | module was just renamed; some doc-strings still say halo-agents. Cosmetic. |
| `/var/log/caddy/halo-access.log` | log file on disk; renaming mid-run loses history. Will flip on next log-rotation. |

## See also

- `CLAUDE.md` — Rules A-E for the engineering discipline
- `docs/wiki/SDD-Workflow.md` — phase-gate model for specs
- `docs/wiki/VPN-Only-API.md` — mesh + bearer posture
- `project_monorepo_fold.md` memory — the fold decision + exact subtree script
- `project_brand_lockin.md` memory — brand naming rule
