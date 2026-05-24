# 1bit-systems — Wiki

> A toolbox-first local inference workbench for AMD Strix Halo: apps connect to a single OpenAI-compatible union endpoint (`1bit-proxy :13306`) that routes to Lemonade (multimodal/OmniRouter), FastFlowLM (XDNA NPU), or a toolbox-backed llama.cpp backend, with the finished single control plane still in progress.

## Current State

The stack is working on the reference Strix Halo box (Ryzen AI MAX+ 395, CachyOS):

- `lemond.service` — Lemonade Server on `:13305`, canonical multimodal/OmniRouter lane
- `flm.service` — FastFlowLM on `:52625`, serving `qwen3:1.7b --embed 1 --socket 20 --q-len 20`
- `1bit-proxy.service` — union OpenAI-compatible endpoint on `:13306`
- `open-webui.service` — secondary browser UI on `:3000`, pointed at `:13306/v1`
- GAIA — primary agent/control surface from `/home/bcloud/Applications/gaia-agent-ui.AppImage`

The toolbox-backed llama.cpp path (Vulkan RADV first, then ROCm 7.2.2) is the repair/bootstrap path for Ubuntu/Fedora hosts. The single control plane (backend registry, toolbox lifecycle management) is roadmap work not yet shipped. `install.sh` is Arch/CachyOS-first and idempotent. Conventional Commits are enforced; recent work includes OpenHands PR review workflow (Gemini Flash) and health checks on the app-integration guide.

## Start Here

- [[Architecture-Deep]] — read first: current runtime topology, request flow, service port map, and kernel/NPU lane boundaries
- [[Repo-Layout]] — directory structure, what lives where, brand naming conventions
- [[Why-Rust]] — the Rust-above / C++-below language split; read before touching any serving or orchestration code
- [[Why-No-Python]] — Rule A in depth with the full list of what is and is not banned, and why; read before touching any serving code
- [[Why-No-NPU-Yet]] — current NPU lane status: FastFlowLM as the intended XDNA serving lane, toolbox llama.cpp as the first repair path, IRON/MLIR-AIE for custom kernel authoring

## Hard Rules

These rules gate every review. Violating them is a bug.

- **Rule A: core serving stays Python-free.** Training, notebooks, build-time conversion, caller-side tools, and compatibility UIs are allowed. The core engine path we own is Python-free: proxy, kernels, native runtimes, and model hot paths. Open WebUI is allowed only as an isolated secondary UI behind the OpenAI-compatible endpoint.
- **Rule B: C++20 for kernels.** HIP code belongs in `rocm-cpp/`. *Only applies on the archived branch `archive/cpp-tower-2026-04-27`; the live tree intentionally has no `rocm-cpp/` tower.*
- **Rule C: hipBLAS is banned in the runtime path.** Port kernels to `rocm-cpp/` instead. *Same scope as Rule B — archive-branch only.*
- **Rule D: Rust 1.88+, edition 2024.** Bump with a reason.
- **Rule E: NPU has two lanes.** FastFlowLM is the live XDNA serving lane. Custom NPU kernels are IRON author-time → MLIR-AIE → Peano → xclbin → libxrt C++ runtime.
- **Compatibility surface is OpenAI.** Anything that breaks `:13306/v1` or `:13306/api/v1` for clients is a bug.
- **Don't add a Rust workspace, a C++ tower, or HIP kernels back.** That was archived as `archive/cpp-tower-2026-04-27`.
- **Don't expand scope.** Match the literal ask. Default to the minimum that delivers 1-bit inference.
- **Don't commit secrets.** CF tokens and gh tokens live in libsecret.

## Open Threads

- **Single control plane** — `1bit up` does not yet manage toolbox-backed `llama-server` lifecycle; backend registry and toolbox lifecycle are the next control-plane milestone (see [[Complete-Pack]] and `docs/control-plane-roadmap.md`).
- **NPU bootstrap dependency** — FastFlowLM is the intended NPU lane but is not the universal repair path; Ubuntu/Fedora users must go through toolbox llama.cpp first, verify `/dev/dri` and `/dev/kfd`, then enable the NPU lane (see [[Why-No-NPU-Yet]]).
- **Fork-everything maintenance** — all runtime dependencies live under `bong-water-water-bong/` forks tracked by 1bit-watchdog with a 24-hour dwell; active forks include lemonade, llamacpp-rocm, whisper.cpp, stable-diffusion.cpp, ggml, and 1bit-tts.cpp (see [[Fork-Everything]]).
- **Cloudflare Tunnel** — `api.1bit.systems` tunnel is staged (templates exist, UUID/auth not filled in) and deliberately disabled until Cloudflare Access or bearer auth is wired in front of the proxy (see [[Cloudflare-Tunnel-Setup]]).

## Article Index

| Article | What it covers |
|---------|----------------|
| [[Architecture-Deep]] | Runtime topology diagram (proxy → Lemonade/FLM/toolbox), service port map, request flow steps, and NPU two-lane boundary (FastFlowLM serving + IRON authoring). Note: kernel boundary (C++20 HIP in `rocm-cpp/`) describes `archive/cpp-tower-2026-04-27` only — live tree has no kernel layer. |
| [[Repo-Layout]] | Top-level directory tree, runtime shape diagram, kernel ownership rules, brand naming (`1bit systems` / `1bit.systems` / `1bit` CLI), and cross-links to related wiki pages |
| [[Why-Rust]] | The Rust-above / C++-below language split: Rust owns native orchestration; includes the current live proxy topology. Note: references to `rocm-cpp/` HIP kernels apply to `archive/cpp-tower-2026-04-27` only. |
| [[Why-No-Python]] | Rule A in full: what is banned in the core serving path, what is allowed (training, notebooks, caller-side, IRON author-time, Open WebUI as isolated UI), why (cold-start, dependency churn, silent failures, GIL, binary size), and enforcement |
| [[Why-No-NPU-Yet]] | Current NPU lane status: FastFlowLM on XDNA/XRT as the intended serving lane, toolbox llama.cpp as the first repair path, IRON→MLIR-AIE→Peano→xclbin→libxrt for custom AIE kernels, and decode bandwidth-vs-compute analysis |
| [[Development]] | The authoritative review baseline: five rules A–E stated in full, current operational endpoint map (toolbox/Lemonade/FLM/proxy/WebUI/GAIA URLs), and the target architecture diagram |
| [[Complete-Pack]] | Current lane inventory table (toolbox GGUF, Lemonade multimodal, FLM NPU, GAIA, Open WebUI, custom NPU kernels, website) with install entry point and Rule A–E fit statement |
| [[Installation]] | Repair-path install guide: toolbox-first path for Ubuntu/Fedora (Vulkan RADV → ROCm 7.2.2), native Arch/CachyOS `./install.sh` path, verification curl commands, endpoint reference table, start/stop commands |
| [[Clients]] | Client connection guide: union endpoint URLs, Open WebUI setup (native service and Docker paths), GAIA config, raw HTTP curl examples (list models, completion, embeddings), and Python/TypeScript/Rust SDK snippets |
| [[FAQ]] | Short answers to common questions: what the project is, hardware targets, current endpoints table, install steps, supported clients, the five rules summary, Python allowances, NPU status, stablecoin note |
| [[AMD-GAIA-Integration]] | AMD GAIA as the primary UI/control surface pointed at `:13306/api/v1`; historical naming-conflict resolution (halo-gaia → 1bit-helm); integration setup snippet; what GAIA does not replace |
| [[Hermes-Integration]] | NousResearch Hermes Agent as an external Python client on the user's laptop pointing at `1bit-proxy :13306`; MCP bridge wiring; Hermes skill/memory format analysis for porting to 1bit-agents; honest feature comparison |
| [[Lemonade-Compat]] | Lemonade's role as the native multimodal/OmniRouter lane on `:13305`; when to use Lemonade direct vs the proxy; app guidance (GAIA, Open WebUI, direct backend, FastFlowLM debugging); Rule A position |
| [[Add-Your-Own-App]] | How to attach caller-side apps to the stack: OpenAI-compatible HTTP contract, model discovery via `/v1/models`, rate-limit handling, TypeScript minimal-agent example, and where serving vs library apps belong |
| [[Cloudflare-Tunnel-Setup]] | Optional `api.1bit.systems` subdomain tunnel plan via `cloudflared`; currently staged but disabled (auth not wired); step-by-step operator setup; what not to expose; coexistence with Caddy |
| [[VPN-Only-API]] | 10-seat private beta access model: Headscale mesh as fence 1, per-user bearer token as fence 2; invite/revoke operator scripts; security posture table; public website vs mesh surface breakdown |
| [[Why-Caddy-Systemd]] | Why Caddy over nginx (ACME + HTTP/3 built in, 20-line config), why `tls internal` on LAN (Headscale mesh, no Let's Encrypt), why `systemd --user` (no root for restarts, `$HOME` state), why not Docker (cold start, GPU passthrough chore, attack surface) |
| [[Why-Strix-Halo]] | Hardware target rationale: 128 GB LPDDR5 unified memory at 256 GB/s, 40-CU RDNA 3.5 iGPU (gfx1151 with WMMA + `v_dot4_i32_i8`), XDNA 2 NPU, why not discrete GPU or Apple Silicon, software stack |
| [[Why-This-Way-How]] | Long-form repair path walkthrough: seven architectural decisions (one front door, toolboxes first, Lemonade owns multimodal, FLM is XDNA lane, Rule A, C++ kernels, NPU two lanes) and step-by-step request flow |
| [[Observability]] | Three observability levels: service logs via `journalctl` (lemond/flm/proxy/webui), kernel profiling via `rocprof` targeting 92% LPDDR5 bandwidth, model-quality PPL harness, live benchmark reference numbers |
| [[Fork-Everything]] | Ownership strategy: all runtime dependencies forked under `bong-water-water-bong/`; active fork table (lemonade, llamacpp-rocm, whisper.cpp, stable-diffusion.cpp, ggml, 1bit-tts.cpp); 24-hour dwell watchdog update flow |
| [[Ternary-on-AIE-Pack-Plan]] | Design doc for ternary weight packing and MAC on the XDNA 2 AIE2P tile array: 2-bit encoding, host-side Rust requantizer, on-tile Peano C++ unpack kernel, DMA descriptor gotchas, bandwidth-vs-compute crossover analysis |
| [[halo-agent-fleet]] | Autonomous agent fleet spec: binary + TOML config per agent, adapter/brain/tools architecture, SQLite conversation persistence, `systemd --user` template units, security defaults (allowlist, rate limit, mutating-tool confirmation) |
| [[tier-jwt-flow]] | Draft payment-to-JWT flow for a future premium API tier: BTCPay Lightning rail (9-step end-to-end diagram), stablecoin machine-client alternative, HS256 JWT shape, 90-day key rotation with dual-key overlap, webhook HMAC auth, threat model, and autonomous agent spend policy design |
| [[Crate-halo-lemonade]] | Historical: `1bit-lemonade` Rust gateway idea; now superseded — current stack is Lemonade direct on `:13305`, FastFlowLM on `:52625`, `1bit-proxy` on `:13306` |
| [[Crate-halo-helm]] | Historical: `1bit-helm` native desktop pane plan (renamed from `halo-gaia` 2026-04-20); current primary UI is AMD GAIA, secondary is Open WebUI |
| [[Crate-halo-landing]] | Historical: `1bit-landing` LAN dashboard plan; current operational checks use `1bit status` and `systemctl status lemond flm 1bit-proxy open-webui`; public website lives in `1bit-site/` |
| [[1bit-website-spec-v2]] | Historical 2026-04-21 WordPress site spec (predates toolbox-first repair path); useful for IA, URL structure, CPT schemas, FAQ bank, and glossary seed — do not treat shipping claims as current |
