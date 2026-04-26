# Why this way + how — the design walkthrough

Long-form explainer paired with the reddit-relaunch announcement. Covers *why* each major decision was made and *how* the pieces actually plug together in practice.

If you've read the landing page and want the engineering story, this is that page.

## The architecture in one picture

```
                 ┌─────────────────────────────────────────────────┐
                 │                your laptop                       │
                 │  Open WebUI · LibreChat · DSPy · Claude Code     │
                 │  1bit-helm (egui) · your python notebook         │
                 └──────────────────────┬───────────────────────────┘
                                        │ OpenAI-compatible HTTP
                                        │
         ┌──────────────────────────────▼──────────────────────────┐
         │              strix halo mini-pc (the engine)            │
         │                                                          │
         │  caddy :443 ─┬─ /studio/* → static landing + wiki       │
         │              ├─ /v2/*     → 1bit-server :8180 (gen-2)    │
         │              ├─ /v1/*     → bitnet_decode :8080 (gen-1)  │
         │              └─ /lemon/*  → 1bit-lemonade :8200          │
         │                                                          │
         │  ┌────────────────── halo-workspace (rust) ─────────────┐│
         │  │ 1bit-cli  · 1bit-server  · lemond · 1bit-mcp    ││
         │  │ 1bit-core · 1bit-agents  · 1bit-landing · 1bit-helm  ││
         │  │ 1bit-lemonade · 1bit-hip · 1bit-mlx    ││
         │  └──────────────────────┬───────────────────────────────┘│
         │                         │ extern "C" FFI                 │
         │  ┌──────────────────────▼───────────────────────────────┐│
         │  │ rocm-cpp (C++ / HIP)                                 ││
         │  │ ternary_gemv · split-KV Flash-Decoding · rotorquant  ││
         │  │ h1b_loader · tokenizer                               ││
         │  └──────────────────────┬───────────────────────────────┘│
         │                         │ hipcc → gfx1151 ISA            │
         │                         ▼                                │
         │              AMD Radeon 8060S iGPU                       │
         │              (+ XDNA 2 NPU, deferred — see below)        │
         │                                                          │
         │  17 1bit-agents (tokio) — kernel rebuilds, changelog,    │
         │    issue triage, PR secret-scan, burnin harness          │
         └──────────────────────────────────────────────────────────┘
```

## Why this way — the six core decisions

### 1. Ternary weights, not FP16 or 1-bit

See [Why-Ternary](./Why-Ternary.md). Short version: 1.58-bit = 10× memory reduction over FP16 + near-zero accuracy loss (Microsoft's 2B-4T training proves it). Binary would save another 0.58 bits but loses the load-bearing *zero* that matches activation sparsity and gives the hardware-efficient branch (sign-flip instead of multiply).

### 2. AMD Strix Halo, not NVIDIA or Apple

See [Why-Strix-Halo](./Why-Strix-Halo.md). Short version: 128 GB of unified LPDDR5 at $2-3k — the only consumer-class box with enough memory for long contexts + enough bandwidth for ternary decode + low enough power for silent closet operation. NVIDIA caps at 24 GB VRAM per card and costs 3-5× more for equivalent memory. Apple M-series is close on specs but ROCm is open; Metal's ternary path lagged BitNet's release by months.

### 3. Rust for orchestration, C++ for kernels

See [Why-Rust](./Why-Rust.md). Short version: hipcc is the only mature toolchain for gfx1151 GPU kernels. Everything above that — HTTP, CLI, agents, formats, clients — wins in Rust: memory safety, `tokio` async, `serde` typed JSON, static binaries, no Python interpreter on the startup path. The FFI boundary between the two layers is 30 extern "C" functions and has been stable for months.

### 4. No Python at runtime

See [Why-No-Python](./Why-No-Python.md). Short version: gen-1 was Python, died on dependency churn and cold-start latency. Rust `1bit-server` starts in 200ms cold, weighs 2.4 MB, and catches JSON shape errors at the deserializer boundary instead of six levels deep in a 500. Python is welcome on the *caller* side (DSPy, notebooks); anything serving an HTTP request is Rust or C++.

### 5. Shadow-traffic burnin before the cutover

See [Why-Shadow-Burnin](./Why-Shadow-Burnin.md). Short version: before we flip `/v1/*` from the proven C++ server to the new Rust server, we fire the same prompts at both and diff byte-for-byte. Currently at 96.66% exact match across 10 000+ rounds. The ~3% that diverge is sub-ULP FP16 noise, not bugs. PPL parity (9.18 vs 9.1607) proves the math is right; burnin proves the stability under load.

### 6. NPU deferred, AMD issue filed

See [Why-No-NPU-Yet](./Why-No-NPU-Yet.md). Short version: XDNA 2 is on the box, `amdxdna` kernel driver works, but AMD Ryzen AI SDK 1.7.1 Linux doesn't list Strix Halo on its supported-SKU set. We filed [`amd/RyzenAI-SW#366`](https://github.com/amd/RyzenAI-SW/issues/366) asking them to add it. Even unblocked, the decode-on-NPU math loses to iGPU on memory bandwidth; the real payoff is prefill acceleration (tier a in the design doc).

## How — the end-to-end path of a request

Follow a single `halo chat "What is the capital of France?"` request through the stack:

1. **`halo chat`** (Rust binary, clap) reads stdin → one line → synthesizes an OpenAI chat-completions payload.
2. **HTTP POST** to `http://127.0.0.1:8180/v1/chat/completions`, bearer-less on loopback.
3. **`1bit-server`** (axum) receives the request. Route handler deserializes with `serde_json` into a strongly-typed `ChatRequest`. Clock starts for the `/metrics` latency histogram.
4. **`lemond`** picks the HIP backend (it's the only one compiled on this box). Locks a tokio Mutex around the shared KV cache. Resets `pos = 0` (per-request — fix from commit `de53544`).
5. **Tokenizer** (`1bit-core`) encodes the prompt. Llama-3 special tokens (`<|eot_id|>`, etc.) are recognized as single IDs, not byte-level BPE — the fix that took parity from 18% to 96%.
6. **Prefill loop** — one forward pass per prompt token, writing K/V into the FP16 KV cache. Uses `1bit-hip`'s safe wrappers around `rcpp_ternary_gemv_halo_f16` + RMSNorm + RoPE + split-KV Flash-Decoding.
7. **Decode loop** — sample from logits at temperature 0 (greedy argmax on host), append to `generated_ids`, write next K/V slot, stop on `<|eot_id|>` or `max_tokens`.
8. **Detokenize** the new IDs into text (1bit-core). Stop check on stop-token list *before* detokenization so `<|eot_id|>` doesn't leak into output bytes.
9. **Response assembly** — `usage` block with `prompt_tokens` / `completion_tokens` / `total_tokens`. Latency logged to `/metrics`.
10. **HTTP 200** with JSON body. Client parses `.choices[0].message.content`. Screen prints the reply.

Total wall-clock: ~200ms for a 10-token prompt + 10-token reply. Scales linearly with both.

## How the agents keep the stack alive

17 1bit-agents run in the background on tokio. Each is a `Specialist` impl with typed I/O (serde + schemars). They're exposed over MCP so Claude Code and DSPy can invoke them as tools. Four relevant to ops:

- **anvil** — polls `bong-water-water-bong/rocm-cpp`. On new commit: git fetch + cmake build + bench_kv_fd. Posts the build+bench summary to `#changelog` via Discord. If build fails, the tail of the log is posted so you see it when you check your phone.
- **librarian** — scans commit history across 1bit systems-core + rocm-cpp + agent-cpp. Appends Conventional-Commits-formatted lines to `CHANGELOG.md` + posts the delta.
- **quartermaster** — polls GitHub for issues with zero labels across our repos. Adds `needs-triage`. State file tracks which issues have been handled so it doesn't re-label.
- **magistrate** — scans open PRs for Conventional-Commits titles + secret patterns in the diff. Flags violations. Never silently drops a `gh pr diff` failure — we learned that lesson.

All four run under systemd timers, idempotent, silent when nothing's changed. You work on your app; they work on the stack.

## How the NPU would plug in (when unblocked)

Single-line change in `lemond`:

```rust
match phase {
    Phase::Prefill if npu_ready() => rcpp_prefill_ternary_aie(..),   // NPU
    Phase::Prefill                 => rcpp_prefill_ternary_hip(..),  // iGPU fallback
    Phase::Decode                  => rcpp_decode_ternary_hip(..),   // iGPU always
}
```

KV cache lives in LPDDR5; neither engine needs to hand off state. Expected prefill gain: 1.5-2× on short prompts. Decode stays iGPU because its bandwidth story wins.

The kernel behind `rcpp_prefill_ternary_aie` is the missing piece. Candidate paths:

- Write it in MLIR for IREE-AMD-AIE (community, open source). Multi-engineer-quarter.
- Write it in AIE assembly + dispatch via `xrt`. No public reference exists.
- Wait for Microsoft's first-party XDNA backend (issue #408 open since Feb 2026, zero replies).
- Route through ONNX Runtime + VitisAI Execution Provider (AMD official, XDNA2) for the ops VitisAI accelerates; fall back to CPU EP for the rest. *(2026-04-21 pivot — FastFlowLM bridge retired.)*

We're watching, not building. The iGPU path has too much runway left — Sherry, rotorquant KV compression, MedusaBitNet speculative heads — all higher-ROI than NPU porting.

## How the install stays simple

One script. What it does:

1. Check GPU is gfx1151, kernel is 7.x, `amdxdna` + `rocm-hip-sdk` installed.
2. Clone (or pull) `bong-water-water-bong/rocm-cpp`, build kernels with `cmake --build`. Takes ~2 minutes on a Strix Halo.
3. `cmake --build --preset release-strix` in cpp/. Another ~1 minute.
4. `install -Dm755 cpp/build/strix/cli/1bit ~/.local/bin/1bit` plus the other tower binaries (helm, landing, voice, echo, mcp). All land in `~/.local/bin/`.
5. `systemctl --user enable --now 1bit-halo-lemonade strix-landing`.
6. Symlink `/home/bcloud/halo-1bit/models/halo-1bit-2b.{h1b,htok}` to the real model dir (compat shim for a hardcoded path in the release binary — will be removed after next rebuild).
7. Run `halo doctor`. Green means you're live.

Visual feedback the whole way — banner, step counter (1/6, 2/6, …), progress-dot lines for long commands, ✓ with elapsed seconds at the end of each step, ⚠/✗ if anything fails.

## How to contribute

See [CONTRIBUTING.md](../../CONTRIBUTING.md). File an issue with `halo doctor` output. Run the bench on your box and post numbers. +1 the AMD issue. Translate a README. All of it helps.

The Rust monorepo is private until launch; request a read-only collaborator invite. The kernels (rocm-cpp) are public MIT.

## What "third time's the charm" actually means

- **v0.0** — Python + AMD Lemonade. Died on MLX ROCm + ternary mismatch.
- **v0.1** — C++ `bitnet_decode` + agent-cpp. Shipped on :8080. Discovered the hardcoded-path bug and the `--server` argv bug when an external tester tried to install on a fresh CachyOS box. Both fixed within 24 hours of report.
- **v0.2** — Rust workspace + gen-1 still running for burn-in. 96.66% parity. 11 crates, 121 tests. This is the one.

Versioned, measurable, reproducible. Each failure documented. Each fix shipped with a commit message explaining why, not just what.
