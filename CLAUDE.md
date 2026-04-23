# CLAUDE.md — conventions for 1bit-systems

Conventions for any Claude (or human) agent working inside this repo.
Keep terse; when in doubt, follow `ARCHITECTURE.md`.

## Hard rules

- **Rule A — no Python at runtime.** Python is fine for scripts that run
  once on a dev box (requantizers, analysis notebooks), never inside a
  systemd unit or a path that serves HTTP. If a component needs orchestration,
  write it in Rust.
- **Rule B — C++20 only for kernels.** All HIP kernels stay in the
  `rocm-cpp/` subtree of this monorepo (folded 2026-04-20 via
  `git subtree add`; history preserved). Do NOT reimplement kernels in
  Rust. FFI through `1bit-hip`. The standalone
  `bong-water-water-bong/rocm-cpp` mirror stays readable for one grace
  week, then archives. Old `stampby/rocm-cpp` already archived,
  `stampby` handle retired.
- **Rule C — hipBLAS is banned** in the runtime path. Native Tensile kernels
  only. If you find yourself reaching for hipBLAS, step back and port the
  kernel to `rocm-cpp` instead.
- **Rule D — Rust 1.88, edition 2024.** `rust-version` is pinned in the
  workspace `Cargo.toml`. Don't bump without a reason. (Bumped 1.86 →
  1.88 on 2026-04-23 because `home`, `image`, `time`, `zbus`, `zvariant`
  need MSRV ≥1.87.)
- **Rule E — NPU stack = ORT C++ with VitisAI Execution Provider (AMD
  official, XDNA2) as the primary lane; Peano + libxrt + aie-rt stays
  as the option for custom AIE kernels we write ourselves. IRON is
  permitted at compile-time.** Primary lane: ONNX Runtime C++ API with
  AMD's VitisAI EP does the .onnx → AIE lowering for us (matches the
  Vitis AI enterprise stack trickling down to consumer Ryzen AI).
  Custom-kernel lane (for ops VitisAI doesn't accelerate): AIE kernel
  authoring in C++ via `Xilinx/llvm-aie` (Peano); runtime dispatch via
  `libxrt` C++ (`xrt::kernel`, `xrt::bo`); tile driver `Xilinx/aie-rt`
  where we need bare-metal control. IRON is permitted at compile-time
  (reference-only). The retired FastFlowLM subprocess bridge +
  `1bit-xdna` crate were removed 2026-04-21 — see
  `project_npu_path_onnx.md`.
- **Target: aspirational 7/7 lanes, ~280 tok/s decode, NPU-prefill
  crossover at L ≥ 33.** Projected in `docs/wiki/Peak-Performance-Projection.md`.
  We do not settle for the conservative tier.

## Layout

```
crates/
  1bit-cli           unified operator CLI (1bit status/logs/doctor/...)
  1bit-core          model + tokenizer parsers (pure, no I/O beyond mmap)
  1bit-router        backend dispatcher + forward pass driver
  1bit-server        axum HTTP, OpenAI-compat, /ppl, /metrics
  1bit-agents        17-specialist async registry (one file, one source of truth)
  1bit-mcp           tokio stdio JSON-RPC bridge → 1bit-agents registry
  1bit-landing       marketing page on :8190, live /metrics probe
  1bit-lemonade      OpenAI-compat model gateway (/v1/models on :8200)
  1bit-helm          desktop client (egui/eframe) — formerly halo-gaia, renamed 2026-04-20
  1bit-hip    FFI into rocm-cpp
  1bit-mlx    Apple Silicon backend (feature-gated)
strixhalo/           dotfiles (systemd, caddy, bin, fish) — see strixhalo/README.md
packages.toml        pkg manifest consumed by `1bit install`
install.sh           fresh-box bootstrap
```

## Testing

- **Every new crate gets ≥3 in-crate tests** before it can be merged.
- **`cargo test --workspace --release` must stay green on main.** Current
  baseline: ~90 tests across 11 crates.
- **Integration tests that need real GPU / model weights** go behind
  `#[ignore]` + `--features real-backend` and are run by hand on the
  strixhalo box. CI on GitHub Actions runs only the default feature set.
- **Parity vs gen-1** is the ultimate cutover gate. Two harnesses:
  - `benchmarks/ppl-gen2.sh` — PPL on wikitext-103. gen-1 baseline 9.1607.
    gen-2 currently 9.1805 (delta +0.02, within ±0.05 tolerance). PASS.
  - `benchmarks/shadow-burnin.sh` — continuous /v1 vs /v2 argmax compare.
    Current rate: ~90% byte-exact after special-token fix. Logs to
    `~/claude output/shadow-burnin.jsonl`, state in `~/.local/share/1bit systems/`.

## Commit conventions

- Conventional Commits prefixes: `feat / fix / perf / docs / refactor / build / ci / chore / test`.
- **One logical change per commit.** "fix: X" PLUS "feat: Y" should be
  two commits, not one.
- **Messages have a "why" line.** Not "add tokenizer" — "add tokenizer
  special-token handling; gen-1 expects 128009 on EOT boundary and argmax
  diverges without it."
- **Always push to `bong` remote** (`git@github-bong:...`).
  `bong-water-water-bong` is the canonical handle. `stampby` is retired.

## What NOT to do

- **Don't commit tokens, session cookies, or bearer secrets.** Caddyfile's
  bearer lives in `/etc/caddy/Caddyfile` (root-only). The `strixhalo/caddy/Caddyfile`
  tracked copy has `sk-halo-REPLACE_ME` placeholders; never replace in git.
- **Don't add Python deps.** If you think you need Python, talk to the user.
- **Don't touch `ternary_gemv_halo.hip`** in rocm-cpp without a rocprof
  trace showing the improvement. Current kernel is at 92% of LPDDR5 peak.
- **Don't skip the KV-cache reset** at the start of each generation. The
  2026-04-19 SEGV was a `pos` accumulator bug that only showed up under
  sustained load (~200 completions in).
- **Don't add new warnings.** CI doesn't gate on `-D warnings` yet, but that
  gate is coming.

## Deploy flow

After changes that need a live restart:

```bash
cargo build --release --workspace                      # or -p <crate>
# 1bit-server: binary is held open by the running unit → stop first
systemctl --user stop strix-server
cp target/release/1bit-server ~/.local/bin/1bit-server-real
systemctl --user start strix-server
# other binaries install via cargo install --path crates/<name>
```

Use `1bit install <component>` from `packages.toml` when in doubt — it
does the stop/copy/start cycle correctly.

## Agent / subagent ground rules

When delegating to a background agent:

- Give it **exact file paths + line numbers** where possible.
- Tell it what to **not** touch (other crates, config files, READMEs).
- Ask for a **≤200-word report**, not a full transcript.
- **Trust but verify.** The agent's summary describes intent, not outcome.
  Always re-run tests + inspect the actual diff before claiming success.

## Memory

Per-project persistent memory lives outside this repo at
`~/.claude/projects/-home-bcloud/memory/`. Don't duplicate that here.
Load-bearing memory entries for this repo: `project_strix_ai_rs.md`,
`project_bitnet_live_bench.md`, `project_ppl_harness.md`,
`project_1bit_paper_techniques.md`.
