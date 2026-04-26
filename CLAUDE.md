# CLAUDE.md — conventions for 1bit-systems

Conventions for any agent here. Terse; when in doubt see
`ARCHITECTURE.md`.

> "I know kung fu." Phase 2 tripped 2026-04-26. Rust is gone;
> everything above the kernels is C++23 now.

## Hard rules

- **Rule A — no Python at runtime.** Dev-box scripts and IRON-py
  offline NPU authoring fine. Never a systemd unit or HTTP path.
- **Rule B — C++23 default; kernels stay in `rocm-cpp/`.** Tower above
  the kernels is C++23 (`std::expected`, `std::span`, `std::format`,
  ranges). HIP kernels stay C++20 in `rocm-cpp/` (folded 2026-04-20
  via `git subtree add`; history preserved). Do NOT reimplement
  kernels above the kernel layer.
- **Rule C — hipBLAS banned** in the runtime path. Native Tensile
  kernels only. If you reach for hipBLAS, port the kernel to
  `rocm-cpp` instead.
- **Rule E — NPU = ORT C++ + VitisAI EP primary; Peano + libxrt +
  aie-rt custom-kernel lane.** IRON permitted at compile time.
- **Rule F — ISO C++ Core Guidelines.** Watch I.27 (pImpl:
  `std::unique_ptr<Impl>`, special members declared in header,
  defaulted in `.cpp`). F.55 exhaustive `std::visit` on variants.
  `std::expected<T, HaloError>` on every fallible path; no
  exceptions in hot paths. `[[nodiscard]]` on factory returns.

> "There is no spoon." The old Rule D (workspace toolchain pin) is
> retired with the Rust workspace.

- **Target: aspirational 7/7 lanes, ~280 tok/s decode, NPU-prefill
  crossover at L ≥ 33.** Projected in
  `docs/wiki/Peak-Performance-Projection.md`. We do not settle for
  the conservative tier.

## Layout

`lemond` lives outside this repo at `/home/bcloud/repos/lemonade/`,
runs as `1bit-halo-lemonade.service` on `:8180`, dispatches per recipe
to the in-process Engine. Tower components in `cpp/`: `core`, `cli`,
`mcp`/`mcp-clients`/`mcp-linuxgsm`, `landing`, `helm`/`helm-tui`,
`voice`/`echo`, `power`, `watchdog`, `retrieval`/`ingest`/`stream`/
`tier-mint`, `onnx`/`aie`/`kokoro`, `tools/`. Plus `strixhalo/`
dotfiles and `packages.toml`.

## Build

```bash
cmake --preset release-strix
cmake --build --preset release-strix
ctest --preset release-strix
```

`release-ryzen` for gfx1201, `debug` for asan.

## Testing

- ≥3 doctest cases per component before merge.
- `ctest --preset release-strix` green on `main`.
- GPU tests gate on `ONEBIT_REAL_BACKEND=1`.
- **Parity vs gen-2** is the cutover gate — see `CUTOVER.md`.

## Deploy flow

`1bit install <component>` reads `packages.toml` and does
stop/copy/start. Use it.

## Commits

Conventional Commits prefixes: `feat / fix / perf / docs / refactor /
build / ci / chore / test`. One logical change per commit. "Why" line
on every message — not "add tokenizer" but "add tokenizer special-token
handling; gen-2 expects 128009 on EOT boundary and argmax diverges
without it." Push to `bong` remote (`git@github-bong:...`).
`bong-water-water-bong` is the canonical handle. `stampby` is retired.

## What NOT to do

- **Don't commit tokens, session cookies, or bearer secrets.**
  Caddyfile's bearer lives in `/etc/caddy/Caddyfile` (root-only). The
  `strixhalo/caddy/Caddyfile` tracked copy has `sk-halo-REPLACE_ME`
  placeholders; never replace in git.
- **Don't add Python deps.** If you think you need Python, talk to
  the user.
- **Don't touch `ternary_gemv_halo.hip`** in `rocm-cpp/` without a
  rocprof trace showing the improvement. Current kernel sits at 92%
  of LPDDR5x peak.
- **Don't skip the KV-cache reset** at the start of each generation.
  The 2026-04-19 SEGV was a `pos` accumulator bug that only showed up
  under sustained load (~200 completions in).
- **Don't add new warnings.** CI doesn't gate on `-Werror` yet, but
  that gate is coming.

## Agent / subagent ground rules

When delegating to a background agent:

- Give it **exact file paths + line numbers** where possible.
- Tell it what to **not** touch (other components, configs, READMEs).
- Ask for a **≤200-word report**, not a full transcript.
- **Trust but verify.** The agent's summary describes intent, not
  outcome. Re-run tests + inspect the actual diff before claiming
  success.

## Memory

`~/.claude/projects/-home-bcloud/memory/`. Load-bearing:
`project_1bit_systems_cpp_port.md`, `feedback_cpp_is_the_standard.md`,
`feedback_time_over_cost.md`, `project_bitnet_live_bench.md`,
`project_ppl_harness.md`.
