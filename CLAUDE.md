# CLAUDE.md — conventions for 1bit-systems

Lean local inference engine on Strix Halo. Apps connect to Lemonade,
FastFlowLM, or the 1bit union endpoint. The single control plane comes
second.

## Hard rules

- **Rule A: core serving stays Python-free.** Training, notebooks,
  build-time conversion, caller-side tools, and compatibility UIs are
  allowed. The core engine path we own is Python-free: proxy, kernels,
  native runtimes, and model hot paths. Open WebUI is allowed only as an
  isolated secondary UI behind the OpenAI-compatible endpoint.
- **Rule B: C++20 for kernels.** HIP code belongs in `rocm-cpp/`.
- **Rule C: hipBLAS is banned in the runtime path.** Port kernels to
  `rocm-cpp/` instead.
- **Rule D: Rust 1.88+, edition 2024.** Bump with a reason.
- **Rule E: NPU has two lanes.** FastFlowLM is the live XDNA serving
  lane. Custom NPU kernels are IRON author-time → MLIR-AIE → Peano →
  xclbin → libxrt C++ runtime.
- **Compatibility surface is OpenAI.** Anything that breaks `:13306/v1`
  or `:13306/api/v1` for clients is a bug.

## Layout

```
.
├── install.sh             # pacman + paru installer, idempotent
├── scripts/1bit           # control-plane CLI (up/down/status/pull/bench/npu)
├── benchmarks/            # bench-1bit-pile.sh + RESULTS-*.md
├── 1bit-site/             # CF Pages site for 1bit.systems
└── docs/                  # architecture notes
```

## What lives outside this repo

- `lemond` (Lemonade Server) — built from the maintained fork, runs on `:13305`,
  binaries cached at `~/.cache/lemonade/bin/`.
- `flm` (FastFlowLM, NPU) — built from the maintained fork, runs on `:52625`.
- `1bit-proxy` — Node service on `:13306`, unifies Lemonade and FastFlowLM.
- ROCm 7.2.x — installed via pacman (`rocm-hip-sdk`).
- XRT + amdxdna — installed via pacman.
- Bonsai / ternary GGUFs — pulled from HuggingFace (`lilyanatia/*`,
  `superkaiii/*`, `gianni-cor/*`) into `~/halo-ai/models/ternary-test/`.

## Test / verify

```sh
./install.sh           # idempotent, run anytime
1bit status            # quick health check
1bit bench             # full pile bench (needs models pulled first)
```

## Deploy

`1bit-site/` deploys to Cloudflare Pages via `wrangler pages deploy`. The
CF project name is `1bit-systems` (not `1bit-site`). There is no GitHub
auto-deploy hook on the lean branch — push to CF manually after edits.

## Commits

Conventional Commits: `feat / fix / perf / docs / refactor / build / ci /
chore / test`. One logical change per commit. Push to `origin`
(`bong-water-water-bong/1bit-systems`).

## What NOT to do

- **Don't add a Rust workspace, a C++ tower, or HIP kernels back.** That
  was the abandoned scaffolding archived as `archive/cpp-tower-2026-04-27`.
  If you need it, branch from there.
- **Don't commit secrets.** CF tokens and gh tokens live in libsecret.
- **Don't expand scope.** Match the literal ask. Default to the minimum
  that delivers 1-bit inference.

## Memory

`~/.claude/projects/-home-bcloud-Projects/memory/`. Load-bearing for this
project: `project_lemonade_install.md`, `project_1bit_real_vision.md`,
`feedback_dont_expand_scope.md`.
