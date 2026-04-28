# CLAUDE.md — conventions for 1bit-systems

Lean 1-bit inference engine on Strix Halo. Three lanes behind one
OpenAI-compatible endpoint at `:13305`.

## Hard rules

- **Lean over scaffolding.** This repo is the install + control plane on
  top of upstream Lemonade Server and FastFlowLM. Do not reimplement
  inference. Do not port kernels in-tree. Upstream PRs are the right
  place for kernel work.
- **No Python at runtime.** Bash for the installer + `1bit` CLI is fine;
  upstream FLM and lemonade-server bring their own runtimes.
- **Compatibility surface is OpenAI.** Anything that breaks the
  `:13305/v1/*` shape is a bug. The whole point is "works with any
  client that speaks OpenAI."

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

- `lemond` (Lemonade Server) — installed via paru/AUR, runs on `:13305`,
  binaries cached at `~/.cache/lemonade/bin/`.
- `flm` (FastFlowLM, NPU) — installed via pacman from `cachyos-extra-znver4`.
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
