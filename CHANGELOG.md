# Changelog

All notable changes to `1bit-systems` are logged here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added (2026-04-28)

- `install.sh`: `patch_lemonade_flm_pin()` — auto-bumps Lemonade's
  pinned `flm.npu` version in `/usr/share/lemonade-server/resources/backend_versions.json`
  to whatever `flm version` reports. Lemonade does *strict equality* on
  backend versions; the AUR `fastflowlm` package can lead the pin (e.g.
  AUR ships `v0.9.39` while Lemonade pins `v0.9.38`), which silently
  flips the `flm:npu` recipe to `update_required` even when `flm validate`
  is fully green. The patch makes `1bit-systems` install all-green
  end-to-end on Arch/CachyOS without manual intervention.
- `benchmarks/RESULTS-stack-2026-04-28.md` — unified-stack bench through
  Lemonade's native recipe routing on `:13305`. Both lanes serve
  (iGPU `llamacpp:rocm` and NPU `flm:npu`), no proxy required. iGPU LFM2-1.2B
  ~217 tok/s decode; iGPU BitNet-b1.58-3B (TQ2_0, sub-2-bit) ~71-76 tok/s;
  NPU qwen3-0.6b ~95 tok/s; NPU qwen3-1.7b ~42 tok/s.

### Fixed (2026-04-28)

- `flm:npu` recipe in Lemonade no longer reports `update_required` after
  install — `install.sh` patches the version pin in place.

### Changed

- **Cutover (2026-04-27):** repo pivoted to a lean install + control
  plane on top of upstream Lemonade Server and FastFlowLM. The C++23
  tower (`cpp/`, `rocm-cpp/`, `npu-kernels/`, `agents/`, `browser/`,
  `strixhalo/`, `packaging/`) is archived in
  `archive/cpp-tower-2026-04-27` and removed from `main`. New
  `install.sh` + `scripts/1bit` CLI replace the old `1bit install`
  package manager.

### Added

- Repository landing-page polish: badges, hero block, TOC,
  benchmark table, client integration snippets, roadmap, acknowledgements.
- `LICENSE` (MIT), `SECURITY.md`, `CODE_OF_CONDUCT.md`, this
  `CHANGELOG.md`.
- `.github/` issue templates (`bug_report.yml`, `feature_request.yml`,
  `config.yml`) and pull-request template.
- Extended `.gitignore` to exclude HIP / Clang-offload compiler
  artifacts (`*.bc`, `*.hipi`, `*.hipfb`, `*.cubin`, `*.fatbin`,
  `*.ptx`, intermediate object/assembly outputs).

### Removed

- 10 stray `bonsai_tq2_gemv_orig-*` compiler artifacts accidentally
  committed at repo root (~5.6 MB of `.hipi` + misc bitcode / objects).

## [0.1.0-dev] — 2026-04-21

Pre-public preview. See `git log` for the full per-commit history. Rough
milestones that land pre-1.0:

- `bitnet_decode` (gen 1) byte-exact parity: **96.66%** across 1500+
  shadow-burnin rounds.
- PPL on wikitext-103 (1024 tokens): **9.1805** (gen-2), vs gen-1's
  9.1607 baseline — within the ±0.05 cutover tolerance.
- Eleven crates green on `cargo test --workspace --release`, ~90 tests.
- Ternary GEMV on gfx1151 reaches ~92% of measured LPDDR5 peak.
- Sherry 1.25-bit packing spike landed (2026-04-18); requantizer
  diagnosed (per-tensor vs per-row scale mismatch), retrain in
  progress.
- RoPE convention migration: interleaved → HF split-half. PPL 524 →
  ~12 on wikitext; 4.29 → 1.04 on repetition (2026-04-19).
- Split-KV Flash-Decoding attention shipped: 6.78× @ L=2048,
  bit-exact (2026-04-19).
- MLX (Apple Silicon) backend feature-gated behind `--features
  mlx-apple`.

## How to add entries

- One bullet per user-visible change.
- Group under `Added / Changed / Fixed / Removed / Security /
  Deprecated`.
- Link to issue / PR where relevant (`(#123)`).
- Move `[Unreleased]` entries under a dated heading at release time;
  update the top comparison link.
