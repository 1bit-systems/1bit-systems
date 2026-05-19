<!--
Thanks for sending a patch. A few ground rules from CLAUDE.md:

- One logical change per PR. "fix X" + "feat Y" = two PRs.
- Commit messages have a "why" line, not just a "what".
- Conventional Commits prefixes (feat / fix / perf / docs / refactor /
  build / ci / chore / test).
- `cargo test --workspace --release` must stay green.
- Any change that adds a new crate needs ≥ 3 in-crate tests.
-->

## Summary

<!-- One-paragraph description of the change and why it's worth landing. -->

## Why

<!-- The motivating problem. Link issues, benchmarks, prior discussion. -->

## Change summary

- <!-- bullet: what changed -->
- <!-- bullet: what changed -->

## Rule check

- [ ] **Rule A** — no Python at runtime (scripts on a dev box are fine; services / HTTP paths stay Rust).
- [ ] **Rule B** — all new kernels live under `rocm-cpp/` (C++20 / HIP); FFI'd through `1bit-hip`.
- [ ] **Rule C** — no new hipBLAS calls on the runtime path.
- [ ] **Rule D** — Rust 1.88+ / edition 2024 (no `rust-version` bump without a reason).

## Tests

- [ ] `cargo test --workspace --release` is green locally.
- [ ] New code ships with tests (unit, integration, or parity fixture as appropriate).
- [ ] If perf-sensitive: ran `halo bench` / `halo ppl` — numbers below.

<!-- paste benchmark diff if relevant -->

## Risk

<!-- What could go wrong after this lands? How would we find out?
     Shadow-burnin? Metric to watch on /metrics? Rollback plan? -->

## Related

<!-- Closes #, follows up on #, supersedes #, references docs/wiki/*.md -->
