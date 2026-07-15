# Task 1 findings (2026-07-14): the router picture needs a correction too

## Correction to the original audit

The original report (and the 1bit-systems README it was based on) said three separate, uncoordinated router mechanisms exist. That's half right. Looking at the actual source:

**`token-router` (Rust) already unifies four of them in one binary**, via `StrategyConfig` (`config.rs`) → `RouterStrategy` trait (`src/strategy/`): `Passthrough`, `Cascade`, `SpecDecode`, `ContentRouter`, and a fifth I hadn't seen mentioned anywhere — **`Performance`**. All five are config-selectable (`router.toml`, `type = "..."`), dispatched through one `AppState { strategy: Box<dyn RouterStrategy> }` in `handlers.rs`. This is not three routers in three places — inside `token-router` it's already one router with pluggable strategies, which is the *shape* the vision wants.

**What's still genuinely separate:** `1bit-systems/tools/token_router.cpp` (C++, 1073 lines) and `1bit-systems/unified-router.py` (230 lines) are real, independently-maintained implementations of overlapping logic (spec_decode and content-routing respectively) in a different repo, in different languages, not calling into `token-router` at all. Both were modified as recently as 2026-07-12 — they're not dead code — and both are referenced in the README, but neither has any CI/systemd wiring I could find, suggesting they're invoked manually/standalone rather than part of an automated path.

## `Performance` strategy — real, but not what the vision describes yet

`PerformanceStrategy::resolve_backend()` (`strategy/performance.rs`):
1. `force_backend` override, if set (testing escape hatch) — takes precedence over everything.
2. Else look up the requested model in a compiled-in `PERFORMANCE_TABLE` (static, per-model-pattern → backend mapping) and use that.
3. Else fall back to `default_backend`.

This is a **static, per-model routing table**, not a live "rank all currently-available backends by measured throughput, pick the fastest, fall back on failure" system. No health check, no live measurement, no runtime fallback if the resolved backend is actually down — it just returns a backend name based on a compile-time table match. It's real infrastructure and the closest existing thing to the vision, but "the router that doesn't exist yet" (per the README) is still accurate for the *live-ranking-with-fallback* behavior specifically — the static-table version of "pick the right backend automatically" does exist.

## Revised recommendation (supersedes the original Task 1 wording)

1. **Don't build a fourth router.** Extend `PerformanceStrategy` to do two things it doesn't yet: (a) pull backend speed from a live/measured source instead of (or in addition to) the compiled-in table — the `site/benchmarks.json` numbers this whole audit has been citing are sitting right there as a starting data source, imperfect as some of them are; (b) add a health check + fallback path so a resolved-but-down backend doesn't just fail, it falls through to the next-best option.
2. **Retire the C++/Python duplicates by porting their unique logic into `token-router`, not by deleting them outright.** `token_router.cpp`'s spec_decode and `unified-router.py`'s content-routing already have Rust equivalents (`SpecDecode`, `ContentRouter` strategies) — diff what's actually different (the C++/Python versions may have fixes or behavior the Rust ones lack, given they were touched more recently, 07-12 vs. `cascade.rs`/`spec_decode.rs` which I didn't check commit dates on) before assuming the Rust version simply wins.
3. **This is a real engineering task, not a documentation fix** — did not attempt the actual Rust code changes (live throughput source + fallback logic) or the cross-language behavior diff in this pass. Scoped and ready for a dedicated session.

## Not done in this pass
- Diffing `token_router.cpp`/`unified-router.py` behavior against the Rust `SpecDecode`/`ContentRouter` strategies to find what's actually different.
- Implementing live throughput measurement or fallback logic in `PerformanceStrategy`.
- Updating the router diagram in both READMEs (blocked on the above being real).
