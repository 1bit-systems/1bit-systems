# CPU Lane Plan

The 7th surface of 1bit systems's APU stack: the 16 Zen5 cores on Strix Halo,
running the sampler + tokenizer + dispatcher in parallel with the iGPU's
next-token grind.

This page covers what we scaffolded today (2026-04-20), why the CPU is
a peer lane (not a fallback), and the three concrete steps that turn the
scaffold into a measured win.

## What the lane is

The CPU lane is **host-side orchestration in parallel with GPU compute**.
On a live decode:

* The iGPU runs `forward_token` — ternary GEMVs + flash-decoding attention
  — and emits logits.
* The CPU has to pick a token id from those logits (argmax or top-k /
  top-p / temperature sampler), detokenize the byte-level BPE delta,
  match stop strings, and hand the next id back to the GPU.

Gen-1 and current gen-2 run all of that on a single tokio task. Every
microsecond the sampler spends on the critical path is a microsecond
the iGPU sits idle waiting to be fed the next token id. The lane gives
that work its own rayon-backed thread pool:

* Pool size = `available_parallelism() - 2` by default (leave 2 logical
  cores for the tokio reactor + HIP stream callback thread).
* `HALO_CPU_THREADS` env override for operators who want to pin it.
* Each rayon thread named `halo-cpu-<idx>` so `rocprof` / `perf top`
  attribute it cleanly.

Surface entry point: [`onebit_router::cpu_lane::CpuLane`] in
`crates/lemond/src/cpu_lane.rs`.

## Why CPU is a peer lane, not a fallback

`project_apu_thesis.md` is the longer form; the short version:

* Strix Halo is a **unified-memory APU**. LPDDR5-8000 is shared by iGPU,
  NPU, and CPU at 256 GB/s peak. A CPU lane doing sampler work doesn't
  compete with the iGPU for bandwidth — the sampler's read pattern is a
  single 32k-element logit scan per token (~128 KB), fits in L2.
* Every lane that idles during decode is a lane that could have been
  running something in parallel. iGPU grinds matmuls, NPU grinds prefill
  GEMMs, CPU grinds the sampler + tokenizer. Amdahl wins when nothing
  idles.
* Peak-Performance-Projection.md tracks this as surface #7. The first six
  are HIP matmul, HIP attention, HIP sampler (today's path), XDNA
  prefill, XDNA decode, shared LPDDR5 DMA. The CPU lane replaces "HIP
  sampler" on the critical path so the iGPU can start the next token's
  matmul while the CPU is still finishing the previous token's sampler.

## What we scaffolded today

2026-04-20:

* `crates/lemond/src/cpu_lane.rs` — new module. `CpuLane` struct
  holds a `rayon::ThreadPool`; `parallel_sample(&self, logits, top_k,
  top_p, temp)` demonstrates the rayon parallel-reduction pattern
  (argmax at `temp=0`, chunked top-k fallback at `temp>0`).
* `Backend::Cpu` in `lemond`'s routing guard now returns
  `BackendError::CpuLaneStub("CPU sampler lane scaffolded, not yet on
  critical path; see docs/wiki/CPU-Lane-Plan.md")` instead of
  `unimplemented!()`. Distinct error kind from `NotYetWired` so ops
  tooling can count them separately.
* `rayon = "1"` added to `crates/lemond/Cargo.toml`. Rayon 1.12 is
  already in the workspace Cargo.lock transitively (via tokenizers);
  adding it directly doesn't grow the build graph.
* Three unit tests in `cpu_lane::tests` — constructor doesn't panic,
  argmax at `temp=0` is deterministic on a 32k-element vector,
  `HALO_CPU_THREADS=4` is respected. Env-var test takes a module-local
  mutex so it can't race with the `HALO_BACKEND` test in
  `backend_config_tests`.

## 2026-04-20 measurement + decision

**Measurement harness** — `crates/1bit-server/benches/sampler.rs`
(plain-`main` `harness = false`, stable Rust, no new deps) plus an
`#[ignore]`-gated `tests/sampler_cost_probe.rs` shim for sandboxed
runs. 500 timed iters × 50 warm-up, vocab = 128,256, median reported.

Numbers on strixhalo (Zen5 × 16) at release opt-level, 2026-04-20:

| path                                     | median      | % of 15 ms fwd | % of 30 ms fwd |
|------------------------------------------|-------------|----------------|----------------|
| inline greedy (temp = 0)                 | **83 µs**   | 0.55%          | 0.28%          |
| inline multinomial (T = 1, k = 0, p = 1) | **402 µs**  | 2.68%          | 1.34%          |
| inline full (T = 0.7, k = 50, p = 0.95)  | **710 µs**  | 4.73%          | 2.37%          |
| parallel argmax (rayon × 1)              | 65 µs       | 0.43%          | 0.22%          |
| parallel argmax (rayon × 4)              | 23 µs       | 0.15%          | 0.08%          |
| parallel argmax (rayon × 8)              | 16 µs       | 0.11%          | 0.05%          |
| parallel argmax (rayon × 14)             | 16 µs       | 0.11%          | 0.05%          |

**Decision — defer pipelining, land env-gated offload.**

The full sampler path sits at 4.73% of a 15 ms forward step — right at
the "pipelining is worth plumbing" threshold we set pre-measurement
(≥5%), and in the common `temp = 0` case the GPU already returns
argmax so the host sampler is skipped entirely (`generate_blocking`
short-circuits when `req.sampler.temperature <= 0.0`). A pipelined
next-forward-over-previous-sampler implementation needs
double-buffered logits scratch + an async handoff between the HIP
stream and the sampler task; we don't spend that complexity for a 4.7%
ceiling that only applies to the non-default sampler config.

**What shipped instead.** `HALO_SAMPLER=inline|parallel` env var, read
once at router construction (`cpu_lane::SAMPLER_MODE_ENV`,
`cpu_lane::sampler_mode_from_env`). Wired through
`onebit_router::RouterConfig::sampler_mode` into
`generate_blocking`. Default is `Inline` (today's behaviour,
bit-identical). `Parallel` routes `Sampler::sample` through
`CpuLane::sample` → `rayon_pool.install()`. Semantics are
bit-identical — the lane's threads reserve cores for tokio + the HIP
stream callback and get named `halo-cpu-<idx>`, so `rocprof` / `perf
top` attribute sampler cycles cleanly. Operators can flip the switch
to measure under real load without a rebuild.

Three parity tests land in `cpu_lane::tests`:
`inline_and_parallel_agree_at_temp_zero`,
`inline_and_parallel_agree_on_topk_topp_path`,
`sampler_mode_env_override_is_respected`.

## Where the real CPU-lane win lives

Sampler hotspot analysis ruled *this* target out. The remaining CPU
lane candidates, in falling priority order:

1. **Tokenizer / detokenizer offload.** `generate_blocking` calls
   `inner.backend.detokenize(&generated_ids)` every step and walks
   the *whole* generated_ids vector on each call — quadratic in
   generation length. Moving the byte-level-BPE decode onto the CPU
   lane (and restructuring to emit only the new-bytes delta) is an
   actual Amdahl win on long generations.
2. **Stop-string match.** Currently a linear scan over every stop
   string against `full_text` on every step. Small constant factor
   today, but it grows with `stop` array length.
3. **Sherry 1.25-bit weight decompression.** When the 3:4 sparsity
   spike ships (see `project_sherry_spike.md`), the on-the-fly
   ternary-from-packed decode is a parallel reduction — fits rayon.

## What we did **not** do today, on purpose

* Wire a pipelined sampler (next forward runs during prior sampler).
  The measurement above put the sampler at the edge of "worth it" —
  below our pre-declared 5% threshold. Punted honestly; no fake win.
* Replace the existing `onebit_core::sampler::Sampler`.
  `CpuLane::sample` now wraps it bit-identically so we ship identical
  tokens regardless of the env knob.
* Add any C++ deps. Rule A + the 2026-04-20 "bare-metal first" lock-in
  keeps orchestration in Rust; kernels stay in rocm-cpp.

## What's next

### (a) Wire `CpuLane::parallel_sample` into 1bit-server's SSE handler — **DONE (2026-04-20)**

Shipped as an env-gated path after the measurement above showed the
pipelined version wasn't worth the plumbing. The sampler is now
dispatched through `CpuLane::sample` when
`HALO_SAMPLER=parallel`; the default (`inline`) is unchanged. See the
"What shipped instead" block above.

### (b) Microbenchmark single-threaded vs rayon-parallel sampler — **DONE (2026-04-20)**

Harness at `crates/1bit-server/benches/sampler.rs` (plain-`main`,
`harness = false`, stable Rust — no criterion dep added). Run with
`cargo bench -p onebit-server --bench sampler`. Numbers are in the
table above.

### (c) Decide whether ZenDNN FFI is worth the dep weight

AMD-AI-ML-Tools-Scan.md flagged ZenDNN 5.2 as having an official
llama.cpp backend. It's AMD's Zen-tuned math library — MLP / GEMM /
reduction kernels AVX-512-VNNI-tuned for Zen4+. Two questions to answer
before we pull it in:

1. Is a parallel sampler + tokenizer even the bottleneck, or is the
   iGPU forward pass eating all the time? (Step b's bench answers this.)
2. Does ZenDNN offer anything for our sampler specifically? The library
   is matmul-first; our sampler is a reduction. If the win is only on
   matmuls, the CPU lane is still Rust-native — we only FFI when the
   kernel class matches.

Decision output: either add ZenDNN as an optional feature-gated dep
behind `--features zendnn` (same shape as `--features real-xdna`), or
skip and stay pure Rust.

## Cross-references

* `/home/bcloud/.claude/projects/-home-bcloud/memory/project_apu_thesis.md`
  — the "why CPU is a peer lane" thesis.
* `docs/wiki/Peak-Performance-Projection.md` — the 7/7 lane aspirational
  target; the CPU lane closes surface #7.
* `docs/wiki/AMD-AI-ML-Tools-Scan.md` — ZenDNN 5.2 notes for step (c).
* `crates/lemond/src/cpu_lane.rs` — the module this page describes.
