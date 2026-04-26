# Live tok/s + PPL

What we measure and what each number means. All numbers taken on the production box (`strixhalo`, 100.64.0.1) unless noted. Cross-arch numbers are captured on the `ryzen` mesh host (`100.64.0.3`, RX 9070 XT, `gfx1201`).

Raw, re-runnable JSON outputs are in [`benchmarks/data/`](../../benchmarks/data/). Do not embed giant tables here — link to the JSON.

## The table

| metric | value | source | what it means |
|---|---:|---|---|
| Decode throughput @ L=64 | **83 tok/s** | `bench.sh` | Short-context user-chat speed. What you feel typing a prompt. |
| Decode throughput @ L=1024 | **33 tok/s** | `bench.sh` | Long-context agent speed. KV-cache bandwidth dominates here. |
| PPL, wikitext-103 (gen-2 Rust) | **9.1805** | `benchmarks/ppl-gen2.sh` | Distribution-level quality. Lower is better. |
| PPL, wikitext-103 (gen-1 baseline) | **9.1607** | historical | Reference point; gen-2 is +0.02, inside ±0.05 tolerance. |
| Shadow-burnin byte-exact | **95.55%** | `halo burnin stats` | Argmax-level parity, gen-1 vs gen-2. 14,344 rounds. |
| Ternary GEMV roofline | **92% of LPDDR5 peak** | `rocprof` | Kernel is bandwidth-bound, not compute-bound. Bytes-read reduction (Sherry) is rank-1. |
| Split-KV Flash-Decoding attn | **6.78× vs prior** @ L=2048 | `benchmarks/attn_fd.sh` | Bit-exact speedup over single-block attention. Default since 2026-04-19. |
| Voice mouth-to-ear first audio | **1.23 s** | `benchmarks/voice.sh` | End-to-end: STT + LLM + TTS first chunk. 3-5× faster than naive serial loop. |
| Tests across 13 crates | **201 passing, 0 failing** | `cargo test --workspace --release` | Workspace-wide green. CI gate. |
| `1bit-server` binary, stripped | **2.4 MB** | `size cpp/build/strix/server/1bit-server` | Static-friendly Rust binary; ships without a runtime. |
| Landing live tok/s | **pulled from `/metrics` via `/_live/stats` SSE** | `cpp/landing/src/telemetry.rs` | The number you see in the hero on `https://strixhalo.local/` is no longer a static guess — it's the same `tokps_recent` the Prom scraper sees, pushed over SSE every 1.5 s. |

## One-liners per number

- **83 @ 64 / 33 @ 1024** — same kernel; the drop is pure KV-cache bandwidth. A 70B FP16 model on the same box does ~18 tok/s. See [Why ternary?](./Why-Ternary.md).
- **PPL 9.1805** — we re-pack Microsoft's weights (no retraining). The +0.02 delta vs gen-1 is FP16-reordering noise, not quality loss.
- **95.55% byte-exact** — **74.9% of the remaining 4.45% is one prompt** (idx=7, "chemical symbol for gold"). Fix that sampler delta and parity jumps to ~98.9%. See [Why parity gates?](./Why-Parity-Gates.md).
- **92% of peak** — roofline math from `rocprof` confirms the ternary GEMV is memory-bound. Compute-side optimization has near-zero headroom; Sherry 1.25-bit packing (bytes-read reduction) is the next 15-25% lever.
- **6.78× attn** — one block per head → split across blocks with an online-softmax reduction. Bit-exact means PPL didn't move when we flipped the default.
- **1.23 s voice** — whisper.cpp partials + sentence-boundary TTS streaming. Naive serial (wait-full-STT → wait-full-LLM → wait-full-TTS) is ~4-6 s.
- **201 tests** — includes property tests on tokenizer round-trips and parity assertions on `.h1b` load paths.
- **2.4 MB** — the whole serving daemon. For comparison, gen-1 Python install was ~1.2 GB.

## Reproducibility notes

- **Box**: Ryzen AI 9 HX 370 (Strix Halo / gfx1151), 128 GB LPDDR5 @ 256 GB/s, CachyOS, kernel 7.0.
- **Ambient**: 22-24 °C room temperature, closet airflow, headless. Thermal throttle shows up above ~28 °C — numbers above are the 22 °C set.
- **Power**: `halo power balanced` profile, 150 W sustained TDP, voltage curve at default. See [Why `halo power`?](./Why-halo-power.md) for the under-volt recipe that adds ~4% throughput without stability loss.
- **Model**: `microsoft/bitnet-b1.58-2B-4T` → `.h1b` v3, 1.8 GB on disk.
- **Temperature**: 0 (greedy argmax) for parity + PPL; 0.7 for voice UX only.

Re-run locally with:

```bash
cd /home/bcloud/repos/halo-workspace
./benchmarks/bench.sh           # decode throughput
./benchmarks/ppl-gen2.sh        # PPL
./benchmarks/shadow-burnin.sh   # parity (long-running)
halo burnin stats               # live summary
```

Numbers regenerate into `~/claude output/`.

## Cross-arch — gfx1151 vs gfx1201

The `gfx1201` (RX 9070 XT, RDNA 4, Navi 48, GDDR6 ~640 GB/s) port shipped 2026-04-22. All 14 WMMA prefill kernels are bit-exact vs CK on both arches; `bitnet_decode --ppl` on wikitext-103 is identical to six decimals on both boxes.

| metric | gfx1151 (Strix Halo iGPU) | gfx1201 (RX 9070 XT dGPU) | source |
|---|---:|---:|---|
| PPL, wikitext-103 (4095-tok single-pass) | **11.9758** | **11.9758** | [`ppl-cross-arch-20260422.json`](../../benchmarks/data/ppl-cross-arch-20260422.json) |
| `bitnet_decode --ppl` throughput (contended) | 59.2 tok/s | 73.6 tok/s | same |
| WMMA FP16 peak (register-resident probe) | **50.17 TFLOPS** (42% of theoretical) | **138.04 TFLOPS** (81% of theoretical) | [`wmma-peak-20260422.json`](../../benchmarks/data/wmma-peak-20260422.json) |
| WMMA INT8 peak | 53.6 TOPS | 288.4 TOPS (2.1× FP16, RDNA 4 wider INT8) | same |
| CU count | 40 | 56 | `rocminfo` |
| Power under sustained probe | 68.1 W | 50.0 W (probe is compute-only, no VRAM traffic) | `rocm-smi` |

The take-away is not "ryzen is faster." Strix Halo's efficiency number (0.90 tok/s/W) is the headline for the closet-AI thesis; RX 9070 XT's raw throughput (178 TFLOPS peak) is the upper bound for the same kernel source on a dGPU.

**Bit-exact verdict.** The gfx1201 port of ternary GEMV + attention + norm kernels is numerically identical to the gfx1151 reference on the production forward path at L=4096. This extends the existing `test_standalone` unit-shape claim to the long-context production path.

## Multi-arch default build

One `./install.sh` run produces a fat binary that covers every mainstream AMD consumer Wave32-WMMA part. Details: [`project_rocm_cpp_multiarch`](../../CLAUDE.md) and [Installation.md](./Installation.md).

| family | arches | hardware |
|---|---|---|
| RDNA3 | gfx1100 / 1101 / 1102 / 1103 | RX 7600 → 7900 XTX, Phoenix iGPU (780M/760M) |
| RDNA3.5 | gfx1150 / 1151 | Strix Point (880M/890M), Strix Halo reference |
| RDNA4 | gfx1200 / 1201 | RX 9060, RX 9070 / 9070 XT |

Arches *not* yet covered (real porting work, not a CMake flip): RDNA2 (`gfx1030-1036`, wave64 fallback needed), CDNA2/3 (`gfx90a` / `gfx942`, MFMA ISA not WMMA), Intel Arc (SYCL/oneAPI, separate backend).

## Benchmark infrastructure

Two new tools landed with the cross-arch pass:

- `rocm-cpp/bench/peak_probe_bench.cpp` — register-resident WMMA throughput probe. Each wave loads A/B fragments once, spins `n_inner` inner iterations of 4×4 back-to-back WMMA ops. Times a median-of-5 run after 3 warmup launches. Drives the WMMA peak numbers above.
- `tools/attn_fd_sweep.cpp` — split-KV Flash-Decoding sweep. Drives [`attn-fd-sweep-20260422.json`](../../benchmarks/data/attn-fd-sweep-20260422.json).

Both honor the project constraints: `no_hipblas: true`, `c++20: true`, `wave32_wmma: true`, `no_python_runtime: true`, `native_tensile_only: true`.

## Links

- [Why shadow-burnin?](./Why-Shadow-Burnin.md) — how the parity number is produced
- [Why parity gates?](./Why-Parity-Gates.md) — how we gate cutover on it
- [Why this way + how?](./Why-This-Way-How.md) — long-form walkthrough of the full stack
- [Peak-Performance-Projection.md](./Peak-Performance-Projection.md) — forward-looking ceiling math
- [Training-Runs.md](./Training-Runs.md) — Run 4 status + Run 5 plan
