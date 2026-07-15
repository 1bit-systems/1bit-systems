# Changelog

## [0.2.1] — 2026-06-26

### Bug fixes & robustness
- `install.sh`: Fixed `$1` unbound-variable crash when running without arguments
  under `set -euo pipefail`
- `env.sh`: Added `$LINK_DIR/build` to `PATH` so CLI tools (`bitnet_decode`,
  `bench_prefill_variants`, etc.) are discoverable after `source env.sh`
- `rust/src/main.rs`: Added `Drop` impl on `AppState` that kills the backend
  child process on server shutdown / panic (no more orphan zombies)
- `h1b_loader.cpp`: Added `f.fail()` checks after every `f.read()` to catch
  truncated or corrupt `.h1b` files early with a clear error message
- `tokenizer.cpp/.h`: Fixed `.htok` merge table field documentation — the third
  field is the *merged token id*, not the rank; rank is derived from insertion
  order. Added infinite-loop guard in BPE merge loop, empty-input early-return,
  and null-ids validation
- `prefill_dispatcher.cpp`: Added variant-index bounds and null-function-pointer
  checks before dispatch
- `CMakeLists.txt`: Removed `src/ck_gemm.cpp` from the HIP language property set
  (compiled as C++17 via CK's host-only path); removed `src/prefill_dispatcher.cpp`
  from the HIP source set (was duplicating `target_sources` entry)
- `.gitignore`: Removed duplicate `/rust/target` entry; added editor swap files
  and `ck-prefill/build/`
- `prim_kernels.hip`: Added `<cstdlib>` include for `std::abs` / `std::round`
  portability

### Documentation
- `tokenizer.h`: Corrected `.htok` merge table field documentation — third field
  is `new_id` (merged token id), not `rank`

## [0.2.0] — 2026-06-23
- Full benchmark on ROCm 7.2.4 (Ubuntu 24.04)
- Prefill 4h kernel: 21.94 TFlops (73% of rocBLAS, 2.9x per-byte)
- Decode halo: 27.01 µs (7.8x rocBLAS)
- Auto-tuner with 7 prefill variants
- CI: headers check + ShellCheck

## [0.1.0] — 2026-04-30
- Initial release on TheRock ROCm 7.13
- BitNet-2B-4T end-to-end decode at 82 tok/s
- Prefill 30.15 TFlops at 1.02x TheRock rocBLAS
- Decode GEMV 4.9-7.2x rocBLAS

## 2026.07.15

- fix(backend_manager): rank_backends now uses benchmark score as primary sort when FASTEST strategy
- fix(backend_manager): select_best respects strategy (FASTEST, LOWEST_POWER, ROUND_ROBIN, MANUAL)
- fix(backend_manager): benchmark_all keeps all benchmarked instances alive so re_evaluate can switch
- fix(backend_manager): generate() updates info.score with EMA so live latency feeds routing
- fix(backend_manager): set_strategy triggers re_evaluate for automatic strategies
- fix(backend_manager): added re_evaluate() — re-ranks and re-selects active backend
- fix(unified_server): removed post-benchmark re-init hack, now uses re_evaluate()
- chore(benchmarks): re-measured kernel benchmarks on Strix Halo (2026-07-15)

## 2026.07.15 (full benchmark sweep)

- feat(benchmarks): full kernel microbenchmark suite on Strix Halo
  - sherry GEMV: 153.0 GB/s | tq1 GEMV: 191.6 GB/s | halo GEMV: 162.8 GB/s
  - prefill 4h: 21.77 TFLOPS | prefill I8-APRE: 38.89 TFLOPS
  - bonsai full-model decode: 425 tok/s (Q1_0 1024-block), 358 tok/s (TQ2_1024)
  - fused TQ2 QKV+GU: 413 tok/s (1.15x speedup over individual launches)
  - KV Flash-Decoding: 12.65× speedup at seq_len=2048
  - RotorQuant PQ3: 9224 tok/s at seq_len=2048
- feat(bonsai): end-to-end real model decode verified on Bonsai 1.7B TQ2
  - Model load: PASS (hs=2048, is=6144, L=28, nh=16, nkv=8, V=151669)
  - Forward pass: coherent logits (argmax=76213, max=327007, min=-396679)
- chore(binary sizes): zaya_server=282KB, unified_server=1.2MB, bitnet_decode=688KB
- doc(benchmarks): published full results to benchmarks/RESULTS-2026-07-15.md
