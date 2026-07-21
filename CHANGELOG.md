# Changelog

## 2026.07.20

- **Mamba1 GPU backend fully wired and fixed.** `backend_mamba1.cpp` + `mamba1_engine.hip` now compile as a first-class backend in `libbackend_manager.a`. Three critical correctness bugs fixed: conv state buffer overflow (shift loop out-of-bounds write), A_log never exponentiated (SSM scan used raw A_log instead of `A = -exp(A_log)`), and HIP device stub linkage (kernel launches wrapped in `extern "C"` helpers). BlackMamba 1.5B runs at **79.8 tok/s**, BlackMamba 2.8B at **46.4 tok/s** on Strix Halo (ROCm HIP, 15+15 MoE layers alternating). Diagnostic tool `tools/test_mamba1_backend.cpp` added for direct backend testing (#579).
- BlackMamba 1.5B and 2.8B GGUF files converted from HF cache (F16, 438/525 tensors) and benchmarked.

## 2026.07.20

- **FastFlowLM fully reverse-engineered and replaced as the default NPU path.** 22 closed-source `.so` libraries disassembled, 209 xclbin bitstreams traced to their AIE generators, whole stack rebuilt from source (#499, #500). `model_router.cpp` now routes qwen3-architecture models to the native, open-source `npu_xrt` engine first, with the FastFlowLM subprocess kept only as a fallback (#567) — its single-core GEMM kernels are correctness-verified on real hardware (`docs/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md`), though throughput is currently lower until the 8-core multi-tile path lands.
- **Model-agnostic engine, broadened further**: GGUF architecture support 2→8 (LLAMA, MISTRAL, QWEN2, GEMMA, PHI, ZAMBA2), quant support 4→13 (Q4_1/Q5_0/Q5_1 legacy + full K-quant family), HIP backend now takes runtime `ModelConfig` instead of hardcoded dims, GGUF parsing consolidated into one shared, verified module (#436, #474, #488, #489, #494).
- **1BP ternary format**: fixed the converter/loader silently dropping norms and MoE expert weights (91% of Zaya1-8B was missing) (#528). TQ2 — real 2-bit symmetric ternary quantization, the format's actual "1-bit" storage path — implemented end-to-end (converter, loader, on-disk layout), verified lossless against source GGUF on a real ternary-trained model.
- **Vision**: Qwen2-VL support, minimal POC — real image-to-text end to end (#491, #492). Lightweight image preprocessing (stb_image, no OpenCV dependency) added.
- **Model catalog**: full Zyphra family showcase (Zamba2, ZR1, Zaya1-74B-preview) plus their 1BP conversions, all uploaded to Hugging Face (#529, #526).
- **NPU toolchain**: switched from Peano/LLVM-AIE to AMD Xilinx IP (Chess) (#527). 8-core INT8 GEMM correctness work reconciled across divergent branches (#344).
- **colibri int4 quantized matmul kernels + PILOT cross-layer prefetch** (#449). **A2A (Agent-to-Agent) protocol v1.0** support added to `zaya_server` (#345).
- Large correctness/security audit sweeps: ~60 numbered issues closed across #362, #415, #417, #436, #495, #496, #498, plus today's fixes (OSCAR attention cross-warp race, NPU worker pipe I/O timeout, concurrent HTTP handler state race).
- **Landing page**: removed a headline "tok/s" claim that the site's own data-integrity quarantine (`benchmarks/latest.json._unverified`) explicitly flagged as having no source — was still driving `<title>`/meta/OG tags and a JS bug that hardcoded it into the meta description on every load, bypassing the quarantine guard entirely.

## 2026.07.16

- feat(hardware-aware): auto-dispatch policy defaulting to N+G pathway
- fix(backend_manager): load_plugins now infers tier from plugin type instead of hardcoding T2_GPU
- doc: fixed stale paths in SECURITY.md, ROCm repo inconsistencies, CI pipeline table
- security: redacted exposed Stripe credentials from ROADMAP.md and site/store/index.html

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
- `tokenizer.cpp/.h`: [redacted]
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
- `tokenizer.h`: [redacted]
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
