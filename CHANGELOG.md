# Changelog

All notable changes to 1bit.systems. Versioning is **date-based** (`YYYY.MM.DD`),
matching the GitHub release tags (`vYYYY.MM.DD`).

## 1.0.0 — 2026.07.26 — First Stable Release 🎉

- **CUDA + Metal GPU backends — cross-platform inference unlocked.** The single
  C++ inference engine now runs on NVIDIA (CUDA), AMD (HIP/ROCm), and Apple (Metal)
  GPUs from one CMake build. All three backends share the same ternary kernel
  library, weight loader, and server frontend (#858).
- **DeepSeek complete family — MLA + MoE architectures.** Full implementation of
  Multi-Head Latent Attention (MLA) and Mixture-of-Experts routing for the
  DeepSeek-v2/v3 model family, integrated into the dense GPU and CPU backends.
- **Vision-language pipelines: Qwen3-VL + ZAYA1-VL-8B.** End-to-end VL inference
  with ViT vision encoder, multimodal projector, and 1BP format support. Text
  decoder handles both Qwen3 and ZAYA1 architectures.
- **GPU Whisper kernels + Pixtral + FLUX feasibility.** Speech-to-text via native
  HIP GPU kernels (FFT, STFT) with whisper.cpp integration, plus Pixtral connector
  and FLUX diffusion model assessment.
- **Jarvis C++ port (Phases 1–3).** Complete C++ rewrite of the Jarvis agent:
  RAG, tool-calling, planner, routing, beacon (Phase 1); STT via whisper.cpp
  (Phase 2); TTS via piper with USB-speaker mirror (Phase 3) (#898, #902, #904).
- **NPU ternary pipeline.** Full TQ2 symmetric-ternary quant path in C++
  (`gguf_to_onebp --tq2`), block-vectorized `mac_8x8_8x8T` NPU kernel, IQ1_M
  and IQ1_S GPU dequant from llama.cpp, and NPU bridge wiring (#812, #887, #892).
- **GPU Render Engine — 2.6× faster prefill.** All GPU operations fused onto a
  single stream with pipelined async copies, eliminating kernel launch overhead
  for the prefill phase (#945).
- **6 paper-based performance improvements.** Integrated research techniques
  across backends for measurable throughput gains on all supported hardware (#917).
- **Bug fix sweep: 33 fixes across the stack.** Highlights:
  - Zamba2 `attention_forward` processed only **1 of 32 heads** — now fixed (#946)
  - MoE tensor shape validation + architecture guard (#947)
  - Server-side generation timeout + `RLIMIT_AS` OOM safety net (#948)
  - OOB token bounds check in Mamba1 forward() + server stability (#935)
  - Tolerate invalid UTF-8 in completion JSON responses (#944)
  - Serialize backend compute calls to stop watchdog/generate() race (#914)
  - HIP context bound on every Mamba1 entry point (#927)
  - NPU engine SIGPIPE ignored so dead workers don't crash the server (AUDIT #3)
  - unified_server data race fix (AUDIT #2)
  - 29 kernel `__shfl_xor_sync` fixes for NPU stability (#954-#962)
- **Windows MSVC build config.** Full CMake + MSVC toolchain for Windows builds,
  including `scripts/build_windows.cmd` and `CMakeLists_windows.txt`.
- **SEO overhaul.** Sitemap, robots.txt, meta tags, Cloudflare Pages Functions,
  auth worker, Web Analytics beacon across all 40 HTML pages (#926).
- **Repo-wide cleanup.** Dead file removal, stale benchmark corrections, honest
  claim validation, untracked build binary cleanup (#890, #892, #906, #908).

| Metric | Value |
|---|---|
| Commits since v2026.07.24 | 88 |
| Features | 16 |
| Bug fixes | 33 |
| Performance | 1 |

**Full changelog**: [v2026.07.24...v2026.07.26](https://github.com/1bit-systems/1bit/compare/v2026.07.24...v2026.07.26)

---

**Monthly cadence from here.** Next: `v1.1.0` — expected ~2026-08-26.

## 2026.07.24

- **Dense GPU inference through the C++ ZINC Vulkan backend — unlocked.** The
  reverse-engineered ZINC stack now runs dense GGUF transformers (Qwen2/ZR1)
  end-to-end on the Radeon 8060S and matches the CPU reference **token-for-token**
  at **~26 tok/s** (vs ~6 on CPU). Twelve bugs fixed to get there — the last was
  a KV cache sized from `context_length` (131072) that overran RADV's 4 GiB
  `maxStorageBufferRange`, silently zeroing cached-V reads and collapsing
  attention (#844, #847, #851, #852, #854). ZINC is enabled by default for
  the architectures it computes correctly (llama/mistral/qwen2) and falls back
  to `cpu_generic` otherwise (#856); `ZINC_DISABLE=1` forces HIP/CPU.
- **~8.5× faster dense CPU decode.** Parallelized the generic-backend matmul
  with OpenMP — ZR1-1.5B 1.5 → ~12 tok/s, deterministic, output unchanged (#849).
- **Engine crash-hardening.** A backend that fails to initialize (e.g. missing
  Vulkan shaders) now fails over to HIP/CPU instead of taking the server down;
  all backend init/benchmark/generate paths are exception-safe (#846, #847).
- **Security**: image-fetch curl restricted to http/https (SSRF/LFI hardening).
- **Pure-C++ 1BP toolchain, end to end.** Ported the `--tq2` symmetric-ternary
  quant path into `tools/gguf_to_onebp.cpp` (was Q4NX-only) and registered it as
  a first-class CMake target — the advertised `gguf_to_onebp model.gguf out.1bp
  --tq2` is now a real C++ binary, zero Python in the convert path. Verified:
  TQ2 output is exactly half of Q4NX and losslessly round-trips ternary input.
- **Engine health confirmed on-device**: BlackMamba-1.5B at **74.8 tok/s** live
  on Strix Halo (Radeon 8060S, gfx1151), three clean runs, no hangs.
- **Landing page**: new flagship 1BP model showcase (measured perf + direct
  Hugging Face download links) and a second-level **Zyphra** + **Poolside Laguna**
  family showcase. "One binary to rule them all" badge restored.
- **Repo hygiene**: untracked 431 committed `build_cmake/` artifacts + stray
  `Desktop/` that had been polluting every diff; reorganized 50 flat docs into
  `docs/{archive,marketing}/` + a navigation index; README marketing refresh.
- **Compiler warnings cleared** (#827 dead watchdog stores, #828 unused params,
  #829 dead `decode_ternary_word`/`qkv_dim`).

## 2026.07.20

- **Mamba1 GPU backend fully wired and fixed.** `backend_mamba1.cpp` + `mamba1_engine.hip` now compile as a first-class backend in `libbackend_manager.a`. Three critical correctness bugs fixed: conv state buffer overflow (shift loop out-of-bounds write), A_log never exponentiated (SSM scan used raw A_log instead of `A = -exp(A_log)`), and HIP device stub linkage (kernel launches wrapped in `extern "C"` helpers). BlackMamba 1.5B runs at **79.8 tok/s**, BlackMamba 2.8B at **46.4 tok/s** on Strix Halo (ROCm HIP, 15+15 MoE layers alternating). Diagnostic tool `tools/test_mamba1_backend.cpp` added for direct backend testing (#579).
- BlackMamba 1.5B and 2.8B GGUF files converted from HF cache (F16, 438/525 tensors) and benchmarked.

## 2026.07.19

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
- Full benchmark on TheRock 7.15.0a (Ubuntu 24.04)
- Prefill 4h kernel: 21.94 TFlops (73% of rocBLAS, 2.9x per-byte)
- Decode halo: 27.01 µs (7.8x rocBLAS)
- Auto-tuner with 7 prefill variants
- CI: headers check + ShellCheck

## [0.1.0] — 2026-04-30
- Initial release on TheRock ROCm 7.13
- BitNet-2B-4T end-to-end decode at 82 tok/s
- Prefill 30.15 TFlops at 1.02x TheRock rocBLAS
- Decode GEMV 4.9-7.2x rocBLAS
