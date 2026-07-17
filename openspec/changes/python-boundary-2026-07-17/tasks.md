# Tasks — Python→Native/Rust boundary

Ordered by value/risk. Each is independent; do not batch.

## Ready now (low risk)

- [x] **T1/T3. Native tokenizer wired (DONE, 2026-07-17).** Re-scoped after inspecting
      canonical: `src/server` already tokenizes natively via the FLM engine (no Python).
      The real gap was `daemon/npu-cppd.py`, which was wired to TWO BROKEN native
      tokenizers: `fused-engine --tokenize-only` (emitted nothing) and
      `engine/npu/tokenizer/detokenize` (mangled byte-level spaces:
      `9707 11 1879 0` -> `Hello,\u0120world\\\`). The C tokenizer
      `engine/npu/tokenizer/tokenize` also mis-encodes (`world!` -> wrong ids).
      Fix: point the daemon at `engine/fusion/tokenize` (pure C++17, verified bit-exact
      encode+decode, clean roundtrip), added `engine/fusion/Makefile` build target,
      gitignored the binary. No Python tokenizer lib anywhere in the serving path now.
- [ ] **T2. Ship the Rust router.** Fold `unified-router.py` routing policy into
      `rust/onebit` (or land `reference/unified-router-rs/` as a bin). Retire the `.py`.
- [ ] **T3. Remove `tokenizers` / tokenize-subprocess from `daemon/npu-cppd.py`** —
      point tokenization at the native `tokenize` CLI/lib (kills the `pip install
      tokenizers` + torch2aie-venv tokenize dependency).

## Blocked / needs fix first (medium risk)

- [x] **T4. Fix `free(): invalid size` in `npu-infer/src/npu_engine_stdio.cpp`.** DONE.
      Root cause (ASan): NOT a data heap bug — a null-vtable SEGV in `I8Ctx::~I8Ctx`
      at exit. XRT global singletons are torn down before the local
      xclbin/hw_context/kernel dtors run (static-destruction-order fiasco). Fix:
      `std::_Exit(0)` after the stdin loop — skip the cross-boundary teardown, let the
      OS reclaim. Verified clean (exit 0, no ASan/glibc error) on token/continue/reset.
      NEW finding surfaced: engine repeats token 9707 (sampling/output-quality bug —
      separate from the crash; see T7/new T10).
- [ ] **T5. Pin known-good xclbin set + add `r.wait()` timeout** in the NPU engine so a
      faulting kernel fails loudly instead of hanging the device. (findings §3, bug 2)
- [ ] **T6. Replace the torch decode loop** (`tools/npu_runner.py`) in
      `daemon/npu-cppd.py`. RETARGETED (2026-07-17): do NOT use the buggy
      `npu_engine_stdio` (T10). Instead proxy to `flm serve` — FastFlowLM is the
      validated native NPU engine and VERIFIED coherent on hardware today
      (`The capital of France is **Paris**.`). It already speaks OpenAI HTTP
      (`flm serve <tag> --port N`). Simplest path: retire the Python daemon entirely,
      run `flm serve` behind the Rust unified-router (which already routes to
      `qwen3-0.6b-FLM`). Eliminates torch + the Python HTTP wrapper + the Python
      decode loop in one move — a fully Python-free serving path.

## B. The correct native engine + what was blocking it (2026-07-17)

B was framed as "K-tile the O/D kernels in npu_engine_stdio." Investigation showed that
is the WRONG engine to invest in, and the right one already exists:

- `npu_engine_stdio.cpp` uses STATIC pre-compiled xclbins that only do K=1024 -> O/D
  return zero (T10). Dead-end for correctness.
- `engine/npu/src/npu_engine_universal.cpp` (999 LoC, the audit's ~42 tok/s auto-detect
  engine) uses **`FlmBridge::gen_gemm_instrs(M,N,K)`** which dlopens FLM's `libgemm.so`
  and generates NPU instructions for **arbitrary K** (incl. O K=2048, D K=3072). It also
  already uses the SAME `_exit(0)` teardown fix we applied in T4 (independent
  confirmation that fix is correct). This is the right "own-C++-engine" base.

Two pre-existing bugs were blocking `flm_bridge` (and thus npu_engine_universal). BOTH
FIXED this change:
1. **Wrong dlopen paths** — hardcoded `/opt/fastflowlm/lib/flm/*.so` no longer exists;
   libs are at `/opt/fastflowlm/lib/`. `init()` returned false ("Cannot load libgemm.so").
   Fix: `flm_dlopen()` helper tries `$FLM_LIB_DIR`, `/opt/fastflowlm/lib`,
   `/opt/fastflowlm/lib/flm`, `~/fastflowlm-build/src/lib`, then bare soname.
2. **Header missing `gen_mha_seq_`** — used in flm_bridge.cpp (init + gen_attn_instrs)
   but never declared in flm_bridge.h, so the file never compiled. Added the member.
   `flm_bridge.cpp` now compiles clean.

Dependency note: npu_engine_universal is "your C++ orchestration + FLM's dlopen'd
kernels" — semi-independent. FULL FastFlowLM independence = your own instruction
generation = the 40-column NPU2 compiler (separate strategic project, weeks+).

**Recommendation:** for a working Python-free serving path, use A (flm serve + Rust
router) now. For your own engine, invest in `npu_engine_universal` (now unblocked), NOT
npu_engine_stdio.

**UPDATE (2026-07-17): built + ran npu_engine_universal end-to-end (findings §3b).**
It builds (build_npu.sh is stale: needs -laiebu -fopenmp; include off-by-one; must pass
--model-tag), initializes via runtime aiebu ELF assembly (handles arbitrary K incl. O/D),
and executes prefill L0-L3 correctly — then HANGS at L4 with fresh AMD-Vi IO_PAGE_FAULT.
Same systemic NPU DMA/IOMMU fault that hits stdio + fusion. FLM does NOT hit it.

So the real B blocker is NOT kernel shapes (solved) but NPU DMA/IOMMU buffer management
(candidate: XRT_BO_FLAGS_HOST_ONLY vs FLM's scheme). New tasks:
- [ ] **T11. Fix build_npu.sh**: add -laiebu -fopenmp; fix flm_bridge.h include path;
      default/require a correct --model-tag (don't derive from model.q4nx filename).
- [~] **T12. Root-cause the IO_PAGE_FAULT** in home-grown engines. INVESTIGATED
      (findings §3c) — 4 hypotheses tested on hardware & ruled out: (1) concurrency
      [serialized O+GU -> still faults], (2) BO type [CACHEABLE -> ENOSPC], (3) weight
      footprint [streamed 420MB->24MB -> still faults], (4) M=512 padding [-> still
      faults]. Narrowed to: the STATIC xclbin + aiebu-ELF + xrt::ext::kernel submission
      path (shared by stdio + universal GEMMs) faults; FLM's live-instruction path does
      NOT. universal already runs ATTENTION through FlmBridge fine.
      DEFINITIVE (findings §3d): dumped EMBEDDED_METADATA — universal's xclbin and FLM's
      mm.xclbin have BYTE-IDENTICAL kernel signatures (MLIR_AIE, id 0x901, bo0..bo4).
      The torch2aie ref run_kernel_main16_q4nx.py runs the SAME kernel class and PASSES
      with 3 BOs (activation,weight,output) = universal's launch. So kernel/xclbin/BOs are
      all fine. The ONLY difference: the ref REGENERATES instructions to match; universal
      ships stale/mismatched static insts_i8_*.txt -> wrong DMA descriptors -> IO_PAGE_FAULT.
      FIX (do this): regenerate matching instructions. TOP EXPERIMENT: init QKV ctx with
      FlmBridge::gen_gemm_instrs(512,N,K,0) + init_with_instrs, pass universal's weight BO
      as bo1; if no fault -> confirmed (then fix weight layout for correctness). Alt:
      rebuild insts+xclbin as a matched pair via torch2aie _build_kernel. Until then, ship
      A (flm serve).

## T12 FIX PROVEN (2026-07-17) + integration plan

The fix is validated on hardware (findings §3e): the torch2aie toolchain builds matched
xclbin+insts that run fault-free & bit-exact on the NPU, including a FULL-LAYER q4nx
kernel (all 6 projections, one hw_context). This is the path to a working own-engine.

- [x] **T12a. Prove toolchain + matched kernel on hardware.** DONE:
      run_kernel_main16_q4nx.py --mode q and --mode full both PASS (mismatches=0, no
      IO_PAGE_FAULT). render group = no sudo. Same 0x901 kernel signature as universal.
- [ ] **T13. Build production full-layer q4nx kernel at qwen3-0.6b dims** via
      cases/full_layer_engine_generate.py + npu_build.compile_mlir (the microbench uses a
      fixed fixture; need real dims). Output: matched design.xclbin + design.bin.
- [ ] **T14. New C++ engine driving the full-layer kernel.** Replace universal's 4 int8
      contexts with ONE full-layer q4nx context: load q4nx weight chunks (make_q4nx_chunk
      layout, NOT int8 packB), one hw_context, launch per layer via xrt::ext::kernel
      (bo0=activation, bo1=q4nx weights, bo2=output records), parse the 17-dword record
      output. This is FLM's layer.xclbin architecture, self-built. Eliminates stale-insts
      fault + multi-hwctx collision + int8 roundtrip in one move.
- [ ] **T15. Wire attention + lm_head + sampler** around the full-layer kernel (attention
      already works via FlmBridge; lm_head can stay CPU initially) and validate coherent
      end-to-end generation, then benchmark vs FLM.

## Follow-ups

- [ ] **T7. Move lm_head off CPU** in the NPU engine (151,936-vocab dot product is
      ~2.2 s/token on CPU today).
- [ ] **T10. Fix repeating-token output** in `npu_engine_stdio.cpp`. ROOT-CAUSED
      (findings §3a): the int8 NPU GEMM only does a single K=1024 tile, so O-proj
      (K=2048) and D-proj (K=3072) return EXACTLY ZERO — attention & MLP contribute
      nothing, residual stays ≈ input embedding, tied lm_head predicts the input token
      back (emits last prompt token forever). NOT a sampler/RoPE bug. Fix = K-tiling +
      accumulation in `I8Ctx::go()`, OR xclbins that handle K>1024, OR retarget the FLM
      engine. Also fix hardcoded `5.0/127` activation scale (D-proj input ~8/elem
      saturates). This is NPU-kernel work, not a CPU one-liner.
- [ ] **T8. Reconcile the "Zero Python" README claim** — it is not yet literally true
      while `daemon/npu-cppd.py` + tokenization Python remain. Update the claim OR
      finish T3/T6 first. Do not edit the claim silently.
- [ ] **T9. De-orphan the good engines.** `npu_engine_stdio.cpp` and `tokenize.cpp`
      are not referenced by any CMake/Makefile/build.sh. Add build targets so the next
      engine attempt starts from the last working one instead of a fresh `engine_final_*`.

## A. Native decode target established (2026-07-17)

FLM (FastFlowLM, `/usr/bin/flm` + `fastflowlm-build`) is the native NPU decode engine:
- Verified coherent: `echo 'The capital of France is' | flm run qwen3:0.6b` -> "**Paris**".
- Native C++/NPU, no Python, no torch in its decode path.
- Exposes `flm run` (interactive) and `flm serve` (OpenAI `/v1/chat/completions`).
- The Rust router's SMALL_MODEL (`qwen3-0.6b-FLM`) already targets it.

Consequence: `npu_engine_stdio` (the orphan we fixed in T4 / diagnosed in T10) is
EXPERIMENTAL ONLY — an independent-engine effort, not required for a working serving
path. B (K-tiling the O/D kernels) is now OPTIONAL/strategic: pursue it only to own the
full stack independent of FastFlowLM; it is not on the critical path to Python-free serving.

## Decisions taken in this change

- Defer the decode-engine swap (T6) until T4+T5 land — do not replace working Python
  with crashing C++.
- Router → Rust (consistent with existing `rust/onebit`; matches the "if you can't kill
  Python, make it Rust" rule).
- Offline tooling (LoRA train, hf_to_q4nx, kernel-gen, export_tokenizer) stays Python —
  not in the serving path, so it does not violate the "no Python in the hotpatch" rule.
