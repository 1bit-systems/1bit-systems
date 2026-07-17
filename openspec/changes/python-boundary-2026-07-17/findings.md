# Findings — Python↔Native boundary, verified on hardware (2026-07-17)

Host: Strix Halo (gfx1151 + XDNA2). All results below are from direct build/run,
not inference from source reading. Reproduction commands included.

## 1. The runtime Python touch-points (what actually sits in the request path)

| File | Role | In LLM hot path? |
|---|---|---|
| `daemon/npu-cppd.py` | OpenAI HTTP API → NPU | **YES** — tokenizes per request AND spawns `python3 tools/npu_runner.py` (torch venv) as the decode loop |
| `unified-router.py` | keyword router / proxy in front of lemond | request path, but **glue only** (no tokenize, no inference) |
| `engine/lora/*.py`, `engine/npu/kernel/*.py`, `engine/fusion/hf_to_q4nx.py`, `engine/fusion/export_tokenizer.py`, `engine/fusion/tokenize.py` | training / model-convert / kernel-gen / tokenizer-export | **NO** — offline/build-time |

The C++/HIP engine core itself is Python-free, as branded. The leak is the **serving
layer**: one daemon still tokenizes in Python and runs its NPU decode loop in torch.

## 2. Native tokenizer — VERIFIED READY (bit-exact)

Source: `engine/fusion/tokenize.cpp` (pure C++17, no deps; C ABI + CLI).

```bash
c++ -std=gnu++17 -O2 -o /tmp/tokenize engine/fusion/tokenize.cpp
TOK=~/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json
/tmp/tokenize --model "$TOK" --encode "Hello, world!"   # -> 9707 11 1879 0
/tmp/tokenize --model "$TOK" --decode 9707 11 1879 0     # -> Hello, world!
```

`9707 11 1879 0` is **identical** to the Python HF reference documented in
`engine/fusion/tokenize.py`. Supports ByteLevel/Metaspace/Split pre-tokenizers,
GPT-2 byte-level, BPE merge-ranks, byte fallback.

**C ABI available for direct linking:**
```c
void* tokenizer_load(const char* json_path);
int   tokenizer_encode(void* tok, const char* text, int* out_ids, int max_ids);
void  tokenizer_free(void* tok);
```

→ Ready to replace `src/server/rest_handler.cpp:518 simple_tokenize()` (a word-split
stub whose own comment says "In production, this would use the model's actual
tokenizer"), and to remove the `from tokenizers import Tokenizer` / tokenize-subprocess
paths from the daemons.

## 3. Native NPU decode engine — BUILDS + RUNS, but BLOCKED by 2 bugs

Source: `npu-infer/src/npu_engine_stdio.cpp` (per-token stdio stepper; protocol
`{"token":N}` / `{"continue":true}` / `{"reset":true}` → `{"token":M,"ms":X}`).

**Build (verified clean):**
```bash
gcc -std=c11 -O3 -march=native -ffast-math -c engine/npu/src/dequant_q4nx.c -o /tmp/dequant.o
c++ -std=gnu++17 -O3 -march=native -ffast-math -g -I npu-infer/include \
    -o /tmp/npu_engine_stdio npu-infer/src/npu_engine_stdio.cpp /tmp/dequant.o -lxrt_coreutil
```

**Run (verified on XDNA2):** expects xclbins in a relative `int8/` dir; hardcoded to
Qwen3-0.6B dims (H=1024, NC=28, NV=151936).

- With `int8_32tile_v6` xclbins → **NPU hang** (>120 s for 1 token). dmesg:
  ```
  amdxdna: AMD-Vi: IO_PAGE_FAULT ...
  aie2_config_cu: Lookup GEM object failed
  aie2_hwctx_restart: Config cu failed, ret -22
  ```
  Root cause class: NPU DMA to IOMMU-rejected addresses; `r.wait()` (line 44, no
  timeout) then blocks forever. Same failure family as the 2026-07-14 fusion audit.

- With base `int8_32tile` xclbins → **works**: `{"token":9707,"ms":2242}` then
  `free(): invalid size` (heap corruption on the C++ side).

**Fixes:**
1. **`free(): invalid size`** — FIXED (2026-07-17). ASan showed it is NOT a data heap
   bug: it's a null-vtable SEGV in `I8Ctx::~I8Ctx()` at exit, destroying
   `shared_ptr<xrt::xclbin_impl>`. Root cause = XRT static-destruction-order fiasco
   (XRT globals torn down before these local dtors run). Fix = `std::_Exit(0)` after
   the stdin loop, skipping the cross-boundary teardown (OS reclaims). Token gen itself
   was already correct; only shutdown crashed. Verified: exit 0, no ASan/glibc error.
   NOTE: output repeats token 9707 — separate sampling/RoPE bug (tracked T10).
2. **xclbin pinning** — the engine hardcodes `int8/` + `qwen3_0_6b` names; `v6`
   IOMMU-faults. Pin a known-good set (base `int8_32tile`) and add a `wait()` timeout
   so a bad kernel dispatch fails loudly instead of hanging.

Also: lm_head runs the full 151,936-vocab dot product on **CPU** (~2.2 s/token). That
belongs on GPU/NPU for a "perfect" engine.

### 3a. T10 root cause (2026-07-17): O/D projections return exactly zero (K>1024)

Repeating-token output was root-caused by instrumenting layer 0 (per-tensor L2):

```
[L0] h_in=0.729                              input embedding OK
[L0] qkv_out=34.493   (K=1024,N=4096)        WORKS
[L0] attn_in(at)=29.312                      O-proj input healthy
[L0] attn_out=0.000   (K=2048,N=1024)        O-proj kernel returns ZERO
[L0] mlp_in(su)=440.123 gu_out=187.838       D-proj input healthy (GU K=1024 WORKS)
[L0] mlp_out=0.000    (K=3072,N=1024)        D-proj kernel returns ZERO
```

**The NPU int8 GEMM only computes a single K=1024 tile.** QKV and GU (K=1024) work;
O (K=2048) and D (K=3072) return exactly zero because the `int8_32tile` xclbin does no
K-accumulation across tiles. With attention and MLP outputs zeroed, the residual stream
stays ≈ the input embedding, so the tied-embedding lm_head predicts the input token back
→ the model emits the last prompt token forever (`374 374 374...` / "is is is").

Secondary issue seen in the same trace: activation scale is hardcoded `5.0/127` for every
matmul, but `mlp_in(su)` L2=440 over 3072 (~8/elem) far exceeds 5.0 — D-proj activations
would saturate even once the kernel works. Per-matmul (ideally dynamic per-token)
activation scale is needed.

**Fix paths (both are real NPU-kernel work, not CPU one-liners):**
1. K-tiling + accumulation in `I8Ctx::go()`: split K into 1024-wide chunks, run the
   K=1024 kernel per chunk against the matching weight slice, accumulate int results
   before dequant. Requires per-shape instruction streams / BO sizing that support this.
2. Build/obtain xclbins that natively handle K=2048 (O) and K=3072 (D).
3. Or retarget the validated FLM engine (94 tok/s) which already generates coherent text.

Until one of these lands, `npu_engine_stdio` builds and runs cleanly (T4) but does NOT
produce correct output. Do not wire it into serving yet.

**Alternative:** retarget the already-validated FLM engine (94 tok/s) instead of
hardening this orphan.

## 3b. npu_engine_universal end-to-end run (2026-07-17): builds + runs, but IOMMU-faults

Built and ran the CORRECT engine (`engine/npu/src/npu_engine_universal.cpp`) on a
freshly-reloaded (clean) NPU. Results:

- **Build**: `build_npu.sh` is STALE — missing `-laiebu` and `-fopenmp`, and the source's
  `#include "../../npu-infer/include/flm_bridge.h"` is off-by-one for the canonical
  layout. Working build:
  ```
  g++ -std=c++23 -O3 -fopenmp -DMODEL_qwen3_0_6b \
    -Iengine/npu -Inpu-infer/include -Iengine/npu/src -I/usr/include \
    -o npu_engine_universal engine/npu/src/npu_engine_universal.cpp \
    npu-infer/src/flm_bridge.cpp dequant_q4nx.o \
    -lxrt_coreutil -lxrt_core -luuid -lm -ldl -laiebu
  ```
- **model_tag gotcha**: derived from the model FILENAME (`model.q4nx` -> "model"), so it
  looks for `insts_i8_QKV_model.txt` and prints `FAIL QKV`. Must pass
  `--model-tag qwen3_0_6b` explicitly (or rename the model file).
- **Runtime**: with the tag fixed it initializes all contexts (aiebu assembles the insts
  into ELF modules at runtime — this IS the arbitrary-shape mechanism, and it handles the
  O/D K>1024 shapes that the static-xclbin stdio engine could not), and correctly
  executes prefill layers L0-L3 (all 5 projections q,a,o,g,d each), then **HANGS at L4
  with fresh `AMD-Vi: IO_PAGE_FAULT` events** — an NPU DMA/IOMMU fault, not a compute
  hang. Same systemic fault class that hits `npu_engine_stdio` and the fusion engine
  (2026-07-14 audit).

**Key contrast:** FastFlowLM (`flm run`) does NOT hit this — it ran clean and coherent on
the same box (A). So the blocker for a home-grown engine is NOT K-tiling or wiring (both
solved: aiebu gives arbitrary K, flm_bridge now compiles). It is the **NPU DMA/IOMMU
buffer management** in the static-xclbin + aiebu submission path.

### 3c. T12 investigation — four hypotheses tested on hardware, all ruled out (2026-07-17)

The IO_PAGE_FAULT is intermittent (faults at varying layer/context: L4-QKV, L2-GU),
always in a `HOST_ONLY` BO page. Tested, each with a fresh `modprobe -r/amdxdna` reload:

1. **Concurrency** (parallel O+GU across 2 hwctxs) — serialized O then GU. STILL FAULTS.
   -> not a concurrent-submission race.
2. **BO type** — `HOST_ONLY` -> `CACHEABLE`. `CREATE_BO err=-28 ENOSPC` (weights exceed
   the NPU carveout). -> can't use CACHEABLE for full-resident weights; not the fix.
3. **Weight footprint** — refactored to STREAM weights (one resident weight BO per
   context, upload current layer on demand, ~24MB vs ~420MB). STILL FAULTS.
   -> not total IOMMU footprint of weights.
4. **M-size overrun** — `flm_bridge.h` says FLM assembles at M=512 padded; padded bA/bC
   to M>=512 (XM default is 128). STILL FAULTS. -> not a simple bA/bC row overrun.

**Narrowed root cause:** the fault is in the **static xclbin + aiebu-assembled-ELF +
`xrt::ext::kernel` submission path**, which BOTH broken engines (npu_engine_stdio,
npu_engine_universal GEMMs) share, and which FLM does NOT use. FLM generates instructions
live via its own libs (`libgemm.so` `Gemm::generate_seq`, `move_weights`) through ONE
context and does not fault. Notably, universal ALREADY routes ATTENTION through FlmBridge
(live gen) successfully — only the projection GEMMs use the faulting static-insts path.

### 3e. T12 FIX PROVEN ON HARDWARE (2026-07-17): toolchain-built matched kernel runs fault-free

The torch2aie/mlir-aie toolchain works on THIS box and builds matched xclbin+insts pairs
that run on the NPU with ZERO IO_PAGE_FAULT and bit-exact output:

```
cd ~/torch2aie/examples/qwen3-decode-layer
export PYTHONPATH=~/torch2aie/toolchain/mlir_aie/python:$PWD:$PWD/cases
export LD_LIBRARY_PATH=~/torch2aie/toolchain/xrt/lib
~/torch2aie/.venv/bin/python3 run_kernel_main16_q4nx.py --mode q      # PASS mismatches=0
~/torch2aie/.venv/bin/python3 run_kernel_main16_q4nx.py --mode full   # PASS mismatches=0
```

- `--mode full` runs ALL SIX projections (q,k,v,o,upgate,down) in ONE kernel / ONE
  hw_context: `PASS ... mismatches=0, max_abs=7.6e-06, NPU 579.7us`, no fault.
- Access: user is in `render` group -> NO sudo needed. Toolchain in
  `~/torch2aie/toolchain/{aietools,mlir_aie,xrt}`, venv `~/torch2aie/.venv`.
- The generated kernel has the SAME `dpu_kernel_id=0x901` / bo0..bo4 signature as
  universal's and FLM's -> universal's `xrt::ext::kernel` launch path can drive it.
- Native **Q4NX** (matches `model.q4nx`) — no int8 roundtrip.
- Artifacts: `build/qwen3-kernel-main16-q4nx-full/{design.xclbin,design.bin}`.

This is the end of the T12 mystery: a freshly-built MATCHED instruction stream runs
fault-free, confirming the fault was the stale/mismatched static `insts_i8_*.txt`. The
full-layer single-kernel design ALSO eliminates the multi-hwctx column collision (one
context, not four). Integration plan in tasks.md (T13).

Note: the `--mode full` microbench uses a fixed test fixture (120 weight chunks). The
production full-layer design is `cases/full_layer_engine_generate.py` — that is the target
to build at real qwen3-0.6b dims for integration.

### 3d. T12 DEFINITIVE root cause (2026-07-17): the static instruction stream

Dumped kernel metadata (`xclbinutil --dump-section EMBEDDED_METADATA`) for both
universal's `final_i8_QKV_qwen3_0_6b.xclbin` and FLM's `mm.xclbin`: **byte-identical
kernel signature** — same `MLIR_AIE`, `dpu_kernel_id=0x901`, same args
`opcode,instr,ninstr,bo0..bo4`. So the xclbin/kernel is NOT the problem.

The generic kernel declares 5 BOs but this GEMM class uses 3. Confirmed by the WORKING
torch2aie reference `run_kernel_main16_q4nx.py`, which runs the SAME kernel class on the
NPU and PASSES numeric validation with exactly 3 buffers:
`npu_build.run(handle,[activation_buf, weight_buf, record_buf])`. universal's 3-BO launch
(`k(3,0,0,*bA,*bB,*bC)`) matches — the 5-vs-3 concern is a red herring.

The decisive difference: the torch2aie reference REGENERATES instructions to match the
xclbin (`_build_kernel`), while universal ships PRE-GENERATED `insts_i8_*.txt` paired with
`final_i8_*.xclbin`. So the fault is the **static instruction stream** — stale/mismatched
vs. what the kernel expects (wrong DMA descriptors -> out-of-range NPU DMA -> IO_PAGE_FAULT).
All other hypotheses (kernel, xclbin, BO count/type, concurrency, weight footprint, M-pad)
are eliminated.

**Two concrete fixes (both are toolchain work, need hardware iteration):**
1. **Regenerate matching instructions via FLM**: `FlmBridge::gen_gemm_instrs(512,N,K,0)`
   produces instructions for THIS kernel. Since the kernel takes weights as a BO arg
   (bo1, per the torch2aie reference), the earlier "weights fused into FLM context" worry
   is likely wrong — `move_weights` should emit DMA-from-bo1. TOP EXPERIMENT: init one
   GEMM ctx (QKV) with `gen_gemm_instrs` + AttnCtx-style `init_with_instrs`, supply
   universal's own weight BO as bo1, verify no fault + correct output. If weight LAYOUT
   differs, fix packB to FLM's tiling (output wrong but NO fault would already prove it).
2. **Regenerate via torch2aie** (`_build_kernel` path) to get a matching xclbin+insts pair.

This supersedes the "mm.xclbin single-kernel" note below (the kernels are already identical;
the issue was always the instruction stream, not the xclbin).

**The fix (concrete, confirmed by FLM's xclbin layout):** FLM ships ONE generic GEMM
kernel per model — `<xclbins>/Qwen3-0.6B-NPU2/{attn,dequant,layer,mm}.xclbin`. `mm.xclbin`
handles ALL GEMM shapes; the K/N are parameterized by instructions generated live by
`libgemm.so` (`Gemm::generate_seq`). universal instead uses 4 shape-baked xclbins
(`final_i8_{QKV,O,GU,D}_*.xclbin`) + static `insts_i8_*.txt` — that is the faulting path.
universal ALREADY runs attention through FLM's `attn.xclbin` + FlmBridge without faulting.

So: replace the 4 static GEMM contexts with a single `mm.xclbin` context driven by
`FlmBridge::gen_gemm_instrs(M,N,K)` + the `init_with_instrs` pattern (as AttnCtx does),
and adopt FLM's weight path (`move_weights`/`gen_dequant`). This swaps the faulting
submission path for FLM's proven one while keeping universal's C++ orchestration — i.e.
a semi-independent own-engine (your C++ + FLM's kernels). Full independence (own
instruction gen) remains the 40-col-compiler project.

Effort: substantial refactor of I8Ctx + the per-layer weight path, with intermittent-fault
hardware validation. Experiments live in /tmp/uni_{serial,cache,stream,pad}.cpp (not
committed; none fixed the fault, so the engine source is unchanged).

**Bottom line:** for a working Python-free NPU serving path, use A (FLM). Pursue B only
as a strategic own-the-stack effort, and target the IOMMU/DMA stability problem, not the
kernel-shape problem.

## 4. Router — PORTED TO RUST + TESTED

`unified-router.py` → std-only Rust (zero external crates, 427 KB static binary).
Reference impl in `reference/unified-router-rs/` in this change dir. Behavior verified
identical against a mock backend:

| Input | Python logic | Rust result |
|---|---|---|
| `auto` + plain greeting | → NPU (SMALL) | ✅ `qwen3-0.6b-FLM` |
| `auto` + "code" keyword | → GPU (BIG) | ✅ `Qwen3-8B-4bit` |
| `user.Unified` + tools | → GPU | ✅ |
| explicit `npu` / `gpu` | → SMALL / BIG | ✅ |
| unknown model | passthrough unchanged | ✅ |
| GET | passthrough | ✅ |

Recommended: fold this routing policy into the existing `rust/onebit` frontend rather
than shipping a second router binary long-term. The std-only version is the portable
reference / drop-in.
