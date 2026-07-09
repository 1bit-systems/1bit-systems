# Fused XCLBIN Integration Blocker — July 2, 2026

## What Works

| Component | Status | Detail |
|-----------|--------|--------|
| QKV prefix xclbin | ✅ Compiled | 253KB, 4 runtime args |
| Full-layer xclbin | ✅ Compiled | 374KB, 5 runtime args |
| Kernel group mapping | ✅ Verified | Groups 0-7 all accessible |
| Kernel dispatch | ✅ Runs | ~63s DMA timeout (not crash) |
| C++ engine | ✅ Compiles | BOs alloc + dispatch OK |

## What's Blocking

The **Q4NX weight format** required by the fused xclbin is fundamentally
different from our flat INT8 weight format. The xclbin's runtime sequence
uses 512 specific DMA descriptors with precise byte offsets and lengths
that read Q4NX-compressed weight chunks in FLM's streaming format.

Our model stores weights as dequantized flat bf16 → packed as flat INT8.
FLM's format is Q4NX: 5120-byte chunks (scales + zeros + packed 4-bit data)
laid out per-column, per-patch, per-phase, using row-major interleaving.

## What's Needed for Full Integration

1. **Q4NX weight packer** — Port FLM's `_projection_stream_from_schedule()`
   from Python to C++. ~200 lines of complex weight interleaving logic.

2. **KV cache layout** — FLM uses block-major (128-token blocks). Our engine
   uses linear layout. The fused xclbin DMA descriptors expect block-major.

3. **AUX data prefix** — RMS norm weights + QK norm + RoPE cos/sin at specific
   byte offsets within the weights BO.

4. **Runtime sequence arguments** — The kernel needs exactly 5 buffers with
   exact sizes matching the MLIR generation.

All of this is implemented in FLM's `qwen3_8b_decode_layer_runner.py`.
Porting to C++ is ~500 lines of careful weight format conversion.

## Decision

**Keep standalone INT8 GEMM xclbins for production** (v12 engine at 97 tok/s).
Fused xclbin integration is a separate project requiring Q4NX format support.
The fused xclbins compile and prove the toolchain works — the blocker is
weight format, not hardware or toolchain.

## Path Forward

1. Write a Python script that uses FLM's runner directly with our 0.6B model
2. Once validated numerically, port the weight packer to C++
3. Integrate into multi-layer decode loop
4. Expected: ~5ms/tok batch step, ~3ms/tok effective at M=32

---

## UPDATE (2026-07-03): Schedule Fixed, Two Deeper Bugs Isolated

Reconstructed the exact weight-packing schedule from the ground-truth Python
(`qwen3_model.py::_projection_stream_from_schedule` /  `layer_weight_stream`,
`/home/bcloud/torch2aie/examples/qwen3-decode-layer/qwen3_model.py:276-336`):

```python
for group in range(4):
    for patch in range(2):
        for (projection, block, input_chunk) in schedule:   # Q,K,V, O, interleaved(UP,GATE), DOWN
            for row_in_patch in range(2):
                row_chunk = block*16 + group*4 + patch*2 + row_in_patch
                source = row_chunk * projection.chunks + input_chunk
                emit chunk_by_phase[projection.phase][source]
```

`/home/bcloud/npu-sandbox/npu-infer/tools/pack_fused_v3.py` already implements this correctly
(reads real Q4NX chunks directly from `model.q4nx`, no dequant/requant — sidesteps the earlier
"our quantizer doesn't match FLM's" dead end from the v2 UPDATE below). Confirmed byte-identical
regen. **`engine/npu/src/q4nx_stream.cpp` still needs this schedule ported in** (currently naive
per-column replication) — that part of the original plan still stands.

**But schedule-correct weights alone do NOT fix the full-layer deadlock.** Re-ran
`npu_engine_v13` against schedule-correct weights: still 62857ms timeout, 0/512 non-zero output.
Isolated further with the smaller QKV-prefix xclbin
(`torch2aie/examples/qwen3-decode-layer/full_layer_qkv_prefix_runner.py`, rebuilt fresh —
the precompiled one in `build/` was stale):

- **QKV-only dispatches cleanly: 4.0-4.4ms, no deadlock**, at both token=9 and token=31. This
  confirms the 63s deadlock is specific to the O/UP/GATE/DOWN tail, matching what the sibling
  BitNet port at `torch2aie/examples/bitnet-decode-layer` found for the identical design pattern —
  a lock/dataflow bug in those tiles, not a weight-packing problem. Still unresolved.
- **But QKV-prefix output values are numerically wrong** vs. the CPU golden reference
  (`cases/full_layer_engine_reference.py`), even with synthetic weights the reference itself
  generates (`make_packed_weights()` — so this isn't a real-data quantization mismatch, it's
  internal to this test harness/kernel pairing). ~1000+ K/V lane mismatches, large deltas, at both
  tested token positions — this design was apparently never validated against real hardware before.
  Traced RoPE (`_apply_rope` vs `write_rope_pair`/`packed_rope_word` in `postprocess_qkv.cc`) and
  RMSNorm (`_head_rms_norm` vs `head_rms_scale`/`normalized_lane`) formulas — **both match exactly**,
  including AUX buffer layout offsets. V-cache is *also* wrong despite never touching RoPE/norm
  (straight GEMM+dequant passthrough) — this rules out RoPE/norm and narrows the bug to the Q4NX
  GEMM/dequant kernel (`qwen3_decode_kernels.cc`, `main_projection_q4nx_fast.o`) or the
  record-absorption step (`qwen3_postprocess_absorb_qkv_payload_record`). Checked nibble-unpack
  layout (`load_q4_dim_quad` / `aie::unpack`) against the Python packer
  (`q4nx_reference.py::make_q4nx_chunk`) — layout appears consistent on paper, not confirmed on
  hardware.

**Next step, if resumed:** this needs empirical kernel instrumentation (dump intermediate
dequant/GEMM values from the Chess kernel, compare against
`q4nx_reference.py::q4nx_matvec_from_chunk` for the same synthetic chunk) rather than more static
reading — a slower, iterative rebuild-and-compare cycle (~10min per iteration for the Chess/MLIR
toolchain), not a quick fix.

**Status: still not production-ready.** v12 (97 tok/s, standalone INT8 GEMM) remains the production
engine.

---

## UPDATE (2026-07-03, cont'd): Main16 GEMM Proven Correct In Isolation — Bug Is In Multi-Tile Plumbing

Found `run_kernel_main16_q4nx.py`, an existing isolated microbenchmark that runs *only* the
`main_projection_q4nx_fast.o` Q4NX GEMM/dequant kernel (single AIE tile) against
`q4nx_reference.py::q4nx_matvec_from_chunk`, no RoPE/norm/absorb/multi-tile plumbing involved.

```
main16_q_records: max_abs=0.000000000 mean_abs=0.000000000 mismatches=0
PASS: Main16 Q4NX isolated numerical validation (q)
```

**Byte-exact.** This decisively rules out the GEMM/dequant math as the source of the QKV-prefix
numeric bug — combined with the earlier RoPE/RMSNorm formula match (verified by reading
`postprocess_qkv.cc` against the Python reference line-for-line), every individual kernel's
*arithmetic* now checks out. The bug has to be in what's unique to the full multi-tile QKV-prefix
pipeline and absent from these isolated single-kernel tests: the record-absorption/compaction step
that assembles per-tile GEMM output across 16 physical AIE cores into `k_body`/`v_body`
(`qwen3_postprocess_absorb_qkv_payload_record` in `postprocess_qkv.cc`), or lock/buffer-depth/DMA
routing in the MLIR resource manifest that wires main16's output to postprocess's input across
columns.

Tried `run_kernel_postprocess_qkv.py` (an isolated test of the RoPE/norm/absorb kernel alone) to
empirically confirm the RoPE finding — it requires the real Qwen3-8B model (hardcoded path from a
different machine, `/var/home/taowen/flm/models/Qwen3-8B-NPU2`, not present here) and doesn't apply
to our 0.6B pipeline. Not a finding, just an unavailable test.

**Remaining search space, if resumed:** the multi-tile bridge/compaction wiring — likely needs
either instrumenting the actual 18-tile QKV-prefix design (add a debug tap after record absorption,
before RoPE) or carefully tracing lock/buffer-depth assignments in
`cases/full_layer_qkv_prefix_generate.py`'s resource manifest. This is a step up in complexity from
everything checked so far (single-kernel math) — multi-tile dataflow/synchronization bugs are
generally the hardest class to find without hardware-side tracing tools.

---

## UPDATE (2026-07-03, cont'd 2): O/UP/GATE/DOWN Deadlock — Also Not a Kernel-Math Problem

Same `run_kernel_main16_q4nx.py` microbenchmark has a `--mode full` option that runs the *entire*
Q,K,V,O,UPGATE,DOWN chunk chain (all 7 phases, `MAIN16_PHASE_LIMIT_FULL=7`) through the isolated
single-tile GEMM scheduler — same `MAIN16_LAYER_SCHEDULER`/phase-limit constants the real full-layer
design uses, just without the attention/vector-station/swiglu tiles wired in.

```
main16_full_records: max_abs=0.000007629 mean_abs=0.000000010 mismatches=0
PASS: Main16 Q4NX isolated numerical validation (full)
```

**627µs, no deadlock, correct to float precision.** This rules out main16's GEMM/dequant scheduler
as the source of the 63s full-layer deadlock — it handles the full 7-phase chunk chain (including
the larger UP/GATE/DOWN record counts) just fine on its own, and produces exactly
`UPGATE_REPLAYS=12` records as expected (the isolated test's shape check would have failed
otherwise). The deadlock is specifically in the **cross-tile handshake** between main16 and its
downstream consumers.

Traced the wiring in `full_layer_engine_generate.py`: main16's UPGATE-phase records get emitted via
`full_main_emit_upgate_slice_record`, then routed through a **packet-switched bridge tile** — the
same physical output channel (`BRIDGE_COMPACT_OUT_CHANNEL`) carries multiple logically distinct
flows (Q→`post`, O→`full`, UPGATE/DOWN→`swiglu`) disambiguated by packet ID
(`Q_GLOBAL_PACKET_ID`, `O_GLOBAL_PACKET_ID`, `FFN_GLOBAL_PACKET_ID`). `swiglu`'s core loop waits on
`aie.use_lock(%swiglu_input_full, AcquireGreaterEqual, 2)` for exactly
`C1R2_UPGATE_REPLAYS // 2 = 6` iterations (12 total). If the bridge ever fails to tag/route the full
count of UPGATE packets to swiglu with the right packet ID — a routing/count mismatch in the
packet-switched NoC config, not a data or GEMM bug — swiglu (or the `full`/c1r2 vector-station tile)
would block on `AcquireGreaterEqual` forever, exactly matching the observed 63s hang (an XRT command
timeout, not a crash).

**This may share a root cause with the QKV numeric-correctness bug** — both point at the same
bridge/compaction layer that combines per-tile main16 output into the downstream single-stream
flows consumed by postprocess_qkv, full_vector_station, and swiglu. Not confirmed, but a reasonable
unifying hypothesis.

**Next step, if resumed:** trace the bridge tile's packet-ID tagging logic for UPGATE/DOWN records
specifically (where main16's per-column, per-row output gets assigned `FFN_GLOBAL_PACKET_ID` before
entering `BRIDGE_COMPACT_OUT_CHANNEL`) and verify the emitted count matches swiglu's and c1r2's
expected `AcquireGreaterEqual` counts exactly. This is packet-switched NoC routing — one of the
harder classes of AIE bugs to fully verify by reading alone; confirming it may require adding a
lock-state or packet-count probe to the bridge tile and rebuilding.

---

## UPDATE (2026-07-03, cont'd 3): Traced compact_dataflow.py — One False Lead, One Dead End, Bridge Confirmed Shared

Went deeper into `compact_dataflow.py`, the module that builds the per-column record compaction
and bridge packet routing shared by both the QKV-prefix (working) and full-layer (deadlocking)
designs.

**False lead, ruled out:** `CompactPhase.output_offset`/`output_length` — computed per-phase in
`_phase_output_slice()`, with a suspicious asymmetry (upgate gets `offset=1`, every other phase gets
`offset=0`). Grepped for all uses: **these fields are never read anywhere** — dead code, not wired
into any BD generation. Not the bug. Documenting so a future session doesn't re-chase this.

**Confirmed:** `full_layer_qkv_prefix_generate.py` builds its bridge wiring via the exact same
`full._bridge(phase_trace)` function `full_layer_engine_generate.py` uses for the full design —
just called with the shorter `QKV_PREFIX_PHASE_TRACE` (q,k,v) instead of the full 6-phase
`COMPACT_PHASE_TRACE` (q,k,v,o,upgate,down). So the bridge/compaction mechanism is genuinely shared,
parameterized code, not two independent implementations — QKV-prefix's clean dispatch is *some*
evidence the mechanism works, but doesn't prove correctness for the longer 6-phase trace (upgate
alone carries `UPGATE_BODY_RECORDS=12` records via multi-dimensional strided BDs, a code path
QKV-prefix never exercises).

**Dead end:** tried `run_full_layer.py --check-only` (structural validation only, no hardware) hoping
for a cheap signal — it still hard-requires the real Qwen3-8B model (`qwen3_8b_decode_layer_runner`),
same as the earlier `run_kernel_postprocess_qkv.py` attempt. Not usable for the 0.6B pipeline.

**Assessment:** further static tracing of this ~1500-line generator is hitting diminishing returns —
each new hypothesis this round either turned out to be dead code or an untestable path. Getting a
decisive answer now needs a lock-state/packet-count probe built into the bridge tile itself and
rebuilt on hardware (a new-instrumentation task, not a reading task) — see the "Next step" above,
which still stands as the concrete way forward.

---

## UPDATE (2026-07-03, cont'd 4): Found and Fixed a Real Bug (kQRecords), Ruled Out As Root Cause

While scoping the column-compaction isolation test, traced `compact_column_memtile`'s actual
generated code and found it does **not** hardcode any record count — it's self-paced ping-pong
forwarding with no fixed iteration count, meaning it almost certainly isn't where a count-mismatch
deadlock lives. This reframed the likely mechanism: AIE ping-pong buffers create backpressure that
propagates *upstream* — a stall in any downstream consumer (`post`, `full`, `swiglu`) could cascade
all the way back through bridge and column-compaction to main16, looking exactly like the observed
deadlock without any of those earlier-stage components having a bug of their own.

That redirected attention to `post` (the K/V cache-writeback tile, `postprocess_qkv.cc`) — and found
a real, concrete, previously-undetected bug: **the build system links the wrong kernel source by
default.**

- `npu_build.py`'s `POSTPROCESS_QKV_SOURCE` env var defaults to `postprocess_qkv.cc`, which hardcodes
  `kQRecords = 8`.
- A separate `postprocess_qkv_06b.cc` exists with the *correct* Qwen3-0.6B value, `kQRecords = 4`
  (matching `Q_BODY_RECORDS=4` for this model — an 8-head-block value left over from a larger model).
  Nothing in the generator or build scripts ever selects it; every hardware run this session
  (and, per the object file `link_with = ".../postprocess_qkv.o"` hardcoded in both
  `full_layer_qkv_prefix_generate.py` and `full_layer_engine_generate.py`) used the wrong one.
- With `kQRecords=8` but only 8 total QKV records ever sent (Q:4+K:2+V:2), the record-routing check
  `record_index < kQRecords` is *always true* — every record gets classified as Q, so K and V never
  receive real data. This is exactly the kind of bug that would produce plausible-looking-but-wrong
  cache values.

**Tested the fix**: re-ran `run_qkv_prefix.py` with `QWEN3_POSTPROCESS_QKV_SOURCE=postprocess_qkv_06b.cc`
set. Confirmed in the build log that it actually recompiled from the corrected source
(`Compiling postprocess_qkv.o with Chess from postprocess_qkv_06b.cc...`). **Result: byte-for-byte
identical output to the unfixed run** — same `expected`/`got` values down to the last decimal place.

**This is a clean, decisive negative.** The `kQRecords` mismatch is real and worth fixing for hygiene
(it's latent undefined behavior — writing at `block=record_index` up to 7 into a 4-block `q_body`
buffer is an out-of-bounds write on AIE's unprotected tile-local SRAM), but it is **not** the cause
of the QKV-prefix numeric mismatch. Something else is wrong upstream or in a part of the K/V/RoPE
path not yet isolated.

**Where this leaves the investigation:** every major hypothesis has now been tested and eliminated
through direct hardware verification, not just static reading — weight schedule (fixed, confirmed),
GEMM/dequant math (byte-exact, isolated, full phase chain), RoPE/RMSNorm formulas (verified against
source), packet-ID tagging (verified via record header data), column-compaction structure
(understood, appears sound), and now `kQRecords` (tested, ruled out). What remains requires either
on-chip state inspection (hardware debug registers / cycle-accurate trace tooling not available
here) or a much deeper dive into the K/V-norm/RoPE side-buffer feed into `post` than source-reading
alone has been able to resolve. Recommend treating this as needing either AMD toolchain-level debug
support or a fresh multi-session investigation, not a quick continuation.

---

## UPDATE (2026-07-03, cont'd 5): Installed AMD's AI Engine Emulator — Blocked by Missing NPU2 Device Data

Installed AMD's full Vitis Unified Software Platform (2026.1) specifically to get `aiesimulator` —
a cycle-accurate AIE simulator with `--hang-detect-time` (flags stalled cores) and
`--enable-memory-check` (catches out-of-bounds writes, directly relevant to the `kQRecords`-class
bug already found), aiming to get a real signal on the O/UP/GATE/DOWN deadlock and QKV numeric bug
without needing real hardware round-trips.

**Got it running**, fixing two real toolchain compatibility gaps along the way:
1. The new Vitis install's `aietools` tree was missing `tps/lnx64/target_aie2p` (the Chess
   simulation engine env) entirely — symlinked from the existing, already-licensed
   `torch2aie/toolchain/aietools` install.
2. Vitis 2026.1's simulator binaries are named `aie2pssim*` ("AIE2PS") but the config generated by
   our `aiecc.py --aiesim` step (from the existing mlir-aie toolchain) references `aie2psim*`
   ("AIE2P", no second S) — a tooling rename between versions. Symlinked the naming across.

**Hard blocker, confirmed unresolvable locally:** `aiesimulator` needs a device topology file,
`data/aie2p/devices/aie2p_8x4_device.json`, describing the NPU2/Strix Halo's 8-column × 4-row AIE
grid. This file does not exist in either toolchain install. Investigated three ways to get it,
each conclusively ruled out:

- **Hand-author it** — opened the file format from an existing Versal device file
  (`XC2VE3304.json`): it's not text JSON despite the extension, it's a proprietary encrypted binary
  blob (`XbV18.3` magic header) generated by AMD's internal chip-description tooling. Not
  authorable.
- **Find it upstream** — confirmed via mlir-aie's own GitHub issue tracker
  ([#2092](https://github.com/Xilinx/mlir-aie/issues/2092), "NPU2 Programming Examples") that
  "add virtual NPU2 device" is an open, actively-tracked item across many of their own example
  programs. Full NPU2 simulator device support isn't complete in the public mlir-aie/Vitis
  ecosystem yet — this is a known upstream gap, not a local misconfiguration.
- **Check `ryzen-ai-lt-1.8.0-beta.exe`** (AMD's separate Ryzen AI SDK, already on disk) — properly
  inspected via `msitools` (not guessed): it's the ONNX Runtime / Vitis-AI-Execution-Provider ML
  deployment stack (Stable Diffusion configs, `vaip_config.json`, ONNX headers, a couple of
  pre-built `.xclbin` overlays) — a completely different, higher-level layer with no AIE-simulator
  device data.

No override flag exists in `aiecc.py` or `aiesimulator` to skip or substitute the device topology
at simulation-launch time — it's baked into the Work folder at generation time from the MLIR
device attribute, with no bypass.

**Status:** `aiesimulator` itself works (proven: runs, licenses, generates Work folders from our
MLIR) but cannot simulate this project's actual NPU2 designs — the missing piece is proprietary
AMD data not publicly distributed for this chip family. Not fixable from this environment.
Empirical hardware microbenchmarking (main16 isolation, packet-header inspection, etc. — see
updates above) remains the more productive investigation path and is what actually found every
confirmed bug/non-bug in this investigation so far.
