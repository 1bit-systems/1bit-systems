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
