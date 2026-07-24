# 4→8 NPU-Column Migration Gap-Map (T13 unblock)

**Scope:** what must change in `~/torch2aie/examples/qwen3-decode-layer/` for
`run_full_layer.py --build-only` to emit a valid real-weight full-layer Q4NX kernel
for Qwen3-0.6B on 8 NPU columns. This is a **checklist**, not a fix — do NOT edit
validators to force a build (see Safety below).

Investigated 2026-07-17 against the failing `--build-only` run (7 structural errors).

## A. Migration state inventory (what's 8-col, what's still 4-col)

| File | State | Evidence |
|---|---|---|
| `compact_dataflow.py` | **8-col (migrated)** | `HUB_Q_OUT_BDS=(25,2,26,3,1,5,7,8)` (8) |
| `compact_dataflow_8col.py` | **8-col (duplicate of compact_dataflow)** | identical constants — likely the migrate-then-rename artifact |
| `attention_dataflow.py` | **still 4-col** | `HUB_Q_OUT_BDS=(2,24,4,26)`, `HUB_RETURN_IN_BDS=(25,6,27,8)` (4) |
| `full_layer_engine_generate.py` constants | **8-col** (imports from compact_dataflow) | line 78 |
| `full_layer_engine_generate.py` validators | **mixed / internally inconsistent** | `HUB_DOWN_OUT_BDS` check is 8-el (matches); `HUB_Q_*`/`HUB_RETURN_*` checks still 4-el (lines ~1562–1567) |
| generation (MLIR emit) | **incomplete for 8-col** | MLIR-inspecting validators fail (sections C, D) |

## B. The 7 failing contracts (from `--build-only`)

```
1. source-side replay first block must acquire aie.use_lock(%hub_return_full, AcquireGreaterEqual, 4)
2. source-side replay last block must release aie.use_lock(%hub_return_empty, Release, 4)
3. hub_dma MM2S1 block ^q4_out uses memtile BD 1 from the wrong bank
4. hub_dma MM2S3 block ^q6_out uses memtile BD 7 from the wrong bank
5. hub_dma S2MM3 block ^return5_in uses memtile BD 10 from the wrong bank
6. hub_dma S2MM5 block ^return7_in uses memtile BD 12 from the wrong bank
7. hub BD contract mismatch
```

## C. Generation gaps (fix FIRST — real NPU-design work)

### C1. Memtile BD bank assignment (errors 3–6)
Validator: `mlir_utils.py::require_memtile_dma_bd_bank` (line 512). Rule:
**odd DMA channels must use BD id > 23; even channels must use BD id ≤ 23**
(`high_bank = bd_id > 23; if high_bank != odd_channel: error`).

Violations in the generated 8-col MLIR:
- MM2S1 `^q4_out`  → channel 1 (odd) uses BD 1  (should be >23)
- MM2S3 `^q6_out`  → channel 3 (odd) uses BD 7  (should be >23)
- S2MM3 `^return5_in` → channel 3 (odd) uses BD 10 (should be >23)
- S2MM5 `^return7_in` → channel 5 (odd) uses BD 12 (should be >23)

**Fix location:** the hub-DMA BD-id assignment in `compact_dataflow.py` around the
`for window, (channel, bd_id) in zip(HUB_Q_OUT_CHANNELS, HUB_Q_OUT_BDS)` loops
(lines ~537, ~548, ~575) and the return/down emitters. The 8-col expansion added
odd channels 5/7 whose BD ids must be remapped into the >23 bank. Current
`HUB_Q_OUT_BDS=(25,2,26,3,1,5,7,8)` puts channels 5/7 (the new odd ones, indices 4–7)
on BDs 1/5/7/8 — all ≤23 — violating the bank rule. Remap so odd channels get BDs >23.

### C2. Source-side replay locks (errors 1–2)
Validator: `mlir_utils.py::require_source_side_packet_replay` (line 219), called at
`full_layer_engine_generate.py:1453` for `attention_out`:
- expects `O_BODY_RECORDS` replay blocks
- first block must contain `aie.use_lock(%hub_return_full, AcquireGreaterEqual, 4)`
- last block must contain  `aie.use_lock(%hub_return_empty, Release, 4)`

The generated `attention_out` replay region is missing those lock ops (with value 4).
**Fix location:** the attention_out source-side replay emitter (search `attention_out`
replay/loop in `compact_dataflow.py` / `attention_dataflow.py`). Must emit the
hub_return_full acquire in the first replay block and hub_return_empty release in the
last. **Open question for the migrator:** is the lock value `4` still correct for 8-col,
or should it be `8`? (If 8, the validator at line ~1459/1461 is also stale — see D2.)

## D. Validator-sync gaps (fix LAST — only after C is correct)

### D1. Hub BD constant checks (error 7)
`full_layer_engine_generate.py` lines ~1562–1567 still assert 4-col tuples:
```
HUB_Q_OUT_CHANNELS != (1,2,3,4)            # actual (1,2,3,4,1,2,3,4)
HUB_Q_OUT_BDS     != (25,2,26,3)           # actual (25,2,26,3,1,5,7,8)
HUB_RETURN_IN_CHANNELS != (2,3,4,5)        # actual (2,3,4,5,2,3,4,5)
HUB_RETURN_IN_BDS != (4,28,6,30)           # actual (4,28,6,30,9,10,11,12)
```
Update the expected tuples to the 8-col values **after** C1 remaps the BD ids (the
expected BD tuples must match whatever C1 lands on — so do C1 first, then set these).

### D2. Possibly-stale lock value (errors 1–2 expectation)
If the migrator intends the source-side replay lock count to scale with columns,
the `4` in the validator call (lines ~1459/1461) and in C2's expected lock ops may
need to become `8`. Resolve with the migrator's intent; do not guess.

## E. Work order

1. **C1** — remap memtile BD ids so odd channels use >23 (unblocks errors 3–6).
2. **C2** — emit the hub_return lock acquire/release in attention_out replay
   first/last blocks; confirm lock value (4 vs 8) with migrator (unblocks 1–2).
3. **D1** — update the hub BD constant checks to the final 8-col tuples (unblocks 7).
4. Re-run `run_full_layer.py --build-only` until clean, then `run_full_layer.py`
   (full NPU run with real weights) and confirm fault-free + coherent output.
5. Then T14 (C++ engine driving this kernel) becomes possible.

### C3. Column-tile BD uniqueness (pre-existing bug, blocks aiecc)

Source: `MLIR-AIE isBdChannelAccessible` for NPU2 memtiles:
```
even channel: bd_id < 24
odd  channel: bd_id >= 24
```
Each memtile has 48 BDs (0-47), banked as 0-23 (even channels) / 24-47 (odd).

The `AIEAssignBufferDescriptorIDs` pass uses ONE `BdIdGenerator` per tile
(each MemTileDMAOp creates a generator for its col/row). It walks ALL DMABDOps
on that tile and calls `assignBdId(bd_id)`. If two DMA blocks on the same tile
(both channels share a tile) specify the same bd_id → assertion failure
`"bdId has already been assigned"`.

The full-layer generator assigns BDs per-channel independently (e.g., column
tiles in the 8-col design assign BDs 0-5 to BOTH the even and odd DMA channels
on the same tile) → duplication → blocks aiecc.

**Fix:** the BD allocation in the generator must ensure UNIQUE bd_ids per tile
across ALL DMA channels on that tile. Track the used-BD set per (col,row) in
the Python generator.

Note: my C1 hub BD remap (BDs 24/34/35/36) is CORRECT per this spec — hub is a
memtile, odd channels need BDs ≥ 24, and the chosen ids don't collide on hub.

## Safety — read before "just make it build"

The validators in sections C/D exist **precisely** to prevent the DMA/IOMMU
`IO_PAGE_FAULT` class of bug that T12 proved is the root cause of every home-grown
engine faulting. **Editing validators to accept invalid MLIR will produce a kernel
that builds and then faults or silently miscomputes on the NPU** — re-creating the
exact T12 failure mode with the safety net removed. Only sync validators (section D)
*after* the generation (section C) genuinely satisfies the contract. If a contract
can't be satisfied without changing the validator's *rule*, that's a design decision
for the migrator, not a quick edit.
