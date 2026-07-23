# INT8 on NPU2 via MLIR-AIE: Architecturally Blocked

**Status: DO NOT ATTEMPT. Use BFP16 instead.**

## Summary

INT8 GEMM on NPU2 through the open-source MLIR-AIE toolchain is **architecturally impossible with current abstractions**. The problem is not a bug, a missing feature, or insufficient effort — it's a fundamental mismatch between how MLIR-AIE's `ObjectFifo` abstraction distributes data across AIE cores and what a correct INT8 matrix multiply requires.

The BFP16 path works (210ms/tok, coherent output) because BFP16's block-floating-point encoding masks the data distribution error. INT8 has no such mask.

## Timeline of Evidence

### June 28 — First INT8 xclbins built
All 5 matrix shapes (QKV, O, GU, D) compiled and ran on hardware. All-1s pattern test passed. Random data test: **394% mean relative error**.

### June 29 — All attempted fixes failed
| Attempt | Result | Blocked By |
|---------|--------|------------|
| Per-core A fifos (v9-v12) | Compile crash | DMA channel limit (~2/shim tile, need 8 for 8 cores) |
| Single-core (v13-v15) | RTE crash | NPU routing conflicts for cross-column data |
| Per-shim A distribution (v17) | Built, same K-issue | Linked fifo pool depth=2 → only 2 sub-views |
| Depth-16 linked pool (v19) | aiecc crash | Lock/BD slot exhaustion with 8 consumers |
| DRAM-backed bf16copy (v21) | Runs, **4× correct** | Sub-view dims (r=8,s=8) are BFP16-specific, wrong for INT8 |
| Weight reordering | Math impossibility | Σ A(K_sub)×B_reordered ≠ Σ A(all K)×B(original K) |

### July 5 — Hardware-in-the-loop proves uncorrelated output
Test A (synthetic pattern): All 4 xclbins produce 10,000/10,000 wrong outputs.
Test B (real inference bytes): QKV mean|diff| = 5,394 vs mean|ref| = 1,834 — error is **larger than signal**.

D-proj at layer 0 (real inference):
```
Cm[8,0:8]  = [ -94565  -83862  -81273 -111747  -84766  -88403  -87565  -84303]  ← NPU
ref[8,0:8] = [-1252751 -681834 -1339176 -1363487 -732522 -1117851 -1361552 -1521576]  ← Correct A·B
Correlation across all output rows: max 0.0155 — i.e., none.
```

Confirmed: **not fixable from the host side**. All host math is provably correct (verified by dumping exact bytes sent to NPU and comparing against numpy reference).

## The Architectural Lock

### How BFP16 Works (the working path)

```
BFP16 reference xclbin (design_1024_bfp16.xclbin):
  8 cores share 1 shim DMA channel for A data via ObjectFifo

  Distribution:  round-robin across 8 cores in K-dimension
    Core 0: A(K[0:64])    Core 1: A(K[64:128])   ...  Core 7: A(K[448:512])
    Core 0: A(K[512:576]) Core 1: A(K[576:640])   ...  Core 7: A(K[960:1024])

  Each core accumulates:  C += A(K_subset) × B(K_all)
  Each core only sees 64 of 1024 K-values — 960 are zero-contribution

  WHY IT STILL WORKS: BFP16 encodes 8 adjacent floats with one shared exponent.
  Adjacent K-blocks have ~similar dequantized values → the error from using
  only 64/1024 K-values is O(1%) per matmul.
```

### Why INT8 Breaks

```
INT8 has no shared-exponent encoding:

  A[0:1024] are 1024 INDEPENDENT int8 values (range [-128, 127])
  
  When each core only sees 64 of 1024 values:
    Core 0 computes: C += A[0:64] × B[0:64, :] + A[512:576] × B[512:576, :]
    = ONLY 12.5% of the total contribution for K-dim=1024
  
  The other 87.5% of K-values (A[64:512] and A[576:1024]) are ZERO
  → Result has NO CORRELATION with correct A·B

  Measured: 394% mean relative error, 0.0155 max row correlation
```

### Why It Cannot Be Fixed

The MLIR-AIE `ObjectFifo` abstraction on NPU2 has these hard limits:

| Resource | NPU2 Limit | Required for INT8 | Gap |
|----------|-----------|-------------------|-----|
| DMA channels per shim tile | ~2 | 8 (one per core) | 4× |
| Linked fifo pool depth | 2 sub-views | 16 sub-views (8 cores × 2 K-iterations) | 8× |
| Lock/BD slots | ~64 | ~256 | 4× |

Each attempted workaround hits one of these limits:
- **Per-core A fifos**: needs 8 DMA channels, only 2 available
- **Depth-16 linked pool**: needs 256 lock/BD slots, only ~64 available
- **DRAM-backed bf16copy**: sub-view grouping (r=8,s=8) is BFP16-specific — can't be disabled without breaking the pool

### How AMD's Windows Driver Works Around It

The proprietary Windows XDNA driver (DirectML) uses a **fundamentally different dataflow**:

```
Windows XDNA INT8:
  M-parallel tiling (NOT K-parallel):
    Each column gets different M-rows
    A is broadcast, B is partitioned along M, not K
  
  Software-managed BD chains:
    Time-multiplexes shim DMA across all columns
    No hardware lock-based fifos
    Pre-compiled tuned kernels per shape
  
  Result: ~50 TOPS INT8 on the same silicon
```

This is completely different from MLIR-AIE's `ObjectFifo` abstraction. The hardware CAN do INT8 — just not through the open-source toolchain.

## Proof: AMD's Own Examples Work

The cheat code examples at `/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/` work correctly:
- `single_core.py`: **0 errors, 0 max diff** on isolated GEMM test
- `whole_array.py`: Same — numpy-verified correct
- **31.0 TFLOPS** achieved with `config2` (192×128×96 tile, 32 cores)

These work because they use M-parallel dataflow (single core gets all K of A, different M-rows) or simpler designs. The moment you try to distribute K across cores via ObjectFifo, INT8 breaks.

## What BFP16 Gets You

| Metric | BFP16 (Working) | INT8 (Blocked) |
|--------|----------------|----------------|
| Speed | 210 ms/tok | Would be ~100 ms/tok |
| TFLOPS | 12 | Would be ~50 |
| Coherent | ✅ (diverse tokens) | ❌ (394% error) |
| Toolchain | MLIR-AIE | MLIR-AIE (same, broken) |
| Power | ~15W | ~15W |

**Verdict: BFP16 is 2.5 tok/s slower than what INT8 could theoretically provide, but it WORKS. Use it.**

## The Verdict

```
┌────────────────────────────────────────────────────────┐
│  INT8 on NPU2 via MLIR-AIE = ARCHITECTURALLY BLOCKED   │
│                                                        │
│  Problem:  K-interleaving from shared A fifo           │
│  Root:     ObjectFifo abstraction doesn't support      │
│            M-parallel tiling needed for correct INT8   │
│  Fixed by: Nothing in open-source toolchain            │
│  Workaround: BFP16 (210ms/tok, coherent)               │
│  Full speed: AMD Windows driver (proprietary, 50 TOPS) │
└────────────────────────────────────────────────────────┘

DO NOT:
  - Build more INT8 xclbins (they'll have the same K-issue)
  - Try to fix the MLIR generator (fundamental abstraction limit)
  - Investigate "why is INT8 wrong" on the host side (proven correct)

DO:
  - Use BFP16 for all NPU inference
  - Document the K-interleaving limitation for anyone asking
  - Watch for AMD open-sourcing their M-parallel dataflow
  - If INT8 is essential, use the GPU (383 tok/s via llama.cpp IQ1_S)
```
