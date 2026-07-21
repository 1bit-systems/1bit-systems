# MLIR-AIR Multi-Launch ELF Integration

Status: **Design proposal** · Target: NPU engine v13

## Problem

The current `npu_xrt` engine (`engine/npu/src/npu_engine_universal.cpp`) dispatches one kernel at a time — one XRT `xclbin` load + `xrt.run()` per GEMM or GEMV call. Each dispatch incurs overhead through the application, runtime, driver, firmware, and hardware layers (measured at 50–200μs per call on XDNA 2).

The Cornell/AMD paper (arXiv:2606.07586) demonstrated that merging consecutive kernels into a **single multi-launch ELF** reduces per-layer dispatches from 15 → 3, achieving a 2.2× prefill and 4.0× decode speedup over the hand-optimized IRON baseline.

## Key Technique: Multi-Launch ELF

A multi-launch ELF packs multiple `air.launch` ops into one MLIR module, which the compiler lowers to a single `xrt.run()` call. Intermediates flow through DDR without CPU round-trip.

### Pre-attention block (6 kernels → 1 dispatch)

```
RMSNorm → Q GEMM → K GEMM → V GEMM → RoPE Q → RoPE K
                     ↓
              All in one .elf, one xrt.run()
```

### Post-attention block (8 kernels → 1 dispatch)

```
O GEMM → Residual Add → FFN RMSNorm → Gate GEMM → Up GEMM → SwiGLU → Down GEMM → FFN Add
                     ↓
              All in one .elf, one xrt.run()
```

## Integration Plan

### Phase 1: Stitch existing kernels (days)

The `mlir-air` repo at `programming_examples/llms/llama32_1b/` already has working multi-launch ELF builders for all 8 LLMs. The stitching logic lives in `llama_kernel_builder/stitching.py`:

1. Extract func bodies from individual `.mlir` modules
2. Rename SSA values with a unique prefix per kernel
3. Remap func args to shared buffer slots
4. Assemble combined module → compile to single `.elf`

**Deliverable:** A `build_xclbins.sh` target that generates fused pre-attention + post-attention ELFs for any Qwen3-architecture model.

### Phase 2: Runtime integration (weeks)

The `npu_engine_universal.cpp` dispatch loop needs to:

1. Recognize consecutive GEMM calls that can be fused (same layer, adjacent in the compute graph)
2. Look up the fused ELF from a cache (keyed by kernel combo + problem shape)
3. Dispatch one `xrt.run()` instead of N individual calls
4. Handle the merged BO write/read pattern (static_input_indices, intermediate_indices)

**Deliverable:** NPU engine v13 with multi-launch dispatch reducing per-layer calls from ~15 → ~3.

### Phase 3: Agent-driven auto-fusion (months)

The Cornell/AMD paper's 8-phase agent skill system can automate the discovery of new fusion patterns. The open-source skill definitions from their work (`plan.md`, `progress.md`, `perf_opt_traj.md`) can be adapted to discover fusion opportunities not yet encoded in the manual stitching logic — for example, fusing the LM Head GEMV with the last layer's output, or cross-layer KV cache management.

## Expected Impact

| Metric | Current (v12) | Target (v13) | Source |
|--------|:------------:|:------------:|--------|
| Per-layer dispatches | 15 | 3 | Cornell/AMD paper §III-B |
| Prefill TTFT (Qwen3-0.6B) | ~2.3s | ~1.0s | Extrapolated from 2.2× speedup |
| Decode (Qwen3-1.7B) | ~6.8 tok/s | ~15 tok/s | Extrapolated from 4.0× speedup |
| NPU engine benchmark | 97 tok/s | ~200+ tok/s | Projected |

## References

- Cornell/AMD paper §III-B: Optimization trajectory (dispatch merging)
- `programming_examples/llms/` — Working multi-launch ELF examples for 8 LLMs
- `llama_kernel_builder/stitching.py` — MLIR stitching implementation
- `docs/HANDOFF-NPU-OPTIMIZATION.md` — Previous optimization handoff notes
