# XCLBIN Weight Layout Bug

## Root Cause

The standalone INT8 xclbin kernels (QKV, O, GU, D) were compiled by the torch2aie
MLIR toolchain. These kernels expect weights in the **Q4NX tile-blocked format**
(specifically: 5120-byte chunks per 32×256 tile with per-group scales and zero-points).

The C++ engine (`npu_engine_universal.cpp`) currently:
1. Dequantizes Q4NX → float `[out_features, in_features]`
2. Transposes to `[in_features, out_features]` row-major
3. Re-quantizes to INT8 with per-matrix scale
4. Packs as flat `[K, N]` row-major in NPU BOs

**Step 4 is wrong.** The xclbin kernels do NOT expect flat row-major `[K, N]`.
They expect the NPU's native tiled format (Q4NX chunk layout or a similar
column-blocked format). This is why:
- The output has valid magnitudes (non-NaN, non-zero)
- But the values are completely scrambled (weights read from wrong positions)
- The fused xclbin works (it reads Q4NX chunks directly, preserving the correct layout)

## Evidence

- **Fused xclbin validation**: single-layer dispatch matches CPU oracle (max_abs=0.0078)
  — this uses Q4NX chunk format directly.
- **Standalone xclbins**: all 8 rounds of host-side math fixes applied, output still garbage
  — the weight layout is the remaining variable.

## Resolution Paths

1. **Use Q4NX chunks directly** (like FLM and the fused runner): Requires the torch2aie
   Python toolchain to generate the correct instruction schedule for packing.
2. **Generate new xclbins** with documented input format: Requires MLIR-AIE compiler
   toolchain, ideally with explicit memory layout control.
3. **Decompile existing xclbins** to determine expected layout: Requires AIE Simulator
   (blocked on this machine — missing `aie2p_8x4_device.json`).

## Current Production Path

The C++23 daemon (`daemon/npu-gpu-cpud.cpp`) proxies to FLM, which handles the
weight format internally. This is the only path with verified coherent output at
94 tok/s. The C++ engine work is valuable for understanding but the weight layout
bug prevents it from replacing FLM with the current xclbin set.
