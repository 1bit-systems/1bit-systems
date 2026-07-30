# Bug: Pure GPU Backend 4x Slower Than Fused Backend

**Filed:** 2026-07-30
**Status:** Open — root cause not identified
**Priority:** Medium
**Tags:** performance, gpu, investigation-needed

## Summary

The pure GPU backend (`src/backend_hip_1bp.cpp`) runs **45 tok/s** while the functionally
identical fused backend (`src/backend_fused.cpp`) runs **185 tok/s** on the same hardware.
Both execute the same operations, use the same custom GEMV kernel, and the same attention
kernel. The 4x performance gap has been exhaustively investigated and the root cause
remains unidentified.

## Hardware

- **GPU:** AMD Radeon 8060S (Strix Halo, gfx1151)
- **ROCm:** 7.15.26302 (TheRock SDK, ROCm 10.0)
- **CPU:** AMD RYZEN AI MAX+ 395
- **RAM:** 122 GB unified

## Benchmarks (Qwen3-0.6B, 100 tokens, 5 warmup)

| Backend | File | tok/s | ms/tok |
|---------|------|-------|--------|
| **Fused GPU+NPU** | `src/backend_fused.cpp` | **185** | **5.4** |
| **Pure GPU** | `src/backend_hip_1bp.cpp` | **45** | **22** |

Both produce correct token IDs (varied — not constant/BOS).

## What Both Backends Do Per Layer

```
RMSNorm(dh) → Q GEMV → K GEMV → V GEMV → RoPE Q → RoPE K
→ F2H(dQ) → KV Store → Flash Attention → H2F
→ Wo GEMV → Add (residual) → Post-RMSNorm
→ W1 GEMV → W2 GEMV → SiLU → W3 GEMV → Add (residual)
```

Both use the same GEMV kernel (tree reduction, `__shared__ double sdata[256]`),
same attention kernel (`kv_cache_attn_decode_kernel` from `src/kv_cache_attn.hip`).

## Investigation — All Ruled Out

### 1. `hipStreamSynchronize` calls
The pure GPU backend had 3 syncs per layer originally (QKV done, KV stored, attention
done). **Removing ALL explicit syncs** (making the entire forward loop async, relying
only on stream ordering): 48→49 tok/s (noise). **Re-adding syncs at the same positions
as the fused backend:** no change.

### 2. `-ffast-math` compiler flag
Changed `bench_hip_1bp` from `-O3 -ffast-math -munsafe-fp-atomics` to `-O2`
(matching `bench_fused`). No change (47 tok/s).

### 3. GPU clock throttling / power management
Forced max GPU clock via:
```bash
echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
```
No change (49 tok/s vs 47 with auto).

### 4. Buffer aliasing / RAW hazards
The pure GPU backend reuses `datt` buffer for Q output, attention f32 output, AND
SiLU output. Changed to use separate `datt2` buffer for attention output, eliminating
any read-after-write or write-after-write hazards. No change (49 tok/s).

### 5. Shared library vs static symbol resolution
The pure GPU backend resolves `rcpp_kv_cache_attn_decode` from `librocm_cpp.so`.
Added `src/kv_cache_attn.hip` to `bench_hip_1bp` sources so the symbol is resolved
from the binary itself (same as fused backend). No change (47 tok/s).

Before: `U rcpp_kv_cache_attn_decode` (from shared lib)
After:  `T rcpp_kv_cache_attn_decode` (from static object)
Performance: 49→47 tok/s (noise).

### 6. Kernel implementation differences
Verified every kernel pair:
- `gemv_kernel` vs `fused_gemv_plain_kernel` — identical (tree reduction, double accum)
- `rmsnorm_kernel` vs `fused_rmsnorm_kernel` — identical logic (warp-shuffle vs tree red.)
- `rope_kernel` vs `fused_rope_kernel` — identical
- `silu_kernel` vs `fused_silu_kernel` — identical (same lines of code)

### 7. GPU architecture target
Both binaries compile for `gfx1151`:
```bash
$ strings build/bench_* | grep gfx1151
amdgcn-amd-amdhsa--gfx1151
```

### 8. Shared library link order
Both link `librocm_cpp.so` from the same build directory. Both show the same ldd output.

## AMD_LOG_LEVEL=3 Trace Observations

**Pure GPU (1 token):** 4646 ShaderName entries, including:
- 1773 gemv_kernel dispatches
- 252 kv_cache_attn_decode_kernel dispatches
- 75 copyBuffer dispatches

**Fused (1 token):** 83 ShaderName entries, including:
- 8 fused_gemv_plain_kernel dispatches
- 1 kv_cache_attn_decode_kernel dispatch
- 59 copyBuffer dispatches

The 56x difference in ShaderName count does NOT correspond to 56x more kernel launches.
Both backends launch exactly 28 attention kernels per token (one per layer).
The trace may be logging events at a different granularity (per-wave vs per-dispatch)
for each backend's compiled code objects.

## Suspected Root Cause

The leading theory is a **subtle interaction between the ROCm 7.15 HIP runtime and
kernel dispatch from different compilation units**.

The fused backend's kernels are defined inline in `backend_fused.cpp` (compiled as a
single HIP translation unit with LANGUAGE HIP). The pure GPU backend's kernels are in
`hip_1bp_kernels.hip` (a separate HIP file). Despite identical source logic, they are
compiled into different code objects. When dispatched, the HIP runtime may take
different paths for each code object — possibly decomposing large 1D grids
(`gemv_kernel<<<M, 256, ...>>>` where M=1024-3072) differently.

A secondary theory: the `rocm_cpp` shared library's code objects (compiled with
`-O3 -ffast-math -munsafe-fp-atomics`) may trigger different GPU scheduler behavior
than the `bench_hip_1bp` binary's own code objects (compiled with `-O2`).

## Reproduction

```bash
cd /home/bcloud/1bit-systems
cmake --build build --target bench_fused bench_hip_1bp

# Fast (185 tok/s)
sudo LD_LIBRARY_PATH=... ./build/bench_fused models/Qwen3-0.6B.1bp 100 5

# Slow (45 tok/s)
sudo LD_LIBRARY_PATH=... ./build/bench_hip_1bp models/Qwen3-0.6B.1bp 100 5
```

## Files

- `src/backend_hip_1bp.cpp` — Pure GPU backend (slow)
- `src/backend_fused.cpp` — Fused GPU+NPU backend (fast, reference implementation)
- `src/hip_1bp_kernels.hip` — GPU kernels used by the pure GPU backend
- `CMakeLists.txt` — Build configuration (both targets at lines ~666 and ~1200)

## Resolution Notes

The fused backend (185 tok/s) is the production path. The pure GPU backend is not
needed for deployment. This bug is filed for documentation and tracking purposes.
If the root cause is found, the fix should be applied to `backend_hip_1bp.cpp` and
potentially `hip_1bp_kernels.hip`.
