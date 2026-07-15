# 2026-04-26 NPU bench results

## TL;DR

HIP path on gfx1151 (Radeon 8060S) is **the production lane**. NPU
kernel is correct + bit-exact but the runtime integration into
`bitnet_decode` isn't finished — naive per-token dispatch through
the AIE2P lane is dominated by xrt::bo allocation overhead.

## HIP decode (halo-1bit-2b, MIT, 1.58 bpw)

| Tokens | Prefill (tok/s) | Decode (tok/s) |
| ------:| ---------------:| --------------:|
|     64 |          91.2   |          86.1  |
|    256 |          91.1   |          75.6  |
|   1024 |          91.4   |          71.4  |

Decode rate scales with context as the KV-cache reads grow. Prefill
is steady because it's a single batched GEMV per layer.

## NPU AIE2P leaf benches (cpp/aie/ doctests, real hardware)

| Tile           | Status     | rel_err    | Wall-clock (single call)            |
| -------------- | ---------- | ---------- | ----------------------------------- |
| 64×64×64       | ✅ PASS    | 4.0e-3     | ~50 ms (bf16 floor)                 |
| 512×512×512    | ✅ PASS    | 0.0        | bit-exact                           |
| 2560×2560 (tiled = 25 × 512×512) | ✅ PASS | 2.7e-3 | ~13 s                          |

The 13 s/2560×2560 number is per `tiled_gemv` call. Each per-tile
dispatch eats ~520 ms — that's BO-allocation + DMA-setup + AIE-tile
wakeup, NOT compute. Persistent BOs + fused dispatch loop should
cut that by 10-100×.

## Integration gap

`bitnet_decode` is built from the standalone `rocm-cpp/build/` tree.
That tree's CMakeLists looks for the `onebit::aie_bitnet_gemm` target,
finds it absent (the AIE lib lives under `cpp/aie/` in the C++23 tower
which builds via a separate CMake), and prints:

> XDNA dispatch: skipped — onebit::aie_bitnet_gemm target not found.
>   Build via the cpp/ tower (release-strix preset) for the engine
>   NPU short-circuit.

The cpp/ tower's `ONEBIT_BUILD_ROCM_CPP=ON` lane requires `TheRock`
(an in-flight ROCm-from-source toolchain) which isn't built on this
box. So the engine's `HALO_NPU=1` short-circuit can't be exercised
end-to-end from `bitnet_decode` until either:
1. TheRock builds, OR
2. The standalone rocm-cpp build gets a `find_library` fallback to
   pick up `cpp/aie/build/strix/aie/libonebit_aie_bitnet_gemm.a`.

## What needs to happen for NPU to beat HIP

1. **Persistent device-resident weights.** Allocate W BOs once at
   model load, copy weights once, never re-upload. This alone should
   move the NPU lane from "13 s per layer GEMV" to "~10 ms per layer
   GEMV" once the per-tile DMA is the only cost.

2. **Fused per-token dispatch loop.** Instead of the host running
   `tiled_gemv` 25 times per matrix, push the n_block × k_block loop
   into a single AIE graph that issues all 25 mmul calls in one
   `kern.run().wait()`. Saves 24 × per-dispatch overhead.

3. **Standalone rocm-cpp link path.** `find_library` for
   `libonebit_aie_bitnet_gemm.a` so `bitnet_decode` actually compiles
   the NPU short-circuit in. Or stop building rocm-cpp standalone and
   require the cpp/ tower path.

4. **Soak run.** ≥ 2 h sustained NPU dispatch to confirm no
   OPTC hangs or `ERT_CMD_STATE_ERROR` regression.

Until 1-3 land, **HIP stays the production lane.**

## Reproduce

```bash
# HIP (default)
cd /home/bcloud/repos/1bit-systems/rocm-cpp/build
./bitnet_decode /home/bcloud/halo-ai/models/halo-1bit-2b.h1b \
    --max-tokens 1024 --prompt "The future of compute is"

# NPU leaf doctests
cd /home/bcloud/repos/1bit-systems/cpp
ONEBIT_REAL_BACKEND=1 ctest --preset release-strix -R onebit_aie
```
