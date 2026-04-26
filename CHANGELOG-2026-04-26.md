# 2026-04-26 — NPU lane lights up

> "I know kung fu." Tripped this morning. The XDNA2 NPU on Strix Halo
> ran our own ternary GEMM end-to-end for the first time.

## NPU `bitnet_gemm` — Phase-1 + Phase-2

First end-to-end ternary GEMM on the AIE2P tile target, fw 1.1.2.65 via
`amdxdna`. `run.wait()` returned `ERT_CMD_STATE_COMPLETED` on every
iteration; issue #103 against the firmware did not trip.

| Tile | Status | Rel-err vs bf16 oracle | xclbin |
|---:|---|---:|---:|
| 64×64×64 | green | 4.0×10⁻³ (bf16 floor) | 28 890 B |
| 512×512×512 | bit-exact | 0.0 | 15 354 B |

Phase-2 is **smaller** than Phase-1 — the fused unpack+mmul kernel is
tighter than Phase-1's unpack→retile→mmul split. Both lanes use
`HALO_V2`-packed weights (16 ternary codes per `uint32`, 2-bit lookup),
bf16 activations with per-row scale folded host-side, fp32 in-tile
accumulator. Sources at
`mlir-aie/programming_examples/basic/bitnet_gemm/single_core/`.
Wiki write-up: `docs/wiki/NPU-bitnet-gemm-2026-04-26.md`.

## libxrt direct-call lane (`7a379ee`)

`cpp/aie/` ships the C++23 `BitnetGemmAIE2P` lib (pImpl,
`std::expected<T, Error>`, `std::span` interface) wired through libxrt
2.21.75 on `/dev/accel/accel0`. Sibling to the existing dlopen-backed
`XrtBackend` — that path stays for the lemonade-routed BitNet GEMV; this
one is the direct-link lane that mirrors `run_pyxrt_bitnet.py` for the
M=K=N=64 xclbin. Build-gated on `ONEBIT_BUILD_XDNA`; off-host CI stays
clean. 64×64×64 dispatch matches the bf16 CPU oracle at
`rel_err = 3.98e-3` (threshold 5e-3).

## mlir-aie fork (`c217623`)

`bong-water-water-bong/mlir-aie` carries Phase-1 + Phase-2 sources +
xclbins so the working tree is reproducible outside the local box.

## Sherry L1-rescale fix (`2e25db9`)

`rocm-cpp/tools/h1b_repack_sherry.cpp` now recomputes per-row scales
after the 3:4 sparsification step
(`S_sherry = S_v2 * |q_v2|_1 / |q_sherry|_1`) instead of pass-through.
PPL pass is the next gate; build clean here.

## lemonade `1bit-systems/` namespace

Multi-recipe registry tidy-up: 1bit recipes (`halo-1bit-2b`, etc.) now
live under a `1bit-systems/` namespace inside `server_models.json` so
the upstream lemonade-sdk fork doesn't trip over our additions.

## Install + fleet plumbing

- `740237a` — install script unsets bad `CC` / `CXX` / `HIPCXX` env
  and wipes stale `CMakeCache.txt` before retry. Removes a recurring
  fresh-clone footgun.
- `1b06698` — bun Discord shim sidecar lands the help-desk bot fleet
  online without dragging discord.js into the C++ tower.

## Not done yet

- Tile widening for production model dims (M=K=N=4096+); Phase-2 caps
  at 512 per launch.
- Runtime engine wire — `BitnetGemmAIE2P` builds, but the decode loop
  still goes through `rcpp_ternary_gemv_halo_f16_devscale` on HIP.
- Soak run on real model weights (not synthetic test fixtures).
- Sherry PPL verification on the L1-rescaled repack.

Ship-gate (`project_ship_gate_npu.md`) moves: kernel half resolved,
runtime + soak still open.
