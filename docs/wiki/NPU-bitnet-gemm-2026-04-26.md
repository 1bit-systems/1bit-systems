# NPU bitnet_gemm — AIE2P ternary GEMM lane

**2026-04-26.** First end-to-end ternary GEMM running on the Strix Halo
XDNA2 NPU (AIE2P tile target, fw 1.1.2.65 via `amdxdna` driver).

## What landed

| Tile size | Status | Rel-err | Staging | xclbin size |
| ---:| --- | ---:| ---:| ---:|
| 64×64×64 | green | 4.0×10⁻³ (bf16 floor) | 16 KiB LDS-equiv | 28 890 B |
| 512×512×512 | **bit-exact** | 0.0×10⁰ | ≈0 (registers only) | 15 354 B |

`run.wait()` returned `ERT_CMD_STATE_COMPLETED` on every iteration —
issue #103 against `amdxdna` fw 1.1.2.65 did not trip on either
workload. Phase-2 is **smaller** than Phase-1 because the fused
unpack+mmul kernel is tighter than the unpack→retile→mmul split.

## Lane

`HALO_V2`-packed weights (16 ternary codes per `uint32`, 2-bit lookup
table), bf16 activations pre-scaled host-side (per-row scale folded
into `A`, since AIE2P compute tiles only have two input DMA channels),
fp32 in-tile accumulator, bf16 output for Phase-1 / fp32-out then
host-side bf16 cast for Phase-2 (K-reduction round-trip noise forced
the higher-precision accumulator).

Files (under `mlir-aie/programming_examples/basic/bitnet_gemm/single_core/`):

```
bitnet_gemm.cc          — Phase-1 (64x64) AIE2P kernel
bitnet_gemm_iron.py     — Phase-1 IRON DSL design
Makefile                — Phase-1 Peano + aiecc build
run_pyxrt_bitnet.py     — Phase-1 host harness via libxrt

bitnet_gemm_512.cc      — Phase-2 (512x512) fused kernel, K-reduction
bitnet_gemm_iron_512.py — Phase-2 IRON DSL with K_div_k=8 inner loop
Makefile.512            — Phase-2 build (separate target_suffix)
run_pyxrt_bitnet_512.py — Phase-2 host harness, weights pre-tiled
```

## Runtime wire (in flight)

`cpp/aie/` carries the C++23 `BitnetGemmAIE2P` lib (pImpl, libxrt
backend) that the engine will dispatch through once the runtime path
ships. Public surface:

```cpp
namespace onebit::aie {
class BitnetGemmAIE2P {
public:
    static std::expected<BitnetGemmAIE2P, Error>
        load(const std::filesystem::path& xclbin,
             const std::filesystem::path& insts);

    [[nodiscard]] std::expected<void, Error>
        gemm(std::span<const std::uint16_t> a_bf16,
             std::span<const std::uint32_t> w_packed,
             std::span<std::uint16_t>       c_bf16);
};
}
```

Built via `cmake --preset release-strix -DONEBIT_BUILD_XDNA=ON`.
XRT 2.21.75 (CachyOS `xrt` package). Headers + libs at
`/usr/include/xrt/` and `/usr/lib/libxrt_{coreutil,driver_xdna}.so`.

## Ship-gate movement

`project_ship_gate_npu.md` previously held public launch hostage to
"NPU is being worked on internally" (Jeremy Fowers DM,
`project_amd_jeremy_npu_signal.md`). The kernel half of that gate is
now resolved — bitnet_gemm runs end-to-end on Strix Halo NPU in our
own lane. The remaining gate is the runtime wire (above) plus a
soak run on real model weights (not the synthetic test fixtures).

## What still needs upstream / fork

The mlir-aie working tree at `/home/bcloud/repos/mlir-aie/` is being
forked to `bong-water-water-bong/mlir-aie` so the Phase-1 + Phase-2
sources + xclbins are tracked outside the local box. See
`project_npu_bitnet_gemm_authored.md` for the GH URL once that
push lands.

## What this does not yet do

- It does not replace `rcpp_ternary_gemv_halo_f16_devscale` in the
  decode loop. The HIP path stays primary.
- It does not handle `K` larger than 512 in one launch. The next
  pass has to wire a K-tile loop in the host runner.
- The `run_pyxrt_bitnet_512.py` harness pre-tiles weights at host time
  so the on-tile re-tile pass could be deleted; that pre-tile cost
  needs to land in the C++ runtime side too.

## Why this matters

Memory `project_npu_path_analysis.md` (2026-04) said the NPU lane was
deferred to Q3 2026 because "no stack runs BitNet-1.58 on XDNA2 today"
and AMD's Linux NPU EP only landed for Phoenix/Krackan, not Strix Halo.
That defer is no longer correct for the kernel lane. We're now the
public reference for ternary GEMM on AIE2P Strix Halo.
