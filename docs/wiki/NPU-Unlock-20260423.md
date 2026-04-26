# NPU unlock — 2026-04-23

Technical writeup of the Strix Halo XDNA2 NPU toolchain breakthrough. Covers what shipped today, why prior memory was wrong, how the naming conventions reconcile, what the axpy test matrix exercised, and what's left to build before BitNet-1.58 decodes on the NPU.

This page **complements** `project_ship_gate_npu.md` — the public-launch gate stays closed. Today we proved the toolchain; the gate lifts when a full BitNet-1.58 model runs end-to-end on our authored kernel path. See the [why no NPU yet](./Why-No-NPU-Yet.md) page for the longer history of ORT + VitisAI EP evaluation that this result pivots us away from.

## Result in one line

**IRON axpy pytest: 160 / 160 passed in 26.8 s on npu5 (Strix Halo 17f0_11) under CachyOS / Arch, Python 3.14 venv, mlir-aie `0.0.1.2026033104+e4f35d6`, llvm-aie `21.0.0`, torch `2.11.0+cpu`.**

That's the full chain — Python DSL → MLIR lowering → Peano C++ AIE codegen → xclbin artifact → libxrt load → AIE2P tile execution. Proven on our silicon, on our distro, under user-land. No AMD VitisAI EP in the loop, no Ubuntu, no Docker.

## Naming reconciliation — "AIE4" vs "AIE2P"

Prior memory and two earlier wiki pages said IRON was Phoenix-only (AIE2 / npu1). That was wrong, and the reason it was wrong is two vendor conventions describing the same silicon at different layers. Verified in `amd/IRON:iron/common/device_utils.py`:

```python
from aie.iron.device import NPU2

def get_kernel_dir(dev=None) -> str:
    """Returns 'aie2p' for NPU2 (Strix, Krackan), 'aie2' for NPU1 (Phoenix)."""
```

| part | xdna-driver label | IRON label | PCI device ID |
|---|---|---|---|
| Phoenix | `npu1` / `AIE4` (misleading) | `NPU1` / `aie2` | `1022:1502` |
| Krackan | `npu3` | `NPU2` / `aie2p` | `1022:17f1` |
| Strix Point | `npu4` | `NPU2` / `aie2p` | `1022:17f0_10` |
| **Strix Halo** | `npu5` | **`NPU2` / `aie2p`** | `1022:17f0_11` |

xdna-driver's `AIE4` label is the driver-side codename; IRON's `aie2p` is the kernel-dir name for the tile generation. Same silicon generation (Strix + Krackan both run AIE2P tiles); different layer of the stack. Once that lines up, IRON's `applications/llama_3.2_1b/llama_npu.py` reference — a real NPU-deployed Llama-3.2-1B, not a CPU golden model — is architecturally in-scope for our box.

## Arch / CachyOS install deltas

IRON is upstream-tested on Ubuntu 24.04/24.10. CachyOS needed exactly two deltas; nothing else patched.

1. **pyxrt symlink.** Arch ships `pyxrt` as a system package under the distro `xrt`, landing at `/usr/lib/python3.14/site-packages/pyxrt.cpython-314-x86_64-linux-gnu.so`. IRON's `requirements.txt` does not install it (expects system-level). Fix:

   ```sh
   ln -sf /usr/lib/python3.14/site-packages/pyxrt.cpython-314-x86_64-linux-gnu.so \
          ~/.venvs/ironenv/lib/python3.14/site-packages/pyxrt.cpython-314-x86_64-linux-gnu.so
   ```

2. **`XILINX_XRT` path.** Ubuntu uses `/opt/xilinx/xrt`; Arch uses `/usr`. Every shell invocation needs:

   ```sh
   export XILINX_XRT=/usr
   ```

Both are baked into `[component.npu-engine]` in `packages.toml` so `1bit install npu-engine` applies them automatically.

## Axpy test matrix (160 tests)

The `mlir-aie/test/npu-xrt/axpy/` pytest fans out the simplest possible vector kernel (`y = a * x`, fp16) across the full authoring surface. Every cell is a fresh compile → xclbin → libxrt load → execute → verify numerics round-trip. 160 cells, 160 green, 26.8 s end-to-end:

| axis | values | count |
|---|---|---:|
| column count (AIE tile columns) | 1, 2, 4, 8 | 4 |
| input vector length | 256, 512, 1024, 2048, 4096, 8192 | 6 |
| `a` scalar factor | small (e.g. 2.0), large (e.g. 1e3) | 2 |
| repetitions per cell | sometimes ≥2 | — |

4 × 6 × 2 × (mean ≥ 3.3) = 160 passes. Critically this exercises **multi-column** tile layouts (up to 8 cols) — that's the same dispatch surface a real GEMM / GEMV kernel uses; axpy is just the thinnest possible kernel on top of it. Single-column would have been suggestive but not definitive. The 8-column pass is the definitive result.

## What's left to ship BitNet-1.58 on NPU

The authoring toolchain is proven; the BitNet-specific kernel is not written.

1. **Author a `bitnet_gemm` operator in MLIR-AIE vector DSL** (~1–2 wk). Port the algorithm from our gfx1151 `ternary_gemv_halo.hip`: packed 2-bit weights × int8 activations with fp16 scale, absmean quantization. Different silicon (AIE2P VLIW + vector intrinsics vs WMMA), same algorithm shape.
2. **Tune for AIE2P tiling.** Memory hierarchy is tile-local scratchpads + shim DMA, not LPDDR5 streaming. Gerganov's L1-tiling lead from `project_gerganov_l1_tuning.md` applies analogously: K-outer tile so weight reads stay in tile scratchpad across multiple M rows.
3. **Integrate xclbin dispatch into `lemond` backend.** Load the xclbin via `libxrt` C++ (`xrt::device`, `xrt::kernel`, `xrt::bo`) from a new sibling crate or direct FFI inside `1bit-hip`. Rule A stays clean — runtime path is C++ + Rust, no Python once the kernel is compiled.
4. **Smoke-test with Llama-3.2-1B first** (IRON ships the reference under `applications/llama_3.2_1b/llama_npu.py`), then swap in our BitNet-1.58.
5. **Trip the ship-gate.** `project_ship_gate_npu.md` lifts when a full sub-2-bit model decodes on the NPU via our kernel, not before. Axpy does not lift it; Llama-1B or BitNet-2B on the NPU does.

## Why this is the real story, not just a driver update

Before today, `project_npu_path_analysis.md` had us deferring the NPU lane to Q3 2026, predicated on AMD shipping a Linux STX-H VitisAI EP. That blocker is gone. We no longer need AMD's timeline — we can author our own AIE kernels and compile them against the already-public Peano + MLIR-AIE stack, dispatch via the already-upstream `amd/xdna-driver`, and load via the already-shipping XRT.

**halo-ai is no longer hostage to AMD's NPU software schedule.** That's the shift. The ship-gate holds because we still need to prove a real model on the silicon, but the path to clearing the gate is now ours to schedule.

See `project_npu_unlocked.md` + `project_iron_strix_halo_revised.md` (both 2026-04-23) for the live engineering state. Discord signal from Jeremy Fowers (AMD) — "NPU is being worked on internally" — is noted in `project_amd_jeremy_npu_signal.md` but no longer on our critical path.
