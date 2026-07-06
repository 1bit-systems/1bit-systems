# NPU Full Unlock Plan

**Goal:** Extract every last drop of performance from the Strix Halo XDNA2 NPU.

## Current State (July 6, 2026)

| Metric | Value |
|--------|-------|
| NPU firmware | 1.1.2.65 |
| XRT version | 2.21.75 |
| Power draw | ~66W |
| Temperature | 43°C |
| Columns | 8 (32 AIE2P tiles) |
| Engine | Fused layer: 291 tok/s (3.4 ms/tok) |
| GEMM | BFP16: ~31 TFLOPS |
| Silicon peak | 50 TOPS INT8 (58.98 theoretical) |

## Prerequisites

- **Training must finish** before any reboot
- USB drive (FAT32) for SREP boot
- ~2 hours of focused work after reboot

---

## Ranked Unlocks (by impact)

### 🔥 #1 — Firmware Upgrade (51 TOPS validated)
**What:** Upgrade NPU firmware from 1.1.2.65 to 255.0.11.71 + matched driver
**Method:**
```
git clone https://github.com/amd/xdna-driver
cd xdna-driver
git checkout v2.23.0
# Build with gcc-15 (matches kernel compiler)
./build.sh
sudo make modules_install
sudo depmod -a
# Firmware 255.0.11.71 ships with the driver
```
**Expected gain:** Validated 51 TOPS INT8 on other Strix Halo systems. Enables the IRON API INT8 path properly.
**Risk:** Low — out-of-tree driver, easy rollback
**Reference:** `xrt-smi validate` PASS with 51.0 TOPS on Ryzen AI 7 350 (same XDNA2 architecture)

### 🔥 #2 — Multi-core MemTile GEMM Dataflow
**What:** Fix the `whole_array` dataflow to distribute GEMM across all 32 AIE2P tiles via shared MemTile
**Current:** Single-core kernel running at ~83 GOP/s (0.17% of 50 TOPS)
**Target:** All 32 tiles → ~50 TOPS
**Method:** The shared MemTile (row 1) acts as a distributor — each compute tile gets a slice of K-dimension. Requires:
- MemTile configured as shared buffer (`ObjectFifo` with `whole_array` direction)
- Each compute tile runs the same kernel on its K-slice
- Results accumulated or written back independently
**Status:** The `muchdevsuchcode/halo` project proved the concept but segfaults on ≥2 cores. The Chess compiler's pyxrt buffer binding is the likely culprit.
**Expected gain:** ≈ 600× throughput on matmul (single-tile → 32-tile)

### 🔥 #3 — INT8 DMA Stride Fix
**What:** Recalibrate DMA stride formulas in the n1_core MLIR generator for 1-byte INT8 elements
**Current:** BFP16 works (1.125 bytes/element packed format). INT8 xclbin builds but hangs — DMA strides are calibrated for BFP16's 9-byte-per-8-elements packing.
**Fix:** In `n1_core_placed.py`, the DMA stride formulas use:
```
# BFP16: 8 values × 9 bytes per chunk = 72 bytes per line
# INT8:  8 values × 8 bytes per chunk = 64 bytes per line  
```
**Files:** `npu-sandbox/npu-infer/bf16_kernel_dev/n1_core_i8.py`
**Expected gain:** 50 TOPS INT8 vs current 31 TFLOPS BFP16 (~1.6×)

### ⚡ #4 — Weight Pre-Quantization
**What:** Keep INT8 weights resident in NPU BOs instead of re-quantizing every matmul call
**Current:** ~96% of wall time is host overhead (re-quant + re-tile weights every call)
**Fix:** Hold pre-quantized weight BOs in the XCLBIN cache. Load once at model init, reuse for all subsequent matmuls.
**Expected gain:** ~25× on matmul dispatch (the 96% overhead eliminated)

### ⚡ #5 — Fused XCLBIN (FLM-style, open)
**What:** Single xclbin that runs all 4 GEMMs per layer (QKV, GU, O, D) in one NPU instruction stream
**Current:** 4 xclbin dispatches per layer = 112 NPU calls/token for 28 layers
**Target:** 1 fused xclbin per layer = 28 NPU calls/token
**Status:** BitNet decode-layer plan already written at `docs/superpowers/plans/2026-06-29-bitnet-decode-layer-xclbin.md`
**Expected gain:** ~4× (fewer dispatches, on-chip streaming between operations)

---

## SREP Configuration

The current SREP config only needs one patch — the NPU unlock. The rest of the "locks" are software/toolchain, not firmware.

**Current SREP_Config.cfg** (on the USB drive):
```
Op LoadFromFV
SetupUtilityApp
Op FastPatch
Pattern: 32C0488B5C2408488B7C2410C3
B001
Op Exec
```

No additional SREP patches are known for Strix Halo NPU. The AMI BIOS v1.07 on the BeyondMax AXB35 board already has all other hardware features exposed. The NPU clock (1.8 GHz) is fixed in silicon — no BIOS knob changes it.

---

## Reboot Checklist

When training finishes, in order:

### Phase 1: SREP USB Boot (5 min)
1. Plug in the SREP USB (already prepared)
2. Boot from USB (FAT32, UEFI mode)
3. SREP loads, applies the NPU unlock patch
4. Reboot into Linux

### Phase 2: Firmware + Driver Upgrade (30 min)
```bash
# Verify current
xrt-smi examine

# Clone and build matched driver
git clone https://github.com/amd/xdna-driver ~/xdna-driver
cd ~/xdna-driver
git checkout v2.23.0
./build.sh
sudo make modules_install
sudo depmod -a

# Reboot to load new firmware + driver
sudo reboot

# Verify
xrt-smi validate   # Should show 51.0 TOPS PASS
```

### Phase 3: INT8 DMA Stride Calibration (1 hour)
```bash
cd ~/npu-sandbox/npu-infer
# Edit n1_core_i8.py with corrected INT8 stride formulas
# Rebuild INT8 xclbin
bash bf16_kernel_dev/build_i8_xclbin.sh
# Test with test_mt_gemm3
./build/test_mt_gemm3 --xclbin build/int8/qwen3_int8.xclbin
```

### Phase 4: Multi-core GEMM (ongoing)
This is the hard part — implementing the shared MemTile dataflow. See the BitNet decode-layer xclbin plan for the tile grid architecture.

---

## Theoretical Max Performance

If ALL unlocks succeed:

| Metric | Current | Potential | Gain |
|--------|---------|-----------|------|
| GEMM throughput | 31 TFLOPS (BFP16) | **50 TOPS (INT8)** | **1.6×** |
| Multi-core utilization | 1 tile (3%) | **32 tiles (100%)** | **32×** |
| Weight overhead | 96% host time | **<1%** | **25×** |
| Dispatch overhead | 112 calls/token | **28 calls/token** | **4×** |
| **Estimated tok/s** | **291 tok/s (fused)** | **~300-500 tok/s** | **~1-1.7×** |

**The CPU/GPU bottleneck:** At some point NPU decode speed becomes memory-bandwidth-bound (LPDDR5X-8000). The 512K L2 on the NPU and 64MB L3 on the iGPU set a ceiling. Expect diminishing returns past ~300 tok/s on a 0.6B model — the model is small enough that the NPU will saturate.

---

## References

- [NPU Unlock SREP](npu-unlock-srep.md) — Firmware unlock via SmokelessRuntimeEFIPatcher
- [BitNet Decode-Layer XCLBIN](superpowers/plans/2026-06-29-bitnet-decode-layer-xclbin.md) — Multi-core tile grid design
- [INT8 XCLBIN Investigation](npu/INT8-HANDOFF.md) — DMA stride root cause analysis
- [Performance Benchmarks](wiki/performance.md) — Current baseline
- [xdna-driver issue #1250](https://github.com/amd/xdna-driver/issues/1250) — Firmware version discussion
- [muchdevsuchcode/halo](https://muchdevsuchcode.github.io/halo/) — Independent Strix Halo NPU bringup with 51 TOPS validated
