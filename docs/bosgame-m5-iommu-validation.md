# Bosgame M5 — IOMMU-Off Validation Report

**Date:** 2026-07-23
**System:** Bosgame M5 (AMD Ryzen AI Max+ 395, Radeon 8060S)
**Kernel:** 7.0.0-28-generic
**Config:** `amd_iommu=off` in GRUB_CMDLINE_LINUX_DEFAULT

## Why IOMMU Off?

Based on [Frontier Lab field report](https://thefrontierlab.ai/strix-halo-tuning-part-two-iommu/):

| Cell | IOMMU On | IOMMU Off | Gain |
|------|----------|-----------|------|
| Dense PP (empty) | 255 t/s | 351 t/s | **+37.5%** |
| Dense PP (32k) | 78 t/s | 105 t/s | **+34.1%** |
| MoE PP (empty) | 1,007 t/s | 1,074 t/s | **+6.7%** |
| Generation | — | — | +0.6–1.4% |

**Scope:** Dedicated inference box only. Keep IOMMU on if running VMs or VFIO passthrough.

## Hardware Validation

| Check | Result |
|-------|--------|
| Kernel cmdline | ✅ `amd_iommu=off` |
| IOMMU groups in sysfs | ✅ Empty (0 entries) |
| AMD-Vi driver in boot log | ✅ Zero references — driver never bound |
| GPU DMA | ✅ Direct — no IOMMU translation layer |
| GPU IOMMU group | ✅ None — GPU has direct PCIe DMA |
| NPU (amdxdna) SVA bind | ✅ Expected failure (IOMMU required for SVA) |

## Live Inference Benchmark

**Backend:** Vulkan (RADV STRIX_HALO)
**Build:** zaya-llama.cpp @ 1a7582b91 (build 9094)
**Model:** Qwen3-4B-Q4_K_M (4B params, ~2.5 GB)
**Offload:** `-ngl 99` (all layers on GPU)
**Batch:** 2048 | **UBatch:** 512 | **Flash Attn:** off

### Prompt Processing Throughput

| Context Length | Tokens/s | Std Dev | Variance |
|---------------|----------|---------|----------|
| 128 tok | 2,134 ± 4 | 0.19% | ✅ |
| 512 tok | 2,191 ± 3 | 0.14% | ✅ |
| 1,024 tok | 1,760 ± 3 | 0.17% | ✅ |
| 2,048 tok | 1,615 ± 1 | 0.06% | ✅ |
| 4,096 tok | 1,338 ± 0.4 | 0.03% | ✅ |
| 8,192 tok | 983 ± 0.3 | 0.03% | ✅ |

### Generation Throughput (from 512-token prompt)

| Tokens to Generate | Tokens/s | Std Dev | Variance |
|-------------------|----------|---------|----------|
| 128 tok | 75.6 ± 0.38 | 0.50% | ✅ |
| 256 tok | 75.6 ± 0.46 | 0.60% | ✅ |
| 512 tok | 73.4 ± 0.11 | 0.15% | ✅ |

### Stability Assessment

- **GPU errors:** None — zero compute ring timeouts, zero DRM errors
- **Sample variance:** <0.6% across all benchmark cells — extremely consistent
- **Device stability:** No resets, no crashes across 15+ benchmark invocations
- **Thermal:** No throttling observed during runs
- **Vulkan backend:** Clean initialization every run, no memory leaks

## Kernel Configuration

```grub
GRUB_CMDLINE_LINUX_DEFAULT="amd_iommu=off amdgpu.no_system_mem_limit=1 amdgpu.noretry=0 amdgpu.gfxoff=0 ttm.pages_limit=31457280"
```

Applied via `/etc/default/grub`, activated with `sudo update-grub && reboot`.

## Verification Commands

```bash
# Confirm IOMMU is off
cat /proc/cmdline | tr ' ' '\n' | grep iommu
ls /sys/kernel/iommu_groups/          # should be empty

# Check GPU is working
journalctl -k --no-pager | grep -i "amdgpu.*initializing"
vulkaninfo --summary | grep -i deviceName

# Run benchmark
./build-vulkan/bin/llama-bench -m model.gguf -ngl 99 -p 512 -n 128
```
