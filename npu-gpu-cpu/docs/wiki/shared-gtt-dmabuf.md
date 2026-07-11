# Shared GTT: Zero-Copy GPU↔NPU via dma-buf

## Approach

Allocate GPU buffers in `TTM_PL_TT` (GTT domain, system RAM) instead of
`TTM_PL_VRAM` (dedicated VRAM). Export via dma-buf to the NPU. Both devices
access the same physical pages — zero copy.

```
┌─────────────────────────────────────────────────────┐
│                   System RAM                         │
│                                                      │
│  ┌───────────────────────────────────────────────┐  │
│  │              Shared GTT Buffer                │  │
│  │         (TTM_PL_TT, physically pinned)        │  │
│  └──────────┬────────────────────┬───────────────┘  │
│             │                    │                   │
│             ▼                    ▼                   │
│  ┌──────────────────┐  ┌────────────────────┐       │
│  │  GPU (amdgpu)    │  │  NPU (amdxdna)     │       │
│  │  via GTT aperture│  │  via IOMMU SVA     │       │
│  │  Same phys pages │  │  Same phys pages   │       │
│  └──────────────────┘  └────────────────────┘       │
│                                                      │
│  ← ← ← ← ← ← ← ZERO COPY → → → → → → → → →         │
└─────────────────────────────────────────────────────┘
```

## Why it works

1. **`TTM_PL_TT`** allocates pages from system RAM, pinned and mapped
   through the GPU's GTT (Graphics Translation Table). The GPU can DMA
   to/from these pages via its GTT aperture.

2. **amdgpufs dma-buf export** creates a dma-buf from the TTM BO. The
   dma-buf's backing storage is the same set of physical pages.

3. **amdxdna dma-buf import** (`amdxdna_gem_prime_import`) attaches to
   the dma-buf and wraps the pages in a shmem GEM object. In SVA mode
   the NPU firmware accesses them via PASID; in IOVA mode via
   `iommu_map_sgtable`.

4. Both sides reference the same physical pages → no data copy.

## When NOT to use this

- Model weights that fit entirely in VRAM should stay in VRAM (faster)
- Only models that exceed VRAM or are shared between GPU+NPU should use GTT
- GTT bandwidth is limited by PCIe/IOMMU throughput (~20-30 GB/s vs
  VRAM's ~500 GB/s on dGPU)

## Integration into Lemonade

`AutoTune` already selects devices based on VRAM. The enhancement:

1. If model fits in VRAM → allocate in TTM_PL_VRAM (fast path)
2. If model exceeds VRAM but fits in system RAM → allocate in TTM_PL_TT,
   export to NPU, run inference on NPU while GPU does embedding/reranking
3. If model exceeds system RAM → CPU fallback (llama.cpp CPU mode)

## Test Plan

```c
// 1. Allocate BO in TTM_PL_TT on amdgpu
// 2. Export as dma-buf (fd)
// 3. Import fd into amdxdna
// 4. GPU writes pattern (0xDEADBEEF)
// 5. NPU reads pattern — verify it matches
// 6. NPU writes pattern (0xCAFEBABE)
// 7. GPU reads pattern — verify it matches
// If both sides see the same data without explicit copy → zero-copy verified
```
