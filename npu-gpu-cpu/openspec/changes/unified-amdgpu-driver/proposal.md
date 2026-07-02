# Unified AMDGPU Driver: NPU + GPU + CPU

**Change ID:** `unified-amdgpu-driver`
**Status:** Implemented (all kernel + userspace tasks complete; inference engine at 28 tok/s)
**Date:** 2026-07-02

## Summary

Fold `amdxdna` (XDNA 2 NPU driver, currently in `drivers/staging/`) into `amdgpu` as a first-class IP block, exposing the NPU through the same DRM file descriptor and TTM memory manager. Then extend ROCm's HIP runtime to recognize the NPU as a compute agent, enabling transparent dispatch to GPU or NPU through a single API.

## Motivation

Strix Halo (gfx1151) has a unified memory controller shared by GPU, NPU, and CPU. The hardware already supports zero-copy sharing — but the kernel presents two separate devices with separate VA spaces and BO heaps. This means:

- A buffer allocated on the GPU (`amdgpu_bo_create`) cannot be directly consumed by the NPU
- Two copies of models in RAM: one for NPU inference, one for GPU inference
- No single driver update covers both (separate build/packaging)
- The NPU cannot leverage ROCm's mature tooling (rocProf, rocGDB, rocBLAS)

## Architecture

### Target: Single `amdgpu` + NPU IP Block

```
Before:                          After:
┌──────────┐  ┌──────────┐     ┌──────────────────┐
│  amdgpu  │  │ amdxdna  │     │     amdgpu       │
│  .ko     │  │ .ko      │     │  ┌────┐ ┌──────┐ │
│  ┌────┐  │  │ ┌─────┐  │     │  │GPU │ │ NPU  │ │
│  │GPU │  │  │ │ NPU │  │     │  │IP  │ │IP    │ │
│  │ BO │  │  │ │ BO  │  │     │  │block│ │block │ │
│  └────┘  │  │ └─────┘  │     │  └────┘ └──────┘ │
│ fd=0     │  │ fd=0     │     │  ┌──────────────┐ │
└──────────┘  └──────────┘     │  │  TTM (shared) │ │
                                │  └──────────────┘ │
                                │  fd=0             │
                                └──────────────────┘
```

### Phase 1: Kernel — Fold NPU into amdgpu

#### 1a. Move amdxdna IP detection into amdgpu

The NPU appears as a PCI function in the same power domain as the GPU. `amdgpu_device_init()` already walks PCI functions to discover IP blocks. Add NPU detection:

```c
/* amdgpu_npu.c — new file */
int amdgpu_npu_init(struct amdgpu_device *adev)
{
    adev->npu.fw = amdgpu_ucode_request(adev, "amdgpu/npu_*.bin");
    amdgpu_npu_ring_init(adev, &adev->npu.ring);
}

int amdgpu_npu_early_init(struct amdgpu_device *adev)
{
    adev->npu.mmio = adev->rmmio + NPU_MMIO_OFFSET;
}
```

#### 1b. Shared TTM memory manager

The NPU currently allocates via `dma_alloc_coherent` (separate from TTM). Replace this with `amdgpu_bo_create` using a new `TTM_PL_NPU` placement domain:

```c
/* amdgpu_ttm.c — add NPU placement */
static const struct ttm_place npu_placement = {
    .fpfn = 0,
    .lpfn = 0,
    .mem_type = TTM_PL_NPU,
    .flags = TTM_PL_FLAG_CONTIGUOUS,
};
```

A BO allocated in `TTM_PL_VRAM` (GPU) or `TTM_PL_NPU` or `TTM_PL_TT` (system RAM, shared) is visible to all three via the shared physical pages.

#### 1c. Single DRM device node

The NPU reuses the GPU's `/dev/dri/card0`. A new DRM IOCTL (`AMDGPU_NPU_CTX`) creates an NPU compute context within the existing fd:

```c
#define DRM_AMDGPU_NPU_CTX 0x20

struct drm_amdgpu_npu_ctx {
    __u64 ctx_id;       /* out */
    __u32 flags;        /* in */
    __u32 pad;
};
```

#### 1d. NPU scheduler ring

The NPU has a hardware scheduler ring (doorbell-based). Add `amdgpu_npu_sched.c`:

```c
int amdgpu_npu_ring_submit(struct amdgpu_ring *ring,
                           struct amdgpu_job *job)
{
    WREG32_NPU(ring->doorbell_offset, job->npu_cmd_header);
}
```

### Phase 2: ROCm — NPU as a HIP Agent

#### 2a. `rocminfo` NPU agent

```c
static hsa_device_type_t
amd_npu_get_device_type(struct amd_gpu_device *dev)
{
    return HSA_DEVICE_TYPE_NPU;
}
```

#### 2b. `hipMalloc` on NPU

```c
hipError_t hipMallocNPU(void **ptr, size_t size)
{
    struct amdgpu_bo_alloc_request req = {
        .alloc_size = size,
        .preferred_heap = AMDGPU_GEM_DOMAIN_NPU,
    };
    amdgpu_bo_alloc(fd, &req, &bo_handle);
    amdgpu_bo_cpu_map(bo_handle, ptr);
}
```

#### 2c. Kernel launch dispatch

```c
void hip_npu_launch(const void *kernel_func, dim3 gridDim,
                    dim3 blockDim, void **args, size_t sharedMem,
                    hipStream_t stream)
{
    struct npu_aql_packet pkt = {
        .opcode = NPU_AQL_KERNEL_DISPATCH,
        .grid = gridDim,
        .workgroup = blockDim,
        .kernel_obj = kernel_func,
        .kernargs_address = args,
    };
    amdgpu_npu_submit_packet(stream->npu_ring, &pkt);
}
```

### Phase 3: Device Selection Policy

| Model Size | GPU VRAM | NPU Mem | Recommended Device |
|---|---|---|---|
| < 2 GB | Any | Any | NPU (lowest power) |
| 2-8 GB | Free | Any | GPU (fastest compute) |
| 8-24 GB | Free | — | GPU |
| 8-24 GB | Full | Any | NPU (fallback) |
| > 24 GB | — | — | CPU |

## Files Changed

### Kernel (against linux.git)

| File | Change |
|------|--------|
| `drivers/gpu/drm/amd/amdgpu/Makefile` | Add `amdgpu_npu.o` |
| `drivers/gpu/drm/amd/amdgpu/amdgpu.h` | Add `struct amdgpu_npu` |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_npu.c` | **New** — NPU init, ring, firmware |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_npu_sched.c` | **New** — NPU scheduler |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_ttm.c` | Add `TTM_PL_NPU` |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_kms.c` | Register `AMDGPU_NPU_CTX` |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_drv.c` | PCI ID — add NPU DID |
| `drivers/staging/amdxdna/` | Remove |

### Userspace (this repo)

| File | Change |
|------|--------|
| `rocm-npu/hip_npu.cpp` | HIP NPU agent |
| `rocm-npu/npu_aql.h` | NPU AQL packet defs |
| `daemon/npu-gpu-cpud.cpp` | Scheduler daemon |

## Risks

1. **NPU firmware is closed** — binary-only blob, stuck with existing mailbox ABI
2. **Upstream resistance** — amdgpu maintainers may resist staging driver absorption
3. **ABI compatibility** — firmware expects specific interface, can't break it

## Success Criteria

1. `cat /sys/kernel/debug/dri/0/amdgpu_npu_info` shows NPU state
2. BO allocated via `amdgpu_bo_create(GEM_DOMAIN_NPU)` accessible from GPU + CPU
3. `rocminfo` lists NPU as an agent
4. `hipLaunchKernel` on NPU runs a simple GEMM
5. Lemonade sees one unified device

## References

- `drivers/staging/amdxdna/`
- `drivers/gpu/drm/amd/amdgpu/`
- https://lists.freedesktop.org/mailman/listinfo/amd-gfx
- https://github.com/ROCm/hip
