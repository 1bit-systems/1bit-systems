# AMD NPU+GPU+CPU Unified Control Plane — Architecture

## Hardware

**AMD Strix Halo (Ryzen AI MAX+ 395)**

| Die | Arch | Compute | Memory | Power | Linux Driver |
|-----|------|---------|--------|-------|-------------|
| **CPU** | Zen 5, 32 cores | ~1 TFLOPS | DDR5 (shared) | 15-35W | `core` |
| **GPU** | RDNA 3.5, gfx1151, 20 CUs | ~12 TFLOPS | DDR5 (shared, 64 GB) | 15-25W | `amdgpu` |
| **NPU** | XDNA 2, aie2, 8 columns | 50 TOPS INT8 | DDR5 (shared) | ~2W | `amdxdna` |

All three share the same DDR5 memory controller. The hardware supports zero-copy — the kernel doesn't.

## Software Stack

```
┌──────────────────────────────────────────────────────┐
│                 User Application                      │
│  (HIP, OpenAI API, llama.cpp, custom C++)            │
└────────┬────────────┬──────────────┬─────────────────┘
         │            │              │
    ┌────▼────┐ ┌─────▼──────┐ ┌────▼─────┐
    │  GPU    │ │   NPU     │ │   CPU    │
    │  ROCm   │ │ INT8      │ │ llama.cpp│
    │  HIP    │ │ Engine    │ │          │
    │         │ │ (XRT)     │ │          │
    └────┬────┘ └─────┬──────┘ └────┬─────┘
         │            │              │
    ┌────▼────┐ ┌─────▼──────┐ ┌────▼─────┐
    │amdgpu.ko│ │amdxdna.ko │ │  native  │
    │  TTM    │ │ IOMMU SVA │ │          │
    │  GTT    │ │  PASID    │ │          │
    └─────────┘ └───────────┘ └──────────┘
         │            │              │
    ┌────▼────────────▼──────────────▼─────┐
    │           IOMMU / SVA                │
    │       Shared Virtual Addressing      │
    └──────────────────────────────────────┘
```

## Key Integration Points

### 1. DMA-buf Zero-Copy (Proven ✅)

```
GPU allocates TTM_PL_TT (GTT, system RAM)
         │
    dma-buf export (drmPrimeHandleToFD)
         │
         ▼
NPU imports (DRM_IOCTL_PRIME_FD_TO_HANDLE)
         │
         ▼
Both access same physical pages: 27 GB/s read, 56 GB/s write
```

**Files:** `tests/test_gtt_dmabuf.cpp`, `tests/bench_gtt_dmabuf.cpp`

### 2. SMU Init Order Fix (PR'd — `lemonade-sdk/amdxdna-dkms#15`)

On Strix Halo, the NPU's SMU (System Management Unit) is embedded in the
firmware package loaded by PSP. The driver initialized SMU before PSP,
causing SMU init to always fail. The community workaround bypassed SMU
entirely, leaving the NPU without power management.

**Fix:** `patches/0001-amdxdna-fix-smu-init-order-strix-halo.patch`
Swaps init order: PSP → SMU. Also adds `stop_psp_no_smu` cleanup label.

### 3. NPU IP Block (Compile-tested on kernel 7.0.0)

`patches/amdgpu_npu.c` — wraps amdxdna into amdgpu's IP block lifecycle:

| Callback | What it does |
|----------|-------------|
| `early_init` | Discover NPU PCI function (bus:00.1) |
| `sw_init` | Request firmware, map BARs, init ring |
| `hw_init` | PSP load → SMU power on → firmware alive check |
| `hw_fini` | Suspend firmware → stop mailbox → stop PSP |
| `sw_fini` | Free ring, release firmware, put PCI dev |

**Integration patch:** `patches/0002-add-npu-ip-block.patch` (6 files)

### 4. HIP NPU Shim (Builds and runs ✅)

`libhip_npu.so` — LD_PRELOAD library that makes HIP see the NPU:

```
Without shim:  hipGetDeviceCount → 1 (GPU only)
With shim:     hipGetDeviceCount → 2 (GPU 0 + NPU 1)
```

**File:** `rocm-npu/hip_npu.cpp`

### 5. Scheduler Daemon (Running on Strix Halo ✅)

`npu-gpu-cpud.py` — REST API gateway at port 8080:

| Endpoint | Description |
|----------|-------------|
| `GET /v1/health` | Device status and policy |
| `GET /v1/models` | Available models (proxied from NPU) |
| `POST /v1/chat/completions` | Routes to NPU/GPU/CPU |

**Policy:** `< 2B→NPU`, `2B-8B→GPU`, `> 8B→CPU`

## Benchmarks

| Metric | Value |
|--------|-------|
| GPU↔NPU bandwidth (read) | 27 GB/s |
| GPU↔NPU bandwidth (write) | 56 GB/s |
| NPU Qwen3-0.6B INT8 (v12 batch) | 97 tok/s (10 ms/tok) |
| NPU Qwen3-0.6B INT8 (all5models) | 28 tok/s (36 ms/tok) |
| NPU Gemma4-E2B INT8 | 16 tok/s (62 ms/tok) |
| NPU Llama-3.1-8B INT8 | 10 tok/s (100 ms/tok) |
| NPU power | ~2W |
| Multi-context scaling | 7.9× (8 contexts) |

## Repos

| Repo | URL | Description |
|------|-----|-------------|
| npu-gpu-cpu | https://github.com/bong-water-water-bong/npu-gpu-cpu | Main project — kernel patches, HIP shim, daemon, docs |
| 1bit-systems | https://github.com/bong-water-water-bong/1bit-systems | Production NPU engine + website (1bit.systems) |
| 1bit-lemonade | https://github.com/bong-water-water-bong/1bit-lemonade | Fork with vLLM fix merged |
| amdxdna-dkms (fork) | https://github.com/bong-water-water-bong/amdxdna-dkms | Fork with SMU init fix |
| npu-sandbox | ~/npu-sandbox/ | XCLBINs, MLIR generators, build artifacts |
