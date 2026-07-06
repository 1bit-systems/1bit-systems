# 🚀 NPU + GPU + CPU = Unified Control Plane

**Single kernel driver + unified memory for AMD Strix Halo on Linux.**

Folding `amdxdna` (XDNA 2 NPU) into `amdgpu` so the NPU, GPU, and CPU share one memory manager, one DRM file descriptor, and one ROCm compute API.

**Live at [1bit.systems](https://1bit.systems)** — "One binary to rule them all. 5 models. 74KB. 28 tok/s NPU."

---

## 🔥 Production: 5-Model NPU Engine — One Binary, 28 tok/s

**Release:** `v2026.07.02-all5models`  
**Engine:** `npu_engine_all` — 74KB stripped C++ binary, auto-detects model dimensions from Q4NX header.

```
g++ -std=c++23 -O3 -o npu_engine_all npu_engine_all.cpp dequant_q4nx.o -lxrt_coreutil
./npu_engine_all model.q4nx 16
```

No Python. No pip. No Docker. No MLIR-AIE toolchain. No torch. Just g++ and run.

### Verified Benchmarks (Strix Halo NPU, INT8 xclbins)

| Model | H | IM | Size | Decode | Tok/s | Layers | Status |
|-------|---|----|------|--------|-------|--------|--------|
| Qwen3-0.6B | 1024 | 3072 | 610 MB | 36 ms/tok | 28 | 28/28 | ✅ |
| Gemma4-E2B | 1536 | 6144 | 4.7 GB | 62 ms/tok | 16 | 35/35 | ✅ |
| Qwen3-VL-4B | 2560 | 9728 | 3.2 GB | 93 ms/tok | 11 | 36/36 | ✅ |
| Llama-3.1-8B | 4096 | 14336 | 5.7 GB | 100 ms/tok | 10 | 32/32 | ✅ |
| Qwen3-8B | 4096 | 12288 | 6.0 GB | 127 ms/tok | 8 | 36/36 | ✅ |

### Fused NPU+GPU Engine (`engine/fusion/`)

The fused engine (`engine/fusion/`) wraps both NPU (XRT xclbin INT8 GEMM) and GPU
(Vulkan flash attention) behind a single API with per-layer dispatch policies.

**Build:** `cd engine/fusion && zig build -Doptimize=ReleaseFast`
**Run:** `./zig-out/bin/fused-engine --model <model.q4nx> --port 8080 --policy auto`

### Dispatch Policies

| --policy | Description |
|----------|-------------|
| auto | FFN+QKV→NPU, Attention→GPU (best throughput) |
| npu_only | All layers → NPU INT8 GEMM |
| gpu_only | All layers → GPU flash attention |
| attention_on_npu | Attention→NPU edge_attention, FFN→GPU DMMV |
| ffn_on_npu | FFN→NPU INT8 GEMM, Attention→GPU flash attn |
| prefill_npu_decode_gpu | Fast prefill on NPU, batch decode on GPU |

The daemon (`npu-gpu-cpud.py`) auto-routes 2B-8B models to the fused engine when
`fused_backend.py` is available. Prefix model name with `fused://` to force
fused routing.

### Engine Evolution (Qwen3-0.6B)
```
v3  (Jul 1): 244 ms/tok   baseline                      1.0×
v6  (Jul 2):  50 ms/tok   batch-4 + OpenMP LM head      4.9×
v8  (Jul 2):  27 ms/tok   M=8 batch decode              9.0×
v9  (Jul 2):  16 ms/tok   M=16 batch decode            15.3×
v11 (Jul 2):  12 ms/tok   M=32 batch decode            20.3×
v12 (Jul 2):  10 ms/tok   M=32 + OpenMP attention      24.4×
all (Jul 2):  36 ms/tok   Model-agnostic, 5 models      —
```

### 23 XCLBINs Across 5 Model Families

| Model Family | Projections | XCLBINs | INT8 Verified |
|-------------|-------------|---------|---------------|
| Qwen3-0.6B | QKV, O, GU, D | 4 | ✅ 28 tok/s |
| Qwen3-8B | QKV, O, G, U, D | 5 | ✅ 8 tok/s |
| Qwen3-VL-4B | QKV, O, G, U, D | 5 | ✅ 11 tok/s |
| Llama-3.1-8B | QKV, O, G, U, D | 5 | ✅ 10 tok/s |
| Gemma4-E2B | QKV, O, GU, D | 4 | ✅ 16 tok/s |

**INT8 is NOT blocked.** The MLIR toolchain was patched (`AIEXDialect.cpp`, `AIETargetModel.cpp`) and INT8 xclbins compile and run. All 5 models use INT8 GEMMs. BFP16 is abandoned (17% fundamental error — see `docs/wiki/bfp16-engine-analysis.md`).

---

## The Vision: One Ring to Rule Them All

```
                          ┌─────────────────────────────────────┐
                          │      Application (any API)          │
                          │  OpenAI-compatible, HIP, raw C++    │
                          └──────────────┬──────────────────────┘
                                         │
                          ┌──────────────▼──────────────────────┐
                          │        ROCm HIP Runtime             │
                          │  (hipMalloc, hipMemcpy, hipLaunch)  │
                          └──────────────┬──────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
    ┌─────────▼─────────┐    ┌──────────▼─────────┐    ┌──────────▼─────────┐
    │     GPU (GFX)     │    │     NPU (XDNA2)    │    │     CPU (x86)      │
    │  amdgpu driver    │    │  amdgpu NPU IP     │    │  Native (Zen 5)    │
    │  RDNA 3.5 CUs     │    │  8 AIE columns     │    │  32 cores          │
    │  80 TFLOPS FP16   │    │  50 TOPS INT8      │    │  ~2-3 tok/s CPU    │
    └─────────┬─────────┘    └──────────┬─────────┘    └──────────┬─────────┘
              │                          │                          │
              └──────────────────────────┼──────────────────────────┘
                                         │
                          ┌──────────────▼──────────────────────┐
                          │       Unified Memory Manager        │
                          │   One DRM fd, one address space     │
                          │   Shared page table (GPU→NPU→CPU)   │
                          └─────────────────────────────────────┘
```

**The problem today**: Three separate drivers. Three memory spaces. Three programming models. The NPU uses XRT + xclbins. The GPU uses ROCm + HIP. The CPU uses llama.cpp. None of them talk to each other.

**The goal**: One `amdgpu` driver that sees the NPU as just another compute engine. One `hipMalloc` that works across all three. One runtime API that schedules work on whatever hardware is available.

---

## 📂 Project Structure

```
├── AGENTS.md                          # Project standards for coding agents
├── Makefile                           # Build everything: binaries, tests, daemon
├── openspec/changes/
│   ├── unified-amdgpu-driver/
│   │   ├── proposal.md                # Kernel driver architecture
│   │   └── tasks.md                   # Implementation tracking (all tasks done)
│   └── 1bit-coding-harness/
│       ├── proposal.md                # 1bit harness design
│       └── tasks.md                   # Harness implementation
├── docs/
│   ├── HANDOFF-NPU-OPTIMIZATION.md    # Complete optimization journey
│   ├── INT8-HANDOFF.md                # INT8 investigation findings
│   ├── REDDIT_POST.md                 # Community call for help
│   └── wiki/                          # Durable project knowledge
│       ├── amdgpu-npu-architecture.md # HW arch + integration points
│       ├── bfp16-engine-analysis.md   # Why BFP16 is broken
│       ├── building-and-testing.md    # Build/run guide
│       ├── shared-gtt-dmabuf.md       # Zero-copy GPU↔NPU design
│       ├── amdxdna-driver-internals.md# Driver reverse-engineering
│       └── smu-init-order.md          # SMU init fix
├── patches/                           # Kernel patches against linux.git
│   ├── amdgpu_npu.c                   # NPU IP block (early_init, sw_init, hw_init)
│   ├── amdgpu_npu.h                   # NPU IP block header
│   ├── amdgpu_npu_mgr.c               # NPU GTT sub-allocator
│   ├── amdgpu_npu_sched.c             # NPU scheduler ring
│   ├── 0001-amdxdna-fix-smu-init-order-strix-halo.patch
│   ├── 0002-add-npu-ip-block.patch
│   ├── apply-and-build.sh             # Apply + build kernel module
│   └── build.sh                       # Kernel module build script
├── rocm-npu/                          # ROCm runtime NPU target
│   ├── hip_npu.cpp                    # LD_PRELOAD shim: HIP sees GPU+NPU
│   ├── hip_npu_memory.cpp             # hipMalloc/hipFree via mmap
│   ├── npu_aql.h                      # NPU AQL packet definitions
│   └── CMakeLists.txt
├── daemon/                            # Control plane daemon
│   ├── npu-gpu-cpud.py                # REST API gateway (port 8080)
│   └── npu_backend.py                 # NPU engine subprocess manager
├── tools/                             # NPU parallelism + server tools
│   ├── npu_multi_context_engine.cpp   # Multi-context verification (4 modes)
│   ├── npu_pipeline_engine.cpp        # Async pipeline submission pool
│   ├── npu_multi_context_server.cpp   # Context pool inference server
│   └── reddit_scraper.py              # Community feedback scraper
├── npu-infer/                         # INT8 kernel generators
│   ├── bf16_kernel_dev/               # MLIR xclbin generators
│   │   ├── n1_core_i8.py              # INT8 MLIR generator
│   │   └── n1_core_i8_v2.py           # INT8 v2 generator
│   └── src/
│       └── npu_engine_i8.cpp          # INT8 inference engine
├── tests/                             # Hardware tests
│   ├── test_gtt_dmabuf.cpp            # dma-buf zero-copy test
│   ├── bench_gtt_dmabuf.cpp           # DMA bandwidth benchmark
│   ├── test_npu_dev.cpp               # NPU device detection
│   ├── test_npu_gemm.cpp              # NPU GEMM via XRT xclbin
│   ├── hip_list_devices.cpp           # HIP device enumeration
│   └── bench_unified.py               # Unified benchmark script
├── build/                             # Build artifacts
└── .github/workflows/                 # CI/CD pipelines
```

### Layered Architecture

```
┌──────────────────────────────────────────────────────────┐
│  USERSPACE                                                │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  OpenAI API     │  │  Custom App  │  │  C++/Python  │ │
│  │  (bridge)       │  │  (HIP)       │  │  (direct)    │ │
│  └────────┬────────┘  └──────┬───────┘  └──────┬───────┘ │
│           │                  │                  │          │
│  ┌────────▼──────────────────▼──────────────────▼───────┐ │
│  │        npu-gpu-cpud (Control Plane Daemon)          │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │ │
│  │  │ NPU      │  │ GPU      │  │ CPU (fallback)   │   │ │
│  │  │ INT8     │  │ ROCm/HIP │  │ llama.cpp        │   │ │
│  │  │ engine   │  │ lemond   │  │                  │   │ │
│  │  └─────┬────┘  └────┬─────┘  └────────┬─────────┘   │ │
│  └────────┼─────────────┼─────────────────┼──────────────┘ │
├───────────┼─────────────┼─────────────────┼────────────────┤
│  KERNEL   │             │                 │                │
│  ┌────────▼─────────────▼─────────────────▼──────────────┐ │
│  │              amdgpu (Unified Driver)                  │ │
│  │  ┌──────────┐  ┌──────────────┐  ┌────────────────┐  │ │
│  │  │ GFX IP   │  │ NPU IP       │  │  DMA-BUF       │  │ │
│  │  │ (gfx1151)│  │ (xDNA2 AIE)  │  │  Shared PT     │  │ │
│  │  └──────────┘  └──────────────┘  └────────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## Status — What's Done

### Kernel (All Complete ✅)
| Task | Status | Artifact |
|------|--------|----------|
| DMA-buf zero-copy GPU↔NPU | ✅ Verified | 27 GB/s read, 56 GB/s write |
| SMU init order fix | ✅ PR'd | lemonade-sdk/amdxdna-dkms#15 |
| NPU IP block (amdgpu_npu.c) | ✅ Written | early/sw/hw init callbacks |
| NPU scheduler ring | ✅ Written | amdgpu_npu_sched.c |
| NPU GTT sub-allocator | ✅ Written | amdgpu_npu_mgr.c |
| PCI DID registration | ✅ Written | 0002-add-npu-ip-block.patch |
| Kernel build scripts | ✅ Written | apply-and-build.sh |

### ROCm Userspace (All Complete ✅)
| Task | Status | Artifact |
|------|--------|----------|
| HIP NPU shim (device enum) | ✅ Built | libhip_npu.so, LD_PRELOAD tested |
| HIP NPU memory alloc | ✅ Built | mmap-based GTT allocation |
| NPU AQL packet defs | ✅ Written | npu_aql.h |

### NPU Inference (All Complete ✅)
| Task | Status | Detail |
|------|--------|--------|
| INT8 engine | ✅ 28 tok/s | 5 models, one binary, auto-detect |
| Multi-context pool | ✅ 7.9× scaling | npu_server, 8 contexts |
| Pipeline engine | ✅ 7.3× scaling | Async submission pool |
| 2-layer batch INT8 | ✅ Built | QKV/O/D xclbins verified |
| Multi-token INT8 | ✅ Built | M=128 batch at 99.9ms/tok |

### Daemon + Tools (All Complete ✅)
| Task | Status | Artifact |
|------|--------|----------|
| Control plane daemon | ✅ Running | npu-gpu-cpud.py, port 8080 |
| NPU backend subprocess | ✅ Running | npu_backend.py, stdin/stdout JSON |
| Multi-context server | ✅ Built | npu_multi_context_server |
| Pipeline benchmark | ✅ Built | npu_pipeline_engine |
| HIP device test | ✅ Passing | hip_list_devices (2 devices) |

---

## Quick Start

```bash
git clone https://github.com/bong-water-water-bong/npu-gpu-cpu.git
cd npu-gpu-cpu

# Build everything
make

# Run hardware tests
sudo make test

# Run benchmarks
sudo make bench

# Build NPU parallelism tools
make npu-tools

# Run multi-context demo
make npu-demo

# Start the control plane daemon
sudo python3 daemon/npu-gpu-cpud.py --port 8080
```

---

**[github.com/bong-water-water-bong/npu-gpu-cpu](https://github.com/bong-water-water-bong/npu-gpu-cpu)** | **[1bit.systems](https://1bit.systems)**
