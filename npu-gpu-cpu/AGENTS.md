# AGENTS.md

## Project

Unified kernel driver + ROCm integration + NPU inference engine for Strix Halo NPU+GPU+CPU on Linux.

## Repo Structure

```
npu-gpu-cpu/
├── AGENTS.md              # This file — agent instruction schema
├── Makefile               # Build everything: tests, tools, rocm-npu, daemon
├── README.md              # Project overview + architecture
├── build/                 # Build artifacts (binaries, libs)
├── openspec/
│   └── changes/
│       ├── unified-amdgpu-driver/
│       │   ├── proposal.md   # Kernel driver architecture
│       │   └── tasks.md      # Implementation tracking
│       └── 1bit-coding-harness/
│           ├── proposal.md   # 1bit harness design
│           └── tasks.md      # Harness tasks
├── docs/
│   ├── AGENTS.md              # NPU session summary
│   ├── HANDOFF-NPU-OPTIMIZATION.md  # Complete optimization journey
│   ├── INT8-HANDOFF.md        # INT8 investigation
│   ├── REDDIT_POST.md         # Community post
│   └── wiki/                  # Durable project knowledge
│       ├── amdgpu-npu-architecture.md
│       ├── bfp16-engine-analysis.md
│       ├── building-and-testing.md
│       ├── shared-gtt-dmabuf.md
│       ├── amdxdna-driver-internals.md
│       └── smu-init-order.md
├── patches/
│   ├── amdgpu_npu.c           # NPU IP block (early_init, sw_init, hw_init)
│   ├── amdgpu_npu.h
│   ├── amdgpu_npu_mgr.c       # NPU GTT sub-allocator
│   ├── amdgpu_npu_sched.c     # NPU scheduler ring
│   ├── 0001-amdxdna-fix-smu-init-order-strix-halo.patch
│   ├── 0002-add-npu-ip-block.patch
│   ├── apply-and-build.sh
│   └── build.sh
├── rocm-npu/                  # ROCm runtime NPU target (userspace)
│   ├── hip_npu.cpp            # LD_PRELOAD shim
│   ├── hip_npu_memory.cpp     # hipMalloc/hipFree
│   ├── npu_aql.h              # AQL packet defs
│   └── CMakeLists.txt
├── daemon/                    # Control plane daemon
│   ├── npu-gpu-cpud.py        # REST API gateway
│   └── npu_backend.py         # NPU engine subprocess manager
├── tools/                     # NPU parallelism tools
│   ├── npu_multi_context_engine.cpp
│   ├── npu_pipeline_engine.cpp
│   ├── npu_multi_context_server.cpp
│   └── reddit_scraper.py
├── npu-infer/                 # XCLBIN generators + INT8 engine source
│   ├── bf16_kernel_dev/       # MLIR xclbin generators
│   │   ├── n1_core_i8.py
│   │   └── n1_core_i8_v2.py
│   └── src/
│       └── npu_engine_i8.cpp
└── tests/
    ├── test_gtt_dmabuf.cpp
    ├── bench_gtt_dmabuf.cpp
    ├── test_npu_dev.cpp
    ├── test_npu_gemm.cpp
    ├── hip_list_devices.cpp
    └── bench_unified.py
```

## Workflow

1. Start non-trivial work with `openspec/changes/<change-id>/proposal.md`
2. Track implementation in `openspec/changes/<change-id>/tasks.md`
3. Update `docs/wiki/` whenever work reveals durable repo knowledge
4. Keep changes surgical, simple, and verified

## References

- `amdgpu` in-tree driver: `drivers/gpu/drm/amd/amdgpu/`
- `amdxdna` staging driver: `drivers/staging/amdxdna/`
- ROCm: `https://github.com/ROCm/`
- NPU engine (source of truth): `/home/bcloud/1bit-systems/engine/npu/`
- XCLBINs: `/home/bcloud/npu-sandbox/npu-infer/build/int8/`

## Key Findings (as of 2026-07-02)

1. **BFP16 is abandoned** — v8bfp16ebs8 format gives 17% per-GEMM error, compounds to degenerate output
   - `docs/wiki/bfp16-engine-analysis.md` for full root cause
   - All production work uses INT8 xclbins (final_i8_*.xclbin)
2. **INT8 engine fully working** — 5 models on one 74KB binary
   - Model-agnostic: auto-detects dims from Q4NX header
   - Build: `g++ -std=c++23 -O3 -mavx2 -mfma -mavx512f -ffast-math -march=native`
   - Source: `/home/bcloud/1bit-systems/engine/npu/src/npu_engine_all.cpp`
   - Benchmarks: Qwen3-0.6B (28 tok/s), Gemma4-E2B (16 tok/s), Llama-3.1-8B (10 tok/s)
3. **INT8 xclbins compile and run** — MLIR toolchain was patched (AIEXDialect.cpp, AIETargetModel.cpp)
   - 23 xclbins across 5 model families
   - INT8 is NOT blocked — the toolchain patches work
4. **Attention K/V pointer bug fixed** — old binary had K/V at wrong offset
   - Fix: pass `&kv[l].k[w*WKVH*HD]` (window offset only)
5. **MultI-context parallelism works** — up to 7.9× scaling with 8 contexts
   - `tools/npu_multi_context_server.cpp` — context pool inference server
   - `tools/npu_pipeline_engine.cpp` — async pipeline submission pool (7.3×)
6. **Daemon integration via subprocess** — `npu_engine_all` uses stdin/stdout JSON
   - Python backend: `daemon/npu_backend.py`
   - Gateway: `daemon/npu-gpu-cpud.py` — REST API on port 8080
7. **AVX-512 build flags critical** — missing `-mavx512f`/`-ffast-math` caused 2× slowdown
   - With flags: 378 AVX-512 instructions, 174ms/tok
   - Without: 0 AVX-512 instructions, ~340ms/tok
