# AGENTS.md — NPU Project Status (Final)

## Session: 2026-07-02 — Production Release

### Engine Status (Verified 2026-07-02)

| Engine | ms/tok | tok/s | Status |
|--------|--------|-------|--------|
| INT8 v12 (Qwen3-0.6B batch) | 10 | 97 | ✅ PRODUCTION |
| INT8 all5models (Qwen3-0.6B) | 36 | 28 | ✅ RELEASED v2026.07.02-all5models |
| INT8 all5models (Gemma4-E2B) | 62 | 16 | ✅ |
| INT8 all5models (Qwen3-VL-4B) | 93 | 11 | ✅ |
| INT8 all5models (Llama-3.1-8B) | 100 | 10 | ✅ |
| INT8 all5models (Qwen3-8B) | 127 | 8 | ✅ |
| GPU (ROCm, Radeon 8060S) | — | 281 | ✅ Bonsai-1.7B IQ1_S |
| Multi-context pool | — | 64 req/s | ✅ 8 ctx, 7.9× speedup |

### All Tasks Complete

| Layer | Deliverable | Status |
|-------|-------------|--------|
| **Kernel patches** | amdgpu_npu.c, mgr.c, sched.c, SMU fix, PCI IDs | ✅ Written & buildable |
| **ROCm userspace** | hip_npu.cpp (shim), npu_aql.h, libhip_npu.so | ✅ Built & tested |
| **NPU inference** | INT8 engine at 28 tok/s (5 models) | ✅ Released |
| **NPU v12 batch** | 10 ms/tok (97 tok/s) on Qwen3-0.6B | ✅ PRODUCTION |
| **Multi-context engine** | npu_server — context pool, 7.9× scaling | ✅ Built & benchmarked |
| **Control plane** | npu-gpu-cpud.py — HTTP gateway with routing policy | ✅ Running |
| **Documentation** | 6 wiki docs + handoff docs | ✅ Complete |
| **Tests** | Device test, dma-buf, benchmark, HIP enumeration | ✅ All passing |
| **Website** | https://1bit.systems — live with all5models numbers | ✅ Live |
| **Release** | v2026.07.02-all5models — 5 models, one binary, 120KB | ✅ Tagged |

### Key Files Map

```
npu-gpu-cpu/
├── tools/
│   ├── npu_multi_context_engine.cpp   # Parallelism verification (4 test modes)
│   ├── npu_pipeline_engine.cpp        # Async submission pool, 7.3× scaling
│   └── npu_multi_context_server.cpp   # Context pool — replaces FLM HTTP server
├── docs/
│   ├── HANDOFF-NPU-OPTIMIZATION.md    # Complete journey
│   ├── INT8-HANDOFF.md                 # INT8 findings
│   └── wiki/
│       ├── amdgpu-npu-architecture.md
│       ├── bfp16-engine-analysis.md
│       ├── building-and-testing.md
│       ├── shared-gtt-dmabuf.md
│       ├── amdxdna-driver-internals.md
│       └── smu-init-order.md
├── patches/  (kernel) — amdgpu_npu.c, mgr.c, sched.c, SMU fix
├── rocm-npu/  (HIP shim) — hip_npu.cpp, npu_aql.h, libhip_npu.so
├── daemon/  (control plane) — npu-gpu-cpud.py, npu_backend.py
├── tests/  (all passing)
└── Makefile
```

### Key External Paths

| Path | Purpose |
|------|---------|
| `/home/bcloud/1bit-systems/` | Production engine + website (source of truth for inference) |
| `/home/bcloud/npu-sandbox/npu-infer/build/int8/` | All 23 xclbins + instruction files |
| `/home/bcloud/torch2aie/` | MLIR-AIE toolchain + IRON API |
| `/home/bcloud/mlir-aie/` | MLIR-AIE source (patched for INT8 support) |

### What's Left For Future Sessions

1. **Fused xclbin port**: 3 xclbins compiled (QKV-prefix, full-layer, unified). Blocked by Q4NX weight format conversion.
2. **16-context stress test**: Push to driver limit, measure scaling ceiling
3. **Unified driver kernel build**: Run `apply-and-build.sh` after kernel update
4. **OpenAI-compatible bridge**: Wire all5models engine through the daemon's chat endpoint
5. **Website update**: Refresh 1bit.systems with v12 standalone numbers (97 tok/s)
