# NPU Engine Benchmarks — 2026-07-25

**Hardware**: AMD Ryzen AI Max+ 395 (Strix Halo), XDNA 2 NPU (32 AIE2P), Radeon 8060S (gfx1151)  
**Model**: Qwen3-0.6B (Q4NX, 28L, H=1024, NH=16, NKV=8, HD=128, IM=3072)  
**GPU Kernels**: Tiled f16 GEMV (col-major), Flash-Decoding attention (via CPU stub)  
**NPU Kernels**: I8 GEMM via 4× xclbins (QKV, O, GU, D)  

## Results

| Engine | Architecture | tok/s | ms/tok | Status |
|--------|-------------|:-----:|:------:|--------|
| v12 (baseline) | Pure NPU, 4× I8 GEMM | — | — | ❌ Crashes (#939) |
| v13 | Pure NPU async pipeline | — | — | ❌ Segfault (#940) |
| **fused** | GPU QKV+O, NPU GU+D | **2.4** | 410 | ✅ Runs, CPU stub |
| **overlap** | 3-buffer pipelined | **5.0** | 200 | ✅ Runs, CPU stub |
| **spec** | GPU draft + NPU verify | **0.9** | 1110 | ✅ Runs, 0% draft |

## Per-Layer Breakdown (Overlap Engine)

| Component | ms/layer | Description |
|-----------|:--------:|-------------|
| GPU QKV gemv | 3.1 | f16 GEMV + H2D/D2H sync |
| CPU norms/RoPE | 0.2 | RMS norm, RoPE, KV cache |
| CPU attention | 8.4 | attn_stub.cpp (sequential) |
| GPU O gemv | 2.8 | f16 GEMV + H2D/D2H sync |
| NPU GU | 7.2 | I8 GEMM on XDNA 2 |
| NPU D | 6.5 | I8 GEMM on XDNA 2 |
| **Total/layer** | **28.2** | ×28 layers = 789ms (theoretical) |

Bottleneck: **CPU attention stub** — 8.4ms/layer × 28 = 235ms/token.  
Fix: Replace with GPU Flash-Decoding → targets 40-60 tok/s.

## Issues

| # | Title | Priority |
|:-:|-------|:--------:|
| 936 | CPU attention stub is primary bottleneck | **Critical** |
| 937 | hipMemcpy per-GEMV sync overhead | High |
| 938 | Spec decode draft 0% acceptance | Medium |
| 939 | v12 crashes: qds_device::wait() error | High |
| 940 | v13 segfaults on xclbin init | High |

## Build

```bash
# Pure NPU (g++ 15):
g++ -O3 -mavx512f -mavx512bw -mavx512vl -mavx512dq -fopenmp \
    src/npu_engine_v13.cpp src/dequant_q4nx.cpp \
    -lxrt_coreutil -lxrt_core -luuid -lpthread -laiebu -lm -ldl \
    -o build/npu_engine_v13

# GPU targets (hipcc / ROCm 7.1, gfx1151):
for t in fused overlap spec; do
  hipcc -O3 -mavx512f -mavx512bw -mavx512vl -mavx512dq \
      -D__HIP_PLATFORM_AMD__=1 -I src -I include -I/opt/rocm/include \
      src/npu_engine_$t.hip src/gpu_kernels_fused.hip \
      src/dequant_q4nx.cpp src/attn_stub.cpp \
      --offload-arch=gfx1151 \
      -lxrt_coreutil -lxrt_core -luuid -lpthread -laiebu -lm -ldl -fopenmp \
      -o build/npu_engine_$t
done
```

## Source Files

```
engine/npu/src/
├── npu_engine_v13.cpp          # Pure NPU async (702 lines)
├── npu_engine_fused.hip        # GPU+NPU fused (1140 lines)
├── npu_engine_overlap.hip      # 3-buffer pipeline (1148 lines)
├── npu_engine_spec.hip         # GPU speculative decode (489 lines)
├── gpu_kernels_fused.hip       # Shared GPU kernels + extern C wrappers
├── attn_stub.cpp               # CPU attention fallback
└── dequant_q4nx.cpp            # Q4NX weight dequantization
```
