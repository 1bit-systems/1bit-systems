# Strix Halo — Full Benchmark Results

**Date:** 2026-07-08  
**Hardware:** AMD Ryzen AI Strix Halo — Radeon 8060S Graphics (gfx1151)  
**Driver:** ROCm 7.2.4 / Mesa 26.0.3 RADV  
**Model:** Qwen3-0.6B (Q4NX, I8 packed weights)

---

## 🔥 54.74 TFLOPS — Full Matrix-Core Peak Confirmed

Raw WMMA intrinsic (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`):

```
65536 ops  │   7.54 ms │  54.66 TFLOPS
262144 ops │  30.09 ms │  54.82 TFLOPS
1048576 ops│ 120.46 ms │  54.76 TFLOPS
```

**Average: 54.74 TFLOPS** — Radeon 8060S spec confirmed (96 CUs × 8 waves/CU).

---

## 📊 Throughput Comparison

| Mode | TFLOPS | Units | Notes |
|------|--------|-------|-------|
| WMMA matrix-core | **54.74** | FP16 matrix | Raw intrinsic, full saturation |
| rocBLAS GemmEx | 7.90 | FP16 matrix | Library path, not fully tuned |
| rocBLAS Sgemv | 3.14 | FP32 vector | Vector units, BW-bound |

---

## 🚀 Engine Performance (M=1 Decode)

| Engine | tok/s | Weight Format | Activations | Status |
|--------|:-----:|:-------------:|:-----------:|:------:|
| engine_final | **43** | FP32 pre-deq | FP32 | Production |
| engine_peak | **75** | I8 packed | FP16 | I8 inline deq + fused QKV/GateUp |

### Per-Kernel Timings (I8 GEMV)

| Kernel | Shape | Time |
|--------|-------|:----:|
| Q_proj | 2048×1024 | 0.003 ms |
| K_proj | 1024×1024 | 0.003 ms |
| V_proj | 1024×1024 | 0.003 ms |
| O_proj | 1024×2048 | 0.003 ms |
| Gate | 3072×1024 | 0.003 ms |
| Up | 3072×1024 | 0.003 ms |
| Down | 1024×3072 | 0.004 ms |

### Decode Scaling

| Tokens | tok/s | ms/tok |
|:------:|:-----:|:------:|
| 16 | 57 | 17.47 |
| 32 | 70 | 14.36 |
| 64 | **75** | **13.33** |
| 128 | 73 | 13.74 |
| 256 | 63 | 15.87 |

---

## 🔧 Build & Run

```bash
cd engine/fusion
./build_peak.sh                    # Build engines
LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib ./bench_all.sh    # Full suite
LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib ./wmma_peak        # Matrix-core peak
LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib ./engine_peak -m model.q4nx -n 128  # Decode
```
