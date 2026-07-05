# Engine Status — July 5, 2026

## Production: C++23 Daemon → FLM Proxy (94 tok/s) ✅

`npu-gpu-cpud` — C++23 zero-dep binary (110 KB). Proxies to FLM on port 52625.

- **Port**: 9090 | **Model**: Qwen3-0.6B | **TTFT**: 529 ms | **Decode**: 94.4 tok/s
- **Output**: Verified coherent | **Zero Python** in runtime path

## C++ Universal Engine (28 tok/s) ✅

Auto-detects 5 models (NPU) + 22 multi-modal models. Coherence bug **FIXED** — root cause was missing AIE micro-tiling in xclbin generator (see [GEMM-KERNEL-CORRECTNESS-CONFIRMED.md](../docs/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md)). After
8 rounds of host-side math fixes. The fused xclbin path is validated correct
(max_abs=0.0078 vs CPU oracle) but runs at 4 tok/s.

## GPU ZINC (22 tok/s) ✅ | GPU llama.cpp 1-bit (70-381 tok/s) ✅

---

*BENCHMARKS.md for full numbers. Daemon: `daemon/npu-gpu-cpud.cpp`.*
