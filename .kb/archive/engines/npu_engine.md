---
type: Engine
title: "[RETIRED] NPU Inference Engine (Fused Layer)"
description: Production inference engine on AMD NPU via XRT. 291 tok/s Qwen3-0.6B, 3.4 ms/tok decode. Fused layer: one xclbin call per transformer layer.
tags: [npu, production, fused, xrt, int8, q4nx, retired]
resource: https://github.com/1bit-systems/1bit/tree/main/engine/npu/
timestamp: 2026-07-06T00:00:00Z
---

> **RETIRED (2026-07-07).** `engine/npu/` was deleted in commit `cd232a091`.
> Its dispatch logic (4-xclbin INT8 GEMM pattern, RMSNorm/RoPE/attention/SwiGLU)
> was ported forward into `spec-decode/engine/npu_target_model.h`
> (`NPUQwen3Target`), which is the current NPU dispatch path. Note both the
> "291 tok/s" (fused layer) and "97 tok/s" (v12) figures below were, per
> `docs/wiki/performance.md`, **raw throughput with output not yet coherent**
> — never a validated/production number, contrary to how this doc frames it.
> Kept for historical reverse-engineering value only.

# Overview

The NPU fused layer engine is the **production inference engine** for 1bit.systems, running on the AMD NPU via XRT (Xilinx Runtime). The fused layer runs the full transformer in one xclbin call (QKV→attention→O→GU→SiLU→D on NPU, no CPU attention), achieving **291 tok/s** on Qwen3-0.6B at 3.4 ms/tok decode — 3× the v12 baseline.

## Key Facts

- **Binary size**: 38 KB — zero external dependencies
- **Fused layer**: one xclbin call per transformer layer (was 4 calls/layer in v12)
- **Auto-detects 5 models** from a single binary
- **All-models mode**: 28 tok/s (C++ all-5)
- **C++ v12 fallback**: 97 tok/s, 10.3 ms/tok
- **FLM fallback v2**: 94 tok/s, 10.6 ms/tok — uses AMD's proprietary runtime
- **INT8 xclbins**: `/home/bcloud/npu-sandbox/npu-infer/build/int8/`
- **Model path**: `~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx`
- **NPU2 firmware**: Supports 8+ simultaneous hw_contexts (firmware 1.1.2.65)

## Engine Evolution

244 → 10 ms/tok in 4 days (24× speedup):

| Version | ms/tok | Strategy |
|---------|--------|----------|
| v7 BFP16 | 1930 | Initial BFP16 prototype |
| i8 swap | 244 | Switched to INT8 quantization |
| v6 batch-4 | 50 | Batch-4 dispatch amortization |
| v9 M=16 | 16 | Batch-16 dispatch amortization |
| v12 M=32 | 10 | Batch-32 — production. Matches FLM on decode. |

> C++ v12's advantage over FLM is per-request TTFT (fused xclbin eliminates per-layer ioctl). FLM's advantage is single-request decode latency.

## Source Layout

- `engine/npu/src/npu_engine_cb.cpp` — Main loop (batched prefill + decode)
- `engine/npu/src/dequant_q4nx.c` — Q4NX dequantizer
- `engine/npu/kernel/edge_attention.cc` — NPU attention (Chess C++)
- `engine/npu/xclbins/n1_core_i8_v2.py` — INT8 MLIR generator

## Build

```bash
g++ -std=c++23 -O3 -o npu_engine engine/npu/src/npu_engine_cb.cpp \
  engine/npu/build/dequant_q4nx.o \
  -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl
```

## Verification

```bash
curl -s http://127.0.0.1:9090/v1/chat/completions \
  -d '{"model":"qwen3:0.6b","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'
```

Expect C++ v12 engine to respond at ~97 tok/s. Falls back to FLM at ~94 tok/s.

## Citations

[1] [Engine bench deep-dive](https://github.com/1bit-systems/1bit/tree/main/engine/npu/BENCHMARKS.md)
[2] [Audit trail](https://github.com/1bit-systems/1bit/blob/main/docs/journey.md)
[3] [Performance Benchmarks](/references/performance_benchmarks.md)
