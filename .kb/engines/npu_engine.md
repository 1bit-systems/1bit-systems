---
type: Engine
title: NPU Inference Engine (C++ v12)
description: Production inference engine on AMD NPU via XRT. 97 tok/s Qwen3-0.6B, 10.3 ms/tok decode.
tags: [npu, production, cpp, xrt, int8, q4nx]
resource: https://github.com/1bit-systems/1bit/tree/main/engine/npu/
timestamp: 2026-07-06T00:00:00Z
---

# Overview

The NPU engine is the **production inference engine** for 1bit.systems, running on the AMD NPU via XRT (Xilinx Runtime). C++ v12 achieves **97 tok/s** on Qwen3-0.6B at 10.3 ms/tok decode — matching and exceeding the AMD FLM runtime on decode through batched dispatch amortization.

## Key Facts

- **Binary size**: 74 KB (73 KB actual) — zero external dependencies
- **Auto-detects 5 models** from a single binary
- **All-models mode**: 28 tok/s (C++ all-5)
- **FLM fallback**: 94 tok/s, 10.6 ms/tok — uses AMD's proprietary runtime
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
