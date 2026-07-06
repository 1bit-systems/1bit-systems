---
type: Reference
title: Fused Engine Dispatch Policies
description: 8 dispatch policies for per-layer routing between NPU (XRT INT8 GEMM) and GPU (Vulkan flash attention/DMMV).
tags: [fusion, dispatch, policies, npu, gpu]
timestamp: 2026-07-06T00:00:00Z
---

# Overview

The Fused Engine dispatcher (`engine/fusion/dispatcher.zig`) supports 8 dispatch policies, controlled via `--policy <name>`.

## Policy Reference

| Policy | Attention | FFN | QKV | Use Case |
|--------|-----------|-----|-----|----------|
| `auto` | GPU | NPU | NPU | **Best throughput** — GPU flash attention + NPU INT8 GEMM |
| `npu_only` | NPU | NPU | NPU | NPU-only mode (models without GPU flash attention kernels) |
| `gpu_only` | GPU | GPU | GPU | GPU-only fallback (when NPU xclbins unavailable) |
| `attention_on_npu` | NPU | GPU | GPU | NPU edge_attention kernel + GPU DMMV |
| `ffn_on_npu` | GPU | NPU | GPU | GPU flash attention + NPU INT8 GEMM (FFN-heavy models) |
| `qkv_on_npu` | GPU | GPU | NPU | QKV projection on NPU, rest on GPU |
| `layer_by_layer` | Per-layer | Per-layer | Per-layer | Benchmark each layer's operations and choose |
| `prefill_npu_decode_gpu` | NPU(prefill) | GPU(decode) | NPU(prefill) | Fast prefill on NPU, batch decode on GPU |

## How It Works

At each layer, the dispatcher examines the operation type (Attention, FFN, QKV projection) and routes it to the appropriate backend based on the active policy.

- **NPU path**: XRT xclbin INT8 GEMM via `engine/npu/`
- **GPU path**: Vulkan flash attention / DMMV via `engine/gpu/`
- **KV cache**: Shared across both via the unified scheduler

## Cross-Backend Memory

- **dma-buf** (preferred): Direct NPU BO ↔ GPU buffer sharing
- **Staging copy** (fallback): Through host memory

## Citations

[1] [Fused Engine Architecture](/engines/fused_engine.md)
[2] [Unified KV Cache](/structures/kv_cache.md)
