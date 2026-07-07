---
type: Structure
title: "[RETIRED] Unified KV Cache (H2O Eviction)"
description: Backend-agnostic KV cache with H2O eviction, RadixAttention prefix sharing, CPU offloading, and dynamic quantization profiles.
tags: [kv-cache, h2o, radix-attention, scheduler, zig, retired]
resource: https://github.com/1bit-systems/1bit/tree/main/engine/gpu/src/scheduler/
timestamp: 2026-07-06T00:00:00Z
---

> **RETIRED (2026-07-07).** `engine/gpu/src/scheduler/` was deleted in commit
> `cd232a091`. The current NPU target (`spec-decode/engine/npu_target_model.h`,
> class `NPUQwen3Target`) uses a much simpler per-layer `std::vector` KV cache
> with no eviction, no RadixAttention prefix sharing, no CPU offload, and no
> dynamic per-layer quantization profiles — none of the machinery described
> below exists in the current codebase. Kept for historical reverse-
> engineering value only.

# Overview

The unified KV cache layer (`engine/gpu/src/scheduler/`) is shared across all three inference paths (NPU, GPU, CPU). It provides a backend-agnostic infrastructure for managing key-value cache memory efficiently.

## Components

### kv_cache.zig — H2O Eviction

Cumulative attention scoring for eviction decisions. Uses a **min-heap** to track the lowest-scoring pages and a **zero-page technique** that remaps evicted pages to a reserved zero-filled page — naturally gets ~0 attention, requires no shader changes.

Three eviction policies (set via `ZINC_KV_EVICTION_POLICY` env var):

| Policy | Description |
|--------|-------------|
| `h2o_attention_score` | Evict pages with lowest cumulative attention scores (default, best quality) |
| `lru` | Evict least recently used pages |
| `fifo` | Evict oldest pages first |

### radix_tree.zig — RadixAttention

Prefix tree for cross-request KV page sharing. When multiple requests share a common prefix (e.g., system prompt), the KV cache for that prefix is computed once and shared. Dramatically reduces TTFT for requests with shared prefixes.

### offload_engine.zig — CPU Offloading

Cold KV pages are evicted to CPU memory. Pages that haven't been accessed recently are moved off the NPU/GPU to free space for active requests. Pages are brought back on demand.

### quant_profile.zig — Dynamic Quantization

Per-layer dynamic quantization scheme selection from 8 schemes. Each layer can use a different quantization format based on its sensitivity to precision loss.

### request.zig + scheduler.zig — Lifecycle & Batching

Continuous batching with page allocation. Requests are scheduled based on available KV cache pages, with fairness and priority support.

## Cross-Backend Sharing

The KV cache is shared across NPU and GPU via:

- **dma-buf** (preferred): Direct memory sharing between NPU BO and GPU buffers
- **Staging copy** (fallback): Copy through host memory when dma-buf is unavailable

## Citations

[1] [Fused Engine Interop](/engines/fused_engine.md)
[2] [Original paper: H2O — Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048)
