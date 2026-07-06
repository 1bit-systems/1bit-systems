---
type: Engine
title: Fused Engine (NPU+GPU Hybrid)
description: Unified NPU+GPU hybrid inference engine dispatching per-layer to XRT (INT8 GEMM) or Vulkan (flash attention) through a single API.
tags: [npu, gpu, fusion, hybrid, zig, dispatcher]
resource: https://github.com/1bit-systems/1bit/tree/main/engine/fusion/
timestamp: 2026-07-06T00:00:00Z
---

# Overview

The FusedEngine (`engine/fusion/`) unifies the NPU (XRT xclbin INT8 GEMM) and GPU (Vulkan flash attention/DMMV) inference paths into one shared serving infrastructure. It dispatches per-layer or per-operation to the optimal backend through a single API.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  FusedEngine                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │ dispatcher │  │  memory    │  │  interop   │ │
│  │ .zig       │  │  .zig      │  │  .zig      │ │
│  ├────────────┤  ├────────────┤  ├────────────┤ │
│  │ 8 policies │  │ dma-buf    │  │ KV cache   │ │
│  │ auto-route │  │ staging cp │  │ sync bridge│ │
│  └────────────┘  └────────────┘  └────────────┘ │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │  server    │  │   main     │  │  KV cache  │ │
│  │  .zig      │  │   .zig     │  │  sched/    │ │
│  └────────────┘  └────────────┘  └────────────┘ │
└──────────────────────────────────────────────────┘
```

## Components

- `engine.zig` — Unified `FusedEngine` wrapping NPU + GPU backends
- `dispatcher.zig` — Layer-level dispatch policy (8 policies)
- `memory.zig` — Cross-backend memory sharing (dma-buf or staging copy)
- `interop.zig` — NPU↔GPU KV cache bridge (sync NPU BO↔GPU buffers)
- `server.zig` — Unified HTTP server (OpenAI-compatible API)
- `main.zig` — CLI entry point

## Build

```bash
cd engine/fusion && zig build -Doptimize=ReleaseFast
```

## Status

- ✅ Unified KV cache scheduler (KvPagePool, H2O eviction, RadixAttention, zero-page)
- ✅ FusedEngine interface
- ✅ Dispatcher with 8 dispatch policies
- ✅ Cross-backend memory (dma-buf + staging fallback)
- ✅ NPU↔GPU interop (KV cache sync bridge)
- ✅ Unified HTTP server (OpenAI-compatible API)
- ⬜ Fused prefix tree — RadixAttention shared across NPU+GPU paths
- ⬜ Dynamic policy switching — Runtime policy change via API
- ⬜ Auto-tuning — Benchmark each operation and choose fastest backend

## Citations

[1] [Dispatch Policies](/references/dispatch_policies.md)
[2] [Unified KV Cache](/structures/kv_cache.md)
