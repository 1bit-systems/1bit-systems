---
type: Engine
title: Unified Serving Daemon
description: Single 74 KB binary serving all models via a HTTP API (OpenAI-compatible), proxying through C++ v12 (NPU) or FLM fallback.
tags: [daemon, serving, api, openai-compatible, http]
resource: https://github.com/1bit-systems/1bit/tree/main/npu-gpu-cpu/
timestamp: 2026-07-06T00:00:00Z
---

# Overview

The unified daemon (`npu-gpu-cpu/`) serves all inference engines through a single HTTP API. A 74 KB binary that auto-detects 5 models and routes requests to the optimal backend.

## Architecture

```
Client ──▶ HTTP API (:9090)
              │
              ▼
        ┌─────────────┐
        │   Router    │
        ├─────────────┤
        │ C++ v12 NPU │──▶ Production path (97 tok/s)
        │ FLM proxy   │──▶ Fallback path (94 tok/s)
        │ GPU backend │──▶ Vulkan/ROCm path
        └─────────────┘
              │
              ▼
        ┌─────────────┐
        │ KV Cache    │── Shared across all backends
        └─────────────┘
```

## API

OpenAI-compatible chat completions endpoint:

```bash
curl -s http://127.0.0.1:9090/v1/chat/completions \
  -d '{
    "model": "qwen3:0.6b",
    "messages": [{"role":"user","content":"Hello"}],
    "max_tokens": 100
  }'
```

## Models Auto-Detected

The 74 KB binary auto-detects all 5 models from the filesystem without configuration.

## Deployment

- **Packaging**: `packaging/` (deb + snap + tarball + docker + ollama)
- **CI**: Pre-commit `/verify` check required before engine changes
- **Release**: `gh release create` + upload deb/snap/tarball + tag vYYYY.MM.DD

## Citations

[1] [NPU Engine](/engines/npu_engine.md)
[2] [Fused Engine](/engines/fused_engine.md)
