---
type: Engine
title: Unified Serving Daemon
description: Small binary serving all models via a HTTP API (OpenAI-compatible), proxying through fused NPU layer (291 tok/s), C++ v12 (97 tok/s), or FLM fallback.
tags: [daemon, serving, api, openai-compatible, http]
resource: https://github.com/1bit-systems/1bit/tree/main/npu-gpu-cpu/
timestamp: 2026-07-06T00:00:00Z
---

# Overview

The unified daemon (`npu-gpu-cpu/`) serves all inference engines through a single HTTP API. Auto-detects 5 models and routes requests to the optimal backend.

## Architecture

```
Client ──▶ HTTP API (:9090)
              │
              ▼
        ┌─────────────┐
        │   Router    │
        ├─────────────┤
        │ Fused layer │──▶ Production path (291 tok/s)
        │ C++ v12 NPU │──▶ Fallback path (97 tok/s)
        │ FLM proxy   │──▶ Fallback v2 (94 tok/s)
        │ FLM proxy   │──▶ Fallback v2 (94 tok/s)
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

The 38 KB binary auto-detects all 5 models from the filesystem without configuration.

## Deployment

- **Packaging**: `packaging/` (deb + snap + tarball + docker + ollama)
- **CI**: Pre-commit `/verify` check required before engine changes
- **Release**: `gh release create` + upload deb/snap/tarball + tag vYYYY.MM.DD

## Citations

[1] [NPU Engine](/engines/npu_engine.md)
[2] [Fused Engine](/engines/fused_engine.md)
