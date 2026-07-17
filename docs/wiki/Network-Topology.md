# Network Topology

> This page exists because issue #1 linked to it before it was written. It
> describes how the pieces talk to each other at runtime.

## Components

```
 client (OpenAI-compatible)
        │  POST /v1/chat/completions
        ▼
 ┌─────────────────────────┐
 │ zaya_server / daemon     │  HTTP server (default 127.0.0.1:8088)
 │ (daemon/npu-gpu-cpud.py) │  routes each request to a backend
 └───────────┬─────────────┘
             │ selects backend per policy
   ┌─────────┼───────────────┬───────────────┐
   ▼         ▼               ▼               ▼
  NPU       GPU             CPU        fused NPU+GPU
 (XDNA2)  (ROCm/Vulkan)   (fallback)  (engine/fusion)
```

## Ports & endpoints

| Endpoint                     | Default            | Purpose                          |
|------------------------------|--------------------|----------------------------------|
| `/v1/chat/completions`       | `127.0.0.1:8088`   | OpenAI-compatible chat API       |
| `/v1/models`                 | `127.0.0.1:8088`   | list available models            |
| `/health`                    | `127.0.0.1:8088`   | liveness check                   |

By default the server binds to **loopback only** (`127.0.0.1`). Expose it on a
LAN only behind a reverse proxy you control; there is no built-in auth.

## Backend selection

The daemon picks a backend per request. See the router discussion in the repo
root docs and `tools/token_router.cpp` / `unified-router.py`. Note the routing
layer is still being consolidated — see the state-of-the-stack notes and the
open router issues.

## See also

- [Installation](Installation.md)
- [Getting Started](../getting-started.md)
