# Lemonade Compatibility

Lemonade is no longer an external thing we point at `1bit-halo-server`. It is
the native multimodal/OmniRouter lane. During the toolbox-first repair path,
`llama-server` may occupy the same `:13305` backend slot.

```text
1bit-proxy :13306
  -> toolbox llama-server :13305  first repaired GGUF backend
  -> Lemonade :13305              native multimodal / OmniRouter lane
  -> FastFlowLM :52625            optional XDNA NPU chat and embeddings
```

## Current Role

Lemonade is the intended OpenAI-compatible multimodal and OmniRouter surface.
It owns direct multimodal workflows and model management for the native
Lemonade lane.

Use Lemonade direct when you specifically want Lemonade behavior with no union routing:

```text
http://127.0.0.1:13305/api/v1
http://127.0.0.1:13305/v1
```

Use the proxy when apps should keep one base URL while the backend is toolbox
llama.cpp, Lemonade, or optional FastFlowLM:

```text
http://127.0.0.1:13306/v1
http://127.0.0.1:13306/api/v1
```

## App Guidance

- GAIA: point at `http://127.0.0.1:13306/api/v1`.
- Open WebUI: point at `http://127.0.0.1:13306/v1`.
- Active toolbox backend: point at `http://127.0.0.1:13305/v1`.
- Direct Lemonade workflows on the native path: point at `http://127.0.0.1:13305/api/v1`.
- FastFlowLM debugging, if enabled: point at `http://127.0.0.1:52625/v1`.

If a client asks for a key, use `local-no-auth` unless Lemonade authentication has been explicitly configured.

## Rule A Position

Lemonade is part of the intended native inference stack. Open WebUI, toolboxes,
and SDK examples are backend/caller/UI surfaces, not the core control plane.
The core code we own remains Python-free: proxy, kernels, native runtime path,
and custom NPU dispatch.

## Historical Note

Older wiki drafts described `1bit-halo-server :8180` as the LLM backend that
Lemonade would call. That is stale. The current repair path keeps
`1bit-proxy :13306` as the app surface and lets the active backend on `:13305`
be toolbox `llama-server` first or Lemonade on the native path.
