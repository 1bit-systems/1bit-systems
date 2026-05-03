# Lemonade Compatibility

Lemonade is no longer an external thing we point at `1bit-halo-server`. It is one of the two live inference engines in the stack.

```text
1bit-proxy :13306
  -> Lemonade :13305     default route, multimodal, OmniRouter
  -> FastFlowLM :52625   XDNA NPU chat and embeddings
```

## Current Role

Lemonade is the canonical OpenAI-compatible multimodal and OmniRouter surface. It owns direct multimodal workflows and model management for the Lemonade lane.

Use Lemonade direct when you specifically want Lemonade behavior with no union routing:

```text
http://127.0.0.1:13305/api/v1
http://127.0.0.1:13305/v1
```

Use the proxy when apps should see Lemonade and FastFlowLM behind one base URL:

```text
http://127.0.0.1:13306/v1
http://127.0.0.1:13306/api/v1
```

## App Guidance

- GAIA: point at `http://127.0.0.1:13306/api/v1`.
- Open WebUI: point at `http://127.0.0.1:13306/v1`.
- Direct Lemonade workflows: point at `http://127.0.0.1:13305/api/v1`.
- FastFlowLM debugging: point at `http://127.0.0.1:52625/v1`.

If a client asks for a key, use `local-no-auth` unless Lemonade authentication has been explicitly configured.

## Rule A Position

Lemonade is part of the current inference stack. Open WebUI and SDK examples are caller/UI surfaces, not the core engine. The core code we own remains Python-free: proxy, kernels, native runtime path, and custom NPU dispatch.

## Historical Note

Older wiki drafts described `1bit-halo-server :8180` as the LLM backend that Lemonade would call. That is stale. The current stack inverts that relationship: Lemonade is a live local server on `:13305`, and `1bit-proxy` unifies it with FastFlowLM on `:13306`.
