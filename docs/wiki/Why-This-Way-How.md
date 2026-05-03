# Why this way + how — current stack walkthrough

Long-form explainer paired with the public docs. This page reflects the current Lemonade + FastFlowLM + `1bit-proxy` stack.

## The architecture in one picture

```
                 your apps
 GAIA · Open WebUI · Hermes · SDK clients · notebooks
                         |
                         | OpenAI-compatible HTTP
                         v
                 1bit-proxy :13306
                    |             |
                    |             `--> FastFlowLM :52625
                    |                  XDNA NPU chat, embeddings, opt-in ASR
                    |
                    `--> Lemonade :13305
                         canonical multimodal + OmniRouter server

                 native runtime boundary
       HIP kernels in rocm-cpp/ are C++20
       custom NPU kernels: IRON author-time -> MLIR-AIE -> Peano
       -> xclbin -> libxrt runtime dispatch
```

GAIA uses `http://127.0.0.1:13306/api/v1`. Generic OpenAI-compatible clients use `http://127.0.0.1:13306/v1`. Open WebUI is secondary and points at the same proxy on `/v1`.

## Why this way

### 1. One OpenAI-compatible front door

Apps should not care whether a request lands on Lemonade or FastFlowLM. `1bit-proxy :13306` keeps the common client surface stable while allowing backend routing to evolve.

### 2. Lemonade owns multimodal and OmniRouter behavior

Lemonade is the canonical local server on `:13305`. The proxy does not replace it; the proxy defaults to it and keeps direct Lemonade access available for workflows that need the exact Lemonade surface.

### 3. FastFlowLM is the live XDNA NPU lane

FastFlowLM serves supported FLM chat and embedding models through XRT on the NPU. It is the live NPU runtime lane today.

### 4. No Python in the core serving path

Training, notebooks, caller-side tools, Open WebUI, and IRON author-time DSL work are allowed outside the core runtime. The serving path stays native and Rule-A clean.

### 5. C++20 for kernels

HIP kernel work belongs in `rocm-cpp/`. Rust stays above the kernel layer. hipBLAS is not allowed in the runtime path.

### 6. NPU has two lanes

The live serving lane is FastFlowLM. The custom-kernel authoring lane is IRON Python DSL at author time, then MLIR-AIE, Peano, `xclbin`, and native `libxrt` runtime dispatch.

## How a request flows

1. A client sends `POST /v1/chat/completions` to `http://127.0.0.1:13306/v1`.
2. `1bit-proxy` normalizes the OpenAI-compatible request.
3. Default requests go to Lemonade on `:13305`.
4. Targeted FLM model families go to FastFlowLM on `:52625`.
5. The selected backend streams or returns the response.
6. The proxy sends the OpenAI-compatible response back to the client.

## How to verify

```bash
1bit status
curl -s http://127.0.0.1:13306/v1/models
curl -s http://127.0.0.1:13305/api/v1/models
curl -s http://127.0.0.1:52625/v1/models
```

Use `local-no-auth` when a local client requires an API key.

## How to contribute

See [CONTRIBUTING.md](../../CONTRIBUTING.md). Keep docs and changes aligned with [Development](./Development.md), especially Rules A-E and the current endpoint map.
