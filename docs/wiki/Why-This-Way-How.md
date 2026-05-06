# Why this way + how — repair path walkthrough

Long-form explainer paired with the public docs. This page reflects the
toolbox-first repair path and the intended Lemonade + FastFlowLM +
`1bit-proxy` product direction.

## The architecture in one picture

```
                 your apps
 GAIA · Open WebUI · Hermes · SDK clients · notebooks
                         |
                         | OpenAI-compatible HTTP
                         v
                 1bit-proxy :13306
                    |             |
                    |             `--> optional FastFlowLM :52625
                    |                  XDNA NPU chat, embeddings, opt-in ASR
                    |
                    `--> toolbox llama-server or Lemonade :13305
                         first repaired GGUF backend or native multimodal lane

                 native runtime boundary
       HIP kernels in rocm-cpp/ are C++20
       custom NPU kernels: IRON author-time -> MLIR-AIE -> Peano
       -> xclbin -> libxrt runtime dispatch
```

GAIA uses `http://127.0.0.1:13306/api/v1`. Generic OpenAI-compatible clients
use `http://127.0.0.1:13306/v1`. Open WebUI is secondary and points at the same
proxy on `/v1`.

## Why this way

### 1. One OpenAI-compatible front door

Apps should not care whether a request lands on toolbox llama.cpp, native
Lemonade, or optional FastFlowLM. `1bit-proxy :13306` keeps the common client
surface stable while allowing backend routing to evolve.

### 2. Toolboxes restore serving first

The Strix Halo llama.cpp toolboxes are the practical repair path. Use
`vulkan-radv` first because it is the most compatible backend, then test
`rocm-7.2.2` after `/dev/dri` and `/dev/kfd` are visible.

### 3. Lemonade owns multimodal and OmniRouter behavior

Lemonade is the native multimodal/OmniRouter lane on `:13305`. The proxy does
not replace it; during repair, toolbox `llama-server` can occupy the same
backend port while Lemonade is brought back under the eventual control plane.

### 4. FastFlowLM is the intended XDNA NPU lane

FastFlowLM serves supported FLM chat and embedding models through XRT on the
NPU when the native host stack is healthy. It is not the first bootstrap
dependency for Ubuntu/Fedora.

### 5. No Python in the core serving path we own

Training, notebooks, caller-side tools, Open WebUI, toolboxes, and IRON
author-time DSL work are allowed outside the core runtime. The serving path we
own stays native and Rule-A clean.

### 6. C++20 for kernels

HIP kernel work belongs in `rocm-cpp/`. Rust stays above the kernel layer. hipBLAS is not allowed in the runtime path.

### 7. NPU has two lanes

The intended serving lane is FastFlowLM. The custom-kernel authoring lane is
IRON Python DSL at author time, then MLIR-AIE, Peano, `xclbin`, and native
`libxrt` runtime dispatch.

## How a request flows

1. A client sends `POST /v1/chat/completions` to `http://127.0.0.1:13306/v1`.
2. `1bit-proxy` normalizes the OpenAI-compatible request.
3. Default requests go to the active backend on `:13305`.
4. Targeted FLM model families can go to FastFlowLM on `:52625` if enabled.
5. The selected backend streams or returns the response.
6. The proxy sends the OpenAI-compatible response back to the client.

## How to verify

```bash
1bit status
curl -s http://127.0.0.1:13306/v1/models
curl -s http://127.0.0.1:13305/v1/models
curl -s http://127.0.0.1:52625/v1/models
```

Use `local-no-auth` when a local client requires an API key.

## How to contribute

See [CONTRIBUTING.md](../../CONTRIBUTING.md). Keep docs and changes aligned with [Development](./Development.md), especially Rules A-E and the current endpoint map.
