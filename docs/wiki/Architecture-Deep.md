# Architecture Deep

This page is the current architecture reference. Older drafts that mention `1bit-server :8180`, `1bit-lemonade :8200`, or gen-1/gen-2 shadow routing are historical and should not be used as live stack guidance.

## Runtime Topology

```
GAIA / Open WebUI / SDK clients / mesh clients
        |
        v
1bit-proxy :13306
   |-- Lemonade :13305
   |     canonical multimodal OpenAI-compatible server
   |     OmniRouter behavior
   |
   `-- FastFlowLM :52625
         XDNA NPU chat
         embeddings
         opt-in ASR
```

GAIA uses `http://127.0.0.1:13306/api/v1`. Generic OpenAI-compatible clients use `http://127.0.0.1:13306/v1`. Open WebUI is a secondary UI on `:3000` and points at the proxy.

## Service Map

| Service | Bind | Role |
|---|---:|---|
| `lemond` | `:13305` | Lemonade Server, canonical multimodal and OmniRouter surface |
| `flm` | `:52625` | FastFlowLM XDNA NPU runtime |
| `1bit-proxy` | `:13306` | Union OpenAI-compatible endpoint |
| `open-webui` | `:3000` | Secondary browser UI/client |
| GAIA | dynamic | Primary UI/control surface |

## Request Flow

1. Client sends OpenAI-compatible request to `1bit-proxy`.
2. Proxy keeps Lemonade as the default route.
3. Proxy routes targeted FLM model families to FastFlowLM.
4. Backend returns standard OpenAI-compatible response or stream.
5. Client sees one stable base URL.

## Kernel Boundary

GPU kernels live in `rocm-cpp/` and use C++20 HIP. Rust does not own the kernel layer. hipBLAS is banned in the runtime path; if a runtime op needs BLAS-like behavior, port or author the kernel in `rocm-cpp/`.

## NPU Boundary

There are two NPU lanes:

- **Live serving lane:** FastFlowLM on XDNA/XRT.
- **Custom authoring lane:** IRON Python DSL at author time -> MLIR-AIE -> Peano -> `xclbin` -> native `libxrt` runtime dispatch.

The runtime surface stays Rule-A clean except explicit caller/UI carve-outs.

## Operational Checks

```bash
1bit status
curl -s http://127.0.0.1:13306/v1/models
curl -s http://127.0.0.1:13305/api/v1/models
curl -s http://127.0.0.1:52625/v1/models
```

Use `local-no-auth` for local clients that require an API key.

## Rules

The review rules live in [Development](./Development.md). This architecture follows the same five-rule statement:

- Rule A: no Python in the core serving path.
- Rule B: C++20 for kernels; HIP code in `rocm-cpp/`.
- Rule C: hipBLAS banned in runtime.
- Rule D: Rust 1.88+, edition 2024.
- Rule E: FastFlowLM live NPU serving lane; IRON author-time lane for custom kernels.
