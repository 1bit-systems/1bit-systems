# Development

This page is the review baseline. It reflects the current 1bit-systems stack:

```text
GAIA / OpenAI clients / Open WebUI
  -> 1bit-proxy :13306
      -> Lemonade :13305     canonical multimodal + OmniRouter surface
      -> FastFlowLM :52625   XDNA NPU chat + embeddings lane
```

Lemonade and FastFlowLM are the inference engines. `1bit-proxy` is the union endpoint. GAIA is the primary control surface; Open WebUI is a secondary compatibility UI pointed at the proxy.

## The Five Rules

Nobody gets through review without observing them.

**Rule A — Core serving stays Python-free.** Training is fine. Notebooks are fine. Build-time conversion is fine. Author-time IRON is fine. The core engine path we own is a Python-free zone: proxy, kernels, native runtimes, and model hot paths. Compatibility UIs and caller-side tools, including Open WebUI while we still ship it, are allowed only as isolated clients behind the OpenAI-compatible endpoint; they do not get to become the engine.

**Rule B — C++20 for kernels.** All HIP code lives in `rocm-cpp/`. Do not port kernels to Rust because "Rust is better." Rust is fine; it is not the right tool for the kernel layer.

**Rule C — hipBLAS is banned in the runtime path.** If you reach for hipBLAS, step back and port the kernel to `rocm-cpp/` instead.

**Rule D — Rust 1.88+, edition 2024.** Bump with a reason.

**Rule E — NPU has two lanes.** The live serving lane is FastFlowLM on XDNA through XRT. The custom-kernel authoring lane is IRON Python DSL at author-time, then MLIR-AIE, Peano, `xclbin`, and `libxrt` from C++ at runtime. Runtime surface stays Rule A-clean except for explicit caller/UI carve-outs.

## Current Operational Baseline

- Lemonade: `http://127.0.0.1:13305/api/v1` or `/v1`
- FastFlowLM: `http://127.0.0.1:52625/v1`
- Union endpoint: `http://127.0.0.1:13306/v1` and `http://127.0.0.1:13306/api/v1`
- Open WebUI: secondary UI on `http://127.0.0.1:3000`, pointed at the union endpoint
- GAIA: primary agent/control surface, also pointed at the union endpoint
