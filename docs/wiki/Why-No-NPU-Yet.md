# NPU Status

The title is historical. The NPU is no longer "not yet" for the current stack.

## Current State

FastFlowLM is the live XDNA NPU serving lane:

```text
FastFlowLM :52625/v1
1bit-proxy :13306/v1 -> FastFlowLM for known FLM model families
```

On the current CachyOS stack, XRT sees the Strix Halo NPU as `RyzenAI-npu5`, and `/dev/accel/accel0` is present. The kernel path is the in-tree `amdxdna` driver on the running CachyOS kernel, with XRT userspace on top.

## What Changed

The older plan said the NPU lane was deferred or pivoting through ONNX Runtime + VitisAI EP. That is no longer the operational baseline.

Current split:

| Lane | Status | Runtime |
|---|---|---|
| FastFlowLM serving | Live | FastFlowLM + XRT / XDNA |
| Custom AIE kernels | Authoring path | IRON at author-time -> MLIR-AIE -> Peano -> xclbin -> libxrt C++ runtime |
| ORT + VitisAI EP | Historical evaluation | Not the live default |

## Rule A Position

IRON is allowed as an author-time Python DSL. Python does not enter the custom NPU runtime surface. The runtime artifact is an `xclbin`, loaded through `libxrt` from C++/native code.

FastFlowLM is a native serving lane. The proxy routes to it as a local OpenAI-compatible backend.

## What NPU Is Good For

Decode remains memory-bandwidth-bound on Strix Halo, so the NPU is not automatically better for every token path. The useful NPU work is:

- FLM chat models that FastFlowLM already supports.
- Embeddings exposed by the FLM / Lemonade model set.
- Future custom prefill or matmul-heavy kernels authored through IRON / MLIR-AIE / Peano.

## Historical Evaluations

Earlier notes compared ONNX Runtime + VitisAI EP, FastFlowLM, IREE-AIE, and direct XRT. Keep those as context, but do not treat them as the current stack definition. The current stack is Lemonade + FastFlowLM behind `1bit-proxy`.
