# NPU Status

The title is historical, but the nuance matters: NPU work exists, while the
out-of-box repair path is GPU-backed toolbox inference first.

## Current State

FastFlowLM is the intended XDNA NPU serving lane when the native host stack is
healthy:

```text
FastFlowLM :52625/v1
1bit-proxy :13306/v1 -> FastFlowLM for known FLM model families
```

On the native CachyOS stack, XRT sees the Strix Halo NPU as `RyzenAI-npu5`, and
`/dev/accel/accel0` is present. That is not the universal bootstrap path for
Ubuntu/Fedora repair. Start with the Strix Halo llama.cpp toolboxes, verify
`/dev/dri`, then test `/dev/kfd` and ROCm before treating the NPU lane as
available.

## What Changed

The older plan said the NPU lane was deferred or pivoting through ONNX Runtime
+ VitisAI EP. The current product direction is FastFlowLM for XDNA, but the
first repaired serving backend is toolbox llama.cpp on the iGPU.

Current split:

| Lane | Status | Runtime |
|---|---|---|
| Toolbox llama.cpp serving | First repair path | Vulkan RADV first, ROCm 7.2.2 after device checks |
| FastFlowLM serving | Native optional path | FastFlowLM + XRT / XDNA |
| Custom AIE kernels | Authoring path | IRON at author-time -> MLIR-AIE -> Peano -> xclbin -> libxrt C++ runtime |
| ORT + VitisAI EP | Historical evaluation | Not the live default |

## Rule A Position

IRON is allowed as an author-time Python DSL. Python does not enter the custom NPU runtime surface. The runtime artifact is an `xclbin`, loaded through `libxrt` from C++/native code.

FastFlowLM is a native serving lane. The proxy can route to it as a local
OpenAI-compatible backend after the service is actually healthy.

## What NPU Is Good For

Decode remains memory-bandwidth-bound on Strix Halo, so the NPU is not automatically better for every token path. The useful NPU work is:

- FLM chat models that FastFlowLM already supports.
- Embeddings exposed by the FLM / Lemonade model set.
- Future custom prefill or matmul-heavy kernels authored through IRON / MLIR-AIE / Peano.

## Historical Evaluations

Earlier notes compared ONNX Runtime + VitisAI EP, FastFlowLM, IREE-AIE, and
direct XRT. Keep those as context, but do not treat them as the current stack
definition. The current repair path is toolbox llama.cpp behind `1bit-proxy`,
with Lemonade and FastFlowLM kept as native/target lanes.
