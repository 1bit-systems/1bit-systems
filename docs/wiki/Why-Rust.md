# Why Rust Above, C++ Below?

**One-line answer**: kernels that win on specific hardware live in C++20/HIP because the compiler and intrinsic surface are there. Orchestration and local tooling use Rust 1.88+ edition 2024 where it fits. Current serving backends are Lemonade and FastFlowLM behind `1bit-proxy`.

## The Split

```
apps / UI / control
  GAIA, Open WebUI, SDK clients, Hermes
        |
        v
1bit-proxy :13306
  -> Lemonade :13305
  -> FastFlowLM :52625

kernel layer
  rocm-cpp/        C++20 HIP kernels
  NPU authoring    IRON -> MLIR-AIE -> Peano -> xclbin -> libxrt
```

## Why Kernels Stay C++20 / HIP

- AMD's mature GPU compiler path is HIP C++.
- The low-level intrinsic surface is exposed there first.
- Existing kernel work and review rules already live in `rocm-cpp/`.
- hipBLAS is banned in the runtime path, so missing ops become owned kernels, not library calls.

## Why Rust Still Matters

Rust is the default for native orchestration code we own when it is above the kernel layer: CLI helpers, typed config, local control tools, MCP surfaces, and small services that are not better handled by the existing Lemonade/FastFlowLM stack.

The floor is Rust 1.88+ with edition 2024. Bump it only with a reason.

## Why Not Python In Serving

Rule A. Python is allowed for training, notebooks, caller-side clients, and IRON author-time DSL work. The core serving path stays Python-free.

## Current Boundary

Do not add new docs or code that resurrect the old Rust gateway/server topology as the current runtime. The live path is:

```text
1bit-proxy :13306 -> Lemonade :13305
                   -> FastFlowLM :52625
```

See [Development](./Development.md) for the full rule statement.
