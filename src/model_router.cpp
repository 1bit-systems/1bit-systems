#include "model_router.h"

// ============================================================
// Backend Router — architecture-aware dispatch
// ============================================================
//
// The router selects one or more backends for a given model based on its
// architecture and format. Backends are tried in priority order: the first
// one that initializes successfully handles inference; subsequent backends
// serve as fallbacks if the primary fails.
//
// Routing hierarchy (as of 2026-07-20):
//
//   MoE (num_experts > 0)
//     └─ hip_gpu (CCA/MoE HIP kernels) + cpu_scalar
//         Zaya-style models with expert routing. The CCA/MoE kernels are
//         architecture-specific (shared memory layouts differ per model)
//         but the router picks the right kernel via the MoE config.
//
//   qwen3 architecture
//     ├─ npu_xrt (native NPU engine — INT8, single-core, ~12 tok/s)
//     ├─ npu_flm (FLM subprocess — fused xclbins, ~95 tok/s, fallback)
//     └─ cpu_generic
//         npu_xrt is the DEFAULT route since PR #567 (2026-07-20) once its
//         single-core GEMM kernels passed correctness verification against
//         the HuggingFace BF16 reference. FLM is kept as a faster fallback
//         until the multi-tile NPU path lands (see docs/mlir-air-integration.md).
//
//   GGUF / H1B format
//     └─ zinc_gpu (Vulkan ZINC runtime) + cpu_generic
//         The ZINC runtime handles multiple quant formats (Q4_0, Q4_K, etc.)
//         and architectures through its IR graph — no per-model specialization.
//
//   Everything else (fallthrough)
//     └─ hip_gpu + cpu_generic
//         Generic HIP GPU kernels cover any model the specific paths don't match.
//
// Adding a new route:
//   1. Add the architecture string to the detect logic in model_discovery.cpp
//   2. Add a new entry in the if-else chain below
//   3. Register the backend factory in backend_factory.cpp
//   4. Add a benchmark entry in bench/record.sh
// ============================================================

BackendRoute select_backend_route(const ModelConfig& cfg) {
    // Zaya-style MoE: any model with expert routing can use the CCA/MoE path
    if (cfg.num_experts > 0) {
        return {{"hip_gpu", "cpu_scalar"}, "MoE model — CCA/MoE kernel path"};
    }
    if (cfg.architecture == "qwen3") {
        return {{"npu_xrt", "npu_flm", "cpu_generic"}, "qwen3 architecture — native NPU engine, FLM subprocess as fallback"};
    }
    if (cfg.format == ModelFormat::GGUF || cfg.format == ModelFormat::H1B) {
        return {{"zinc_gpu", "cpu_generic"}, "GGUF/H1B model — ZINC GPU, generic CPU fallback"};
    }
    // Default: try HIP GPU first, fall back to generic CPU
    return {{"hip_gpu", "cpu_generic"}, "generic model — HIP GPU, generic CPU fallback"};
}
