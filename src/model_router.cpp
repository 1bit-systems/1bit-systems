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
//     ├─ npu_xrt (native NPU engine — INT8, single-core)
//     └─ cpu_generic
//         npu_xrt is the sole NPU route since PR #567 (2026-07-20), once its
//         single-core GEMM kernels passed correctness verification against
//         the HuggingFace BF16 reference. The FastFlowLM subprocess fallback
//         (a proprietary AMD binary) was removed entirely — this project
//         ships zero proprietary code, not "zero by default."
//
//   zamba / mamba / zamba2 architecture (Mamba1 SSM + MoE or shared attn)
//     └─ hip_gpu (Mamba1/Mamba2 HIP kernels) + cpu_generic
//         Mamba1 models (Zamba-7B-v1, BlackMamba) use the kernels from
//         mamba1_engine.hip; Mamba2 models use mamba2_kernels.hip.
//         Backend selection is per-layer in the MoE case (BlackMamba:
//         alternating SSM layers and MoE FFN layers).
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
// Fallback behavior:
//   The caller (backend_manager.cpp) iterates the returned backend list in order.
//   Each backend tries to initialize (load model, allocate GPU buffers, etc.).
//   If init fails, the caller tries the next backend in the list.
//   If ALL backends fail, inference returns an error — there is no silent fallback
//   to CPU that produces wrong/empty output.
//
// Adding a new route:
//   1. Add the architecture string to the detect logic in model_discovery.cpp
//   2. Add a new entry in the if-else chain below
//   3. Register the backend factory in backend_factory.cpp
//   4. Add a benchmark entry in bench/record.sh
// ============================================================

BackendRoute select_backend_route(const ModelConfig& cfg) {
    // Zaya-style MoE: any model with expert routing that's NOT a Mamba1 MoE
    // (BlackMamba) uses the CCA/MoE kernel path.
    if (cfg.num_experts > 0 && cfg.arch != RCPP_ARCH_MAMBA && cfg.arch != RCPP_ARCH_LAGUNA) {
        return {{"hip_gpu", "cpu_scalar"}, "MoE model — CCA/MoE kernel path"};
    }

    // Laguna (poolside): sigmoid-routed MoE with hybrid SWA/global attention.
    // Uses the ZINC GPU backend for GGUF/1BP (general GPU kernels handle the
    // standard ops; softplus gate + token-choice MoE need the specialized path
    // from backend_laguna.cpp when available).
    if (cfg.arch == RCPP_ARCH_LAGUNA) {
        return {{"laguna_gpu", "zinc_gpu", "cpu_generic"},
                "Laguna model — specialized Laguna HIP backend, ZINC GPU fallback"};
    }
    // Mamba1 models (Zamba-7B-v1, BlackMamba): Mamba1 SSM HIP kernels,
    // with per-layer MoE expert dispatch for BlackMamba.
    if (cfg.arch == RCPP_ARCH_MAMBA || cfg.arch == RCPP_ARCH_ZAMBA) {
        return {{"mamba1_gpu", "cpu_generic"}, "Mamba1 model — Mamba1 HIP kernels, generic CPU fallback"};
    }
    if (cfg.architecture == "qwen3") {
        return {{"npu_xrt", "cpu_generic"}, "qwen3 architecture — native NPU engine, no proprietary fallback"};
    }
    if (cfg.format == ModelFormat::GGUF || cfg.format == ModelFormat::H1B) {
        return {{"zinc_gpu", "cpu_generic"}, "GGUF/H1B model — ZINC GPU, generic CPU fallback"};
    }
    // Default: try HIP GPU first, fall back to generic CPU
    return {{"hip_gpu", "cpu_generic"}, "generic model — HIP GPU, generic CPU fallback"};
}
  // Falcon (tiiuae) — parallel attention+ffn, MQA
  if (cfg.arch == RCPP_ARCH_FALCON) {
      return {{"hip_gpu", "cpu_generic"}, "Falcon — dense, parallel attn+ffn"};
  }
  // OLMo (AI2) — LayerNorm, no RoPE
  if (cfg.arch == RCPP_ARCH_OLMO) {
      return {{"hip_gpu", "cpu_generic"}, "OLMo — LayerNorm, learned positions"};
  }
