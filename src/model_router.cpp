#include "model_router.h"

// Phase 4: universal routing. The HIP/CPU backends now accept runtime
// ZayaConfig from any ModelConfig. Single-token inference kernels (CCA prep,
// EDA router) are fully dynamic. Batch-path speculative-decode kernels still
// have architecture-specific shared memory and are documented in zaya_router_moe.hip.
//
// Routes:
//   (a) Zaya-style MoE models → hip_gpu + cpu_scalar (fast CCA/MoE kernels)
//   (b) qwen3 architecture → npu_xrt + npu_flm + cpu_generic (native NPU engine,
//       FLM subprocess as fallback only — see backend_manager.cpp, npu_xrt is
//       the default now that its single-core GEMM kernels are correctness-verified,
//       docs/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md)
//   (c) GGUF/H1B format → zinc_gpu + cpu_generic (multi-arch, multi-quant)
//   (d) Everything else → hip_gpu + cpu_generic (dynamic engine, generic fallback)

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
