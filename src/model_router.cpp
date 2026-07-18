#include "model_router.h"

// Phase 2: two rules. (a) exact Zaya1-8B dims (the only architecture the
// hand-tuned HIP/CPU kernels understand) keeps today's behavior unchanged.
// (b) everything else GGUF/H1B-shaped goes to cpu_generic, the one backend
// that actually parses arbitrary llama.cpp-style tensor layouts. Q4NX/ZINC
// routing is added in later phases (see docs/plans — Phase 3/4/5) as those
// backends land; until then a Q4NX model simply has no route here and falls
// through to BackendManager's normal (all-fail) behavior, which is honest
// given no working backend exists for it yet.
static bool is_exact_zaya_dims(const ModelConfig& cfg) {
    return cfg.hidden == 2048 && cfg.n_layers == 40 && cfg.n_heads == 8 &&
           cfg.n_kv_heads == 2 && cfg.head_dim == 128 && cfg.vocab == 262272;
}

BackendRoute select_backend_route(const ModelConfig& cfg) {
    if (is_exact_zaya_dims(cfg)) {
        return {{"hip_gpu", "cpu_scalar"}, "exact Zaya1-8B dims — hand-tuned kernel path"};
    }
    if (cfg.architecture == "qwen3") {
        // FLM only serves its own bundled Qwen3 catalog (see backend_flm.cpp's
        // tag_for_hidden) — cpu_generic as fallback if FLM/NPU isn't available
        // or the exact size isn't a good match.
        return {{"npu_flm", "cpu_generic"}, "qwen3 architecture — FastFlowLM NPU path"};
    }
    if (cfg.format == ModelFormat::GGUF || cfg.format == ModelFormat::H1B) {
        return {{"zinc_gpu", "cpu_generic"}, "GGUF/H1B model — ZINC GPU, generic CPU fallback"};
    }
    return {{}, "no route for this format yet — falling back to default priority order"};
}
