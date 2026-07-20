#pragma once
// model_router.h — model -> backend selection.
//
// Given a discovered ModelConfig, decides which Backend(s) (by BackendManager
// id, in preference order) should try to handle it. This is a static table in
// the spirit of engine/fusion/arch_registry.zig's architecture registry,
// ported to C++: it only decides ORDER, reusing BackendManager's existing
// try-until-one-succeeds init loop (see BackendManager::init's preferred_ids
// overload) rather than introducing a new selection primitive.
//
// npu_xrt is the default NPU route for qwen3 models as of 2026-07-20 — its
// single-core GEMM kernels are correctness-verified on real hardware
// (docs/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md); npu_flm (FastFlowLM subprocess)
// is kept only as a fallback. The 8-core multi-tile path is still unverified
// in combination, so throughput is currently lower (~12 tok/s single-core vs
// FLM's ~95 tok/s) until that lands.

#include "common.h"
#include <string>
#include <vector>

struct BackendRoute {
    std::vector<std::string> backend_ids_in_order;
    std::string reason; // human-readable, for logging
};

BackendRoute select_backend_route(const ModelConfig& cfg);
