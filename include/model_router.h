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
// npu_xrt is the sole NPU route for qwen3 models as of 2026-07-21 — its
// single-core GEMM kernels are correctness-verified on real hardware
// (docs/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md). The npu_flm backend (a
// subprocess wrapper around AMD's proprietary FastFlowLM binary) has been
// removed entirely, not just deprioritized — this project ships zero
// proprietary code. The 8-core multi-tile path is still unverified in
// combination, so single-core throughput is what ships until that lands.

#include "common.h"
#include <string>
#include <vector>

struct BackendRoute {
    std::vector<std::string> backend_ids_in_order;
    std::string reason; // human-readable, for logging
};

BackendRoute select_backend_route(const ModelConfig& cfg);
