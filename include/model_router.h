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
// npu_xrt is deliberately never returned here — its GEMM kernels are confirmed
// broken (docs/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md) and it stays manual-opt-in
// only (see BackendInfo::auto_selectable in backend_manager.h).

#include "common.h"
#include <string>
#include <vector>

struct BackendRoute {
    std::vector<std::string> backend_ids_in_order;
    std::string reason; // human-readable, for logging
};

BackendRoute select_backend_route(const ModelConfig& cfg);
