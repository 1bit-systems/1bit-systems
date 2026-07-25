// pipeline_overlap.h — 2-slot pipelined NPU-FFN ∥ GPU-attention.
//
// Architecture:
//   Two SharedBO slots (A and B) — each NPU-owned, GPU-imported, host-coherent.
//   GPU does attention for layer L on slot A while NPU does FFN for layer L-1
//   on slot B.  On each layer boundary the roles swap (A↔B).
//
//   ┌──────────────┐         ┌──────────────┐
//   │  GPU attn L0 ├────────>│  NPU FFN L0  │──┐
//   └──────────────┘         └──────────────┘  │
//                          ┌───────────────────┘
//                          ▼
//   ┌──────────────┐    ┌──────────────┐
//   │  GPU attn L1 │<───│  NPU FFN L0  │  (overlap: GPU attn L1 & NPU FFN L1 run in parallel)
//   └──────────────┘    └──────────────┘
//
// This skeleton compiles without xclbins (pure XRT + HIP).  It exercises the
// double-buffering + synchronization pattern that the real fused engine needs.
//
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include "shared_bo.h"

namespace fusion {

// Configuration for a 2-slot pipelined executor.
struct PipelineConfig {
    size_t layer_count   = 40;   // NC
    size_t hidden_dim    = 2048; // H
    size_t inter_size    = 2048; // FFN intermediate dim
    size_t batch_size    = 1;
    size_t attn_scratch  = 0;    // widest per-half slot width in floats, if > hidden_dim
                                  // (e.g. gpu_attn_fn's raw NH*HD output before a projection
                                  // back down to hidden_dim); 0 means "same as hidden_dim"
};

// Result from one forward pass (for instrumentation/profiling).
struct PipelineMetrics {
    double total_ms       = 0;
    double gpu_ms_per_layer = 0;
    double npu_ms_per_layer = 0;
    double overlap_efficiency = 0;  // fraction of total time where both ran simultaneously
};

// 2-slot pipelined executor.
// Does NOT implement the actual GPU attention or NPU FFN — those callbacks are
// injected so this skeleton can be tested standalone with dummy work.
class PipelineOverlap {
public:
    PipelineOverlap(const PipelineConfig& cfg, xrt::device& npu_dev);
    ~PipelineOverlap();

    // Run one full forward pass with pipelined overlap.
    // @param gpu_attn_fn  f(layer, slot_idx, h_in, h_out)  — GPU attention kernel
    // @param npu_ffn_fn   f(layer, slot_idx, h_in, h_out)  — NPU FFN kernel
    PipelineMetrics run(
        std::function<void(int layer, int slot, float* h, float* out)> gpu_attn_fn,
        std::function<void(int layer, int slot, float* h, float* out)> npu_ffn_fn
    );

    // Access the shared slots for inspection.
    SharedBO* slot_a() { return slot_[0]; }
    SharedBO* slot_b() { return slot_[1]; }

private:
    PipelineConfig cfg_;
    SharedBO* slot_[2] = {};

    // Double-buffer state.
    int active_slot_ = 0;  // GPU writes here for layer L
    int drain_slot_  = 1;  // NPU reads from here for layer L-1
};

} // namespace fusion
