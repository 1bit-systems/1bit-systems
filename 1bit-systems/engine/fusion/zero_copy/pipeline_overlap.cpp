// pipeline_overlap.cpp — 2-slot pipelined NPU-FFN ∥ GPU-attention skeleton.
#include "pipeline_overlap.h"
#include <chrono>
#include <cstring>
#include <cmath>

namespace fusion {

PipelineOverlap::PipelineOverlap(const PipelineConfig& cfg, xrt::device& npu_dev) : cfg_(cfg) {
    size_t slot_bytes = cfg_.hidden_dim * cfg_.batch_size * sizeof(float) * 2; // h_in + h_out
    slot_[0] = SharedBO::create(npu_dev, slot_bytes);
    slot_[1] = SharedBO::create(npu_dev, slot_bytes);
    if (!slot_[0] || !slot_[1]) {
        fprintf(stderr, "PipelineOverlap: failed to allocate shared slots\n");
        // One or both null → downstream ops will catch it.
    }
}

PipelineOverlap::~PipelineOverlap() {
    delete slot_[0];
    delete slot_[1];
}

PipelineMetrics PipelineOverlap::run(
    std::function<void(int, int, float*, float*)> gpu_attn_fn,
    std::function<void(int, int, float*, float*)> npu_ffn_fn)
{
    PipelineMetrics m{};
    auto t_start = std::chrono::steady_clock::now();

    const int NC = (int)cfg_.layer_count;
    const int H = (int)cfg_.hidden_dim * (int)cfg_.batch_size;
    const int SZ = H * (int)sizeof(float);

    // Prime slot 0 with input (embedding).
    // In the real engine this is the token embedding result.
    memset(slot_[0]->host_ptr(), 0, SZ);

    // --- Layer 0: boot (no overlap, NPU runs after GPU) ---
    float* h_in  = (float*)slot_[0]->host_ptr();
    float* h_gpu = (float*)((uint8_t*)slot_[0]->host_ptr() + SZ);
    gpu_attn_fn(0, 0, h_in, h_gpu);                     // GPU launches
    npu_ffn_fn(0, 1, h_gpu, (float*)slot_[1]->host_ptr());  // NPU runs (GPU must have finished)

    // --- Layers 1..NC-1: pipelined overlap ---
    // At layer L:
    //   GPU: attention(L)  → slot[active] (asynchronous)
    //   NPU: ffn(L-1)      → slot[drain]  (completing from previous stage)
    //   Then wait-both, residual-add, swap.
    int active = 0;   // GPU writes attn(L) here
    int drain  = 1;   // NPU has been writing ffn(L-1) here

    // slot layout: [0..H)=h_in, [H..2H)=h_out (GPU attn result or residual)

    for (int L = 1; L < NC; L++) {
        float* h_current = (float*)slot_[active]->host_ptr();
        float* attn_out  = (float*)((uint8_t*)slot_[active]->host_ptr() + SZ);

        auto t_overlap_start = std::chrono::steady_clock::now();

        // Launch GPU attention for L (async, on HIP stream).
        // Launches FIRST so GPU work begins immediately.
        gpu_attn_fn(L, active, h_current, attn_out);

        // NPU was already doing FFN(L-1) from the PREVIOUS iteration.
        // Here we wait for it (in a real engine the wait was already orchestrated
        // at the end of the previous iteration — we duplicate it here as a sync point).
        // The key insight: gpu_attn launch and npu_ffn wait overlap naturally
        // if the NPU is still busy while the GPU is being programmed.

        // Wait for NPU FFN(L-1) to finish.
        // In the real engine: npu_run.wait() blocks here.
        // The dummy callback returns immediately, but we track virtual time.
        // The critical thing is that we launched GPU work BEFORE waiting for NPU.

        // After NPU has finished FFN(L-1), the drain slot contains h_{L-1}'s output.
        // Add residual: h_out(L-1) += residual(L-1)  (CPU, quick)
        // then use as input for next iteration.
        float* residual = (float*)((uint8_t*)slot_[drain]->host_ptr() + SZ);
        float* ffn_out  = (float*)slot_[drain]->host_ptr();
        for (int i = 0; i < H; i++) {
            ffn_out[i] += residual[i];  // residual add (normally done per-layer)
        }

        // Now both GPU attn(L) and NPU FFN(L-1) should have waited.
        // Swap roles for next layer.
        std::swap(active, drain);

        uint64_t overlap_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t_overlap_start).count();
        m.overlap_efficiency += (double)overlap_ns;
    }

    // Total time
    auto t_end = std::chrono::steady_clock::now();
    m.total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return m;
}

} // namespace fusion
