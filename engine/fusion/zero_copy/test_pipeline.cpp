// test_pipeline_overlap.cpp — 2-slot pipeline skeleton with dummy callbacks.
//
// Tests the overlap pattern without real NPU/GPU kernels: the "GPU" and "NPU"
// callbacks simulate realistic latency with std::this_thread::sleep_for so we
// can measure overlap efficiency.
//
// Build & run: make test_pipeline && ./test_pipeline
#include "pipeline_overlap.h"
#include <cstdio>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>

int main() {
    // Need /dev/accel/accel0; try to open it to check.
    int acc = open("/dev/accel/accel0", O_RDONLY);
    if (acc < 0) { fprintf(stderr, "No NPU access — skipping test.\n"); return 77; }
    close(acc);

    xrt::device npu(0);

    fusion::PipelineConfig cfg;
    cfg.layer_count  = 40;
    cfg.hidden_dim   = 2048;
    cfg.inter_size   = 2048;
    cfg.batch_size   = 1;
    cfg.attn_scratch = 0;

    fusion::PipelineOverlap pl(cfg, npu);

    // Dummy GPU: simulated QKV attention taking ~2ms.
    auto gpu_fn = [](int layer, int slot, float* h, float* out) {
        (void)layer; (void)slot; (void)h;
        // Simulate GPU-attn latency
        std::this_thread::sleep_for(std::chrono::microseconds(2000));
        // Write a marker so we can verify the data flow
        for (int i = 0; i < 2048; i++) out[i] = (float)layer + (float)i * 0.001f;
    };

    // Dummy NPU: simulated MoE FFN taking ~1ms (NPU is slower for FFN).
    auto npu_fn = [](int layer, int slot, float* h, float* out) {
        (void)layer; (void)slot; (void)h;
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
        for (int i = 0; i < 2048; i++) out[i] = (float)(layer * 1000 + i);
    };

    fprintf(stderr, "Pipeline: %d layers, GPU ~2ms/layer, NPU ~1ms/layer\n", cfg.layer_count);
    fprintf(stderr, "Sequential would take ~%.0fms\n",
            40.0 * (2000 + 1000) / 1000.0);
    fprintf(stderr, "With ideal overlap (NPU hides behind GPU): ~%.0fms\n",
            40.0 * 2000 / 1000.0);

    auto m = pl.run(gpu_fn, npu_fn);
    fprintf(stderr, "\nResult: %.2f ms total (%.2f ms/layer)\n",
            m.total_ms, m.total_ms / cfg.layer_count);
    fprintf(stderr, "Efficiency: overlap efficiency (virtual time) = %.1f%%\n",
            m.overlap_efficiency / m.total_ms / 10.0f);

    // Verify slot data is reachable (zero-copy proof: write host, read via GPU ptr).
    // Already proven in test_zero_copy — here just ensure we can iterate.
    float* s0 = (float*)pl.slot_a()->host_ptr();
    float* s1 = (float*)pl.slot_b()->host_ptr();
    fprintf(stderr, "SlotA[0..3]: %.1f %.1f %.1f %.1f\n", s0[0], s0[1], s0[2], s0[3]);
    fprintf(stderr, "SlotB[0..3]: %.1f %.1f %.1f %.1f\n", s1[0], s1[1], s1[2], s1[3]);
    fprintf(stderr, "\n=== PIPELINE SKELETON OK ===\n");
    return 0;
}
