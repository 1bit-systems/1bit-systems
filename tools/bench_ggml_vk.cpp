// bench_ggml_vk.cpp — Benchmark the GGML-Vulkan backend (llama.cpp, MIT License).
// Build: cmake --build build --target bench_ggml_vk
// Run:   LD_LIBRARY_PATH=... ./build/bench_ggml_vk models/Qwen3-0.6B.Q4_K_M.gguf 100 5

#include "backend.h"
#include "backend_ggml_vulkan.h"
#include <cstdio>
#include <chrono>

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "models/Qwen3-0.6B.Q4_K_M.gguf";
    int tokens = argc > 2 ? atoi(argv[2]) : 100;
    int warmup = argc > 3 ? atoi(argv[3]) : 5;

    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  GGML-Vulkan (llama.cpp) Benchmark      ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    Backend* b = create_ggml_vulkan_backend();
    if (!b) { fprintf(stderr, "FAIL: create\n"); return 1; }

    ModelConfig cfg;
    cfg.model_path = path;
    cfg.format = ModelFormat::GGUF;

    auto t0 = std::chrono::steady_clock::now();
    if (!b->init(cfg, "")) { fprintf(stderr, "FAIL: init\n"); return 1; }
    auto t1 = std::chrono::steady_clock::now();
    printf("  Init: %.0f ms\n\n", std::chrono::duration<double,std::milli>(t1-t0).count());

    b->reset(); int tok = 1;
    for (int i = 0; i < warmup; i++) tok = b->generate(tok);
    b->reset(); tok = 1;
    t0 = std::chrono::steady_clock::now();
    int ok = 0;
    for (int i = 0; i < tokens && tok >= 0; i++) { tok = b->generate(tok); if (tok >= 0) ok++; }
    t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();

    printf("╔══════════════════════════════════════════╗\n");
    printf("║           RESULTS                        ║\n");
    printf("╚══════════════════════════════════════════╝\n");
    printf("  Tokens:     %d\n", ok);
    printf("  Time:       %.0f ms\n", ms);
    printf("  Per token:  %.1f ms\n", ms/ok);
    printf("  Throughput: %.0f tok/s\n", ok/(ms/1000.0));

    delete b;
    return 0;
}
