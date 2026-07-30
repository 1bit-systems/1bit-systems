// bench_fused.cpp — Benchmark fused GPU attention ∥ NPU FFN pipeline.
// Build: cmake --build build --target bench_fused
// Run:   LD_LIBRARY_PATH=... ./build/bench_fused models/Qwen3-0.6B.1bp 10 3
#include "backend.h"
#include <cstdio>
#include <chrono>
extern "C" Backend* create_fused_backend();

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "models/Qwen3-0.6B.1bp";
    int tokens = argc > 2 ? atoi(argv[2]) : 10;
    int warmup = argc > 3 ? atoi(argv[3]) : 3;
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  Fused GPU Attn ∥ NPU FFN Benchmark     ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");
    Backend* b = create_fused_backend();
    if (!b) { fprintf(stderr, "FAIL: create_fused_backend\n"); return 1; }
    ModelConfig cfg; cfg.model_path = path; cfg.format = ModelFormat::ONEBP;
    uint8_t hdr[256];
    FILE* f = fopen(path, "rb"); fread(hdr, 1, 256, f); fclose(f);
    memcpy(&cfg.hidden_size, hdr+20, 4); memcpy(&cfg.num_layers, hdr+24, 4);
    cfg.num_heads = 16; cfg.num_kv_heads = 8; cfg.head_dim = 128;
    cfg.intermediate_size = 3072; cfg.vocab_size = 151936;
    printf("  Model: %s\n", path);
    printf("  Dims:  H=%d NC=%d\n", cfg.hidden_size, cfg.num_layers);
    auto t0 = std::chrono::steady_clock::now();
    if (!b->init(cfg, path)) { fprintf(stderr, "FAIL: init\n"); return 1; }
    auto t1 = std::chrono::steady_clock::now();
    printf("  Init:  %.0f ms\n\n", std::chrono::duration<double,std::milli>(t1-t0).count());
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
    delete b;  // destructor calls destroy()
    return 0;
}
