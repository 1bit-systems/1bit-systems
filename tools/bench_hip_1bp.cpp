#include "backend.h"
#include <cstdio>
#include <chrono>
#include <string>

extern "C" Backend* create_hip_1bp_backend();

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "models/Qwen3-0.6B.1bp";
    int tokens = argc > 2 ? atoi(argv[2]) : 10;
    int warmup = argc > 3 ? atoi(argv[3]) : 3;

    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║    GPU 1BP Engine Benchmark              ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    Backend* b = create_hip_1bp_backend();
    if (!b) { fprintf(stderr, "FAIL: create_hip_1bp_backend\n"); return 1; }

    ModelConfig cfg;
    cfg.model_path = path;
    cfg.format = ModelFormat::ONEBP;
    uint8_t hdr[256];
    FILE* f = fopen(path, "rb"); if (!f) { fprintf(stderr, "Cannot open %s\n", path); return 1; }
    fread(hdr, 1, 256, f); fclose(f);
    memcpy(&cfg.hidden_size, hdr+20, 4); memcpy(&cfg.num_layers, hdr+24, 4);
    memcpy(&cfg.num_heads, hdr+28, 4); memcpy(&cfg.num_kv_heads, hdr+32, 4);
    memcpy(&cfg.head_dim, hdr+36, 4); memcpy(&cfg.intermediate_size, hdr+40, 4);
    memcpy(&cfg.vocab_size, hdr+44, 4);

    printf("  Model: %s\n", path);
    printf("  Dims:  H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d V=%d\n",
           cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.num_kv_heads,
           cfg.head_dim, cfg.intermediate_size, cfg.vocab_size);
    printf("  Tokens: %d (%d warmup + %d measured)\n\n", tokens + warmup, warmup, tokens);

    // Init
    auto t0 = std::chrono::steady_clock::now();
    if (!b->init(cfg, path)) { fprintf(stderr, "FAIL: init\n"); return 1; }
    auto t1 = std::chrono::steady_clock::now();
    printf("  Init: %.0f ms\n", std::chrono::duration<double,std::milli>(t1-t0).count());

    // Warmup
    b->reset();
    int tok = 1;
    for (int i = 0; i < warmup; i++) { tok = b->generate(tok); if (tok < 0) break; }
    printf("  Warmup: %d tokens (last token id: %d)\n", warmup, tok);

    // Benchmark
    b->reset();
    tok = 1;
    t0 = std::chrono::steady_clock::now();
    int generated = 0;
    for (int i = 0; i < tokens && tok >= 0; i++) {
        tok = b->generate(tok);
        if (tok >= 0) generated++;
    }
    t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();

    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║           RESULTS                        ║\n");
    printf("╚══════════════════════════════════════════╝\n");
    printf("  Tokens:     %d\n", generated);
    printf("  Time:       %.0f ms\n", ms);
    printf("  Per token:  %.1f ms\n", ms / generated);
    printf("  Throughput: %.0f tok/s\n", generated / (ms/1000.0));
    printf("  Tokens:    ");
    b->reset();
    tok = 1;
    for (int i = 0; i < 8 && tok >= 0; i++) {
        tok = b->generate(tok);
        printf(" %d", tok);
    }
    printf("\n\n");

    b->destroy();
    delete b;
    return 0;
}
