#include "engine.h"
#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

int main(int argc, char** argv) {
    printf("╔═══════════════════════════════════════════════╗\n");
    printf("║  NPU Inference Engine — Qwen3-0.6B           ║\n");
    printf("║  Strix Halo XDNA 2 NPU — Full Pipeline       ║\n");
    printf("╚═══════════════════════════════════════════════╝\n\n");
    
    const char* model_path = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    if (argc > 1) model_path = argv[1];
    
    if (argc > 1 && strcmp(argv[1], "--help") == 0) {
        printf("Usage: %s [model_path]\n", model_path);
        return 0;
    }
    
    // Init engine
    NpuInferenceEngine engine;
    if (!engine.init(model_path)) {
        fprintf(stderr, "❌ Engine initialization failed\n");
        return 1;
    }
    
    // Test generate: BOS token → a few output tokens
    const int prompt[] = {151643}; // BOS=151643 for Qwen3
    int output[32];
    
    auto t0 = std::chrono::steady_clock::now();
    int num_out = engine.generate(prompt, 1, output, 16);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    if (num_out > 0) {
        printf("\n✅ Generated %d tokens (%.0f ms, %.0f ms/tok):\n  ", 
               num_out, ms, ms / num_out);
        for (int i = 0; i < num_out; i++) {
            printf("%d ", output[i]);
        }
        printf("\n");
    } else {
        printf("\n❌ No tokens generated\n");
    }
    
    return 0;
}
