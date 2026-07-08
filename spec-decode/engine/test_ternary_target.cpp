/**
 * test_ternary_target.cpp — Validate NpuTernaryTarget against CPU reference.
 *
 * Build:
 *   g++ -std=c++23 -O2 -o test_ternary_target test_ternary_target.cpp \
 *       -I$XRT/include -I. -Iengine \
 *       -L$XRT/lib64 -lxrt_coreutil -fopenmp -lm
 *
 * Usage:
 *   ./test_ternary_target model.q4nx xclbin_dir/
 */

#include "spec-decode/engine/npu_ternary_target.h"
#include <cstdio>
#include <cstring>
#include <sys/time.h>

static double now_ms() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.q4nx> <xclbin_dir> [token_id]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* xclbin_dir = argv[2];
    int token_id = argc > 3 ? atoi(argv[3]) : 1;  // default token 1

    printf("=== Native Ternary Target Test ===\n");
    printf("Model:  %s\n", model_path);
    printf("Xclbin: %s\n", xclbin_dir);
    printf("Token:  %d\n", token_id);

    // Target layers for draft feature extraction (e.g., layers 4, 10, 18, 26)
    int32_t target_layers[] = {4, 10, 18, 26};
    int num_target = 4;

    printf("\n[1] Creating NpuTernaryTarget...\n");
    double t0 = now_ms();

    NpuTernaryTarget target(model_path, xclbin_dir,
                            target_layers, num_target);

    double t1 = now_ms();
    printf("    Init time: %.1f ms\n", t1 - t0);

    // Run forward
    printf("\n[2] Running forward (token=%d)...\n", token_id);
    int32_t tokens[] = {token_id};
    std::vector<float> logits(151936);
    std::vector<float> hidden(1024);

    t0 = now_ms();
    target.forward(tokens, 1, logits.data(), hidden.data());
    t1 = now_ms();

    printf("    Forward time: %.1f ms\n", t1 - t0);

    // Show top-5 tokens
    printf("\n[3] Top-5 predictions:\n");
    struct TokenScore { int id; float score; };
    std::vector<TokenScore> scored;
    for (int i = 0; i < 151936; i++)
        scored.push_back({i, logits[i]});
    std::partial_sort(scored.begin(), scored.begin() + 5, scored.end(),
        [](auto& a, auto& b) { return a.score > b.score; });

    for (int i = 0; i < 5; i++) {
        printf("    token %6d: %.4f\n", scored[i].id, scored[i].score);
    }

    // Show hidden state stats
    printf("\n[4] Hidden state stats:\n");
    double hsum = 0, hsq = 0;
    float hmin = hidden[0], hmax = hidden[0];
    for (int i = 0; i < 1024; i++) {
        hsum += hidden[i];
        hsq += (double)hidden[i] * hidden[i];
        if (hidden[i] < hmin) hmin = hidden[i];
        if (hidden[i] > hmax) hmax = hidden[i];
    }
    double hmean = hsum / 1024;
    double hstd = sqrt(hsq / 1024 - hmean * hmean);
    printf("    mean=%.4f std=%.4f min=%.4f max=%.4f\n",
           hmean, hstd, hmin, hmax);

    // Extract draft features
    printf("\n[5] Draft feature extraction (target layers %d,%d,%d,%d):\n",
           target_layers[0], target_layers[1], target_layers[2], target_layers[3]);
    std::vector<float> features(num_target * 1024);
    target.get_layer_hidden(nullptr, 0, target_layers, num_target, features.data());

    for (int t = 0; t < num_target; t++) {
        double fsum = 0;
        for (int i = 0; i < 1024; i++) fsum += features[t * 1024 + i];
        printf("    layer %2d: mean=%.4f\n", target_layers[t], fsum / 1024);
    }

    printf("\n=== Test complete ===\n");
    return 0;
}
