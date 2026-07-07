/**
 * test_fused_target.cpp — Test the NpuFusedTarget (single xclbin per layer).
 *
 * Build:
 *   g++ -std=c++23 -O3 -o build/test_fused_target tools/test_fused_target.cpp \
 *     /home/bcloud/engine/npu/build/dequant_q4nx.o \
 *     -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl -fopenmp -I.
 *
 * Run:
 *   sudo ./build/test_fused_target [max_tokens]
 *     OR (for power/efficiency measurement):
 *   FUSED_DBG=1 sudo ./build/test_fused_target [max_tokens]
 */
#include "engine/npu_fused_target.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>

// Correct paths for the fused layer xclbin + capref weights
static const char* kModelPath  = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
static const char* kXclbinDir  = "/home/bcloud/torch2aie/examples/qwen3-decode-layer/build/qwen3-decode-layer-capacity-token127";
static const char* kWeightsDir = "/home/bcloud/npu-sandbox/npu-infer/build/int8/capref";

static constexpr int32_t kTargetLayerIds[] = {1, 6, 12, 18, 24};

int main(int argc, char* argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int max_new = argc > 1 ? atoi(argv[1]) : 16;

    printf("═══ NPU Fused Target Test ═══\n");
    printf("Model:    %s\n", kModelPath);
    printf("XCLBINs:  %s\n", kXclbinDir);
    printf("Weights:  %s\n", kWeightsDir);
    printf("Generate: %d tokens\n\n", max_new);

    // Load the fused target
    printf("Loading...\n");
    auto t0 = std::chrono::steady_clock::now();

    NpuFusedTarget target(kModelPath, kXclbinDir, kWeightsDir,
                           kTargetLayerIds, 5);

    double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    printf("Loaded in %.0f ms\n\n", load_ms);

    // Short chat prompt
    int32_t prompt[] = {151643, 872, 198, 11852, 151644, 198, 151643, 77091, 198};
    int prompt_len = 9;

    const int H = 1024, NV = 151936;

    // Prefill
    printf("Prefill (%d tokens)...\n", prompt_len);
    std::vector<float> logits(NV);
    std::vector<float> hidden(28 * H);

    auto t1 = std::chrono::high_resolution_clock::now();
    target.forward(prompt, prompt_len, logits.data(), hidden.data());

    // Greedy decode
    printf("Generating %d tokens...\n", max_new);
    std::vector<int32_t> output(prompt_len + max_new);
    std::copy(prompt, prompt + prompt_len, output.begin());
    int generated = prompt_len;

    for (int i = 0; i < max_new; i++) {
        int tok = 0;
        float mv = logits[0];
        for (int v = 1; v < NV; v++) {
            if (logits[v] > mv) { mv = logits[v]; tok = v; }
        }
        output[generated++] = tok;
        if (tok == 151645) break;

        target.forward_with_kv(&tok, 1, generated - 1, logits.data(), hidden.data());
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    int new_tokens = generated - prompt_len;

    printf("\n═══ Results ═══\n");
    printf("  Generated: %d tokens\n", new_tokens);
    printf("  Time:      %.0f ms (%.1f tok/s)\n", ms, new_tokens / (ms / 1000.0));
    printf("  Tokens:    ");
    for (int i = prompt_len; i < generated && i < prompt_len + 10; i++)
        printf("%d ", output[i]);
    printf("%s\n", new_tokens > 10 ? "..." : "");

    return 0;
}
