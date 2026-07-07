/**
 * test_fused_server_target.cpp — Test FusedServerTarget (subprocess approach).
 * Build:
 *   g++ -std=c++23 -O3 -o build/test_fused_server_target tools/test_fused_server_target.cpp \
 *     /home/bcloud/engine/npu/build/dequant_q4nx.o \
 *     -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl -fopenmp -I.
 * Run: sudo ./build/test_fused_server_target
 */
#include "engine/fused_server_target.h"
#include <cstdio>
#include <chrono>

static const char* kModelPath  = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
static const char* kXclbinDir  = "/home/bcloud/torch2aie/examples/qwen3-decode-layer/build/qwen3-decode-layer-capacity-token127";
static const char* kWeightsDir = "/home/bcloud/npu-sandbox/npu-infer/build/int8/capref";
static constexpr int32_t kTargetLayerIds[] = {1, 6, 12, 18, 24};

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    const int H = 1024, NV = 151936;

    printf("═══ FusedServerTarget Test ═══\n\n");
    
    FusedServerTarget target(kModelPath, kXclbinDir, kWeightsDir,
                              kTargetLayerIds, 5);
    
    if (!target.ready()) {
        fprintf(stderr, "Server failed to start\n");
        return 1;
    }
    
    // Single token forward
    int32_t token = 151643;  // <|im_start|>
    std::vector<float> logits(NV);
    std::vector<float> hidden(28 * H);
    
    printf("Forward single token...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    target.forward(&token, 1, logits.data(), hidden.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    // Logit stats
    float lmin = logits[0], lmax = logits[0];
    int argmax = 0, lnz = 0;
    for (int v = 0; v < NV; v++) {
        if (logits[v] < lmin) lmin = logits[v];
        if (logits[v] > lmax) { lmax = logits[v]; argmax = v; }
        if (logits[v] != 0) lnz++;
    }
    printf("  %.0f ms | logits: min=%.2f max=%.2f argmax=%d nonzero=%d/%d\n",
           ms, lmin, lmax, argmax, lnz, NV);
    
    // Hidden state stats
    float hmin = hidden[0], hmax = hidden[0], hsum = 0;
    int hnz = 0;
    for (int i = 0; i < 28 * H; i++) {
        if (hidden[i] < hmin) hmin = hidden[i];
        if (hidden[i] > hmax) hmax = hidden[i];
        hsum += hidden[i];
        if (hidden[i] != 0) hnz++;
    }
    printf("  hidden: min=%.4f max=%.4f mean=%.4f nonzero=%d\n",
           hmin, hmax, hsum / (28 * H), hnz);
    
    // Verify no NaN
    bool has_nan = false;
    for (int i = 0; i < 28 * H; i++)
        if (std::isnan(hidden[i])) { has_nan = true; break; }
    printf("  NaN in hidden: %s\n", has_nan ? "YES ⚠️" : "no ✅");
    
    // Generate a few tokens
    printf("\nGenerating 8 tokens...\n");
    int32_t prompt[] = {151643, 872, 198, 11852, 151644, 198, 151643, 77091, 198};
    int prompt_len = 9;
    std::vector<int32_t> output(prompt_len + 8);
    std::copy(prompt, prompt + prompt_len, output.begin());
    
    // Recreate for fresh KV
    FusedServerTarget target2(kModelPath, kXclbinDir, kWeightsDir,
                               kTargetLayerIds, 5);
    
    std::vector<float> lg(NV), hd(28 * H);
    target2.forward(prompt, prompt_len, lg.data(), hd.data());
    
    int gen = prompt_len;
    for (int i = 0; i < 8; i++) {
        int tok = 0; float mv = lg[0];
        for (int v = 1; v < NV; v++) if (lg[v] > mv) { mv = lg[v]; tok = v; }
        output[gen++] = tok;
        if (tok == 151645) break;
        target2.forward_with_kv(&tok, 1, gen - 1, lg.data(), hd.data());
    }
    
    printf("Tokens: ");
    for (int i = prompt_len; i < gen; i++) printf("%d ", output[i]);
    printf("\n");
    
    // target2 destroyed automatically at scope exit
    return 0;
}
