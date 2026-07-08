/**
 * test_acceptance.cpp — Test trained Eagle3 draft acceptance on 4-xclbin target.
 * Build: g++ -std=c++23 -O3 -o build/test_acceptance tools/test_acceptance.cpp \
 *   /home/bcloud/engine/npu/build/dequant_q4nx.o \
 *   -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl -fopenmp -I.
 * Run: sudo ./build/test_acceptance
 */
#include "engine/spec_decode.h"
#include "engine/npu_target_model.h"
#include "draft/mtp_draft.h"
#include <cstdio>
#include <chrono>

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char* kModel = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    const char* kXclbin = "/home/bcloud/npu-sandbox/npu-infer/build/int8";
    const char* kCheckpoint = "/home/bcloud/spec-decode/checkpoints/eagle3_draft_npu_1k.bin";
    // Real Qwen3-0.6B chat prompt tokens for "<|im_start|>user\nHello, what is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    int prompt[] = {151644, 872, 198, 9707, 11, 1128, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198};
    int prompt_len = 17;

    printf("═══ Acceptance Test ═══\n");
    printf("Draft: %s\n\n", kCheckpoint);

    MTPDraftConfig draft_cfg;
    MTPDraftModel draft(draft_cfg);
    bool loaded = draft.load_weights(kCheckpoint);
    printf("Draft loaded: %s\n\n", loaded ? "YES ✅" : "NO ❌");

    SpecDecodeConfig spec_cfg;
    spec_cfg.max_new_tokens = 16;
    spec_cfg.num_draft_layers = 1;

    NPUQwen3Target target(kModel, kXclbin, spec_cfg.target_layer_ids, spec_cfg.num_target_layers);
    using MTPSpec = SpeculativeDecoderT<MTPDraftModel, MTPDraftState, MTPDraftConfig>;
    MTPSpec decoder(target, draft, spec_cfg);

    std::vector<int32_t> output(100);
    auto t0 = std::chrono::high_resolution_clock::now();
    int gen = decoder.generate(prompt, prompt_len, output.data(), 16);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto& s = decoder.stats();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Generated: %d tokens in %.0f ms (%.1f tok/s)\n", gen - prompt_len, ms, (gen - prompt_len) / (ms / 1000));
    printf("Accept:   %.1f%%\n", s.acceptance_rate() * 100);
    printf("Speedup:  %.2fx (verifies: %ld)\n", s.speedup_factor(), (long)s.verify_calls);
    if (s.acceptance_rate() > 0.01f)
        printf("✅ ACCEPTANCE > 0%%. Training worked!\n");
    else
        printf("❌ Still 0%% acceptance\n");
    return s.acceptance_rate() > 0.01f ? 0 : 1;
}
