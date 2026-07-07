/**
 * test_eagle3_npu.cpp — Test the NPU-trained Eagle3 draft on real hardware.
 *
 * Loads NPU target + Eagle3 draft (MTP) and runs speculative decoding.
 * Measures acceptance rate and speedup.
 *
 * Build:
 *   g++ -std=c++23 -O3 -o build/test_eagle3_npu tools/test_eagle3_npu.cpp \
 *     /home/bcloud/engine/npu/build/dequant_q4nx.o \
 *     -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl -I.
 *
 * Run:
 *   sudo ./build/test_eagle3_npu [max_tokens] [num_examples]
 */

#include "engine/spec_decode.h"
#include "engine/npu_target_model.h"
#include "draft/mtp_draft.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>

static const char* kModelPath    = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
static const char* kXclbinDir    = "/home/bcloud/npu-sandbox/npu-infer/build/int8";
static const char* kCheckpoint   = "/home/bcloud/spec-decode/checkpoints/eagle3_draft_npu.bin";

// Precomputed prompts (short, varied)
static int32_t kPrompts[][9] = {
    {151643, 872, 198, 11852, 151644, 198, 151643, 77091, 198},  // "hi"
    {151643, 872, 198, 9826, 151644, 198, 151643, 77091, 198},   // "hello"
    {151643, 872, 198, 11353, 151644, 198, 151643, 77091, 198},  // "what"
};
static constexpr int kNumPrompts = 3;
static constexpr int kPromptLen = 9;

int main(int argc, char* argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int max_new = argc > 1 ? atoi(argv[1]) : 32;
    int num_examples = argc > 2 ? atoi(argv[2]) : kNumPrompts;
    if (num_examples > kNumPrompts) num_examples = kNumPrompts;

    printf("═══ NPU Eagle3 Draft Test ═══\n");
    printf("Checkpoint: %s\n", kCheckpoint);
    printf("Max new:    %d tokens\n", max_new);
    printf("Examples:   %d\n\n", num_examples);

    // Load NPU target
    printf("Loading NPU target...\n");
    auto t0 = std::chrono::steady_clock::now();
    
    MTPDraftConfig draft_cfg;
    MTPDraftModel draft(draft_cfg);
    bool loaded = draft.load_weights(kCheckpoint);
    
    SpecDecodeConfig spec_cfg;
    spec_cfg.max_new_tokens = max_new;
    spec_cfg.num_draft_layers = 1;  // Eagle3 = 1 layer
    
    NPUQwen3Target target(kModelPath, kXclbinDir,
                           spec_cfg.target_layer_ids, spec_cfg.num_target_layers);
    
    double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    printf("  Loaded in %.0f ms\n", load_ms);
    printf("  Draft loaded: %s\n\n", loaded ? "YES (NPU-trained)" : "NO");

    // Test with MTP draft (Eagle3)
    using MTPDecoder = SpeculativeDecoderT<MTPDraftModel, MTPDraftState, MTPDraftConfig>;
    
    int total_accepted = 0;
    int total_proposed = 0;
    int total_verify = 0;
    double total_ms = 0;
    int total_tokens = 0;

    for (int ex = 0; ex < num_examples; ex++) {
        printf("── Example %d ──\n", ex);

        // Re-create decoder for each example (fresh KV cache)
        // (Need fresh target for each because NPUQwen3Target's internal state
        //  is tied to the decoder's loop — a proper API would reuse it)
        MTPDraftModel draft_ex(draft_cfg);
        draft_ex.load_weights(kCheckpoint);
        NPUQwen3Target target_ex(kModelPath, kXclbinDir,
                                  spec_cfg.target_layer_ids, spec_cfg.num_target_layers);
        MTPDecoder decoder(target_ex, draft_ex, spec_cfg);

        std::vector<int32_t> prompt(kPrompts[ex], kPrompts[ex] + kPromptLen);
        std::vector<int32_t> output(max_new + kPromptLen);

        auto t1 = std::chrono::high_resolution_clock::now();
        int generated = decoder.generate(prompt.data(), kPromptLen,
                                          output.data(), max_new);
        auto t2 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        auto& s = decoder.stats();

        printf("  Generated: %d tokens in %.0f ms (%.1f tok/s)\n",
               generated - kPromptLen, ms,
               (generated - kPromptLen) / (ms / 1000.0));
        printf("  Accept:   %.1f%% | Speedup: %.2fx | Verifies: %ld\n",
               s.acceptance_rate() * 100, s.speedup_factor(), (long)s.verify_calls);

        total_accepted += s.accepted_draft_tokens;
        total_proposed += s.total_draft_proposed;
        total_verify += s.verify_calls;
        total_ms += ms;
        total_tokens += (generated - kPromptLen);
    }

    printf("\n═══ Summary ═══\n");
    float overall_accept = total_proposed > 0 ? (float)total_accepted / total_proposed * 100 : 0;
    float avg_speedup = total_verify > 0 ? (float)total_tokens / total_verify : 1.0f;
    printf("  Overall acceptance:   %.1f%% (%d/%d)\n", overall_accept, total_accepted, total_proposed);
    printf("  Overall speedup:      %.2fx\n", avg_speedup);
    printf("  Total tokens:         %d\n", total_tokens);
    printf("  Total verifies:       %d\n", total_verify);
    printf("  Avg throughput:       %.1f tok/s\n",
           total_tokens / (total_ms / 1000.0));
    
    printf("\n");
    if (overall_accept > 5.0f) {
        printf("✅ ACCEPTANCE IMPROVED from 0%% to %.1f%%!\n", overall_accept);
    } else {
        printf("⚠️  Acceptance still low (%.1f%%). Need more NPU training data.\n", overall_accept);
    }

    return overall_accept > 5.0f ? 0 : 1;
}
