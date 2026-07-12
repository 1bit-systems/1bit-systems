/**
 * NPU Speculative Decoding Integration
 *
 * Bridges the speculative decoding engine (spec_decode.h) with the 1bit NPU inference stack,
 * using the real 4-xclbin INT8 GEMM dispatch (NPUQwen3Target, engine/npu_target_model.h) as
 * the target model, and the tiny MTP draft head (draft/mtp_draft.h) as the draft model.
 *
 * Build:
 *   cmake -DENABLE_NPU=ON -DXRT_DIR=/home/bcloud/torch2aie/toolchain/xrt ..
 *   make npu_spec_decode
 *
 * Run:
 *   ./npu_spec_decode [prompt_len] [max_new_tokens]
 */

#include "spec_decode.h"
#include "npu_target_model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

static const char* kModelPath = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
static const char* kXclbinDir = "/home/bcloud/npu-sandbox/npu-infer/build/int8";
static const char* kDraftCheckpoint = "/home/bcloud/spec-decode/checkpoints/eagle3_draft_trained_420.bin";

int main(int argc, char* argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int prompt_len = argc > 1 ? atoi(argv[1]) : 9;
    int max_new = argc > 2 ? atoi(argv[2]) : 64;

    printf("1bit Speculative Decoding — NPU Benchmark\n");
    printf("=========================================\n");
    printf("Model:  %s\n", kModelPath);
    printf("XCLBINs: %s\n", kXclbinDir);
    printf("Prompt: %d tokens\n", prompt_len);
    printf("Generate: %d tokens\n\n", max_new);

    printf("Loading target model + 4 GEMM xclbins...\n");
    auto t_load = std::chrono::steady_clock::now();

    SpecDecodeConfig spec_cfg;
    spec_cfg.max_new_tokens = max_new;
    // NPUQwen3Target::NC=28, H=1024 — keep target_layer_ids within [0,27].
    NPUQwen3Target target(kModelPath, kXclbinDir, spec_cfg.target_layer_ids, spec_cfg.num_target_layers);

    printf("  Loaded in %.0fms\n\n",
           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load).count());

    MTPDraftConfig draft_cfg;
    MTPDraftModel draft_model(draft_cfg);
    bool draft_trained = draft_model.load_weights(kDraftCheckpoint);
    if (!draft_trained) {
        // Hard-fail instead of silently falling back to the untrained passthrough path
        // (mtp_draft.h always predicts token 0/1 with no weights loaded, which guarantees
        // ~0% acceptance and previously got misreported as a real speculative-decode result).
        fprintf(stderr, "ERROR: failed to load draft checkpoint at %s — refusing to run an "
                        "untrained benchmark. Fix kDraftCheckpoint or train a checkpoint first.\n",
                kDraftCheckpoint);
        return 1;
    }
    printf("Loaded trained draft checkpoint: %s\n", kDraftCheckpoint);

    SpeculativeDecoder decoder(target, draft_model, spec_cfg);

    // Chat-template prompt matching the production engine's default (bos, im_start, etc).
    int chat_prompt[] = {151643, 872, 198, 11852, 151644, 198, 151643, 77091, 198};
    prompt_len = prompt_len < 9 ? prompt_len : 9;
    std::vector<int32_t> prompt(chat_prompt, chat_prompt + prompt_len);
    std::vector<int32_t> output(max_new + prompt_len);

    if (draft_trained) {
        printf("Generating with trained Eagle3 draft (%s)...\n", kDraftCheckpoint);
    } else {
        printf("Generating (draft weights are untrained — expect ~0%% acceptance until an Eagle3\n");
        printf("checkpoint is trained via train_draft.py; this run validates the NPU dispatch path)...\n");
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    int generated = decoder.generate(prompt.data(), prompt_len, output.data(), max_new);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    int new_tokens = generated - prompt_len;
    auto& s = decoder.stats();

    printf("\n=== Results ===\n");
    printf("Generated %d tokens in %.0f ms = %.1f tok/s\n", new_tokens, ms, new_tokens / (ms / 1000.0));
    printf("Acceptance rate: %.1f%%  |  Effective speedup: %.2fx\n",
           s.acceptance_rate() * 100, s.speedup_factor());
    printf("Vs non-speculative baseline: ~94-97 tok/s\n");

    return 0;
}
