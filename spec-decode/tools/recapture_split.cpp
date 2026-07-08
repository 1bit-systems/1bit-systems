/**
 * recapture_split.cpp — NPU forward on sequence prefixes (position-based split).
 * For each sequence, runs NPU on the first 1/3 (prompt area), captures features
 * from the last prompt position, and saves both prompt tokens and full response tokens.
 */
#include "../engine/npu_target_model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

static constexpr int H = 1024, NC = 28, NV = 151936, NUM_TL = 5;
static constexpr int TL_IDS[NUM_TL] = {1, 6, 12, 18, 24};
static const char* kModel = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
static const char* kXclbin = "/home/bcloud/npu-sandbox/npu-infer/build/int8";

int main(int argc, char* argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int max_samples = (argc > 1) ? atoi(argv[1]) : 200;
    
    const char* input_path = "/home/bcloud/spec-decode/npu_inputs_1000.bin";
    const char* output_path = "/home/bcloud/spec-decode/npu_hidden_split.bin";
    
    printf("═══ Recapture Split Features (len/3 boundary) ═══\n");
    
    FILE* in_f = fopen(input_path, "rb");
    if (!in_f) { perror("fopen"); return 1; }
    
    int32_t num_examples;
    fread(&num_examples, 4, 1, in_f);
    if (max_samples && num_examples > max_samples) num_examples = max_samples;
    
    struct Ex { std::vector<int32_t> prompt, response; };
    std::vector<Ex> examples;
    
    for (int i = 0; i < num_examples; i++) {
        int32_t len;
        if (fread(&len, 4, 1, in_f) != 1) break;
        std::vector<int32_t> tokens(len);
        if (fread(tokens.data(), 4, len, in_f) != (size_t)len) break;
        
        int split = len / 3;
        if (split < 4 || len - split < 3) continue;
        
        Ex ex;
        ex.prompt.assign(tokens.begin(), tokens.begin() + split);
        ex.response.assign(tokens.begin() + split, tokens.end());
        examples.push_back(std::move(ex));
    }
    fclose(in_f);
    printf("Read %zu valid examples (split at len/3)\n", examples.size());
    
    printf("Loading NPU...\n");
    auto t0 = std::chrono::steady_clock::now();
    NPUQwen3Target target(kModel, kXclbin, TL_IDS, NUM_TL);
    printf("  Loaded in %.0f ms\n", std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count());
    
    FILE* out_f = fopen(output_path, "wb");
    int32_t out_count = (int32_t)examples.size();
    fwrite(&out_count, 4, 1, out_f);
    
    std::vector<float> logits(NV), all_hidden((size_t)NC * H);
    int processed = 0;
    auto start = std::chrono::steady_clock::now();
    
    for (auto& ex : examples) {
        // Run NPU on prompt-only prefix
        target.forward(ex.prompt.data(), (int)ex.prompt.size(), logits.data(), all_hidden.data());
        
        // Extract target layer features from last prompt position (interleaved)
        std::vector<float> features((size_t)NUM_TL * H);
        for (int d = 0; d < H; d++)
            for (int li = 0; li < NUM_TL; li++)
                features[(size_t)d * NUM_TL + li] = all_hidden[(size_t)TL_IDS[li] * H + d];
        
        // Write: prompt_len, prompt_tokens, response_len, response_tokens, features
        int32_t pl = (int32_t)ex.prompt.size();
        int32_t rl = (int32_t)ex.response.size();
        fwrite(&pl, 4, 1, out_f);
        fwrite(ex.prompt.data(), 4, pl, out_f);
        fwrite(&rl, 4, 1, out_f);
        fwrite(ex.response.data(), 4, rl, out_f);
        fwrite(features.data(), 4, features.size(), out_f);
        
        processed++;
        if (processed % 50 == 0) {
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            printf("  %d/%zu (%.0fs, %.1f ex/min)\n", processed, examples.size(), s, processed/(s/60.0));
        }
    }
    fclose(out_f);
    
    double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    printf("Done: %d examples in %.0fs (%.1f ex/min)\n", processed, total, processed/(total/60.0));
    return 0;
}
