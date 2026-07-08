/**
 * recapture_prompt_features.cpp — Re-run NPU on PROMPT-ONLY prefixes
 * to get features from the correct (prompt boundary) position for MTP training.
 *
 * Reads npu_inputs_1000.bin, finds the <|im_start|>assistant boundary (token 77091),
 * runs NPU forward only on the prompt prefix, and saves features from the last
 * prompt position.
 */
#include "../engine/npu_target_model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

static constexpr int H = 1024, NC = 28, NV = 151936;
static constexpr int NUM_TL = 5;
static constexpr int TL_IDS[NUM_TL] = {1, 6, 12, 18, 24};
static const char* kModel = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
static const char* kXclbin = "/home/bcloud/npu-sandbox/npu-infer/build/int8";

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    
    const char* input_path = "/home/bcloud/spec-decode/npu_inputs_1000.bin";
    const char* output_path = "/home/bcloud/spec-decode/npu_hidden_prompt_1000.bin";
    int max_examples = 200;  // Quick run with 200 samples
    
    printf("═══ Recapture Prompt Features ═══\n");
    
    // Read inputs
    FILE* in_f = fopen(input_path, "rb");
    if (!in_f) { perror("fopen"); return 1; }
    
    int32_t num_examples;
    fread(&num_examples, 4, 1, in_f);
    if (max_examples && num_examples > max_examples) num_examples = max_examples;
    
    struct Example { std::vector<int32_t> tokens; };
    std::vector<Example> examples;
    
    for (int i = 0; i < num_examples; i++) {
        int32_t len;
        if (fread(&len, 4, 1, in_f) != 1) break;
        Example ex;
        ex.tokens.resize(len);
        if (fread(ex.tokens.data(), 4, len, in_f) != (size_t)len) break;
        examples.push_back(std::move(ex));
    }
    fclose(in_f);
    printf("Read %zu examples\n", examples.size());
    
    // Find prompt boundary (token 77091 = "assistant" after <|im_start|>)
    int skipped = 0, no_boundary = 0;
    for (auto& ex : examples) {
        // Find first occurrence of 151644 (assistant <|im_start|>) followed by 77091
        int boundary = -1;
        for (int j = 0; j < (int)ex.tokens.size() - 1; j++) {
            if (ex.tokens[j] == 151644 && ex.tokens[j+1] == 77091) {
                boundary = j;  // Position of <|im_start|> before "assistant"
                break;
            }
        }
        if (boundary < 1) {
            no_boundary++;
            continue;
        }
        // Truncate to prompt-only (everything before the assistant marker)
        ex.tokens.resize(boundary);
        if (ex.tokens.size() < 2) { skipped++; continue; }
    }
    printf("After boundary detection: %zu examples (%d skipped, %d no-boundary)\n",
           examples.size() - skipped - no_boundary, skipped, no_boundary);
    // Remove invalid
    examples.erase(std::remove_if(examples.begin(), examples.end(),
        [](const Example& e) { return e.tokens.size() < 2; }), examples.end());
    printf("Valid: %zu examples\n", examples.size());
    
    // Load NPU
    printf("Loading NPU...\n");
    auto t0 = std::chrono::steady_clock::now();
    NPUQwen3Target target(kModel, kXclbin, TL_IDS, NUM_TL);
    printf("  Loaded in %.0f ms\n", std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count());
    
    // Run forward on prompt-only prefixes
    FILE* out_f = fopen(output_path, "wb");
    int32_t out_count = (int32_t)examples.size();
    fwrite(&out_count, 4, 1, out_f);
    
    std::vector<float> logits(NV), all_hidden((size_t)NC * H);
    int processed = 0;
    auto start = std::chrono::steady_clock::now();
    
    for (auto& ex : examples) {
        target.forward(ex.tokens.data(), (int)ex.tokens.size(), logits.data(), all_hidden.data());
        
        // Extract target layer features (interleaved format to match training)
        std::vector<float> features((size_t)NUM_TL * H);
        for (int d = 0; d < H; d++)
            for (int li = 0; li < NUM_TL; li++)
                features[(size_t)d * NUM_TL + li] = all_hidden[(size_t)TL_IDS[li] * H + d];
        
        int32_t sl = (int32_t)ex.tokens.size();
        fwrite(&sl, 4, 1, out_f);
        fwrite(ex.tokens.data(), 4, sl, out_f);
        int32_t nl = NUM_TL, hs = H;
        fwrite(&nl, 4, 1, out_f);
        fwrite(&hs, 4, 1, out_f);
        fwrite(features.data(), 4, features.size(), out_f);
        
        processed++;
        if (processed % 20 == 0) {
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            printf("  %d/%zu (%.0fs, %.1f ex/min)\n", processed, examples.size(), s, processed/(s/60.0));
        }
    }
    fclose(out_f);
    
    double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    printf("Done: %d examples in %.0fs (%.1f ex/min)\n", processed, total, processed/(total/60.0));
    return 0;
}
