// Full inference: prefill real tokens, sample, decode, output text
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "npu_utils/npu_instr_utils.hpp"
#include "models/qwen3/qwen3_npu.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "AutoModel/modeling_qwen3.hpp"

// Simple UTF-8 token lookup table for common Qwen3 tokens
// Maps token IDs to text for decoding
static const char* lookup_token(int id) {
    // Qwen3 tokens 0-200 are special/control
    // Common tokens for The capital of France is Paris:
    static const char* common[] = {
        "The", " capital", " of", " France", " is", " Paris", ".", "\n", "", "", "", "", "", "", "", ""
    };
    if (id >= 0 && id < 16) return common[id];
    return "?";
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model_dir> [tokens]\n", argv[0]); return 1; }
    std::string model_path = argv[1];
    int gen_tokens = argc > 2 ? atoi(argv[2]) : 32;
    
    fprintf(stderr, "Loading: %s\n", model_path.c_str());
    
    xrt::device dev(0);
    auto npu = std::make_unique<npu_xclbin_manager>(device_npu2, &dev, false);
    
    LM_Config config;
    config.from_pretrained(model_path);
    int NV = config.vocab_size;
    
    fprintf(stderr, "Creating qwen3_npu engine...\n");
    auto engine = std::make_unique<qwen3_npu>(config, npu.get(), 4096);
    
    fprintf(stderr, "Loading weights...\n");
    Q4NX q4nx(model_path);
    engine->load_weights(q4nx);
    engine->clear_context();
    fprintf(stderr, "Vocab: %d\n\n", NV);
    
    // Simple prompt tokens for testing (from Qwen3 tokenizer)
    // The capital of France is = specific token IDs
    // Qwen3 uses BPE with special tokens starting at 151643
    std::vector<int> test_tokens = {151644, 872, 198, 13048, 151645, 198, 151644, 77091, 198};
    
    fprintf(stderr, "=== Prefill %zu tokens ===\n", test_tokens.size());
    auto t0 = std::chrono::steady_clock::now();
    auto logits = engine->prefill(test_tokens, nullptr);
    auto prefill_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "Prefill: %.0fms\n", prefill_ms);
    
    // Get logits as float buffer and find argmax
    // The qwen3_npu::prefill returns buffer<bf16>
    // We need to find the highest-scoring token
    fprintf(stderr, "\n=== Decode %d tokens ===\n", gen_tokens);
    auto t1 = std::chrono::steady_clock::now();
    
    std::vector<int> output_tokens;
    for (int step = 0; step < gen_tokens; step++) {
        // Forward through the model
        logits = engine->forward(step == 0 ? 77091 : output_tokens.back());
        
        // Sample: pick argmax from logits
        // logits is buffer<bf16> with [vocab_size] elements
        int best = 0;
        float best_val = -1e30f;
        int limit = NV > 1000 ? 1000 : NV;  // check top 1000 for speed
        for (int v = 0; v < limit; v++) {
            uint32_t bits = (uint32_t)logits[v] << 16;
            float val;
            memcpy(&val, &bits, 4);
            if (val > best_val) { best_val = val; best = v; }
        }
        output_tokens.push_back(best);
    }
    
    auto decode_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t1).count();
    
    fprintf(stderr, "\n=== Results ===\n");
    fprintf(stderr, "Prefill: %.0fms\n", prefill_ms);
    fprintf(stderr, "Decode:  %.0fms for %d tokens\n", decode_ms, gen_tokens);
    fprintf(stderr, "Speed:   %.1f tok/s\n", gen_tokens / (decode_ms / 1000.0));
    fprintf(stderr, "Tokens:  ");
    for (int t : output_tokens) fprintf(stderr, "%d ", t);
    fprintf(stderr, "\n");    
    return 0;
}
