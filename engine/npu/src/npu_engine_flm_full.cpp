// npu_engine_flm_full.cpp — Full inference using FastFlowLM .so, with real token sampling
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
#include "modules/sampler.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

// Simple BPE tokenizer (minimal — enough to test)
// Uses the model directory's tokenizer files
#include "tokenizer/tokenizer.hpp"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model_dir> [prompt]\n", argv[0]); return 1; }
    
    std::string model_path = argv[1];
    std::string prompt = argc > 2 ? argv[2] : "The capital of France is";
    int gen_tokens = 32;
    
    fprintf(stderr, "Loading: %s\n", model_path.c_str());
    
    // Init NPU
    xrt::device dev(0);
    auto npu = std::make_unique<npu_xclbin_manager>(device_npu2, &dev, false);
    
    // Load config
    LM_Config config;
    config.from_pretrained(model_path);
    
    // Create engine
    fprintf(stderr, "Creating engine...\n");
    auto engine = std::make_unique<qwen3_npu>(config, npu.get(), 4096);
    
    // Load weights
    fprintf(stderr, "Loading weights...\n");
    Q4NX q4nx(model_path);
    engine->load_weights(q4nx);
    engine->clear_context();
    
    // Sampler
    sampler_config scfg;
    scfg.top_k = 10; scfg.top_p = 0.9; scfg.min_p = 0.1; scfg.temperature = 0.3;
    Sampler sampler(config.vocab_size, scfg);
    
    // Tokenizer
    fprintf(stderr, "Loading tokenizer...\n");
    Tokenizer tokenizer(model_path);
    
    // Encode prompt
    fprintf(stderr, "Tokenizing: %s\n", prompt.c_str());
    auto tokens = tokenizer.encode(prompt);
    fprintf(stderr, "  %zu tokens\n", tokens.size());
    
    // Prefill
    fprintf(stderr, "\nPrefill...\n");
    auto t0 = std::chrono::steady_clock::now();
    auto logits = engine->prefill(tokens, nullptr);
    auto prefill_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "  %.0fms\n", prefill_ms);
    
    // Sample first token
    int token = sampler.sample(logits); if (token < 0) token = 0;
    fprintf(stderr, "First token: %d\n", token);
    
    // Decode loop
    fprintf(stderr, "\nDecode %d tokens...\n", gen_tokens);
    std::string output;
    auto t1 = std::chrono::steady_clock::now();
    
    for (int i = 0; i < gen_tokens; i++) {
        logits = engine->forward(token);
        token = sampler.sample(logits);
        auto piece = tokenizer.decode(std::vector<int>{token});
        output += piece;
        printf("%s", piece.c_str());
        fflush(stdout);
    }
    
    auto decode_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t1).count();
    
    fprintf(stderr, "\n\n=== Results ===\n");
    fprintf(stderr, "Prefill: %.0fms\n", prefill_ms);
    fprintf(stderr, "Decode:  %.0fms for %d tokens\n", decode_ms, gen_tokens);
    fprintf(stderr, "Speed:   %.1f tok/s\n", gen_tokens / (decode_ms / 1000.0));
    fprintf(stderr, "Output:  %s\n", output.c_str());
    
    return 0;
}
