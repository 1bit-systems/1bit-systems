// tools/run_zamba2.cpp — Standalone Zamba2 inference test
//
// Build: g++ -std=c++17 -O3 -I. -o run_zamba2 \
//   tools/run_zamba2.cpp src/mamba2_kernels.cpp src/zamba2_engine.cpp
//
// Run:   ./run_zamba2 /path/to/zamba2.gguf "Hello, world!"
//
// This is a minimal test that loads a Zamba2 model from GGUF and
// generates text. It does not depend on the 1bit.systems engine,
// making it safe to compile on any machine with a C++17 compiler.

#include "../src/zamba2_engine.h"
#include "../src/gguf_zamba2_loader.cpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>

// ── Minimal BPE tokenizer (Mistral v0.1 compatible) ──
// For now, we use a simple byte-level fallback.
// In production, use the tokenizer from the GGUF file or HuggingFace tokenizers.
struct SimpleTokenizer {
    std::vector<std::string> id_to_token;
    int vocab_size = 32000;
    int bos = 1, eos = 2;

    void init(int vocab) {
        vocab_size = vocab;
        id_to_token.resize(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            id_to_token[i] = "<" + std::to_string(i) + ">";
        }
        id_to_token[bos] = "<s>";
        id_to_token[eos] = "</s>";
    }

    std::vector<int> encode(const std::string& text) {
        // Stub: just use byte-level fallback
        std::vector<int> ids = {bos};
        for (char c : text) {
            ids.push_back((unsigned char)c + 3); // shift to avoid special tokens
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) {
        std::string out;
        for (int id : ids) {
            if (id > 0 && id < vocab_size) {
                // Skip special tokens for display
                if (id >= 3) {
                    out += (char)(id - 3);
                }
            }
        }
        return out;
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [prompt]\n", argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string prompt = argc > 2 ? argv[2] : "Hello, world!";

    fprintf(stderr, "╔══════════════════════════════════════╗\n");
    fprintf(stderr, "║   Zamba2 Inference Test              ║\n");
    fprintf(stderr, "╚══════════════════════════════════════╝\n");
    fprintf(stderr, "Model: %s\n", model_path.c_str());
    fprintf(stderr, "Prompt: %s\n\n", prompt.c_str());

    // ── Load model ──
    Zamba2Model model;
    if (!load_zamba2_from_gguf(model_path, model)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // ── Init tokenizer ──
    SimpleTokenizer tokenizer;
    tokenizer.init(model.cfg.vocab_size);

    // ── Encode prompt ──
    auto input_ids = tokenizer.encode(prompt);
    fprintf(stderr, "Input tokens: ");
    for (int id : input_ids) fprintf(stderr, "%d ", id);
    fprintf(stderr, "\n\n");

    // ── Generate ──
    const int max_tokens = 100;
    std::vector<int> output_ids;
    std::vector<float> logits(model.cfg.vocab_size);

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < max_tokens; ++t) {
        int token_id;
        if (t < (int)input_ids.size()) {
            token_id = input_ids[t];
        } else {
            // Argmax
            token_id = 0;
            float max_val = logits[0];
            for (int i = 1; i < model.cfg.vocab_size; ++i) {
                if (logits[i] > max_val) {
                    max_val = logits[i];
                    token_id = i;
                }
            }
        }

        // Forward
        if (!model.forward(token_id, logits.data())) {
            fprintf(stderr, "Forward failed at token %d\n", t);
            break;
        }

        if (t >= (int)input_ids.size()) {
            output_ids.push_back(token_id);

            // Print as we go
            std::string chunk = tokenizer.decode({token_id});
            printf("%s", chunk.c_str());
            fflush(stdout);

            // Stop at EOS
            if (token_id == tokenizer.eos) break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(end - start).count();
    int gen_tokens = (int)output_ids.size();
    float tok_per_sec = gen_tokens > 0 ? (gen_tokens / (ms / 1000.0f)) : 0;

    printf("\n\n╔══════════════════════════════════════╗\n");
    printf("║   Generation Complete                ║\n");
    printf("╠══════════════════════════════════════╣\n");
    printf("║ Tokens generated: %d                  ║\n", gen_tokens);
    printf("║ Time: %.1f ms                        ║\n", ms);
    printf("║ Speed: %.1f tok/s                    ║\n", tok_per_sec);
    printf("╚══════════════════════════════════════╝\n");

    return 0;
}
