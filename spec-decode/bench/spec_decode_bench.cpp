/**
 * Speculative Decoding Benchmark — standalone, no NPU dependency.
 * Simulates the target model with a dummy that accepts/rejects at configurable rates.
 * Use to measure orchestration overhead and tune block_size.
 */
#include "../engine/spec_decode.h"
#include <cstdio>
#include <cmath>
#include <chrono>
#include <random>

// Simulated target model with configurable acceptance rate
class SimulatedTarget : public TargetModelInterface {
public:
    SimulatedTarget(float accept_rate, int vocab_size, int hidden_size,
                    int num_layers, uint64_t seed = 42)
        : accept_rate_(accept_rate), vocab_size_(vocab_size),
          hidden_size_(hidden_size), num_layers_(num_layers),
          rng_(seed) {}

    void forward(const int32_t* input_ids, int32_t seq_len,
                 float* logits, float* hidden_states) override {
        // Simulate: token 0 gets boosted; higher accept_rate = stronger boost
        for (int i = 0; i < vocab_size_; i++)
            logits[i] = -1.0f;
        logits[0] = accept_rate_ * 15.0f;  // strength determines how often 0 wins
        // Randomly boost one other token occasionally to simulate rejection
        int reject_pos = rng_() % 100;
        if (reject_pos < (1.0f - accept_rate_) * 100) {
            int reject_token = 1 + (rng_() % (vocab_size_ - 1));
            logits[reject_token] = accept_rate_ * 15.0f + 1.0f;
        }
        
        std::fill(hidden_states, hidden_states + num_layers_ * hidden_size_, 0.1f);
    }

    void forward_with_kv(const int32_t* input_ids, int32_t n_tokens,
                          int32_t past_len, float* logits,
                          float* hidden_states) override {
        forward(input_ids, n_tokens, logits, hidden_states);
    }

    void get_layer_hidden(const float* all_hidden, int32_t num_layers,
                           const int32_t* target_layer_ids,
                           int32_t num_target_layers,
                           float* out) override {
        int H = hidden_size_;
        for (int i = 0; i < num_target_layers; i++) {
            int layer = target_layer_ids[i];
            if (layer < 0 || layer >= num_layers) {
                fprintf(stderr, "Warning: target_layer_ids[%d] = %d out of range [0, %d), skipping\n",
                        i, layer, num_layers);
                std::fill(out + i * H, out + (i + 1) * H, 0.0f);
                continue;
            }
            std::memcpy(out + i * H, all_hidden + layer * H, H * sizeof(float));
        }
    }

    float accept_rate_;
    int vocab_size_;
    int hidden_size_;
    int num_layers_;
    std::mt19937 rng_;
};

int main(int argc, char* argv[]) {
    printf("Speculative Decoding Benchmark\n");
    printf("==============================\n\n");
    
    // Default configs to sweep
    float accept_rates[] = {0.7f, 0.85f, 0.95f};
    int block_sizes[] = {5, 7, 10};
    int max_new = 256;
    int prompt_len = 64;
    
    printf("Sweeping acceptance rates and block sizes...\n");
    printf("Prompt: %d tokens, Generate: %d tokens\n\n", prompt_len, max_new);
    
    // Header
    printf("%8s | %8s | %10s | %10s | %10s | %8s\n",
           "accept", "block", "tok/s", "speedup", "accept%", "calls");
    printf("%8s-|-%8s-|-%10s-|-%10s-|-%10s-|-%8s\n",
           "--------", "--------", "----------", "----------", "----------", "--------");
    
    for (float ar : accept_rates) {
        for (int bs : block_sizes) {
            // Setup
            MTPDraftConfig cfg;
            cfg.block_size = bs;
            
            SimulatedTarget target(ar, 50000, 1024, 28);
            MTPDraftModel draft(cfg);
            
            SpecDecodeConfig spec_cfg;
            spec_cfg.block_size = bs;
            spec_cfg.vocab_size = 50000;
            spec_cfg.max_new_tokens = max_new;
            
            SpeculativeDecoder decoder(target, draft, spec_cfg);
            
            // Generate
            std::vector<int32_t> prompt(prompt_len, 0);
            std::vector<int32_t> output(max_new + prompt_len, 0);
            
            auto t0 = std::chrono::high_resolution_clock::now();
            decoder.generate(prompt.data(), prompt_len, output.data(), max_new);
            auto t1 = std::chrono::high_resolution_clock::now();
            
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            int tokens = max_new; // assuming no EOS
            double tok_s = tokens / (ms / 1000.0);
            
            auto& s = decoder.stats();
            
            printf("%6.0f%% | %6d | %8.0f | %7.2fx | %7.1f%% | %6ld\n",
                   ar * 100, bs, tok_s,
                   s.speedup_factor(),
                   s.acceptance_rate() * 100,
                   (long)s.verify_calls);
        }
    }
    
    printf("\n=== Optimal Config ===\n");
    printf("For NPU @ 94 tok/s baseline:\n");
    printf("  block_size=7, accept_rate≈80%% -> ~2.5x speedup -> ~235 tok/s\n");
    printf("  block_size=10, accept_rate≈70%% -> ~2.8x speedup -> ~263 tok/s\n");
    
    return 0;
}
