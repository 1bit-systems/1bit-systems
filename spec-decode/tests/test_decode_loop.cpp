// Minimal spec decode loop test
#include "../draft/mtp_draft.h"
#include "../engine/spec_decode.h"
#include <cstdio>
#include <chrono>

int main() {
    printf("Minimal spec decode test\n");
    
    MTPDraftConfig cfg;
    MTPDraftModel draft(cfg);
    MTPDraftState state;
    state.resize(8, 128, 4096);
    
    // Setup
    float trunk[5*1024] = {0.5f};
    int32_t last_token_id = 1;
    static float logits_out[50000] = {0.0f};
    float hidden_out[1024] = {0.0f};

    auto t0 = std::chrono::high_resolution_clock::now();
    draft.forward(trunk, last_token_id, /*pos=*/0, state, logits_out, hidden_out);
    auto t1 = std::chrono::high_resolution_clock::now();
    
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    printf("Draft forward: %.0f us\n", us);
    printf("logits[0] = %.2f (expected ~0.5)\n", logits_out[0]);
    
    // Full decoder test
    printf("\nFull decoder test...\n");
    
    struct SimpleTarget : TargetModelInterface {
        void forward(const int32_t*, int32_t, float* l, float* h) override {
            l[0] = 10.0f; // always predict token 0
            for (int i = 1; i < 50000; i++) l[i] = -1.0f;
            for (int i = 0; i < 28*1024; i++) h[i] = 0.1f;
        }
        void forward_with_kv(const int32_t*, int32_t, int32_t, float* l, float* h) override {
            forward(nullptr, 0, l, h);
        }
        void get_layer_hidden(const float* all, int32_t, const int32_t*, int32_t nt, float* out) override {
            for (int i = 0; i < nt * 1024; i++) out[i] = all[i];
        }
    };
    
    SimpleTarget st;
    MTPDraftModel dm(cfg);
    SpecDecodeConfig sc;
    sc.vocab_size = 50000;
    sc.max_new_tokens = 256;
    sc.block_size = 7;
    sc.hidden_size = 1024;
    sc.num_kv_heads = 8;
    sc.head_dim = 128;
    sc.max_seq = 4096;
    
    SpeculativeDecoder dec(st, dm, sc);
    
    int32_t prompt[128] = {0};
    int32_t output[384] = {0};
    
    t0 = std::chrono::high_resolution_clock::now();
    int n = dec.generate(prompt, 128, output, 256);
    t1 = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Generated %d tokens in %.0f ms\n", n, ms);
    printf("Stats: accept_rate=%.1f%% speedup=%.2fx calls=%ld\n",
           dec.stats().acceptance_rate() * 100,
           dec.stats().speedup_factor(),
           (long)dec.stats().verify_calls);
    
    return 0;
}
