/**
 * NPU Speculative Decoding Integration
 * 
 * Bridges the speculative decoding engine with the 1bit NPU inference stack.
 * 
 * Architecture:
 *   ┌──────────────────────────────────────────────────────┐
 *   │  spec_decode.h — orchestrator                        │
 *   │  ┌─────────────┐   ┌──────────────────────────────┐  │
 *   │  │ Draft Model  │   │ Target Model (NPU via XRT)  │  │
 *   │  │ (1 layer)    │──►│ Qwen3-0.6B on XDNA 2       │  │
 *   │  │ ~8.5M params ├──►│                             │  │
 *   │  │ ON NPU       │   │ Rejection sampling          │  │
 *   │  └─────────────┘   └──────────────────────────────┘  │
 *   └──────────────────────────────────────────────────────┘
 * 
 * Integration points:
 *   1. After prefill: extract hidden states from N target layers
 *   2. Draft: run tiny 1-layer MTP draft on NPU (fused kernel)
 *   3. Verify: target model forward on prompt + draft tokens
 *   4. Accept/reject: greedy comparison (temp=0) or rejection sampling
 * 
 * Build:
 *   g++ -std=c++23 -O3 -o npu_spec_decode engine/npu_spec_integration.cpp \
 *       -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -lm -ldl
 */

#include "spec_decode.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

// XRT headers for NPU interaction
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

class NPUTargetModel : public TargetModelInterface {
public:
    NPUTargetModel(const char* xclbin_path, const char* kernel_name) {
        // Open NPU device
        device_ = xrt::device(0);
        xclbin_uuid_ = device_.load_xclbin(xclbin_path);
        
        // Get kernel handles
        prefill_kernel_ = xrt::kernel(device_, xclbin_uuid_, "prefill");
        decode_kernel_ = xrt::kernel(device_, xclbin_uuid_, "decode");
        spec_kernel_ = xrt::kernel(device_, xclbin_uuid_, "spec_decode");
        
        // Allocate buffers
        input_bo_ = xrt::bo(device_, 4096 * sizeof(int32_t), XCL_BO_FLAGS_NONE,
                            prefill_kernel_.group_id(0));
        logits_bo_ = xrt::bo(device_, 151936 * sizeof(float), XCL_BO_FLAGS_NONE,
                             prefill_kernel_.group_id(1));
        hidden_bo_ = xrt::bo(device_, 28 * 1024 * sizeof(float), XCL_BO_FLAGS_NONE,
                             prefill_kernel_.group_id(2));
        
        printf("[NPU] Device opened, XCLBIN loaded\n");
    }

    void forward(const int32_t* input_ids, int32_t seq_len,
                 float* logits, float* hidden_states) override {
        // Upload input
        input_bo_.write(input_ids, seq_len * sizeof(int32_t));
        input_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        
        // Run prefill kernel
        auto run = prefill_kernel_(input_bo_, seq_len, logits_bo_, hidden_bo_);
        run.wait();
        
        // Download results
        logits_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        logits_bo_.read(logits, 151936 * sizeof(float));
        
        hidden_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        hidden_bo_.read(hidden_states, 28 * 1024 * sizeof(float));
    }

    void forward_with_kv(const int32_t* input_ids, int32_t n_tokens,
                          int32_t past_len, float* logits,
                          float* hidden_states) override {
        // Upload input
        input_bo_.write(input_ids, n_tokens * sizeof(int32_t));
        input_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        
        // Run decode kernel (uses KV cache from past context)
        auto run = decode_kernel_(input_bo_, n_tokens, past_len,
                                  logits_bo_, hidden_bo_);
        run.wait();
        
        // Download results
        logits_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        logits_bo_.read(logits, 151936 * sizeof(float) * n_tokens);
        
        hidden_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        hidden_bo_.read(hidden_states, 28 * 1024 * sizeof(float) * n_tokens);
    }

    void get_layer_hidden(const float* all_hidden, int32_t num_layers,
                           const int32_t* target_layer_ids,
                           int32_t num_target_layers,
                           float* out) override {
        const int H = 1024; // hidden_size
        for (int i = 0; i < num_target_layers; i++) {
            int layer = target_layer_ids[i];
            std::memcpy(out + i * H, all_hidden + layer * H, H * sizeof(float));
        }
    }

private:
    xrt::device device_;
    xrt::uuid xclbin_uuid_;
    xrt::kernel prefill_kernel_;
    xrt::kernel decode_kernel_;
    xrt::kernel spec_kernel_;
    xrt::bo input_bo_;
    xrt::bo logits_bo_;
    xrt::bo hidden_bo_;
};


// Benchmark harness
int main(int argc, char* argv[]) {
    const char* xclbin_path = argc > 1 ? argv[1] : "/home/bcloud/npu-sandbox/npu-infer/build/int8/int8.xclbin";
    int prompt_len = argc > 2 ? atoi(argv[2]) : 128;
    int max_new = argc > 3 ? atoi(argv[3]) : 512;
    
    printf("1bit Speculative Decoding — NPU Benchmark\n");
    printf("=========================================\n");
    printf("XCLBIN: %s\n", xclbin_path);
    printf("Prompt: %d tokens\n", prompt_len);
    printf("Generate: %d tokens\n\n", max_new);
    
    // Initialize
    MTPDraftConfig draft_cfg;
    MTPDraftWeights draft_weights;
    MTPDraftModel draft_model(draft_cfg);
    
    SpecDecodeConfig spec_cfg;
    spec_cfg.max_new_tokens = max_new;
    
    NPUTargetModel target(xclbin_path, "qwen3_0_6b");
    SpeculativeDecoder decoder(target, draft_model, spec_cfg);
    
    // Dummy prompt (in production, from tokenizer)
    std::vector<int32_t> prompt(prompt_len, 151643); // bos_token_id
    std::vector<int32_t> output(max_new + prompt_len);
    
    // Warmup
    printf("Warmup...\n");
    decoder.generate(prompt.data(), prompt_len, output.data(), 64);
    
    // Benchmark
    printf("Benchmarking...\n");
    int num_runs = 5;
    double total_time = 0.0;
    int total_tokens = 0;
    
    for (int run = 0; run < num_runs; run++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        int generated = decoder.generate(
            prompt.data(), prompt_len, output.data(), max_new
        );
        auto t1 = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        int new_tokens = generated - prompt_len;
        total_time += ms;
        total_tokens += new_tokens;
        
        auto& s = decoder.stats();
        printf("  Run %d: %d tokens in %.0f ms = %.1f tok/s | "
               "accept=%.1f%% speedup=%.2fx\n",
               run + 1, new_tokens, ms, new_tokens / (ms / 1000.0),
               s.acceptance_rate() * 100, s.speedup_factor());
    }
    
    printf("\n=== Results ===\n");
    printf("Average: %.1f tok/s\n", total_tokens / (total_time / 1000.0));
    printf("Vs baseline (no spec): ~94 tok/s\n");
    printf("Speedup vs baseline: %.2fx\n",
           total_tokens / (total_time / 1000.0) / 94.0);
    
    return 0;
}
