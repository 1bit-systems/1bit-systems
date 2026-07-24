// backend_cuda.cpp — CUDA GPU backend
//
// Wraps the CUDA engine (cuda_engine.cu) into the Backend interface.
// Mirrors backend_hip.cpp in structure and semantics.

#include "backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>

#include "cuda_engine.h"

// ── CUDA Backend implementation ──
struct CudaBackend : Backend {
    CudaState* cs = nullptr;
    CudaConfig ccfg;
    std::vector<float> embed, iscale, ibias;
    float* logits_buf = nullptr;
    int pos = 0;

    CudaBackend() { type = BackendType::CUDA_GPU; name = "CUDA GPU (NVIDIA)"; }

    ~CudaBackend() override { destroy(); }

    bool init(const ModelConfig& cfg, const std::string& weights_dir) override {
        this->cfg = cfg;

        // Build CudaConfig from the model's actual dimensions
        ccfg = CudaConfig::from_model(
            cfg.hidden_size > 0 ? cfg.hidden_size : cfg.hidden,
            cfg.num_layers > 0 ? cfg.num_layers : cfg.n_layers,
            cfg.num_heads > 0 ? cfg.num_heads : cfg.n_heads,
            cfg.num_kv_heads > 0 ? cfg.num_kv_heads : cfg.n_kv_heads,
            cfg.head_dim,
            cfg.vocab_size > 0 ? cfg.vocab_size : cfg.vocab,
            cfg.num_experts > 0 ? cfg.num_experts : 16,
            cfg.intermediate_size > 0 ? cfg.intermediate_size : (cfg.hidden_size > 0 ? cfg.hidden_size : cfg.hidden),
            cfg.router_hidden > 0 ? cfg.router_hidden : 256);

        printf("CUDA: Initializing engine (H=%d, L=%d, NH=%d, NKV=%d, V=%d)...\n",
               ccfg.h, ccfg.n_layers, ccfg.nq, ccfg.nkv, ccfg.vocab);

        // Ensure trailing slash
        std::string wd = weights_dir;
        if (!wd.empty() && wd.back() != '/') wd += '/';

        cs = cuda_init(wd.c_str(), &ccfg);
        if (!cs) { fprintf(stderr, "CUDA: cuda_init failed\n"); return false; }

        // Keep copies for lm_head
        embed = cs->embed;
        iscale = cs->ibias;    // Note: ibias is reused as iscale in this context
        ibias = cs->ibias;

        try {
            logits_buf = new float[ccfg.vocab];
        } catch (std::bad_alloc&) {
            fprintf(stderr, "CUDA: failed to allocate logits buffer (%zu bytes)\n",
                    size_t(ccfg.vocab) * sizeof(float));
            cuda_destroy(cs);
            return false;
        }

        initialized = true;
        printf("CUDA: Engine ready\n");
        return true;
    }

    bool reset() override {
        if (!cs) return false;
        cuda_reset(cs);
        pos = 0;
        return true;
    }

    bool forward(int token_id, float* hidden_out) override {
        (void)token_id; (void)hidden_out;
        // Same limitation as HIP backend: forward()/lm_head() split not wired
        // for CUDA. generate() works. See issue #82.
        static bool warned = false;
        if (!warned) {
            fprintf(stderr, "CUDA Backend: forward() not implemented (generate() works)\n");
            warned = true;
        }
        return false;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        (void)hidden; (void)logits; (void)argmax;
        static bool warned = false;
        if (!warned) {
            fprintf(stderr, "CUDA Backend: lm_head() not implemented (generate() works)\n");
            warned = true;
        }
        return false;
    }

    int generate(int token_id) override {
        if (!cs || !initialized) return -1;
        return cuda_forward_greedy(cs, token_id);
    }

    float benchmark(int tokens = 10) override {
        if (!cs) return 0;
        reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        int tok = 100;
        for (int i = 0; i < tokens; i++) {
            tok = cuda_forward_greedy(cs, tok);
        }
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return ms / tokens;
    }

    void destroy() override {
        if (cs) cuda_destroy(cs);
        delete[] logits_buf;
        cs = nullptr;
        initialized = false;
    }
};

extern "C" Backend* create_cuda_backend() { return new CudaBackend(); }
