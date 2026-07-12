// backend_hip.cpp — HIP GPU backend for Zaya1-8B
// Wraps the existing zaya_engine HIP kernels into the Backend interface.

#include "backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>

// ── Include the existing Zaya HIP engine ──
// This includes all kernel definitions from zaya_cca_attn.hip and zaya_gpu_router.hip
#include "zaya_engine.cpp"  // reuse the whole C-callable HIP engine

// ── HIP Backend implementation ──
struct HIPBackend : Backend {
    ZayaState* zs = nullptr;
    std::vector<float> embed, fnorm, iscale, ibias;
    float* logits_buf = nullptr;
    int pos = 0;

    HIPBackend() { type = BackendType::HIP_GPU; name = "HIP GPU (ROCm)"; }

    ~HIPBackend() override { destroy(); }

    bool init(const ModelConfig& cfg, const std::string& weights_dir) override {
        this->cfg = cfg;
        printf("HIP: Initializing Zaya engine...\n");
        zs = zaya_init();  // loads weights, allocates GPU memory
        if (!zs) { fprintf(stderr,"HIP: zaya_init failed\n"); return false; }

        // Keep copies for lm_head (tied embeddings)
        embed = zs->embed;
        fnorm = load_bin(weights_dir + "/model_norm_weight.bin");
        iscale = zs->iscale;
        ibias = zs->ibias;

        logits_buf = new float[VOCAB];
        initialized = true;
        printf("HIP: Engine ready\n");
        return true;
    }

    bool reset() override {
        if (!zs) return false;
        zaya_reset(zs);
        pos = 0;
        return true;
    }

    bool forward(int token_id, float* hidden_out) override {
        (void)token_id; (void)hidden_out;
        // The Backend forward()/lm_head() split is not wired for HIP: zaya_forward
        // fuses forward+lm_head and does not expose the pre-head hidden state, so the
        // split needs a zaya API change to implement correctly. generate() (the path
        // BackendManager actually uses) IS implemented. Fail loudly here (fixes #82).
        static bool warned = false;
        if (!warned) { fprintf(stderr, "HIP Backend: forward() not implemented on the adapter (generate() works); see #82\n"); warned = true; }
        return false;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        (void)hidden; (void)logits; (void)argmax;
        static bool warned = false;
        if (!warned) { fprintf(stderr, "HIP Backend: lm_head() not implemented on the adapter (generate() works); see #82\n"); warned = true; }
        return false;
    }

    int generate(int token_id) override {
        if (!zs || !initialized) return -1;
        zaya_forward(zs, token_id, logits_buf);
        int best = 0;
        for (int i = 1; i < VOCAB; i++) if (logits_buf[i] > logits_buf[best]) best = i;
        return best;
    }

    float benchmark(int tokens = 10) override {
        if (!zs) return 0;
        reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        // Use the server's forward function which does full pipeline
        int tok = 100;
        for (int i = 0; i < tokens; i++) {
            zaya_forward(zs, tok, logits_buf);
            tok = 0;
            for (int v = 1; v < VOCAB; v++) if (logits_buf[v] > logits_buf[tok]) tok = v;
        }
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return ms / tokens;
    }

    void destroy() override {
        if (zs) zaya_destroy(zs);
        delete[] logits_buf;
        zs = nullptr;
        initialized = false;
    }
};

Backend* create_hip_backend() { return new HIPBackend(); }
