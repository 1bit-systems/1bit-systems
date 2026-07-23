// backend_hip.cpp — HIP GPU backend via Zaya engine
// Wraps the zaya_engine HIP kernels into the Backend interface.
// Architecture is now runtime-configurable: builds ZayaConfig from ModelConfig
// and passes it to zaya_init(). Kernels that use compile-time shared memory
// (CCA attention, MoE router) still require Zaya1-8B dimensions; the engine
// validates this at init time and fails gracefully for unsupported configs.

#include "backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>

// ── Zaya engine API (declared in zaya_engine.h, compiled in zaya_engine.cpp) ──
#include "zaya_engine.h"

// ── HIP Backend implementation ──
struct HIPBackend : Backend {
    ZayaState* zs = nullptr;
    ZayaConfig zcfg;
    std::vector<float> embed, iscale, ibias;
    float* logits_buf = nullptr;
    int pos = 0;

    HIPBackend() { type = BackendType::HIP_GPU; name = "HIP GPU (ROCm)"; }

    ~HIPBackend() override { destroy(); }

    bool init(const ModelConfig& cfg, const std::string& weights_dir) override {
        this->cfg = cfg;

        // Build ZayaConfig from the model's actual dimensions.
        // Single-token inference kernels (CCA prep, router) are fully dynamic.
        // Batch-path kernels (zaya_router_moe.hip, zaya_moe_batch_union.hip)
        // still have architecture-specific shared memory and are only used
        // for speculative decode — the main generate() path is unaffected.
        zcfg = ZayaConfig::from_model(
            cfg.hidden_size > 0 ? cfg.hidden_size : cfg.hidden,
            cfg.num_layers > 0 ? cfg.num_layers : cfg.n_layers,
            cfg.num_heads > 0 ? cfg.num_heads : cfg.n_heads,
            cfg.num_kv_heads > 0 ? cfg.num_kv_heads : cfg.n_kv_heads,
            cfg.head_dim,
            cfg.vocab_size > 0 ? cfg.vocab_size : cfg.vocab,
            /*n_exp=*/cfg.num_experts > 0 ? cfg.num_experts : 16,
            /*n_ff=*/cfg.intermediate_size > 0 ? cfg.intermediate_size : (cfg.hidden_size > 0 ? cfg.hidden_size : cfg.hidden),
            /*rtr_h=*/cfg.router_hidden > 0 ? cfg.router_hidden : 256);

        printf("HIP: Initializing Zaya engine (H=%d, L=%d, NH=%d, NKV=%d, V=%d)...\n",
               zcfg.h, zcfg.n_layers, zcfg.nq, zcfg.nkv, zcfg.vocab);

        // Ensure trailing slash for zaya_engine.cpp's filename concatenation
        std::string wd = weights_dir;
        if (!wd.empty() && wd.back() != '/') wd += '/';
        zs = zaya_init(wd.c_str(), &zcfg);
        if (!zs) { fprintf(stderr,"HIP: zaya_init failed\n"); return false; }

        // Apply optional LoRA adapter after base weights are loaded
        if (!cfg.lora_path.empty()) {
            printf("HIP: Applying LoRA adapter from %s...\n", cfg.lora_path.c_str());
            if (zaya_apply_lora(zs, cfg.lora_path.c_str()) != 0) {
                fprintf(stderr, "HIP: zaya_apply_lora failed — continuing without LoRA\n");
            } else {
                printf("HIP: LoRA adapter applied successfully\n");
            }
        }

        // Keep copies for lm_head (tied embeddings) — loaded by zaya_init
        embed = zs->embed;
        iscale = zs->iscale;
        ibias = zs->ibias;

        try {
            logits_buf = new float[zcfg.vocab];
        } catch (std::bad_alloc&) {
            fprintf(stderr, "HIP: failed to allocate logits buffer (%zu bytes)\n",
                    size_t(zcfg.vocab) * sizeof(float));
            zaya_destroy(zs);
            return false;
        }
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
        // zaya_forward_greedy does GPU argmax — copies only 4 bytes instead of
        // full vocab logits. Avoids the full-logit copy (fixes #64).
        return zaya_forward_greedy(zs, token_id);
    }

    float benchmark(int tokens = 10) override {
        if (!zs) return 0;
        reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        int tok = 100;
        for (int i = 0; i < tokens; i++) {
            tok = zaya_forward_greedy(zs, tok);
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

extern "C" Backend* create_hip_backend() { return new HIPBackend(); }
