// backend_zamba2_adapter.cpp — Zamba2 backend adapter
// Wraps create_zamba2_backend() (src/backend_zamba2.cpp) into the
// InferenceBackend interface used by zaya_server.
//
// Build: compiled as C++17, linked with src/backend_zamba2.cpp

#include "../../include/common.h"     // ModelConfig, BackendType, InferenceResult
#include "../../src/backend.h"         // Backend struct (canonical)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

// ── InferenceBackend interface (mirrors tests/backends/backend.h) ──
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;
    virtual BackendType type() const = 0;
    virtual const char* name() const = 0;
    virtual bool is_available() = 0;
    virtual bool load_model(const ModelConfig& cfg) = 0;
    virtual void unload_model() = 0;
    virtual int forward(int token_id, int pos) = 0;
    virtual void reset_state() = 0;
    virtual float estimated_tok_s() const { return 0; }
    virtual bool is_coherent() const { return true; }
};

// Factory from src/backend_zamba2.cpp (linked into librocm_cpp.so)
extern "C" Backend* create_zamba2_backend();

// ── Adapter: wraps Backend* into InferenceBackend ──
class Zamba2BackendAdapter : public InferenceBackend {
    Backend* backend_ = nullptr;
    ModelConfig cfg_;
    bool loaded_ = false;

public:
    Zamba2BackendAdapter() = default;
    ~Zamba2BackendAdapter() override { unload_model(); }

    BackendType type() const override { return BackendType::GENERIC; }
    const char* name() const override { return "Zamba2 GPU"; }

    bool is_available() override {
        fprintf(stderr, "  Zamba2 adapter: checking availability\n");
        return true;
    }

    bool load_model(const ModelConfig& cfg) override {
        cfg_ = cfg;
        unload_model();

        fprintf(stderr, "  Zamba2 adapter: load_model called, arch=%d (want RCPP_ARCH_ZAMBA2=%d), name=%s\n",
                (int)cfg.arch, (int)RCPP_ARCH_ZAMBA2, cfg.model_name.c_str());

        // Only handle Zamba2 architecture
        if (cfg.arch != RCPP_ARCH_ZAMBA2) {
            fprintf(stderr, "  Zamba2 adapter: arch mismatch (%d != %d), skipping\n",
                    (int)cfg.arch, (int)RCPP_ARCH_ZAMBA2);
            return false;
        }

        backend_ = create_zamba2_backend();
        if (!backend_) {
            fprintf(stderr, "  Zamba2 adapter: create_zamba2_backend() failed\n");
            return false;
        }

        std::string wd = cfg.weights_dir;
        if (!wd.empty() && wd.back() != '/') wd += '/';

        if (!backend_->init(cfg, wd)) {
            fprintf(stderr, "  Zamba2 adapter: backend init failed\n");
            delete backend_;
            backend_ = nullptr;
            return false;
        }

        loaded_ = true;
        fprintf(stderr, "  Zamba2: loaded via adapter (H=%d L=%d V=%d)\n",
                cfg.hidden_size, cfg.num_layers, cfg.vocab_size);
        return true;
    }

    void unload_model() override {
        if (backend_) {
            backend_->destroy();
            delete backend_;
            backend_ = nullptr;
        }
        loaded_ = false;
    }

    int forward(int token_id, int pos) override {
        (void)pos;
        if (!backend_ || !loaded_) return -1;
        return backend_->generate(token_id);
    }

    void reset_state() override {
        if (backend_) backend_->reset();
    }

    float estimated_tok_s() const override { return 40.0f; }
    bool is_coherent() const override { return true; }
};

// ── Detection entry point ──
std::vector<InferenceBackend*> detect_backends_zamba2() {
    std::vector<InferenceBackend*> backends;
    static Zamba2BackendAdapter zamba2;
    backends.push_back(&zamba2);
    return backends;
}
