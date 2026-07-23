// backend_hip_adapter.cpp — Adapter from src::Backend → InferenceBackend
//
// Wraps create_hip_backend() (src/backend_hip.cpp + src/zaya_engine.cpp)
// into the InferenceBackend interface used by zaya_server.
// Replaces the old tests/backends/backend_hip.cpp which duplicated kernels.
//
// Build: compiled as HIP, linked with src/backend_hip.cpp + src/zaya_engine.cpp

// Pull in common types + canonical Backend. We avoid including
// tests/backends/backend.h directly to dodge detect_backends() overload
// conflicts. Instead we define InferenceBackend manually below.
#include "../../include/common.h"     // ModelConfig, BackendType, InferenceResult
#include "../../src/backend.h"         // Backend struct (canonical)

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

// ── InferenceBackend interface (mirrors tests/backends/backend.h) ──
// Declared here instead of including tests/backends/backend.h to avoid
// the detect_backends() overload collision with src/backend.h.
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

// Factory from src/backend_hip.cpp (linked into the same target)
extern "C" Backend* create_hip_backend();

// ── Adapter: wraps Backend* into InferenceBackend ──
class HipBackendAdapter : public InferenceBackend {
    Backend* backend_ = nullptr;
    ModelConfig cfg_;
    bool loaded_ = false;

public:
    HipBackendAdapter() = default;
    ~HipBackendAdapter() override { unload_model(); }

    BackendType type() const override { return BackendType::HIP_GPU; }
    const char* name() const override { return "ROCm HIP (Zaya)"; }

    bool is_available() override {
        int count = 0;
        hipError_t e = hipGetDeviceCount(&count);
        if (e != hipSuccess || count == 0) return false;
        hipDeviceProp_t props;
        if (hipGetDeviceProperties(&props, 0) != hipSuccess) return false;
        fprintf(stderr, "  HIP: found %s (%d CU, %zu MB VRAM)\n",
                props.name, props.multiProcessorCount,
                props.totalGlobalMem / (1024 * 1024));
        return true;
    }

    bool load_model(const ModelConfig& cfg) override {
        cfg_ = cfg;
        unload_model();

        backend_ = create_hip_backend();
        if (!backend_) {
            fprintf(stderr, "  HIP adapter: create_hip_backend() failed\n");
            return false;
        }

        std::string wd = cfg.weights_dir;
        if (!wd.empty() && wd.back() != '/') wd += '/';

        if (!backend_->init(cfg, wd)) {
            fprintf(stderr, "  HIP adapter: backend init failed\n");
            delete backend_;
            backend_ = nullptr;
            return false;
        }

        loaded_ = true;
        fprintf(stderr, "  HIP: loaded via adapter (H=%d L=%d V=%d)\n",
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

    float estimated_tok_s() const override { return 64.0f; }
    bool is_coherent() const override { return true; }
};

// ── Detection entry point (called by backend_cpu.cpp's detect_backends()) ──
std::vector<InferenceBackend*> detect_backends_hip() {
    std::vector<InferenceBackend*> backends;
    static HipBackendAdapter hip;
    backends.push_back(&hip);
    return backends;
}
