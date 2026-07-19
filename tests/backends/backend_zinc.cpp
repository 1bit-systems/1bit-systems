// backend_zinc.cpp — ZINC Vulkan GPU backend (InferenceBackend wrapper)
// Part of the unified zaya_server binary. Wraps the src/backend_zinc.cpp ZINC engine.
// ZINC is a multi-arch, multi-quant Vulkan compute engine (github.com/zolotukhin/zinc).
#include "backend.h"
#include <cstdio>
#include <dlfcn.h>

// ZINC is accessed via dlopen/dlsym — no link-time dependency.
class ZincInferenceBackend : public InferenceBackend {
    void* zinc_ctx_ = nullptr;
    void* zinc_lib_ = nullptr;
    void (*zinc_destroy_fn_)(void*) = nullptr;
    bool loaded_ = false;

public:
    BackendType type() const override { return BackendType::ZINC_GPU; }
    const char* name() const override { return "ZINC GPU (Vulkan)"; }
    float estimated_tok_s() const override { return 22.0f; }
    bool is_coherent() const override { return true; }

    bool is_available() override {
        // ZINC requires libzinc.so at runtime — check via dlopen
        void* h = dlopen("libzinc.so", RTLD_NOW | RTLD_GLOBAL);
        if (h) { dlclose(h); return true; }
        // Try from zig-out path
        h = dlopen("/home/bcloud/zinc/zig-out/lib/libzinc.so", RTLD_NOW | RTLD_GLOBAL);
        if (h) { dlclose(h); return true; }
        return false;
    }

    bool load_model(const ModelConfig& cfg) override {
        unload_model();
        zinc_lib_ = dlopen("/home/bcloud/zinc/zig-out/lib/libzinc.so", RTLD_NOW | RTLD_GLOBAL);
        if (!zinc_lib_) return false;
        auto zinc_create_fn = (void*(*)())dlsym(zinc_lib_, "zinc_create");
        auto zinc_load_fn = (bool(*)(void*,const char*))dlsym(zinc_lib_, "zinc_load");
        zinc_destroy_fn_ = (void(*)(void*))dlsym(zinc_lib_, "zinc_destroy");
        if (!zinc_create_fn || !zinc_load_fn || !zinc_destroy_fn_) { dlclose(zinc_lib_); zinc_lib_ = nullptr; return false; }
        if (cfg.model_path.empty()) { dlclose(zinc_lib_); zinc_lib_ = nullptr; return false; }
        zinc_ctx_ = zinc_create_fn();
        if (!zinc_ctx_) { dlclose(zinc_lib_); zinc_lib_ = nullptr; return false; }
        if (!zinc_load_fn(zinc_ctx_, cfg.model_path.c_str())) {
            zinc_destroy_fn_(zinc_ctx_);
            zinc_ctx_ = nullptr;
            return false;
        }
        loaded_ = true;
        fprintf(stderr, "  ZINC: model loaded (%s)\n", cfg.model_name.c_str());
        return true;
    }

    void unload_model() override {
        if (zinc_ctx_ && zinc_destroy_fn_) {
            zinc_destroy_fn_(zinc_ctx_);
            zinc_ctx_ = nullptr;
        }
        if (zinc_lib_) { dlclose(zinc_lib_); zinc_lib_ = nullptr; }
        loaded_ = false;
    }

    void reset_state() override {}

    int forward(int token_id, int pos) override {
        // ZINC does full-sequence inference — not per-token.
        // This backend is used via generate(), not forward().
        return 0;
    }
};

std::vector<InferenceBackend*> detect_backends_zinc() {
    static ZincInferenceBackend backend;
    // Only include if ZINC is available (libzinc.so exists)
    if (backend.is_available()) return {&backend};
    return {};
}
