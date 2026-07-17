// backend_cpu_plugin.cpp — Sample backend plugin (like an ONNX Runtime EP)
// Demonstrates the plugin ABI: compile into a .so and load at runtime.
// Self-contained CPU backend implementation (does not include backend_cpu.cpp).
//
// Build:
//   g++ -shared -fPIC -O3 -std=c++17 -I include \
//       backends/backend_cpu_plugin.cpp \
//       -o backends/zaya_cpu_backend.so
//
// Run:
//   ./build/backend_demo  (auto-discovers plugins in ./backends/)

#include "backend.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

// ── Minimal CPU backend (reference implementation) ──
// A real plugin would load weights and run the model.
// This demo plugin shows the architecture: ONNX Runtime EP pattern.

struct PluginCPUBackend : Backend {
    std::vector<float> cache;
    bool loaded = false;

    PluginCPUBackend() {
        type = BackendType::CPU_SCALAR;
        name = "CPU Plugin (EP demo)";
    }

    ~PluginCPUBackend() override { destroy(); }

    bool init(const ModelConfig& cfg, const std::string& weights_dir) override {
        this->cfg = cfg;
        printf("PluginCPU: init (cfg: H=%d L=%d V=%d)\n",
               cfg.hidden, cfg.n_layers, cfg.vocab);

        // Real implementation would load weights from weights_dir
        cache.resize(cfg.hidden);
        loaded = true;
        initialized = true;
        return true;
    }

    bool reset() override {
        printf("PluginCPU: reset\n");
        return true;
    }

    bool forward(int token_id, float* hidden_out) override {
        (void)token_id;
        // Real implementation: run all 40 layers
        // Demo: fill hidden with a constant
        for (int i = 0; i < cfg.hidden; i++)
            hidden_out[i] = 0.5f;
        return true;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        (void)hidden;
        // Demo: return token 100
        if (logits) memset(logits, 0, cfg.vocab * sizeof(float));
        if (argmax) *argmax = 100;
        return true;
    }

    int generate(int token_id) override {
        (void)token_id;
        // Demo: always predict token 42
        return 42;
    }

    float benchmark(int tokens = 10) override {
        (void)tokens;
        // Demo: claim 42 ms/tok
        return 42.0f;
    }

    void destroy() override {
        cache.clear();
        loaded = false;
        initialized = false;
    }
};

// ── Plugin ABI (v1) ──
// These symbols are what BackendPluginLoader looks for via dlsym.
extern "C" {

Backend* create_backend() {
    printf("Plugin CPU: create_backend() called\n");
    return new PluginCPUBackend();
}

void destroy_backend(Backend* b) {
    if (b) {
        b->destroy();
        delete b;
        printf("Plugin CPU: destroy_backend() done\n");
    }
}

int backend_type() {
    return static_cast<int>(BackendType::CPU_SCALAR);
}

const char* backend_id() {
    return "cpu_plugin_v1";
}

const char* backend_version() {
    return "1.0.0";
}

const char* backend_description() {
    return "CPU Backend Plugin (ORT EP demo)";
}

} // extern "C"
