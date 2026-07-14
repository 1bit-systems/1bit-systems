// backend.h — Unified inference backend interface
// One binary compiles all backends; runtime auto-detects the best available.
#pragma once
#include <vector>
#include <string>
#include <cstdio>

enum class BackendType {
    HIP,        // AMD ROCm HIP (Radeon / RDNA 3.5+)
    Vulkan,     // Cross-platform GPU (GLSL compute shaders)
    NPU,        // AMD XDNA 2 NPU (Strix Halo)
    CPU,        // x86-64 scalar fallback (always available)
};

struct ModelConfig {
    int hidden_size       = 2048;
    int num_heads         = 16;
    int num_kv_heads      = 2;
    int head_dim          = 128;
    int num_layers        = 40;
    int vocab_size        = 262272;
    int intermediate_size = 2048;
    int num_experts       = 16;
    int num_experts_top   = 17;
    int router_hidden     = 256;
    float rope_theta      = 500000.0f;
    float rms_norm_eps    = 1e-5f;
    int max_seq_len       = 2048;
    std::string model_name = "unknown";
    std::string model_path;
    std::string weights_dir;
};

struct InferenceResult {
    std::vector<int> tokens;
    std::string text;
    float gen_ms = 0;
    float tok_s = 0;
};

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

// Detect all available backends on this hardware
std::vector<InferenceBackend*> detect_backends();

// Pick the fastest available backend
InferenceBackend* select_best_backend();

// Simple JSON helpers shared across backends
namespace json_helpers {
    inline std::string get_str(const std::string& j, const std::string& k) {
        auto p = j.find("\"" + k + "\"");
        if (p == std::string::npos) return "";
        p = j.find(':', p);
        if (p == std::string::npos) return "";
        p = j.find_first_of("\"", p);
        if (p == std::string::npos || j[p] != '\"') {
            auto ns = j.find_first_of("-0123456789", p + 1);
            if (ns != std::string::npos) {
                auto ne = j.find_first_not_of("0123456789.e-+", ns);
                return j.substr(ns, ne - ns);
            }
            return "";
        }
        auto e = j.find('\"', p + 1);
        if (e == std::string::npos) return "";
        return j.substr(p + 1, e - p - 1);
    }
    inline int get_int(const std::string& j, const std::string& k, int d = 0) {
        auto s = get_str(j, k);
        if (s.empty()) return d;
        return atoi(s.c_str());
    }
    inline float get_float(const std::string& j, const std::string& k, float d = 0.0f) {
        auto s = get_str(j, k);
        if (s.empty()) return d;
        return (float)atof(s.c_str());
    }
}
