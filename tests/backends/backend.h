// backend.h — Inference backend interface (tests/ version)
//
// NOTE: This is a SIMPLIFIED interface for zaya_server and test backends.
// The CANONICAL backend interface lives in src/backend.h (struct Backend).
// This file is NOT merged with src/backend.h due to conflicting function
// signatures (detect_backends() returns different types).
//
// Key difference:
//   InferenceBackend::forward() = fuse forward+lm_head -> returns token_id
//   src::Backend::forward()     = returns hidden state, separate lm_head()+generate()
//
// BackendType values MUST match src/backend.h (verified by static_assert below).
// TODO(#unify): reconcile detect_backends() signatures
#pragma once
#include <vector>
#include <string>
#include <cstdio>
#include <cstdint>

// ── Backend type ──
// Values MUST match src/backend.h (HIP_GPU=1, VULKAN=2, NPU_XRT=3, CPU_AVX512=4).
enum class BackendType : uint8_t {
    HIP = 1,        // AMD ROCm GPU via HIP (canonical: HIP_GPU)
    Vulkan = 2,     // Any Vulkan 1.2+ GPU (canonical: VULKAN)
    NPU = 3,        // AMD XDNA NPU via XRT (canonical: NPU_XRT)
    CPU = 4,        // CPU with AVX-512 (canonical: CPU_AVX512)
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
