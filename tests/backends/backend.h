// backend.h — Inference backend interface (tests/ version)
//
// Uses the canonical BackendType and ModelConfig from include/common.h.
// This file provides the simplified InferenceBackend interface used by
// zaya_server. The canonical src::Backend interface is in src/backend.h.
//
// Key difference:
//   InferenceBackend::forward() = fuse forward+lm_head -> returns token_id
//   src::Backend::forward()     = returns hidden state, separate lm_head+generate
//
#pragma once
#include "../../include/common.h"
#include <vector>
#include <string>
#include <cstdio>

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
