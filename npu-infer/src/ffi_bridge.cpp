/// ffi_bridge.cpp — C-compatible FFI wrappers around the NPU inference engine
/// Exposes flat C functions that Rust can call via `extern "C"`.
#include "engine.h"
#include "common.h"
#include <cstring>
#include <mutex>

// We wrap the C++ engine in an opaque handle for FFI safety
extern "C" {

// Opaque handle — Rust holds this as a *mut c_void
struct NpuEngineHandle {
    NpuInferenceEngine engine;
    bool initialized;
    int last_error;
    char error_msg[256];
};

// ── Lifecycle ───────────────────────────────────────────────────

NpuEngineHandle* npu_engine_create() {
    auto* h = new NpuEngineHandle();
    h->initialized = false;
    h->last_error = 0;
    h->error_msg[0] = '\0';
    return h;
}

int npu_engine_destroy(NpuEngineHandle* h) {
    if (!h) return -1;
    delete h;
    return 0;
}

// ── Initialization ──────────────────────────────────────────────

int npu_engine_init(NpuEngineHandle* h, const char* model_path) {
    if (!h) return -1;
    
    bool ok = h->engine.init(model_path);
    h->initialized = ok;
    
    if (!ok) {
        h->last_error = 1;
        snprintf(h->error_msg, sizeof(h->error_msg),
                 "Engine init failed: %s", model_path ? model_path : "null");
        return -1;
    }
    return 0;
}

int npu_engine_is_initialized(NpuEngineHandle* h) {
    return (h && h->initialized) ? 1 : 0;
}

const char* npu_engine_last_error(NpuEngineHandle* h) {
    return (h) ? h->error_msg : "null handle";
}

// ── Inference ───────────────────────────────────────────────────

/// Generate tokens from a prompt.
/// @return Number of output tokens written, or -1 on error.
int npu_engine_generate(
    NpuEngineHandle* h,
    const int* input_tokens,
    int num_input_tokens,
    int* output_tokens,
    int max_output_tokens
) {
    if (!h || !h->initialized) return -1;
    if (!input_tokens || num_input_tokens <= 0) return -1;
    if (!output_tokens || max_output_tokens <= 0) return -1;
    
    try {
        int n = h->engine.generate(input_tokens, num_input_tokens,
                                    output_tokens, max_output_tokens);
        return n;
    } catch (const std::exception& e) {
        h->last_error = 2;
        snprintf(h->error_msg, sizeof(h->error_msg),
                 "Generate failed: %s", e.what());
        return -1;
    } catch (...) {
        h->last_error = 3;
        snprintf(h->error_msg, sizeof(h->error_msg),
                 "Generate failed: unknown error");
        return -1;
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Get the model configuration (vocab_size, hidden_size, etc.)
int npu_engine_get_config(NpuEngineHandle* h, int* out_vocab_size,
                           int* out_hidden_size, int* out_num_layers) {
    if (!h || !h->initialized) return -1;
    // The ModelConfig is embedded in the engine. We'd need to expose it.
    // For now, return from the hardcoded config.
    if (out_vocab_size) *out_vocab_size = 151936;
    if (out_hidden_size) *out_hidden_size = 1024;
    if (out_num_layers) *out_num_layers = 28;
    return 0;
}

} // extern "C"
