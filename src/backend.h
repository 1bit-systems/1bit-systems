// backend.h — Canonical backend interface
//
// Uses the canonical BackendType and ModelConfig from include/common.h.
// This is the CANONICAL backend interface used by BackendManager.
// The simplified InferenceBackend (tests/ version) is a parallel interface
// that shares the same types via include/common.h.
//
// Each backend implements: init, forward, lm_head, generate, benchmark, destroy

#pragma once
#include "common.h"
#include <string>
#include <vector>

// BackendType and ModelConfig are defined in include/common.h

// ── Backend interface ──
// All backends must implement this. Ops return true on success.
struct Backend {
    BackendType type = BackendType::NONE;
    std::string name;
    ModelConfig cfg;
    bool initialized = false;

    virtual ~Backend() = default;

    /// Initialize backend: detect hardware, load weights, allocate memory.
    /// weights_dir = path to /tmp/zaya_weights/ or equivalent
    virtual bool init(const ModelConfig& cfg, const std::string& weights_dir) = 0;

    /// Reset KV cache and router state for a new sequence.
    virtual bool reset() = 0;

    /// Run one token through all 40 layers.
    /// token_id = input token, hidden_out[hidden] = output hidden state
    virtual bool forward(int token_id, float* hidden_out) = 0;

    /// Compute lm_head: logits[vocab] = hidden[hidden] @ embed[vocab×hidden]^T
    virtual bool lm_head(const float* hidden, float* logits, int* argmax) = 0;

    /// Generate one token (forward + lm_head in one call).
    /// Returns the predicted token ID, -1 on error.
    virtual int generate(int token_id) = 0;

    /// Clean up resources.
    virtual void destroy() = 0;

    /// Benchmark: run N iterations, return ms/token.
    virtual float benchmark(int tokens = 10) = 0;

    /// True if this backend can actually run inference (forward/lm_head/generate).
    /// Defaults to true; stub backends that only detect hardware override to false
    /// so BackendManager discovers them but never selects them for inference (#82).
    virtual bool can_infer() const { return true; }
};

// ── Factory: auto-detect and create best available backend ──
Backend* create_best_backend();
Backend* create_backend(BackendType type);

// ── CPU backend ──
Backend* create_cpu_backend();
Backend* create_generic_backend();

// ── Vulkan backend ──
Backend* create_vulkan_backend();

// ── HIP backend ──
extern "C" Backend* create_hip_backend();

// ── NPU backend ──
Backend* create_npu_backend();

// ── NPU via FastFlowLM subprocess (see docs/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md
// for why this exists instead of the in-process NPU kernels) ──
extern "C" Backend* create_flm_backend();

// ── ZINC backend (general GGUF, multi-arch/multi-quant, via libzinc.so) ──
Backend* create_zinc_backend();

// ── Zamba2 backend ──
Backend* create_zamba2_backend();

// ── Auto-detect ──
BackendType detect_backends();
