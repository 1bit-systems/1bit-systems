/// backend_zinc.cpp — C++ ZINC GPU backend.
/// Fixed: #763 proper includes via CMake, shader path from build config
#include "backend.h"
#include "simple_tokenizer.h"
#include "vulkan_wrapper.h"
#include "compute_engine.h"
#include "model_loader.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <cstdlib>
#include <sys/stat.h>

// Resolve a shader directory that actually contains the compiled ZINC .spv
// files. backend_zinc.cpp is compiled into libbackend_manager (which does NOT
// get the ZINC_SHADER_DIR compile define that the standalone zinc_cpp target
// gets), so the old `#ifdef ZINC_SHADER_DIR / else "shaders"` fell back to a
// CWD-relative "shaders" that rarely exists. When no valid dir is found we
// return "" so the caller can fail the backend GRACEFULLY instead of letting
// a missing-shader std::runtime_error escape a worker thread and call
// std::terminate on the whole server.
// Every .spv the ZINC compute path dispatches (post shader_map translation in
// compute_engine.cpp). If ANY is missing the backend will throw mid-decode on
// the PILOT worker thread — an uncatchable cross-thread std::terminate — so we
// require the whole set up front and otherwise disable ZINC cleanly.
static const char* kZincRequiredShaders[] = {
    "embed", "gemv_f32", "rms_norm_mul", "rope_fused", "flash_attn",
    "swiglu", "argmax", "vadd", "copy_buffer",
};

static std::string zinc_resolve_shader_dir() {
    auto has_shaders = [](const std::string& d) -> bool {
        if (d.empty()) return false;
        struct stat st;
        return stat((d + "/embed.spv").c_str(), &st) == 0;
    };
    // 1. Explicit runtime override always wins.
    if (const char* env = std::getenv("ZINC_SHADER_DIR"); env && has_shaders(env))
        return env;
    // 2. Compile-time location (set for the zinc_cpp target's own build).
#ifdef ZINC_SHADER_DIR
    if (has_shaders(ZINC_SHADER_DIR)) return ZINC_SHADER_DIR;
#endif
    // 3. Common build / install locations, relative to CWD and typical prefixes.
    static const char* candidates[] = {
        "shaders",
        "build_cmake/zinc_cpp_build/shaders",
        "build/zinc_cpp_build/shaders",
        "engine/gpu/zinc_cpp/build/shaders",
        "engine/gpu/shaders",
        "/usr/share/1bit-systems/shaders",
        "/usr/local/share/1bit-systems/shaders",
    };
    for (const char* c : candidates)
        if (has_shaders(c)) return c;
    return "";
}

struct ZincBackend : Backend {
    std::unique_ptr<ZincEngine> engine_;
    std::unique_ptr<ComputeEngine> compute_;
    std::unique_ptr<ModelLoader> loader_;
    std::unique_ptr<ModelGPU> model_;
    std::unique_ptr<InferenceEngine> infer_;
    bool vulkan_initialized_ = false;

    ZincBackend() { type = BackendType::ZINC_GPU; name = "ZINC GPU (Vulkan, C++)"; }
    ~ZincBackend() override { destroy(); }

    bool init(const ModelConfig& model_cfg, const std::string& weights_dir) override {
        (void)weights_dir;
        cfg = model_cfg;
        destroy();

        if (cfg.model_path.empty()) {
            fprintf(stderr, "ZINC: no GGUF model path available\n");
            return false;
        }

        // The dense ZINC GPU path runs end-to-end (embed, Q4_K matmuls, RoPE,
        // GQA attention, SwiGLU, argmax) and is fast, but its crude Q4_K
        // re-quantization (no sub-block scales) still degrades quality vs the
        // full-precision cpu_generic path (#844). Opt-in only so dense models
        // keep using the correct backend by default. Set ZINC_ENABLE=1 to try.
        if (!getenv("ZINC_ENABLE")) {
            fprintf(stderr, "ZINC: dense GPU path is experimental — disabled by "
                "default (set ZINC_ENABLE=1). Using HIP/CPU. See #844.\n");
            return false;
        }

        printf("ZINC C++: initializing Vulkan...\n");

        // Validate shaders are present BEFORE we spin up Vulkan + the PILOT
        // prefetch worker thread. If they're missing, a later lazy shader load
        // throws on a worker thread and std::terminates the whole process
        // (there's no way to catch a cross-thread throw at the call site).
        // Failing here just marks ZINC unavailable and lets BackendManager
        // fall back to HIP/CPU — the server stays up.
        std::string shader_dir = zinc_resolve_shader_dir();
        if (shader_dir.empty()) {
            fprintf(stderr,
                "ZINC: compiled Vulkan shaders (embed.spv) not found — disabling ZINC "
                "backend (set ZINC_SHADER_DIR to enable). Falling back to HIP/CPU.\n");
            return false;
        }
        // Require the FULL dispatch set before we touch Vulkan / start the PILOT
        // worker — a shader that's missing but only requested mid-decode throws
        // on a worker thread and std::terminates the whole server (#844).
        {
            std::string missing;
            for (const char* s : kZincRequiredShaders) {
                struct stat st;
                if (stat((shader_dir + "/" + s + ".spv").c_str(), &st) != 0) {
                    if (!missing.empty()) missing += ", ";
                    missing += s;
                }
            }
            if (!missing.empty()) {
                fprintf(stderr,
                    "ZINC: incomplete Vulkan shader set in %s (missing: %s) — disabling "
                    "ZINC backend, falling back to HIP/CPU. See issue #844.\n",
                    shader_dir.c_str(), missing.c_str());
                return false;
            }
        }

        try {
            engine_ = std::make_unique<ZincEngine>();
            engine_->init(shader_dir, -1);

            loader_ = std::make_unique<ModelLoader>(
                engine_->device(), engine_->queue(),
                engine_->queue_family(), *engine_->cmd_pool());

            model_ = std::make_unique<ModelGPU>();
            if (!loader_->load(cfg.model_path, *model_)) {
                fprintf(stderr, "ZINC: failed to load model\n");
                return false;
            }

            compute_ = std::make_unique<ComputeEngine>(
                engine_->device(), engine_->queue(),
                engine_->queue_family(), *engine_->cmd_pool(),
                *engine_->pipeline_cache());

            infer_ = std::make_unique<InferenceEngine>();
            if (!infer_->init(*compute_, *model_)) {
                fprintf(stderr, "ZINC: failed to init inference engine\n");
                return false;
            }

            vulkan_initialized_ = true;
            initialized = true;
            printf("ZINC C++: ready — %d layers, H=%d\n",
                   model_->dims.n_layers, model_->dims.hidden);
            return true;

        } catch (const std::exception& e) {
            fprintf(stderr, "ZINC C++: init failed: %s\n", e.what());
            destroy();
            return false;
        }
    }

    bool reset() override { if (infer_) { infer_->reset(); return true; } return false; }
    bool forward(int, float*) override { return false; }
    bool lm_head(const float*, float*, int*) override { return false; }

    int generate(int token_id) override {
        if (!initialized || !infer_) return -1;
        return infer_->generate(token_id);
    }

    void destroy() override {
        infer_.reset();
        compute_.reset();
        model_.reset();
        loader_.reset();
        engine_.reset();
        vulkan_initialized_ = false;
        initialized = false;
    }

    float benchmark(int tokens) override {
        if (!initialized || !infer_) return 0.0f;
        reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        int tok = 1; // BOS (Llama convention)
        for (int i = 0; i < tokens; i++) {
            tok = infer_->generate(tok);
            if (tok < 0) break;
        }
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return tokens > 0 ? ms / tokens : 0.0f;
    }

    bool can_infer() const override { return initialized && vulkan_initialized_; }
};

Backend* create_zinc_backend() { return new ZincBackend(); }
