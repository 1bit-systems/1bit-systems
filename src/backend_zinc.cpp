/// backend_zinc.cpp — C++ ZINC GPU backend.
/// Replaces the old Zig-based ZINC backend. Uses the C++ zinc_cpp library
/// from engine/gpu/zinc_cpp/ instead of loading libzinc.so via C ABI.
///
/// ZincEngine manages Vulkan init, model loading, and GPU inference
/// through pre-compiled .spv compute shaders. Supports any GGUF model
/// through the generic GGUF pipeline (Llama, Mistral, Qwen2, Gemma, Phi,
/// etc.) without hardcoded architecture limits.
#include "backend.h"
#include "simple_tokenizer.h"
#include "../engine/gpu/zinc_cpp/include/vulkan_wrapper.h"
#include "../engine/gpu/zinc_cpp/include/compute_engine.h"
#include "../engine/gpu/zinc_cpp/include/model_loader.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <memory>

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

        if (cfg.model_path.empty() && cfg.format != ModelFormat::GGUF) {
            fprintf(stderr, "ZINC: no GGUF model path available\n");
            return false;
        }
        std::string path = cfg.model_path.empty() ? cfg.model_name : cfg.model_path;

        printf("ZINC C++: initializing Vulkan...\n");

        try {
            // 1. Init Vulkan
            engine_ = std::make_unique<ZincEngine>();
            engine_->init("shaders", -1);

            // 2. Load model weights to GPU
            loader_ = std::make_unique<ModelLoader>(
                engine_->device(), engine_->queue(),
                engine_->queue_family(), *engine_->cmd_pool());

            model_ = std::make_unique<ModelGPU>();
            if (!loader_->load(path, *model_)) {
                fprintf(stderr, "ZINC: failed to load model\n");
                return false;
            }

            // 3. Create compute engine
            compute_ = std::make_unique<ComputeEngine>(
                engine_->device(), engine_->queue(),
                engine_->queue_family(), *engine_->cmd_pool(),
                *engine_->pipeline_cache());

            // 4. Create inference engine
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

    bool reset() override {
        if (infer_) infer_->reset();
        return true;
    }

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
        int tok = 0; // BOS
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
