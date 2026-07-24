// diffusion_bridge.h — stable-diffusion.cpp integration layer
//
// Embeds stable-diffusion.cpp and exposes its image/video generation
// capabilities through OpenAI-compatible endpoints.
//
// Uses the struct-based sd.cpp C API (stable-diffusion.h):
//   new_sd_ctx() / free_sd_ctx() / generate_image() / generate_video()
//
// All model types supported: SD1.x, SDXL, SD3, FLUX, Wan, LTX, etc.

#pragma once

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <cstdint>

// sd.cpp types (stable-diffusion.h is in include path of sd.cpp submodule)
#include "stable-diffusion.h"

// ─── Generation parameters ────────────────────────────────────────
struct DiffusionParams {
    std::string prompt;
    std::string negative_prompt;
    int width              = 512;
    int height             = 512;
    int steps              = 20;
    float cfg_scale        = 7.0f;
    int seed               = -1;
    
    // LoRA
    std::vector<std::string> lora_paths;
    std::vector<float> lora_weights;
    
    // Control / conditioning
    std::string control_net_path;
    float control_strength  = 1.0f;
    std::string control_image_path;
    
    // IP-Adapter
    std::string ip_adapter_path;
    std::string ip_adapter_image;
    float ip_adapter_strength = 0.5f;
    
    // Image-to-image
    std::string init_image_path;
    float strength            = 0.75f;
    std::string mask_image_path;
    
    // Output format
    std::string output_format = "png";
    int output_quality        = 95;
    
    // Video params
    int video_frames          = 81;
    int video_fps             = 16;
    std::string video_output_format = "mp4";
};

// ─── Generated result ─────────────────────────────────────────────
struct DiffusionResult {
    std::vector<uint8_t> data;
    std::string mime_type;
    int width  = 0;
    int height = 0;
    int frames = 1;
    int64_t generation_time_ms = 0;
    int seed_used = -1;
};

// ─── Progress callback ────────────────────────────────────────────
using DiffusionProgressFn = std::function<void(int step, int total)>;

// ─── Diffusion engine ─────────────────────────────────────────────
class DiffusionEngine {
public:
    DiffusionEngine();
    ~DiffusionEngine();
    
    DiffusionEngine(const DiffusionEngine&) = delete;
    DiffusionEngine& operator=(const DiffusionEngine&) = delete;
    
    // ── Model lifecycle ──
    bool load_model(const std::string& model_path,
                    const std::string& vae_path = "");
    void unload_model();
    bool is_loaded() const;
    bool supports_video() const;
    std::string model_path() const { return model_path_; }

    // ── Generation ──
    DiffusionResult txt2img(const DiffusionParams& params,
                            DiffusionProgressFn progress = nullptr);
    DiffusionResult img2img(const DiffusionParams& params,
                            DiffusionProgressFn progress = nullptr);
    DiffusionResult txt2vid(const DiffusionParams& params,
                            DiffusionProgressFn progress = nullptr);
    DiffusionResult img2vid(const DiffusionParams& params,
                            DiffusionProgressFn progress = nullptr);

    // ── Upscaling ──
    bool load_upscaler(const std::string& path);
    DiffusionResult upscale(const uint8_t* rgb, int w, int h, int factor = 2);

    // ── LoRA ──
    bool load_lora(const std::string& path, float weight = 1.0f);
    void clear_loras();

private:
    std::string model_path_;
    std::string vae_path_;
    sd_ctx_t* sd_ctx_     = nullptr;
    upscaler_ctx_t* up_ctx_ = nullptr;
};

// ─── Singleton ────────────────────────────────────────────────────
DiffusionEngine& diffusion_engine();
