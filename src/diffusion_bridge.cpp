// diffusion_bridge.cpp — stable-diffusion.cpp integration (correct C API)
// Uses the struct-based C API from stable-diffusion.h

#include "diffusion_bridge.h"
#include "stable-diffusion.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <vector>

// ─── Engine lifecycle ─────────────────────────────────────────────

DiffusionEngine::DiffusionEngine() = default;

DiffusionEngine::~DiffusionEngine() {
    unload_model();
}

bool DiffusionEngine::load_model(const std::string& model_path,
                                  const std::string& vae_path) {
    unload_model();
    if (model_path.empty()) return false;
    
    model_path_ = model_path;
    vae_path_ = vae_path;
    
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    
    params.model_path = model_path.c_str();
    params.vae_path = vae_path.empty() ? nullptr : vae_path.c_str();
    params.n_threads = 8;
    params.wtype = SD_TYPE_F32;
    params.rng_type = CUDA_RNG;
    params.flash_attn = true;
    
    sd_ctx_ = new_sd_ctx(&params);
    if (!sd_ctx_) {
        fprintf(stderr, "diffusion: new_sd_ctx failed for %s\n",
                model_path.c_str());
        return false;
    }
    
    printf("diffusion: loaded %s\n", model_path.c_str());
    return true;
}

void DiffusionEngine::unload_model() {
    if (sd_ctx_) { ::free_sd_ctx(sd_ctx_); sd_ctx_ = nullptr; }
    if (up_ctx_) { ::free_upscaler_ctx(up_ctx_); up_ctx_ = nullptr; }
    model_path_.clear();
    vae_path_.clear();
}

bool DiffusionEngine::is_loaded() const { return sd_ctx_ != nullptr; }
bool DiffusionEngine::supports_video() const {
    return sd_ctx_ && sd_ctx_supports_video_generation(sd_ctx_);
}

// ─── Image generation ─────────────────────────────────────────────

DiffusionResult DiffusionEngine::txt2img(const DiffusionParams& params,
                                          DiffusionProgressFn progress) {
    if (!sd_ctx_) return {};
    auto t0 = std::chrono::steady_clock::now();
    
    sd_img_gen_params_t gp;
    sd_img_gen_params_init(&gp);
    
    gp.prompt = params.prompt.c_str();
    gp.negative_prompt = params.negative_prompt.empty() ? nullptr
                         : params.negative_prompt.c_str();
    gp.width = params.width;
    gp.height = params.height;
    gp.seed = params.seed;
    gp.batch_count = 1;
    gp.strength = params.cfg_scale;
    
    sd_sample_params_t sp;
    sd_sample_params_init(&sp);
    sp.sample_steps = params.steps;
    sp.sample_method = EULER_A_SAMPLE_METHOD;
    sp.scheduler = KARRAS_SCHEDULER;
    gp.sample_params = sp;
    
    // LoRAs
    std::vector<sd_lora_t> loras;
    for (size_t i = 0; i < params.lora_paths.size(); i++) {
        sd_lora_t l;
        l.path = params.lora_paths[i].c_str();
        l.multiplier = i < params.lora_weights.size() ?
                       params.lora_weights[i] : 1.0f;
        l.is_high_noise = false;
        loras.push_back(l);
    }
    gp.loras = loras.data();
    gp.lora_count = (uint32_t)loras.size();
    
    sd_image_t* images_out = nullptr;
    int num_images = 0;
    bool ok = generate_image(sd_ctx_, &gp, &images_out, &num_images);
    
    auto t1 = std::chrono::steady_clock::now();
    
    if (!ok || !images_out || num_images < 1) {
        fprintf(stderr, "diffusion: generate_image failed\n");
        return {};
    }
    
    DiffusionResult result;
    result.width = (int)images_out[0].width;
    result.height = (int)images_out[0].height;
    result.mime_type = "image/png";
    result.frames = 1;
    result.generation_time_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(t1 - t0).count();
    result.seed_used = gp.seed;
    
    // Copy image data
    size_t data_size = (size_t)images_out[0].width *
                       images_out[0].height *
                       images_out[0].channel;
    if (images_out[0].data && data_size > 0) {
        result.data.assign(images_out[0].data,
                           images_out[0].data + data_size);
    }
    
    free_sd_images(images_out, num_images);
    return result;
}

DiffusionResult DiffusionEngine::img2img(const DiffusionParams& params,
                                          DiffusionProgressFn progress) {
    if (!sd_ctx_) return {};
    auto t0 = std::chrono::steady_clock::now();
    
    sd_img_gen_params_t gp;
    sd_img_gen_params_init(&gp);
    
    gp.prompt = params.prompt.c_str();
    gp.negative_prompt = params.negative_prompt.empty() ? nullptr
                         : params.negative_prompt.c_str();
    gp.width = params.width;
    gp.height = params.height;
    gp.seed = params.seed;
    gp.strength = params.cfg_scale;
    gp.batch_count = 1;
    
    sd_sample_params_t sp;
    sd_sample_params_init(&sp);
    sp.sample_steps = params.steps;
    sp.sample_method = EULER_A_SAMPLE_METHOD;
    sp.scheduler = KARRAS_SCHEDULER;
    gp.sample_params = sp;
    
    // TODO: load init_image from params.init_image_path
    // sd_image_t init = load_image(params.init_image_path);
    // gp.init_image = init;
    // gp.strength (img2img) = params.strength;
    
    sd_image_t* images_out = nullptr;
    int num_images = 0;
    bool ok = generate_image(sd_ctx_, &gp, &images_out, &num_images);
    auto t1 = std::chrono::steady_clock::now();
    
    if (!ok || !images_out || num_images < 1) return {};
    
    DiffusionResult result;
    result.width = (int)images_out[0].width;
    result.height = (int)images_out[0].height;
    result.mime_type = "image/png";
    result.generation_time_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(t1 - t0).count();
    
    size_t sz = (size_t)images_out[0].width * images_out[0].height * images_out[0].channel;
    if (images_out[0].data && sz > 0)
        result.data.assign(images_out[0].data, images_out[0].data + sz);
    
    free_sd_images(images_out, num_images);
    return result;
}

DiffusionResult DiffusionEngine::txt2vid(const DiffusionParams& params,
                                          DiffusionProgressFn progress) {
    if (!sd_ctx_ || !supports_video()) return {};
    // TODO: video generation via generate_video()
    fprintf(stderr, "diffusion: video generation not yet wired\n");
    return {};
}

DiffusionResult DiffusionEngine::img2vid(const DiffusionParams& params,
                                          DiffusionProgressFn progress) {
    return {};
}

// ─── Upscaling ────────────────────────────────────────────────────

bool DiffusionEngine::load_upscaler(const std::string& path) {
    if (up_ctx_) { free_upscaler_ctx(up_ctx_); up_ctx_ = nullptr; }
    up_ctx_ = ::new_upscaler_ctx(path.c_str(), false, 8, 0, nullptr, nullptr);
    return up_ctx_ != nullptr;
}

DiffusionResult DiffusionEngine::upscale(const uint8_t* rgb, int w, int h,
                                          int factor) {
    if (!up_ctx_) return {};
    sd_image_t input_img = {(uint32_t)w, (uint32_t)h, 3, (uint8_t*)rgb};
    sd_image_t* result_imgs = nullptr;
    int num_out = 0;
    bool ok = ::upscale(up_ctx_, input_img, (uint32_t)factor,
                        &result_imgs, &num_out);
    if (!ok || !result_imgs || num_out < 1) return {};
    
    DiffusionResult r;
    r.width = (int)result_imgs[0].width;
    r.height = (int)result_imgs[0].height;
    r.mime_type = "image/png";
    size_t sz = (size_t)result_imgs[0].width * result_imgs[0].height * result_imgs[0].channel;
    if (result_imgs[0].data && sz > 0)
        r.data.assign(result_imgs[0].data, result_imgs[0].data + sz);
    
    free_sd_images(result_imgs, num_out);
    return r;
}

// ─── LoRA ─────────────────────────────────────────────────────────

bool DiffusionEngine::load_lora(const std::string& path, float weight) {
    printf("diffusion: lora %s weight=%.2f\n", path.c_str(), weight);
    return true;
}

void DiffusionEngine::clear_loras() {}

// ─── Singleton ────────────────────────────────────────────────────

DiffusionEngine& diffusion_engine() {
    static DiffusionEngine engine;
    return engine;
}
