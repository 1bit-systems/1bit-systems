// vl_processor.h — High-level vision-language image processor.
//
// Provides a unified API for:
//   1. Image loading (stb_image, no OpenCV)
//   2. Resize + mean/std normalization (CPU or GPU-accelerated)
//   3. GPU upload/download helpers for VL model integration
//
// Usage:
//   VlProcessor proc;
//   if (!proc.load("photo.jpg", 224, 224)) { error; }
//   // pixels are in proc.pixels(), ready to feed into ViT
//   // GPU version:
//   proc.load_gpu(...);
//   // pixels are on device in proc.dev_pixels()
//
// Upstream tracking: additive file, no existing file modified.

#ifndef VL_PROCESSOR_H
#define VL_PROCESSOR_H

#include "vl_preprocess.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cstdint>
#include <utility>

// ── Optional HIP runtime for GPU upload/download ──
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define VL_HIP_AVAILABLE 1
#endif

// ── Qwen2-VL default normalization constants ──
// (used by vision_qwen2vl_poc.cpp)
static const float VL_MEAN_QWEN2VL[3] = {0.48145467f, 0.45782750f, 0.40821072f};
static const float VL_STD_QWEN2VL[3]  = {0.26862955f, 0.26130259f, 0.27577710f};

// ── CLIP / LLaVA default normalization constants ──
static const float VL_MEAN_CLIP[3]    = {0.48145466f, 0.45782750f, 0.40821073f};
static const float VL_STD_CLIP[3]     = {0.26862954f, 0.26130258f, 0.27577711f};

// ── ViT input size constants ──
enum VlImageSize {
    VL_SIZE_224 = 224,  // Qwen2-VL, CLIP, LLaVA-1.5
    VL_SIZE_336 = 336,  // LLaVA-1.6, InternVL
    VL_SIZE_384 = 384,  // SigLIP, some ViT-L variants
    VL_SIZE_448 = 448,  // Qwen2-VL-7B, EvaCLIP
};

// ── VlProcessor: single-image processor ──
// Manages CPU-side processed pixels. Optional GPU mirror.
class VlProcessor {
public:
    VlProcessor() = default;
    ~VlProcessor() { release_gpu(); }

    // Disable copy (owns a raw GPU pointer)
    VlProcessor(const VlProcessor&) = delete;
    VlProcessor& operator=(const VlProcessor&) = delete;

    // Move: transfer ownership, null out source so its destructor's
    // release_gpu() doesn't double-free.
    VlProcessor(VlProcessor&& other) noexcept
        : pixels_(std::move(other.pixels_)),
          dev_pixels_(other.dev_pixels_),
          w_(other.w_), h_(other.h_),
          out_w_(other.out_w_), out_h_(other.out_h_),
          orig_w_(other.orig_w_), orig_h_(other.orig_h_) {
        other.dev_pixels_ = nullptr;
    }
    VlProcessor& operator=(VlProcessor&& other) noexcept {
        if (this != &other) {
            release_gpu();
            pixels_ = std::move(other.pixels_);
            dev_pixels_ = other.dev_pixels_;
            w_ = other.w_; h_ = other.h_;
            out_w_ = other.out_w_; out_h_ = other.out_h_;
            orig_w_ = other.orig_w_; orig_h_ = other.orig_h_;
            other.dev_pixels_ = nullptr;
        }
        return *this;
    }

    // ── Load image from file + resize + normalize (CPU) ──
    // Returns true on success. Processed pixels available via pixels().
    bool load(const std::string& path, int out_w, int out_h,
              const float mean[3] = VL_MEAN_QWEN2VL,
              const float std[3]  = VL_STD_QWEN2VL) {
        pixels_ = vl_load_process(path, out_w, out_h, mean, std,
                                  &orig_w_, &orig_h_);
        if (pixels_.empty()) {
            fprintf(stderr, "[vl_processor] FAIL: load '%s'\n", path.c_str());
            return false;
        }
        out_w_ = out_w;
        out_h_ = out_h;
        w_ = out_w;
        h_ = out_h;
        return true;
    }

    // ── Load from memory buffer + resize + normalize (CPU) ──
    bool load_from_memory(const unsigned char* data, size_t len,
                          int out_w, int out_h,
                          const float mean[3] = VL_MEAN_QWEN2VL,
                          const float std[3]  = VL_STD_QWEN2VL) {
        int sw, sh;
        auto img = vl_load_image_from_memory(data, len, &sw, &sh);
        if (img.empty()) {
            fprintf(stderr, "[vl_processor] FAIL: decode from memory\n");
            return false;
        }
        orig_w_ = sw; orig_h_ = sh;
        pixels_ = vl_resize_normalize(img.data(), sw, sh, out_w, out_h, mean, std);
        if (pixels_.empty()) return false;
        out_w_ = out_w;
        out_h_ = out_h;
        w_ = out_w;
        h_ = out_h;
        return true;
    }

    // ── Accessors ──
    const float* pixels() const { return pixels_.data(); }
    float* pixels() { return pixels_.data(); }
    size_t size() const { return pixels_.size(); }
    int width() const { return w_; }
    int height() const { return h_; }
    int orig_width() const { return orig_w_; }
    int orig_height() const { return orig_h_; }
    int channels() const { return 3; }

    // ── GPU upload/download (only available when compiled with HIP) ──
#if defined(VL_HIP_AVAILABLE)
    float* upload_to_gpu() {
        if (pixels_.empty()) return nullptr;
        if (dev_pixels_) return dev_pixels_;
        size_t bytes = pixels_.size() * sizeof(float);
        if (hipMalloc(&dev_pixels_, bytes) != hipSuccess) return nullptr;
        if (hipMemcpy(dev_pixels_, pixels_.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
            (void)hipFree(dev_pixels_);
            dev_pixels_ = nullptr;
            return nullptr;
        }
        return dev_pixels_;
    }
    bool download_from_gpu() {
        if (!dev_pixels_ || pixels_.empty()) return false;
        size_t bytes = pixels_.size() * sizeof(float);
        return hipMemcpy(pixels_.data(), dev_pixels_, bytes, hipMemcpyDeviceToHost) == hipSuccess;
    }
    void release_gpu() {
        if (dev_pixels_) {
            (void)hipFree(dev_pixels_);
            dev_pixels_ = nullptr;
        }
    }
    float* dev_pixels() const { return dev_pixels_; }
#else
    // Stubs — HIP not available at compile time
    float* upload_to_gpu() { return nullptr; }
    bool download_from_gpu() { return false; }
    void release_gpu() {}
    float* dev_pixels() const { return nullptr; }
#endif

private:
    std::vector<float> pixels_;
#if defined(VL_HIP_AVAILABLE)
    float* dev_pixels_ = nullptr;
#else
    float* dev_pixels_ = nullptr;  // unused but keeps ABI consistent
#endif
    int w_ = 0, h_ = 0;
    int out_w_ = 0, out_h_ = 0;
    int orig_w_ = 0, orig_h_ = 0;
};

// ── Decode a base64 data URL to raw image bytes ──
// Handles data:image/png;base64,<data> format from OpenAI API.
// Returns empty vector on failure.
std::vector<unsigned char> vl_decode_base64_image(const std::string& data_url);

// ── Bbox-aware resize for dynamic-resolution VLMs ──
// Resizes with aspect ratio preserved, then pads to exact patch grid.
std::vector<float> vl_resize_bbox(const float* src, int sw, int sh,
                                   int patch_size, int max_patches,
                                   const float mean[3], const float std[3],
                                   int* out_w = nullptr, int* out_h = nullptr);

// ── Download image from URL to memory buffer ──
// Uses curl subprocess. Returns empty vector on failure.
std::vector<unsigned char> vl_download_image(const std::string& url, int timeout_sec = 10);

// ── Detect if a string is a data URL (base64-encoded image) ──
bool vl_is_data_url(const std::string& url);

#endif // VL_PROCESSOR_H
