// vl_preprocess.h — Lightweight image loading + preprocessing for VL models.
//
// Replaces the OpenCV-dependent path (cv::imread → cv::resize → mat conversion)
// with stb_image.h (already vendored in third_party/stb/) + hand-written
// bilinear resize + mean/std normalization.
//
// Zero OpenCV dependency. ~120 lines of C++.
//
// Usage:
//   #include "vl_preprocess.h"
//   int w, h;
//   std::vector<float> pixels = vl_load_image("photo.jpg", &w, &h);
//   std::vector<float> processed = vl_resize_normalize(pixels, w, h, 224, 224,
//       {0.48145467f, 0.45782750f, 0.40821072f},
//       {0.26862955f, 0.26130259f, 0.27577710f});
//
// Upstream tracking: this file is additive — no existing file is modified.
// Cherry-pick clobber-free: if upstream adds their own vl_preprocess, rename
// this to vl_preprocess_ovr.h and #error at the top.

#ifndef VL_PREPROCESS_H
#define VL_PREPROCESS_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <cstdint>

// ── stb_image forward declaration ──
// We don't #define STB_IMAGE_IMPLEMENTATION here — that goes in exactly one
// .cpp file (vl_processor.cpp). This header only declares the API.
extern "C" {
    unsigned char* stbi_load(const char* filename, int* x, int* y, int* comp, int req_comp);
    unsigned char* stbi_load_from_memory(const unsigned char* buffer, int len, int* x, int* y, int* comp, int req_comp);
    void stbi_image_free(void* data);
}

// ── Load image file → float RGB pixels, interleaved, [0..1] ──
// Returns empty vector on failure. Output dimensions in *w, *h.
inline std::vector<float> vl_load_image(const std::string& path, int* w, int* h) {
    int comp;
    unsigned char* u8 = stbi_load(path.c_str(), w, h, &comp, 3);
    if (!u8) {
        fprintf(stderr, "[vl] ERROR: could not load image '%s'\n", path.c_str());
        return {};
    }
    size_t n = (size_t)(*w) * (*h) * 3;
    std::vector<float> out(n);
    for (size_t i = 0; i < n; i++)
        out[i] = u8[i] / 255.0f;
    stbi_image_free(u8);
    return out;
}

// ── Load image from raw uint8 buffer → float RGB pixels, interleaved ──
inline std::vector<float> vl_load_image_from_memory(const unsigned char* data, size_t len,
                                                      int* w, int* h) {
    int comp;
    unsigned char* u8 = stbi_load_from_memory(data, (int)len, w, h, &comp, 3);
    if (!u8) {
        fprintf(stderr, "[vl] ERROR: could not decode image from memory\n");
        return {};
    }
    size_t n = (size_t)(*w) * (*h) * 3;
    std::vector<float> out(n);
    for (size_t i = 0; i < n; i++)
        out[i] = u8[i] / 255.0f;
    stbi_image_free(u8);
    return out;
}

// ── Bilinear resize + per-channel mean/std normalization ──
// Input:  float RGB interleaved [0..1], sw x sh pixels
// Output: float RGB interleaved, normalized, out_w x out_h pixels
//
// Mean/std are the standard ImageNet/VL model values. For Qwen2-VL:
//   mean = {0.48145467f, 0.45782750f, 0.40821072f}
//   std  = {0.26862955f, 0.26130259f, 0.27577710f}
//
// Supports variable input sizes; always produces exact out_w x out_h output.
inline std::vector<float> vl_resize_normalize(const float* src, int sw, int sh,
                                                int out_w, int out_h,
                                                const float mean[3], const float std[3]) {
    std::vector<float> dst((size_t)out_w * out_h * 3);
    for (int y = 0; y < out_h; y++) {
        float sy = (y + 0.5f) * sh / out_h - 0.5f;
        int y0 = (int)std::floor(sy);
        float fy = sy - y0;
        int y1 = std::min(std::max(y0 + 1, 0), sh - 1);
        y0 = std::min(std::max(y0, 0), sh - 1);
        for (int x = 0; x < out_w; x++) {
            float sx = (x + 0.5f) * sw / out_w - 0.5f;
            int x0 = (int)std::floor(sx);
            float fx = sx - x0;
            int x1 = std::min(std::max(x0 + 1, 0), sw - 1);
            x0 = std::min(std::max(x0, 0), sw - 1);
            for (int c = 0; c < 3; c++) {
                float v00 = src[((size_t)y0 * sw + x0) * 3 + c];
                float v01 = src[((size_t)y0 * sw + x1) * 3 + c];
                float v10 = src[((size_t)y1 * sw + x0) * 3 + c];
                float v11 = src[((size_t)y1 * sw + x1) * 3 + c];
                float v0 = v00 * (1 - fx) + v01 * fx;
                float v1 = v10 * (1 - fx) + v11 * fx;
                float v = v0 * (1 - fy) + v1 * fy;
                dst[((size_t)y * out_w + x) * 3 + c] = (v - mean[c]) / std[c];
            }
        }
    }
    return dst;
}

// ── Convenience: load + resize + normalize in one call ──
// Returns empty vector on load failure.
inline std::vector<float> vl_load_process(const std::string& path, int out_w, int out_h,
                                            const float mean[3], const float std[3],
                                            int* orig_w = nullptr, int* orig_h = nullptr) {
    int sw, sh;
    std::vector<float> img = vl_load_image(path, &sw, &sh);
    if (img.empty()) return {};
    if (orig_w) *orig_w = sw;
    if (orig_h) *orig_h = sh;
    return vl_resize_normalize(img.data(), sw, sh, out_w, out_h, mean, std);
}

#endif // VL_PREPROCESS_H
