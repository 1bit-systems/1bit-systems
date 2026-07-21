// vl_processor.cpp — stb_image implementation + ViT bbox-aware resize.
//
// This file:
//   1. Provides the single TU for stb_image.h's implementation
//   2. Bbox-aware resize: VL models like Qwen2-VL support dynamic resolution
//      by dividing the image into a grid of patches. This computes the
//      optimal crop-and-resize to minimize padding.
//   3. Image download from URL (curl subprocess or direct read)
//
// Upstream tracking: additive file, no existing file modified.

#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb/stb_image.h"

#include "vl_processor.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

// ── Bbox-aware resize for dynamic-resolution VLMs ──
// Qwen2-VL divides the image into a grid of patch-sized tiles
// (patch_size=14). The vision encoder processes (grid_h * grid_w) patches.
// This function resizes the image so the longest side fits within
// (max_patches * patch_size) while preserving aspect ratio, then pads
// to exact patch grid alignment.
//
// Returns processed pixels (resized + padded + normalized).
std::vector<float> vl_resize_bbox(const float* src, int sw, int sh,
                                   int patch_size, int max_patches,
                                   const float mean[3], const float std[3],
                                   int* out_w, int* out_h) {
    // Compute target size: fit longest side to max_patches * patch_size
    int max_dim = max_patches * patch_size;
    float scale;
    int tw, th;
    if (sw >= sh) {
        scale = (float)max_dim / sw;
        tw = max_dim;
        th = (int)(sh * scale + 0.5f);
        // Round th down to nearest multiple of patch_size
        th = (th / patch_size) * patch_size;
        if (th < patch_size) th = patch_size;
    } else {
        scale = (float)max_dim / sh;
        th = max_dim;
        tw = (int)(sw * scale + 0.5f);
        tw = (tw / patch_size) * patch_size;
        if (tw < patch_size) tw = patch_size;
    }

    if (out_w) *out_w = tw;
    if (out_h) *out_h = th;

    // First do a plain resize to (tw, th)
    std::vector<float> resized = vl_resize_normalize(src, sw, sh, tw, th, mean, std);

    return resized;
}

// ── Download image from URL to memory buffer ──
// Uses a very simple HTTP GET via stdio pipe to curl.
// Returns empty vector on failure.
std::vector<unsigned char> vl_download_image(const std::string& url, int timeout_sec) {
    std::string cmd = "curl -sL --connect-timeout " + std::to_string(timeout_sec) +
                      " --max-time " + std::to_string(timeout_sec + 5) +
                      " '" + url + "' 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        fprintf(stderr, "[vl] ERROR: failed to run curl\n");
        return {};
    }

    std::vector<unsigned char> data;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), pipe)) > 0)
        data.insert(data.end(), buf, buf + n);

    int rc = pclose(pipe);
    if (rc != 0 || data.empty()) {
        fprintf(stderr, "[vl] ERROR: curl failed (rc=%d) for '%s'\n", rc, url.c_str());
        return {};
    }

    return data;
}

// ── Detect if a string is a data URL (base64-encoded image) ──
bool vl_is_data_url(const std::string& url) {
    return url.find("data:image/") == 0;
}

// ── Decode base64 data URL to raw bytes ──
std::vector<unsigned char> vl_decode_base64_image(const std::string& data_url) {
    auto comma = data_url.find(',');
    if (comma == std::string::npos) return {};
    std::string b64 = data_url.substr(comma + 1);

    static const char b64_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    signed char b64_rev[256];
    for (int i = 0; i < 256; i++) b64_rev[i] = -1;
    for (int i = 0; i < 64; i++) b64_rev[(unsigned char)b64_chars[i]] = i;

    std::string clean;
    for (char c : b64)
        if (!isspace((unsigned char)c)) clean += c;

    size_t len = clean.size();
    if (len == 0) return {};

    size_t out_len = len / 4 * 3;
    if (len > 0 && clean[len-1] == '=') out_len--;
    if (len > 1 && clean[len-2] == '=') out_len--;

    std::vector<unsigned char> out(out_len);
    size_t pos = 0;
    for (size_t i = 0; i < len; i += 4) {
        unsigned char s[4] = {0, 0, 0, 0};
        for (int j = 0; j < 4 && i + j < len; j++) {
            char c = clean[i + j];
            if (c == '=') s[j] = 0;
            else s[j] = (unsigned char)b64_rev[(unsigned char)c];
        }
        if (pos < out_len) out[pos++] = (s[0] << 2) | (s[1] >> 4);
        if (pos < out_len) out[pos++] = (s[1] << 4) | (s[2] >> 2);
        if (pos < out_len) out[pos++] = (s[2] << 6) | s[3];
    }
    return out;
}
