// test_vit_encoder.cpp — Standalone ViT encoder test
#include "vision_encoder.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <mmproj.gguf> <image.jpg>\n", argv[0]);
        return 1;
    }
    
    fprintf(stderr, "Loading vision encoder from: %s\n", argv[1]);
    VisionWeights vision;
    if (!vision.load_from_gguf(argv[1])) {
        fprintf(stderr, "FAIL: vision encoder load\n");
        return 1;
    }
    fprintf(stderr, "\nVision encoder: H=%d L=%d NH=%d P=%d FF=%d\n",
            vision.config.hidden_size, vision.config.num_layers,
            vision.config.num_heads, vision.config.patch_size,
            vision.config.intermediate_size);
    
    auto check_weights = [](const std::vector<float>& w, const char* name) {
        if (w.empty()) return true;
        int n_nan = 0, n_inf = 0, n_zero = 0;
        double mean = 0, mx = -1e30, mn = 1e30;
        for (float v : w) {
            if (std::isnan(v)) n_nan++;
            else if (!std::isfinite(v)) n_inf++;
            else { mean += v; mx = std::max((double)v, mx); mn = std::min((double)v, mn); if (v == 0) n_zero++; }
        }
        mean /= (w.size() - n_nan - n_inf);
        fprintf(stderr, "  %s: size=%zu mean=%.4f range=[%.4f,%.4f] NaN=%d Inf=%d Zero=%d\n",
                name, w.size(), mean, mn, mx, n_nan, n_inf, n_zero);
        return n_nan == 0 && n_inf == 0;
    };
    
    check_weights(vision.patch_embd0, "patch_embd0");
    check_weights(vision.patch_bias, "patch_bias");
    
    if (!vision.layers.empty()) {
        auto& l0 = vision.layers[0];
        check_weights(l0.ln1_w, "l0.ln1_w");
        check_weights(l0.attn_q_w, "l0.attn_q_w");
        check_weights(l0.attn_k_w, "l0.attn_k_w");
        check_weights(l0.attn_v_w, "l0.attn_v_w");
        check_weights(l0.attn_o_w, "l0.attn_o_w");
        check_weights(l0.ffn_up_w, "l0.ffn_up_w");
        check_weights(l0.ffn_down_w, "l0.ffn_down_w");
    }
    
    check_weights(vision.post_ln_w, "post_ln_w");
    check_weights(vision.mm0_w, "mm0_w");
    check_weights(vision.mm2_w, "mm2_w");
    
    // Quick patch embed test
    fprintf(stderr, "\nTesting patch_embed on synthetic input...\n");
    int P = vision.config.patch_size;
    int H = vision.config.hidden_size;
    
    // Use non-constant for fill
    std::vector<float> test_img((size_t)P * P * 3, 0.0f);
    for (size_t i = 0; i < test_img.size(); i++) test_img[i] = 0.5f;
    std::vector<float> patch_out(H);
    vit_patch_embed(vision, test_img.data(), P, P, 0, 0, patch_out.data());
    
    double mean_p = 0, mx_p = -1e30, mn_p = 1e30;
    int nan_p = 0;
    for (float v : patch_out) {
        if (std::isnan(v)) nan_p++;
        else { mean_p += v; mx_p = std::max((double)v, mx_p); mn_p = std::min((double)v, mn_p); }
    }
    mean_p /= H;
    fprintf(stderr, "  patch_embed output: mean=%.4f range=[%.4f,%.4f] NaN=%d\n", mean_p, mn_p, mx_p, nan_p);
    
    if (nan_p > 0) {
        fprintf(stderr, "\n❌ patch_embed produces NaN — Q8_0 dequant or weight loading bug\n");
        return 1;
    }
    
    // Full forward on real image
    int img_size = P * 16;
    if (img_size < 224) img_size = 224;
    
    float clip_mean[3] = {0.48145466f, 0.45782750f, 0.40821073f};
    float clip_std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};
    const float* mean = vision.mean[0] != 0 ? vision.mean : clip_mean;
    const float* std  = vision.std[0]  != 0 ? vision.std  : clip_std;
    
    fprintf(stderr, "\nLoading + preprocessing image: %s (%dx%d)\n", argv[2], img_size, img_size);
    auto img = vit_load_and_preprocess(argv[2], img_size, img_size, mean, std);
    if (img.empty()) {
        fprintf(stderr, "FAIL: image load\n");
        return 1;
    }
    fprintf(stderr, "Image pixels: first 5 = %.4f %.4f %.4f %.4f %.4f\n", img[0], img[1], img[2], img[3], img[4]);
    
    fprintf(stderr, "Running ViT forward (24 layers)...\n");
    auto embeds = vit_forward(vision, img.data(), img_size, img_size);
    if (embeds.empty()) {
        fprintf(stderr, "FAIL: vit_forward returned empty\n");
        return 1;
    }
    
    int th = vision.proj_config.text_hidden;
    int n_tokens = (int)embeds.size() / th;
    
    double mean_v = 0, var = 0, mx = -1e30, mn = 1e30;
    int nans = 0, infs = 0;
    for (float v : embeds) {
        if (std::isnan(v)) { nans++; continue; }
        if (!std::isfinite(v)) { infs++; continue; }
        mean_v += v; mx = std::max((double)v, mx); mn = std::min((double)v, mn);
    }
    size_t valid = embeds.size() - nans - infs;
    mean_v /= valid > 0 ? valid : 1;
    for (float v : embeds) { if (std::isfinite(v)) var += (v - mean_v) * (v - mean_v); }
    var /= valid > 0 ? valid : 1;
    
    fprintf(stderr, "\n=== Vision Encoder Results ===\n");
    fprintf(stderr, "  Tokens:     %d x %d\n", n_tokens, th);
    fprintf(stderr, "  Mean:       %.4f  Std: %.4f\n", mean_v, sqrt(var));
    fprintf(stderr, "  Range:      [%.4f, %.4f]\n", mn, mx);
    fprintf(stderr, "  NaN: %d  Inf: %d\n", nans, infs);
    fprintf(stderr, "  First 5: ");
    for (int i = 0; i < 5 && i < (int)embeds.size(); i++) fprintf(stderr, "%.4f ", embeds[i]);
    fprintf(stderr, "\n");
    
    if (nans == 0 && infs == 0 && mx > mn)
        fprintf(stderr, "\n✅ ViT PASSED\n");
    else
        fprintf(stderr, "\n❌ ViT FAILED\n");
    
    return (nans == 0 && infs == 0 && mx > mn) ? 0 : 1;
}
