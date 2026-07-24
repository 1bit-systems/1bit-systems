// pixtral.cpp — Pixtral vision-language connector implementation
#include "pixtral.h"
#include "gguf_reader.h"
#include "vision_encoder.h"

bool PixtralWeights::load_vision(const std::string& gguf_path) {
    fprintf(stderr, "[pixtral] Loading vision encoder from %s\n", gguf_path.c_str());
    
    VitConfig vcfg;
    vcfg.hidden_size = cfg.vision_hidden;
    vcfg.num_layers = cfg.n_vision_layers;
    vcfg.num_heads = cfg.n_vision_heads;
    vcfg.patch_size = cfg.patch_size;
    vcfg.intermediate_size = cfg.vision_hidden * 4;  // approximate
    
    if (!vision.load_from_gguf(gguf_path, &vcfg)) {
        fprintf(stderr, "[pixtral] FAIL: could not load vision encoder\n");
        return false;
    }
    fprintf(stderr, "[pixtral] Vision encoder loaded: H=%d L=%d NH=%d P=%d\n",
            vision.config.hidden_size, vision.config.num_layers,
            vision.config.num_heads, vision.config.patch_size);
    return true;
}

bool PixtralWeights::load_connector(const std::string& gguf_path) {
    fprintf(stderr, "[pixtral] Loading connector from %s\n", gguf_path.c_str());
    
    GgufReader r;
    if (!r.open(gguf_path)) {
        fprintf(stderr, "[pixtral] FAIL: could not open %s\n", gguf_path.c_str());
        return false;
    }
    
    auto get = [&](const std::string& name, std::vector<float>& dst, size_t expect) -> bool {
        size_t n = 0;
        if (!r.get_tensor_f32(name, dst, &n)) {
            fprintf(stderr, "  [pixtral] missing: %s\n", name.c_str());
            return false;
        }
        if (expect > 0 && n != expect) {
            fprintf(stderr, "  [pixtral] %s: expected %zu, got %zu\n", name.c_str(), expect, n);
            return false;
        }
        return true;
    };
    
    int V = cfg.vision_hidden, M = cfg.mlp_hidden, T = cfg.text_hidden;
    bool ok = true;
    
    // Pre-connector LayerNorm (optional)
    ok &= get("conn.ln.weight", conn_ln_w, (size_t)V) || true;
    if (conn_ln_w.empty()) conn_ln_w.resize(V, 1.0f);
    ok &= get("conn.ln.bias", conn_ln_b, (size_t)V) || true;
    if (conn_ln_b.empty()) conn_ln_b.resize(V, 0.0f);
    
    // Connector MLP layers (following LLaVA-style: vision → mlp_hidden → text_hidden)
    ok &= get("conn.mm.0.weight", conn_mm0_w, (size_t)V * M);
    ok &= get("conn.mm.0.bias",   conn_mm0_b, (size_t)M);
    ok &= get("conn.mm.1.weight", conn_mm1_w, (size_t)M * M);
    ok &= get("conn.mm.1.bias",   conn_mm1_b, (size_t)M);
    ok &= get("conn.mm.2.weight", conn_mm2_w, (size_t)M * T);
    ok &= get("conn.mm.2.bias",   conn_mm2_b, (size_t)T);
    
    if (!ok) {
        fprintf(stderr, "[pixtral] FAIL: connector weights incomplete\n");
        return false;
    }
    
    fprintf(stderr, "[pixtral] Connector loaded: V=%d → M=%d → T=%d\n", V, M, T);
    return true;
}

std::vector<float> PixtralWeights::forward(const float* vision_embeds, int n_tokens) const {
    using namespace vit_math;
    int V = cfg.vision_hidden, M = cfg.mlp_hidden, T = cfg.text_hidden;
    
    std::vector<float> result((size_t)n_tokens * T);
    std::vector<float> ln_out(V), hidden(M), hidden2(M);
    
    for (int t = 0; t < n_tokens; t++) {
        const float* vt = &vision_embeds[(size_t)t * V];
        
        // LayerNorm
        layernorm(ln_out.data(), vt, conn_ln_w.data(), conn_ln_b.data(), V, 1e-6f);
        
        // mm.0: V → M
        matmul(hidden.data(), ln_out.data(), conn_mm0_w.data(), M, V);
        for (int i = 0; i < M; i++) hidden[i] += conn_mm0_b[i];
        if (use_gelu) gelu(hidden.data(), hidden.data(), M);
        else silu(hidden.data(), hidden.data(), M);
        
        // mm.1: M → M
        matmul(hidden2.data(), hidden.data(), conn_mm1_w.data(), M, M);
        for (int i = 0; i < M; i++) hidden2[i] += conn_mm1_b[i];
        if (use_gelu) gelu(hidden2.data(), hidden2.data(), M);
        else silu(hidden2.data(), hidden2.data(), M);
        
        // mm.2: M → T
        matmul(&result[(size_t)t * T], hidden2.data(), conn_mm2_w.data(), T, M);
        for (int i = 0; i < T; i++) result[(size_t)t * T + i] += conn_mm2_b[i];
    }
    
    return result;
}

std::vector<float> pixtral_process_image(PixtralWeights& pw,
                                          const uint8_t* image_data, int img_w, int img_h,
                                          int target_size) {
    // Preprocess image to target size
    float clip_mean[3] = {0.48145466f, 0.45782750f, 0.40821073f};
    float clip_std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};
    
    auto img_f = vit_preprocess(image_data, img_w, img_h, target_size, target_size, clip_mean, clip_std);
    if (img_f.empty()) return {};
    
    // Run vision encoder
    auto vision_embeds = vit_forward(pw.vision, img_f.data(), target_size, target_size);
    if (vision_embeds.empty()) return {};
    
    int n_tokens = (int)vision_embeds.size() / pw.vision.config.hidden_size;
    
    // Run connector
    return pw.forward(vision_embeds.data(), n_tokens);
}
