// pixtral.h — Pixtral (Mistral) vision-language model connector
//
// Pixtral = Pixtral ViT vision encoder + MLP connector + Mistral text decoder
//
// Architecture:
//   Image → Pixtral ViT (2D-RoPE) → MLP connector → Mistral decoder
//
// The vision encoder reuses the generic ViT from vision_encoder.h.
// The connector is a 2-layer MLP specific to Pixtral's vision-text bridge.
// The text decoder uses the existing Mistral backend (RCPP_ARCH_MISTRAL).

#pragma once

#include "vision_encoder.h"
#include <cstdio>
#include <vector>
#include <string>

// ─── Pixtral connector config ─────────────────────────────────────
struct PixtralConfig {
    int vision_hidden   = 1024;   // Pixtral ViT hidden dimension
    int text_hidden     = 5120;   // Mistral-7B hidden dimension  
    int mlp_hidden      = 4096;   // Connector MLP width
    int n_vision_layers = 24;     // ViT layers
    int n_vision_heads  = 16;     // ViT heads
    int patch_size      = 14;     // Patch size
    int image_size      = 672;    // Pixtral native resolution (supports dynamic)

    static PixtralConfig pixtral_12b() {
        PixtralConfig c;
        c.vision_hidden = 1024;
        c.text_hidden = 5120;
        c.mlp_hidden = 4096;
        c.n_vision_layers = 24;
        c.n_vision_heads = 16;
        c.patch_size = 14;
        return c;
    }
};

// ─── Pixtral connector weights ────────────────────────────────────
struct PixtralWeights {
    PixtralConfig cfg;
    
    // Vision encoder (loaded from Pixtral's vision GGUF)
    VisionWeights vision;
    
    // Connector MLP weights (separate file or embedded in vision GGUF)
    std::vector<float> conn_ln_w, conn_ln_b;      // pre-connector LayerNorm
    std::vector<float> conn_mm0_w, conn_mm0_b;    // [vision_hidden, mlp_hidden]
    std::vector<float> conn_mm1_w, conn_mm1_b;    // [mlp_hidden, mlp_hidden]
    std::vector<float> conn_mm2_w, conn_mm2_b;    // [mlp_hidden, text_hidden]
    bool use_gelu = true;

    bool load_vision(const std::string& gguf_path);
    bool load_connector(const std::string& gguf_path);
    
    // Run connector: vision_embeds[n_tokens, vision_hidden] → text_embeds[n_tokens, text_hidden]
    std::vector<float> forward(const float* vision_embeds, int n_tokens) const;
};

// ─── Pixtral full inference ───────────────────────────────────────
// Load Pixtral vision encoder, process image, project to text space.
// Returns vision embeddings projected to text hidden dimension,
// ready to feed into a Mistral text decoder via forward_embed().
std::vector<float> pixtral_process_image(PixtralWeights& pw,
                                          const uint8_t* image_data, int img_w, int img_h,
                                          int target_size = 672);
