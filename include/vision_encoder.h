// vision_encoder.h — Vision Transformer (ViT) encoder + vision-text projector
// for VLMs like ZAYA1-VL-8B, Qwen2-VL, and LLaVA-style architectures.
//
// Architecture:
//   image → patch_embed (conv2d) → [ViT layers (self-attn + FFN)] × N
//       → post_ln → merger/connector → vision-text projector
//
// Two projector styles:
//   - CLIP-style: single linear projection (vision_hidden -> text_hidden)
//   - LLaVA-style: 2-layer MLP (vision_hidden -> mlp_hidden -> text_hidden)
//   - Qwen2-VL-style: merger (4-token group) → mm.0 (4*hidden) → GELU → mm.2 (proj_dim)
//   - Identity: vision_hidden == text_hidden, no projection
//
// The ViT config struct defines all hyperparameters so this works for any
// VLM architecture — just set the right values and load the right weights.
//
// Weights are loaded via GGUF tensor names following the llama.cpp convention
// (v.patch_embd.weight, v.blk.0.attn_q.weight, mm.0.weight, etc.) for
// compatibility with existing vision projector GGUF files.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

// ─── 1BP vision tensor name prefixes ──────────────────────────────
// These are the canonical names used when storing vision weights in a
// 1BP file. Matches the GGUF naming convention from llama.cpp's clip.cpp.
// Vision encoder tensors are stored with "v." prefix in the 1BP index.
static constexpr const char* VIT_TENSOR_PATCH_EMBD    = "v.patch_embd.weight";    // [patch*patch*3, hidden]
static constexpr const char* VIT_TENSOR_PATCH_EMBD1   = "v.patch_embd.weight.1";  // [patch*patch*3, hidden]
static constexpr const char* VIT_TENSOR_POS_EMBD      = "v.position_embd.weight"; // [n_positions, hidden]
static constexpr const char* VIT_TENSOR_CLS_EMBD      = "v.class_embd.weight";    // [1, hidden]
static constexpr const char* VIT_TENSOR_PRE_LN_W      = "v.pre_ln.weight";        // [hidden]
static constexpr const char* VIT_TENSOR_PRE_LN_B      = "v.pre_ln.bias";          // [hidden]
static constexpr const char* VIT_TENSOR_POST_LN_W     = "v.post_ln.weight";       // [hidden]
static constexpr const char* VIT_TENSOR_POST_LN_B     = "v.post_ln.bias";         // [hidden]

// Per-layer tensor names (substitute layer index for %d):
// v.blk.%d.ln1.weight, v.blk.%d.ln1.bias
// v.blk.%d.ln2.weight, v.blk.%d.ln2.bias
// v.blk.%d.attn_q.weight, v.blk.%d.attn_q.bias
// v.blk.%d.attn_k.weight, v.blk.%d.attn_k.bias
// v.blk.%d.attn_v.weight, v.blk.%d.attn_v.bias
// v.blk.%d.attn_out.weight, v.blk.%d.attn_out.bias
// v.blk.%d.ffn_up.weight, v.blk.%d.ffn_up.bias      (real up-projection)
// v.blk.%d.ffn_down.weight, v.blk.%d.ffn_down.bias  (real down-projection)

// Projector tensor names:
static constexpr const char* VIT_TENSOR_MM0_W = "mm.0.weight";  // [proj_in, proj_mid]
static constexpr const char* VIT_TENSOR_MM0_B = "mm.0.bias";    // [proj_mid]
static constexpr const char* VIT_TENSOR_MM1_W = "mm.1.weight";  // [proj_mid, proj_mid] (3-layer MLP only)
static constexpr const char* VIT_TENSOR_MM1_B = "mm.1.bias";    // [proj_mid]
static constexpr const char* VIT_TENSOR_MM2_W = "mm.2.weight";  // [proj_mid, text_hidden]
static constexpr const char* VIT_TENSOR_MM2_B = "mm.2.bias";    // [text_hidden]

// ─── Vision encoder types ─────────────────────────────────────────
enum class VitArch : uint8_t {
    CLIP_VIT_L = 0,    // CLIP ViT-L/14 (hidden=1024, heads=16, layers=24, patch=14)
    CLIP_VIT_B = 1,    // CLIP ViT-B/16 (hidden=768, heads=12, layers=12, patch=16)
    SIGLIP_VIT = 2,    // SigLIP ViT (same architecture as CLIP, different training)
    QWEN2_VL   = 3,    // Qwen2-VL vision encoder
    CUSTOM     = 4,    // Custom config (set all fields manually)
};

enum class ProjectorType : uint8_t {
    LINEAR    = 0,     // Single linear: vision_hidden -> text_hidden
    MLP_2LAYER = 1,    // 2-layer MLP: vision_hidden -> mid -> text_hidden (GELU)
    MLP_3LAYER = 2,    // 3-layer MLP: vision_hidden -> mid -> mid -> text_hidden (GELU)
    QWEN2_MERGER = 3,  // Qwen2-VL-style: group 4 tokens -> mm.0 -> GELU -> mm.2
    IDENTITY  = 4,     // No projection (vision_hidden == text_hidden)
};

// ─── Default vision architectures ─────────────────────────────────
struct VitDefaults {
    static constexpr int CLIP_VIT_L_HIDDEN   = 1024;
    static constexpr int CLIP_VIT_L_LAYERS   = 24;
    static constexpr int CLIP_VIT_L_HEADS    = 16;
    static constexpr int CLIP_VIT_L_PATCH    = 14;
    static constexpr int CLIP_VIT_L_FF       = 4096;

    static constexpr int CLIP_VIT_B_HIDDEN   = 768;
    static constexpr int CLIP_VIT_B_LAYERS   = 12;
    static constexpr int CLIP_VIT_B_HEADS    = 12;
    static constexpr int CLIP_VIT_B_PATCH    = 16;
    static constexpr int CLIP_VIT_B_FF       = 3072;

    static constexpr int SIGLIP_HIDDEN       = 1152;
    static constexpr int SIGLIP_LAYERS       = 27;
    static constexpr int SIGLIP_HEADS        = 16;
    static constexpr int SIGLIP_PATCH        = 14;
    static constexpr int SIGLIP_FF           = 4304;
};

// ─── ViT config ───────────────────────────────────────────────────
struct VitConfig {
    int hidden_size     = 1024;   // ViT hidden dimension
    int num_layers      = 24;     // Number of transformer layers in ViT
    int num_heads       = 16;     // Number of attention heads
    int patch_size      = 14;     // Patch size (14 for CLIP/SigLIP, 14 for Qwen2-VL)
    int intermediate_size = 4096; // FFN intermediate dimension
    int max_positions   = 1024;   // Max patch positions (for position embeddings)
    float layer_norm_eps = 1e-6f;

    // MLP activation
    bool use_gelu    = true;     // true = GELU (CLIP, SigLIP), false = ReLU
    bool use_bias    = true;     // whether attention/FFN use bias
    bool use_pre_ln  = false;    // whether there's a pre-LN layer before the transformer

    // Image normalization constants (for preprocessing)
    float mean[3] = {0.48145466f, 0.45782750f, 0.40821073f}; // CLIP default
    float std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};

    // Factory helpers for known architectures
    static VitConfig clip_vit_l() {
        VitConfig c;
        c.hidden_size = VitDefaults::CLIP_VIT_L_HIDDEN;
        c.num_layers  = VitDefaults::CLIP_VIT_L_LAYERS;
        c.num_heads   = VitDefaults::CLIP_VIT_L_HEADS;
        c.patch_size  = VitDefaults::CLIP_VIT_L_PATCH;
        c.intermediate_size = VitDefaults::CLIP_VIT_L_FF;
        c.use_gelu = true; c.use_bias = true;
        return c;
    }
    static VitConfig siglip_vit() {
        VitConfig c;
        c.hidden_size = VitDefaults::SIGLIP_HIDDEN;
        c.num_layers  = VitDefaults::SIGLIP_LAYERS;
        c.num_heads   = VitDefaults::SIGLIP_HEADS;
        c.patch_size  = VitDefaults::SIGLIP_PATCH;
        c.intermediate_size = VitDefaults::SIGLIP_FF;
        c.use_gelu = true; c.use_bias = true;
        return c;
    }
    static VitConfig mage_vit() {
        VitConfig c;
        c.hidden_size = 1024;
        c.num_layers  = 24;
        c.num_heads   = 16;
        c.patch_size  = 16;
        c.intermediate_size = 4096;
        c.use_gelu = true; c.use_bias = true; c.use_pre_ln = true;
        return c;
    }
};

// ─── Projector config ─────────────────────────────────────────────
struct ProjectorConfig {
    ProjectorType type = ProjectorType::LINEAR;
    int vision_hidden  = 1024;    // ViT output dimension
    int text_hidden    = 2048;    // Text decoder input dimension (Zaya1: 2048)
    int mlp_hidden     = 4096;    // MLP intermediate dimension (2/3-layer MLP)
    int merge_group    = 0;       // Tokens to merge (0 = no merge, 4 = Qwen2-VL style)
};

// ─── Per-layer ViT weights ────────────────────────────────────────
struct VitLayerWeights {
    std::vector<float> ln1_w, ln1_b;       // pre-attention LayerNorm
    std::vector<float> ln2_w, ln2_b;       // pre-FFN LayerNorm
    std::vector<float> attn_q_w, attn_q_b; // [hidden, hidden]
    std::vector<float> attn_k_w, attn_k_b;
    std::vector<float> attn_v_w, attn_v_b;
    std::vector<float> attn_o_w, attn_o_b;
    std::vector<float> ffn_up_w, ffn_up_b;   // up-projection [hidden, ff]  (real up, not swapped)
    std::vector<float> ffn_down_w, ffn_down_b; // down-projection [ff, hidden] (real down)
};

// ─── Full vision encoder weights ──────────────────────────────────
struct VisionWeights {
    VitConfig config;

    // Patch embedding (conv2d weights: flat [patch*patch*3, hidden] or 4D [kw,kh,cin,cout])
    std::vector<float> patch_embd0;
    std::vector<float> patch_embd1;  // optional second temporal kernel
    std::vector<float> patch_bias;   // optional bias [hidden]

    // Position embeddings
    std::vector<float> pos_embd;   // [max_positions, hidden] (abs pos emb, optional)
    std::vector<float> cls_embd;   // [1, hidden] (CLS token, optional)

    // Pre/post layer norms
    std::vector<float> pre_ln_w, pre_ln_b;
    std::vector<float> post_ln_w, post_ln_b;

    // Transformer layers
    std::vector<VitLayerWeights> layers;

    // Projector
    ProjectorConfig proj_config;
    std::vector<float> mm0_w, mm0_b;  // layer 0 of projector
    std::vector<float> mm1_w, mm1_b;  // layer 1 of projector (3-layer MLP only)
    std::vector<float> mm2_w, mm2_b;  // final projection layer

    // Normalization constants (overrides config if set)
    float mean[3] = {0.0f, 0.0f, 0.0f}; // 0 = use config defaults
    float std[3]  = {0.0f, 0.0f, 0.0f};

    bool has_patch_embd1 = false;  // true if second temporal kernel exists
    bool has_pos_embd    = false;
    bool has_cls_embd    = false;
    bool has_pre_ln      = false;
    bool has_fused_qkv = false;    // true if attn_qkv.weight instead of separate attn_q/k/v
    bool ffn_names_swapped = false; // true if llama.cpp clip bug (ffn_up <-> ffn_down)

    // Load all weights from a GGUF vision tower file.
    // Returns true on success, false on any missing required tensor.
    bool load_from_gguf(const std::string& gguf_path, const VitConfig* cfg_override = nullptr);

    // Clear all weights
    void clear();
};

// ─── Math helpers (match backend_generic conventions) ─────────────
namespace vit_math {
    // out[i] = sum_j in[j] * w[i*K + j]  (row-major: w[out_dim, in_dim])
    static inline void matmul(float* out, const float* in, const float* w, int M, int K) {
        for (int i = 0; i < M; i++) {
            float s = 0;
            for (int j = 0; j < K; j++) s += in[j] * w[(size_t)i * K + j];
            out[i] = s;
        }
    }

    // RMSNorm (used in newer architectures)
    static inline void rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
        double ss = 0;
        for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
        float inv = 1.0f / sqrtf((float)(ss / n) + eps);
        for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
    }

    // LayerNorm (used in CLIP/SigLIP)
    static inline void layernorm(float* out, const float* x, const float* w, const float* b, int n, float eps) {
        float mean = 0; for (int i = 0; i < n; i++) mean += x[i]; mean /= n;
        float var = 0; for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; } var /= n;
        float inv = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
    }

    // GELU approximation (tanh)
    static inline void gelu(float* out, const float* x, int n) {
        const float c = 0.7978845608f;
        for (int i = 0; i < n; i++) {
            float v = x[i];
            out[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
        }
    }

    // SiLU / Swish
    static inline void silu(float* out, const float* x, int n) {
        for (int i = 0; i < n; i++) out[i] = x[i] / (1.0f + expf(-x[i]));
    }

    // ReLU
    static inline void relu(float* out, const float* x, int n) {
        for (int i = 0; i < n; i++) out[i] = x[i] > 0 ? x[i] : 0;
    }

    // Softmax in-place over n elements
    static inline void softmax_inplace(float* x, int n) {
        float mx = x[0]; for (int i = 1; i < n; i++) mx = std::max(mx, x[i]);
        float sum = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
        float inv = 1.0f / sum;
        for (int i = 0; i < n; i++) x[i] *= inv;
    }
}

// ─── ViT encoder forward pass ─────────────────────────────────────
// Runs the full ViT encoder on a preprocessed image.
// Input:  img[image_h * image_w * 3] — normalized floats [0..1], HWC interleaved
// Output: vision_embeds[n_patches * vision_hidden] — one vector per patch
//
// If proj_config is provided, also runs the projector to map to text_hidden.
// Returns projected embeddings if projector configured, raw ViT output otherwise.
std::vector<float> vit_forward(
    const VisionWeights& weights,
    const float* img, int img_w, int img_h,
    const ProjectorConfig* proj_override = nullptr);

// ─── Patch embedding helper ───────────────────────────────────────
// Embeds one patch of the image using conv2d-style weights.
// Used internally by vit_forward, also callable standalone.
void vit_patch_embed(
    const VisionWeights& weights,
    const float* img, int img_w, int img_h,
    int patch_row, int patch_col,
    float* out /* [hidden_size] */);

// ─── 2D RoPE (M-RoPE vision mode) for position encoding ──────────
// Applies rotary position embeddings using row/col positions.
// Used by Qwen2-VL-style and other 2D-aware ViTs.
void vit_rope2d_apply(float* x, int head_dim, int row, int col, float freq_base);

// ─── Preprocess image for ViT ─────────────────────────────────────
// Resize + normalize. Returns float HWC image [out_h * out_w * 3].
// Uses the normalization constants from weights.config (or an override).
std::vector<float> vit_preprocess(
    const uint8_t* src, int src_w, int src_h,
    int out_w, int out_h,
    const float* mean, const float* std);

// ─── Load preprocessed image from file ────────────────────────────
// Decodes image file, resizes, normalizes. Returns empty on failure.
std::vector<float> vit_load_and_preprocess(
    const std::string& image_path,
    int out_w, int out_h,
    const float* mean, const float* std);

// ─── Mage-ViT: 3D RoPE ────────────────────────────────────────────
void vit_rope3d_apply(float* q, float* k, int head_dim,
                       int t, int h, int w, float freq_base);

// ─── Mage-ViT forward pass ─────────────────────────────────────────
std::vector<float> mage_vit_forward(
    const VisionWeights& weights,
    const float* pixels, int channels, int time, int height, int width,
    int frame_window_size = 4);

// ─── Q4NX tile dequantizers ────────────────────────────────────────
void dequant_q4nx_row(const uint8_t* row_data, float* out,
                       int n_groups, int gs = 32);
void dequant_q4nx_tile(const uint8_t* tile_data, float* out,
                        int tr, int tc, int gs = 32);

// ─── Load Mage-ViT vision weights from 1BP file ───────────────────
bool mage_vit_load_weights_1bp(const char* path, VisionWeights& vw);
