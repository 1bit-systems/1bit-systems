// vision_encoder.cpp — Vision Transformer encoder + projector implementation
//
// Implements the forward passes declared in vision_encoder.h.
// Pure C++23, no external dependencies beyond stb_image for image loading.
// Matches the Qwen2-VL vision POC (vision_qwen2vl_poc.cpp) conventions
// but generalized for any ViT architecture (CLIP, SigLIP, Qwen2-VL, custom).

#include "vision_encoder.h"
#include "vl_processor.h"
#include <cstring>
#include <cmath>
#include <numeric>

// ─── Read tensor via GgufReader (uses 1bit-systems' built-in GGUF reader) ──
#include "gguf_reader.h"

static bool read_tensor_f32(const std::string& gguf_path, const std::string& name,
                             std::vector<float>& dst, size_t* out_n = nullptr) {
    GgufReader r;
    if (!r.open(gguf_path)) {
        fprintf(stderr, "  [vit] FAIL: could not open GGUF file %s\n", gguf_path.c_str());
        return false;
    }
    bool ok = r.get_tensor_f32(name, dst, out_n);
    return ok;
    // Destructor handles file close
}
// ─── Read a single GGUF uint32 metadata value ──────────────────
static bool read_gguf_u32(const std::string& path, const std::string& key, uint32_t& out) {
    GgufReader r;
    if (!r.open(path)) return false;
    bool ok = r.get_u32(key, out);
    // Also try with general. prefix
    if (!ok) {
        std::string arch = r.architecture();
        if (!arch.empty()) ok = r.get_u32(arch + "." + key, out);
    }
    return ok;
    // Destructor handles file close
}

// ====================================================================
// VisionWeights implementation
// ====================================================================

bool VisionWeights::load_from_gguf(const std::string& gguf_path, const VitConfig* cfg_override) {
    // First try to read the GGUF metadata for vision config
    uint32_t u32v;
    VitConfig detected;

    if (read_gguf_u32(gguf_path, "vision.hidden_size", u32v)) detected.hidden_size = (int)u32v;
    if (read_gguf_u32(gguf_path, "vision.num_layers", u32v)) detected.num_layers = (int)u32v;
    if (read_gguf_u32(gguf_path, "vision.num_heads", u32v)) detected.num_heads = (int)u32v;
    if (read_gguf_u32(gguf_path, "vision.patch_size", u32v)) detected.patch_size = (int)u32v;
    if (read_gguf_u32(gguf_path, "vision.intermediate_size", u32v)) detected.intermediate_size = (int)u32v;

    // Use override if provided, otherwise detected values
    if (cfg_override) {
        config = *cfg_override;
    } else {
        config = detected;
    }

    int H = config.hidden_size;
    int P = config.patch_size;
    int FF = config.intermediate_size;

    auto get = [&](const std::string& name, std::vector<float>& dst, size_t expect) -> bool {
        size_t n = 0;
        if (!read_tensor_f32(gguf_path, name, dst, &n)) {
            fprintf(stderr, "  [vit] missing tensor: %s\n", name.c_str());
            return false;
        }
        if (expect > 0 && n != expect) {
            fprintf(stderr, "  [vit] %s: expected %zu floats, got %zu — ACCEPTING (auto-detected size)\n", name.c_str(), expect, n);
            // Accept the tensor at its actual size (auto-detected config may differ)
        }
        return true;
    };

    auto get_opt = [&](const std::string& name, std::vector<float>& dst, size_t expect) -> bool {
        size_t n = 0;
        if (!read_tensor_f32(gguf_path, name, dst, &n)) {
            dst.clear();
            return false;
        }
        if (expect > 0 && n != expect) {
            fprintf(stderr, "  [vit] %s: expected %zu floats, got %zu (OPTIONAL — ignoring)\n", name.c_str(), expect, n);
            dst.clear();
            return false;
        }
        return true;
    };

    // ── Step 1: Probe patch_embd to auto-detect config before any `get()`` ──
    {
        std::vector<float> probe;
        size_t n = 0;
        // Don't use the expectation-checking `get` — probe raw, then decide
        if (read_tensor_f32(gguf_path, "v.patch_embd.weight", probe, &n) && n > 0) {
            int inferred_patch = 14, inferred_hidden = 1024;  // CLIP default
            bool found = false;
            for (int test_p = 14; test_p <= 16; test_p++) {
                size_t test = (size_t)test_p * test_p * 3;
                if (n % test == 0) {
                    int h = (int)(n / test);
                    if (h > 512 && h <= 8192) {
                        inferred_patch = test_p; inferred_hidden = h;
                        found = true; break;
                    }
                }
            }
            if (found && (inferred_patch != config.patch_size || inferred_hidden != config.hidden_size)) {
                fprintf(stderr, "  [vit] auto-detected P=%d H=%d from patch_embd shape (override P=%d H=%d)\n",
                        inferred_patch, inferred_hidden, config.patch_size, config.hidden_size);
                config.patch_size = inferred_patch;
                config.hidden_size = inferred_hidden;
                int hd = inferred_hidden % 72 == 0 ? 72 : 64;  // SigLIP or CLIP head_dim
                config.num_heads = inferred_hidden / hd;
            }
            // Store the probe result instead of loading again
            patch_embd0 = std::move(probe);
        } else {
            fprintf(stderr, "  [vit] missing patch_embd.weight\n");
            return false;
        }
    }
    // Reload config locals now that auto-detection may have changed them
    H = config.hidden_size;
    P = config.patch_size;
    FF = config.intermediate_size;

    has_patch_embd1 = get_opt("v.patch_embd.weight.1", patch_embd1, (size_t)P * P * 3 * H);
    get_opt("v.patch_embd.bias", patch_bias, (size_t)H);

    // Position and CLS embeddings (optional)
    has_pos_embd = get_opt("v.position_embd.weight", pos_embd, (size_t)config.max_positions * H);
    has_cls_embd = get_opt("v.class_embd.weight", cls_embd, (size_t)H);

    // Pre-LN (optional)
    has_pre_ln = get_opt("v.pre_ln.weight", pre_ln_w, (size_t)H);
    if (has_pre_ln) get_opt("v.pre_ln.bias", pre_ln_b, (size_t)H);
    if (has_pre_ln && pre_ln_b.empty()) {
        pre_ln_b.resize(H, 0.0f);
    }

    // Detect if FFN names are swapped (llama.cpp clip bug: ffn_down is really up, ffn_up is really down)
    // We detect by checking shapes: if v.blk.0.ffn_down.weight has shape [ff, H] instead of [H, ff],
    // it's the real up-projection and names are swapped.
    has_fused_qkv = false;
    // Check if this model uses fused QKV (attn_qkv.weight) or separate Q/K/V
    {
        std::vector<float> probe;
        if (!read_tensor_f32(gguf_path, "v.blk.0.attn_q.weight", probe)) {
            has_fused_qkv = true;
            fprintf(stderr, "  [vit] detected fused QKV format (attn_qkv.weight) — splitting at load\n");
        }
    }

    // Use the ffn_down probe to detect FF size AND check if names are swapped.
    // We read ffn_down.weight once and use its size for both purposes.
    {
        std::vector<float> probe_down;
        size_t n_down = 0;
        if (read_tensor_f32(gguf_path, "v.blk.0.ffn_down.weight", probe_down, &n_down)) {
            ffn_names_swapped = (n_down == (size_t)config.intermediate_size * H);
            // Auto-detect intermediate_size from ffn_down: if not swapped,
            // shape is [ff, H], so ff = n_down / H.
            // If swapped, shape is [H, ff] (same), so still ff = n_down / H.
            if (H > 0) {
                int detected_ff = (int)(n_down / H);
                if (detected_ff * H == (int)n_down && detected_ff > 0 && detected_ff != config.intermediate_size) {
                    fprintf(stderr, "  [vit] auto-detected FF=%d from ffn_down (%zu el, H=%d)\n", detected_ff, n_down, H);
                    config.intermediate_size = detected_ff;
                    FF = detected_ff;
                }
            }
        }
    }
    if (ffn_names_swapped) {
        fprintf(stderr, "  [vit] detected swapped FFN names (llama.cpp clip bug) — auto-correcting\n");
    }

    fprintf(stderr, "  [vit] FF=%d H=%d at layer load start\n", FF, H);

    // Transformer layers
    layers.resize(config.num_layers);
    for (int il = 0; il < config.num_layers; il++) {
        auto& l = layers[il];
        std::string p = "v.blk." + std::to_string(il) + ".";
        bool ok = true;

        ok &= get(p + "ln1.weight", l.ln1_w, (size_t)H);
        ok &= get(p + "ln1.bias",   l.ln1_b, (size_t)H);
        ok &= get(p + "ln2.weight", l.ln2_w, (size_t)H);
        ok &= get(p + "ln2.bias",   l.ln2_b, (size_t)H);

        if (has_fused_qkv) {
            // Fused QKV: attn_qkv.weight [H, 3*H] — split into Q, K, V
            std::vector<float> qkv_w, qkv_b;
            ok &= get(p + "attn_qkv.weight", qkv_w, (size_t)H * 3 * H);
            ok &= get(p + "attn_qkv.bias",   qkv_b, (size_t)3 * H);
            if (ok) {
                l.attn_q_w.resize((size_t)H * H);
                l.attn_k_w.resize((size_t)H * H);
                l.attn_v_w.resize((size_t)H * H);
                l.attn_q_b.resize((size_t)H);
                l.attn_k_b.resize((size_t)H);
                l.attn_v_b.resize((size_t)H);
                // Split: first H cols = Q, next H = K, last H = V
                for (int i = 0; i < H; i++) {
                    for (int j = 0; j < H; j++) {
                        l.attn_q_w[(size_t)i * H + j] = qkv_w[(size_t)i * 3 * H + j];
                        l.attn_k_w[(size_t)i * H + j] = qkv_w[(size_t)i * 3 * H + H + j];
                        l.attn_v_w[(size_t)i * H + j] = qkv_w[(size_t)i * 3 * H + 2 * H + j];
                    }
                    l.attn_q_b[i] = qkv_b[i];
                    l.attn_k_b[i] = qkv_b[H + i];
                    l.attn_v_b[i] = qkv_b[2 * H + i];
                }
            }
        } else {
            ok &= get(p + "attn_q.weight", l.attn_q_w, (size_t)H * H);
            ok &= get(p + "attn_q.bias",   l.attn_q_b, (size_t)H);
            ok &= get(p + "attn_k.weight", l.attn_k_w, (size_t)H * H);
            ok &= get(p + "attn_k.bias",   l.attn_k_b, (size_t)H);
            ok &= get(p + "attn_v.weight", l.attn_v_w, (size_t)H * H);
            ok &= get(p + "attn_v.bias",   l.attn_v_b, (size_t)H);
        }
        ok &= get(p + "attn_out.weight", l.attn_o_w, (size_t)H * H);
        ok &= get(p + "attn_out.bias",   l.attn_o_b, (size_t)H);

        if (ffn_names_swapped) {
            // Names are swapped: "ffn_down.weight" is really the up-projection (H->ff)
            // and "ffn_up.weight" is really the down-projection (ff->H)
            ok &= get(p + "ffn_down.weight", l.ffn_up_w,   (size_t)FF * H);
            ok &= get(p + "ffn_down.bias",   l.ffn_up_b,   (size_t)FF);
            ok &= get(p + "ffn_up.weight",   l.ffn_down_w, (size_t)H * FF);
            ok &= get(p + "ffn_up.bias",     l.ffn_down_b, (size_t)H);
        } else {
            ok &= get(p + "ffn_up.weight",   l.ffn_up_w,   (size_t)FF * H);
            ok &= get(p + "ffn_up.bias",     l.ffn_up_b,   (size_t)FF);
            ok &= get(p + "ffn_down.weight", l.ffn_down_w, (size_t)H * FF);
            ok &= get(p + "ffn_down.bias",   l.ffn_down_b, (size_t)H);
        }

        if (!ok) {
            fprintf(stderr, "  [vit] layer %d: missing required tensor\n", il);
            return false;
        }
    }

    // Post-LN
    if (!get("v.post_ln.weight", post_ln_w, (size_t)H)) return false;
    if (!get("v.post_ln.bias",   post_ln_b, (size_t)H)) return false;
    
    // Debug: print first few patch_embd values
    if (!patch_embd0.empty()) {
        double mean_w = 0, max_w = -1e30, min_w = 1e30;
        int n_nan = 0;
        for (size_t i = 0; i < patch_embd0.size() && i < 100000; i++) {
            float v = patch_embd0[i];
            if (std::isnan(v)) n_nan++;
            else { mean_w += v; max_w = std::max((double)v, max_w); min_w = std::min((double)v, min_w); }
        }
        fprintf(stderr, "  [vit] patch_embd0 sample: mean=%.4f range=[%.4f, %.4f] NaN=%d\n",
                mean_w / std::min((size_t)100000, patch_embd0.size()), min_w, max_w, n_nan);
    }

    // Auto-detect intermediate_size from loaded layer 0
    if (!layers.empty() && H > 0) {
        size_t ff_size = layers[0].ffn_up_w.size();
        if (ff_size > 0 && ff_size % (size_t)H == 0) {
            int detected_ff = (int)(ff_size / H);
            if (detected_ff != config.intermediate_size) {
                fprintf(stderr, "  [vit] auto-detected intermediate_size=%d from layer 0 weights (override %d)\n",
                        detected_ff, config.intermediate_size);
                config.intermediate_size = detected_ff;
            }
        }
    }

    // Projector — try to detect type from available tensors
    {
        size_t n = 0;
        if (read_tensor_f32(gguf_path, "mm.0.weight", mm0_w, &n)) {
            // mm.0 exists — this is a projector model
            int d0 = (int)sqrtf((float)n); // square matrix: [d0, d0]
            if (d0 * d0 != (int)n) d0 = n / config.hidden_size; // fallback

            // Check for mm.1 (3-layer MLP)
            if (read_tensor_f32(gguf_path, "mm.1.weight", mm1_w, &n)) {
                proj_config.type = ProjectorType::MLP_3LAYER;
                proj_config.mlp_hidden = (int)n / (int)sqrtf((float)n);
            }

            // Check for merger pattern: mm.2 expects [4*H, text_hidden] or similar
            if (read_tensor_f32(gguf_path, "mm.2.weight", mm2_w, &n)) {
                int proj_dim = (int)n / (4 * H);
                if ((size_t)proj_dim * 4 * H == n) {
                    // Qwen2-VL merger: mm.0[4*H, 4*H], mm.2[4*H, text_hidden]
                    proj_config.type = ProjectorType::QWEN2_MERGER;
                    proj_config.merge_group = 4;
                    proj_config.text_hidden = proj_dim;
                    proj_config.vision_hidden = H;
                    proj_config.mlp_hidden = 4 * H;
                } else {
                    // 2-layer MLP: mm.0[vision_hidden, mlp_hidden], mm.2[mlp_hidden, text_hidden]
                    proj_config.type = ProjectorType::MLP_2LAYER;
                    proj_config.mlp_hidden = d0;
                    proj_config.text_hidden = (int)n / d0;
                    proj_config.vision_hidden = H;
                }
            } else if (proj_config.type == ProjectorType::MLP_3LAYER) {
                proj_config.text_hidden = d0;
                proj_config.vision_hidden = H;
            } else {
                // Single linear: mm.0[vision_hidden, text_hidden]
                proj_config.type = ProjectorType::LINEAR;
                proj_config.text_hidden = (int)n / H;
                proj_config.vision_hidden = H;
            }

            // Load bias tensors
            get_opt("mm.0.bias", mm0_b, (size_t)proj_config.mlp_hidden > 0 ? (size_t)proj_config.mlp_hidden : (size_t)proj_config.text_hidden);
            if (proj_config.type == ProjectorType::MLP_3LAYER || proj_config.type == ProjectorType::QWEN2_MERGER) {
                if (mm1_w.size() > 0) get_opt("mm.1.bias", mm1_b, (size_t)proj_config.mlp_hidden);
                get_opt("mm.2.bias", mm2_b, (size_t)proj_config.text_hidden);
            }
            if (proj_config.type == ProjectorType::MLP_2LAYER) {
                get_opt("mm.2.bias", mm2_b, (size_t)proj_config.text_hidden);
            }
        } else {
            // No projector — identity mapping
            proj_config.type = ProjectorType::IDENTITY;
            proj_config.vision_hidden = H;
            proj_config.text_hidden = H;
        }
    }

    // Copy normalization constants if the GGUF file provides them
    float f32v;
    // (could read from GGUF metadata, but for now use config defaults)

    fprintf(stderr, "[vit] loaded ViT: H=%d L=%d NH=%d P=%d FF=%d projector=%d\n",
            config.hidden_size, config.num_layers, config.num_heads,
            config.patch_size, config.intermediate_size, (int)proj_config.type);
    return true;
}

void VisionWeights::clear() {
    patch_embd0.clear(); patch_embd1.clear(); patch_bias.clear();
    pos_embd.clear(); cls_embd.clear();
    pre_ln_w.clear(); pre_ln_b.clear();
    post_ln_w.clear(); post_ln_b.clear();
    layers.clear();
    mm0_w.clear(); mm0_b.clear();
    mm1_w.clear(); mm1_b.clear();
    mm2_w.clear(); mm2_b.clear();
}

// ====================================================================
// Core math implementations
// ====================================================================

void vit_patch_embed(const VisionWeights& weights, const float* img, int img_w, int img_h,
                      int patch_row, int patch_col, float* out) {
    int P = weights.config.patch_size;
    int H = weights.config.hidden_size;
    std::fill(out, out + H, 0.0f);

    const float* k0_base = weights.patch_embd0.data();
    const float* k1_base = weights.has_patch_embd1 ? weights.patch_embd1.data() : nullptr;

    for (int cin = 0; cin < 3; cin++) {
        for (int kh = 0; kh < P; kh++) {
            int py = patch_row * P + kh;
            if (py >= img_h) continue;
            for (int kw = 0; kw < P; kw++) {
                int px = patch_col * P + kw;
                if (px >= img_w) continue;
                float pix = img[((size_t)py * img_w + px) * 3 + cin];
                size_t kbase = (size_t)kw + (size_t)P * kh + (size_t)P * P * cin;
                for (int o = 0; o < H; o++) {
                    out[o] += pix * k0_base[(size_t)o * (size_t)P * P * 3 + kbase];
                    if (k1_base) {
                        out[o] += pix * k1_base[(size_t)o * (size_t)P * P * 3 + kbase];
                    }
                }
    // Add bias if present
    if (!weights.patch_bias.empty()) {
        for (int o = 0; o < H; o++) out[o] += weights.patch_bias[o];
    }
            }
        }
    }
}

void vit_rope2d_apply(float* x, int head_dim, int row, int col, float freq_base) {
    int half = head_dim / 2;
    int quarter = head_dim / 4;
    float theta_scale = powf(freq_base, -4.0f / head_dim);
    for (int ic = 0; ic < half; ic++) {
        int pos = (ic < quarter) ? row : col;
        int local = (ic < quarter) ? ic : (ic - quarter);
        float freq = pos * powf(theta_scale, (float)local);
        float c = cosf(freq), s = sinf(freq);
        float x0 = x[ic], x1 = x[ic + half];
        x[ic] = x0 * c - x1 * s;
        x[ic + half] = x0 * s + x1 * c;
    }
}

// ====================================================================
// ViT forward (full encoder)
// ====================================================================

std::vector<float> vit_forward(const VisionWeights& weights, const float* img,
                                int img_w, int img_h, const ProjectorConfig* proj_override) {
    using namespace vit_math;
    const auto& cfg = weights.config;
    int H = cfg.hidden_size, NH = cfg.num_heads, HD = H / NH;
    int P = cfg.patch_size, FF = cfg.intermediate_size;
    int pw = img_w / P, ph = img_h / P;
    int n_patches = pw * ph;
    int n_positions = pw * ph;

    // Use CLS token if available, which adds one position
    if (weights.has_cls_embd) n_positions += 1;

    // Allocate sequence buffer
    std::vector<float> seq((size_t)n_positions * H);
    std::vector<float> qkv_buf((size_t)n_positions * 3 * H); // Q, K, V storage for full attention
    std::vector<float> x2(H), att(H), scores(n_positions);
    std::vector<float> up(FF), down(H);

    // Pre-LN (optional)
    if (weights.has_pre_ln && !weights.pre_ln_w.empty()) {
        // Apply to each patch before transformer
    }

    // Patch embedding
    int ptr = 0;
    if (weights.has_cls_embd) {
        // CLS token at position 0
        std::copy(weights.cls_embd.begin(), weights.cls_embd.end(), seq.begin());
        ptr = 1;
    }
    for (int row = 0; row < ph; row++) {
        for (int col = 0; col < pw; col++) {
            vit_patch_embed(weights, img, img_w, img_h, row, col, &seq[(size_t)ptr * H]);
            ptr++;
        }
    }

    // Add absolute position embeddings (if available)
    if (weights.has_pos_embd && !weights.pos_embd.empty()) {
        for (int i = 0; i < n_positions && i < (int)weights.pos_embd.size() / H; i++) {
            for (int j = 0; j < H; j++) {
                seq[(size_t)i * H + j] += weights.pos_embd[(size_t)i * H + j];
            }
        }
    }

    // Transformer layers
    for (int il = 0; il < cfg.num_layers; il++) {
        const auto& l = weights.layers[il];

        // Pre-attention LayerNorm
        for (int t = 0; t < n_positions; t++) {
            float* xt = &seq[(size_t)t * H];

            if (cfg.use_bias) {
                layernorm(x2.data(), xt, l.ln1_w.data(), l.ln1_b.data(), H, cfg.layer_norm_eps);
            } else {
                rmsnorm(x2.data(), xt, l.ln1_w.data(), H, cfg.layer_norm_eps);
            }

            // QKV projection
            std::vector<float> q(H), k(H), v(H);
            matmul(q.data(), x2.data(), l.attn_q_w.data(), H, H);
            matmul(k.data(), x2.data(), l.attn_k_w.data(), H, H);
            matmul(v.data(), x2.data(), l.attn_v_w.data(), H, H);
            for (int i = 0; i < H; i++) {
                if (cfg.use_bias) {
                    q[i] += l.attn_q_b[i];
                    k[i] += l.attn_k_b[i];
                    v[i] += l.attn_v_b[i];
                }
            }

            // Store Q, K, V for full attention
            std::copy(q.begin(), q.end(), &qkv_buf[(size_t)t * 3 * H]);
            std::copy(k.begin(), k.end(), &qkv_buf[(size_t)t * 3 * H + H]);
            std::copy(v.begin(), v.end(), &qkv_buf[(size_t)t * 3 * H + 2*H]);

            // TODO: 2D RoPE could be applied here per-head for Qwen2-VL style
        }

        // Bidirectional full attention
        float scale = 1.0f / sqrtf((float)HD);
        for (int t = 0; t < n_positions; t++) {
            std::fill(att.begin(), att.end(), 0.0f);

            for (int h = 0; h < NH; h++) {
                float* Qh = &qkv_buf[(size_t)t * 3 * H] + (size_t)h * HD;

                // Attention scores for this head
                for (int s = 0; s < n_positions; s++) {
                    float* Kh = &qkv_buf[(size_t)s * 3 * H + H] + (size_t)h * HD;
                    float acc = 0;
                    for (int d = 0; d < HD; d++) acc += Qh[d] * Kh[d];
                    scores[s] = acc * scale;
                }

                // Softmax
                softmax_inplace(scores.data(), n_positions);

                // Weighted sum of values
                for (int d = 0; d < HD; d++) {
                    float acc = 0;
                    for (int s = 0; s < n_positions; s++) {
                        acc += scores[s] * qkv_buf[(size_t)s * 3 * H + 2*H + (size_t)h * HD + d];
                    }
                    att[(size_t)h * HD + d] = acc;
                }
            }

            // Output projection + residual
            std::vector<float> att_out(H);
            matmul(att_out.data(), att.data(), l.attn_o_w.data(), H, H);
            for (int i = 0; i < H; i++) {
                if (cfg.use_bias) att_out[i] += l.attn_o_b[i];
            }
            float* xt = &seq[(size_t)t * H];
            for (int i = 0; i < H; i++) xt[i] += att_out[i];
        }

        // FFN
        for (int t = 0; t < n_positions; t++) {
            float* xt = &seq[(size_t)t * H];

            // Pre-FFN LayerNorm
            if (cfg.use_bias) {
                layernorm(x2.data(), xt, l.ln2_w.data(), l.ln2_b.data(), H, cfg.layer_norm_eps);
            } else {
                rmsnorm(x2.data(), xt, l.ln2_w.data(), H, cfg.layer_norm_eps);
            }

            // Up-projection
            matmul(up.data(), x2.data(), l.ffn_up_w.data(), FF, H);
            for (int i = 0; i < FF; i++) if (cfg.use_bias) up[i] += l.ffn_up_b[i];

            // Activation
            if (cfg.use_gelu) {
                gelu(up.data(), up.data(), FF);
            } else {
                // Default to GELU (fallback)
                gelu(up.data(), up.data(), FF);
            }

            // Down-projection + residual
            matmul(down.data(), up.data(), l.ffn_down_w.data(), H, FF);
            for (int i = 0; i < H; i++) {
                if (cfg.use_bias) down[i] += l.ffn_down_b[i];
                xt[i] += down[i];
            }
        }
    }

    // Post-LN
    for (int t = 0; t < n_positions; t++) {
        float* xt = &seq[(size_t)t * H];
        if (cfg.use_bias) {
            layernorm(x2.data(), xt, weights.post_ln_w.data(), weights.post_ln_b.data(), H, cfg.layer_norm_eps);
        } else {
            rmsnorm(x2.data(), xt, weights.post_ln_w.data(), H, cfg.layer_norm_eps);
        }
        std::copy(x2.begin(), x2.end(), xt);
    }

    // Remove CLS token if present (only keep patch tokens)
    std::vector<float> patch_outputs((size_t)n_patches * H);
    int src_offset = weights.has_cls_embd ? 1 : 0;
    std::copy(seq.begin() + (size_t)src_offset * H,
              seq.begin() + (size_t)(src_offset + n_patches) * H,
              patch_outputs.begin());

    // Apply projector
    const auto& proj = proj_override ? *proj_override : weights.proj_config;
    if (proj.type == ProjectorType::IDENTITY || proj.text_hidden == H) {
        return patch_outputs;
    }

    int text_hidden = proj.text_hidden;
    std::vector<float> projected((size_t)n_patches * text_hidden);
    std::vector<float> hidden_buf(proj.mlp_hidden > 0 ? (size_t)proj.mlp_hidden : (size_t)text_hidden);

    for (int t = 0; t < n_patches; t++) {
        const float* patch = &patch_outputs[(size_t)t * H];

        if (proj.type == ProjectorType::QWEN2_MERGER) {
            // Qwen2-VL merger: group 4 tokens -> mm.0 -> GELU -> mm.2
            // This is handled by the caller for group-level merging
            // For single-token (fallback), just apply linear
            matmul(hidden_buf.data(), patch, weights.mm0_w.data(), proj.mlp_hidden, H);
            for (int i = 0; i < proj.mlp_hidden; i++) {
                if (!weights.mm0_b.empty()) hidden_buf[i] += weights.mm0_b[i];
            }
            gelu(hidden_buf.data(), hidden_buf.data(), proj.mlp_hidden);
            matmul(&projected[(size_t)t * text_hidden], hidden_buf.data(), weights.mm2_w.data(), text_hidden, proj.mlp_hidden);
            for (int i = 0; i < text_hidden; i++) {
                if (!weights.mm2_b.empty()) projected[(size_t)t * text_hidden + i] += weights.mm2_b[i];
            }
        } else if (proj.type == ProjectorType::MLP_2LAYER) {
            // 2-layer MLP: mm.0 [H, mlp_hidden], mm.2 [mlp_hidden, text_hidden]
            matmul(hidden_buf.data(), patch, weights.mm0_w.data(), proj.mlp_hidden, H);
            for (int i = 0; i < proj.mlp_hidden; i++) {
                if (!weights.mm0_b.empty()) hidden_buf[i] += weights.mm0_b[i];
            }
            gelu(hidden_buf.data(), hidden_buf.data(), proj.mlp_hidden);
            matmul(&projected[(size_t)t * text_hidden], hidden_buf.data(), weights.mm2_w.data(), text_hidden, proj.mlp_hidden);
            for (int i = 0; i < text_hidden; i++) {
                if (!weights.mm2_b.empty()) projected[(size_t)t * text_hidden + i] += weights.mm2_b[i];
            }
        } else {
            // Linear: mm.0 [H, text_hidden]
            matmul(&projected[(size_t)t * text_hidden], patch, weights.mm0_w.data(), text_hidden, H);
            for (int i = 0; i < text_hidden; i++) {
                if (!weights.mm0_b.empty()) projected[(size_t)t * text_hidden + i] += weights.mm0_b[i];
            }
        }
    }

    return projected;
}

// ====================================================================
// Image preprocessing
// ====================================================================

std::vector<float> vit_preprocess(const uint8_t* src, int src_w, int src_h,
                                   int out_w, int out_h,
                                   const float* mean, const float* std) {
    std::vector<float> dst((size_t)out_w * out_h * 3);
    for (int y = 0; y < out_h; y++) {
        float sy = (y + 0.5f) * src_h / out_h - 0.5f;
        int y0 = (int)floorf(sy); float fy = sy - y0;
        int y1 = std::min(std::max(y0 + 1, 0), src_h - 1);
        y0 = std::min(std::max(y0, 0), src_h - 1);
        for (int x = 0; x < out_w; x++) {
            float sx = (x + 0.5f) * src_w / out_w - 0.5f;
            int x0 = (int)floorf(sx); float fx = sx - x0;
            int x1 = std::min(std::max(x0 + 1, 0), src_w - 1);
            x0 = std::min(std::max(x0, 0), src_w - 1);
            for (int c = 0; c < 3; c++) {
                float v00 = src[(y0 * src_w + x0) * 3 + c];
                float v01 = src[(y0 * src_w + x1) * 3 + c];
                float v10 = src[(y1 * src_w + x0) * 3 + c];
                float v11 = src[(y1 * src_w + x1) * 3 + c];
                float v0 = v00 * (1 - fx) + v01 * fx;
                float v1 = v10 * (1 - fx) + v11 * fx;
                float v = (v0 * (1 - fy) + v1 * fy) / 255.0f;
                v = (v - mean[c]) / std[c];
                dst[((size_t)y * out_w + x) * 3 + c] = v;
            }
        }
    }
    return dst;
}

// stb_image is already implemented in vl_processor.cpp (single TU rule)
#include "../third_party/stb/stb_image.h"

std::vector<float> vit_load_and_preprocess(const std::string& image_path,
                                            int out_w, int out_h,
                                            const float* mean, const float* std) {

    int sw, sh, comp;
    uint8_t* img_u8 = stbi_load(image_path.c_str(), &sw, &sh, &comp, 3);
    if (!img_u8) {
        fprintf(stderr, "[vit] FAIL: could not load image '%s'\n", image_path.c_str());
        return {};
    }

    auto result = vit_preprocess(img_u8, sw, sh, out_w, out_h, mean, std);
    stbi_image_free(img_u8);
    return result;
}

// ═══════════════════════════════════════════════════════════════════
// Q4NX tile dequant helpers (matching 1BP tile layout)
// ═══════════════════════════════════════════════════════════════════

static uint16_t bf16_from_bytes(const uint8_t* b) {
    return (uint16_t)b[0] | ((uint16_t)b[1] << 8);
}

static float bf16_to_f32(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

void dequant_q4nx_row(const uint8_t* row_data, float* out, int n_groups, int gs) {
    for (int g = 0; g < n_groups; g++) {
        float scale = bf16_to_f32(bf16_from_bytes(row_data + g * 2));
        float zp    = bf16_to_f32(bf16_from_bytes(row_data + n_groups * 2 + g * 2));
        if (!std::isfinite(scale)) scale = 0.0f;
        if (!std::isfinite(zp)) zp = 0.0f;
        const uint8_t* pk = row_data + n_groups * 4;
        for (int i = 0; i < gs && g * gs + i < n_groups * gs; i++) {
            int byte_idx = g * gs / 2 + i / 2;
            int nibble = (i & 1) ? (pk[byte_idx] >> 4) : (pk[byte_idx] & 0x0F);
            float dqv = (float)nibble * scale + zp;
            out[g * gs + i] = std::isfinite(dqv) ? dqv : 0.0f;
        }
    }
}

void dequant_q4nx_tile(const uint8_t* tile_data, float* out,
                        int tr, int tc, int gs) {
    int n_groups = tc / gs;
    int row_bytes = n_groups * 4 + tc / 2;
    for (int r = 0; r < tr; r++)
        dequant_q4nx_row(tile_data + r * row_bytes, out + r * tc, n_groups, gs);
}

#include "onebp_loader.h"

static bool find_tensor_1bp(OnebpModel& mdl, const std::string& suffix,
                             std::vector<float>& dst) {
    for (auto& t : mdl.tensors) {
        bool match = t.name == suffix || t.name == "model." + suffix ||
            (t.name.size() >= suffix.size() &&
             t.name.compare(t.name.size() - suffix.size(), suffix.size(), suffix) == 0);
        if (!match) continue;
        uint8_t* raw = mdl.tensor_data(t);
        if (!raw) return false;
        if (t.ndim == 1) {
            int n = (int)t.dims[0];
            dst.resize(n);
            const uint16_t* f16 = (const uint16_t*)raw;
            for (int i = 0; i < n; i++) {
                uint32_t bits = (uint32_t)f16[i] << 16;
                memcpy(&dst[i], &bits, 4);
            }
            return true;
        }
        if (t.ndim != 2) continue;
        int rows = (int)t.dims[0];
        int cols = (int)t.dims[1];
        size_t f16_sz = (size_t)rows * cols * 2;
        size_t f32_sz = (size_t)rows * cols * 4;
        bool is_f16 = (t.bytes == f16_sz || t.bytes == f32_sz);
        if (is_f16) {
            dst.resize((size_t)rows * cols);
            if (t.bytes == f32_sz) {
                memcpy(dst.data(), raw, f32_sz);
            } else {
                const uint16_t* f16 = (const uint16_t*)raw;
                for (size_t i = 0; i < (size_t)rows * cols; i++) {
                    uint32_t bits = (uint32_t)f16[i] << 16;
                    memcpy(&dst[i], &bits, 4);
                }
            }
            return true;
        }
        int tr = 32, tc2 = 256, gs = 32;
        int ntr = (rows + tr - 1) / tr;
        int ntc = (cols + tc2 - 1) / tc2;
        int row_bytes = (tc2 / gs) * 4 + tc2 / 2;
        int tile_bytes = tr * row_bytes;
        dst.resize((size_t)rows * cols);
        size_t data_off = 0;
        for (int ti = 0; ti < ntr; ti++) {
            for (int tj = 0; tj < ntc; tj++) {
                int r0 = ti * tr, c0 = tj * tc2;
                int rh = std::min(tr, rows - r0);
                int ch = std::min(tc2, cols - c0);
                std::vector<float> tile_buf((size_t)tr * tc2);
                dequant_q4nx_tile(raw + data_off, tile_buf.data(), tr, tc2, gs);
                data_off += tile_bytes;
                for (int r = 0; r < rh; r++)
                    for (int c = 0; c < ch; c++)
                        dst[(size_t)(r0 + r) * cols + (c0 + c)] = tile_buf[(size_t)r * tc2 + c];
            }
        }
        return true;
    }
    return false;
}

bool mage_vit_load_weights_1bp(const char* path, VisionWeights& vw) {
    OnebpModel mdl;
    if (!mdl.load(path)) {
        fprintf(stderr, "[mage_vit] FAIL: load %s\n", path);
        return false;
    }
    auto& h = mdl.header;
    // Model dimensions come from OnebpHeader::reserved[0..5].
    // If ALL are 0 (e.g., freshly converted .1bp without ViT metadata),
    // fail with a clear message rather than silently using wrong fallback (issue #1158).
    bool all_zero = true;
    for (int i = 0; i < 6; i++) { if (h.reserved[i] != 0) { all_zero = false; break; } }
    if (all_zero) {
        fprintf(stderr, "[mage_vit] FAIL: 1BP reserved fields all zero — "
                "ViT model dimensions not populated.\n"
                "Re-convert with a converter that writes ViT metadata.\n");
        return false;
    }
    int H = h.reserved[0] > 0 ? h.reserved[0] : 1024;
    int NL = h.reserved[1] > 0 ? h.reserved[1] : 24;
    int NH = h.reserved[2] > 0 ? h.reserved[2] : 16;
    int FF = h.reserved[3] > 0 ? h.reserved[3] : 4096;
    int PS = h.reserved[5] > 0 ? h.reserved[5] : 16;
    vw.config = VitConfig::mage_vit();
    vw.config.hidden_size = H;
    vw.config.num_layers = NL;
    vw.config.num_heads = NH;
    vw.config.intermediate_size = FF;
    vw.config.patch_size = PS;
    vw.layers.resize(NL);
    fprintf(stderr, "[mage_vit] 1BP: H=%d L=%d NH=%d FF=%d P=%d\n", H, NL, NH, FF, PS);
    auto find = [&](const std::string& suf, std::vector<float>& dst) { return find_tensor_1bp(mdl, suf, dst); };
    find("visual.embeddings.patch_embedding.weight", vw.patch_embd0);
    vw.has_pre_ln = find("visual.layernorm_pre.weight", vw.pre_ln_w);
    if (vw.has_pre_ln) {
        find("visual.layernorm_pre.bias", vw.pre_ln_b);
        if (vw.pre_ln_b.empty()) vw.pre_ln_b.resize(H, 0.0f);
    }
    find("visual.layernorm_pre.weight", vw.post_ln_w);
    find("visual.layernorm_pre.bias", vw.post_ln_b);
    if (vw.post_ln_b.empty() && !vw.post_ln_w.empty()) vw.post_ln_b.resize(H, 0.0f);
    find("visual.merger.ln_q.weight", vw.mm1_w);
    find("visual.merger.ln_q.bias", vw.mm1_b);
    find("visual.merger.mlp.0.weight", vw.mm0_w);
    find("visual.merger.mlp.0.bias", vw.mm0_b);
    find("visual.merger.mlp.2.weight", vw.mm2_w);
    find("visual.merger.mlp.2.bias", vw.mm2_b);
    if (vw.post_ln_w.empty() && !vw.pre_ln_w.empty() && NL > 0) {
        vw.post_ln_w = vw.pre_ln_w;
        vw.post_ln_b = vw.pre_ln_b;
    }
    for (int il = 0; il < NL; il++) {
        auto& l = vw.layers[il];
        std::string p = "visual.encoder.layers." + std::to_string(il) + ".";
        find(p + "layer_norm1.weight", l.ln1_w);
        find(p + "layer_norm1.bias", l.ln1_b);
        if (l.ln1_b.empty() && !l.ln1_w.empty()) l.ln1_b.resize(H, 0.0f);
        find(p + "layer_norm2.weight", l.ln2_w);
        find(p + "layer_norm2.bias", l.ln2_b);
        if (l.ln2_b.empty() && !l.ln2_w.empty()) l.ln2_b.resize(H, 0.0f);
        std::vector<float> qkv_w, qkv_b;
        if (find(p + "self_attn.qkv.weight", qkv_w)) {
            l.attn_q_w.resize((size_t)H * H);
            l.attn_k_w.resize((size_t)H * H);
            l.attn_v_w.resize((size_t)H * H);
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < H; j++) {
                    l.attn_q_w[(size_t)i * H + j] = qkv_w[(size_t)i * 3 * H + j];
                    l.attn_k_w[(size_t)i * H + j] = qkv_w[(size_t)i * 3 * H + H + j];
                    l.attn_v_w[(size_t)i * H + j] = qkv_w[(size_t)i * 3 * H + 2*H + j];
                }
            }
            if (find(p + "self_attn.qkv.bias", qkv_b)) {
                l.attn_q_b.resize(H); l.attn_k_b.resize(H); l.attn_v_b.resize(H);
                for (int i = 0; i < H; i++) {
                    l.attn_q_b[i] = qkv_b[i];
                    l.attn_k_b[i] = qkv_b[H + i];
                    l.attn_v_b[i] = qkv_b[2*H + i];
                }
            } else {
                l.attn_q_b.resize(H, 0.0f); l.attn_k_b.resize(H, 0.0f); l.attn_v_b.resize(H, 0.0f);
            }
        }
        find(p + "self_attn.proj.weight", l.attn_o_w);
        find(p + "self_attn.proj.bias", l.attn_o_b);
        if (l.attn_o_b.empty() && !l.attn_o_w.empty()) l.attn_o_b.resize(H, 0.0f);
        find(p + "mlp.fc1.weight", l.ffn_up_w);
        find(p + "mlp.fc1.bias", l.ffn_up_b);
        if (l.ffn_up_b.empty() && !l.ffn_up_w.empty()) l.ffn_up_b.resize(FF, 0.0f);
        find(p + "mlp.fc2.weight", l.ffn_down_w);
        find(p + "mlp.fc2.bias", l.ffn_down_b);
        if (l.ffn_down_b.empty() && !l.ffn_down_w.empty()) l.ffn_down_b.resize(H, 0.0f);
    }
    fprintf(stderr, "[mage_vit] Load complete: %d layers, %s merger\n", NL, vw.mm0_w.empty() ? "no" : "yes");
    return true;
}

// ═══ Mage-ViT: 3D interleaved RoPE ═══════════════════════════════
static void mage_vit_rope_one(float* x, int head_dim,
                               int t, int h, int w, float theta_base) {
    int half = head_dim / 2;
    int t_pairs = head_dim * 4 / 32;
    int h_pairs = head_dim * 6 / 32;
    int w_pairs = head_dim * 6 / 32;
    int off = 0;
    for (int i = 0; i < t_pairs; i++) {
        float freq = t * powf(theta_base, -2.0f * (float)(off + i) / (float)head_dim);
        float c = cosf(freq), s = sinf(freq);
        float x0 = x[off + i], x1 = x[off + i + half];
        x[off + i] = x0 * c - x1 * s;
        x[off + i + half] = x0 * s + x1 * c;
    }
    off += t_pairs;
    for (int i = 0; i < h_pairs; i++) {
        float freq = h * powf(theta_base, -2.0f * (float)(off + i) / (float)head_dim);
        float c = cosf(freq), s = sinf(freq);
        float x0 = x[off + i], x1 = x[off + i + half];
        x[off + i] = x0 * c - x1 * s;
        x[off + i + half] = x0 * s + x1 * c;
    }
    off += h_pairs;
    for (int i = 0; i < w_pairs; i++) {
        float freq = w * powf(theta_base, -2.0f * (float)(off + i) / (float)head_dim);
        float c = cosf(freq), s = sinf(freq);
        float x0 = x[off + i], x1 = x[off + i + half];
        x[off + i] = x0 * c - x1 * s;
        x[off + i + half] = x0 * s + x1 * c;
    }
}

void vit_rope3d_apply(float* q, float* k, int head_dim,
                       int t, int h, int w, float freq_base) {
    mage_vit_rope_one(q, head_dim, t, h, w, freq_base);
    mage_vit_rope_one(k, head_dim, t, h, w, freq_base);
}

// ═══ Mage-ViT full forward ════════════════════════════════════════
std::vector<float> mage_vit_forward(
    const VisionWeights& weights,
    const float* pixels, int channels, int time, int height, int width,
    int frame_window_size) {
    using namespace vit_math;
    const auto& cfg = weights.config;
    int H = cfg.hidden_size, NH = cfg.num_heads, HD = H / NH;
    int P = cfg.patch_size, FF = cfg.intermediate_size;
    int ph = height / P, pw = width / P;
    int ppf = ph * pw, tp = ppf * time;
    std::vector<float> seq((size_t)tp * H);
    std::vector<float> qb((size_t)tp * H), kb((size_t)tp * H), vb((size_t)tp * H);
    std::vector<float> x2(H), att(H), up(FF), dn(H);
    int cp3 = P * P * channels;
    for (int t = 0; t < time; t++)
        for (int r = 0; r < ph; r++)
            for (int c = 0; c < pw; c++) {
                int pi = t * ppf + r * pw + c;
                float* out = &seq[(size_t)pi * H];
                std::fill(out, out + H, 0.0f);
                const float* w = weights.patch_embd0.data();
                if (!w || weights.patch_embd0.empty()) return {};
                for (int ch = 0; ch < channels; ch++)
                    for (int ky = 0; ky < P; ky++) {
                        int py = r * P + ky;
                        if (py >= height) continue;
                        for (int kx = 0; kx < P; kx++) {
                            int px = c * P + kx;
                            if (px >= width) continue;
                            float pix = pixels[((size_t)t * height * width + (size_t)py * width + px) * channels + ch];
                            size_t wo = (size_t)ch * P * P + (size_t)ky * P + kx;
                            for (int o = 0; o < H; o++)
                                out[o] += pix * w[(size_t)o * (size_t)cp3 + wo];
                        }
                    }
                if (!weights.patch_bias.empty())
                    for (int o = 0; o < H; o++) out[o] += weights.patch_bias[o];
            }
    if (weights.has_pre_ln && !weights.pre_ln_w.empty())
        for (int i = 0; i < tp; i++) {
            layernorm(x2.data(), &seq[(size_t)i * H],
                      weights.pre_ln_w.data(), weights.pre_ln_b.data(), H, cfg.layer_norm_eps);
            std::copy(x2.begin(), x2.end(), &seq[(size_t)i * H]);
        }
    float ascale = 1.0f / sqrtf((float)HD);
    int nw = (time + frame_window_size - 1) / frame_window_size;
    float attn_scr[4096];
    for (int il = 0; il < cfg.num_layers; il++) {
        const auto& l = weights.layers[il];
        for (int i = 0; i < tp; i++) {
            float* xt = &seq[(size_t)i * H];
            layernorm(x2.data(), xt, l.ln1_w.data(), l.ln1_b.data(), H, cfg.layer_norm_eps);
            matmul(&qb[(size_t)i * H], x2.data(), l.attn_q_w.data(), H, H);
            matmul(&kb[(size_t)i * H], x2.data(), l.attn_k_w.data(), H, H);
            matmul(&vb[(size_t)i * H], x2.data(), l.attn_v_w.data(), H, H);
            for (int j = 0; j < H; j++) {
                qb[(size_t)i * H + j] += l.attn_q_b[j];
                kb[(size_t)i * H + j] += l.attn_k_b[j];
                vb[(size_t)i * H + j] += l.attn_v_b[j];
            }
            int t = i / ppf, ipf = i % ppf, row = ipf / pw, col = ipf % pw;
            for (int h = 0; h < NH; h++) {
                mage_vit_rope_one(&qb[(size_t)i * H + (size_t)h * HD], HD, t, row, col, 10000.0f);
                mage_vit_rope_one(&kb[(size_t)i * H + (size_t)h * HD], HD, t, row, col, 10000.0f);
            }
        }
        for (int w = 0; w < nw; w++) {
            int ts = w * frame_window_size, te = std::min(ts + frame_window_size, time);
            int nt = (te - ts) * ppf, base = ts * ppf;
            for (int i = 0; i < nt; i++) {
                int ai = base + i;
                std::fill(att.begin(), att.end(), 0.0f);
                for (int h = 0; h < NH; h++) {
                    float* Qh = &qb[(size_t)ai * H + (size_t)h * HD];
                    for (int sj = 0; sj < nt; sj++) {
                        int aj = base + sj;
                        float* Kh = &kb[(size_t)aj * H + (size_t)h * HD];
                        float acc = 0;
                        for (int d = 0; d < HD; d++) acc += Qh[d] * Kh[d];
                        attn_scr[sj] = acc * ascale;
                    }
                    softmax_inplace(attn_scr, nt);
                    for (int d = 0; d < HD; d++) {
                        float acc = 0;
                        for (int sj = 0; sj < nt; sj++) {
                            int aj = base + sj;
                            acc += attn_scr[sj] * vb[(size_t)aj * H + (size_t)h * HD + d];
                        }
                        att[(size_t)h * HD + d] = acc;
                    }
                }
                float* xt = &seq[(size_t)ai * H];
                float att_out[4096];
                matmul(att_out, att.data(), l.attn_o_w.data(), H, H);
                for (int j = 0; j < H; j++) {
                    if (cfg.use_bias) att_out[j] += l.attn_o_b[j];
                    xt[j] += att_out[j];
                }
            }
        }
        for (int i = 0; i < tp; i++) {
            float* xt = &seq[(size_t)i * H];
            layernorm(x2.data(), xt, l.ln2_w.data(), l.ln2_b.data(), H, cfg.layer_norm_eps);
            matmul(up.data(), x2.data(), l.ffn_up_w.data(), FF, H);
            if (cfg.use_bias) for (int j = 0; j < FF; j++) up[j] += l.ffn_up_b[j];
            gelu(up.data(), up.data(), FF);
            matmul(dn.data(), up.data(), l.ffn_down_w.data(), H, FF);
            if (cfg.use_bias) for (int j = 0; j < H; j++) dn[j] += l.ffn_down_b[j];
            for (int j = 0; j < H; j++) xt[j] += dn[j];
        }
    }
    // (No post-LN — goes directly to merger)
    // 2x2 spatial merger
    if (!weights.mm0_w.empty()) {
        int pm = (int)(weights.mm0_w.size() / (4 * H));
        int mph = ph / 2, mpw = pw / 2;
        int mpf = mph * mpw, tm = mpf * time;
        int th = weights.mm2_w.empty() ? pm : (int)(weights.mm2_w.size() / pm);
        std::vector<float> merged((size_t)tm * th), mb(4*H), lb(4*H), hid(pm);
        for (int t = 0; t < time; t++)
            for (int mr = 0; mr < mph; mr++)
                for (int mc = 0; mc < mpw; mc++) {
                    for (int dr = 0; dr < 2; dr++)
                        for (int dc = 0; dc < 2; dc++) {
                            int si = t * ppf + (mr * 2 + dr) * pw + (mc * 2 + dc);
                            std::copy(seq.begin() + (size_t)si * H, seq.begin() + (size_t)si * H + H,
                                      mb.begin() + (size_t)(dr * 2 + dc) * H);
                        }
                    float mn = 0; for (int i = 0; i < 4*H; i++) mn += mb[i];
                    mn /= (4*H);
                    float vr = 0; for (int i = 0; i < 4*H; i++) { float d = mb[i] - mn; vr += d*d; }
                    vr /= (4*H);
                    float inv = 1.0f / sqrtf(vr + 1e-6f);
                    for (int i = 0; i < 4*H; i++) lb[i] = (mb[i] - mn) * inv;
                    if (!weights.mm1_w.empty() && (int)weights.mm1_w.size() == 4*H) {
                        for (int i = 0; i < 4*H; i++)
                            lb[i] = lb[i] * weights.mm1_w[i] + (weights.mm1_b.empty() ? 0.0f : weights.mm1_b[i % (int)weights.mm1_b.size()]);
                    }
                    matmul(hid.data(), lb.data(), weights.mm0_w.data(), pm, 4*H);
                    if (!weights.mm0_b.empty() && (int)weights.mm0_b.size() == pm)
                        for (int i = 0; i < pm; i++) hid[i] += weights.mm0_b[i];
                    gelu(hid.data(), hid.data(), pm);
                    int di = t * mpf + mr * mpw + mc;
                    matmul(&merged[(size_t)di * th], hid.data(), weights.mm2_w.data(), th, pm);
                    if (!weights.mm2_b.empty() && (int)weights.mm2_b.size() == th)
                        for (int i = 0; i < th; i++)
                            merged[(size_t)di * th + i] += weights.mm2_b[i];
                }
        return merged;
    }
    return seq;
}
