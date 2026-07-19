// vision_qwen2vl_poc.cpp — minimal proof-of-concept: real image in, real text out,
// through a from-scratch Qwen2-VL vision encoder (ViT + qwen2vl_merger) spliced
// into the existing GenericBackend (CPU) Qwen2 text decoder.
//
// Reference for the vision tower's exact math: llama.cpp's tools/mtmd/clip.cpp
// and tools/mtmd/models/qwen2vl.cpp (clip_graph_qwen2vl::build()). Key facts
// this implementation depends on, verified against that source:
//   - patch embed = conv2d(patch_embd.weight, img) + conv2d(patch_embd.weight.1, img)
//     (both temporal-frame kernels applied to the SAME still image)
//   - patch sequence is reordered into consecutive 2x2 spatial blocks (not raster
//     order) so the later "group every 4 tokens" merger step is a contiguous reshape
//   - self-attention RoPE is 2D ("M-RoPE" vision mode): within each head_dim/2 pair
//     (x[i], x[i+head_dim/2]), the first quarter of frequency bands uses the patch's
//     row position, the second quarter uses its column position — see
//     ggml_mrope_cache_init / GGML_ROPE_TYPE_VISION in ggml-cpu/ops.cpp
//   - this specific GGUF export has ffn_up.weight/ffn_down.weight NAMES SWAPPED
//     (a known legacy llama.cpp clip.cpp bug, auto-corrected there via
//     is_ffn_swapped; replicated here by loading "ffn_down.weight" as the real
//     up-projection and "ffn_up.weight" as the real down-projection)
//   - no pre_ln tensor exists for this projector type — skipped
//   - merger: reshape every 4 consecutive tokens -> mm.0 (5120->5120) -> GELU -> mm.2 (5120->1536)

#include "backend.h"
#include "model_discovery.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb/stb_image.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

// ── Minimal GGUF string-array metadata reader (just for tokenizer.ggml.tokens) ──
static bool read_gguf_string_array(const std::string& path, const std::string& key,
                                    std::vector<std::string>& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GGUF", 4) != 0) { fclose(f); return false; }
    uint32_t ver; fread(&ver, 4, 1, f);
    uint64_t tc, kc; fread(&tc, 8, 1, f); fread(&kc, 8, 1, f);
    auto read_str = [&](std::string& s) {
        uint64_t l; fread(&l, 8, 1, f); s.resize(l); if (l) fread(&s[0], 1, l, f);
    };
    bool found = false;
    for (uint64_t i = 0; i < kc && !found; i++) {
        std::string k; read_str(k);
        uint32_t vt; fread(&vt, 4, 1, f);
        if (vt == 9 && k == key) {
            uint32_t at; fread(&at, 4, 1, f);
            uint64_t an; fread(&an, 8, 1, f);
            out.resize(an);
            for (uint64_t j = 0; j < an; j++) {
                if (at == 8) read_str(out[j]);
                else { uint8_t tmp[8]; fread(tmp, 1, 8, f); }
            }
            found = true;
        } else {
            switch (vt) {
                case 0: case 1: case 7: fseek(f, 1, SEEK_CUR); break;
                case 2: case 3: fseek(f, 2, SEEK_CUR); break;
                case 4: case 5: case 6: fseek(f, 4, SEEK_CUR); break;
                case 8: { std::string tmp; read_str(tmp); break; }
                case 9: {
                    uint32_t at; fread(&at, 4, 1, f);
                    uint64_t an; fread(&an, 8, 1, f);
                    if (at == 8) { for (uint64_t j = 0; j < an; j++) { std::string tmp; read_str(tmp); } }
                    else {
                        int sz = 4;
                        if (at <= 7) { static const int s[] = {1,1,2,2,4,4,4,1}; sz = s[at]; }
                        else if (at >= 10 && at <= 12) sz = 8;
                        fseek(f, an * sz, SEEK_CUR);
                    }
                    break;
                }
                case 10: case 11: case 12: fseek(f, 8, SEEK_CUR); break;
                default: break;
            }
        }
    }
    fclose(f);
    return found;
}

// ── Shared math helpers (same conventions as GenericBackend) ──
// out[i] = sum_j in[j] * w[i*K+j] — w stored [out_dim, in_dim] row-major,
// matching GGUF's native flat tensor layout (ne[0]=in_dim fastest).
static void matmul(float* out, const float* in, const float* w, int M, int K) {
    for (int i = 0; i < M; i++) {
        float s = 0;
        for (int j = 0; j < K; j++) s += in[j] * w[(size_t)i * K + j];
        out[i] = s;
    }
}

static void layernorm(float* out, const float* x, const float* w, const float* b, int n, float eps) {
    float mean = 0; for (int i = 0; i < n; i++) mean += x[i]; mean /= n;
    float var = 0; for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; } var /= n;
    float inv = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
}

static void gelu(float* out, const float* x, int n) {
    const float c = 0.7978845608f;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        out[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
    }
}

// ── Vision tower weights ──
struct VitLayer {
    std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b;
    std::vector<float> q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b;
    std::vector<float> up_w, up_b, down_w, down_b; // real orientation, post name-swap-fix
};

struct VisionModel {
    int hidden = 1280, n_layers = 32, n_heads = 16, patch = 14;
    int ff = 5120, proj_dim = 1536;
    float eps = 1e-6f;
    float img_mean[3] = {0.48145467f, 0.45782750f, 0.40821072f};
    float img_std[3]  = {0.26862955f, 0.26130259f, 0.27577710f};
    std::vector<float> patch_embd0, patch_embd1; // each [14,14,3,1280] flat
    std::vector<VitLayer> layers;
    std::vector<float> post_ln_w, post_ln_b;
    std::vector<float> mm0_w, mm0_b, mm2_w, mm2_b;

    bool load(const std::string& path) {
        auto get = [&](const std::string& name, std::vector<float>& dst, size_t expect) -> bool {
            size_t n = 0;
            if (!read_gguf_tensor(path, name, dst, &n)) {
                fprintf(stderr, "  missing tensor: %s\n", name.c_str());
                return false;
            }
            if (n != expect) {
                fprintf(stderr, "  %s: expected %zu got %zu\n", name.c_str(), expect, n);
                return false;
            }
            return true;
        };
        int H = hidden;
        if (!get("v.patch_embd.weight", patch_embd0, (size_t)patch * patch * 3 * H)) return false;
        if (!get("v.patch_embd.weight.1", patch_embd1, (size_t)patch * patch * 3 * H)) return false;
        layers.resize(n_layers);
        for (int il = 0; il < n_layers; il++) {
            auto& l = layers[il];
            std::string p = "v.blk." + std::to_string(il) + ".";
            bool ok = true;
            ok &= get(p + "ln1.weight", l.ln1_w, H);
            ok &= get(p + "ln1.bias", l.ln1_b, H);
            ok &= get(p + "ln2.weight", l.ln2_w, H);
            ok &= get(p + "ln2.bias", l.ln2_b, H);
            ok &= get(p + "attn_q.weight", l.q_w, (size_t)H * H);
            ok &= get(p + "attn_q.bias", l.q_b, H);
            ok &= get(p + "attn_k.weight", l.k_w, (size_t)H * H);
            ok &= get(p + "attn_k.bias", l.k_b, H);
            ok &= get(p + "attn_v.weight", l.v_w, (size_t)H * H);
            ok &= get(p + "attn_v.bias", l.v_b, H);
            ok &= get(p + "attn_out.weight", l.o_w, (size_t)H * H);
            ok &= get(p + "attn_out.bias", l.o_b, H);
            // NOTE: names are swapped in this export (legacy clip.cpp bug,
            // auto-corrected there via is_ffn_swapped) — "ffn_down.weight" is
            // really the up-projection [H->ff], "ffn_up.weight" is really the
            // down-projection [ff->H]. Confirmed via tensor shapes: the file's
            // "ffn_up.weight" is [5120,1280] (ne0=5120=ff, i.e. ff->H) and its
            // "ffn_down.weight" is [1280,5120] (ne0=1280=H, i.e. H->ff).
            ok &= get(p + "ffn_down.weight", l.up_w, (size_t)ff * H);
            ok &= get(p + "ffn_down.bias", l.up_b, ff);
            ok &= get(p + "ffn_up.weight", l.down_w, (size_t)H * ff);
            ok &= get(p + "ffn_up.bias", l.down_b, H);
            if (!ok) return false;
        }
        if (!get("v.post_ln.weight", post_ln_w, H)) return false;
        if (!get("v.post_ln.bias", post_ln_b, H)) return false;
        if (!get("mm.0.weight", mm0_w, (size_t)(4 * H) * (4 * H))) return false;
        if (!get("mm.0.bias", mm0_b, 4 * H)) return false;
        if (!get("mm.2.weight", mm2_w, (size_t)(4 * H) * proj_dim)) return false;
        if (!get("mm.2.bias", mm2_b, proj_dim)) return false;
        return true;
    }
};

// Bilinear resize an interleaved RGB uint8 image to (out_w, out_h), producing
// normalized (mean/std) interleaved float RGB.
static std::vector<float> preprocess_image(const uint8_t* src, int sw, int sh,
                                            int out_w, int out_h, const VisionModel& vm) {
    std::vector<float> dst((size_t)out_w * out_h * 3);
    for (int y = 0; y < out_h; y++) {
        float sy = (y + 0.5f) * sh / out_h - 0.5f;
        int y0 = (int)floorf(sy); float fy = sy - y0;
        int y1 = std::min(std::max(y0 + 1, 0), sh - 1); y0 = std::min(std::max(y0, 0), sh - 1);
        for (int x = 0; x < out_w; x++) {
            float sx = (x + 0.5f) * sw / out_w - 0.5f;
            int x0 = (int)floorf(sx); float fx = sx - x0;
            int x1 = std::min(std::max(x0 + 1, 0), sw - 1); x0 = std::min(std::max(x0, 0), sw - 1);
            for (int c = 0; c < 3; c++) {
                float v00 = src[(y0 * sw + x0) * 3 + c];
                float v01 = src[(y0 * sw + x1) * 3 + c];
                float v10 = src[(y1 * sw + x0) * 3 + c];
                float v11 = src[(y1 * sw + x1) * 3 + c];
                float v0 = v00 * (1 - fx) + v01 * fx;
                float v1 = v10 * (1 - fx) + v11 * fx;
                float v = (v0 * (1 - fy) + v1 * fy) / 255.0f;
                v = (v - vm.img_mean[c]) / vm.img_std[c];
                dst[((size_t)y * out_w + x) * 3 + c] = v;
            }
        }
    }
    return dst;
}

// Compute one patch's embedding (both temporal-frame kernels summed, applied
// to the same still image) — direct port of ggml_conv_2d's index convention:
// kernel flat index = kw + 14*kh + 196*cin + 588*cout.
static void patch_embed(const VisionModel& vm, const float* img, int W, int H,
                         int prow, int pcol, float* out /*[hidden]*/) {
    int P = vm.patch, Hd = vm.hidden;
    std::fill(out, out + Hd, 0.0f);
    for (int cin = 0; cin < 3; cin++) {
        for (int kh = 0; kh < P; kh++) {
            int py = prow * P + kh;
            for (int kw = 0; kw < P; kw++) {
                int px = pcol * P + kw;
                float pix = img[((size_t)py * W + px) * 3 + cin];
                size_t kbase = (size_t)kw + (size_t)P * kh + (size_t)P * P * cin;
                const float* k0 = &vm.patch_embd0[kbase];
                const float* k1 = &vm.patch_embd1[kbase];
                for (int o = 0; o < Hd; o++)
                    out[o] += pix * (k0[(size_t)o * P * P * 3] + k1[(size_t)o * P * P * 3]);
            }
        }
    }
}

// 2D (M-RoPE vision-mode) rotation applied in-place to one head's vector x
// (size head_dim). row/col are the patch's grid position. Derived from
// ggml_mrope_cache_init + rotate_pairs with GGML_ROPE_TYPE_VISION in
// ggml-cpu/ops.cpp: first head_dim/4 frequency bands keyed to row, next
// head_dim/4 keyed to col, "rotate-half" pairing (i, i+head_dim/2).
static void rope2d_apply(float* x, int head_dim, int row, int col, float freq_base) {
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

// Full ViT forward: returns merged embeddings, n_merged x proj_dim (1536).
static std::vector<float> vit_forward(const VisionModel& vm, const float* img, int W, int H) {
    int P = vm.patch;
    int pw = W / P, ph = H / P;
    int n_patches = pw * ph;
    int Hd = vm.hidden, NH = vm.n_heads, HD = Hd / NH;

    // Sequence order: consecutive 2x2 spatial blocks (matches merger's
    // "group every 4 tokens" reshape and the position array used for RoPE).
    std::vector<float> seq((size_t)n_patches * Hd);
    std::vector<int> row_of(n_patches), col_of(n_patches);
    int ptr = 0;
    for (int by = 0; by < ph; by += 2) {
        for (int bx = 0; bx < pw; bx += 2) {
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    int row = by + dy, col = bx + dx;
                    patch_embed(vm, img, W, H, row, col, &seq[(size_t)ptr * Hd]);
                    row_of[ptr] = row; col_of[ptr] = col;
                    ptr++;
                }
            }
        }
    }

    std::vector<float> x2(Hd), q(Hd), k(Hd), v(Hd), att(Hd), scores(n_patches);
    std::vector<float> up(vm.ff), down(Hd);
    std::vector<float> Qall((size_t)n_patches * Hd), Kall((size_t)n_patches * Hd), Vall((size_t)n_patches * Hd);

    for (int il = 0; il < vm.n_layers; il++) {
        auto& l = vm.layers[il];

        // Project Q/K/V for every token first (needed since attention is
        // full/bidirectional over the whole sequence, unlike causal decode).
        for (int t = 0; t < n_patches; t++) {
            float* xt = &seq[(size_t)t * Hd];
            layernorm(x2.data(), xt, l.ln1_w.data(), l.ln1_b.data(), Hd, vm.eps);
            matmul(&Qall[(size_t)t * Hd], x2.data(), l.q_w.data(), Hd, Hd);
            matmul(&Kall[(size_t)t * Hd], x2.data(), l.k_w.data(), Hd, Hd);
            matmul(&Vall[(size_t)t * Hd], x2.data(), l.v_w.data(), Hd, Hd);
            for (int i = 0; i < Hd; i++) { Qall[(size_t)t*Hd+i] += l.q_b[i]; Kall[(size_t)t*Hd+i] += l.k_b[i]; Vall[(size_t)t*Hd+i] += l.v_b[i]; }
            for (int h = 0; h < NH; h++) {
                rope2d_apply(&Qall[(size_t)t * Hd + (size_t)h * HD], HD, row_of[t], col_of[t], 10000.0f);
                rope2d_apply(&Kall[(size_t)t * Hd + (size_t)h * HD], HD, row_of[t], col_of[t], 10000.0f);
            }
        }

        float scale = 1.0f / sqrtf((float)HD);
        for (int t = 0; t < n_patches; t++) {
            std::fill(att.begin(), att.end(), 0.0f);
            for (int h = 0; h < NH; h++) {
                float* Q = &Qall[(size_t)t * Hd + (size_t)h * HD];
                for (int s = 0; s < n_patches; s++) {
                    float* K = &Kall[(size_t)s * Hd + (size_t)h * HD];
                    float acc = 0; for (int d = 0; d < HD; d++) acc += Q[d] * K[d];
                    scores[s] = acc * scale;
                }
                float mx = scores[0]; for (int s = 1; s < n_patches; s++) mx = std::max(mx, scores[s]);
                float sum = 0; for (int s = 0; s < n_patches; s++) { scores[s] = expf(scores[s] - mx); sum += scores[s]; }
                float inv = 1.0f / sum;
                for (int d = 0; d < HD; d++) {
                    float acc = 0;
                    for (int s = 0; s < n_patches; s++) acc += scores[s] * inv * Vall[(size_t)s * Hd + (size_t)h * HD + d];
                    att[h * HD + d] = acc;
                }
            }
            matmul(x2.data(), att.data(), l.o_w.data(), Hd, Hd);
            for (int i = 0; i < Hd; i++) x2[i] += l.o_b[i];
            float* xt = &seq[(size_t)t * Hd];
            for (int i = 0; i < Hd; i++) xt[i] += x2[i];
        }

        for (int t = 0; t < n_patches; t++) {
            float* xt = &seq[(size_t)t * Hd];
            layernorm(x2.data(), xt, l.ln2_w.data(), l.ln2_b.data(), Hd, vm.eps);
            matmul(up.data(), x2.data(), l.up_w.data(), vm.ff, Hd);
            for (int i = 0; i < vm.ff; i++) up[i] += l.up_b[i];
            gelu(up.data(), up.data(), vm.ff);
            matmul(down.data(), up.data(), l.down_w.data(), Hd, vm.ff);
            for (int i = 0; i < Hd; i++) xt[i] += down[i] + l.down_b[i];
        }
    }

    for (int t = 0; t < n_patches; t++) {
        float* xt = &seq[(size_t)t * Hd];
        layernorm(x2.data(), xt, vm.post_ln_w.data(), vm.post_ln_b.data(), Hd, vm.eps);
        std::copy(x2.begin(), x2.end(), xt);
    }

    // Merger: every 4 consecutive tokens (one 2x2 block) -> mm.0 -> GELU -> mm.2
    int n_merged = n_patches / 4;
    std::vector<float> merged((size_t)n_merged * vm.proj_dim);
    std::vector<float> cat(4 * Hd), hid(4 * Hd);
    for (int m = 0; m < n_merged; m++) {
        std::copy(&seq[(size_t)m * 4 * Hd], &seq[(size_t)m * 4 * Hd] + 4 * Hd, cat.begin());
        matmul(hid.data(), cat.data(), vm.mm0_w.data(), 4 * Hd, 4 * Hd);
        for (int i = 0; i < 4 * Hd; i++) hid[i] += vm.mm0_b[i];
        gelu(hid.data(), hid.data(), 4 * Hd);
        matmul(&merged[(size_t)m * vm.proj_dim], hid.data(), vm.mm2_w.data(), vm.proj_dim, 4 * Hd);
        for (int i = 0; i < vm.proj_dim; i++) merged[(size_t)m * vm.proj_dim + i] += vm.mm2_b[i];
    }
    return merged;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <text.gguf> <mmproj.gguf> <image.png/jpg> [prompt text]\n", argv[0]);
        return 1;
    }
    std::string text_path = argv[1], mmproj_path = argv[2], image_path = argv[3];
    std::string prompt = argc > 4 ? argv[4] : "Describe this image in one sentence.";

    VisionModel vm;
    fprintf(stderr, "Loading vision tower from %s ...\n", mmproj_path.c_str());
    if (!vm.load(mmproj_path)) { fprintf(stderr, "FAIL: could not load vision tower\n"); return 1; }
    fprintf(stderr, "Vision tower loaded: %d layers, hidden=%d\n", vm.n_layers, vm.hidden);

    int sw, sh, comp;
    uint8_t* img_u8 = stbi_load(image_path.c_str(), &sw, &sh, &comp, 3);
    if (!img_u8) { fprintf(stderr, "FAIL: could not load image %s\n", image_path.c_str()); return 1; }
    fprintf(stderr, "Image loaded: %dx%d, %d comp\n", sw, sh, comp);

    const int OUT = 224; // 16x16 patches -> 8x8=64 merged tokens
    std::vector<float> img_f = preprocess_image(img_u8, sw, sh, OUT, OUT, vm);
    stbi_image_free(img_u8);

    fprintf(stderr, "Running ViT forward (32 layers, %dx%d patches)...\n", OUT / 14, OUT / 14);
    std::vector<float> vision_emb = vit_forward(vm, img_f.data(), OUT, OUT);
    int n_vis_tokens = (int)(vision_emb.size() / vm.proj_dim);
    fprintf(stderr, "Vision embeddings: %d tokens x %d dims\n", n_vis_tokens, vm.proj_dim);
    {
        double mean = 0, mx = -1e30, mn = 1e30;
        for (float v : vision_emb) { mean += v; mx = std::max((double)v, mx); mn = std::min((double)v, mn); }
        mean /= vision_emb.size();
        fprintf(stderr, "Vision embedding stats: mean=%.4f min=%.4f max=%.4f\n", mean, mn, mx);
    }

    fprintf(stderr, "Loading text model from %s ...\n", text_path.c_str());
    ModelConfig cfg;
    if (!read_gguf_header(text_path, cfg)) { fprintf(stderr, "FAIL: could not read text model header\n"); return 1; }
    cfg.max_seq_len = 256;
    Backend* be = create_generic_backend();
    if (!be->init(cfg, text_path)) { fprintf(stderr, "FAIL: text model load failed\n"); return 1; }
    fprintf(stderr, "Text model loaded: hidden=%d layers=%d\n", cfg.hidden, cfg.n_layers);
    if (vm.proj_dim != cfg.hidden) {
        fprintf(stderr, "FAIL: vision proj_dim (%d) != text hidden (%d)\n", vm.proj_dim, cfg.hidden);
        return 1;
    }

    // Feed vision embeddings through the text decoder first (establishing
    // KV-cache context), then a real tokenized text prompt, then generate.
    std::vector<std::string> vocab;
    read_gguf_string_array(text_path, "tokenizer.ggml.tokens", vocab);
    std::unordered_map<std::string, int> vocab_ix;
    for (size_t i = 0; i < vocab.size(); i++) vocab_ix[vocab[i]] = (int)i;
    fprintf(stderr, "Vocab loaded: %zu tokens\n", vocab.size());

    auto greedy_tokenize = [&](const std::string& text) {
        std::vector<int> ids;
        std::string s;
        s.reserve(text.size() * 2);
        for (char c : text) {
            if (c == ' ') s += "\xC4\xA0"; // U+0120 'Ġ' — GPT2 byte-level BPE space marker
            else s += c;
        }
        size_t pos = 0;
        while (pos < s.size()) {
            size_t best_len = 0; int best_id = -1;
            size_t max_try = std::min((size_t)24, s.size() - pos);
            for (size_t len = max_try; len >= 1; len--) {
                auto it = vocab_ix.find(s.substr(pos, len));
                if (it != vocab_ix.end()) { best_len = len; best_id = it->second; break; }
            }
            if (best_id < 0) { pos++; continue; } // skip unmappable byte
            ids.push_back(best_id);
            pos += best_len;
        }
        return ids;
    };
    auto detok = [&](const std::vector<int>& ids) {
        std::string out;
        for (int id : ids) {
            if (id < 0 || (size_t)id >= vocab.size()) continue;
            std::string tok = vocab[id];
            size_t p;
            while ((p = tok.find("\xC4\xA0")) != std::string::npos) tok.replace(p, 2, " ");
            out += tok;
        }
        return out;
    };

    // Qwen2-VL wraps the image embedding sequence with <|vision_start|> /
    // <|vision_end|> special tokens (151652/151653) in training — feed them
    // as normal token-embedding lookups around the spliced vision embeddings
    // so the model sees the input in the format it was actually trained on.
    const int VISION_START = 151652, VISION_END = 151653;
    bool skip_vision = argc > 5 && std::string(argv[5]) == "--novision";
    if (!skip_vision) {
        be->generate(VISION_START);
        for (int t = 0; t < n_vis_tokens; t++) {
            be->forward_embed(&vision_emb[(size_t)t * vm.proj_dim]);
        }
        be->generate(VISION_END);
        fprintf(stderr, "Fed <|vision_start|> + %d vision tokens + <|vision_end|> through text decoder.\n", n_vis_tokens);
    } else {
        fprintf(stderr, "Skipping vision tokens (--novision A/B check).\n");
    }

    std::vector<int> prompt_ids = greedy_tokenize(prompt);
    fprintf(stderr, "Prompt tokenized to %zu ids: [", prompt_ids.size());
    for (int id : prompt_ids) fprintf(stderr, "%d ", id);
    fprintf(stderr, "]\n");

    int last = prompt_ids.empty() ? 0 : prompt_ids.front();
    for (size_t i = 1; i < prompt_ids.size(); i++) last = be->generate(prompt_ids[i - 1]);
    if (!prompt_ids.empty()) last = be->generate(prompt_ids.back());

    std::vector<int> generated;
    for (int i = 0; i < 30; i++) {
        last = be->generate(last);
        generated.push_back(last);
    }

    printf("=== Generated token ids ===\n");
    for (int id : generated) printf("%d ", id);
    printf("\n=== Detokenized ===\n%s\n", detok(generated).c_str());

    return 0;
}
