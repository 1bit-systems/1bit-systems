// mage_vl_server.cpp — Full Mage-VL inference server
// Vision encoder + Qwen3-4B text decoder from a single .1bp file.
//
// Build:
//   g++ -std=c++17 -O2 -I include -I src -I third_party/stb \
//       tools/mage_vl_server.cpp src/vision_encoder.cpp src/vl_processor.cpp \
//       src/onebp_model.cpp src/gguf_reader.cpp src/backend_cpu.cpp \
//       -o build/mage_vl_server -lpthread
//
// Run: (no pip, no Python, pure C++)
//   ./build/mage_vl_server models/Mage-VL-4B.1bp

#include "vision_encoder.h"
#include "vl_processor.h"
#include "onebp_loader.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <unordered_map>

// ─── Constants ────────────────────────────────────────────────────
static constexpr int IMG_SIZE = 448;
static constexpr int VISION_START_ID = 151652;
static constexpr int VISION_END_ID = 151653;
static constexpr int EOS_ID = 151645;
static constexpr int BOS_ID = 151643;
static constexpr int IMAGE_PAD_ID = 151655;
static constexpr int IMAGE_TOKEN_COUNT = 196;  // 14x14 merged patches
static constexpr int TEXT_HIDDEN = 2560;

static const float MAGE_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float MAGE_STD[3]  = {0.229f, 0.224f, 0.225f};

// ═══════════════════════════════════════════════════════════════════
// 1BP Weight access helpers
// ═══════════════════════════════════════════════════════════════════

struct TextWeights {
    std::vector<float> embed;       // [vocab, hidden]
    std::vector<float> final_norm;  // [hidden]
    std::vector<float> output_w;    // [vocab, hidden] or empty for tied
    
    struct Layer {
        std::vector<float> rms_attn;
        std::vector<float> rms_ffn;
        std::vector<float> attn_q_w, attn_k_w, attn_v_w, attn_o_w;
        std::vector<float> q_norm, k_norm;
        std::vector<float> ffn_gate_w, ffn_up_w, ffn_down_w;
    };
    std::vector<Layer> layers;
    int hidden, n_layers, n_heads, n_kv, head_dim, ff_dim, vocab;
};

// Load text decoder weights from 1BP
static bool load_text_weights(const char* path, TextWeights& tw, OnebpModel& mdl) {
    if (!mdl.load(path)) { fprintf(stderr, "FAIL: load %s\n", path); return false; }
    auto& h = mdl.header;
    tw.hidden = h.hidden_size;
    tw.n_layers = h.num_layers;
    tw.n_heads = h.num_attention_heads;
    tw.n_kv = h.num_kv_heads;
    tw.head_dim = h.head_dim;
    tw.ff_dim = h.intermediate_size;
    tw.vocab = h.vocab_size;
    
    auto find = [&](const std::string& name, std::vector<float>& dst) -> bool {
        for (auto& t : mdl.tensors) {
            if (t.name.find(name) == std::string::npos &&
                t.name.find("model." + name) == std::string::npos) continue;
            if (t.ndim != 2 && t.ndim != 1) continue;
            uint8_t* raw = mdl.tensor_data(t);
            if (!raw) return false;
            size_t n = 1; for (int d = 0; d < t.ndim; d++) n *= t.dims[d];
            
            if (t.ndim == 1 && t.bytes == n * 2) {
                dst.resize(n);
                const uint16_t* f16 = (const uint16_t*)raw;
                for (size_t i = 0; i < n; i++) {
                    uint32_t bits = (uint32_t)f16[i] << 16;
                    memcpy(&dst[i], &bits, 4);
                }
                return true;
            }
            
            // Q4NX tile dequant for 2D weights
            if (t.ndim == 2) {
                int rows = (int)t.dims[0], cols = (int)t.dims[1];
                int tr=32, tc=256, gs=32;
                int ntr=(rows+tr-1)/tr, ntc=(cols+tc-1)/tc;
                int ng=tc/gs, rb=ng*4+tc/2, tb=tr*rb;
                dst.resize((size_t)rows*cols);
                size_t doff=0;
                for (int ti=0; ti<ntr; ti++) for (int tj=0; tj<ntc; tj++) {
                    int r0=ti*tr, c0=tj*tc, rh=std::min(tr,rows-r0), ch=std::min(tc,cols-c0);
                    std::vector<float> buf((size_t)tr*tc);
                    for (int r=0; r<tr; r++) {
                        const uint8_t* rd = raw + doff + r*rb;
                        for (int g=0; g<ng; g++) {
                            auto b2f=[](uint16_t b){uint32_t x=(uint32_t)b<<16;float f;memcpy(&f,&x,4);return f;};
                            float s=b2f(*(const uint16_t*)(rd+g*2));
                            float zp=b2f(*(const uint16_t*)(rd+ng*2+g*2));
                            if (!std::isfinite(s)) s=0; if (!std::isfinite(zp)) zp=0;
                            const uint8_t* pk=rd+ng*4;
                            for (int i=0; i<gs && g*gs+i<tc; i++) {
                                int bi=g*gs/2+i/2, nib=(i&1)?(pk[bi]>>4):(pk[bi]&0x0F);
                                buf[(size_t)r*tc+g*gs+i]=(float)nib*s+zp;
                            }
                        }
                    }
                    for (int r=0; r<rh; r++) for (int c=0; c<ch; c++)
                        dst[(size_t)(r0+r)*cols+(c0+c)] = buf[(size_t)r*tc+c];
                    doff += tb;
                }
                return true;
            }
        }
        return false;
    };
    
    // Load embed
    find("token_embd.weight", tw.embed);
    find("output_norm.weight", tw.final_norm);
    find("output.weight", tw.output_w);
    if (tw.output_w.empty()) tw.output_w = tw.embed;  // tied
    
    // Load layers
    tw.layers.resize(tw.n_layers);
    for (int i = 0; i < tw.n_layers; i++) {
        auto& l = tw.layers[i];
        std::string p = "blk." + std::to_string(i) + ".";
        find(p + "rms_attn_w", l.rms_attn);
        find(p + "rms_ffn_w", l.rms_ffn);
        find(p + "attn_q.weight", l.attn_q_w);
        find(p + "attn_k.weight", l.attn_k_w);
        find(p + "attn_v.weight", l.attn_v_w);
        find(p + "attn_o.weight", l.attn_o_w);
        find(p + "ffn_gate.weight", l.ffn_gate_w);
        find(p + "ffn_up.weight", l.ffn_up_w);
        find(p + "ffn_down.weight", l.ffn_down_w);
        find(p + "attn_q_norm.weight", l.q_norm);
        if (l.q_norm.empty()) find(p + "attn_q_norm", l.q_norm);
        find(p + "attn_k_norm.weight", l.k_norm);
        if (l.k_norm.empty()) find(p + "attn_k_norm", l.k_norm);
    }
    
    fprintf(stderr, "[text] loaded %d layers, H=%d NH=%d NKV=%d V=%d\n",
            tw.n_layers, tw.hidden, tw.n_heads, tw.n_kv, tw.vocab);
    fprintf(stderr, "[text] embed=%zu final_norm=%zu layers=%zu\n",
            tw.embed.size(), tw.final_norm.size(), tw.layers.size());
    return true;
}

// ═══════════════════════════════════════════════════════════════════
// Math helpers
// ═══════════════════════════════════════════════════════════════════

static float silu(float x) { return x / (1.0f + expf(-x)); }

static void rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

static void matmul(float* out, const float* x, const float* w, int M, int K) {
    for (int i = 0; i < M; i++) {
        float s = 0; for (int j = 0; j < K; j++) s += x[j] * w[(size_t)i * K + j];
        out[i] = s;
    }
}

static void rope_qwen3(float* q, float* k, int pos, int head_dim, float theta) {
    for (int i = 0; i < head_dim/2; i++) {
        float freq = pos / powf(theta, 2.0f * i / head_dim);
        float c = cosf(freq), s = sinf(freq);
        float q0 = q[i], q1 = q[i + head_dim/2];
        q[i] = q0*c - q1*s; q[i + head_dim/2] = q0*s + q1*c;
        float k0 = k[i], k1 = k[i + head_dim/2];
        k[i] = k0*c - k1*s; k[i + head_dim/2] = k0*s + k1*c;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Qwen3 text decoder forward
// ═══════════════════════════════════════════════════════════════════

struct Qwen3Decoder {
    TextWeights& w;
    int H, NH, NKV, HD, FF;
    std::vector<float> k_cache, v_cache;
    int seq_len = 0;
    
    Qwen3Decoder(TextWeights& tw) : w(tw), H(tw.hidden), NH(tw.n_heads), 
        NKV(tw.n_kv), HD(tw.head_dim), FF(tw.ff_dim) {
        int max_len = 2048;  // reasonable for demo
        k_cache.resize((size_t)w.n_layers * max_len * NKV * HD, 0.0f);
        v_cache.resize((size_t)w.n_layers * max_len * NKV * HD, 0.0f);
    }
    
    void reset() { seq_len = 0; }
    
    // Forward a single token with given embedding
    void forward(const float* emb, float* logits) {
        std::vector<float> x(H), x2(H), q(H), k(H), v(H), att(NKV*HD), score(262144);
        std::vector<float> up(FF), gate(FF);
        
        memcpy(x.data(), emb, H * sizeof(float));
        
        for (int il = 0; il < w.n_layers; il++) {
            auto& l = w.layers[il];
            
            // RMSNorm + QKV
            rmsnorm(x2.data(), x.data(), l.rms_attn.data(), H, 1e-6f);
            matmul(q.data(), x2.data(), l.attn_q_w.data(), H, H);
            matmul(k.data(), x2.data(), l.attn_k_w.data(), H, H);
            matmul(v.data(), x2.data(), l.attn_v_w.data(), H, H);
            
            // QK norm
            int hd = HD;
            if (!l.q_norm.empty()) {
                for (int h = 0; h < NH; h++)
                    rmsnorm(q.data() + (size_t)h*hd, q.data() + (size_t)h*hd, l.q_norm.data(), hd, 1e-6f);
                for (int h = 0; h < NKV; h++)
                    rmsnorm(k.data() + (size_t)h*hd, k.data() + (size_t)h*hd, l.k_norm.data(), hd, 1e-6f);
            }
            
            // RoPE
            rope_qwen3(q.data(), k.data(), seq_len, hd, 5000000.0f);
            
            // Store in KV cache
            size_t kv_off = (size_t)il * 262144 * NKV * HD + (size_t)seq_len * NKV * HD;
            memcpy(&k_cache[kv_off], k.data(), NKV * HD * sizeof(float));
            memcpy(&v_cache[kv_off], v.data(), NKV * HD * sizeof(float));
            
            // Attention
            float scale = 1.0f / sqrtf((float)hd);
            std::fill(att.begin(), att.end(), 0.0f);
            int n_kv = NKV;
            for (int h = 0; h < NH; h++) {
                int kv_h = h % n_kv;
                float* Qh = q.data() + (size_t)h * hd;
                
                for (int s = 0; s <= seq_len; s++) {
                    float* Kh = &k_cache[(size_t)il * 262144 * NKV * HD + (size_t)s * NKV * HD + (size_t)kv_h * hd];
                    float acc = 0;
                    for (int d = 0; d < hd; d++) acc += Qh[d] * Kh[d];
                    score[s] = acc * scale;
                }
                
                // Softmax
                float mx = score[0]; for (int s = 1; s <= seq_len; s++) mx = std::max(mx, score[s]);
                float sum = 0; for (int s = 0; s <= seq_len; s++) { score[s] = expf(score[s] - mx); sum += score[s]; }
                float inv = 1.0f / sum;
                
                // Weighted sum of V
                for (int d = 0; d < hd; d++) {
                    float acc = 0;
                    for (int s = 0; s <= seq_len; s++) {
                        float* Vh = &v_cache[(size_t)il * 262144 * NKV * HD + (size_t)s * NKV * HD + (size_t)kv_h * hd];
                        acc += score[s] * Vh[d];
                    }
                    att[(size_t)h * hd + d] = acc;
                }
            }
            
            // Output projection + residual
            std::vector<float> att_out(H);
            matmul(att_out.data(), att.data(), l.attn_o_w.data(), H, NH * hd);
            for (int j = 0; j < H; j++) x[j] += att_out[j];
            
            // FFN: gate + up -> SiLU(gate)*up -> down
            rmsnorm(x2.data(), x.data(), l.rms_ffn.data(), H, 1e-6f);
            matmul(gate.data(), x2.data(), l.ffn_gate_w.data(), FF, H);
            matmul(up.data(), x2.data(), l.ffn_up_w.data(), FF, H);
            for (int j = 0; j < FF; j++) up[j] *= silu(gate[j]);
            
            std::vector<float> down(H);
            matmul(down.data(), up.data(), l.ffn_down_w.data(), H, FF);
            for (int j = 0; j < H; j++) x[j] += down[j];
        }
        
        // Final norm + lm_head
        rmsnorm(x2.data(), x.data(), w.final_norm.data(), H, 1e-6f);
        matmul(logits, x2.data(), w.output_w.data(), w.vocab, H);
        seq_len++;
    }
    
    // Generate: find argmax of logits
    int argmax(float* logits) {
        int best = 0; float best_v = logits[0];
        for (int i = 1; i < w.vocab; i++) if (logits[i] > best_v) { best_v = logits[i]; best = i; }
        return best;
    }
};

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.1bp> <image.jpg> [prompt]\n", argv[0]);
        return 1;
    }
    
    printf("╔═══════════════════════════════════════════════════╗\n");
    printf("║  Mage-VL Server — Full Image→Text Pipeline      ║\n");
    printf("╚═══════════════════════════════════════════════════╝\n\n");
    
    // Load weights
    OnebpModel mdl;
    VisionWeights vw;
    TextWeights tw;
    
    printf("[1/3] Loading vision encoder...\n");
    if (!mage_vit_load_weights_1bp(argv[1], vw)) return 1;
    
    printf("[2/3] Loading text decoder...\n");
    if (!load_text_weights(argv[1], tw, mdl)) return 1;
    
    printf("[3/3] Creating decoder...\n");
    Qwen3Decoder dec(tw);
    printf("[3/3] Loading image...\n");
    VlProcessor proc;
    if (!proc.load(argv[2], IMG_SIZE, IMG_SIZE, MAGE_MEAN, MAGE_STD)) return 1;
    
    // ── Run vision encoder ──
    printf("\nRunning vision encoder...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    auto vision_embeds = mage_vit_forward(vw, proc.pixels(), 3, 1, IMG_SIZE, IMG_SIZE, 4);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Vision encoder: %.1f ms (196 patches x 2560-dim)\n", ms);
    
    // ── Text generation ──
    printf("\nGenerating text...\n");
    dec.reset();
    
    // Build prompt tokens: BOS + vision tokens + text
    std::vector<int> prompt_tokens = {BOS_ID, VISION_START_ID};
    for (int i = 0; i < IMAGE_TOKEN_COUNT; i++) prompt_tokens.push_back(IMAGE_PAD_ID);
    prompt_tokens.push_back(VISION_END_ID);
    
    // Add instruction text (simple tokenization for demo)
    const char* default_prompt = "Describe this image in a few words.";
    const char* user_prompt = argc > 3 ? argv[3] : default_prompt;
    
    // Basic ASCII tokenization (split on space, lookup)
    std::string p = user_prompt;
    // For proper results, use full HF tokenizer. Here we approximate with BPE-free.
    // Add raw text tokens
    for (char c : p) prompt_tokens.push_back((int)(unsigned char)c + 3);  // crude ascii mapping
    
    prompt_tokens.push_back(EOS_ID);
    
    // Prefill: feed all prompt tokens, replace IMAGE_PAD with vision embeddings
    std::vector<float> logits(tw.vocab);
    int vision_idx = 0;
    
    for (int pos = 0; pos < (int)prompt_tokens.size(); pos++) {
        int tok = prompt_tokens[pos];
        
        if (tok == IMAGE_PAD_ID && vision_idx < (int)(vision_embeds.size() / TEXT_HIDDEN)) {
            // Use vision embedding instead of token embedding
            dec.forward(&vision_embeds[(size_t)vision_idx * TEXT_HIDDEN], logits.data());
            vision_idx++;
        } else {
            // Use token embedding
            if (tok >= 0 && tok < tw.vocab && !tw.embed.empty()) {
                dec.forward(&tw.embed[(size_t)tok * TEXT_HIDDEN], logits.data());
            } else {
                std::vector<float> zero_emb(TEXT_HIDDEN, 0.0f);
                dec.forward(zero_emb.data(), logits.data());
            }
        }
    }
    
    // Autoregressive generation  
    std::vector<int> output_tokens;
    int next_tok = dec.argmax(logits.data());
    int max_new = 64;
    
    auto t2 = std::chrono::high_resolution_clock::now();
    int gen_count = 0;
    
    while (next_tok != EOS_ID && gen_count < max_new) {
        output_tokens.push_back(next_tok);
        dec.forward(&tw.embed[(size_t)next_tok * TEXT_HIDDEN], logits.data());
        next_tok = dec.argmax(logits.data());
        gen_count++;
    }
    
    auto t3 = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    
    printf("\n=== Generated %d tokens in %.0f ms (%.1f tok/s) ===\n", gen_count, gen_ms, gen_count/(gen_ms/1000.0));
    printf("Token IDs: ");
    for (int t : output_tokens) printf("%d ", t);
    printf("\n");
    
    return 0;
}
