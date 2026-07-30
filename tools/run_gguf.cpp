/** run_gguf.cpp — Minimal GGUF inference runner using generic CPU backend
 *  Usage: ./build/run_gguf model.gguf [prompt] [tokens]
 */
#include "gguf_reader.h"
#include <cstdio>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>

// Simple sampler
int sample(const float* logits, int n, float temp) {
    if (temp <= 0) {
        int max_i = 0;
        for (int i = 1; i < n; i++) if (logits[i] > logits[max_i]) max_i = i;
        return max_i;
    }
    std::vector<float> p(n);
    float sum = 0;
    for (int i = 0; i < n; i++) { p[i] = expf(logits[i] / temp); sum += p[i]; }
    float r = (float)rand() / RAND_MAX * sum;
    for (int i = 0; i < n; i++) { r -= p[i]; if (r <= 0) return i; }
    return n - 1;
}

// SiLU
static inline float silu(float x) { return x / (1.0f + expf(-x)); }

struct Model {
    int H, L, NH, NKV, HD, FF, V;
    std::vector<float> tok_embd, out_norm, lm_head;
    struct Layer {
        std::vector<float> attn_norm, q, k, v, o, ffn_norm, gate, up, down;
        // MoE
        std::vector<float> gate_router;
        int n_experts = 0, n_active = 0;
        bool is_moe = false;
    };
    std::vector<Layer> layers;
    GgufReader* reader = nullptr;
    
    bool load(const std::string& path) {
        reader = new GgufReader();
        if (!reader->open(path)) return false;
        
        uint32_t tmp;
        reader->get_u32("block_count", tmp); L = tmp;
        reader->get_u32("embedding_length", tmp); H = tmp;
        reader->get_u32("attention.head_count", tmp); NH = tmp;
        reader->get_u32("attention.head_count_kv", tmp); NKV = tmp ? tmp : NH;
        reader->get_u32("feed_forward_length", tmp); FF = tmp;
        V = reader->vocab_count();
        if (V <= 0) V = 151936;
        HD = H / NH;
        
        printf("  Model: %s | H=%d L=%d NH=%d NKV=%d FF=%d V=%d\n",
               reader->architecture().c_str(), H, L, NH, NKV, FF, V);
        
        auto load_t = [&](const std::string& n) -> std::vector<float> {
            std::vector<float> b; size_t sz = 0;
            reader->get_tensor_f32(n, b, &sz);
            if (b.empty()) { const float* d = reader->get_tensor(n, &sz); if (d) b.assign(d, d+sz); }
            return b;
        };
        
        tok_embd = load_t("token_embd.weight");
        out_norm = load_t("output_norm.weight");
        lm_head = load_t("output.weight");
        printf("  Embed: %zu, Norm: %zu, Head: %zu\n", tok_embd.size(), out_norm.size(), lm_head.size());
        
        layers.resize(L);
        for (int i = 0; i < L; i++) {
            auto& ly = layers[i];
            std::string p = "blk." + std::to_string(i) + ".";
            ly.attn_norm = load_t(p + "attn_norm.weight");
            ly.q = load_t(p + "attn_q.weight");
            ly.k = load_t(p + "attn_k.weight");
            ly.v = load_t(p + "attn_v.weight");
            ly.o = load_t(p + "attn_output.weight");
            ly.ffn_norm = load_t(p + "ffn_norm.weight");
            ly.gate = load_t(p + "ffn_gate.weight");
            ly.up = load_t(p + "ffn_up.weight");
            ly.down = load_t(p + "ffn_down.weight");
            
            // MoE tensors
            ly.gate_router = load_t(p + "ffn_gate_inp.weight");
            if (!ly.gate_router.empty()) {
                ly.is_moe = true;
                ly.n_experts = ly.gate_router.size() / H;
                ly.n_active = 8; // top-8 for Qwen MoE
                printf("    Layer %d: MoE (%d experts)\n", i, ly.n_experts);
            }
        }
        return true;
    }
    
    int forward(const std::vector<int>& tokens, int predict_idx,
                std::vector<std::vector<float>>& k_cache,
                std::vector<std::vector<float>>& v_cache) {
        int N = (int)tokens.size();
        
        for (int pos = 0; pos < N; pos++) {
            int tok = tokens[pos];
            // Embed
            std::vector<float> x(H);
            if ((size_t)tok * H + H <= tok_embd.size())
                memcpy(x.data(), &tok_embd[(size_t)tok * H], H * sizeof(float));
            
            for (int l = 0; l < L && l < (int)layers.size(); l++) {
                auto& ly = layers[l];
                
                // RMS Norm
                auto rmsnorm = [&](std::vector<float>& o, const std::vector<float>& w) {
                    float ss = 0; for (int i = 0; i < H; i++) ss += x[i] * x[i];
                    float rms = sqrtf(ss / H + 1e-6f);
                    for (int i = 0; i < H; i++) o[i] = (w.size() > (size_t)i ? w[i] : 1.0f) * x[i] / rms;
                };
                
                // Self-attention
                std::vector<float> q(H), k(H), v(H), attn_out(H);
                std::vector<float> x_norm(H);
                rmsnorm(x_norm, ly.attn_norm);
                
                // QKV projections
                auto matmul = [](std::vector<float>& out, const std::vector<float>& w, 
                                 const std::vector<float>& in, int d_out, int d_in) {
                    for (int i = 0; i < d_out; i++) {
                        float s = 0;
                        for (int j = 0; j < d_in; j++) s += w[(size_t)i * d_in + j] * in[j];
                        out[i] = s;
                    }
                };
                
                if (ly.q.size() >= (size_t)H * H) matmul(q, ly.q, x_norm, H, H);
                if (ly.k.size() >= (size_t)(NKV * HD) * H) matmul(k, ly.k, x_norm, NKV * HD, H);
                if (ly.v.size() >= (size_t)(NKV * HD) * H) matmul(v, ly.v, x_norm, NKV * HD, H);
                
                // Store in KV cache
                int kv_size = NKV * HD;
                if (pos < (int)k_cache[l].size()) {
                    memcpy(&k_cache[l][(size_t)pos * kv_size], k.data(), kv_size * sizeof(float));
                    memcpy(&v_cache[l][(size_t)pos * kv_size], v.data(), kv_size * sizeof(float));
                }
                
                // Simple attention over cached keys/values
                int seq_len = pos + 1;
                if (seq_len > 1 && seq_len <= (int)k_cache[l].size() / kv_size) {
                    std::vector<float> scores(seq_len);
                    for (int t = 0; t < seq_len; t++) {
                        float s = 0;
                        for (int h = 0; h < NH; h++) {
                            for (int d = 0; d < HD; d++)
                                s += q[(size_t)h * HD + d] * k_cache[l][(size_t)t * kv_size + (h % NKV) * HD + d];
                        }
                        scores[t] = s / sqrtf((float)HD);
                    }
                    // Softmax
                    float max_s = scores[0]; for (int t = 1; t < seq_len; t++) if (scores[t] > max_s) max_s = scores[t];
                    float sum_s = 0; for (int t = 0; t < seq_len; t++) { scores[t] = expf(scores[t] - max_s); sum_s += scores[t]; }
                    for (int t = 0; t < seq_len; t++) scores[t] /= sum_s;
                    
                    // Weighted sum of values
                    for (int h = 0; h < NH; h++) {
                        for (int d = 0; d < HD; d++) {
                            float s = 0;
                            for (int t = 0; t < seq_len; t++)
                                s += scores[t] * v_cache[l][(size_t)t * kv_size + (h % NKV) * HD + d];
                            attn_out[(size_t)h * HD + d] = s;
                        }
                    }
                    
                    // Output projection
                    std::vector<float> attn_proj(H);
                    if (ly.o.size() >= (size_t)H * H) matmul(attn_proj, ly.o, attn_out, H, H);
                    for (int i = 0; i < H; i++) x[i] += attn_proj[i];
                }
                
                // FFN
                std::vector<float> ffn_normed(H);
                rmsnorm(ffn_normed, ly.ffn_norm);
                
                if (ly.is_moe && ly.n_experts > 0) {
                    // Simplified MoE: just use first expert
                    size_t expert_size = (size_t)FF * H;
                    std::vector<float> gate_out(FF), up_out(FF), down_out(H);
                    
                    if (ly.gate.size() >= expert_size)
                        matmul(gate_out, ly.gate, ffn_normed, FF, H);
                    if (ly.up.size() >= expert_size)
                        matmul(up_out, ly.up, ffn_normed, FF, H);
                    
                    for (int i = 0; i < FF; i++) gate_out[i] = silu(gate_out[i]);
                    for (int i = 0; i < FF; i++) up_out[i] *= gate_out[i];
                    
                    if (ly.down.size() >= (size_t)H * FF)
                        matmul(down_out, ly.down, up_out, H, FF);
                    for (int i = 0; i < H; i++) x[i] += down_out[i];
                } else {
                    // Dense FFN
                    std::vector<float> gate_out(FF), up_out(FF), down_out(H);
                    if (ly.gate.size() >= expert_size(FF, H))
                        matmul(gate_out, ly.gate, ffn_normed, FF, H);
                    if (ly.up.size() >= expert_size(FF, H))
                        matmul(up_out, ly.up, ffn_normed, FF, H);
                    
                    for (int i = 0; i < FF; i++) gate_out[i] = silu(gate_out[i]);
                    for (int i = 0; i < FF; i++) up_out[i] *= gate_out[i];
                    
                    if (ly.down.size() >= (size_t)H * FF)
                        matmul(down_out, ly.down, up_out, H, FF);
                    for (int i = 0; i < H; i++) x[i] += down_out[i];
                }
            }
            
            // Final norm
            float ss = 0; for (int i = 0; i < H; i++) ss += x[i] * x[i];
            float rms = sqrtf(ss / H + 1e-6f);
            for (int i = 0; i < H; i++) x[i] = (out_norm.size() > (size_t)i ? out_norm[i] : 1.0f) * x[i] / rms;
            
            // LM head (or tied embedding)
            if (pos == predict_idx || N == 1) {
                std::vector<float> logits(V);
                if (lm_head.size() >= (size_t)V * H) {
                    for (int i = 0; i < V; i++) {
                        float s = 0;
                        for (int j = 0; j < H; j++) s += lm_head[(size_t)i * H + j] * x[j];
                        logits[i] = s;
                    }
                } else {
                    // Tied embeddings
                    for (int i = 0; i < V; i++) {
                        float s = 0;
                        for (int j = 0; j < H; j++) s += tok_embd[(size_t)i * H + j] * x[j];
                        logits[i] = s;
                    }
                }
                
                int next = sample(logits.data(), V, 0.0f);
                return next;
            }
        }
        return -1;
    }
    
private:
    size_t expert_size(int ff, int h) { return (size_t)ff * h; }
};

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf [tokens=10]\n", argv[0]); return 1; }
    
    srand(time(0));
    Model model;
    if (!model.load(argv[1])) { fprintf(stderr, "Load failed\n"); return 1; }
    
    int n_tokens = argc > 2 ? atoi(argv[2]) : 10;
    
    // KV cache
    int max_seq = 2048;
    int kv_size = model.NKV * model.HD;
    std::vector<std::vector<float>> k_cache(model.L, std::vector<float>((size_t)max_seq * kv_size));
    std::vector<std::vector<float>> v_cache(model.L, std::vector<float>((size_t)max_seq * kv_size));
    
    // Start with BOS
    std::vector<int> tokens = {151643}; // Qwen BOS
    
    printf("\nGenerating %d tokens...\n\n", n_tokens);
    auto t0 = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < n_tokens; i++) {
        int next = model.forward(tokens, (int)tokens.size() - 1, k_cache, v_cache);
        if (next < 0 || next >= model.V) break;
        tokens.push_back(next);
        printf("%d ", next);
        fflush(stdout);
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    printf("\n\nDone: %d tokens in %.0f ms (%.1f ms/tok, %.1f tok/s)\n",
           n_tokens, ms, ms/n_tokens, n_tokens*1000.0f/ms);
    
    return 0;
}
