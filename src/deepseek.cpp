// deepseek.cpp — DeepSeek model family implementation with MLA
#include "deepseek.h"
#include "gguf_reader.h"

bool DeepSeekModel::load_from_gguf(const std::string& path, const DeepSeekConfig* override_cfg) {
    GgufReader r;
    if (!r.open(path)) {
        fprintf(stderr, "[deepseek] FAIL: could not open %s\n", path.c_str());
        return false;
    }
    
    auto gu32 = [&](const std::string& key, int def) -> int {
        uint32_t v;
        if (r.get_u32(key, v)) return (int)v;
        std::string arch = r.architecture();
        if (!arch.empty() && r.get_u32(arch + "." + key, v)) return (int)v;
        return def;
    };
    
    if (override_cfg) {
        cfg = *override_cfg;
    } else {
        cfg.hidden_size       = gu32("embedding_length", 2048);
        cfg.num_layers        = gu32("block_count", 24);
        cfg.num_heads         = gu32("attention.head_count", 16);
        cfg.num_kv_heads      = gu32("attention.head_count_kv", 16);
        cfg.head_dim          = gu32("attention.key_length", 128);
        cfg.vocab_size        = gu32("vocab_size", 102400);
        cfg.max_seq_len       = gu32("context_length", 4096);
        cfg.qk_nope_dim       = gu32("attention.qk_nope_head_dim", 128);
        cfg.qk_rope_dim       = gu32("attention.qk_rope_head_dim", 64);
        cfg.v_dim             = gu32("attention.v_head_dim", 128);
        cfg.kv_lora_rank      = gu32("attention.kv_lora_rank", 512);
        cfg.q_lora_rank       = gu32("attention.q_lora_rank", 1536);
        cfg.n_routed_experts  = gu32("feed_forward.moe.layer.moe.expert_count", 64);
        cfg.n_shared_experts  = gu32("feed_forward.moe.layer.moe.shared_expert_count", 1);
        cfg.top_k             = gu32("feed_forward.moe.layer.moe.routed_scaling_factor", 6);
        cfg.moe_intermediate  = gu32("feed_forward.moe.layer.moe.intermediate_size", 1408);
        // Derive qk_compressed = kv_lora_rank
        cfg.qk_compressed = cfg.kv_lora_rank;
        cfg.q_compressed = cfg.q_lora_rank;
    }
    
    auto get = [&](const std::string& name, std::vector<float>& dst, size_t expect) -> bool {
        size_t n = 0;
        if (!r.get_tensor_f32(name, dst, &n)) {
            fprintf(stderr, "  [deepseek] missing: %s\n", name.c_str());
            return false;
        }
        if (expect > 0 && n != expect) {
            fprintf(stderr, "  [deepseek] %s: expected %zu, got %zu\n", name.c_str(), expect, n);
            return false;
        }
        return true;
    };
    
    int H = cfg.hidden_size;
    
    // Embeddings
    if (!get("token_embd.weight", token_emb, (size_t)cfg.vocab_size * H)) return false;
    if (!get("output_norm.weight", final_norm_w, (size_t)H)) {
        get("final_norm.weight", final_norm_w, (size_t)H);
    }
    if (final_norm_w.empty()) final_norm_w.resize(H, 1.0f);
    
    // Output weights (may be tied with embeddings)
    get("output.weight", output_w, (size_t)cfg.vocab_size * H);
    
    // Layers
    layers.resize(cfg.num_layers);
    for (int il = 0; il < cfg.num_layers; il++) {
        auto& l = layers[il];
        std::string p = "blk." + std::to_string(il) + ".";
        bool ok = true;
        
        // RMSNorm
        ok &= get(p + "attn_norm.weight", l.rms_attn_w, (size_t)H);
        ok &= get(p + "ffn_norm.weight",  l.rms_ffn_w,  (size_t)H);
        
        // MLA projections
        ok &= get(p + "attn_q_a.weight", l.w_q_a, (size_t)H * cfg.q_lora_rank);
        get(p + "attn_q_a.bias", l.w_q_a_bias, (size_t)cfg.q_lora_rank);
        if (l.w_q_a_bias.empty()) l.w_q_a_bias.resize(cfg.q_lora_rank, 0.0f);
        
        ok &= get(p + "attn_kv_a.weight", l.w_kv_a, (size_t)H * (cfg.kv_lora_rank + cfg.qk_rope_dim));
        get(p + "attn_kv_a.bias", l.w_kv_a_bias, (size_t)(cfg.kv_lora_rank + cfg.qk_rope_dim));
        if (l.w_kv_a_bias.empty()) l.w_kv_a_bias.resize(cfg.kv_lora_rank + cfg.qk_rope_dim, 0.0f);
        
        ok &= get(p + "attn_kv_b.weight", l.w_kv_b, 
                  (size_t)cfg.kv_lora_rank * cfg.num_heads * (cfg.qk_nope_dim + cfg.v_dim));
        
        ok &= get(p + "attn_q_b.weight", l.w_q_b,
                  (size_t)cfg.q_lora_rank * cfg.num_heads * cfg.qk_nope_dim);
        
        // Output projection
        ok &= get(p + "attn_o.weight", l.w_o, (size_t)cfg.num_heads * cfg.v_dim * H);
        
        // MoE FFN
        ok &= get(p + "moe.gate.weight", l.w_gate, (size_t)H * cfg.n_routed_experts);
        
        // Shared expert
        ok &= get(p + "shared_expert.gate.weight", l.w_shared_gate, (size_t)H * cfg.moe_intermediate);
        ok &= get(p + "shared_expert.up.weight",   l.w_shared_up,   (size_t)H * cfg.moe_intermediate);
        ok &= get(p + "shared_expert.down.weight", l.w_shared_down, (size_t)cfg.moe_intermediate * H);
        
        // Routed experts (flat: all experts concatenated)
        size_t exp_size = (size_t)cfg.n_routed_experts * H * cfg.moe_intermediate;
        ok &= get(p + "experts.gate.weight", l.exp_gate, exp_size);
        ok &= get(p + "experts.up.weight",   l.exp_up,   exp_size);
        // Down projection is [moe_intermediate, hidden] per expert
        size_t down_size = (size_t)cfg.n_routed_experts * cfg.moe_intermediate * H;
        ok &= get(p + "experts.down.weight", l.exp_down, down_size);
        if (l.exp_down.size() != down_size) l.exp_down.resize(down_size, 0.0f);
        
        if (!ok) {
            fprintf(stderr, "  [deepseek] layer %d incomplete\n", il);
            return false;
        }
    }
    
    fprintf(stderr, "[deepseek] loaded: %s (%d layers, H=%d, MLA kv_lora=%d, experts=%d, top_k=%d)\n",
            r.architecture().c_str(), cfg.num_layers, H, cfg.kv_lora_rank, cfg.n_routed_experts, cfg.top_k);
    return true;
}

void DeepSeekModel::clear() {
    token_emb.clear(); final_norm_w.clear(); output_w.clear();
    layers.clear();
}

std::vector<float> deepseek_forward(
    const DeepSeekModel& model, int token_id,
    std::vector<std::vector<float>>& mla_kv_cache,
    int& pos)
{
    using namespace deepseek_math;
    const auto& cfg = model.cfg;
    int H = cfg.hidden_size;
    
    // Token embedding
    std::vector<float> x(H);
    if (token_id >= 0 && token_id < cfg.vocab_size) {
        for (int i = 0; i < H; i++)
            x[i] = model.token_emb[(size_t)token_id * H + i];
    }
    
    // Allocate per-layer buffers
    std::vector<float> norm(H), q_rope(cfg.qk_rope_dim);
    std::vector<float> q_comp(cfg.q_lora_rank), k_comp(cfg.kv_lora_rank + cfg.qk_rope_dim);
    std::vector<float> k_nope(cfg.num_heads * cfg.qk_nope_dim);
    std::vector<float> q_nope(cfg.num_heads * cfg.qk_nope_dim);
    std::vector<float> v(cfg.num_heads * cfg.v_dim);
    std::vector<float> attn_out(cfg.num_heads * cfg.v_dim);
    std::vector<float> scores(cfg.max_seq_len);
    std::vector<float> shared_gate(cfg.moe_intermediate), shared_up(cfg.moe_intermediate), shared_down(H);
    std::vector<float> expert_gate(cfg.moe_intermediate), expert_up(cfg.moe_intermediate), expert_down(H);
    std::vector<float> expert_wts(cfg.top_k);
    int expert_ids[64]; // top_k <= 64
    
    for (int il = 0; il < cfg.num_layers; il++) {
        auto& l = model.layers[il];
        
        // Pre-attention RMSNorm
        rmsnorm(norm.data(), x.data(), l.rms_attn_w.data(), H, 1e-6f);
        
        // ── MLA: compressed KV ──
        // c = x @ W_kv_a [hidden] → [kv_lora_rank + qk_rope_dim]
        matmul(k_comp.data(), norm.data(), l.w_kv_a.data(), cfg.kv_lora_rank + cfg.qk_rope_dim, H);
        for (int i = 0; i < cfg.kv_lora_rank + cfg.qk_rope_dim; i++)
            k_comp[i] += l.w_kv_a_bias[i];
        
        // Extract k_rope (last qk_rope_dim elements of c) and apply RoPE
        for (int i = 0; i < cfg.qk_rope_dim; i++)
            q_rope[i] = k_comp[cfg.kv_lora_rank + i];
        rope(q_rope.data(), cfg.qk_rope_dim, pos, 10000.0f);
        
        // Decompress K and V from cached latent
        // K_nope = c[:kv_lora_rank] @ W_kv_b[:, :n_heads * qk_nope_dim]
        // V      = c[:kv_lora_rank] @ W_kv_b[:, n_heads * qk_nope_dim:]
        int kv_head_stride = cfg.qk_nope_dim + cfg.v_dim;
        for (int h = 0; h < cfg.num_heads; h++) {
            for (int d = 0; d < cfg.qk_nope_dim; d++) {
                float sum = 0;
                for (int j = 0; j < cfg.kv_lora_rank; j++)
                    sum += k_comp[j] * l.w_kv_b[(size_t)j * cfg.num_heads * kv_head_stride + (size_t)h * kv_head_stride + d];
                k_nope[(size_t)h * cfg.qk_nope_dim + d] = sum;
            }
            for (int d = 0; d < cfg.v_dim; d++) {
                float sum = 0;
                for (int j = 0; j < cfg.kv_lora_rank; j++)
                    sum += k_comp[j] * l.w_kv_b[(size_t)j * cfg.num_heads * kv_head_stride + (size_t)h * kv_head_stride + cfg.qk_nope_dim + d];
                v[(size_t)h * cfg.v_dim + d] = sum;
            }
        }
        
        // Store in KV cache
        if ((int)mla_kv_cache.size() <= il) mla_kv_cache.resize(cfg.num_layers);
        mla_kv_cache[il].resize((size_t)(pos + 1) * (cfg.qk_nope_dim + cfg.qk_rope_dim + cfg.v_dim) * cfg.num_heads);
        int kv_stride = cfg.num_heads * (cfg.qk_nope_dim + cfg.qk_rope_dim + cfg.v_dim);
        for (int h = 0; h < cfg.num_heads; h++) {
            for (int d = 0; d < cfg.qk_nope_dim; d++)
                mla_kv_cache[il][(size_t)pos * kv_stride + (size_t)h * kv_stride + d] = k_nope[(size_t)h * cfg.qk_nope_dim + d];
            for (int d = 0; d < cfg.qk_rope_dim; d++)
                mla_kv_cache[il][(size_t)pos * kv_stride + (size_t)h * kv_stride + cfg.qk_nope_dim + d] = q_rope[d]; // cached k_rope
            for (int d = 0; d < cfg.v_dim; d++)
                mla_kv_cache[il][(size_t)pos * kv_stride + (size_t)h * kv_stride + cfg.qk_nope_dim + cfg.qk_rope_dim + d] = v[(size_t)h * cfg.v_dim + d];
        }
        
        // ── MLA: compressed Q ──
        // q_comp = x @ W_q_a [hidden] → [q_lora_rank]
        matmul(q_comp.data(), norm.data(), l.w_q_a.data(), cfg.q_lora_rank, H);
        for (int i = 0; i < cfg.q_lora_rank; i++) q_comp[i] += l.w_q_a_bias[i];
        
        // Q decompress: q_nope = q_comp @ W_q_b [q_lora_rank] → [n_heads * qk_nope_dim]
        for (int h = 0; h < cfg.num_heads; h++) {
            for (int d = 0; d < cfg.qk_nope_dim; d++) {
                float sum = 0;
                for (int j = 0; j < cfg.q_lora_rank; j++)
                    sum += q_comp[j] * l.w_q_b[(size_t)j * cfg.num_heads * cfg.qk_nope_dim + (size_t)h * cfg.qk_nope_dim + d];
                q_nope[(size_t)h * cfg.qk_nope_dim + d] = sum;
            }
        }
        
        // Apply RoPE to Q rope part (q_rope was set above from k_comp - not correct for Q)
        // Actually: q_rope = x @ W_q_a[:,-qk_rope_dim:] — last qk_rope_dim of W_q_a output
        // We already have k_comp for KV. For Q rope, we need separate projection.
        // Re-use q_comp but only the rope portion
        matmul(q_rope.data(), norm.data(), &l.w_q_a[(size_t)cfg.q_lora_rank * H - cfg.qk_rope_dim], cfg.qk_rope_dim, H);
        rope(q_rope.data(), cfg.qk_rope_dim, pos, 10000.0f);
        
        // ── MLA attention ──
        float scale = 1.0f / sqrtf((float)(cfg.qk_nope_dim + cfg.qk_rope_dim));
        const auto& cache = mla_kv_cache[il];
        int seq_len = pos + 1;
        
        for (int h = 0; h < cfg.num_heads; h++) {
            std::fill(scores.begin(), scores.begin() + seq_len, 0.0f);
            
            float max_score = -1e30f;
            for (int s = 0; s < seq_len; s++) {
                float acc = 0;
                for (int d = 0; d < cfg.qk_nope_dim; d++)
                    acc += q_nope[(size_t)h * cfg.qk_nope_dim + d] * cache[(size_t)s * kv_stride + (size_t)h * kv_stride + d];
                for (int d = 0; d < cfg.qk_rope_dim; d++)
                    acc += q_rope[d] * cache[(size_t)s * kv_stride + (size_t)h * kv_stride + cfg.qk_nope_dim + d];
                scores[s] = acc * scale;
                max_score = std::max(max_score, scores[s]);
            }
            
            // Softmax
            float sum = 0;
            for (int s = 0; s < seq_len; s++) { scores[s] = expf(scores[s] - max_score); sum += scores[s]; }
            float inv = 1.0f / sum;
            
            // Weighted V
            for (int d = 0; d < cfg.v_dim; d++) {
                float acc = 0;
                for (int s = 0; s < seq_len; s++)
                    acc += scores[s] * inv * cache[(size_t)s * kv_stride + (size_t)h * kv_stride + cfg.qk_nope_dim + cfg.qk_rope_dim + d];
                attn_out[(size_t)h * cfg.v_dim + d] = acc;
            }
        }
        
        // Output projection
        std::vector<float> attn_out_proj(H);
        matmul(attn_out_proj.data(), attn_out.data(), l.w_o.data(), H, cfg.num_heads * cfg.v_dim);
        for (int i = 0; i < H; i++) x[i] += attn_out_proj[i];
        
        // ── MoE FFN ──
        rmsnorm(norm.data(), x.data(), l.rms_ffn_w.data(), H, 1e-6f);
        
        // Shared expert
        matmul(shared_gate.data(), norm.data(), l.w_shared_gate.data(), cfg.moe_intermediate, H);
        matmul(shared_up.data(), norm.data(), l.w_shared_up.data(), cfg.moe_intermediate, H);
        for (int i = 0; i < cfg.moe_intermediate; i++) {
            shared_gate[i] = silu(shared_gate[i]);
            shared_gate[i] *= shared_up[i];
        }
        matmul(shared_down.data(), shared_gate.data(), l.w_shared_down.data(), H, cfg.moe_intermediate);
        
        std::fill(expert_down.begin(), expert_down.end(), 0.0f);
        
        if (cfg.n_routed_experts > 0 && cfg.top_k > 0) {
            // Router
            std::vector<float> router_scores(cfg.n_routed_experts);
            matmul(router_scores.data(), norm.data(), l.w_gate.data(), cfg.n_routed_experts, H);
            
            // Softmax
            float mx = router_scores[0]; for (int i = 1; i < cfg.n_routed_experts; i++) mx = std::max(mx, router_scores[i]);
            float sum = 0; for (int i = 0; i < cfg.n_routed_experts; i++) { router_scores[i] = expf(router_scores[i] - mx); sum += router_scores[i]; }
            for (int i = 0; i < cfg.n_routed_experts; i++) router_scores[i] /= sum;
            
            // Top-k selection
            for (int k = 0; k < cfg.top_k; k++) {
                int best = 0; float best_v = -1e30f;
                for (int i = 0; i < cfg.n_routed_experts; i++) {
                    if (router_scores[i] > best_v) { best_v = router_scores[i]; best = i; }
                }
                expert_ids[k] = best;
                expert_wts[k] = router_scores[best];
                router_scores[best] = -1e30f;
            }
            
            // Process each selected expert
            for (int k = 0; k < cfg.top_k; k++) {
                int eid = expert_ids[k];
                float wt = expert_wts[k];
                
                // Expert gate
                for (int i = 0; i < cfg.moe_intermediate; i++) {
                    float sum = 0;
                    for (int j = 0; j < H; j++)
                        sum += norm[j] * l.exp_gate[(size_t)eid * H * cfg.moe_intermediate + (size_t)j * cfg.moe_intermediate + i];
                    expert_gate[i] = sum;
                }
                // Expert up
                for (int i = 0; i < cfg.moe_intermediate; i++) {
                    float sum = 0;
                    for (int j = 0; j < H; j++)
                        sum += norm[j] * l.exp_up[(size_t)eid * H * cfg.moe_intermediate + (size_t)j * cfg.moe_intermediate + i];
                    expert_up[i] = sum;
                }
                
                // SiLU gate
                for (int i = 0; i < cfg.moe_intermediate; i++) {
                    expert_gate[i] = silu(expert_gate[i]);
                    expert_gate[i] *= expert_up[i];
                }
                
                // Expert down
                for (int i = 0; i < H; i++) {
                    float sum = 0;
                    for (int j = 0; j < cfg.moe_intermediate; j++)
                        sum += expert_gate[j] * l.exp_down[(size_t)eid * cfg.moe_intermediate * H + (size_t)j * H + i];
                    expert_down[i] += sum * wt;
                }
            }
        }
        
        // Combine shared + routed + residual
        for (int i = 0; i < H; i++)
            x[i] += shared_down[i] + expert_down[i];
    }
    
    // Final RMSNorm + output projection
    rmsnorm(norm.data(), x.data(), model.final_norm_w.data(), H, 1e-6f);
    
    std::vector<float> logits(cfg.vocab_size);
    if (!model.output_w.empty()) {
        matmul(logits.data(), norm.data(), model.output_w.data(), cfg.vocab_size, H);
    } else {
        // Tied embeddings
        for (int i = 0; i < cfg.vocab_size; i++) {
            float s = 0;
            for (int j = 0; j < H; j++) s += norm[j] * model.token_emb[(size_t)i * H + j];
            logits[i] = s;
        }
    }
    
    pos++;
    return logits;
}
