// ZAYA1-8B Fast C++ Inference — pre-dequantized FP16 weights
// Build: g++ -O3 -I. -o zaya_fast zaya_fast.cpp -lpthread
// Run:   ./zaya_fast

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <dirent.h>

// Architecture constants
constexpr int H = 2048, NQ = 8, NKV = 2, HD = 128;
constexpr int QD = NQ * HD, KD = NKV * HD, QKV = QD + KD;
constexpr int N_LAYERS = 40, VOCAB = 262272;
constexpr int N_EXP = 16, N_FF = 2048, N_EXP_T = 17, RTR_H = 256;

// Tensor storage
struct Tensor {
    int rows, cols;
    std::vector<float> data;  // FP32 for now (converted from FP16 cache)
};

std::vector<Tensor> g_weights;

// Load FP16 cache file
Tensor load_fp16(const std::string& name, int rows, int cols) {
    std::string path = "/tmp/zaya_fp16_cache/" + name + ".fp16";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Missing: %s\n", path.c_str()); return {rows, cols, {}}; }
    
    fseek(f, 0, SEEK_END);
    int n = ftell(f) / 2;
    fseek(f, 0, SEEK_SET);
    
    std::vector<uint16_t> fp16(n);
    fread(fp16.data(), 2, n, f);
    fclose(f);
    
    // Convert to FP32
    std::vector<float> f32(n);
    for (int i = 0; i < n; i++) {
        uint32_t bits = ((uint32_t)fp16[i]) << 16;
        memcpy(&f32[i], &bits, 4);
    }
    
    return {rows, cols, f32};
}

// Load all weights
void load_all() {
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // Embed
    g_weights.push_back(load_fp16("model_embed_tokens_weight", VOCAB, H));
    // Final norm
    g_weights.push_back(load_fp16("model_norm_weight", H, 1));
    // Input scale
    g_weights.push_back(load_fp16("model_input_hidden_states_scale", H, 1));
    g_weights.push_back(load_fp16("model_input_hidden_states_bias", H, 1));
    
    // Per-layer weights
    for (int l = 0; l < N_LAYERS; l++) {
        char prefix[64]; sprintf(prefix, "model_layers_%d_", l);
        
        auto& ln = g_weights.emplace_back(load_fp16(std::string(prefix) + "input_layernorm_weight", H, 1));
        auto& pa = g_weights.emplace_back(load_fp16(std::string(prefix) + "post_attention_layernorm_weight", H, 1));
        
        // Attention
        auto& q = g_weights.emplace_back(load_fp16(std::string(prefix) + "self_attn_q_proj_weight", QD, H));
        auto& k = g_weights.emplace_back(load_fp16(std::string(prefix) + "self_attn_k_proj_weight", KD, H));
        auto& v1 = g_weights.emplace_back(load_fp16(std::string(prefix) + "self_attn_v_proj_current_weight", KD/2, H));
        auto& v2 = g_weights.emplace_back(load_fp16(std::string(prefix) + "self_attn_v_proj_delayed_weight", KD/2, H));
        auto& o = g_weights.emplace_back(load_fp16(std::string(prefix) + "self_attn_o_proj_weight", H, QD));
        
        // Conv
        auto& cdw = g_weights.emplace_back(load_fp16(std::string(prefix) + "self_attn_conv_qk_depthwise_weight", QKV, 2));
        auto& cdb = g_weights.emplace_back(load_fp16(std::string(prefix) + "self_attn_conv_qk_depthwise_bias", QKV, 1));
        auto& cgw = g_weights.emplace_back(load_fp16(std::string(prefix) + "self_attn_conv_qk_grouped_weight", QKV, 256));
        auto& cgb = g_weights.emplace_back(load_fp16(std::string(prefix) + "self_attn_conv_qk_grouped_bias", QKV, 1));
        
        // MoE gate
        auto& gd = g_weights.emplace_back(load_fp16(std::string(prefix) + "mlp_gate_down_proj_weight", H, RTR_H));
        auto& gdb = g_weights.emplace_back(load_fp16(std::string(prefix) + "mlp_gate_down_proj_bias", RTR_H, 1));
        auto& rn = g_weights.emplace_back(load_fp16(std::string(prefix) + "mlp_gate_router_mlp_norm_weight", RTR_H, 1));
        auto& rf1 = g_weights.emplace_back(load_fp16(std::string(prefix) + "mlp_gate_router_mlp_fc1_weight", RTR_H, RTR_H));
        auto& rf1b = g_weights.emplace_back(load_fp16(std::string(prefix) + "mlp_gate_router_mlp_fc1_bias", RTR_H, 1));
        auto& rf2 = g_weights.emplace_back(load_fp16(std::string(prefix) + "mlp_gate_router_mlp_fc2_weight", RTR_H, RTR_H));
        auto& rf2b = g_weights.emplace_back(load_fp16(std::string(prefix) + "mlp_gate_router_mlp_fc2_bias", RTR_H, 1));
        auto& ro = g_weights.emplace_back(load_fp16(std::string(prefix) + "mlp_gate_router_mlp_out_proj_weight", N_EXP_T, RTR_H));
        auto& bb = g_weights.emplace_back(load_fp16(std::string(prefix) + "mlp_gate_balancing_biases", N_EXP_T, 1));
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("Loaded %zu tensors in %.1f ms\n", g_weights.size(),
           std::chrono::duration<float, std::milli>(t1 - t0).count());
}

int main() {
    printf("ZAYA1-8B Fast C++ Inference\n");
    printf("===========================\n\n");
    
    load_all();
    
    return 0;
}
