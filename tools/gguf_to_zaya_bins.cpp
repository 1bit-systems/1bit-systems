/** gguf_to_zaya_bins.cpp — Extract GGUF model weights to Zaya-format .bin files
 *  Usage: ./build/gguf_to_zaya_bins model.gguf output_dir/
 */
#include "gguf_reader.h"
#include <cstdio>
#include <string>
#include <filesystem>
#include <vector>
#include <cstdint>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s model.gguf output_dir/\n", argv[0]);
        return 1;
    }
    
    GgufReader reader;
    if (!reader.open(argv[1])) {
        fprintf(stderr, "Failed to open GGUF: %s\n", argv[1]);
        return 1;
    }
    
    std::string out_dir = argv[2];
    std::filesystem::create_directories(out_dir);
    
    printf("GGUF: %s\n", argv[1]);
    printf("  Architecture: %s\n", reader.architecture().c_str());
    
    // Read model config (use uint32_t for get_u32)
    uint32_t H = 0, L = 0, NH = 0, NKV = 0, HD = 0, FF = 0, V = 0;
    reader.get_u32("block_count", L);
    reader.get_u32("embedding_length", H);
    reader.get_u32("attention.head_count", NH);
    reader.get_u32("attention.head_count_kv", NKV);
    reader.get_u32("feed_forward_length", FF);
    // head_dim may be stored as attention.key_length
    reader.get_u32("attention.key_length", HD);
    if (!HD) reader.get_u32("head_dim", HD);
    
    // Get vocab size
    V = reader.vocab_count();
    if (V <= 0) V = 151936;
    
    // Infer head_dim from H / NH if still unknown
    if (HD <= 0 && NH > 0) HD = H / NH;
    
    printf("  H=%u L=%u NH=%u NKV=%u HD=%u FF=%u V=%u\n", H, L, NH, NKV, HD, FF, V);
    
    if (L <= 0) L = 40; // fallback
    if (HD <= 0) HD = 128;
    if (NKV <= 0) NKV = NH;
    
    // Save manifest
    {
        std::string mp = out_dir + "/manifest.json";
        FILE* f = fopen(mp.c_str(), "w");
        fprintf(f, "{\n");
        fprintf(f, "  \"hidden_size\": %d,\n", H);
        fprintf(f, "  \"num_layers\": %d,\n", L);
        fprintf(f, "  \"num_attention_heads\": %d,\n", NH);
        fprintf(f, "  \"num_kv_heads\": %d,\n", NKV);
        fprintf(f, "  \"head_dim\": %d,\n", HD);
        fprintf(f, "  \"intermediate_size\": %d,\n", FF);
        fprintf(f, "  \"vocab_size\": %d\n", V);
        fprintf(f, "}\n");
        fclose(f);
        printf("  Manifest: %s\n", mp.c_str());
    }
    
    // Save a tokenizer stub
    {
        std::string tp = out_dir + "/tokenizer.json";
        FILE* f = fopen(tp.c_str(), "w");
        fprintf(f, "{\"bos_token_id\": 151643, \"eos_token_id\": 151643}\n");
        fclose(f);
    }
    
    // Helper: read tensor and write as f32 .bin file
    auto save_tensor = [&](const std::string& gguf_name, const std::string& zaya_name) {
        std::vector<float> buf;
        size_t n = 0;
        if (!reader.get_tensor_f32(gguf_name, buf, &n)) {
            // Try saved tensors
            const float* data = reader.get_tensor(gguf_name, &n);
            if (!data) {
                printf("  MISS %s\n", gguf_name.c_str());
                return false;
            }
            buf.assign(data, data + n);
        }
        
        std::string out_path = out_dir + "/" + zaya_name + ".bin";
        FILE* f = fopen(out_path.c_str(), "wb");
        if (!f) { printf("  FAIL %s\n", out_path.c_str()); return false; }
        fwrite(buf.data(), 4, buf.size(), f);
        fclose(f);
        
        printf("  %s (%zu MB)\n", zaya_name.c_str(), (buf.size() * 4) / (1024*1024));
        return true;
    };
    
    // Zaya naming conventions for common tensors
    // Embedding
    save_tensor("token_embd.weight", "model_embed_tokens_weight");
    save_tensor("output_norm.weight", "model_norm_weight");
    save_tensor("output.weight", "model_lm_head_weight");
    
    // Per-layer weights
    for (int i = 0; i < L; i++) {
        std::string p = "blk." + std::to_string(i) + ".";
        
        save_tensor(p + "attn_norm.weight", 
            "model_layers_" + std::to_string(i) + "_input_layernorm_weight");
        save_tensor(p + "attn_q.weight", 
            "model_layers_" + std::to_string(i) + "_self_attn_q_proj_weight");
        save_tensor(p + "attn_k.weight", 
            "model_layers_" + std::to_string(i) + "_self_attn_k_proj_weight");
        save_tensor(p + "attn_v.weight", 
            "model_layers_" + std::to_string(i) + "_self_attn_v_proj_weight");
        save_tensor(p + "attn_output.weight", 
            "model_layers_" + std::to_string(i) + "_self_attn_o_proj_weight");
        save_tensor(p + "ffn_norm.weight", 
            "model_layers_" + std::to_string(i) + "_post_attention_layernorm_weight");
        save_tensor(p + "ffn_gate.weight", 
            "model_layers_" + std::to_string(i) + "_mlp_gate_proj_weight");
        save_tensor(p + "ffn_up.weight", 
            "model_layers_" + std::to_string(i) + "_mlp_up_proj_weight");
        save_tensor(p + "ffn_down.weight", 
            "model_layers_" + std::to_string(i) + "_mlp_down_proj_weight");
        
        // MoE-specific (Qwen3.6-35B uses MoE)
        save_tensor(p + "ffn_gate_inp.weight",
            "model_layers_" + std::to_string(i) + "_mlp_gate_router_weight");
    }
    
    printf("\nDone. Extracted to %s/\n", out_dir.c_str());
    return 0;
}
