// dump_model_dims.cpp — Read Q4NX model file and print xclbin dimensions
// Compile: g++ -std=c++23 -O2 -I../src -I../include -o dump_model_dims dump_model_dims.cpp ../src/dequant_q4nx.cpp -luuid
// Usage: ./dump_model_dims /path/to/model.q4nx <model_tag>
#include <cstdio>
#include <cstring>
#include "model_config.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.q4nx> [model_tag]\n", argv[0]);
        return 1;
    }
    const char* tag = argc > 2 ? argv[2] : "auto";
    auto cfg = parse_q4nx_header(argv[1], tag);
    
    if (!cfg.valid()) {
        fprintf(stderr, "❌ Invalid model config\n");
        return 1;
    }
    
    printf("Model: %s\n", cfg.model_tag.c_str());
    printf("═══════════════════════════════════════════\n");
    printf("  H = %d  (hidden_size)\n", cfg.H);
    printf("  NC = %d  (num_layers)\n", cfg.NC);
    printf("  NH = %d  (num_heads)\n", cfg.NH);
    printf("  NKV = %d  (num_kv_heads)\n", cfg.NKV);
    printf("  HD = %d  (head_dim)\n", cfg.HD);
    printf("  IM = %d  (intermediate_size)\n", cfg.IM);
    printf("  NV = %d  (vocab_size)\n", cfg.NV);
    printf("  GQA = %d\n", cfg.GQA);
    printf("  ROPE_THETA = %.0f\n", cfg.rope_theta);
    printf("  GU_FUSED = %s\n", cfg.gu_split ? "no (separate G/U)" : "yes");
    printf("\n");
    printf("XCLBIN dimensions:\n");
    printf("  QKV: K=%5d, N=%5d, cols=%d\n", cfg.xclbin_qkv_k, cfg.xclbin_qkv_n, 
           cfg.xclbin_qkv_n / 128 <= 8 ? cfg.xclbin_qkv_n / 128 : 8);
    printf("  O:   K=%5d, N=%5d, cols=%d\n", cfg.xclbin_o_k, cfg.xclbin_o_n,
           cfg.xclbin_o_n / 128 <= 8 ? cfg.xclbin_o_n / 128 : 4);
    if (cfg.gu_split) {
        printf("  G:   K=%5d, N=%5d, cols=%d\n", cfg.xclbin_g_k, cfg.xclbin_g_n,
               cfg.xclbin_g_n / 128 <= 8 ? cfg.xclbin_g_n / 128 : 8);
        printf("  U:   K=%5d, N=%5d, cols=%d\n", cfg.xclbin_u_k, cfg.xclbin_u_n,
               cfg.xclbin_u_n / 128 <= 8 ? cfg.xclbin_u_n / 128 : 8);
    } else {
        printf("  GU:  K=%5d, N=%5d, cols=%d\n", cfg.xclbin_gu_k, cfg.xclbin_gu_n,
               cfg.xclbin_gu_n / 128 <= 8 ? cfg.xclbin_gu_n / 128 : 8);
    }
    printf("  D:   K=%5d, N=%5d, cols=%d\n", cfg.xclbin_d_k, cfg.xclbin_d_n,
           cfg.xclbin_d_n / 128 <= 8 ? cfg.xclbin_d_n / 128 : 4);
    printf("\n");
    printf("I8 tile row counts:\n");
    printf("  Q_I8R=%d, KV_I8R=%d, O_I8R=%d\n", 
           cfg.H * cfg.xclbin_qkv_n / 8192,
           cfg.H * (cfg.NKV * cfg.HD) / 8192,
           cfg.xclbin_o_k * cfg.H / 8192);
    printf("  GU_I8R=%d, D_I8R=%d, LM_I8R=%d\n",
           cfg.H * cfg.IM / 8192,
           cfg.IM * cfg.H / 8192,
           cfg.NV * cfg.H / 8192);
    return 0;
}
