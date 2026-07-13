#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../include/model_config.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.q4nx [tag]\n", argv[0]);
        return 1;
    }
    const char* tag = argc > 2 ? argv[2] : "test";
    
    ModelConfig cfg = parse_q4nx_header(argv[1], tag);
    if (!cfg.valid()) {
        fprintf(stderr, "FAILED to parse %s\n", argv[1]);
        return 1;
    }
    
    printf("Model: %s\n", tag);
    printf("  H=%d, NC=%d, NH=%d, NKV=%d, HD=%d, IM=%d, NV=%d\n",
           cfg.H, cfg.NC, cfg.NH, cfg.NKV, cfg.HD, cfg.IM, cfg.NV);
    printf("  GQA=%d, WQH=%d, WKVH=%d\n", cfg.GQA, cfg.WQH, cfg.WKVH);
    printf("  has_q_norm=%d, has_k_norm=%d, has_rope_freqs=%d, has_lm_head=%d\n",
           cfg.has_q_norm, cfg.has_k_norm, cfg.has_rope_freqs_file, cfg.has_lm_head);
    printf("  gu_split=%d, rope_theta=%.0f, rope_factor=%.1f\n",
           cfg.gu_split, cfg.rope_theta, cfg.rope_factor);
    printf("  QKV: K=%d, N=%d [q@%d, k@%d, v@%d]\n",
           cfg.xclbin_qkv_k, cfg.xclbin_qkv_n,
           0, cfg.qkv_k_offset, cfg.qkv_v_offset);
    printf("  O: K=%d, N=%d\n", cfg.xclbin_o_k, cfg.xclbin_o_n);
    if (cfg.gu_split) {
        printf("  G: K=%d, N=%d\n", cfg.xclbin_g_k, cfg.xclbin_g_n);
        printf("  U: K=%d, N=%d\n", cfg.xclbin_u_k, cfg.xclbin_u_n);
    } else {
        printf("  GU: K=%d, N=%d\n", cfg.xclbin_gu_k, cfg.xclbin_gu_n);
    }
    printf("  D: K=%d, N=%d\n", cfg.xclbin_d_k, cfg.xclbin_d_n);
    
    // Verify against known values
    struct Expected {
        const char* tag;
        int H, NC, NH, NKV, HD, IM, NV;
        bool qn, kn, rf, lm, gs;
    } expected[] = {
        {"qwen3_0_6b",   1024, 28, 16,  8, 128,  3072, 151936, true,  true,  false, true,  false},
        {"llama",        4096, 32, 32,  8, 128, 14336, 128256, false, false, true,  true,  true},
        {"qwen3_vl_4b",  2560, 36, 32,  8, 128,  9728, 151936, true,  true,  false, true,  true},
        {"gemma4_e2b",   1536, 35, 8,   1, 256,  6144, 262144, true,  true,  true,  true,  false},
        {"qwen3_8b",     4096, 36, 32,  8, 128, 12288, 151936, true,  true,  false, true,  true},
    };
    
    int n_exp = sizeof(expected) / sizeof(expected[0]);
    bool found = false;
    for (int i = 0; i < n_exp; i++) {
        if (strcmp(expected[i].tag, tag) == 0) {
            found = true;
            int errors = 0;
            #define CHECK(field, exp) do { if (cfg.field != exp) { \
                fprintf(stderr, "  MISMATCH %s: got %d, expected %d\n", #field, cfg.field, exp); errors++; } } while(0)
            #define CHECKB(field, exp) do { if (cfg.field != exp) { \
                fprintf(stderr, "  MISMATCH %s: got %d, expected %d\n", #field, (int)cfg.field, (int)exp); errors++; } } while(0)
            CHECK(H, expected[i].H);
            CHECK(NC, expected[i].NC);
            CHECK(NH, expected[i].NH);
            CHECK(NKV, expected[i].NKV);
            CHECK(HD, expected[i].HD);
            CHECK(IM, expected[i].IM);
            CHECK(NV, expected[i].NV);
            CHECKB(has_q_norm, expected[i].qn);
            CHECKB(has_k_norm, expected[i].kn);
            CHECKB(has_rope_freqs_file, expected[i].rf);
            CHECKB(has_lm_head, expected[i].lm);
            CHECKB(gu_split, expected[i].gs);
            #undef CHECK
            #undef CHECKB
            if (errors == 0) {
                printf("  ✅ ALL CHECKS PASS\n");
            } else {
                printf("  ❌ %d MISMATCHES\n", errors);
            }
            break;
        }
    }
    if (!found) {
        printf("  ⚠️  No expected values for tag '%s'\n", tag);
    }
    
    return cfg.valid() ? 0 : 1;
}
