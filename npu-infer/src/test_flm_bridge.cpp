/**
 * test_flm_bridge.cpp — Verify FLM bridge generates non-empty instruction sequences
 * 
 * Build: via CMake (test_flm_bridge target)
 */

#include "flm_bridge.h"
#include <cstdio>
#include <cstring>

int main() {
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  FLM Bridge Test                                 ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

    auto& bridge = FlmBridge::instance();
    
    FlmBridge::Config cfg;
    cfg.hidden_size = 1024;
    cfg.num_heads = 16;
    cfg.num_kv_heads = 8;
    cfg.num_layers = 28;
    cfg.intermediate_size = 3072;
    cfg.max_seq_len = 40960;
    cfg.vocab_size = 151936;
    cfg.head_dim = 128;
    cfg.batch_padded = 512;
    cfg.xclbin_path = "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2";
    
    if (!bridge.init(cfg)) {
        fprintf(stderr, "FAIL: bridge init failed\n");
        return 1;
    }
    printf("✅ Bridge initialized\n\n");
    
    // Test 1: q_proj (M=512, N=1024, K=1024) - standard GEMM
    printf("=== Test 1: q_proj (M=512, N=1024, K=1024) ===\n");
    auto q_instrs = bridge.gen_gemm_instrs(512, 1024, 1024, 0);
    if (q_instrs.empty()) {
        fprintf(stderr, "FAIL: q_proj generated empty instructions\n");
        return 1;
    }
    printf("✅ %zu instructions (%zu bytes)\n", q_instrs.size(), q_instrs.size() * 4);
    printf("   First 8: ");
    for (int i = 0; i < 8 && i < (int)q_instrs.size(); i++)
        printf("0x%08x ", q_instrs[i]);
    printf("\n\n");
    
    // Test 2: gate_proj (M=512, N=3072, K=1024)
    printf("=== Test 2: gate_proj (M=512, N=3072, K=1024) ===\n");
    auto g_instrs = bridge.gen_gemm_instrs(512, 3072, 1024, 0);
    if (g_instrs.empty()) {
        fprintf(stderr, "FAIL: gate_proj generated empty instructions\n");
        return 1;
    }
    printf("✅ %zu instructions (%zu bytes)\n", g_instrs.size(), g_instrs.size() * 4);
    
    // Test 3: down_proj (M=512, N=1024, K=3072)
    printf("\n=== Test 3: down_proj (M=512, N=1024, K=3072) ===\n");
    auto d_instrs = bridge.gen_gemm_instrs(512, 1024, 3072, 0);
    if (d_instrs.empty()) {
        fprintf(stderr, "FAIL: down_proj generated empty instructions\n");
        return 1;
    }
    printf("✅ %zu instructions (%zu bytes)\n", d_instrs.size(), d_instrs.size() * 4);
    
    // Test 4: lm_head (M=512, N=151936, K=1024) - large output
    printf("\n=== Test 4: lm_head (M=512, N=151936, K=1024) ===\n");
    auto l_instrs = bridge.gen_gemm_instrs(512, 151936, 1024, 0);
    if (l_instrs.empty()) {
        fprintf(stderr, "FAIL: lm_head generated empty instructions\n");
        return 1;
    }
    printf("✅ %zu instructions (%zu bytes)\n", l_instrs.size(), l_instrs.size() * 4);
    
    // Test 5: padded single token (M=1 auto-padded to 512, N=1024, K=1024)
    printf("\n=== Test 5: auto-padded (M=1→512, N=1024, K=1024) ===\n");
    auto p_instrs = bridge.gen_gemm_instrs(1, 1024, 1024, 0);
    if (p_instrs.empty()) {
        fprintf(stderr, "FAIL: padded generated empty instructions\n");
        return 1;
    }
    printf("✅ %zu instructions (%zu bytes)\n", p_instrs.size(), p_instrs.size() * 4);
    
    // Save all instructions for analysis
    auto save = [](const std::vector<uint32_t>& data, const char* path) {
        FILE* f = fopen(path, "wb");
        if (!f) { perror(path); return false; }
        fwrite(data.data(), 4, data.size(), f);
        fclose(f);
        printf("  Saved to %s\n", path);
        return true;
    };
    
    save(q_instrs, "/tmp/flm_q_proj.bin");
    save(g_instrs, "/tmp/flm_gate_proj.bin");
    save(d_instrs, "/tmp/flm_down_proj.bin");
    save(l_instrs, "/tmp/flm_lm_head.bin");
    
    printf("\n=== ALL TESTS PASSED ===\n");
    return 0;
}
