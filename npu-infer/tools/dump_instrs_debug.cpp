// dump_instrs_debug.cpp — Debug version
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <dlfcn.h>
#include <link.h>
#include <unistd.h>

struct LM_Config {
    std::string str0;
    std::string str1;
    std::string str2;
    std::string str3;
    uint64_t pad_80[1];
    std::string str4;
    uint64_t field_a8;
    uint64_t field_b0;
    uint64_t field_b8;
    uint64_t field_c0;
    uint64_t field_c8;
    uint64_t field_d0;
    uint64_t field_d8;
    uint64_t field_e0;
};

int main() {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    
    fprintf(stderr, "Opening libraries...\n");
    
    void* qwen_handle = dlopen("/opt/fastflowlm/lib/flm/libqwen3_npu.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!qwen_handle) { fprintf(stderr, "ERROR: %s\n", dlerror()); return 1; }
    
    void* gemm_handle = dlopen("/opt/fastflowlm/lib/flm/libgemm.so", RTLD_LAZY | RTLD_GLOBAL);
    void* mha_handle = dlopen("/opt/fastflowlm/lib/flm/libmha.so", RTLD_LAZY | RTLD_GLOBAL);
    void* dequant_handle = dlopen("/opt/fastflowlm/lib/flm/libdequant.so", RTLD_LAZY | RTLD_GLOBAL);
    
    fprintf(stderr, "All libraries loaded\n");
    
    // Get constructor and test
    auto ctor = (void (*)(void*, LM_Config, unsigned int))
        dlsym(qwen_handle, "_ZN18qwen3_npu_sequenceC1E9LM_Configj");
    auto dtor = (void (*)(void*))
        dlsym(qwen_handle, "_ZN18qwen3_npu_sequenceD1Ev");
    
    auto gen_layer = (void (*)(void*, void*, unsigned int))
        dlsym(qwen_handle, "_ZN18qwen3_npu_sequence4Impl13gen_layer_seqEP12npu_sequencej");
    
    fprintf(stderr, "ctor=%p dtor=%p gen_layer=%p\n", (void*)ctor, (void*)dtor, (void*)gen_layer);
    
    if (!ctor) { fprintf(stderr, "Missing ctor\n"); return 1; }
    
    // Build LM_Config
    LM_Config config;
    memset(&config, 0, sizeof(config));
    
    config.str0 = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    config.str1 = "qwen3:0.6b";
    config.str2 = "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2";
    config.str3 = "/opt/fastflowlm/lib/flm";
    
    config.field_a8 = 1024;      // hidden_size
    config.field_b0 = 28;        // num_layers
    config.field_b8 = 16;        // num_attention_heads
    config.field_c0 = 8;         // num_key_value_heads
    config.field_c8 = 3072;      // intermediate_size
    config.field_d0 = 128;       // head_dim
    config.field_d8 = 151936;    // vocab_size
    config.field_e0 = 40960;     // max_position_embeddings
    
    fprintf(stderr, "Config built, sizeof=%zu\n", sizeof(config));
    fprintf(stderr, "Calling constructor...\n");
    
    // Allocate sequence object on heap
    void* seq = malloc(1024);
    memset(seq, 0, 1024);
    
    ctor(seq, config, 4096);
    fprintf(stderr, "Constructor returned!\n");
    
    // Get Impl pointer
    void** pp = (void**)seq;
    fprintf(stderr, "seq[0]=%p seq[1]=%p seq[2]=%p\n", pp[0], pp[1], pp[2]);
    
    // Try to generate layer sequence
    if (gen_layer) {
        fprintf(stderr, "Calling gen_layer_seq...\n");
        gen_layer(pp[0], seq, 0);
        fprintf(stderr, "gen_layer_seq returned!\n");
    }
    
    dtor(seq);
    free(seq);
    
    dlclose(qwen_handle);
    return 0;
}
