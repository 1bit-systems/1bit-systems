#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <dlfcn.h>
#include <chrono>
#include <new>

struct npu_sequence { uint8_t data[65536]; };

// Stubs (exported so dlopen finds them)
struct bytes { void* impl; bytes() : impl(nullptr) {} };
extern "C" void _ZN11SafeTensors12load_weightsER5bytesNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(
    bytes& b, std::string path) 
{
    fprintf(stderr, "STUB: SafeTensors::load_weights(%s)\n", path.c_str());
    b.impl = malloc(16);
}

extern "C" void _ZN3MHAC1ER10mha_type_ti(void* self, const void* t, int i) { memset(self, 0, 8); }
extern "C" void _ZN3MHAD1Ev(void* self) {}
extern "C" void _ZN5bytesC1Em(void* self, size_t sz) { 
    *(void**)self = malloc(sz ? sz : 16); 
    fprintf(stderr, "STUB: bytes(%zu)\n", sz);
}
extern "C" void _ZN5bytesC1ERN3xrt6deviceEm(void* self, void* dev, size_t sz) { 
    *(void**)self = malloc(sz ? sz : 16);
    fprintf(stderr, "STUB: bytes(device&,%zu)\n", sz);
}
extern "C" void _ZN5bytesD1Ev(void* self) { void* p = *(void**)self; if (p) free(p); }
extern "C" void _ZN6LMHeadD1Ev(void* self) {}
extern "C" void _ZN7DequantD1Ev(void* self) {}

int main() {
    printf("=== Qwen3 Layer with Proper Stubs v2 ===\n\n");
    
    // Load libs 
    dlopen("/opt/fastflowlm/lib/flm/libgemm.so", RTLD_LAZY | RTLD_GLOBAL);
    dlopen("/opt/fastflowlm/lib/flm/libmha.so", RTLD_LAZY | RTLD_GLOBAL);
    dlopen("/opt/xilinx/xrt/lib/libxrt_coreutil.so", RTLD_LAZY | RTLD_GLOBAL);
    
    void* qwen_lib = dlopen("/opt/fastflowlm/lib/flm/libqwen3_npu.so", RTLD_NOW | RTLD_GLOBAL);
    if (!qwen_lib) { printf("FAIL: %s\n", dlerror()); return 1; }
    printf("✅ libqwen3_npu.so loaded\n");
    
    // Resolve ImplC1
    auto pImplCtor = (void(*)(void*, void*, unsigned int))
        dlsym(qwen_lib, "_ZN18qwen3_npu_sequence4ImplC1E9LM_Configj");
    auto pSendX = (void(*)(void*, npu_sequence*))
        dlsym(qwen_lib, "_ZN18qwen3_npu_sequence4Impl7_send_xEP12npu_sequence");
    auto pMoveWeights = (void(*)(void*, npu_sequence*, unsigned int, unsigned int, unsigned int))
        dlsym(qwen_lib, "_ZN18qwen3_npu_sequence4Impl13_move_weightsEP12npu_sequencemmm");
    auto pSendRMS = (void(*)(void*, npu_sequence*))
        dlsym(qwen_lib, "_ZN18qwen3_npu_sequence4Impl17_send_rms_weightsEP12npu_sequence");
    auto pGenLayer = (void(*)(void*, npu_sequence*, unsigned int))
        dlsym(qwen_lib, "_ZN18qwen3_npu_sequence4Impl13gen_layer_seqEP12npu_sequencej");
    
    auto cmds2seq = (void(*)(npu_sequence*))
        dlsym(RTLD_DEFAULT, "_ZN12npu_sequence8cmds2seqEv");
    
    if (!pImplCtor || !pSendX || !pMoveWeights || !pSendRMS || !pGenLayer || !cmds2seq) {
        printf("FAIL: ImplCtor=%p SendX=%p MoveW=%p SendRMS=%p GenLayer=%p cmds2seq=%p\n",
               pImplCtor, pSendX, pMoveWeights, pSendRMS, pGenLayer, cmds2seq);
        return 1;
    }
    printf("✅ Symbols resolved\n");
    
    // Allocate Impl memory (0x70 = 112 bytes)
    void* impl_mem = aligned_alloc(16, 0x80);
    memset(impl_mem, 0, 0x80);
    
    // LM_Config - need proper string values since constructor accesses them
    // From disasm: accesses offsets 0, 8, 0x20, 0x28, 0x40, 0x48, 0x60, 0x68, 0xa8
    // These are std::string objects (each 32 bytes)
    struct LMConfig {
        std::string s1;  // 0x00 - model_path
        std::string s2;  // 0x20 - model_name
        std::string s3;  // 0x40 - model_type
        uint64_t v1;     // 0x60
        uint64_t v2;     // 0x68
        uint32_t v3;     // 0x70
        uint32_t v4;     // 0x74
        uint32_t npu_col; // 0xa8
    };
    
    // Use aligned storage
    alignas(16) char cfg_buf[0x200];
    LMConfig* cfg = new (cfg_buf) LMConfig();
    
    // Set some reasonable values
    cfg->s1 = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    cfg->s2 = "qwen3:0.6b";
    cfg->s3 = "Qwen3-0.6B";
    cfg->v1 = 0;
    cfg->v2 = 0;
    cfg->npu_col = 8; // npu_columns
    
    printf("Constructing Impl...\n");
    pImplCtor(impl_mem, cfg, 0);
    printf("  impl constructed\n");
    
    // Create sequence and test
    npu_sequence seq;
    memset(&seq, 0, sizeof(seq));
    *(uint32_t*)(seq.data + 0x00) = 4;
    *(uint32_t*)(seq.data + 0x04) = 4;
    *(uint32_t*)(seq.data + 0x20) = 4;
    
    printf("\n=== Layer Pipeline ===\n");
    
    auto t0 = std::chrono::steady_clock::now();
    
    printf("1. _send_x... "); fflush(stdout);
    pSendX(impl_mem, &seq);
    printf("ncmds=%u\n", *(uint32_t*)(seq.data + 0x1C));
    
    printf("2. _move_weights... "); fflush(stdout);
    pMoveWeights(impl_mem, &seq, 0, 0, 0);
    printf("ncmds=%u\n", *(uint32_t*)(seq.data + 0x1C));
    
    printf("3. _send_rms_weights... "); fflush(stdout);
    pSendRMS(impl_mem, &seq);
    printf("ncmds=%u\n", *(uint32_t*)(seq.data + 0x1C));
    
    printf("4. gen_layer_seq... "); fflush(stdout);
    pGenLayer(impl_mem, &seq, 0);
    printf("ncmds=%u\n", *(uint32_t*)(seq.data + 0x1C));
    
    printf("5. cmds2seq... "); fflush(stdout);
    cmds2seq(&seq);
    
    uint32_t* instr_data = *(uint32_t**)(seq.data + 0x40);
    size_t ninstr = (*(uint32_t**)(seq.data + 0x48) - instr_data);
    
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    
    printf("%zu words, %.3f ms\n", ninstr, ms);
    
    if (ninstr > 0 && ninstr < 1000000) {
        printf("First 8: ");
        for (int i = 0; i < 8; i++) printf("0x%08x ", instr_data[i]);
        printf("\n");
        
        FILE* f = fopen("/tmp/full_layer_stub.bin", "wb");
        fwrite(instr_data, 4, ninstr, f);
        fclose(f);
        printf("Saved (%zu words, %.1f KB)\n", ninstr, ninstr * 4.0 / 1024);
        
        // Sort of BD headers
        int bd = 0;
        for (size_t i = 2; i < ninstr; i++) if ((instr_data[i] & 0xFF) == 0x91) bd++;
        printf("BD(0x91) headers: %d\n", bd);
    }
    
    // Cleanup (trivial destructor)
    free(impl_mem);
    cfg->s1.~basic_string();
    cfg->s2.~basic_string();
    cfg->s3.~basic_string();
    
    printf("\nDONE\n");
    return 0;
}
