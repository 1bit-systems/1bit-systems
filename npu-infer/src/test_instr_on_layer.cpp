#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <dlfcn.h>
#include <chrono>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

static inline uint16_t float_to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    return (uint16_t)((bits + rounding_bias) >> 16);
}
static inline float bf16_to_float(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

struct npu_sequence { uint8_t data[65536]; };

int main() {
    printf("=== Instructions on layer.xclbin ===\n\n");
    
    // Load libgemm.so and libmha.so for instruction generation
    void* gemm_lib = dlopen("/opt/fastflowlm/lib/flm/libgemm.so", RTLD_LAZY | RTLD_LOCAL);
    void* mha_lib  = dlopen("/opt/fastflowlm/lib/flm/libmha.so", RTLD_LAZY | RTLD_LOCAL);
    
    auto gemm_ctor = (void(*)(void*, void*))dlsym(gemm_lib, "_ZN4GemmC1ER9LM_Config");
    auto gemm_gen = (void(*)(void*, npu_sequence*, uint32_t, uint32_t, uint32_t, uint32_t, bool, uint32_t))dlsym(gemm_lib,
        "_ZN4Gemm12generate_seqEP12npu_sequencejjjjbNS_17Activation_Type_tEj");
    auto cmds2seq = (void(*)(npu_sequence*))dlsym(mha_lib, "_ZN12npu_sequence8cmds2seqEv");
    
    struct LM_Config { char dummy[0x200]; };
    struct Gemm { void* impl; };
    
    Gemm gemm; memset(&gemm, 0, sizeof(gemm));
    LM_Config cfg; memset(&cfg, 0, sizeof(cfg));
    gemm_ctor(&gemm, &cfg);
    
    // Load layer.xclbin (maybe this one handles compute instructions)
    xrt::device device(0);
    std::string path = "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/layer.xclbin";
    FILE* f = fopen(path.c_str(), "rb");
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> raw(fsize);
    fread(raw.data(), 1, fsize, f); fclose(f);
    xrt::xclbin xclbin(raw);
    device.register_xclbin(xclbin);
    xrt::kernel kernel(device, xclbin.get_uuid(), "MLIR_AIE");
    
    printf("Kernel ready\n");
    
    // Create BOs
    size_t BO_SIZE = 8 * 1024 * 1024;
    size_t INSTR_SIZE = 512 * 1024;
    
    xrt::bo act_bo(device, BO_SIZE, xrt::bo::flags::host_only, (xrt::memory_group)0);
    xrt::bo ws_bo(device, BO_SIZE, xrt::bo::flags::host_only, (xrt::memory_group)0);
    xrt::bo w1_bo(device, BO_SIZE, xrt::bo::flags::host_only, (xrt::memory_group)0);
    xrt::bo w2_bo(device, BO_SIZE, xrt::bo::flags::host_only, (xrt::memory_group)0);
    xrt::bo kv_bo(device, BO_SIZE, xrt::bo::flags::host_only, (xrt::memory_group)0);
    xrt::bo instr_bo(device, INSTR_SIZE, xrt::bo::flags::host_only, (xrt::memory_group)1);
    
    printf("BOs: act=0x%llx w1=0x%llx\n",
           (unsigned long long)act_bo.address(), (unsigned long long)w1_bo.address());
    
    // Fill act with test data
    uint16_t* act_map = (uint16_t*)act_bo.map();
    for (int i = 0; i < 4096; i++) act_map[i] = float_to_bf16((float)(i % 100 + 1));
    act_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, 8192, 0);
    
    // Fill weights
    uint16_t* w1_map = (uint16_t*)w1_bo.map();
    for (size_t i = 0; i < BO_SIZE/2; i++) w1_map[i] = float_to_bf16(0.5f);
    w1_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, BO_SIZE, 0);
    
    // Generate GEMM instructions for large GEMM
    printf("\n=== Generating GEMM instructions (4096,1024,256) ===\n");
    
    npu_sequence seq;
    memset(&seq, 0, sizeof(seq));
    *(uint32_t*)(seq.data + 0x00) = 4;
    *(uint32_t*)(seq.data + 0x04) = 4;
    *(uint32_t*)(seq.data + 0x08) = (uint32_t)(act_bo.address() & 0xFFFFFFFF);
    *(uint32_t*)(seq.data + 0x0C) = (uint32_t)(act_bo.address() & 0xFFFFFFFF);
    *(uint32_t*)(seq.data + 0x10) = (uint32_t)(w1_bo.address() & 0xFFFFFFFF);
    *(uint32_t*)(seq.data + 0x14) = (uint32_t)(w1_bo.address() & 0xFFFFFFFF);
    *(uint32_t*)(seq.data + 0x18) = 0;
    
    gemm_gen(&gemm, &seq, 4096, 1024, 256, 4096, false, 3);
    cmds2seq(&seq);
    
    uint32_t* instr = *(uint32_t**)(seq.data + 0x40);
    size_t ninstr = (*(uint32_t**)(seq.data + 0x48) - instr);
    
    printf("Instructions: %zu words (%.1f KB)\n", ninstr, ninstr * 4.0 / 1024);
    printf("First 8: ");
    for (int i = 0; i < 8; i++) printf("0x%08x ", instr[i]);
    printf("\n");
    
    // Count BDs
    int bds = 0;
    for (size_t i = 2; i < ninstr; i++) {
        uint8_t b = instr[i] & 0xFF;
        if (b == 0x91 || (b >= 1 && b <= 15)) bds++;
    }
    printf("BD-like headers: %d\n", bds);
    
    // Submit to layer.xclbin with opcode=0
    if (ninstr > 0 && ninstr < 100000) {
        memcpy(instr_bo.map(), instr, ninstr * 4);
        instr_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, ninstr * 4, 0);
        
        printf("\n=== Submitting to layer.xclbin (opcode=0) ===\n");
        auto t0 = std::chrono::steady_clock::now();
        try {
            auto run = kernel((uint64_t)0, instr_bo, (uint32_t)ninstr,
                             act_bo, ws_bo, w1_bo, w2_bo, kv_bo);
            run.wait();
        } catch (const std::exception& e) {
            printf("FAILED: %s\n", e.what());
            return 1;
        }
        auto t1 = std::chrono::steady_clock::now();
        
        act_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, 128, 0);
        
        printf("Exec: %.3f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());
        printf("Out[0..7]: ");
        for (int i = 0; i < 8; i++) printf("%.3f ", bf16_to_float(instr[i]));
        printf("\n\n   (Oops, printed instr data not output data)\n");
        
        uint16_t* out = (uint16_t*)act_bo.map();
        printf("Output[0..7]: ");
        for (int i = 0; i < 8; i++) printf("%.3f ", bf16_to_float(out[i]));
        printf("\n");
        
        int diff = 0;
        for (int i = 0; i < 256; i++) if (out[i] != float_to_bf16((float)(i % 100 + 1))) diff++;
        printf("Non-identity: %d/256\n", diff);
    }
    
    printf("\nDONE\n");
    return 0;
}
