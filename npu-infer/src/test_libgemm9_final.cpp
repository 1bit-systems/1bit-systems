// test_libgemm9_final.cpp — Fixed sync calls + full libgemm + XRT integration
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <dlfcn.h>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/experimental/xrt_xclbin.h>

struct npu_sequence { uint8_t data[2048]; };
struct LM_Config { char dummy[0x200]; };
struct Gemm { void* impl; };

typedef void (*GemmCtor_t)(void*, LM_Config&);
typedef void (*GemmDtor_t)(void*);
typedef void (*GemmGenSeq_t)(void*, npu_sequence*, unsigned int, unsigned int, unsigned int, unsigned int, bool, unsigned int, unsigned int);

static uint16_t float_to_bf16(float v) {
    uint32_t bits; memcpy(&bits, &v, sizeof(bits));
    uint32_t bias = ((bits >> 16) & 1) + 0x7FFF;
    return (uint16_t)((bits + bias) >> 16);
}
static float bf16_to_float(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f; memcpy(&f, &bits, sizeof(f)); return f;
}

int main() {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    
    printf("=== libgemm.so + XRT Integration ===\n\n");
    
    // Step 1: Load libgemm.so
    void* gemm_lib = dlopen("/opt/fastflowlm/lib/flm/libgemm.so", RTLD_LAZY | RTLD_LOCAL);
    if (!gemm_lib) { printf("FAIL dlopen: %s\n", dlerror()); return 1; }
    auto gemm_ctor = (GemmCtor_t)dlsym(gemm_lib, "_ZN4GemmC1ER9LM_Config");
    auto gemm_dtor = (GemmDtor_t)dlsym(gemm_lib, "_ZN4GemmD1Ev");
    auto gemm_gen  = (GemmGenSeq_t)dlsym(gemm_lib, "_ZN4Gemm12generate_seqEP12npu_sequencejjjjbNS_17Activation_Type_tEj");
    printf("  libgemm.so loaded. gen=%p\n", (void*)gemm_gen);
    
    // Step 2: Create Gemm object
    Gemm gemm; memset(&gemm, 0, sizeof(gemm));
    LM_Config cfg; memset(&cfg, 0, sizeof(cfg));
    gemm_ctor(&gemm, cfg);
    printf("  Gemm created\n");
    
    // Step 3: Setup XRT
    xrt::device device;
    try { device = xrt::device(0); }
    catch (...) { try { device = xrt::device(1); } catch(...) { return 1; } }
    
    std::ifstream ifs("/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/mm.xclbin", std::ios::binary);
    std::vector<char> xclbin_data((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    xrt::xclbin xclbin(xclbin_data);
    device.register_xclbin(xclbin);
    xrt::kernel kernel(device, xclbin.get_uuid(), "MLIR_AIE");
    printf("  Kernel ready\n");
    
    // Step 4: Create BOs
    const size_t BO_SIZE = 4 * 1024 * 1024;
    const size_t INSTR_SIZE = 1024 * 1024;
    
    xrt::bo act_bo(device, BO_SIZE, xrt::bo::flags::host_only, (xrt::memory_group)0);
    xrt::bo ws_bo(device, BO_SIZE, xrt::bo::flags::host_only, (xrt::memory_group)0);
    xrt::bo w1_bo(device, BO_SIZE, xrt::bo::flags::host_only, (xrt::memory_group)0);
    xrt::bo w2_bo(device, BO_SIZE, xrt::bo::flags::host_only, (xrt::memory_group)0);
    xrt::bo kv_bo(device, BO_SIZE, xrt::bo::flags::host_only, (xrt::memory_group)0);
    xrt::bo instr_bo(device, INSTR_SIZE, xrt::bo::flags::host_only, (xrt::memory_group)1);
    printf("  BOs created (instr at 0x%llx)\n", (unsigned long long)instr_bo.address());
    
    // Step 5: Fill activation with known data
    uint16_t* act_map = (uint16_t*)act_bo.map();
    for (int i = 0; i < 1024; i++) act_map[i] = float_to_bf16((float)(i + 1));
    // Use sync(dir, sz, offset) with non-zero sz to avoid XRT bug
    act_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, 4096, 0);
    
    uint16_t* ws_map = (uint16_t*)ws_bo.map();
    memset(ws_map, 0, 4096);
    ws_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, 4096, 0);
    
    // Fill weight BOs
    uint16_t* w1_map = (uint16_t*)w1_bo.map();
    for (size_t i = 0; i < 1024 * 256; i++) w1_map[i] = float_to_bf16(0.001f * (float)(i % 100));
    w1_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, BO_SIZE, 0);
    
    memcpy(w2_bo.map(), w1_bo.map(), BO_SIZE);
    w2_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, BO_SIZE, 0);
    
    // Step 6: Generate instructions for each GEMM shape
    struct TestCase {
        const char* name;
        unsigned int M, K, N;
    };
    
    TestCase tests[] = {
        {"Qproj (256,1024,1024)", 256, 1024, 1024},
        {"Oproj (1024,1024,256)", 1024, 1024, 256},
        {"gate (256,1024,2048)", 256, 1024, 2048},
        {"up (256,1024,2048)", 256, 1024, 2048},
        {"down (2048,1024,256)", 2048, 1024, 256},
    };
    
    for (auto& t : tests) {
        printf("\n--- %s ---\n", t.name);
        
        npu_sequence seq;
        memset(&seq, 0, sizeof(seq));
        *(uint32_t*)(seq.data + 0x00) = 4;
        *(uint32_t*)(seq.data + 0x04) = 4;
        
        auto t0 = std::chrono::steady_clock::now();
        gemm_gen(&gemm, &seq, t.M, t.K, t.N, t.M, false, 3, 1);
        auto t1 = std::chrono::steady_clock::now();
        
        uint32_t* instr_data = *(uint32_t**)(seq.data + 0x38);
        uint32_t* instr_end  = *(uint32_t**)(seq.data + 0x40);
        size_t ninstr = (size_t)(instr_end - instr_data);
        
        if (ninstr == 0 || ninstr > 100000) {
            printf("  No valid instrs (ninstr=%zu)\n", ninstr);
            // Try vec@0x28 instead
            instr_data = *(uint32_t**)(seq.data + 0x28);
            instr_end  = *(uint32_t**)(seq.data + 0x30);
            ninstr = (size_t)(instr_end - instr_data);
            printf("  Trying vec@0x28: ninstr=%zu\n", ninstr);
            if (ninstr == 0 || ninstr > 100000) continue;
        }
        
        double gen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("  Gen: %zu instr (%zu KB) in %.3f ms\n", 
               ninstr, ninstr * 4 / 1024, gen_ms);
        
        // Show first few instruction words
        printf("  First 4: 0x%08x 0x%08x 0x%08x 0x%08x\n",
               instr_data[0], instr_data[1], instr_data[2], instr_data[3]);
        
        // Copy instructions to SRAM BO
        memcpy(instr_bo.map(), instr_data, ninstr * 4);
        instr_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, ninstr * 4, 0);
        
        // Run kernel
        auto t2 = std::chrono::steady_clock::now();
        try {
            auto run = kernel(
                (uint64_t)0,        // opcode = 0 (dynamic instructions)
                instr_bo,           // instr BO on SRAM bank
                (uint32_t)ninstr,   // number of instructions
                act_bo, ws_bo, w1_bo, w2_bo, kv_bo
            );
            run.wait();
            auto t3 = std::chrono::steady_clock::now();
            double exec_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
            
            // Read back results
            act_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, 128, 0);
            uint16_t* output = (uint16_t*)act_bo.map();
            
            printf("  Exec: %.3f ms\n", exec_ms);
            printf("  Out[0..7]: ");
            for (int i = 0; i < 8; i++) printf("%.3f ", bf16_to_float(output[i]));
            printf("\n");
            
            // Sum of first 1024 outputs
            double sum = 0;
            for (int i = 0; i < 1024; i++) sum += bf16_to_float(output[i]);
            printf("  Sum[0..1023]: %.3f\n", sum);
            
        } catch (const std::exception& e) {
            printf("  FAILED: %s\n", e.what());
        }
    }
    
    gemm_dtor(&gemm);
    dlclose(gemm_lib);
    printf("\nDONE\n");
    return 0;
}
