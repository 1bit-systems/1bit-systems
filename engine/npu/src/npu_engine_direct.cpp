/**
 * NPU Engine — Direct FLM Integration via npu_app_manager
 *
 * Uses FLM's weak npu_app_manager constructor directly to create
 * the NPU application context, then calls qwen3_npu to load weights.
 * After weight loading, BOs are accessed through the manager.
 *
 * Build:
 *   g++ -std=c++23 -O3 -o npu_engine_direct npu_engine_direct.cpp \
 *       -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -ldl -luuid -lm
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

// ─── Load FLM symbols ───
struct Flm {
    void* lib = nullptr;
    
    // npu_app_manager (weak, can call directly)
    void (*app_mgr_ctor)(void*, int, void*, const std::string&, bool) = nullptr;
    
    // qwen3_npu
    void (*model_ctor)(void*, const void*, void*, int) = nullptr;
    void (*model_dtor)(void*) = nullptr;
    
    // qwen3_npu_sequence
    void (*seq_gen_layer)(void*, void*, unsigned) = nullptr;
    void (*seq_gen_dequant)(void*, void*, uint64_t, uint64_t, uint64_t) = nullptr;
    void (*cmds2seq)(void*) = nullptr;
    
    // npu_xclbin_manager
    void (*mgr_register)(void*, const std::string&) = nullptr;
    
    bool load() {
        lib = dlopen("/opt/fastflowlm/lib/libqwen3_npu.so", RTLD_LAZY | RTLD_GLOBAL);
        dlopen("/opt/fastflowlm/lib/libgemm.so", RTLD_LAZY | RTLD_GLOBAL);
        dlopen("/opt/fastflowlm/lib/libmha.so", RTLD_LAZY | RTLD_GLOBAL);
        if (!lib) { fprintf(stderr, "FAIL: dlopen\n"); return false; }
        
        auto sym = [](const char* n) { return dlsym(RTLD_DEFAULT, n); };
        
        // npu_app_manager constructor (weak — direct call works!)
        // npu_app_manager(npu_device, xrt::device*, string, bool)
        // mangled: _ZN15npu_app_managerC1E10npu_devicePN3xrt6deviceENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEb
        app_mgr_ctor = (void(*)(void*,int,void*,const std::string&,bool))
            sym("_ZN15npu_app_managerC1E10npu_devicePN3xrt6deviceENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEb");
        
        // qwen3_npu::Impl::Impl(LM_Config, npu_xclbin_manager*, int)
        model_ctor = (void(*)(void*,const void*,void*,int))
            sym("_ZN9qwen3_npu4ImplC1E9LM_ConfigP18npu_xclbin_manageri");
        model_dtor = (void(*)(void*))
            sym("_ZN9qwen3_npu4ImplD1Ev");
        
        // qwen3_npu_sequence::gen_layer_seq(npu_sequence*, unsigned)
        seq_gen_layer = (void(*)(void*,void*,unsigned))
            sym("_ZN18qwen3_npu_sequence4Impl13gen_layer_seqEP12npu_sequencej");
        seq_gen_dequant = (void(*)(void*,void*,uint64_t,uint64_t,uint64_t))
            sym("_ZN18qwen3_npu_sequence4Impl15gen_dequant_seqEP12npu_sequencemmm");
        cmds2seq = (void(*)(void*))
            sym("_ZN12npu_sequence8cmds2seqEv");
        
        // npu_xclbin_manager::register_xclbin(string)
        mgr_register = (void(*)(void*,const std::string&))
            sym("_ZN18npu_xclbin_manager15register_xclbinENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
        
        if (!app_mgr_ctor) { fprintf(stderr, "FAIL: no app_mgr_ctor\n"); return false; }
        if (!model_ctor)   { fprintf(stderr, "FAIL: no model_ctor\n"); return false; }
        printf("  ✅ Symbols loaded\n");
        printf("  app_mgr_ctor=%p  model_ctor=%p  seq_gen=%p\n",
               (void*)app_mgr_ctor, (void*)model_ctor, (void*)seq_gen_layer);
        return true;
    }
    
    ~Flm() { if (lib) dlclose(lib); }
};

int main() {
    printf("╔═══════════════════════════════════════════════╗\n");
    printf("║  NPU Engine — Direct FLM Integration          ║\n");
    printf("╚═══════════════════════════════════════════════╝\n\n");
    
    Flm flm;
    if (!flm.load()) return 1;
    
    // ── 1. Open XRT device ──
    printf("Opening XRT device...\n");
    xrt::device dev(0);
    
    // ── 2. Create npu_app_manager ──
    // Size = 0x328 = 808 bytes from stack allocation
    // Allocate on heap to be safe
    uint8_t* mgr = new uint8_t[2048];
    memset(mgr, 0, 2048);
    
    printf("Creating npu_app_manager...\n");
    try {
        flm.app_mgr_ctor(mgr, 0, &dev, 
            "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/layer.xclbin",
            true);
        printf("  ✅ npu_app_manager created\n");
    } catch (const std::exception& e) {
        printf("  ❌ %s\n", e.what());
        delete[] mgr;
        return 1;
    }
    
    // ── 3. Register xclbin via npu_xclbin_manager wrapper ──
    // npu_xclbin_manager wraps npu_app_manager. We need to create it
    // properly for qwen3_npu to accept it. The npu_xclbin_manager
    // stores a pointer to npu_app_manager as its first internal field.
    // Allocate npu_xclbin_manager (size unknown, try 1024)
    uint8_t* xclbin_mgr = new uint8_t[4096];
    memset(xclbin_mgr, 0, 4096);
    
    // Store app_manager pointer at offset 0 of xclbin_mgr
    // npu_xclbin_manager has npu_app_manager* as first member
    *(void**)xclbin_mgr = mgr;
    
    printf("  xclbin_mgr at %p, app_mgr in first field\n", xclbin_mgr);
    
    // Register xclbin through the manager wrapper
    try {
        flm.mgr_register(xclbin_mgr,
            "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/layer.xclbin");
        printf("  ✅ xclbin registered\n");
    } catch (const std::exception& e) {
        printf("  ⚠️  register: %s (continuing)\n", e.what());
    }
    
    // ── 4. Create qwen3_npu model ──
    // qwen3_npu takes npu_xclbin_manager*, which we've set up above.
    uint8_t* model = new uint8_t[4096];
    memset(model, 0, 4096);
    
    // Build LM_Config
    char cfg[4096] = {};
    static std::string xp = "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/layer.xclbin";
    static std::string mn = "qwen3:0.6b";
    auto ss = [&](int po, int lo, const std::string& s) {
        *(const char**)(cfg + po) = s.c_str();
        *(size_t*)(cfg + lo) = s.size();
    };
    ss(0x00, 0x08, xp); ss(0x20, 0x28, mn); ss(0x40, 0x48, mn);
    ss(0x68, 0x70, mn); ss(0xd8, 0xe0, mn); ss(0xf8, 0x100, mn);
    *(int*)(cfg + 0x60) = 1024; *(int*)(cfg + 0x64) = 16;
    *(int*)(cfg + 0x88) = 3072; *(int*)(cfg + 0x8c) = 8;
    *(int*)(cfg + 0x94) = 28; *(int*)(cfg + 0xd4) = 4096;
    *(uint64_t*)(cfg + 0xa8) = 4096;
    
    printf("Creating model (loads weights)...\n");
    try {
        flm.model_ctor(model, cfg, mgr, 0);
        printf("  ✅ Model created with weights loaded\n");
    } catch (const std::exception& e) {
        printf("  ❌ %s\n", e.what());
        delete[] mgr; delete[] model;
        return 1;
    }
    
    printf("\n✅ Engine ready! Model at %p, manager at %p\n", model, mgr);
    printf("Next: extract BOs from manager, generate layer seqs, submit\n");
    
    // Cleanup
    if (flm.model_dtor) flm.model_dtor(model);
    delete[] mgr; delete[] model;
    return 0;
}
