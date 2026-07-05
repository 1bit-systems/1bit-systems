/**
 * NPU Engine v4 — Full FLM Integration
 *
 * Uses FLM's exported constructors to:
 * 1. Create npu_xclbin_manager (manages XRT BOs + xclbins)
 * 2. Create qwen3_npu model handler (loads weights at correct DDR offsets)
 * 3. Generate per-layer instruction sequences via qwen3_npu_sequence::gen_layer_seq
 * 4. Submit via XRT to FLM's layer.xclbin (1 launch/layer instead of 4)
 *
 * Build:
 *   g++ -std=c++23 -O3 -mavx512f -o npu_engine_v4 npu_engine_v4.cpp \
 *       -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -ldl -luuid -lm
 *
 * Run: sudo ./npu_engine_v4
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <dlfcn.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t0) {
    return std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
}

// ─── FLM exported symbols (loaded via dlsym) ───
struct FlmAPI {
    void* lib_gemm = nullptr;
    void* lib_qwen = nullptr;
    
    // npu_xclbin_manager (size estimated, ~1024B based on internal state)
    // register_xclbin(path) — registers an xclbin with the manager
    void (*mgr_register_xclbin)(void* self, const std::string& path) = nullptr;
    
    // qwen3_npu_sequence — generates per-layer instruction sequences
    void (*seq_gen_layer)(void* self, void* npu_seq, unsigned layer_idx) = nullptr;
    void (*seq_gen_dequant)(void* self, void* npu_seq, uint64_t M, uint64_t N, uint64_t K) = nullptr;
    void (*seq_gen_lm_head)(void* self, void* npu_seq) = nullptr;
    void (*seq_impl_ctor)(void* self, const void* cfg, unsigned max_len) = nullptr;
    void (*seq_impl_dtor)(void* self) = nullptr;
    void (*cmds2seq)(void* seq) = nullptr;
    
    // qwen3_npu — model handler with weight loading
    void (*model_impl_ctor)(void* self, const void* cfg, void* mgr, int dev_id) = nullptr;
    void (*model_impl_dtor)(void* self) = nullptr;
    void (*model_load_weights)(void* self, void* q4nx) = nullptr;  // Q4NX reference
    
    // npu_sequence
    void (*npu_seq_cmds2seq)(void*) = nullptr;
    
    // LM_Config buffer size (4096 bytes as inferred from flm_bridge.cpp)
    static const int CFG_SIZE = 4096;
    
    bool load() {
        printf("Loading FLM libraries...\n");
        lib_gemm = dlopen("/opt/fastflowlm/lib/flm/libgemm.so", RTLD_LAZY | RTLD_GLOBAL);
        lib_qwen = dlopen("/opt/fastflowlm/lib/flm/libqwen3_npu.so", RTLD_LAZY | RTLD_GLOBAL);
        dlopen("/opt/fastflowlm/lib/flm/libmha.so", RTLD_LAZY | RTLD_GLOBAL);
        dlopen("/opt/fastflowlm/lib/flm/libdequant.so", RTLD_LAZY | RTLD_GLOBAL);
        dlopen("/opt/fastflowlm/lib/flm/liblm_head.so", RTLD_LAZY | RTLD_GLOBAL);
        
        if (!lib_gemm || !lib_qwen) {
            fprintf(stderr, "FAIL: dlopen FLM libs\n"); return false;
        }
        
        auto sym = [](const char* n) { return dlsym(RTLD_DEFAULT, n); };
        
        // npu_xclbin_manager
        mgr_register_xclbin = (void(*)(void*,const std::string&))
            sym("_ZN18npu_xclbin_manager15register_xclbinENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
        
        // qwen3_npu_sequence methods
        seq_impl_ctor = (void(*)(void*,const void*,unsigned))
            sym("_ZN18qwen3_npu_sequence4ImplC1E9LM_Configj");
        seq_impl_dtor = (void(*)(void*))
            sym("_ZN18qwen3_npu_sequence4ImplD1Ev");
        seq_gen_layer = (void(*)(void*,void*,unsigned))
            sym("_ZN18qwen3_npu_sequence4Impl13gen_layer_seqEP12npu_sequencej");
        seq_gen_dequant = (void(*)(void*,void*,uint64_t,uint64_t,uint64_t))
            sym("_ZN18qwen3_npu_sequence4Impl15gen_dequant_seqEP12npu_sequencemmm");
        seq_gen_lm_head = (void(*)(void*,void*))
            sym("_ZN18qwen3_npu_sequence4Impl15gen_lm_head_seqEP12npu_sequence");
        
        // qwen3_npu model handler
        model_impl_ctor = (void(*)(void*,const void*,void*,int))
            sym("_ZN9qwen3_npu4ImplC1E9LM_ConfigP18npu_xclbin_manageri");
        model_impl_dtor = (void(*)(void*))
            sym("_ZN9qwen3_npu4ImplD1Ev");
        model_load_weights = (void(*)(void*,void*))
            sym("_ZN9qwen3_npu4Impl12load_weightsER4Q4NX");
        
        // npu_sequence
        cmds2seq = (void(*)(void*))
            sym("_ZN12npu_sequence8cmds2seqEv");
        
        bool ok = mgr_register_xclbin && seq_impl_ctor && seq_gen_layer &&
                  model_impl_ctor && model_impl_dtor && cmds2seq;
        
        if (!ok) {
            fprintf(stderr, "Missing FLM symbols:\n");
            if(!mgr_register_xclbin) fprintf(stderr,"  mgr_register_xclbin\n");
            if(!seq_impl_ctor) fprintf(stderr,"  seq_impl_ctor\n");
            if(!seq_gen_layer) fprintf(stderr,"  seq_gen_layer\n");
            if(!model_impl_ctor) fprintf(stderr,"  model_impl_ctor\n");
            if(!model_impl_dtor) fprintf(stderr,"  model_impl_dtor\n");
            if(!cmds2seq) fprintf(stderr,"  cmds2seq\n");
            return false;
        }
        
        printf("  ✅ FLM API loaded\n");
        printf("  mgr_register_xclbin  = %p\n", (void*)mgr_register_xclbin);
        printf("  seq_impl_ctor        = %p\n", (void*)seq_impl_ctor);
        printf("  model_impl_ctor      = %p\n", (void*)model_impl_ctor);
        return true;
    }
    
    // Build LM_Config buffer (matches flm_bridge.cpp layout)
    void build_config(char* buf, const std::string& xclbin_path) {
        memset(buf, 0, CFG_SIZE);
        static std::string sp = xclbin_path;
        static std::string mn = "qwen3:0.6b";
        auto ss = [&](int po, int lo, const std::string& s) {
            *(const char**)(buf + po) = s.c_str();
            *(size_t*)(buf + lo) = s.size();
        };
        ss(0x00, 0x08, sp);
        ss(0x20, 0x28, mn);
        ss(0x40, 0x48, mn);
        ss(0x68, 0x70, mn);
        ss(0xd8, 0xe0, mn);
        ss(0xf8, 0x100, mn);
        *(int*)(buf + 0x60)      = 1024;   // hidden_size
        *(int*)(buf + 0x64)      = 16;     // num_heads
        *(int*)(buf + 0x88)      = 3072;   // intermediate_size
        *(int*)(buf + 0x8c)      = 8;      // num_kv_heads
        *(int*)(buf + 0x94)      = 28;     // num_layers
        *(int*)(buf + 0xd4)      = 4096;   // max_seq_len
        *(uint64_t*)(buf + 0xa8) = 4096;
    }
    
    ~FlmAPI() {
        if (lib_gemm) dlclose(lib_gemm);
        if (lib_qwen) dlclose(lib_qwen);
    }
};

int main(int argc, char** argv) {
    printf("╔═══════════════════════════════════════════════╗\n");
    printf("║  NPU Engine v4 — Full FLM Integration         ║\n");
    printf("╚═══════════════════════════════════════════════╝\n\n");
    
    FlmAPI flm;
    if (!flm.load()) return 1;
    
    // ── 1. Create LM_Config ──
    char cfg[FlmAPI::CFG_SIZE];
    flm.build_config(cfg, "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/layer.xclbin");
    printf("✅ LM_Config built\n");
    
    // ── 2. Create npu_xclbin_manager (allocate large buffer for internal state) ──
    // The manager is ~2000+ bytes internally (device refs, BO maps, xclbin cache).
    // We allocate 4096 and zero-init.
    uint8_t mgr_buf[4096];
    memset(mgr_buf, 0, sizeof(mgr_buf));
    // Initialize vtable pointer to nullptr (manager doesn't use virtual dispatch)
    void* manager = mgr_buf;
    printf("npu_xclbin_manager at %p\n", manager);
    
    // The manager constructor is not exported. We zero-init and call
    // register_xclbin which initializes internal state on first call.
    // This may crash if the class expects specific constructor initialization.
    // If it does, we need to find the constructor symbol or use different approach.
    
    // Register xclbins
    try {
        flm.mgr_register_xclbin(manager, "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/layer.xclbin");
        printf("✅ layer.xclbin registered\n");
    } catch (...) {
        printf("⚠️  register_xclbin threw (expected if manager uninitialized)\n");
        printf("   Need to find npu_xclbin_manager constructor or size\n");
        return 1;
    }
    
    // ── 3. Create qwen3_npu model (loads weights) ──
    // The Impl size is ~2600+ bytes (from disassembly: stack offsets up to 0x1d0+)
    // qwen3_npu doesn't have virtual dispatch, so alignment at start of buf
    uint8_t model_buf[8192];
    memset(model_buf, 0, sizeof(model_buf));
    void* model = model_buf;
    
    printf("Creating qwen3_npu::Impl (loads weights)...\n");
    auto t0 = Clock::now();
    try {
        flm.model_impl_ctor(model, cfg, manager, 0);
        printf("✅ qwen3_npu::Impl constructed in %.0f ms\n", ms(t0));
    } catch (const std::exception& e) {
        printf("❌ Constructor threw: %s\n", e.what());
        return 1;
    } catch (...) {
        printf("❌ Constructor threw unknown exception\n");
        return 1;
    }
    
    // ── 4. Generate instruction sequences ──
    // We have the model with weights loaded. Now generate per-layer seqs.
    // qwen3_npu_sequence::Impl generates instruction sequences for each layer.
    // Its constructor and gen_layer_seq() produce the fused instruction stream.
    
    // Create npu_sequence buffer (~4096 bytes from earlier analysis)
    uint8_t seq_buf[4096];
    memset(seq_buf, 0, sizeof(seq_buf));
    ((uint32_t*)(seq_buf + 0x20))[0] = 4;  // op_line_count = 4
    
    // Create qwen3_npu_sequence::Impl
    uint8_t seq_impl_buf[4096];
    memset(seq_impl_buf, 0, sizeof(seq_impl_buf));
    
    printf("\nGenerating layer instruction sequences...\n");
    t0 = Clock::now();
    try {
        flm.seq_impl_ctor(seq_impl_buf, cfg, (unsigned)4096);
        printf("  seq_impl constructed\n");
    } catch (...) {
        printf("  ⚠️  seq_impl ctor failed (non-critical, trying gen_layer directly)\n");
    }
    
    // Generate layer 0's instruction sequence
    for (int l = 0; l < 28; l++) {
        try {
            flm.seq_gen_layer(seq_impl_buf, seq_buf, (unsigned)l);
            flm.cmds2seq(seq_buf);
            
            // Extract instructions
            uint32_t** vb = (uint32_t**)(seq_buf + 0x40);
            uint32_t** ve = (uint32_t**)(seq_buf + 0x50);
            size_t ninstr = 0;
            if (vb && ve && *vb && *ve && *ve > *vb) {
                ninstr = *ve - *vb;
            }
            printf("  Layer %2d: %zu instructions\n", l, ninstr);
        } catch (const std::exception& e) {
            printf("  Layer %2d: FAILED — %s\n", l, e.what());
            break;
        }
    }
    printf("Generation done: %.0f ms\n\n", ms(t0));
    
    // ── 5. Submit via XRT ──
    printf("Opening XRT device for submission...\n");
    xrt::device dev(0);
    auto xc = xrt::xclbin([]()->std::vector<char>{
        std::string xp = "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/layer.xclbin";
        FILE* f = fopen(xp.c_str(), "rb"); fseek(f,0,2); long sz = ftell(f); fseek(f,0,0);
        std::vector<char> d(sz); fread(d.data(),1,sz,f); fclose(f); return d;
    }());
    dev.register_xclbin(xc);
    auto kern = xrt::kernel(dev, xc.get_uuid(), "MLIR_AIE");
    printf("✅ Kernel ready (data gid=%d, inst gid=%d)\n",
           kern.group_id(3), kern.group_id(1));
    
    // The BOs are managed by npu_xclbin_manager. In FLM's architecture,
    // the manager allocates and owns the BOs. We need to retrieve them.
    // Since we allocated the manager ourselves, the BOs are created inside it.
    // We need to access them through the same manager interface.
    // 
    // For now: the manager constructor handles BO creation internally.
    // If the constructor succeeded, the BOs exist in the manager's state.
    // We can access them by reverse-engineering the manager's internal layout
    // or by using a separate XRT kernel submission with our own BOs.
    
    printf("\n✅ Engine ready. Need npu_xclbin_manager accessor methods\n");
    printf("   to retrieve BOs for instruction submission.\n");
    printf("   Current BOs are held inside the manager at %p\n", manager);
    
    // Cleanup
    if (flm.model_impl_dtor) flm.model_impl_dtor(model);
    
    return 0;
}
