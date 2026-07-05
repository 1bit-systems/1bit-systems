/**
 * NPU Engine v4 — Fused Layer via FLM Instruction Generators
 *
 * Generates a single NPU instruction sequence per layer (QKV+O+GU+D)
 * using FLM's libgemm.so + libqwen3_npu.so instruction generators,
 * then submits via XRT to FLM's layer.xclbin.
 *
 * Eliminates 3/4 of XRT kernel launches vs per-GEMM approach.
 *
 * Build:
 *   g++ -std=c++23 -O3 -o npu_engine_fused npu_engine_fused.cpp \
 *       -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -ldl -luuid -lm
 *
 * Run (as root): sudo ./npu_engine_fused [M_batch]
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
static double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
}

// ─── FLM instruction generator ───
struct FlmGen {
    void* handle_ge = nullptr;
    void* handle_qw = nullptr;
    
    void (*Gemm_C1)(void*,void*) = nullptr;
    void (*Gemm_D1)(void*) = nullptr;
    void (*Gemm_generate_seq)(void*,void*,uint32_t,uint32_t,uint32_t,uint32_t,bool,int,uint32_t) = nullptr;
    void (*qwen3_gen_dequant)(void*,uint32_t,uint32_t,uint32_t) = nullptr;
    void (*qwen3_send_x)(void*,void*) = nullptr;
    void (*qwen3_move_weights)(void*,void*,uint32_t,uint32_t,uint32_t) = nullptr;
    void (*npu_seq_c1)(void*,void*,unsigned) = nullptr;
    void (*npu_seq_d1)(void*) = nullptr;
    void (*cmds2seq)(void*) = nullptr;
    
    bool ok = false;
    const int BATCH_PAD = 512;  // FLM requires M >= 512
    
    bool load() {
        printf("Loading FLM libraries...\n");
        handle_ge = dlopen("/opt/fastflowlm/lib/flm/libgemm.so", RTLD_LAZY | RTLD_GLOBAL);
        handle_qw = dlopen("/opt/fastflowlm/lib/flm/libqwen3_npu.so", RTLD_LAZY | RTLD_GLOBAL);
        dlopen("/opt/fastflowlm/lib/flm/libmha.so", RTLD_LAZY | RTLD_GLOBAL);
        dlopen("/opt/fastflowlm/lib/flm/libdequant.so", RTLD_LAZY | RTLD_GLOBAL);
        if (!handle_ge || !handle_qw) { fprintf(stderr, "FAIL: dlopen\n"); return false; }
        
        auto sym = [](const char* n) { return dlsym(RTLD_DEFAULT, n); };
        #define LOAD(n) do { \
            auto p = sym("_ZN" n); \
            if (!p) { fprintf(stderr, "  MISSING: " n "\n"); return false; } \
            *(void**)(&Gemm_C1) = p; /* first one gets type-cast trick */ \
        } while(0)
        // Use exact assignment per symbol
        Gemm_C1 = (void(*)(void*,void*))sym("_ZN4GemmC1ER9LM_Config");
        Gemm_D1 = (void(*)(void*))sym("_ZN4GemmD1Ev");
        Gemm_generate_seq = (void(*)(void*,void*,uint32_t,uint32_t,uint32_t,uint32_t,bool,int,uint32_t))
            sym("_ZN4Gemm12generate_seqEP12npu_sequencejjjjbNS_17Activation_Type_tEj");
        if (!Gemm_generate_seq)
            Gemm_generate_seq = (void(*)(void*,void*,uint32_t,uint32_t,uint32_t,uint32_t,bool,int,uint32_t))
                sym("_ZN4Gemm12generate_seqEP12npu_sequencejjjjbNS_17Activation_Type_tEjj");
        qwen3_gen_dequant = (void(*)(void*,uint32_t,uint32_t,uint32_t))
            sym("_ZN18qwen3_npu_sequence15gen_dequant_seqEP12npu_sequencemmm");
        qwen3_send_x = (void(*)(void*,void*))
            sym("_ZN18qwen3_npu_sequence4Impl7_send_xEP12npu_sequence");
        qwen3_move_weights = (void(*)(void*,void*,uint32_t,uint32_t,uint32_t))
            sym("_ZN18qwen3_npu_sequence4Impl13_move_weightsEP12npu_sequencemmm");
        npu_seq_c1 = (void(*)(void*,void*,unsigned))
            sym("_ZN18qwen3_npu_sequenceC1E9LM_Configj");
        npu_seq_d1 = (void(*)(void*))
            sym("_ZN18qwen3_npu_sequenceD1Ev");
        cmds2seq = (void(*)(void*))
            sym("_ZN12npu_sequence8cmds2seqEv");
        
        ok = Gemm_C1 && Gemm_D1 && Gemm_generate_seq && qwen3_gen_dequant &&
             qwen3_send_x && qwen3_move_weights && npu_seq_c1 && npu_seq_d1 && cmds2seq;
        if (!ok) { fprintf(stderr, "FAIL: missing symbols\n"); return false; }
        printf("  ✅ FLM libs loaded\n");
        return true;
    }
    
    // Build LM_Config buffer matching FLM's expected layout
    void build_config(char* buf, const char* xclbin_path) {
        memset(buf, 0, 4096);
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
    
    // Generate instruction sequence for one GEMM shape
    // Returns vector of instruction words
    std::vector<uint32_t> gen_gemm_instrs(int M, int N, int K) {
        if (!ok) return {};
        
        uint32_t actual_M = (uint32_t)(M < BATCH_PAD ? BATCH_PAD : M);
        
        char cfg[4096];
        build_config(cfg, "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/layer.xclbin");
        
        char gemm_buf[64] = {};
        Gemm_C1(gemm_buf, cfg);
        
        char seq_buf[4096] = {};
        npu_seq_c1(seq_buf, cfg, (unsigned)4096);
        
        void* impl = *(void**)seq_buf;
        if (!impl) { Gemm_D1(gemm_buf); npu_seq_d1(seq_buf); return {}; }
        
        // FLM pipeline
        qwen3_gen_dequant(seq_buf, actual_M, (uint32_t)N, (uint32_t)K);
        qwen3_send_x(impl, seq_buf);
        qwen3_move_weights(impl, seq_buf, actual_M, (uint32_t)N, (uint32_t)K);
        Gemm_generate_seq(gemm_buf, seq_buf, actual_M, (uint32_t)N, (uint32_t)K, actual_M, false, 3, 0);
        cmds2seq(seq_buf);
        
        // Extract instructions
        uint32_t** vb = (uint32_t**)(seq_buf + 0x40);
        uint32_t** ve = (uint32_t**)(seq_buf + 0x50);
        std::vector<uint32_t> result;
        if (vb && ve && *vb && *ve && *ve > *vb) {
            size_t cnt = *ve - *vb;
            if (cnt > 0 && cnt < 1000000) {
                result.assign(*vb, *ve);
            }
        }
        
        Gemm_D1(gemm_buf);
        npu_seq_d1(seq_buf);
        return result;
    }
    
    ~FlmGen() {
        if (handle_ge) dlclose(handle_ge);
        if (handle_qw) dlclose(handle_qw);
    }
};

// ─── NPU Engine ───
struct NpuEngine {
    xrt::device dev;
    xrt::kernel kern;
    xrt::xclbin xc;
    
    // BOs matching layer.xclbin arg layout:
    // group_id(3) = input activations (bo0)
    // group_id(5) = output activations (bo1)
    // group_id(4) = weights (bo2) + scratch (bo3)
    // group_id(1) = instructions (instr)
    xrt::bo bo_in, bo_out, bo_w, bo_scratch, bo_instr;
    
    // Per-layer instruction sequences
    std::vector<uint32_t> layer_instrs[28];
    
    FlmGen flm;
    
    bool init() {
        if (!flm.load()) return false;
        
        printf("Opening NPU device...\n");
        try { dev = xrt::device(0); }
        catch (const std::exception& e) { fprintf(stderr, "FAIL: device: %s\n", e.what()); return false; }
        
        printf("Loading layer.xclbin...\n");
        std::string xp = "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/layer.xclbin";
        FILE* f = fopen(xp.c_str(), "rb"); if (!f) { fprintf(stderr, "FAIL: open %s\n", xp.c_str()); return false; }
        fseek(f, 0, 2); long xsz = ftell(f); fseek(f, 0, 0);
        std::vector<char> xd(xsz); fread(xd.data(), 1, xsz, f); fclose(f);
        
        try {
            xc = xrt::xclbin(xd);
            dev.register_xclbin(xc);
            kern = xrt::kernel(dev, xc.get_uuid(), "MLIR_AIE");
        } catch (const std::exception& e) {
            fprintf(stderr, "FAIL: kernel: %s\n", e.what()); return false;
        }
        printf("  ✅ Kernel MLIR_AIE ready\n");
        
        // layer.xclbin DDR bank layout:
        //   group_id(1)=65537 (DDR bank 1, for instruction buffer)
        //   group_id(3)=65536 (DDR bank 0, for weight/activation BOs)
        // Instruction and data MUST be in SEPARATE banks!
        const size_t SZ_DATA = 1024 * 1024 * 1024ULL; // 1GB shared data (bank 0)
        const size_t SZ_INSTR = 512 * 1024;           // 512KB instructions (bank 1)
        
        printf("Allocating BOs...\n");
        int gid_data = kern.group_id(3);  // 65536 = DDR bank 0
        int gid_inst = kern.group_id(1);  // 65537 = DDR bank 1
        printf("  data group_id=%d, inst group_id=%d\n", gid_data, gid_inst);
        try {
            // All data BOs in the SAME DDR bank (bank 0). The FLM instructions
            // reference different regions within a single DDR contiguous space.
            // All 4 data BOs point to ONE shared buffer so FLM instructions
            // can address any DDR offset within it regardless of BO arg index.
            // XRT allocates one 1GB physically-contiguous region in DDR bank 0.
            xrt::bo bo_all(dev, SZ_DATA, XRT_BO_FLAGS_HOST_ONLY, gid_data);
            bo_in     = bo_all;
            bo_w      = bo_all;
            bo_out    = bo_all;
            bo_scratch= bo_all;
            // Instruction BO in SEPARATE bank (bank 1, cacheable)
            bo_instr  = xrt::bo(dev, SZ_INSTR, XCL_BO_FLAGS_CACHEABLE, gid_inst);
        } catch (const std::exception& e) {
            fprintf(stderr, "FAIL: BO: %s\n", e.what()); return false;
        }
        printf("  ✅ BOs allocated\n");
        return true;
    }
    
    // Generate fused instruction sequence for one layer
    bool gen_layer(int l, int M) {
        const int H = 1024, NH = 16, NKV = 8, HD = 128, IM = 3072;
        const int QOUT = NH * HD;       // 2048
        const int KVOUT = NKV * HD;     // 1024
        const int NQKV = QOUT + 2 * KVOUT; // 4096
        
        std::vector<uint32_t> all;
        
        auto add = [&](const char* name, int N, int K) {
            auto instrs = flm.gen_gemm_instrs(M, N, K);
            if (instrs.empty()) {
                fprintf(stderr, "  FAIL: %s(M=%d,N=%d,K=%d)\n", name, M, N, K);
                return false;
            }
            printf("  %-10s M=%d N=%5d K=%5d → %4zu instr\n", name, M, N, K, instrs.size());
            all.insert(all.end(), instrs.begin(), instrs.end());
            return true;
        };
        
        if (!add("QKV", NQKV, H)) return false;
        if (!add("O",   H,    NH*HD)) return false;
        if (!add("GU",  IM+IM, H)) return false;
        if (!add("D",   H,    IM)) return false;
        
        printf("  Layer %2d: %zu total instructions\n", l, all.size());
        layer_instrs[l] = std::move(all);
        return true;
    }
    
    bool gen_all_layers(int M) {
        printf("\nGenerating fused instruction sequences (M=%d)...\n\n", M);
        auto t0 = Clock::now();
        for (int l = 0; l < 28; l++) {
            if (!gen_layer(l, M)) return false;
        }
        printf("\nGeneration: %.0f ms\n", elapsed_ms(t0));
        return true;
    }
    
    // Submit one layer's instructions
    double submit_layer(int l) {
        auto& instrs = layer_instrs[l];
        if (instrs.empty()) return 0;
        
        memcpy(bo_instr.map(), instrs.data(), instrs.size() * 4);
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        
        auto t0 = Clock::now();
        auto run = kern((uint64_t)3, bo_instr, (unsigned)instrs.size(),
                        bo_in, bo_w, bo_out, bo_scratch);
        run.wait();
        return elapsed_ms(t0);
    }
    
    // Run all layers and return total time
    double run_all() {
        printf("\n=== Running all 28 layers (fused, 1 launch/layer) ===\n");
        auto t0 = Clock::now();
        for (int l = 0; l < 28; l++) {
            double ms = submit_layer(l);
            printf("  Layer %2d: %4zu instr, %7.3f ms\n", l, layer_instrs[l].size(), ms);
        }
        return elapsed_ms(t0);
    }
};

int main(int argc, char** argv) {
    printf("╔═══════════════════════════════════════════════╗\n");
    printf("║  NPU Engine v4 — Fused Layer (FLM generators) ║\n");
    printf("╚═══════════════════════════════════════════════╝\n\n");
    
    int M = 1;
    if (argc > 1) M = atoi(argv[1]);
    if (M < 1) M = 1;
    if (M > 128) M = 128;
    
    NpuEngine eng;
    if (!eng.init()) return 1;
    if (!eng.gen_all_layers(M)) return 1;
    
    double total = eng.run_all();
    printf("\n═══════════════════════════════════════════════\n");
    printf("  Total: %.1f ms (%d layers, M=%d)\n", total, 28, M);
    printf("  Per layer: %.3f ms\n", total / 28.0);
    printf("  Per token: %.3f ms → %.1f tok/s\n", total / M / 28.0 * 28, M * 28.0 / (total / 1000.0));
    printf("═══════════════════════════════════════════════\n");
    
    return 0;
}
