#include "flm_bridge.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <new>

#define LOG_INFO(fmt, ...)  fprintf(stdout, "[FLM] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) fprintf(stderr, "[FLM] ERROR: " fmt "\n", ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) if (0) fprintf(stdout, "[FLM] " fmt "\n", ##__VA_ARGS__)

// ============================================================================
// Singleton
// ============================================================================

FlmBridge& FlmBridge::instance() {
    static FlmBridge bridge;
    return bridge;
}

FlmBridge::~FlmBridge() {
    if (handle_qw_) dlclose(handle_qw_);
    if (handle_ge_) dlclose(handle_ge_);
}

// ============================================================================
// Init
// ============================================================================

bool FlmBridge::init(const Config& cfg) {
    if (initialized_) return true;
    cfg_ = cfg;
    
    LOG_INFO("Loading FLM libraries...");
    
    handle_ge_ = dlopen("/opt/fastflowlm/lib/flm/libgemm.so", RTLD_LAZY | RTLD_GLOBAL);
    handle_qw_ = dlopen("/opt/fastflowlm/lib/flm/libqwen3_npu.so", RTLD_LAZY | RTLD_GLOBAL);
    dlopen("/opt/fastflowlm/lib/flm/libmha.so", RTLD_LAZY | RTLD_GLOBAL);
    dlopen("/opt/fastflowlm/lib/flm/libdequant.so", RTLD_LAZY | RTLD_GLOBAL);
    dlopen("/opt/fastflowlm/lib/flm/liblm_head.so", RTLD_LAZY | RTLD_GLOBAL);
    
    if (!handle_ge_) { LOG_ERROR("Cannot load libgemm.so: %s", dlerror()); return false; }
    if (!handle_qw_) { LOG_ERROR("Cannot load libqwen3_npu.so: %s", dlerror()); return false; }
    
    // Load all symbols from RTLD_DEFAULT (visible through RTLD_GLOBAL)
    auto sym = [](const char* name) {
        return dlsym(RTLD_DEFAULT, name);
    };
    
    qwen3_npu_seq_c1_ = (void(*)(void*, void*, unsigned))
        sym("_ZN18qwen3_npu_sequenceC1E9LM_Configj");
    qwen3_npu_seq_d1_ = (void(*)(void*))
        sym("_ZN18qwen3_npu_sequenceD1Ev");
    gemm_c1_ = (void* (*)(void*, void*))
        sym("_ZN4GemmC1ER9LM_Config");
    gemm_d1_ = (void(*)(void*))
        sym("_ZN4GemmD1Ev");
    send_rope_rms_ = (void(*)(void*, void*))
        sym("_ZN18qwen3_npu_sequence4Impl22_send_rope_rms_weightsEP12npu_sequence");
    send_rms_ = (void(*)(void*, void*))
        sym("_ZN18qwen3_npu_sequence4Impl17_send_rms_weightsEP12npu_sequence");
    gen_dequant_ = (void(*)(void*, unsigned, unsigned, unsigned))
        sym("_ZN18qwen3_npu_sequence15gen_dequant_seqEP12npu_sequencemmm");
    send_x_ = (void(*)(void*, void*))
        sym("_ZN18qwen3_npu_sequence4Impl7_send_xEP12npu_sequence");
    move_weights_ = (void(*)(void*, void*, unsigned, unsigned, unsigned))
        sym("_ZN18qwen3_npu_sequence4Impl13_move_weightsEP12npu_sequencemmm");
    gemm_generate_seq_ = (void(*)(void*, void*, unsigned, unsigned, unsigned,
                                   unsigned, bool, int, unsigned))
        sym("_ZN4Gemm12generate_seqEP12npu_sequencejjjjbNS_17Activation_Type_tEj");
    cmds2seq_ = (void(*)(void*))
        sym("_ZN12npu_sequence8cmds2seqEv");
    
    // Verify all loaded
    bool ok = qwen3_npu_seq_c1_ && qwen3_npu_seq_d1_ && gemm_c1_ && gemm_d1_ &&
              send_rope_rms_ && send_rms_ && gen_dequant_ && send_x_ &&
              move_weights_ && gemm_generate_seq_ && cmds2seq_;
    
    if (!ok) {
        LOG_ERROR("Missing FLM symbols:");
        #define CHECK(s, name) if (!s) LOG_ERROR("  " name)
        CHECK(qwen3_npu_seq_c1_, "qwen3_npu_sequence C1");
        CHECK(qwen3_npu_seq_d1_, "qwen3_npu_sequence D1");
        CHECK(gemm_c1_, "Gemm C1");
        CHECK(gemm_d1_, "Gemm D1");
        CHECK(send_rope_rms_, "send_rope_rms_weights");
        CHECK(send_rms_, "send_rms_weights");
        CHECK(gen_dequant_, "gen_dequant_seq");
        CHECK(send_x_, "send_x");
        CHECK(move_weights_, "move_weights");
        CHECK(gemm_generate_seq_, "generate_seq");
        CHECK(cmds2seq_, "cmds2seq");
        #undef CHECK
        return false;
    }
    
    LOG_INFO("FLM bridge initialized (M=%d, hidden=%d)",
             cfg_.batch_padded, cfg_.hidden_size);
    initialized_ = true;
    return true;
}

// ============================================================================
// Helper: create LM_Config buffer
// ============================================================================
// Layout from full_pipeline.cpp:
// String fields at offsets: 0x00/0x08, 0x20/0x28, 0x40/0x48, 0x68/0x70, 0xd8/0xe0, 0xf8/0x100
// Each is a (pointer, length) pair
// Numeric: 0x60=hidden, 0x64=heads, 0x88=intermediate, 0x8c=kv_heads, 0x94=layers, 0xd4=max_seq_len, 0xa8=max_seq_len(u64)

struct LmConfigBuf {
    char data[4096];
    
    void init(const FlmBridge::Config& cfg) {
        memset(data, 0, sizeof(data));
        
        // Static strings for LM_Config
        static std::string saved_path = cfg.xclbin_path;
        static std::string saved_name = "qwen3:0.6b";
        
        // String pointer/length pairs
        auto set_str = [this](int ptr_off, int len_off, const std::string& s) {
            *(const char**)(data + ptr_off) = s.c_str();
            *(size_t*)(data + len_off) = s.size();
        };
        
        set_str(0x00, 0x08, saved_path);
        set_str(0x20, 0x28, saved_name);
        set_str(0x40, 0x48, saved_name);
        set_str(0x68, 0x70, saved_name);
        set_str(0xd8, 0xe0, saved_name);
        set_str(0xf8, 0x100, saved_name);
        
        // Numeric fields
        *(int*)(data + 0x60)      = cfg.hidden_size;
        *(int*)(data + 0x64)      = cfg.num_heads;
        *(int*)(data + 0x88)      = cfg.intermediate_size;
        *(int*)(data + 0x8c)      = cfg.num_kv_heads;
        *(int*)(data + 0x94)      = cfg.num_layers;
        *(int*)(data + 0xd4)      = cfg.max_seq_len;
        *(uint64_t*)(data + 0xa8)  = (uint64_t)cfg.max_seq_len;
    }
};

// ============================================================================
// Generate GEMM pipeline instructions
// ============================================================================

std::vector<uint32_t> FlmBridge::gen_gemm_instrs(uint32_t M, uint32_t N, uint32_t K,
                                                   int activation_type) {
    if (!initialized_) return {};
    
    if (M < (uint32_t)cfg_.batch_padded && M < 512) {
        LOG_DEBUG("Padding M from %u to %u for FLM compatibility", M, cfg_.batch_padded);
        // FLM needs M >= 512 for _move_weights
        // We'll use cfg_.batch_padded instead
    }
    
    // Use the padded M if our M is too small
    uint32_t actual_M = M;
    if (actual_M < 512) {
        actual_M = cfg_.batch_padded;
    }
    
    // Create LM_Config
    LmConfigBuf cfg_buf;
    cfg_buf.init(cfg_);
    
    // Create Gemm object
    char gemm_buf[64] = {};
    gemm_c1_(gemm_buf, &cfg_buf.data);
    
    // Create sequence object
    char seq_buf[4096] = {};
    qwen3_npu_seq_c1_(seq_buf, &cfg_buf.data, (unsigned)cfg_.max_seq_len);
    
    auto *seq = (void*)seq_buf;
    auto *impl = *(void**)seq_buf;  // First field = Impl pointer
    
    if (!impl) {
        LOG_ERROR("Sequence constructor failed: Impl pointer is null");
        gemm_d1_(gemm_buf);
        qwen3_npu_seq_d1_(seq_buf);
        return {};
    }
    
    // Execute pipeline in FLM order (from full_pipeline.cpp)
    send_rope_rms_(impl, seq);
    send_rms_(impl, seq);
    gen_dequant_(seq, actual_M, N, K);
    send_x_(impl, seq);
    move_weights_(impl, seq, actual_M, N, K);
    gemm_generate_seq_(gemm_buf, seq, actual_M, N, K, actual_M, false, activation_type, 0);
    cmds2seq_(seq);
    
    // Extract instructions
    uint32_t** vb = (uint32_t**)(seq_buf + 0x40);
    uint32_t** ve = (uint32_t**)(seq_buf + 0x50);
    
    std::vector<uint32_t> result;
    if (vb && ve && *vb && *ve && *ve > *vb) {
        size_t count = *ve - *vb;
        if (count > 0 && count < 1000000) {
            result.assign(*vb, *ve);
        }
    }
    
    // Cleanup
    gemm_d1_(gemm_buf);
    qwen3_npu_seq_d1_(seq_buf);
    
    LOG_DEBUG("GEMM instrs (M=%u, N=%u, K=%u, act=%d): %zu",
              actual_M, N, K, activation_type, result.size());
    
    return result;
}
