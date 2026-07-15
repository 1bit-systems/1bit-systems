#ifndef FLM_BRIDGE_H
#define FLM_BRIDGE_H

#include <cstdint>
#include <vector>
#include <string>

/**
 * FlmBridge wraps FLM's instruction-generation functions via dlopen.
 *
 * Key insight: FLM generates instructions at M=512 (padded batch),
 * even for single-token decode. The real data is at row 0, 
 * rows 1-511 are padding.
 *
 * Usage:
 *   auto& bridge = FlmBridge::instance();
 *   if (!bridge.init(cfg)) { ... }
 *   
 *   // Generate GEMM instructions for a projection
 *   // M=512 (padded), N=output_dim, K=1024 (hidden)
 *   auto instrs = bridge.gen_gemm_instrs(512, 1024, 1024);  // q_proj
 *   auto instrs2 = bridge.gen_gemm_instrs(512, 3072, 1024); // gate_proj
 *   
 *   // Submit via XRT:
 *   bo_instr.write(instrs.data());
 *   kernel(3, bo_instr, instrs.size()*4, act_bo, weight_bo, out_bo, kv_bo);
 */
class FlmBridge {
public:
    struct Config {
        int hidden_size = 1024;
        int num_heads = 16;
        int num_kv_heads = 8;
        int num_layers = 28;
        int intermediate_size = 3072;
        int max_seq_len = 40960;
        int vocab_size = 151936;
        int head_dim = 128;
        int batch_padded = 512;
        std::string xclbin_path = "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2";
    };

    static FlmBridge& instance();
    
    bool init(const Config& cfg);
    bool initialized() const { return initialized_; }
    const Config& config() const { return cfg_; }
    
    // Generate a complete GEMM pipeline instruction sequence
    // Calls: send_rope_rms → send_rms → gen_dequant → send_x → move_weights → generate_seq → cmds2seq
    std::vector<uint32_t> gen_gemm_instrs(uint32_t M, uint32_t N, uint32_t K,
                                           int activation_type = 0);
    
    // Convenience: generate instructions for specific projection types
    std::vector<uint32_t> gen_qkv_instrs(uint32_t N) { 
        return gen_gemm_instrs(cfg_.batch_padded, N, cfg_.hidden_size); 
    }
    std::vector<uint32_t> gen_o_instrs() { 
        return gen_gemm_instrs(cfg_.batch_padded, cfg_.hidden_size, cfg_.hidden_size); 
    }
    std::vector<uint32_t> gen_gate_up_instrs() { 
        return gen_gemm_instrs(cfg_.batch_padded, cfg_.intermediate_size, cfg_.hidden_size, 
                               1); // Activation_Type_t::SILU
    }
    std::vector<uint32_t> gen_down_instrs() { 
        return gen_gemm_instrs(cfg_.batch_padded, cfg_.hidden_size, cfg_.intermediate_size); 
    }
    std::vector<uint32_t> gen_lm_head_instrs() { 
        return gen_gemm_instrs(cfg_.batch_padded, cfg_.vocab_size, cfg_.hidden_size); 
    }

private:
    FlmBridge() = default;
    ~FlmBridge();
    FlmBridge(const FlmBridge&) = delete;
    FlmBridge& operator=(const FlmBridge&) = delete;
    
    bool initialized_ = false;
    Config cfg_;
    
    // DL open handles
    void* handle_qw_ = nullptr;
    void* handle_ge_ = nullptr;
    
    // Function pointers
    void (*qwen3_npu_seq_c1_)(void*, void*, unsigned) = nullptr;
    void (*qwen3_npu_seq_d1_)(void*) = nullptr;
    void* (*gemm_c1_)(void*, void*) = nullptr;
    void (*gemm_d1_)(void*) = nullptr;
    void (*send_rope_rms_)(void*, void*) = nullptr;
    void (*send_rms_)(void*, void*) = nullptr;
    void (*gen_dequant_)(void*, unsigned, unsigned, unsigned) = nullptr;
    void (*send_x_)(void*, void*) = nullptr;
    void (*move_weights_)(void*, void*, unsigned, unsigned, unsigned) = nullptr;
    void (*gemm_generate_seq_)(void*, void*, unsigned, unsigned, unsigned,
                                unsigned, bool, int, unsigned) = nullptr;
    void (*cmds2seq_)(void*) = nullptr;
};

#endif // FLM_BRIDGE_H
