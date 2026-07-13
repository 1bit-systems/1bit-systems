#ifndef NPU_INFER_ENGINE_H
#define NPU_INFER_ENGINE_H

#include "common.h"
#include "model.h"
#include <cstdint>
#include <vector>
#include <string>
#include <memory>

namespace xrt {
    class device;
    class kernel;
    class bo;
    class run;
    class hw_context;
    class xclbin;
}

struct NpuBo {
    std::unique_ptr<xrt::bo> bo;
    uint8_t* map = nullptr;
    size_t size = 0;
    std::string label;
    
    NpuBo() = default;
    ~NpuBo();
    NpuBo(NpuBo&& other) noexcept;
    NpuBo& operator=(NpuBo&& other) noexcept;
    NpuBo(const NpuBo&) = delete;
    NpuBo& operator=(const NpuBo&) = delete;
    
    bool create(xrt::device& device, size_t sz, const char* label_str = nullptr);
    void sync_to_device(size_t offset = 0, size_t sz = 0);
    void sync_from_device(size_t offset = 0, size_t sz = 0);
};

class WeightPacker {
public:
    WeightPacker(ModelWeights* mw, ModelConfig* config);
    ~WeightPacker();
    int num_bos(const TensorDesc* desc) const;
    size_t bo_size(const TensorDesc* desc) const;
    void pack_block(uint8_t* buffer, const TensorDesc* desc, int block_idx) const;
    int pack_to_bos(const TensorDesc* desc, NpuBo* bos, int max_bos, xrt::device& device) const;
private:
    ModelWeights* mw_;
    ModelConfig* config_;
};

class XclbinManager {
public:
    XclbinManager(xrt::device& device);
    ~XclbinManager();
    bool load(XclbinType type);
    xrt::kernel* kernel(XclbinType type);
private:
    xrt::device& device_;
    struct Entry {
        std::unique_ptr<xrt::xclbin> xclbin;
        std::unique_ptr<xrt::kernel> kernel;
        bool loaded = false;
    };
    Entry entries_[XCLBIN_COUNT];
};

struct WeightCacheLayer {
    std::vector<NpuBo> q_proj_blocks;
    std::vector<NpuBo> k_proj_blocks;
    std::vector<NpuBo> v_proj_blocks;
    std::vector<NpuBo> o_proj_blocks;
    std::vector<NpuBo> gate_proj_blocks;
    std::vector<NpuBo> up_proj_blocks;
    std::vector<NpuBo> down_proj_blocks;
};

class NpuInferenceEngine {
public:
    NpuInferenceEngine();
    ~NpuInferenceEngine();
    
    bool init(const char* model_path);
    int generate(const int* input_tokens, int num_input_tokens,
                 int* output_tokens, int max_output_tokens);
    
private:
    ModelConfig config_;
    ModelWeights* model_ = nullptr;
    std::unique_ptr<xrt::device> device_;
    std::unique_ptr<XclbinManager> xclbins_;
    std::unique_ptr<WeightPacker> packer_;
    
    struct HwCtxState {
        NpuBo act_bo;
        NpuBo act_workspace;
        NpuBo kv_cache;
        std::vector<WeightCacheLayer*>* weight_cache_ptr = nullptr; // pointer to shared cache
        int current_seq_len = 0;
        
        WeightCacheLayer& weight_cache(int layer) { 
            return *(*weight_cache_ptr)[layer]; 
        }
    };
    
    HwCtxState hwctx_[3];
    
    // Master weight cache (shared across all hwctx)
    std::vector<std::unique_ptr<WeightCacheLayer>> weight_cache_master_;
    std::vector<WeightCacheLayer*> weight_cache_ptrs_;
    
    NpuBo lm_head_bo_;
    std::vector<NpuBo> lm_head_blocks_;
    std::vector<float> lm_head_buffer_;
    int current_token_ = 0;
    
    bool cache_all_weights();
    bool pack_tensor_blocks(std::vector<NpuBo>& blocks, const TensorDesc* desc, const char* label_prefix);
    bool run_prefill(const int* input_tokens, int num_input_tokens);
    int run_decode_step(int last_token);
    void run_layer_mm(HwCtxState& ctx, int layer_idx);
    void run_layer_attn(HwCtxState& ctx, int layer_idx);
    void run_layer_mlp(HwCtxState& ctx, int layer_idx);
    void run_mm_blocked(xrt::kernel* kern, HwCtxState& ctx,
                        const std::vector<NpuBo>& weight_blocks,
                        int opcode = 3, int num_runs_limit = 5);
    int sample_token(const float* logits, int vocab_size, float temperature);
    void embed_lookup(int token, NpuBo& dest);
};

#endif // NPU_INFER_ENGINE_H
