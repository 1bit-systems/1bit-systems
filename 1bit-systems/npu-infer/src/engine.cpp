#include "engine.h"
#include "common.h"
#include "model.h"

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_xclbin.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>

// ========= BF16 helpers =========
static inline float bf16_to_float_cpp(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}
static inline uint16_t float_to_bf16_cpp(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    return (uint16_t)((bits + rounding_bias) >> 16);
}

// ========= NpuBo =========
NpuBo::~NpuBo() {}
NpuBo::NpuBo(NpuBo&& other) noexcept 
    : bo(std::move(other.bo)), map(other.map), size(other.size), label(std::move(other.label)) {
    other.map = nullptr; other.size = 0;
}
NpuBo& NpuBo::operator=(NpuBo&& other) noexcept {
    bo = std::move(other.bo); map = other.map; size = other.size; label = std::move(other.label);
    other.map = nullptr; other.size = 0; return *this;
}
bool NpuBo::create(xrt::device& device, size_t sz, const char* label_str) {
    try {
        bo = std::make_unique<xrt::bo>(device, sz, xrt::bo::flags::host_only, 0);
        map = (uint8_t*)bo->map(); size = sz;
        if (label_str) label = label_str;
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("BO create failed (%s): %s", label_str ? label_str : "?", e.what());
        return false;
    }
}
void NpuBo::sync_to_device(size_t offset, size_t sz) {
    if (sz == 0) sz = size - offset;
    bo->sync(XCL_BO_SYNC_BO_TO_DEVICE, sz, offset);
}
void NpuBo::sync_from_device(size_t offset, size_t sz) {
    if (sz == 0) sz = size - offset;
    bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE, sz, offset);
}

// ========= WeightPacker =========
WeightPacker::WeightPacker(ModelWeights* mw, ModelConfig* config) : mw_(mw), config_(config) {}
WeightPacker::~WeightPacker() {}
int WeightPacker::num_bos(const TensorDesc* desc) const { return npu_weight_num_blocks(desc, config_); }
size_t WeightPacker::bo_size(const TensorDesc* desc) const { (void)desc; return config_->npu_weight_bo_size; }
void WeightPacker::pack_block(uint8_t* buffer, const TensorDesc* desc, int block_idx) const {
    void* data = model_tensor_data(mw_, const_cast<TensorDesc*>(desc));
    npu_pack_weight_bo(buffer, data, desc, config_, block_idx);
}
int WeightPacker::pack_to_bos(const TensorDesc* desc, NpuBo* bos, int max_bos, xrt::device& device) const {
    int n = num_bos(desc); if (n > max_bos) n = max_bos;
    size_t bsize = bo_size(desc);
    for (int i = 0; i < n; i++) {
        char label[128]; snprintf(label, sizeof(label), "%s_b%d", desc->name, i);
        if (!bos[i].create(device, bsize, label)) return i;
        pack_block(bos[i].map, desc, i); bos[i].sync_to_device();
    }
    return n;
}

// ========= XclbinManager =========
XclbinManager::XclbinManager(xrt::device& device) : device_(device) {}
XclbinManager::~XclbinManager() {}
bool XclbinManager::load(XclbinType type) {
    if (type < 0 || type >= XCLBIN_COUNT) return false;
    Entry& e = entries_[type]; if (e.loaded) return true;
    const char* path = XCLBIN_PATHS[type];
    LOG_DEBUG("Loading %s ...", path);
    FILE* f = fopen(path, "rb"); if (!f) { LOG_ERROR("Cannot open %s", path); return false; }
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> raw_data(fsize);
    size_t br = fread(raw_data.data(), 1, fsize, f); (void)br; fclose(f);
    try {
        auto xclbin = std::make_unique<xrt::xclbin>(raw_data);
        device_.register_xclbin(*xclbin);
        auto kernel = std::make_unique<xrt::kernel>(device_, xclbin->get_uuid(), "MLIR_AIE");
        e.xclbin = std::move(xclbin); e.kernel = std::move(kernel); e.loaded = true;
        LOG_INFO("Loaded %s", path); return true;
    } catch (const std::exception& ex) {
        LOG_ERROR("Failed to load %s: %s", path, ex.what()); return false;
    }
}
xrt::kernel* XclbinManager::kernel(XclbinType type) {
    if (type < 0 || type >= XCLBIN_COUNT) return nullptr;
    return entries_[type].kernel.get();
}

// ========= NpuInferenceEngine =========
NpuInferenceEngine::NpuInferenceEngine() {}
NpuInferenceEngine::~NpuInferenceEngine() { model_free(model_); }

bool NpuInferenceEngine::pack_tensor_blocks(std::vector<NpuBo>& blocks, 
                                             const TensorDesc* desc,
                                             const char* label_prefix) {
    int n = packer_->num_bos(desc);
    blocks.resize(n);
    for (int i = 0; i < n; i++) {
        char label[128];
        snprintf(label, sizeof(label), "%s_b%d", label_prefix, i);
        if (!blocks[i].create(*device_, packer_->bo_size(desc), label)) return false;
        packer_->pack_block(blocks[i].map, desc, i);
        blocks[i].sync_to_device();
    }
    return true;
}

bool NpuInferenceEngine::cache_all_weights() {
    LOG_INFO("=== Pre-packing all weights into BOs ===");
    auto t0 = std::chrono::steady_clock::now();
    
    weight_cache_master_.resize(config_.num_layers);
    for (int l = 0; l < config_.num_layers; l++) {
        weight_cache_master_[l] = std::make_unique<WeightCacheLayer>();
    }
    
    weight_cache_ptrs_.resize(config_.num_layers);
    for (int l = 0; l < config_.num_layers; l++)
        weight_cache_ptrs_[l] = weight_cache_master_[l].get();
    for (int c = 0; c < 3; c++)
        hwctx_[c].weight_cache_ptr = &weight_cache_ptrs_;
    
    for (int l = 0; l < config_.num_layers; l++) {
        LayerWeights* lw = &model_->layers[l];
        WeightCacheLayer* cache = weight_cache_master_[l].get();
        
        if (!pack_tensor_blocks(cache->q_proj_blocks, &lw->q_proj_weight, "q_proj")) return false;
        if (!pack_tensor_blocks(cache->k_proj_blocks, &lw->k_proj_weight, "k_proj")) return false;
        if (!pack_tensor_blocks(cache->v_proj_blocks, &lw->v_proj_weight, "v_proj")) return false;
        if (!pack_tensor_blocks(cache->o_proj_blocks, &lw->o_proj_weight, "o_proj")) return false;
        if (!pack_tensor_blocks(cache->gate_proj_blocks, &lw->gate_proj_weight, "gate_proj")) return false;
        if (!pack_tensor_blocks(cache->up_proj_blocks, &lw->up_proj_weight, "up_proj")) return false;
        if (!pack_tensor_blocks(cache->down_proj_blocks, &lw->down_proj_weight, "down_proj")) return false;
        
        if (l % 10 == 0) LOG_DEBUG("  Layer %d weights cached", l);
    }
    
    TensorDesc* lm_head = &model_->lm_head_weight;
    if (!pack_tensor_blocks(lm_head_blocks_, lm_head, "lm_head")) return false;
    
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    int total_bos = 0;
    for (auto& c : weight_cache_master_) {
        total_bos += c->q_proj_blocks.size() + c->k_proj_blocks.size() + c->v_proj_blocks.size()
                   + c->o_proj_blocks.size() + c->gate_proj_blocks.size() + c->up_proj_blocks.size()
                   + c->down_proj_blocks.size();
    }
    total_bos += lm_head_blocks_.size();
    
    LOG_INFO("All weights cached: %d layers, ~%d BOs in %.0f ms",
             config_.num_layers, total_bos, ms);
    return true;
}

bool NpuInferenceEngine::init(const char* model_path) {
    LOG_INFO("=== NPU Inference Engine Init ===");
    config_ = QWEN3_0_6B_CONFIG;
    
    model_ = model_load(model_path, config_);
    if (!model_) return false;
    
    device_ = std::make_unique<xrt::device>();
    try { *device_ = xrt::device(0); }
    catch (...) {
        try { *device_ = xrt::device(1); }
        catch (const std::exception& e) { LOG_ERROR("Cannot open NPU: %s", e.what()); return false; }
    }
    
    xclbins_ = std::make_unique<XclbinManager>(*device_);
    for (int i = 0; i < XCLBIN_COUNT; i++)
        if (!xclbins_->load((XclbinType)i)) return false;
    
    packer_ = std::make_unique<WeightPacker>(model_, &config_);
    
    for (int c = 0; c < 3; c++) {
        hwctx_[c].current_seq_len = 0;
        if (!hwctx_[c].act_bo.create(*device_, config_.npu_activation_bo_size, "act")) return false;
        if (!hwctx_[c].act_workspace.create(*device_, 10485760, "workspace")) return false;
        if (!hwctx_[c].kv_cache.create(*device_, config_.npu_kv_cache_bo_size, "kv_cache")) return false;
        hwctx_[c].kv_cache.sync_to_device();
    }
    
    if (!cache_all_weights()) return false;
    
    int vocab_size = config_.vocab_size;
    lm_head_buffer_.resize(vocab_size);
    
    LOG_INFO("=== Init Complete ===");
    LOG_INFO("Model: %s (%d layers, %d hidden)",
             model_path, config_.num_layers, config_.hidden_size);
    return true;
}

// === Sequential GEMM ===
// Individual kernel call with wait
static void run_gemm(xrt::kernel* kern, xrt::bo& act, xrt::bo& ws,
                      xrt::bo& w1, xrt::bo& w2, xrt::bo& kv) {
    auto r = (*kern)(
        (uint64_t)3,
        (uint64_t)0,
        (uint32_t)0,
        act, ws, w1, w2, kv
    );
    r.wait();
}

static void run_blocked_gemm(xrt::kernel* kern, xrt::bo& act, xrt::bo& ws,
                              xrt::bo& kv, std::vector<NpuBo>& weights) {
    for (auto& w : weights) {
        run_gemm(kern, act, ws, *w.bo, *w.bo, kv);
    }
}

// === Layer MM pipeline ===
void NpuInferenceEngine::run_layer_mm(HwCtxState& ctx, int layer_idx) {
    WeightCacheLayer& wc = ctx.weight_cache(layer_idx);
    xrt::kernel* mm_kern = xclbins_->kernel(XCLBIN_MM);
    if (!mm_kern) return;
    xrt::bo& act = *ctx.act_bo.bo;
    xrt::bo& ws = *ctx.act_workspace.bo;
    xrt::bo& kv = *ctx.kv_cache.bo;
    run_blocked_gemm(mm_kern, act, ws, kv, wc.q_proj_blocks);
    run_blocked_gemm(mm_kern, act, ws, kv, wc.k_proj_blocks);
    run_blocked_gemm(mm_kern, act, ws, kv, wc.v_proj_blocks);
}

// === Attention pipeline ===
void NpuInferenceEngine::run_layer_attn(HwCtxState& ctx, int layer_idx) {
    WeightCacheLayer& wc = ctx.weight_cache(layer_idx);
    xrt::kernel* attn_kern = xclbins_->kernel(XCLBIN_ATTN);
    if (!attn_kern) return;
    xrt::bo& act = *ctx.act_bo.bo;
    xrt::bo& ws = *ctx.act_workspace.bo;
    xrt::bo& kv = *ctx.kv_cache.bo;
    xrt::bo& w = (!wc.o_proj_blocks.empty()) ? *wc.o_proj_blocks[0].bo : act;
    run_gemm(attn_kern, act, ws, w, w, kv);
    if (!wc.o_proj_blocks.empty()) {
        run_blocked_gemm(xclbins_->kernel(XCLBIN_MM), act, ws, kv, wc.o_proj_blocks);
    }
}

// === MLP pipeline ===
void NpuInferenceEngine::run_layer_mlp(HwCtxState& ctx, int layer_idx) {
    WeightCacheLayer& wc = ctx.weight_cache(layer_idx);
    xrt::kernel* mm_kern = xclbins_->kernel(XCLBIN_MM);
    if (!mm_kern) return;
    xrt::bo& act = *ctx.act_bo.bo;
    xrt::bo& ws = *ctx.act_workspace.bo;
    xrt::bo& kv = *ctx.kv_cache.bo;
    run_blocked_gemm(mm_kern, act, ws, kv, wc.gate_proj_blocks);
    run_blocked_gemm(mm_kern, act, ws, kv, wc.up_proj_blocks);
    run_blocked_gemm(mm_kern, act, ws, kv, wc.down_proj_blocks);
}

// === Prefill ===
bool NpuInferenceEngine::run_prefill(const int* input_tokens, int num_input_tokens) {
    LOG_INFO("=== Prefill %d tokens ===", num_input_tokens);
    if (num_input_tokens < 1) return false;
    
    for (int t = 0; t < num_input_tokens; t++) {
        embed_lookup(input_tokens[t], hwctx_[0].act_bo);
        
        for (int l = 0; l < config_.num_layers; l++) {
            run_layer_mm(hwctx_[0], l);
            run_layer_attn(hwctx_[1], l);
            run_layer_mlp(hwctx_[2], l);
            if (l % 7 == 0) LOG_DEBUG("  Layer %d done", l);
        }
    }
    
    LOG_INFO("Prefill complete");
    return true;
}

// === Decode ===
int NpuInferenceEngine::run_decode_step(int last_token) {
    embed_lookup(last_token, hwctx_[0].act_bo);
    
    for (int l = 0; l < config_.num_layers; l++) {
        run_layer_mm(hwctx_[0], l);
        run_layer_attn(hwctx_[1], l);
        run_layer_mlp(hwctx_[2], l);
    }
    
    // LM head
    xrt::kernel* mm_kern = xclbins_->kernel(XCLBIN_MM);
    if (mm_kern && !lm_head_blocks_.empty()) {
        xrt::bo& act = *hwctx_[0].act_bo.bo;
        xrt::bo& ws = *hwctx_[0].act_workspace.bo;
        xrt::bo& kv = *hwctx_[0].kv_cache.bo;
        for (auto& w : lm_head_blocks_) {
            run_gemm(mm_kern, act, ws, *w.bo, *w.bo, kv);
        }
    }
    
    // Read logits (first vocab_size BF16 values from activation BO)
    hwctx_[0].act_bo.sync_from_device(0, config_.vocab_size * 2);
    uint16_t* logits_bf16 = (uint16_t*)hwctx_[0].act_bo.map;
    
    for (int i = 0; i < config_.vocab_size; i++) {
        lm_head_buffer_[i] = bf16_to_float_cpp(logits_bf16[i]);
    }
    
    return sample_token(lm_head_buffer_.data(), config_.vocab_size, 1.0f);
}

// === Sample ===
int NpuInferenceEngine::sample_token(const float* logits, int vocab_size, float temperature) {
    (void)temperature;
    int max_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// === Embed lookup ===
void NpuInferenceEngine::embed_lookup(int token, NpuBo& dest) {
    TensorDesc* emb = &model_->embed_tokens;
    if (emb->ndim == 2 && token >= 0 && token < emb->shape[0]) {
        uint64_t offset = (uint64_t)token * emb->shape[1] * 2;
        size_t copy_size = emb->shape[1] * 2;
        if (copy_size > dest.size) copy_size = dest.size;
        memcpy(dest.map, model_->file_data + emb->data_offset + offset, copy_size);
        dest.sync_to_device(0, copy_size);
    }
}

// === Generate ===
int NpuInferenceEngine::generate(const int* input_tokens, int num_input_tokens,
                                  int* output_tokens, int max_output_tokens) {
    LOG_INFO("=== Generate ===");
    auto t_total = std::chrono::steady_clock::now();
    
    // Reset KV cache
    for (int c = 0; c < 3; c++) {
        memset(hwctx_[c].kv_cache.map, 0, hwctx_[c].kv_cache.size);
        hwctx_[c].kv_cache.sync_to_device();
        hwctx_[c].current_seq_len = 0;
    }
    
    if (!run_prefill(input_tokens, num_input_tokens)) {
        LOG_ERROR("Prefill failed");
        return 0;
    }
    
    current_token_ = input_tokens[num_input_tokens - 1];
    int num_out = 0;
    
    for (int i = 0; i < max_output_tokens; i++) {
        current_token_ = run_decode_step(current_token_);
        output_tokens[num_out++] = current_token_;
        
        if (current_token_ == 0) break;
    }
    
    auto t_end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_total).count();
    
    LOG_INFO("Generated %d tokens in %.0f ms (%.1f ms/tok)",
             num_out, total_ms, total_ms / (num_out > 0 ? num_out : 1));
    
    return num_out;
}
