// GGUF model loader — pure C++ + HIP, no Python.
// Reads GGUF format models and uploads weights to GPU for inference.
//
// Low-level GGUF parsing/dequantization lives in gguf_reader.h now (the
// single shared implementation every GGUF-consuming file in this codebase
// should use). This file keeps only what's specific to it: HIP upload,
// the BitNet/Halo-v2 tensor naming convention, and block-scaled-ternary
// (BST) detection — BST is a project-specific extension that reuses the
// real ggml_type numeric value 16 (which collides with GGML_TYPE_IQ2_XXS),
// so it deliberately isn't part of the shared module's real-dtype enum.
#include "rocm_cpp/bitnet_model.h"
#include "gguf_reader.h"
#include "block_scaled_ternary.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>

#define HIP_CHECK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP %d %s:%d\n",_s,__FILE__,__LINE__); return RCPP_HIP_ERROR;}} while(0)

namespace {
constexpr uint32_t BST_DTYPE = 16; // project-specific; collides with real GGML_TYPE_IQ2_XXS
} // namespace

extern "C" {

rcpp_status_t rcpp_bitnet_load_gguf(const char* path, rcpp_bitnet_model_t* out_model) {
    if (!path || !out_model) return RCPP_INVALID_ARG;
    memset(out_model, 0, sizeof(*out_model));
    GgufReader reader;
    if (!reader.open(path)) return RCPP_INVALID_ARG;
    fprintf(stderr, "[gguf] Loading: %s\n", path);

    auto gu = [&](const std::string& k, int def) -> int {
        uint32_t v;
        return reader.get_u32(k, v) ? (int)v : def;
    };

    int hidden_size = gu("llm.embedding_length", 2048);
    int n_layers = gu("llm.block_count", 40);
    int n_heads = gu("llm.attention.head_count", 8);
    int inter_size = gu("llm.feed_forward_length", 2048);
    int vocab_size = gu("llm.vocab_size", 262272);
    if (n_heads <= 0) { fprintf(stderr, "[gguf] invalid head_count=%d\n", n_heads); return RCPP_INVALID_ARG; }
    if (hidden_size <= 0) { fprintf(stderr, "[gguf] invalid hidden_size=%d\n", hidden_size); return RCPP_INVALID_ARG; }
    int head_dim = hidden_size / n_heads;

    out_model->hidden_size = hidden_size;
    out_model->intermediate_size = inter_size;
    out_model->num_layers = n_layers;
    out_model->num_heads = n_heads;
    out_model->vocab_size = vocab_size;

    bool has_bst_tensor = false;
    for (const auto& name : reader.tensor_names()) {
        const GgufTensorInfo* ti = reader.tensor_info(name);
        if (ti && ti->dtype == BST_DTYPE) { has_bst_tensor = true; break; }
    }
    out_model->weight_format = has_bst_tensor
        ? RCPP_WEIGHT_FORMAT_BLOCK_SCALED_TERNARY
        : RCPP_WEIGHT_FORMAT_HALO_V2;
    if (has_bst_tensor) out_model->flags |= H1B_FLAG_BLOCK_SCALED;
    std::string arch = reader.architecture();
    out_model->arch = rcpp_arch_from_string(arch.c_str());
    out_model->is_qwen3 = (out_model->arch == RCPP_ARCH_QWEN3) ? 1 : 0;

    const size_t MAX_EL = 1ULL << 34; // 16B elements ~64 GB

    // Reads a tensor to float32 (BST tensors use their own dequant on raw
    // block bytes; everything else goes through the shared reader).
    auto read_tensor_f32 = [&](const std::string& name, std::vector<float>& out) -> bool {
        const GgufTensorInfo* ti = reader.tensor_info(name);
        if (ti && ti->dtype == BST_DTYPE) {
            std::vector<uint8_t> raw;
            uint64_t numel = 0;
            if (!reader.get_tensor_raw(name, BST_BLOCK_K, BST_BLOCK_BYTES, raw, &numel)) return false;
            out.resize(numel);
            uint64_t n_blocks = (numel + BST_BLOCK_K - 1) / BST_BLOCK_K;
            for (uint64_t b = 0; b < n_blocks; b++) {
                uint64_t start = b * BST_BLOCK_K;
                uint64_t end = std::min<uint64_t>(start + BST_BLOCK_K, numel);
                block_scaled_ternary_dequant_row(raw.data() + b * BST_BLOCK_BYTES, out.data() + start, (int)(end - start));
            }
            return true;
        }
        return reader.get_tensor_f32(name, out);
    };

    {
        std::vector<float> emb;
        if (!read_tensor_f32("token_embd.weight", emb))
            read_tensor_f32("model.embed_tokens.weight", emb);
        if (emb.empty() || emb.size() > MAX_EL) { fprintf(stderr, "GGUF: invalid embedding size %zu\n", emb.size()); return RCPP_INVALID_ARG; }
        std::vector<_Float16> f16(emb.size());
        for (size_t i = 0; i < emb.size(); ++i) f16[i] = (_Float16)emb[i];
        HIP_CHECK(hipMalloc(&out_model->embedding_dev, f16.size() * sizeof(_Float16)));
        HIP_CHECK(hipMemcpy(out_model->embedding_dev, f16.data(), f16.size() * sizeof(_Float16), hipMemcpyHostToDevice));
    }
    {
        std::vector<float> fn;
        read_tensor_f32("output_norm.weight", fn);
        std::vector<_Float16> f16(fn.size());
        for (size_t i = 0; i < fn.size(); ++i) f16[i] = (_Float16)fn[i];
        HIP_CHECK(hipMalloc(&out_model->final_norm_weight_dev, f16.size() * sizeof(_Float16)));
        HIP_CHECK(hipMemcpy(out_model->final_norm_weight_dev, f16.data(), f16.size() * sizeof(_Float16), hipMemcpyHostToDevice));
    }
    out_model->layers = static_cast<rcpp_bitnet_layer_t*>(std::calloc(n_layers, sizeof(rcpp_bitnet_layer_t)));
    if (!out_model->layers) return RCPP_INTERNAL;
    for (int l = 0; l < n_layers; ++l) {
        auto& layer = out_model->layers[l];
        auto prefix = [&](const char* n) -> std::string {
            return std::string("model.layers.") + std::to_string(l) + "." + n;
        };
        // Select target pointers based on weight format
        // BST format writes to bst_*_packed_dev, standard writes to *_packed_dev
        auto lw = [&](const std::string& gn, void** std_ptr, void** bst_ptr, int r, int c) -> rcpp_status_t {
            std::vector<float> data;
            if (!read_tensor_f32(gn, data)) return RCPP_OK;
            size_t expected = (size_t)r * c;
            if (data.size() != expected) {
                fprintf(stderr, "GGUF: tensor %s dim mismatch: expected %dx%d=%zu, got %zu\n", gn.c_str(), r, c, expected, data.size());
                return RCPP_INVALID_ARG;
            }
            if (data.size() > MAX_EL) { fprintf(stderr, "GGUF: tensor %s too large (%zu el)\n", gn.c_str(), data.size()); return RCPP_INVALID_ARG; }
            std::vector<_Float16> f16(data.size());
            for (size_t i = 0; i < data.size(); ++i) f16[i] = (_Float16)data[i];
            void** target = has_bst_tensor ? bst_ptr : std_ptr;
            HIP_CHECK(hipMalloc(target, f16.size() * sizeof(_Float16)));
            HIP_CHECK(hipMemcpy(*target, f16.data(), f16.size() * sizeof(_Float16), hipMemcpyHostToDevice));
            return RCPP_OK;
        };
        lw(prefix("input_layernorm.weight"), &layer.input_norm_dev, &layer.input_norm_dev, hidden_size, 1);
        lw(prefix("post_attention_layernorm.weight"), &layer.post_attn_norm_dev, &layer.post_attn_norm_dev, hidden_size, 1);
        // Weight projections: select pointer based on format
        lw(prefix("self_attn.q_proj.weight"), &layer.q_packed_dev, &layer.bst_q_packed_dev, n_heads * head_dim, hidden_size);
        lw(prefix("self_attn.k_proj.weight"), &layer.k_packed_dev, &layer.bst_k_packed_dev, hidden_size, hidden_size);
        lw(prefix("self_attn.v_proj.weight"), &layer.v_packed_dev, &layer.bst_v_packed_dev, hidden_size, hidden_size);
        lw(prefix("self_attn.o_proj.weight"), &layer.o_packed_dev, &layer.bst_o_packed_dev, hidden_size, n_heads * head_dim);
        lw(prefix("mlp.gate_proj.weight"), &layer.gate_packed_dev, &layer.bst_gate_packed_dev, inter_size, hidden_size);
        lw(prefix("mlp.up_proj.weight"), &layer.up_packed_dev, &layer.bst_up_packed_dev, inter_size, hidden_size);
        lw(prefix("mlp.down_proj.weight"), &layer.down_packed_dev, &layer.bst_down_packed_dev, hidden_size, inter_size);
    }
    fprintf(stderr, "[gguf] Model load complete\n");
    return RCPP_OK;
}
} // extern "C"
