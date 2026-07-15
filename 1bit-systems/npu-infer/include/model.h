#ifndef NPU_INFER_MODEL_H
#define NPU_INFER_MODEL_H

#include <stdint.h>
#include <stddef.h>
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Tensor descriptor from Q4NX metadata
typedef struct {
    char     name[128];
    char     dtype[16];
    int64_t  shape[4];
    int      ndim;
    uint64_t data_offset;
    uint64_t data_size;
    uint64_t num_elements;
} TensorDesc;

// Per-layer weights
typedef struct {
    TensorDesc input_layernorm_weight;
    TensorDesc post_attention_layernorm_weight;
    TensorDesc q_norm_weight;
    TensorDesc k_norm_weight;
    TensorDesc q_proj_weight;
    TensorDesc k_proj_weight;
    TensorDesc v_proj_weight;
    TensorDesc o_proj_weight;
    TensorDesc gate_proj_weight;
    TensorDesc up_proj_weight;
    TensorDesc down_proj_weight;
} LayerWeights;

// Full model weights
typedef struct {
    ModelConfig config;
    
    // Embedding/LM Head (tied)
    TensorDesc embed_tokens;   // BF16 [vocab_size, hidden_size]
    
    TensorDesc lm_head_weight; // I8 tied to embed_tokens
    
    // Per-layer
    LayerWeights* layers;
    
    // Final norm
    TensorDesc norm_weight;
    
    // Raw file data (mmap)
    uint8_t* file_data;
    uint64_t file_size;
} ModelWeights;

// ========== Model Loader ==========
ModelWeights* model_load(const char* path, ModelConfig config);
void model_free(ModelWeights* mw);
void* model_tensor_data(ModelWeights* mw, TensorDesc* desc);
void model_print_info(ModelWeights* mw);
int model_find_tensor(const char* name, ModelWeights* mw);

// ========== NPU Weight Packer ==========

// Get number of NPU blocks needed for a weight tensor
// in_features is the input dimension (column dimension that gets blocked to 1024)
int npu_weight_num_blocks(const TensorDesc* desc, const ModelConfig* config);

// Dequantize I8 tensor to BF16, one block at a time
// out: pre-allocated buffer of block_rows * block_cols BF16 values (only first half filled if padded)
// in: Q4NX I8 weight data pointer
// block_idx: which of the N column blocks (0..num_blocks-1)
// Returns number of BF16 values written
int npu_dequant_block(void* out, const void* in, 
                       const TensorDesc* desc, const ModelConfig* config,
                       int block_idx);

// Pack a single 1MB BO from a tensor
// bo_buffer: pre-allocated 1MB buffer (will be filled with [block] + zeros)
// in: Q4NX weight data pointer
// block_idx: which column block to pack
// Returns 0 on success
int npu_pack_weight_bo(uint8_t* bo_buffer, const void* in,
                        const TensorDesc* desc, const ModelConfig* config,
                        int block_idx);

#ifdef __cplusplus
}
#endif

// Convert BF16 buffer back to float for verification
float bf16_to_float(uint16_t v);
uint16_t float_to_bf16(float v);

#endif // NPU_INFER_MODEL_H
