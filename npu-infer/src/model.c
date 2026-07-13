#include "model.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>

// ========= BF16 conversion helpers =========
float bf16_to_float(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

uint16_t float_to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    // Round to nearest even for BF16
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    uint32_t truncated = (bits + rounding_bias) >> 16;
    return (uint16_t)truncated;
}

// ========= Q4NX I8 format =========
// Q4NX stores I8 weights as raw int8 values with NO embedded scale factors.
// For inference, I8 values are dequantized using per-group absmax:
//   For each group of 32 I8 values:
//     scale = max(|I8_values|) / 127
//     BF16_val = I8_val * scale
// This converts [-128, 127] I8 range to [-absmax, absmax] BF16 range.
//
// FLM's reorder_cpy rearranges I8 data into NPU's blocked format.
// Our engine: convert I8→BF16 per group, then pack into [npu_block_rows, npu_block_cols] blocks.

// ========= NPU blocked format =========
//
// NPU expects weights in blocked format:
// For weight matrix [out_features, in_features]:
//   - Column-blocks of 1024 columns
//   - Row-blocks of 256 rows
//   - Each block = [min(256,out_rem), 1024] BF16 values in row-major
//   - Padded within a 1MB BO (second half zero padding)

// ========= Q4NX format =========
// Q4NX stores weights in a hybrid I8/BF16 format.
// Each 2 consecutive I8 bytes form one BF16 value: [lo_byte, hi_byte] little-endian.
// The shape [rows, cols] in the metadata refers to I8 ELEMENTS (1 byte each).
// Actual BF16 elements = rows * cols / 2.
// NPU blocking: each BO holds [npu_block_rows, npu_block_cols] BF16 values = 512KB data + 512KB zeros.
// Number of column blocks = ceil(cols / 2 / block_cols).

int npu_weight_num_blocks(const TensorDesc* desc, const ModelConfig* config) {
    if (desc->ndim != 2) return 0;
    int64_t i8_cols = desc->shape[1];
    int64_t bf16_cols = i8_cols / 2;  // 2 I8 bytes per BF16
    return (int)((bf16_cols + config->npu_block_cols - 1) / config->npu_block_cols);
}

int npu_dequant_block(void* out, const void* in,
                       const TensorDesc* desc, const ModelConfig* config,
                       int block_idx) {
    int64_t rows = desc->shape[0];
    int64_t i8_cols = desc->shape[1];
    int64_t bf16_cols = i8_cols / 2;  // Elements per row in BF16
    int block_cols = config->npu_block_cols;  // 1024
    
    const uint8_t* data = (const uint8_t*)in;
    uint16_t* bf16_out = (uint16_t*)out;
    
    int col_start = block_idx * block_cols;
    int col_end = col_start + block_cols;
    if (col_end > bf16_cols) col_end = (int)bf16_cols;
    int num_cols = col_end - col_start;
    int num_rows = (int)rows;
    if (num_rows > (int)config->npu_block_rows) 
        num_rows = (int)config->npu_block_rows;
    
    // Zero entire output block
    memset(bf16_out, 0, num_rows * block_cols * 2);
    
    // For each BF16 value, read 2 consecutive I8 bytes as little-endian BF16
    // I8 data layout: [lo_byte_0, hi_byte_0, lo_byte_1, hi_byte_1, ...] per row
    for (int r = 0; r < num_rows; r++) {
        for (int c = 0; c < num_cols; c++) {
            // Position in BF16 space
            int bf16_idx = r * bf16_cols + col_start + c;
            // Position in I8 byte space (2 bytes per BF16)
            int i8_byte_idx = bf16_idx * 2;
            if (i8_byte_idx + 1 >= rows * i8_cols) break;
            
            // Read as little-endian uint16 = BF16
            uint16_t bf16_val;
            memcpy(&bf16_val, &data[i8_byte_idx], 2);
            bf16_out[r * block_cols + c] = bf16_val;
        }
    }
    
    return num_rows * num_cols;
}

int npu_pack_weight_bo(uint8_t* bo_buffer, const void* in,
                        const TensorDesc* desc, const ModelConfig* config,
                        int block_idx) {
    int bo_size = config->npu_weight_bo_size;
    memset(bo_buffer, 0, bo_size);
    
    int num_written = npu_dequant_block(bo_buffer, in, desc, config, block_idx);
    if (num_written < 0) return num_written;
    
    return 0;
}

// ========= Simple JSON Parser =========

static int parse_json_metadata(const uint8_t* json_data, uint64_t json_len,
                                TensorDesc* tensors, int max_tensors);
static int find_tensor(const char* name, TensorDesc* tensors, int count);

// ========= Model Loader =========

ModelWeights* model_load(const char* path, ModelConfig config) {
    ModelWeights* mw = calloc(1, sizeof(ModelWeights));
    if (!mw) return NULL;
    
    memcpy(&mw->config, &config, sizeof(config));
    
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        LOG_ERROR("Cannot open model file: %s", path);
        free(mw);
        return NULL;
    }
    
    struct stat st;
    fstat(fd, &st);
    mw->file_size = st.st_size;
    
    mw->file_data = mmap(NULL, mw->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    
    if (mw->file_data == MAP_FAILED) {
        LOG_ERROR("mmap failed: %s", strerror(errno));
        free(mw);
        return NULL;
    }
    
    uint64_t header_size;
    memcpy(&header_size, mw->file_data, 8);
    
    LOG_INFO("Model file: %s (%lu MB)", path, (unsigned long)(mw->file_size / 1024 / 1024));
    LOG_INFO("Header: %lu bytes JSON", (unsigned long)header_size);
    
    const char* json_start = (const char*)(mw->file_data + 8);
    size_t json_len = header_size;
    
    int max_tensors = 512;
    TensorDesc* tensors = calloc(max_tensors, sizeof(TensorDesc));
    int num_tensors = parse_json_metadata((const uint8_t*)json_start, json_len,
                                           tensors, max_tensors);
    
    LOG_INFO("Found %d tensors in metadata", num_tensors);
    
    // Embed tokens
    int idx_emb = find_tensor("model.embed_tokens.weight", tensors, num_tensors);
    if (idx_emb >= 0) memcpy(&mw->embed_tokens, &tensors[idx_emb], sizeof(TensorDesc));
    
    // Allocate per-layer weights
    mw->layers = calloc(config.num_layers, sizeof(LayerWeights));
    
    char name_buf[128];
    for (int l = 0; l < config.num_layers; l++) {
        LayerWeights* layer = &mw->layers[l];
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.input_layernorm.weight", l);
        int idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->input_layernorm_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.post_attention_layernorm.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->post_attention_layernorm_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.q_norm.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->q_norm_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.k_norm.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->k_norm_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.q_proj.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->q_proj_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.k_proj.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->k_proj_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.v_proj.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->v_proj_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.o_proj.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->o_proj_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.gate_proj.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->gate_proj_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.up_proj.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->up_proj_weight, &tensors[idx], sizeof(TensorDesc));
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.down_proj.weight", l);
        idx = find_tensor(name_buf, tensors, num_tensors);
        if (idx >= 0) memcpy(&layer->down_proj_weight, &tensors[idx], sizeof(TensorDesc));
    }
    
    // Final norm
    int idx_fn = find_tensor("model.norm.weight", tensors, num_tensors);
    if (idx_fn >= 0) memcpy(&mw->norm_weight, &tensors[idx_fn], sizeof(TensorDesc));
    
    // LM head
    int idx_lm = find_tensor("lm_head.weight", tensors, num_tensors);
    if (idx_lm >= 0) memcpy(&mw->lm_head_weight, &tensors[idx_lm], sizeof(TensorDesc));
    
    free(tensors);
    
    LOG_INFO("Model loaded: %d tensors, %d layers", num_tensors, config.num_layers);
    return mw;
}

void model_free(ModelWeights* mw) {
    if (!mw) return;
    if (mw->file_data) munmap(mw->file_data, mw->file_size);
    free(mw->layers);
    free(mw);
}

void* model_tensor_data(ModelWeights* mw, TensorDesc* desc) {
    if (!mw || !desc) return NULL;
    return mw->file_data + desc->data_offset;
}

int model_find_tensor(const char* name, ModelWeights* mw) {
    (void)name;
    (void)mw;
    return -1;
}

static int parse_json_metadata(const uint8_t* json_data, uint64_t json_len,
                                TensorDesc* tensors, int max_tensors) {
    const char* s = (const char*)json_data;
    uint64_t len = json_len;
    int count = 0;
    
    const char* p = s;
    const char* end = s + len;
    
    while (p < end && count < max_tensors) {
        while (p < end && *p != '"') p++;
        if (p >= end) break;
        
        const char* key_start = p + 1;
        const char* key_end = key_start;
        while (key_end < end && *key_end != '"') key_end++;
        if (key_end >= end) break;
        
        ptrdiff_t key_len = key_end - key_start;
        const char* key_str = key_start;
        
        bool ends_with_weight = (key_len > 7 && memcmp(key_end - 7, ".weight", 7) == 0);
        bool is_lm_head = (key_len == 12 && memcmp(key_str, "lm_head.weight", 12) == 0);
        
        if (!ends_with_weight && !is_lm_head) {
            p = key_end + 1;
            continue;
        }
        
        TensorDesc* t = &tensors[count];
        memset(t, 0, sizeof(TensorDesc));
        
        int name_len = key_len < (int)sizeof(t->name) - 1 ? key_len : (int)sizeof(t->name) - 1;
        memcpy(t->name, key_str, name_len);
        t->name[name_len] = '\0';
        
        p = key_end + 1;
        while (p < end && *p != '{') p++;
        if (p >= end) break;
        
        const char* dtype_pos = strstr(p, "\"dtype\"");
        if (dtype_pos) {
            const char* val_start = strchr(dtype_pos, ':');
            if (val_start) {
                val_start++;
                while (*val_start == ' ' || *val_start == '"') val_start++;
                const char* val_end = val_start;
                while (*val_end && *val_end != '"') val_end++;
                int dt_len = val_end - val_start;
                int copy_len = dt_len < (int)sizeof(t->dtype) - 1 ? dt_len : (int)sizeof(t->dtype) - 1;
                memcpy(t->dtype, val_start, copy_len);
                t->dtype[copy_len] = '\0';
            }
        }
        
        const char* shape_pos = strstr(p, "\"shape\"");
        if (shape_pos) {
            const char* arr_start = strchr(shape_pos, '[');
            if (arr_start) {
                arr_start++;
                t->ndim = 0;
                const char* sp = arr_start;
                while (*sp != ']' && sp < end && t->ndim < 4) {
                    while (*sp == ' ' || *sp == ',') sp++;
                    if (*sp >= '0' && *sp <= '9') {
                        t->shape[t->ndim] = strtol(sp, (char**)&sp, 10);
                        t->ndim++;
                    } else break;
                }
            }
        }
        
        const char* off_pos = strstr(p, "\"data_offsets\"");
        if (off_pos) {
            const char* arr_start = strchr(off_pos, '[');
            if (arr_start) {
                uint64_t offsets[2] = {0, 0};
                int off_count = 0;
                const char* sp = arr_start + 1;
                while (*sp != ']' && sp < end && off_count < 2) {
                    while (*sp == ' ' || *sp == ',') sp++;
                    if (*sp >= '0' && *sp <= '9') {
                        offsets[off_count] = strtoull(sp, (char**)&sp, 10);
                        off_count++;
                    } else break;
                }
                if (off_count == 2) {
                    t->data_offset = offsets[0];
                    t->data_size = offsets[1] - offsets[0];
                }
            }
        }
        
        t->num_elements = 1;
        for (int d = 0; d < t->ndim; d++) {
            t->num_elements *= t->shape[d];
        }
        
        count++;
        p = key_end + 1;
    }
    
    return count;
}

static int find_tensor(const char* name, TensorDesc* tensors, int count) {
    for (int i = 0; i < count; i++) {
        size_t nlen = strlen(name);
        if (strncmp(tensors[i].name, name, nlen) == 0 &&
            strlen(tensors[i].name) == nlen) {
            return i;
        }
    }
    return -1;
}
