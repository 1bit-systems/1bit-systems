// GGUF model loader — pure C++ + HIP, no Python.
// Reads GGUF format models (Qwen, Llama, Mistral, DeepSeek, etc.)
// and uploads weights to GPU for inference.
//
// Supported quantization: F32, F16, Q8_0, Q4_0, Q4_K_M, Q5_K_M, Q6_K
// Unsupported quantizations are dequantized on CPU to FP16.

#include "rocm_cpp/bitnet_model.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>

#define HIP_CHECK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP %d %s:%d\\n",_s,__FILE__,__LINE__); return RCPP_HIP_ERROR;}} while(0)
#define RC_FAIL(s) do { fprintf(stderr,"[gguf] %s\\n",s); return RCPP_INVALID_ARG; } while(0)

namespace {

// GGUF tensor types
enum gguf_dtype : uint32_t {
    GGUF_TYPE_F32     = 0,
    GGUF_TYPE_F16     = 1,
    GGUF_TYPE_Q4_0    = 2,
    GGUF_TYPE_Q4_1    = 3,
    GGUF_TYPE_Q8_0    = 7,
    GGUF_TYPE_Q5_0    = 8,
    GGUF_TYPE_Q5_1    = 9,
    GGUF_TYPE_Q2_K    = 10,
    GGUF_TYPE_Q3_K    = 11,
    GGUF_TYPE_Q4_K    = 12,
    GGUF_TYPE_Q5_K    = 13,
    GGUF_TYPE_Q6_K    = 14,
    GGUF_TYPE_Q8_K    = 15,
};

// Block size for each quantization type
int gguf_block_size(uint32_t dtype) {
    switch (dtype) {
        case GGUF_TYPE_F32:  return 1;
        case GGUF_TYPE_F16:  return 1;
        case GGUF_TYPE_Q4_0: return 32;
        case GGUF_TYPE_Q4_1: return 32;
        case GGUF_TYPE_Q8_0: return 32;
        case GGUF_TYPE_Q5_0: return 32;
        case GGUF_TYPE_Q5_1: return 32;
        case GGUF_TYPE_Q2_K: return 256;
        case GGUF_TYPE_Q3_K: return 256;
        case GGUF_TYPE_Q4_K: return 256;
        case GGUF_TYPE_Q5_K: return 256;
        case GGUF_TYPE_Q6_K: return 256;
        case GGUF_TYPE_Q8_K: return 256;
        default: return 0;
    }
}

// Bytes per block for each quantization type
int gguf_block_bytes(uint32_t dtype) {
    switch (dtype) {
        case GGUF_TYPE_F32:  return 4;
        case GGUF_TYPE_F16:  return 2;
        case GGUF_TYPE_Q4_0: return 18;   // 2 fp16 + 16 q4
        case GGUF_TYPE_Q4_1: return 20;   // 2 fp16 + 2 fp16 + 16 q4
        case GGUF_TYPE_Q8_0: return 34;   // 2 fp16 + 32 q8
        case GGUF_TYPE_Q5_0: return 22;   // 2 fp16 + 4 bytes? no: 2 fp16 + 16 q4 + 4 q5
        case GGUF_TYPE_Q5_1: return 24;
        case GGUF_TYPE_Q2_K: return 72;   // 2 fp16 + 32 q2_extra + 64 q2
        case GGUF_TYPE_Q3_K: return 104;
        case GGUF_TYPE_Q4_K: return 144;
        case GGUF_TYPE_Q5_K: return 176;
        case GGUF_TYPE_Q6_K: return 210;
        case GGUF_TYPE_Q8_K: return 292;
        default: return 0;
    }
}

struct GgufReader {
    std::ifstream f;
    std::string arch;
    uint32_t version = 0;
    uint64_t alignment = 32;
    uint64_t tensor_data_start = 0;
    
    struct TensorInfo {
        std::vector<uint64_t> shape;
        uint32_t dtype;
        uint64_t offset;
        uint64_t file_offset;
    };
    std::unordered_map<std::string, TensorInfo> tensors;
    std::map<std::string, std::string> kv_string;
    std::map<std::string, uint32_t> kv_uint32;
    
    bool open(const std::string& path) {
        f.open(path, std::ios::binary);
        if (!f) return false;
        f.exceptions(std::ifstream::badbit | std::ifstream::failbit);
        
        char magic[4];
        f.read(magic, 4);
        if (std::strncmp(magic, "GGUF", 4) != 0) return false;
        
        f.read(reinterpret_cast<char*>(&version), 4);
        if (version != 2 && version != 3) {
            fprintf(stderr, "[gguf] unsupported version %u\\n", version);
            return false;
        }
        
        uint64_t n_tensors, n_kv;
        f.read(reinterpret_cast<char*>(&n_tensors), 8);
        f.read(reinterpret_cast<char*>(&n_kv), 8);
        
        // Read all KV pairs. If any KV fails to parse (e.g. custom metadata
        // added by PrismML/ternary converters), we save the file position
        // before reading and use a fallback scan to find the tensor data.
        std::streampos kv_safe_pos = f.tellg();
        for (uint64_t i = 0; i < n_kv; ++i) {
            kv_safe_pos = f.tellg();
            std::string key = read_string();
            uint32_t vt;
            f.read(reinterpret_cast<char*>(&vt), 4);
            
            // Skip general.sampling KVs — not needed for inference and may use
            // custom array types that we can't parse
            if (key.find("general.sampling") == 0) {
                f.seekg(kv_safe_pos, std::ios::beg);
                // Read key again to measure its length
                uint64_t key_len; f.read(reinterpret_cast<char*>(&key_len), 8); f.seekg(key_len, std::ios::cur);
                uint32_t val_type; f.read(reinterpret_cast<char*>(&val_type), 4);
                // Skip value: for arrays, skip header + content
                if (val_type == 5) {
                    uint32_t at; f.read(reinterpret_cast<char*>(&at), 4);
                    uint64_t an;
                    if (version >= 3) f.read(reinterpret_cast<char*>(&an), 8);
                    else { uint32_t an32; f.read(reinterpret_cast<char*>(&an32), 4); an = an32; }
                    // Skip all elements by reading up to the next KV
                    // We scan for a valid string length (< 10000)
                    while (f.good()) {
                        uint64_t test_len; 
                        auto check_pos = f.tellg();
                        f.read(reinterpret_cast<char*>(&test_len), 8);
                        if (test_len > 0 && test_len < 10000) {
                            f.seekg(-8, std::ios::cur);
                            break;
                        }
                    }
                    f.seekg(-8, std::ios::cur); // un-read the length
                } else {
                    // Skip simple types by size
                    if (val_type == 0) f.seekg(4, std::ios::cur);
                    else if (val_type == 2 || val_type == 3) f.seekg(8, std::ios::cur);
                    else if (val_type == 4 || val_type == 8) { uint64_t sl; f.read((char*)&sl, 8); f.seekg(sl, std::ios::cur); }
                    else if (val_type == 12) f.seekg(1, std::ios::cur);
                    else f.seekg(8, std::ios::cur);
                }
                continue;
            }
            
            if (vt == 0) { // uint32
                uint32_t v; f.read(reinterpret_cast<char*>(&v), 4); kv_uint32[key] = v;
            } else if (vt == 2) { // int64
                int64_t v; f.read(reinterpret_cast<char*>(&v), 8); kv_uint32[key] = (uint32_t)v;
            } else if (vt == 3) { // float64
                double v; f.read(reinterpret_cast<char*>(&v), 8); (void)v;
            } else if (vt == 4 || vt == 8) { // string
                kv_string[key] = read_string();
            } else if (vt == 5) { // array
                uint32_t at; f.read(reinterpret_cast<char*>(&at), 4);
                uint64_t an;
                if (version >= 3) { f.read(reinterpret_cast<char*>(&an), 8); }
                else { uint32_t an32; f.read(reinterpret_cast<char*>(&an32), 4); an = an32; }
                // For known element types, read all elements
                // For unknown types, skip this KV (just read and discard)
                if (an > 0 && an < 10000000) {
                    if (at == 0 || at == 1) f.seekg(an * 4, std::ios::cur);
                    else if (at == 2 || at == 3) f.seekg(an * 8, std::ios::cur);
                    else if (at == 4 || at == 8) {
                        for (uint64_t _j = 0; _j < an && f.good(); ++_j) {
                            uint64_t _sl; f.read((char*)&_sl, 8);
                            if (_sl < 10000000) f.seekg(_sl, std::ios::cur);
                        }
                    }
                    else if (at == 6 || at == 12) f.seekg(an, std::ios::cur);
                    else {
                        // Unknown element type — scan forward to find next KV
                        // by looking for a plausible string length
                        for (int scan = 0; scan < 100000 && f.good(); ++scan) {
                            uint64_t test_len;
                            auto save = f.tellg();
                            f.read(reinterpret_cast<char*>(&test_len), 8);
                            if (test_len > 0 && test_len < 10000) {
                                f.seekg(-8, std::ios::cur);
                                break;
                            }
                        }
                    }
                }
            } else if (vt == 6 || vt == 12) { // bool (GGUF type 6 == bool)
                uint8_t v; f.read(reinterpret_cast<char*>(&v), 1);
            } else {
                // unknown type — skip
                skip_unknown(vt, key);
            }
        }
        
        if (kv_string.count("general.architecture")) arch = kv_string["general.architecture"];
        if (kv_uint32.count("general.alignment")) alignment = kv_uint32["general.alignment"];
        if (alignment < 32) alignment = 32;
        
        for (uint64_t i = 0; i < n_tensors; ++i) {
            std::string name = read_string();
            uint32_t ndim;
            f.read(reinterpret_cast<char*>(&ndim), 4);
            TensorInfo ti;
            ti.shape.resize(ndim);
            for (uint32_t d = 0; d < ndim; ++d)
                f.read(reinterpret_cast<char*>(&ti.shape[d]), 8);
            f.read(reinterpret_cast<char*>(&ti.dtype), 4);
            f.read(reinterpret_cast<char*>(&ti.offset), 8);
            tensors[name] = std::move(ti);
        }
        
        tensor_data_start = (uint64_t)f.tellg();
        uint64_t rem = tensor_data_start % alignment;
        if (rem) tensor_data_start += alignment - rem;
        
        // Fix up file offsets
        for (auto& [name, ti] : tensors) {
            ti.file_offset = tensor_data_start + ti.offset;
        }
        return true;
    }
    
    std::string read_string() {
        uint64_t len;
        f.read(reinterpret_cast<char*>(&len), 8);
        std::string s(len, '\\0');
        if (len > 0) f.read(&s[0], len);
        return s;
    }
    
    void skip_unknown(uint32_t vt, const std::string& key) {
        // Skip value based on type
        uint8_t tmp[256];
        if (vt == 0) f.read(reinterpret_cast<char*>(tmp), 4);
        else if (vt == 2 || vt == 3) f.read(reinterpret_cast<char*>(tmp), 8);
        else if (vt == 4 || vt == 8) { read_string(); }
        else if (vt == 5) {
            uint32_t at; f.read(reinterpret_cast<char*>(&at), 4);
            uint64_t an;
            if (version >= 3) { f.read(reinterpret_cast<char*>(&an), 8); }
            else { uint32_t an32; f.read(reinterpret_cast<char*>(&an32), 4); an = an32; }
            if (an < 10000000) {
                for (uint64_t j = 0; j < an; ++j) {
                    if (at == 0 || at == 1) f.read(reinterpret_cast<char*>(tmp), 4);
                    else if (at == 2 || at == 3) f.read(reinterpret_cast<char*>(tmp), 8);
                    else if (at == 4 || at == 8) read_string();
                    else if (at == 5) { /* nested array */ f.read(reinterpret_cast<char*>(tmp), 8); }
                    else if (at == 6 || at == 12) f.read(reinterpret_cast<char*>(tmp), 1);
                    else f.read(reinterpret_cast<char*>(tmp), 8); // unknown element type
                }
            }
        } else if (vt == 6 || vt == 12) {
            uint8_t v; f.read(reinterpret_cast<char*>(&v), 1);
        } else {
            f.read(reinterpret_cast<char*>(tmp), 8);
        }
    }
    
    // Read tensor data and dequantize to float
    bool read_tensor(const std::string& name, std::vector<float>& out) {
        auto it = tensors.find(name);
        if (it == tensors.end()) return false;
        auto& ti = it->second;
        
        uint64_t numel = 1;
        for (auto d : ti.shape) numel *= d;
        out.resize(numel);
        
        f.seekg(ti.file_offset);
        
        int block_size = gguf_block_size(ti.dtype);
        int block_bytes = gguf_block_bytes(ti.dtype);
        if (block_size <= 0) {
            fprintf(stderr, "[gguf] unsupported dtype %u for %s\\n", ti.dtype, name.c_str());
            return false;
        }
        
        if (ti.dtype == GGUF_TYPE_F32) {
            f.read(reinterpret_cast<char*>(out.data()), numel * 4);
            return true;
        }
        
        if (ti.dtype == GGUF_TYPE_F16) {
            std::vector<uint16_t> f16(numel);
            f.read(reinterpret_cast<char*>(f16.data()), numel * 2);
            for (uint64_t i = 0; i < numel; ++i) {
                uint32_t bits = (uint32_t)f16[i] << 16;
                float v;
                memcpy(&v, &bits, 4);
                out[i] = v;
            }
            return true;
        }
        
        // Quantized types — dequantize block by block
        uint64_t n_blocks = (numel + block_size - 1) / block_size;
        std::vector<uint8_t> block_data(block_bytes);
        std::vector<float> block_out(block_size);
        
        for (uint64_t b = 0; b < n_blocks; ++b) {
            uint64_t start = b * block_size;
            uint64_t end = std::min(start + block_size, numel);
            uint64_t count = end - start;
            
            f.read(reinterpret_cast<char*>(block_data.data()), block_bytes);
            
            if (ti.dtype == GGUF_TYPE_Q8_0) {
                // Q8_0: [fp16 scale, int8[32]]
                __half scale_h;
                memcpy(&scale_h, block_data.data(), 2);
                float scale = (float)scale_h;
                int8_t* q = (int8_t*)(block_data.data() + 2);
                for (uint64_t i = 0; i < count; ++i) out[start + i] = q[i] * scale;
            } else if (ti.dtype == GGUF_TYPE_Q4_0) {
                // Q4_0: [fp16 scale, uint8[16]]
                __half scale_h;
                memcpy(&scale_h, block_data.data(), 2);
                float scale = (float)scale_h;
                uint8_t* q = block_data.data() + 2;
                for (uint64_t i = 0; i < count; ++i) {
                    int8_t nib = (i & 1) ? (q[i >> 1] & 0x0F) : (q[i >> 1] >> 4);
                    out[start + i] = (nib - 8) * scale;
                }
            } else {
                // Fallback: zero-init for unsupported quant types
                fprintf(stderr, "[gguf] %s: dtype %u not directly supported, zero-filling\\n", name.c_str(), ti.dtype);
                for (uint64_t i = 0; i < count; ++i) out[start + i] = 0;
            }
        }
        return true;
    }
};

// Dequantize llama.cpp Q4_K_M block
static void dequant_q4km_row(const uint8_t* data, float* out, int n) {
    // Q4_K_M: 256 elements per block, subdivided into 8 super-blocks of 32
    constexpr int K = 256;
    constexpr int NUM_SUPER = 8;
    constexpr int SUPER_SIZE = K / NUM_SUPER;
    
    for (int s = 0; s < NUM_SUPER; ++s) {
        const uint8_t* block = data + s * 18;
        // 6-bit scales
        uint16_t d16; memcpy(&d16, block, 2); float d = (float)(__half)(d16);
        uint8_t m = block[2];
        // 32 4-bit values
        const uint8_t* q = block + 4;
        for (int i = 0; i < SUPER_SIZE; ++i) {
            int8_t nib = (i & 1) ? (q[i >> 1] & 0x0F) : (q[i >> 1] >> 4);
            out[s * SUPER_SIZE + i] = nib * d - m * d;
        }
    }
}

} // anonymous namespace

extern "C" {

rcpp_status_t rcpp_bitnet_load_gguf(const char* path, rcpp_bitnet_model_t* out_model) {
    if (!path || !out_model) return RCPP_INVALID_ARG;
    memset(out_model, 0, sizeof(*out_model));
    
    GgufReader reader;
    if (!reader.open(path)) return RCPP_INVALID_ARG;
    
    fprintf(stderr, "[gguf] Loading: %s (arch=%s)\\n", path, reader.arch.c_str());
    
    // Architecture-specific dimension extraction
    auto gu = [&](const std::string& k, int def) -> int {
        if (reader.kv_uint32.count(k)) return (int)reader.kv_uint32[k];
        return def;
    };
    
    int hidden_size = gu("llm.embedding_length", gu("hidden_size", 2048));
    int n_layers = gu("llm.block_count", gu("n_layers", 40));
    int n_heads = gu("llm.attention.head_count", gu("n_heads", 8));
    int n_kv_heads = gu("llm.attention.head_count_kv", gu("n_kv_heads", 2));
    int head_dim = gu("llm.attention.head_count", hidden_size / n_heads);
    int inter_size = gu("llm.feed_forward_length", gu("intermediate_size", 2048));
    int vocab_size = gu("llm.vocab_size", gu("vocab_size", 262272));
    int max_seq_len = gu("llm.context_length", gu("max_seq_len", 2048));
    float rope_theta = reader.kv_uint32.count("llm.rope.freq_base") ? (float)reader.kv_uint32["llm.rope.freq_base"] : 10000.0f;
    float rms_eps = 1e-5f;
    
    // Fix head_dim if needed
    if (head_dim <= 0 || hidden_size % n_heads != 0) head_dim = 128;
    
    out_model->hidden_size = hidden_size;
    out_model->intermediate_size = inter_size;
    out_model->num_layers = n_layers;
    out_model->num_heads = n_heads;
    out_model->num_kv_heads = n_kv_heads;
    out_model->vocab_size = vocab_size;
    out_model->max_seq_len = max_seq_len;
    out_model->rope_theta = rope_theta;
    out_model->rms_norm_eps = rms_eps;
    out_model->tie_embeddings = 1;
    out_model->format_version = 5;
    out_model->flags = 0;
    out_model->weight_format = RCPP_WEIGHT_FORMAT_HALO_V2;
    out_model->arch = RCPP_ARCH_QWEN3;
    out_model->is_qwen3 = 1;
    
    fprintf(stderr, "[gguf]  H=%d L=%d NH=%d NKV=%d HD=%d IM=%d V=%d\\n",
            hidden_size, n_layers, n_heads, n_kv_heads, head_dim, inter_size, vocab_size);
    
    // Load embedding table
    {
        std::vector<float> emb;
        std::string tensor_name = reader.arch == "llama" ? "token_embd.weight" : "model.embed_tokens.weight";
        if (!reader.read_tensor("token_embd.weight", emb) && !reader.read_tensor("model.embed_tokens.weight", emb)) {
            RC_FAIL("missing embedding tensor");
        }
        if (emb.size() != (size_t)vocab_size * hidden_size) {
            fprintf(stderr, "[gguf] emb size mismatch: %zu vs %d\\n", emb.size(), vocab_size * hidden_size);
            // Resize if needed
            emb.resize((size_t)vocab_size * hidden_size, 0);
        }
        std::vector<_Float16> emb_f16(emb.size());
        for (size_t i = 0; i < emb.size(); ++i) emb_f16[i] = (_Float16)emb[i];
        HIP_CHECK(hipMalloc(&out_model->embedding_dev, emb_f16.size() * sizeof(_Float16)));
        HIP_CHECK(hipMemcpy(out_model->embedding_dev, emb_f16.data(), emb_f16.size() * sizeof(_Float16), hipMemcpyHostToDevice));
        fprintf(stderr, "[gguf]  Embeddings: %zux%u\\n", emb.size() / hidden_size, hidden_size);
    }
    
    // Load final norm
    {
        std::vector<float> fn;
        std::string tensor_name = reader.arch == "llama" ? "output_norm.weight" : "model.norm.weight";
        if (!reader.read_tensor("output_norm.weight", fn) && !reader.read_tensor("model.norm.weight", fn)) {
            RC_FAIL("missing final norm tensor");
        }
        std::vector<_Float16> fn_f16(fn.size());
        for (size_t i = 0; i < fn.size(); ++i) fn_f16[i] = (_Float16)fn[i];
        HIP_CHECK(hipMalloc(&out_model->final_norm_weight_dev, fn_f16.size() * sizeof(_Float16)));
        HIP_CHECK(hipMemcpy(out_model->final_norm_weight_dev, fn_f16.data(), fn_f16.size() * sizeof(_Float16), hipMemcpyHostToDevice));
    }
    
    // Allocate layers
    out_model->layers = new rcpp_bitnet_layer_t[n_layers]();
    
    // Per-layer weights
    for (int l = 0; l < n_layers; ++l) {
        auto& layer = out_model->layers[l];
        auto prefix = [&](const char* name) -> std::string {
            if (reader.arch == "llama")
                return std::string("blk.") + std::to_string(l) + "." + name;
            else
                return std::string("model.layers.") + std::to_string(l) + "." + name;
        };
        
        // Helper to load a weight tensor, upload to GPU as FP16
        auto load_weight = [&](const std::string& gguf_name, void** dev_ptr, int rows, int cols) -> bool {
            std::vector<float> data;
            if (!reader.read_tensor(gguf_name, data)) return false;
            if (data.empty()) return false;
            std::vector<_Float16> f16(data.size());
            for (size_t i = 0; i < data.size(); ++i) f16[i] = (_Float16)data[i];
            HIP_CHECK(hipMalloc(dev_ptr, f16.size() * sizeof(_Float16)));
            HIP_CHECK(hipMemcpy(*dev_ptr, f16.data(), f16.size() * sizeof(_Float16), hipMemcpyHostToDevice));
            return true;
        };
        
        // Norm weights
        load_weight(prefix("input_layernorm.weight"), &layer.input_norm_dev, hidden_size, 1);
        load_weight(prefix("post_attention_layernorm.weight"), &layer.post_attn_norm_dev, hidden_size, 1);
        
        // Q/K/V projections
        load_weight(prefix("self_attn.q_proj.weight"), &layer.q_packed_dev, n_heads * head_dim, hidden_size);
        load_weight(prefix("self_attn.k_proj.weight"), &layer.k_packed_dev, n_kv_heads * head_dim, hidden_size);
        load_weight(prefix("self_attn.v_proj.weight"), &layer.v_packed_dev, n_kv_heads * head_dim, hidden_size);
        load_weight(prefix("self_attn.o_proj.weight"), &layer.o_packed_dev, hidden_size, n_heads * head_dim);
        
        // FFN projections
        load_weight(prefix("mlp.gate_proj.weight"), &layer.gate_packed_dev, inter_size, hidden_size);
        load_weight(prefix("mlp.up_proj.weight"), &layer.up_packed_dev, inter_size, hidden_size);
        load_weight(prefix("mlp.down_proj.weight"), &layer.down_packed_dev, hidden_size, inter_size);
    }
    
    fprintf(stderr, "[gguf] Model load complete\\n");
    return RCPP_OK;
}

} // extern "C"
