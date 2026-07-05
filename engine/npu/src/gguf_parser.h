/**
 * GGUF Parser — reads GGUF model files and loads weights into NPU BOs.
 *
 * Format spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
 *
 * Build: included directly into npu_engine_cb.cpp or standalone.
 */
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

// ─── GGUF types ───
enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// ─── GGML tensor types ───
enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
};

// Quantization block sizes and type sizes
static int ggml_blck_size(enum ggml_type t) {
    switch(t) {
        case GGML_TYPE_Q4_0:    return 32;
        case GGML_TYPE_Q4_1:    return 32;
        case GGML_TYPE_Q5_0:    return 32;
        case GGML_TYPE_Q5_1:    return 32;
        case GGML_TYPE_Q8_0:    return 32;
        case GGML_TYPE_Q8_1:    return 32;
        case GGML_TYPE_Q4_K:    return 256;
        case GGML_TYPE_Q5_K:    return 256;
        case GGML_TYPE_Q6_K:    return 256;
        case GGML_TYPE_Q8_K:    return 256;
        case GGML_TYPE_IQ1_S:   return 256;
        case GGML_TYPE_IQ2_XXS: return 256;
        case GGML_TYPE_IQ2_XS:  return 256;
        case GGML_TYPE_IQ3_XXS: return 256;
        case GGML_TYPE_IQ3_S:   return 256;
        case GGML_TYPE_IQ2_S:   return 256;
        case GGML_TYPE_IQ4_NL:  return 32;
        case GGML_TYPE_IQ4_XS:  return 256;
        case GGML_TYPE_F32:     return 1;
        case GGML_TYPE_F16:     return 1;
        case GGML_TYPE_I8:      return 1;
        case GGML_TYPE_I16:     return 1;
        case GGML_TYPE_I32:     return 1;
        default: return 1;
    }
}

static int ggml_type_size(enum ggml_type t) {
    switch(t) {
        case GGML_TYPE_Q4_0:    return 20;    // 2 bytes scale + 16 bytes (32*4/8)
        case GGML_TYPE_Q4_1:    return 24;    // 2*2 bytes scale + 16 bytes
        case GGML_TYPE_Q5_0:    return 24;    // 2 bytes scale + 2 bytes + 16 bytes
        case GGML_TYPE_Q5_1:    return 28;    // 2*2 bytes scale + 2 bytes + 16 bytes
        case GGML_TYPE_Q8_0:    return 34;    // 2 bytes scale + 32 bytes
        case GGML_TYPE_Q8_1:    return 40;    // 2*2 bytes scale + 32 bytes
        case GGML_TYPE_Q4_K:    return 144;   // 2+12 scales + 128 bytes
        case GGML_TYPE_Q5_K:    return 176;   // 2+12 scales + 160 bytes  
        case GGML_TYPE_Q6_K:    return 210;   // 2+12 scales + 192 bytes
        case GGML_TYPE_Q8_K:    return 274;   // 2+12 scales + 256 bytes
        case GGML_TYPE_IQ1_S:   return 34;    // scales + data
        case GGML_TYPE_IQ2_XXS: return 66;    // scales + data
        case GGML_TYPE_IQ2_XS:  return 80;    // scales + data
        case GGML_TYPE_IQ3_XXS: return 104;   // scales + data
        case GGML_TYPE_IQ3_S:   return 128;   // scales + data
        case GGML_TYPE_IQ2_S:   return 128;   // scales + data
        case GGML_TYPE_IQ4_NL:  return 18;    // scales + data
        case GGML_TYPE_IQ4_XS:  return 144;   // scales + data
        case GGML_TYPE_F32:     return 4;
        case GGML_TYPE_F16:     return 2;
        case GGML_TYPE_I8:      return 1;
        case GGML_TYPE_I16:     return 2;
        case GGML_TYPE_I32:     return 4;
        default: return 0;
    }
}

// ─── GGUF reader ───
struct GGUFReader {
    const uint8_t* data = nullptr;
    size_t size = 0;
    size_t offset = 0;
    
    bool open(const char* path) {
        int fd = ::open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "GGUF: cannot open %s\n", path); return false; }
        struct stat st; ::fstat(fd, &st);
        size = st.st_size;
        data = (const uint8_t*)::mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (!data) { fprintf(stderr, "GGUF: mmap failed\n"); return false; }
        offset = 0;
        return true;
    }
    
    void close() {
        if (data) ::munmap((void*)data, size);
        data = nullptr; size = 0;
    }
    
    // Read primitives
    uint8_t  read_u8()  { uint8_t  v; memcpy(&v, data+offset, 1); offset+=1; return v; }
    int8_t   read_i8()  { int8_t   v; memcpy(&v, data+offset, 1); offset+=1; return v; }
    uint16_t read_u16() { uint16_t v; memcpy(&v, data+offset, 2); offset+=2; return v; }
    int16_t  read_i16() { int16_t  v; memcpy(&v, data+offset, 2); offset+=2; return v; }
    uint32_t read_u32() { uint32_t v; memcpy(&v, data+offset, 4); offset+=4; return v; }
    int32_t  read_i32() { int32_t  v; memcpy(&v, data+offset, 4); offset+=4; return v; }
    uint64_t read_u64() { uint64_t v; memcpy(&v, data+offset, 8); offset+=8; return v; }
    int64_t  read_i64() { int64_t  v; memcpy(&v, data+offset, 8); offset+=8; return v; }
    float    read_f32() { float    v; memcpy(&v, data+offset, 4); offset+=4; return v; }
    double   read_f64() { double   v; memcpy(&v, data+offset, 8); offset+=8; return v; }
    float    read_f16() { uint16_t h=read_u16(); uint32_t s=(h&0x8000)<<16; uint32_t e=(h&0x7C00)>>10; uint32_t m=h&0x03FF; uint32_t f; if(e==0)f=s|(m<<13);else if(e==31)f=s|0x7F800000|(m<<13);else f=s|((e+112)<<23)|(m<<13); float v; memcpy(&v,&f,4); return v; }
    
    std::string read_string() {
        uint64_t len = read_u64();
        std::string s((const char*)data+offset, len);
        offset += len;
        return s;
    }
    
    void skip(size_t n) { offset += n; }
    void seek(size_t pos) { offset = pos; }
    size_t tell() { return offset; }
    const void* ptr() { return data + offset; }
    
    // Read a value of given GGUF type
    double read_value(int type) {
        switch(type) {
            case GGUF_TYPE_UINT8:   return read_u8();
            case GGUF_TYPE_INT8:    return read_i8();
            case GGUF_TYPE_UINT16:  return read_u16();
            case GGUF_TYPE_INT16:   return read_i16();
            case GGUF_TYPE_UINT32:  return read_u32();
            case GGUF_TYPE_INT32:   return read_i32();
            case GGUF_TYPE_FLOAT32: return read_f32();
            case GGUF_TYPE_UINT64:  return (double)read_u64();
            case GGUF_TYPE_INT64:   return (double)read_i64();
            case GGUF_TYPE_FLOAT64: return read_f64();
            case GGUF_TYPE_BOOL:    return read_u8();
            default: return 0;
        }
    }
};

// ─── GGUF model info ───
struct GGUFModel {
    // Metadata
    std::string arch;
    int64_t hidden_size = 0;
    int64_t n_layers = 0;
    int64_t n_heads = 0;
    int64_t n_kv_heads = 0;
    int64_t head_dim = 0;
    int64_t intermediate_size = 0;
    int64_t vocab_size = 0;
    int64_t n_embd = 0;
    uint32_t alignment = 32;
    
    // Tensor info
    struct Tensor {
        std::string name;
        std::vector<int64_t> dims;
        int type;
        uint64_t file_offset;  // offset to tensor data in file
        uint64_t file_size;    // size of tensor data in file
    };
    std::vector<Tensor> tensors;
    std::unordered_map<std::string, int> tensor_map;
    
    // Data offsets
    uint64_t tensor_data_offset = 0;  // where tensor data starts in file
    
    // All metadata KV pairs (for debugging)
    struct KV { std::string key; int type; double value; std::string str; };
    std::vector<KV> kv_pairs;
    
    size_t file_size = 0;
    
    bool parse(GGUFReader& r) {
        // Check magic
        if (r.size < 4 || memcmp(r.data, "GGUF", 4) != 0) {
            fprintf(stderr, "GGUF: bad magic\n"); return false;
        }
        r.offset = 4;
        
        uint32_t version = r.read_u32();
        if (version != 3) {
            fprintf(stderr, "GGUF: unsupported version %u (expected 3)\n", version);
            // Try anyway — format is stable
        }
        
        uint64_t tensor_count = r.read_u64();
        uint64_t kv_count = r.read_u64();
        
        // Read KV pairs
        for (uint64_t i = 0; i < kv_count; i++) {
            std::string key = r.read_string();
            int type = r.read_i32();
            
            KV kv;
            kv.key = key;
            kv.type = type;
            
            if (type == GGUF_TYPE_STRING) {
                kv.str = r.read_string();
                apply_metadata(key, kv.str);
            } else if (type == GGUF_TYPE_ARRAY) {
                int arr_type = r.read_i32();
                uint64_t arr_n = r.read_u64();
                if (key == "tokenizer.ggml.bos_token_id" && arr_type == GGUF_TYPE_INT32 && arr_n > 0) {
                    kv.value = r.read_i32(); // first element
                    if (arr_n > 1) {
                        // Skip remaining
                        if (arr_type == GGUF_TYPE_INT32) r.skip((arr_n-1)*4);
                        else r.skip((arr_n-1)*size_of_type(arr_type));
                    }
                } else if (arr_type == GGUF_TYPE_STRING) {
                    // String arrays: each element has length+data, skip individually
                    for (uint64_t si = 0; si < arr_n; si++) {
                        uint64_t slen = r.read_u64();
                        r.skip(slen);
                    }
                } else {
                    r.skip(arr_n * size_of_type(arr_type));
                }
            } else {
                kv.value = r.read_value(type);
                apply_metadata(key, kv.value);
            }
            kv_pairs.push_back(kv);
        }
        
        // Compute head_dim if not set
        if (head_dim == 0 && n_heads > 0 && hidden_size > 0) {
            head_dim = hidden_size / n_heads;
        }
        // Compute intermediate_size for common arch patterns
        if (intermediate_size == 0) {
            intermediate_size = hidden_size * 4;  // fallback for older models
        }
        
        // Read tensor info
        for (uint64_t i = 0; i < tensor_count; i++) {
            Tensor t;
            t.name = r.read_string();
            uint32_t n_dims = r.read_u32();
            for (uint32_t d = 0; d < n_dims; d++) {
                t.dims.push_back(r.read_i64());
            }
            t.type = r.read_i32();
            t.file_offset = r.read_u64();
            
            // Compute tensor data size
            int n_elems = 1;
            for (auto d : t.dims) n_elems *= d;
            int bs = ggml_blck_size((ggml_type)t.type);
            int ts = ggml_type_size((ggml_type)t.type);
            t.file_size = (n_elems / bs) * ts;
            
            tensors.push_back(t);
            tensor_map[t.name] = i;
        }
        
        // Compute where tensor data actually starts
        tensor_data_offset = r.offset;
        // Align to alignment
        if (tensor_data_offset % alignment != 0) {
            tensor_data_offset += alignment - (tensor_data_offset % alignment);
        }
        
        file_size = r.size;
        return true;
    }
    
    static int size_of_type(int t) {
        switch(t) {
            case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return 1;
            case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
            case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
            case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
            default: return 4;
        }
    }
    
    void apply_metadata(const std::string& key, const std::string& val) {
        if (key == "general.architecture") arch = val;
        else if (key == "general.alignment") alignment = 32; // default
    }
    
    void apply_metadata(const std::string& key, double val) {
        int64_t iv = (int64_t)val;
        auto dot = key.find('.');
        std::string suffix = (dot != std::string::npos) ? key.substr(dot) : key;
        if (suffix == ".block_count") n_layers = iv;
        else if (suffix == ".attention.head_count") n_heads = iv;
        else if (suffix == ".attention.head_count_kv") n_kv_heads = iv;
        else if (suffix == ".embedding_length") hidden_size = iv;
        else if (suffix == ".feed_forward_length") intermediate_size = iv;
        else if (suffix == ".vocab_size") vocab_size = iv;
        else if (suffix == ".attention.key_length") head_dim = iv;
        else if (suffix == ".attention.layer_norm_rms_epsilon") {}
        else if (suffix == ".context_length") {}
    }
    
    // Check if a tensor exists
    bool has_tensor(const char* name) {
        return tensor_map.find(name) != tensor_map.end();
    }
    
    // Get tensor info
    const Tensor* get_tensor(const char* name) {
        auto it = tensor_map.find(name);
        if (it == tensor_map.end()) return nullptr;
        return &tensors[it->second];
    }
    
    // Read raw tensor data from file (for the reader to seek to)
    uint64_t get_tensor_file_offset(const Tensor& t) {
        return tensor_data_offset + t.file_offset;
    }
    
    // Dequantize a tensor to float32 (allocate with new[] )
    float* dequantize_to_f32(GGUFReader& r, const Tensor& t) {
        int n_elems = 1;
        for (auto d : t.dims) n_elems *= d;
        float* out = new float[n_elems];
        
        uint64_t file_off = get_tensor_file_offset(t);
        r.seek(file_off);
        
        int bs = ggml_blck_size((ggml_type)t.type);
        int ts = ggml_type_size((ggml_type)t.type);
        int n_blocks = n_elems / bs;
        
        switch (t.type) {
            case GGML_TYPE_F32: {
                for (int i = 0; i < n_elems; i++) out[i] = r.read_f32();
                break;
            }
            case GGML_TYPE_F16: {
                for (int i = 0; i < n_elems; i++) {
                    uint16_t h = r.read_u16();
                    // F16 to F32
                    uint32_t sign = (h & 0x8000) << 16;
                    uint32_t exp = (h & 0x7C00) >> 10;
                    uint32_t mant = h & 0x03FF;
                    uint32_t f;
                    if (exp == 0) {
                        f = sign | (mant << 13); // subnormal → zero
                    } else if (exp == 31) {
                        f = sign | 0x7F800000 | (mant << 13); // inf/nan
                    } else {
                        f = sign | ((exp + 112) << 23) | (mant << 13);
                    }
                    memcpy(&out[i], &f, 4);
                }
                break;
            }
            case GGML_TYPE_Q8_0: {
                for (int b = 0; b < n_blocks; b++) {
                    float d = r.read_f16();  // scale (stored as F16)
                    for (int j = 0; j < 32; j++) {
                        out[b*32+j] = d * (int8_t)r.read_u8();
                    }
                }
                break;
            }
            case GGML_TYPE_Q4_0: {
                for (int b = 0; b < n_blocks; b++) {
                    float d = r.read_f16();
                    for (int j = 0; j < 16; j++) {
                        uint8_t byte = r.read_u8();
                        out[b*32+j*2+0] = d * ((int8_t)((byte & 0xF0) >> 4));
                        out[b*32+j*2+1] = d * ((int8_t)(byte & 0x0F));
                    }
                }
                break;
            }
            case GGML_TYPE_I8: {
                for (int i = 0; i < n_elems; i++) out[i] = (float)(int8_t)r.read_u8();
                break;
            }
            default: {
                fprintf(stderr, "GGUF: unsupported quant type %d for %s\n", t.type, t.name.c_str());
                delete[] out;
                return nullptr;
            }
        }
        return out;
    }
};
