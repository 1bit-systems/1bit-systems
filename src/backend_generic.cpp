// backend_generic.cpp — Universal CPU inference backend
// Reads ModelConfig from any discovered GGUF/H1B/BIN model and runs inference.
// Supports Llama, Mistral, Qwen2, Qwen3, Gemma, Phi architectures with:
//   RMSNorm / LayerNorm, RoPE (partial/full), GQA/MHA, SiLU/SwiGLU/GeGLU, KV cache,
//   optional QKV bias, optional per-head Q/K-norm (Qwen3), MoE FFN routing
//   (top-k softmax gating over "_exps" stacked expert tensors — Qwen2-MoE/
//   Qwen3-MoE/Mixtral convention; no shared-expert or expert-bias support).

#include "backend.h"
#include <sys/stat.h>
#include <dirent.h>
#include "model_discovery.h"
#include "rocm_cpp/tokenizer.h"

// ── Minimal GGUF weight reader ──────────────────────────────────────────────
// Reads tensor data from a GGUF file into host float vectors.
// Handles F32, F16, Q8_0, Q4_0 quantizations (dequantizes to F32).
// Tensor lookup by name: gguf_tensor("blk.0.attn_q.weight", data, shape)

#include <cstring>
#include <chrono>
#include <cstdint>
#include <algorithm>

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    if (exp == 0) {
        uint32_t adj = mant ? __builtin_clz(mant) - 10 : 0;
        float r; uint32_t f32 = (sign << 31) | ((127 - 15 - adj) << 23) | ((mant << (adj + 13)) & 0x7fffff); memcpy(&r, &f32, 4); return r;
    } else if (exp == 31) {
        float r; uint32_t f32 = (sign << 31) | 0x7f800000 | (mant << 13); memcpy(&r, &f32, 4); return r;
    } else {
        float r; uint32_t f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13); memcpy(&r, &f32, 4); return r;
    }
}

#include <map>
#include <unordered_map>

// GGUF constants
#define GGUF_MAGIC    0x46554747
#define GGUF_TYPE_F32  0
#define GGUF_TYPE_F16  1
#define GGUF_TYPE_Q4_0 2
#define GGUF_TYPE_Q4_1 3
#define GGUF_TYPE_Q5_0 6
#define GGUF_TYPE_Q8_0 7
#define GGUF_TYPE_Q5_1 8
#define GGUF_TYPE_Q2_K 10
#define GGUF_TYPE_Q3_K 11
#define GGUF_TYPE_Q4_K 12
#define GGUF_TYPE_Q5_K 13
#define GGUF_TYPE_Q6_K 14
#define GGUF_TYPE_Q8_K 15

struct GgufTensor {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t dtype;
    uint64_t file_offset;  // absolute byte offset of data
};

struct GgufReader {
    FILE* f = nullptr;
    std::unordered_map<std::string, GgufTensor> tensors;
    std::vector<float> scratch;
    int vocab_size = 0;

    // Bounds to prevent OOM from malformed/crafted GGUF files (issue #366)
    static constexpr uint64_t MAX_STRING_LEN   = 1ULL * 1024 * 1024;   // 1 MiB per individual string
    static constexpr uint64_t MAX_TENSOR_COUNT = 200000;               // generous: Llama-405B ≈ 3.5k tensors
    static constexpr uint64_t MAX_KV_COUNT     = 200000;               // generous
    static constexpr uint32_t MAX_NDIM         = 16;                   // GGUF tensors are ≤ 5-d; 16 is very safe
    static constexpr uint64_t MAX_ARRAY_COUNT  = 1000000;              // 1M elements max in metadata arrays
    static constexpr uint64_t MAX_DIM_SIZE     = 1ULL << 24;           // ~16.7M per dimension — far beyond any real model

    bool open(const std::string& path) {
        f = fopen(path.c_str(), "rb");
        if (!f) return false;
        uint32_t magic; fread(&magic, 4, 1, f);
        if (magic != GGUF_MAGIC) { fclose(f); return false; }
        uint32_t version; fread(&version, 4, 1, f);
        uint64_t tensor_count, kv_count;
        fread(&tensor_count, 8, 1, f); fread(&kv_count, 8, 1, f);
        if (tensor_count > MAX_TENSOR_COUNT) {
            fprintf(stderr, "[GGUF] FATAL: tensor count %llu exceeds max %llu — file may be corrupted\n",
                    (unsigned long long)tensor_count, (unsigned long long)MAX_TENSOR_COUNT);
            fclose(f); f = nullptr; return false;
        }
        if (kv_count > MAX_KV_COUNT) {
            fprintf(stderr, "[GGUF] FATAL: KV count %llu exceeds max %llu — file may be corrupted\n",
                    (unsigned long long)kv_count, (unsigned long long)MAX_KV_COUNT);
            fclose(f); f = nullptr; return false;
        }
        // Skip metadata KVs
        for (uint64_t i = 0; i < kv_count; i++) {
            uint64_t klen; fread(&klen, 8, 1, f);
            if (klen > MAX_STRING_LEN) {
                fprintf(stderr, "[GGUF] FATAL: KV key length %llu exceeds max %llu — file may be corrupted\n",
                        (unsigned long long)klen, (unsigned long long)MAX_STRING_LEN);
                fclose(f); f = nullptr; return false;
            }
            fseek(f, klen, SEEK_CUR);
            uint32_t vtype; fread(&vtype, 4, 1, f);
            switch (vtype) {
                case 0: fseek(f, 1, SEEK_CUR); break;   // uint8
                case 1: fseek(f, 1, SEEK_CUR); break;   // int8
                case 2: fseek(f, 2, SEEK_CUR); break;   // uint16
                case 3: fseek(f, 2, SEEK_CUR); break;   // int16
                case 4: fseek(f, 4, SEEK_CUR); break;   // uint32
                case 5: fseek(f, 4, SEEK_CUR); break;   // int32
                case 6: fseek(f, 4, SEEK_CUR); break;   // float32
                case 7: fseek(f, 1, SEEK_CUR); break;   // bool
                case 8: { uint64_t sl; fread(&sl, 8, 1, f);
                    if (sl > MAX_STRING_LEN) {
                        fprintf(stderr, "[GGUF] FATAL: KV string length %llu exceeds max %llu — file may be corrupted\n",
                                (unsigned long long)sl, (unsigned long long)MAX_STRING_LEN);
                        fclose(f); f = nullptr; return false;
                    }
                    fseek(f, sl, SEEK_CUR); break; } // string
                case 9: {  // array: uint32 elem_type + uint64 count + elements
                    uint32_t at; fread(&at, 4, 1, f);
                    uint64_t an; fread(&an, 8, 1, f);
                    if (an > MAX_ARRAY_COUNT) {
                        fprintf(stderr, "[GGUF] FATAL: KV array count %llu exceeds max %llu — file may be corrupted\n",
                                (unsigned long long)an, (unsigned long long)MAX_ARRAY_COUNT);
                        fclose(f); f = nullptr; return false;
                    }
                    if (at == 8) { // array of strings — skip each string (len prefix + data)
                        for (uint64_t j = 0; j < an; j++) { 
                            uint64_t sl; fread(&sl, 8, 1, f);
                            if (sl > MAX_STRING_LEN) {
                                fprintf(stderr, "[GGUF] FATAL: array string[%llu] length %llu exceeds max %llu — file may be corrupted\n",
                                        (unsigned long long)j, (unsigned long long)sl, (unsigned long long)MAX_STRING_LEN);
                                fclose(f); f = nullptr; return false;
                            }
                            fseek(f, sl, SEEK_CUR); 
                        }
                    } else {
                        // Element sizes by GGUF type
                        static const int elem_sizes[] = {1,1,2,2,4,4,4,1,0,8,8,8,8};
                        int es = (at < 13) ? elem_sizes[at] : 4;
                        if (es == 0) { // should not happen
                            fprintf(stderr, "[GGUF] skipping array of type %u, %llu elements\n", at, (unsigned long long)an);
                            fseek(f, an * 4, SEEK_CUR);
                        } else {
                            fseek(f, an * es, SEEK_CUR);
                        }
                    }
                    break;
                }
                case 10: fseek(f, 8, SEEK_CUR); break;  // uint64
                case 11: fseek(f, 8, SEEK_CUR); break;  // int64
                case 12: fseek(f, 8, SEEK_CUR); break;  // float64
                default: break;
            }
        }
        // Read tensor info
        uint64_t data_offset = ftell(f) + tensor_count * (8+8+4+4*4); // estimate
        for (uint64_t i = 0; i < tensor_count; i++) {
            GgufTensor t;
            uint64_t nlen; fread(&nlen, 8, 1, f);
            if (nlen > 512) { fprintf(stderr, "[GGUF] bad nlen=%llu at tensor %llu, stopping\n", (unsigned long long)nlen, (unsigned long long)i); break; }
            t.name.resize(nlen); fread(&t.name[0], 1, nlen, f);
            uint32_t n_dims; fread(&n_dims, 4, 1, f);
            if (n_dims > MAX_NDIM) {
                fprintf(stderr, "[GGUF] FATAL: tensor '%s' has %u dimensions (max %u) — file may be corrupted\n",
                        t.name.c_str(), n_dims, MAX_NDIM);
                fclose(f); f = nullptr; return false;
            }
            t.dtype = 0; fread(&t.dtype, 4, 1, f);
            t.shape.resize(n_dims);
            bool dim_ok = true;
            for (uint32_t j = 0; j < n_dims; j++) {
                fread(&t.shape[j], 8, 1, f);
                if (t.shape[j] > MAX_DIM_SIZE) {
                    fprintf(stderr, "[GGUF] FATAL: tensor '%s' dim[%u]=%llu exceeds max %llu — file may be corrupted\n",
                            t.name.c_str(), j, (unsigned long long)t.shape[j], (unsigned long long)MAX_DIM_SIZE);
                    dim_ok = false;
                }
            }
            if (!dim_ok) { fclose(f); f = nullptr; return false; }
            tensors[t.name] = t;
        }
        // GGUF block sizes and bytes per block for each quantization type
        auto block_info = [](uint32_t dtype) -> std::pair<int,int> {
            switch (dtype) {
                case 0: return {1, 4};      // F32
                case 1: return {1, 2};      // F16
                case 2: return {32, 18};    // Q4_0
                case 3: return {32, 20};    // Q4_1
                case 6: return {32, 34};    // Q8_0
                case 7: return {32, 22};    // Q5_0
                case 8: return {32, 24};    // Q5_1
                case 9: return {256, 72};   // Q2_K
                case 10: return {256, 104}; // Q3_K
                case 11: return {256, 144}; // Q4_K
                case 12: return {256, 176}; // Q5_K
                case 13: return {256, 210}; // Q6_K
                case 14: return {256, 292}; // Q8_K
                case 15: return {256, 0};   // unknown
                default: return {32, 0};    // unknown
            }
        };
        
        // Compute actual data offsets — align to 32 bytes
        data_offset = ftell(f);
        data_offset = (data_offset + 31) & ~31;
        for (auto& [name, t] : tensors) {
            t.file_offset = data_offset;
            uint64_t n_elems = 1;
            for (auto s : t.shape) {
                if (s != 0 && n_elems > UINT64_MAX / s) {
                    fprintf(stderr, "[GGUF] FATAL: tensor '%s' numel overflow — file may be corrupted\n", name.c_str());
                    fclose(f); f = nullptr; return false;
                }
                n_elems *= s;
            }
            auto [block_size, bytes_per_block] = block_info(t.dtype);
            if (bytes_per_block == 0) {
                fprintf(stderr, "[GGUF] unknown dtype %u for tensor %s, treating as F32\n", t.dtype, name.c_str());
                bytes_per_block = 4; block_size = 1;
            }
            uint64_t n_blocks = (n_elems + block_size - 1) / block_size;
            data_offset += n_blocks * bytes_per_block;
            // Align to 32 bytes (GGUF alignment)
            data_offset = (data_offset + 31) & ~31;
        }
        return true;
    }

    // Get tensor data, dequantized to float. Returns pointer to internal buffer.
    float* get(const std::string& name, size_t* out_n = nullptr) {
        auto it = tensors.find(name);
        if (it == tensors.end()) return nullptr;
        auto& t = it->second;
        uint64_t n = 1;
        for (auto s : t.shape) {
            if (s != 0 && n > UINT64_MAX / s) {
                fprintf(stderr, "[GGUF] FATAL: tensor '%s' numel overflow — file may be corrupted\n", name.c_str());
                return nullptr;
            }
            n *= s;
        }
        if (out_n) *out_n = n;
        scratch.resize(n);
        fseek(f, t.file_offset, SEEK_SET);
        if (t.dtype == GGUF_TYPE_F32) {
            fread(scratch.data(), 4, n, f);
        } else if (t.dtype == GGUF_TYPE_F16) {
            std::vector<uint16_t> buf(n);
            fread(buf.data(), 2, n, f);
            for (size_t i = 0; i < n; i++) scratch[i] = fp16_to_fp32(buf[i]);
        } else if (t.dtype == GGUF_TYPE_Q4_0) {
            // Q4_0: 2 bytes scale + 16 bytes of 4-bit nibbles per 32 elements
            int blocks = (n + 31) / 32;
            for (int b = 0; b < blocks; b++) {
                uint16_t scale_h; fread(&scale_h, 2, 1, f);
                float scale = fp16_to_fp32(scale_h);
                uint8_t q[16]; fread(q, 1, 16, f);
                for (int j = 0; j < 32 && b*32+j < n; j++) {
                    int8_t v = (j & 1) ? (q[j>>1] >> 4) : (q[j>>1] & 0xf);
                    scratch[b*32+j] = (v - 8) * scale;
                }
            }
        } else if (t.dtype == GGUF_TYPE_Q8_0) {
            // Q8_0: 2 bytes scale + 32 bytes int8 per 32 elements
            int blocks = (n + 31) / 32;
            for (int b = 0; b < blocks; b++) {
                uint16_t scale_h; fread(&scale_h, 2, 1, f);
                float scale = fp16_to_fp32(scale_h);
                int8_t q[32]; fread(q, 1, 32, f);
                for (int j = 0; j < 32 && b*32+j < n; j++)
                    scratch[b*32+j] = q[j] * scale;
            }
        } else if (t.dtype == GGUF_TYPE_Q4_1) {
            // Q4_1: fp16 scale + fp16 min + 16 bytes 4-bit nibbles per 32 elements
            int blocks = (n + 31) / 32;
            for (int b = 0; b < blocks; b++) {
                uint16_t sh, mh; fread(&sh, 2, 1, f); fread(&mh, 2, 1, f);
                float scale = fp16_to_fp32(sh), min = fp16_to_fp32(mh);
                uint8_t q[16]; fread(q, 1, 16, f);
                for (int j = 0; j < 32 && b*32+j < n; j++) {
                    int8_t v = (j & 1) ? (q[j>>1] >> 4) : (q[j>>1] & 0xf);
                    scratch[b*32+j] = v * scale + min;
                }
            }
        } else if (t.dtype == GGUF_TYPE_Q5_0) {
            // Q5_0: 2B fp16 scale + 2B fp16 min + 4B packed high bits + 16B 4-bit low nibbles per 32
            int blocks = (n + 31) / 32;
            for (int b = 0; b < blocks; b++) {
                uint16_t sh, mh; fread(&sh, 2, 1, f); fread(&mh, 2, 1, f);
                float scale = fp16_to_fp32(sh), min = fp16_to_fp32(mh);
                uint32_t qh; fread(&qh, 4, 1, f);
                uint8_t ql[16]; fread(ql, 1, 16, f);
                for (int j = 0; j < 32 && b*32+j < n; j++) {
                    int8_t v = ((qh >> j) & 1) ? 16 : 0;
                    v += (j & 1) ? (ql[j>>1] >> 4) : (ql[j>>1] & 0xf);
                    scratch[b*32+j] = (v - 16) * scale + min;
                }
            }
        } else if (t.dtype == GGUF_TYPE_Q5_1) {
            // Q5_1: fp16 scale + fp16 min + 4B high bits + 16B 4-bit low nibbles per 32
            int blocks = (n + 31) / 32;
            for (int b = 0; b < blocks; b++) {
                uint16_t sh, mh; fread(&sh, 2, 1, f); fread(&mh, 2, 1, f);
                float scale = fp16_to_fp32(sh), min = fp16_to_fp32(mh);
                uint32_t qh; fread(&qh, 4, 1, f);
                uint8_t ql[16]; fread(ql, 1, 16, f);
                for (int j = 0; j < 32 && b*32+j < n; j++) {
                    int8_t v = ((qh >> j) & 1) ? 16 : 0;
                    v += (j & 1) ? (ql[j>>1] >> 4) : (ql[j>>1] & 0xf);
                    scratch[b*32+j] = v * scale + min;
                }
            }
        } else if (t.dtype == GGUF_TYPE_Q8_K) {
            // Q8_K: 2B fp16 dscale + 256 int8 quants
            int blocks = (n + 255) / 256;
            for (int b = 0; b < blocks; b++) {
                uint16_t dsh; fread(&dsh, 2, 1, f);
                float dscale = fp16_to_fp32(dsh);
                int8_t q[256]; fread(q, 1, 256, f);
                for (int j = 0; j < 256 && b*256+j < n; j++)
                    scratch[b*256+j] = q[j] * dscale;
            }
        } else if (t.dtype == GGUF_TYPE_Q6_K) {
            // Q6_K: super-block fp16 dscale + 16 sub-blocks, each: 1B 6-bit sub-scale + 16B 6-bit quants (packed as 3 uint16 per 8)
            int sblocks = (n + 255) / 256;
            for (int sb = 0; sb < sblocks; sb++) {
                uint16_t dsh; fread(&dsh, 2, 1, f);
                float dscale = fp16_to_fp32(dsh);
                uint8_t ql[16*16], qh[16*2]; fread(ql, 1, 16*16, f); fread(qh, 1, 16*2, f);
                for (int sub = 0; sub < 16; sub++) {
                    uint8_t ss = ((qh[sub*2] >> ((sub & 1) * 4)) & 0xf) | ((ql[16*16-16+sub] & 0x3f) << 4);
                    float scl = dscale * (ss > 0 ? ss : 1.0f);
                    for (int j = 0; j < 16 && sb*256+sub*16+j < n; j++) {
                        int idx = sub*16 + j;
                        int8_t v = ((ql[idx] | ((qh[sub*2+(j/8)] & 0x3f) << 8)) >> (j & 7) * 3) & 7;
                        scratch[sb*256+sub*16+j] = (v - 3) * scl;
                    }
                }
            }
        } else if (t.dtype == GGUF_TYPE_Q4_K) {
            // Q4_K: super-block fp16 dscale + fp16 dmin + packed sub-scales + 16B 4-bit nibbles per sub-block
            int sblocks = (n + 255) / 256;
            for (int sb = 0; sb < sblocks; sb++) {
                uint16_t dsh, dmh; fread(&dsh, 2, 1, f); fread(&dmh, 2, 1, f);
                float dscale = fp16_to_fp32(dsh), dmin = fp16_to_fp32(dmh);
                uint8_t ss[12]; fread(ss, 1, 12, f);
                uint8_t sm[4]; fread(sm, 1, 4, f);
                uint8_t q[16*16]; fread(q, 1, 16*16, f);
                for (int sub = 0; sub < 16; sub++) {
                    int si = sub * 3 / 2, shift = (sub & 1) * 4;
                    uint8_t sub_scale = (si < 12) ? ((ss[si] >> shift) & 0xf) : 0;
                    sub_scale |= ((sm[sub/2] >> ((sub & 1) * 4)) & 0xf) << 4;
                    float scl = dscale * sub_scale, mn = dmin * ((sm[sub/2] >> ((sub & 1) * 4)) & 0xf);
                    for (int j = 0; j < 16 && sb*256+sub*16+j < n; j++) {
                        int idx = sub*16 + j;
                        int8_t v = (j & 1) ? (q[idx>>1] >> 4) : (q[idx>>1] & 0xf);
                        scratch[sb*256+sub*16+j] = v * scl - mn;
                    }
                }
            }
        } else if (t.dtype == GGUF_TYPE_Q3_K) {
            // Q3_K: super-block fp16 dscale + packed sub-scales + 2B high quants + 16B 2-bit low quants per sub-block
            int sblocks = (n + 255) / 256;
            for (int sb = 0; sb < sblocks; sb++) {
                uint16_t dsh; fread(&dsh, 2, 1, f);
                float dscale = fp16_to_fp32(dsh);
                uint8_t ss_high[2]; fread(ss_high, 1, 2, f);
                uint8_t qh[16*2], ql[16*16]; fread(qh, 1, 16*2, f); fread(ql, 1, 16*16, f);
                for (int sub = 0; sub < 16; sub++) {
                    uint8_t ss_low = (sub & 1) ? (qh[sub/2] >> 4) : (qh[sub/2] & 0xf);
                    uint8_t ss = ((ss_high[sub/8] >> ((sub & 7) * 3)) & 7) | (ss_low << 4);
                    float scl = dscale * (ss > 0 ? ss : 1.0f);
                    for (int j = 0; j < 16 && sb*256+sub*16+j < n; j++) {
                        int idx = sub*16 + j;
                        int8_t vh = ((qh[sub*2+(j/8)] >> ((j & 7) * 3)) & 7) << 2;
                        int8_t vl = (ql[idx] >> ((j & 3) * 2)) & 3;
                        scratch[sb*256+sub*16+j] = (vh + vl - 9) * scl;
                    }
                }
            }
        } else if (t.dtype == GGUF_TYPE_Q2_K) {
            // Q2_K: super-block fp16 dscale + fp16 dmin + packed sub-scales + 16B 2-bit quants per sub-block
            int sblocks = (n + 255) / 256;
            for (int sb = 0; sb < sblocks; sb++) {
                uint16_t dsh, dmh; fread(&dsh, 2, 1, f); fread(&dmh, 2, 1, f);
                float dscale = fp16_to_fp32(dsh), dmin = fp16_to_fp32(dmh);
                uint8_t ss[16]; fread(ss, 1, 16, f);
                uint8_t q[16*16]; fread(q, 1, 16*16, f);
                for (int sub = 0; sub < 16; sub++) {
                    float scl = dscale * (ss[sub] & 0xf), mn = dmin * (ss[sub] >> 4);
                    for (int j = 0; j < 16 && sb*256+sub*16+j < n; j++) {
                        int idx = sub*16 + j;
                        int8_t v = (q[idx] >> ((j & 3) * 2)) & 3;
                        scratch[sb*256+sub*16+j] = v * scl - mn;
                    }
                }
            }
        } else if (t.dtype == GGUF_TYPE_Q5_K) {
            // Q5_K: super-block fp16 dscale + fp16 dmin + packed sub-scales + 2B high + 16B 4-bit low per sub-block
            int sblocks = (n + 255) / 256;
            for (int sb = 0; sb < sblocks; sb++) {
                uint16_t dsh, dmh; fread(&dsh, 2, 1, f); fread(&dmh, 2, 1, f);
                float dscale = fp16_to_fp32(dsh), dmin = fp16_to_fp32(dmh);
                uint8_t ss[12]; fread(ss, 1, 12, f);
                uint8_t sm[4]; fread(sm, 1, 4, f);
                uint8_t qh[16*2], ql[16*16]; fread(qh, 1, 16*2, f); fread(ql, 1, 16*16, f);
                for (int sub = 0; sub < 16; sub++) {
                    int si = sub * 3 / 2, shift = (sub & 1) * 4;
                    uint8_t sub_scale = (si < 12) ? ((ss[si] >> shift) & 0xf) : 0;
                    sub_scale |= ((sm[sub/2] >> ((sub & 1) * 4)) & 0xf) << 4;
                    float scl = dscale * sub_scale, mn = dmin * ((sm[sub/2] >> ((sub & 1) * 4)) & 0xf);
                    for (int j = 0; j < 16 && sb*256+sub*16+j < n; j++) {
                        int idx = sub*16 + j;
                        int8_t vh = (qh[sub*2+(j/8)] >> (j & 7)) & 1;
                        int8_t vl = (j & 1) ? (ql[idx>>1] >> 4) : (ql[idx>>1] & 0xf);
                        scratch[sb*256+sub*16+j] = (vh * 16 + vl) * scl - mn;
                    }
                }
            }
        }
        return scratch.data();
    }

    void close() { if (f) fclose(f); }
};

#include <cstdio>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>

// ── SafeTensors reader ──────────────────────────────────────────────────────
// Reads HuggingFace SafeTensors format (torchtune / HF output) directly.
// No Python dependency. Parses header JSON + reads raw tensor data.
// Supports F32, F16, BF16 dtypes.

#include <cstring>
#include <chrono>

struct SafeTensorsReader {
    FILE* f = nullptr;
    size_t data_start = 0;
    std::unordered_map<std::string, std::pair<size_t, size_t>> tensors; // name -> (offset, nbytes)
    std::unordered_map<std::string, std::vector<uint64_t>> shapes;
    std::unordered_map<std::string, std::string> dtypes;
    std::vector<float> scratch;
    
    ~SafeTensorsReader() { close(); }
    
    bool open(const std::string& path) {
        f = fopen(path.c_str(), "rb");
        if (!f) return false;
        uint64_t hdr_size;
        if (fread(&hdr_size, 8, 1, f) != 1) { fclose(f); return false; }
        std::string json_str(hdr_size, '\0');
        if (fread(&json_str[0], 1, hdr_size, f) != hdr_size) { fclose(f); return false; }
        
        // Parse tensor entries from JSON
        size_t pos = 0;
        while ((pos = json_str.find('"', pos)) != std::string::npos) {
            size_t ks = pos + 1, ke = json_str.find('"', ks);
            if (ke == std::string::npos) break;
            std::string key = json_str.substr(ks, ke - ks);
            pos = ke + 1;
            if (key == "__metadata__") continue;
            
            auto dq = json_str.find("\"dtype\"", pos);
            if (dq == std::string::npos) continue;
            auto vs = json_str.find('"', dq + 7) + 1, ve = json_str.find('"', vs);
            dtypes[key] = json_str.substr(vs, ve - vs);
            
            auto shq = json_str.find("\"shape\"", ve);
            if (shq == std::string::npos) continue;
            auto sb = json_str.find('[', shq), eb = json_str.find(']', sb);
            std::vector<uint64_t> sh;
            for (size_t sp = sb + 1; sp < eb;) {
                while (sp < eb && (json_str[sp] == ' ' || json_str[sp] == ',')) sp++;
                if (sp >= eb) break;
                sh.push_back(std::stoull(&json_str[sp]));
                while (sp < eb && isdigit(json_str[sp])) sp++;
            }
            shapes[key] = sh;
            
            auto doq = json_str.find("\"data_offsets\"", eb);
            if (doq == std::string::npos) continue;
            auto dob = json_str.find('[', doq), doeb = json_str.find(']', dob);
            std::string ds = json_str.substr(dob + 1, doeb - dob - 1);
            uint64_t os, oe; sscanf(ds.c_str(), "%lu, %lu", &os, &oe);
            tensors[key] = {os, oe - os};
        }
        data_start = 8 + hdr_size;
        return !tensors.empty();
    }
    
    float* get(const std::string& name, size_t* out_n = nullptr) {
        auto it = tensors.find(name);
        if (it == tensors.end()) return nullptr;
        auto [offset, nbytes] = it->second;
        auto dit = dtypes.find(name);
        std::string dtype = (dit != dtypes.end()) ? dit->second : "F32";
        auto sit = shapes.find(name);
        size_t n = 1;
        if (sit != shapes.end())
            for (auto d : sit->second) n *= d;
        if (out_n) *out_n = n;
        
        scratch.resize(n);
        fseek(f, data_start + offset, SEEK_SET);
        
        if (dtype == "F32" || dtype == "float32") {
            fread(scratch.data(), 4, n, f);
        } else if (dtype == "F16" || dtype == "float16") {
            std::vector<uint16_t> buf(n);
            fread(buf.data(), 2, n, f);
            for (size_t i = 0; i < n; i++) scratch[i] = fp16_to_fp32(buf[i]);
        } else if (dtype == "BF16" || dtype == "bfloat16") {
            std::vector<uint16_t> buf(n);
            fread(buf.data(), 2, n, f);
            for (size_t i = 0; i < n; i++) {
                uint32_t f32 = (uint32_t)buf[i] << 16;
                memcpy(&scratch[i], &f32, 4);
            }
        }
        return scratch.data();
    }
    
    void close() { if (f) { fclose(f); f = nullptr; } }
};

// ── Generic CPU Backend ──────────────────────────────────────────────────────
struct GenericBackend : Backend {
    ModelConfig cfg;
    std::vector<float> embed, final_norm, output_weight;
    std::vector<std::vector<float>> layer_w;  // flat per-layer weights
    std::vector<std::vector<float>> k_cache, v_cache; // KV cache [n_layers][max_seq * n_kv * hd]
    int pos = 0;
    std::vector<float> logits_buf;

    // Per-layer weight indices. bq/bk/bv are optional QKV biases (Qwen2 and
    // some other architectures use biased attention projections, unlike
    // Llama) — SIZE_MAX means "not present in this model", distinct from a
    // legitimate index 0 into flat_weights.
    // q_norm/k_norm: optional per-head QK RMSNorm (Qwen3 and others), applied
    // to each head's head_dim-sized slice with a shared [head_dim] weight,
    // right after QKV bias and before RoPE.
    // moe_*: present only on MoE layers (cfg.n_experts > 0). w1/w2/w3 are
    // unused for such layers — the FFN routes through the expert tensors
    // instead. moe_gate_exps/moe_up_exps/moe_down_exps are flat [n_expert *
    // n_ff * hidden]-sized blocks (GGUF's stacked "_exps" 3D tensors); expert
    // e's slice starts at index e * n_ff * hidden and is row-major [n_ff,
    // hidden] — i.e. the exact same layout matmul() already expects for the
    // dense case, just offset per-expert.
    struct LayerW {
        size_t wq, wk, wv, wo, rms_attn, rms_ffn;
        size_t w1 = SIZE_MAX, w2 = SIZE_MAX, w3 = SIZE_MAX;
        size_t bq = SIZE_MAX, bk = SIZE_MAX, bv = SIZE_MAX;
        size_t q_norm = SIZE_MAX, k_norm = SIZE_MAX;
        size_t moe_gate_inp = SIZE_MAX, moe_gate_exps = SIZE_MAX, moe_up_exps = SIZE_MAX, moe_down_exps = SIZE_MAX;
    };
    std::vector<LayerW> layers;

    GenericBackend() { type = BackendType::GENERIC; name = "Generic CPU (GGUF)"; }

    void load_weights(const std::string& base) {
        // Weights stored as flat float vectors: model_layers_N_name.bin
        // Read by the existing W() macro pattern
        auto W = [&](const std::string& name) -> std::vector<float> {
            std::string path = base + "/" + name;
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f) return {};
            size_t n = f.tellg() / sizeof(float); f.seekg(0);
            std::vector<float> d(n); f.read((char*)d.data(), n * sizeof(float));
            return d;
        };
        embed = W("model_embed_tokens_weight.bin");
        final_norm = W("model_norm_weight.bin");

        int L = cfg.n_layers;
        layers.resize(L);
        for (int i = 0; i < L; i++) {
            std::string p = "model_layers_" + std::to_string(i) + "_";
            layers[i] = {
                push(W(p + "self_attn_q_proj.weight")),
                push(W(p + "self_attn_k_proj.weight")),
                push(W(p + "self_attn_v_proj.weight")),
                push(W(p + "self_attn_o_proj.weight")),
                push(W(p + "mlp_gate_proj.weight")),
                push(W(p + "mlp_up_proj.weight")),
                push(W(p + "mlp_down_proj.weight")),
                push(W(p + "input_layernorm.weight")),
                push(W(p + "post_attention_layernorm.weight")),
            };
        }
    }

    size_t push(std::vector<float>&& v) {
        size_t idx = flat_weights.size();
        flat_weights.insert(flat_weights.end(), v.begin(), v.end());
        return idx;
    }
    std::vector<float> flat_weights;
    float* w(size_t idx) { return flat_weights.data() + idx; }

    bool init(const ModelConfig& model_cfg, const std::string& weights_dir) override {
        cfg = model_cfg;
        printf("Generic: initializing %s (%d layers, %d hidden, %d heads)\n",
               cfg.model_name.c_str(), cfg.n_layers, cfg.hidden, cfg.n_heads);

        // Try loading weights from a GGUF file first
        bool loaded = false;
        if (!cfg.model_path.empty()) {
            std::string gguf_path = cfg.model_path;
            // If model_path is a directory, look for a .gguf inside
            struct stat st;
            if (stat(gguf_path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
                // Find first .gguf in the directory
                DIR* d = opendir(gguf_path.c_str());
                if (d) {
                    struct dirent* e;
                    while ((e = readdir(d)) != nullptr) {
                        std::string n(e->d_name);
                        if (n.size() > 5 && n.substr(n.size()-5) == ".gguf") {
                            gguf_path = gguf_path + "/" + n;
                            break;
                        }
                    }
                    closedir(d);
                }
            }
            printf("Generic: trying GGUF path: %s\n", gguf_path.c_str());
            loaded = load_gguf(gguf_path);
        }
        if (!loaded) {
            // Fall back: old .bin format
            load_weights(weights_dir);
            loaded = !embed.empty();
        }
        if (!loaded) {
            fprintf(stderr, "Generic: could not load weights from %s\n", weights_dir.c_str());
            return false;
        }
        logits_buf.resize(cfg.vocab);
        k_cache.resize(cfg.n_layers);
        v_cache.resize(cfg.n_layers);
        for (auto& k : k_cache) k.resize(cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim);
        for (auto& v : v_cache) v.resize(cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim);
        initialized = true;
        return true;
    }

    bool load_gguf(const std::string& path) {
        ModelConfig hdr_cfg;
        if (!read_gguf_header(path, hdr_cfg)) return false;
        fprintf(stderr, "load_gguf: %s, %d layers, %d hidden\n", hdr_cfg.model_name.c_str(), hdr_cfg.n_layers, hdr_cfg.hidden);
        
        int H = hdr_cfg.hidden_size, L = hdr_cfg.n_layers, NH = hdr_cfg.n_heads;
        int NKV = hdr_cfg.n_kv_heads, HD = hdr_cfg.head_dim;
        int FF = hdr_cfg.intermediate_size;
        [[maybe_unused]] int V = hdr_cfg.vocab_size;
        
        auto load = [&](const std::string& name, std::vector<float>& dst, size_t expected) -> bool {
            std::vector<float> buf;
            size_t n = 0;
            if (!read_gguf_tensor(path, name, buf, &n)) return false;
            if (n != expected) { fprintf(stderr, "  %s: expected %zu, got %zu\n", name.c_str(), expected, n); return false; }
            dst = std::move(buf);
            return true;
        };
        
        // Embedding
        int real_vocab = read_gguf_vocab(path);
        if (real_vocab > 0) cfg.vocab = cfg.vocab_size = real_vocab;
        load("token_embd.weight", embed, (size_t)real_vocab * H);
        
        // Final norm
        load("output_norm.weight", final_norm, H);

        // LM head — optional. Many GGUF exports omit it entirely when the
        // source model ties embeddings (tie_word_embeddings: true); when
        // present, it's a genuinely different matrix from token_embd.weight
        // and using the tied embedding instead silently produces wrong
        // logits (issue #319). output_weight stays empty when absent, and
        // forward() falls back to the tied embedding in that case.
        load("output.weight", output_weight, (size_t)real_vocab * H);

        
        // Per-layer weights
        layers.resize(L);
        flat_weights.clear();
        
        auto load_tensor = [&](const std::string& name, size_t expected) -> size_t {
            std::vector<float> buf;
            size_t n = 0;
            if (!read_gguf_tensor(path, name, buf, &n)) return 0;
            if (n != expected) { fprintf(stderr, "  %s: expected %zu, got %zu\n", name.c_str(), expected, n); return 0; }
            size_t idx = flat_weights.size();
            flat_weights.insert(flat_weights.end(), buf.begin(), buf.end());
            return idx;
        };
        // Like load_tensor, but returns SIZE_MAX (not 0) when the tensor
        // simply isn't present — for genuinely optional tensors (QKV bias),
        // where "absent" and "present at index 0" must stay distinguishable.
        auto load_tensor_optional = [&](const std::string& name, size_t expected) -> size_t {
            std::vector<float> buf;
            size_t n = 0;
            if (!read_gguf_tensor(path, name, buf, &n)) return SIZE_MAX;
            if (n != expected) { fprintf(stderr, "  %s: expected %zu, got %zu\n", name.c_str(), expected, n); return SIZE_MAX; }
            size_t idx = flat_weights.size();
            flat_weights.insert(flat_weights.end(), buf.begin(), buf.end());
            return idx;
        };

        int NE = cfg.n_experts;
        for (int i = 0; i < L; i++) {
            std::string p = "blk." + std::to_string(i) + ".";
            LayerW lw;
            lw.rms_attn = load_tensor(p + "attn_norm.weight", H);
            lw.rms_ffn  = load_tensor(p + "ffn_norm.weight", H);
            lw.wq = load_tensor(p + "attn_q.weight", NH*HD*H);
            lw.wk = load_tensor(p + "attn_k.weight", NKV*HD*H);
            lw.wv = load_tensor(p + "attn_v.weight", NKV*HD*H);
            lw.wo = load_tensor(p + "attn_output.weight", H*NH*HD);
            if (NE > 0) {
                // MoE layer: no dense ffn_gate/up/down — route through the
                // stacked per-expert "_exps" tensors instead.
                lw.moe_gate_inp  = load_tensor(p + "ffn_gate_inp.weight", (size_t)NE*H);
                lw.moe_gate_exps = load_tensor(p + "ffn_gate_exps.weight", (size_t)NE*FF*H);
                lw.moe_up_exps   = load_tensor(p + "ffn_up_exps.weight", (size_t)NE*FF*H);
                lw.moe_down_exps = load_tensor(p + "ffn_down_exps.weight", (size_t)NE*H*FF);
            } else {
                lw.w1 = load_tensor(p + "ffn_gate.weight", FF*H);
                lw.w2 = load_tensor(p + "ffn_up.weight", FF*H);
                lw.w3 = load_tensor(p + "ffn_down.weight", H*FF);
            }
            lw.bq = load_tensor_optional(p + "attn_q.bias", NH*HD);
            lw.bk = load_tensor_optional(p + "attn_k.bias", NKV*HD);
            lw.bv = load_tensor_optional(p + "attn_v.bias", NKV*HD);
            lw.q_norm = load_tensor_optional(p + "attn_q_norm.weight", HD);
            lw.k_norm = load_tensor_optional(p + "attn_k_norm.weight", HD);
            layers[i] = lw;
        }
        {
            int with_bias = 0, with_qknorm = 0;
            for (auto& lw : layers) {
                if (lw.bq != SIZE_MAX) with_bias++;
                if (lw.q_norm != SIZE_MAX) with_qknorm++;
            }
            if (with_bias > 0) printf("Generic: %d/%d layers have biased QKV projections\n", with_bias, L);
            if (with_qknorm > 0) printf("Generic: %d/%d layers have Q/K-norm\n", with_qknorm, L);
            if (NE > 0) printf("Generic: MoE model — %d experts, %d used per token\n", NE, cfg.num_experts_top);
        }
        
        printf("Generic: loaded %zu layers, embed=%zu, final_norm=%zu, lm_head=%s\n",
               layers.size(), embed.size(), final_norm.size(),
               output_weight.empty() ? "tied" : "untied");
        return !embed.empty() && layers.size() == (size_t)L;
    }

    size_t push_vec(float* data, size_t n) {
        if (!data) return 0;
        size_t idx = flat_weights.size();
        flat_weights.insert(flat_weights.end(), data, data + n);
        return idx;
    }

    bool reset() override {
        pos = 0;
        for (auto& k : k_cache) std::fill(k.begin(), k.end(), 0.0f);
        for (auto& v : v_cache) std::fill(v.begin(), v.end(), 0.0f);
        return true;
    }

    static void rmsnorm(float* o, const float* x, const float* w, int n, float eps) {
        float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
        float r = 1.0f / sqrtf(ss / n + eps);
        for (int i = 0; i < n; i++) o[i] = x[i] * r * w[i];
    }

    // NeoX-style (half-split) RoPE — the convention GGUF/llama.cpp-family
    // models (Llama, Qwen, Mistral, ...) actually use: pairs element i with
    // i+rot_dim/2, not adjacent elements (i, i+1). Cross-checked against
    // ZINC's shaders (src/shaders/rope_fused.comp and siblings, all
    // independently confirm half_rot = rope_dim/2 pairing) since ZINC is
    // independently verified to produce coherent output on these same
    // models — this file's previous adjacent-pair version was the GPT-J
    // convention, wrong for this model family, and produced incoherent
    // (real-vocabulary but semantically scrambled) output as a result.
    static void rope(float* q, float* k, int pos, int n_heads, int n_kv, int hd, int rot_dim, float theta) {
        int half = rot_dim / 2;
        for (int h = 0; h < n_heads; h++) {
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / powf(theta, (2.0f * i) / (float)rot_dim);
                float t = pos * freq;
                float cosv = cosf(t), sinv = sinf(t);
                int i0 = h * hd + i, i1 = h * hd + i + half;
                float q0 = q[i0], q1 = q[i1];
                q[i0] = q0 * cosv - q1 * sinv;
                q[i1] = q0 * sinv + q1 * cosv;
            }
        }
        for (int h = 0; h < n_kv; h++) {
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / powf(theta, (2.0f * i) / (float)rot_dim);
                float t = pos * freq;
                float cosv = cosf(t), sinv = sinf(t);
                int i0 = h * hd + i, i1 = h * hd + i + half;
                float k0 = k[i0], k1 = k[i1];
                k[i0] = k0 * cosv - k1 * sinv;
                k[i1] = k0 * sinv + k1 * cosv;
            }
        }
    }

    static void matmul(float* out, const float* in, const float* w, int M, int K) {
        for (int i = 0; i < M; i++) {
            float s = 0;
            for (int j = 0; j < K; j++) s += in[j] * w[i * (size_t)K + j];
            out[i] = s;
        }
    }

    static void silu(float* out, const float* gate, const float* up, int n) {
        for (int i = 0; i < n; i++) {
            float g = gate[i];
            out[i] = (g / (1.0f + expf(-g))) * up[i];
        }
    }

    // GELU activation (tanh approximation): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    static void gelu(float* out, const float* x, int n) {
        const float c = 0.7978845608f; // sqrt(2/pi)
        for (int i = 0; i < n; i++) {
            float v = x[i];
            float x3 = v * v * v;
            float inner = c * (v + 0.044715f * x3);
            out[i] = 0.5f * v * (1.0f + tanhf(inner));
        }
    }

    // GeGLU: gelu(gate) * up — used by Gemma
    static void geglu(float* out, const float* gate, const float* up, int n) {
        gelu(out, gate, n);
        for (int i = 0; i < n; i++) out[i] *= up[i];
    }

    // Squared ReLU GLU: relu(gate)^2 * up — used by Phi
    static void squared_relu_glu(float* out, const float* gate, const float* up, int n) {
        for (int i = 0; i < n; i++) {
            float g = gate[i];
            float r = g > 0.0f ? g : 0.0f;
            out[i] = r * r * up[i];
        }
    }

    // Architecture-specific FFN activation dispatch
    static void ffn_activate(float* out, const float* gate, const float* up, int n, rcpp_arch_t arch) {
        switch (arch) {
            case RCPP_ARCH_GEMMA:
                // GeGLU: gelu(gate) * up
                geglu(out, gate, up, n);
                break;
            case RCPP_ARCH_PHI:
                // Squared ReLU GLU: relu(gate)^2 * up
                squared_relu_glu(out, gate, up, n);
                break;
            default:
                // SwiGLU: silu(gate) * up — Llama, Mistral, Qwen2, Qwen3, BitNet, fallback
                silu(out, gate, up, n);
                break;
        }
    }

    static void softmax(float* x, int n) {
        float mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
        float sum = 0; for (int i = 0; i < n; i++) sum += expf(x[i] - mx);
        float inv = 1.0f / (sum + 1e-10f);
        for (int i = 0; i < n; i++) x[i] = expf(x[i] - mx) * inv;
    }

    int generate(int token_id) override {
        if (!initialized) return -1;
        return forward(token_id);
    }

    bool forward(int token_id, float* hidden_out) override {
        int tok = forward(token_id);
        if (hidden_out) *hidden_out = 0.0f;
        return tok >= 0;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        return false;  // not implemented — use generate() instead
    }

    int forward(int token) {
        std::vector<float> x0(cfg.hidden);
        for (int i = 0; i < cfg.hidden; i++) x0[i] = embed[token * (size_t)cfg.hidden + i];
        return forward_embed(x0.data());
    }

    // Same transformer body as forward(int), but takes a precomputed
    // embedding vector directly instead of doing a token_embd lookup —
    // the splice point for injecting vision embeddings (mm.2 output) at
    // image-placeholder positions instead of a text token's row.
    int forward_embed(const float* x_in) override {
        int H = cfg.hidden, NH = cfg.n_heads, NKV = cfg.n_kv_heads, HD = cfg.head_dim;
        int GQA = NH / NKV, FF = cfg.intermediate_size, V = cfg.vocab;
        float eps = cfg.rms_norm_eps, theta = cfg.rope_theta;
        int rot_dim = cfg.head_dim;  // full RoPE by default

        std::vector<float> x(H), x2(H), q(NH*HD), k(NKV*HD), v(NKV*HD), scores(HD);
        std::vector<float> att(NH*HD);
        std::vector<float> gate_up(FF*2);

        for (int i = 0; i < H; i++) x[i] = x_in[i];

        for (int il = 0; il < cfg.n_layers; il++) {
            auto& l = layers[il];
            int kv_begin = pos * NKV * HD;

            // RMSNorm → QKV
            rmsnorm(x2.data(), x.data(), w(l.rms_attn), H, eps);
            matmul(q.data(), x2.data(), w(l.wq), NH*HD, H);
            matmul(k.data(), x2.data(), w(l.wk), NKV*HD, H);
            matmul(v.data(), x2.data(), w(l.wv), NKV*HD, H);

            // Optional QKV bias (Qwen2 and others use biased attention
            // projections; absent for architectures like Llama).
            if (l.bq != SIZE_MAX) { float* b = w(l.bq); for (int i = 0; i < NH*HD; i++) q[i] += b[i]; }
            if (l.bk != SIZE_MAX) { float* b = w(l.bk); for (int i = 0; i < NKV*HD; i++) k[i] += b[i]; }
            if (l.bv != SIZE_MAX) { float* b = w(l.bv); for (int i = 0; i < NKV*HD; i++) v[i] += b[i]; }

            // Optional per-head QK-norm (Qwen3 and others): RMSNorm applied
            // independently to each head's head_dim-sized slice with a
            // shared [head_dim] weight. Must happen before RoPE.
            if (l.q_norm != SIZE_MAX) {
                float* qn = w(l.q_norm);
                for (int h = 0; h < NH; h++) rmsnorm(&q[h*HD], &q[h*HD], qn, HD, eps);
            }
            if (l.k_norm != SIZE_MAX) {
                float* kn = w(l.k_norm);
                for (int h = 0; h < NKV; h++) rmsnorm(&k[h*HD], &k[h*HD], kn, HD, eps);
            }

            // RoPE
            rope(q.data(), k.data(), pos, NH, NKV, HD, rot_dim, theta);

            // KV cache
            memcpy(&k_cache[il][kv_begin], k.data(), NKV * HD * sizeof(float));
            memcpy(&v_cache[il][kv_begin], v.data(), NKV * HD * sizeof(float));

            // Attention: GQA
            std::fill(att.begin(), att.end(), 0.0f);
            for (int h = 0; h < NH; h++) {
                int kv_h = h / GQA;
                float* Q = &q[h * HD];
                // Score over all past positions
                for (int t = 0; t <= pos; t++) {
                    float* K = &k_cache[il][t * NKV * HD + kv_h * HD];
                    float s = 0;
                    for (int d = 0; d < HD; d++) s += Q[d] * K[d];
                    scores[t] = s / sqrtf((float)HD);
                }
                softmax(scores.data(), pos + 1);
                // Weighted sum of V
                for (int d = 0; d < HD; d++) {
                    float sum = 0;
                    for (int t = 0; t <= pos; t++) {
                        float* V = &v_cache[il][t * NKV * HD + kv_h * HD];
                        sum += scores[t] * V[d];
                    }
                    att[h * HD + d] = sum;
                }
            }

            // O proj
            matmul(x2.data(), att.data(), w(l.wo), H, NH*HD);
            // Residual
            for (int i = 0; i < H; i++) x[i] += x2[i];

            // FFN: RMSNorm → gate/up → activation (arch-specific) → down → residual (dense), or
            // RMSNorm → router top-k → per-expert gate/up/activation/down,
            // weighted sum → residual (MoE).
            rmsnorm(x2.data(), x.data(), w(l.rms_ffn), H, eps);
            if (l.moe_gate_inp != SIZE_MAX) {
                int NE = cfg.n_experts, NEU = cfg.num_experts_top;
                std::vector<float> router_probs(NE);
                matmul(router_probs.data(), x2.data(), w(l.moe_gate_inp), NE, H);
                softmax(router_probs.data(), NE);

                // Top-k expert selection by router probability (descending),
                // then renormalize just the selected weights to sum to 1 —
                // matches llama.cpp's build_moe_ffn with norm_w=true (the
                // convention used by Qwen2-MoE/Qwen3-MoE/Mixtral).
                std::vector<int> idx(NE);
                for (int e = 0; e < NE; e++) idx[e] = e;
                std::partial_sort(idx.begin(), idx.begin() + NEU, idx.end(),
                                   [&](int a, int b) { return router_probs[a] > router_probs[b]; });
                float wsum = 0.0f;
                for (int t = 0; t < NEU; t++) wsum += router_probs[idx[t]];
                if (wsum < 6.103515625e-5f) wsum = 6.103515625e-5f; // match ggml's clamp

                std::vector<float> ffn_acc(H, 0.0f);
                std::vector<float> gate_buf(FF), up_buf(FF), silu_buf(FF), down_buf(H);
                for (int t = 0; t < NEU; t++) {
                    int e = idx[t];
                    float we = router_probs[e] / wsum;
                    float* wg = w(l.moe_gate_exps) + (size_t)e * FF * H;
                    float* wu = w(l.moe_up_exps)   + (size_t)e * FF * H;
                    float* wd = w(l.moe_down_exps) + (size_t)e * H * FF;
                    matmul(gate_buf.data(), x2.data(), wg, FF, H);
                    matmul(up_buf.data(), x2.data(), wu, FF, H);
                    ffn_activate(silu_buf.data(), gate_buf.data(), up_buf.data(), FF, cfg.arch);
                    matmul(down_buf.data(), silu_buf.data(), wd, H, FF);
                    for (int i = 0; i < H; i++) ffn_acc[i] += we * down_buf[i];
                }
                for (int i = 0; i < H; i++) x[i] += ffn_acc[i];
            } else {
                matmul(gate_up.data(), x2.data(), w(l.w1), FF, H);
                matmul(&gate_up[FF], x2.data(), w(l.w2), FF, H);
                std::vector<float> silu_buf(FF);
                ffn_activate(silu_buf.data(), gate_up.data(), &gate_up[FF], FF, cfg.arch);
                matmul(x2.data(), silu_buf.data(), w(l.w3), H, FF);
                for (int i = 0; i < H; i++) x[i] += x2[i];
            }
        }

        // Final RMSNorm
        rmsnorm(x2.data(), x.data(), final_norm.data(), H, eps);

        // LM head — untied output.weight when the model has one, else tied embedding.
        const float* lm_head = output_weight.empty() ? embed.data() : output_weight.data();
        matmul(logits_buf.data(), x2.data(), lm_head, V, H);

        pos++;

        // Argmax
        int best = 0; float bestv = logits_buf[0];
        for (int i = 1; i < V; i++) {
            if (logits_buf[i] > bestv) { bestv = logits_buf[i]; best = i; }
        }
        return best;
    }

    void destroy() override { initialized = false; }

    ~GenericBackend() override { destroy(); }

    float benchmark(int tokens = 10) override {
        if (!initialized) return 0;
        reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        int tok = 100;
        for (int i = 0; i < tokens; i++) tok = forward(tok);
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return ms / tokens;
    }
};

Backend* create_generic_backend() { return new GenericBackend(); }
