// GGUF model loader — pure C++ + HIP, no Python.
// Reads GGUF format models and uploads weights to GPU for inference.
#include "rocm_cpp/bitnet_model.h"
#include "block_scaled_ternary.h"
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

#define HIP_CHECK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP %d %s:%d\n",_s,__FILE__,__LINE__); return RCPP_HIP_ERROR;}} while(0)
#define RC_FAIL(s) do { fprintf(stderr,"[gguf] %s\n",s); return RCPP_INVALID_ARG; } while(0)

namespace {

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
    GGUF_TYPE_BLOCK_SCALED_TERNARY = 16,
};

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
        case GGUF_TYPE_BLOCK_SCALED_TERNARY: return BST_BLOCK_K;
        default: return 0;
    }
}

int gguf_block_bytes(uint32_t dtype) {
    switch (dtype) {
        case GGUF_TYPE_F32:  return 4;
        case GGUF_TYPE_F16:  return 2;
        case GGUF_TYPE_Q4_0: return 18;
        case GGUF_TYPE_Q4_1: return 20;
        case GGUF_TYPE_Q8_0: return 34;
        case GGUF_TYPE_Q5_0: return 22;
        case GGUF_TYPE_Q5_1: return 24;
        // Real block_q2_K is scales[16]+qs[64]+d(2)+dmin(2) = 84 bytes, and
        // block_q3_K is hmask[32]+qs[64]+scales[12]+d(2) = 110 bytes — this
        // table previously had both wrong (72/104), which would misalign
        // every read for a K-quant format not even wired up yet at the time.
        case GGUF_TYPE_Q2_K: return 84;
        case GGUF_TYPE_Q3_K: return 110;
        case GGUF_TYPE_Q4_K: return 144;
        case GGUF_TYPE_Q5_K: return 176;
        case GGUF_TYPE_Q6_K: return 210;
        case GGUF_TYPE_Q8_K: return 292;
        case GGUF_TYPE_BLOCK_SCALED_TERNARY: return BST_BLOCK_BYTES;
        default: return 0;
    }
}

// ── K-quant dequantization ──────────────────────────────────────────────
// Reference: llama.cpp ggml-quants.c block_q4_K / block_q6_K / block_q8_K

// Reads a float16 value from a byte pointer
// Was `(uint32_t)bits << 16` reinterpreted as f32 — correct for widening a
// bfloat16 (whose bits ARE float32's truncated upper half), but every caller
// here reads real IEEE754 half-precision float16 (the K-quant super-block
// d/dmin scale fields per the GGUF/GGML spec), which has a different
// exponent width/bias (5 bits, bias 15) than float32 (8 bits, bias 127) —
// the bit-shift silently produced the wrong value for every K-quant block
// in every model loaded through this file. See issue #473.
static inline float read_f16(const uint8_t* p) {
    uint16_t h; memcpy(&h, p, 2);
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float sign = s ? -1.0f : 1.0f;
    if (e == 0) return sign * (float)m * 5.9604644775390625e-08f;  // subnormal: m * 2^-24
    if (e == 31) return m ? NAN : sign * INFINITY;
    return sign * (1.0f + (float)m / 1024.0f) * powf(2.0f, (float)((int)e - 15));
}

// Shared 6-bit scale/min unpacking scheme used by both Q4_K and Q5_K: 8
// (scale,min) pairs packed into 12 bytes (12 bytes * 8 bits / 8 pairs / 2
// values = 6 bits each).
static inline void k_get_scale_min(const uint8_t scales[12], int j, uint8_t& sc, uint8_t& m) {
    if (j < 4) { sc = scales[j] & 63; m = scales[j + 4] & 63; }
    else { sc = (uint8_t)((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
           m = (uint8_t)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)); }
}

static bool dequant_q4_k(const uint8_t* bd, float* out, int count) {
    // Q4_K: 256-elem superblock, 144 bytes
    // d(2) + dmin(2) + scales(12) + qs(128)
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        float d = read_f16(p); p += 2;
        float dmin = read_f16(p); p += 2;
        uint8_t scales[12]; memcpy(scales, p, 12); p += 12;
        uint8_t qs[128]; memcpy(qs, p, 128); p += 128;

        int base = b * BS;
        int is = 0; const uint8_t* q = qs;
        for (int off = 0; off < BS && base + off < count; off += 64) {
            uint8_t sc, m;
            k_get_scale_min(scales, is, sc, m);
            float d1 = d * sc, m1 = dmin * m;
            k_get_scale_min(scales, is + 1, sc, m);
            float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32 && base + off + l < count; l++)
                out[base + off + l] = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32 && base + off + 32 + l < count; l++)
                out[base + off + 32 + l] = d2 * (q[l] >> 4) - m2;
            q += 32; is += 2;
        }
    }
    return true;
}

static bool dequant_q6_k(const uint8_t* bd, float* out, int count) {
    // Q6_K: 256-elem superblock, 210 bytes
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        uint8_t ql[128]; memcpy(ql, p, 128); p += 128;
        uint8_t qh[64]; memcpy(qh, p, 64); p += 64;
        int8_t scales[16]; memcpy(scales, p, 16); p += 16;
        float d = read_f16(p); p += 2;

        int base = b * BS;
        for (int n = 0; n < BS; n += 128) {
            for (int l = 0; l < 32 && base + n + l < count; l++) {
                int8_t sc = scales[l / 2];
                int v = (ql[n / 2 + l] & 0xF) | ((qh[n / 2 + l / 4] >> (4 * (l & 1))) & 0x30);
                out[base + n + l] = d * sc * (v - 32);
            }
        }
    }
    return true;
}

static bool dequant_q8_k(const uint8_t* bd, float* out, int count) {
    // Q8_K: 256-elem superblock, 292 bytes
    // d(4) + qs(256) + bsums(32)
    // Note: d is a full f32 (not f16) in Q8_K per ggml layout
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        float d; memcpy(&d, p, 4); p += 4;
        int8_t qs[256]; memcpy(qs, p, 256); p += 256;
        // Skip bsums (32 bytes) — not needed for dequant
        p += 32;
        int base = b * BS;
        for (int l = 0; l < BS && base + l < count; l++)
            out[base + l] = d * qs[l];
    }
    return true;
}

static bool dequant_q2_k(const uint8_t* bd, float* out, int count) {
    // Q2_K: 256-elem superblock, 84 bytes
    // scales(16) + qs(64) + d(2) + dmin(2)
    // Each byte of `scales` packs one 16-element sub-block's (scale, min):
    // low 4 bits = scale, high 4 bits = min. qs packs 2-bit quants, 4 per byte.
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        uint8_t scales[16]; memcpy(scales, p, 16); p += 16;
        uint8_t qs[64]; memcpy(qs, p, 64); p += 64;
        float d = read_f16(p); p += 2;
        float dmin = read_f16(p); p += 2;

        int base = b * BS;
        int is = 0; const uint8_t* q = qs;
        for (int n = 0; n < BS && base + n < count; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);
                for (int l = 0; l < 16 && base + n + j * 32 + l < count; l++)
                    out[base + n + j * 32 + l] = dl * ((q[l] >> shift) & 3) - ml;
                sc = scales[is++];
                dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
                for (int l = 0; l < 16 && base + n + j * 32 + 16 + l < count; l++)
                    out[base + n + j * 32 + 16 + l] = dl * ((q[l + 16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
    return true;
}

static bool dequant_q3_k(const uint8_t* bd, float* out, int count) {
    // Q3_K: 256-elem superblock, 110 bytes
    // hmask(32) + qs(64) + scales(12) + d(2)
    // 3-bit quants: 2 low bits from `qs`, 1 high bit from `hmask`. `scales`
    // packs 16 signed 6-bit sub-block scales (bias 32) across 12 bytes using
    // the same kmask1/kmask2 bit-repacking as llama.cpp's ggml-quants.c —
    // this is inherently little-endian (uint32_t reinterpretation of raw
    // bytes), consistent with the rest of this file's binary parsing.
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        uint8_t hmask[32]; memcpy(hmask, p, 32); p += 32;
        uint8_t qs[64]; memcpy(qs, p, 64); p += 64;
        uint8_t raw_scales[12]; memcpy(raw_scales, p, 12); p += 12;
        float d_all = read_f16(p); p += 2;

        uint32_t aux[4] = {0, 0, 0, 0};
        memcpy(aux, raw_scales, 12);
        const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int8_t scales[16]; memcpy(scales, aux, 16);
        for (int j = 0; j < 16; j++) scales[j] -= 32;

        int base = b * BS;
        int is = 0; const uint8_t* q = qs;
        uint8_t m = 1;
        for (int n = 0; n < BS && base + n < count; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d_all * scales[is++];
                for (int l = 0; l < 16 && base + n + j * 32 + l < count; l++)
                    out[base + n + j * 32 + l] =
                        dl * (((int8_t)((q[l] >> shift) & 3)) - ((hmask[l] & m) ? 0 : 4));
                dl = d_all * scales[is++];
                for (int l = 0; l < 16 && base + n + j * 32 + 16 + l < count; l++)
                    out[base + n + j * 32 + 16 + l] =
                        dl * (((int8_t)((q[l + 16] >> shift) & 3)) - ((hmask[l + 16] & m) ? 0 : 4));
                shift += 2;
                m = (uint8_t)(m << 1);
            }
            q += 32;
        }
    }
    return true;
}

static bool dequant_q5_k(const uint8_t* bd, float* out, int count) {
    // Q5_K: 256-elem superblock, 176 bytes
    // d(2) + dmin(2) + scales(12) + qh(32) + qs(128)
    // Same 6-bit scale/min packing as Q4_K; qs holds the low 4 bits, qh the
    // 5th (high) bit, one per element.
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        float d = read_f16(p); p += 2;
        float dmin = read_f16(p); p += 2;
        uint8_t scales[12]; memcpy(scales, p, 12); p += 12;
        uint8_t qh[32]; memcpy(qh, p, 32); p += 32;
        uint8_t ql[128]; memcpy(ql, p, 128); p += 128;

        int base = b * BS;
        int is = 0; const uint8_t* q = ql;
        uint8_t u1 = 1, u2 = 2;
        for (int n = 0; n < BS && base + n < count; n += 64) {
            uint8_t sc, m;
            k_get_scale_min(scales, is, sc, m);
            float d1 = d * sc, m1 = dmin * m;
            k_get_scale_min(scales, is + 1, sc, m);
            float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32 && base + n + l < count; l++)
                out[base + n + l] = d1 * ((q[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32 && base + n + 32 + l < count; l++)
                out[base + n + 32 + l] = d2 * ((q[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            q += 32; is += 2;
            u1 = (uint8_t)(u1 << 2); u2 = (uint8_t)(u2 << 2);
        }
    }
    return true;
}

struct GgufReader {
    std::ifstream f;
    std::string arch;
    uint32_t version = 0;
    uint64_t alignment = 32;
    uint64_t tensor_data_start = 0;
    bool has_bst_tensor = false;
    
    struct TensorInfo {
        std::vector<uint64_t> shape;
        uint32_t dtype;
        uint64_t offset;
        uint64_t file_offset;
    };
    std::unordered_map<std::string, TensorInfo> tensors;
    std::map<std::string, std::string> kv_string;
    std::map<std::string, uint32_t> kv_uint32;
    
    // Bounds to prevent OOM from malformed/crafted GGUF files (issue #366)
    static constexpr uint64_t MAX_STRING_LEN   = 1ULL * 1024 * 1024;   // 1 MiB per individual string
    static constexpr uint64_t MAX_TENSOR_COUNT = 200000;               // generous: Llama-405B ≈ 3.5k tensors
    static constexpr uint64_t MAX_KV_COUNT     = 200000;               // generous
    static constexpr uint32_t MAX_NDIM         = 16;                   // GGUF tensors are ≤ 5-d; 16 is very safe
    static constexpr uint64_t MAX_ARRAY_COUNT  = 1000000;              // 1M elements max in metadata arrays
    static constexpr uint64_t MAX_DIM_SIZE     = 1ULL << 24;           // ~16.7M per dimension — far beyond any real model
    bool m_corrupted = false;
    
    bool open(const std::string& path) {
        f.open(path, std::ios::binary);
        if (!f) return false;
        char magic[4];
        f.read(magic, 4);
        if (std::strncmp(magic, "GGUF", 4) != 0) return false;
        f.read(reinterpret_cast<char*>(&version), 4);
        if (version != 2 && version != 3) return false;
        uint64_t n_tensors, n_kv;
        f.read(reinterpret_cast<char*>(&n_tensors), 8);
        f.read(reinterpret_cast<char*>(&n_kv), 8);
        if (n_tensors > MAX_TENSOR_COUNT) {
            fprintf(stderr, "[gguf] FATAL: tensor count %llu exceeds max %llu — file may be corrupted\n",
                    (unsigned long long)n_tensors, (unsigned long long)MAX_TENSOR_COUNT);
            return false;
        }
        if (n_kv > MAX_KV_COUNT) {
            fprintf(stderr, "[gguf] FATAL: KV count %llu exceeds max %llu — file may be corrupted\n",
                    (unsigned long long)n_kv, (unsigned long long)MAX_KV_COUNT);
            return false;
        }
        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key = read_string();
            uint32_t vt;
            f.read(reinterpret_cast<char*>(&vt), 4);
            if (vt == 0) { uint32_t v; f.read(reinterpret_cast<char*>(&v), 4); kv_uint32[key] = v; }
            else if (vt == 8) { kv_string[key] = read_string(); }
            else if (vt == 4) { uint32_t v; f.read(reinterpret_cast<char*>(&v), 4); kv_uint32[key] = v; }
            else { skip_unknown(vt, key); }
        }
        if (m_corrupted) return false;
        if (kv_string.count("general.architecture")) arch = kv_string["general.architecture"];
        if (kv_uint32.count("general.alignment")) alignment = kv_uint32["general.alignment"];
        if (alignment < 32) alignment = 32;
        for (uint64_t i = 0; i < n_tensors; ++i) {
            std::string name = read_string();
            if (m_corrupted) return false;
            uint32_t ndim;
            f.read(reinterpret_cast<char*>(&ndim), 4);
            if (ndim > MAX_NDIM) {
                fprintf(stderr, "[gguf] FATAL: tensor '%s' has %u dimensions (max %u) — file may be corrupted\n",
                        name.c_str(), ndim, MAX_NDIM);
                return false;
            }
            TensorInfo ti;
            ti.shape.resize(ndim);
            bool dim_ok = true;
            for (uint32_t d = 0; d < ndim; ++d) {
                f.read(reinterpret_cast<char*>(&ti.shape[d]), 8);
                if (ti.shape[d] > MAX_DIM_SIZE) {
                    fprintf(stderr, "[gguf] FATAL: tensor '%s' dim[%u]=%llu exceeds max %llu — file may be corrupted\n",
                            name.c_str(), d, (unsigned long long)ti.shape[d], (unsigned long long)MAX_DIM_SIZE);
                    dim_ok = false;
                }
            }
            if (!dim_ok) return false;
            f.read(reinterpret_cast<char*>(&ti.dtype), 4);
            f.read(reinterpret_cast<char*>(&ti.offset), 8);
            if (ti.dtype == GGUF_TYPE_BLOCK_SCALED_TERNARY)
                has_bst_tensor = true;
            tensors[name] = std::move(ti);
        }
        tensor_data_start = (uint64_t)f.tellg();
        uint64_t rem = tensor_data_start % alignment;
        if (rem) tensor_data_start += alignment - rem;
        for (auto& [name, ti] : tensors)
            ti.file_offset = tensor_data_start + ti.offset;
        return true;
    }
    
    std::string read_string() {
        uint64_t len;
        f.read(reinterpret_cast<char*>(&len), 8);
        if (len > MAX_STRING_LEN) {
            fprintf(stderr, "[gguf] FATAL: string length %llu exceeds max %llu — file may be corrupted\n",
                    (unsigned long long)len, (unsigned long long)MAX_STRING_LEN);
            m_corrupted = true;
            return {};
        }
        std::string s(len, '\0');
        if (len > 0) f.read(&s[0], len);
        return s;
    }
    
    void skip_unknown(uint32_t vt, const std::string& key) {
        // GGUF value types (llama.cpp gguf.h):
        // 0=UINT8  1=INT8  2=UINT16  3=INT16  4=UINT32  5=INT32
        // 6=FLOAT32  7=BOOL  8=STRING  9=ARRAY  10=UINT64  11=INT64  12=FLOAT64
        uint8_t tmp[256];
        if (vt == 0 || vt == 1 || vt == 7) f.read(reinterpret_cast<char*>(tmp), 1);
        else if (vt == 2 || vt == 3) f.read(reinterpret_cast<char*>(tmp), 2);
        else if (vt == 4 || vt == 5 || vt == 6) f.read(reinterpret_cast<char*>(tmp), 4);
        else if (vt == 8) { read_string(); }
        else if (vt == 9) {
            uint32_t at; f.read(reinterpret_cast<char*>(&at), 4);
            uint64_t an;
            if (version >= 3) { f.read(reinterpret_cast<char*>(&an), 8); }
            else { uint32_t an32; f.read(reinterpret_cast<char*>(&an32), 4); an = an32; }
            if (an > MAX_ARRAY_COUNT) {
                fprintf(stderr, "[gguf] FATAL: KV '%s' array count %llu exceeds max %llu — file may be corrupted\n",
                        key.c_str(), (unsigned long long)an, (unsigned long long)MAX_ARRAY_COUNT);
                m_corrupted = true;
                return;
            }
            for (uint64_t j = 0; j < an; ++j) {
                if (at == 0 || at == 1 || at == 7 || at == 4 || at == 5 || at == 6) f.read(reinterpret_cast<char*>(tmp), 4);
                else if (at == 2 || at == 3) f.read(reinterpret_cast<char*>(tmp), 2);
                else if (at == 8) read_string();
                else f.read(reinterpret_cast<char*>(tmp), 8);
            }
        } else if (vt == 10 || vt == 11 || vt == 12) { f.read(reinterpret_cast<char*>(tmp), 8); }
        else { f.read(reinterpret_cast<char*>(tmp), 4); }
    }
    
    bool read_tensor(const std::string& name, std::vector<float>& out) {
        auto it = tensors.find(name);
        if (it == tensors.end()) return false;
        auto& ti = it->second;
        uint64_t numel = 1;
        for (auto d : ti.shape) {
            if (d != 0 && numel > UINT64_MAX / d) {
                fprintf(stderr, "[gguf] FATAL: tensor '%s' numel overflow — file may be corrupted\n", name.c_str());
                return false;
            }
            numel *= d;
        }
        // Sanity cap on total elements to prevent OOM on allocation
        static constexpr uint64_t MAX_TENSOR_ELEMENTS = 1ULL << 30;  // ~1G elements ≈ 4 GiB in F32
        if (numel > MAX_TENSOR_ELEMENTS) {
            fprintf(stderr, "[gguf] FATAL: tensor '%s' numel %llu exceeds max %llu — file may be corrupted\n",
                    name.c_str(), (unsigned long long)numel, (unsigned long long)MAX_TENSOR_ELEMENTS);
            return false;
        }
        out.resize(numel);
        f.seekg(ti.file_offset);
        int block_size = gguf_block_size(ti.dtype);
        int block_bytes = gguf_block_bytes(ti.dtype);
        if (block_size <= 0) return false;
        if (ti.dtype == GGUF_TYPE_F32) {
            f.read(reinterpret_cast<char*>(out.data()), numel * 4); return true;
        }
        if (ti.dtype == GGUF_TYPE_F16) {
            std::vector<uint16_t> f16(numel);
            f.read(reinterpret_cast<char*>(f16.data()), numel * 2);
            for (uint64_t i = 0; i < numel; ++i) {
                uint32_t bits = (uint32_t)f16[i] << 16;
                float v; memcpy(&v, &bits, 4); out[i] = v;
            }
            return true;
        }
        if (ti.dtype == GGUF_TYPE_BLOCK_SCALED_TERNARY) {
            uint64_t n_blocks = (numel + block_size - 1) / block_size;
            std::vector<uint8_t> bd(block_bytes);
            for (uint64_t b = 0; b < n_blocks; ++b) {
                uint64_t start = b * block_size;
                uint64_t end = std::min(start + block_size, numel);
                f.read(reinterpret_cast<char*>(bd.data()), block_bytes);
                block_scaled_ternary_dequant_row(bd.data(), out.data() + start, (int)(end - start));
            }
            return true;
        }
        uint64_t n_blocks = (numel + block_size - 1) / block_size;
        std::vector<uint8_t> bd(block_bytes);
        for (uint64_t b = 0; b < n_blocks; ++b) {
            uint64_t start = b * block_size;
            uint64_t end = std::min(start + block_size, numel);
            uint64_t count = end - start;
            f.read(reinterpret_cast<char*>(bd.data()), block_bytes);
            if (ti.dtype == GGUF_TYPE_Q8_0) {
                __half scale_h; memcpy(&scale_h, bd.data(), 2);
                float scale = (float)scale_h;
                int8_t* q = (int8_t*)(bd.data() + 2);
                for (uint64_t i = 0; i < count; ++i) out[start + i] = q[i] * scale;
            } else if (ti.dtype == GGUF_TYPE_Q4_0) {
                __half scale_h; memcpy(&scale_h, bd.data(), 2);
                float scale = (float)scale_h;
                uint8_t* q = bd.data() + 2;
                for (uint64_t i = 0; i < count; ++i) {
                    int8_t nib = (i & 1) ? (q[i >> 1] & 0x0F) : (q[i >> 1] >> 4);
                    out[start + i] = (nib - 8) * scale;
                }
            } else if (ti.dtype == GGUF_TYPE_Q4_K) {
                // NOTE: this dispatch runs once per 256-element super-block
                // (the enclosing for-loop already steps block-by-block).
                // `return`ing here used to exit after the tensor's FIRST
                // block only, leaving every subsequent block un-dequantized
                // (zero-initialized) for any tensor bigger than 256
                // elements — i.e. essentially every real weight matrix.
                dequant_q4_k(bd.data(), out.data() + start, (int)count);
            } else if (ti.dtype == GGUF_TYPE_Q6_K) {
                dequant_q6_k(bd.data(), out.data() + start, (int)count);
            } else if (ti.dtype == GGUF_TYPE_Q8_K) {
                dequant_q8_k(bd.data(), out.data() + start, (int)count);
            } else if (ti.dtype == GGUF_TYPE_Q2_K) {
                dequant_q2_k(bd.data(), out.data() + start, (int)count);
            } else if (ti.dtype == GGUF_TYPE_Q3_K) {
                dequant_q3_k(bd.data(), out.data() + start, (int)count);
            } else if (ti.dtype == GGUF_TYPE_Q5_K) {
                dequant_q5_k(bd.data(), out.data() + start, (int)count);
            } else {
                // Unsupported quantization (Q2_K, Q3_K, Q5_K, Q1_0, etc.)
                fprintf(stderr, "  [gguf] unsupported quant type %d for tensor, aborting load\n", ti.dtype);
                return false;
            }
        }
        return true;
    }
};
} // anonymous namespace

extern "C" {

rcpp_status_t rcpp_bitnet_load_gguf(const char* path, rcpp_bitnet_model_t* out_model) {
    if (!path || !out_model) return RCPP_INVALID_ARG;
    memset(out_model, 0, sizeof(*out_model));
    GgufReader reader;
    if (!reader.open(path)) return RCPP_INVALID_ARG;
    fprintf(stderr, "[gguf] Loading: %s\n", path);
    
    auto gu = [&](const std::string& k, int def) -> int {
        if (reader.kv_uint32.count(k)) return (int)reader.kv_uint32[k];
        return def;
    };
    
    int hidden_size = gu("llm.embedding_length", 2048);
    int n_layers = gu("llm.block_count", 40);
    int n_heads = gu("llm.attention.head_count", 8);
    int inter_size = gu("llm.feed_forward_length", 2048);
    int vocab_size = gu("llm.vocab_size", 262272);
    int head_dim = hidden_size / n_heads;
    
    out_model->hidden_size = hidden_size;
    out_model->intermediate_size = inter_size;
    out_model->num_layers = n_layers;
    out_model->num_heads = n_heads;
    out_model->vocab_size = vocab_size;
    out_model->weight_format = reader.has_bst_tensor
        ? RCPP_WEIGHT_FORMAT_BLOCK_SCALED_TERNARY
        : RCPP_WEIGHT_FORMAT_HALO_V2;
    if (reader.has_bst_tensor)
        out_model->flags |= H1B_FLAG_BLOCK_SCALED;
    out_model->arch = rcpp_arch_from_string(reader.arch.c_str());
    out_model->is_qwen3 = (out_model->arch == RCPP_ARCH_QWEN3) ? 1 : 0;
    
    const size_t MAX_EL = 1ULL << 34; // 16B elements ~64 GB
    {
        std::vector<float> emb;
        if (!reader.read_tensor("token_embd.weight", emb))
            reader.read_tensor("model.embed_tokens.weight", emb);
        if (emb.empty() || emb.size() > MAX_EL) { fprintf(stderr, "GGUF: invalid embedding size %zu\n", emb.size()); return RCPP_INVALID_ARG; }
        std::vector<_Float16> f16(emb.size());
        for (size_t i = 0; i < emb.size(); ++i) f16[i] = (_Float16)emb[i];
        HIP_CHECK(hipMalloc(&out_model->embedding_dev, f16.size() * sizeof(_Float16)));
        HIP_CHECK(hipMemcpy(out_model->embedding_dev, f16.data(), f16.size() * sizeof(_Float16), hipMemcpyHostToDevice));
    }
    {
        std::vector<float> fn;
        reader.read_tensor("output_norm.weight", fn);
        std::vector<_Float16> f16(fn.size());
        for (size_t i = 0; i < fn.size(); ++i) f16[i] = (_Float16)fn[i];
        HIP_CHECK(hipMalloc(&out_model->final_norm_weight_dev, f16.size() * sizeof(_Float16)));
        HIP_CHECK(hipMemcpy(out_model->final_norm_weight_dev, f16.data(), f16.size() * sizeof(_Float16), hipMemcpyHostToDevice));
    }
    out_model->layers = new rcpp_bitnet_layer_t[n_layers]();
    for (int l = 0; l < n_layers; ++l) {
        auto& layer = out_model->layers[l];
        auto prefix = [&](const char* n) -> std::string {
            return std::string("model.layers.") + std::to_string(l) + "." + n;
        };
        // Select target pointers based on weight format
        // BST format writes to bst_*_packed_dev, standard writes to *_packed_dev
        auto lw = [&](const std::string& gn, void** std_ptr, void** bst_ptr, int r, int c) -> rcpp_status_t {
            std::vector<float> data;
            if (!reader.read_tensor(gn, data)) return RCPP_OK;
            if (data.size() > MAX_EL) { fprintf(stderr, "GGUF: tensor %s too large (%zu el)\n", gn.c_str(), data.size()); return RCPP_INVALID_ARG; }
            std::vector<_Float16> f16(data.size());
            for (size_t i = 0; i < data.size(); ++i) f16[i] = (_Float16)data[i];
            void** target = reader.has_bst_tensor ? bst_ptr : std_ptr;
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
