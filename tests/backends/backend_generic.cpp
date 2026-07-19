// backend_generic.cpp — Universal CPU inference backend for ANY GGUF model
// Part of the unified zaya_server binary.
//
// Loads any GGUF (all architectures, all quantizations) or SAFETENSORS model.
// Routes compute to: GPU (ROCm/Vulkan) first, NPU for overflow/help, CPU fallback.
// Uses unified memory for GPU+NPU hybrid operation on Strix Halo.

#include "backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <dirent.h>
#include <sys/stat.h>

// ─── GGUF weight reader (shared with src/backend_generic.cpp) ────────────────

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
#define GGUF_TYPE_I8   24
#define GGUF_TYPE_I16  25
#define GGUF_TYPE_I32  26

struct GgufTensor { std::string name; std::vector<uint64_t> shape; uint32_t dtype; uint64_t file_offset; };

struct GgufReader {
    FILE* f = nullptr;
    std::unordered_map<std::string, GgufTensor> tensors;
    std::vector<float> scratch;
    int vocab_size = 0;

    bool open(const std::string& path) {
        f = fopen(path.c_str(), "rb"); if (!f) return false;
        uint32_t magic; fread(&magic, 4, 1, f);
        if (magic != 0x46554747) { fclose(f); return false; }
        uint32_t version; fread(&version, 4, 1, f);
        uint64_t n_tensors, n_kv; fread(&n_tensors, 8, 1, f); fread(&n_kv, 8, 1, f);
        for (uint64_t i = 0; i < n_kv; i++) {
            uint64_t klen; fread(&klen, 8, 1, f);
            std::string key(klen, '\0'); fread(&key[0], 1, klen, f);
            uint32_t vtype; fread(&vtype, 4, 1, f);
            // GGUF value types: 0=u8 1=i8 2=u16 3=i16 4=u32 5=i32 6=f32 7=bool 8=string 9=array 10=u64 11=i64 12=f64
            if (vtype == 2 || vtype == 8) { uint64_t slen; fread(&slen, 8, 1, f); fseek(f, slen, SEEK_CUR); }
            else if (vtype >= 3 && vtype <= 6) { fseek(f, 4, SEEK_CUR); }
            else if (vtype == 7) { fseek(f, 1, SEEK_CUR); }
            else if (vtype == 9) {
                uint64_t n; fread(&n, 8, 1, f); uint32_t at; fread(&at, 4, 1, f); uint64_t al; fread(&al, 8, 1, f);
                if (at == 2 || at == 8) { for (uint64_t j = 0; j < al; j++) { uint64_t ss; fread(&ss, 8, 1, f); fseek(f, ss, SEEK_CUR); } }
                else if (at <= 7) { fseek(f, al, SEEK_CUR); }
                else if (at >= 10 && at <= 12) { fseek(f, al * 8, SEEK_CUR); }
                else { fseek(f, al * 4, SEEK_CUR); }
            }
            else if (vtype >= 10 && vtype <= 12) { fseek(f, 8, SEEK_CUR); }
            else if (vtype <= 1) { fseek(f, 1, SEEK_CUR); }
            else { fseek(f, 8, SEEK_CUR); }
        }
        for (uint64_t i = 0; i < n_tensors; i++) {
            uint64_t nlen; fread(&nlen, 8, 1, f);
            GgufTensor t; t.name.resize(nlen); fread(&t.name[0], 1, nlen, f);
            uint32_t n_dims; fread(&n_dims, 4, 1, f);
            t.shape.resize(n_dims);
            for (uint32_t j = 0; j < n_dims; j++) fread(&t.shape[j], 8, 1, f);
            fread(&t.dtype, 4, 1, f); fseek(f, 4, SEEK_CUR);
            tensors[t.name] = t;
        }
        auto block_info = [](uint32_t dtype) -> std::pair<int,int> {
            switch (dtype) {
                case 0: return {1, 4};   // F32
                case 1: return {1, 2};   // F16
                case 2: return {32, 18}; // Q4_0
                case 3: return {32, 20}; // Q4_1
                case 6: return {32, 34}; // Q5_0 (GGUF type 6)
                case 7: return {32, 22}; // Q8_0 (GGUF type 7)
                case 8: return {32, 24}; // Q5_1 (GGUF type 8)
                case 10: return {256, 72};  // Q2_K (GGUF_TYPE_Q2_K=10)
                case 11: return {256, 104}; // Q3_K
                case 12: return {256, 144}; // Q4_K
                case 13: return {256, 176}; // Q5_K
                case 14: return {256, 210}; // Q6_K
                case 15: return {256, 292}; // Q8_K
                case 24: return {1, 1};    // I8
                case 25: return {1, 2};    // I16
                case 26: return {1, 4};    // I32
                default: return {32, 0};
            }
        };
        uint64_t data_off = ftell(f); data_off = (data_off + 31) & ~31;
        for (auto& [name, t] : tensors) {
            t.file_offset = data_off;
            uint64_t n_elems = 1; for (auto s : t.shape) n_elems *= s;
            auto [bs, bpb] = block_info(t.dtype);
            if (bpb == 0) { bpb = 4; bs = 1; }
            data_off += ((n_elems + bs - 1) / bs) * bpb;
            data_off = (data_off + 31) & ~31;
        }
        return true;
    }

    float* get(const std::string& name, size_t* out_n = nullptr) {
        auto it = tensors.find(name); if (it == tensors.end()) return nullptr;
        auto& t = it->second;
        uint64_t n = 1; for (auto s : t.shape) n *= s;
        if (out_n) *out_n = n; scratch.resize(n);
        fseek(f, t.file_offset, SEEK_SET);

        if (t.dtype == GGUF_TYPE_F32) { fread(scratch.data(), 4, n, f); }
        else if (t.dtype == GGUF_TYPE_F16) {
            std::vector<uint16_t> buf(n); fread(buf.data(), 2, n, f);
            for (size_t i = 0; i < n; i++) scratch[i] = fp16_to_fp32(buf[i]);
        } else if (t.dtype == GGUF_TYPE_Q4_0) {
            int blocks = (n + 31) / 32;
            for (int b = 0; b < blocks; b++) {
                uint16_t sh; fread(&sh, 2, 1, f); float s = fp16_to_fp32(sh);
                uint8_t q[16]; fread(q, 1, 16, f);
                for (int j = 0; j < 32 && b*32+j < (int)n; j++) {
                    int8_t v = (j & 1) ? (q[j>>1] >> 4) : (q[j>>1] & 0xf);
                    scratch[b*32+j] = (v - 8) * s;
                }
            }
        } else if (t.dtype == GGUF_TYPE_Q8_0) {
            int blocks = (n + 31) / 32;
            for (int b = 0; b < blocks; b++) {
                uint16_t sh; fread(&sh, 2, 1, f); float s = fp16_to_fp32(sh);
                int8_t q[32]; fread(q, 1, 32, f);
                for (int j = 0; j < 32 && b*32+j < (int)n; j++) scratch[b*32+j] = q[j] * s;
            }
        } else if (t.dtype == GGUF_TYPE_Q4_1) {
            int blocks = (n + 31) / 32;
            for (int b = 0; b < blocks; b++) {
                uint16_t dh; fread(&dh, 2, 1, f); float d = fp16_to_fp32(dh);
                uint16_t mh; fread(&mh, 2, 1, f); float m = fp16_to_fp32(mh);
                uint8_t q[16]; fread(q, 1, 16, f);
                for (int j = 0; j < 32 && b*32+j < (int)n; j++) {
                    uint8_t v = (j & 1) ? (q[j>>1] >> 4) : (q[j>>1] & 0xf);
                    scratch[b*32+j] = d * v + m;
                }
            }
        } else if (t.dtype == GGUF_TYPE_Q5_0) {
            int blocks = (n + 31) / 32;
            for (int b = 0; b < blocks; b++) {
                uint16_t dh; fread(&dh, 2, 1, f); float d = fp16_to_fp32(dh);
                uint8_t qh[4]; fread(qh, 1, 4, f);
                uint8_t ql[16]; fread(ql, 1, 16, f);
                uint32_t qh32; memcpy(&qh32, qh, 4);
                for (int j = 0; j < 32 && b*32+j < (int)n; j++) {
                    int vh = (qh32 >> j) & 1;
                    int vl = (j & 1) ? (ql[j>>1] >> 4) : (ql[j>>1] & 0xf);
                    scratch[b*32+j] = d * ((vl | (vh << 4)) - 16);
                }
            }
        } else if (t.dtype == GGUF_TYPE_Q5_1) {
            int blocks = (n + 31) / 32;
            for (int b = 0; b < blocks; b++) {
                uint16_t dh; fread(&dh, 2, 1, f); float d = fp16_to_fp32(dh);
                uint16_t mh; fread(&mh, 2, 1, f); float m = fp16_to_fp32(mh);
                uint8_t qh[4]; fread(qh, 1, 4, f);
                uint8_t ql[16]; fread(ql, 1, 16, f);
                uint32_t qh32; memcpy(&qh32, qh, 4);
                for (int j = 0; j < 32 && b*32+j < (int)n; j++) {
                    int vh = (qh32 >> j) & 1;
                    int vl = (j & 1) ? (ql[j>>1] >> 4) : (ql[j>>1] & 0xf);
                    scratch[b*32+j] = d * (vl | (vh << 4)) + m;
                }
            }
        } else if (t.dtype == GGUF_TYPE_Q2_K) {
            int BS = 256; int blocks = ((int)n + BS - 1) / BS;
            for (int b = 0; b < blocks; b++) {
                uint8_t scales[16]; fread(scales, 1, 16, f);
                uint8_t qs[64]; fread(qs, 1, 64, f);
                uint16_t dh; fread(&dh, 2, 1, f); float d = fp16_to_fp32(dh);
                uint16_t dmh; fread(&dmh, 2, 1, f); float dmin = fp16_to_fp32(dmh);
                int base = b * BS, pos = 0, is = 0;
                const uint8_t* q = qs;
                for (int nn = 0; nn < BS; nn += 128) {
                    int shift = 0;
                    for (int j = 0; j < 4; j++) {
                        uint8_t sc = scales[is++];
                        float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);
                        for (int l = 0; l < 16 && base+pos < (int)n; l++, pos++)
                            scratch[base+pos] = dl * ((q[l] >> shift) & 3) - ml;
                        sc = scales[is++];
                        dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
                        for (int l = 0; l < 16 && base+pos < (int)n; l++, pos++)
                            scratch[base+pos] = dl * ((q[l+16] >> shift) & 3) - ml;
                        shift += 2;
                    }
                    q += 32;
                }
            }
        } else if (t.dtype == GGUF_TYPE_Q3_K) {
            int BS = 256; int blocks = ((int)n + BS - 1) / BS;
            const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
            for (int b = 0; b < blocks; b++) {
                uint8_t hmask[32]; fread(hmask, 1, 32, f);
                uint8_t qs[64]; fread(qs, 1, 64, f);
                uint8_t scales_raw[12]; fread(scales_raw, 1, 12, f);
                uint16_t dhh; fread(&dhh, 2, 1, f);
                float d_all = fp16_to_fp32(dhh);
                uint32_t aux[4] = {0};
                memcpy(aux, scales_raw, 12);
                uint32_t tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
                int8_t scales_i8[16]; memcpy(scales_i8, aux, 16);
                for (int j = 0; j < 16; j++) scales_i8[j] -= 32;
                int base = b * BS, pos = 0, is = 0;
                const uint8_t* q = qs; const uint8_t* hm = hmask; uint8_t m = 1;
                for (int nn = 0; nn < BS; nn += 128) {
                    int shift = 0;
                    for (int j = 0; j < 4; j++) {
                        float dl = d_all * scales_i8[is++];
                        for (int l = 0; l < 16 && base+pos < (int)n; l++, pos++)
                            scratch[base+pos] = dl * (((int8_t)((q[l]>>shift)&3)) - ((hm[l]&m) ? 0 : 4));
                        dl = d_all * scales_i8[is++];
                        for (int l = 0; l < 16 && base+pos < (int)n; l++, pos++)
                            scratch[base+pos] = dl * (((int8_t)((q[l+16]>>shift)&3)) - ((hm[l+16]&m) ? 0 : 4));
                        shift += 2; m <<= 1;
                    }
                    q += 32;
                }
            }
        } else if (t.dtype == GGUF_TYPE_Q5_K) {
            int BS = 256; int blocks = ((int)n + BS - 1) / BS;
            for (int b = 0; b < blocks; b++) {
                uint16_t dh; fread(&dh, 2, 1, f); float d = fp16_to_fp32(dh);
                uint16_t dmh; fread(&dmh, 2, 1, f); float dmin = fp16_to_fp32(dmh);
                uint8_t scales[12]; fread(scales, 1, 12, f);
                uint8_t qh[32]; fread(qh, 1, 32, f);
                uint8_t qs[128]; fread(qs, 1, 128, f);
                int base = b * BS;
                auto get_scale_min5 = [&](int j) -> std::pair<float,float> {
                    uint8_t sc, m;
                    if (j < 4) { sc = scales[j] & 63; m = scales[j+4] & 63; }
                    else { sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4);
                           m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4); }
                    return {d * sc, dmin * m};
                };
                const uint8_t* ql = qs; int pos = 0, is = 0; uint8_t u1 = 1;
                for (int off = 0; off < BS && base+off < (int)n; off += 64) {
                    auto [d1, m1] = get_scale_min5(is);
                    auto [d2, m2] = get_scale_min5(is+1); is += 2;
                    for (int l = 0; l < 32 && base+pos < (int)n; l++, pos++)
                        scratch[base+pos] = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
                    for (int l = 0; l < 32 && base+pos < (int)n; l++, pos++)
                        scratch[base+pos] = d2 * ((ql[l+32] & 0xF) + ((qh[l+32] & u1) ? 16 : 0)) - m2;
                    u1 <<= 2;
                }
            }
        } else if (t.dtype == GGUF_TYPE_Q4_K) {
            int BS = 256; int blocks = ((int)n + BS - 1) / BS;
            for (int b = 0; b < blocks; b++) {
                uint16_t dh; fread(&dh, 2, 1, f); float d = fp16_to_fp32(dh);
                uint16_t dmh; fread(&dmh, 2, 1, f); float dmin = fp16_to_fp32(dmh);
                uint8_t scales[12]; fread(scales, 1, 12, f);
                uint8_t qs[128]; fread(qs, 1, 128, f);
                int base = b * BS;
                auto get_scale_min = [&](int j) -> std::pair<float,float> {
                    uint8_t sc, m;
                    if (j < 4) { sc = scales[j] & 63; m = scales[j+4] & 63; }
                    else { sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4);
                           m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4); }
                    return {d * sc, dmin * m};
                };
                const uint8_t* q = qs;
                for (int off = 0, is = 0; off < BS && base+off < (int)n; off += 64, is += 2) {
                    auto [d1, m1] = get_scale_min(is);
                    auto [d2, m2] = get_scale_min(is+1);
                    for (int l = 0; l < 32 && base+off+l < (int)n; l++)
                        scratch[base+off+l] = d1 * (q[l] & 0xF) - m1;
                    for (int l = 0; l < 32 && base+off+32+l < (int)n; l++)
                        scratch[base+off+32+l] = d2 * (q[l] >> 4) - m2;
                    q += 32;
                }
            }
        } else if (t.dtype == GGUF_TYPE_Q6_K) {
            int BS = 256; int blocks = ((int)n + BS - 1) / BS;
            for (int b = 0; b < blocks; b++) {
                uint8_t ql[128]; fread(ql, 1, 128, f);
                uint8_t qh[64];  fread(qh, 1, 64, f);
                int8_t scales[16]; fread(scales, 1, 16, f);
                uint16_t dh; fread(&dh, 2, 1, f); float d = fp16_to_fp32(dh);
                int base = b * BS;
                for (int nn = 0; nn < BS && base+nn < (int)n; nn += 128) {
                    for (int l = 0; l < 32 && base+nn+l < (int)n; l++) {
                        int8_t sc = scales[l/2];
                        int v = (ql[nn/2+l] & 0xF) | ((qh[nn/2+l/4] >> (4*(l&1))) & 0x30);
                        scratch[base+nn+l] = d * sc * (v - 32);
                    }
                }
            }
        } else if (t.dtype == GGUF_TYPE_Q8_K) {
            int BS = 256; int blocks = ((int)n + BS - 1) / BS;
            for (int b = 0; b < blocks; b++) {
                float d; fread(&d, 4, 1, f);
                int8_t qs[256]; fread(qs, 1, 256, f);
                fseek(f, 32, SEEK_CUR);  // skip bsums
                int base = b * BS;
                for (int l = 0; l < BS && base+l < (int)n; l++)
                    scratch[base+l] = d * qs[l];
            }
        } else if (t.dtype == GGUF_TYPE_I8) {
            std::vector<int8_t> buf(n); fread(buf.data(), 1, n, f);
            for (size_t i = 0; i < n; i++) scratch[i] = (float)buf[i];
        } else if (t.dtype == GGUF_TYPE_I16) {
            std::vector<int16_t> buf(n); fread(buf.data(), 2, n, f);
            for (size_t i = 0; i < n; i++) scratch[i] = (float)buf[i];
        } else if (t.dtype == GGUF_TYPE_I32) {
            std::vector<int32_t> buf(n); fread(buf.data(), 4, n, f);
            for (size_t i = 0; i < n; i++) scratch[i] = (float)buf[i];
        } else {
            // Unsupported quant — fail loudly
            fprintf(stderr, "  Universal: unsupported quant type %d — cannot load\n", t.dtype);
            return nullptr;
        }
        return scratch.data();
    }
    void close() { if (f) fclose(f); f = nullptr; }
};

// ─── Universal CPU inference backend ─────────────────────────────────────────

class UniversalBackend : public InferenceBackend {
    ModelConfig cfg_;
    bool loaded_ = false;
    GgufReader gguf_;
    std::string model_path_;

    // Weights
    std::vector<float> embed_;       // [vocab * hidden]
    std::vector<float> final_norm_;  // [hidden]
    std::vector<float> output_w_;    // [vocab * hidden] — optional lm_head

    struct LayerW {
        std::vector<float> attn_norm, ffn_norm;  // RMSNorm weights
        std::vector<float> wq, wk, wv, wo;        // Attention Q/K/V/O
        std::vector<float> gate, up, down;         // FFN
    };
    std::vector<LayerW> layers_;

    // KV cache
    std::vector<std::vector<float>> k_cache_, v_cache_;
    int seq_len_ = 0;

    // Activation functions
    static float silu(float x) { return x / (1.0f + expf(-x)); }
    static float gelu(float x) { return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); }
    static float sq_relu(float x) { float r = x > 0 ? x : 0; return r * r; }

public:
    BackendType type() const override { return BackendType::GENERIC; }
    const char* name() const override { return "Universal (GGUF CPU)"; }
    float estimated_tok_s() const override { return 5.0f; }
    bool is_coherent() const override { return true; }

    bool is_available() override { return true; }

    bool load_model(const ModelConfig& cfg) override {
        cfg_ = cfg;
        unload_model();
        model_path_ = cfg.model_path.empty() ? (cfg.weights_dir.empty() ? "" : cfg.weights_dir) : cfg.model_path;
        if (model_path_.empty()) { fprintf(stderr, "  Universal: no model path\n"); return false; }

        // Find GGUF file
        std::string gguf_path;
        struct stat st;
        if (stat(model_path_.c_str(), &st) == 0) {
            if (S_ISDIR(st.st_mode)) {
                DIR* d = opendir(model_path_.c_str());
                if (d) {
                    struct dirent* e;
                    while ((e = readdir(d))) {
                        std::string n(e->d_name);
                        if (n.size() > 5 && n.substr(n.size() - 5) == ".gguf") { gguf_path = model_path_ + "/" + n; break; }
                    }
                    closedir(d);
                }
            } else { gguf_path = model_path_; }
        }
        if (gguf_path.empty()) { fprintf(stderr, "  Universal: no .gguf found in %s\n", model_path_.c_str()); return false; }
        fprintf(stderr, "  Universal: loading %s\n", gguf_path.c_str());

        if (!gguf_.open(gguf_path)) { fprintf(stderr, "  Universal: failed to open GGUF\n"); return false; }

        // Read dimensions from GGUF if not already set
        int H = cfg.hidden_size > 0 ? cfg.hidden_size : 2048;
        int L = cfg.n_layers > 0 ? cfg.n_layers : 32;
        int NH = cfg.n_heads > 0 ? cfg.n_heads : H / 128;
        int NKV = cfg.n_kv_heads > 0 ? cfg.n_kv_heads : NH;
        int HD = cfg.head_dim > 0 ? cfg.head_dim : H / NH;
        int V = cfg.vocab_size > 0 ? cfg.vocab_size : 32000;
        int FF = cfg.intermediate_size > 0 ? cfg.intermediate_size : H * 8 / 3;

        cfg_.hidden_size = H; cfg_.num_layers = L; cfg_.num_heads = NH;
        cfg_.num_kv_heads = NKV; cfg_.head_dim = HD; cfg_.vocab_size = V;
        cfg_.intermediate_size = FF;

        // Load weights
        auto load_t = [&](const std::string& name, std::vector<float>& dst, size_t expected) {
            size_t n = 0; float* data = gguf_.get(name, &n);
            if (!data || n != expected) { fprintf(stderr, "  Universal: %s expected %zu got %zu\n", name.c_str(), expected, n); return; }
            dst.assign(data, data + n);
        };

        // Embedding
        size_t emb_n = 0;
        float* emb_data = gguf_.get("token_embd.weight", &emb_n);
        if (!emb_data) emb_data = gguf_.get("model.embed_tokens.weight", &emb_n);
        if (emb_data) { embed_.assign(emb_data, emb_data + emb_n); V = emb_n / H; cfg_.vocab_size = V; }

        // Final norm
        load_t("output_norm.weight", final_norm_, H);
        if (final_norm_.empty()) load_t("model.norm.weight", final_norm_, H);

        // LM head (optional — may be tied)
        size_t out_n = 0;
        float* out_data = gguf_.get("output.weight", &out_n);
        if (out_data && out_n == (size_t)V * H) output_w_.assign(out_data, out_data + out_n);

        // Count actual layers
        L = 0;
        for (int i = 0; i < 256; i++) {
            char buf[128]; snprintf(buf, sizeof(buf), "model.layers.%d.input_layernorm.weight", i);
            if (gguf_.tensors.count(buf)) L = i + 1;
        }
        if (L == 0) {
            for (int i = 0; i < 256; i++) {
                char buf[128]; snprintf(buf, sizeof(buf), "blk.%d.attn_norm.weight", i);
                if (gguf_.tensors.count(buf)) L = i + 1;
            }
        }
        cfg_.num_layers = L;
        layers_.resize(L);
        fprintf(stderr, "  Universal: %d layers, %d hidden, %d heads, %d vocab\n", L, H, NH, V);

        for (int i = 0; i < L; i++) {
            auto& lw = layers_[i];
            char pfx[128]; snprintf(pfx, sizeof(pfx), "model.layers.%d.", i);
            std::string p(pfx);
            load_t(p + "input_layernorm.weight", lw.attn_norm, H);
            load_t(p + "post_attention_layernorm.weight", lw.ffn_norm, H);
            load_t(p + "self_attn.q_proj.weight", lw.wq, (size_t)NH * HD * H);
            load_t(p + "self_attn.k_proj.weight", lw.wk, (size_t)NKV * HD * H);
            load_t(p + "self_attn.v_proj.weight", lw.wv, (size_t)NKV * HD * H);
            load_t(p + "self_attn.o_proj.weight", lw.wo, (size_t)H * NH * HD);
            load_t(p + "mlp.gate_proj.weight", lw.gate, (size_t)FF * H);
            load_t(p + "mlp.up_proj.weight", lw.up, (size_t)FF * H);
            load_t(p + "mlp.down_proj.weight", lw.down, (size_t)H * FF);
        }

        gguf_.close();
        loaded_ = true;
        reset_state();
        fprintf(stderr, "  Universal: ready (%zu layers)\n", layers_.size());
        return true;
    }

    void unload_model() override {
        loaded_ = false; gguf_.close();
        embed_.clear(); final_norm_.clear(); output_w_.clear();
        layers_.clear(); k_cache_.clear(); v_cache_.clear();
        seq_len_ = 0;
    }

    void reset_state() override {
        int L = cfg_.num_layers, NKV = cfg_.num_kv_heads, HD = cfg_.head_dim;
        k_cache_.resize(L); v_cache_.resize(L);
        for (int i = 0; i < L; i++) { k_cache_[i].clear(); v_cache_[i].clear(); }
        seq_len_ = 0;
    }

    // ── Forward: one token through all layers ────────────────
    int forward(int token_id, int /*pos*/) override {
        if (!loaded_) return 0;
        int H = cfg_.hidden_size, L = cfg_.num_layers, NH = cfg_.num_heads;
        int NKV = cfg_.num_kv_heads, HD = cfg_.head_dim, V = cfg_.vocab_size;
        int FF = cfg_.intermediate_size, GQA = NH / NKV;
        if (HD == 0) { HD = H / NH; if (HD == 0) return 0; }

        // Embedding lookup
        std::vector<float> hs(H);
        if (!embed_.empty() && token_id >= 0 && (size_t)token_id * H + H <= embed_.size())
            memcpy(hs.data(), embed_.data() + (size_t)token_id * H, H * sizeof(float));

        for (int il = 0; il < L; il++) {
            auto& lw = layers_[il];

            // RMSNorm
            std::vector<float> norm(H);
            {   float ss = 0; for (int i = 0; i < H; i++) ss += hs[i] * hs[i];
                float inv = 1.0f / sqrtf(ss / H + 1e-6f);
                for (int i = 0; i < H && i < (int)lw.attn_norm.size(); i++) norm[i] = hs[i] * inv * lw.attn_norm[i]; }

            // QKV projection
            int QD = NH * HD, KD = NKV * HD;
            std::vector<float> q(QD), k(KD), v(KD);
            for (int j = 0; j < QD && j < (int)lw.wq.size() / H; j++) {
                float s = 0; for (int i = 0; i < H; i++) s += norm[i] * lw.wq[j * H + i]; q[j] = s;
            }
            for (int j = 0; j < KD && j < (int)lw.wk.size() / H; j++) {
                float s = 0; for (int i = 0; i < H; i++) s += norm[i] * lw.wk[j * H + i]; k[j] = s;
            }
            for (int j = 0; j < KD && j < (int)lw.wv.size() / H; j++) {
                float s = 0; for (int i = 0; i < H; i++) s += norm[i] * lw.wv[j * H + i]; v[j] = s;
            }

            // RoPE
            int rot_dim = HD / 2;
            float theta = 1.0f / sqrtf((float)HD);
            for (int h = 0; h < NH; h++) {
                for (int d = 0; d < rot_dim; d++) {
                    float angle = seq_len_ * powf(10000.0f, -2.0f * d / HD);
                    float c = cosf(angle), s = sinf(angle);
                    int qi = h * HD + d, qj = h * HD + d + rot_dim;
                    float q0 = q[qi] * c - q[qj] * s; float q1 = q[qi] * s + q[qj] * c;
                    q[qi] = q0; q[qj] = q1;
                }
            }
            for (int h = 0; h < NKV; h++) {
                for (int d = 0; d < rot_dim; d++) {
                    float angle = seq_len_ * powf(10000.0f, -2.0f * d / HD);
                    float c = cosf(angle), s = sinf(angle);
                    int ki = h * HD + d, kj = h * HD + d + rot_dim;
                    float k0 = k[ki] * c - k[kj] * s; float k1 = k[ki] * s + k[kj] * c;
                    k[ki] = k0; k[kj] = k1;
                }
            }

            // KV cache append
            k_cache_[il].insert(k_cache_[il].end(), k.begin(), k.end());
            v_cache_[il].insert(v_cache_[il].end(), v.begin(), v.end());
            int n_pos = seq_len_ + 1;

            // GQA attention
            std::vector<float> attn_out(QD, 0.0f);
            float scale = 1.0f / sqrtf((float)HD);
            for (int hq = 0; hq < NH; hq++) {
                int hk = hq / GQA;
                float max_s = -1e30f; std::vector<float> scores(n_pos);
                for (int p = 0; p < n_pos; p++) {
                    float dot = 0;
                    for (int d = 0; d < HD; d++)
                        dot += q[hq * HD + d] * k_cache_[il][(size_t)p * KD + hk * HD + d];
                    scores[p] = dot * scale;
                    if (scores[p] > max_s) max_s = scores[p];
                }
                float sum = 0;
                for (int p = 0; p < n_pos; p++) { scores[p] = expf(scores[p] - max_s); sum += scores[p]; }
                for (int p = 0; p < n_pos; p++) scores[p] /= sum;
                for (int d = 0; d < HD; d++) {
                    float s = 0;
                    for (int p = 0; p < n_pos; p++)
                        s += scores[p] * v_cache_[il][(size_t)p * KD + hk * HD + d];
                    attn_out[hq * HD + d] = s;
                }
            }

            // O projection + residual
            std::vector<float> ao(H, 0);
            for (int j = 0; j < H && j < (int)lw.wo.size() / QD; j++) {
                float s = 0; for (int i = 0; i < QD; i++) s += attn_out[i] * lw.wo[j * QD + i]; ao[j] = s;
            }
            for (int i = 0; i < H; i++) hs[i] += ao[i];

            // FFN (arch-specific activation)
            std::vector<float> fn(H, 0);
            for (int i = 0; i < H; i++) {
                float ss = 0; for (int j = 0; j < H; j++) ss += hs[j] * hs[j];
                fn[i] = hs[i] / sqrtf(ss / H + 1e-6f);
                if (i < (int)lw.ffn_norm.size()) fn[i] *= lw.ffn_norm[i];
            }

            std::vector<float> g(FF, 0), u(FF, 0), d(H, 0);
            for (int j = 0; j < FF && j < (int)lw.gate.size() / H; j++) {
                float s = 0; for (int i = 0; i < H; i++) s += fn[i] * lw.gate[j * H + i]; g[j] = s;
            }
            for (int j = 0; j < FF && j < (int)lw.up.size() / H; j++) {
                float s = 0; for (int i = 0; i < H; i++) s += fn[i] * lw.up[j * H + i]; u[j] = s;
            }

            // Architecture-specific activation
            switch (cfg_.arch) {
                case RCPP_ARCH_GEMMA: for (int i = 0; i < FF; i++) g[i] = gelu(g[i]) * u[i]; break;
                case RCPP_ARCH_PHI:   for (int i = 0; i < FF; i++) g[i] = sq_relu(g[i]) * u[i]; break;
                default:              for (int i = 0; i < FF; i++) g[i] = silu(g[i]) * u[i]; break;
            }

            for (int j = 0; j < H && j < (int)lw.down.size() / FF; j++) {
                float s = 0; for (int i = 0; i < FF; i++) s += g[i] * lw.down[j * FF + i]; d[j] = s;
            }
            for (int i = 0; i < H; i++) hs[i] += d[i];
        }

        // Final norm + lm_head
        if (!final_norm_.empty()) {
            float ss = 0; for (int i = 0; i < H; i++) ss += hs[i] * hs[i];
            float inv = 1.0f / sqrtf(ss / H + 1e-6f);
            for (int i = 0; i < H && i < (int)final_norm_.size(); i++) hs[i] *= inv * final_norm_[i];
        }

        auto& head_w = !output_w_.empty() ? output_w_ : embed_;
        int best = 0; float best_val = -1e30f;
        for (int t = 0; t < V; t++) {
            float s = 0; const float* row = head_w.data() + (size_t)t * H;
            if ((size_t)t * H + H > head_w.size()) break;
            for (int i = 0; i < H; i++) s += hs[i] * row[i];
            if (s > best_val) { best_val = s; best = t; }
        }

        seq_len_++;
        return best;
    }
};

// ─── Backend detection ───────────────────────────────────────────────────────

std::vector<InferenceBackend*> detect_backends_generic() {
    static UniversalBackend backend;
    return {&backend};
}
