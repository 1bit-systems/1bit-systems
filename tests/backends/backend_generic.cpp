// backend_generic.cpp — Universal CPU inference backend for ANY GGUF model
// Part of the unified zaya_server binary.
//
// Loads any GGUF (all architectures, all quantizations) or SAFETENSORS model.
// Routes compute to: GPU (ROCm/Vulkan) first, NPU for overflow/help, CPU fallback.
// Uses unified memory for GPU+NPU hybrid operation on Strix Halo.

#include "backend.h"
#include "colibri_kernels.h"
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

// USE_COLIBRI_Q4=1: quantize GGUF weights to int4 at load time (8x smaller)
// Set to 0 for original f32 behavior.
#ifndef USE_COLIBRI_Q4
#define USE_COLIBRI_Q4 1
#endif

// Forward declaration of gguf_dequant from the shared gguf_reader module
// (libgguf_reader, compiled from src/gguf_reader.cpp). Not including
// gguf_reader.h here to avoid ::GgufReader class name collision with the
// local generic_backend::GgufReader defined below.
struct GgufBlockInfo { int block_size; int block_bytes; };
bool gguf_dequant(uint32_t dtype, const uint8_t* data, float* out, int count);

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
#define GGUF_TYPE_Q5_1 7
#define GGUF_TYPE_Q8_0 8
#define GGUF_TYPE_Q2_K 10
#define GGUF_TYPE_Q3_K 11
#define GGUF_TYPE_Q4_K 12
#define GGUF_TYPE_Q5_K 13
#define GGUF_TYPE_Q6_K 14
#define GGUF_TYPE_Q8_K 15

// NOTE: this backend has its OWN hand-rolled GGUF parser. It must NOT live at
// global scope under the names GgufReader / GgufTensor, because include/
// gguf_reader.h defines different classes with the same (global) names and a
// different memory layout. Both TUs end up in the same binary (backend_manager
// links backend_generic.cpp), so the linker kept one ::GgufReader::open for
// both, corrupting memory at the call site in backend_hip.cpp and crashing
// every GGUF load with SIGFPE (issue #799). The namespace below makes these
// distinct symbols.
namespace generic_backend {
struct GgufTensor { std::string name; std::vector<uint64_t> shape; uint32_t dtype; uint64_t file_offset; };

// Shared block_info function (was a lambda inside open(), but needs to be
// callable from both open() and get()). Must match ggml_type block sizes.
static std::pair<int,int> gguf_block_info(uint32_t dtype) {
    switch (dtype) {
        case 0: return {1, 4}; case 1: return {1, 2}; case 2: return {32, 18};
        case 3: return {32, 20}; case 6: return {32, 22}; case 7: return {32, 24};
        case 8: return {32, 34}; case 9: return {256, 72}; case 10: return {256, 104};
        case 11: return {256, 144}; case 12: return {256, 176}; case 13: return {256, 210};
        case 14: return {256, 292}; case 15: return {256, 0};
        default: return {32, 0};
    }
}

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
                uint32_t n_arr; fread(&n_arr, 4, 1, f); uint32_t at; fread(&at, 4, 1, f); uint64_t al = n_arr;
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
        uint64_t data_off = ftell(f); data_off = (data_off + 31) & ~31;
        for (auto& [name, t] : tensors) {
            t.file_offset = data_off;
            uint64_t n_elems = 1; for (auto s : t.shape) n_elems *= s;
            auto [bs, bpb] = gguf_block_info(t.dtype);
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
        } else if (t.dtype == GGUF_TYPE_Q5_0) {
            // Q5_0: scale[2] + high_bits[4] + nibbles[16] = 22 bytes / 32 elements
            int blocks = (n + 31) / 32;
            for (int b = 0; b < blocks; b++) {
                uint16_t sh; fread(&sh, 2, 1, f); float s = fp16_to_fp32(sh);
                uint32_t qh; fread(&qh, 4, 1, f);
                uint8_t q[16]; fread(q, 1, 16, f);
                for (int j = 0; j < 32 && b*32+j < (int)n; j++) {
                    int lo = (j < 16) ? (q[j] & 0xF) : (q[j-16] >> 4);
                    int hi = (qh >> j) & 1;
                    scratch[b*32+j] = (int8_t)(lo | (hi << 4)) * s;
                }
            }
        } else if (t.dtype == GGUF_TYPE_Q5_1) {
            // Q5_1: scale[2] + min[2] + high_bits[4] + nibbles[16] = 24 bytes / 32 elements
            int blocks = (n + 31) / 32;
            for (int b = 0; b < blocks; b++) {
                uint16_t sh; fread(&sh, 2, 1, f); float s = fp16_to_fp32(sh);
                uint16_t mh; fread(&mh, 2, 1, f); float m = fp16_to_fp32(mh);
                uint32_t qh; fread(&qh, 4, 1, f);
                uint8_t q[16]; fread(q, 1, 16, f);
                for (int j = 0; j < 32 && b*32+j < (int)n; j++) {
                    int lo = (j < 16) ? (q[j] & 0xF) : (q[j-16] >> 4);
                    int hi = (qh >> j) & 1;
                    scratch[b*32+j] = (int8_t)(lo | (hi << 4)) * s + m;
                }
            }
        } else {
            // Unknown quant type — use the shared gguf_dequant function
            // from gguf_reader.h (included at file scope). This handles
            // all K-quants (Q2_K through Q8_K) and any future dtypes that
            // the shared dequant module knows about, without duplicating
            // hundreds of lines of superblock dequant math here.
            auto [bs, bpb] = gguf_block_info(t.dtype);
            if (bpb == 0) { bpb = 4; bs = 1; }
            uint64_t n_blocks = (n + bs - 1) / bs;
            std::vector<uint8_t> raw(n_blocks * bpb);
            fseek(f, t.file_offset, SEEK_SET);
            fread(raw.data(), 1, raw.size(), f);
            if (!gguf_dequant(t.dtype, raw.data(), scratch.data(), (int)n)) {
                // gguf_dequant returned false (unrecognized dtype) — fall
                // back to raw float32 read from the same file position.
                fprintf(stderr, "  Universal: dtype=%u not handled by gguf_dequant, trying raw f32\n", t.dtype);
                fseek(f, t.file_offset, SEEK_SET);
                fread(scratch.data(), 4, std::min(n, (uint64_t)scratch.size()), f);
            }
        }
        return scratch.data();
    }
    void close() { if (f) fclose(f); f = nullptr; }
};
}  // namespace generic_backend

// Local aliases so the rest of this file (UniversalBackend::gguf_, etc.) keeps
// using the unqualified names without touching the shared header's types.
using GgufReader = generic_backend::GgufReader;

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
#if USE_COLIBRI_Q4
    std::vector<uint8_t> q4_embed_, q4_output_;
    std::vector<float> qs_embed_, qs_output_;
#endif

    struct LayerW {
        std::vector<float> attn_norm, ffn_norm;  // RMSNorm weights
        std::vector<float> wq, wk, wv, wo;        // Attention Q/K/V/O
        std::vector<float> gate, up, down;         // FFN
#if USE_COLIBRI_Q4
        // int4 quantized versions — 8x smaller, cosim > 0.997
        std::vector<uint8_t> q4_wq, q4_wk, q4_wv, q4_wo;
        std::vector<float>  qs_wq, qs_wk, qs_wv, qs_wo;
        std::vector<uint8_t> q4_gate, q4_up, q4_down;
        std::vector<float>  qs_gate, qs_up, qs_down;
#endif
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

#if USE_COLIBRI_Q4
        // Quantize embedding to int4
        if (!embed_.empty()) {
            int VO = (int)embed_.size() / H;
            q4_embed_.resize((size_t)VO * ((H+1)/2));
            qs_embed_.resize(VO);
            colibri_quantize_matrix_q4(embed_.data(), q4_embed_.data(), qs_embed_.data(), VO, H);
        }
#endif

        // Final norm
        load_t("output_norm.weight", final_norm_, H);
        if (final_norm_.empty()) load_t("model.norm.weight", final_norm_, H);

        // LM head (optional — may be tied)
        size_t out_n = 0;
        float* out_data = gguf_.get("output.weight", &out_n);
        if (out_data && out_n == (size_t)V * H) output_w_.assign(out_data, out_data + out_n);
#if USE_COLIBRI_Q4
        if (!output_w_.empty()) {
            q4_output_.resize((size_t)V * ((H+1)/2));
            qs_output_.resize(V);
            colibri_quantize_matrix_q4(output_w_.data(), q4_output_.data(), qs_output_.data(), V, H);
        }
#endif

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

#if USE_COLIBRI_Q4
            // Quantize this layer's weights to int4
            auto q4_row = [&](std::vector<float>& f32, int I,
                              std::vector<uint8_t>& q4, std::vector<float>& qs) {
                if (f32.empty()) return;
                int O = (int)f32.size() / I;
                q4.resize((size_t)O * ((I+1)/2));
                qs.resize(O);
                colibri_quantize_matrix_q4(f32.data(), q4.data(), qs.data(), O, I);
            };
            int QK = NH * HD, KK = NKV * HD, VK = NKV * HD;
            q4_row(lw.wq, H, lw.q4_wq, lw.qs_wq);
            q4_row(lw.wk, H, lw.q4_wk, lw.qs_wk);
            q4_row(lw.wv, H, lw.q4_wv, lw.qs_wv);
            q4_row(lw.wo, QK, lw.q4_wo, lw.qs_wo);
            q4_row(lw.gate, H, lw.q4_gate, lw.qs_gate);
            q4_row(lw.up, H, lw.q4_up, lw.qs_up);
            q4_row(lw.down, FF, lw.q4_down, lw.qs_down);
#endif
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
#if USE_COLIBRI_Q4
            if (!lw.q4_wq.empty()) {
                colibri_matmul_q4(q.data(), norm.data(), lw.q4_wq.data(), lw.qs_wq.data(), 1, H, QD);
                colibri_matmul_q4(k.data(), norm.data(), lw.q4_wk.data(), lw.qs_wk.data(), 1, H, KD);
                colibri_matmul_q4(v.data(), norm.data(), lw.q4_wv.data(), lw.qs_wv.data(), 1, H, KD);
            } else
#endif
            {
                for (int j = 0; j < QD && j < (int)lw.wq.size() / H; j++) {
                    float s = 0; for (int i = 0; i < H; i++) s += norm[i] * lw.wq[j * H + i]; q[j] = s;
                }
                for (int j = 0; j < KD && j < (int)lw.wk.size() / H; j++) {
                    float s = 0; for (int i = 0; i < H; i++) s += norm[i] * lw.wk[j * H + i]; k[j] = s;
                }
                for (int j = 0; j < KD && j < (int)lw.wv.size() / H; j++) {
                    float s = 0; for (int i = 0; i < H; i++) s += norm[i] * lw.wv[j * H + i]; v[j] = s;
                }
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
#if USE_COLIBRI_Q4
            if (!lw.q4_wo.empty())
                colibri_matmul_q4(ao.data(), attn_out.data(), lw.q4_wo.data(), lw.qs_wo.data(), 1, QD, H);
            else
#endif
            {
                for (int j = 0; j < H && j < (int)lw.wo.size() / QD; j++) {
                    float s = 0; for (int i = 0; i < QD; i++) s += attn_out[i] * lw.wo[j * QD + i]; ao[j] = s;
                }
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
#if USE_COLIBRI_Q4
            if (!lw.q4_gate.empty() && !lw.q4_up.empty()) {
                colibri_matmul_q4(g.data(), fn.data(), lw.q4_gate.data(), lw.qs_gate.data(), 1, H, FF);
                colibri_matmul_q4(u.data(), fn.data(), lw.q4_up.data(), lw.qs_up.data(), 1, H, FF);
            } else
#endif
            {
                for (int j = 0; j < FF && j < (int)lw.gate.size() / H; j++) {
                    float s = 0; for (int i = 0; i < H; i++) s += fn[i] * lw.gate[j * H + i]; g[j] = s;
                }
                for (int j = 0; j < FF && j < (int)lw.up.size() / H; j++) {
                    float s = 0; for (int i = 0; i < H; i++) s += fn[i] * lw.up[j * H + i]; u[j] = s;
                }
            }

            // Architecture-specific activation
            switch (cfg_.arch) {
                case RCPP_ARCH_GEMMA: for (int i = 0; i < FF; i++) g[i] = gelu(g[i]) * u[i]; break;
                case RCPP_ARCH_PHI:   for (int i = 0; i < FF; i++) g[i] = sq_relu(g[i]) * u[i]; break;
                default:              for (int i = 0; i < FF; i++) g[i] = silu(g[i]) * u[i]; break;
            }

#if USE_COLIBRI_Q4
            if (!lw.q4_down.empty())
                colibri_matmul_q4(d.data(), g.data(), lw.q4_down.data(), lw.qs_down.data(), 1, FF, H);
            else
#endif
            {
                for (int j = 0; j < H && j < (int)lw.down.size() / FF; j++) {
                    float s = 0; for (int i = 0; i < FF; i++) s += g[i] * lw.down[j * FF + i]; d[j] = s;
                }
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
#if USE_COLIBRI_Q4
        bool has_head_q4 = !output_w_.empty() && !q4_output_.empty();
        bool has_embed_q4 = !q4_embed_.empty();
        if (has_head_q4) {
            std::vector<float> logits(V, 0);
            colibri_matmul_q4(logits.data(), hs.data(), q4_output_.data(), qs_output_.data(), 1, H, V);
            for (int t = 0; t < V; t++) if (logits[t] > best_val) { best_val = logits[t]; best = t; }
        } else if (has_embed_q4) {
            std::vector<float> logits(V, 0);
            colibri_matmul_q4(logits.data(), hs.data(), q4_embed_.data(), qs_embed_.data(), 1, H, V);
            for (int t = 0; t < V; t++) if (logits[t] > best_val) { best_val = logits[t]; best = t; }
        } else
#endif
        {
            for (int t = 0; t < V; t++) {
                float s = 0; const float* row = head_w.data() + (size_t)t * H;
                if ((size_t)t * H + H > head_w.size()) break;
                for (int i = 0; i < H; i++) s += hs[i] * row[i];
                if (s > best_val) { best_val = s; best = t; }
            }
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
