#include "gguf_parser.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <map>

static uint16_t f32bf16(float f) { uint32_t u; memcpy(&u, &f, 4); return u >> 16; }

// TQ1_0 (type 36) dequant
#define QKK 256
static void deq_tq1(const uint8_t* d, float* o, int n) {
    const uint8_t p3[6] = {1,3,9,27,81,243};
    int nb = n / QKK, qb = (QKK - 4*QKK/64)/5, qh = QKK/64, bs = qb + qh + 2;
    for (int i = 0; i < nb; i++) {
        const uint8_t* blk = d + i * bs;
        uint16_t dh; memcpy(&dh, blk + qb + qh, 2);
        uint32_t bv = (dh & 0x7FFF) << 13 | (dh & 0x8000) << 16;
        if ((dh & 0x7C00) == 0x7C00) bv |= 0x7F800000;
        float s; memcpy(&s, &bv, 4);
        const uint8_t* qs = blk, *qh2 = blk + qb;
        auto emit = [&](uint8_t qv) { int z = (((uint16_t)qv * 3) >> 8) - 1; *o++ = (float)z * s; };
        for (int j = 0; j < 32; j += 32) for (int nn = 0; nn < 5; nn++) for (int m = 0; m < 32; m++) emit(qs[j+m] * p3[nn]);
        for (int j = 32; j < qb; j += 16) for (int nn = 0; nn < 5; nn++) for (int m = 0; m < 16; m++) emit(qs[j+m] * p3[nn]);
        for (int nn = 0; nn < 4; nn++) for (int j = 0; j < qh; ++j) emit(qh2[j] * p3[nn]);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s input.gguf output.q4nx\n", argv[0]); return 1; }
    
    // Hardcoded BitNet b1.58-2B-4T dimensions
    const int H=2560, NC=30, NH=20, NKV=5, HD=128, IM=6912, NV=128256;
    const int QOUT=NH*HD, KVOUT=NKV*HD;  // 2560, 640
    
    GGUFReader r;
    if (!r.open(argv[1])) { fprintf(stderr, "FAIL open\n"); return 1; }
    GGUFModel info;
    if (!info.parse(r)) { fprintf(stderr, "FAIL parse\n"); return 1; }
    
    fprintf(stderr, "GGUF: %s, %zu tensors\n", info.arch.c_str(), info.tensors.size());
    fprintf(stderr, "Using dimensions: H=%d NC=%d NH=%d NKV=%d IM=%d NV=%d\n", H, NC, NH, NKV, IM, NV);
    
    // Build Q4NX output
    std::vector<uint8_t> od;
    std::string json = "{";
    
    auto add = [&](const std::string& name, const std::string& dtype,
                   int s0, int s1, const void* data, size_t n) {
        if (json.size() > 1) json += ",";
        uint64_t off = od.size(); od.resize(od.size() + n);
        memcpy(od.data() + off, data, n);
        char buf[4096];
        snprintf(buf, sizeof(buf), "\"%s\":{\"dtype\":\"%s\",\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]}",
                 name.c_str(), dtype.c_str(), s0, s1, (size_t)off, (size_t)(off+n));
        json += buf;
    };
    
    // Find tensor by name
    auto find = [&](const std::string& suffix) -> const GGUFModel::Tensor* {
        for (auto& t : info.tensors)
            if (t.name.size() >= suffix.size() &&
                t.name.compare(t.name.size()-suffix.size(), suffix.size(), suffix) == 0)
                return &t;
        return nullptr;
    };
    auto find_layer = [&](int l, const std::string& suffix) -> const GGUFModel::Tensor* {
        std::string prefix = "blk." + std::to_string(l) + ".";
        for (auto& t : info.tensors) {
            if (t.name.size() >= prefix.size() + suffix.size() &&
                t.name.compare(0, prefix.size(), prefix) == 0 &&
                t.name.compare(t.name.size()-suffix.size(), suffix.size(), suffix) == 0)
                return &t;
        }
        return nullptr;
    };
    
    // Dequant tensor
    auto deq = [&](const GGUFModel::Tensor& t) -> float* {
        int n = 1; for (auto d : t.dims) n *= d;
        float* o = new float[n];
        if (t.type == 0) { memcpy(o, r.data + info.tensor_data_offset + t.file_offset, n*4); }
        else if (t.type == 1) {
            auto p = (const uint16_t*)(r.data + info.tensor_data_offset + t.file_offset);
            for (int i = 0; i < n; i++) {
                uint32_t b = (p[i] & 0x7FFF) << 13 | (p[i] & 0x8000) << 16;
                if ((p[i] & 0x7C00) == 0x7C00) b |= 0x7F800000;
                memcpy(&o[i], &b, 4);
            }
        } else if (t.type == 36) {
            deq_tq1(r.data + info.tensor_data_offset + t.file_offset, o, n);
        } else { fprintf(stderr, "Unknown type %d\n", t.type); delete[] o; return nullptr; }
        return o;
    };
    
    // Float to INT8
    auto toi8 = [](const float* f, int n) -> std::vector<int8_t> {
        float mx = 0; for (int i = 0; i < n; i++) { float a = fabsf(f[i]); if (a > mx) mx = a; }
        if (mx < 1e-12f) mx = 1.0f; float sc = 127.0f / mx;
        std::vector<int8_t> i8(n);
        for (int i = 0; i < n; i++) {
            float v = f[i]; if (!std::isfinite(v)) v = 0;
            int q = (int)roundf(v * sc); if (q > 127) q = 127; else if (q < -127) q = -127;
            i8[i] = (int8_t)q;
        }
        return i8;
    };
    
    // 1. Embeddings
    fprintf(stderr, "Embeddings...\n");
    auto te = find("token_embd.weight");
    if (te) {
        float* f = deq(*te); if (f) {
            std::vector<uint16_t> bf16((size_t)NV*H);
            for (size_t i = 0; i < (size_t)NV*H; i++) bf16[i] = f32bf16(f[i]);
            add("model.embed_tokens.weight", "BF16", NV, H, bf16.data(), bf16.size()*2);
            delete[] f;
        }
    }
    
    // 2. Output norm
    auto tn = find("output_norm.weight");
    if (tn) {
        float* f = deq(*tn); if (f) {
            std::vector<uint16_t> b(H); for (int i = 0; i < H; i++) b[i] = f32bf16(f[i]);
            add("model.norm.weight", "BF16", H, 0, b.data(), b.size()*2);
            delete[] f;
        }
    }
    
    // 3. Layer norms (BF16) + Q/K norms
    for (int l = 0; l < NC; l++) {
        auto t = find_layer(l, "attn_norm.weight");
        if (t) { float* f = deq(*t); if (f) {
            std::vector<uint16_t> b(H); for (int i = 0; i < H; i++) b[i] = f32bf16(f[i]);
            char buf[256]; snprintf(buf, sizeof(buf), "model.layers.%d.input_layernorm.weight", l);
            add(buf, "BF16", H, 0, b.data(), b.size()*2); delete[] f;
        }}
        t = find_layer(l, "ffn_norm.weight");
        if (t) { float* f = deq(*t); if (f) {
            std::vector<uint16_t> b(H); for (int i = 0; i < H; i++) b[i] = f32bf16(f[i]);
            char buf[256]; snprintf(buf, sizeof(buf), "model.layers.%d.post_attention_layernorm.weight", l);
            add(buf, "BF16", H, 0, b.data(), b.size()*2); delete[] f;
        }}
        // Also handle sub-norms if they exist
        t = find_layer(l, "attn_sub_norm.weight");
        if (t) { fprintf(stderr, "  L%d has attn_sub_norm\n", l); }
    }
    
    // 4. Per-layer projections + Q/K norms
    struct Proj { const char* gguf; const char* q4nx; int of; int inf; };
    std::vector<Proj> projs = {
        {"attn_q.weight", "self_attn.q_proj.weight", QOUT, H},
        {"attn_k.weight", "self_attn.k_proj.weight", KVOUT, H},
        {"attn_v.weight", "self_attn.v_proj.weight", KVOUT, H},
        {"attn_output.weight", "self_attn.o_proj.weight", H, QOUT},
        {"ffn_gate.weight", "mlp.gate_proj.weight", IM, H},
        {"ffn_up.weight", "mlp.up_proj.weight", IM, H},
        {"ffn_down.weight", "mlp.down_proj.weight", H, IM},
    };
    
    // Q/K norms (BitNet has attn_sub_norm for Q and K)
    // These are stored as sub-norm weights; the engine handles them if present
    
    for (int l = 0; l < NC; l++) {
        for (auto& p : projs) {
            auto t = find_layer(l, p.gguf);
            if (!t) { fprintf(stderr, "  L%d %s NOT FOUND\n", l, p.gguf); continue; }
            float* f = deq(*t);
            if (!f) continue;
            int n = t->dims[0] * t->dims[1];
            auto i8 = toi8(f, n);
            char buf[256];
            snprintf(buf, sizeof(buf), "model.layers.%d.%s", l, p.q4nx);
            // Q4NX format: shape=[i8_tile_rows, 5120]
            int tile_rows = (p.of + 31) / 32;
            int tile_cols = (p.inf + 255) / 256;
            int i8_rows = tile_rows * tile_cols;
            add(buf, "I8", i8_rows, 5120, i8.data(), i8.size());
            delete[] f;
        }
        if (l % 10 == 0) fprintf(stderr, "  L%d\n", l);
    }
    
    json += "}";
    
    fprintf(stderr, "\nWriting %s...\n", argv[2]);
    FILE* fout = fopen(argv[2], "wb");
    uint64_t hs = json.size(); fwrite(&hs, 8, 1, fout);
    fwrite(json.data(), 1, json.size(), fout);
    fwrite(od.data(), 1, od.size(), fout);
    fclose(fout);
    fprintf(stderr, "✅ %s (JSON: %zu bytes, Data: %.0f KB, Total: %.0f MB)\n",
            argv[2], json.size(), od.size()/1024.0, (8.0+json.size()+od.size())/1048576.0);
    r.close();
    return 0;
}
