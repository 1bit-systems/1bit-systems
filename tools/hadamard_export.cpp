// hadamard_export — Convert F16 GGUF → Hadamard-rotated INT8 .h1b
//
// Pipeline:
//   1. Read F16 GGUF weights
//   2. Per-row Walsh-Hadamard butterfly rotation (B=128 blocks)
//   3. Per-row INT8 quantization (symmetric, max_abs / 127)
//   4. Write .h1b v5 with H1B_FLAG_HADAMARD_ROTATED flag
//
// The output .h1b uses format_version=5, weight_format=WMMA_I8.
// Weights are stored as flat int8_t + per-row fp32 scales.
// Layer norms are preserved as fp32 sidecar (same as gguf_to_h1b).
//
// Usage:
//   hadamard_export --input model-f16.gguf --output model-hadamard

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <cmath>

// ── GGUF reader (same pattern as gguf_to_h1b.cpp) ───────────────────────

struct GgufTensor {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t dtype;
    uint64_t offset;
};

class GgufReader {
public:
    std::ifstream f;
    uint64_t data_start = 0;
    uint32_t version = 0, alignment = 32;
    std::string arch;
    uint32_t hs=0, is_=0, layers=0, heads=0, kv=0, vocab=0, maxseq=0;
    float rope_theta = 1000000.0f;
    std::map<std::string,GgufTensor> tensors;

    bool open(const std::string& path) {
        f.open(path, std::ios::binary); if (!f) return false;
        char magic[4]; f.read(magic,4);
        if (std::strncmp(magic,"GGUF",4)!=0) return false;
        auto ru32=[&](uint32_t&v){f.read((char*)&v,4);return!!f;};
        auto ru64=[&](uint64_t&v){f.read((char*)&v,8);return!!f;};
        auto rstr=[&](std::string&s){
            uint64_t l; if(!ru64(l))return false;
            if(l>0){s.resize((size_t)l);f.read(&s[0],l);}else s.clear();
            return!!f;
        };
        auto skip=[&](uint32_t vt){
            switch(vt){
                case 0:case 1:case 7:f.seekg(1,std::ios::cur);break;
                case 2:case 3:f.seekg(2,std::ios::cur);break;
                case 4:case 5:case 6:f.seekg(4,std::ios::cur);break;
                case 8:{std::string s;rstr(s);}break;
                case 9:{uint32_t at;ru32(at);uint64_t al;ru64(al);
                    if(at==8)for(uint64_t j=0;j<al;++j){std::string s;rstr(s);}
                    else f.seekg((long)(al*4),std::ios::cur);
                }break;
                case 10:case 11:case 12:f.seekg(8,std::ios::cur);break;
            }
        };

        ru32(version);
        uint64_t nt, nk; ru64(nt); ru64(nk);
        for(uint64_t i=0;i<nk;++i){
            std::string k; rstr(k); uint32_t vt; ru32(vt);
            if(vt==8){std::string v;rstr(v);
                if(k=="general.architecture")arch=v;
                else if(k.find("hidden_size")!=std::string::npos
                     || k.find("embedding_length")!=std::string::npos)hs=(uint32_t)std::stoul(v);
                else if(k.find("intermediate_size")!=std::string::npos
                     || k.find("feed_forward_length")!=std::string::npos)is_=(uint32_t)std::stoul(v);
                else if(k=="qwen3.block_count")layers=(uint32_t)std::stoul(v);
                else if(k.find("head_count")!=std::string::npos&&k.find("kv")==std::string::npos)heads=(uint32_t)std::stoul(v);
                else if(k.find("head_count_kv")!=std::string::npos)kv=(uint32_t)std::stoul(v);
                else if(k.find("vocab_size")!=std::string::npos)vocab=(uint32_t)std::stoul(v);
            }else if(vt==4){uint32_t vu;ru32(vu);
                if(k.find("hidden_size")!=std::string::npos
                || k.find("embedding_length")!=std::string::npos)hs=vu;
                else if(k.find("intermediate_size")!=std::string::npos
                || k.find("feed_forward_length")!=std::string::npos)is_=vu;
                else if(k=="qwen3.block_count")layers=vu;
                else if(k.find("head_count")!=std::string::npos&&k.find("kv")==std::string::npos)heads=vu;
                else if(k.find("head_count_kv")!=std::string::npos)kv=vu;
                else if(k.find("vocab_size")!=std::string::npos)vocab=vu;
                else if(k.find("max_position_embeddings")!=std::string::npos)maxseq=vu;
            }else if(vt==6){float vf;f.read((char*)&vf,4);
                if(k.find("rope.freq_base")!=std::string::npos)rope_theta=vf;
            }else skip(vt);
        }
        for(uint64_t i=0;i<nt;++i){
            GgufTensor t; rstr(t.name); uint32_t nd; ru32(nd);
            t.shape.resize(nd);
            for(uint32_t d=0;d<nd;++d)ru64(t.shape[d]);
            ru32(t.dtype); ru64(t.offset);
            tensors[t.name]=t;
        }
        if(vocab==0){
            auto it=tensors.find("token_embd.weight");
            if(it!=tensors.end()&&it->second.shape.size()>=2)
                vocab=(uint32_t)it->second.shape[1];
        }
        data_start=(uint64_t)f.tellg();
        uint64_t rem=data_start%alignment;
        if(rem)data_start+=alignment-rem;
        return true;
    }

    bool read_f32(const std::string& name, std::vector<float>& out) {
        auto it=tensors.find(name);
        if(it==tensors.end())return false;
        const auto& t=it->second;
        size_t n=1; for(auto d:t.shape)n*=(size_t)d;
        if(t.dtype!=0&&t.dtype!=1)return false;
        f.seekg((long)(data_start+t.offset));
        if(t.dtype==0){out.resize(n);f.read((char*)out.data(),n*4);return(size_t)f.gcount()==n*4;}
        std::vector<uint16_t> f16(n); f.read((char*)f16.data(),n*2);
        if((size_t)f.gcount()!=n*2)return false;
        out.resize(n);
        for(size_t i=0;i<n;++i){
            uint16_t h=f16[i]; uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff;
            uint32_t b;
            if(e==0){if(m==0)b=s<<31;else{int e2=-1;uint32_t m2=m;while(!(m2&0x400)){m2<<=1;--e2;}b=(s<<31)|((e2+127)<<23)|((m2&0x3ff)<<13);}}
            else if(e==31)b=(s<<31)|0x7f800000|(m<<13);
            else b=(s<<31)|((e+112)<<23)|(m<<13);
            memcpy(&out[i],&b,4);
        }
        return true;
    }
};

// ── Hadamard butterfly transform ────────────────────────────────────────
constexpr int HADAMARD_BLOCK = 128;

static void hadamard_rotate_row(float* row, int K) {
    // In-place butterfly FWHT per block of B=128.
    for (int base = 0; base < K; base += HADAMARD_BLOCK) {
        float* blk = row + base;
        for (int stage = 0; (1 << stage) < HADAMARD_BLOCK; ++stage) {
            int dist = 1 << stage;
            for (int i = 0; i < HADAMARD_BLOCK; ++i) {
                if ((i & dist) == 0) {
                    int j = i | dist;
                    float a = blk[i], b = blk[j];
                    blk[i] = a + b;
                    blk[j] = a - b;
                }
            }
        }
        constexpr float inv_sqrt_b = 1.0f / 11.31370849898476f;
        for (int i = 0; i < HADAMARD_BLOCK; ++i)
            blk[i] *= inv_sqrt_b;
    }
}

static void quantize_row_i8(const float* row, int K,
                            std::vector<int8_t>& out_i8, float& out_scale) {
    float max_abs = 0.0f;
    for (int i = 0; i < K; ++i) {
        float a = fabsf(row[i]);
        if (a > max_abs) max_abs = a;
    }
    out_scale = max_abs / 127.0f;
    if (out_scale < 1e-10f) out_scale = 1e-10f;
    out_i8.resize(K);
    for (int i = 0; i < K; ++i) {
        float v = row[i] / out_scale;
        v = fminf(fmaxf(roundf(v), -128.0f), 127.0f);
        out_i8[i] = (int8_t)v;
    }
}

// ── Main ─────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    const char* in_path = nullptr;
    const char* out_base = nullptr;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--input" && i + 1 < argc) in_path = argv[++i];
        else if (a == "--output" && i + 1 < argc) out_base = argv[++i];
    }
    if (!in_path || !out_base) {
        fprintf(stderr, "usage: hadamard_export --input model-f16.gguf --output model-hadamard\n");
        return 1;
    }

    GgufReader re;
    if (!re.open(in_path)) { fprintf(stderr, "[err] cannot open %s\n", in_path); return 1; }
    uint32_t hs = re.hs, is = re.is_, nl = re.layers, nh = re.heads, nkv = re.kv, voc = re.vocab;
    uint32_t hd = hs / nh;
    float rt = re.rope_theta;
    fprintf(stderr, "[hadamard] hs=%u is=%u L=%u nh=%u nkv=%u hd=%u voc=%u\n", hs, is, nl, nh, nkv, hd, voc);
    if (hs == 0 || is == 0 || nl == 0 || voc == 0) { fprintf(stderr, "[err] bad config\n"); return 1; }

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

    auto rd = [&](const std::string& nm) -> std::vector<float> {
        std::vector<float> v;
        re.read_f32(nm, v);
        if (v.empty()) fprintf(stderr, "[warn] missing %s\n", nm.c_str());
        return v;
    };

    // Read token_embd and output_norm (preserved as fp32 in sidecar)
    auto te = rd("token_embd.weight");
    auto on = rd("output_norm.weight");

    struct N { std::vector<float> an, fn, qn, kn; };
    struct W { std::vector<float> q, k, v, o, g, u, d; };
    std::vector<N> nn(nl);
    std::vector<W> ww(nl);
    for (uint32_t l = 0; l < nl; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";
        nn[l].an = rd(p + "attn_norm.weight");
        nn[l].fn = rd(p + "ffn_norm.weight");
        nn[l].qn = rd(p + "attn_q_norm.weight");
        nn[l].kn = rd(p + "attn_k_norm.weight");
        ww[l].q  = rd(p + "attn_q.weight");
        ww[l].k  = rd(p + "attn_k.weight");
        ww[l].v  = rd(p + "attn_v.weight");
        ww[l].o  = rd(p + "attn_output.weight");
        ww[l].g  = rd(p + "ffn_gate.weight");
        ww[l].u  = rd(p + "ffn_up.weight");
        ww[l].d  = rd(p + "ffn_down.weight");
    }

    // ── Hadamard-rotate + INT8-quantize each weight matrix per-row ───────
    struct Packed { std::vector<int8_t> data; std::vector<float> scales; int rows, cols; };
    auto process = [&](const std::vector<float>& w, int rows, int cols) -> Packed {
        Packed p; p.rows = rows; p.cols = cols;
        p.data.resize((size_t)rows * cols);
        p.scales.resize(rows);
        std::vector<float> row(cols);
        std::vector<int8_t> row_i8;
        for (int r = 0; r < rows; ++r) {
            memcpy(row.data(), w.data() + (size_t)r * cols, cols * sizeof(float));
            hadamard_rotate_row(row.data(), cols);
            quantize_row_i8(row.data(), cols, row_i8, p.scales[r]);
            memcpy(p.data.data() + (size_t)r * cols, row_i8.data(), cols);
        }
        return p;
    };

    fprintf(stderr, "[hadamard] rotating + quantizing weights...\n");
    std::vector<Packed> pw(nl * 7);
    for (uint32_t l = 0; l < nl; ++l) {
        pw[l*7+0] = process(ww[l].q, nh*hd, (int)hs);
        pw[l*7+1] = process(ww[l].k, nkv*hd, (int)hs);
        pw[l*7+2] = process(ww[l].v, nkv*hd, (int)hs);
        pw[l*7+3] = process(ww[l].o, (int)hs, nh*hd);
        pw[l*7+4] = process(ww[l].g, (int)is, (int)hs);
        pw[l*7+5] = process(ww[l].u, (int)is, (int)hs);
        pw[l*7+6] = process(ww[l].d, (int)hs, (int)is);
    }

    // ── Write .h1b v5 ────────────────────────────────────────────────────
    std::string h1b_path = std::string(out_base) + ".h1b";
    fprintf(stderr, "[hadamard] writing %s...\n", h1b_path.c_str());
    {
        std::ofstream f(h1b_path, std::ios::binary);
        if (!f) { fprintf(stderr, "[err] cannot write %s\n", h1b_path.c_str()); return 1; }

        // Header: magic, version=5, config[9], flags=0x1 (HADAMARD_ROTATED), extras
        f.write("H1B\0", 4);
        int32_t ver = 5; f.write((const char*)&ver, 4);
        int32_t cfg[9] = {(int32_t)hs, (int32_t)is, (int32_t)nl, (int32_t)nh,
                          (int32_t)nkv, (int32_t)voc, (int32_t)4096, 1,
                          (int32_t)0x1u};  // flags = H1B_FLAG_HADAMARD_ROTATED
        f.write((const char*)cfg, 36);
        float extras[2] = {rt, 1e-5f};
        f.write((const char*)extras, 8);

        // Embedding: fp32 zeros placeholder (normed by sidecar GGUF)
        std::vector<float> zv((size_t)voc * hs, 0);
        f.write((const char*)zv.data(), zv.size() * 4);
        std::vector<float> zh(hs, 0);
        f.write((const char*)zh.data(), zh.size() * 4);

        // Per-layer norm zeros (norms go in sidecar GGUF)
        for (uint32_t l = 0; l < nl; ++l) {
            for (int i = 0; i < 8; ++i) f.write((const char*)zh.data(), hs * 4);
            std::vector<float> zi(is, 0);
            f.write((const char*)zi.data(), is * 4);
        }

        // INT8 weights: flat int8_t [rows, cols] + fp32 scales [rows]
        for (uint32_t l = 0; l < nl; ++l) {
            for (int slot = 0; slot < 7; ++slot) {
                const auto& p = pw[l*7+slot];
                f.write((const char*)p.data.data(), p.data.size());
                f.write((const char*)p.scales.data(), p.scales.size() * sizeof(float));
            }
        }
    }

    // ── Write sidecar GGUF (norms only) ──────────────────────────────────
    std::string gguf_path = std::string(out_base) + ".gguf";
    fprintf(stderr, "[hadamard] writing %s...\n", gguf_path.c_str());
    {
        std::ofstream f(gguf_path, std::ios::binary);
        if (!f) { fprintf(stderr, "[err] cannot write %s\n", gguf_path.c_str()); return 1; }

        auto w32 = [&](uint32_t v) { f.write((const char*)&v, 4); };
        auto w64 = [&](uint64_t v) { f.write((const char*)&v, 8); };
        auto wstr = [&](const std::string& s) { w64(s.size()); f.write(s.data(), s.size()); };
        auto wkvstr = [&](const std::string& k, const std::string& v) { wstr(k); w32(8); wstr(v); };
        auto wkvu32 = [&](const std::string& k, uint32_t v) { wstr(k); w32(4); w32(v); };
        auto wtinfo = [&](const std::string& n, const std::vector<uint64_t>& sh,
                          uint32_t dt, uint64_t off) {
            wstr(n); w32((uint32_t)sh.size());
            for (auto d : sh) w64(d);
            w32(dt); w64(off);
        };

        uint64_t nt = 4 * nl + 2, nk = 9;
        f.write("GGUF", 4); w32(3); w64(nt); w64(nk);
        wkvstr("general.architecture", re.arch.c_str());
        wkvu32("qwen3.hidden_size", hs);
        wkvu32("qwen3.feed_forward_length", is);
        wkvu32("qwen3.block_count", nl);
        wkvu32("qwen3.attention.head_count", nh);
        wkvu32("qwen3.attention.head_count_kv", nkv);
        wkvu32("qwen3.vocab_size", voc);
        wkvu32("qwen3.max_position_embeddings", 4096);
        wkvu32("qwen3.rope.freq_base", (uint32_t)rt);

        uint64_t off = 0;
        for (uint32_t l = 0; l < nl; ++l) {
            wtinfo("blk." + std::to_string(l) + ".attn_norm.weight", {(uint64_t)hs}, 0, off); off += hs * 4;
            wtinfo("blk." + std::to_string(l) + ".ffn_norm.weight",   {(uint64_t)hs}, 0, off); off += hs * 4;
            wtinfo("blk." + std::to_string(l) + ".attn_q_norm.weight",{(uint64_t)hd}, 0, off); off += hd * 4;
            wtinfo("blk." + std::to_string(l) + ".attn_k_norm.weight",{(uint64_t)hd}, 0, off); off += hd * 4;
        }
        wtinfo("output_norm.weight", {(uint64_t)hs}, 0, off); off += hs * 4;
        wtinfo("token_embd.weight", {(uint64_t)hs, (uint64_t)voc}, 0, off);

        // Align to 32 bytes, then write norm data.
        uint64_t pos = (uint64_t)f.tellp();
        uint64_t rem = pos % 32;
        if (rem) for (uint64_t i = 0; i < 32 - rem; ++i) f.put(0);

        for (uint32_t l = 0; l < nl; ++l) {
            f.write((const char*)nn[l].an.data(), hs * 4);
            f.write((const char*)nn[l].fn.data(), hs * 4);
            f.write((const char*)nn[l].qn.data(), hd * 4);
            f.write((const char*)nn[l].kn.data(), hd * 4);
        }
        f.write((const char*)on.data(), hs * 4);
        f.write((const char*)te.data(), (size_t)voc * hs * 4);
    }

    fprintf(stderr, "[hadamard] done: %s + %s\n", h1b_path.c_str(), gguf_path.c_str());
    return 0;
}
