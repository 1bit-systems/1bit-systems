// convert_zaya_to_h1b — Convert Zaya F16 GGUF → ternary-packed .h1b v6
// Run once (build-time tool). Inference uses bitnet_decode (pure C++, no Python).
//
// Build:
//   g++ -std=c++17 -O3 -o convert_zaya_to_h1b convert_zaya_to_h1b.cpp -lgguf
//   ./convert_zaya_to_h1b /path/to/model.gguf /path/to/output.h1b
//
// Or without libgguf (standalone GGUF reader):
//   g++ -std=c++17 -O3 -o convert_zaya_to_h1b convert_zaya_to_h1b.cpp
//   ./convert_zaya_to_h1b model.gguf model.h1b

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>

// ═══════════════════════════════════════════════════════════════
//  GGUF reader (zero external dependencies)
// ═══════════════════════════════════════════════════════════════

struct GgufTensor {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t dtype;
    uint64_t offset;
    size_t file_offset; // absolute byte offset in GGUF file
};

class GgufReader {
public:
    std::ifstream f;
    uint64_t data_start = 0;
    uint32_t alignment = 32;
    std::string arch;
    // Zaya-specific config
    uint32_t hs=0, is_=0, layers=0, heads=0, kv=0, vocab=0, maxseq=0;
    uint32_t num_experts=0, top_k=0, expert_hidden=0;
    uint32_t conv_kernel=0, cca_heads=0;
    uint32_t head_dim=0;
    float rope_theta=5000000.0f, rms_eps=1e-5f;
    float rope_dim=64.0f;
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
                    else f.seekg((long)(al*4),std::ios::cur);}break;
                case 10:case 11:case 12:f.seekg(8,std::ios::cur);break;
            }
        };
        auto rufloat=[&](float&v)->bool{
            uint32_t u; if(!ru32(u))return false;
            memcpy(&v,&u,4); return true;
        };

        uint32_t ver; ru32(ver);
        uint64_t nt, nk; ru64(nt); ru64(nk);
        for(uint64_t i=0;i<nk;++i){
            std::string k; rstr(k); uint32_t vt; ru32(vt);
            auto match=[&](const std::string& pat)->bool{
                if(pat.size()>k.size())return false;
                // Check if k ends with pat (for "zaya.*" keys)
                return k.size()>=pat.size() && k.substr(k.size()-pat.size())==pat;
            };
            auto read_val=[&]()->std::string{
                std::string vs; rstr(vs); return vs;
            };
            if(vt==8){std::string v;rstr(v);
                if(k=="general.architecture")arch=v;
                else if(k.find("hidden_size")!=std::string::npos||k.find("embedding_length")!=std::string::npos)hs=(uint32_t)std::stoul(v);
                else if(k.find("feed_forward_length")!=std::string::npos)is_=(uint32_t)std::stoul(v);
                else if(k.find("block_count")!=std::string::npos)layers=(uint32_t)std::stoul(v);
                else if(k.find("head_count")!=std::string::npos&&k.find("kv")==std::string::npos)heads=(uint32_t)std::stoul(v);
                else if(k.find("head_count_kv")!=std::string::npos)kv=(uint32_t)std::stoul(v);
                else if(k.find("vocab_size")!=std::string::npos)vocab=(uint32_t)std::stoul(v);
                else if(k.find("context_length")!=std::string::npos)maxseq=(uint32_t)std::stoul(v);
                else if(k.find("expert_count")!=std::string::npos)num_experts=(uint32_t)std::stoul(v);
                else if(k.find("expert_used_count")!=std::string::npos)top_k=(uint32_t)std::stoul(v);
                else if(k.find("expert_feed_forward_length")!=std::string::npos)expert_hidden=(uint32_t)std::stoul(v);
                else if(k.find("key_length")!=std::string::npos)head_dim=(uint32_t)std::stoul(v);
                else if(k.find("conv_kernel")!=std::string::npos)conv_kernel=(uint32_t)std::stoul(v);
            }else if(vt==4){uint32_t vu;ru32(vu);
                if(k.find("expert_feed_forward_length")!=std::string::npos){}
                else if(k.find("feed_forward_length")!=std::string::npos)is_=vu;
                else if(k.find("embedding_length")!=std::string::npos)hs=vu;
                else if(k.find("hidden_size")!=std::string::npos)hs=vu;
                else if(k.find("block_count")!=std::string::npos)layers=vu;
                else if(k.find("head_count")!=std::string::npos&&k.find("kv")==std::string::npos)heads=vu;
                else if(k.find("head_count_kv")!=std::string::npos)kv=vu;
                else if(k.find("vocab_size")!=std::string::npos)vocab=vu;
                else if(k.find("context_length")!=std::string::npos)maxseq=vu;
                else if(k.find("expert_count")!=std::string::npos)num_experts=vu;
                else if(k.find("expert_used_count")!=std::string::npos)top_k=vu;
                else if(k.find("key_length")!=std::string::npos)head_dim=vu;
                else if(k.find("value_length")!=std::string::npos){}
                else if(k.find("dimension_count")!=std::string::npos)rope_dim=(float)vu;
                else if(k.find("conv_kernel")!=std::string::npos)conv_kernel=vu;
            }else if(vt==6){float vf;f.read((char*)&vf,4);
                if(k.find("freq_base")!=std::string::npos)rope_theta=vf;
                else if(k.find("rms_epsilon")!=std::string::npos)rms_eps=vf;
            }else skip(vt);
        }
        cca_heads = heads + kv; // default: total Q+K heads for conv

        uint64_t tens_off = (uint64_t)f.tellg();
        for(uint64_t i=0;i<nt;++i){
            GgufTensor t; rstr(t.name); uint32_t nd; ru32(nd);
            t.shape.resize(nd);
            for(uint32_t d=0;d<nd;++d)ru64(t.shape[d]);
            ru32(t.dtype); ru64(t.offset);
            t.file_offset = tens_off + t.offset;
            tensors[t.name] = t;
        }
        // data_start is implicit — tensor offsets are from after the tensor info
        data_start = (uint64_t)f.tellg();
        return true;
    }

    // Read a tensor's raw data into a float vector (handles FP16 GGUF dtype).
    std::vector<float> read_tensor_f32(const std::string& name) {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            fprintf(stderr, "[WARN] tensor not found: %s\n", name.c_str());
            return {};
        }
        auto& t = it->second;
        size_t n = 1;
        for (auto s : t.shape) n *= (size_t)s;

        std::vector<float> result(n);
        if (t.dtype == 0) { // FP32
            f.seekg((long)(data_start + t.offset));
            f.read((char*)result.data(), n * 4);
        } else if (t.dtype == 1) { // FP16
            f.seekg((long)(data_start + t.offset));
            std::vector<uint16_t> half(n);
            f.read((char*)half.data(), n * 2);
            for (size_t i = 0; i < n; i++) {
                // FP16 → FP32
                uint32_t sign = (half[i] >> 15) & 1;
                uint32_t exp  = (half[i] >> 10) & 0x1F;
                uint32_t mant = half[i] & 0x3FF;
                uint32_t f32;
                if (exp == 0) {
                    f32 = sign << 31 | (mant != 0 ? 0x7B800000 | (mant << 13) : 0);
                } else if (exp == 31) {
                    f32 = sign << 31 | 0x7F800000 | (mant << 13);
                } else {
                    f32 = sign << 31 | ((exp + 112) << 23) | (mant << 13);
                }
                memcpy(&result[i], &f32, 4);
            }
        } else {
            fprintf(stderr, "[WARN] unsupported dtype %d for %s\n", t.dtype, name.c_str());
            return {};
        }
        return result;
    }

    // Read tensor as FP16 (native half)
    std::vector<uint16_t> read_tensor_f16(const std::string& name) {
        auto it = tensors.find(name);
        if (it == tensors.end()) return {};
        auto& t = it->second;
        size_t n = 1;
        for (auto s : t.shape) n *= (size_t)s;

        std::vector<uint16_t> result(n);
        if (t.dtype == 0) { // FP32 → truncate to FP16
            auto f32 = read_tensor_f32(name);
            for (size_t i = 0; i < n; i++) {
                uint32_t u; memcpy(&u, &f32[i], 4);
                uint32_t sign = (u >> 31) & 1;
                int32_t exp  = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
                uint32_t mant = (u >> 13) & 0x3FF;
                if (exp >= 31) { exp = 31; mant = 0; } // Inf
                else if (exp <= 0) { exp = 0; mant = 0; } // Zero
                result[i] = (uint16_t)((sign << 15) | (exp << 10) | mant);
            }
        } else if (t.dtype == 1) { // FP16 native
            f.seekg((long)(data_start + t.offset));
            f.read((char*)result.data(), n * 2);
        }
        return result;
    }
};

// ═══════════════════════════════════════════════════════════════
//  TQ1 packing helper — matches read_ternary_tq1 in h1b_loader.cpp
// ═══════════════════════════════════════════════════════════════

// TQ1: base-3 ternary packing (1.6 bpw, 5 ternaries per byte).
//   d_i  ∈ {-1, 0, +1}  →  t_i ∈ {0, 1, 2}
//   byte = t0 + t1*3 + t2*9 + t3*27 + t4*81
static int8_t encode_ternary(float v) {
    if (v > 0.5f) return 2;  // +1
    if (v < -0.5f) return 0;  // -1
    return 1;                  // 0
}

static void tq1_pack(const float* weights, int rows, int cols,
                     std::vector<uint8_t>& packed, std::vector<float>& scales) {
    // K padded to multiple of 20 (5 ternaries per byte × 4 bytes per u32 row).
    int k_pad = ((cols + 19) / 20) * 20;
    packed.resize((size_t)rows * k_pad / 5);
    scales.resize((size_t)rows);
    memset(packed.data(), 0, packed.size());

    for (int r = 0; r < rows; r++) {
        const float* row = weights + (size_t)r * cols;

        // Compute per-row scale: 1/mean(|w|)
        double sum_abs = 0;
        for (int c = 0; c < cols; c++) sum_abs += fabs(row[c]);
        float scale = (float)(sum_abs / cols);
        if (scale < 1e-10f) scale = 1.0f;
        float inv_scale = 1.0f / scale;
        scales[r] = scale;

        // Quantize to ternary and pack
        for (int c = 0; c < k_pad; c += 5) {
            int byte_idx = (r * k_pad + c) / 5;
            uint8_t byte = 0;
            for (int i = 0; i < 5; i++) {
                float val = (c + i < cols) ? row[c + i] * inv_scale : 0.0f;
                int8_t t = encode_ternary(val);
                byte += (uint8_t)(t * (int)pow(3, i));
            }
            packed[byte_idx] = byte;
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  .h1b writer (v6 — Zaya format)
// ═══════════════════════════════════════════════════════════════

static void write_f32_as_f16(std::ofstream& f, const float* data, int n) {
    for (int i = 0; i < n; i++) {
        float v = data[i];
        uint32_t u; memcpy(&u, &v, 4);
        uint32_t sign = (u >> 31) & 1;
        int32_t exp  = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (u >> 13) & 0x3FF;
        if (exp >= 31) { exp = 31; mant = 0; }
        else if (exp <= 0) { exp = 0; mant = 0; }
        uint16_t h = (uint16_t)((sign << 15) | (exp << 10) | mant);
        f.write((char*)&h, 2);
    }
}

static void write_fp16(std::ofstream& f, const uint16_t* data, int n) {
    f.write((const char*)data, (size_t)n * 2);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <output.h1b>\n", argv[0]);
        return 1;
    }

    // ── 1. Read GGUF ──────────────────────────────────────────
    fprintf(stderr, "[convert] Reading %s ...\n", argv[1]);
    GgufReader gguf;
    if (!gguf.open(argv[1])) {
        fprintf(stderr, "[ERROR] Failed to open GGUF file\n");
        return 1;
    }

    int H = gguf.hs, IS = gguf.is_, L = gguf.layers,
        NH = gguf.heads, NKV = gguf.kv, V = gguf.vocab;
    int NE = gguf.num_experts, TK = gguf.top_k, EH = gguf.expert_hidden;
    int CK = gguf.conv_kernel, CCA = gguf.cca_heads;
    // Zaya stores head_dim in attention.key_length (not derived from H/NH).
    int HD = gguf.head_dim;
    if (HD <= 0) HD = H / NH;  // fallback for non-Zaya models
    fprintf(stderr, "[convert] HD=%d (head_dim)\n", HD);

    fprintf(stderr, "[convert] Zaya: H=%d IS=%d L=%d NH=%d NKV=%d V=%d\n",
            H, IS, L, NH, NKV, V);
    // Zaya's expert_hidden = hidden_size (each expert replaces the full FFN).
    // The GGUF field expert_feed_forward_length is actually the router hidden size (256),
    // not the expert's working dimension. Patch expert_hidden from H when it looks wrong.
    if (EH == 0 || EH < 64) EH = H;

    fprintf(stderr, "[convert] MoE: experts=%d top_k=%d expert_hidden=%d\n",
            NE, TK, EH);
    fprintf(stderr, "[convert] CCA: conv_kernel=%d cca_heads=%d\n", CK, CCA);
    fprintf(stderr, "[convert] Config: rope_theta=%.0f eps=%g\n",
            gguf.rope_theta, gguf.rms_eps);

    if (gguf.arch != "zaya") {
        fprintf(stderr, "[ERROR] Expected arch=zaya, got %s\n", gguf.arch.c_str());
        return 1;
    }

    // ── 2. Open output .h1b ────────────────────────────────────
    std::ofstream out(argv[2], std::ios::binary);
    if (!out) { fprintf(stderr, "[ERROR] Cannot write %s\n", argv[2]); return 1; }

    // ── 3. Write standard .h1b header ─────────────────────────
    int32_t cfg[9] = { H, IS, L, NH, NKV, V, 4096, 6, 0 };  // v6
    out.write((char*)cfg, sizeof(cfg));

    // Zaya extended config (5 int32_t)
    int32_t zcfg[5] = { NE, TK, EH, CK, CCA };
    out.write((char*)zcfg, sizeof(zcfg));

    // Helper: write a layer's norm weights (FP32 → FP16)
    auto write_norm = [&](const std::vector<float>& w) {
        write_f32_as_f16(out, w.data(), (int)w.size());
    };

    // ── 4. Process each layer ──────────────────────────────────
    for (int l = 0; l < L; l++) {
        std::string p = "blk." + std::to_string(l) + ".";
        fprintf(stderr, "[convert] Layer %d/%d\r", l+1, L);

        // ── Norms ────────────────────────────────────────────────
        // Zaya has: attn_norm (pre-attn), attn_norm_2 (2nd attn), ffn_norm (router)
        auto in_norm  = gguf.read_tensor_f32(p + "attn_norm.weight");
        auto pa_norm  = gguf.read_tensor_f32(p + "attn_norm_2.weight"); // 2nd attn norm
        auto fn_norm  = gguf.read_tensor_f32(p + "ffn_norm.weight");    // router norm
        if (in_norm.empty() || pa_norm.empty()) {
            fprintf(stderr, "\n[ERROR] Missing norms for layer %d\n", l);
            return 1;
        }
        // Write input_norm + 6 filler slots + post_attn_norm to match BitNet layout.
        // Zaya only has 2 attn norms; we pad to match the larger expected layout.
        write_norm(in_norm);  // input_norm
        write_norm(pa_norm);  // attn_norm_2 (used as 2nd attention norm)
        for (int i = 0; i < 4; i++) { // 4 more filler slots
            for (int j = 0; j < H; j++) { uint16_t z = 0; out.write((char*)&z, 2); }
        }
        if (!fn_norm.empty()) { // ffn_norm (router norm) — stored in ffn_sub_norm slot
            write_norm(fn_norm);
        } else {
            for (int j = 0; j < H; j++) { uint16_t z = 0; out.write((char*)&z, 2); }
        }

        // ── 7 standard projections (TQ1-packed, with transpose) ─
        // Zaya stores weights as [input_dim, output_dim] instead of [output_dim, input_dim].
        // We transpose during conversion so existing ternary_gemv kernels work.
        auto read_and_transpose = [&](const std::string& name, int rows_out, int cols_in) {
            auto w = gguf.read_tensor_f32(name);
            if (w.empty()) return w;
            // w is currently [cols_in, rows_out] (GGUF layout)
            // We need [rows_out, cols_in] (our layout)
            std::vector<float> t(w.size());
            for (int r = 0; r < rows_out; r++)
                for (int c = 0; c < cols_in; c++)
                    t[(size_t)r * cols_in + c] = w[(size_t)c * rows_out + r];
            return t;
        };

        auto write_tq1 = [&](const std::vector<float>& w, int rows, int cols) {
            std::vector<uint8_t> packed;
            std::vector<float> scales;
            tq1_pack(w.data(), rows, cols, packed, scales);
            out.write((char*)packed.data(), packed.size());
            out.write((char*)scales.data(), (size_t)rows * 4);
        };

        int qd = NH * HD, kd = NKV * HD;

        // Q proj: GGUF has attn_q.weight [H, QD] → transpose to [QD, H]
        { auto w = read_and_transpose(p + "attn_q.weight", qd, H);
          if (w.empty()) return 1; write_tq1(w, qd, H); }
        // K proj: GGUF has attn_k.weight [H, KD] → transpose to [KD, H]
        { auto w = read_and_transpose(p + "attn_k.weight", kd, H);
          if (w.empty()) return 1; write_tq1(w, kd, H); }
        // V proj: Zaya uses val_proj1 + val_proj2 [H, 128] each.
        // Combine by stacking: result = [val_proj1_T, val_proj2_T] = [256, H]
        // But we need [nkv*HD, H] = [256, H]. Concatenate transposed projs.
        {
            auto v1 = gguf.read_tensor_f32(p + "cca_val_proj1.weight"); // [H, 128]
            auto v2 = gguf.read_tensor_f32(p + "cca_val_proj2.weight"); // [H, 128]
            if (v1.empty() || v2.empty()) return 1;
            std::vector<float> v_combined((size_t)kd * H);
            // Transpose each and concatenate
            for (int r = 0; r < kd/2; r++)
                for (int c = 0; c < H; c++) {
                    v_combined[(size_t)r * H + c] = v1[(size_t)c * (kd/2) + r];
                    v_combined[((size_t)r + kd/2) * H + c] = v2[(size_t)c * (kd/2) + r];
                }
            write_tq1(v_combined, kd, H);
        }
        // O proj: attn_output.weight [QD/2, H] — only half-dim for CCA!
        { auto w = gguf.read_tensor_f32(p + "attn_output.weight"); // [QD/2, H]
          if (w.empty()) return 1;
          // Transpose [QD/2, H] → [H, QD/2]… actually no. We need [H, QD/2] for the kernel.
          // The GGUF stores it as [oh, H] where oh = QD/2 = 1024.
          // O proj output dim = H (residual), input dim = QD/2.
          // Our layout: [output_dims=M, K] = [H, QD/2]
          int oh = (int)w.size() / H;
          std::vector<float> w_t((size_t)H * oh);
          for (int r = 0; r < H; r++)
              for (int c = 0; c < oh; c++)
                  w_t[(size_t)r * oh + c] = w[(size_t)c * H + r];
          write_tq1(w_t, H, oh);
        }
        // Standard gate/up/down are not used by Zaya (MoE replaces FFN).
        // Write dummy TQ1-packed tensors to fill the expected slots.
        { std::vector<float> w((size_t)IS * H, 0); write_tq1(w, IS, H); }
        { std::vector<float> w((size_t)IS * H, 0); write_tq1(w, IS, H); }
        { std::vector<float> w((size_t)H * IS, 0); write_tq1(w, H, IS); }

        // ── MoE expert weights ──────────────────────────────────
        // Zaya stores combined gate+up experts: ffn_gate_up_exps.weight [H, 2*EH, NE]
        // We split into separate TQ1-packed gate_experts [NE, EH, H] and up_experts [NE, EH, H].
        auto expert_gu = gguf.read_tensor_f32(p + "ffn_gate_up_exps.weight");
        if (expert_gu.empty()) {
            fprintf(stderr, "\n[ERROR] Missing ffn_gate_up_exps.weight for layer %d\n", l);
            return 1;
        }
        // expert_gu is flat [H * 2*EH * NE]. Reshape: [NE, 2*EH, H]
        // where dim 1 = [0..EH-1] is gate, [EH..2*EH-1] is up
        {
            std::vector<float> gate_experts((size_t)NE * EH * H);
            std::vector<float> up_experts((size_t)NE * EH * H);
            for (int e = 0; e < NE; e++) {
                for (int i = 0; i < EH; i++) {
                    for (int j = 0; j < H; j++) {
                        size_t src_idx = (size_t)e * 2 * EH * H + (size_t)i * H + j;
                        size_t dst_idx = (size_t)e * EH * H + (size_t)i * H + j;
                        gate_experts[dst_idx] = expert_gu[src_idx];
                        up_experts[dst_idx]   = expert_gu[src_idx + (size_t)EH * H];
                    }
                }
            }
            // TQ1-pack gate experts
            { std::vector<uint8_t> p; std::vector<float> s;
              tq1_pack(gate_experts.data(), NE, EH * H, p, s);
              out.write((char*)p.data(), p.size());
              out.write((char*)s.data(), NE * 4); }
            // TQ1-pack up experts
            { std::vector<uint8_t> p; std::vector<float> s;
              tq1_pack(up_experts.data(), NE, EH * H, p, s);
              out.write((char*)p.data(), p.size());
              out.write((char*)s.data(), NE * 4); }
        }

        // ── EDA router (3-layer MLP) ───────────────────────────
        // Zaya's router is: inp[H] → W1[H,256] → ReLU → W2[256,256] → ReLU → W3[256,NE+1]
        // Store all 3 weight matrices + biases as FP16.
        // b/c the decode path composes them as 3 sequential GEMVs.
        auto w1 = gguf.read_tensor_f32(p + "ffn_gate_inp.weight");   // [H, 256]
        auto b1 = gguf.read_tensor_f32(p + "ffn_gate_inp.bias");     // [256]
        auto w2 = gguf.read_tensor_f32(p + "zaya_router_mlp2.weight"); // [256, 256]
        auto b2 = gguf.read_tensor_f32(p + "zaya_router_mlp2.bias");   // [256]
        auto w3 = gguf.read_tensor_f32(p + "zaya_router_mlp4.weight"); // [256, NE+1]
        auto b3 = gguf.read_tensor_f32(p + "zaya_router_biases.weight"); // [NE+1]
        int n_route = NE + 1; // experts + 1 for "no expert" slot

        if (w1.empty() || w3.empty()) {
            fprintf(stderr, "\n[ERROR] Missing router weights for layer %d\n", l);
            return 1;
        }
        // Write W1: [256, H] (transposed from [H, 256])
        {
            std::vector<float> wt((size_t)256 * H);
            for (int r = 0; r < 256; r++)
                for (int c = 0; c < H; c++)
                    wt[(size_t)r * H + c] = w1[(size_t)c * 256 + r];
            write_f32_as_f16(out, wt.data(), 256 * H);
        }
        write_f32_as_f16(out, b1.data(), 256);
        // Write W2: [256, 256] (transposed from [256, 256])
        {
            std::vector<float> wt((size_t)256 * 256);
            for (int r = 0; r < 256; r++)
                for (int c = 0; c < 256; c++)
                    wt[(size_t)r * 256 + c] = w2[(size_t)c * 256 + r];
            write_f32_as_f16(out, wt.data(), 256 * 256);
        }
        write_f32_as_f16(out, b2.data(), 256);
        // Write W3: [n_route, 256] (transposed from [256, n_route])
        {
            std::vector<float> wt((size_t)n_route * 256);
            for (int r = 0; r < n_route; r++)
                for (int c = 0; c < 256; c++)
                    wt[(size_t)r * 256 + c] = w3[(size_t)c * n_route + r];
            write_f32_as_f16(out, wt.data(), n_route * 256);
        }
        if (!b3.empty()) write_f32_as_f16(out, b3.data(), n_route);
        else for (int i = 0; i < n_route; i++) { uint16_t z = 0; out.write((char*)&z, 2); }

        // ── CCA grouped conv weights ────────────────────────────
        // Zaya uses cca_conv_grp.weight [2, 128, 1280] — depthwise grouped conv
        // across 1280 channel groups. Store as-is for the CCA kernel.
        auto cg = gguf.read_tensor_f32(p + "cca_conv_grp.weight");
        if (cg.empty()) {
            fprintf(stderr, "\n[ERROR] Missing cca_conv_grp.weight for layer %d\n", l);
            return 1;
        }
        write_f32_as_f16(out, cg.data(), (int)cg.size());
        // cca_conv_grp bias
        auto cgb = gguf.read_tensor_f32(p + "cca_conv_grp.bias");
        if (!cgb.empty()) write_f32_as_f16(out, cgb.data(), (int)cgb.size());
        else for (int i = 0; i < 1280; i++) { uint16_t z = 0; out.write((char*)&z, 2); }

        // ── SSM conv1d weights ─────────────────────────────────
        auto sc = gguf.read_tensor_f32(p + "ssm_conv1d.weight"); // [2, 1280]
        auto sb = gguf.read_tensor_f32(p + "ssm_conv1d.bias");   // [1280]
        if (sc.empty()) {
            fprintf(stderr, "\n[ERROR] Missing ssm_conv1d.weight for layer %d\n", l);
            return 1;
        }
        write_f32_as_f16(out, sc.data(), 2 * 1280);
        if (!sb.empty()) write_f32_as_f16(out, sb.data(), 1280);
        else for (int i = 0; i < 1280; i++) { uint16_t z = 0; out.write((char*)&z, 2); }

        // ── CCA k_scale ────────────────────────────────────────
        auto ks = gguf.read_tensor_f32(p + "cca_k_scale.weight"); // [2]
        if (!ks.empty()) write_f32_as_f16(out, ks.data(), 2);
        else { uint16_t z[2] = {0,0}; out.write((char*)z, 4); }

        // ── Residual scales (res_scale_hs, res_scale_res) ──────
        auto rsh = gguf.read_tensor_f32(p + "res_scale_hs.weight");
        auto rsr = gguf.read_tensor_f32(p + "res_scale_res.weight");
        if (!rsh.empty()) write_f32_as_f16(out, rsh.data(), H);
        else for (int i = 0; i < H; i++) { uint16_t z = 0x3C00; out.write((char*)&z, 2); } // 1.0
        if (!rsr.empty()) write_f32_as_f16(out, rsr.data(), H);
        else for (int i = 0; i < H; i++) { uint16_t z = 0x3C00; out.write((char*)&z, 2); } // 1.0
    }
    fprintf(stderr, "\n[convert] All %d layers written.\n", L);

    // ── 5. Embedding / LM head ─────────────────────────────────
    // token_embd.weight shape: [2048, 262272] = [H, V] — transposed!
    // We need [V, H].
    fprintf(stderr, "[convert] Writing embeddings...\n");
    auto embed = gguf.read_tensor_f32("token_embd.weight");
    if (embed.empty()) {
        fprintf(stderr, "[ERROR] token_embd.weight not found\n");
        return 1;
    }
    // transpose [H, V] → [V, H]
    {
        std::vector<uint16_t> embed_t((size_t)V * H);
        for (int r = 0; r < V; r++)
            for (int c = 0; c < H; c++) {
                float v = embed[(size_t)c * V + r];
                uint32_t u; memcpy(&u, &v, 4);
                uint32_t sign = (u >> 31) & 1;
                int32_t exp  = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
                uint32_t mant = (u >> 13) & 0x3FF;
                if (exp >= 31) { exp = 31; mant = 0; }
                else if (exp <= 0) { exp = 0; mant = 0; }
                embed_t[(size_t)r * H + c] = (uint16_t)((sign << 15) | (exp << 10) | mant);
            }
        write_fp16(out, embed_t.data(), V * H);
    }

    // ── 6. Final norm ──────────────────────────────────────────
    fprintf(stderr, "[convert] Writing final norm...\n");
    auto final_norm = gguf.read_tensor_f32("output_norm.weight");
    if (final_norm.empty()) final_norm = gguf.read_tensor_f32("final_norm.weight");
    if (final_norm.empty()) {
        fprintf(stderr, "[WARN] final_norm not found, writing identity\n");
        for (int i = 0; i < H; i++) { float one = 1.0f; write_f32_as_f16(out, &one, 1); }
    } else {
        write_f32_as_f16(out, final_norm.data(), H);
    }

    // ── 7. LM head (output weight) ────────────────────────────
    fprintf(stderr, "[convert] Writing output weight...\n");
    // The LM head is tied to embedding (tie_word_embeddings=true).
    // We already wrote the transposed embedding above.
    // For the LM head, the decode path uses the same embedding weight
    // via rcpp_embedding_lookup_fp16/rcpp_fp16_gemv.
    // Write it again as output weight (same data, already transposed).
    // Actually — skip this since tied embeddings mean LM head = embedding.
    // The .h1b format expects output weight separately though.
    // Write a zero-length sentinel (0 elements) to indicate tied.
    {
        int zero = 0;
        out.write((char*)&zero, 4);
    }

    out.close();
    fprintf(stderr, "[convert] ✅ Written %s\n", argv[2]);
    return 0;
}
