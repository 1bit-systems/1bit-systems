// gguf_zamba2_loader.cpp — Load Zamba2/Mamba2/Mamba model weights from GGUF
//
// Supports the GGUF tensor naming convention used by llama.cpp's GGUF converter:
//   - blk.N.attn_norm.weight     — input RMS norm (all layers)
//   - blk.N.ssm_in.weight         — Mamba2 in_proj
//   - blk.N.ssm_conv1d.weight/bias — Mamba2 conv1d
//   - blk.N.ssm_dt.bias           — Mamba2 dt bias
//   - blk.N.ssm_a                 — Mamba2 A_log
//   - blk.N.ssm_d                 — Mamba2 D
//   - blk.N.ssm_norm.weight       — Mamba2 norm
//   - blk.N.ssm_out.weight        — Mamba2 out_proj
//   - blk.N.attn_q/k/v.weight     — Attention QKV (hybrid layers only)
//   - blk.N.attn_output.weight    — Attention output (hybrid layers only)
//   - blk.N.post_attention_norm.weight  — Post-attn norm (hybrid layers only)
//   - blk.N.ffn_gate/up/down.weight     — FFN (hybrid layers only)
//   - blk.N.ffn_norm.weight       — FFN norm (hybrid layers only)
//   - blk.N.ssm_mix.weight        — Mixing projection (hybrid layers only)
//   - token_embd.weight           — Embedding
//   - output_norm.weight          — Final RMS norm
//   - output.weight               — LM head (may be tied)

#include "zamba2_engine.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <cmath>

// ── Minimal GGUF reader ──
struct Zamba2GgufReader {
    std::ifstream f;
    uint32_t version = 0;
    uint64_t alignment = 32;
    uint64_t tensor_data_start = 0;
    std::string arch;

    struct TensorInfo {
        std::vector<uint64_t> shape;
        uint32_t dtype;
        uint64_t offset;
        uint64_t file_offset;
    };
    std::unordered_map<std::string, TensorInfo> tensors;
    std::unordered_map<std::string, uint64_t> kv_uint64;
    std::unordered_map<std::string, uint32_t> kv_uint32;
    std::unordered_map<std::string, float> kv_float;
    std::unordered_map<std::string, std::string> kv_string;

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

        for (uint64_t i = 0; i < n_kv; ++i) {
            uint64_t key_len; f.read(reinterpret_cast<char*>(&key_len), 8);
            std::string key(key_len, '\0');
            if (key_len > 0) f.read(&key[0], key_len);
            uint32_t vt; f.read(reinterpret_cast<char*>(&vt), 4);

            auto read_string = [&]() {
                uint64_t sl; f.read(reinterpret_cast<char*>(&sl), 8);
                std::string sv(sl, '\0');
                if (sl > 0) f.read(&sv[0], sl);
                return sv;
            };

            switch (vt) {
                case 0: case 4: { uint32_t v; f.read(reinterpret_cast<char*>(&v), 4); kv_uint32[key] = v; break; }
                case 2: { int64_t v; f.read(reinterpret_cast<char*>(&v), 8); kv_uint64[key] = (uint64_t)v; break; }
                case 3: { double v; f.read(reinterpret_cast<char*>(&v), 8); kv_float[key] = (float)v; break; }
                case 5: { int32_t v; f.read(reinterpret_cast<char*>(&v), 4); kv_uint32[key] = (uint32_t)v; break; }
                case 6: { float v; f.read(reinterpret_cast<char*>(&v), 4); kv_float[key] = v; break; }
                case 7: { uint8_t v; f.read(reinterpret_cast<char*>(&v), 1); kv_uint32[key] = v; break; }
                case 8: kv_string[key] = read_string(); break;
                case 9: {
                    uint32_t at; f.read(reinterpret_cast<char*>(&at), 4);
                    uint64_t an; f.read(reinterpret_cast<char*>(&an), 8);
                    if (at == 8) { for (uint64_t j = 0; j < an; ++j) read_string(); }
                    else { f.seekg(an * 4, std::ios::cur); }
                    break;
                }
                default: f.seekg(8, std::ios::cur); break;
            }
        }

        if (kv_string.count("general.architecture")) arch = kv_string["general.architecture"];
        if (kv_uint32.count("general.alignment")) alignment = kv_uint32["general.alignment"];
        if (alignment < 32) alignment = 32;

        for (uint64_t i = 0; i < n_tensors; ++i) {
            uint64_t name_len; f.read(reinterpret_cast<char*>(&name_len), 8);
            std::string name(name_len, '\0');
            if (name_len > 0) f.read(&name[0], name_len);
            uint32_t ndim; f.read(reinterpret_cast<char*>(&ndim), 4);
            TensorInfo ti; ti.shape.resize(ndim);
            for (uint32_t d = 0; d < ndim; ++d) f.read(reinterpret_cast<char*>(&ti.shape[d]), 8);
            f.read(reinterpret_cast<char*>(&ti.dtype), 4);
            f.read(reinterpret_cast<char*>(&ti.offset), 8);
            tensors[name] = std::move(ti);
        }

        tensor_data_start = (uint64_t)f.tellg();
        uint64_t rem = tensor_data_start % alignment;
        if (rem) tensor_data_start += alignment - rem;

        for (auto& [name, ti] : tensors) ti.file_offset = tensor_data_start + ti.offset;
        return true;
    }

    // ── Read and dequantize a tensor ──
    bool read_tensor(const std::string& name, std::vector<float>& out) {
        auto it = tensors.find(name);
        if (it == tensors.end()) return false;
        auto& ti = it->second;

        uint64_t numel = 1;
        for (auto d : ti.shape) numel *= d;
        out.resize(numel);
        f.seekg(ti.file_offset);

        if (ti.dtype == 0) { // F32
            f.read(reinterpret_cast<char*>(out.data()), numel * 4);
            return true;
        }
        if (ti.dtype == 1) { // F16
            std::vector<uint16_t> f16(numel);
            f.read(reinterpret_cast<char*>(f16.data()), numel * 2);
            for (uint64_t i = 0; i < numel; ++i) {
                uint32_t bits = (uint32_t)f16[i] << 16;
                float v; memcpy(&v, &bits, 4); out[i] = v;
            }
            return true;
        }
        // Q4_0: 32 elements/block, 2 bytes header + 16 bytes quads
        if (ti.dtype == 2) {
            const int bs = 32, bb = 18;
            uint64_t nb = (numel + bs - 1) / bs;
            std::vector<uint8_t> blk(bb);
            for (uint64_t b = 0; b < nb; ++b) {
                uint64_t start = b * bs, end = std::min(start + bs, numel), cnt = end - start;
                f.read(reinterpret_cast<char*>(blk.data()), bb);
                // FP16 → FP32 conversion
                uint16_t sh_bits; memcpy(&sh_bits, blk.data(), 2);
                uint32_t f32_bits = (uint32_t)sh_bits << 16;
                float s; memcpy(&s, &f32_bits, 4);
                uint8_t* q = blk.data() + 2;
                for (uint64_t i = 0; i < cnt; ++i) {
                    int8_t nib = (i & 1) ? (q[i >> 1] & 0x0F) : (q[i >> 1] >> 4);
                    out[start + i] = (nib - 8) * s;
                }
            }
            return true;
        }
        // Q8_0: 32 elements/block, 2 bytes header + 32 bytes quads  
        if (ti.dtype == 6 || ti.dtype == 7) {
            const int bs = 32, bb = 34;
            uint64_t nb = (numel + bs - 1) / bs;
            std::vector<uint8_t> blk(bb);
            for (uint64_t b = 0; b < nb; ++b) {
                uint64_t start = b * bs, end = std::min(start + bs, numel), cnt = end - start;
                f.read(reinterpret_cast<char*>(blk.data()), bb);
                uint16_t sh_bits; memcpy(&sh_bits, blk.data(), 2);
                uint32_t f32_bits = (uint32_t)sh_bits << 16;
                float s; memcpy(&s, &f32_bits, 4);
                int8_t* q = (int8_t*)(blk.data() + 2);
                for (uint64_t i = 0; i < cnt; ++i) out[start + i] = q[i] * s;
            }
            return true;
        }
        // Q6_K: 256 elements/block, 14 bytes header + 192 bytes quads + super-scales
        if (ti.dtype == 13 || ti.dtype == 14) {
            const int bs = 256, bb = 210;
            uint64_t nb = (numel + bs - 1) / bs;
            std::vector<uint8_t> blk(bb);
            for (uint64_t b = 0; b < nb; ++b) {
                uint64_t start = b * bs, end = std::min(start + bs, numel), cnt = end - start;
                f.read(reinterpret_cast<char*>(blk.data()), bb);
                // Q6_K: 4 super-blocks of 64, each with 6-bit values
                // Header: d (__half, 2 bytes) + 4 dms (uint8, 1 each) + 4 scales (uint8, 1 each)
                // = 2 + 4 + 4 = 10 bytes header
                // Then 256 * 6 bits = 192 bytes of packed quants
                uint16_t dh_bits; memcpy(&dh_bits, blk.data(), 2);
                uint32_t df32_bits = (uint32_t)dh_bits << 16;
                float d; memcpy(&d, &df32_bits, 4);
                // Simplified dequant: extract 6-bit values
                uint8_t* q6 = blk.data() + 10; // skip header (d + 4 dms + 4 scales = 10)
                for (uint64_t i = 0; i < cnt && i < 256; ++i) {
                    // 6 bits packed across bytes: every 4 values use 3 bytes
                    int byte_idx = (i * 6) / 8;
                    int bit_off = (i * 6) % 8;
                    int val;
                    if (bit_off <= 2) {
                        val = (q6[byte_idx] >> bit_off) & 0x3F;
                    } else {
                        val = ((q6[byte_idx] >> bit_off) | (q6[byte_idx + 1] << (8 - bit_off))) & 0x3F;
                    }
                    // Sign extend 6-bit to 8-bit
                    if (val & 0x20) val |= ~0x3F;
                    out[start + i] = (float)val * d;
                }
            }
            return true;
        }
        fprintf(stderr, "[gguf] tensor %s: dtype %u not supported, zero-filling\n", name.c_str(), ti.dtype);
        std::fill(out.begin(), out.end(), 0.0f);
        return false;
    }

    bool has_tensor(const std::string& name) const {
        return tensors.count(name) > 0;
    }
};

// ── Load Zamba2 model from GGUF ──
bool load_zamba2_from_gguf(const std::string& path, Zamba2Model& model) {
    Zamba2GgufReader reader;
    if (!reader.open(path)) {
        fprintf(stderr, "[zamba2] Failed to open GGUF: %s\n", path.c_str());
        return false;
    }

    fprintf(stderr, "[zamba2] Loading Zamba2 from GGUF (arch=%s)\n", reader.arch.c_str());

    auto& cfg = model.cfg;

    // ── Read hyperparams from GGUF KV ──
    auto gu32 = [&](const std::string& k, int def) -> int {
        if (reader.kv_uint32.count(k)) return (int)reader.kv_uint32[k];
        std::string ak = reader.arch + "." + k;
        if (reader.kv_uint32.count(ak)) return (int)reader.kv_uint32[ak];
        return def;
    };
    auto gf32 = [&](const std::string& k, float def) -> float {
        if (reader.kv_float.count(k)) return reader.kv_float[k];
        std::string ak = reader.arch + "." + k;
        if (reader.kv_float.count(ak)) return reader.kv_float[ak];
        return def;
    };

    cfg.d_model       = gu32("embedding_length", 2560);
    cfg.d_state       = gu32("ssm.state_size", 64);
    cfg.d_conv        = gu32("ssm.conv_kernel", 4);
    cfg.d_inner       = gu32("ssm.inner_size", 5120);
    cfg.n_head        = cfg.d_inner / 64;  // head_dim=64
    cfg.n_group       = gu32("ssm.group_count", 1);
    cfg.head_dim      = cfg.d_inner / cfg.n_head;
    cfg.n_layers      = gu32("block_count", 54);
    cfg.n_attn_heads  = gu32("attention.head_count", 32);
    cfg.n_kv_heads    = gu32("attention.head_count_kv", 32);
    cfg.attn_head_dim = gu32("attention.head_dim", 80);
    cfg.vocab_size    = gu32("vocab_size", gu32("llm.vocab_size", 32000));
    cfg.max_seq_len   = gu32("context_length", 4096);
    cfg.rope_theta    = gf32("rope.freq_base", 10000.0f);
    cfg.rms_norm_eps  = gf32("attention.layer_norm_rms_epsilon", 1e-5f);

    // Fix vocabulary size from embedding if possible
    int actual_vocab = 0;
    if (reader.has_tensor("token_embd.weight")) {
        auto& ti = reader.tensors.at("token_embd.weight");
        if (ti.shape.size() >= 2) actual_vocab = (int)ti.shape[1];
    }
    if (actual_vocab > 0) cfg.vocab_size = actual_vocab;

    fprintf(stderr, "[zamba2] Config: H=%d L=%d d_state=%d d_conv=%d d_inner=%d "
                    "n_head=%d n_group=%d head_dim=%d V=%d\n",
            cfg.d_model, cfg.n_layers, cfg.d_state, cfg.d_conv, cfg.d_inner,
            cfg.n_head, cfg.n_group, cfg.head_dim, cfg.vocab_size);
    fprintf(stderr, "[zamba2] Attn: NH=%d NKV=%d HD=%d rope_theta=%.0f\n",
            cfg.n_attn_heads, cfg.n_kv_heads, cfg.attn_head_dim, cfg.rope_theta);

    // ── Detect which layers are hybrid (have attention/FFN weights) ──
    auto is_hybrid = [&](int layer) -> bool {
        std::string q_name = "blk." + std::to_string(layer) + ".attn_q.weight";
        return reader.has_tensor(q_name);
    };

    // ── Load embedding ──
    if (!reader.read_tensor("token_embd.weight", model.embed_w)) {
        fprintf(stderr, "[zamba2] Missing embedding tensor\n");
        return false;
    }
    // Transpose if needed: GGUF stores [d_model, vocab], we want [vocab, d_model]
    if (reader.tensors.at("token_embd.weight").shape[0] == (uint64_t)cfg.d_model) {
        // Stored as [d_model, vocab] — transpose to [vocab, d_model]
        std::vector<float> orig = model.embed_w;
        for (int i = 0; i < cfg.vocab_size; ++i)
            for (int j = 0; j < cfg.d_model; ++j)
                model.embed_w[i * cfg.d_model + j] = orig[j * cfg.vocab_size + i];
    }

    // ── Load final norm ──
    reader.read_tensor("output_norm.weight", model.final_norm_w);

    // ── Load per-layer weights ──
    model.mamba_layers.clear();
    model.hybrid_layers.clear();

    int n_hybrid = 0, n_mamba = 0;

    for (int l = 0; l < cfg.n_layers; ++l) {
        auto p = [&](const std::string& name) -> std::string {
            return "blk." + std::to_string(l) + "." + name;
        };

        if (is_hybrid(l)) {
            // ── Hybrid layer: Mamba2 + attention + FFN ──
            HybridLayerWeights hl;

            // Input norm + mamba decoder input norm (same in GGUF Zamba2 format)
            reader.read_tensor(p("attn_norm.weight"), hl.input_norm_w);
            hl.mamba_input_norm_w = hl.input_norm_w;  // shared norm

            // Mamba2 weights
            reader.read_tensor(p("ssm_in.weight"), hl.mamba.in_proj_w);
            reader.read_tensor(p("ssm_conv1d.weight"), hl.mamba.conv1d_w);
            reader.read_tensor(p("ssm_conv1d.bias"), hl.mamba.conv1d_b);
            reader.read_tensor(p("ssm_dt.bias"), hl.mamba.dt_bias);
            reader.read_tensor(p("ssm_a"), hl.mamba.A_log);
            reader.read_tensor(p("ssm_d"), hl.mamba.D);
            reader.read_tensor(p("ssm_norm.weight"), hl.mamba.norm_w);
            reader.read_tensor(p("ssm_out.weight"), hl.mamba.out_proj_w);
            hl.mamba.input_norm_w = hl.input_norm_w;  // share with hybrid input norm
            hl.mamba.loaded = true;

            // Linear projection (mixing)  
            reader.read_tensor(p("ssm_mix.weight"), hl.linear_w);

            // Shared block idx (ABAB pattern)
            hl.shared_block_idx = n_hybrid % 2;

            // Self-attention
            // Use a temporary SharedBlockWeights for loading attn/ffn
            SharedBlockWeights sb;
            reader.read_tensor(p("attn_q.weight"), sb.q_proj_w);
            reader.read_tensor(p("attn_k.weight"), sb.k_proj_w);
            reader.read_tensor(p("attn_v.weight"), sb.v_proj_w);
            reader.read_tensor(p("attn_output.weight"), sb.o_proj_w);
            reader.read_tensor(p("post_attention_norm.weight"), sb.pre_ff_norm_w);
            reader.read_tensor(p("ffn_gate.weight"), sb.gate_up_proj_w);
            // The GGUF has separate gate/up; we need them fused (or separate)
            // For now, store gate in gate_up_proj_w and up separately
            std::vector<float> up_w;
            reader.read_tensor(p("ffn_up.weight"), up_w);
            reader.read_tensor(p("ffn_down.weight"), sb.down_proj_w);
            reader.read_tensor(p("ffn_norm.weight"), sb.input_norm_w);

            // Store as per-hybrid-layer weights (the converter duplicated shared blocks)
            hl.shared_transformer_q = sb.q_proj_w;
            hl.shared_transformer_k = sb.k_proj_w;
            hl.shared_transformer_v = sb.v_proj_w;
            hl.shared_transformer_o = sb.o_proj_w;
            hl.shared_transformer_pre_ff_norm = sb.pre_ff_norm_w;
            hl.shared_transformer_ffn_norm = sb.input_norm_w;
            hl.shared_transformer_gate = sb.gate_up_proj_w;
            hl.shared_transformer_up = up_w;
            hl.shared_transformer_down = sb.down_proj_w;

            hl.loaded = true;
            model.hybrid_layers[l] = std::move(hl);
            n_hybrid++;
        } else {
            // ── Pure Mamba2 layer ──
            Mamba2LayerWeights ml;
            reader.read_tensor(p("attn_norm.weight"), ml.input_norm_w);
            reader.read_tensor(p("ssm_in.weight"), ml.in_proj_w);
            reader.read_tensor(p("ssm_conv1d.weight"), ml.conv1d_w);
            reader.read_tensor(p("ssm_conv1d.bias"), ml.conv1d_b);
            reader.read_tensor(p("ssm_dt.bias"), ml.dt_bias);
            reader.read_tensor(p("ssm_a"), ml.A_log);
            reader.read_tensor(p("ssm_d"), ml.D);
            reader.read_tensor(p("ssm_norm.weight"), ml.norm_w);
            reader.read_tensor(p("ssm_out.weight"), ml.out_proj_w);
            ml.loaded = !ml.in_proj_w.empty();
            model.mamba_layers[l] = std::move(ml);
            n_mamba++;
        }
    }

    // ── Verify ──
    bool ok = !model.embed_w.empty() && !model.final_norm_w.empty()
           && (n_mamba + n_hybrid) == cfg.n_layers;

    if (ok) {
        model.loaded = true;
        model.init_state();
        fprintf(stderr, "[zamba2] Model loaded: %d mamba + %d hybrid layers\n", n_mamba, n_hybrid);
    } else {
        fprintf(stderr, "[zamba2] Load incomplete: %d mamba + %d hybrid of %d layers\n",
                n_mamba, n_hybrid, cfg.n_layers);
    }

    return ok;
}
