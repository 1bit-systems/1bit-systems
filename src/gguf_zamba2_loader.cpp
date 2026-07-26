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
// ── Proper IEEE 754 half-precision (__half) → float32 conversion ──
// NOT simply shifting to upper 16 bits (which only works for zero and denormals).
// The half exponent (5-bit, bias 15) must be rebias'd to float exponent (8-bit, bias 127).
static inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t mant = (uint32_t)(h & 0x03FF) << 13;
    uint32_t exp  = (h >> 10) & 0x1F;
    if (exp == 0) {
        if (mant == 0) return 0.0f;     // zero
        // Denormal: fits in float32 subnormal range (shift mant to position)
        uint32_t bits = sign | mant;
        float f; memcpy(&f, &bits, 4);
        return f;
    }
    if (exp == 31) {  // NaN or Inf
        uint32_t bits = sign | 0x7F800000 | mant;
        float f; memcpy(&f, &bits, 4);
        return f;
    }
    // Normal: rebias exponent from half's 15 to float's 127
    uint32_t rebias = (uint32_t)(exp - 15 + 127) << 23;
    uint32_t bits = sign | rebias | mant;
    float f; memcpy(&f, &bits, 4);
    return f;
}

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

        // A crafted/corrupt GGUF can carry a 64-bit length of ~2^62; allocating
        // std::string(len) then blows up the heap (DoS). Mirror gguf_reader.cpp's
        // caps: 1 MiB per string, 1M elements per array. Oversized fields are
        // skipped (stream kept in sync) rather than trusted (AUDIT_ISSUES.md #4).
        static constexpr uint64_t MAX_STRING_LEN  = 1ULL * 1024 * 1024;
        static constexpr uint64_t MAX_ARRAY_COUNT = 1000000ULL;

        for (uint64_t i = 0; i < n_kv; ++i) {
            uint64_t key_len; f.read(reinterpret_cast<char*>(&key_len), 8);
            std::string key;
            if (key_len > MAX_STRING_LEN) {
                f.seekg((std::streamoff)key_len, std::ios::cur);  // skip, keep in sync
            } else {
                key.resize((size_t)key_len);
                if (key_len > 0) f.read(&key[0], key_len);
            }
            uint32_t vt; f.read(reinterpret_cast<char*>(&vt), 4);

            auto read_string = [&]() -> std::string {
                uint64_t sl; f.read(reinterpret_cast<char*>(&sl), 8);
                if (sl > MAX_STRING_LEN) { f.seekg((std::streamoff)sl, std::ios::cur); return {}; }
                std::string sv((size_t)sl, '\0');
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
                    if (an > MAX_ARRAY_COUNT) return false;  // malformed — refuse
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
            if (name_len > MAX_STRING_LEN) { f.seekg((std::streamoff)name_len, std::ios::cur); return false; }
            std::string name((size_t)name_len, '\0');
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
                out[i] = half_to_float(f16[i]);
            }
            return true;
        }
        // Q4_0: 32 elements/block, 2 bytes header + 16 bytes quads
        // Layout: qs[0..15] stores 32 4-bit values. Lower nibbles of qs[0..15]
        // are elements 0..15, upper nibbles of qs[0..15] are elements 16..31.
        if (ti.dtype == 2) {
            const int bs = 32, bb = 18;
            uint64_t nb = (numel + bs - 1) / bs;
            std::vector<uint8_t> blk(bb);
            for (uint64_t b = 0; b < nb; ++b) {
                uint64_t start = b * bs, end = std::min(start + bs, numel), cnt = end - start;
                f.read(reinterpret_cast<char*>(blk.data()), bb);
                uint16_t sh_bits; memcpy(&sh_bits, blk.data(), 2);
                float s = half_to_float(sh_bits);
                uint8_t* q = blk.data() + 2;
                // First 16 elements: lower nibble of q[0..15]
                for (uint64_t i = 0; i < cnt && i < 16; ++i) {
                    int8_t nib = (int8_t)(q[i] & 0x0F);
                    out[start + i] = (float)(nib - 8) * s;
                }
                // Next 16 elements: upper nibble of q[0..15]
                for (uint64_t i = 16; i < cnt && i < 32; ++i) {
                    int8_t nib = (int8_t)(q[i - 16] >> 4);
                    out[start + i] = (float)(nib - 8) * s;
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
                float s = half_to_float(sh_bits);
                int8_t* q = (int8_t*)(blk.data() + 2);
                for (uint64_t i = 0; i < cnt; ++i) out[start + i] = q[i] * s;
            }
            return true;
        }
        // Q4_K: 256 elements/block, 144 bytes (2+2+12+128), K-quant 4-bit
        // bit layout per get_scale_min_k4 in ggml-quants.c
        if (ti.dtype == 12) {
            const int bs = 256, bb = 144;
            uint64_t nb = (numel + bs - 1) / bs;
            std::vector<uint8_t> blk(bb);
            for (uint64_t b = 0; b < nb; ++b) {
                uint64_t start = b * bs, end = std::min(start + bs, numel), cnt = end - start;
                f.read(reinterpret_cast<char*>(blk.data()), bb);
                // Header: d (half, 2) + dmin (half, 2) + scales (12) + qs (128)
                uint16_t dh, dmh;
                memcpy(&dh, blk.data(), 2);
                memcpy(&dmh, blk.data() + 2, 2);
                float d_block = half_to_float(dh);
                float min_block = half_to_float(dmh);
                uint8_t* sc = blk.data() + 4;
                uint8_t* qs = blk.data() + 16;
                for (int sub = 0; sub < 8 && sub * 32 < (int)cnt; ++sub) {
                    uint8_t sv, mv;
                    if (sub < 4) {
                        sv = sc[sub] & 63; mv = sc[sub + 4] & 63;
                    } else {
                        int j = sub;
                        sv = (sc[j+4] & 0xF) | ((sc[j-4] >> 6) << 4);
                        mv = (sc[j+4] >> 4) | ((sc[j]   >> 6) << 4);
                    }
                    float d = d_block * (float)sv;
                    float m = min_block * (float)mv;
                    int qb = (sub / 2) * 32;
                    bool up = sub & 1;
                    for (int l = 0; l < 32; ++l) {
                        int idx = sub * 32 + l;
                        if (idx >= (int)cnt) break;
                        int nib = up ? (qs[qb + l] >> 4) : (qs[qb + l] & 0xF);
                        out[start + idx] = d * (float)nib - m;
                    }
                }
            }
            return true;
        }
        // Q6_K: 256 elements/block, 210 bytes (block_q6_K from ggml-quants)
        // Layout: d(__half,2) + ql[128] + qh[64] + sc[16]
        // Each element: low 4 bits from ql + high 2 bits from qh = 6-bit signed
        // scaled by d * sc[sub_block]
        if (ti.dtype == 13 || ti.dtype == 14) {
            const int bs = 256, bb = 210;
            uint64_t nb = (numel + bs - 1) / bs;
            std::vector<uint8_t> blk(bb);
            for (uint64_t b = 0; b < nb; ++b) {
                uint64_t start = b * bs, end = std::min(start + bs, numel), cnt = end - start;
                f.read(reinterpret_cast<char*>(blk.data()), bb);
                uint16_t dh_bits; memcpy(&dh_bits, blk.data(), 2);
                float d = half_to_float(dh_bits);
                uint8_t* ql = blk.data() + 2;   // ql[128], 4-bit low quants
                uint8_t* qh = blk.data() + 130;  // qh[64], 2-bit high quants
                int8_t*  sc = (int8_t*)(blk.data() + 194); // sc[16], int8 scales
                // Process 128 elements at a time (half super-block), matching
                // dequantize_row_q6_K in ggml-quants.c exactly.
                for (int half = 0; half < 2 && half * 128 < (int)cnt; ++half) {
                    int off = half * 128;
                    if (off >= (int)cnt) break;
                    for (int l = 0; l < 32; ++l) {
                        // Map l to 4 elements per byte iteration
                        // q1: ql[l] lower nibble + qh[l] bits 0-1
                        int8_t q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        // q2: ql[l+32] lower nibble + qh[l] bits 2-3
                        int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        // q3: ql[l] upper nibble + qh[l] bits 4-5
                        int8_t q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        // q4: ql[l+32] upper nibble + qh[l] bits 6-7
                        int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        int is = l / 16;
                        // Map output indices, bounded by cnt
                        int idx0 = off + l;
                        int idx1 = off + l + 32;
                        int idx2 = off + l + 64;
                        int idx3 = off + l + 96;
                        if (idx0 < (int)cnt) out[start + idx0] = d * (float)sc[is + 0] * (float)q1;
                        if (idx1 < (int)cnt) out[start + idx1] = d * (float)sc[is + 2] * (float)q2;
                        if (idx2 < (int)cnt) out[start + idx2] = d * (float)sc[is + 4] * (float)q3;
                        if (idx3 < (int)cnt) out[start + idx3] = d * (float)sc[is + 6] * (float)q4;
                    }
                    ql += 64;
                    qh += 32;
                    sc += 8;
                }
            }
            return true;
        }
        fprintf(stderr, "[gguf] tensor %s: dtype %u not supported, zero-filling\n", name.c_str(), ti.dtype);
        std::fill(out.begin(), out.end(), 0.0f);
        return false;
    }

    // Like read_tensor but transposes the matrix from GGUF [input, output] layout
    // to the engine's expected [output, input] layout. Uses the tensor's own shape
    // to determine dimensions. Works for any 2D weight matrix.
    bool read_tensor_transposed(const std::string& name, std::vector<float>& out) {
        if (!read_tensor(name, out)) return false;
        auto it = tensors.find(name);
        if (it == tensors.end() || it->second.shape.size() != 2) return true; // not 2D, skip
        int input_dim  = (int)it->second.shape[0];
        int output_dim = (int)it->second.shape[1];
        std::vector<float> orig = out;
        for (int o = 0; o < output_dim; ++o)
            for (int i = 0; i < input_dim; ++i)
                out[o * input_dim + i] = orig[i * output_dim + o];
        return true;
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

    // ── Architecture guard: reject non-SSM models ──
    // The Zamba2 GGUF reader expects Mamba2/SSM tensor names (ssm_in, ssm_conv1d, etc.).
    // Transformer models (llama, qwen2, qwen3, zaya, etc.) silently load with zero/garbage
    // weights because read_tensor() returns false for missing keys — producing garbage
    // output rather than a clear error. Reject them early.
    static const std::vector<std::string> supported_archs = {"zamba2", "zamba", "mamba"};
    bool arch_ok = false;
    for (auto& a : supported_archs) {
        if (reader.arch == a) { arch_ok = true; break; }
    }
    if (!arch_ok) {
        fprintf(stderr, "[zamba2] ERROR: architecture '%s' is not supported by this backend.\n"
                        "        Supported: zamba2, zamba, mamba.\n"
                        "        Refusing to load — model would produce garbage.\n",
                reader.arch.c_str());
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

    // Pre-compute Mamba2 projection dimensions for weight transpose
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

            // Mamba2 weights — transpose from GGUF [input, output] to engine [output, input]
            reader.read_tensor_transposed(p("ssm_in.weight"), hl.mamba.in_proj_w);
            reader.read_tensor(p("ssm_conv1d.weight"), hl.mamba.conv1d_w);  // [d_conv, conv_dim] — access matches
            reader.read_tensor(p("ssm_conv1d.bias"), hl.mamba.conv1d_b);
            reader.read_tensor(p("ssm_dt.bias"), hl.mamba.dt_bias);
            reader.read_tensor(p("ssm_a"), hl.mamba.A_log);
            reader.read_tensor(p("ssm_d"), hl.mamba.D);
            reader.read_tensor(p("ssm_norm.weight"), hl.mamba.norm_w);
            reader.read_tensor_transposed(p("ssm_out.weight"), hl.mamba.out_proj_w);
            hl.mamba.input_norm_w = hl.input_norm_w;  // share with hybrid input norm
            hl.mamba.loaded = true;

            // Linear projection (mixing) — transpose from [d_model, d_model] to [d_model, d_model]
            reader.read_tensor_transposed(p("ssm_mix.weight"), hl.linear_w);

            // Shared block idx (ABAB pattern)
            hl.shared_block_idx = n_hybrid % 2;

            // Self-attention — all weight matrices need transpose from [input, output]
            SharedBlockWeights sb;
            std::vector<float> up_w;
            reader.read_tensor_transposed(p("attn_q.weight"), sb.q_proj_w);
            reader.read_tensor_transposed(p("attn_k.weight"), sb.k_proj_w);
            reader.read_tensor_transposed(p("attn_v.weight"), sb.v_proj_w);
            reader.read_tensor_transposed(p("attn_output.weight"), sb.o_proj_w);
            reader.read_tensor(p("post_attention_norm.weight"), sb.pre_ff_norm_w);
            reader.read_tensor_transposed(p("ffn_gate.weight"), sb.gate_up_proj_w);
            // Separate gate/up in GGUF; up stored separately
            reader.read_tensor_transposed(p("ffn_up.weight"), up_w);
            reader.read_tensor_transposed(p("ffn_down.weight"), sb.down_proj_w);
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
            reader.read_tensor_transposed(p("ssm_in.weight"), ml.in_proj_w);
            reader.read_tensor(p("ssm_conv1d.weight"), ml.conv1d_w);  // [d_conv, conv_dim] — already correct
            reader.read_tensor(p("ssm_conv1d.bias"), ml.conv1d_b);
            reader.read_tensor(p("ssm_dt.bias"), ml.dt_bias);
            reader.read_tensor(p("ssm_a"), ml.A_log);
            reader.read_tensor(p("ssm_d"), ml.D);
            reader.read_tensor(p("ssm_norm.weight"), ml.norm_w);
            reader.read_tensor_transposed(p("ssm_out.weight"), ml.out_proj_w);
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
