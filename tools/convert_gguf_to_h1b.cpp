// convert_gguf_to_h1b — Convert a Bonsai F16 GGUF to .h1b + sidecar GGUF.
//
// Reads a Bonsai-format F16 GGUF, ternarizes the weights to TQ2_0_g128 format,
// and writes:
//   output.h1b         — BONSAI_TQ2 weight format, zeroed norms/embedding
//   output.gguf        — sidecar GGUF with norms + embedding (FP32)
//
// The output.h1b + output.gguf pair is loadable by rcpp_bitnet_load_h1b().
//
// Usage:
//   convert_gguf_to_h1b --input model-f16.gguf --output model [--config model-tq2.gguf]
//
// --config is optional: if the F16 GGUF doesn't have config, read from
// a separate TQ2 GGUF config file.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

// ========================================================================
// Minimal GGUF v3 reader (same structure as h1b_loader.cpp)
// ========================================================================
struct GgufTensorInfo {
    std::vector<uint64_t> shape;
    uint32_t dtype;
    uint64_t offset;
};

class GgufReader {
public:
    bool open(const std::string& path) {
        f_.open(path, std::ios::binary);
        if (!f_) return false;
        char magic[4];
        f_.read(magic, 4);
        if (std::strncmp(magic, "GGUF", 4) != 0) return false;
        if (!read_u32(version_)) return false;
        if (version_ != 2 && version_ != 3) {
            fprintf(stderr, "[gguf] unsupported version %u\n", version_);
            return false;
        }
        uint64_t n_tensors, n_kv;
        if (!read_u64(n_tensors) || !read_u64(n_kv)) return false;

        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key;
            if (!read_string(key)) return false;
            uint32_t vt;
            if (!read_u32(vt)) return false;
            if (key == "general.architecture" && vt == 8) {
                if (!read_string(arch_)) return false;
            } else if (key == "general.alignment" && vt == 4) {
                uint32_t a;
                if (!read_u32(a)) return false;
                alignment_ = a ? a : 32;
            } else if ((key == "llm.hidden_size" || key == "qwen3.hidden_size") && vt == 4) {
                read_u32(cfg_.hidden_size);
            } else if ((key == "llm.intermediate_size" || key == "qwen3.intermediate_size") && vt == 4) {
                read_u32(cfg_.intermediate_size);
            } else if ((key == "llm.block_count" || key == "qwen3.block_count") && vt == 4) {
                read_u32(cfg_.num_layers);
            } else if ((key == "llm.attention.head_count" || key == "qwen3.attention.head_count") && vt == 4) {
                read_u32(cfg_.num_heads);
            } else if ((key == "llm.attention.head_count_kv" || key == "qwen3.attention.head_count_kv") && vt == 4) {
                read_u32(cfg_.num_kv_heads);
            } else if ((key == "llm.vocab_size" || key == "qwen3.vocab_size") && vt == 4) {
                read_u32(cfg_.vocab_size);
            } else if ((key == "llm.max_position_embeddings" || key == "qwen3.max_position_embeddings") && vt == 4) {
                read_u32(cfg_.max_seq_len);
            } else if ((key == "llm.rope.freq_base" || key == "qwen3.rope.freq_base") && vt == 4) {
                read_u32(cfg_.rope_theta_u32);
            } else if ((key == "llm.rope.freq_base" || key == "qwen3.rope.freq_base") && vt == 10) {
                float v; read_raw(&v, 4); cfg_.rope_theta_float = v;
            } else if (key == "general.name" && vt == 8) {
                read_string(cfg_.model_name);
            } else {
                if (!skip_value(vt)) return false;
            }
        }

        for (uint64_t i = 0; i < n_tensors; ++i) {
            std::string name;
            if (!read_string(name)) return false;
            uint32_t ndim;
            if (!read_u32(ndim)) return false;
            GgufTensorInfo info;
            info.shape.resize(ndim);
            for (uint32_t d = 0; d < ndim; ++d) {
                if (!read_u64(info.shape[d])) return false;
            }
            if (!read_u32(info.dtype)) return false;
            if (!read_u64(info.offset)) return false;
            tensors_[name] = info;
        }

        data_start_ = (uint64_t)f_.tellg();
        uint64_t rem = data_start_ % alignment_;
        if (rem) data_start_ += alignment_ - rem;
        return true;
    }

    const std::string& arch() const { return arch_; }

    struct Config {
        uint32_t hidden_size = 0;
        uint32_t intermediate_size = 0;
        uint32_t num_layers = 0;
        uint32_t num_heads = 0;
        uint32_t num_kv_heads = 0;
        uint32_t vocab_size = 0;
        uint32_t max_seq_len = 0;
        uint32_t rope_theta_u32 = 0;
        float rope_theta_float = 500000.0f;
        std::string model_name;
    };

    const Config& config() const { return cfg_; }

    const GgufTensorInfo* info(const std::string& name) const {
        auto it = tensors_.find(name);
        return it != tensors_.end() ? &it->second : nullptr;
    }

    bool read_tensor_bytes(const std::string& name, size_t expected_bytes,
                           std::vector<uint8_t>& out) {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) return false;
        const GgufTensorInfo& ti = it->second;
        uint64_t off = data_start_ + ti.offset;
        f_.seekg(off);
        if (!f_) return false;
        out.resize(expected_bytes);
        f_.read(reinterpret_cast<char*>(out.data()), expected_bytes);
        return (size_t)f_.gcount() == expected_bytes;
    }

    bool read_tensor_f32(const std::string& name, std::vector<float>& out) {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) return false;
        const GgufTensorInfo& ti = it->second;
        if (ti.dtype != 0) { // F32
            fprintf(stderr, "[gguf] tensor %s dtype=%u (expected 0=F32)\n", name.c_str(), ti.dtype);
            return false;
        }
        size_t n = 1;
        for (auto d : ti.shape) n *= d;
        uint64_t off = data_start_ + ti.offset;
        f_.seekg(off);
        if (!f_) return false;
        out.resize(n);
        f_.read(reinterpret_cast<char*>(out.data()), n * sizeof(float));
        return (size_t)f_.gcount() == n * sizeof(float);
    }

    bool read_tensor_f16(const std::string& name, std::vector<uint16_t>& out) {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) return false;
        const GgufTensorInfo& ti = it->second;
        if (ti.dtype != 1) { // F16
            fprintf(stderr, "[gguf] tensor %s dtype=%u (expected 1=F16)\n", name.c_str(), ti.dtype);
            return false;
        }
        size_t n = 1;
        for (auto d : ti.shape) n *= d;
        uint64_t off = data_start_ + ti.offset;
        f_.seekg(off);
        if (!f_) return false;
        out.resize(n);
        f_.read(reinterpret_cast<char*>(out.data()), n * sizeof(uint16_t));
        return (size_t)f_.gcount() == n * sizeof(uint16_t);
    }

private:
    bool read_u32(uint32_t& v) { return read_raw(&v, 4); }
    bool read_u64(uint64_t& v) { return read_raw(&v, 8); }
    bool read_raw(void* buf, size_t n) {
        f_.read(static_cast<char*>(buf), n);
        return !!f_;
    }
    bool read_string(std::string& s) {
        uint64_t len;
        if (!read_u64(len)) return false;
        s.resize((size_t)len);
        f_.read(s.data(), len);
        return !!f_;
    }
    bool skip_value(uint32_t vt) {
        switch (vt) {
            case 0: { uint8_t v; return read_raw(&v, 1); }
            case 1: { uint8_t v; return read_raw(&v, 1); }
            case 2: { uint16_t v; return read_raw(&v, 2); }
            case 3: { int16_t v; return read_raw(&v, 2); }
            case 4: { uint32_t v; return read_raw(&v, 4); }
            case 5: { int32_t v; return read_raw(&v, 4); }
            case 6: { float v; return read_raw(&v, 4); }
            case 7: { bool v; return read_raw(&v, 1); }
            case 8: { std::string s; return read_string(s); }
            case 10: { double v; return read_raw(&v, 8); }
            case 11: { uint64_t v; return read_raw(&v, 8); }
            case 12: { int64_t v; return read_raw(&v, 8); }
            case 13: { uint64_t v; return read_raw(&v, 8); } // array
            default: return false;
        }
    }

    std::ifstream f_;
    uint32_t version_ = 0;
    uint64_t data_start_ = 0;
    uint32_t alignment_ = 32;
    std::string arch_;
    Config cfg_;
    std::map<std::string, GgufTensorInfo> tensors_;
};

// ========================================================================
// FP16 helpers — must be defined before TQ2 packer that uses them
// ========================================================================
static inline float fp16_to_float(uint16_t h) {
    // FP16 -> FP32 conversion
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        // Subnormal or zero
        if (mant == 0) {
            f = sign << 31;
        } else {
            int e = -1;
            uint32_t m = mant;
            while ((m & 0x400) == 0) { m <<= 1; --e; }
            exp = e + 127;
            mant = m & 0x3ff;
            f = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        // Inf/NaN
        f = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, 4);
    return result;
}

static inline uint16_t float_to_fp16(float v) {
    uint32_t f;
    memcpy(&f, &v, 4);
    uint32_t sign = (f >> 31) & 1;
    int32_t exp  = (f >> 23) & 0xff;
    uint32_t mant = f & 0x7fffff;
    uint16_t h;
    if (exp == 0) {
        h = (sign << 15);
    } else if (exp == 0xff) {
        h = (sign << 15) | 0x7c00;
        if (mant) h |= 0x200; // NaN
    } else {
        int32_t newexp = exp - 127 + 15;
        if (newexp >= 31) {
            h = (sign << 15) | 0x7c00;
        } else if (newexp <= 0) {
            // Flush to zero
            h = (sign << 15);
        } else {
            h = (sign << 15) | (newexp << 10) | (mant >> 13);
        }
    }
    return h;
}

static inline float valRatio(float w, float d) {
    if (d < 1e-10f) return 0.0f;
    return w / d;
}

// ========================================================================
// TQ2_0_g128 packer — ternarize float weights to 34-byte blocks of 128
// ========================================================================

// Convert float to ternary {-1, 0, +1} with threshold around 2 std devs
// that ensures ~50% zeros for BitNet-style models.
static inline int8_t ternarize(float v, float threshold) {
    if (v > threshold) return 1;
    if (v < -threshold) return -1;
    return 0;
}

// Pack a row of 128 floats into TQ2_0_g128 block (34 bytes).
// Layout: 32 bytes of 2-bit codes (weights in {-1,0,+1} → codes {0,1,2})
//         + 2 bytes FP16 block scale d.
void pack_tq2_block(const float* vals, int n, uint8_t* out) {
    // Find the right threshold for 50% zeros
    float max_abs = 0.0f;
    double sum_abs = 0.0;
    for (int i = 0; i < n; ++i) {
        float a = std::fabs(vals[i]);
        if (a > max_abs) max_abs = a;
        sum_abs += a;
    }
    float mean_abs = (float)(sum_abs / n);
    float threshold = 0.7f * mean_abs;
    if (threshold < 1e-10f) threshold = 1e-10f;

    int8_t ternary[128];
    float max_mag = 0.0f;
    for (int i = 0; i < n; ++i) {
        ternary[i] = ternarize(vals[i], threshold);
        if (ternary[i] != 0) {
            float mag = std::fabs(vals[i]);
            if (mag > max_mag) max_mag = mag;
        }
    }
    if (max_mag < 1e-10f) max_mag = 1.0f;

    uint16_t d_fp16 = float_to_fp16(max_mag);
    float d = fp16_to_float(d_fp16);

    for (int i = 0; i < 32; ++i) {
        uint8_t byte = 0;
        for (int j = 0; j < 4; ++j) {
            int idx = i * 4 + j;
            int code;
            float w = (float)ternary[idx] * max_mag / d;
            float ratio = valRatio(w, d);
            (void)ratio;
            if (w > d * 0.5f) code = 2;
            else if (w < -d * 0.5f) code = 0;
            else code = 1;
            byte |= (uint8_t)(code << (j * 2));
        }
        out[i] = byte;
    }
    memcpy(out + 32, &d_fp16, 2);
}

void ternarize_and_pack_tq2(const float* data, int rows, int cols,
                            std::vector<uint8_t>& packed) {
    const int gs = 128;
    const int block_bytes = 34;
    const int blocks_per_row = cols / gs;
    const size_t row_bytes = blocks_per_row * block_bytes;
    packed.resize(rows * row_bytes);

    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < blocks_per_row; ++b) {
            pack_tq2_block(data + r * cols + b * gs, gs,
                          packed.data() + r * row_bytes + b * block_bytes);
        }
    }
}

// ========================================================================
// Sidecar GGUF writer — writes FP32 norms + embedding 
// ========================================================================
class GgufWriter {
public:
    GgufWriter(const std::string& path) : f_(path, std::ios::binary) {}

    bool write_header(uint64_t n_tensors, uint64_t n_kv) {
        f_.write("GGUF", 4);
        write_u32(3); // version 3
        write_u64(n_tensors);
        write_u64(n_kv);
        return !!f_;
    }

    void write_kv_string(const std::string& key, const std::string& val) {
        write_string(key);
        write_u32(8); // string type
        write_string(val);
    }

    void write_kv_u32(const std::string& key, uint32_t val) {
        write_string(key);
        write_u32(4); // u32 type
        write_u32(val);
    }

    void write_kv_f32(const std::string& key, float val) {
        write_string(key);
        write_u32(6); // float32 type
        write_raw(&val, 4);
    }

    void write_tensor_info(const std::string& name,
                           const std::vector<uint64_t>& shape,
                           uint32_t dtype,
                           uint64_t offset) {
        write_string(name);
        write_u32((uint32_t)shape.size());
        for (auto d : shape) write_u64(d);
        write_u32(dtype);
        write_u64(offset);
    }

    void write_tensor_data(const void* data, size_t bytes) {
        f_.write(static_cast<const char*>(data), bytes);
    }

    void align_to(uint32_t alignment) {
        uint64_t pos = f_.tellp();
        uint64_t rem = pos % alignment;
        if (rem) {
            uint64_t pad = alignment - rem;
            for (uint64_t i = 0; i < pad; ++i) f_.put(0);
        }
    }

    uint64_t tell() { return (uint64_t)f_.tellp(); }

private:
    void write_u32(uint32_t v) { write_raw(&v, 4); }
    void write_u64(uint64_t v) { write_raw(&v, 8); }
    void write_string(const std::string& s) {
        write_u64(s.size());
        write_raw(s.data(), s.size());
    }
    void write_raw(const void* d, size_t n) {
        f_.write(static_cast<const char*>(d), n);
    }
    std::ofstream f_;
};

// ========================================================================
// Main
// ========================================================================
int main(int argc, char** argv) {
    const char* input_path = nullptr;
    const char* output_path = nullptr;
    const char* config_path = nullptr;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--input" && i + 1 < argc) input_path = argv[++i];
        else if (a == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (a == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (a == "--help" || a == "-h") {
            fprintf(stderr, "Usage: %s --input model-f16.gguf --output basename [--config model-tq2.gguf]\n", argv[0]);
            return 0;
        }
    }

    if (!input_path || !output_path) {
        fprintf(stderr, "Usage: %s --input model-f16.gguf --output basename\n", argv[0]);
        return 1;
    }

    std::string h1b_path = std::string(output_path) + ".h1b";
    std::string gguf_path = std::string(output_path) + ".gguf";

    fprintf(stderr, "[convert] loading GGUF: %s\n", input_path);
    GgufReader reader;
    if (!reader.open(input_path)) {
        fprintf(stderr, "[convert] failed to open GGUF: %s\n", input_path);
        return 1;
    }

    const auto& cfg = reader.config();
    fprintf(stderr, "[convert] arch=%s hs=%d is=%d layers=%d heads=%d kv_heads=%d vocab=%d\n",
            reader.arch().c_str(),
            cfg.hidden_size, cfg.intermediate_size, cfg.num_layers,
            cfg.num_heads, cfg.num_kv_heads, cfg.vocab_size);

    uint32_t hs = cfg.hidden_size;
    uint32_t is_ = cfg.intermediate_size;
    uint32_t nh = cfg.num_heads;
    uint32_t nkv = cfg.num_kv_heads;
    uint32_t hd = hs / nh;
    uint32_t num_layers = cfg.num_layers;
    uint32_t vocab_size = cfg.vocab_size;

    if (hs == 0 || is_ == 0 || num_layers == 0) {
        fprintf(stderr, "[convert] incomplete config from GGUF\n");
        return 1;
    }

    // ==================================================================
    // Step 1: Ternarize all weight matrices → TQ2 packed format
    // ==================================================================
    struct TensorData {
        std::vector<float> data_f32;
        std::vector<uint16_t> data_f16;
    };

    // Read all F16 weight tensors
    auto read_weight_f16 = [&](const std::string& name) -> TensorData {
        TensorData td;
        // Try F16 first
        if (reader.read_tensor_f16(name, td.data_f16)) {
            return td;
        }
        // Fall back to F32
        if (reader.read_tensor_f32(name, td.data_f32)) {
            return td;
        }
        fprintf(stderr, "[convert] WARNING: could not read tensor %s\n", name.c_str());
        return td;
    };

    auto get_float = [&](const TensorData& td, size_t i) -> float {
        if (!td.data_f32.empty()) return td.data_f32[i];
        if (!td.data_f16.empty()) {
            uint16_t h = td.data_f16[i];
            return fp16_to_float(h);
        }
        return 0.0f;
    };

    auto tensor_size = [&](const TensorData& td) -> size_t {
        if (!td.data_f32.empty()) return td.data_f32.size();
        return td.data_f16.size();
    };

    // Collect weights per layer
    struct LayerWeights {
        TensorData q, k, v, o, gate, up, down;
        TensorData attn_norm, ffn_norm, attn_q_norm, attn_k_norm;
    };
    std::vector<LayerWeights> layers(num_layers);

    for (uint32_t l = 0; l < num_layers; ++l) {
        std::string prefix = "blk." + std::to_string(l) + ".";
        layers[l].q       = read_weight_f16(prefix + "attn_q.weight");
        layers[l].k       = read_weight_f16(prefix + "attn_k.weight");
        layers[l].v       = read_weight_f16(prefix + "attn_v.weight");
        layers[l].o       = read_weight_f16(prefix + "attn_output.weight");
        layers[l].gate    = read_weight_f16(prefix + "ffn_gate.weight");
        layers[l].up      = read_weight_f16(prefix + "ffn_up.weight");
        layers[l].down    = read_weight_f16(prefix + "ffn_down.weight");
        layers[l].attn_norm   = read_weight_f16(prefix + "attn_norm.weight");
        layers[l].ffn_norm    = read_weight_f16(prefix + "ffn_norm.weight");
        layers[l].attn_q_norm = read_weight_f16(prefix + "attn_q_norm.weight");
        layers[l].attn_k_norm = read_weight_f16(prefix + "attn_k_norm.weight");

        fprintf(stderr, "[convert] layer %d: q=%zu k=%zu v=%zu o=%zu gate=%zu up=%zu down=%zu\n",
                l,
                tensor_size(layers[l].q) / (hs * 2),  // rough: rows * cols div by 2 for F16
                tensor_size(layers[l].k) / (hs * 2),
                tensor_size(layers[l].v) / (hs * 2),
                tensor_size(layers[l].o) / (nh * hd * 2),
                tensor_size(layers[l].gate) / (hs * 2),
                tensor_size(layers[l].up) / (hs * 2),
                tensor_size(layers[l].down) / (is_ * 2));
    }

    // Read embedding + output_norm
    TensorData token_embd = read_weight_f16("token_embd.weight");
    TensorData output_norm = read_weight_f16("output_norm.weight");

    // ==================================================================
    // Common vars shared by step 2 (h1b) and step 3 (sidecar GGUF)
    float rope_theta = cfg.rope_theta_float;
    float rms_norm_eps = 1e-5f;

    // Step 2: Write .h1b file
    // ==================================================================
    {
        fprintf(stderr, "[convert] writing h1b: %s\n", h1b_path.c_str());
        std::ofstream f(h1b_path, std::ios::binary);
        if (!f) { fprintf(stderr, "[convert] cannot write: %s\n", h1b_path.c_str()); return 1; }

        // Magic "H1B\0"
        f.write("H1B\0", 4);

        // Version (use version 1 for BONSAI_TQ2)
        int32_t version = 1;
        f.write(reinterpret_cast<const char*>(&version), 4);

        // cfg[9]
        int32_t cfg9[9] = {
            (int32_t)hs, (int32_t)is_, (int32_t)num_layers,
            (int32_t)nh, (int32_t)nkv, (int32_t)vocab_size,
            (int32_t)cfg.max_seq_len ? (int32_t)cfg.max_seq_len : 4096,
            1, // tie_embeddings
            (int32_t)0x8u // H1B_FLAG_BONSAI_TQ2
        };
        // If max_seq_len is 0, use a reasonable default
        if (cfg9[6] <= 0) cfg9[6] = 4096;
        f.write(reinterpret_cast<const char*>(cfg9), sizeof(cfg9));

        // Extra fp32[2]: rope_theta, rms_norm_eps (version >= 2)
        f.write(reinterpret_cast<const char*>(&rope_theta), 4);
        f.write(reinterpret_cast<const char*>(&rms_norm_eps), 4);

        // Embedding: vocab_size * hs floats (zeros — sidecar will fill)
        std::vector<float> zero_embd(vocab_size * hs, 0.0f);
        f.write(reinterpret_cast<const char*>(zero_embd.data()),
                zero_embd.size() * sizeof(float));

        // Final norm: hs floats (zeros)
        std::vector<float> zero_norm(hs, 0.0f);
        f.write(reinterpret_cast<const char*>(zero_norm.data()),
                zero_norm.size() * sizeof(float));

        // Per-layer norms (zeros for BONSAI_QWEN3)
        const int norms_per_layer = 2 + 4 + 2; // input_norm, post_attn_norm, 
                                                 // attn_sub_norm, 4 skip, ffn_sub_norm
        for (uint32_t l = 0; l < num_layers; ++l) {
            // input_norm (hs floats)
            f.write(reinterpret_cast<const char*>(zero_norm.data()), hs * sizeof(float));
            // post_attn_norm (hs floats)
            f.write(reinterpret_cast<const char*>(zero_norm.data()), hs * sizeof(float));
            // attn_sub_norm (hs floats)
            f.write(reinterpret_cast<const char*>(zero_norm.data()), hs * sizeof(float));
            // 4 skip copies of attn_sub (4 * hs floats)
            f.write(reinterpret_cast<const char*>(zero_norm.data()), hs * sizeof(float));
            f.write(reinterpret_cast<const char*>(zero_norm.data()), hs * sizeof(float));
            f.write(reinterpret_cast<const char*>(zero_norm.data()), hs * sizeof(float));
            f.write(reinterpret_cast<const char*>(zero_norm.data()), hs * sizeof(float));
            // ffn_sub_norm (is floats)
            std::vector<float> zero_is_norm(is_, 0.0f);
            f.write(reinterpret_cast<const char*>(zero_is_norm.data()), is_ * sizeof(float));
        }

        // Ternary weights — TQ2 packed blocks
        const int block_bytes = 34;
        const int gs = 128;

        auto write_tq2_weight = [&](const TensorData& td, int rows, int cols) {
            if (tensor_size(td) == 0) {
                fprintf(stderr, "[convert] WARNING: empty tensor, writing zeros\n");
                std::vector<uint8_t> zeros(rows * (cols / gs) * block_bytes);
                f.write(reinterpret_cast<const char*>(zeros.data()), zeros.size());
                return;
            }
            size_t n = tensor_size(td);
            std::vector<float> f32(n);
            for (size_t i = 0; i < n; ++i) {
                f32[i] = get_float(td, i);
            }
            std::vector<uint8_t> packed;
            ternarize_and_pack_tq2(f32.data(), rows, cols, packed);
            f.write(reinterpret_cast<const char*>(packed.data()), packed.size());
        };

        for (uint32_t l = 0; l < num_layers; ++l) {
            write_tq2_weight(layers[l].q,    nh * hd, hs);
            write_tq2_weight(layers[l].k,    nkv * hd, hs);
            write_tq2_weight(layers[l].v,    nkv * hd, hs);
            write_tq2_weight(layers[l].o,    hs, nh * hd);
            write_tq2_weight(layers[l].gate, is_, hs);
            write_tq2_weight(layers[l].up,   is_, hs);
            write_tq2_weight(layers[l].down, hs, is_);
        }

        f.close();
        fprintf(stderr, "[convert] h1b written: %s\n", h1b_path.c_str());
    }

    // ==================================================================
    // Step 3: Write sidecar GGUF with FP32 norms + embedding
    // ==================================================================
    {
        fprintf(stderr, "[convert] writing sidecar GGUF: %s\n", gguf_path.c_str());

        // Count tensors we'll write
        // Per layer: attn_norm, ffn_norm, attn_q_norm, attn_k_norm (4) × num_layers
        // + output_norm + token_embd (TQ2_0_g128, dtype 42)
        uint64_t n_tensors = 4 * num_layers + 2;
        uint64_t n_kv = 8; // architecture + model dimensions

        GgufWriter w(gguf_path);
        w.write_header(n_tensors, n_kv);

        // KV metadata
        w.write_kv_string("general.architecture", "qwen3");
        if (!cfg.model_name.empty())
            w.write_kv_string("general.name", cfg.model_name);
        w.write_kv_u32("qwen3.hidden_size", hs);
        w.write_kv_u32("qwen3.intermediate_size", is_);
        w.write_kv_u32("qwen3.block_count", num_layers);
        w.write_kv_u32("qwen3.attention.head_count", nh);
        w.write_kv_u32("qwen3.attention.head_count_kv", nkv);
        w.write_kv_u32("qwen3.vocab_size", vocab_size);
        uint32_t max_seq = cfg.max_seq_len ? cfg.max_seq_len : 4096;
        w.write_kv_u32("qwen3.max_position_embeddings", max_seq);
        w.write_kv_f32("qwen3.rope.freq_base", rope_theta);

        w.align_to(32);
        uint64_t data_start = w.tell();
        uint64_t offset = 0;

        // Write tensor info headers first
        // Per-layer norms: F32
        for (uint32_t l = 0; l < num_layers; ++l) {
            std::vector<uint64_t> shape_1d = {hs};
            w.write_tensor_info("blk." + std::to_string(l) + ".attn_norm.weight",
                                shape_1d, 0, offset);
            offset += hs * sizeof(float);

            w.write_tensor_info("blk." + std::to_string(l) + ".ffn_norm.weight",
                                shape_1d, 0, offset);
            offset += hs * sizeof(float);

            std::vector<uint64_t> shape_hd = {hd};
            w.write_tensor_info("blk." + std::to_string(l) + ".attn_q_norm.weight",
                                shape_hd, 0, offset);
            offset += hd * sizeof(float);

            w.write_tensor_info("blk." + std::to_string(l) + ".attn_k_norm.weight",
                                shape_hd, 0, offset);
            offset += hd * sizeof(float);
        }

        // token_embd.weight: [vocab_size, hs] F32
        {
            std::vector<uint64_t> shape_2d = {(uint64_t)vocab_size, (uint64_t)hs};
            w.write_tensor_info("token_embd.weight", shape_2d, 0, offset);
            offset += (uint64_t)vocab_size * hs * sizeof(float);
        }

        // output_norm.weight: [hs] F32
        {
            std::vector<uint64_t> shape_1d = {(uint64_t)hs};
            w.write_tensor_info("output_norm.weight", shape_1d, 0, offset);
            offset += hs * sizeof(float);
        }

        // ── Write actual data ─────────────────────────────────────
        w.align_to(32);

        // Per-layer norms: read from source GGUF and write to sidecar
        for (uint32_t l = 0; l < num_layers; ++l) {
            auto write_norm = [&](const TensorData& td) {
                if (tensor_size(td) == 0) {
                    // Fallback: write identity-like weights (all 1s for RMSNorm)
                    std::vector<float> ones(tensor_size(layers[l].attn_norm) > 0
                        ? tensor_size(layers[l].attn_norm) : hs, 1.0f);
                    w.write_tensor_data(ones.data(), ones.size() * sizeof(float));
                } else {
                    size_t n = tensor_size(td);
                    std::vector<float> f32(n);
                    for (size_t i = 0; i < n; ++i) f32[i] = get_float(td, i);
                    w.write_tensor_data(f32.data(), n * sizeof(float));
                }
            };
            write_norm(layers[l].attn_norm);
            write_norm(layers[l].ffn_norm);
            // attn_q_norm and attn_k_norm are shape [hd], not [hs]
            {
                auto& td = layers[l].attn_q_norm;
                if (tensor_size(td) == 0) {
                    std::vector<float> ones(hd, 1.0f);
                    w.write_tensor_data(ones.data(), hd * sizeof(float));
                } else {
                    size_t n = tensor_size(td);
                    std::vector<float> f32(n);
                    for (size_t i = 0; i < n; ++i) f32[i] = get_float(td, i);
                    w.write_tensor_data(f32.data(), n * sizeof(float));
                }
            }
            {
                auto& td = layers[l].attn_k_norm;
                if (tensor_size(td) == 0) {
                    std::vector<float> ones(hd, 1.0f);
                    w.write_tensor_data(ones.data(), hd * sizeof(float));
                } else {
                    size_t n = tensor_size(td);
                    std::vector<float> f32(n);
                    for (size_t i = 0; i < n; ++i) f32[i] = get_float(td, i);
                    w.write_tensor_data(f32.data(), n * sizeof(float));
                }
            }
        }

        // token_embd.weight
        {
            size_t n = tensor_size(token_embd);
            if (n == 0) {
                std::vector<float> zeros(vocab_size * hs, 0.0f);
                w.write_tensor_data(zeros.data(), zeros.size() * sizeof(float));
            } else {
                std::vector<float> f32(n);
                for (size_t i = 0; i < n; ++i) f32[i] = get_float(token_embd, i);
                w.write_tensor_data(f32.data(), n * sizeof(float));
            }
        }

        // output_norm.weight
        {
            size_t n = tensor_size(output_norm);
            if (n == 0) {
                std::vector<float> ones(hs, 1.0f);
                w.write_tensor_data(ones.data(), hs * sizeof(float));
            } else {
                std::vector<float> f32(n);
                for (size_t i = 0; i < n; ++i) f32[i] = get_float(output_norm, i);
                w.write_tensor_data(f32.data(), n * sizeof(float));
            }
        }

        w.align_to(32);
        fprintf(stderr, "[convert] sidecar GGUF written: %s (data=%lu bytes)\n",
                gguf_path.c_str(), (unsigned long)w.tell());
    }

    fprintf(stderr, "[convert] done. Run with: zaya_server --model %s\n",
            h1b_path.c_str());
    return 0;
}
