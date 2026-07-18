// model_discovery.cpp — Scan weights directory for model files (GGUF, H1B, safetensors)
// and read their headers to populate ModelConfig without loading full weights.

#include "model_discovery.h"
#include "q4nx_reader.h"
#include "safetensors_reader.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <limits>
#include <dirent.h>
#include <sys/stat.h>

// ── Minimal GGUF header reader ──────────────────────────────────────────────
// Reads just enough of a GGUF file to extract architecture + dimensions.
// The GGUF spec: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8)
// Then kv pairs: key_length(8) + key_data + value_type(4) + value_data

// Best-effort mapping of GGUF's general.file_type (ggml_ftype) to a human-readable
// quantization tag. Not exhaustive — covers the common cases, falls back to a
// numbered "unknown" tag for anything else rather than silently guessing.
static std::string ggml_ftype_name(uint32_t ft) {
    switch (ft) {
        case 0:  return "F32";
        case 1:  return "F16";
        case 2:  return "Q4_0";
        case 3:  return "Q4_1";
        case 7:  return "Q8_0";
        case 8:  return "Q5_0";
        case 9:  return "Q5_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K_S";
        case 12: return "Q3_K_M";
        case 13: return "Q3_K_L";
        case 14: return "Q4_K_S";
        case 15: return "Q4_K_M";
        case 16: return "Q5_K_S";
        case 17: return "Q5_K_M";
        case 18: return "Q6_K";
        case 32: return "BF16";
        default: return "unknown(" + std::to_string(ft) + ")";
    }
}

static bool read_gguf_metadata(const std::string& path, ModelConfig& cfg) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    // Magic: "GGUF" (0x46554747 little-endian)
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GGUF", 4) != 0) {
        fclose(f);
        // Try .h1b magic
        f = fopen(path.c_str(), "rb");
        uint32_t h1b_magic;
        if (fread(&h1b_magic, 4, 1, f) != 1 || h1b_magic != 0x48314248) {
            fclose(f);
            return false;
        }
        // H1B format — read config from header
        uint32_t version, hs, is, n_layers, n_heads, n_kv, max_seq;
        fseek(f, 8, SEEK_SET);
        fread(&version, 4, 1, f); fread(&hs, 4, 1, f); fread(&is, 4, 1, f);
        fread(&n_layers, 4, 1, f); fread(&n_heads, 4, 1, f); fread(&n_kv, 4, 1, f);
        fread(&max_seq, 4, 1, f);
        cfg.hidden = cfg.hidden_size = hs;
        cfg.n_ff = cfg.intermediate_size = is;
        cfg.n_layers = cfg.num_layers = n_layers;
        cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = n_heads;
        cfg.n_kv_heads = cfg.num_kv_heads = n_kv ? n_kv : n_heads;
        cfg.head_dim = hs / n_heads;
        cfg.max_seq_len = max_seq ? max_seq : 2048;
        cfg.model_path = path;
        cfg.format = ModelFormat::H1B;
        // Derive name from filename
        auto slash = path.find_last_of('/');
        auto dot = path.find_last_of('.');
        cfg.model_name = path.substr(slash + 1, dot - slash - 1);
        fclose(f);
        return true;
    }

    uint32_t version;
    fread(&version, 4, 1, f);
    uint64_t tensor_count, kv_count;
    fread(&tensor_count, 8, 1, f);
    fread(&kv_count, 8, 1, f);

    // Defaults
    cfg.hidden = cfg.hidden_size = 2048;
    cfg.n_layers = cfg.num_layers = 32;
    cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = 32;
    cfg.n_kv_heads = cfg.num_kv_heads = 32;
    cfg.head_dim = 128;
    cfg.n_ff = cfg.intermediate_size = 8192;
    cfg.vocab = cfg.vocab_size = 32000;
    cfg.max_seq_len = 2048;
    cfg.rope_theta = 10000.0f;
    cfg.rms_norm_eps = 1e-6f;
    // 0 experts means "dense, not MoE" — distinct from ModelConfig's default
    // constructor value (16), which exists only for the hardcoded Zaya .bin
    // path. A real MoE GGUF overwrites this via general.expert_count below.
    cfg.n_experts = cfg.num_experts = 0;
    cfg.num_experts_top = 0;
    cfg.model_path = path;
    cfg.format = ModelFormat::GGUF;
    auto slash = path.find_last_of('/');
    auto dot = path.find_last_of('.');
    cfg.model_name = path.substr(slash + 1, dot - slash - 1);

    bool explicit_head_dim = false;

    // Parse metadata key-value pairs
    char key_buf[256];
    for (uint64_t i = 0; i < kv_count; i++) {
        uint64_t key_len;
        if (fread(&key_len, 8, 1, f) != 1) break;
        if (key_len > 255) { fseek(f, key_len, SEEK_CUR); continue; }
        if (fread(key_buf, 1, key_len, f) != key_len) break;
        key_buf[key_len] = '\0';

        uint32_t val_type;
        if (fread(&val_type, 4, 1, f) != 1) break;

        auto read_str = [&]() -> std::string {
            uint64_t len; if (fread(&len, 8, 1, f) != 1) return "";
            std::string s(len, '\0');
            fread(&s[0], 1, len, f);
            return s;
        };
        auto read_u32 = [&]() -> uint32_t {
            uint32_t v; fread(&v, 4, 1, f); return v;
        };
        auto read_f32 = [&]() -> float {
            float v; fread(&v, 4, 1, f); return v;
        };
        auto read_arr = [&]() -> uint32_t {
            uint32_t vtype, n; fread(&vtype, 4, 1, f); fread(&n, 4, 1, f);
            if (vtype == 4 && n >= 1) { uint32_t v; fread(&v, 4, 1, f); return v; }
            fseek(f, n * 4, SEEK_CUR);
            return 0;
        };

        std::string key(key_buf);
        if (key == "general.architecture") {
            // Keep separate from model_name (general.name), which commonly appears
            // later in the same file and would otherwise clobber this value.
            cfg.architecture = read_str();
        }
        else if (key == "general.name") { cfg.model_name = read_str(); }
        else if (key == "tokenizer.ggml.model") { /* ignore */ }
        else if (key == "general.file_type") { cfg.quantization = ggml_ftype_name(read_u32()); }
        else {
            // Check for dimension keys (architecture-agnostic by checking suffixes)
            auto ends_with = [&](const std::string& s) -> bool {
                if (key.size() < s.size()) return false;
                return key.substr(key.size() - s.size()) == s;
            };
            bool matched = true;
            if (ends_with(".attention.head_count") || ends_with(".attention.head_count_kv")) {
                uint32_t v = read_u32();
                if (ends_with(".attention.head_count"))
                    cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = v;
                else
                    cfg.n_kv_heads = cfg.num_kv_heads = v;
            } else if (key == "tokenizer.ggml.tokens") {
                // Read array element count as vocab size
                uint32_t at; fread(&at, 4, 1, f); uint64_t an; fread(&an, 8, 1, f);
                cfg.vocab = cfg.vocab_size = (int)an;
                // skip all token strings
                if (at == 8) for (uint64_t j = 0; j < an; j++) { uint64_t sl; fread(&sl, 8, 1, f); fseek(f, sl, SEEK_CUR); }
                else fseek(f, an * 4, SEEK_CUR);
            } else if (ends_with(".block_count")) { 
                cfg.n_layers = cfg.num_layers = read_u32(); 
            } else if (ends_with(".feed_forward_length")) { 
                cfg.n_ff = cfg.intermediate_size = read_u32(); 
            } else if (ends_with(".embedding_length")) { 
                cfg.hidden = cfg.hidden_size = read_u32(); 
            } else if (ends_with(".rope.freq_base")) { cfg.rope_theta = read_f32(); }
            else if (ends_with(".expert_count")) { cfg.n_experts = cfg.num_experts = read_u32(); }
            else if (ends_with(".expert_used_count")) { cfg.num_experts_top = read_u32(); }
            else if (ends_with(".attention.key_length")) {
                // Authoritative head_dim when present — some architectures
                // (e.g. Qwen3: hidden=1024, heads=16, but head_dim=128) are
                // NOT hidden/n_heads. Falls back to that derivation below
                // only when this key is absent.
                cfg.head_dim = read_u32();
                explicit_head_dim = true;
            }
            else { matched = false; }
        if (!matched) {
            // Skip unknown values based on type. GGUF value types: 0=u8,
            // 1=i8, 2=u16, 3=i16, 4=u32, 5=i32, 6=f32, 7=bool, 8=string
            // (length-prefixed, NOT fixed-size — a prior version of this
            // switch treated it as a fixed 8-byte int64 and silently desynced
            // every subsequent KV read whenever a real-world GGUF file had
            // any unmatched string key, e.g. general.type/general.size_label
            // — verified against a real llama.cpp-converted Qwen3 GGUF,
            // where this corrupted parsing badly enough that every dimension
            // field silently fell back to defaults), 9=array, 10=u64,
            // 11=i64, 12=f64. Matches the already-correct skip logic in
            // read_gguf_tensor()/read_gguf_vocab() below.
            switch (val_type) {
                case 0: case 1: fseek(f, 1, SEEK_CUR); break;
                case 2: case 3: fseek(f, 2, SEEK_CUR); break;
                case 4: case 5: case 6: fseek(f, 4, SEEK_CUR); break;
                case 7: fseek(f, 1, SEEK_CUR); break; // bool
                case 8: { uint64_t slen; fread(&slen, 8, 1, f); fseek(f, slen, SEEK_CUR); break; } // string
                case 9: { // array
                    uint32_t atype; uint64_t an;
                    fread(&atype, 4, 1, f); fread(&an, 8, 1, f);
                    if (atype == 8) {
                        for (uint64_t j = 0; j < an; j++) { uint64_t sl; fread(&sl, 8, 1, f); fseek(f, sl, SEEK_CUR); }
                    } else {
                        int sz = 4;
                        if (atype <= 7) { static const int s[] = {1,1,2,2,4,4,4,1}; sz = s[atype]; }
                        else if (atype >= 10 && atype <= 12) sz = 8;
                        fseek(f, an * sz, SEEK_CUR);
                    }
                    break;
                }
                case 10: case 11: case 12: fseek(f, 8, SEEK_CUR); break;
                default: break;
            }
        }
    }

    }

    // Derive head_dim from hidden / heads — unless the file gave an explicit
    // attention.key_length (authoritative; not always equal to hidden/heads).
    if (!explicit_head_dim) cfg.head_dim = cfg.hidden / cfg.n_heads;
    // Default KV heads to full if not set
    if (cfg.n_kv_heads == 0) cfg.n_kv_heads = cfg.n_heads;
    cfg.num_kv_heads = cfg.n_kv_heads;

    fclose(f);
    return true;
}

// ── Scan directory for model files ──────────────────────────────────────────
std::vector<ModelConfig> discover_models(const std::string& dir) {
    std::vector<ModelConfig> models;
    DIR* d = opendir(dir.c_str());
    if (!d) {
        fprintf(stderr, "[discover] could not open %s\n", dir.c_str());
        return models;
    }

    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        std::string name(entry->d_name);
        if (name == "." || name == "..") continue;

        // Check extension
        auto dot = name.find_last_of('.');
        if (dot == std::string::npos) continue;
        std::string ext = name.substr(dot);
        if (ext != ".gguf" && ext != ".h1b" && ext != ".safetensors" && ext != ".bin" && ext != ".q4nx") continue;

        std::string full = dir + "/" + name;
        struct stat st;
        if (stat(full.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;

        ModelConfig cfg;
        bool ok = false;
        if (ext == ".gguf") ok = read_gguf_metadata(full, cfg);
        else if (ext == ".h1b") ok = read_gguf_metadata(full, cfg);
        else if (ext == ".q4nx") ok = read_q4nx_metadata(full, cfg);
        else if (ext == ".safetensors") ok = read_safetensors_metadata(full, cfg);
        else if (ext == ".bin") {
            // .bin files: use the directory name as model name, Zaya defaults
            cfg.model_name = dir.substr(dir.find_last_of('/') + 1);
            cfg.model_path = full;
            cfg.format = ModelFormat::RAW_BIN;
            cfg.architecture = "zaya1";
            cfg.hidden = cfg.hidden_size = 2048;
            cfg.n_layers = cfg.num_layers = 40;
            cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = 8;
            cfg.n_kv_heads = cfg.num_kv_heads = 2;
            cfg.head_dim = 128;
            cfg.n_ff = cfg.intermediate_size = 2048;
            cfg.vocab = cfg.vocab_size = 262272;
            cfg.n_experts = cfg.num_experts = 16;
            ok = true;
        }
        else continue;

        if (ok) {
            models.push_back(cfg);
            printf("  📦 %-30s %d layers, %d hidden, %d heads%s — %s/%s/%s\n",
                   cfg.model_name.c_str(), cfg.n_layers, cfg.hidden, cfg.n_heads,
                   cfg.n_kv_heads != cfg.n_heads ? " (GQA)" : "",
                   ext.c_str(),
                   cfg.architecture.empty() ? "?" : cfg.architecture.c_str(),
                   cfg.quantization.empty() ? "?" : cfg.quantization.c_str());
        }
    }
    closedir(d);

    printf("[discover] %zu model(s) found in %s\n", models.size(), dir.c_str());
    return models;
}



// ── Read vocab size from GGUF embedding tensor shape ──────────────────────
// Faster than parsing all KV pairs. Reads the first tensor's second dimension.
int read_gguf_vocab(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return 0;
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GGUF", 4) != 0) { fclose(f); return 0; }
    uint32_t ver; fread(&ver, 4, 1, f);
    uint64_t tc, kc; fread(&tc, 8, 1, f); fread(&kc, 8, 1, f);
    // Skip all KVs (use correct v3 types)
    char kbuf[256];
    for (uint64_t i = 0; i < kc; i++) {
        uint64_t kl; fread(&kl, 8, 1, f);
        if (kl > 255) { fseek(f, kl, SEEK_CUR); } else { fread(kbuf, 1, kl, f); }
        uint32_t vt; fread(&vt, 4, 1, f);
        switch (vt) {
            case 0: fseek(f,1,SEEK_CUR);break; case 1: fseek(f,1,SEEK_CUR);break;
            case 2: fseek(f,2,SEEK_CUR);break; case 3: fseek(f,2,SEEK_CUR);break;
            case 4:case 5:case 6: fseek(f,4,SEEK_CUR);break;
            case 7: fseek(f,1,SEEK_CUR);break;
            case 8: { uint64_t sl; fread(&sl,8,1,f); fseek(f,sl,SEEK_CUR); break; }
            case 9: {
                uint32_t at; fread(&at,4,1,f); uint64_t an; fread(&an,8,1,f);
                if (at == 8) for (uint64_t j=0;j<an;j++){uint64_t sl;fread(&sl,8,1,f);fseek(f,sl,SEEK_CUR);}
                else fseek(f, an*4, SEEK_CUR);
                break;
            }
            case 10:case 11:case 12: fseek(f,8,SEEK_CUR);break;
            default: fseek(f,4,SEEK_CUR);break;
        }
    }
    // Read first tensor's shape
    for (uint64_t i = 0; i < tc && i < 1; i++) {
        uint64_t nl; fread(&nl, 8, 1, f); if (nl > 512) break;
        fseek(f, nl, SEEK_CUR); // skip name
        uint32_t nd; fread(&nd, 4, 1, f);
        for (int j = 0; j < nd; j++) {
            uint64_t dim; fread(&dim, 8, 1, f);
            if (j == 0) { fclose(f); return (int)dim; } // first dim = vocab (embed shape is [vocab, hidden])
        }
    }
    fclose(f);
    return 0;
}
bool read_gguf_header(const std::string& path, ModelConfig& cfg) {
    return read_gguf_metadata(path, cfg);
}


// ── Read a GGUF tensor's data ──────────────────────────────────────────────
// Uses the same GGUF header parsing as read_gguf_metadata, then reads the actual
// tensor data at the correct offset. Handles F32, F16, Q4_0, Q5_0, Q5_1, Q8_0,
// Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, BF16.
bool read_gguf_tensor(const std::string& path, const std::string& tensor_name,
                      std::vector<float>& output, size_t* out_n) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GGUF", 4) != 0) { fclose(f); return false; }
    uint32_t ver; fread(&ver, 4, 1, f);
    uint64_t tc, kc; fread(&tc, 8, 1, f); fread(&kc, 8, 1, f);

    // Skip KV metadata
    for (uint64_t i = 0; i < kc; i++) {
        uint64_t klen; fread(&klen, 8, 1, f); fseek(f, klen, SEEK_CUR);
        uint32_t vt; fread(&vt, 4, 1, f);
        switch (vt) {
            case 0: fseek(f, 1, SEEK_CUR); break;   case 1: fseek(f, 1, SEEK_CUR); break;
            case 2: fseek(f, 2, SEEK_CUR); break;   case 3: fseek(f, 2, SEEK_CUR); break;
            case 4: case 5: case 6: fseek(f, 4, SEEK_CUR); break;
            case 7: fseek(f, 1, SEEK_CUR); break;  // bool
            case 8: { uint64_t sl; fread(&sl, 8, 1, f); fseek(f, sl, SEEK_CUR); break; } // string
            case 9: { // array
                uint32_t at; fread(&at, 4, 1, f); uint64_t an; fread(&an, 8, 1, f);
                if (at == 8) for (uint64_t j = 0; j < an; j++) { uint64_t sl; fread(&sl, 8, 1, f); fseek(f, sl, SEEK_CUR); }
                else { int sz = 4; if (at <= 7) { static const int s[]={1,1,2,2,4,4,4,1}; sz = s[at]; } else if (at >= 10 && at <= 12) sz = 8; fseek(f, an * sz, SEEK_CUR); }
                break;
            }
            case 10: fseek(f, 8, SEEK_CUR); break;
            case 11: fseek(f, 8, SEEK_CUR); break;
            case 12: fseek(f, 8, SEEK_CUR); break;
            default: break;
        }
    }

    // Read tensor descriptions — GGUF v3: name, n_dims, shape[], dtype, data_offset
    struct T { std::string name; uint32_t dtype; uint64_t n; uint64_t data_off; };
    std::vector<T> tensors;
    for (uint64_t i = 0; i < tc; i++) {
        uint64_t nl; fread(&nl, 8, 1, f); if (!f || nl > 512) break;
        T ti; ti.name.resize(nl); fread(&ti.name[0], 1, nl, f);
        uint32_t nd; fread(&nd, 4, 1, f);
        uint64_t ne = 1;
        for (int j = 0; j < nd; j++) { uint64_t s; fread(&s, 8, 1, f); ne *= s; }
        ti.n = ne;
        fread(&ti.dtype, 4, 1, f);  // dtype AFTER shape
        fread(&ti.data_off, 8, 1, f); // data_offset AFTER dtype
        tensors.push_back(ti);
    }

    // Compute data blob base offset (aligned end of tensor info section)
    long info_end = ftell(f);
    uint64_t data_base = (uint64_t)((info_end + 31) & ~31);
    // Add base to each tensor's data_offset
    for (auto& ti : tensors) ti.data_off += data_base;
    
    // Find target tensor
    T* target = nullptr;
    for (auto& ti : tensors) { if (ti.name == tensor_name) { target = &ti; break; } }
    if (!target) { fclose(f); return false; }

    // Read data using the file's own data_offset
    output.resize(target->n);
    fseek(f, target->data_off, SEEK_SET);
    
    // Helper: read fp16 from file. Handles subnormals correctly — the naive
    // "rebias the exponent bits" trick silently produces wrong values when
    // exp==0 (subnormal), which real quantization scale factors do hit in
    // practice (verified: a real Q6_K super-block scale in a downloaded
    // model was off by ~2.3x from the naive conversion until this fix).
    auto rd_f16 = [&]() -> float {
        uint16_t h; fread(&h, 2, 1, f);
        int sign = (h & 0x8000) ? -1 : 1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        float val;
        if (exp == 0) {
            val = std::ldexp((float)mant, -24); // subnormal (incl. zero): mant * 2^-24
        } else if (exp == 0x1F) {
            val = mant ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
        } else {
            val = std::ldexp((float)(mant | 0x400), exp - 25); // (1.mant) * 2^(exp-15-10)
        }
        return sign * val;
    };

    switch (target->dtype) {
        case 0: // F32
            fread(output.data(), 4, target->n, f); break;
        case 1: { // F16
            for (size_t i = 0; i < target->n; i++) output[i] = rd_f16();
            break;
        }
        case 2: { // Q4_0: 32 elems, 2 bytes scale + 16 bytes nibbles = 18 bytes/block
            // ggml's block_q4_0 de-interleaves nibbles into two halves, NOT
            // sequential (j, j+1, j+2...): output[j] is qs[j]'s low nibble,
            // output[j+16] is qs[j]'s high nibble (see dequantize_row_q4_0
            // in ggml-quants.c). Verified against the independent `gguf`
            // Python reference (bit-exact) for Q4_K/Q6_K — same verification
            // needed here caught this was using the wrong pattern.
            uint64_t nb = (target->n + 31) / 32;
            for (uint64_t b = 0; b < nb; b++) {
                float scale = rd_f16();
                uint8_t q[16]; fread(q, 1, 16, f);
                for (int j = 0; j < 16; j++) {
                    size_t i0 = b*32+j, i1 = b*32+j+16;
                    if (i0 < target->n) output[i0] = ((int)(q[j] & 0xF) - 8) * scale;
                    if (i1 < target->n) output[i1] = ((int)(q[j] >> 4) - 8) * scale;
                }
            }
            break;
        }
        case 6: { // Q5_0: 32 elems, 2 bytes scale + 4 bytes qh + 16 bytes qs = 22 bytes/block
            // Same de-interleaved low/high split as Q4_0, plus a 5th bit from
            // qh (read as one little-endian u32, bit j for the low half,
            // bit j+16 for the high half — see dequantize_row_q5_0).
            uint64_t nb = (target->n + 31) / 32;
            for (uint64_t b = 0; b < nb; b++) {
                float d = rd_f16();
                uint8_t qh_bytes[4]; fread(qh_bytes, 1, 4, f);
                uint32_t qh; memcpy(&qh, qh_bytes, 4);
                uint8_t qs[16]; fread(qs, 1, 16, f);
                for (int j = 0; j < 16; j++) {
                    uint8_t xh_0 = (uint8_t)(((qh >> j) << 4) & 0x10);
                    uint8_t xh_1 = (uint8_t)((qh >> (j + 12)) & 0x10);
                    int32_t x0 = (int32_t)((qs[j] & 0xF) | xh_0) - 16;
                    int32_t x1 = (int32_t)((qs[j] >> 4) | xh_1) - 16;
                    size_t i0 = b*32+j, i1 = b*32+j+16;
                    if (i0 < target->n) output[i0] = x0 * d;
                    if (i1 < target->n) output[i1] = x1 * d;
                }
            }
            break;
        }
        case 7: { // Q5_1: 32 elems, 2 bytes d + 2 bytes m + 4 bytes qh + 16 bytes qs = 24 bytes/block
            uint64_t nb = (target->n + 31) / 32;
            for (uint64_t b = 0; b < nb; b++) {
                float d = rd_f16(); float m = rd_f16();
                uint8_t qh_bytes[4]; fread(qh_bytes, 1, 4, f);
                uint32_t qh; memcpy(&qh, qh_bytes, 4);
                uint8_t qs[16]; fread(qs, 1, 16, f);
                for (int j = 0; j < 16; j++) {
                    uint8_t xh_0 = (uint8_t)(((qh >> j) << 4) & 0x10);
                    uint8_t xh_1 = (uint8_t)((qh >> (j + 12)) & 0x10);
                    int32_t x0 = (int32_t)((qs[j] & 0xF) | xh_0);
                    int32_t x1 = (int32_t)((qs[j] >> 4) | xh_1);
                    size_t i0 = b*32+j, i1 = b*32+j+16;
                    if (i0 < target->n) output[i0] = x0 * d + m;
                    if (i1 < target->n) output[i1] = x1 * d + m;
                }
            }
            break;
        }
        case 8: { // Q8_0: 32 elems, 2 bytes scale + 32 bytes int8 = 34 bytes/block
            uint64_t nb = (target->n + 31) / 32;
            for (uint64_t b = 0; b < nb; b++) {
                float scale = rd_f16();
                int8_t q[32]; fread(q, 1, 32, f);
                for (int j = 0; j < 32 && b*32+j < target->n; j++)
                    output[b*32+j] = (float)q[j] * scale;
            }
            break;
        }
        case 12: { // Q4_K: 256-elem superblock, 144 bytes: d(2) + dmin(2) + scales[12] + qs[128]
            // Reference layout: ggml's block_q4_K / get_scale_min_k4 (llama.cpp ggml-quants.c).
            // Sub-block (32-elem) scale/min are 6-bit values packed across 12 bytes; qs holds
            // 4-bit quants, low/high nibble = elements [0..32)/[32..64) of each 64-elem pair.
            uint64_t nb = (target->n + 255) / 256;
            for (uint64_t b = 0; b < nb; b++) {
                float d = rd_f16();
                float dmin = rd_f16();
                uint8_t scales[12]; fread(scales, 1, 12, f);
                uint8_t qs[128]; fread(qs, 1, 128, f);

                auto get_scale_min = [&](int j, uint8_t& sc, uint8_t& m) {
                    if (j < 4) {
                        sc = scales[j] & 63;
                        m = scales[j + 4] & 63;
                    } else {
                        sc = (uint8_t)((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
                        m = (uint8_t)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
                    }
                };

                size_t base = b * 256;
                int is = 0;
                const uint8_t* q = qs;
                for (int off = 0; off < 256; off += 64) {
                    uint8_t sc, m;
                    get_scale_min(is + 0, sc, m);
                    float d1 = d * sc, m1 = dmin * m;
                    get_scale_min(is + 1, sc, m);
                    float d2 = d * sc, m2 = dmin * m;
                    for (int l = 0; l < 32 && base + off + l < target->n; l++)
                        output[base + off + l] = d1 * (q[l] & 0xF) - m1;
                    for (int l = 0; l < 32 && base + off + 32 + l < target->n; l++)
                        output[base + off + 32 + l] = d2 * (q[l] >> 4) - m2;
                    q += 32;
                    is += 2;
                }
            }
            break;
        }
        case 14: { // Q6_K: 256-elem superblock, 210 bytes: ql[128] + qh[64] + scales[16] + d(2)
            // Reference layout: ggml's block_q6_K (llama.cpp ggml-quants.c). 6-bit quants:
            // low 4 bits from ql, high 2 bits from qh; 16 signed 8-bit per-16-elem scales.
            uint64_t nb = (target->n + 255) / 256;
            for (uint64_t b = 0; b < nb; b++) {
                uint8_t ql[128]; fread(ql, 1, 128, f);
                uint8_t qh[64]; fread(qh, 1, 64, f);
                int8_t scales[16]; fread(scales, 1, 16, f);
                float d = rd_f16();

                size_t base = b * 256;
                const uint8_t* qlp = ql;
                const uint8_t* qhp = qh;
                const int8_t* sc = scales;
                for (int n = 0; n < 256; n += 128) {
                    for (int l = 0; l < 32; l++) {
                        int is = l / 16;
                        int8_t q1 = (int8_t)((qlp[l] & 0xF) | (((qhp[l] >> 0) & 3) << 4)) - 32;
                        int8_t q2 = (int8_t)((qlp[l + 32] & 0xF) | (((qhp[l] >> 2) & 3) << 4)) - 32;
                        int8_t q3 = (int8_t)((qlp[l] >> 4) | (((qhp[l] >> 4) & 3) << 4)) - 32;
                        int8_t q4 = (int8_t)((qlp[l + 32] >> 4) | (((qhp[l] >> 6) & 3) << 4)) - 32;
                        size_t i1 = base + n + l, i2 = base + n + l + 32, i3 = base + n + l + 64, i4 = base + n + l + 96;
                        if (i1 < target->n) output[i1] = d * sc[is + 0] * q1;
                        if (i2 < target->n) output[i2] = d * sc[is + 2] * q2;
                        if (i3 < target->n) output[i3] = d * sc[is + 4] * q3;
                        if (i4 < target->n) output[i4] = d * sc[is + 6] * q4;
                    }
                    qlp += 64;
                    qhp += 32;
                    sc += 8;
                }
            }
            break;
        }
        case 10: { // Q2_K: 256-elem superblock, 84 bytes: scales[16] + qs[64] + d(2) + dmin(2)
            // Reference layout: ggml's block_q2_K. 2-bit quants; each 16-elem
            // sub-block has a 4-bit scale + 4-bit min packed into one scales[] byte.
            uint64_t nb = (target->n + 255) / 256;
            for (uint64_t b = 0; b < nb; b++) {
                uint8_t scales[16]; fread(scales, 1, 16, f);
                uint8_t qs[64]; fread(qs, 1, 64, f);
                float d = rd_f16();
                float dmin = rd_f16();

                size_t base = b * 256;
                size_t pos = 0;
                int is = 0;
                const uint8_t* q = qs;
                for (int n = 0; n < 256; n += 128) {
                    int shift = 0;
                    for (int j = 0; j < 4; j++) {
                        uint8_t sc = scales[is++];
                        float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);
                        for (int l = 0; l < 16; l++, pos++)
                            if (base + pos < target->n) output[base + pos] = dl * ((q[l] >> shift) & 3) - ml;
                        sc = scales[is++];
                        dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
                        for (int l = 0; l < 16; l++, pos++)
                            if (base + pos < target->n) output[base + pos] = dl * ((q[l + 16] >> shift) & 3) - ml;
                        shift += 2;
                    }
                    q += 32;
                }
            }
            break;
        }
        case 11: { // Q3_K: 256-elem superblock, 110 bytes: hmask[32] + qs[64] + scales[12] + d(2)
            // Reference layout: ggml's block_q3_K. 3-bit quants (2 low bits from qs,
            // 1 high bit from hmask); scales are 6-bit values packed across 12 bytes
            // via 32-bit-word manipulation (see ggml-quants.c dequantize_row_q3_K).
            uint64_t nb = (target->n + 255) / 256;
            for (uint64_t b = 0; b < nb; b++) {
                uint8_t hmask[32]; fread(hmask, 1, 32, f);
                uint8_t qs[64]; fread(qs, 1, 64, f);
                uint8_t scales_raw[12]; fread(scales_raw, 1, 12, f);
                float d_all = rd_f16();

                const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
                uint32_t aux[4] = {0, 0, 0, 0};
                memcpy(aux, scales_raw, 12);
                uint32_t tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
                int8_t scales[16];
                memcpy(scales, aux, 16);
                for (int j = 0; j < 16; j++) scales[j] -= 32;

                size_t base = b * 256;
                size_t pos = 0;
                int is = 0;
                const uint8_t* q = qs;
                const uint8_t* hm = hmask;
                uint8_t m = 1;
                for (int n = 0; n < 256; n += 128) {
                    int shift = 0;
                    for (int j = 0; j < 4; j++) {
                        float dl = d_all * scales[is++];
                        for (int l = 0; l < 16; l++, pos++)
                            if (base + pos < target->n)
                                output[base + pos] = dl * (((int8_t)((q[l] >> shift) & 3)) - ((hm[l] & m) ? 0 : 4));
                        dl = d_all * scales[is++];
                        for (int l = 0; l < 16; l++, pos++)
                            if (base + pos < target->n)
                                output[base + pos] = dl * (((int8_t)((q[l + 16] >> shift) & 3)) - ((hm[l + 16] & m) ? 0 : 4));
                        shift += 2;
                        m <<= 1;
                    }
                    q += 32;
                }
            }
            break;
        }
        case 13: { // Q5_K: 256-elem superblock, 176 bytes: d(2) + dmin(2) + scales[12] + qh[32] + qs[128]
            // Reference layout: ggml's block_q5_K. Same 6-bit scale/min packing as
            // Q4_K (get_scale_min_k4); 5-bit quants (4 low bits from qs, 1 high bit from qh).
            uint64_t nb = (target->n + 255) / 256;
            for (uint64_t b = 0; b < nb; b++) {
                float d = rd_f16();
                float dmin = rd_f16();
                uint8_t scales[12]; fread(scales, 1, 12, f);
                uint8_t qh[32]; fread(qh, 1, 32, f);
                uint8_t qs[128]; fread(qs, 1, 128, f);

                auto get_scale_min = [&](int j, uint8_t& sc, uint8_t& m) {
                    if (j < 4) {
                        sc = scales[j] & 63;
                        m = scales[j + 4] & 63;
                    } else {
                        sc = (uint8_t)((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
                        m = (uint8_t)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
                    }
                };

                size_t base = b * 256;
                size_t pos = 0;
                int is = 0;
                const uint8_t* ql = qs;
                uint8_t u1 = 1, u2 = 2;
                for (int off = 0; off < 256; off += 64) {
                    uint8_t sc, m;
                    get_scale_min(is + 0, sc, m);
                    float d1 = d * sc, m1 = dmin * m;
                    get_scale_min(is + 1, sc, m);
                    float d2 = d * sc, m2 = dmin * m;
                    for (int l = 0; l < 32; l++, pos++)
                        if (base + pos < target->n)
                            output[base + pos] = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
                    for (int l = 0; l < 32; l++, pos++)
                        if (base + pos < target->n)
                            output[base + pos] = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
                    ql += 32;
                    is += 2;
                    u1 <<= 2; u2 <<= 2;
                }
            }
            break;
        }
        case 15: { // Q8_K: 256-elem superblock, 292 bytes: d(4, f32) + qs[256] int8 + bsums[16] int16
            // Reference layout: ggml's block_q8_K. Note d is a full f32 here (not f16
            // like every other K-quant) — bsums are precomputed dot-product sums, not
            // needed for dequantization.
            uint64_t nb = (target->n + 255) / 256;
            for (uint64_t b = 0; b < nb; b++) {
                float d; fread(&d, 4, 1, f);
                int8_t qs[256]; fread(qs, 1, 256, f);
                fseek(f, 32, SEEK_CUR); // skip bsums[16] (int16)
                size_t base = b * 256;
                for (int l = 0; l < 256 && base + l < target->n; l++)
                    output[base + l] = d * (float)qs[l];
            }
            break;
        }
        case 30: { // BF16: 1 elem, 2 bytes/block — top 16 bits of an f32 (1 sign + 8 exp + 7 mantissa)
            for (size_t i = 0; i < target->n; i++) {
                uint16_t h; fread(&h, 2, 1, f);
                uint32_t bits = (uint32_t)h << 16;
                float v; memcpy(&v, &bits, 4);
                output[i] = v;
            }
            break;
        }
        default:
            fprintf(stderr, "[gguf] unhandled dtype %u for %s\n", target->dtype, tensor_name.c_str());
            fclose(f); return false;
    }    if (out_n) *out_n = target->n;
    fclose(f); return true;

}