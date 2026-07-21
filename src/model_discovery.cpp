// model_discovery.cpp — Scan weights directory for model files (GGUF, H1B, safetensors)
// and read their headers to populate ModelConfig without loading full weights.

#include "model_discovery.h"
#include "gguf_reader.h"
#include "q4nx_reader.h"
#include "safetensors_reader.h"
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>

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

static bool ends_with(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool read_h1b_metadata(const std::string& path, ModelConfig& cfg) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t h1b_magic;
    if (fread(&h1b_magic, 4, 1, f) != 1 || h1b_magic != 0x48314248) { fclose(f); return false; }
    uint32_t version, hs, is, n_layers, n_heads, n_kv, max_seq;
    fseek(f, 8, SEEK_SET);
    if (fread(&version, 4, 1, f) != 1 || fread(&hs, 4, 1, f) != 1 ||
        fread(&is, 4, 1, f) != 1 || fread(&n_layers, 4, 1, f) != 1 ||
        fread(&n_heads, 4, 1, f) != 1 || fread(&n_kv, 4, 1, f) != 1 ||
        fread(&max_seq, 4, 1, f) != 1) {
        fclose(f);
        return false;
    }
    cfg.hidden = cfg.hidden_size = hs;
    cfg.n_ff = cfg.intermediate_size = is;
    cfg.n_layers = cfg.num_layers = n_layers;
    cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = n_heads;
    cfg.n_kv_heads = cfg.num_kv_heads = n_kv ? n_kv : n_heads;
    cfg.head_dim = (n_heads > 0) ? (hs / n_heads) : 128;
    cfg.max_seq_len = max_seq ? max_seq : 2048;
    cfg.model_path = path;
    cfg.format = ModelFormat::H1B;
    auto slash = path.find_last_of('/');
    auto dot = path.find_last_of('.');
    cfg.model_name = path.substr(slash + 1, dot - slash - 1);
    fclose(f);
    return true;
}

static bool read_gguf_metadata(const std::string& path, ModelConfig& cfg) {
    GgufReader r;
    if (!r.open(path)) return read_h1b_metadata(path, cfg);

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

    cfg.architecture = r.architecture();
    cfg.arch = rcpp_arch_from_string(cfg.architecture.c_str());

    std::string name;
    if (r.get_string("general.name", name)) cfg.model_name = name;

    uint32_t ft;
    if (r.get_u32("general.file_type", ft)) cfg.quantization = ggml_ftype_name(ft);

    std::vector<std::string> tokens;
    if (r.get_string_array("tokenizer.ggml.tokens", tokens)) cfg.vocab = cfg.vocab_size = (int)tokens.size();

    bool explicit_head_dim = false;
    for (const auto& key : r.kv_keys()) {
        uint32_t u32v; float f32v;
        if (ends_with(key, ".attention.head_count")) {
            if (r.get_u32(key, u32v)) cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = u32v;
        } else if (ends_with(key, ".attention.head_count_kv")) {
            if (r.get_u32(key, u32v)) cfg.n_kv_heads = cfg.num_kv_heads = u32v;
        } else if (ends_with(key, ".block_count")) {
            if (r.get_u32(key, u32v)) cfg.n_layers = cfg.num_layers = u32v;
        } else if (ends_with(key, ".feed_forward_length")) {
            if (r.get_u32(key, u32v)) cfg.n_ff = cfg.intermediate_size = u32v;
        } else if (ends_with(key, ".embedding_length")) {
            if (r.get_u32(key, u32v)) cfg.hidden = cfg.hidden_size = u32v;
        } else if (ends_with(key, ".rope.freq_base")) {
            if (r.get_f32(key, f32v)) cfg.rope_theta = f32v;
        } else if (ends_with(key, ".expert_count")) {
            if (r.get_u32(key, u32v)) cfg.n_experts = cfg.num_experts = u32v;
        } else if (ends_with(key, ".expert_used_count")) {
            if (r.get_u32(key, u32v)) cfg.num_experts_top = u32v;
        } else if (ends_with(key, ".attention.key_length")) {
            // Authoritative head_dim when present — some architectures
            // (e.g. Qwen3: hidden=1024, heads=16, but head_dim=128) are
            // NOT hidden/n_heads. Falls back to that derivation below
            // only when this key is absent.
            if (r.get_u32(key, u32v)) { cfg.head_dim = u32v; explicit_head_dim = true; }
        }
    }

    // Derive head_dim from hidden / heads — unless the file gave an explicit
    // attention.key_length (authoritative; not always equal to hidden/heads).
    if (!explicit_head_dim) cfg.head_dim = (cfg.n_heads > 0) ? (cfg.hidden / cfg.n_heads) : 128;
    // Default KV heads to full if not set
    if (cfg.n_kv_heads == 0) cfg.n_kv_heads = cfg.n_heads;
    cfg.num_kv_heads = cfg.n_kv_heads;

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
        else if (ext == ".h1b") ok = read_h1b_metadata(full, cfg);
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
int read_gguf_vocab(const std::string& path) {
    GgufReader r;
    if (!r.open(path)) return 0;
    const GgufTensorInfo* ti = r.tensor_info("token_embd.weight");
    if (!ti || ti->shape.empty()) return 0;
    return (int)ti->shape.back();
}

bool read_gguf_header(const std::string& path, ModelConfig& cfg) {
    return read_gguf_metadata(path, cfg);
}

// ── Read a GGUF tensor's data ──────────────────────────────────────────────
bool read_gguf_tensor(const std::string& path, const std::string& tensor_name,
                      std::vector<float>& output, size_t* out_n) {
    GgufReader r;
    if (!r.open(path)) return false;
    return r.get_tensor_f32(tensor_name, output, out_n);
}
