// model_discovery.cpp — Scan weights directory for model files (GGUF, H1B, safetensors)
// and read their headers to populate ModelConfig without loading full weights.

#include "model_discovery.h"
#include "gguf_reader.h"
#include "q4nx_reader.h"
#include "safetensors_reader.h"
#include <cstdio>
#include <cstring>
#ifndef _WIN32
#include <dirent.h>
#include <sys/stat.h>
#endif
#ifdef _WIN32
#include <io.h>
#include <windows.h>
// Windows equivalent for struct dirent iteration
struct WIN32_FIND_DATAA;
#endif

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
    if (r.get_u32("general.file_type", ft)) {
        cfg.quantization = ggml_ftype_name(ft);
    } else {
        // No explicit file_type — scan tensor dtypes to detect binary/ternary formats
        bool has_q1_0 = false, has_tq2_0 = false, has_tq1_0 = false;
        bool has_iq1_s = false, has_iq1_m = false;
        for (const auto& tn : r.tensor_names()) {
            auto* ti = r.tensor_info(tn);
            if (!ti) continue;
            if (ti->dtype == GGUF_DTYPE_Q1_0_G128)   has_q1_0 = true;
            if (ti->dtype == GGUF_DTYPE_TQ2_0_G128)  has_tq2_0 = true;
            if (ti->dtype == GGUF_DTYPE_TQ1_0_LLAMA)  has_tq1_0 = true;
            if (ti->dtype == GGUF_DTYPE_TQ2_0_LLAMA)  has_tq2_0 = true;
            if (ti->dtype == GGUF_DTYPE_IQ1_S)        has_iq1_s = true;
            if (ti->dtype == GGUF_DTYPE_IQ1_M)        has_iq1_m = true;
        }
        if (has_q1_0)      cfg.quantization = "Q1_0 (binary 1-bit)";
        else if (has_tq1_0) cfg.quantization = "TQ1_0 (ternary 1.69bpw)";
        else if (has_tq2_0) cfg.quantization = "TQ2_0 (ternary 2.06bpw)";
        else if (has_iq1_s) cfg.quantization = "IQ1_S (1.5bpw)";
        else if (has_iq1_m) cfg.quantization = "IQ1_M (1.75bpw)";
    }

    std::vector<std::string> tokens;
    if (r.get_string_array("tokenizer.ggml.tokens", tokens)) cfg.vocab = cfg.vocab_size = (int)tokens.size();

    bool explicit_head_dim = false;
    for (const auto& key : r.kv_keys()) {
        uint32_t u32v; float f32v;
        if (ends_with(key, ".ssm.state_size")) {
            // d_state — not stored in ModelConfig currently;
            // architecture-specific loaders read it directly from the GGUF file.
            // Don't reuse head_dim for this because it would corrupt attention
            // head_dim for hybrid Mamba2+attention models (Zamba2) when the
            // KV key ordering in GGUF puts ssm.state_size after attention.key_length.
        } else if (ends_with(key, ".ssm.conv_kernel")) {
            // d_conv — not stored in ModelConfig currently
        } else if (ends_with(key, ".attention.head_count")) {
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

// ── Read .1bp (oneBP) metadata header ────────────────────────────────────
static bool read_onebp_metadata(const std::string& path, ModelConfig& cfg) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    // Read the 256-byte OnebpHeader
    uint8_t hdr_buf[256];
    if (fread(hdr_buf, 1, 256, f) != 256) { fclose(f); return false; }
    fclose(f);

    // Validate magic: "1BP\0" = 0x00504231 (little-endian)
    uint32_t magic;
    memcpy(&magic, hdr_buf, 4);
    if (magic != 0x00504231) return false;

    // Read version
    uint32_t version;
    memcpy(&version, hdr_buf + 4, 4);
    if (version < 1 || version > 3) return false;

    // Extract fields from header at known offsets (OnebpHeader layout)
    int32_t hidden_size;        memcpy(&hidden_size, hdr_buf + 16, 4);
    int32_t num_layers;         memcpy(&num_layers, hdr_buf + 20, 4);
    int32_t num_heads;          memcpy(&num_heads, hdr_buf + 24, 4);
    int32_t num_kv_heads;       memcpy(&num_kv_heads, hdr_buf + 28, 4);
    int32_t head_dim;           memcpy(&head_dim, hdr_buf + 32, 4);
    int32_t intermediate_size;  memcpy(&intermediate_size, hdr_buf + 36, 4);
    int32_t vocab_size;         memcpy(&vocab_size, hdr_buf + 40, 4);
    int32_t max_seq_len;        memcpy(&max_seq_len, hdr_buf + 44, 4);
    uint32_t tensor_count;      memcpy(&tensor_count, hdr_buf + 84, 4);
    uint32_t num_experts;       memcpy(&num_experts, hdr_buf + 88, 4);
    uint32_t n_expert_used;     memcpy(&n_expert_used, hdr_buf + 92, 4);
    uint32_t arch_raw;          memcpy(&arch_raw, hdr_buf + 8, 4);
    uint32_t quant_raw;         memcpy(&quant_raw, hdr_buf + 12, 4);

    if (hidden_size <= 0 || num_layers <= 0 || vocab_size <= 0) return false;

    cfg.hidden = cfg.hidden_size = hidden_size;
    cfg.n_layers = cfg.num_layers = num_layers;
    cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = num_heads;
    cfg.n_kv_heads = cfg.num_kv_heads = num_kv_heads ? num_kv_heads : num_heads;
    cfg.head_dim = head_dim ? head_dim : (num_heads > 0 ? hidden_size / num_heads : 128);
    cfg.n_ff = cfg.intermediate_size = intermediate_size;
    cfg.vocab = cfg.vocab_size = vocab_size;
    cfg.max_seq_len = max_seq_len ? max_seq_len : 2048;
    cfg.n_experts = cfg.num_experts = num_experts;
    cfg.num_experts_top = n_expert_used;
    cfg.model_path = path;
    cfg.format = ModelFormat::ONEBP;

    // Read model_tag from offset 192 (64 chars)
    char tag[65];
    memcpy(tag, hdr_buf + 192, 64);
    tag[64] = '\0';
    cfg.model_name = tag;
    if (cfg.model_name.empty()) {
        auto slash = path.find_last_of('/');
        auto dot = path.find_last_of('.');
        cfg.model_name = path.substr(slash + 1, dot - slash - 1);
    }

    // Architecture string from enum
    switch (arch_raw) {
        case 0:  cfg.architecture = "qwen2"; break;
        case 1:  cfg.architecture = "llama"; break;
        case 2:  cfg.architecture = "mistral"; break;
        case 3:  cfg.architecture = "phi3"; break;
        case 4:  cfg.architecture = "gemma"; break;
        case 5:  cfg.architecture = "falcon"; break;
        case 6:  cfg.architecture = "starcoder"; break;
        case 7:  cfg.architecture = "deepseek2"; break;
        case 8:  cfg.architecture = "qwen2moe"; break;
        case 9:  cfg.architecture = "qwen3moe"; break;
        case 10: cfg.architecture = "qwen35"; break;
        case 11: cfg.architecture = "qwen35moe"; break;
        case 12: cfg.architecture = "zamba"; break;
        case 13: cfg.architecture = "zamba2"; break;
        case 14: cfg.architecture = "mamba"; break;
        case 15: cfg.architecture = "gemma3"; break;
        case 16: cfg.architecture = "gemma4"; break;
        case 17: cfg.architecture = "olmo"; break;
        case 18: cfg.architecture = "laguna"; break;
        case 19: cfg.architecture = "zaya1"; break;
        default: cfg.architecture = "unknown(" + std::to_string(arch_raw) + ")"; break;
    }

    // Quantization tag from enum
    switch (quant_raw) {
        case 0:  cfg.quantization = "BF16"; break;
        case 1:  cfg.quantization = "Q1_0 (binary 1-bit)"; break;
        case 2:  cfg.quantization = "TQ2_0 (ternary 2.06bpw)"; break;
        case 3:  cfg.quantization = "TQ1_0 (ternary 1.69bpw)"; break;
        case 4:  cfg.quantization = "IQ1_S (1.5bpw)"; break;
        case 5:  cfg.quantization = "IQ1_M (1.75bpw)"; break;
        case 6:  cfg.quantization = "FP16_Sherry"; break;
        case 7:  cfg.quantization = "I8_Sherry"; break;
        case 8:  cfg.quantization = "Q4_0"; break;
        default: cfg.quantization = "unknown(" + std::to_string(quant_raw) + ")"; break;
    }

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
        if (ext != ".gguf" && ext != ".h1b" && ext != ".safetensors" && ext != ".bin" && ext != ".q4nx" && ext != ".1bp") continue;

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
            // .bin weight files: only discover once per directory.
            // Check for the sentinel weight file that marks a Zaya model directory.
            // Skip all other .bin files to avoid duplicate model entries.
            static std::string last_bin_dir;
            if (name != "model_embed_tokens_weight.bin") continue;
            if (dir == last_bin_dir) continue;  // already discovered this directory
            last_bin_dir = dir;

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
        else if (ext == ".1bp") ok = read_onebp_metadata(full, cfg);
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
