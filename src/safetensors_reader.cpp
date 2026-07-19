// safetensors_reader.cpp — method bodies moved out of include/safetensors_reader.h
// to avoid recompilation cascades (issue #375).
#include "safetensors_reader.h"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace safetensors_detail {

// Extract the first string value for a top-level JSON key: "key":"value" or
// "key": ["value", ...] (returns the first array element). Not a general
// JSON parser — sufficient for the flat HF config.json fields used here.
bool json_find_string(const std::string& text, const std::string& key, std::string& out) {
    auto pos = text.find("\"" + key + "\"");
    if (pos == std::string::npos) return false;
    pos = text.find(':', pos);
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '[')) pos++;
    if (pos >= text.size() || text[pos] != '"') return false;
    pos++;
    auto end = text.find('"', pos);
    if (end == std::string::npos) return false;
    out = text.substr(pos, end - pos);
    return true;
}

bool json_find_int(const std::string& text, const std::string& key, int& out) {
    auto pos = text.find("\"" + key + "\"");
    if (pos == std::string::npos) return false;
    pos = text.find(':', pos);
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n')) pos++;
    if (pos >= text.size() || !(isdigit((unsigned char)text[pos]) || text[pos] == '-')) return false;
    out = atoi(text.c_str() + pos);
    return true;
}

bool json_find_float(const std::string& text, const std::string& key, float& out) {
    auto pos = text.find("\"" + key + "\"");
    if (pos == std::string::npos) return false;
    pos = text.find(':', pos);
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n')) pos++;
    if (pos >= text.size() || !(isdigit((unsigned char)text[pos]) || text[pos] == '-')) return false;
    out = (float)atof(text.c_str() + pos);
    return true;
}

std::string read_small_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return "";
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz <= 0 || sz > 8L * 1024 * 1024) { fclose(f); return ""; } // config.json is never this large
    fseek(f, 0, SEEK_SET);
    std::string s(sz, '\0');
    size_t got = fread(&s[0], 1, sz, f);
    fclose(f);
    s.resize(got);
    return s;
}

} // namespace safetensors_detail

// ── read_safetensors_metadata ───────────────────────────────────────────────

bool read_safetensors_metadata(const std::string& path, ModelConfig& cfg) {
    Q4nxReader r;
    if (!r.open(path)) return false;
    if (r.size < 16) { r.close(); return false; }

    uint64_t hdr_len = 0;
    memcpy(&hdr_len, r.data, 8);
    if (hdr_len == 0 || 8 + hdr_len > r.size || hdr_len > (r.size > (16u << 20) ? (16u << 20) : r.size)) {
        r.close();
        return false; // not a safetensors-style container
    }
    std::string header(r.data + 8, (size_t)hdr_len);
    r.close();

    // Sane defaults (same shape as read_gguf_metadata's) — overwritten below
    // by config.json and/or tensor-shape inference where available.
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
    cfg.model_path = path;
    cfg.format = ModelFormat::SAFETENSORS;

    auto slash = path.find_last_of('/');
    auto dot = path.find_last_of('.');
    cfg.model_name = path.substr(slash + 1, (dot == std::string::npos ? path.size() : dot) - slash - 1);
    std::string dir = slash == std::string::npos ? "." : path.substr(0, slash);

    // Prefer the HuggingFace-standard sibling config.json — it's the
    // authoritative source real tooling (transformers/vLLM) relies on, since
    // safetensors itself carries no architecture/dimension fields.
    bool got_config = false;
    std::string config_text = safetensors_detail::read_small_file(dir + "/config.json");
    if (!config_text.empty()) {
        using namespace safetensors_detail;
        std::string arch;
        if (json_find_string(config_text, "architectures", arch)) {
            // HF class names are like "Qwen2ForCausalLM" / "LlamaForCausalLM" —
            // strip the trailing suffix to match GGUF's lowercase family tag
            // convention ("qwen2", "llama", version digit kept).
            std::string low = arch;
            for (auto& c : low) c = (char)tolower((unsigned char)c);
            for (const char* suf : {"forcausallm", "lmheadmodel", "model"}) {
                size_t sl = strlen(suf);
                if (low.size() > sl && low.compare(low.size() - sl, sl, suf) == 0) {
                    low = low.substr(0, low.size() - sl);
                    break;
                }
            }
            cfg.architecture = low;
        }
        int iv;
        if (json_find_int(config_text, "hidden_size", iv)) cfg.hidden = cfg.hidden_size = iv;
        if (json_find_int(config_text, "num_hidden_layers", iv)) cfg.n_layers = cfg.num_layers = iv;
        if (json_find_int(config_text, "num_attention_heads", iv)) cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = iv;
        if (json_find_int(config_text, "num_key_value_heads", iv)) cfg.n_kv_heads = cfg.num_kv_heads = iv;
        else cfg.n_kv_heads = cfg.num_kv_heads = cfg.n_heads;
        if (json_find_int(config_text, "intermediate_size", iv)) cfg.n_ff = cfg.intermediate_size = iv;
        if (json_find_int(config_text, "vocab_size", iv)) cfg.vocab = cfg.vocab_size = iv;
        if (json_find_int(config_text, "max_position_embeddings", iv)) cfg.max_seq_len = iv;
        float fv;
        if (json_find_float(config_text, "rope_theta", fv)) cfg.rope_theta = fv;
        if (json_find_float(config_text, "rms_norm_eps", fv)) cfg.rms_norm_eps = fv;
        // Prefer an explicit head_dim key when present — some architectures
        // (e.g. Qwen3: hidden_size=1024, num_attention_heads=16, but
        // head_dim=128) are NOT hidden_size/num_attention_heads.
        if (json_find_int(config_text, "head_dim", iv)) cfg.head_dim = iv;
        else if (cfg.n_heads > 0) cfg.head_dim = cfg.hidden / cfg.n_heads;
        got_config = true;
    }

    // Fall back to (or supplement with) tensor-name/shape inference straight
    // from the safetensors header — same technique as q4nx_reader.h — for
    // whatever config.json didn't cover, or when it's absent entirely (a bare
    // weights dump with no HF metadata alongside).
    auto find_shape_after = [&](const std::string& needle, int& a, int& b) -> bool {
        auto pos = header.find(needle);
        if (pos == std::string::npos) return false;
        auto shape_pos = header.find("\"shape\":[", pos);
        if (shape_pos == std::string::npos) return false;
        shape_pos += strlen("\"shape\":[");
        return sscanf(header.c_str() + shape_pos, "%d,%d", &a, &b) == 2;
    };
    if (!got_config || cfg.vocab <= 0 || cfg.hidden <= 0) {
        int vocab = 0, hidden = 0;
        if (find_shape_after("\"model.embed_tokens.weight\"", vocab, hidden) ||
            find_shape_after("\"transformer.wte.weight\"", vocab, hidden) ||
            find_shape_after("\"lm_head.weight\"", vocab, hidden)) {
            cfg.vocab = cfg.vocab_size = vocab;
            cfg.hidden = cfg.hidden_size = hidden;
        }
    }
    if (!got_config) {
        // Tensor naming isn't fully standardized across HF checkpoint families
        // — most use "model.layers.N." but some (e.g. custom draft/EAGLE
        // architectures) use bare "layers.N." with no "model." prefix.
        int max_layer = -1;
        for (const char* marker : {"\"model.layers.", "\"layers."}) {
            size_t mlen = strlen(marker);
            size_t search = 0;
            while ((search = header.find(marker, search)) != std::string::npos) {
                int n = atoi(header.c_str() + search + mlen);
                if (n > max_layer) max_layer = n;
                search += mlen;
            }
            if (max_layer >= 0) break;
        }
        if (max_layer >= 0) cfg.n_layers = cfg.num_layers = max_layer + 1;
    }

    // Architecture last-resort: filename, same convention as q4nx_reader.h.
    if (cfg.architecture.empty()) {
        auto sep = cfg.model_name.find_first_of("-_");
        cfg.architecture = sep == std::string::npos ? cfg.model_name : cfg.model_name.substr(0, sep);
    }

    // Quantization: dtype of the first tensor found in the safetensors header
    // itself (ground truth for the on-disk data, not config.json's
    // torch_dtype which can describe the compute dtype instead).
    auto dtype_pos = header.find("\"dtype\":\"");
    if (dtype_pos != std::string::npos) {
        dtype_pos += strlen("\"dtype\":\"");
        auto end = header.find('"', dtype_pos);
        if (end != std::string::npos) cfg.quantization = header.substr(dtype_pos, end - dtype_pos);
    }

    return true;
}
