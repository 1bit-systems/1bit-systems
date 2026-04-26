// model.cpp — ArtifactPaths discovery + GenAiConfig JSON parsing.
//
// Filesystem + nlohmann/json only. Builds without ORT.

#include "onebit/onnx/model.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <system_error>

namespace onebit::onnx {

std::expected<ArtifactPaths, Error>
ArtifactPaths::discover(const std::filesystem::path& root) {
    namespace fs = std::filesystem;

    std::error_code ec;
    if (!fs::exists(root, ec) || !fs::is_directory(root, ec)) {
        return std::unexpected(Error{ErrorKind::NotAnArtifactDir, root,
                                     "root does not exist or is not a directory"});
    }

    ArtifactPaths p;
    p.root         = root;
    p.model        = root / "model.onnx";
    p.weights      = root / "model.onnx.data";
    p.genai_config = root / "genai_config.json";
    p.tokenizer    = root / "tokenizer.json";

    if (!fs::exists(p.model, ec)) {
        return std::unexpected(Error{ErrorKind::NotAnArtifactDir, root,
                                     "no model.onnx in root"});
    }

    for (const auto& f : {&p.weights, &p.genai_config, &p.tokenizer}) {
        if (!fs::exists(*f, ec)) {
            return std::unexpected(Error{ErrorKind::MissingArtifact, *f, {}});
        }
    }

    return p;
}

std::expected<GenAiConfig, Error>
GenAiConfig::load(const std::filesystem::path& path) {
    std::ifstream in{path, std::ios::binary};
    if (!in) {
        return std::unexpected(Error{ErrorKind::Io, path, "could not open genai_config.json"});
    }
    std::stringstream buf;
    buf << in.rdbuf();
    const auto text = buf.str();

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(text);
    } catch (const nlohmann::json::parse_error& e) {
        return std::unexpected(Error{ErrorKind::InvalidGenAiConfig, path, e.what()});
    }

    GenAiConfig cfg;
    try {

    // model section is required.
    auto m_it = j.find("model");
    if (m_it == j.end() || !m_it->is_object()) {
        return std::unexpected(Error{ErrorKind::InvalidGenAiConfig, path,
                                     "missing 'model' object"});
    }
    const auto& m = *m_it;

    auto get_or = [&](const char* key, auto def) {
        auto it = m.find(key);
        using T = decltype(def);
        if (it == m.end()) return def;
        return it->template get<T>();
    };

    cfg.model.bos_token_id  = get_or("bos_token_id", std::uint32_t{0});
    cfg.model.eos_token_id  = get_or("eos_token_id", std::uint32_t{0});
    cfg.model.context_length = get_or("context_length", std::size_t{2048});
    cfg.model.vocab_size    = get_or("vocab_size", std::uint32_t{0});
    cfg.model.arch          = get_or("type", std::string{});

    // decoder is required for any real session use; absent in some
    // pure-encoder graphs. We tolerate absence for config-only probes.
    if (auto d_it = m.find("decoder"); d_it != m.end() && d_it->is_object()) {
        const auto& d = *d_it;
        auto dget = [&](const char* k, std::size_t def) {
            auto it = d.find(k);
            return it == d.end() ? def : it->template get<std::size_t>();
        };
        cfg.model.decoder.head_size           = dget("head_size", 0);
        cfg.model.decoder.hidden_size         = dget("hidden_size", 0);
        cfg.model.decoder.num_attention_heads = dget("num_attention_heads", 0);
        cfg.model.decoder.num_key_value_heads = dget("num_key_value_heads",
                                                      cfg.model.decoder.num_attention_heads);
        cfg.model.decoder.num_hidden_layers   = dget("num_hidden_layers", 0);
    }

    // search section is optional.
    if (auto s_it = j.find("search"); s_it != j.end() && s_it->is_object()) {
        const auto& s = *s_it;
        if (auto it = s.find("do_sample"); it != s.end()) cfg.search.do_sample = it->get<bool>();
        if (auto it = s.find("temperature"); it != s.end()) cfg.search.temperature = it->get<float>();
        if (auto it = s.find("top_k"); it != s.end())       cfg.search.top_k = it->get<std::uint32_t>();
    }
    } catch (const nlohmann::json::type_error& e) {
        return std::unexpected(Error{ErrorKind::InvalidGenAiConfig, path, e.what()});
    } catch (const nlohmann::json::out_of_range& e) {
        return std::unexpected(Error{ErrorKind::InvalidGenAiConfig, path, e.what()});
    }

    return cfg;
}

} // namespace onebit::onnx
