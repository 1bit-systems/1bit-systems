// onebit::onnx::ArtifactPaths + GenAiConfig — OGA Model Builder output
// directory parsing.
//
// An OGA output directory has a fixed shape (model.onnx, model.onnx.data,
// genai_config.json, tokenizer.json[, tokenizer_config.json]). This
// header pins the in-memory representation we use across the rest of
// the crate.
//
// Filesystem + JSON only — no ORT dependency. Buildable on every host.

#pragma once

#include <expected>
#include <filesystem>
#include <string>

#include "onebit/onnx/error.hpp"

namespace onebit::onnx {

// Resolved paths for every file in an OGA output directory. Constructed
// via `ArtifactPaths::discover`; fields are public for log lines.
struct ArtifactPaths {
    std::filesystem::path root;
    std::filesystem::path model;        // model.onnx
    std::filesystem::path weights;      // model.onnx.data
    std::filesystem::path genai_config; // genai_config.json
    std::filesystem::path tokenizer;    // tokenizer.json

    // Probe `root` for the four files above. Returns
    // ErrorKind::NotAnArtifactDir if model.onnx is missing,
    // ErrorKind::MissingArtifact for any other absent file.
    [[nodiscard]] static std::expected<ArtifactPaths, Error>
    discover(const std::filesystem::path& root);
};

// Subset of genai_config.json we actually consume. Anything we don't
// touch is intentionally omitted so OGA exporter bumps don't break us
// unless the core fields move.
struct GenAiDecoder {
    std::size_t head_size{};
    std::size_t hidden_size{};
    std::size_t num_attention_heads{};
    std::size_t num_key_value_heads{};
    std::size_t num_hidden_layers{};
};

struct GenAiModel {
    std::uint32_t bos_token_id{};
    std::uint32_t eos_token_id{};
    std::size_t   context_length{};
    std::uint32_t vocab_size{};
    std::string   arch;             // "llama" for TriLM 3.9B
    GenAiDecoder  decoder{};
};

struct GenAiSearch {
    bool          do_sample{false};
    float         temperature{1.0f};
    std::uint32_t top_k{50};
};

struct GenAiConfig {
    GenAiModel  model;
    GenAiSearch search;

    // Parse genai_config.json at `path`.
    [[nodiscard]] static std::expected<GenAiConfig, Error>
    load(const std::filesystem::path& path);
};

} // namespace onebit::onnx
