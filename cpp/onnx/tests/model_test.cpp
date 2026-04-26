// model_test.cpp — ArtifactPaths::discover + GenAiConfig::load.
//
// All filesystem fixtures are temp dirs we build ourselves; no
// dependency on the 2.9 GB TriLM artifact.

#include <doctest/doctest.h>

#include "onebit/onnx/model.hpp"

#include <filesystem>
#include <fstream>

using namespace onebit::onnx;
namespace fs = std::filesystem;

namespace {
fs::path make_tmp(const char* tag) {
    auto base = fs::temp_directory_path() /
                ("onebit_onnx_test_" + std::string{tag} + "_" +
                 std::to_string(static_cast<unsigned long long>(
                     reinterpret_cast<std::uintptr_t>(&tag))));
    fs::remove_all(base);
    fs::create_directories(base);
    return base;
}

void write(const fs::path& p, std::string_view body) {
    std::ofstream o{p, std::ios::binary};
    o.write(body.data(), static_cast<std::streamsize>(body.size()));
}
} // namespace

TEST_CASE("discover rejects non-artifact dir")
{
    auto td = make_tmp("not_artifact");
    auto r  = ArtifactPaths::discover(td);
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().kind() == ErrorKind::NotAnArtifactDir);
    fs::remove_all(td);
}

TEST_CASE("discover flags partial sync")
{
    auto td = make_tmp("partial");
    write(td / "model.onnx", "fake-graph");
    // intentionally omit weights + config + tokenizer
    auto r = ArtifactPaths::discover(td);
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().kind() == ErrorKind::MissingArtifact);
    fs::remove_all(td);
}

TEST_CASE("discover returns full layout when complete")
{
    auto td = make_tmp("complete");
    write(td / "model.onnx",         "graph");
    write(td / "model.onnx.data",    "weights");
    write(td / "genai_config.json",  R"({"model":{}})");
    write(td / "tokenizer.json",     "{}");

    auto r = ArtifactPaths::discover(td);
    REQUIRE(r.has_value());
    CHECK(r->root         == td);
    CHECK(r->model        == td / "model.onnx");
    CHECK(r->weights      == td / "model.onnx.data");
    CHECK(r->genai_config == td / "genai_config.json");
    CHECK(r->tokenizer    == td / "tokenizer.json");
    fs::remove_all(td);
}

TEST_CASE("GenAiConfig::load parses TriLM-shaped fixture")
{
    auto td = make_tmp("genai_cfg");
    auto cfg_path = td / "genai_config.json";
    write(cfg_path, R"({
        "model": {
            "bos_token_id": 0,
            "eos_token_id": 0,
            "context_length": 2048,
            "vocab_size": 50688,
            "type": "llama",
            "decoder": {
                "head_size": 128,
                "hidden_size": 3072,
                "num_attention_heads": 24,
                "num_key_value_heads": 24,
                "num_hidden_layers": 30
            }
        },
        "search": {
            "do_sample": false,
            "temperature": 1.0,
            "top_k": 50
        }
    })");
    auto cfg = GenAiConfig::load(cfg_path);
    REQUIRE(cfg.has_value());
    CHECK(cfg->model.arch                       == "llama");
    CHECK(cfg->model.context_length             == 2048);
    CHECK(cfg->model.vocab_size                 == 50688u);
    CHECK(cfg->model.decoder.num_hidden_layers  == 30);
    CHECK(cfg->model.decoder.num_attention_heads == 24);
    CHECK(cfg->model.decoder.hidden_size        == 3072);
    CHECK(cfg->search.top_k                     == 50u);
    fs::remove_all(td);
}

TEST_CASE("GenAiConfig::load surfaces parse errors typed")
{
    auto td = make_tmp("genai_bad");
    auto cfg_path = td / "genai_config.json";
    write(cfg_path, "not json {{{");
    auto cfg = GenAiConfig::load(cfg_path);
    REQUIRE_FALSE(cfg.has_value());
    CHECK(cfg.error().kind() == ErrorKind::InvalidGenAiConfig);
    fs::remove_all(td);
}

TEST_CASE("GenAiConfig::load fails IO on missing file")
{
    auto cfg = GenAiConfig::load("/no/such/file.json");
    REQUIRE_FALSE(cfg.has_value());
    CHECK(cfg.error().kind() == ErrorKind::Io);
}
