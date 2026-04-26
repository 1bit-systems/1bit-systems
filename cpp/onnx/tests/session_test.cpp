// session_test.cpp — Session lifecycle without paying the cost of a
// real Ort::Session. config-only mode + runtime-availability probe.

#include <doctest/doctest.h>

#include "onebit/onnx/session.hpp"

#include <filesystem>
#include <fstream>

using namespace onebit::onnx;
namespace fs = std::filesystem;

namespace {
fs::path make_tmp(const char* tag) {
    auto base = fs::temp_directory_path() /
                ("onebit_onnx_session_" + std::string{tag} + "_" +
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

fs::path build_minimal_artifact(const char* tag) {
    auto td = make_tmp(tag);
    write(td / "model.onnx",        "graph");
    write(td / "model.onnx.data",   "weights");
    write(td / "genai_config.json", R"({
        "model": {"bos_token_id":0, "eos_token_id":0, "context_length":128,
                  "vocab_size":256, "type":"llama",
                  "decoder":{"head_size":8,"hidden_size":16,
                             "num_attention_heads":2,"num_key_value_heads":2,
                             "num_hidden_layers":1}}
    })");
    write(td / "tokenizer.json", "{}");
    return td;
}
} // namespace

TEST_CASE("load_config_only round-trips paths + config")
{
    auto td = build_minimal_artifact("cfg_only");
    auto s  = Session::load_config_only(td);
    REQUIRE(s.has_value());
    CHECK(s->paths().root      == td);
    CHECK(s->config().model.arch == "llama");
    CHECK(s->lane()            == ExecutionLane::Cpu);
    CHECK(s->has_runtime()     == false);
    fs::remove_all(td);
}

TEST_CASE("load_config_only on missing dir returns NotAnArtifactDir")
{
    auto s = Session::load_config_only("/tmp/onebit_no_such_artifact_dir");
    REQUIRE_FALSE(s.has_value());
    CHECK(s.error().kind() == ErrorKind::NotAnArtifactDir);
}

TEST_CASE("execution lane labels stable")
{
    CHECK(label(ExecutionLane::Cpu)     == "ort-cpu");
    CHECK(label(ExecutionLane::Vitisai) == "ort-vitisai");
}

TEST_CASE("runtime_available reflects build configuration")
{
    // Self-consistency test: whichever way the build went, the value
    // is observable and stable.
    bool a = Session::runtime_available();
    bool b = Session::runtime_available();
    CHECK(a == b);
    if (a) {
        DOCTEST_WARN_MESSAGE(true, "ORT linked — full session paths exercised in integration");
    } else {
        DOCTEST_WARN_MESSAGE(true, "ORT NOT linked — Session::load returns OrtRuntimeUnavailable (expected)");
    }
}

TEST_CASE("Session::load returns OrtRuntimeUnavailable when ORT not linked")
{
    if (Session::runtime_available()) {
        DOCTEST_WARN_MESSAGE(true, "ORT linked — skipping no-runtime guard test");
        return;
    }
    auto td = build_minimal_artifact("no_runtime");
    auto s  = Session::load(td);
    REQUIRE_FALSE(s.has_value());
    CHECK(s.error().kind() == ErrorKind::OrtRuntimeUnavailable);
    fs::remove_all(td);
}

TEST_CASE("Session is move-only; move preserves config")
{
    auto td = build_minimal_artifact("move");
    auto s1 = Session::load_config_only(td);
    REQUIRE(s1.has_value());
    auto s2 = std::move(*s1);
    CHECK(s2.config().model.context_length == 128);
    CHECK(s2.has_runtime() == false);
    fs::remove_all(td);
}
