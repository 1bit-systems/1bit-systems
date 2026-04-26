// error_test.cpp — labels + what() formatting.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/onnx/error.hpp"

using namespace onebit::onnx;

TEST_CASE("every kind has a stable label")
{
    CHECK(label(ErrorKind::MissingArtifact)       == "missing_artifact");
    CHECK(label(ErrorKind::NotAnArtifactDir)      == "not_an_artifact_dir");
    CHECK(label(ErrorKind::InvalidGenAiConfig)    == "invalid_genai_config");
    CHECK(label(ErrorKind::OrtRuntimeUnavailable) == "ort_runtime_unavailable");
    CHECK(label(ErrorKind::VitisaiUnavailable)    == "vitisai_unavailable");
    CHECK(label(ErrorKind::SessionInit)           == "session_init");
    CHECK(label(ErrorKind::TokenizerLoad)         == "tokenizer_load");
    CHECK(label(ErrorKind::Io)                    == "io");
}

TEST_CASE("what() includes path and detail when both are set")
{
    Error e{ErrorKind::MissingArtifact, "/tmp/foo.onnx", "weights blob gone"};
    auto s = e.what();
    CHECK(s.find("missing_artifact") != std::string::npos);
    CHECK(s.find("/tmp/foo.onnx")    != std::string::npos);
    CHECK(s.find("weights blob gone") != std::string::npos);
}

TEST_CASE("what() omits empty path / detail")
{
    Error e{ErrorKind::SessionInit, std::string{"session create failed"}};
    auto s = e.what();
    CHECK(s.find("session_init")            != std::string::npos);
    CHECK(s.find("session create failed")   != std::string::npos);
    // Path field empty — should not produce ": :" sequence.
    CHECK(s.find(": :") == std::string::npos);
}

TEST_CASE("Error fields are read-only after construction")
{
    Error e{ErrorKind::Io, "/no/such", "ENOENT"};
    CHECK(e.kind()         == ErrorKind::Io);
    CHECK(e.path().string() == "/no/such");
    CHECK(e.detail()       == "ENOENT");
}

TEST_CASE("unknown kind cast doesn't crash label()")
{
    auto bogus = static_cast<ErrorKind>(999);
    CHECK(label(bogus) == "unknown");
}
