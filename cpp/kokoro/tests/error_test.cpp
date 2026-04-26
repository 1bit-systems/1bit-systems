// error_test.cpp — labels + what() formatting for kokoro::Error.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/kokoro/error.hpp"

using namespace onebit::kokoro;

TEST_CASE("kokoro labels are stable")
{
    CHECK(label(ErrorKind::UnsupportedStub) == "unsupported_stub");
    CHECK(label(ErrorKind::ModelLoadFailed) == "model_load_failed");
    CHECK(label(ErrorKind::VoiceNotFound)   == "voice_not_found");
    CHECK(label(ErrorKind::InvalidText)     == "invalid_text");
    CHECK(label(ErrorKind::InvalidVoice)    == "invalid_voice");
    CHECK(label(ErrorKind::InvalidSpeed)    == "invalid_speed");
    CHECK(label(ErrorKind::ShimError)       == "shim_error");
    CHECK(label(ErrorKind::InvalidPath)     == "invalid_path");
}

TEST_CASE("ShimError preserves status code")
{
    Error e{ErrorKind::ShimError, "onnxruntime forward failed", -5};
    CHECK(e.kind()   == ErrorKind::ShimError);
    CHECK(e.code()   == -5);
    CHECK(e.detail() == "onnxruntime forward failed");
}

TEST_CASE("InvalidSpeed preserves the offending value")
{
    Error e{ErrorKind::InvalidSpeed, "speed out of (0,4]", 9.0f};
    CHECK(e.kind()  == ErrorKind::InvalidSpeed);
    CHECK(e.speed() == 9.0f);
}

TEST_CASE("what() includes label and detail")
{
    Error e{ErrorKind::UnsupportedStub, "build is stub-only"};
    auto s = e.what();
    CHECK(s.find("unsupported_stub")   != std::string::npos);
    CHECK(s.find("build is stub-only") != std::string::npos);
}

TEST_CASE("default construction without detail still produces label")
{
    Error e{ErrorKind::InvalidText, ""};
    auto s = e.what();
    CHECK(s == "invalid_text");
}
