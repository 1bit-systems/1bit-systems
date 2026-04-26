#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/voice/pipeline.hpp"
#include "onebit/voice/splitter.hpp"

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

using onebit::voice::parse_sse_delta;
using onebit::voice::parse_url;
using onebit::voice::PipelineError;
using onebit::voice::SentenceSplitter;
using onebit::voice::VoiceChunk;
using onebit::voice::VoiceConfig;
using onebit::voice::VoicePipeline;

// =====================================================================
// SentenceSplitter — same matrix as Rust splitter::tests, bit-exact.
// =====================================================================

TEST_CASE("splitter: single complete sentence")
{
    SentenceSplitter s;
    auto             out = s.feed("Hello world.");
    REQUIRE(out.size() == 1);
    CHECK(out[0] == "Hello world.");
    CHECK(s.buffered() == "");
    CHECK(!s.finish().has_value());
}

TEST_CASE("splitter: partial sentence buffers")
{
    SentenceSplitter s;
    CHECK(s.feed("Hello wor").empty());
    CHECK(s.buffered() == "Hello wor");
    auto out = s.feed("ld.");
    REQUIRE(out.size() == 1);
    CHECK(out[0] == "Hello world.");
}

TEST_CASE("splitter: multiple sentences in one delta")
{
    SentenceSplitter s;
    auto             out = s.feed("One. Two! Three?");
    REQUIRE(out.size() == 3);
    CHECK(out[0] == "One.");
    CHECK(out[1] == "Two!");
    CHECK(out[2] == "Three?");
    CHECK(s.buffered() == "");
}

TEST_CASE("splitter: newline is a boundary")
{
    SentenceSplitter s;
    auto             out = s.feed("first line\nsecond line\n");
    REQUIRE(out.size() == 2);
    CHECK(out[0] == "first line");
    CHECK(out[1] == "second line");
}

TEST_CASE("splitter: trailing partial flushed on finish")
{
    SentenceSplitter s;
    CHECK(s.feed("trailing no punct").empty());
    auto fin = s.finish();
    REQUIRE(fin.has_value());
    CHECK(*fin == "trailing no punct");
    CHECK(s.buffered() == "");
}

TEST_CASE("splitter: empty deltas are no-ops")
{
    SentenceSplitter s;
    CHECK(s.feed("").empty());
    CHECK(s.feed("").empty());
    CHECK(!s.finish().has_value());
}

TEST_CASE("splitter: lone punctuation delta is dropped")
{
    SentenceSplitter s;
    auto             a = s.feed("Hi.");
    REQUIRE(a.size() == 1);
    CHECK(a[0] == "Hi.");
    auto b = s.feed("!");
    CHECK(b.empty());
}

TEST_CASE("splitter: ellipsis emits at most one speakable sentence")
{
    SentenceSplitter s;
    auto             out = s.feed("Hello...");
    REQUIRE(out.size() == 1);
    CHECK(out[0] == "Hello.");
    CHECK(!s.finish().has_value());
}

TEST_CASE("splitter: whitespace between sentences swallowed")
{
    SentenceSplitter s;
    auto             out = s.feed("First.   Second.");
    REQUIRE(out.size() == 2);
    CHECK(out[0] == "First.");
    CHECK(out[1] == "Second.");
}

TEST_CASE("splitter: abbreviation false positive (documented)")
{
    SentenceSplitter s;
    auto             out = s.feed("e.g. this.");
    REQUIRE(out.size() == 3);
    CHECK(out[0] == "e.");
    CHECK(out[1] == "g.");
    CHECK(out[2] == "this.");
}

// =====================================================================
// SSE delta parser
// =====================================================================

TEST_CASE("parse_sse_delta: happy path")
{
    auto d = parse_sse_delta(R"(data: {"choices":[{"delta":{"content":"Paris"}}]})");
    REQUIRE(d.has_value());
    CHECK(*d == "Paris");
}

TEST_CASE("parse_sse_delta: [DONE] sentinel returns empty string")
{
    auto a = parse_sse_delta("data: [DONE]");
    REQUIRE(a.has_value());
    CHECK(a->empty());
    auto b = parse_sse_delta("data:[DONE]");
    REQUIRE(b.has_value());
    CHECK(b->empty());
}

TEST_CASE("parse_sse_delta: role-only frame returns nullopt")
{
    auto d = parse_sse_delta(R"(data: {"choices":[{"delta":{"role":"assistant"}}]})");
    CHECK(!d.has_value());
}

TEST_CASE("parse_sse_delta: malformed returns nullopt")
{
    CHECK(!parse_sse_delta("data: not-json").has_value());
    CHECK(!parse_sse_delta("comment: whatever").has_value());
    CHECK(!parse_sse_delta("").has_value());
}

// =====================================================================
// VoiceConfig + URL parser + VoiceChunk
// =====================================================================

TEST_CASE("VoiceConfig: defaults point at local lemond + kokoro")
{
    const VoiceConfig c{};
    CHECK(c.llm_url.find("127.0.0.1:8180") != std::string::npos);
    CHECK(c.tts_url.find("127.0.0.1:8083") != std::string::npos);
    CHECK(c.voice == "af_sky");
}

TEST_CASE("parse_url: explicit port + path")
{
    auto r = parse_url("http://127.0.0.1:8180/v1/chat/completions");
    REQUIRE(r.has_value());
    CHECK(r->scheme == "http");
    CHECK(r->host == "127.0.0.1");
    CHECK(r->port == 8180);
    CHECK(r->path == "/v1/chat/completions");
}

TEST_CASE("parse_url: missing scheme defaults to http:80")
{
    auto r = parse_url("example.com/x");
    REQUIRE(r.has_value());
    CHECK(r->scheme == "http");
    CHECK(r->host == "example.com");
    CHECK(r->port == 80);
    CHECK(r->path == "/x");
}

TEST_CASE("VoiceChunk: trivial round-trip")
{
    VoiceChunk c{3, "Hi.", {'R', 'I', 'F', 'F'}};
    CHECK(c.index == 3);
    CHECK(c.sentence == "Hi.");
    REQUIRE(c.wav.size() == 4);
    CHECK(c.wav[0] == 'R');
}

// =====================================================================
// VoicePipeline — drives splitter + injected fake SSE + injected TTS,
// no network, no httplib calls.
// =====================================================================

TEST_CASE("pipeline: injected SSE + fake TTS yields chunks in sentence order")
{
    VoicePipeline pipe(VoiceConfig{});
    pipe.set_test_tts(
        [](std::string_view text,
           std::string_view) -> std::expected<std::vector<std::uint8_t>, PipelineError> {
            std::vector<std::uint8_t> wav(text.begin(), text.end());
            return wav;
        });

    // Mock a chat-completions SSE body: three deltas plus DONE.
    std::string sse;
    sse += R"(data: {"choices":[{"delta":{"role":"assistant"}}]})" "\n\n";
    sse += R"(data: {"choices":[{"delta":{"content":"Hello"}}]})" "\n\n";
    sse += R"(data: {"choices":[{"delta":{"content":" world. How are"}}]})" "\n\n";
    sse += R"(data: {"choices":[{"delta":{"content":" you?"}}]})" "\n\n";
    sse += "data: [DONE]\n\n";
    pipe.inject_test_sse(std::move(sse));

    std::vector<VoiceChunk> got;
    auto rc = pipe.speak(
        "ignored",
        [&](const VoiceChunk& c) {
            got.push_back(c);
            return true;
        });
    REQUIRE(rc.has_value());
    REQUIRE(got.size() == 2);
    CHECK(got[0].sentence == "Hello world.");
    CHECK(got[0].index == 0);
    CHECK(got[1].sentence == "How are you?");
    CHECK(got[1].index == 1);
    // Body of WAV is the sentence (because of fake TTS).
    CHECK(std::string(got[0].wav.begin(), got[0].wav.end()) == "Hello world.");
}

TEST_CASE("pipeline: handler returning false cancels and surfaces Cancelled")
{
    VoicePipeline pipe(VoiceConfig{});
    pipe.set_test_tts(
        [](std::string_view, std::string_view)
            -> std::expected<std::vector<std::uint8_t>, PipelineError> {
            return std::vector<std::uint8_t>{0u};
        });
    std::string sse;
    sse += R"(data: {"choices":[{"delta":{"content":"A. B. C."}}]})" "\n\n";
    sse += "data: [DONE]\n\n";
    pipe.inject_test_sse(std::move(sse));

    std::size_t count = 0;
    auto        rc    = pipe.speak("x", [&](const VoiceChunk&) {
        ++count;
        return false;     // bail after the first
    });
    REQUIRE(!rc.has_value());
    CHECK(rc.error().kind == PipelineError::Kind::Cancelled);
    CHECK(count == 1);
}

TEST_CASE("pipeline: tail without trailing punct flushed via finish()")
{
    VoicePipeline pipe(VoiceConfig{});
    pipe.set_test_tts(
        [](std::string_view t,
           std::string_view) -> std::expected<std::vector<std::uint8_t>, PipelineError> {
            return std::vector<std::uint8_t>(t.begin(), t.end());
        });
    std::string sse;
    sse += R"(data: {"choices":[{"delta":{"content":"trailing no punct"}}]})" "\n\n";
    sse += "data: [DONE]\n\n";
    pipe.inject_test_sse(std::move(sse));

    std::vector<std::string> sentences;
    auto rc = pipe.speak("x", [&](const VoiceChunk& c) {
        sentences.push_back(c.sentence);
        return true;
    });
    REQUIRE(rc.has_value());
    REQUIRE(sentences.size() == 1);
    CHECK(sentences[0] == "trailing no punct");
}
