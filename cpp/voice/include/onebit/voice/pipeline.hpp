#pragma once

// VoicePipeline — port of crates/1bit-voice/src/pipeline.rs.
//
// LLM SSE → SentenceSplitter → kokoro `/tts` POST per sentence →
// callback per VoiceChunk (sentence + WAV bytes). The interleave is
// the whole point: TTS fires the moment the splitter yields a sentence,
// in parallel with the LLM still generating.
//
// C++ shape vs Rust:
//   * Rust returned a Pin<Box<dyn Stream>>; C++23 std::generator over a
//     fallible chunk would need a coroutine wrapper. We instead expose
//     a synchronous `speak(prompt, callback)` that invokes the user's
//     `chunk_handler` once per emitted chunk. Streaming is preserved at
//     the wire level — the SSE consumer streams via std::string_view
//     slices (no copy of inbound bytes into a per-event std::string).
//   * std::expected<void, Error> for the overall return.

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::voice {

struct VoiceConfig {
    // 1bit-server base URL with /v1/chat/completions path.
    std::string llm_url     = "http://127.0.0.1:8180/v1/chat/completions";
    // Model id for the LLM request.
    std::string model       = "1bit-monster-2b";
    // Max tokens for the completion.
    std::uint32_t max_tokens = 256;
    // Sampling temperature; 0 = greedy.
    float temperature        = 0.7F;
    // 1bit-halo-kokoro `/tts` endpoint.
    std::string tts_url     = "http://127.0.0.1:8083/tts";
    // Voice id passed to kokoro.
    std::string voice       = "af_sky";
    // Per-HTTP-call timeout, seconds.
    std::uint32_t timeout_secs = 60;
};

struct VoiceChunk {
    std::size_t  index;
    std::string  sentence;
    std::vector<std::uint8_t> wav;
};

struct PipelineError {
    enum class Kind {
        LlmConnect,
        LlmStatus,
        TtsConnect,
        TtsStatus,
        Cancelled,
    };
    Kind        kind;
    std::string message;
};

// Minimal URL split helper (host, port, path). Public for tests.
struct ParsedUrl {
    std::string scheme;   // "http" or "https"
    std::string host;
    std::uint16_t port = 0;
    std::string path;     // includes leading '/'; "/" when absent
};
[[nodiscard]] std::expected<ParsedUrl, std::string>
parse_url(std::string_view url);

// Parse a single SSE event block; return the OpenAI-compat
// choices[0].delta.content if present. Empty string on the [DONE] sentinel.
// std::nullopt for role-only frames or malformed payloads — callers skip.
// `event` is a non-owning view into the SSE buffer; we slice deeper, no copy.
[[nodiscard]] std::optional<std::string>
parse_sse_delta(std::string_view event);

// Forward declaration; impl lives in pipeline.cpp via pImpl (Core
// Guidelines I.27).
class VoicePipeline {
public:
    using ChunkHandler =
        std::function<bool(const VoiceChunk& chunk)>;
    // Returning false from ChunkHandler cancels the pipeline; speak()
    // resolves with std::unexpected{Cancelled}.

    explicit VoicePipeline(VoiceConfig cfg);
    ~VoicePipeline();
    VoicePipeline(const VoicePipeline&)            = delete;
    VoicePipeline& operator=(const VoicePipeline&) = delete;
    VoicePipeline(VoicePipeline&&) noexcept;
    VoicePipeline& operator=(VoicePipeline&&) noexcept;

    [[nodiscard]] const VoiceConfig& config() const noexcept;

    // Drive the full pipeline against the configured backends. `handler`
    // receives chunks in sentence order. Synchronous; intended to be
    // run on a worker thread by the caller (echo/server.cpp does this).
    [[nodiscard]] std::expected<void, PipelineError>
    speak(std::string_view prompt, const ChunkHandler& handler);

    // Inject a fake LLM SSE body for tests / off-network use. When set,
    // the next speak() skips the HTTP POST and feeds `body` to the
    // splitter as if it had arrived from the LLM. Cleared after use.
    void inject_test_sse(std::string body);

    // Inject a fake TTS handler. When set, replaces the kokoro POST with
    // a local lambda that returns the WAV bytes for any (text, voice)
    // pair. Stays installed across calls.
    using TestTts = std::function<
        std::expected<std::vector<std::uint8_t>, PipelineError>(
            std::string_view text, std::string_view voice)>;
    void set_test_tts(TestTts tts);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace onebit::voice
