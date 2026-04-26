#include "onebit/voice/pipeline.hpp"

#include "onebit/voice/splitter.hpp"
#include "onebit/voice/sse_client.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <charconv>
#include <cstring>
#include <optional>

namespace onebit::voice {

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// URL parser — host/port/path; scheme assumed http unless https://.
// ---------------------------------------------------------------------------

std::expected<ParsedUrl, std::string> parse_url(std::string_view url)
{
    ParsedUrl out{};
    auto scheme_end = url.find("://");
    std::string_view rest;
    if (scheme_end != std::string_view::npos) {
        out.scheme = std::string{url.substr(0, scheme_end)};
        rest       = url.substr(scheme_end + 3);
    } else {
        out.scheme = "http";
        rest       = url;
    }
    out.port = (out.scheme == "https") ? 443U : 80U;

    const auto slash = rest.find('/');
    std::string_view authority =
        (slash == std::string_view::npos) ? rest : rest.substr(0, slash);
    out.path =
        (slash == std::string_view::npos) ? "/" : std::string{rest.substr(slash)};

    const auto colon = authority.find(':');
    if (colon == std::string_view::npos) {
        out.host = std::string{authority};
    } else {
        out.host = std::string{authority.substr(0, colon)};
        const auto port_sv = authority.substr(colon + 1);
        std::uint16_t p    = 0;
        const auto*  begin = port_sv.data();
        const auto*  end   = begin + port_sv.size();
        const auto   r     = std::from_chars(begin, end, p);
        if (r.ec != std::errc{} || p == 0) {
            return std::unexpected(std::string("bad port in url"));
        }
        out.port = p;
    }
    if (out.host.empty()) {
        return std::unexpected(std::string("empty host"));
    }
    return out;
}

// ---------------------------------------------------------------------------
// SSE event → delta content
// ---------------------------------------------------------------------------

std::optional<std::string> parse_sse_delta(std::string_view event)
{
    // Find the last `data:` line in the event block.
    std::optional<std::string_view> payload;
    std::size_t i = 0;
    while (i < event.size()) {
        const auto nl   = event.find('\n', i);
        const auto end  = (nl == std::string_view::npos) ? event.size() : nl;
        std::string_view line = event.substr(i, end - i);
        // Strip optional trailing \r.
        if (!line.empty() && line.back() == '\r') {
            line = line.substr(0, line.size() - 1);
        }
        constexpr std::string_view kPfxA = "data: ";
        constexpr std::string_view kPfxB = "data:";
        if (line.size() >= kPfxA.size() &&
            line.substr(0, kPfxA.size()) == kPfxA) {
            payload = line.substr(kPfxA.size());
        } else if (line.size() >= kPfxB.size() &&
                   line.substr(0, kPfxB.size()) == kPfxB) {
            payload = line.substr(kPfxB.size());
        }
        if (nl == std::string_view::npos) break;
        i = nl + 1;
    }
    if (!payload) return std::nullopt;

    // trim leading whitespace from payload
    std::string_view p = *payload;
    while (!p.empty() && (p.front() == ' ' || p.front() == '\t')) {
        p.remove_prefix(1);
    }
    if (p == "[DONE]") return std::string{};

    json v;
    try {
        v = json::parse(p);
    } catch (const json::exception&) {
        return std::nullopt;
    }
    const auto* choices = v.contains("choices") ? &v["choices"] : nullptr;
    if (choices == nullptr || !choices->is_array() || choices->empty()) {
        return std::nullopt;
    }
    const auto& first = (*choices)[0];
    if (!first.contains("delta")) return std::nullopt;
    const auto& delta = first["delta"];
    if (!delta.contains("content") || !delta["content"].is_string()) {
        return std::nullopt;
    }
    return delta["content"].get<std::string>();
}

// ---------------------------------------------------------------------------
// Pipeline impl
// ---------------------------------------------------------------------------

struct VoicePipeline::Impl {
    VoiceConfig                cfg;
    std::optional<std::string> injected_sse;
    TestTts                    test_tts;
};

VoicePipeline::VoicePipeline(VoiceConfig cfg)
    : p_(std::make_unique<Impl>())
{
    p_->cfg = std::move(cfg);
}

VoicePipeline::~VoicePipeline()                              = default;
VoicePipeline::VoicePipeline(VoicePipeline&&) noexcept       = default;
VoicePipeline& VoicePipeline::operator=(VoicePipeline&&) noexcept = default;

const VoiceConfig& VoicePipeline::config() const noexcept { return p_->cfg; }

void VoicePipeline::inject_test_sse(std::string body)
{
    p_->injected_sse = std::move(body);
}

void VoicePipeline::set_test_tts(TestTts tts)
{
    p_->test_tts = std::move(tts);
}

namespace {

[[nodiscard]] std::expected<std::vector<std::uint8_t>, PipelineError>
synthesize_real(const VoiceConfig& cfg, std::string_view text)
{
    auto tts_url = parse_url(cfg.tts_url);
    if (!tts_url) {
        return std::unexpected(PipelineError{
            PipelineError::Kind::TtsConnect, tts_url.error()});
    }
    httplib::Client cli(tts_url->scheme + "://" + tts_url->host + ":" +
                        std::to_string(tts_url->port));
    cli.set_connection_timeout(static_cast<time_t>(cfg.timeout_secs), 0);
    cli.set_read_timeout(static_cast<time_t>(cfg.timeout_secs), 0);

    json body = {
        {"text",  std::string{text}},
        {"voice", cfg.voice},
    };
    const std::string body_s = body.dump();
    auto              res    = cli.Post(tts_url->path, body_s, "application/json");
    if (!res) {
        return std::unexpected(PipelineError{
            PipelineError::Kind::TtsConnect,
            "POST " + cfg.tts_url + " failed"});
    }
    if (res->status < 200 || res->status >= 300) {
        return std::unexpected(PipelineError{
            PipelineError::Kind::TtsStatus,
            "kokoro " + std::to_string(res->status) + ": " + res->body});
    }
    std::vector<std::uint8_t> wav(res->body.begin(), res->body.end());
    return wav;
}

} // namespace

std::expected<void, PipelineError>
VoicePipeline::speak(std::string_view prompt, const ChunkHandler& handler)
{
    const auto& cfg = p_->cfg;
    SentenceSplitter splitter;
    std::size_t      idx = 0;

    // Fire one TTS request and dispatch the chunk to the user callback.
    // Returns false if the user asked us to stop.
    auto handle_sentence =
        [&](std::string sentence) -> std::expected<bool, PipelineError> {
        std::expected<std::vector<std::uint8_t>, PipelineError> wav_r;
        if (p_->test_tts) {
            wav_r = p_->test_tts(sentence, cfg.voice);
        } else {
            wav_r = synthesize_real(cfg, sentence);
        }
        if (!wav_r) return std::unexpected(wav_r.error());
        VoiceChunk chunk{idx, std::move(sentence), std::move(*wav_r)};
        ++idx;
        return handler(chunk);
    };

    // ------------------------------------------------------------------
    // Path A — injected test SSE body. No network. Treat it as a single
    // already-buffered SSE stream and split by "\n\n".
    // ------------------------------------------------------------------
    if (p_->injected_sse) {
        std::string body = std::move(*p_->injected_sse);
        p_->injected_sse.reset();

        std::size_t i = 0;
        while (i < body.size()) {
            auto nl = body.find("\n\n", i);
            const std::string_view event =
                (nl == std::string::npos) ?
                std::string_view{body}.substr(i) :
                std::string_view{body}.substr(i, nl - i);
            const auto delta = parse_sse_delta(event);
            if (delta && !delta->empty()) {
                for (auto& sentence : splitter.feed(*delta)) {
                    auto rc = handle_sentence(std::move(sentence));
                    if (!rc)         return std::unexpected(rc.error());
                    if (!*rc) {
                        return std::unexpected(PipelineError{
                            PipelineError::Kind::Cancelled, "cancelled"});
                    }
                }
            }
            if (nl == std::string::npos) break;
            i = nl + 2;
        }
        if (auto tail = splitter.finish()) {
            auto rc = handle_sentence(std::move(*tail));
            if (!rc)         return std::unexpected(rc.error());
            if (!*rc) {
                return std::unexpected(PipelineError{
                    PipelineError::Kind::Cancelled, "cancelled"});
            }
        }
        return {};
    }

    // ------------------------------------------------------------------
    // Path B — live LLM. Stream SSE via our minimal TCP client.
    // ------------------------------------------------------------------
    auto llm_url = parse_url(cfg.llm_url);
    if (!llm_url) {
        return std::unexpected(PipelineError{
            PipelineError::Kind::LlmConnect, llm_url.error()});
    }

    json body = {
        {"model",       cfg.model},
        {"messages",    json::array({ json{
            {"role",    "user"},
            {"content", std::string{prompt}}}})},
        {"max_tokens",  cfg.max_tokens},
        {"temperature", cfg.temperature},
        {"stream",      true},
    };
    const std::string body_s = body.dump();

    std::expected<bool, PipelineError> last_handler_ok = true;
    bool                               cancelled       = false;

    auto sse_rc = post_sse(
        llm_url->host, llm_url->port, llm_url->path,
        body_s, cfg.timeout_secs,
        [&](std::string_view event) -> bool {
            const auto delta = parse_sse_delta(event);
            if (!delta || delta->empty()) {
                return true;
            }
            for (auto& sentence : splitter.feed(*delta)) {
                auto rc = handle_sentence(std::move(sentence));
                if (!rc) {
                    last_handler_ok = std::unexpected(rc.error());
                    return false;
                }
                if (!*rc) {
                    cancelled = true;
                    return false;
                }
            }
            return true;
        });

    if (!sse_rc) {
        const auto& e = sse_rc.error();
        const auto kind =
            (e.kind == SseError::Kind::Status)
                ? PipelineError::Kind::LlmStatus
                : PipelineError::Kind::LlmConnect;
        return std::unexpected(PipelineError{kind, e.message});
    }
    if (cancelled) {
        return std::unexpected(PipelineError{
            PipelineError::Kind::Cancelled, "cancelled"});
    }
    if (!last_handler_ok) {
        return std::unexpected(last_handler_ok.error());
    }

    if (auto tail = splitter.finish()) {
        auto rc = handle_sentence(std::move(*tail));
        if (!rc)        return std::unexpected(rc.error());
        if (!*rc) {
            return std::unexpected(PipelineError{
                PipelineError::Kind::Cancelled, "cancelled"});
        }
    }
    return {};
}

} // namespace onebit::voice
