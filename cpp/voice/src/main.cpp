// 1bit-voice — CLI: reads a prompt from --prompt or stdin, streams audio
// chunks to stdout (or --out file). Pipe into paplay/aplay/ffplay.

#include "onebit/voice/pipeline.hpp"

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv)
{
    using namespace onebit::voice;

    CLI::App app{"1bit-voice — sentence-boundary streaming voice CLI"};

    std::string prompt;
    std::string llm_url     = "http://127.0.0.1:8180/v1/chat/completions";
    std::string model       = "1bit-monster-2b";
    std::string tts_url     = "http://127.0.0.1:8083/tts";
    std::string voice       = "af_sky";
    std::string out_path;
    std::uint32_t max_tokens   = 256;
    float         temperature  = 0.7F;
    std::uint32_t timeout_secs = 120;
    bool          timing       = false;

    app.add_option("--prompt",      prompt,      "Prompt text or `-` for stdin")
        ->required();
    app.add_option("--llm-url",     llm_url,     "LLM /v1/chat/completions URL");
    app.add_option("--model",       model,       "Model id");
    app.add_option("--tts-url",     tts_url,     "Kokoro /tts URL");
    app.add_option("--voice",       voice,       "Voice id");
    app.add_option("--max-tokens",  max_tokens,  "Max generation tokens");
    app.add_option("--temperature", temperature, "Sampling temperature");
    app.add_option("--out",         out_path,    "Output file (default stdout)");
    app.add_option("--timeout",     timeout_secs,"HTTP timeout seconds");
    app.add_flag("--timing",        timing,      "Log per-sentence latency");
    CLI11_PARSE(app, argc, argv);

    if (prompt == "-") {
        std::stringstream ss;
        ss << std::cin.rdbuf();
        prompt = ss.str();
        while (!prompt.empty() &&
               (prompt.back() == '\n' || prompt.back() == '\r' ||
                prompt.back() == ' '  || prompt.back() == '\t')) {
            prompt.pop_back();
        }
    }

    VoiceConfig cfg{
        .llm_url = llm_url, .model = model,
        .max_tokens = max_tokens, .temperature = temperature,
        .tts_url = tts_url, .voice = voice,
        .timeout_secs = timeout_secs,
    };
    VoicePipeline pipeline(std::move(cfg));

    std::ofstream                 file_sink;
    std::ostream*                 sink = &std::cout;
    if (!out_path.empty()) {
        file_sink.open(out_path, std::ios::binary | std::ios::out);
        if (!file_sink) {
            spdlog::error("could not open --out path {}", out_path);
            return 2;
        }
        sink = &file_sink;
    }

    const auto start = std::chrono::steady_clock::now();
    std::size_t total_bytes = 0;
    std::size_t n_chunks    = 0;

    auto handler = [&](const VoiceChunk& chunk) -> bool {
        sink->write(reinterpret_cast<const char*>(chunk.wav.data()),
                    static_cast<std::streamsize>(chunk.wav.size()));
        sink->flush();
        total_bytes += chunk.wav.size();
        ++n_chunks;
        if (timing) {
            const auto ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count();
            std::fprintf(stderr,
                         "[1bit-voice] chunk %zu (%4lld ms, %zu bytes wav) "
                         "sentence=\"%s\"\n",
                         chunk.index, static_cast<long long>(ms),
                         chunk.wav.size(), chunk.sentence.c_str());
        }
        return true;
    };

    auto rc = pipeline.speak(prompt, handler);
    if (!rc) {
        spdlog::error("voice pipeline failed: {}", rc.error().message);
        return 1;
    }
    if (timing) {
        const auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start)
                .count();
        std::fprintf(stderr,
                     "[1bit-voice] done · %zu chunks · %zu bytes · %.2f s\n",
                     n_chunks, total_bytes,
                     static_cast<double>(ms) / 1000.0);
    }
    return 0;
}
