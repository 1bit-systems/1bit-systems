// 1bit-echo — browser-side WebSocket gateway over 1bit-voice.

#include "onebit/echo/server.hpp"

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include <string>

int main(int argc, char** argv)
{
    using namespace onebit::echo;

    CLI::App app{"1bit-echo — WebSocket gateway over 1bit-voice"};

    std::string  bind_host    = "127.0.0.1";
    std::uint16_t bind_port   = 8085;
    std::string  llm_url      = "http://127.0.0.1:8180/v1/chat/completions";
    std::string  tts_url      = "http://127.0.0.1:8083/tts";
    std::string  voice        = "af_sky";
    std::string  codec_str    = "wav";

    app.add_option("--bind-host", bind_host, "Bind host (default 127.0.0.1)");
    app.add_option("--bind-port", bind_port, "Bind port (default 8085)");
    app.add_option("--llm-url",   llm_url,   "LLM /v1/chat/completions URL");
    app.add_option("--tts-url",   tts_url,   "Kokoro /tts URL");
    app.add_option("--voice",     voice,     "Voice id");
    app.add_option("--codec",     codec_str, "Wire codec: wav|pcm");
    CLI11_PARSE(app, argc, argv);

    auto codec = codec_from_str(codec_str);
    if (!codec) {
        spdlog::error("{}", codec.error());
        return 2;
    }
    EchoConfig cfg{};
    cfg.bind_host          = bind_host;
    cfg.bind_port          = bind_port;
    cfg.voice_cfg.llm_url  = llm_url;
    cfg.voice_cfg.tts_url  = tts_url;
    cfg.voice_cfg.voice    = voice;
    cfg.codec              = *codec;

    EchoServer server(std::move(cfg));
    auto rc = server.run();
    if (!rc) {
        spdlog::error("1bit-echo: {}", rc.error());
        return 1;
    }
    return 0;
}
