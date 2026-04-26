#include "onebit/echo/server.hpp"

#include "onebit/echo/codec.hpp"
#include "onebit/echo/ws.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace onebit::echo {

using json = nlohmann::json;

std::string_view codec_name(Codec c) noexcept
{
    switch (c) {
        case Codec::Wav: return "wav";
        case Codec::Pcm: return "pcm";
    }
    return "wav";
}

std::expected<Codec, std::string> codec_from_str(std::string_view s) noexcept
{
    if (s == "wav" || s == "WAV") return Codec::Wav;
    if (s == "pcm" || s == "PCM") return Codec::Pcm;
    // accept "opus" as alias for backward compat with Rust CLI flag —
    // serves the same wire purpose with a different payload.
    if (s == "opus" || s == "OPUS") return Codec::Pcm;
    return std::unexpected(std::string{"unknown codec `"} + std::string{s} +
                           "` (want: wav|pcm)");
}

std::string preamble_json(Codec codec)
{
    std::uint32_t sr;
    std::uint32_t fr;
    std::string   name;
    if (codec == Codec::Pcm) {
        sr   = kTargetSr;
        fr   = kFrameMs;
        name = "pcm";
    } else {
        sr   = 24'000U;
        fr   = 0U;
        name = "wav";
    }
    return json{
        {"sample_rate", sr},
        {"channels",    1},
        {"frame_ms",    fr},
        {"codec",       name},
    }
        .dump();
}

namespace {

[[nodiscard]] int make_listening_socket(const std::string& host,
                                        std::uint16_t port,
                                        std::uint16_t* actual,
                                        std::string* err)
{
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        if (err) *err = std::string("socket: ") + std::strerror(errno);
        return -1;
    }
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = ::htons(port);
    if (host.empty() || host == "0.0.0.0") {
        addr.sin_addr.s_addr = ::htonl(INADDR_ANY);
    } else {
        if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
            ::close(fd);
            if (err) *err = "bad bind host: " + host;
            return -1;
        }
    }
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        if (err) *err = std::string("bind: ") + std::strerror(errno);
        ::close(fd);
        return -1;
    }
    if (::listen(fd, 32) < 0) {
        if (err) *err = std::string("listen: ") + std::strerror(errno);
        ::close(fd);
        return -1;
    }
    if (actual != nullptr) {
        sockaddr_in bound{};
        socklen_t   bl = sizeof(bound);
        if (::getsockname(fd, reinterpret_cast<sockaddr*>(&bound), &bl) == 0) {
            *actual = ::ntohs(bound.sin_port);
        }
    }
    return fd;
}

} // namespace

struct EchoServer::Impl {
    EchoConfig                cfg;
    int                       listen_fd = -1;
    std::atomic<bool>         stopping{false};
    std::mutex                threads_m;
    std::vector<std::thread>  workers;

    explicit Impl(EchoConfig c) : cfg(std::move(c)) {}

    ~Impl()
    {
        if (listen_fd >= 0) {
            ::close(listen_fd);
            listen_fd = -1;
        }
        std::lock_guard<std::mutex> lk(threads_m);
        for (auto& t : workers) {
            if (t.joinable()) t.detach();
        }
    }
};

EchoServer::EchoServer(EchoConfig cfg)
    : p_(std::make_unique<Impl>(std::move(cfg)))
{}
EchoServer::~EchoServer()                                     = default;
EchoServer::EchoServer(EchoServer&&) noexcept                 = default;
EchoServer& EchoServer::operator=(EchoServer&&) noexcept      = default;

const EchoConfig& EchoServer::config() const noexcept { return p_->cfg; }

std::expected<std::uint16_t, std::string> EchoServer::listen()
{
    std::string err;
    std::uint16_t actual = 0;
    const int     fd     =
        make_listening_socket(p_->cfg.bind_host, p_->cfg.bind_port,
                              &actual, &err);
    if (fd < 0) return std::unexpected(err);
    p_->listen_fd = fd;
    if (p_->cfg.bind_port == 0) p_->cfg.bind_port = actual;
    return actual;
}

namespace {

// Drive one VoicePipeline::speak against `socket_fd`, forwarding chunks.
void run_pipeline(int                                 fd,
                  const onebit::voice::VoiceConfig&   vcfg,
                  Codec                               codec,
                  std::string                         prompt,
                  std::atomic<bool>&                  cancel_flag)
{
    onebit::voice::VoicePipeline pipe(vcfg);
    auto rc = pipe.speak(prompt,
        [&](const onebit::voice::VoiceChunk& chunk) -> bool {
            if (cancel_flag.load(std::memory_order_acquire)) return false;
            if (codec == Codec::Wav) {
                auto r = ws_send_binary(fd, chunk.wav.data(),
                                        chunk.wav.size());
                return r.has_value();
            }
            auto frames =
                wav_to_pcm_frames(chunk.wav.data(), chunk.wav.size());
            if (!frames) {
                spdlog::warn("pcm encode failed: {}", frames.error().message);
                (void)ws_send_text(
                    fd, std::string{"error: "} + frames.error().message);
                return true;
            }
            for (const auto& f : *frames) {
                if (auto r = ws_send_binary(fd, f.data(), f.size()); !r) {
                    return false;
                }
            }
            return true;
        });
    if (!rc) {
        spdlog::warn("voice pipeline error: {}", rc.error().message);
        (void)ws_send_text(fd, std::string{"error: "} + rc.error().message);
    }
}

void handle_one_socket(int                                socket_fd,
                       const onebit::voice::VoiceConfig&  vcfg,
                       Codec                              codec)
{
    struct FdGuard {
        int fd;
        ~FdGuard() { if (fd >= 0) ::close(fd); }
    } g{socket_fd};

    auto path = ws_server_handshake(socket_fd);
    if (!path) {
        spdlog::debug("ws handshake failed: {}", path.error().message);
        return;
    }

    auto first = ws_recv(socket_fd);
    if (!first) {
        spdlog::debug("recv prompt failed: {}", first.error().message);
        return;
    }
    if (first->opcode != WsOpcode::Text) {
        spdlog::warn("expected text prompt frame, got opcode {}",
                     static_cast<int>(first->opcode));
        (void)ws_send_close(socket_fd);
        return;
    }
    std::string prompt(first->payload.begin(), first->payload.end());

    if (auto r = ws_send_text(socket_fd, preamble_json(codec)); !r) {
        return;
    }

    // Pipeline thread + a tiny watcher that flips cancel_flag on Close
    // / cancel sentinel. The pipeline's chunk handler checks the flag
    // between chunks and aborts.
    std::atomic<bool> cancel_flag{false};
    std::thread pipe_thread(
        [&, prompt = std::move(prompt)]() mutable {
            run_pipeline(socket_fd, vcfg, codec,
                         std::move(prompt), cancel_flag);
        });

    while (!cancel_flag.load(std::memory_order_acquire)) {
        auto m = ws_recv(socket_fd);
        if (!m) {
            cancel_flag.store(true, std::memory_order_release);
            break;
        }
        if (m->opcode == WsOpcode::Close) {
            cancel_flag.store(true, std::memory_order_release);
            break;
        }
        if (m->opcode == WsOpcode::Text) {
            const std::string_view t = m->text();
            if (t.find("cancel") != std::string_view::npos) {
                cancel_flag.store(true, std::memory_order_release);
                break;
            }
        }
    }
    if (pipe_thread.joinable()) pipe_thread.join();
    (void)ws_send_close(socket_fd);
}

} // namespace

void EchoServer::stop()
{
    p_->stopping.store(true, std::memory_order_release);
    if (p_->listen_fd >= 0) {
        ::shutdown(p_->listen_fd, SHUT_RDWR);
        ::close(p_->listen_fd);
        p_->listen_fd = -1;
    }
}

void EchoServer::serve_forever()
{
    if (p_->listen_fd < 0) return;
    while (!p_->stopping.load(std::memory_order_acquire)) {
        sockaddr_in peer{};
        socklen_t   pl = sizeof(peer);
        const int   fd = ::accept(p_->listen_fd,
                                  reinterpret_cast<sockaddr*>(&peer), &pl);
        if (fd < 0) {
            if (p_->stopping.load(std::memory_order_acquire)) break;
            if (errno == EINTR) continue;
            spdlog::warn("accept: {}", std::strerror(errno));
            break;
        }
        const auto vcfg  = p_->cfg.voice_cfg;
        const auto codec = p_->cfg.codec;
        std::thread t([fd, vcfg, codec]() {
            handle_one_socket(fd, vcfg, codec);
        });
        std::lock_guard<std::mutex> lk(p_->threads_m);
        p_->workers.push_back(std::move(t));
    }
    std::lock_guard<std::mutex> lk(p_->threads_m);
    for (auto& t : p_->workers) {
        if (t.joinable()) t.join();
    }
    p_->workers.clear();
}

std::expected<void, std::string> EchoServer::run()
{
    auto port = listen();
    if (!port) return std::unexpected(port.error());
    spdlog::info("1bit-echo listening on {}:{} codec={}",
                 p_->cfg.bind_host, *port, codec_name(p_->cfg.codec));
    serve_forever();
    return {};
}

void EchoServer::handle_one_for_tests()
{
    if (p_->listen_fd < 0) return;
    sockaddr_in peer{};
    socklen_t   pl = sizeof(peer);
    const int   fd = ::accept(p_->listen_fd,
                              reinterpret_cast<sockaddr*>(&peer), &pl);
    if (fd < 0) return;
    handle_one_socket(fd, p_->cfg.voice_cfg, p_->cfg.codec);
}

} // namespace onebit::echo
