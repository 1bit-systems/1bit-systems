// audio_stream.cpp — Streaming audio codec decoder + WebSocket server.
//
// Two paths for StreamingDecoder:
//   1. ONNX Runtime (USE_ONNXRUNTIME) — frame-by-frame decoding (one codec
//      frame at a time → 312 samples, 13ms @ 24kHz).
//   2. Fallback — decode the full audio with CodecDecoder, then chunk
//      into frames and call the callback.
//
// WebSocketServer: a minimal RFC 6455 WebSocket server using raw POSIX
// sockets.  Runs on a separate port so it doesn't conflict with httplib
// (v0.18.1 has no built-in WebSocket support).  Accepts one connection at
// a time, upgrades it, streams audio frames, then closes.

#include "jarvis/audio_stream.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#include <openssl/sha.h>
#endif

#include "jarvis/codec_tts.h"
#include "codec_decoder.h"

namespace jarvis {

// ── Debug logging ──────────────────────────────────────────────────────
#ifndef NDEBUG
#define STREAM_LOG(fmt, ...) fprintf(stderr, "[audio-stream] " fmt "\n", ##__VA_ARGS__)
#else
#define STREAM_LOG(fmt, ...) ((void)0)
#endif

static constexpr int kSampleRate = 24000;
static constexpr int kFrameSamples = 312; // 13ms @ 24kHz

// ── StreamingDecoderImpl ───────────────────────────────────────────────

struct StreamingDecoderImpl {
    std::unique_ptr<CodecDecoder> decoder;
    bool loaded = false;
};

StreamingDecoder::StreamingDecoder()
    : impl_(std::make_unique<StreamingDecoderImpl>()) {}

StreamingDecoder::~StreamingDecoder() = default;

bool StreamingDecoder::load(const std::string& onnx_model_path) {
    impl_->decoder = std::make_unique<CodecDecoder>();
    if (!impl_->decoder->load(onnx_model_path)) {
        STREAM_LOG("Failed to load decoder from '%s'", onnx_model_path.c_str());
        impl_->decoder.reset();
        return false;
    }
    impl_->loaded = true;
    STREAM_LOG("StreamingDecoder loaded from '%s'", onnx_model_path.c_str());
    return true;
}

bool StreamingDecoder::is_loaded() const {
    return impl_->loaded;
}

void StreamingDecoder::decode_streaming(
    const int32_t* tokens, int n_tokens, const float* speaker_emb,
    AudioChunkCallback callback)
{
    if (!impl_->loaded || !impl_->decoder) {
        STREAM_LOG("Decoder not loaded, cannot stream");
        return;
    }

    // Decode full audio (ONNX Runtime or fallback), then chunk into frames.
    // In the future when the ONNX model supports per-frame inference, this
    // will feed one codec frame at a time.
    auto pcm = impl_->decoder->decode(tokens, n_tokens, speaker_emb);
    if (pcm.empty()) {
        STREAM_LOG("decode() returned empty");
        return;
    }

    // Chunk the PCM into frames and call callback for each, pacing at real-time
    size_t offset = 0;
    while (offset < pcm.size()) {
        int chunk_size = (int)std::min(kFrameSamples, (int)(pcm.size() - offset));
        callback(pcm.data() + offset, chunk_size, kSampleRate);
        offset += chunk_size;

        // Pace at real-time: each frame = samples/sample_rate seconds
        int sleep_ms = (chunk_size * 1000) / kSampleRate;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    }
}

// ── WebSocket helpers ─────────────────────────────────────────────────

// Base64 encode binary data.
static std::string base64_encode(const unsigned char* data, size_t len) {
    static const char b64[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve((len + 2) / 3 * 4);
    for (size_t i = 0; i < len; i += 3) {
        unsigned char a = data[i];
        unsigned char b = (i + 1 < len) ? data[i + 1] : 0;
        unsigned char c = (i + 2 < len) ? data[i + 2] : 0;
        result += b64[a >> 2];
        result += b64[((a & 3) << 4) | (b >> 4)];
        result += (i + 1 < len) ? b64[((b & 0xF) << 2) | (c >> 6)] : '=';
        result += (i + 2 < len) ? b64[c & 0x3F] : '=';
    }
    return result;
}

// Compute the WebSocket accept key (SHA-1 of key + magic GUID → base64).
static std::string compute_accept_key(const std::string& ws_key) {
    static const char* kMagicGUID = "258EAFA5-E914-47DA-95CA-5AB9DC11B85B";
    std::string concat = ws_key + kMagicGUID;
    unsigned char hash[20];

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    SHA1(reinterpret_cast<const unsigned char*>(concat.data()), concat.size(), hash);
#else
    // Minimal SHA-1 (FIPS 180-4, public domain)
    struct {
        uint32_t state[5];
    } ctx;
    ctx.state[0] = 0x67452301;
    ctx.state[1] = 0xEFCDAB89;
    ctx.state[2] = 0x98BADCFE;
    ctx.state[3] = 0x10325476;
    ctx.state[4] = 0xC3D2E1F0;

    auto transform = [](uint32_t state[5], const unsigned char block[64]) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++)
            w[i] = ((uint32_t)block[i*4]<<24)|((uint32_t)block[i*4+1]<<16)|
                   ((uint32_t)block[i*4+2]<<8)|(uint32_t)block[i*4+3];
        for (int i = 16; i < 80; i++)
            w[i] = (w[i-3]^w[i-8]^w[i-14]^w[i-16]), w[i]=(w[i]<<1)|(w[i]>>31);
        uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4];
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i<20)       { f=(b&c)|(~b&d); k=0x5A827999; }
            else if (i<40)  { f=b^c^d;        k=0x6ED9EBA1; }
            else if (i<60)  { f=(b&c)|(b&d)|(c&d); k=0x8F1BBCDC; }
            else            { f=b^c^d;        k=0xCA62C1D6; }
            uint32_t temp=((a<<5)|(a>>27))+f+e+k+w[i];
            e=d; d=c; c=(b<<30)|(b>>2); b=a; a=temp;
        }
        state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d; state[4]+=e;
    };

    // Process full 64-byte blocks
    size_t len = concat.size();
    size_t offset = 0;
    while (len - offset >= 64) {
        transform(ctx.state, (const unsigned char*)(concat.data() + offset));
        offset += 64;
    }

    // Final block with padding
    unsigned char fb[64] = {0};
    size_t rem = len - offset;
    std::memcpy(fb, concat.data() + offset, rem);
    fb[rem] = 0x80;
    if (rem >= 56) { transform(ctx.state, fb); std::memset(fb, 0, 56); }
    uint64_t bits = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) fb[56 + i] = (unsigned char)(bits >> (56 - i * 8));
    transform(ctx.state, fb);

    // Big-endian output
    for (int i = 0; i < 5; i++)
        hash[i*4]=(unsigned char)(ctx.state[i]>>24), hash[i*4+1]=(unsigned char)(ctx.state[i]>>16),
        hash[i*4+2]=(unsigned char)(ctx.state[i]>>8), hash[i*4+3]=(unsigned char)(ctx.state[i]);
#endif

    return base64_encode(hash, 20);
}

// ── Raw WebSocket frame I/O ───────────────────────────────────────────

static bool ws_read_exact(int fd, void* buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t n = read(fd, (char*)buf + total, len - total);
        if (n <= 0) return false;
        total += (size_t)n;
    }
    return true;
}

static bool ws_send_frame(int fd, const void* data, size_t len, uint8_t opcode) {
    uint8_t header[10];
    size_t hlen;
    header[0] = 0x80 | (opcode & 0x0F); // FIN + opcode
    if (len < 126) {
        header[1] = (uint8_t)len; hlen = 2;
    } else if (len <= 0xFFFF) {
        header[1] = 126;
        header[2] = (uint8_t)(len >> 8);
        header[3] = (uint8_t)(len);
        hlen = 4;
    } else {
        header[1] = 127;
        uint64_t n = (uint64_t)len;
        for (int i = 7; i >= 0; i--) header[2 + i] = (uint8_t)(n >> (i * 8));
        hlen = 10;
    }
    if (write(fd, header, hlen) != (ssize_t)hlen) return false;
    if (len > 0 && write(fd, data, len) != (ssize_t)len) return false;
    return true;
}

// Returns: 0 = success with payload, 1 = ping (pong sent), 2 = close, -1 = error
// Payload written to out_payload, opcode written to out_opcode.
static int ws_recv_frame(int fd, std::string* out_payload, uint8_t* out_opcode) {
    uint8_t h[2];
    if (!ws_read_exact(fd, h, 2)) return -1;

    uint8_t opcode = h[0] & 0x0F;
    bool masked = (h[1] & 0x80) != 0;
    uint64_t payload_len = h[1] & 0x7F;

    if (payload_len == 126) {
        uint8_t ext[2];
        if (!ws_read_exact(fd, ext, 2)) return -1;
        payload_len = ((uint64_t)ext[0] << 8) | (uint64_t)ext[1];
    } else if (payload_len == 127) {
        uint8_t ext[8];
        if (!ws_read_exact(fd, ext, 8)) return -1;
        payload_len = 0;
        for (int i = 0; i < 8; i++) payload_len = (payload_len << 8) | (uint64_t)ext[i];
    }

    uint8_t mask_key[4] = {0};
    if (masked && !ws_read_exact(fd, mask_key, 4)) return -1;

    // Handle control frames
    if (opcode == 0x09) { // ping
        ws_send_frame(fd, nullptr, 0, 0x0A); // pong
        if (payload_len > 0) { std::vector<uint8_t> d((size_t)payload_len); ws_read_exact(fd, d.data(), (size_t)payload_len); }
        return 1;
    }
    if (opcode == 0x0A) { // pong
        if (payload_len > 0) { std::vector<uint8_t> d((size_t)payload_len); ws_read_exact(fd, d.data(), (size_t)payload_len); }
        return 1;
    }
    if (opcode == 0x08) { // close
        // Read and discard payload
        if (payload_len > 0) { std::vector<uint8_t> d((size_t)payload_len); ws_read_exact(fd, d.data(), (size_t)payload_len); }
        return 2;
    }

    out_payload->resize((size_t)payload_len);
    if (payload_len > 0 && !ws_read_exact(fd, out_payload->data(), (size_t)payload_len)) return -1;

    if (masked) {
        for (size_t i = 0; i < (size_t)payload_len; i++)
            (*out_payload)[i] ^= mask_key[i & 3];
    }

    if (out_opcode) *out_opcode = opcode;
    return 0;
}

// ── WebSocket upgrade handler ─────────────────────────────────────────
//
// Handles one WebSocket connection: performs upgrade handshake, streams
// audio frames, cleans up.

static void handle_ws_connection(int client_fd, void* codec_tts_ptr) {
    if (client_fd < 0) return;

    // ── Read HTTP upgrade request ────────────────────────────
    std::string request;
    char buf[4096];
    ssize_t n;
    while ((n = read(client_fd, buf, sizeof(buf))) > 0) {
        request.append(buf, (size_t)n);
        if (request.find("\r\n\r\n") != std::string::npos) break;
        if (request.size() > 8192) break; // safety limit
    }

    if (request.empty()) {
        close(client_fd);
        return;
    }

    // ── Parse headers ────────────────────────────────────────
    // Crude HTTP header parser — good enough for the WebSocket upgrade
    std::string ws_key;
    std::string path;
    bool has_upgrade = false;

    // Parse first line: GET /path?query HTTP/1.1
    auto first_line_end = request.find("\r\n");
    if (first_line_end != std::string::npos) {
        auto first_line = request.substr(0, first_line_end);
        auto space1 = first_line.find(' ');
        auto space2 = first_line.rfind(' ');
        if (space1 != std::string::npos && space2 != std::string::npos && space2 > space1) {
            path = first_line.substr(space1 + 1, space2 - space1 - 1);
        }
    }

    // Parse headers
    auto header_start = request.find("\r\n") + 2;
    auto header_end = request.find("\r\n\r\n");
    if (header_end == std::string::npos) { close(client_fd); return; }

    auto headers_section = request.substr(header_start, header_end - header_start);
    size_t pos = 0;
    while (pos < headers_section.size()) {
        auto line_end = headers_section.find("\r\n", pos);
        if (line_end == std::string::npos) break;
        auto line = headers_section.substr(pos, line_end - pos);
        auto colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = line.substr(0, colon);
            std::string value = line.substr(colon + 1);
            // Trim
            while (!value.empty() && (value[0] == ' ' || value[0] == '\t')) value = value.substr(1);

            if (key == "Upgrade" && (value == "websocket" || value == "WebSocket"))
                has_upgrade = true;
            if (key == "Sec-WebSocket-Key")
                ws_key = value;
        }
        pos = line_end + 2;
    }

    // ── Extract voice and text from query params ─────────────
    std::string voice, text;
    auto qmark = path.find('?');
    if (qmark != std::string::npos) {
        auto query = path.substr(qmark + 1);
        size_t qpos = 0;
        while (qpos < query.size()) {
            auto amp = query.find('&', qpos);
            auto seg = (amp != std::string::npos) ? query.substr(qpos, amp - qpos) : query.substr(qpos);
            auto eq = seg.find('=');
            if (eq != std::string::npos) {
                std::string k = seg.substr(0, eq);
                std::string v = seg.substr(eq + 1);
                // URL decode: replace + with space, %XX decode
                std::string decoded;
                for (size_t i = 0; i < v.size(); i++) {
                    if (v[i] == '+') decoded += ' ';
                    else if (v[i] == '%' && i + 2 < v.size()) {
                        unsigned int c;
                        sscanf(v.substr(i+1, 2).c_str(), "%x", &c);
                        decoded += (char)c;
                        i += 2;
                    } else decoded += v[i];
                }
                if (k == "voice") voice = decoded;
                else if (k == "text") text = decoded;
            }
            if (amp == std::string::npos) break;
            qpos = amp + 1;
        }
    }

    if (!has_upgrade || ws_key.empty() || voice.empty() || text.empty()) {
        // Send HTTP error response
        const char* err = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        write(client_fd, err, strlen(err));
        close(client_fd);
        return;
    }

    // ── Send 101 Switching Protocols ─────────────────────────
    {
        std::string accept = compute_accept_key(ws_key);
        std::string response =
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Accept: " + accept + "\r\n"
            "\r\n";
        if (write(client_fd, response.data(), response.size()) != (ssize_t)response.size()) {
            close(client_fd);
            return;
        }
    }

    // Set TCP_NODELAY for low-latency streaming
    int flag = 1;
    setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    // ── Send metadata frame ──────────────────────────────────
    std::string meta = R"({"type":"meta","sample_rate":24000,"channels":1,"format":"float32"})";
    if (!ws_send_frame(client_fd, meta.data(), meta.size(), 0x01)) { // text frame
        close(client_fd);
        return;
    }

    // ── Stream audio ─────────────────────────────────────────
    if (!codec_tts_ptr) {
        std::string err = R"({"type":"end","reason":"no TTS engine"})";
        ws_send_frame(client_fd, err.data(), err.size(), 0x01);
        close(client_fd);
        return;
    }

    auto* tts = static_cast<jarvis::CodecTts*>(codec_tts_ptr);

    if (!tts->has_voice(voice)) {
        std::string err = R"({"type":"end","reason":"voice pack not found"})";
        ws_send_frame(client_fd, err.data(), err.size(), 0x01);
        close(client_fd);
        return;
    }

    // Synthesize full audio, then stream in chunks
    std::string wav = tts->synthesize(text, voice);
    if (wav.empty()) {
        std::string err = R"({"type":"end","reason":"synthesis failed"})";
        ws_send_frame(client_fd, err.data(), err.size(), 0x01);
        close(client_fd);
        return;
    }

    // Parse WAV header to find PCM data
    if (wav.size() < 44) {
        std::string err = R"({"type":"end","reason":"invalid WAV"})";
        ws_send_frame(client_fd, err.data(), err.size(), 0x01);
        close(client_fd);
        return;
    }

    // Find "data" chunk
    size_t pcm_offset = 0;
    size_t pcm_size = 0;
    size_t data_start = 12;
    while (data_start + 8 <= wav.size()) {
        uint32_t chunk_size = *(const uint32_t*)(wav.data() + data_start + 4);
        if (wav.substr(data_start, 4) == "data") {
            pcm_offset = data_start + 8;
            pcm_size = (size_t)chunk_size;
            break;
        }
        data_start += 8 + (size_t)chunk_size;
    }

    if (pcm_offset == 0 || pcm_offset >= wav.size()) {
        std::string err = R"({"type":"end","reason":"no PCM data in WAV"})";
        ws_send_frame(client_fd, err.data(), err.size(), 0x01);
        close(client_fd);
        return;
    }

    size_t num_samples = pcm_size / 2; // S16LE, mono
    const int16_t* s16 = reinterpret_cast<const int16_t*>(wav.data() + pcm_offset);

    // Set socket receive timeout so we can check for cancel
    timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 50000; // 50ms
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    size_t sample_offset = 0;
    bool cancelled = false;

    while (sample_offset < num_samples) {
        // Check for cancel frame (non-blocking due to SO_RCVTIMEO)
        std::string incoming;
        uint8_t opcode;
        int ret = ws_recv_frame(client_fd, &incoming, &opcode);
        if (ret == 2) { // close frame
            break;
        }
        if (ret == 0 && opcode == 0x01) { // text frame
            if (incoming.find("cancel") != std::string::npos) {
                cancelled = true;
                std::string done = R"({"type":"end","reason":"cancelled"})";
                ws_send_frame(client_fd, done.data(), done.size(), 0x01);
                break;
            }
        }

        // Send audio chunk
        size_t chunk_samples = std::min((size_t)kFrameSamples, num_samples - sample_offset);
        std::vector<float> float_chunk(chunk_samples);
        for (size_t i = 0; i < chunk_samples; i++)
            float_chunk[i] = s16[sample_offset + i] / 32768.0f;

        if (!ws_send_frame(client_fd, float_chunk.data(), chunk_samples * sizeof(float), 0x02)) {
            STREAM_LOG("Client disconnected during send");
            break;
        }

        sample_offset += chunk_samples;

        // Pace at real-time
        int sleep_ms = (int)(chunk_samples * 1000 / kSampleRate);
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    }

    if (!cancelled) {
        std::string done = R"({"type":"end","reason":"done"})";
        ws_send_frame(client_fd, done.data(), done.size(), 0x01);
    }

    close(client_fd);
}

// ── WebSocketServer implementation ────────────────────────────────────

struct WebSocketServerImpl {
    // All state is in the owning WebSocketServer's fields for thread safety
};

WebSocketServer::WebSocketServer() = default;
WebSocketServer::~WebSocketServer() { stop(); }

int WebSocketServer::start(int port, void* codec_tts_ptr) {
    if (running_) return port_;

    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        STREAM_LOG("Failed to create WebSocket server socket");
        return -1;
    }

    // Reuse address
    int opt = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // localhost only
    addr.sin_port = htons((uint16_t)port);

    if (bind(listen_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        STREAM_LOG("Failed to bind WebSocket server to port %d", port);
        close(listen_fd_);
        listen_fd_ = -1;
        return -1;
    }

    if (listen(listen_fd_, 1) < 0) { // backlog=1, we handle one at a time
        STREAM_LOG("Failed to listen on WebSocket port");
        close(listen_fd_);
        listen_fd_ = -1;
        return -1;
    }

    // Get actual port (in case port was 0)
    struct sockaddr_in bound_addr;
    socklen_t addr_len = sizeof(bound_addr);
    if (getsockname(listen_fd_, (struct sockaddr*)&bound_addr, &addr_len) == 0) {
        port_ = ntohs(bound_addr.sin_port);
    } else {
        port_ = port;
    }

    running_ = true;

    // Store the codec_tts pointer for the thread
    auto* tts = static_cast<jarvis::CodecTts*>(codec_tts_ptr);

    server_thread_ = std::make_unique<std::thread>([this, tts]() {
        STREAM_LOG("WebSocket server started on port %d", port_);

        while (running_ && listen_fd_ >= 0) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            int client_fd = accept(listen_fd_, (struct sockaddr*)&client_addr, &client_len);

            if (client_fd < 0) {
                if (errno == EINTR || errno == EAGAIN) continue;
                if (!running_) break;
                STREAM_LOG("WebSocket accept failed: %s", strerror(errno));
                continue;
            }

            STREAM_LOG("WebSocket client connected");
            handle_ws_connection(client_fd, tts);
            STREAM_LOG("WebSocket client disconnected");
        }

        STREAM_LOG("WebSocket server stopped");
    });

    return port_;
}

void WebSocketServer::stop() {
    running_ = false;
    if (listen_fd_ >= 0) {
        shutdown(listen_fd_, SHUT_RDWR);
        close(listen_fd_);
        listen_fd_ = -1;
    }
    if (server_thread_ && server_thread_->joinable()) {
        server_thread_->join();
        server_thread_.reset();
    }
}

} // namespace jarvis
