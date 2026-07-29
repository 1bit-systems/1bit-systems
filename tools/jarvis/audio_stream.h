// audio_stream.h — Real-time streaming audio codec decoder + WebSocket server.
//
// StreamingDecoder generates audio frame-by-frame (13ms frames at 24kHz =
// 312 samples/frame) from codec tokens.  Uses ONNX Runtime when available
// (USE_ONNXRUNTIME), falls back to frame-chunking the full decode.
//
// WebSocketServer runs a minimal WebSocket server on a separate port (so it
// doesn't conflict with httplib which lacks built-in WebSocket support in
// v0.18.1).  jarvis_server registers /v1/audio/stream as a redirect / info
// endpoint pointing to this server.

#pragma once
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>

namespace jarvis {

/// Callback type for streaming audio chunks
using AudioChunkCallback = std::function<void(const float* samples, int num_samples, int sample_rate)>;

/// Streaming audio codec decoder.
/// Generates audio frame-by-frame (13ms frames at 24kHz = 312 samples).
struct StreamingDecoderImpl;

class StreamingDecoder {
public:
    StreamingDecoder();
    ~StreamingDecoder();

    /// Load decoder model
    bool load(const std::string& onnx_model_path);

    /// Start decoding codec tokens, calling callback for each audio frame.
    /// @param tokens: codec token indices [n_codebooks, seq_len]
    /// @param n_tokens: number of codec frames
    /// @param speaker_emb: [512] float speaker embedding
    /// @param callback: called with each audio chunk as it's decoded
    void decode_streaming(
        const int32_t* tokens, int n_tokens, const float* speaker_emb,
        AudioChunkCallback callback
    );

    bool is_loaded() const;
    int sample_rate() const { return 24000; }

private:
    std::unique_ptr<StreamingDecoderImpl> impl_;
};

// ── WebSocket audio stream server ────────────────────────────────────
//
// Implemented using raw POSIX sockets because httplib v0.18.1 does not
// include built-in WebSocket support.  Runs on a configurable port
// (default 8082).  jarvis_server starts it on a background thread.
//
// Protocol (RFC 6455):
//   1. Client sends HTTP GET upgrade with Sec-WebSocket-Key
//   2. Server replies 101 Switching Protocols
//   3. Server sends text frame: {"type":"meta","sample_rate":24000,...}
//   4. Server sends binary frames: raw float32 PCM (~312 samples = 13ms)
//   5. Server sends text frame: {"type":"end","reason":"done"}
//   6. Client can send text frame: {"type":"cancel"} to abort

struct WebSocketServerImpl;

class WebSocketServer {
public:
    WebSocketServer();
    ~WebSocketServer();

    /// Start the WebSocket server on a background thread.
    /// @param port  listening port (0 = auto-assign)
    /// @param codec_tts_ptr  pointer to a jarvis::CodecTts instance (void* to avoid include)
    /// @return the actual port number, or -1 on failure
    int start(int port, void* codec_tts_ptr);

    /// Stop the server and join the background thread.
    void stop();

    /// Check if the server is running.
    bool is_running() const { return running_; }

    /// Get the port the server is listening on.
    int port() const { return port_; }

private:
    std::atomic<bool> running_{false};
    std::atomic<int> port_{-1};
    int listen_fd_{-1};
    std::unique_ptr<std::thread> server_thread_;
};

} // namespace jarvis
