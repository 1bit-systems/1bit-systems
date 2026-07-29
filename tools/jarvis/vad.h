// vad.h — Voice Activity Detection using energy-based threshold.
// Simple, no external dependencies. Operates on 16kHz mono float32 audio.
#pragma once

#include <cstdint>
#include <vector>
#include <functional>
#include <memory>

namespace jarvis {

/// Callback when speech starts/stops
using VADCallback = std::function<void(bool is_speech)>;

struct VADConfig {
    int sample_rate = 16000;       // VAD operates at 16kHz
    int frame_ms = 20;             // frame size in ms
    float threshold = 0.01f;       // energy threshold (RMS)
    float min_speech_ms = 200.0f;  // minimum speech duration to trigger
    float min_silence_ms = 500.0f; // silence duration to trigger end-of-speech
    float ramp_up_ms = 300.0f;     // lookback after speech start (include preceding audio)
    float ramp_down_ms = 200.0f;   // hold after speech end (keep including trailing audio)
};

class VAD {
public:
    VAD(VADConfig config = VADConfig{});
    ~VAD();

    // Process audio samples (float32, mono)
    // Calls callback when speech/silence transitions occur
    void process(const float* samples, int num_samples);

    // Reset all internal state
    void reset();

    // Current state
    bool is_speaking() const;

    // Get the most recent speech buffer (everything captured since speech started,
    // including ramp-up lookback). Returns empty if not currently speaking.
    std::vector<float> get_speech_buffer() const;

    // Get the last completed utterance (cleared when next utterance starts)
    std::vector<float> get_last_utterance() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace jarvis
