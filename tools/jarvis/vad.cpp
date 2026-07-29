// vad.cpp — Energy-based Voice Activity Detection implementation.
// Simple RMS energy threshold with debounce and lookback/ramp-down periods.
// No external dependencies — pure C++23 with standard library.

#include "jarvis/vad.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <vector>

namespace jarvis {

// ── Constants ──────────────────────────────────────────────────────────

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Implementation struct ──────────────────────────────────────────────

struct VAD::Impl {
    VADConfig config;

    // Ring buffer of recent RMS energy values (one per frame)
    std::deque<float> energy_history;

    // Audio sample ring buffer for lookback/ramp-down
    std::deque<float> audio_ring;

    // Current speech state
    bool speaking = false;

    // Counters (in frames)
    int speech_frames = 0;       // consecutive frames above threshold
    int silence_frames = 0;      // consecutive frames below threshold
    int ramp_up_frames = 0;      // frames to go back on speech start
    int ramp_down_frames = 0;    // frames to hold after speech end
    int ramp_down_remaining = 0; // frames left in ramp-down

    // Speech buffer — accumulates audio while speaking (+ ramp-up)
    std::vector<float> speech_buffer;

    // Last completed utterance
    std::vector<float> last_utterance;

    // Callback
    VADCallback callback;

    // Pre-computed frame sizes
    int frame_samples;  // samples per frame
    int min_speech_frames;
    int min_silence_frames;
    int ramp_up_samples;
    int ramp_down_samples;

    explicit Impl(VADConfig cfg)
        : config(std::move(cfg))
        , frame_samples(cfg.sample_rate * cfg.frame_ms / 1000)
        , min_speech_frames((int)(cfg.min_speech_ms / cfg.frame_ms))
        , min_silence_frames((int)(cfg.min_silence_ms / cfg.frame_ms))
        , ramp_up_samples((int)(cfg.ramp_up_ms * cfg.sample_rate / 1000))
        , ramp_down_samples((int)(cfg.ramp_down_ms * cfg.sample_rate / 1000))
    {
        if (frame_samples < 1) frame_samples = 1;
        if (min_speech_frames < 1) min_speech_frames = 1;
        if (min_silence_frames < 1) min_silence_frames = 1;
    }

    // Compute RMS energy of a block of samples
    static float compute_energy(const float* samples, int n) {
        if (n <= 0) return 0.0f;
        double sum_sq = 0.0;
        for (int i = 0; i < n; i++) {
            sum_sq += (double)samples[i] * (double)samples[i];
        }
        return (float)std::sqrt(sum_sq / n);
    }

    void process_frame(const float* frame_samples_ptr) {
        float energy = compute_energy(frame_samples_ptr, frame_samples);

        // Store energy in history (keep last 100 frames)
        energy_history.push_back(energy);
        if (energy_history.size() > 100) {
            energy_history.pop_front();
        }

        // Store audio in ring buffer (for lookback/ramp-down)
        for (int i = 0; i < frame_samples; i++) {
            audio_ring.push_back(frame_samples_ptr[i]);
        }
        // Keep maximum of ramp_up_samples + ramp_down_samples + some extra
        size_t max_ring = (size_t)(ramp_up_samples + ramp_down_samples + frame_samples * 10);
        while (audio_ring.size() > max_ring) {
            audio_ring.pop_front();
        }

        if (energy >= config.threshold) {
            // Above threshold
            silence_frames = 0;
            speech_frames++;

            if (!speaking && speech_frames >= min_speech_frames) {
                // Speech just started (debounced)
                speaking = true;
                ramp_down_remaining = 0;

                // Build speech buffer with ramp-up lookback
                speech_buffer.clear();
                size_t lookback = (size_t)ramp_up_samples;
                if (lookback > audio_ring.size()) lookback = audio_ring.size();
                auto start_it = audio_ring.end() - lookback;
                speech_buffer.insert(speech_buffer.end(), start_it, audio_ring.end());

                if (callback) callback(true);
            }

            // Accumulate audio into speech buffer
            if (speaking) {
                speech_buffer.insert(speech_buffer.end(),
                                     frame_samples_ptr,
                                     frame_samples_ptr + frame_samples);
            }
        } else {
            // Below threshold
            speech_frames = 0;

            if (speaking) {
                silence_frames++;

                // Still accumulating during ramp-down
                if (ramp_down_remaining > 0) {
                    speech_buffer.insert(speech_buffer.end(),
                                         frame_samples_ptr,
                                         frame_samples_ptr + frame_samples);
                    ramp_down_remaining -= frame_samples;
                }

                if (silence_frames >= min_silence_frames && ramp_down_remaining <= 0) {
                    // Speech ended
                    speaking = false;
                    last_utterance = speech_buffer;

                    if (callback) callback(false);
                }

                // Set ramp-down on first silence frame after speech
                if (silence_frames == 1) {
                    ramp_down_remaining = ramp_down_samples + frame_samples; // include this frame
                }
            }
        }
    }
};

// ── Public API ─────────────────────────────────────────────────────────

VAD::VAD(VADConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
VAD::~VAD() = default;

void VAD::process(const float* samples, int num_samples) {
    if (!samples || num_samples <= 0) return;

    int frame_sz = impl_->frame_samples;
    int offset = 0;

    while (offset + frame_sz <= num_samples) {
        impl_->process_frame(samples + offset);
        offset += frame_sz;
    }
}

void VAD::reset() {
    impl_->energy_history.clear();
    impl_->audio_ring.clear();
    impl_->speaking = false;
    impl_->speech_frames = 0;
    impl_->silence_frames = 0;
    impl_->ramp_up_frames = 0;
    impl_->ramp_down_frames = 0;
    impl_->ramp_down_remaining = 0;
    impl_->speech_buffer.clear();
    impl_->last_utterance.clear();
}

bool VAD::is_speaking() const {
    return impl_->speaking;
}

std::vector<float> VAD::get_speech_buffer() const {
    return impl_->speech_buffer;
}

std::vector<float> VAD::get_last_utterance() const {
    return impl_->last_utterance;
}

} // namespace jarvis
