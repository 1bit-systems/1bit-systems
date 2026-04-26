#pragma once

// Audio framing for 1bit-echo's WebSocket gateway.
//
// Kokoro emits 24 kHz mono s16le PCM in a 44-byte RIFF WAV. The browser
// path wants a sample-rate that matches AudioContext defaults (48 kHz)
// and frames small enough to keep latency low (20 ms). The Rust crate
// piped raw frames into libopus; here we own the WAV parse + linear
// 24→48 kHz upsample + 20 ms framing only, and emit raw s16le PCM
// frames. A real Opus encoder is left as a future ticket — adding
// libopus would violate the pure-C++23 deps rule. Wav mode (forwarding
// the RIFF blob verbatim) remains the default.

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace onebit::echo {

inline constexpr std::uint32_t kTargetSr  = 48'000U;
inline constexpr std::uint32_t kFrameMs   = 20U;
// 48000 * 0.020 = 960 samples/frame, 1920 bytes for s16le mono.
inline constexpr std::size_t   kFrameSamples =
    (static_cast<std::size_t>(kTargetSr) * kFrameMs) / 1'000U;

struct WavInfo {
    std::uint32_t sample_rate     = 0;
    std::uint16_t channels        = 0;
    std::uint16_t bits_per_sample = 0;
    std::size_t   data_offset     = 0;
    std::size_t   data_len        = 0;
};

struct CodecError {
    std::string message;
};

// Parse the 44-byte RIFF/WAVE/fmt header kokoro emits. Anything fancier
// (extensible fmt, non-PCM, leading aux chunks) is rejected.
[[nodiscard]] std::expected<WavInfo, CodecError>
parse_wav(const std::uint8_t* data, std::size_t len);

// 16-bit linear-interp resample (mono). Output length tracks the ratio.
[[nodiscard]] std::vector<std::int16_t>
linear_resample(const std::int16_t* in, std::size_t in_n,
                std::uint32_t src_sr, std::uint32_t dst_sr);

// Split a mono RIFF WAV into 20 ms s16le PCM frames at kTargetSr. One
// std::vector<std::uint8_t> per frame, zero-padded tail. Caller chooses
// what to do with each frame (wire-encode as Binary, etc.).
[[nodiscard]] std::expected<std::vector<std::vector<std::uint8_t>>, CodecError>
wav_to_pcm_frames(const std::uint8_t* wav, std::size_t len);

// Build a minimal RIFF WAV blob. Used by tests + as a fixture builder.
[[nodiscard]] std::vector<std::uint8_t>
build_wav(std::uint32_t sample_rate, std::uint16_t channels,
          const std::int16_t* pcm, std::size_t n);

} // namespace onebit::echo
