#include "onebit/echo/codec.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace onebit::echo {

namespace {

[[nodiscard]] std::uint16_t le16(const std::uint8_t* p)
{
    return static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(p[0]) |
        (static_cast<std::uint16_t>(p[1]) << 8));
}
[[nodiscard]] std::uint32_t le32(const std::uint8_t* p)
{
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

void put_le16(std::vector<std::uint8_t>& out, std::uint16_t v)
{
    out.push_back(static_cast<std::uint8_t>(v & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xFFU));
}
void put_le32(std::vector<std::uint8_t>& out, std::uint32_t v)
{
    out.push_back(static_cast<std::uint8_t>(v & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((v >> 16) & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((v >> 24) & 0xFFU));
}

} // namespace

std::expected<WavInfo, CodecError>
parse_wav(const std::uint8_t* d, std::size_t n)
{
    if (n < 44) {
        return std::unexpected(
            CodecError{"wav too short: " + std::to_string(n) + " bytes"});
    }
    if (std::memcmp(d, "RIFF", 4) != 0 ||
        std::memcmp(d + 8, "WAVE", 4) != 0 ||
        std::memcmp(d + 12, "fmt ", 4) != 0) {
        return std::unexpected(CodecError{"not a RIFF/WAVE/fmt file"});
    }
    const std::uint16_t fmt = le16(d + 20);
    if (fmt != 1) {
        return std::unexpected(
            CodecError{"unsupported WAV format " + std::to_string(fmt)});
    }
    if (std::memcmp(d + 36, "data", 4) != 0) {
        return std::unexpected(CodecError{"expected `data` at offset 36"});
    }
    WavInfo w;
    w.channels        = le16(d + 22);
    w.sample_rate     = le32(d + 24);
    w.bits_per_sample = le16(d + 34);
    w.data_offset     = 44;
    w.data_len        = static_cast<std::size_t>(le32(d + 40));
    return w;
}

std::vector<std::int16_t>
linear_resample(const std::int16_t* in, std::size_t n,
                std::uint32_t src_sr, std::uint32_t dst_sr)
{
    if (n == 0 || src_sr == dst_sr) {
        return std::vector<std::int16_t>(in, in + n);
    }
    const double      ratio   = static_cast<double>(dst_sr) / src_sr;
    const std::size_t out_len =
        static_cast<std::size_t>(static_cast<double>(n) * ratio);
    const std::size_t last    = n - 1;
    std::vector<std::int16_t> out(out_len, 0);
    for (std::size_t j = 0; j < out_len; ++j) {
        const double      src_pos = static_cast<double>(j) / ratio;
        const std::size_t idx     = static_cast<std::size_t>(src_pos);
        const double      frac    = src_pos - static_cast<double>(idx);
        const double      a = in[std::min(idx, last)];
        const double      b = in[std::min(idx + 1, last)];
        const double      v = std::clamp(a + (b - a) * frac, -32768.0, 32767.0);
        out[j] = static_cast<std::int16_t>(v);
    }
    return out;
}

std::expected<std::vector<std::vector<std::uint8_t>>, CodecError>
wav_to_pcm_frames(const std::uint8_t* wav, std::size_t len)
{
    auto info = parse_wav(wav, len);
    if (!info) return std::unexpected(info.error());
    if (info->channels != 1) {
        return std::unexpected(CodecError{
            "expected mono, got " + std::to_string(info->channels)});
    }
    if (info->bits_per_sample != 16) {
        return std::unexpected(CodecError{
            "expected 16-bit PCM, got " +
            std::to_string(info->bits_per_sample)});
    }
    const std::size_t end =
        std::min(info->data_offset + info->data_len, len);
    const std::size_t pcm_bytes = end - info->data_offset;
    const std::size_t n         = pcm_bytes / 2;

    std::vector<std::int16_t> samples(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint8_t lo = wav[info->data_offset + i * 2];
        const std::uint8_t hi = wav[info->data_offset + i * 2 + 1];
        samples[i] =
            static_cast<std::int16_t>(static_cast<std::uint16_t>(lo) |
                                      (static_cast<std::uint16_t>(hi) << 8));
    }
    auto upsampled =
        linear_resample(samples.data(), samples.size(),
                        info->sample_rate, kTargetSr);

    std::vector<std::vector<std::uint8_t>> frames;
    if (upsampled.empty()) return frames;
    frames.reserve(upsampled.size() / kFrameSamples + 1);
    const std::size_t frame_bytes = kFrameSamples * 2;
    for (std::size_t i = 0; i < upsampled.size(); i += kFrameSamples) {
        std::vector<std::uint8_t> f(frame_bytes, 0);
        const std::size_t take =
            std::min<std::size_t>(kFrameSamples, upsampled.size() - i);
        for (std::size_t k = 0; k < take; ++k) {
            const std::int16_t v = upsampled[i + k];
            f[k * 2]     = static_cast<std::uint8_t>(v & 0xFF);
            f[k * 2 + 1] = static_cast<std::uint8_t>((v >> 8) & 0xFF);
        }
        frames.push_back(std::move(f));
    }
    return frames;
}

std::vector<std::uint8_t>
build_wav(std::uint32_t sample_rate, std::uint16_t channels,
          const std::int16_t* pcm, std::size_t n)
{
    const std::uint32_t byte_rate =
        sample_rate * static_cast<std::uint32_t>(channels) * 2U;
    const std::uint32_t data_len = static_cast<std::uint32_t>(n * 2U);
    std::vector<std::uint8_t> out;
    out.reserve(44 + n * 2U);
    out.insert(out.end(), {'R', 'I', 'F', 'F'});
    put_le32(out, 36U + data_len);
    out.insert(out.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
    put_le32(out, 16U);
    put_le16(out, 1U);
    put_le16(out, channels);
    put_le32(out, sample_rate);
    put_le32(out, byte_rate);
    put_le16(out, static_cast<std::uint16_t>(channels * 2U));
    put_le16(out, 16U);
    out.insert(out.end(), {'d', 'a', 't', 'a'});
    put_le32(out, data_len);
    for (std::size_t i = 0; i < n; ++i) {
        const std::int16_t v = pcm[i];
        out.push_back(static_cast<std::uint8_t>(v & 0xFF));
        out.push_back(static_cast<std::uint8_t>((static_cast<std::uint16_t>(v) >> 8) & 0xFF));
    }
    return out;
}

} // namespace onebit::echo
