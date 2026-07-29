// codec_decoder.h — Lightweight C++ audio codec decoder for RVQ-VAE (5.87M params)
//
// Primary path: onnxruntime C API for cross-platform inference.
// Fallback path: returns empty PCM when onnxruntime is not available.
//
// Input:  codec tokens (8 × seq_len int32 codes) + speaker embedding (512 floats)
// Output: 24kHz mono PCM float samples
//
// Guard: USE_ONNXRUNTIME must be defined at build time for real inference.
// Without it, decode() returns an empty vector (no crash, no leak).

#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

struct CodecDecoderImpl;

class CodecDecoder {
public:
    CodecDecoder();
    ~CodecDecoder();

    // Prevent copy
    CodecDecoder(const CodecDecoder&) = delete;
    CodecDecoder& operator=(const CodecDecoder&) = delete;

    /// Load a decoder ONNX model. Returns true on success.
    /// The model should accept:
    ///   input[0]: (int32) codec_tokens [8, seq_len]
    ///   input[1]: (float)  speaker_emb   [512]
    ///   output[0]: (float) audio         [1, output_samples]
    bool load(const std::string& onnx_model_path);

    /// Check if model is loaded and ready
    bool is_loaded() const;

    /// Decode codec tokens to audio waveform.
    /// @param tokens: flattened [8 * seq_len] int32 codes
    /// @param n_tokens: number of code frames (seq_len)
    /// @param speaker_emb: [512] float speaker embedding
    /// @return: float PCM samples at 24kHz (mono)
    std::vector<float> decode(const int32_t* tokens, int n_tokens, const float* speaker_emb);

    /// Get sample rate (always 24000)
    int sample_rate() const { return 24000; }

    /// Get expected number of output samples for given token count
    int expected_output_samples(int n_tokens) const;

    /// Convert float PCM to WAV bytes (mono, 16-bit, 24kHz)
    static std::string pcm_to_wav(const float* samples, size_t num_samples, int sample_rate = 24000);

private:
    std::unique_ptr<CodecDecoderImpl> impl_;
};
