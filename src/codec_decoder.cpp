// codec_decoder.cpp — Audio codec decoder implementation
//
// Uses onnxruntime C++ API (Ort::Session, Ort::Value, etc.) when
// USE_ONNXRUNTIME is defined at build time. Falls back to a no-op
// stub that returns empty PCM — safe to ship in builds without
// onnxruntime, the caller handles the empty result as "TTS unavailable".
//
// Thread safety: load() is not thread-safe (call once at init). decode()
// is thread-safe — each call creates its own Ort::Value tensors
// on the stack and does not mutate the session.

#include "codec_decoder.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef NDEBUG
#define CODEC_LOG(fmt, ...) fprintf(stderr, "[codec] " fmt "\n", ##__VA_ARGS__)
#else
#define CODEC_LOG(fmt, ...) ((void)0)
#endif

// ── ONNX Runtime path — only compiled when USE_ONNXRUNTIME is set ──────
#ifdef USE_ONNXRUNTIME

#include <onnxruntime_cxx_api.h>

struct CodecDecoderImpl {
    std::string model_path;
    Ort::Env env{nullptr};
    Ort::Session session{nullptr};
    bool loaded{false};

    ~CodecDecoderImpl() = default;
};

bool CodecDecoder::load(const std::string& onnx_model_path) {
    try {
        impl_ = std::make_unique<CodecDecoderImpl>();
        impl_->model_path = onnx_model_path;

        // Create ONNX Runtime environment and session using C++ API
        impl_->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "codec_decoder");

        Ort::SessionOptions session_opts;
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_opts.SetIntraOpNumThreads(4);

        // CUDA EP available when built with CUDA; fine if not — CPU works
        CODEC_LOG("Using CPU execution provider");

        // Load model
        impl_->session = Ort::Session(impl_->env, onnx_model_path.c_str(), session_opts);
        impl_->loaded = true;
        CODEC_LOG("Loaded codec decoder from '%s'", onnx_model_path.c_str());
        return true;

    } catch (const Ort::Exception& e) {
        CODEC_LOG("ORT Exception during load: %s", e.what());
        impl_.reset();
        return false;
    }
}

bool CodecDecoder::is_loaded() const {
    return impl_ && impl_->loaded;
}

int CodecDecoder::expected_output_samples(int n_tokens) const {
    (void)n_tokens;
    if (!impl_) return 0;
    // The decoder produces one output tensor per inference call.
    // Output shape is [1, output_samples]. We return the full output
    // size since the model processes all codec frames at once.
    try {
        auto output_type_info = impl_->session.GetOutputTypeInfo(0);
        auto tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        if (shape.size() >= 2 && shape[1] > 0) return (int)shape[1];
    } catch (...) {}
    return 0;
}

std::vector<float> CodecDecoder::decode(const int32_t* tokens, int n_tokens, const float* speaker_emb) {
    if (!impl_ || !impl_->loaded) {
        CODEC_LOG("decode() called but model is not loaded");
        return {};
    }

    try {
        constexpr int64_t kNumCodebooks = 8;
        constexpr int64_t kSpeakerEmbDim = 512;

        // Input tensor shapes
        std::vector<int64_t> token_shape = {kNumCodebooks, n_tokens};
        std::vector<int64_t> speaker_shape = {kSpeakerEmbDim};

        // Create input tensors using C++ API
        auto token_tensor = Ort::Value::CreateTensor<int32_t>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
            const_cast<int32_t*>(tokens),
            (size_t)(kNumCodebooks * n_tokens),
            token_shape.data(), token_shape.size());

        auto speaker_tensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
            const_cast<float*>(speaker_emb),
            (size_t)kSpeakerEmbDim,
            speaker_shape.data(), speaker_shape.size());

        // Run inference
        Ort::RunOptions run_opts;
        const char* input_names[] = {"codec_tokens", "speaker_emb"};
        Ort::Value input_tensors[] = {std::move(token_tensor), std::move(speaker_tensor)};
        const char* output_names[] = {"audio"};

        auto output_tensors = impl_->session.Run(
            run_opts,
            input_names, input_tensors, 2,
            output_names, 1);

        if (output_tensors.empty()) {
            CODEC_LOG("Run() returned empty output");
            return {};
        }

        // Extract output data
        auto& output_tensor = output_tensors[0];
        auto type_info = output_tensor.GetTensorTypeAndShapeInfo();
        auto elem_count = type_info.GetElementCount();
        float* output_data = output_tensor.GetTensorMutableData<float>();

        if (!output_data || elem_count == 0) {
            CODEC_LOG("Output tensor has no data");
            return {};
        }

        return std::vector<float>(output_data, output_data + elem_count);

    } catch (const Ort::Exception& e) {
        CODEC_LOG("ORT decode exception: %s", e.what());
        return {};
    }
}

// ── Fallback path: ONNX Runtime not available ──────────────────────────
#else

struct CodecDecoderImpl {
    // Minimal stub — nothing to do
};

bool CodecDecoder::load(const std::string& onnx_model_path) {
    (void)onnx_model_path;
    CODEC_LOG("ONNX Runtime not available (USE_ONNXRUNTIME not defined)");
    impl_ = std::make_unique<CodecDecoderImpl>();
    return false;
}

bool CodecDecoder::is_loaded() const { return false; }

int CodecDecoder::expected_output_samples(int n_tokens) const {
    (void)n_tokens;
    return 0;
}

std::vector<float> CodecDecoder::decode(const int32_t* tokens, int n_tokens, const float* speaker_emb) {
    (void)tokens; (void)n_tokens; (void)speaker_emb;
    CODEC_LOG("ONNX Runtime not available — decode() returns empty");
    return {};
}

#endif // USE_ONNXRUNTIME

// ── Common: constructor / destructor / pcm_to_wav ──────────────────────

CodecDecoder::CodecDecoder() = default;
CodecDecoder::~CodecDecoder() = default;

std::string CodecDecoder::pcm_to_wav(const float* samples, size_t num_samples, int sample_rate) {
    if (!samples || num_samples == 0) return {};

    std::vector<int16_t> pcm(num_samples);
    for (size_t i = 0; i < num_samples; i++) {
        float clamped = samples[i];
        if (clamped < -1.0f) clamped = -1.0f;
        if (clamped > 1.0f) clamped = 1.0f;
        pcm[i] = (int16_t)(clamped * 32767.0f);
    }

    const int channels = 1;
    const int bits_per_sample = 16;
    uint32_t data_size = (uint32_t)(pcm.size() * sizeof(int16_t));
    uint32_t byte_rate = (uint32_t)(sample_rate * channels * bits_per_sample / 8);
    uint16_t block_align = (uint16_t)(channels * bits_per_sample / 8);
    uint32_t riff_size = 36 + data_size;

    std::string out;
    out.reserve(44 + data_size);

    auto put_u32 = [&](uint32_t v) { out.append(reinterpret_cast<const char*>(&v), 4); };
    auto put_u16 = [&](uint16_t v) { out.append(reinterpret_cast<const char*>(&v), 2); };

    out += "RIFF"; put_u32(riff_size); out += "WAVE";
    out += "fmt "; put_u32(16); put_u16(1); put_u16((uint16_t)channels);
    put_u32((uint32_t)sample_rate); put_u32(byte_rate); put_u16(block_align);
    put_u16((uint16_t)bits_per_sample);
    out += "data"; put_u32(data_size);
    out.append(reinterpret_cast<const char*>(pcm.data()), data_size);

    return out;
}
