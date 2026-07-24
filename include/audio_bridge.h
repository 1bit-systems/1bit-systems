// audio_bridge.h — audio.cpp integration layer
//
// Embeds audio.cpp as a submodule and exposes TTS, STT, music generation,
// voice cloning, VAD, and source separation through native C++ APIs and
// OpenAI-compatible endpoints.
//
// Design:
//   - wraps audio.cpp's model loaders and inference paths
//   - replaces the current Piper fork/exec in jarvis
//   - single unified API for all audio tasks
//   - GGUF-first model loading, same as 1BP / stable-diffusion.cpp

#pragma once

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <cstdint>

// ─── Audio task types ─────────────────────────────────────────────
enum class AudioTask : uint8_t {
    TTS         = 0,  // Text-to-Speech
    STT_ASR     = 1,  // Speech-to-Text / ASR
    VOICE_CLONE = 2,  // Voice cloning
    VOICE_CONV  = 3,  // Voice conversion
    MUSIC_GEN   = 4,  // Music generation
    VAD         = 5,  // Voice Activity Detection
    DIARIZATION = 6,  // Speaker diarization
    SEPARATION  = 7,  // Source separation
    SFX         = 8,  // Sound effects generation
    ALIGNMENT   = 9,  // Forced alignment
};

// ─── Audio model families (mapping to audio.cpp) ──────────────────
enum class AudioModelFamily : uint8_t {
    // TTS
    POCKET_TTS     = 0,  // PocketTTS-100M
    QWEN3_TTS      = 1,  // Qwen3-TTS 0.6B/1.7B
    VIBEVOICE      = 2,  // VibeVoice 1.5B/7B
    OMNI_VOICE     = 3,  // OmniVoice (Qwen3-0.6B based)
    FISH_AUDIO     = 4,  // Fish Audio S2 Pro
    SUPERTONIC     = 5,  // Supertonic 3
    VOXCPM2        = 6,  // VoxCPM2-2B
    CHATTERBOX     = 7,  // Chatterbox 0.5B
    MIOTTS         = 8,  // MioTTS-1.7B
    INDEX_TTS2     = 9,  // IndexTTS-2
    MOSS_TTS       = 10, // MOSS-TTS
    IRODORI_TTS    = 11, // Irodori-TTS
    HIGGS_TTS      = 12, // Higgs Audio v3 TTS 4B
    
    // ASR / STT
    QWEN3_ASR      = 20, // Qwen3-ASR
    NEMOTRON_ASR   = 21, // Nemotron 3.5 ASR
    HIGGS_STT      = 22, // Higgs Audio v3 STT
    VOXTRAAL       = 23, // Voxtral Realtime ASR
    VIBEVOICE_ASR  = 24, // VibeVoice ASR
    
    // Music / Audio gen
    STABLE_AUDIO   = 30, // Stable Audio 3
    HEARTMULA      = 31, // HeartMuLa
    ACE_STEP       = 32, // ACE-Step
    VEVO2          = 33, // VeVo2
    
    // Voice conversion
    SEED_VC        = 40, // SeedVC
    MIOCODEC       = 41, // MioCodec
    
    // Separation
    HTDEMUCS       = 50, // HTDemucs
    MEL_ROFORMER   = 51, // Mel-Band RoFormer
    
    // VAD / Diarization
    SILERO_VAD     = 60, // Silero VAD
    MARBLENET_VAD  = 61, // MarbleNet VAD
    SORTFORMER     = 62, // Sortformer diarization
};

// ─── Audio generation parameters ──────────────────────────────────
struct AudioParams {
    // Text content (TTS, voice clone)
    std::string text;
    
    // Voice reference (voice clone, VC)
    std::string voice_ref_path;   // WAV file for voice cloning
    std::string voice_id;          // Built-in voice ID (PocketTTS, etc.)
    std::string language = "en";   // Language code
    
    // STT/ASR
    std::string audio_path;        // Input audio for ASR
    
    // Music gen
    std::string genre;             // Music genre/style prompt
    int duration_seconds = 30;     // Music duration
    
    // Voice conversion
    std::string source_audio_path; // Source audio for VC
    float similarity_ratio = 0.9f; // Voice similarity (0-1)
    
    // Emotion / style control
    std::string emotion;           // "happy", "sad", "neutral", etc.
    float speed = 1.0f;            // Playback speed multiplier
    
    // VAD
    float vad_threshold = 0.5f;    // VAD sensitivity
    int vad_min_speech_ms = 100;   // Min speech segment length
    
    // Output
    int sample_rate = 24000;       // Output sample rate
    std::string output_format = "wav"; // "wav", "mp3", "opus"
};

// ─── Audio result ─────────────────────────────────────────────────
struct AudioResult {
    std::vector<float> samples;     // PCM float samples [-1..1]
    std::vector<uint8_t> encoded;   // Encoded audio bytes (WAV/MP3)
    std::string mime_type;          // "audio/wav", "audio/mpeg"
    int sample_rate;
    int channels = 1;
    double duration_seconds;
    int64_t generation_time_ms;
    
    // For ASR
    std::string transcription;
    
    // For VAD
    std::vector<std::pair<double, double>> speech_segments;  // start/end seconds
};

// ─── Progress callback ────────────────────────────────────────────
using AudioProgressFn = std::function<void(AudioTask task, int step, int total)>;

// ─── Audio engine ─────────────────────────────────────────────────
// Wraps audio.cpp's model loaders and inference paths.
class AudioEngine {
public:
    AudioEngine();
    ~AudioEngine();
    
    // No copy
    AudioEngine(const AudioEngine&) = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;
    
    // ── Model lifecycle ──
    /// Load an audio model for a specific task.
    /// family: model family identifier
    /// model_path: path to model file or directory
    bool load_model(AudioModelFamily family, const std::string& model_path);
    
    /// Unload all models.
    void unload_all();
    
    /// Check if a specific model family is loaded.
    bool is_loaded(AudioModelFamily family) const;
    
    /// List loaded models.
    std::vector<AudioModelFamily> loaded_models() const;

    // ── TTS ──
    /// Text-to-Speech synthesis.
    AudioResult synthesize(const AudioParams& params,
                           AudioProgressFn progress = nullptr);
    
    /// Voice cloning: synthesize speech in a reference voice.
    AudioResult clone_voice(const AudioParams& params,
                            AudioProgressFn progress = nullptr);

    // ── ASR ──
    /// Speech-to-Text transcription.
    AudioResult transcribe(const std::string& audio_path,
                           const std::string& language = "auto");
    
    /// Streaming ASR (for real-time).
    void begin_streaming_asr(AudioModelFamily family);
    AudioResult streaming_asr_chunk(const float* pcm, int n_samples);
    AudioResult end_streaming_asr();

    // ── Music / Audio generation ──
    /// Generate music from text description.
    AudioResult generate_music(const AudioParams& params,
                               AudioProgressFn progress = nullptr);
    
    /// Generate sound effects.
    AudioResult generate_sfx(const AudioParams& params,
                             AudioProgressFn progress = nullptr);

    // ── Voice conversion ──
    /// Convert source audio to target voice.
    AudioResult voice_convert(const AudioParams& params,
                              AudioProgressFn progress = nullptr);

    // ── VAD ──
    /// Detect speech segments in audio.
    AudioResult detect_voice_activity(const float* pcm, int n_samples,
                                       int sample_rate = 16000);
    
    /// Real-time VAD on a single frame.
    bool vad_frame(const float* pcm, int n_samples, int sample_rate = 16000);

    // ── Source separation ──
    /// Separate audio into stems (vocals, drums, bass, etc.).
    AudioResult separate(const float* pcm, int n_samples, int sample_rate = 44100);

    // ── Backend selection ──
    enum Backend { CPU, CUDA, VULKAN, METAL, AUTO };
    void set_backend(Backend backend);
    Backend backend() const;
    
    // ── Utility ──
    /// Encode float PCM to WAV bytes.
    static std::vector<uint8_t> encode_wav(const float* pcm, int n_samples,
                                            int sample_rate, int channels = 1);
    
    /// Load WAV file to float PCM.
    static std::vector<float> load_wav(const std::string& path,
                                        int* sample_rate = nullptr,
                                        int* channels = nullptr);

private:
    Backend backend_ = Backend::AUTO;
    void* audio_context_ = nullptr;  // Opaque audio.cpp context
    
    // Per-family model handles
    struct ModelHandle {
        AudioModelFamily family;
        std::string path;
        void* handle = nullptr;  // audio.cpp model handle
    };
    std::vector<ModelHandle> models_;
};

// ─── Global audio engine (singleton) ─────────────────────────────
AudioEngine& audio_engine();
