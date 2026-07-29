// codec_tts.h — Voice pack + TTS wrapper using the native codec decoder.
//
// Replaces the Piper subprocess with native ONNX inference via CodecDecoder.
// Manages voice packs (.voice files containing decoder ONNX model + speaker
// embedding + config) with in-memory registry and hot-reload via mtime.
//
// When ONNX Runtime is not available, all synthesis calls return empty
// string — jarvis_server treats this as "TTS unavailable" and falls back
// to Piper.

#pragma once
#include <string>
#include <vector>
#include <memory>

namespace jarvis {

/// Information about a loaded voice pack
struct VoicePackInfo {
    std::string name;
    std::string speaker_name;
    std::string language;
    int sample_rate;
    std::string path;
};

/// High-level TTS using the codec decoder and voice packs.
/// Replaces the Piper subprocess with native ONNX inference.

struct CodecTtsImpl;

class CodecTts {
public:
    CodecTts();
    ~CodecTts();

    /// Set directory to scan for .voice packs (default: ~/voice-packs)
    void set_voice_packs_dir(const std::string& dir);

    /// Scan voice packs directory, load all .voice files
    bool scan_voice_packs();

    /// Get list of available voice packs
    std::vector<VoicePackInfo> list_voice_packs() const;

    /// Load a specific voice pack by name or path
    bool load_voice_pack(const std::string& name_or_path);

    /// Unload a voice pack
    void unload_voice_pack(const std::string& name);

    /// Synthesize speech with the loaded voice pack.
    /// @param text: text to synthesize
    /// @param voice: voice pack name (must be loaded)
    /// @return: complete WAV file bytes (mono, 16-bit, 24kHz)
    std::string synthesize(const std::string& text, const std::string& voice);

    /// Check if a voice pack is loaded
    bool has_voice(const std::string& name) const;

private:
    std::unique_ptr<CodecTtsImpl> impl_;
};

} // namespace jarvis
