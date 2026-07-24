// audio_bridge.cpp — audio.cpp integration layer
//
// Bridges 1bit-systems with the audio.cpp framework for TTS, STT,
// music generation, voice cloning, VAD, and source separation.
// Requires the audio.cpp submodule at third_party/audio.cpp/.

#include "audio_bridge.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>

// ─── Implementation ───────────────────────────────────────────────

AudioEngine::AudioEngine() = default;
AudioEngine::~AudioEngine() {
    unload_all();
}

bool AudioEngine::load_model(AudioModelFamily family, const std::string& model_path) {
    // Check if already loaded
    for (auto& m : models_) {
        if (m.family == family) {
            printf("audio: model family %d already loaded\n", (int)family);
            return true;
        }
    }
    
    // Create new model handle
    ModelHandle mh;
    mh.family = family;
    mh.path = model_path;
    mh.handle = nullptr;  // audio.cpp model handle (from audiocpp_model_load)
    
    // In production: call audio.cpp's model loader
    // mh.handle = audiocpp_load_model(family, model_path, backend_);
    
    models_.push_back(mh);
    printf("audio: loaded model family %d from %s\n", (int)family, model_path.c_str());
    return true;
}

void AudioEngine::unload_all() {
    for (auto& m : models_) {
        // In production: call audio.cpp's model unloader
        // if (m.handle) audiocpp_free_model(m.handle);
    }
    models_.clear();
}

bool AudioEngine::is_loaded(AudioModelFamily family) const {
    for (auto& m : models_) {
        if (m.family == family) return true;
    }
    return false;
}

std::vector<AudioModelFamily> AudioEngine::loaded_models() const {
    std::vector<AudioModelFamily> result;
    for (auto& m : models_) result.push_back(m.family);
    return result;
}

// ─── TTS ──────────────────────────────────────────────────────────

AudioResult AudioEngine::synthesize(const AudioParams& params,
                                     AudioProgressFn progress) {
    auto t0 = std::chrono::steady_clock::now();
    
    // In production: call audio.cpp's TTS path
    // AudioResult result = audiocpp_tts(text, voice_id, language, speed, emotion);
    
    // Placeholder: generate sine wave as dummy output
    int sample_rate = params.sample_rate;
    double duration = std::min(30.0, params.text.length() * 0.1);  // rough estimate
    int n_samples = (int)(sample_rate * duration);
    
    AudioResult result;
    result.samples.resize(n_samples);
    for (int i = 0; i < n_samples; i++) {
        double t = (double)i / sample_rate;
        result.samples[i] = 0.3f * sinf(2.0f * M_PI * 220.0f * t);  // 220Hz tone
    }
    result.sample_rate = sample_rate;
    result.channels = 1;
    result.duration_seconds = duration;
    result.encoded = encode_wav(result.samples.data(), n_samples, sample_rate);
    result.mime_type = "audio/wav";
    
    auto t1 = std::chrono::steady_clock::now();
    result.generation_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    
    return result;
}

AudioResult AudioEngine::clone_voice(const AudioParams& params,
                                      AudioProgressFn progress) {
    // Voice cloning via audio.cpp (Qwen3-TTS CustomVoice, OmniVoice, etc.)
    auto t0 = std::chrono::steady_clock::now();
    
    AudioResult result;
    result.mime_type = "audio/wav";
    result.sample_rate = params.sample_rate;
    
    // In production: load voice ref -> extract embedding -> generate with voice
    result.transcription = "[voice clone: " + params.text + "]";
    
    auto t1 = std::chrono::steady_clock::now();
    result.generation_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    
    return result;
}

// ─── ASR ──────────────────────────────────────────────────────────

AudioResult AudioEngine::transcribe(const std::string& audio_path,
                                     const std::string& language) {
    auto t0 = std::chrono::steady_clock::now();
    
    // Load audio file
    int sr = 0, ch = 0;
    auto pcm = load_wav(audio_path, &sr, &ch);
    
    AudioResult result;
    if (pcm.empty()) {
        result.transcription = "";
        return result;
    }
    
    // In production: run STT model (Qwen3-ASR, Nemotron, etc.)
    // Currently using the existing whisper.cpp in 1bit-systems
    // as a fallback, or audio.cpp's ASR models when available.
    
    result.transcription = "[transcription via audio.cpp ASR]";
    result.sample_rate = sr;
    result.channels = ch;
    result.duration_seconds = (double)pcm.size() / sr;
    
    auto t1 = std::chrono::steady_clock::now();
    result.generation_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    
    return result;
}

void AudioEngine::begin_streaming_asr(AudioModelFamily family) {
    // Initialize streaming ASR session
}

AudioResult AudioEngine::streaming_asr_chunk(const float* pcm, int n_samples) {
    AudioResult result;
    // Process chunk through ASR model
    return result;
}

AudioResult AudioEngine::end_streaming_asr() {
    AudioResult result;
    // Finalize and return complete transcript
    return result;
}

// ─── Music generation ─────────────────────────────────────────────

AudioResult AudioEngine::generate_music(const AudioParams& params,
                                         AudioProgressFn progress) {
    auto t0 = std::chrono::steady_clock::now();
    
    // In production: Stable Audio 3, HeartMuLa, or ACE-Step via audio.cpp
    AudioResult result;
    result.mime_type = "audio/wav";
    result.sample_rate = 44100;
    result.channels = 2;  // stereo for music
    result.duration_seconds = params.duration_seconds;
    
    int n_samples = params.duration_seconds * result.sample_rate * result.channels;
    result.samples.resize(n_samples, 0.0f);
    result.encoded = encode_wav(result.samples.data(), n_samples,
                                 result.sample_rate, result.channels);
    
    auto t1 = std::chrono::steady_clock::now();
    result.generation_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    
    return result;
}

AudioResult AudioEngine::generate_sfx(const AudioParams& params,
                                       AudioProgressFn progress) {
    return AudioResult{};  // Placeholder
}

// ─── Voice conversion ─────────────────────────────────────────────

AudioResult AudioEngine::voice_convert(const AudioParams& params,
                                        AudioProgressFn progress) {
    return AudioResult{};  // Placeholder
}

// ─── VAD ──────────────────────────────────────────────────────────

AudioResult AudioEngine::detect_voice_activity(const float* pcm, int n_samples,
                                                 int sample_rate) {
    AudioResult result;
    result.sample_rate = sample_rate;
    
    // Simple energy-based VAD as placeholder
    // In production: Silero VAD or MarbleNet VAD via audio.cpp
    const float threshold = 0.01f;
    bool in_speech = false;
    double seg_start = 0;
    
    for (int i = 0; i < n_samples; i += sample_rate / 100) {  // 10ms frames
        double energy = 0;
        int frame_end = std::min(i + sample_rate / 100, n_samples);
        for (int j = i; j < frame_end; j++) {
            energy += fabs(pcm[j]);
        }
        energy /= (frame_end - i);
        
        double t = (double)i / sample_rate;
        if (energy > threshold && !in_speech) {
            in_speech = true;
            seg_start = t;
        } else if (energy <= threshold && in_speech) {
            in_speech = false;
            result.speech_segments.push_back({seg_start, t});
        }
    }
    if (in_speech) {
        result.speech_segments.push_back({seg_start, (double)n_samples / sample_rate});
    }
    
    return result;
}

bool AudioEngine::vad_frame(const float* pcm, int n_samples, int sample_rate) {
    // Placeholder: energy-based VAD
    double energy = 0;
    for (int i = 0; i < n_samples; i++) energy += fabs(pcm[i]);
    energy /= n_samples;
    return energy > 0.01f;
}

// ─── Source separation ────────────────────────────────────────────

AudioResult AudioEngine::separate(const float* pcm, int n_samples, int sample_rate) {
    return AudioResult{};  // Placeholder
}

// ─── Backend ──────────────────────────────────────────────────────

void AudioEngine::set_backend(Backend backend) { backend_ = backend; }
AudioEngine::Backend AudioEngine::backend() const { return backend_; }

// ─── Utility ──────────────────────────────────────────────────────

std::vector<uint8_t> AudioEngine::encode_wav(const float* pcm, int n_samples,
                                              int sample_rate, int channels) {
    // WAV header + 16-bit PCM data
    int bytes_per_sample = 2;
    int data_bytes = n_samples * bytes_per_sample;
    int header_size = 44;
    
    std::vector<uint8_t> wav(header_size + data_bytes);
    
    // RIFF header
    memcpy(&wav[0], "RIFF", 4);
    uint32_t file_size = header_size + data_bytes - 8;
    memcpy(&wav[4], &file_size, 4);
    memcpy(&wav[8], "WAVE", 4);
    
    // fmt chunk
    memcpy(&wav[12], "fmt ", 4);
    uint32_t fmt_size = 16;
    uint16_t audio_fmt = 1;  // PCM
    uint16_t num_channels = (uint16_t)channels;
    uint32_t sample_rate_u32 = (uint32_t)sample_rate;
    uint16_t block_align = (uint16_t)(channels * bytes_per_sample);
    uint32_t byte_rate = sample_rate_u32 * block_align;
    uint16_t bits_per_sample = 16;
    
    memcpy(&wav[16], &fmt_size, 4);
    memcpy(&wav[20], &audio_fmt, 2);
    memcpy(&wav[22], &num_channels, 2);
    memcpy(&wav[24], &sample_rate_u32, 4);
    memcpy(&wav[28], &byte_rate, 4);
    memcpy(&wav[32], &block_align, 2);
    memcpy(&wav[34], &bits_per_sample, 2);
    
    // data chunk
    memcpy(&wav[36], "data", 4);
    memcpy(&wav[40], &data_bytes, 4);
    
    // Convert float PCM to 16-bit integer
    for (int i = 0; i < n_samples; i++) {
        float s = std::max(-1.0f, std::min(1.0f, pcm[i]));
        int16_t sample = (int16_t)(s * 32767.0f);
        memcpy(&wav[header_size + i * 2], &sample, 2);
    }
    
    return wav;
}

std::vector<float> AudioEngine::load_wav(const std::string& path,
                                          int* sample_rate, int* channels) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        fprintf(stderr, "audio: failed to open %s\n", path.c_str());
        return {};
    }
    
    // Read WAV header
    char riff[4]; file.read(riff, 4);
    if (memcmp(riff, "RIFF", 4) != 0) return {};
    
    file.seekg(0, std::ios::end);
    auto file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (file_size < 44) return {};
    
    // Read header
    std::vector<uint8_t> header(44);
    file.read((char*)header.data(), 44);
    
    uint16_t audio_fmt;
    uint16_t num_channels;
    uint32_t sample_rate_u32;
    uint16_t bits_per_sample;
    
    memcpy(&audio_fmt, &header[20], 2);
    memcpy(&num_channels, &header[22], 2);
    memcpy(&sample_rate_u32, &header[24], 4);
    memcpy(&bits_per_sample, &header[34], 2);
    
    if (audio_fmt != 1) {
        fprintf(stderr, "audio: only PCM WAV supported (format=%d)\n", audio_fmt);
        return {};
    }
    
    if (sample_rate) *sample_rate = (int)sample_rate_u32;
    if (channels) *channels = num_channels;
    
    // Read data chunk
    // Skip any extra chunks between header and data
    char chunk_id[4];
    uint32_t chunk_size;
    
    while (file.read(chunk_id, 4) && file.read((char*)&chunk_size, 4)) {
        if (memcmp(chunk_id, "data", 4) == 0) {
            // Found data chunk
            int n_samples = chunk_size / (bits_per_sample / 8) / num_channels;
            std::vector<float> pcm(n_samples);
            
            if (bits_per_sample == 16) {
                std::vector<int16_t> raw(n_samples * num_channels);
                file.read((char*)raw.data(), chunk_size);
                // Mix down to mono
                for (int i = 0; i < n_samples; i++) {
                    if (num_channels == 1) {
                        pcm[i] = (float)raw[i] / 32768.0f;
                    } else {
                        int sum = 0;
                        for (int c = 0; c < num_channels; c++) {
                            sum += raw[i * num_channels + c];
                        }
                        pcm[i] = (float)sum / (32768.0f * num_channels);
                    }
                }
            }
            return pcm;
        }
        file.seekg(chunk_size, std::ios::cur);
    }
    
    return {};
}

// ─── Singleton ────────────────────────────────────────────────────

AudioEngine& audio_engine() {
    static AudioEngine engine;
    return engine;
}
