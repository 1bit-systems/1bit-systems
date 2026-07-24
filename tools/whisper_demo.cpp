// whisper_demo.cpp — Whisper speech-to-text demo
// Build: cmake --build . --target whisper_demo -j8
// Run:   ./build/whisper_demo <whisper.gguf> <audio.wav>
//
// If no audio file provided, generates a synthetic test tone.

#include "whisper.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

// Generate a synthetic test tone (440Hz sine wave, 3 seconds)
static std::vector<float> generate_test_tone(int sample_rate = 16000, int duration_sec = 3, float freq = 440.0f) {
    int n = sample_rate * duration_sec;
    std::vector<float> pcm(n);
    for (int i = 0; i < n; i++) {
        float t = (float)i / sample_rate;
        pcm[i] = 0.3f * sinf(2.0f * M_PI * freq * t);  // 440Hz A4 tone
        // Add a bit of harmonic
        pcm[i] += 0.15f * sinf(2.0f * M_PI * freq * 2 * t);
        pcm[i] += 0.1f * sinf(2.0f * M_PI * freq * 3 * t);
    }
    return pcm;
}

int main(int argc, char** argv) {
    fprintf(stderr, "╔══════════════════════════════════════════╗\n");
    fprintf(stderr, "║        Whisper Speech-to-Text Demo       ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════╝\n\n");
    
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <whisper.gguf> [audio.wav]\n", argv[0]);
        fprintf(stderr, "\n  If no audio file given, generates a synthetic 440Hz test tone.\n");
        return 1;
    }
    
    std::string model_path = argv[1];
    
    // 1. Load model
    fprintf(stderr, "1. Loading Whisper model: %s\n", model_path.c_str());
    WhisperModel model;
    if (!model.load_from_gguf(model_path)) {
        fprintf(stderr, "❌ Failed to load model\n");
        return 1;
    }
    fprintf(stderr, "   ✅ Model loaded: %d enc layers, %d dec layers, %d hidden\n",
            model.cfg.n_audio_layer, model.cfg.n_text_layer, model.cfg.n_audio_state);
    
    // 2. Load or generate audio
    std::vector<float> audio_pcm;
    int sample_rate = 16000;
    
    if (argc > 2) {
        fprintf(stderr, "\n2. Loading audio: %s\n", argv[2]);
        audio_pcm = whisper_load_wav(argv[2], &sample_rate);
        if (audio_pcm.empty()) {
            fprintf(stderr, "❌ Failed to load WAV file\n");
            return 1;
        }
    } else {
        fprintf(stderr, "\n2. Generating test tone (440Hz, 3 seconds)\n");
        audio_pcm = generate_test_tone();
        sample_rate = 16000;
    }
    
    fprintf(stderr, "   Audio: %d samples @ %d Hz (%.1f seconds)\n",
            (int)audio_pcm.size(), sample_rate, (float)audio_pcm.size() / sample_rate);
    
    // 3. Compute mel spectrogram
    fprintf(stderr, "\n3. Computing mel spectrogram (%d bands)...\n", model.cfg.n_mels);
    auto mel = whisper_log_mel_spectrogram(audio_pcm.data(), (int)audio_pcm.size(), 
                                            sample_rate, model.cfg.n_mels);
    int n_frames = (int)(mel.size() / model.cfg.n_mels);
    fprintf(stderr, "   Mel spectrogram: %d frames × %d bands\n", n_frames, model.cfg.n_mels);
    
    // 4. Run encoder
    fprintf(stderr, "\n4. Running encoder (%d layers)...\n", model.cfg.n_audio_layer);
    auto enc_out = whisper_encode(model, mel.data(), n_frames);
    int n_enc_ctx = (int)(enc_out.size() / model.cfg.n_audio_state);
    fprintf(stderr, "   Encoder output: %d tokens × %d dims\n", n_enc_ctx, model.cfg.n_audio_state);
    
    // 5. Run decoder loop
    fprintf(stderr, "\n5. Running decoder loop...\n");
    auto text = whisper_transcribe(model, audio_pcm.data(), (int)audio_pcm.size());
    
    // 6. Output
    fprintf(stderr, "\n══════════════════════════════════════════\n");
    fprintf(stderr, "   Transcription: \"%s\"\n", text.c_str());
    fprintf(stderr, "══════════════════════════════════════════\n");
    
    return 0;
}
