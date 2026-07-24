// gen_test_wav.cpp — Generate a test WAV file for Whisper demo
// Build: g++ -O3 tools/gen_test_wav.cpp -o build/gen_test_wav
// Run:   ./build/gen_test_wav [output.wav] [duration_sec] [freq_hz]

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    const char* out_path = argc > 1 ? argv[1] : "test_tone.wav";
    int duration = argc > 2 ? atoi(argv[2]) : 3;
    float freq = argc > 3 ? atof(argv[3]) : 440.0f;
    
    int sample_rate = 16000;
    int n_samples = sample_rate * duration;
    int bits = 16;
    int channels = 1;
    
    // Generate sine wave with harmonics
    std::vector<int16_t> pcm(n_samples);
    for (int i = 0; i < n_samples; i++) {
        float t = (float)i / sample_rate;
        float val = 0.5f * sinf(2.0f * M_PI * freq * t);
        val += 0.25f * sinf(2.0f * M_PI * freq * 2 * t);  // 2nd harmonic
        val += 0.125f * sinf(2.0f * M_PI * freq * 3 * t);  // 3rd harmonic
        pcm[i] = (int16_t)(val * 32767.0f);
    }
    
    // Write WAV
    FILE* f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "FAIL: could not write %s\n", out_path); return 1; }
    
    int data_bytes = n_samples * (bits / 8);
    int file_size = 36 + data_bytes;
    
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    
    int fmt_size = 16;
    short audio_fmt = 1;  // PCM
    short num_channels = channels;
    int byte_rate = sample_rate * channels * (bits / 8);
    short block_align = channels * (bits / 8);
    
    fwrite(&fmt_size, 4, 1, f);
    fwrite(&audio_fmt, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits, 2, 1, f);
    
    fwrite("data", 1, 4, f);
    fwrite(&data_bytes, 4, 1, f);
    fwrite(pcm.data(), 2, n_samples, f);
    
    fclose(f);
    fprintf(stderr, "Generated: %s (%d samples, %d Hz, %d sec)\n", out_path, n_samples, sample_rate, duration);
    return 0;
}
