// whisper.cpp — Whisper speech-to-text implementation
#include "whisper.h"
#include "gguf_reader.h"
#include <cstring>
#include <cmath>
#include <numeric>

// ====================================================================
// Audio processing: WAV loader
// ====================================================================
std::vector<float> whisper_load_wav(const std::string& path, int* out_sample_rate) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    
    char riff[4]; fread(riff, 1, 4, f);
    if (memcmp(riff, "RIFF", 4) != 0) { fclose(f); return {}; }
    fseek(f, 8, SEEK_SET);
    char wave[4]; fread(wave, 1, 4, f);
    if (memcmp(wave, "WAVE", 4) != 0) { fclose(f); return {}; }
    
    // Find fmt chunk
    uint16_t channels = 1, bits = 16;
    int sample_rate = 16000;
    while (true) {
        char chunk_id[4]; 
        if (fread(chunk_id, 1, 4, f) != 4) { fclose(f); return {}; }
        uint32_t chunk_size; fread(&chunk_size, 4, 1, f);
        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            uint16_t audio_fmt; fread(&audio_fmt, 2, 1, f);
            fread(&channels, 2, 1, f);
            fread(&sample_rate, 4, 1, f);
            fseek(f, 6, SEEK_CUR); // skip byte rate + block align
            fread(&bits, 2, 1, f);
            if (chunk_size > 16) fseek(f, chunk_size - 16, SEEK_CUR);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            int n_samples = chunk_size / (bits / 8);
            std::vector<float> pcm(n_samples);
            if (bits == 16) {
                std::vector<int16_t> buf(n_samples);
                fread(buf.data(), 2, n_samples, f);
                for (int i = 0; i < n_samples; i++) pcm[i] = buf[i] / 32768.0f;
            } else if (bits == 32) {
                std::vector<float> buf(n_samples);
                fread(buf.data(), 4, n_samples, f);
                pcm = std::move(buf);
            }
            fclose(f);
            if (out_sample_rate) *out_sample_rate = sample_rate;
            if (channels > 1) {
                // Downmix to mono
                std::vector<float> mono(n_samples / channels);
                for (int i = 0; i < (int)mono.size(); i++) {
                    float s = 0; for (int c = 0; c < channels; c++) s += pcm[(size_t)i * channels + c];
                    mono[i] = s / channels;
                }
                return mono;
            }
            return pcm;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }
}

// ====================================================================
// Audio processing: Mel spectrogram
// ====================================================================
// Uses the standard Whisper mel filterbank (80 bands, 400 FFT, 160 hop)
// Following OpenAI's implementation conventions.

// Pre-computed mel filterbank (80 bands × 201 FFT bins, normalized)
struct MelFilterbank {
    int n_mels, n_fft, n_fft_bins;
    float fmin, fmax;
    std::vector<std::vector<std::pair<int, float>>> banks; // per-mel: [(bin, weight), ...]
    
    MelFilterbank(int n_mels = 80, int n_fft = 400, float fmin = 0, float fmax = 8000, int sr = 16000)
        : n_mels(n_mels), n_fft(n_fft), fmin(fmin), fmax(fmax) {
        n_fft_bins = n_fft / 2 + 1;
        banks.resize(n_mels);
        
        float mel_fmin = 1127.0f * logf(1.0f + fmin / 700.0f);
        float mel_fmax = 1127.0f * logf(1.0f + fmax / 700.0f);
        
        for (int m = 0; m < n_mels; m++) {
            float mel = mel_fmin + (float)m / (n_mels - 1) * (mel_fmax - mel_fmin);
            float freq = 700.0f * (expf(mel / 1127.0f) - 1.0f);
            float bin = freq / sr * (n_fft - 1);
            
            int bin_lo = (int)floorf(bin);
            int bin_hi = std::min(bin_lo + 1, n_fft_bins - 1);
            
            if (bin_lo >= 0 && bin_lo < n_fft_bins) {
                banks[m].push_back({bin_lo, bin - bin_lo});
            }
            if (bin_hi >= 0 && bin_hi < n_fft_bins && bin_hi != bin_lo) {
                banks[m].push_back({bin_hi, (float)bin_hi - bin});
            }
        }
    }
    
    void apply(const float* stft_mag, int n_frames, float* out) const {
        for (int m = 0; m < n_mels; m++) {
            for (int t = 0; t < n_frames; t++) {
                float val = 0;
                for (auto& [bin, weight] : banks[m]) {
                    val += stft_mag[(size_t)bin * n_frames + t] * (1.0f - fabsf(weight));
                }
                // Log: log10(max(val, 1e-10)) * 10 (or clamp)
                val = std::max(val, 1e-10f);
                out[(size_t)m * n_frames + t] = log10f(val) * 10.0f;
            }
        }
    }
};

// Hann window
static std::vector<float> hann_window(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; i++) w[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (n - 1)));
    return w;
}

std::vector<float> whisper_log_mel_spectrogram(const float* audio, int n_samples,
                                                int sample_rate, int n_mels, int n_fft, int hop) {
    // Resample to 16kHz if needed
    std::vector<float> pcm;
    if (sample_rate != 16000) {
        float ratio = 16000.0f / sample_rate;
        int n_out = (int)(n_samples * ratio);
        pcm.resize(n_out);
        for (int i = 0; i < n_out; i++) {
            float src_pos = i / ratio;
            int idx = (int)src_pos;
            float frac = src_pos - idx;
            if (idx + 1 < n_samples)
                pcm[i] = audio[idx] * (1 - frac) + audio[idx + 1] * frac;
            else
                pcm[i] = audio[std::min(idx, n_samples - 1)];
        }
    } else {
        pcm.assign(audio, audio + n_samples);
    }
    
    // Pad to at least n_fft
    if ((int)pcm.size() < n_fft) pcm.resize(n_fft, 0.0f);
    
    int n_frames = (pcm.size() - n_fft) / hop + 1;
    int n_fft_bins = n_fft / 2 + 1;
    
    auto hann = hann_window(n_fft);
    MelFilterbank filterbank(n_mels, n_fft, 0, 8000, 16000);
    
    // STFT → magnitude using pre-computed twiddle factors (FFT-based)
    std::vector<float> stft_mag((size_t)n_fft_bins * n_frames, 0.0f);
    
    // Pre-compute cos/sin tables for this FFT size (once)
    struct TwiddleCache {
        std::vector<float> cos_t, sin_t;
        int n_fft_cached = 0;
    };
    static TwiddleCache tw;
    if (tw.n_fft_cached != n_fft) {
        tw.cos_t.assign((size_t)n_fft_bins * n_fft, 0.0f);
        tw.sin_t.assign((size_t)n_fft_bins * n_fft, 0.0f);
        tw.n_fft_cached = n_fft;
        for (int k = 0; k < n_fft_bins; k++) {
            for (int i = 0; i < n_fft; i++) {
                float angle = 2.0f * M_PI * k * i / n_fft;
                tw.cos_t[(size_t)k * n_fft + i] = cosf(angle);
                tw.sin_t[(size_t)k * n_fft + i] = -sinf(angle);
            }
        }
    }
    
    #pragma omp parallel for
    for (int t = 0; t < n_frames; t++) {
        // Windowed frame
        std::vector<float> frame(n_fft);
        for (int i = 0; i < n_fft; i++)
            frame[i] = pcm[(size_t)t * hop + i] * hann[i];
        
        // DFT using pre-computed twiddle factors
        for (int k = 0; k < n_fft_bins; k++) {
            float re = 0, im = 0;
            const float* ct = &tw.cos_t[(size_t)k * n_fft];
            const float* st = &tw.sin_t[(size_t)k * n_fft];
            for (int i = 0; i < n_fft; i++) {
                re += frame[i] * ct[i];
                im += frame[i] * st[i];
            }
            stft_mag[(size_t)k * n_frames + t] = sqrtf(re * re + im * im);
        }
    }
    
    // Apply mel filterbank
    std::vector<float> mel((size_t)n_mels * n_frames);
    filterbank.apply(stft_mag.data(), n_frames, mel.data());
    
    // Normalize: subtract mean (Whisper-style)
    double mean = 0;
    for (float v : mel) mean += v;
    mean /= mel.size();
    for (float& v : mel) v -= (float)mean;
    
    // Pad or truncate to n_audio_ctx (1500 for 30s)
    int target_frames = 1500;
    if (n_frames < target_frames) {
        mel.resize((size_t)n_mels * target_frames, 0.0f);
    }
    
    return mel;
}

// ====================================================================
// GGUF model loader
// ====================================================================
bool WhisperModel::load_from_gguf(const std::string& path, const WhisperConfig* override_cfg) {
    GgufReader r;
    if (!r.open(path)) {
        fprintf(stderr, "[whisper] FAIL: could not open %s\n", path.c_str());
        return false;
    }
    
    // Auto-detect config from model
    auto gu32 = [&](const std::string& key, int def) -> int {
        uint32_t v;
        if (r.get_u32(key, v)) return (int)v;
        std::string arch = r.architecture();
        if (!arch.empty() && r.get_u32(arch + "." + key, v)) return (int)v;
        return def;
    };
    
    if (override_cfg) {
        cfg = *override_cfg;
    } else {
        cfg.n_audio_state = gu32("audio.embedding_length", 384);
        cfg.n_audio_layer = gu32("audio.block_count", 4);
        cfg.n_audio_head  = gu32("audio.head_count", 6);
        cfg.n_text_state  = gu32("text.embedding_length", 384);
        cfg.n_text_layer  = gu32("text.block_count", 4);
        cfg.n_text_head   = gu32("text.head_count", 6);
        cfg.n_vocab       = gu32("vocab_size", 51865);
        cfg.n_mels        = gu32("audio.feature_length", 80);
        // Derive heads from hidden if not specified
        if (cfg.n_audio_head == 0) cfg.n_audio_head = cfg.n_audio_state / 64;
        if (cfg.n_text_head == 0)  cfg.n_text_head  = cfg.n_text_state / 64;
    }
    
    auto get = [&](const std::string& name, std::vector<float>& dst, size_t expect) -> bool {
        size_t n = 0;
        if (!r.get_tensor_f32(name, dst, &n)) {
            fprintf(stderr, "  [whisper] missing: %s\n", name.c_str());
            return false;
        }
        if (expect > 0 && n != expect) {
            fprintf(stderr, "  [whisper] %s: expected %zu, got %zu\n", name.c_str(), expect, n);
            return false;
        }
        return true;
    };
    
    auto get_opt = [&](const std::string& name, std::vector<float>& dst, size_t expect) -> bool {
        size_t n = 0;
        if (!r.get_tensor_f32(name, dst, &n)) return false;
        if (expect > 0 && n != expect) {
            dst.clear(); return false;
        }
        return true;
    };
    
    int DA = cfg.n_audio_state, DT = cfg.n_text_state;
    
    // Encoder conv layers
    if (!get("encoder.conv1.weight", enc_conv1_w, (size_t)cfg.n_mels * DA * 3)) return false;
    if (!get("encoder.conv1.bias", enc_conv1_b, (size_t)DA)) return false;
    if (!get("encoder.conv2.weight", enc_conv2_w, (size_t)DA * DA * 3)) return false;
    if (!get("encoder.conv2.bias", enc_conv2_b, (size_t)DA)) return false;
    
    // Encoder position embedding (sinusoidal, not stored as tensor)
    enc_pos_emb.resize((size_t)cfg.n_audio_ctx * DA);
    for (int p = 0; p < cfg.n_audio_ctx; p++) {
        for (int d = 0; d < DA; d++) {
            float theta = 1.0f / powf(10000.0f, (float)d / DA);
            enc_pos_emb[(size_t)p * DA + d] = (d % 2 == 0) ? sinf(p * theta) : cosf(p * theta);
        }
    }
    
    // Encoder layers
    enc_layers.resize(cfg.n_audio_layer);
    for (int il = 0; il < cfg.n_audio_layer; il++) {
        auto& l = enc_layers[il];
        std::string p = "encoder.blocks." + std::to_string(il) + ".";
        bool ok = true;
        ok &= get(p + "attn_ln.weight", l.attn_ln_w, (size_t)DA);
        ok &= get(p + "attn_ln.bias",   l.attn_ln_b, (size_t)DA);
        ok &= get(p + "attn.query.weight", l.attn_q_w, (size_t)DA * DA);
        ok &= get(p + "attn.query.bias",   l.attn_q_b, (size_t)DA);
        ok &= get(p + "attn.key.weight", l.attn_k_w, (size_t)DA * DA);
        ok &= get(p + "attn.key.bias",     l.attn_k_b, (size_t)DA);
        ok &= get(p + "attn.value.weight", l.attn_v_w, (size_t)DA * DA);
        ok &= get(p + "attn.value.bias",   l.attn_v_b, (size_t)DA);
        ok &= get(p + "attn.output.weight", l.attn_o_w, (size_t)DA * DA);
        ok &= get(p + "attn.output.bias",   l.attn_o_b, (size_t)DA);
        ok &= get(p + "mlp_ln.weight", l.ffn_ln_w, (size_t)DA);
        ok &= get(p + "mlp_ln.bias",   l.ffn_ln_b, (size_t)DA);
        ok &= get(p + "mlp.0.weight", l.ffn_gate_w, (size_t)DA * DA * 4);
        ok &= get(p + "mlp.0.bias",   l.ffn_gate_b, (size_t)DA * 4);
        ok &= get(p + "mlp.2.weight", l.ffn_down_w, (size_t)DA * 4 * DA);
        ok &= get(p + "mlp.2.bias",   l.ffn_down_b, (size_t)DA);
        if (!ok) { fprintf(stderr, "  encoder layer %d failed\n", il); return false; }
    }
    
    // Encoder final LN
    if (!get("encoder.ln.weight", enc_ln_w, (size_t)DA)) return false;
    if (!get("encoder.ln.bias",   enc_ln_b, (size_t)DA)) return false;
    
    // Decoder token + position embeddings
    if (!get("decoder.token_embedding.weight", dec_token_emb, (size_t)cfg.n_vocab * DT)) return false;
    if (!get("decoder.position_embedding.weight", dec_pos_emb, (size_t)cfg.n_text_ctx * DT)) return false;
    
    // Decoder layers
    dec_layers.resize(cfg.n_text_layer);
    for (int il = 0; il < cfg.n_text_layer; il++) {
        auto& l = dec_layers[il];
        std::string p = "decoder.blocks." + std::to_string(il) + ".";
        bool ok = true;
        // Self-attention
        ok &= get(p + "attn_ln.weight", l.attn_ln_w, (size_t)DT);
        ok &= get(p + "attn_ln.bias",   l.attn_ln_b, (size_t)DT);
        ok &= get(p + "attn.query.weight", l.attn_q_w, (size_t)DT * DT);
        ok &= get(p + "attn.query.bias",   l.attn_q_b, (size_t)DT);
        ok &= get(p + "attn.key.weight", l.attn_k_w, (size_t)DT * DT);
        ok &= get(p + "attn.key.bias",     l.attn_k_b, (size_t)DT);
        ok &= get(p + "attn.value.weight", l.attn_v_w, (size_t)DT * DT);
        ok &= get(p + "attn.value.bias",   l.attn_v_b, (size_t)DT);
        ok &= get(p + "attn.output.weight", l.attn_o_w, (size_t)DT * DT);
        ok &= get(p + "attn.output.bias",   l.attn_o_b, (size_t)DT);
        // Cross-attention
        ok &= get(p + "cross_attn_ln.weight", l.cross_ln_w, (size_t)DT);
        ok &= get(p + "cross_attn_ln.bias",   l.cross_ln_b, (size_t)DT);
        ok &= get(p + "cross_attn.query.weight", l.cross_q_w, (size_t)DT * DT);
        ok &= get(p + "cross_attn.query.bias",   l.cross_q_b, (size_t)DT);
        ok &= get(p + "cross_attn.key.weight", l.cross_k_w, (size_t)DT * DT);
        ok &= get(p + "cross_attn.key.bias",     l.cross_k_b, (size_t)DT);
        ok &= get(p + "cross_attn.value.weight", l.cross_v_w, (size_t)DT * DT);
        ok &= get(p + "cross_attn.value.bias",   l.cross_v_b, (size_t)DT);
        ok &= get(p + "cross_attn.output.weight", l.cross_o_w, (size_t)DT * DT);
        ok &= get(p + "cross_attn.output.bias",   l.cross_o_b, (size_t)DT);
        // FFN
        ok &= get(p + "mlp_ln.weight", l.ffn_ln_w, (size_t)DT);
        ok &= get(p + "mlp_ln.bias",   l.ffn_ln_b, (size_t)DT);
        ok &= get(p + "mlp.0.weight", l.ffn_gate_w, (size_t)DT * DT * 4);
        ok &= get(p + "mlp.0.bias",   l.ffn_gate_b, (size_t)DT * 4);
        ok &= get(p + "mlp.2.weight", l.ffn_down_w, (size_t)DT * 4 * DT);
        ok &= get(p + "mlp.2.bias",   l.ffn_down_b, (size_t)DT);
        if (!ok) { fprintf(stderr, "  decoder layer %d failed\n", il); return false; }
    }
    
    // Decoder final LN + output
    if (!get("decoder.ln.weight", dec_ln_w, (size_t)DT)) return false;
    if (!get("decoder.ln.bias",   dec_ln_b, (size_t)DT)) return false;
    get_opt("model.output.weight", output_w, (size_t)cfg.n_vocab * DT);
    
    fprintf(stderr, "[whisper] loaded: %d enc layers, %d dec layers, %d hidden, %d vocab\n",
            cfg.n_audio_layer, cfg.n_text_layer, cfg.n_audio_state, cfg.n_vocab);
    return true;
}

void WhisperModel::clear() {
    enc_conv1_w.clear(); enc_conv1_b.clear();
    enc_conv2_w.clear(); enc_conv2_b.clear();
    enc_pos_emb.clear();
    enc_ln_w.clear(); enc_ln_b.clear();
    enc_layers.clear();
    dec_pos_emb.clear();
    dec_token_emb.clear();
    dec_ln_w.clear(); dec_ln_b.clear();
    dec_layers.clear();
    output_w.clear();
}

// ====================================================================
// Encoder forward
// ====================================================================
std::vector<float> whisper_encode(const WhisperModel& model, const float* mel, int n_frames) {
    using namespace whisper_math;
    const auto& cfg = model.cfg;
    int DA = cfg.n_audio_state;
    int n_ctx = std::min(n_frames, cfg.n_audio_ctx);
    int L = cfg.n_audio_layer;
    
    // Conv1D: n_mels → DA, kernel=3, stride=1
    int c1_out = n_frames - 3 + 1;
    std::vector<float> h1((size_t)DA * c1_out);
    conv1d(h1.data(), mel, cfg.n_mels, n_frames,
           model.enc_conv1_w.data(), model.enc_conv1_b.data(), DA, 3);
    for (size_t i = 0; i < h1.size(); i++) h1[i] = gelu(h1[i]);
    
    // Conv1D: DA → DA, kernel=3, stride=2 (downsample)
    int c2_out = (c1_out - 3) / 2 + 1;
    std::vector<float> h2((size_t)DA * c2_out);
    conv1d(h2.data(), h1.data(), DA, c1_out,
           model.enc_conv2_w.data(), model.enc_conv2_b.data(), DA, 3, 2);
    for (size_t i = 0; i < h2.size(); i++) h2[i] = gelu(h2[i]);
    
    // Pad to n_audio_ctx
    n_ctx = std::min(c2_out, cfg.n_audio_ctx);
    std::vector<float> x((size_t)n_ctx * DA);
    for (int t = 0; t < n_ctx; t++)
        for (int d = 0; d < DA; d++)
            x[(size_t)t * DA + d] = h2[(size_t)t * DA + d] + model.enc_pos_emb[(size_t)t * DA + d];
    
    // Transformer encoder layers
    std::vector<float> x2(DA), attn_out(DA), ffn_up(DA * 4), ffn_gate(DA * 4);
    for (int il = 0; il < L; il++) {
        auto& l = model.enc_layers[il];
        
        // Self-attention with residual
        for (int t = 0; t < n_ctx; t++) {
            float* xt = &x[(size_t)t * DA];
            layernorm(x2.data(), xt, l.attn_ln_w.data(), l.attn_ln_b.data(), DA, 1e-5f);
            std::copy(x2.begin(), x2.end(), xt);
        }
        self_attn(attn_out.data(), x.data(), n_ctx, DA,
                  l.attn_q_w.data(), l.attn_q_b.data(),
                  l.attn_k_w.data(), l.attn_k_b.data(),
                  l.attn_v_w.data(), l.attn_v_b.data(),
                  l.attn_o_w.data(), l.attn_o_b.data(),
                  cfg.n_audio_head);
        for (int t = 0; t < n_ctx; t++)
            for (int d = 0; d < DA; d++)
                x[(size_t)t * DA + d] += attn_out[(size_t)t * DA + d];
        
        // FFN with residual
        for (int t = 0; t < n_ctx; t++) {
            float* xt = &x[(size_t)t * DA];
            layernorm(x2.data(), xt, l.ffn_ln_w.data(), l.ffn_ln_b.data(), DA, 1e-5f);
            matmul(ffn_gate.data(), x2.data(), l.ffn_gate_w.data(), DA * 4, DA);
            if (!l.ffn_gate_b.empty()) for (int i = 0; i < DA * 4; i++) ffn_gate[i] += l.ffn_gate_b[i];
            for (int i = 0; i < DA * 4; i++) ffn_gate[i] = gelu(ffn_gate[i]);
            matmul(ffn_up.data(), ffn_gate.data(), l.ffn_down_w.data(), DA, DA * 4);
            if (!l.ffn_down_b.empty()) for (int i = 0; i < DA; i++) ffn_up[i] += l.ffn_down_b[i];
            for (int d = 0; d < DA; d++) xt[d] += ffn_up[d];
        }
    }
    
    // Final encoder LN
    for (int t = 0; t < n_ctx; t++) {
        float* xt = &x[(size_t)t * DA];
        layernorm(x2.data(), xt, model.enc_ln_w.data(), model.enc_ln_b.data(), DA, 1e-5f);
        std::copy(x2.begin(), x2.end(), xt);
    }
    
    return x;
}

// ====================================================================
// Decoder single step
// ====================================================================
std::vector<float> whisper_decode_step(const WhisperModel& model, const std::vector<int>& tokens,
                                        const float* enc_out, int n_enc_ctx,
                                        std::vector<float>& kv_cache) {
    using namespace whisper_math;
    const auto& cfg = model.cfg;
    int DT = cfg.n_text_state;
    int N = (int)tokens.size();
    
    // Token embedding + position embedding
    std::vector<float> x((size_t)N * DT);
    for (int t = 0; t < N; t++) {
        int tok = tokens[t];
        if (tok >= 0 && tok < cfg.n_vocab) {
            for (int d = 0; d < DT; d++)
                x[(size_t)t * DT + d] = model.dec_token_emb[(size_t)tok * DT + d] + model.dec_pos_emb[(size_t)t * DT + d];
        }
    }
    
    std::vector<float> x2(DT), attn_out(DT), ca_out(DT), ffn_up(DT * 4), ffn_gate(DT * 4);
    
    for (int il = 0; il < cfg.n_text_layer; il++) {
        auto& l = model.dec_layers[il];
        
        // Self-attention (causal mask)
        for (int t = 0; t < N; t++) {
            float* xt = &x[(size_t)t * DT];
            layernorm(x2.data(), xt, l.attn_ln_w.data(), l.attn_ln_b.data(), DT, 1e-5f);
            std::copy(x2.begin(), x2.end(), xt);
        }
        // Self-attention with causal mask
        int H = DT / cfg.n_text_head;
        std::vector<float> Q((size_t)N * DT), K((size_t)N * DT), V((size_t)N * DT);
        std::vector<float> scores(N);
        for (int t = 0; t < N; t++) {
            matmul(&Q[(size_t)t * DT], &x[(size_t)t * DT], l.attn_q_w.data(), DT, DT);
            matmul(&K[(size_t)t * DT], &x[(size_t)t * DT], l.attn_k_w.data(), DT, DT);
            matmul(&V[(size_t)t * DT], &x[(size_t)t * DT], l.attn_v_w.data(), DT, DT);
            if (!l.attn_q_b.empty()) for (int d = 0; d < DT; d++) { Q[(size_t)t*DT+d] += l.attn_q_b[d]; K[(size_t)t*DT+d] += l.attn_k_b[d]; V[(size_t)t*DT+d] += l.attn_v_b[d]; }
        }
        float scale = 1.0f / sqrtf((float)H);
        for (int t = 0; t < N; t++) {
            std::fill(attn_out.begin(), attn_out.end(), 0.0f);
            for (int h = 0; h < cfg.n_text_head; h++) {
                float* Qh = &Q[(size_t)t * DT + (size_t)h * H];
                for (int s = 0; s <= t; s++) { // causal: only attend to past
                    float* Kh = &K[(size_t)s * DT + (size_t)h * H];
                    float acc = 0; for (int d = 0; d < H; d++) acc += Qh[d] * Kh[d];
                    scores[s] = acc * scale;
                }
                for (int s = t + 1; s < N; s++) scores[s] = -1e30f; // mask future
                softmax_inplace(scores.data(), N);
                for (int d = 0; d < H; d++) {
                    float acc = 0;
                    for (int s = 0; s < N; s++) acc += scores[s] * V[(size_t)s * DT + (size_t)h * H + d];
                    attn_out[(size_t)h * H + d] = acc;
                }
            }
            matmul(x2.data(), attn_out.data(), l.attn_o_w.data(), DT, DT);
            if (!l.attn_o_b.empty()) for (int d = 0; d < DT; d++) x2[d] += l.attn_o_b[d];
            for (int d = 0; d < DT; d++) x[(size_t)t * DT + d] += x2[d];
        }
        
        // Cross-attention
        for (int t = 0; t < N; t++) {
            float* xt = &x[(size_t)t * DT];
            layernorm(x2.data(), xt, l.cross_ln_w.data(), l.cross_ln_b.data(), DT, 1e-5f);
            std::copy(x2.begin(), x2.end(), xt);
        }
        cross_attn(ca_out.data(), x.data(), N, DT, enc_out, n_enc_ctx,
                   l.cross_q_w.data(), l.cross_q_b.data(),
                   l.cross_k_w.data(), l.cross_k_b.data(),
                   l.cross_v_w.data(), l.cross_v_b.data(),
                   l.cross_o_w.data(), l.cross_o_b.data(),
                   cfg.n_text_head);
        for (int t = 0; t < N; t++)
            for (int d = 0; d < DT; d++)
                x[(size_t)t * DT + d] += ca_out[(size_t)t * DT + d];
        
        // FFN with residual
        for (int t = 0; t < N; t++) {
            float* xt = &x[(size_t)t * DT];
            layernorm(x2.data(), xt, l.ffn_ln_w.data(), l.ffn_ln_b.data(), DT, 1e-5f);
            matmul(ffn_gate.data(), x2.data(), l.ffn_gate_w.data(), DT * 4, DT);
            if (!l.ffn_gate_b.empty()) for (int i = 0; i < DT * 4; i++) ffn_gate[i] += l.ffn_gate_b[i];
            for (int i = 0; i < DT * 4; i++) ffn_gate[i] = gelu(ffn_gate[i]);
            matmul(ffn_up.data(), ffn_gate.data(), l.ffn_down_w.data(), DT, DT * 4);
            if (!l.ffn_down_b.empty()) for (int i = 0; i < DT; i++) ffn_up[i] += l.ffn_down_b[i];
            for (int d = 0; d < DT; d++) xt[d] += ffn_up[d];
        }
    }
    
    // Final LN + output projection
    std::vector<float> logits((size_t)cfg.n_vocab);
    {
        float* xt = &x[(size_t)(N - 1) * DT];
        layernorm(x2.data(), xt, model.dec_ln_w.data(), model.dec_ln_b.data(), DT, 1e-5f);
        if (!model.output_w.empty()) {
            matmul(logits.data(), x2.data(), model.output_w.data(), cfg.n_vocab, DT);
        } else {
            // Tied embeddings
            for (int i = 0; i < cfg.n_vocab; i++) {
                float s = 0;
                for (int j = 0; j < DT; j++) s += x2[j] * model.dec_token_emb[(size_t)i * DT + j];
                logits[i] = s;
            }
        }
    }
    
    return logits;
}

// ====================================================================
// Full transcription
// ====================================================================
std::string whisper_transcribe(const WhisperModel& model, const float* audio_pcm, int n_samples) {
    // Compute log-mel spectrogram
    int sr = 16000;
    auto mel = whisper_log_mel_spectrogram(audio_pcm, n_samples, sr, model.cfg.n_mels);
    if (mel.empty()) return "";
    
    // Run encoder
    int n_frames = (int)(mel.size() / model.cfg.n_mels);
    auto enc_out = whisper_encode(model, mel.data(), n_frames);
    int n_enc_ctx = (int)(enc_out.size() / model.cfg.n_audio_state);
    
    // Greedy decoder loop
    std::vector<int> tokens = {WHISPER_SOT, WHISPER_TRANSCRIBE, WHISPER_ENGLISH};
    std::vector<float> kv_cache; // not used in CPU impl
    std::vector<int> output;
    
    for (int step = 0; step < model.cfg.n_text_ctx; step++) {
        auto logits = whisper_decode_step(model, tokens, enc_out.data(), n_enc_ctx, kv_cache);
        
        // Greedy: argmax
        int next = 0;
        float max_val = logits[0];
        for (int i = 1; i < model.cfg.n_vocab; i++) {
            if (logits[i] > max_val) { max_val = logits[i]; next = i; }
        }
        
        if (next == WHISPER_EOT) break;
        output.push_back(next);
        tokens.push_back(next);
    }
    
    // Basic token → text (byte-level mapping for English BPE)
    std::string text;
    for (int id : output) {
        if (id < 256) {
            text += (char)id;
        } else {
            char buf[32]; snprintf(buf, sizeof(buf), "[%d]", id);
            text += buf;
        }
    }
    return text;
}
