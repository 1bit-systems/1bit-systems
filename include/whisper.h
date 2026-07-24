// whisper.h — OpenAI Whisper speech-to-text model
//
// Architecture:
//   Audio → log-mel spectrogram → Conv1D + GELU → Sinusoidal pos → Encoder × N
//     → Cross-attention Decoder × N → Linear → text tokens
//
// Model sizes:
//   tiny.en  (39M)  : 4 enc layers, 4 dec layers, 384 hidden, 6 heads
//   base.en  (74M)  : 6 enc layers, 6 dec layers, 512 hidden, 8 heads
//   small.en (244M) : 12 enc layers, 12 dec layers, 768 hidden, 12 heads
//   medium   (769M) : 24 enc layers, 24 dec layers, 1024 hidden, 16 heads
//   large    (1550M): 32 enc layers, 32 dec layers, 1280 hidden, 20 heads
//
// Input: 80-channel log-mel spectrogram, 30 seconds = 3000 frames @ 100Hz
// Output: text tokens (tokenizer: 51866 multilingual / 51865 English-only)
//
// Weights are loaded from a single GGUF file (the llama.cpp whisper format).

#pragma once

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

// ─── Whisper model config ──────────────────────────────────────────
struct WhisperConfig {
    int n_mels      = 80;     // mel filterbank channels
    int n_audio_ctx = 1500;   // audio context size (30s @ 50Hz after conv stride)
    int n_audio_state = 384;  // encoder hidden dim
    int n_audio_head = 6;     // encoder attention heads
    int n_audio_layer = 4;    // encoder layers
    int n_text_state = 384;   // decoder hidden dim
    int n_text_head = 6;      // decoder attention heads
    int n_text_layer = 4;     // decoder layers
    int n_vocab     = 51865;  // vocabulary size (English)
    int n_text_ctx  = 448;    // max text context (tokens)
    int max_seg     = 0;       // loaded from model

    // Factory for known sizes
    static WhisperConfig tiny_en() {
        WhisperConfig c;
        c.n_audio_state = 384; c.n_audio_head = 6;  c.n_audio_layer = 4;
        c.n_text_state  = 384; c.n_text_head  = 6;  c.n_text_layer  = 4;
        c.n_vocab = 51865;
        return c;
    }
    static WhisperConfig base_en() {
        WhisperConfig c;
        c.n_audio_state = 512; c.n_audio_head = 8;  c.n_audio_layer = 6;
        c.n_text_state  = 512; c.n_text_head  = 8;  c.n_text_layer  = 6;
        c.n_vocab = 51865;
        return c;
    }
    static WhisperConfig small_en() {
        WhisperConfig c;
        c.n_audio_state = 768; c.n_audio_head = 12; c.n_audio_layer = 12;
        c.n_text_state  = 768; c.n_text_head  = 12; c.n_text_layer  = 12;
        c.n_vocab = 51865;
        return c;
    }
    static WhisperConfig medium() {
        WhisperConfig c;
        c.n_audio_state = 1024; c.n_audio_head = 16; c.n_audio_layer = 24;
        c.n_text_state  = 1024; c.n_text_head  = 16; c.n_text_layer  = 24;
        c.n_vocab = 51865;
        return c;
    }
    static WhisperConfig large() {
        WhisperConfig c;
        c.n_audio_state = 1280; c.n_audio_head = 20; c.n_audio_layer = 32;
        c.n_text_state  = 1280; c.n_text_head  = 20; c.n_text_layer  = 32;
        c.n_vocab = 51865;
        return c;
    }
};

// ─── Whisper special tokens ────────────────────────────────────────
enum WhisperToken : int {
    WHISPER_SOT      = 50257,  // start-of-transcript
    WHISPER_ENC      = 50258,  // no-timestamps
    WHISPER_NONE     = 50259,  // (unused)
    WHISPER_EOT      = 50256,  // end-of-transcript
    WHISPER_TRANSCRIBE = 50359, // transcribe task
    WHISPER_ENGLISH  = 50259,  // English language token
};

// ─── Whisper weights (per layer) ───────────────────────────────────
struct WhisperLayerWeights {
    // Self-attention (encoder & decoder)
    std::vector<float> attn_q_w, attn_q_b;
    std::vector<float> attn_k_w, attn_k_b;
    std::vector<float> attn_v_w, attn_v_b;
    std::vector<float> attn_o_w, attn_o_b;
    // Cross-attention (decoder only)
    std::vector<float> cross_q_w, cross_q_b;
    std::vector<float> cross_k_w, cross_k_b;
    std::vector<float> cross_v_w, cross_v_b;
    std::vector<float> cross_o_w, cross_o_b;
    // FFN
    std::vector<float> ffn_gate_w, ffn_gate_b; // GELU gate
    std::vector<float> ffn_up_w, ffn_up_b;
    std::vector<float> ffn_down_w, ffn_down_b;
    // LayerNorm
    std::vector<float> attn_ln_w, attn_ln_b;
    std::vector<float> cross_ln_w, cross_ln_b; // decoder only
    std::vector<float> ffn_ln_w, ffn_ln_b;
};

// ─── Full Whisper model weights ────────────────────────────────────
struct WhisperModel {
    WhisperConfig cfg;

    // Encoder
    std::vector<float> enc_conv1_w, enc_conv1_b;  // conv1d: [n_mels, n_audio_state] × 3
    std::vector<float> enc_conv2_w, enc_conv2_b;  // conv1d: [n_audio_state, n_audio_state] × 3
    std::vector<float> enc_pos_emb;                // sinusoidal pos emb [n_audio_ctx, n_audio_state]
    std::vector<float> enc_ln_w, enc_ln_b;         // final encoder LN
    std::vector<WhisperLayerWeights> enc_layers;

    // Decoder
    std::vector<float> dec_pos_emb;                // learned pos emb [n_text_ctx, n_text_state]
    std::vector<float> dec_token_emb;              // token embedding [n_vocab, n_text_state]
    std::vector<float> dec_ln_w, dec_ln_b;         // final decoder LN
    std::vector<WhisperLayerWeights> dec_layers;

    // Output projection (tied or separate)
    std::vector<float> output_w;                   // [n_vocab, n_text_state]

    // Token id -> raw vocab string (GPT2 byte-level-BPE alphabet, i.e. NOT
    // yet decoded back to real bytes — see whisper_decode_token in
    // whisper.cpp for that). Loaded from the "tokenizer.ggml.tokens" GGUF
    // array; empty if the GGUF has none (falls back to the byte-literal
    // placeholder decode).
    std::vector<std::string> vocab;

    bool load_from_gguf(const std::string& path, const WhisperConfig* override_cfg = nullptr);
    void clear();
};

// ─── Audio processing ──────────────────────────────────────────────
// Load WAV file and return 16-bit PCM samples
std::vector<float> whisper_load_wav(const std::string& path, int* out_sample_rate = nullptr);

// Compute log-mel spectrogram: 80 mel bands × N frames
// Input: float PCM mono audio at 16000 Hz
// Output: log-mel spectrogram [n_mels × n_frames]
std::vector<float> whisper_log_mel_spectrogram(
    const float* audio, int n_samples, int sample_rate,
    int n_mels = 80, int n_fft = 400, int hop = 160);

// ─── Math helpers (shared with vision_encoder conventions) ─────────
namespace whisper_math {
    static inline float gelu(float x) {
        const float c = 0.7978845608f;
        return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
    }
    static inline float silu(float x) { return x / (1.0f + expf(-x)); }
    static inline float relu(float x) { return x > 0 ? x : 0; }

    static inline void matmul(float* out, const float* in, const float* w, int M, int K) {
        for (int i = 0; i < M; i++) {
            float s = 0;
            for (int j = 0; j < K; j++) s += in[j] * w[(size_t)i * K + j];
            out[i] = s;
        }
    }

    static inline void layernorm(float* out, const float* x, const float* w, const float* b, int n, float eps) {
        float mean = 0; for (int i = 0; i < n; i++) mean += x[i]; mean /= n;
        float var = 0; for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; } var /= n;
        float inv = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
    }

    static inline void softmax_inplace(float* x, int n) {
        float mx = x[0]; for (int i = 1; i < n; i++) mx = std::max(mx, x[i]);
        float sum = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
        float inv = 1.0f / sum; for (int i = 0; i < n; i++) x[i] *= inv;
    }

    // 1D convolution (conv1d): out[C_out, L_out] = in[C_in, L_in] * weight[C_out, C_in, K]
    // K = kernel_size, stride = 1 (Whisper uses stride=1 for conv2, stride not important for our CPU impl)
    static inline void conv1d(float* out, const float* in, int C_in, int L_in,
                               const float* w, const float* b, int C_out, int K, int stride = 1) {
        int L_out = (L_in - K) / stride + 1;
        for (int co = 0; co < C_out; co++) {
            for (int lo = 0; lo < L_out; lo++) {
                float s = b ? b[co] : 0;
                for (int ci = 0; ci < C_in; ci++) {
                    for (int ki = 0; ki < K; ki++) {
                        int li = lo * stride + ki;
                        s += in[ci * (size_t)L_in + li] * w[(size_t)co * C_in * K + (size_t)ci * K + ki];
                    }
                }
                out[co * (size_t)L_out + lo] = s;
            }
        }
    }

    // Full bidirectional self-attention
    static inline void self_attn(float* out, const float* x, int N, int D,
                                  const float* q_w, const float* q_b,
                                  const float* k_w, const float* k_b,
                                  const float* v_w, const float* v_b,
                                  const float* o_w, const float* o_b,
                                  int n_heads) {
        int H = D / n_heads;
        std::vector<float> Q((size_t)N * D), K((size_t)N * D), V((size_t)N * D);
        std::vector<float> scores(N), attn(D);

        // Project Q, K, V
        for (int t = 0; t < N; t++) {
            matmul(&Q[(size_t)t * D], &x[(size_t)t * D], q_w, D, D);
            matmul(&K[(size_t)t * D], &x[(size_t)t * D], k_w, D, D);
            matmul(&V[(size_t)t * D], &x[(size_t)t * D], v_w, D, D);
            // Real Whisper: key projection has no bias (k_b is null/empty
            // for a correctly-loaded model) — checked independently, not
            // bundled under q_b, so a missing k_b doesn't skip q_b/v_b or
            // dereference a null k_b.
            if (q_b) for (int i = 0; i < D; i++) Q[(size_t)t * D + i] += q_b[i];
            if (k_b) for (int i = 0; i < D; i++) K[(size_t)t * D + i] += k_b[i];
            if (v_b) for (int i = 0; i < D; i++) V[(size_t)t * D + i] += v_b[i];
        }

        float scale = 1.0f / sqrtf((float)H);
        for (int t = 0; t < N; t++) {
            std::fill(attn.begin(), attn.end(), 0.0f);
            for (int h = 0; h < n_heads; h++) {
                float* Qh = &Q[(size_t)t * D + (size_t)h * H];
                for (int s = 0; s < N; s++) {
                    float* Kh = &K[(size_t)s * D + (size_t)h * H];
                    float acc = 0; for (int d = 0; d < H; d++) acc += Qh[d] * Kh[d];
                    scores[s] = acc * scale;
                }
                softmax_inplace(scores.data(), N);
                for (int d = 0; d < H; d++) {
                    float acc = 0;
                    for (int s = 0; s < N; s++) acc += scores[s] * V[(size_t)s * D + (size_t)h * H + d];
                    attn[(size_t)h * H + d] = acc;
                }
            }
            matmul(&out[(size_t)t * D], attn.data(), o_w, D, D);
            if (o_b) for (int i = 0; i < D; i++) out[(size_t)t * D + i] += o_b[i];
        }
    }

    // Cross-attention: queries from decoder, keys/values from encoder
    static inline void cross_attn(float* out, const float* q_from, int N_dec, int D,
                                   const float* enc_out, int N_enc,
                                   const float* q_w, const float* q_b,
                                   const float* k_w, const float* k_b,
                                   const float* v_w, const float* v_b,
                                   const float* o_w, const float* o_b,
                                   int n_heads) {
        int H = D / n_heads;
        std::vector<float> Q((size_t)N_dec * D), K((size_t)N_enc * D), V((size_t)N_enc * D);
        std::vector<float> scores(N_enc), attn(D);

        for (int t = 0; t < N_dec; t++) matmul(&Q[(size_t)t * D], &q_from[(size_t)t * D], q_w, D, D);
        for (int t = 0; t < N_enc; t++) matmul(&K[(size_t)t * D], &enc_out[(size_t)t * D], k_w, D, D);
        for (int t = 0; t < N_enc; t++) matmul(&V[(size_t)t * D], &enc_out[(size_t)t * D], v_w, D, D);
        if (q_b) for (int t = 0; t < N_dec; t++) for (int i = 0; i < D; i++) Q[(size_t)t * D + i] += q_b[i];
        if (k_b) for (int t = 0; t < N_enc; t++) for (int i = 0; i < D; i++) K[(size_t)t * D + i] += k_b[i];
        if (v_b) for (int t = 0; t < N_enc; t++) for (int i = 0; i < D; i++) V[(size_t)t * D + i] += v_b[i];

        float scale = 1.0f / sqrtf((float)H);
        for (int t = 0; t < N_dec; t++) {
            std::fill(attn.begin(), attn.end(), 0.0f);
            for (int h = 0; h < n_heads; h++) {
                float* Qh = &Q[(size_t)t * D + (size_t)h * H];
                for (int s = 0; s < N_enc; s++) {
                    float* Kh = &K[(size_t)s * D + (size_t)h * H];
                    float acc = 0; for (int d = 0; d < H; d++) acc += Qh[d] * Kh[d];
                    scores[s] = acc * scale;
                }
                softmax_inplace(scores.data(), N_enc);
                for (int d = 0; d < H; d++) {
                    float acc = 0;
                    for (int s = 0; s < N_enc; s++) acc += scores[s] * V[(size_t)s * D + (size_t)h * H + d];
                    attn[(size_t)h * H + d] = acc;
                }
            }
            matmul(&out[(size_t)t * D], attn.data(), o_w, D, D);
            if (o_b) for (int i = 0; i < D; i++) out[(size_t)t * D + i] += o_b[i];
        }
    }
}

// ─── Whisper encoder forward ───────────────────────────────────────
// audio: log-mel spectrogram [n_mels × n_frames] (80 × ~3000 for 30s)
// output: encoder features [n_audio_ctx × n_audio_state] (1500 × hidden)
std::vector<float> whisper_encode(const WhisperModel& model, const float* mel, int n_frames);

// ─── Whisper decoder forward (single step) ─────────────────────────
// tokens: current token sequence [n_tokens]
// enc_out: encoder output from whisper_encode
// kv_cache: persistent key/value cache (caller manages, pass same pointer each step)
// Returns logits over vocabulary [n_vocab]
std::vector<float> whisper_decode_step(const WhisperModel& model, const std::vector<int>& tokens,
                                        const float* enc_out, int n_enc_ctx,
                                        std::vector<float>& kv_cache);

// ─── Whisper full transcription ────────────────────────────────────
// audio_pcm: 16kHz mono PCM audio samples
// Returns transcribed text
std::string whisper_transcribe(const WhisperModel& model, const float* audio_pcm, int n_samples);

// Decodes one GPT2 byte-level-BPE vocab entry (as stored in the GGUF —
// UTF-8 text using the standard byte<->unicode remapping alphabet, e.g.
// a literal space is the two-byte UTF-8 sequence for U+0120 'Ġ', not an
// ASCII space) back to its original raw bytes. Malformed input passes
// through unchanged rather than throwing, since this feeds directly into
// user-visible transcription output.
std::string whisper_decode_bpe_token(const std::string& token_text);
