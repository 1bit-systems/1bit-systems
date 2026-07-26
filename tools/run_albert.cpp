// run_albert.cpp — Pure C++ Albert-MoE-13 inference (CPU, libc only)
// Build: g++ -std=c++17 -O3 run_albert.cpp -o run_albert && ./run_albert "prompt"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

constexpr int kHidden    = 256;
constexpr int kNHeads    = 4;
constexpr int kHeadDim   = 64;
constexpr int kNExp      = 12;
constexpr int kTopK      = 3;
constexpr int kVocab     = 32000;
constexpr int kFfnDim    = 1024;
constexpr int kMaxSeq    = 256;

// ── Model types ─────────────────────────────────────────────────────────
struct Tensor {
    char     key[128]{};
    float   *data   = nullptr;
    int64_t  numel  = 0;
};

struct Model {
    Tensor *t        = nullptr;
    int     count    = 0;
    float  *raw      = nullptr;
    size_t  raw_size = 0;
};

// ── Load model weights via binary index ─────────────────────────────────
static Model load_model(const char *sf_path, const char *idx_path) {
    Model m{};
    // Read index
    FILE *fi = std::fopen(idx_path, "rb");
    if (!fi) { std::perror(idx_path); return m; }
    int count;
    std::fread(&count, 4, 1, fi);
    m.t     = static_cast<Tensor *>(std::calloc(count, sizeof(Tensor)));
    m.count = count;

    uint8_t *name_buf = nullptr;
    size_t   name_cap = 0;
    for (int i = 0; i < count; i++) {
        int name_len;
        if (!std::fread(&name_len, 4, 1, fi)) break;
        if (name_len > static_cast<int>(name_cap)) {
            name_cap = name_len + 1;
            name_buf = static_cast<uint8_t *>(std::realloc(name_buf, name_cap));
        }
        if (!std::fread(name_buf, 1, name_len, fi)) break;
        name_buf[name_len] = 0;
        std::snprintf(m.t[i].key, sizeof(m.t[i].key), "%s",
                     reinterpret_cast<char *>(name_buf));
        int64_t off, nel;
        if (!std::fread(&off, 8, 1, fi)) break;
        if (!std::fread(&nel, 8, 1, fi)) break;
        m.t[i].numel = nel;
        m.t[i].data  = nullptr;
    }
    std::free(name_buf);
    std::fclose(fi);
    std::printf("Read %d index entries\n", count);

    // Read safetensors data
    FILE *fs = std::fopen(sf_path, "rb");
    if (!fs) { std::perror(sf_path); return m; }
    std::fseek(fs, 0, SEEK_END);
    m.raw_size = std::ftell(fs);
    std::fseek(fs, 0, SEEK_SET);
    m.raw = static_cast<float *>(std::malloc(m.raw_size));
    std::fread(m.raw, 1, m.raw_size, fs);
    std::fclose(fs);

    // Map data pointers — re-read index for proper offsets
    fi = std::fopen(idx_path, "rb");
    std::fread(&count, 4, 1, fi);
    for (int i = 0; i < count && i < m.count; i++) {
        int name_len;
        std::fread(&name_len, 4, 1, fi);
        std::fseek(fi, name_len, SEEK_CUR);
        int64_t offset, numel;
        std::fread(&offset, 8, 1, fi);
        std::fread(&numel, 8, 1, fi);
        if (i < m.count) {
            m.t[i].data  = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(m.raw) + offset);
            m.t[i].numel = numel;
        }
    }
    std::fclose(fi);
    std::printf("Model: %d tensors, %.1f MB\n", count, m.raw_size / 1e6);
    return m;
}

static float *get_w(Model *m, const char *name) {
    for (int i = 0; i < m->count; i++)
        if (!std::strcmp(m->t[i].key, name)) return m->t[i].data;
    std::fprintf(stderr, "Missing: %s\n", name);
    return nullptr;
}

// Helper: get weight or return zero-filled fallback (replaces GCC statement-expression macro)
static float *W_or_zero(Model *m, const char *name) {
    float *w = get_w(m, name);
    if (w) return w;
    static float zero_buf[kHidden]{};
    return zero_buf;
}

// ── Math ────────────────────────────────────────────────────────────────
static float gelu(float x) {
    float t = std::tanh(0.79788456f * (x + 0.044715f * x * x * x));
    return 0.5f * x * (1.0f + t);
}

static void layer_norm(float *x, float *w, float *b, int n) {
    float mean = 0, var = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var = std::sqrt(var / n + 1e-5f);
    for (int i = 0; i < n; i++) x[i] = (x[i] - mean) / var * w[i] + b[i];
}

static void matmul_t(float *out, float *a, float *b, int m, int n, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int t = 0; t < k; t++) s += a[i * k + t] * b[j * k + t];
            out[i * n + j] = s;
        }
}

// ── Forward buffers ─────────────────────────────────────────────────────
struct ForwardBuf {
    float  *buf     = nullptr;
    float  *q       = nullptr;
    float  *k       = nullptr;
    float  *v       = nullptr;
    float  *attn    = nullptr;
    float  *tmp     = nullptr;
    float  *gs      = nullptr;
    float  *fc_w    = nullptr;
    float  *fc_b    = nullptr;
    float  *proj_w  = nullptr;
    float  *proj_b  = nullptr;
    int     max_seq = 0;
};

static ForwardBuf forward_buf_alloc(int max_seq) {
    ForwardBuf b;
    b.max_seq = max_seq;
    b.buf    = static_cast<float *>(std::calloc(static_cast<size_t>(max_seq) * kHidden, sizeof(float)));
    b.q      = static_cast<float *>(std::calloc(static_cast<size_t>(max_seq) * kHidden, sizeof(float)));
    b.k      = static_cast<float *>(std::calloc(static_cast<size_t>(max_seq) * kHidden, sizeof(float)));
    b.v      = static_cast<float *>(std::calloc(static_cast<size_t>(max_seq) * kHidden, sizeof(float)));
    b.attn   = static_cast<float *>(std::calloc(static_cast<size_t>(max_seq) * max_seq, sizeof(float)));
    b.tmp    = static_cast<float *>(std::calloc(static_cast<size_t>(max_seq) * kHidden, sizeof(float)));
    b.gs     = static_cast<float *>(std::calloc(static_cast<size_t>(max_seq) * kNExp, sizeof(float)));
    b.fc_w   = static_cast<float *>(std::calloc(static_cast<size_t>(kHidden) * kFfnDim, sizeof(float)));
    b.fc_b   = static_cast<float *>(std::calloc(kFfnDim, sizeof(float)));
    b.proj_w = static_cast<float *>(std::calloc(static_cast<size_t>(kFfnDim) * kHidden, sizeof(float)));
    b.proj_b = static_cast<float *>(std::calloc(kHidden, sizeof(float)));
    return b;
}

static void forward_buf_free(ForwardBuf *b) {
    std::free(b->buf); std::free(b->q); std::free(b->k); std::free(b->v);
    std::free(b->attn); std::free(b->tmp); std::free(b->gs);
    std::free(b->fc_w); std::free(b->fc_b);
    std::free(b->proj_w); std::free(b->proj_b);
    std::memset(b, 0, sizeof(*b));
}

// ── Forward pass ────────────────────────────────────────────────────────
static void forward(float *x, int seq, Model *m, ForwardBuf *fb) {
    if (seq > fb->max_seq) {
        std::fprintf(stderr, "seq=%d exceeds max_seq=%d\n", seq, fb->max_seq);
        std::exit(1);
    }
    float *buf    = fb->buf;
    float *q_buf  = fb->q;
    float *k_buf  = fb->k;
    float *v_buf  = fb->v;
    float *attn   = fb->attn;
    float *tmp_b  = fb->tmp;
    float *gs     = fb->gs;
    float *fc_w   = fb->fc_w;
    float *fc_b   = fb->fc_b;
    float *proj_w = fb->proj_w;
    float *proj_b = fb->proj_b;

    std::printf("DEBUG: m->count=%d raw=%p raw_size=%zu\n",
                m->count, static_cast<void *>(m->raw), m->raw_size);
    std::fflush(stdout);
    std::printf("DEBUG: first key='%s'\n", m->t[0].key);
    std::fflush(stdout);

    // Embed
    float *embed_w = W_or_zero(m, "model.embed_tokens.weight");
    for (int i = 0; i < seq; i++)
        for (int j = 0; j < kHidden; j++)
            x[i * kHidden + j] = embed_w[0 * kHidden + j];  // simplified: use token 0 embedding

    // Layer loop
    for (int l = 0; l < 28; l++) {
        char prefix[64];
        std::snprintf(prefix, sizeof(prefix), "model.layers.%d.", l);

        // Self-attention
        float *qkv_w = W_or_zero(m, (std::string(prefix) + "self_attn.q_proj.weight").c_str());
        float *o_w   = W_or_zero(m, (std::string(prefix) + "self_attn.o_proj.weight").c_str());

        for (int i = 0; i < seq; i++) {
            for (int j = 0; j < kHidden; j++) {
                float s = 0;
                for (int t = 0; t < kHidden; t++)
                    s += x[i * kHidden + t] * qkv_w[j * kHidden + t];
                q_buf[i * kHidden + j] = s;
            }
        }

        // Simple attention: just pass through Q (placeholder for full MHA)
        for (int i = 0; i < seq * kHidden; i++) buf[i] = q_buf[i];

        for (int i = 0; i < seq; i++)
            for (int j = 0; j < kHidden; j++) {
                float s = 0;
                for (int t = 0; t < kHidden; t++)
                    s += buf[i * kHidden + t] * o_w[j * kHidden + t];
                tmp_b[i * kHidden + j] = x[i * kHidden + j] + s;
            }

        // Input layer norm
        float *ln_w = W_or_zero(m, (std::string(prefix) + "input_layernorm.weight").c_str());
        float *ln_b = W_or_zero(m, (std::string(prefix) + "input_layernorm.bias").c_str());
        for (int i = 0; i < seq; i++)
            layer_norm(tmp_b + i * kHidden, ln_w, ln_b, kHidden);

        // MoE FFN: gate + expert dispatch
        float *gate_w = W_or_zero(m, (std::string(prefix) + "mlp.gate.weight").c_str());
        for (int i = 0; i < seq; i++)
            for (int e = 0; e < kNExp; e++) {
                float s = 0;
                for (int t = 0; t < kHidden; t++)
                    s += tmp_b[i * kHidden + t] * gate_w[e * kHidden + t];
                gs[i * kNExp + e] = s;
            }

        // Top-K expert selection (simplified: always use expert 0)
        for (int i = 0; i < seq; i++) {
            float *up_w   = W_or_zero(m, (std::string(prefix) + "mlp.experts.0.up_proj.weight").c_str());
            float *down_w = W_or_zero(m, (std::string(prefix) + "mlp.experts.0.down_proj.weight").c_str());

            for (int j = 0; j < kFfnDim; j++) {
                float s = 0;
                for (int t = 0; t < kHidden; t++)
                    s += tmp_b[i * kHidden + t] * up_w[j * kHidden + t];
                fc_b[j] = gelu(s);
            }
            for (int j = 0; j < kHidden; j++) {
                float s = 0;
                for (int t = 0; t < kFfnDim; t++)
                    s += fc_b[t] * down_w[j * kFfnDim + t];
                x[i * kHidden + j] = tmp_b[i * kHidden + j] + s;
            }
        }
    }

    // Final layer norm
    float *final_ln_w = W_or_zero(m, "model.norm.weight");
    float *final_ln_b = W_or_zero(m, "model.norm.bias");
    for (int i = 0; i < seq; i++)
        layer_norm(x + i * kHidden, final_ln_w, final_ln_b, kHidden);

    // LM head
    float *lm_head_w = W_or_zero(m, "lm_head.weight");

    // Sample last token
    float *logits = static_cast<float *>(std::calloc(kVocab, sizeof(float)));
    if (!logits) { std::fprintf(stderr, "Out of memory allocating logits\n"); std::exit(1); }
    for (int v = 0; v < kVocab; v++) {
        float s = 0;
        for (int j = 0; j < kHidden; j++)
            s += x[(seq - 1) * kHidden + j] * lm_head_w[v * kHidden + j];
        logits[v] = s;
    }

    int next = 0;
    for (int v = 1; v < kVocab; v++)
        if (logits[v] > logits[next]) next = v;
    std::printf("  next token: %d\n", next);
    std::free(logits);
}

int main(int argc, char **argv) {
    std::printf("Albert-MoE-13 C++ Inference\n");
    Model m = load_model(
        "/home/bcloud/models/albert-moe/albert_v3.0.best.safetensors",
        "/tmp/albert_idx.bin");
    if (!m.count) return 1;

    int tokens[kMaxSeq]{};
    int seq = 3;

    ForwardBuf fb = forward_buf_alloc(kMaxSeq);
    float *x = fb.buf;  // reuse the heap buffer for input

    clock_t t0 = std::clock();
    forward(x, seq, &m, &fb);
    float ms = static_cast<float>(std::clock() - t0) / CLOCKS_PER_SEC * 1000;
    std::printf("Forward: %.0fms\n", ms);

    forward_buf_free(&fb);
    std::free(m.raw);
    std::free(m.t);
    return 0;
}
