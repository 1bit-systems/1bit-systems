// run_albert_opt.cpp — Optimized Albert-MoE-13 C++ inference
// Pre-loads all weights, supports top-3 routing, RoPE, KV cache
//
// Build: g++ -std=c++17 -O3 -march=native run_albert_opt.cpp -lm -o run_albert_opt
// Run:   ./run_albert_opt "your prompt"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cstdint>
#include <sys/stat.h>

// ── Architecture ──
constexpr int H = 256;
constexpr int N_HEADS = 4;
constexpr int HEAD_DIM = 64;
constexpr int N_EXP = 12;
constexpr int TOP_K = 3;
constexpr int VOCAB = 32000;
constexpr int FFN_HIDDEN = (H * 4);  // 1024
constexpr int MAX_SEQ = 256;
constexpr int N_BLOCKS = 31;
constexpr int N_STREAMS = 2;
constexpr float ROPE_THETA = 10000.0f;

// ── Weight store ──
struct Tensor { char key[128]; float *data; int64_t numel; };
struct WeightStore {
    Tensor *tensors;
    int count;
    float *raw;
    size_t raw_size;
};

static WeightStore ws = {0};

static int tensor_cmp(const void *a, const void *b) {
    return std::strcmp(static_cast<const Tensor*>(a)->key, static_cast<const Tensor*>(b)->key);
}

static float *get_w(const char *name) {
    int lo = 0, hi = ws.count - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int cmp = std::strcmp(ws.tensors[mid].key, name);
        if (cmp == 0) return ws.tensors[mid].data;
        else if (cmp < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return nullptr;
}

static int load_all_weights(const char *dir) {
    // Validate dir: reject shell metacharacters to prevent command injection
    if (!dir || !dir[0]) { std::fprintf(stderr, "Invalid directory\n"); return 0; }
    for (const char *p = dir; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '/' || c == '.' ||
              c == '-' || c == '_')) {
            std::fprintf(stderr, "Invalid character in directory path: '%c'\n", c);
            return 0;
        }
    }
    char cmd[512];
    std::snprintf(cmd, sizeof(cmd), "ls %s/*.bin 2>/dev/null | wc -l", dir);
    FILE *f = popen(cmd, "r");
    if (!f) { std::fprintf(stderr, "Can't list %s\n", dir); return 0; }
    int n_files = 0;
    std::fscanf(f, "%d", &n_files);
    pclose(f);
    if (n_files == 0) { std::fprintf(stderr, "No .bin files in %s\n", dir); return 0; }

    ws.tensors = static_cast<Tensor*>(std::calloc(n_files, sizeof(Tensor)));
    ws.count = n_files;

    std::snprintf(cmd, sizeof(cmd), "ls %s/*.bin", dir);
    f = popen(cmd, "r");
    if (!f) return 0;

    char filepath[512];
    int idx = 0;
    size_t total_bytes = 0;
    while (std::fgets(filepath, sizeof(filepath), f) && idx < n_files) {
        size_t len = std::strlen(filepath);
        if (len > 0 && filepath[len-1] == '\n') filepath[len-1] = 0;

        const char *base = std::strrchr(filepath, '/');
        base = base ? base + 1 : filepath;
        const char *dot = std::strrchr(base, '.');
        if (!dot || std::strcmp(dot, ".bin") != 0) continue;

        size_t key_len = dot - base;
        if (key_len >= sizeof(ws.tensors[idx].key)) key_len = sizeof(ws.tensors[idx].key) - 1;
        std::memcpy(ws.tensors[idx].key, base, key_len);
        ws.tensors[idx].key[key_len] = 0;

        struct stat st;
        if (stat(filepath, &st) != 0) { continue; }
        ws.tensors[idx].numel = st.st_size / 4;
        ws.tensors[idx].data = nullptr;
        total_bytes += st.st_size;
        idx++;
    }
    pclose(f);
    ws.count = idx;

    ws.raw = static_cast<float*>(std::malloc(total_bytes));
    if (!ws.raw) { std::fprintf(stderr, "Failed to allocate %.1f MB\n", total_bytes/1e6); return 0; }
    ws.raw_size = total_bytes;

    std::snprintf(cmd, sizeof(cmd), "ls %s/*.bin", dir);
    f = popen(cmd, "r");
    if (!f) return 0;

    size_t offset = 0;
    idx = 0;
    while (std::fgets(filepath, sizeof(filepath), f) && idx < ws.count) {
        size_t len = std::strlen(filepath);
        if (len > 0 && filepath[len-1] == '\n') filepath[len-1] = 0;

        if (std::strncmp(ws.tensors[idx].key, std::strrchr(filepath, '/')+1, std::strlen(ws.tensors[idx].key)) != 0) continue;

        FILE *bin = std::fopen(filepath, "rb");
        if (!bin) { continue; }
        std::fseek(bin, 0, SEEK_END);
        size_t sz = std::ftell(bin);
        std::fseek(bin, 0, SEEK_SET);
        size_t got = std::fread(static_cast<uint8_t*>(static_cast<void*>(ws.raw)) + offset, 1, sz, bin);
        (void)got;
        std::fclose(bin);

        ws.tensors[idx].data = reinterpret_cast<float*>(static_cast<uint8_t*>(static_cast<void*>(ws.raw)) + offset);
        offset += sz;
        idx++;
    }
    pclose(f);

    std::qsort(ws.tensors, ws.count, sizeof(Tensor), tensor_cmp);
    std::printf("Loaded %d weights, %.1f MB\n", ws.count, ws.raw_size/1e6);
    return 1;
}

static float gelu(float x) {
    float t = std::tanh(0.79788456f * (x + 0.044715f * x * x * x));
    return 0.5f * x * (1.0f + t);
}

static void layer_norm(float *x, float *w, float *b, int n) {
    float mean = 0, var = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var = std::sqrt(var / n + 1e-5f);
    for (int i = 0; i < n; i++) x[i] = (x[i] - mean) / var * w[i] + b[i];
}

static void matmul_t(float *out, float *a, float *b, int m, int n, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int t = 0; t < k; t++) s += a[i*k+t] * b[j*k+t];
            out[i*n+j] = s;
        }
}

static void precompute_rope(float *cos_cache, float *sin_cache, int max_seq, int dim) {
    for (int pos = 0; pos < max_seq; pos++)
        for (int d = 0; d < dim; d++) {
            float theta = pos * std::pow(ROPE_THETA, -2.0f * d / dim);
            cos_cache[pos*dim + d] = std::cos(theta);
            sin_cache[pos*dim + d] = std::sin(theta);
        }
}

static void attention(float *out, float *q, float *k, float *k_cache, float *v_cache,
                      int pos, float *cos_cache, float *sin_cache) {
    for (int h = 0; h < N_HEADS; h++) {
        float *q_h = &q[h * HEAD_DIM];
        float *k_h = &k[h * HEAD_DIM];
        float *cos = &cos_cache[pos * HEAD_DIM];
        float *sin = &sin_cache[pos * HEAD_DIM];
        for (int d = 0; d < HEAD_DIM/2; d++) {
            float q1 = q_h[d], q2 = q_h[d + HEAD_DIM/2];
            q_h[d] = q1 * cos[d] - q2 * sin[d];
            q_h[d + HEAD_DIM/2] = q1 * sin[d] + q2 * cos[d];
            float k1 = k_h[d], k2 = k_h[d + HEAD_DIM/2];
            k_h[d] = k1 * cos[d] - k2 * sin[d];
            k_h[d + HEAD_DIM/2] = k1 * sin[d] + k2 * cos[d];
        }
    }
    std::memcpy(&k_cache[pos * N_HEADS * HEAD_DIM], k, N_HEADS * HEAD_DIM * sizeof(float));

    float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    for (int h = 0; h < N_HEADS; h++) {
        float *q_h = &q[h * HEAD_DIM];
        float scores[MAX_SEQ];
        for (int t = 0; t <= pos; t++) {
            float *k_h = &k_cache[(t * N_HEADS + h) * HEAD_DIM];
            float s = 0;
            for (int d = 0; d < HEAD_DIM; d++) s += q_h[d] * k_h[d];
            scores[t] = s * scale;
        }
        float mv = scores[0];
        for (int t = 1; t <= pos; t++) if (scores[t] > mv) mv = scores[t];
        float sum = 0;
        for (int t = 0; t <= pos; t++) { scores[t] = std::exp(scores[t] - mv); sum += scores[t]; }
        float inv_sum = 1.0f / sum;
        for (int t = 0; t <= pos; t++) scores[t] *= inv_sum;

        float *out_h = &out[h * HEAD_DIM];
        std::memset(out_h, 0, HEAD_DIM * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            float *v_h = &v_cache[(t * N_HEADS + h) * HEAD_DIM];
            float w = scores[t];
            for (int d = 0; d < HEAD_DIM; d++) out_h[d] += w * v_h[d];
        }
    }
}

static void block_forward(float *h, int blk, int stream, int pos, int is_prompt,
                          float *k_cache, float *v_cache,
                          float *cos_cache, float *sin_cache,
                          float *fc_w_avg, float *fc_b_avg,
                          float *proj_w_avg, float *proj_b_avg) {
    char s = stream ? 'b' : 'a';
    int cid = (blk * N_STREAMS + stream);
    char key[256];
    float buf[H];

    // LN1
    std::snprintf(key, sizeof(key), "blocks_%d_stream_%c_ln1_weight", blk, s);
    float *ln1_w = get_w(key);
    std::snprintf(key, sizeof(key), "blocks_%d_stream_%c_ln1_bias", blk, s);
    float *ln1_b = get_w(key);
    std::memcpy(buf, h, H * sizeof(float));
    layer_norm(buf, ln1_w, ln1_b, H);

    // QKV
    std::snprintf(key, sizeof(key), "blocks_%d_stream_%c_attn_q_proj_weight", blk, s);
    float *q_w = get_w(key);
    std::snprintf(key, sizeof(key), "blocks_%d_stream_%c_attn_k_proj_weight", blk, s);
    float *k_w = get_w(key);
    std::snprintf(key, sizeof(key), "blocks_%d_stream_%c_attn_v_proj_weight", blk, s);
    float *v_w = get_w(key);
    std::snprintf(key, sizeof(key), "blocks_%d_stream_%c_attn_o_proj_weight", blk, s);
    float *o_w = get_w(key);

    float q_proj[H], k_proj[H], v_in[H];
    matmul_t(q_proj, buf, q_w, 1, H, H);
    matmul_t(k_proj, buf, k_w, 1, H, H);
    matmul_t(v_in, buf, v_w, 1, H, H);

    std::memcpy(&v_cache[(cid * MAX_SEQ + pos) * N_HEADS * HEAD_DIM],
           v_in, N_HEADS * HEAD_DIM * sizeof(float));

    // Attention
    float attn_out[N_HEADS * HEAD_DIM];
    attention(attn_out, q_proj, k_proj,
              &k_cache[cid * MAX_SEQ * N_HEADS * HEAD_DIM],
              &v_cache[cid * MAX_SEQ * N_HEADS * HEAD_DIM],
              pos, cos_cache, sin_cache);

    // O proj + residual
    float o_buf[H];
    matmul_t(o_buf, attn_out, o_w, 1, H, H);
    for (int j = 0; j < H; j++) h[j] += o_buf[j];

    // LN2
    std::snprintf(key, sizeof(key), "blocks_%d_stream_%c_ln2_weight", blk, s);
    float *ln2_w = get_w(key);
    std::snprintf(key, sizeof(key), "blocks_%d_stream_%c_ln2_bias", blk, s);
    float *ln2_b = get_w(key);
    std::memcpy(buf, h, H * sizeof(float));
    layer_norm(buf, ln2_w, ln2_b, H);

    // MoE gate
    std::snprintf(key, sizeof(key), "blocks_%d_stream_%c_moe_gate_weight", blk, s);
    float *gate_w = get_w(key);
    if (!gate_w) return;

    float gs[N_EXP];
    matmul_t(gs, buf, gate_w, 1, N_EXP, H);

    float mv = gs[0]; for (int e = 1; e < N_EXP; e++) if (gs[e] > mv) mv = gs[e];
    float sum = 0; for (int e = 0; e < N_EXP; e++) { gs[e] = std::exp(gs[e]-mv); sum += gs[e]; }
    float inv_sum = 1.0f / sum;
    for (int e = 0; e < N_EXP; e++) gs[e] *= inv_sum;

    int top_i[TOP_K]; float top_v[TOP_K];
    for (int k = 0; k < TOP_K; k++) { top_i[k] = 0; top_v[k] = -1e9f; }
    for (int e = 0; e < N_EXP; e++)
        for (int k = 0; k < TOP_K; k++)
            if (gs[e] > top_v[k]) {
                for (int k2 = TOP_K-1; k2 > k; k2--) { top_i[k2] = top_i[k2-1]; top_v[k2] = top_v[k2-1]; }
                top_i[k] = e; top_v[k] = gs[e]; break;
            }

    float moe_out[H] = {0};
    for (int k = 0; k < TOP_K; k++) {
        int e = top_i[k];
        float w = top_v[k];
        int eb = (blk < 3) ? 3 + (blk % 3) : blk;
        if (eb >= 31) eb = 31;

        std::snprintf(key, sizeof(key), "blocks_%d_experts_%d_c_fc_weight", eb, e);
        float *fc_w_p = get_w(key);
        std::snprintf(key, sizeof(key), "blocks_%d_experts_%d_c_proj_weight", eb, e);
        float *pj_w_p = get_w(key);

        if (fc_w_p && pj_w_p) {
            std::snprintf(key, sizeof(key), "blocks_%d_experts_%d_c_fc_bias", eb, e);
            float *fc_b_p = get_w(key);
            std::snprintf(key, sizeof(key), "blocks_%d_experts_%d_c_proj_bias", eb, e);
            float *pj_b_p = get_w(key);

            float fc[FFN_HIDDEN];
            for (int j = 0; j < FFN_HIDDEN; j++) {
                float s = fc_b_p[j];
                for (int d = 0; d < H; d++) s += buf[d] * fc_w_p[j*H+d];
                fc[j] = gelu(s);
            }
            float proj[H];
            for (int j = 0; j < H; j++) {
                float s = pj_b_p[j];
                for (int d = 0; d < FFN_HIDDEN; d++) s += fc[d] * pj_w_p[j*FFN_HIDDEN+d];
                proj[j] = s;
            }
            for (int j = 0; j < H; j++) moe_out[j] += w * proj[j];
        } else {
            float fc[FFN_HIDDEN];
            for (int j = 0; j < FFN_HIDDEN; j++) {
                float s = fc_b_avg[j];
                for (int d = 0; d < H; d++) s += buf[d] * fc_w_avg[j*H+d];
                fc[j] = gelu(s);
            }
            float proj[H];
            for (int j = 0; j < H; j++) {
                float s = proj_b_avg[j];
                for (int d = 0; d < FFN_HIDDEN; d++) s += fc[d] * proj_w_avg[j*FFN_HIDDEN+d];
                proj[j] = s;
            }
            for (int j = 0; j < H; j++) moe_out[j] += w * proj[j];
        }
    }
    for (int j = 0; j < H; j++) h[j] += moe_out[j];
}

static int vocab[VOCAB];
static char id_to_token[VOCAB][64];
static int vocab_size = 0;

static int load_tokenizer(const char *path) {
    FILE *f = std::fopen(path, "r");
    if (!f) { std::fprintf(stderr, "No vocab at %s\n", path); return 0; }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    char *json = static_cast<char*>(std::malloc(sz + 1));
    size_t got = std::fread(json, 1, sz, f);
    (void)got;
    json[sz] = 0;
    std::fclose(f);

    std::memset(vocab, 0, sizeof(vocab));
    char *p = json;
    vocab_size = 0;
    while (*p && vocab_size < VOCAB) {
        p = std::strchr(p, '"');
        if (!p) break;
        p++;
        char *key_start = p;
        p = std::strchr(p, '"');
        if (!p) break;
        int key_len = p - key_start;
        if (key_len >= 64) key_len = 63;

        p = std::strchr(p, ':');
        if (!p) break;
        p++;
        while (*p == ' ') p++;
        int id = std::atoi(p);
        if (id >= 0 && id < VOCAB) {
            std::memcpy(id_to_token[id], key_start, key_len);
            id_to_token[id][key_len] = 0;
            vocab[id] = 1;
            vocab_size++;
        }
        p = std::strchr(p, ',');
        if (!p) break;
    }
    std::free(json);
    std::printf("Loaded %d vocab entries\n", vocab_size);
    return vocab_size > 0;
}

int main(int argc, char **argv) {
    std::printf("=== Albert-MoE-13 Optimized C++ Inference ===\n");
    if (argc < 2) { std::printf("Usage: %s \"prompt\"\n", argv[0]); return 1; }

    if (!load_all_weights("/tmp/albert_weights")) return 1;
    load_tokenizer("/home/bcloud/models/albert-moe/vocab.json");

    float cos_cache[MAX_SEQ * HEAD_DIM];
    float sin_cache[MAX_SEQ * HEAD_DIM];
    precompute_rope(cos_cache, sin_cache, MAX_SEQ, HEAD_DIM);

    // Tokenize
    int tokens[MAX_SEQ];
    int seq = 0;
    tokens[seq++] = 0;
    int has_vocab = (vocab_size > 0);
    if (has_vocab) {
        // Simple: use char ID
        for (const char *p = argv[1]; *p && seq < MAX_SEQ - 1; p++)
            tokens[seq++] = static_cast<int>(static_cast<unsigned char>(*p));
    } else {
        for (const char *p = argv[1]; *p && seq < MAX_SEQ - 1; p++)
            tokens[seq++] = static_cast<int>(static_cast<unsigned char>(*p));
    }
    std::printf("Prompt: %s (%d tokens)\n", argv[1], seq);

    // KV cache
    float *k_cache = static_cast<float*>(std::calloc(N_BLOCKS * N_STREAMS * MAX_SEQ * N_HEADS * HEAD_DIM, sizeof(float)));
    float *v_cache = static_cast<float*>(std::calloc(N_BLOCKS * N_STREAMS * MAX_SEQ * N_HEADS * HEAD_DIM, sizeof(float)));

    // Pre-compute averaged expert weights
    float *fc_w_avg = static_cast<float*>(std::calloc(H * FFN_HIDDEN, sizeof(float)));
    float *fc_b_avg = static_cast<float*>(std::calloc(FFN_HIDDEN, sizeof(float)));
    float *proj_w_avg = static_cast<float*>(std::calloc(FFN_HIDDEN * H, sizeof(float)));
    float *proj_b_avg = static_cast<float*>(std::calloc(H, sizeof(float)));
    int exp_count = 0;
    for (int eb = 3; eb < N_BLOCKS; eb++) {
        for (int e = 0; e < N_EXP; e++) {
            char key[256];
            std::snprintf(key, sizeof(key), "blocks_%d_experts_%d_c_fc_weight", eb, e);
            float *w = get_w(key); if (!w) continue;
            std::snprintf(key, sizeof(key), "blocks_%d_experts_%d_c_fc_bias", eb, e);
            float *bi = get_w(key);
            std::snprintf(key, sizeof(key), "blocks_%d_experts_%d_c_proj_weight", eb, e);
            float *pw = get_w(key);
            std::snprintf(key, sizeof(key), "blocks_%d_experts_%d_c_proj_bias", eb, e);
            float *pb = get_w(key);
            for (int j = 0; j < H*FFN_HIDDEN; j++) fc_w_avg[j] += w[j];
            for (int j = 0; j < FFN_HIDDEN; j++) fc_b_avg[j] += bi[j];
            for (int j = 0; j < FFN_HIDDEN*H; j++) proj_w_avg[j] += pw[j];
            for (int j = 0; j < H; j++) proj_b_avg[j] += pb[j];
            exp_count++;
        }
    }
    if (exp_count > 0) {
        float iv = 1.0f / exp_count;
        for (int j = 0; j < H*FFN_HIDDEN; j++) fc_w_avg[j] *= iv;
        for (int j = 0; j < FFN_HIDDEN; j++) fc_b_avg[j] *= iv;
        for (int j = 0; j < FFN_HIDDEN*H; j++) proj_w_avg[j] *= iv;
        for (int j = 0; j < H; j++) proj_b_avg[j] *= iv;
    }
    std::printf("Avg expert weights: %d experts\n", exp_count);

    // Load required weights
    float *embed_w = get_w("embed_weight");
    float *ln_f_w = get_w("ln_f_weight");
    float *ln_f_b = get_w("ln_f_bias");
    float *lm_head_w = get_w("lm_head_weight");
    if (!embed_w || !lm_head_w) { std::fprintf(stderr, "Missing embed/lm_head\n"); return 1; }

    float h_buf[H], buf[H];

    // ── First: prompt processing (position by position, KV cache fill) ──
    clock_t t0 = std::clock();
    for (int pos = 0; pos < seq; pos++) {
        int tok = tokens[pos];
        std::memcpy(h_buf, &embed_w[tok * H], H * sizeof(float));

        for (int blk = 0; blk < N_BLOCKS; blk++)
            for (int st = 0; st < N_STREAMS; st++)
                block_forward(h_buf, blk, st, pos, 1,
                              k_cache, v_cache, cos_cache, sin_cache,
                              fc_w_avg, fc_b_avg, proj_w_avg, proj_b_avg);

        if (pos == seq - 1) std::memcpy(buf, h_buf, H * sizeof(float));
    }

    // ── Generation ──
    int gen_tokens = 0;
    int output_tokens[32];
    int n_out = 0;

    for (int gen = 0; gen < 10; gen++) {
        std::memcpy(h_buf, buf, H * sizeof(float));
        layer_norm(h_buf, ln_f_w, ln_f_b, H);

        float logits[VOCAB];
        for (int v = 0; v < VOCAB; v++) {
            float s = 0;
            for (int j = 0; j < H; j++) s += h_buf[j] * lm_head_w[v*H+j];
            logits[v] = s;
        }
        int next = 0;
        for (int v = 1; v < VOCAB; v++) if (logits[v] > logits[next]) next = v;
        output_tokens[n_out++] = next;
        gen_tokens++;
        if (next == 2 || n_out >= 10) break;

        // Next token
        int pos = seq + n_out - 1;
        std::memcpy(h_buf, &embed_w[next * H], H * sizeof(float));
        for (int blk = 0; blk < N_BLOCKS; blk++)
            for (int st = 0; st < N_STREAMS; st++)
                block_forward(h_buf, blk, st, pos, 0,
                              k_cache, v_cache, cos_cache, sin_cache,
                              fc_w_avg, fc_b_avg, proj_w_avg, proj_b_avg);
        std::memcpy(buf, h_buf, H * sizeof(float));
    }

    float ms = static_cast<float>(std::clock() - t0) / CLOCKS_PER_SEC * 1000;
    std::printf("Generated %d tokens in %.0f ms (%.1f tok/s)\n",
           gen_tokens, ms, gen_tokens / (ms/1000.0f));
    std::printf("Tokens:");
    for (int i = 0; i < n_out; i++) std::printf(" %d", output_tokens[i]);
    std::printf("\n");
    if (has_vocab) {
        std::printf("Text:");
        for (int i = 0; i < n_out; i++)
            std::printf("%s", id_to_token[output_tokens[i]]);
        std::printf("\n");
    }

    std::free(k_cache); std::free(v_cache);
    std::free(fc_w_avg); std::free(fc_b_avg); std::free(proj_w_avg); std::free(proj_b_avg);
    std::free(ws.raw); std::free(ws.tensors);
    return 0;
}
