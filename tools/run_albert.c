// run_albert.c — Pure C Albert-MoE-13 inference (CPU, libc only)
// Build: gcc -O3 run_albert.c -lm -o run_albert && ./run_albert "prompt"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define W(name) ({ float *_w = get_w(m, name); if (!_w) { fprintf(stderr,"WARN: missing %s\n",name); static float _z[H]; memset(_z,0,sizeof(_z)); _w = _z; } _w; })

#define H 256
#define N_HEADS 4
#define HEAD_DIM 64
#define N_EXP 12
#define TOP_K 3
#define VOCAB 32000
#define FFN_HIDDEN 1024
#define MAX_SEQ 256

// ── Load model weights via binary index ──
typedef struct { char key[128]; float *data; int64_t numel; } Tensor;
typedef struct { Tensor *t; int count; float *raw; size_t raw_size; } Model;

static Model load_model(const char *sf_path, const char *idx_path) {
    Model m = {0};
    // Read index
    FILE *fi = fopen(idx_path, "rb"); if (!fi) { perror(idx_path); return m; }
    int count; fread(&count, 4, 1, fi);
    m.t = calloc(count, sizeof(Tensor));
    m.count = count;
    
    uint8_t *name_buf = NULL; size_t name_cap = 0;
    int64_t off, nel;
    for (int i = 0; i < count; i++) {
        int name_len; if (!fread(&name_len, 4, 1, fi)) break;
        if (name_len > (int)name_cap) { name_cap = name_len + 1; name_buf = realloc(name_buf, name_cap); }
        if (!fread(name_buf, 1, name_len, fi)) break; name_buf[name_len] = 0;
        strcpy(m.t[i].key, (char*)name_buf);
        if (!fread(&off, 8, 1, fi)) break;  // offset
        if (!fread(&nel, 8, 1, fi)) break;  // numel
        m.t[i].numel = nel;
        m.t[i].data = NULL; // will be set in second pass
    }
    free(name_buf); fclose(fi);
    printf("Read %d index entries\n", count);
    
    // Read safetensors data
    FILE *fs = fopen(sf_path, "rb"); if (!fs) { perror(sf_path); return m; }
    fseek(fs, 0, SEEK_END); m.raw_size = ftell(fs); fseek(fs, 0, SEEK_SET);
    m.raw = malloc(m.raw_size); fread(m.raw, 1, m.raw_size, fs); fclose(fs);
    
    // Map data pointers — re-read index for proper offsets
    fi = fopen(idx_path, "rb"); fread(&count, 4, 1, fi);
    for (int i = 0; i < count && i < m.count; i++) {
        int name_len; fread(&name_len, 4, 1, fi);
        fseek(fi, name_len, SEEK_CUR);
        int64_t offset, numel;
        fread(&offset, 8, 1, fi);
        fread(&numel, 8, 1, fi);
        if (i < m.count) {
            m.t[i].data = (float*)(m.raw + offset);
            m.t[i].numel = numel;
        }
    }
    fclose(fi);
    printf("Model: %d tensors, %.1f MB\n", count, m.raw_size/1e6);
    return m;
}

static float *get_w(Model *m, const char *name) {
    for (int i = 0; i < m->count; i++)
        if (!strcmp(m->t[i].key, name)) return m->t[i].data;
    fprintf(stderr, "Missing: %s\n", name); return NULL;
}

// ── Math ──
static float gelu(float x) {
    float t = tanhf(0.79788456f * (x + 0.044715f * x * x * x));
    return 0.5f * x * (1.0f + t);
}

static void layer_norm(float *x, float *w, float *b, int n) {
    float mean = 0, var = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var = sqrtf(var / n + 1e-5f);
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

// ═══ Forward ───
static void forward(float *x, int seq, Model *m, int *tokens) {
    float buf[MAX_SEQ * H]; // temp buffer
    
    // Embed
    printf("DEBUG: m->count=%d raw=%p raw_size=%zu\n", m->count, (void*)m.raw, m.raw_size); fflush(stdout);
    printf("DEBUG: first key='%s'\n", m->t[0].key); fflush(stdout);
    printf("DEBUG: first data ptr=%p\n", (void*)m->t[0].data); fflush(stdout);
    float *embed_w = get_w(m, "embed.weight");
    printf("DEBUG: embed_w=%p\n", (void*)embed_w); fflush(stdout);
    if (!embed_w) { fprintf(stderr,"Missing embed.weight\n"); exit(1); }
    ptrdiff_t offset = (char*)embed_w - (char*)m.raw;
    printf("DEBUG: offset=%td numel=%ld\n", offset, (long)(m->t[0].numel)); fflush(stdout);
    if (offset < 0 || offset + H*4 > (ptrdiff_t)m.raw_size) {
        fprintf(stderr,"ERROR: embed_w out of bounds!\n"); exit(1);
    }
    printf("DEBUG: embed_w[0]=%.4f\n", embed_w[0]); fflush(stdout);
    for (int i = 0; i < seq; i++) {
        int tok = tokens[i]; if (tok < 0 || tok >= 32000) tok = 0;
        memcpy(&x[i*H], &embed_w[tok*H], H*sizeof(float));
    }
    
    // Run all blocks (both streams)
    for (int blk = 0; blk < 33; blk++) {
        char key[256];
        
        for (int stream = 0; stream < 2; stream++) {
            char s = stream ? 'b' : 'a';
            
            // ── Attention sublayer ──
            snprintf(key, sizeof(key), "blocks.%d.stream_%c.ln1.weight", blk, s);
            float *ln1_w = get_w(m, key);
            snprintf(key, sizeof(key), "blocks.%d.stream_%c.ln1.bias", blk, s);
            float *ln1_b = get_w(m, key);
            snprintf(key, sizeof(key), "blocks.%d.stream_%c.attn.q_proj.weight", blk, s);
            float *q_w = get_w(m, key);
            snprintf(key, sizeof(key), "blocks.%d.stream_%c.attn.k_proj.weight", blk, s);
            float *k_w = get_w(m, key);
            snprintf(key, sizeof(key), "blocks.%d.stream_%c.attn.v_proj.weight", blk, s);
            float *v_w = get_w(m, key);
            snprintf(key, sizeof(key), "blocks.%d.stream_%c.attn.o_proj.weight", blk, s);
            float *o_w = get_w(m, key);
            
            memcpy(buf, x, seq*H*sizeof(float));
            for (int i = 0; i < seq; i++) layer_norm(&buf[i*H], ln1_w, ln1_b, H);
            
            // Standard attention
            float q[seq*H], k[seq*H], v[seq*H], attn[seq*seq];
            matmul_t(q, buf, q_w, seq, H, H);
            matmul_t(k, buf, k_w, seq, H, H);
            matmul_t(v, buf, v_w, seq, H, H);
            
            float scale = 1.0f / sqrtf(HEAD_DIM);
            for (int h = 0; h < N_HEADS; h++) {
                for (int i = 0; i < seq; i++)
                    for (int j = 0; j < seq; j++) {
                        float s = 0;
                        for (int d = 0; d < HEAD_DIM; d++)
                            s += q[i*H+h*HEAD_DIM+d] * k[j*H+h*HEAD_DIM+d];
                        attn[i*seq+j] = s * scale;
                    }
                for (int i = 0; i < seq; i++) {
                    float mv = attn[i*seq]; for (int j = 1; j < seq; j++) if (attn[i*seq+j] > mv) mv = attn[i*seq+j];
                    float sum = 0; for (int j = 0; j < seq; j++) { attn[i*seq+j] = expf(attn[i*seq+j] - mv); sum += attn[i*seq+j]; }
                    for (int j = 0; j < seq; j++) attn[i*seq+j] /= sum;
                }
                for (int i = 0; i < seq; i++) {
                    float out_h[HEAD_DIM] = {0};
                    for (int j = 0; j < seq; j++)
                        for (int d = 0; d < HEAD_DIM; d++)
                            out_h[d] += attn[i*seq+j] * v[j*H+h*HEAD_DIM+d];
                    for (int d = 0; d < HEAD_DIM; d++) buf[i*H+h*HEAD_DIM+d] = out_h[d];
                }
            }
            // Output projection + residual
            float tmp[seq*H];
            matmul_t(tmp, buf, o_w, seq, H, H);
            for (int i = 0; i < seq*H; i++) x[i] += tmp[i];
            
            // ── MoE sublayer ──
            snprintf(key, sizeof(key), "blocks.%d.stream_%c.ln2.weight", blk, s);
            float *ln2_w = get_w(m, key);
            snprintf(key, sizeof(key), "blocks.%d.stream_%c.ln2.bias", blk, s);
            float *ln2_b = get_w(m, key);
            snprintf(key, sizeof(key), "blocks.%d.stream_%c.moe.gate.weight", blk, s);
            float *gate_w = get_w(m, key);
            
            memcpy(buf, x, seq*H*sizeof(float));
            for (int i = 0; i < seq; i++) layer_norm(&buf[i*H], ln2_w, ln2_b, H);
            
            // Router
            float gs[seq*N_EXP];
            matmul_t(gs, buf, gate_w, seq, N_EXP, H);
            
            // Expert FFN (use averaged expert weights)
            float fc_w[H*FFN_HIDDEN], fc_b[FFN_HIDDEN], proj_w[FFN_HIDDEN*H], proj_b[H];
            int exp_count = 0;
            memset(fc_w, 0, sizeof(fc_w)); memset(fc_b, 0, sizeof(fc_b));
            memset(proj_w, 0, sizeof(proj_w)); memset(proj_b, 0, sizeof(proj_b));
            
            for (int eb = 3; eb < 33; eb++) {
                for (int e = 0; e < N_EXP; e++) {
                    snprintf(key, sizeof(key), "blocks.%d.experts.%d.c_fc.weight", eb, e);
                    float *w = get_w(m, key); if (!w) continue;
                    snprintf(key, sizeof(key), "blocks.%d.experts.%d.c_fc.bias", eb, e);
                    float *bi = get_w(m, key);
                    snprintf(key, sizeof(key), "blocks.%d.experts.%d.c_proj.weight", eb, e);
                    float *pw = get_w(m, key);
                    snprintf(key, sizeof(key), "blocks.%d.experts.%d.c_proj.bias", eb, e);
                    float *pb = get_w(m, key);
                    
                    for (int j = 0; j < H*FFN_HIDDEN; j++) fc_w[j] += w[j];
                    for (int j = 0; j < FFN_HIDDEN; j++) fc_b[j] += bi[j];
                    for (int j = 0; j < FFN_HIDDEN*H; j++) proj_w[j] += pw[j];
                    for (int j = 0; j < H; j++) proj_b[j] += pb[j];
                    exp_count++;
                }
            }
            if (exp_count > 0) {
                float iv = 1.0f/exp_count;
                for (int j = 0; j < H*FFN_HIDDEN; j++) fc_w[j] *= iv;
                for (int j = 0; j < FFN_HIDDEN; j++) fc_b[j] *= iv;
                for (int j = 0; j < FFN_HIDDEN*H; j++) proj_w[j] *= iv;
                for (int j = 0; j < H; j++) proj_b[j] *= iv;
            }
            
            for (int i = 0; i < seq; i++) {
                // Softmax gate
                float mv = gs[i*N_EXP]; for (int e = 1; e < N_EXP; e++) if (gs[i*N_EXP+e] > mv) mv = gs[i*N_EXP+e];
                float sum = 0; for (int e = 0; e < N_EXP; e++) { gs[i*N_EXP+e] = expf(gs[i*N_EXP+e]-mv); sum += gs[i*N_EXP+e]; }
                for (int e = 0; e < N_EXP; e++) gs[i*N_EXP+e] /= sum;
                
                // Top-3
                int top_i[TOP_K]; float top_v[TOP_K];
                for (int k = 0; k < TOP_K; k++) { top_i[k] = 0; top_v[k] = -1e9f; }
                for (int e = 0; e < N_EXP; e++) {
                    for (int k = 0; k < TOP_K; k++) {
                        if (gs[i*N_EXP+e] > top_v[k]) {
                            for (int k2 = TOP_K-1; k2 > k; k2--) { top_i[k2] = top_i[k2-1]; top_v[k2] = top_v[k2-1]; }
                            top_i[k] = e; top_v[k] = gs[i*N_EXP+e]; break;
                        }
                    }
                }
                
                float moe_out[H] = {0};
                for (int k = 0; k < TOP_K; k++) {
                    float fc[FFN_HIDDEN];
                    for (int j = 0; j < FFN_HIDDEN; j++) {
                        float s = fc_b[j]; for (int d = 0; d < H; d++) s += buf[i*H+d] * fc_w[j*H+d];
                        fc[j] = gelu(s);
                    }
                    float proj[H];
                    for (int j = 0; j < H; j++) {
                        float s = proj_b[j]; for (int d = 0; d < FFN_HIDDEN; d++) s += fc[d] * proj_w[j*FFN_HIDDEN+d];
                        proj[j] = s;
                    }
                    for (int j = 0; j < H; j++) moe_out[j] += top_v[k] * proj[j];
                }
                for (int j = 0; j < H; j++) x[i*H+j] += moe_out[j];
            }
        }
    }
    
    // Final LN + lm_head
    float *ln_f_w = get_w(m, "ln_f.weight");
    float *ln_f_b = get_w(m, "ln_f.bias");
    float *lm_head_w = get_w(m, "lm_head.weight");
    
    for (int i = 0; i < seq; i++) layer_norm(&x[i*H], ln_f_w, ln_f_b, H);
    
    // Sample last token
    float logits[VOCAB];
    for (int v = 0; v < VOCAB; v++) {
        float s = 0;
        for (int j = 0; j < H; j++) s += x[(seq-1)*H+j] * lm_head_w[v*H+j];
        logits[v] = s;
    }
    
    int next = 0; for (int v = 1; v < VOCAB; v++) if (logits[v] > logits[next]) next = v;
    printf("  next token: %d\n", next);
}

int main(int argc, char **argv) {
    printf("Albert-MoE-13 C Inference\n");
    Model m = load_model(
        "/home/bcloud/models/albert-moe/albert_v3.0.best.safetensors",
        "/tmp/albert_idx.bin"
    );
    if (!m.count) return 1;
    
    int tokens[MAX_SEQ] = {1, 100, 101}; // BOS + dummy
    int seq = 3;
    
    clock_t t0 = clock();
    float x[MAX_SEQ * H];
    forward(x, seq, &m, tokens);
    float ms = (float)(clock() - t0) / CLOCKS_PER_SEC * 1000;
    printf("Forward: %.0fms\n", ms);
    
    free(m.raw); free(m.t);
    return 0;
}
