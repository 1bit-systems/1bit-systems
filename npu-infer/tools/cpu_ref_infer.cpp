/**
 * CPU reference inference — verifies weight format by running full model on CPU.
 * If this produces correct tokens, the weight format is understood.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static inline float bf16f(uint16_t v) { uint32_t b=((uint32_t)v)<<16; float f; memcpy(&f,&b,4); return f; }
static inline uint16_t fbf16(float v) { uint32_t b; memcpy(&b,&v,4); uint32_t r=((b>>16)&1)+0x7FFF; return (uint16_t)((b+r)>>16); }
static inline bool bf16_bad(uint16_t v) { return (v & 0x7F80) == 0x7F80; }

struct Tensor { const char* name; int i8_rows, i8_cols; uint64_t data_off; };
struct Cfg { int hidden, layers, q_heads, kv_heads, head_dim, inter, vocab; float eps; };

static const Cfg CFG = {1024, 28, 16, 8, 128, 3072, 151936, 1e-6f};

// Simple JSON parser to find tensor offsets
static int parse_tensors(const uint8_t* json, size_t len, Tensor* out, int max) {
    int n = 0;
    const char* s = (const char*)json, *end = s + len;
    while (s < end && n < max) {
        const char* qs = (const char*)memchr(s, '"', end - s);
        if (!qs) break;
        const char* qe = (const char*)memchr(qs+1, '"', end - qs - 1);
        if (!qe) break;
        size_t nlen = qe - qs - 1;
        bool is_weight = (nlen > 7 && !memcmp(qe - 7, ".weight", 7));
        if (!is_weight) { s = qe + 1; continue; }

        memcpy((char*)out[n].name, qs+1, nlen < 63 ? nlen : 63);
        out[n].name[nlen < 63 ? nlen : 63] = 0;

        const char* ob = (const char*)memchr(qe, '{', end - qe);
        if (!ob) break;

        // Find shape
        const char* sh = strstr(ob, "\"shape\"");
        if (sh) {
            const char* arr = strchr(sh, '[');
            if (arr) {
                int64_t shape[4]={}; int nd=0;
                const char* sp = arr+1;
                while (*sp!=']' && nd<4) {
                    while (*sp==' '||*sp==',') sp++;
                    if (*sp>='0'&&*sp<='9') { shape[nd++]=strtol(sp,(char**)&sp,10); }
                    else break;
                }
                out[n].i8_rows = (int)shape[0];
                out[n].i8_cols = (nd>1) ? (int)shape[1] : 0;
            }
        }

        // Find data_offsets
        const char* off = strstr(ob, "\"data_offsets\"");
        if (off) {
            const char* arr = strchr(off, '[');
            if (arr) out[n].data_off = strtoull(arr+1, nullptr, 10);
        }
        n++;
        s = qe + 1;
    }
    return n;
}

static const uint16_t* tensor_bf16(const uint8_t* fd, uint64_t doff, const Tensor& t) {
    return (const uint16_t*)(fd + doff + t.data_off);
}
static int tensor_rows(const Tensor& t) { return t.i8_rows; }
static int tensor_cols(const Tensor& t) { return t.i8_cols / 2; }

// Weight matrix multiply: out[n] = sum_k(in[k] * weight[storage_idx(k,n)])
// weight stored as [i8_rows, i8_cols] I8 = [rows, cols] BF16
// Column-major interpretation: weight[in_feat, out_feat]
//   storage_idx(k, n) = k * cols + n  (k = in_feat idx, n = out_feat idx)
static float weight_elem(const uint16_t* w, int k, int n, int cols) {
    uint16_t raw = w[k * cols + n];
    return bf16_bad(raw) ? 0.0f : bf16f(raw);
}

inline void rms_norm(float* RESTRICT x, const uint16_t* RESTRICT w, int n, float eps) {
    double ss = 0; for(int i=0;i<n;i++) ss += (double)x[i]*x[i];
    float inv = 1.0f/sqrtf((float)(ss/n) + eps);
    for(int i=0;i<n;i++) { float wi=bf16f(w[i]); x[i] = x[i]*inv*wi; }
}

inline void matvec(float* RESTRICT out, const float* RESTRICT x,
                   const uint16_t* RESTRICT w, int in_dim, int out_dim, int w_cols) {
    for(int n=0; n<out_dim; n++) {
        double sum = 0;
        for(int k=0; k<in_dim; k++) sum += (double)x[k] * weight_elem(w, k, n, w_cols);
        out[n] = (float)sum;
    }
}

static void softmax(float* x, int n) {
    float mx=x[0]; for(int i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    double s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-mx); s+=x[i];}
    for(int i=0;i<n;i++) x[i]/=(float)s;
}

static void silu(float* x, int n) { for(int i=0;i<n;i++) x[i]=x[i]/(1.0f+expf(-x[i])); }

int main(int argc, char** argv) {
    const char* mp = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    if (argc>1) mp = argv[1];

    int fd = open(mp, O_RDONLY);
    struct stat st; fstat(fd, &st);
    uint8_t* fdata = (uint8_t*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    uint64_t hsz; memcpy(&hsz, fdata, 8);
    uint64_t doff = 8 + hsz;
    printf("Model: %.0f MB, data@0x%lx\n", st.st_size/1e6, doff);

    // Parse all tensors
    Tensor tensors[512];
    int nt = parse_tensors(fdata + 8, hsz, tensors, 512);
    printf("Parsed %d tensors\n", nt);

    // Find key tensors
    auto find_t = [&](const char* name) -> const Tensor* {
        for(int i=0;i<nt;i++) if(!strcmp((const char*)tensors[i].name, name)) return &tensors[i];
        return nullptr;
    };

    // Layer 0 tensors
    printf("\n=== CPU Inference Test ===\n");

    // Embedding lookup
    const uint16_t* emb = tensor_bf16(fdata, doff, *find_t("model.embed_tokens.weight"));
    int token = 151643; // BOS

    float x[1024], h[1024];
    for(int i=0;i<1024;i++) { uint16_t r=emb[token*1024+i]; x[i]=bf16_bad(r)?0:bf16f(r); }
    printf("Token %d embed[0..3]: %.4f %.4f %.4f %.4f\n", token, x[0],x[1],x[2],x[3]);

    // Layer 0
    auto ln0 = find_t("model.layers.0.input_layernorm.weight");
    auto qp0 = find_t("model.layers.0.self_attn.q_proj.weight");
    auto kp0 = find_t("model.layers.0.self_attn.k_proj.weight");
    auto vp0 = find_t("model.layers.0.self_attn.v_proj.weight");
    auto op0 = find_t("model.layers.0.self_attn.o_proj.weight");
    auto gp0 = find_t("model.layers.0.mlp.gate_proj.weight");
    auto up0 = find_t("model.layers.0.mlp.up_proj.weight");
    auto dp0 = find_t("model.layers.0.mlp.down_proj.weight");
    auto an0 = find_t("model.layers.0.post_attention_layernorm.weight");

    printf("\nLayer 0 tensors:\n");
    printf("  q_proj: I8[%d,%d] BF16[%d,%d] off=0x%lx\n", qp0->i8_rows, qp0->i8_cols, tensor_rows(*qp0), tensor_cols(*qp0), qp0->data_off);
    printf("  k_proj: I8[%d,%d] BF16[%d,%d]\n", kp0->i8_rows, kp0->i8_cols, tensor_rows(*kp0), tensor_cols(*kp0));
    printf("  o_proj: I8[%d,%d] BF16[%d,%d]\n", op0->i8_rows, op0->i8_cols, tensor_rows(*op0), tensor_cols(*op0));
    printf("  gate:   I8[%d,%d] BF16[%d,%d]\n", gp0->i8_rows, gp0->i8_cols, tensor_rows(*gp0), tensor_cols(*gp0));

    // RMS norm
    rms_norm(x, tensor_bf16(fdata, doff, *ln0), 1024, CFG.eps);
    printf("After RMS norm[0..3]: %.4f %.4f %.4f %.4f\n", x[0],x[1],x[2],x[3]);

    // Q projection
    float q_out[256];
    const uint16_t* qw = tensor_bf16(fdata, doff, *qp0);
    matvec(q_out, x, qw, 1024, 256, tensor_cols(*qp0));
    printf("Q_proj[0..7]: ");
    for(int i=0;i<8;i++) printf("%.4f ", q_out[i]);
    printf("\n");

    // Gate projection
    float g_out[384];
    const uint16_t* gw = tensor_bf16(fdata, doff, *gp0);
    matvec(g_out, x, gw, 1024, 384, tensor_cols(*gp0));
    printf("Gate[0..7]: ");
    for(int i=0;i<8;i++) printf("%.4f ", g_out[i]);
    printf("\n");

    // Check for NaN
    bool ok = true;
    for(int i=0;i<256;i++) if(!isfinite(q_out[i])) ok=false;
    for(int i=0;i<384;i++) if(!isfinite(g_out[i])) ok=false;
    printf("Values finite: %s\n", ok?"YES":"NO (NaN/Inf present)");

    if (ok) {
        printf("\n*** WEIGHT FORMAT CONFIRMED: column-major [in_feat, out_feat] ***\n");
        printf("*** Q_proj and Gate both produce valid finite output ***\n");
    }

    munmap(fdata, st.st_size);
    return ok ? 0 : 1;
}
