/**
 * NPU Engine v6 — 5-Variant Fused Backend.
 * Uses dimension-optimized M=128 xclbins for each GEMM in Qwen3-0.6B.
 * Variants: 1024×1024, 1024×2048, 1024×3072, 2048×1024, 3072×1024
 *
 * Full: dequant→BFP16 pack→NPU GEMM→Q/K norms+RoPE→KV cache attn→MLP→token gen.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include "helper.h"
#include "gemm_atb_layout.h"
extern "C" float* dequant_i8_to_float(const uint8_t*,int,int*,int*);
extern "C" float* dequant_i8_to_float_ex(const uint8_t*,int,int,int*,int*);
extern "C" float bf16_to_float(uint16_t);
extern "C" uint16_t float_to_bf16(float);

// ---------------------------------------------------------------------------
// xclbin variant table
// ---------------------------------------------------------------------------
// For each (M_xclbin, K_xclbin, N_xclbin) pair we store a dedicated xclbin
// + instruction file.  M_xclbin is always 128 (padded from actual M=1 decode).
// K_xclbin and N_xclbin are chosen per weight dimension to minimise waste.
// ---------------------------------------------------------------------------
struct XclbinVariant {
    const char *xclbin;
    const char *insts;
    int XK;   // K (input-feature) dimension of the xclbin
    int XN;   // N (output-feature) dimension of the xclbin
};

// The table is keyed by (K_target, N_target) — i.e. the logical matvec dims.
// We embed the 128x...x... xclbin paths directly.
static const XclbinVariant VARIANTS[] = {
    //   K,N    xclbin suffix          inst suffix
    {nullptr, nullptr, 1024, 1024},   // index 0: K,V
    {nullptr, nullptr, 1024, 2048},   // index 1: Q
    {nullptr, nullptr, 1024, 3072},   // index 2: gate,up
    {nullptr, nullptr, 2048, 1024},   // index 3: O
    {nullptr, nullptr, 3072, 1024},   // index 4: down
};

static const int NVARIANTS = 5;
static const int M_PAD = 128;   // padded activation rows
static const int MT = 128;      // m_tile for MLIR
static const int KT = 64;       // k_tile
static const int NT = 128;      // n_tile

static const char *XCLBIN_BASE =
    "/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build";
static const char *INST_BASE  =
    "/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static float bf16f(uint16_t v) {
    uint32_t b = v << 16; float f; memcpy(&f, &b, 4); return f;
}
static float bf16_g(uint16_t v) {
    return ((v & 0x7F80) == 0x7F80) ? 0.0f : bf16f(v);
}
static uint16_t fbf16(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    uint32_t r = ((b >> 16) & 1) + 0x7FFF;
    return (uint16_t)((b + r) >> 16);
}

static std::vector<float> rope_cos, rope_sin;
static void rope_init(int hd, float theta, int max_pos) {
    rope_cos.resize(max_pos * hd); rope_sin.resize(max_pos * hd);
    for (int p = 0; p < max_pos; p++)
        for (int d = 0; d < hd; d += 2) {
            float f = 1.0f / powf(theta, (float)d / hd), a = p * f;
            rope_cos[p * hd + d] = cosf(a); rope_sin[p * hd + d] = sinf(a);
            rope_cos[p * hd + d + 1] = cosf(a); rope_sin[p * hd + d + 1] = sinf(a);
        }
}
static void rope_apply(float *x, int hd, int pos) {
    for (int d = 0; d < hd; d += 2) {
        float a = x[d], b = x[d + 1], c = rope_cos[pos * hd + d], s = rope_sin[pos * hd + d];
        x[d] = a * c - b * s; x[d + 1] = b * c + a * s;
    }
}

// ---------------------------------------------------------------------------
// pre_pack_variant — pack weight matrix for a specific (K,N) xclbin dimension
// ---------------------------------------------------------------------------
static std::vector<uint8_t> pre_pack_variant(const float *w,
                                              int out_feat,   // logical output dim
                                              int in_feat,    // logical input dim
                                              int deq_rows,   // rows from dequant
                                              int XK, int XN) // xclbin dims
{
    std::vector<float> Bf(XK * XN);  // transposed weight (K×N padded to XK×XN)
    bool trans = (deq_rows == in_feat && in_feat != out_feat);
    for (int k = 0; k < in_feat && k < XK; k++)
        for (int n = 0; n < out_feat && n < XN; n++) {
            float v = trans ? w[k * out_feat + n] : w[n * in_feat + k];
            if (std::isfinite(v)) Bf[k * XN + n] = v;  // no extra scaling
        }
    auto Bs = gemm_atb::layout_transpose_L1_1x2_8x8block(Bf, XK, XN, KT, NT);
    auto Be = floatToBfp16(8, XK * XN, Bs.data(), 0);
    return {Be.data(), Be.data() + Be.size()};
}

// ---------------------------------------------------------------------------
// NpuMultiVariant — manages 5 kernels with dimension-optimised xclbins
// ---------------------------------------------------------------------------
struct NpuMultiVariant {
    xrt::device dev{0};
    // Per-variant resources
    struct Var {
        std::unique_ptr<xrt::xclbin> xc;
        std::unique_ptr<xrt::hw_context> hc;
        std::unique_ptr<xrt::kernel> k;
        std::vector<uint32_t> ins;
        std::unique_ptr<xrt::bo> bI;   // instructions  (CACHEABLE, grp 1)
        std::unique_ptr<xrt::bo> bA;   // activations   (HOST_ONLY, grp 3) — M_PAD × XK
        std::unique_ptr<xrt::bo> bB;   // weights       (HOST_ONLY, grp 4) — packed B
        std::unique_ptr<xrt::bo> bC;   // output        (HOST_ONLY, grp 5) — M_PAD × XN
        int XK, XN;   // xclbin dims this variant uses
    } v[NVARIANTS];

    bool init() {
        dev = xrt::device(0);
        for (int i = 0; i < NVARIANTS; i++) {
            int XK = VARIANTS[i].XK, XN = VARIANTS[i].XN;
            char path[256], inspath[256], suffix[64];
            snprintf(suffix, sizeof(suffix), "128x%dx%d_128x64x128", XK, XN);
            snprintf(path, sizeof(path), "%s/final_%s.xclbin", XCLBIN_BASE, suffix);
            snprintf(inspath, sizeof(inspath), "%s/insts_%s.txt", INST_BASE, suffix);

            // Load instructions
            FILE *f = fopen(inspath, "rb");
            if (!f) { printf("ERR: cannot open %s\n", inspath); return false; }
            fseek(f, 0, 2); long sz = ftell(f); fseek(f, 0, 0);
            v[i].ins.resize(sz / 4);
            size_t nr = fread(v[i].ins.data(), 4, v[i].ins.size(), f); (void)nr;
            fclose(f);

            // Load xclbin
            v[i].xc = std::make_unique<xrt::xclbin>(std::string(path));
            auto &xc = *v[i].xc;
            auto uuid = xc.get_uuid();
            dev.register_xclbin(xc);
            v[i].hc = std::make_unique<xrt::hw_context>(dev, uuid);
            v[i].k = std::make_unique<xrt::kernel>(*v[i].hc, "MLIR_AIE");

            // Allocate BOs
            v[i].bI = std::make_unique<xrt::bo>(dev, (size_t)v[i].ins.size() * 4,
                                                 XCL_BO_FLAGS_CACHEABLE, v[i].k->group_id(1));
            v[i].bA = std::make_unique<xrt::bo>(dev, (size_t)M_PAD * XK * 2,
                                                 XRT_BO_FLAGS_HOST_ONLY, v[i].k->group_id(3));
            v[i].bB = std::make_unique<xrt::bo>(dev, (size_t)XK * XN * 9 / 8,
                                                 XRT_BO_FLAGS_HOST_ONLY, v[i].k->group_id(4));
            v[i].bC = std::make_unique<xrt::bo>(dev, (size_t)M_PAD * XN * 2,
                                                 XRT_BO_FLAGS_HOST_ONLY, v[i].k->group_id(5));
            // Upload instructions
            memcpy(v[i].bI->map(), v[i].ins.data(), v[i].ins.size() * 4);
            v[i].bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);

            v[i].XK = XK; v[i].XN = XN;
            printf("  Variant %d: %dx%d (%s)\n", i, XK, XN, path);
        }
        return true;
    }

    // Lookup variant index by (K, N) logical matvec dimensions
    int lookup(int K, int N) const {
        for (int i = 0; i < NVARIANTS; i++)
            if (v[i].XK == K && v[i].XN == N) return i;
        printf("WARN: no variant for K=%d N=%d, using closest\n", K, N);
        return 0;
    }

    // Run one GEMM using the variant that matches (K, N).
    // A: input activations (M x K logical, M=1 for decode)
    // pB: pre-packed BFP16 weight blob
    // psz: size of pB in bytes (must match variant)
    // out_feat: logical output dimension for result
    // C: output buffer (M x out_feat)
    void gemm(int vi, const float *A, int M, int K,
              const uint8_t *pB, size_t psz,
              int out_feat, float *C)
    {
        auto &ctx = v[vi];
        int XK = ctx.XK, XN = ctx.XN;

        // Pad activations to M_PAD × XK (M=1 → M_PAD=128)
        std::vector<float> Af(M_PAD * XK);
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++)
                Af[m * XK + k] = A[m * K + k];

        // Shuffle A (same layout function as before)
        auto As = gemm_atb::layout_A_L1_2x1_8x8block(Af, M_PAD, XK, MT, KT);
        uint16_t *Ab = (uint16_t *)ctx.bA->map();
        for (int i = 0; i < M_PAD * XK; i++) Ab[i] = fbf16(As[i]);
        ctx.bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Upload weights
        memcpy(ctx.bB->map(), pB, psz);
        ctx.bB->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Clear output
        memset(ctx.bC->map(), 0, (size_t)M_PAD * XN * 2);
        ctx.bC->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Launch
        auto run = (*ctx.k)((unsigned)3, *ctx.bI, (unsigned)ctx.ins.size(),
                            *ctx.bA, *ctx.bB, *ctx.bC);
        run.wait();
        ctx.bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        // C is already row-major from the NPU (no unshuffle needed for M=128 single-row design)
        uint16_t *Cb = (uint16_t *)ctx.bC->map();

        // Extract logical result (first M rows, first out_feat columns)
        for (int m = 0; m < M; m++)
            for (int n = 0; n < out_feat; n++)
                C[m * out_feat + n] = bf16f(Cb[m * XN + n]);
    }

    // Convenience: lookup + gemm in one call
    void gemm_by_dims(const float *A, int M, int K,
                      const uint8_t *pB, size_t psz,
                      int out_feat, float *C)
    {
        int vi = lookup(K, out_feat);
        gemm(vi, A, M, K, pB, psz, out_feat, C);
    }
};

// ---------------------------------------------------------------------------
// Model structure
// ---------------------------------------------------------------------------
struct LO {
    uint64_t qp, kp, vp, op, gp, up, dp, in_off, pa_off, qn_off, kn_off;
};
struct PW {
    std::vector<uint8_t> q, k, v, o, g, u, d;
};
struct KV {
    std::vector<float> k, v;
    int n;
    KV() : k(64 * 8 * 128), v(64 * 8 * 128), n(0) {}
};

static uint64_t joff(const char *js, size_t jl, const char *nm) {
    size_t nl = strlen(nm);
    const char *end = js + jl;
    while (js < end) {
        auto q = (const char *)memmem(js, end - js, nm, nl);
        if (!q) return 0;
        if (q > js && *(q - 1) == '"' && *(q + nl) == '"') {
            auto oo = strstr(q, "\"data_offsets\"");
            if (oo) { auto ar = strchr(oo, '['); if (ar) return strtoull(ar + 1, NULL, 10); }
        }
        js = q + 1;
    }
    return 0;
}

static void rms_norm(float *__restrict x, const uint16_t *__restrict w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float ir = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) x[i] = x[i] * ir * bf16_g(w[i]);
}

// ---------------------------------------------------------------------------
// forward_layer — uses gemm_by_dims which auto-selects the right variant
// ---------------------------------------------------------------------------
static void forward_layer(NpuMultiVariant &npu, int l, int seq_pos,
                          const LO &lo, const PW &pw,
                          float *h, float *qo, float *ko, float *vo,
                          float *atto, float *oo, float *g, float *u,
                          float *su, float *dow, KV &kvc,
                          float eps, int H, int NH, int NKV, int HD, int IM,
                          int GQA, const uint8_t *d, uint64_t df)
{
    float save[H];
    memcpy(save, h, H * sizeof(float));

    // Input RMS norm
    rms_norm(h, (const uint16_t *)(d + df + lo.in_off), H, eps);
    if (l == 19 && seq_pos < 2) {
      printf("  L%d after in_norm h[0..4]=", l);
      for (int i = 0; i < 5; i++) printf("%.1f ", h[i]);
      printf(" save[0..4]="); for (int i = 0; i < 5; i++) printf("%.1f ", save[i]);
      printf("\n");
    }

    // Q, K, V projections — each auto-selects the right xclbin variant
    npu.gemm_by_dims(h, 1, H, pw.q.data(), pw.q.size(), 2048, qo);
    if (l == 0 && seq_pos < 2) { printf("  Q[0..4]: "); for (int i = 0; i < 5; i++) printf("%.1f ", qo[i]); printf("\n"); }
    // Check for NaN in input to each gemm
    bool hn2 = false; for (int i = 0; i < H; i++) if (std::isnan(h[i])) hn2 = true;
    if (hn2) printf("  L%d: NAN in h BEFORE Q!\n", l);
    npu.gemm_by_dims(h, 1, H, pw.k.data(), pw.k.size(), 1024, ko);
    if (l == 0 && seq_pos < 2) { printf("  K[0..4]: "); for (int i = 0; i < 5; i++) printf("%.1f ", ko[i]); printf("\n"); }
    npu.gemm_by_dims(h, 1, H, pw.v.data(), pw.v.size(), 1024, vo);
    if (l == 0 && seq_pos < 2) { printf("  V[0..4]: "); for (int i = 0; i < 5; i++) printf("%.1f ", vo[i]); printf("\n"); }

    // Q/K per-head norms + RoPE
    auto qnw = (const uint16_t *)(d + df + lo.qn_off);
    auto knw = (const uint16_t *)(d + df + lo.kn_off);
    for (int hh = 0; hh < NH; hh++) {
        // Q norm
        double sq = 0;
        for (int d = 0; d < HD; d++) sq += (double)qo[hh * HD + d] * qo[hh * HD + d];
        float iq = 1.0f / sqrtf((float)(sq / HD) + eps);
        for (int d = 0; d < HD; d++) qo[hh * HD + d] *= iq * bf16_g(qnw[d]);
        rope_apply(&qo[hh * HD], HD, seq_pos);

        // K norm (shared per GQA group)
        if (hh % GQA == 0) {
            int kv = hh / GQA;
            double sk = 0;
            for (int d = 0; d < HD; d++) sk += (double)ko[kv * HD + d] * ko[kv * HD + d];
            float ik = 1.0f / sqrtf((float)(sk / HD) + eps);
            for (int d = 0; d < HD; d++) ko[kv * HD + d] *= ik * bf16_g(knw[d]);
            rope_apply(&ko[kv * HD], HD, seq_pos);

            // Store into KV cache
            memcpy(&kvc.k[seq_pos * NKV * HD + kv * HD], &ko[kv * HD], HD * 4);
            memcpy(&kvc.v[seq_pos * NKV * HD + kv * HD], &vo[kv * HD], HD * 4);
        }
    }
    kvc.n = seq_pos + 1;
    int cl = kvc.n;

    // Attention: softmax(Q·K^T) · V  (CPU, GQA)
    for (int hh = 0; hh < NH; hh++) {
        int kv = hh / GQA;
        std::vector<float> sc(cl);
        for (int p = 0; p < cl; p++) {
            double s = 0;
            for (int d = 0; d < HD; d++)
                s += qo[hh * HD + d] * kvc.k[p * NKV * HD + kv * HD + d];
            sc[p] = (float)(s / sqrtf((float)HD));
        }
        float mx = sc[0];
        for (int p = 1; p < cl; p++) if (sc[p] > mx) mx = sc[p];
        // Clamp to avoid exp overflow
        if (mx > 64.0f) mx = 64.0f;
        double sum = 0;
        for (int p = 0; p < cl; p++) { sc[p] = expf(fmaxf(sc[p] - mx, -64.0f)); sum += sc[p]; }
        for (int p = 0; p < cl; p++) sc[p] /= (float)sum;
        for (int d = 0; d < HD; d++) {
            float s = 0;
            for (int p = 0; p < cl; p++) s += sc[p] * kvc.v[p * NKV * HD + kv * HD + d];
            atto[hh * HD + d] = s;
        }
    }

    // O projection
    npu.gemm_by_dims(atto, 1, NH * HD, pw.o.data(), pw.o.size(), H, oo);
    bool in_nan = false; for (int i = 0; i < NH*HD; i++) if (std::isnan(atto[i])) in_nan = true;
    bool nan_detected = false;
    for (int i = 0; i < H; i++) if (std::isnan(oo[i])) nan_detected = true;
    if (l == 0) {
      printf("  L%d: atto[0..4]=", l); for (int i=0;i<5;i++) printf("%.1f ", atto[i]);
      printf("  oo[0..4]="); for (int i=0;i<5;i++) printf("%.1f ", oo[i]);
      printf("  input_nan=%d o_nan=%d\n", in_nan, nan_detected);
    }
    if (nan_detected) printf("  L%d: NAN in O!\n", l);

    // Residual
    for (int i = 0; i < H; i++) h[i] = save[i] + oo[i];

    // Post-attention RMS norm
    rms_norm(h, (const uint16_t *)(d + df + lo.pa_off), H, eps);

    // MLP: gate, up, SiLU(gate)*up, down
    npu.gemm_by_dims(h, 1, H, pw.g.data(), pw.g.size(), IM, g);
    bool gn = false; for (int i = 0; i < IM; i++) if (std::isnan(g[i])) gn = true;
    if (gn) printf("  L%d: NAN in G!\n", l);
    npu.gemm_by_dims(h, 1, H, pw.u.data(), pw.u.size(), IM, u);
    bool un = false; for (int i = 0; i < IM; i++) if (std::isnan(u[i])) un = true;
    if (un) printf("  L%d: NAN in U!\n", l);
    for (int i = 0; i < IM; i++) su[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
    npu.gemm_by_dims(su, 1, IM, pw.d.data(), pw.d.size(), H, dow);
    bool dn = false; for (int i = 0; i < H; i++) if (std::isnan(dow[i])) dn = true;
    if (dn) printf("  L%d: NAN in D!\n", l);

    // Full residual
    for (int i = 0; i < H; i++) h[i] = save[i] + oo[i] + dow[i];
    bool hn = false; for (int i = 0; i < H; i++) if (std::isnan(h[i])) hn = true;
    if (hn) printf("  L%d: NAN after layer! oo[0]=%.1f dow[0]=%.1f\n", l, oo[0], dow[0]);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== NPU v6 — Qwen3-0.6B (5-Variant Fused Backend) ===\n\n");

    const char *mp = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    int fd = open(mp, O_RDONLY);
    struct stat st;
    fstat(fd, &st);
    uint8_t *d = (uint8_t *)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    uint64_t hsz;
    memcpy(&hsz, d, 8);
    uint64_t df = 8 + hsz;
    auto i8p = [&](uint64_t o) { return d + df + o; };
    auto emb = (const uint16_t *)(d + df);
    const char *js = (const char *)(d + 8);
    size_t jl = hsz;

    const int H = 1024, L = 28, NH = 16, NKV = 8, HD = 128, IM = 3072, V = 151936, GQA = 2;
    const float eps = 1e-6f;

    // Parse weight offsets
    LO lo[28];
    for (int l = 0; l < L; l++) {
        char b[128];
        snprintf(b, sizeof(b), "model.layers.%d.self_attn.q_proj.weight", l);   lo[l].qp = joff(js, jl, b);
        snprintf(b, sizeof(b), "model.layers.%d.self_attn.k_proj.weight", l);   lo[l].kp = joff(js, jl, b);
        snprintf(b, sizeof(b), "model.layers.%d.self_attn.v_proj.weight", l);   lo[l].vp = joff(js, jl, b);
        snprintf(b, sizeof(b), "model.layers.%d.self_attn.o_proj.weight", l);   lo[l].op = joff(js, jl, b);
        snprintf(b, sizeof(b), "model.layers.%d.mlp.gate_proj.weight", l);      lo[l].gp = joff(js, jl, b);
        snprintf(b, sizeof(b), "model.layers.%d.mlp.up_proj.weight", l);        lo[l].up = joff(js, jl, b);
        snprintf(b, sizeof(b), "model.layers.%d.mlp.down_proj.weight", l);      lo[l].dp = joff(js, jl, b);
        snprintf(b, sizeof(b), "model.layers.%d.input_layernorm.weight", l);    lo[l].in_off = joff(js, jl, b);
        snprintf(b, sizeof(b), "model.layers.%d.post_attention_layernorm.weight", l); lo[l].pa_off = joff(js, jl, b);
        snprintf(b, sizeof(b), "model.layers.%d.self_attn.q_norm.weight", l);   lo[l].qn_off = joff(js, jl, b);
        snprintf(b, sizeof(b), "model.layers.%d.self_attn.k_norm.weight", l);   lo[l].kn_off = joff(js, jl, b);
    }
    uint64_t nm_off = joff(js, jl, "model.norm.weight");
    uint64_t lm_off = joff(js, jl, "lm_head.weight");

    // Init NPU multi-variant backend
    NpuMultiVariant npu;
    if (!npu.init()) { printf("NPU init failed\n"); return 1; }

    // Pre-dequant + pre-pack weights
    printf("\nPre-dequant + pre-pack...\n");
    auto t0 = std::chrono::steady_clock::now();
    PW pw[28];
    float *lm_f; int lm_r, lm_c;
    for (int l = 0; l < L; l++) {
        int unused;
        // Q: H×2048 → K=H=1024, N=2048
        // Q: H×2048 → dequant (in_feat=1024), K=1024, N=2048
        float *qw = dequant_i8_to_float(i8p(lo[l].qp), 256, &lm_r, &unused);
        pw[l].q = pre_pack_variant(qw, lm_r, H, lm_r, 1024, 2048); free(qw);
        // K: H×1024 → dequant (in_feat=1024), K=1024, N=1024
        float *kw = dequant_i8_to_float(i8p(lo[l].kp), 128, &lm_r, &unused);
        pw[l].k = pre_pack_variant(kw, lm_r, H, lm_r, 1024, 1024); free(kw);
        // V: H×1024 → same as K
        float *vw = dequant_i8_to_float(i8p(lo[l].vp), 128, &lm_r, &unused);
        pw[l].v = pre_pack_variant(vw, lm_r, H, lm_r, 1024, 1024); free(vw);
        // O: NH*HD×H (2048×1024) → FIXED: dequant with in_feat=2048 via _ex
        //    The old code used 4-arg dequant which hardcodes in_feat=1024,
        //    scrambling the O weight tile grid. This caused hidden state blowup.
        float *ow = dequant_i8_to_float_ex(i8p(lo[l].op), 256, 2048, &lm_r, &unused);
        pw[l].o = pre_pack_variant(ow, H, NH * HD, lm_r, 2048, 1024); free(ow);
        // Gate: H×3072 → dequant (in_feat=1024), K=1024, N=3072
        float *gw = dequant_i8_to_float(i8p(lo[l].gp), 384, &lm_r, &unused);
        pw[l].g = pre_pack_variant(gw, lm_r, H, lm_r, 1024, 3072); free(gw);
        // Up: H×3072 → dequant (in_feat=1024), K=1024, N=3072
        float *uw = dequant_i8_to_float(i8p(lo[l].up), 384, &lm_r, &unused);
        pw[l].u = pre_pack_variant(uw, lm_r, H, lm_r, 1024, 3072); free(uw);
        // Down: 3072×H → dequant (in_feat=1024), K=3072, N=1024
        float *dw = dequant_i8_to_float(i8p(lo[l].dp), 384, &lm_r, &unused);
        pw[l].d = pre_pack_variant(dw, H, IM, lm_r, 3072, 1024); free(dw);
        if (l % 7 == 0) printf("  L%d\n", l);
    }
    // LM head (in_feat=1024)
    lm_f = dequant_i8_to_float(i8p(lm_off), 18992, &lm_r, &lm_c);

    printf("Pre-packed in %.0f ms\n",
           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());

    // Init RoPE
    rope_init(HD, 1000000.0f, 4096);

    // KV cache per layer (max 64 tokens)
    std::vector<KV> kvc(L);

    // Working buffers
    std::vector<float> h(H), qo(2048), ko(1024), vo(1024), atto(2048),
                       oo(H), g(IM), u(IM), su(IM), dow(H), logits(V);

    // 9-token chat template prefill
    int prompt_tokens[] = {151643, 872, 198, 11852, 151644, 198, 151643, 77091, 198};
    int n_prompt = 9;
    int seq_pos = 0;

    printf("\nPrefilling %d tokens...\n", n_prompt);
    auto t_prefill = std::chrono::steady_clock::now();
    for (int pi = 0; pi < n_prompt; pi++) {
        int tok = prompt_tokens[pi];
        for (int i = 0; i < H; i++) h[i] = bf16_g(emb[tok * H + i]);
        for (int l = 0; l < L; l++)
            forward_layer(npu, l, seq_pos, lo[l], pw[l],
                          h.data(), qo.data(), ko.data(), vo.data(),
                          atto.data(), oo.data(), g.data(), u.data(),
                          su.data(), dow.data(), kvc[l],
                          eps, H, NH, NKV, HD, IM, GQA, d, df);
        seq_pos++;
        if (pi % 3 == 0) printf("  Prefill %d/%d\n", pi + 1, n_prompt);
    }
    double prefill_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_prefill).count();
    printf("  Done: %.0f ms (%.0f ms/tok)\n", prefill_ms, prefill_ms / n_prompt);

    // Generate
    printf("\nGenerating...\n");
    auto ti = std::chrono::steady_clock::now();
    for (int step = 0; step < 8; step++) {
        auto ts = std::chrono::steady_clock::now();

        // 28 transformer layers
        for (int l = 0; l < L; l++)
            forward_layer(npu, l, seq_pos, lo[l], pw[l],
                          h.data(), qo.data(), ko.data(), vo.data(),
                          atto.data(), oo.data(), g.data(), u.data(),
                          su.data(), dow.data(), kvc[l],
                          eps, H, NH, NKV, HD, IM, GQA, d, df);

        // Final RMS norm
        auto fnw = (const uint16_t *)(d + df + nm_off);
        std::vector<float> no(H);
        memcpy(no.data(), h.data(), H * sizeof(float));
        printf("  h[0..4] before norm: "); for (int i = 0; i < 5; i++) printf("%.1f ", h[i]); printf("\n");
        rms_norm(no.data(), fnw, H, eps);
        printf("  h[0..4] after norm: "); for (int i = 0; i < 5; i++) printf("%.1f ", no[i]); printf("\n");

        // LM head (CPU matvec)
        for (int n = 0; n < V; n++) {
            double s = 0;
            for (int k = 0; k < H; k++) {
                uint16_t r = emb[n * H + k];
                if ((r & 0x7F80) != 0x7F80) s += no[k] * bf16f(r);
            }
            logits[n] = (float)s;
        }

        // Greedy decode
        int token = 0;
        float mx = logits[0];
        for (int i = 1; i < V; i++) if (logits[i] > mx) { mx = logits[i]; token = i; }
        float mn = logits[0];
        for (int i = 1; i < V; i++) if (logits[i] < mn) mn = logits[i];

        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - ts).count();
        printf("  [%d] %d (%.0f ms) [%.1f,%.1f]\n", step, token, ms, mx, mn);

        // Next token embedding
        for (int i = 0; i < H; i++) h[i] = bf16_g(emb[token * H + i]);
        seq_pos++;
    }

    double total_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - ti).count();
    printf("\nAverage: %.0f ms/tok\n", total_s * 1000 / 8);

    // Cleanup
    free(lm_f);
    munmap(d, st.st_size);
    return 0;
}