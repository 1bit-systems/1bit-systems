// NPU Fused Engine — INT8 GEMMs, 28-layer pipeline, CPU attention
// Based on proven npu_engine_i8.cpp architecture (174ms/tok verified)
// Removes BFP16 entirely — uses INT8 xclbins with per-layer scale factors
//
// Build:
//   c++ -std=gnu++17 -O2 -g -I../include \
//       -o ../build/npu_fused_i8 npu_engine_fused_i8.cpp \
//       -L/home/bcloud/torch2aie/toolchain/xrt/lib64 \
//       -Wl,-rpath,/home/bcloud/torch2aie/toolchain/xrt/lib64 \
//       -lxrt_coreutil
//
// Run:
//   sudo LD_LIBRARY_PATH=/home/bcloud/torch2aie/toolchain/xrt/lib64:\
//     /home/bcloud/torch2aie/toolchain/mlir_aie.libs:\
//     /home/bcloud/torch2aie/toolchain/sysroot/usr/lib64 \
//     ./npu_fused_i8

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

extern "C" float* dequant_i8_to_float(const uint8_t*,int,int*,int*);

// BF16 helpers
static inline float bf16f(uint16_t v) {
    uint32_t b = v << 16; float f; memcpy(&f, &b, 4); return f;
}
static inline float bf16g(uint16_t v) {
    return (v & 0x7F80) == 0x7F80 ? 0.0f : bf16f(v);
}

// Qwen3-0.6B constants
static constexpr int H = 1024;       // hidden size
static constexpr int NC = 28;        // num layers
static constexpr int NH = 16;        // num Q heads
static constexpr int NKV = 8;        // num KV heads
static constexpr int HD = 128;       // head dim
static constexpr int IM = 3072;      // intermediate size
static constexpr int NV = 151936;    // vocab size
static constexpr int GQA = 2;        // GQA ratio
static constexpr int XM = 128;       // NPU tile M
static constexpr int AW = 4;         // attention windows
static constexpr int WQH = NH / AW;  // Q heads per window (4)
static constexpr int WKVH = NKV / AW;// KV heads per window (2)
static constexpr float EPS = 1e-6f;

// Clean NaN/inf
static inline void cn(float* x, int n) {
    for (int i = 0; i < n; i++)
        if (!std::isfinite(x[i])) x[i] = 0.0f;
}

// Softmax
static inline void sm(float* sc, int n) {
    if (n <= 0) return;
    cn(sc, n);
    float mx = sc[0];
    for (int i = 1; i < n; i++) if (sc[i] > mx) mx = sc[i];
    double s = 0;
    for (int i = 0; i < n; i++) {
        float d = sc[i] - mx;
        if (d > 80) d = 80; else if (d < -80) d = -80;
        sc[i] = expf(d); s += sc[i];
    }
    if (s <= 0) { float iv = 1.0f / n; for (int i = 0; i < n; i++) sc[i] = iv; return; }
    float is = 1.0f / (float)s;
    for (int i = 0; i < n; i++) sc[i] *= is;
}

// RMSNorm with weight
static inline void rn_c(float* x, const float* w, int n) {
    cn(x, n);
    double ss = 0;
    for (int i = 0; i < n; i++)
        if (std::isfinite(x[i])) ss += (double)x[i] * x[i];
    float ir = 1.0f / sqrtf((float)(ss / n) + EPS);
    for (int i = 0; i < n; i++)
        x[i] = std::isfinite(x[i]) ? x[i] * ir * w[i] : 0.0f;
}

// RoPE precomputation
static std::vector<float> rc, rs;
static void ri(int hd, float th, int mp) {
    rc.resize(mp * hd); rs.resize(mp * hd);
    for (int p = 0; p < mp; p++)
        for (int d = 0; d < hd; d += 2) {
            float f = 1.0f / powf(th, (float)d / hd), a = p * f;
            rc[p * hd + d] = cosf(a); rs[p * hd + d] = sinf(a);
            rc[p * hd + d + 1] = cosf(a); rs[p * hd + d + 1] = sinf(a);
        }
}

// Apply RoPE
static inline void ra(float* x, int hd, int p) {
    for (int d = 0; d < hd; d += 2) {
        float a = x[d], b = x[d + 1];
        float c = rc[p * hd + d], s = rs[p * hd + d];
        x[d] = a * c - b * s;
        x[d + 1] = b * c + a * s;
    }
}

// JSON offset lookup
static uint64_t jo(const char* js, size_t jl, const char* nm) {
    size_t nl = strlen(nm);
    const char* p = js, * e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, nm, nl);
        if (!q) return 0;
        if (q > js && *(q - 1) == '"' && *(q + nl) == '"') {
            auto o = strstr(q, "\"data_offsets\"");
            if (o) { auto a = strchr(o, '['); if (a) return strtoull(a + 1, NULL, 10); }
        }
        p = q + 1;
    }
    return 0;
}

// ── INT8 GEMM context ──────────────────────────────────────────────
struct I8Ctx {
    const char* name;
    int MD, KD, ND;
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::kernel> k;
    std::vector<uint32_t> ins;
    std::unique_ptr<xrt::bo> bI, bA, bC;
    std::vector<std::unique_ptr<xrt::bo>> layerB;
    int8_t* Am;
    int16_t* Cm;

    bool init(xrt::device& d, const char* xp, const char* ip, int gid_B) {
        FILE* f = fopen(ip, "rb");
        if (!f) { fprintf(stderr, "  FAIL: cannot open %s\n", ip); return false; }
        fseek(f, 0, 2); long sz = ftell(f); fseek(f, 0, 0);
        ins.resize(sz / 4);
        fread(ins.data(), 4, ins.size(), f); fclose(f);
        xc = std::make_unique<xrt::xclbin>(std::string(xp));
        d.register_xclbin(*xc);
        hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
        k = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");
        bI = std::make_unique<xrt::bo>(d, ins.size() * 4, XCL_BO_FLAGS_CACHEABLE, k->group_id(1));
        memcpy(bI->map(), ins.data(), ins.size() * 4);
        bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bA = std::make_unique<xrt::bo>(d, (size_t)MD * KD, XRT_BO_FLAGS_HOST_ONLY, k->group_id(3));
        bC = std::make_unique<xrt::bo>(d, (size_t)MD * ND * 2, XRT_BO_FLAGS_HOST_ONLY, k->group_id(5));
        Am = (int8_t*)bA->map();
        Cm = (int16_t*)bC->map();
        for (int l = 0; l < NC; l++) {
            layerB.emplace_back(std::make_unique<xrt::bo>(
                d, (size_t)KD * ND, XRT_BO_FLAGS_HOST_ONLY, k->group_id(gid_B)));
        }
        return true;
    }

    // Pack weight matrix w[K][N] into INT8 with per-matrix scale sout
    void packB(int l, const float* w, int K, int N, float& sout) {
        float amax = 0;
        for (int i = 0; i < K * N; i++) {
            float a = fabsf(w[i]);
            if (std::isfinite(a) && a > amax) amax = a;
        }
        if (amax < 1e-12f) amax = 1.0f;
        sout = amax / 127.0f;
        float is = 127.0f / amax;
        auto* Bm = (int8_t*)layerB[l]->map();
        for (int i = 0; i < K * N; i++) {
            float v = w[i];
            if (!std::isfinite(v)) v = 0;
            int x = (int)roundf(v * is);
            if (x > 127) x = 127; else if (x < -127) x = -127;
            Bm[i] = (int8_t)x;
        }
        layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // Run INT8 GEMM: A[M][K] * B[K][N] → C[M][N]
    inline void go(int l, const float* A, int am, int ak,
                   float ascale, float Bscale, float* C, int an) {
        float ais = 1.0f / ascale;
        memset(Am, 0, (size_t)MD * KD);
        for (int m = 0; m < am; m++)
            for (int k = 0; k < ak; k++) {
                float v = A[m * ak + k];
                if (!std::isfinite(v)) v = 0;
                int q = (int)roundf(v * ais);
                if (q > 127) q = 127; else if (q < -127) q = -127;
                Am[m * KD + k] = (int8_t)q;
            }
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r = (*k)((unsigned)3, *bI, (unsigned)ins.size(), *bA, *layerB[l], *bC);
        r.wait();
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        float cs = ascale * Bscale;
        for (int m = 0; m < am; m++)
            for (int n = 0; n < an; n++) {
                float val = (float)Cm[m * ND + n] * cs;
                if (!std::isfinite(val)) val = 0;
                C[m * an + n] = val;
            }
    }
};

// ── CPU Attention (proven fast for <32 tokens) ────────────────────
struct AttnCPU {
    static void run(const float* Q, const float* K, const float* V,
                    int n_tokens, float* out) {
        constexpr int WQH = 4,  WKVH = 2, HD = 128, GQA = 2;
        for (int h = 0; h < WQH; h++) {
            float scores[32]; // max 32 tokens
            float mx = -1e30f;
            for (int p = 0; p < n_tokens; p++) {
                double s = 0;
                for (int d = 0; d < HD; d++)
                    s += Q[h * HD + d] * K[p * NKV * HD + (h / GQA) * HD + d];
                scores[p] = (float)(s / sqrtf((float)HD));
                if (scores[p] > mx) mx = scores[p];
            }
            double sum = 0;
            for (int p = 0; p < n_tokens; p++) {
                scores[p] = expf(scores[p] - mx);
                sum += scores[p];
            }
            if (sum <= 0) sum = 1;
            float is = 1.0f / (float)sum;
            for (int d = 0; d < HD; d++) {
                float s = 0;
                for (int p = 0; p < n_tokens; p++)
                    s += scores[p] * is * V[p * NKV * HD + (h / GQA) * HD + d];
                out[h * HD + d] = s;
            }
        }
    }
};

// ── Main ──────────────────────────────────────────────────────────
int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== NPU Fused INT8 — Qwen3-0.6B (28 layers, INT8 GEMMs, CPU attn) ===\n\n");

    // Open model
    const char* mp = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    int fd = open(mp, O_RDONLY);
    if (fd < 0) { perror("open model"); return 1; }
    struct stat st; fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    uint64_t hsz; memcpy(&hsz, md, 8);
    uint64_t df = 8 + hsz;
    auto i8p = [&](uint64_t o) { return md + df + o; };
    auto emb = (const uint16_t*)(md + df);
    const char* js = (const char*)(md + 8);
    size_t jl = hsz;

    // Parse layer weight offsets
    struct LO {
        uint64_t qp, kp, vp, op, gp, up, dp, in_off, pa_off, qn_off, kn_off;
    } lo[NC];
    char b[128];
    for (int l = 0; l < NC; l++) {
        snprintf(b, 128, "model.layers.%d.self_attn.q_proj.weight", l);
        lo[l].qp = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.self_attn.k_proj.weight", l);
        lo[l].kp = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.self_attn.v_proj.weight", l);
        lo[l].vp = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.self_attn.o_proj.weight", l);
        lo[l].op = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.mlp.gate_proj.weight", l);
        lo[l].gp = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.mlp.up_proj.weight", l);
        lo[l].up = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.mlp.down_proj.weight", l);
        lo[l].dp = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.input_layernorm.weight", l);
        lo[l].in_off = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.post_attention_layernorm.weight", l);
        lo[l].pa_off = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.self_attn.q_norm.weight", l);
        lo[l].qn_off = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.self_attn.k_norm.weight", l);
        lo[l].kn_off = jo(js, jl, b);
    }
    uint64_t no = jo(js, jl, "model.norm.weight");
    uint64_t lo_off = jo(js, jl, "lm_head.weight");

    // Load norm weights
    float in_n[NC][H], pa_n[NC][H], fin[H], qn_w[NC][HD], kn_w[NC][HD];
    for (int l = 0; l < NC; l++) {
        auto iw = (const uint16_t*)(md + df + lo[l].in_off);
        auto pw_ = (const uint16_t*)(md + df + lo[l].pa_off);
        auto qw = (const uint16_t*)(md + df + lo[l].qn_off);
        auto kw = (const uint16_t*)(md + df + lo[l].kn_off);
        for (int i = 0; i < H; i++) { in_n[l][i] = bf16g(iw[i]); pa_n[l][i] = bf16g(pw_[i]); }
        for (int i = 0; i < HD; i++) { qn_w[l][i] = bf16g(qw[i]); kn_w[l][i] = bf16g(kw[i]); }
    }
    {
        auto fw = (const uint16_t*)(md + df + no);
        for (int i = 0; i < H; i++) fin[i] = bf16g(fw[i]);
    }

    // Init NPU device + INT8 contexts
    printf("Init INT8 GEMM contexts...\n");
    xrt::device dev(0);

#define D "/home/bcloud/npu-sandbox/npu-infer/build/int8"
    I8Ctx cq{ "QKV", XM, H, 4096 };
    I8Ctx co{ "O",   XM, NH*HD, H };
    I8Ctx cg{ "GU",  XM, H, 6144 };
    I8Ctx cd{ "D",   XM, IM, H };

    if (!cq.init(dev, D "/final_i8_QKV_v.xclbin", D "/insts_i8_QKV_v.txt", 4)) return 1;
    if (!co.init(dev, D "/final_i8_O_v.xclbin",   D "/insts_i8_O_v.txt",   4)) return 1;
    if (!cg.init(dev, D "/final_i8_GU_v.xclbin",  D "/insts_i8_GU_v.txt",  4)) return 1;
    if (!cd.init(dev, D "/final_i8_D_v.xclbin",   D "/insts_i8_D_v.txt",   4)) return 1;
#undef D

    // Pre-dequantize + pack all weights
    printf("Dequant + Pack weights...\n");
    auto tp = std::chrono::steady_clock::now();

    struct { float qk, o_, g_, d_; } wsc[NC];
    for (int l = 0; l < NC; l++) {
        int qr, kr, vr, or_, gr, ur, dr, unused;
        float* qw = dequant_i8_to_float(i8p(lo[l].qp), 256, &qr, &unused);
        float* kw = dequant_i8_to_float(i8p(lo[l].kp), 128, &kr, &unused);
        float* vw = dequant_i8_to_float(i8p(lo[l].vp), 128, &vr, &unused);
        int t = qr + kr + vr;
        std::vector<float> w((size_t)H * t);
        for (int k = 0; k < H; k++) {
            memcpy(&w[k * t],        &qw[k * qr], qr * 4);
            memcpy(&w[k * t + qr],   &kw[k * kr], kr * 4);
            memcpy(&w[k * t + qr + kr], &vw[k * vr], vr * 4);
        }
        cq.packB(l, w.data(), H, t, wsc[l].qk);
        free(qw); free(kw); free(vw);

        float* ow = dequant_i8_to_float(i8p(lo[l].op), 256, &or_, &unused);
        co.packB(l, ow, or_, H, wsc[l].o_);
        free(ow);

        float* gw = dequant_i8_to_float(i8p(lo[l].gp), 384, &gr, &unused);
        float* uw = dequant_i8_to_float(i8p(lo[l].up), 384, &ur, &unused);
        int t2 = gr + ur;
        std::vector<float> w2((size_t)H * t2);
        for (int k = 0; k < H; k++) {
            memcpy(&w2[k * t2],      &gw[k * gr], gr * 4);
            memcpy(&w2[k * t2 + gr], &uw[k * ur], ur * 4);
        }
        cg.packB(l, w2.data(), H, t2, wsc[l].g_);
        free(gw); free(uw);

        float* dw = dequant_i8_to_float(i8p(lo[l].dp), 384, &dr, &unused);
        cd.packB(l, dw, dr, H, wsc[l].d_);
        free(dw);
    }
    // Pre-read but free lm_head to check offset validity
    int lr, lc;
    free(dequant_i8_to_float(i8p(lo_off), 18992, &lr, &lc));

    printf("  %.0f ms\n",
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tp).count());

    // Precompute RoPE
    ri(HD, 1000000.0f, 4096);

    // KV cache
    struct KV {
        std::vector<float> k, v;
        int n;
        KV() : k(4096 * NKV * HD), v(4096 * NKV * HD), n(0) {}
    };
    std::vector<KV> kv(NC);

    // Per-layer activation buffers
    std::vector<float> h(H);
    std::vector<float> qo(4096);
    std::vector<float> ko(1024);
    std::vector<float> vo(1024);
    std::vector<float> at(2048);   // attention output (NH*HD = 2048)
    std::vector<float> oo(H);
    std::vector<float> gt(6144);
    std::vector<float> su(IM);     // SiLU output
    std::vector<float> dwo(H);
    std::vector<float> lg(NV);
    std::vector<float> sb(H);      // skip connection buffer
    std::vector<float> sc(4096);   // score buffer (unused here but kept)
    int sp = 0;  // sequence position

    // ── Single layer forward ──────────────────────────────────────
    auto layer = [&](int l) {
        // Self-attention
        memcpy(sb.data(), h.data(), H * 4);
        rn_c(h.data(), in_n[l], H);

        // QKV projection (fused: Q 1024×2048 + K 1024×1024 + V 1024×1024 = 1024×4096)
        cq.go(l, h.data(), 1, H, 5.0f / 127.0f, wsc[l].qk, qo.data(), 4096);
        cn(qo.data(), 4096);

        // Split Q, K, V
        memcpy(ko.data(), &qo[2048], 4096);
        memcpy(vo.data(), &qo[3072], 4096);

        // QK-norm + RoPE
        float* qn = qn_w[l];
        float* kn = kn_w[l];
        for (int hh = 0; hh < NH; hh++) {
            // Q-norm
            double sq = 0;
            for (int d = 0; d < HD; d++)
                sq += qo[hh * HD + d] * qo[hh * HD + d];
            float iq = 1.0f / sqrtf((float)(sq / HD) + EPS);
            for (int d = 0; d < HD; d++) qo[hh * HD + d] *= iq * qn[d];
            ra(&qo[hh * HD], HD, sp);

            // K-norm + store KV (GQA: 1 K head per GQA Q heads)
            if (hh % GQA == 0) {
                int kvh = hh / GQA;
                double sk = 0;
                for (int d = 0; d < HD; d++)
                    sk += ko[kvh * HD + d] * ko[kvh * HD + d];
                float ik = 1.0f / sqrtf((float)(sk / HD) + EPS);
                for (int d = 0; d < HD; d++) ko[kvh * HD + d] *= ik * kn[d];
                ra(&ko[kvh * HD], HD, sp);
                memcpy(&kv[l].k[sp * NKV * HD + kvh * HD], &ko[kvh * HD], HD * 4);
                memcpy(&kv[l].v[sp * NKV * HD + kvh * HD], &vo[kvh * HD], HD * 4);
            }
        }
        kv[l].n = sp + 1;
        int cl = kv[l].n;

        // CPU attention (faster than NPU for <32 tokens)
        // Kw/Vw start at offset 0 — K/V are stored positionally (token p at offset p*NKV*HD)
        for (int w = 0; w < AW; w++) {
            float* Qw = &qo[w * WQH * HD];
            float* Kw = &kv[l].k[w * WKVH * HD];
            float* Vw = &kv[l].v[w * WKVH * HD];
            AttnCPU::run(Qw, Kw, Vw, cl, &at[w * WQH * HD]);
        }

        // O projection
        co.go(l, at.data(), 1, NH * HD, 5.0f / 127.0f, wsc[l].o_, oo.data(), H);
        cn(oo.data(), H);
        for (int i = 0; i < H; i++) h[i] = sb[i] + oo[i];

        // MLP
        memcpy(sb.data(), h.data(), H * 4);
        rn_c(h.data(), pa_n[l], H);

        // Gate + Up projection (fused)
        cg.go(l, h.data(), 1, H, 5.0f / 127.0f, wsc[l].g_, gt.data(), 6144);
        cn(gt.data(), 6144);
        for (int i = 0; i < IM; i++) {
            float gv = gt[i];
            if (!std::isfinite(gv)) gv = 0;
            su[i] = (gv / (1.0f + expf(-gv))) * gt[IM + i];  // SiLU(gate) * up
        }

        // Down projection
        cd.go(l, su.data(), 1, IM, 5.0f / 127.0f, wsc[l].d_, dwo.data(), H);
        cn(dwo.data(), H);
        for (int i = 0; i < H; i++) h[i] = sb[i] + dwo[i];
    };

    // ── Prefill tokens ────────────────────────────────────────────
    int pt[] = { 151643, 872, 198, 11852, 151644, 198, 151643, 77091, 198 };
    int npt = 9;
    printf("\n=== Prefill %d tokens ===\n", npt);
    for (int pi = 0; pi < npt; pi++) {
        for (int i = 0; i < H; i++) h[i] = bf16g(emb[pt[pi] * H + i]);
        for (int l = 0; l < NC; l++) layer(l);
        sp++;
        if (pi % 3 == 0) printf("  %d/%d\n", pi + 1, npt);
    }
    printf("Done\n\n");

    // ── Generate tokens ───────────────────────────────────────────
    printf("=== Generate ===\n");
    int ng = 8;
    auto tgs = std::chrono::steady_clock::now();

    for (int st = 0; st < ng; st++) {
        auto ts = std::chrono::steady_clock::now();

        // Run all 28 layers
        for (int l = 0; l < NC; l++) layer(l);

        // Final RMSNorm
        memcpy(sb.data(), h.data(), H * 4);
        rn_c(sb.data(), fin, H);

        // LM head (dot product with embedding weight)
        for (int n = 0; n < NV; n++) {
            double s = 0;
            for (int k = 0; k < H; k++) {
                uint16_t r = emb[n * H + k];
                if ((r & 0x7F80) != 0x7F80)  // skip NaN/Inf
                    s += sb[k] * bf16f(r);
            }
            lg[n] = (float)s;
        }

        // Softmax sampling
        float mx = lg[0];
        for (int i = 1; i < NV; i++) if (lg[i] > mx) mx = lg[i];
        double sum = 0;
        for (int i = 0; i < NV; i++) {
            float d = lg[i] - mx;
            if (d < -80) d = -80;
            lg[i] = expf(d);
            sum += lg[i];
        }
        float rr = (float)rand() / RAND_MAX * (float)sum, acc = 0;
        int tok = 0;
        for (int i = 0; i < NV; i++) { acc += lg[i]; if (acc >= rr) { tok = i; break; } }

        double mss = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - ts).count();
        printf("  [%d] %d (%.0fms)\n", st, tok, mss);

        // Next token embedding
        for (int i = 0; i < H; i++) h[i] = bf16g(emb[tok * H + i]);
        sp++;
    }

    double tts = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - tgs).count();
    printf("\n=== %.0f ms/tok ===\n", tts * 1000.0 / ng);

    munmap(md, st.st_size);
    return 0;
}
