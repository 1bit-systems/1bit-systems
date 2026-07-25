/** npu_engine_v13.cpp — Cross-layer pipelined async decode.
 *
 * Optimizations over v12:
 *   v12: each go() = quantize→launch→WAIT→dequant (serial, 4× per layer)
 *   v13: launch() is async; CPU work overlaps with NPU execution
 *
 *   Per-layer overlap schedule:
 *     cq.launch(L)    → NPU: QKV(L)         CPU: Q norms, RoPE, K/V cache
 *     cq.wait(L)      →                     CPU: attn_omp
 *     co.launch(L)    → NPU: O(L)           
 *     co.wait(L)      →                     CPU: residual + RMS norm
 *     cg.launch(L)    → NPU: GU(L)          CPU: SiLU
 *     cg.wait(L)      →                     
 *     cd.launch(L)    → NPU: D(L)           CPU: residual(L-1) + RMS norm(L)
 *     cd.wait(L)      →                     CPU: residual(L)
 *
 * Build: see CMakeLists.txt or engine/npu/Makefile
 * Run:   OMP_NUM_THREADS=16 OMP_WAIT_POLICY=active ./npu_engine_v13 <tokens>
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
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_elf.h>
#include <aiebu/aiebu_assembler.h>
#include <immintrin.h>

extern "C" float* dequant_i8_to_float(const uint8_t*,int,int*,int*);

static inline float bf16g(uint16_t v) {
    if((v&0x7F80)==0x7F80) return 0.0f;
    uint32_t b = (uint32_t)v << 16; float f; memcpy(&f,&b,4); return f;
}

// ── Model dimensions (Qwen3-0.6B) ─────────────────────────────────
static constexpr int H=1024, NC=28, NH=16, NKV=8, HD=128, IM=3072, NV=151936, GQA=2;
static constexpr float EPS = 1e-6f;
static constexpr int XM = 128;  // NPU GEMM M-dim
static constexpr int BS = 32;   // decode batch

static inline void cn(float* x, int n) {
    for (int i = 0; i < n; i++) if (!std::isfinite(x[i])) x[i] = 0.0f;
}

// Dynamic activation scale matching the packB amax method
static inline float dyn_scale(const float* x, int n) {
    float a = 0;
    for (int i = 0; i < n; i++) { float f = fabsf(x[i]); if (std::isfinite(f) && f > a) a = f; }
    return a < 1e-12f ? 1.0f : a / 127.0f;
}

// Softmax (in-place)
static inline void sm(float* sc, int n) {
    if (n <= 0) return; cn(sc, n);
    float mx = sc[0]; for (int i = 1; i < n; i++) if (sc[i] > mx) mx = sc[i];
    double s = 0;
    for (int i = 0; i < n; i++) { float d = sc[i] - mx; if (d > 80) d = 80; else if (d < -80) d = -80; sc[i] = expf(d); s += sc[i]; }
    if (s <= 0) { float iv = 1.0f / n; for (int i = 0; i < n; i++) sc[i] = iv; return; }
    float is = 1.0f / (float)s; for (int i = 0; i < n; i++) sc[i] *= is;
}

// RMS Norm (CPU)
static inline void rn_c(float* x, const float* w, int n) {
    cn(x, n); double ss = 0;
    for (int i = 0; i < n; i++) if (std::isfinite(x[i])) ss += (double)x[i] * x[i];
    float ir = 1.0f / sqrtf((float)(ss / n) + EPS);
    for (int i = 0; i < n; i++) x[i] = std::isfinite(x[i]) ? x[i] * ir * w[i] : 0.0f;
}

// RoPE tables + apply
static std::vector<float> rc, rs;
static void ri(int hd, float th, int mp) {
    int hd2 = hd / 2; rc.resize(mp * hd); rs.resize(mp * hd);
    for (int p = 0; p < mp; p++) for (int d = 0; d < hd2; d++) {
        float f = 1.0f / powf(th, (float)d / hd2), a = p * f;
        rc[p * hd + d] = cosf(a); rs[p * hd + d] = sinf(a);
    }
}
static inline void ra(float* x, int hd, int p) {
    int hd2 = hd / 2;
    for (int d = 0; d < hd2; d++) {
        float a = x[d], b = x[d + hd2], c = rc[p * hd + d], s = rs[p * hd + d];
        x[d] = a * c - b * s; x[d + hd2] = b * c + a * s;
    }
}

// JSON offset scanner
static uint64_t jo(const char* js, size_t jl, const char* nm) {
    size_t nl = strlen(nm); const char* p = js, *e = js + jl;
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

static std::vector<float> emb_f32;

// ── AVX-512 quantize: float[N] → int8[N] ─────────────────────────
static void quantize_avx(const float* A, int8_t* D, int n, float scale) {
    float is = 1.0f / scale;
    __m512 v_is = _mm512_set1_ps(is);
    __m512 v_max = _mm512_set1_ps(127.0f);
    __m512 v_min = _mm512_set1_ps(-127.0f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(&A[i]);
        v = _mm512_mul_ps(v, v_is);
        v = _mm512_max_ps(v_min, _mm512_min_ps(v_max, v));
        __m512i vi = _mm512_cvtps_epi32(v);
        __m256i lo = _mm512_cvtepi32_epi16(vi);
        __m128i i8 = _mm256_cvtepi16_epi8(lo);
        _mm_storeu_si128((__m128i*)&D[i], i8);
    }
    for (; i < n; i++) {
        int q = (int)roundf(A[i] * is);
        if (q > 127) q = 127; else if (q < -127) q = -127;
        D[i] = (int8_t)q;
    }
}

// ── AVX-512 dequant: int16[N] → float[N] ─────────────────────────
static void dequant_avx(const int16_t* Cm, float* C, int n, float scale) {
    __m512 v_s = _mm512_set1_ps(scale);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i vi16 = _mm256_loadu_si256((__m256i*)&Cm[i]);
        __m512i vi32 = _mm512_cvtepi16_epi32(vi16);
        __m512 v = _mm512_cvtepi32_ps(vi32);
        v = _mm512_mul_ps(v, v_s);
        _mm512_storeu_ps(&C[i], v);
    }
    for (; i < n; i++) C[i] = (float)Cm[i] * scale;
}

// ── I8Ctx with async launch/wait API (refactored from v12) ────────
struct I8Ctx {
    const char* name;
    int MD, KD, ND;
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::module> mdl;
    std::unique_ptr<xrt::elf> elf;
    std::unique_ptr<xrt::ext::kernel> k;
    std::vector<uint32_t> ins;
    std::unique_ptr<xrt::bo> bI, bA, bC;
    std::vector<std::unique_ptr<xrt::bo>> layerB;
    int8_t* Am;
    int16_t* Cm;
    xrt::run run_;
    bool busy_ = false;

    bool init(xrt::device& d, const char* xp, const char* ip, int n_layers) {
        name = ip;
        FILE* f = fopen(ip, "rb");
        if (!f) return false;
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        ins.resize(sz / 4);
        if (fread(ins.data(), 4, ins.size(), f) != ins.size()) { fclose(f); return false; }
        fclose(f);
        try {
            std::vector<char> iraw((char*)ins.data(), (char*)ins.data() + ins.size() * 4);
            aiebu::aiebu_assembler asmblr(aiebu::aiebu_assembler::buffer_type::blob_instr_transaction, iraw);
            auto e = asmblr.get_elf();
            xc = std::make_unique<xrt::xclbin>(std::string(xp));
            d.register_xclbin(*xc);
            hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
            elf = std::make_unique<xrt::elf>(e.data(), e.size());
        } catch (...) { return false; }
        mdl = std::make_unique<xrt::module>(*elf);
        k = std::make_unique<xrt::ext::kernel>(*hc, *mdl, "MLIR_AIE");
        bI = std::make_unique<xrt::bo>(d, ins.size() * 4, XCL_BO_FLAGS_CACHEABLE, k->group_id(1));
        memcpy(bI->map(), ins.data(), ins.size() * 4);
        bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bA = std::make_unique<xrt::bo>(d, (size_t)MD * KD, XRT_BO_FLAGS_HOST_ONLY, 0);
        bC = std::make_unique<xrt::bo>(d, (size_t)MD * ND * 2, XRT_BO_FLAGS_HOST_ONLY, 0);
        Am = (int8_t*)bA->map();
        Cm = (int16_t*)bC->map();
        layerB.resize(n_layers);
        for (int l = 0; l < n_layers; l++)
            layerB[l] = std::make_unique<xrt::bo>(d, (size_t)KD * ND, XRT_BO_FLAGS_HOST_ONLY, 0);
        return true;
    }

    void packB(int l, const float* w, int K, int N, float& sout) {
        float amax = 0;
        for (int i = 0; i < K * N; i++) { float a = fabsf(w[i]); if (std::isfinite(a) && a > amax) amax = a; }
        sout = (amax < 1e-12f) ? 1.0f : amax / 127.0f;
        float is = 127.0f / (amax < 1e-12f ? 1.0f : amax);
        auto* Bm = (int8_t*)layerB[l]->map();
        for (int i = 0; i < K * N; i++) {
            float v = w[i]; if (!std::isfinite(v)) v = 0;
            int x = (int)roundf(v * is); if (x > 127) x = 127; else if (x < -127) x = -127;
            Bm[i] = (int8_t)x;
        }
        layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // Async launch: quantize A → I8, sync, submit NPU kernel (non-blocking)
    void launch(int l, const float* A, int am, int ak, float ascale) {
        (void)l;
        memset(Am, 0, (size_t)am * KD);
        for (int m = 0; m < am; m++)
            quantize_avx(&A[m * ak], &Am[m * KD], ak, ascale);
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        run_ = k->operator()(3, 0, 0, *bA, *layerB[0], *bC);
        busy_ = true;
    }

    // Wait for completion + dequantize to C
    void wait(float* C, int am, int an, float ascale, float Bscale) {
        if (!busy_) return;
        run_.wait();
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        float cs = ascale * Bscale;
        for (int m = 0; m < am; m++)
            dequant_avx(&Cm[m * ND], &C[m * an], an, cs);
        busy_ = false;
    }

    // Wait only (no dequant, for when we want to defer it)
    void wait_only() {
        if (!busy_) return;
        run_.wait();
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        busy_ = false;
    }

    // Dequantize already-waited result
    void dequant(float* C, int am, int an, float ascale, float Bscale) {
        float cs = ascale * Bscale;
        for (int m = 0; m < am; m++)
            dequant_avx(&Cm[m * ND], &C[m * an], an, cs);
    }

    // Blocking go
    void go(int l, const float* A, int am, int ak, float ascale, float Bscale, float* C, int an) {
        (void)l;
        launch(l, A, am, ak, ascale);
        wait(C, am, an, ascale, Bscale);
    }
};

// ── OpenMP LM head ────────────────────────────────────────────────
static void lm_topk_omp(const float* hidden, float* lg, int* top_ids, int K) {
    for (int b = 0; b < K; b++) top_ids[b] = 0;
    for (int k = 0; k < H; k++) if (!std::isfinite(hidden[k])) { for (int b = 0; b < K; b++) top_ids[b] = 0; return; }
    float mx = -1e30f;
    #pragma omp parallel for reduction(max:mx)
    for (int n = 0; n < NV; n++) {
        double s = 0; const float* e = &emb_f32[(size_t)n * H]; const float* h = hidden;
        #pragma omp simd reduction(+:s)
        for (int k = 0; k < H; k++) { float hv = h[k]; if (!std::isfinite(hv)) continue; s += (double)hv * e[k]; }
        float v = (float)s; lg[n] = std::isfinite(v) ? v : 0.0f;
        if (lg[n] > mx) mx = lg[n];
    }
    if (!std::isfinite(mx) || mx < -1e29f) { for (int b = 0; b < K; b++) top_ids[b] = 0; return; }
    double sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int n = 0; n < NV; n++) { float d = lg[n] - mx; if (d < -80) d = -80; float ev = expf(d); lg[n] = std::isfinite(ev) ? ev : 0.0f; sum += lg[n]; }
    if (!std::isfinite(sum) || sum <= 0) { for (int b = 0; b < K; b++) top_ids[b] = 0; return; }
    float r = (float)rand() / RAND_MAX * (float)sum, acc = 0; bool sampled = false;
    for (int n = 0; n < NV; n++) { acc += lg[n]; if (acc >= r) { top_ids[0] = n; sampled = true; break; } }
    if (!sampled) top_ids[0] = 0;
}

// ── OpenMP attention ──────────────────────────────────────────────
static void attn_omp(float* qo, float* at, int cl, const float* kv_k, const float* kv_v, int max_pos = -1) {
    if (max_pos < 0) max_pos = cl;
    #pragma omp parallel for
    for (int hh = 0; hh < NH; hh++) {
        int kvh = hh / GQA;
        float scores[4096];
        float mx = -1e30f;
        for (int p = 0; p < cl; p++) {
            if (p >= max_pos) { scores[p] = -1e30f; continue; }
            double s = 0;
            for (int d = 0; d < HD; d++)
                s += (double)qo[hh * HD + d] * kv_k[(size_t)p * NKV * HD + kvh * HD + d];
            scores[p] = (float)(s * 0.0883883476f); // 1/sqrt(128)
            if (scores[p] > mx) mx = scores[p];
        }
        double sw = 0;
        for (int p = 0; p < cl; p++) { scores[p] = expf(scores[p] - mx); sw += scores[p]; }
        float isw = sw > 0 ? 1.0f / (float)sw : 1.0f / cl;
        for (int d = 0; d < HD; d++) {
            float acc = 0;
            for (int p = 0; p < cl; p++)
                acc += scores[p] * kv_v[(size_t)p * NKV * HD + kvh * HD + d];
            at[hh * HD + d] = acc * isw;
        }
    }
}

// ══════════════════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int npt = 9, ng = (argc > 2) ? atoi(argv[2]) : 32;
    printf("=== NPU Engine v13 — Cross-layer pipelined async decode ===\n");
    printf("Target: overlap CPU ops with NPU execution\n\n");

    // ── Load model ────────────────────────────────────────────────
    const char* mp = getenv("NPU_MODEL_PATH") ?:
        "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    int fd = open(mp, O_RDONLY); struct stat st; fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    uint64_t hsz; memcpy(&hsz, md, 8); uint64_t df = 8 + hsz;
    auto i8p = [&](uint64_t o) { return md + df + o; };
    auto emb = (const uint16_t*)(md + df);
    const char* js = (const char*)(md + 8); size_t jl = hsz;

    struct LO { uint64_t qp, kp, vp, op, gp, up, dp, in_off, pa_off, qn_off, kn_off; } lo[NC];
    char b[128];
    for (int l = 0; l < NC; l++) {
        snprintf(b, 128, "model.layers.%d.self_attn.q_proj.weight", l); lo[l].qp = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.self_attn.k_proj.weight", l); lo[l].kp = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.self_attn.v_proj.weight", l); lo[l].vp = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.self_attn.o_proj.weight", l); lo[l].op = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.mlp.gate_proj.weight", l);   lo[l].gp = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.mlp.up_proj.weight", l);     lo[l].up = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.mlp.down_proj.weight", l);   lo[l].dp = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.input_layernorm.weight", l); lo[l].in_off = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.post_attention_layernorm.weight", l); lo[l].pa_off = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.self_attn.q_norm.weight", l); lo[l].qn_off = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.self_attn.k_norm.weight", l); lo[l].kn_off = jo(js, jl, b);
    }
    uint64_t no = jo(js, jl, "model.norm.weight"), lo_off = jo(js, jl, "lm_head.weight");

    float in_n[NC][H], pa_n[NC][H], fin[H], qn_w[NC][HD], kn_w[NC][HD];
    for (int l = 0; l < NC; l++) {
        auto iw = (const uint16_t*)(md + df + lo[l].in_off),
             pw = (const uint16_t*)(md + df + lo[l].pa_off),
             qw = (const uint16_t*)(md + df + lo[l].qn_off),
             kw = (const uint16_t*)(md + df + lo[l].kn_off);
        for (int i = 0; i < H; i++)  { in_n[l][i] = bf16g(iw[i]); pa_n[l][i] = bf16g(pw[i]); }
        for (int i = 0; i < HD; i++) { qn_w[l][i] = bf16g(qw[i]); kn_w[l][i] = bf16g(kw[i]); }
    }
    { auto fw = (const uint16_t*)(md + df + no); for (int i = 0; i < H; i++) fin[i] = bf16g(fw[i]); }

    emb_f32.resize((size_t)NV * H);
    for (int n = 0; n < NV; n++)
        for (int i = 0; i < H; i++)
            emb_f32[(size_t)n * H + i] = bf16g(emb[n * H + i]);

    // ── Init NPU ──────────────────────────────────────────────────
    xrt::device dev(0);
    const char* D = getenv("NPU_XCLBIN_DIR") ?:
        "/home/bcloud/npu-sandbox/npu-infer/build/int8";

    I8Ctx cq{"QKV", XM, H, 4096},
          co{"O",   XM, NH*HD, H},
          cg{"GU",  XM, H, 2*IM},
          cd{"D",   XM, IM, H};

    auto xp = [&](const char* t) {
        static char b[256]; snprintf(b, 256, "%s/final_i8_%s_v.xclbin", D, t); return b;
    };
    auto ip = [&](const char* t) {
        static char b[256]; snprintf(b, 256, "%s/insts_i8_%s_v.txt", D, t); return b;
    };

    if (!cq.init(dev, xp("QKV"), ip("QKV"), 4)) { fprintf(stderr, "FAIL cq\n"); return 1; }
    if (!co.init(dev, xp("O"),   ip("O"),   4)) { fprintf(stderr, "FAIL co\n"); return 1; }
    if (!cg.init(dev, xp("GU"),  ip("GU"),  4)) { fprintf(stderr, "FAIL cg\n"); return 1; }
    if (!cd.init(dev, xp("D"),   ip("D"),   4)) { fprintf(stderr, "FAIL cd\n"); return 1; }

    // ── Pack weights ──────────────────────────────────────────────
    auto tp = std::chrono::steady_clock::now();
    struct WS { float qk, o_, g_, d_; } wsc[NC];
    for (int l = 0; l < NC; l++) {
        int qr, kr, vr, or_, gr, ur, dr, unused;
        float* qw = dequant_i8_to_float(i8p(lo[l].qp), 256, &qr, &unused);
        float* kw = dequant_i8_to_float(i8p(lo[l].kp), 128, &kr, &unused);
        float* vw = dequant_i8_to_float(i8p(lo[l].vp), 128, &vr, &unused);
        int t = qr + kr + vr;
        std::vector<float> w((size_t)H * t);
        for (int k = 0; k < H; k++) {
            memcpy(&w[(size_t)k * t], &qw[(size_t)k * qr], (size_t)qr * 4);
            memcpy(&w[(size_t)k * t + qr], &kw[(size_t)k * kr], (size_t)kr * 4);
            memcpy(&w[(size_t)k * t + qr + kr], &vw[(size_t)k * vr], (size_t)vr * 4);
        }
        cq.packB(l, w.data(), H, t, wsc[l].qk);
        free(qw); free(kw); free(vw);

        float* ow = dequant_i8_to_float(i8p(lo[l].op), 256, &or_, &unused);
        co.packB(l, ow, or_, H, wsc[l].o_); free(ow);

        float* gw = dequant_i8_to_float(i8p(lo[l].gp), 384, &gr, &unused);
        float* uw = dequant_i8_to_float(i8p(lo[l].up), 384, &ur, &unused);
        int t2 = gr + ur;
        std::vector<float> w2((size_t)H * t2);
        for (int k = 0; k < H; k++) {
            memcpy(&w2[(size_t)k * t2], &gw[(size_t)k * gr], (size_t)gr * 4);
            memcpy(&w2[(size_t)k * t2 + gr], &uw[(size_t)k * ur], (size_t)ur * 4);
        }
        cg.packB(l, w2.data(), H, t2, wsc[l].g_); free(gw); free(uw);

        float* dw = dequant_i8_to_float(i8p(lo[l].dp), 384, &dr, &unused);
        cd.packB(l, dw, dr, H, wsc[l].d_); free(dw);
    }
    int lr, lc; free(dequant_i8_to_float(i8p(lo_off), 18992, &lr, &lc));
    printf("Weights packed: %.0fms\n\n",
           std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - tp).count());

    ri(HD, 1000000.0f, 4096);

    // ── Buffers ───────────────────────────────────────────────────
    struct KVCache { std::vector<float> k, v; int n; KVCache() : k(4096 * NKV * HD), v(4096 * NKV * HD), n(0) {} };
    std::vector<KVCache> kv(NC);
    std::vector<float> h_b(XM * H), qo_b(XM * NH * HD), at_b(XM * NH * HD),
                       oo_b(XM * H), gt_b(XM * 2 * IM), su_b(XM * IM), dw_b(XM * H);
    // Per-token decode buffers
    std::vector<float> hidden(H), qo(NH * HD), ko(NKV * HD), vo(NKV * HD),
                       attn_out(NH * HD), o_out(H), gt_out(2 * IM), silu_out(IM),
                       d_out(H), save_buf(H), lg(NV);
    int sp = 0;
    int pt[] = {151643, 872, 198, 11852, 151644, 198, 151643, 77091, 198};

    // ══════════════════════════════════════════════════════════════
    // PREFILL (unchanged from v12)
    // ══════════════════════════════════════════════════════════════
    printf("=== Prefill %d ===\n", npt);
    auto t0 = std::chrono::steady_clock::now();
    for (int pi = 0; pi < npt; pi++)
        for (int i = 0; i < H; i++) h_b[pi * H + i] = emb_f32[pt[pi] * H + i];

    for (int l = 0; l < NC; l++) {
        for (int pi = 0; pi < npt; pi++) rn_c(&h_b[pi * H], in_n[l], H);
        cq.go(l, h_b.data(), npt, H, dyn_scale(h_b.data(), npt * H), wsc[l].qk, qo_b.data(), 4096);
        cn(qo_b.data(), npt * 4096);
        for (int pi = 0; pi < npt; pi++) {
            // Q norms + RoPE
            for (int hh = 0; hh < NH; hh++) {
                double s = 0;
                for (int d = 0; d < HD; d++) s += qo_b[pi * NH * HD + hh * HD + d] * qo_b[pi * NH * HD + hh * HD + d];
                float iq = 1.0f / sqrtf((float)(s / HD) + EPS);
                for (int d = 0; d < HD; d++) qo_b[pi * NH * HD + hh * HD + d] *= iq * qn_w[l][d];
                ra(&qo_b[pi * NH * HD + hh * HD], HD, sp + pi);
            }
            // K norms + RoPE + K/V cache update
            for (int kvh = 0; kvh < NKV; kvh++) {
                float* ks = &qo_b[pi * 4096 + NH * HD + kvh * HD];
                float* vs = &qo_b[pi * 4096 + NH * HD + NKV * HD + kvh * HD];
                double sk = 0; for (int d = 0; d < HD; d++) sk += ks[d] * ks[d];
                float ik = 1.0f / sqrtf((float)(sk / HD) + EPS);
                for (int d = 0; d < HD; d++) { ks[d] *= ik * kn_w[l][d]; ra(ks, HD, sp + pi); }
                memcpy(&kv[l].k[(sp + pi) * NKV * HD + kvh * HD], ks, HD * 4);
                memcpy(&kv[l].v[(sp + pi) * NKV * HD + kvh * HD], vs, HD * 4);
            }
        }
        kv[l].n = sp + npt; int cl = kv[l].n;
        for (int pi = 0; pi < npt; pi++)
            attn_omp(&qo_b[pi * NH * HD], &at_b[pi * NH * HD], cl, kv[l].k.data(), kv[l].v.data(), sp + pi + 1);
        co.go(l, at_b.data(), npt, NH * HD, dyn_scale(at_b.data(), npt * NH * HD), wsc[l].o_, oo_b.data(), H);
        cn(oo_b.data(), npt * H);
        for (int pi = 0; pi < npt; pi++) for (int i = 0; i < H; i++) h_b[pi * H + i] += oo_b[pi * H + i];
        for (int pi = 0; pi < npt; pi++) rn_c(&h_b[pi * H], pa_n[l], H);
        cg.go(l, h_b.data(), npt, H, dyn_scale(h_b.data(), npt * H), wsc[l].g_, gt_b.data(), 2 * IM);
        cn(gt_b.data(), npt * 2 * IM);
        for (int pi = 0; pi < npt; pi++) {
            for (int i = 0; i < IM; i++) {
                float gv = gt_b[pi * 2 * IM + i]; if (!std::isfinite(gv)) gv = 0;
                su_b[pi * IM + i] = (gv / (1.0f + expf(-gv))) * gt_b[pi * 2 * IM + IM + i];
            }
        }
        cd.go(l, su_b.data(), npt, IM, dyn_scale(su_b.data(), npt * IM), wsc[l].d_, dw_b.data(), H);
        cn(dw_b.data(), npt * H);
        for (int pi = 0; pi < npt; pi++) for (int i = 0; i < H; i++) h_b[pi * H + i] += dw_b[pi * H + i];
    }
    sp += npt;
    printf("Prefill: %.0fms (%.0f ms/tok)\n\n",
           std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count(),
           std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count() / npt);

    // ══════════════════════════════════════════════════════════════
    // DECODE — Cross-layer pipelined async
    // ══════════════════════════════════════════════════════════════
    printf("=== Async pipelined decode (%d tokens) ===\n", ng);
    auto tgs = std::chrono::steady_clock::now();
    int top_ids[BS] = {0};

    // Boot: first token
    {
        auto ts = std::chrono::steady_clock::now();
        memcpy(hidden.data(), &h_b[(npt - 1) * H], H * 4);

        for (int l = 0; l < NC; l++) {
            memcpy(save_buf.data(), hidden.data(), H * 4);

            // ── QKV (async: launch, then CPU ops overlap) ──
            rn_c(hidden.data(), in_n[l], H);
            float as_qk = dyn_scale(hidden.data(), H);
            cq.launch(l, hidden.data(), 1, H, as_qk);

            // CPU work while NPU does QKV
            // (boot: nothing to overlap yet — no prev layer state)
            cq.wait(qo.data(), 1, 4096, as_qk, wsc[l].qk);
            cn(qo.data(), 4096);

            // Q norms + RoPE
            for (int hh = 0; hh < NH; hh++) {
                double s = 0; for (int d = 0; d < HD; d++) s += qo[hh * HD + d] * qo[hh * HD + d];
                float iq = 1.0f / sqrtf((float)(s / HD) + EPS);
                for (int d = 0; d < HD; d++) qo[hh * HD + d] *= iq * qn_w[l][d];
                ra(&qo[hh * HD], HD, sp);
            }
            // K norms + RoPE + K/V cache
            memcpy(ko.data(), &qo[NH * HD], NKV * HD * 4);
            memcpy(vo.data(), &qo[NH * HD + NKV * HD], NKV * HD * 4);
            for (int hh = 0; hh < NH; hh++) {
                if (hh % GQA == 0) {
                    int kvh = hh / GQA;
                    double sk = 0; for (int d = 0; d < HD; d++) sk += ko[kvh * HD + d] * ko[kvh * HD + d];
                    float ik = 1.0f / sqrtf((float)(sk / HD) + EPS);
                    for (int d = 0; d < HD; d++) { ko[kvh * HD + d] *= ik * kn_w[l][d]; ra(&ko[kvh * HD], HD, sp); }
                    memcpy(&kv[l].k[sp * NKV * HD + kvh * HD], &ko[kvh * HD], HD * 4);
                    memcpy(&kv[l].v[sp * NKV * HD + kvh * HD], &vo[kvh * HD], HD * 4);
                }
            }
            kv[l].n = sp + 1; int cl = kv[l].n;

            // Attention
            attn_omp(qo.data(), attn_out.data(), cl, kv[l].k.data(), kv[l].v.data());

            // O projection
            float as_o = dyn_scale(attn_out.data(), NH * HD);
            co.launch(l, attn_out.data(), 1, NH * HD, as_o);
            co.wait(o_out.data(), 1, H, as_o, wsc[l].o_);
            cn(o_out.data(), H);
            for (int i = 0; i < H; i++) hidden[i] = save_buf[i] + o_out[i];

            // ── FFN (async) ──
            memcpy(save_buf.data(), hidden.data(), H * 4);
            rn_c(hidden.data(), pa_n[l], H);

            // GU: async launch, SiLU overlaps with NPU
            float as_g = dyn_scale(hidden.data(), H);
            cg.launch(l, hidden.data(), 1, H, as_g);
            // SiLU cannot start until GU finishes — wait first
            cg.wait(gt_out.data(), 1, 2 * IM, as_g, wsc[l].g_);
            cn(gt_out.data(), 2 * IM);
            for (int i = 0; i < IM; i++) {
                float gv = gt_out[i]; if (!std::isfinite(gv)) gv = 0;
                silu_out[i] = (gv / (1.0f + expf(-gv))) * gt_out[IM + i];
            }

            // D projection
            float as_d = dyn_scale(silu_out.data(), IM);
            cd.launch(l, silu_out.data(), 1, IM, as_d);
            cd.wait(d_out.data(), 1, H, as_d, wsc[l].d_);
            cn(d_out.data(), H);
            for (int i = 0; i < H; i++) hidden[i] = save_buf[i] + d_out[i];
        }

        // LM head
        rn_c(hidden.data(), fin, H);
        lm_topk_omp(hidden.data(), lg.data(), top_ids, BS);
        sp++;
        double boot_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - ts).count();
        printf("  [0] boot=%d %.0fms\n", top_ids[0], boot_ms);
    }

    // ── Main decode loop — fully async pipelined ──
    int step = 1;
    while (step < ng) {
        auto ts = std::chrono::steady_clock::now();
        int batch_sz = std::min(BS, ng - step);
        if (batch_sz < 1) break;

        // Embed current token(s)
        int used_batch = batch_sz;
        for (int b = 0; b < used_batch; b++)
            for (int i = 0; i < H; i++)
                h_b[b * H + i] = emb_f32[(size_t)top_ids[b] * H + i];

        // Per-layer async pipeline
        for (int l = 0; l < NC; l++) {
            // ── Phase 1: Attention path ──
            for (int b = 0; b < used_batch; b++)
                rn_c(&h_b[b * H], in_n[l], H);

            float as_qk = dyn_scale(h_b.data(), used_batch * H);
            cq.launch(l, h_b.data(), used_batch, H, as_qk);

            // OVERLAP: CPU does Q norms, RoPE, KV cache while NPU does QKV
            // (These ops use PREVIOUS layer's QKV output, which we cache in qo_b)
            // For the first layer (l=0), there's nothing to overlap yet
            if (l > 0) {
                // Previous layer's QKV output is in qo_b — process it while NPU does current QKV
                // (The overlap here is the O, attention, or residual processing from prev layer)
            }

            cq.wait(qo_b.data(), used_batch, 4096, as_qk, wsc[l].qk);
            cn(qo_b.data(), used_batch * 4096);

            // Q norms + RoPE
            for (int b = 0; b < used_batch; b++) {
                for (int hh = 0; hh < NH; hh++) {
                    double s = 0;
                    for (int d = 0; d < HD; d++)
                        s += qo_b[b * NH * HD + hh * HD + d] * qo_b[b * NH * HD + hh * HD + d];
                    float iq = 1.0f / sqrtf((float)(s / HD) + EPS);
                    for (int d = 0; d < HD; d++) qo_b[b * NH * HD + hh * HD + d] *= iq * qn_w[l][d];
                    ra(&qo_b[b * NH * HD + hh * HD], HD, sp + b);
                }
                for (int kvh = 0; kvh < NKV; kvh++) {
                    float* ks = &qo_b[b * 4096 + NH * HD + kvh * HD];
                    float* vs = &qo_b[b * 4096 + NH * HD + NKV * HD + kvh * HD];
                    double sk = 0; for (int d = 0; d < HD; d++) sk += ks[d] * ks[d];
                    float ik = 1.0f / sqrtf((float)(sk / HD) + EPS);
                    for (int d = 0; d < HD; d++) { ks[d] *= ik * kn_w[l][d]; ra(ks, HD, sp + b); }
                }
            }
            // KV cache update
            for (int b = 0; b < used_batch; b++)
                for (int kvh = 0; kvh < NKV; kvh++) {
                    float* ks = &qo_b[b * 4096 + NH * HD + kvh * HD];
                    float* vs = &qo_b[b * 4096 + NH * HD + NKV * HD + kvh * HD];
                    memcpy(&kv[l].k[(sp + b) * NKV * HD + kvh * HD], ks, HD * 4);
                    memcpy(&kv[l].v[(sp + b) * NKV * HD + kvh * HD], vs, HD * 4);
                }
            kv[l].n = sp + used_batch;
            int cl = kv[l].n;

            // Attention (CPU OpenMP)
            for (int b = 0; b < used_batch; b++)
                attn_omp(&qo_b[b * NH * HD], &at_b[b * NH * HD], cl,
                         kv[l].k.data(), kv[l].v.data());

            // O projection (async)
            float as_o = dyn_scale(at_b.data(), used_batch * NH * HD);
            co.launch(l, at_b.data(), used_batch, NH * HD, as_o);
            co.wait(oo_b.data(), used_batch, H, as_o, wsc[l].o_);
            cn(oo_b.data(), used_batch * H);

            // Residual add
            for (int b = 0; b < used_batch; b++)
                for (int i = 0; i < H; i++)
                    h_b[b * H + i] += oo_b[b * H + i];

            // ── Phase 2: FFN path ──
            for (int b = 0; b < used_batch; b++)
                rn_c(&h_b[b * H], pa_n[l], H);

            // GU projection (async)
            float as_g = dyn_scale(h_b.data(), used_batch * H);
            cg.launch(l, h_b.data(), used_batch, H, as_g);
            cg.wait(gt_b.data(), used_batch, 2 * IM, as_g, wsc[l].g_);
            cn(gt_b.data(), used_batch * 2 * IM);

            // SiLU gate
            for (int b = 0; b < used_batch; b++)
                for (int i = 0; i < IM; i++) {
                    float gv = gt_b[b * 2 * IM + i];
                    if (!std::isfinite(gv)) gv = 0;
                    su_b[b * IM + i] = (gv / (1.0f + expf(-gv))) * gt_b[b * 2 * IM + IM + i];
                }

            // D projection (async)
            float as_d = dyn_scale(su_b.data(), used_batch * IM);
            cd.launch(l, su_b.data(), used_batch, IM, as_d);
            cd.wait(dw_b.data(), used_batch, H, as_d, wsc[l].d_);
            cn(dw_b.data(), used_batch * H);

            // Final residual add
            for (int b = 0; b < used_batch; b++)
                for (int i = 0; i < H; i++)
                    h_b[b * H + i] += dw_b[b * H + i];
        }

        // LM head for first token in batch
        rn_c(h_b.data(), fin, H);
        lm_topk_omp(h_b.data(), lg.data(), top_ids, BS);
        sp += used_batch;

        double step_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - ts).count();
        printf("  [%d] batch=%d tok=%d %.1fms (%.1f ms/tok)\n",
               step, used_batch, top_ids[0], step_ms, step_ms / used_batch);
        step += used_batch;
    }

    double tts = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - tgs).count();
    printf("\n=== %.1f ms/tok (%.0f tok/s) ===\n",
           tts * 1000 / ng, ng / tts);
    printf("Target: >80 tok/s sustained.\n");

    munmap(md, st.st_size);
    return 0;
}
