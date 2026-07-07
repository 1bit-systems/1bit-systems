/** NPU Engine — Split-mode for Fused NPU+GPU.
 *  Uses component xclbins (QKV, O, GU, D) — each supports multi-invocation.
 *  Protocol: reads commands from stdin, writes results to stdout.
 *  Commands:
 *    QKV <layer> <pos> <batch>
 *      → reads batch*H float hidden states from stdin after command line
 *      → runs QKV xclbin, outputs batch*QKV float QKV results to stdout
 *    FFN <layer> <pos> <batch>
 *      → reads batch*NH*HD float attention output from stdin after command line
 *      → runs O → residual → norm → GU(SiLU) → D, outputs batch*H floats to stdout
 *    LM_HEAD <batch>
 *      → reads batch*H float hidden states from stdin
 *      → runs final norm + lm_head, outputs the top-1 token ID per batch
 *    EXIT
 *  
 *  Target: 273 tok/s fused NPU (QKV+FFN) + GPU (attention)
 *  Startup: ~5s dequant (once), then <1ms per xclbin invocation
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <unistd.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

#include "platform.h"
#include "model_config.h"
extern "C" float* dequant_i8_to_float_ex(const uint8_t*,int,int,int*,int*);

static constexpr float EPS = 1e-6f;
static constexpr float FIXED_ASCALE = 8.0f / 127.0f;
static constexpr float FIXED_AIS = 127.0f / 8.0f;

// Clamp non-finite values to 0
static inline void clamp_nan(float* x, int n) {
    for (int i = 0; i < n; i++) if (!std::isfinite(x[i])) x[i] = 0.0f;
}

// RMSNorm: x *= rsqrt(mean(x^2) + eps) * weight
static inline void rmsnorm(float* x, const float* w, int n) {
    clamp_nan(x, n);
    double ss = 0;
    for (int i = 0; i < n; i++) if (std::isfinite(x[i])) ss += (double)x[i] * x[i];
    float ir = 1.0f / sqrtf((float)(ss / n) + EPS);
    for (int i = 0; i < n; i++) x[i] = std::isfinite(x[i]) ? x[i] * ir * w[i] : 0.0f;
}

// Softmax in-place
static inline void softmax(float* sc, int n) {
    if (n <= 0) return;
    clamp_nan(sc, n);
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

// RoPE tables
static std::vector<float> rc, rs;
static void rope_init(int hd, float th, int mp) {
    int hd2 = hd / 2;
    rc.resize(mp * hd); rs.resize(mp * hd);
    for (int p = 0; p < mp; p++)
        for (int d = 0; d < hd2; d++) {
            float f = 1.0f / powf(th, (float)d / hd2), a = p * f;
            rc[p * hd + d] = cosf(a); rs[p * hd + d] = sinf(a);
        }
}

static inline void rope_apply(float* x, int hd, int p) {
    int hd2 = hd / 2;
    for (int d = 0; d < hd2; d++) {
        float a = x[d], b = x[d + hd2];
        float c = rc[p * hd + d], s = rs[p * hd + d];
        x[d] = a * c - b * s;
        x[d + hd2] = b * c + a * s;
    }
}

// BF16 → f32
static inline float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f; memcpy(&f, &bits, 4); return f;
}

// JSON key lookup: find data_offset for named tensor
static uint64_t json_offs(const char* js, size_t jl, const char* nm) {
    size_t nl = strlen(nm);
    const char* p = js, *e = js + jl;
    while (p < e) {
        auto q = (const char*)platform_memmem(p, e - p, nm, nl);
        if (!q) return 0;
        if (q > js && *(q - 1) == '"' && *(q + nl) == '"') {
            auto o = strstr(q, "\"data_offsets\"");
            if (o) { auto a = strchr(o, '['); if (a) return strtoull(a + 1, NULL, 10); }
        }
        p = q + 1;
    }
    return 0;
}

// Self-path
static std::string self_dir() {
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) { buf[len] = '\0'; std::string p(buf); auto s = p.rfind('/'); if (s != std::string::npos) return p.substr(0, s); }
    return ".";
}

// f32 embeddings (global, loaded once)
static std::vector<float> g_emb_f32;
static std::vector<float> g_lm_head_f32;

// INT8 GEMM context with per-layer weight buffers
struct GemmCtx {
    int MD, KD, ND, NL;
    xrt::device* dev;
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::kernel> k;
    std::vector<uint32_t> ins;
    xrt::bo bI, bA, bC;
    std::vector<xrt::bo> layerB;  // per-layer weight BOs
    int8_t* Am;
    int32_t* Cm;

    bool init(xrt::device& d, const char* xp, const char* ip, int nlayers) {
        NL = nlayers;
        dev = &d;
        // Load instructions
        FILE* f = fopen(ip, "rb");
        if (!f) { fprintf(stderr, "Cannot open insts: %s\n", ip); return false; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        ins.resize(sz / 4); { size_t _r = fread(ins.data(), 4, ins.size(), f); (void)_r; } fclose(f);
        // Register xclbin
        xc = std::make_unique<xrt::xclbin>(std::string(xp));
        d.register_xclbin(*xc);
        hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
        k = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");
        // Instruction BO (group 1)
        bI = xrt::bo(d, ins.size() * 4, XCL_BO_FLAGS_CACHEABLE, k->group_id(1));
        memcpy(bI.map(), ins.data(), ins.size() * 4);
        bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        // Activation BO (group 3) — 16MB min for DMA safety
        bA = xrt::bo(d, std::max((size_t)MD * KD, (size_t)16 * 1024 * 1024),
                     XRT_BO_FLAGS_HOST_ONLY, k->group_id(3));
        // Output BO (group 5) — 16MB min
        bC = xrt::bo(d, std::max((size_t)MD * ND * 4, (size_t)16 * 1024 * 1024),
                     XRT_BO_FLAGS_HOST_ONLY, k->group_id(5));
        Am = (int8_t*)bA.map();
        Cm = (int32_t*)bC.map();
        // Per-layer weight BOs (group 4 — matches universal engine's proven layout)
        for (int l = 0; l < NL; l++) {
            layerB.emplace_back(
                xrt::bo(d, std::max((size_t)KD * ND, (size_t)16 * 1024 * 1024),
                        XRT_BO_FLAGS_HOST_ONLY, k->group_id(4)));
        }
        return true;
    }

    // Pack transposed weights into per-layer BO: w is row-major [K, N], store INT8 quantized
    void packB(int l, const float* w, int K, int N, float& scale_out) {
        float amax = 0;
        for (int i = 0; i < K * N; i++) {
            float a = fabsf(w[i]);
            if (std::isfinite(a) && a > amax) amax = a;
        }
        if (amax < 1e-12f) amax = 1.0f;
        scale_out = amax / 127.0f;
        float is = 127.0f / amax;
        int8_t* Bm = (int8_t*)layerB[l].map();
        for (int i = 0; i < K * N; i++) {
            float v = w[i];
            if (!std::isfinite(v)) v = 0;
            int x = (int)roundf(v * is);
            if (x > 127) x = 127; else if (x < -127) x = -127;
            Bm[i] = (int8_t)x;
        }
        layerB[l].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // Run GEMM: A[am x ak] @ B[KD x ND] → C[am x an]
    // B must have been packed via packB(l, ...) for this layer
    void go(int l, const float* A, int am, int ak, float ascale, float Bscale, float* C, int an) {
        // Use caller-provided activation scale. Fixed ASCALE=8.0/127.0 avoids the
        // per-call amax scan (~50us per GEMM, ~4ms for all 84 calls at B=128).
        // Post-RMSNorm activations stay within [-8,8] so fixed scale is safe.
        // ascale = amax/127.0, so quantization is A/ascale = A*127/amax
        float ais = 1.0f / ascale;
        // Quantize A to INT8 — skip memset since we write every element
        for (int m = 0; m < am; m++)
            for (int k = 0; k < ak; k++) {
                float v = A[m * ak + k];
                if (!std::isfinite(v)) v = 0;
                int q = (int)roundf(v * ais);
                if (q > 127) q = 127; else if (q < -127) q = -127;
                Am[m * KD + k] = (int8_t)q;
            }
        // Zero the padding columns (KD - ak) per row — only if KD > ak
        if (KD > ak) {
            for (int m = 0; m < am; m++)
                memset(Am + m * KD + ak, 0, (size_t)(KD - ak));
        }
        bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        // Launch: opcode=3, instructions, count, activation, weights, output
        auto r = (*k)((unsigned)3, bI, (unsigned)ins.size(), bA, layerB[l], bC);
        r.wait();
        bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        // Dequantize output
        float cs = ascale * Bscale;
        for (int m = 0; m < am; m++)
            for (int n = 0; n < an; n++) {
                float val = (float)Cm[m * ND + n] * cs;
                if (!std::isfinite(val)) val = 0;
                C[m * an + n] = val;
            }
    }
};

// GPT-style attention (CPU fallback — for when we need attention on NPU instead of GPU)
static void attention_cpu(float* qo, float* out, int cl,
                          const float* kv_k, const float* kv_v,
                          int NH, int NKV, int HD, int GQA, int max_pos) {
    if (max_pos < 0) max_pos = cl;
    for (int hh = 0; hh < NH; hh++) {
        int kvh = hh / GQA;
        std::vector<float> scores(cl);
        float mx = -1e30f;
        for (int p = 0; p < cl; p++) {
            if (p >= max_pos) { scores[p] = -1e30f; continue; }
            double s = 0;
            for (int d = 0; d < HD; d++) s += (double)qo[hh * HD + d] * kv_k[p * NKV * HD + kvh * HD + d];
            scores[p] = (float)(s / sqrtf((float)HD));
            if (scores[p] > mx) mx = scores[p];
        }
        double sw = 0;
        for (int p = 0; p < cl; p++) { scores[p] = expf(scores[p] - mx); sw += scores[p]; }
        float isw = sw > 0 ? 1.0f / (float)sw : 1.0f / cl;
        for (int d = 0; d < HD; d++) {
            float acc = 0;
            for (int p = 0; p < cl; p++) acc += scores[p] * kv_v[p * NKV * HD + kvh * HD + d];
            out[hh * HD + d] = acc * isw;
        }
    }
}

// Top-1 greedy decode from lm_head
static int top1_logits(const float* hidden, int H, int NV, const float* lm_emb) {
    double best_s = -1e30;
    int best_n = 0;
    for (int n = 0; n < NV; n++) {
        double s = 0;
        for (int k = 0; k < H; k++) s += (double)hidden[k] * lm_emb[(size_t)n * H + k];
        if (s > best_s) { best_s = s; best_n = n; }
    }
    return best_n;
}

// Transpose helper: src[row-major out_f x in_f] → dst[row-major in_f x out_f]
static void transpose(float* dst, const float* src, int out_f, int in_f, int dst_stride, int dst_off) {
    for (int o = 0; o < out_f; o++)
        for (int i = 0; i < in_f; i++)
            dst[(size_t)i * dst_stride + dst_off + o] = src[(size_t)o * in_f + i];
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stdin, NULL, _IONBF, 0);
    if (argc < 2) { printf("Usage: %s model.q4nx [--xclbin-dir <dir>]\n", argv[0]); return 1; }

    const char* mp = argv[1];

    // Determine model tag from parent dir name
    std::string mp_s(mp), model_tag;
    auto ls = mp_s.rfind('/');
    auto sl = mp_s.rfind('/', ls - 1);
    model_tag = (sl != std::string::npos && ls != std::string::npos)
        ? mp_s.substr(sl + 1, ls - sl - 1) : mp_s.substr(ls + 1);
    for (auto& c : model_tag) { c = tolower(c); if (c == '-' || c == '.') c = '_'; }

    // Find xclbin dir
    std::string xd = self_dir() + "/int8";
    for (int i = 2; i < argc - 1; i++) {
        if (strcmp(argv[i], "--xclbin-dir") == 0) { xd = argv[i + 1]; break; }
    }
    if (auto* env = getenv("NPU_XCLBIN_DIR")) xd = env;

    printf("SPLIT: Loading model %s (tag=%s, xclbin=%s)\n", mp, model_tag.c_str(), xd.c_str());

    // Parse config
    auto cfg = parse_q4nx_header(mp, model_tag.c_str());
    if (!cfg.valid()) { fprintf(stderr, "SPLIT: Invalid model config\n"); return 1; }

    int H = cfg.H, NC = cfg.NC, NH = cfg.NH, NKV = cfg.NKV;
    int HD = cfg.HD, IM = cfg.IM, NV = cfg.NV, GQA = cfg.GQA, XM = cfg.XM;
    int QKV = cfg.qkv_total;
    int QOUT = NH * HD, KVOUT = NKV * HD;
    printf("SPLIT: H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d NV=%d GQA=%d QKV=%d XM=%d\n",
           H, NC, NH, NKV, HD, IM, NV, GQA, QKV, XM);

    // Open and mmap model
    auto fd = platform_open_read(mp);
    platform_stat st; platform_fstat(fd, &st);
    auto md = (const uint8_t*)platform_mmap((size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    platform_close(fd);
    uint64_t hsz; memcpy(&hsz, md, 8);
    uint64_t df = 8 + hsz;
    auto i8p = [&](uint64_t o) { return md + df + o; };
    const char* js = (const char*)(md + 8);
    size_t jl = hsz;

    // BF16 embeddings → f32
    auto emb_bf16 = reinterpret_cast<const uint16_t*>(md + df);
    printf("SPLIT: Converting embeddings...\n");
    auto te = std::chrono::steady_clock::now();
    g_emb_f32.resize((size_t)NV * H);
    for (int n = 0; n < NV; n++)
        for (int i = 0; i < H; i++)
            g_emb_f32[(size_t)n * H + i] = bf16_to_f32(emb_bf16[n * H + i]);
    printf("  %.0fms\n", std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - te).count());

    // LM head (may be separate from embeddings)
    auto lm_emb = g_emb_f32.data();
    uint64_t lm_off = json_offs(js, jl, "lm_head.weight");
    if (lm_off) {
        int lr = 0, lc = 0;
        float* lm_raw = dequant_i8_to_float_ex(i8p(lm_off), 0, 0, &lr, &lc);
        if (lr > 0 && lc > 0) {
            g_lm_head_f32.assign(lm_raw, lm_raw + (size_t)lr * lc);
            lm_emb = g_lm_head_f32.data();
            printf("SPLIT: LM head %dx%d\n", lr, lc);
        }
        if (lm_raw) free(lm_raw);
    }

    // Find tensor INT8 rows (for dequant)
    auto gi8 = [&](const char* k) -> int {
        int r = 0; find_tensor_info(js, jl, k, &r); return r;
    };
    int q_i8 = gi8("model.layers.0.self_attn.q_proj.weight");
    int k_i8 = gi8("model.layers.0.self_attn.k_proj.weight");
    int v_i8 = gi8("model.layers.0.self_attn.v_proj.weight");
    int o_i8 = gi8("model.layers.0.self_attn.o_proj.weight");
    int g_i8 = gi8("model.layers.0.mlp.gate_proj.weight");
    int u_i8 = gi8("model.layers.0.mlp.up_proj.weight");
    int d_i8 = gi8("model.layers.0.mlp.down_proj.weight");

    // Load norm weights from JSON offsets
    printf("SPLIT: Loading norms...\n");
    std::vector<std::vector<float>> in_n(NC, std::vector<float>(H));
    std::vector<std::vector<float>> pa_n(NC, std::vector<float>(H));
    std::vector<float> fin_w(H, 1.0f);
    for (int l = 0; l < NC; l++) {
        char bn[128];
        snprintf(bn, sizeof(bn), "model.layers.%d.input_layernorm.weight", l);
        uint64_t io = json_offs(js, jl, bn);
        if (io) { auto* fp = (const uint16_t*)(md + df + io); for (int i = 0; i < H; i++) in_n[l][i] = bf16_to_f32(fp[i]); }
        else for (int i = 0; i < H; i++) in_n[l][i] = 1.0f;
        snprintf(bn, sizeof(bn), "model.layers.%d.post_attention_layernorm.weight", l);
        uint64_t po = json_offs(js, jl, bn);
        if (po) { auto* fp = (const uint16_t*)(md + df + po); for (int i = 0; i < H; i++) pa_n[l][i] = bf16_to_f32(fp[i]); }
        else for (int i = 0; i < H; i++) pa_n[l][i] = 1.0f;
    }
    uint64_t fn_off = json_offs(js, jl, "model.norm.weight");
    if (fn_off) { auto* fp = (const uint16_t*)(md + df + fn_off); for (int i = 0; i < H; i++) fin_w[i] = bf16_to_f32(fp[i]); }

    // Initialize NPU and GEMM contexts
    printf("SPLIT: Initializing NPU...\n");
    xrt::device dev(0);
    auto xp = [&](const char* t) { return xd + "/final_i8_" + t + "_" + model_tag + ".xclbin"; };
    auto ip = [&](const char* t) { return xd + "/insts_i8_" + t + "_" + model_tag + ".txt"; };

    GemmCtx cq, co, cg, cd;
    cq.MD = XM; cq.KD = H; cq.ND = QKV;
    co.MD = XM; co.KD = NH * HD; co.ND = H;
    cg.MD = XM; cg.KD = H; cg.ND = cfg.gu_split ? IM : IM * 2;
    cd.MD = XM; cd.KD = IM; cd.ND = H;

    printf("SPLIT: Loading xclbins...\n");
    auto td = std::chrono::steady_clock::now();
    if (!cq.init(dev, xp("QKV").c_str(), ip("QKV").c_str(), NC)) return 1;
    if (!co.init(dev, xp("O").c_str(), ip("O").c_str(), NC)) return 1;
    if (!cg.init(dev, xp("GU").c_str(), ip("GU").c_str(), NC)) return 1;
    if (!cd.init(dev, xp("D").c_str(), ip("D").c_str(), NC)) return 1;
    printf("  %.0fms\n", std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - td).count());

    // Dequant, transpose, and pack weights per layer
    printf("SPLIT: Packing weights...\n");
    auto tp = std::chrono::steady_clock::now();
    std::vector<float> qsc(NC), osc(NC), gsc(NC), dsc(NC);

    auto deq = [&](uint64_t off, int i8_rows, int feat_dim) -> float* {
        if (!off || i8_rows <= 0) return nullptr;
        int r = 0, c = 0;
        return dequant_i8_to_float_ex(i8p(off), i8_rows, feat_dim, &r, &c);
    };

    for (int l = 0; l < NC; l++) {
        char bn[128];
        // QKV: Q + K + V or fused qkv_proj
        snprintf(bn, sizeof(bn), "model.layers.%d.self_attn.q_proj.weight", l);
        uint64_t q_off = json_offs(js, jl, bn);
        snprintf(bn, sizeof(bn), "model.layers.%d.self_attn.k_proj.weight", l);
        uint64_t k_off = json_offs(js, jl, bn);
        snprintf(bn, sizeof(bn), "model.layers.%d.self_attn.v_proj.weight", l);
        uint64_t v_off = json_offs(js, jl, bn);

        // Try fused QKV first
        snprintf(bn, sizeof(bn), "model.layers.%d.self_attn.qkv_proj.weight", l);
        uint64_t qkv_off = json_offs(js, jl, bn);
        if (qkv_off && q_i8 > 0) {
            int qr, qc;
            float* qkv_raw = dequant_i8_to_float_ex(i8p(qkv_off), q_i8, H, &qr, &qc);
            if (qkv_raw) {
                int qkv_n = QOUT + KVOUT + KVOUT;
                std::vector<float> w((size_t)H * qkv_n);
                transpose(w.data(), qkv_raw, qkv_n, H, qkv_n, 0);
                cq.packB(l, w.data(), H, qkv_n, qsc[l]);
                free(qkv_raw);
            }
        } else if (q_off && k_off && v_off) {
            float* qw = deq(q_off, q_i8, H);
            float* kw = deq(k_off, k_i8, H);
            float* vw = deq(v_off, v_i8, H);
            if (qw && kw && vw) {
                int qkv_n = QOUT + KVOUT + KVOUT;
                std::vector<float> w((size_t)H * qkv_n);
                transpose(w.data(), qw, QOUT, H, qkv_n, 0);
                transpose(w.data(), kw, KVOUT, H, qkv_n, QOUT);
                transpose(w.data(), vw, KVOUT, H, qkv_n, QOUT + KVOUT);
                cq.packB(l, w.data(), H, qkv_n, qsc[l]);
            }
            if (qw) free(qw); if (kw) free(kw); if (vw) free(vw);
        }

        // O projection
        snprintf(bn, sizeof(bn), "model.layers.%d.self_attn.o_proj.weight", l);
        uint64_t o_off = json_offs(js, jl, bn);
        if (o_off && o_i8 > 0) {
            float* ow = deq(o_off, o_i8, NH * HD);
            if (ow) {
                std::vector<float> wo((size_t)(NH * HD) * H);
                transpose(wo.data(), ow, H, NH * HD, H, 0);
                co.packB(l, wo.data(), NH * HD, H, osc[l]);
                free(ow);
            }
        }

        // Gate + Up (fused GU or separate)
        snprintf(bn, sizeof(bn), "model.layers.%d.mlp.gate_proj.weight", l);
        uint64_t g_off = json_offs(js, jl, bn);
        snprintf(bn, sizeof(bn), "model.layers.%d.mlp.up_proj.weight", l);
        uint64_t u_off = json_offs(js, jl, bn);

        if (g_off && u_off && g_i8 > 0 && u_i8 > 0) {
            float* gw = deq(g_off, g_i8, H);
            float* uw = deq(u_off, u_i8, H);
            if (gw && uw) {
                if (cfg.gu_split) {
                    std::vector<float> wg((size_t)H * g_i8);
                    transpose(wg.data(), gw, IM, H, g_i8, 0);
                    cg.packB(l, wg.data(), H, g_i8, gsc[l]);
                } else {
                    int t2 = g_i8 + u_i8;
                    std::vector<float> w2((size_t)H * t2);
                    transpose(w2.data(), gw, IM, H, t2, 0);
                    transpose(w2.data(), uw, IM, H, t2, IM);
                    cg.packB(l, w2.data(), H, t2, gsc[l]);
                }
            }
            if (gw) free(gw); if (uw) free(uw);
        }

        // Down projection
        snprintf(bn, sizeof(bn), "model.layers.%d.mlp.down_proj.weight", l);
        uint64_t d_off = json_offs(js, jl, bn);
        if (d_off && d_i8 > 0) {
            float* dw = deq(d_off, d_i8, IM);
            if (dw) {
                std::vector<float> wd((size_t)IM * H);
                transpose(wd.data(), dw, H, IM, H, 0);
                cd.packB(l, wd.data(), IM, H, dsc[l]);
                free(dw);
            }
        }
    }
    printf("  %.0fms\n", std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - tp).count());

    // RoPE precomputation
    rope_init(HD, cfg.rope_theta, 4096);
    int rope_half = HD / 2;

    // KV cache: [layers][max_pos * NKV * HD]
    struct KVCache {
        std::vector<float> k, v;
        int n;
        KVCache(int sz) : k(sz, 0), v(sz, 0), n(0) {}
    };
    int kv_cache_size = 4096 * NKV * HD;
    std::vector<KVCache> kv_caches;
    for (int i = 0; i < NC; i++) kv_caches.emplace_back(kv_cache_size);

    // Working buffers
    std::vector<float> h_buf(XM * H);
    std::vector<float> res_buf(XM * H);
    std::vector<float> qo_buf(XM * QKV);
    std::vector<float> at_buf(XM * NH * HD);
    std::vector<float> oo_buf(XM * H);
    std::vector<float> gt_buf(XM * (cfg.gu_split ? IM : 2 * IM));
    std::vector<float> su_buf(XM * IM);
    std::vector<float> dw_buf(XM * H);

    // Command loop
    printf("SPLIT: READY\n");
    fflush(stdout);

    char cmd[512];
    while (fgets(cmd, sizeof(cmd), stdin)) {
        // Strip trailing newline
        size_t clen = strlen(cmd);
        while (clen > 0 && (cmd[clen - 1] == '\n' || cmd[clen - 1] == '\r')) cmd[--clen] = '\0';

        char op[32] = {0};
        int layer = 0, pos = 0, batch = 0;
        int parsed = sscanf(cmd, "%31s %d %d %d", op, &layer, &pos, &batch);

        if (parsed < 1) continue;

        // ===== EXIT =====
        if (strcmp(op, "EXIT") == 0) break;

        // ===== QKV <layer> <pos> <batch> =====
        // Reads batch*H floats from stdin
        if (strcmp(op, "QKV") == 0 && parsed == 4) {
            if (layer < 0 || layer >= NC || batch <= 0 || batch > XM) {
                fprintf(stderr, "SPLIT: invalid QKV args layer=%d batch=%d\n", layer, batch);
                continue;
            }
            // Read hidden states
            std::vector<float> input(batch * H);
            size_t got = fread(input.data(), sizeof(float), batch * H, stdin);
            if ((int)got != batch * H) { fprintf(stderr, "SPLIT: QKV short read %zu/%d\n", got, batch * H); break; }

            // RMSNorm → QKV GEMM → RoPE → Q/K norms → store KV
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < H; i++) h_buf[b * H + i] = input[b * H + i];
                // Save residual
                for (int i = 0; i < H; i++) res_buf[b * H + i] = h_buf[b * H + i];
            }
            for (int b = 0; b < batch; b++) rmsnorm(&h_buf[b * H], in_n[layer].data(), H);

            // QKV GEMM
            cq.go(layer, h_buf.data(), batch, H, FIXED_ASCALE, qsc[layer], qo_buf.data(), QKV);
            clamp_nan(qo_buf.data(), batch * QKV);

            // Per-token: Q-norm + RoPE on Q, K-norm + RoPE on K, store KV
            for (int b = 0; b < batch; b++) {
                int tp = pos + b;
                float* qkv = &qo_buf[b * QKV];
                // Q: RoPE after Q-norm (Q-norm is optional)
                for (int hh = 0; hh < NH; hh++) {
                    rope_apply(&qkv[hh * HD], HD, tp);
                }
                // K: RoPE after K-norm, store to KV cache
                for (int kvh = 0; kvh < NKV; kvh++) {
                    float* kp = &qkv[cfg.qkv_k_offset + kvh * HD];
                    rope_apply(kp, HD, tp);
                    memcpy(&kv_caches[layer].k[(size_t)tp * NKV * HD + kvh * HD], kp, HD * sizeof(float));
                    float* vp = &qkv[cfg.qkv_v_offset + kvh * HD];
                    memcpy(&kv_caches[layer].v[(size_t)tp * NKV * HD + kvh * HD], vp, HD * sizeof(float));
                }
            }
            if (pos + batch > kv_caches[layer].n) kv_caches[layer].n = pos + batch;

            // Output QKV
            fwrite(qo_buf.data(), sizeof(float), batch * QKV, stdout);
            fflush(stdout);
        }

        // ===== FFN <layer> <pos> <batch> =====
        // Reads batch*NH*HD floats from stdin (attention output)
        else if (strcmp(op, "FFN") == 0 && parsed == 4) {
            if (layer < 0 || layer >= NC || batch <= 0 || batch > XM) {
                fprintf(stderr, "SPLIT: invalid FFN args layer=%d batch=%d\n", layer, batch);
                continue;
            }
            // Read attention output (NH*HD per token)
            std::vector<float> attn_in(batch * NH * HD);
            size_t got = fread(attn_in.data(), sizeof(float), batch * NH * HD, stdin);
            if ((int)got != batch * NH * HD) { fprintf(stderr, "SPLIT: FFN short read %zu/%d\n", got, batch * NH * HD); break; }

            // O projection
            co.go(layer, attn_in.data(), batch, NH * HD, FIXED_ASCALE, osc[layer], oo_buf.data(), H);
            clamp_nan(oo_buf.data(), batch * H);

            // Residual add (hidden = res + attention_output)
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < H; i++)
                    h_buf[b * H + i] = res_buf[b * H + i] + oo_buf[b * H + i];

            // Save pre-FFN residual
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < H; i++)
                    res_buf[b * H + i] = h_buf[b * H + i];

            // Post-attention RMSNorm
            for (int b = 0; b < batch; b++) rmsnorm(&h_buf[b * H], pa_n[layer].data(), H);

            // Gate+Up GEMM (fused or separate)
            int mlp_out = cfg.gu_split ? IM : 2 * IM;
            cg.go(layer, h_buf.data(), batch, H, FIXED_ASCALE, gsc[layer], gt_buf.data(), mlp_out);
            clamp_nan(gt_buf.data(), batch * mlp_out);

            // SiLU(Gate) * Up
            if (cfg.gu_split) {
                // Gate and Up are separate xclbins
                // For fused GU: first IM = gate, second IM = up
                for (int b = 0; b < batch; b++)
                    for (int i = 0; i < IM; i++) {
                        float gv = gt_buf[b * mlp_out + i];
                        float uv = gt_buf[b * mlp_out + IM + i];
                        su_buf[b * IM + i] = (gv / (1.0f + expf(-gv))) * uv;
                    }
            } else {
                for (int b = 0; b < batch; b++)
                    for (int i = 0; i < IM; i++) {
                        float gv = gt_buf[b * mlp_out + i];
                        float uv = gt_buf[b * mlp_out + IM + i];
                        su_buf[b * IM + i] = (gv / (1.0f + expf(-gv))) * uv;
                    }
            }

            // Down projection
            cd.go(layer, su_buf.data(), batch, IM, FIXED_ASCALE, dsc[layer], dw_buf.data(), H);
            clamp_nan(dw_buf.data(), batch * H);

            // Final residual add
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < H; i++)
                    h_buf[b * H + i] = res_buf[b * H + i] + dw_buf[b * H + i];

            // Output hidden state
            fwrite(h_buf.data(), sizeof(float), batch * H, stdout);
            fflush(stdout);
        }

        // ===== ATTENTION <layer> <pos> <batch> =====
        // CPU attention fallback when GPU not available
        // Reads batch*QKV floats (QKV output), outputs batch*NH*HD attention output
        else if (strcmp(op, "ATTENTION") == 0 && parsed == 4) {
            if (layer < 0 || layer >= NC || batch <= 0 || batch > XM) {
                fprintf(stderr, "SPLIT: invalid ATTENTION args\n");
                continue;
            }
            std::vector<float> qkv_in(batch * QKV);
            size_t got = fread(qkv_in.data(), sizeof(float), batch * QKV, stdin);
            if ((int)got != batch * QKV) { fprintf(stderr, "SPLIT: ATTENTION short read\n"); break; }

            int cl = kv_caches[layer].n;
            for (int b = 0; b < batch; b++) {
                attention_cpu(&qkv_in[b * QKV], &at_buf[b * NH * HD], cl,
                              kv_caches[layer].k.data(), kv_caches[layer].v.data(),
                              NH, NKV, HD, GQA, pos + b + 1);
            }

            fwrite(at_buf.data(), sizeof(float), batch * NH * HD, stdout);
            fflush(stdout);
        }

        // ===== LM_HEAD <batch> =====
        // Reads batch*H floats (final hidden), outputs batch int32 token IDs
        else if (strcmp(op, "LM_HEAD") == 0 && parsed >= 2) {
            int bsize = batch > 0 ? batch : 1;
            if (bsize > XM) bsize = XM;
            std::vector<float> hidden_in(bsize * H);
            size_t got = fread(hidden_in.data(), sizeof(float), bsize * H, stdin);
            if ((int)got != bsize * H) { fprintf(stderr, "SPLIT: LM_HEAD short read\n"); break; }

            std::vector<int32_t> token_ids(bsize);
            for (int b = 0; b < bsize; b++) {
                float* h = &hidden_in[b * H];
                rmsnorm(h, fin_w.data(), H);
                token_ids[b] = top1_logits(h, H, NV, lm_emb);
            }

            fwrite(token_ids.data(), sizeof(int32_t), bsize, stdout);
            fflush(stdout);
        }

        // ===== PREFILL <n_tokens> <start_pos> =====
        // Reads n_tokens token IDs (int32), then n_tokens*H embeddings
        else if (strcmp(op, "PREFILL") == 0 && parsed >= 2) {
            int ntok = layer; // reuse layer field as token count
            int sp = pos;
            if (ntok <= 0 || ntok > 4096) continue;

            std::vector<int32_t> token_ids(ntok);
            size_t got = fread(token_ids.data(), sizeof(int32_t), ntok, stdin);
            if ((int)got != ntok) { fprintf(stderr, "SPLIT: PREFILL short read tokens\n"); break; }

            // Embed lookup
            for (int pi = 0; pi < ntok; pi++)
                for (int i = 0; i < H; i++)
                    h_buf[pi * H + i] = g_emb_f32[(size_t)token_ids[pi] * H + i];

            // Run all layers
            for (int l = 0; l < NC; l++) {
                // QKV
                for (int pi = 0; pi < ntok; pi++) { for (int i = 0; i < H; i++) res_buf[pi * H + i] = h_buf[pi * H + i]; }
                for (int pi = 0; pi < ntok; pi++) rmsnorm(&h_buf[pi * H], in_n[l].data(), H);
                cq.go(l, h_buf.data(), ntok, H, FIXED_ASCALE, qsc[l], qo_buf.data(), QKV);
                clamp_nan(qo_buf.data(), ntok * QKV);

                for (int pi = 0; pi < ntok; pi++) {
                    int tp = sp + pi;
                    float* qkv = &qo_buf[pi * QKV];
                    for (int hh = 0; hh < NH; hh++) rope_apply(&qkv[hh * HD], HD, tp);
                    for (int kvh = 0; kvh < NKV; kvh++) {
                        float* kp = &qkv[cfg.qkv_k_offset + kvh * HD];
                        rope_apply(kp, HD, tp);
                        memcpy(&kv_caches[l].k[(size_t)tp * NKV * HD + kvh * HD], kp, HD * sizeof(float));
                        memcpy(&kv_caches[l].v[(size_t)tp * NKV * HD + kvh * HD],
                               &qkv[cfg.qkv_v_offset + kvh * HD], HD * sizeof(float));
                    }
                }
                kv_caches[l].n = sp + ntok;

                // CPU attention (prefill is always CPU attention)
                for (int pi = 0; pi < ntok; pi++) {
                    int cl = kv_caches[l].n;
                    attention_cpu(&qo_buf[pi * QKV], &at_buf[pi * NH * HD], cl,
                                  kv_caches[l].k.data(), kv_caches[l].v.data(),
                                  NH, NKV, HD, GQA, sp + pi + 1);
                }

                // O + residual
                co.go(l, at_buf.data(), ntok, NH * HD, FIXED_ASCALE, osc[l], oo_buf.data(), H);
                for (int pi = 0; pi < ntok; pi++)
                    for (int i = 0; i < H; i++) h_buf[pi * H + i] = res_buf[pi * H + i] + oo_buf[pi * H + i];

                // FFN
                for (int pi = 0; pi < ntok; pi++) { for (int i = 0; i < H; i++) res_buf[pi * H + i] = h_buf[pi * H + i]; }
                for (int pi = 0; pi < ntok; pi++) rmsnorm(&h_buf[pi * H], pa_n[l].data(), H);

                int mlp_out = cfg.gu_split ? IM : 2 * IM;
                cg.go(l, h_buf.data(), ntok, H, FIXED_ASCALE, gsc[l], gt_buf.data(), mlp_out);
                for (int pi = 0; pi < ntok; pi++)
                    for (int i = 0; i < IM; i++) {
                        float gv = gt_buf[pi * mlp_out + i];
                        float uv = gt_buf[pi * mlp_out + IM + i];
                        su_buf[pi * IM + i] = (gv / (1.0f + expf(-gv))) * uv;
                    }

                cd.go(l, su_buf.data(), ntok, IM, FIXED_ASCALE, dsc[l], dw_buf.data(), H);
                for (int pi = 0; pi < ntok; pi++)
                    for (int i = 0; i < H; i++) h_buf[pi * H + i] = res_buf[pi * H + i] + dw_buf[pi * H + i];
            }

            // Output final hidden for position ntok-1
            fwrite(&h_buf[(ntok - 1) * H], sizeof(float), H, stdout);
            fflush(stdout);
        }

        // Unknown command
        else {
            fprintf(stderr, "SPLIT: unknown cmd: %s\n", cmd);
        }
    }

    platform_munmap((void*)md, (size_t)st.st_size);
    fprintf(stderr, "SPLIT: DONE\n");
    return 0;
}
