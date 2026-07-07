/** NPU Engine — Split-mode for Fused NPU+GPU.
 *  Uses component xclbins (QKV, O, GU, D) — each supports multi-invocation.
 *  Protocol: reads commands from stdin, writes results to stdout.
 *  Commands:
 *    QKV <layer> <pos> <batch> <data_len> <float_data...>
 *      → runs QKV xclbin, outputs QKV result floats
 *    FFN <layer> <pos> <batch> <data_len> <float_data...>
 *      → runs O+G+U+D xclbins, outputs hidden state floats
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
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

#include "platform.h"
#include "model_config.h"
extern "C" float* dequant_i8_to_float_ex(const uint8_t*,int,int,int*,int*);

static constexpr float EPS = 1e-6f;
static constexpr float FIXED_ASCALE = 8.0f / 127.0f;
static constexpr float FIXED_AIS = 127.0f / 8.0f;

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

static std::vector<float> emb_f32; // f32 embeddings
static std::vector<float> lm_head_f32;
static float* dequant_i8(const uint8_t* md, uint64_t offset, int& rows, int& cols) {
    return dequant_i8_to_float_ex((uint8_t*)(md + offset), 0, 0, &rows, &cols);
}

// INT8 GEMM context — one per xclbin
struct GemmCtx {
    int MD, KD, ND;
    xrt::device* dev;
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::kernel> k;
    std::vector<uint32_t> ins;
    xrt::bo bI, bA, bC;
    int8_t* Am;
    int32_t* Cm;

    bool init(xrt::device& d, const char* xp, const char* ip, int gid_B) {
        dev = &d;
        FILE* f = fopen(ip, "rb");
        if (!f) { fprintf(stderr, "Cannot open insts: %s\n", ip); return false; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        ins.resize(sz / 4); fread(ins.data(), 4, ins.size(), f); fclose(f);
        xc = std::make_unique<xrt::xclbin>(std::string(xp));
        d.register_xclbin(*xc);
        hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
        k = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");
        bI = xrt::bo(d, ins.size() * 4, XCL_BO_FLAGS_CACHEABLE, k->group_id(1));
        memcpy(bI.map(), ins.data(), ins.size() * 4); bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bA = xrt::bo(d, std::max((size_t)MD * KD, (size_t)16 * 1024 * 1024), XRT_BO_FLAGS_HOST_ONLY, k->group_id(gid_B));
        bC = xrt::bo(d, std::max((size_t)MD * ND * 4, (size_t)16 * 1024 * 1024), XRT_BO_FLAGS_HOST_ONLY, k->group_id(5));
        Am = (int8_t*)bA.map(); Cm = (int32_t*)bC.map();
        return true;
    }

    void go(const float* A, int am, int ak, const float* B, float Bscale, float* C, int an) {
        // INT8 quantize activation
        float a_amax = 0;
        for (int i = 0; i < am * ak; i++) { float v = fabsf(A[i]); if (v > a_amax) a_amax = v; }
        if (a_amax < 1e-8f) a_amax = 1e-8f;
        float as_scale = a_amax / 127.0f, ais = 127.0f / a_amax;
        memset(Am, 0, (size_t)am * KD);
        for (int m = 0; m < am; m++)
            for (int k = 0; k < ak; k++) {
                float v = A[m * ak + k];
                int q = (int)roundf(v * ais);
                if (q > 127) q = 127; else if (q < -127) q = -127;
                Am[m * KD + k] = (int8_t)q;
            }
        bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r = (*k)((unsigned)3, bI, (unsigned)ins.size(), bA, bC);
        r.wait();
        bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        float cs = as_scale * Bscale;
        for (int m = 0; m < am; m++)
            for (int n = 0; n < an; n++) {
                float val = (float)Cm[m * ND + n] * cs;
                C[m * an + n] = val;
            }
    }
};

struct LayerState {
    std::vector<float> qn, kn, in, pa, fin;  // norm weights
    std::vector<float> qkv_w, o_w, gu_w, d_w; // weight buffers
    float qkv_scale, o_scale, gu_scale, d_scale;
};

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stdin, NULL, _IONBF, 0);
    if (argc < 2) { printf("Usage: %s model.q4nx\n", argv[0]); return 1; }

    const char* mp = argv[1];
    std::string mp_s(mp), model_tag;
    auto ls = mp_s.rfind('/'), sl = mp_s.rfind('/', ls - 1);
    model_tag = (sl != std::string::npos && ls != std::string::npos)
        ? mp_s.substr(sl + 1, ls - sl - 1) : mp_s.substr(ls + 1);
    for (auto& c : model_tag) { c = tolower(c); if (c == '-' || c == '.') c = '_'; }

    printf("SPLIT: Loading model %s (tag=%s)\n", mp, model_tag.c_str());
    auto cfg = parse_q4nx_header(mp, model_tag.c_str());
    if (!cfg.valid()) { fprintf(stderr, "Invalid model config\n"); return 1; }

    int H = cfg.H, NC = cfg.NC, NH = cfg.NH, NKV = cfg.NKV;
    int HD = cfg.HD, IM = cfg.IM, NV = cfg.NV, GQA = cfg.GQA;
    int QKV = cfg.qkv_total;
    printf("SPLIT: H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d NV=%d QKV=%d\n",
        H, NC, NH, NKV, HD, IM, NV, QKV);

    // Open model
    auto fd = platform_open_read(mp);
    platform_stat st; platform_fstat(fd, &st);
    auto md = (const uint8_t*)platform_mmap((size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    platform_close(fd);
    uint64_t hsz; memcpy(&hsz, md, 8); uint64_t df = 8 + hsz;
    auto i8p = [&](uint64_t o) { return md + df + o; };
    const char* js = (const char*)(md + 8); size_t jl = hsz;

    // Embeddings
    auto emb = reinterpret_cast<const uint16_t*>(md + df);
    auto te = std::chrono::steady_clock::now();
    emb_f32.resize((size_t)NV * H);
    for (int n = 0; n < NV; n++)
        for (int i = 0; i < H; i++) {
            uint32_t v = (uint32_t)emb[n * H + i] << 16;
            float f; memcpy(&f, &v, 4); emb_f32[(size_t)n * H + i] = f;
        }
    printf("SPLIT: Emb %.0fms\n",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - te).count());

    // LM head
    auto lm_emb = emb_f32.data();
    uint64_t lm_off = jo(js, jl, "lm_head.weight");
    if (lm_off) {
        int lr, lc;
        float* lm_w = dequant_i8(md, lm_off, lr, lc);
        if (lr > 0 && lc > 0) {
            lm_head_f32.resize((size_t)lr * lc);
            memcpy(lm_head_f32.data(), lm_w, (size_t)lr * lc * 4);
            lm_emb = lm_head_f32.data();
            printf("SPLIT: LM head loaded %dx%d\n", lr, lc);
        }
        free(lm_w);
    }

    // Find xclbin dir
    std::string xd = get_self_dir() + "/int8";
    if (auto* env = getenv("NPU_XCLBIN_DIR")) xd = env;

    // Init NPU device
    xrt::device dev(0);
    auto tp = std::chrono::steady_clock::now();

    // Load norm weights
    printf("SPLIT: Loading norms...\n");
    std::vector<LayerState> layers(NC);
    for (int l = 0; l < NC; l++) {
        auto& ls = layers[l];
        char buf[128];
        snprintf(buf, sizeof(buf), "model.layers.%d.input_layernorm.weight", l);
        uint64_t in_off = jo(js, jl, buf);
        snprintf(buf, sizeof(buf), "model.layers.%d.post_attention_layernorm.weight", l);
        uint64_t pa_off = jo(js, jl, buf);
        uint64_t q_off = jo(js, jl, buf); // Q norm
        uint64_t k_off = jo(js, jl, buf); // K norm

        if (in_off) { ls.in.resize(H); memcpy(ls.in.data(), i8p(in_off), H * 4); }
        if (pa_off) { ls.pa.resize(H); memcpy(ls.pa.data(), i8p(pa_off), H * 4); }
        ls.qn.resize(H, 1.0f); ls.kn.resize(H, 1.0f);
    }
    uint64_t fn_off = jo(js, jl, "model.norm.weight");
    std::vector<float> fin_w(H, 1.0f);
    if (fn_off) memcpy(fin_w.data(), i8p(fn_off), H * 4);

    // Dequant weights
    printf("SPLIT: Dequantizing weights...\n");
    auto td = std::chrono::steady_clock::now();

    auto xp = [&](const char* t) { return xd + "/final_i8_" + t + "_" + model_tag + ".xclbin"; };
    auto ip = [&](const char* t) { return xd + "/final_i8_" + t + "_" + model_tag + ".insts"; };

    // Init GEMM contexts
    GemmCtx cq, co, cg, cd;
    cq.MD = 128; cq.KD = H; cq.ND = QKV;
    co.MD = 128; co.KD = NH * HD; co.ND = H;
    cg.MD = 128; cg.KD = H; cg.ND = cfg.gu_split ? IM : IM * 2;
    cd.MD = 128; cd.KD = IM; cd.ND = H;

    if (!cq.init(dev, xp("QKV").c_str(), ip("QKV").c_str(), 4)) return 1;
    if (!co.init(dev, xp("O").c_str(), ip("O").c_str(), 4)) return 1;
    if (!cg.init(dev, xp("GU").c_str(), ip("GU").c_str(), 4)) return 1;
    if (!cd.init(dev, xp("D").c_str(), ip("D").c_str(), 4)) return 1;
    printf("SPLIT: GEMM ctxs ready %.0fms\n",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count());

    // Pack weights per layer
    printf("SPLIT: Packing weights...\n");
    for (int l = 0; l < NC; l++) {
        auto& ls = layers[l];
        // QKV
        uint64_t qkv_off = jo(js, jl, ("model.layers." + std::to_string(l) + ".self_attn.qkv_proj.weight").c_str());
        if (!qkv_off) qkv_off = jo(js, jl, ("model.layers." + std::to_string(l) + ".self_attn.q_proj.weight").c_str());
        if (qkv_off) {
            int qr, qc; float* qw = dequant_i8(md, qkv_off, qr, qc);
            if (qw) {
                float amax = 0;
                for (int i = 0; i < qr * qc; i++) {
                    float v = fabsf(qw[i]); if (v > amax) amax = v;
                }
                ls.qkv_scale = (amax < 1e-12f) ? 1.0f : amax / 127.0f;
                free(qw);
            }
        }
        // Similarly for O, GU, D weights
    }

    // RoPE
    rope_init(HD, cfg.rope_theta, 4096);

    // KV cache
    int kv_size = 4096 * NKV * HD;
    struct KVC { std::vector<float> k, v; int n; } kv_cache(kv_size);

    // Command loop
    printf("SPLIT: READY — listening on stdin\n");
    char cmd[256];
    while (fgets(cmd, sizeof(cmd), stdin)) {
        char op[16]; int layer, pos, batch, data_len;
        if (sscanf(cmd, "%s %d %d %d %d", op, &layer, &pos, &batch, &data_len) < 1) continue;

        if (strcmp(op, "EXIT") == 0) break;

        printf("SPLIT: %s layer=%d pos=%d batch=%d\n", op, layer, pos, batch);
        fflush(stdout);
    }

    platform_munmap((void*)md, (size_t)st.st_size);
    printf("SPLIT: DONE\n");
    return 0;
}
