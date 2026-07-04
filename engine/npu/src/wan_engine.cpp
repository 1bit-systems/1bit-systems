/** WAN 2.1 DiT NPU Engine — INT32 accumulator. 
 * Uses INT32 output xclbins to avoid INT16 overflow in per-tile accumulation.
 * 
 * Shapes (WAN 2.1, 30 blocks):
 *   Self/Cross-attn QKV/O: [32, 1536] G [1536, 1536] ! [32, 1536]  (n=192, i32)
 *   FFN Gate/Up:          [32, 1536] G [8960, 1536] ! [32, 8960]  (n=160, i32)
 *   FFN Down:             [32, 8960] G [1536, 8960] ! [32, 1536]  (n=192, i32)
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

static inline void cn(float* x, int n) { for (int i = 0; i < n; i++) if (!std::isfinite(x[i])) x[i] = 0.0f; }

struct WAN_GEMM {
    int MD, KD, ND;
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::kernel> k;
    std::vector<uint32_t> ins;
    std::unique_ptr<xrt::bo> bI, bA, bC;
    std::unique_ptr<xrt::bo> bB;
    int8_t* Am;
    int32_t* Cm;
    float Bscale;

    bool init(xrt::device& d, const char* xp, const char* ip, int MD_, int KD_, int ND_) {
        MD = MD_; KD = KD_; ND = ND_;
        FILE* f = fopen(ip, "rb"); if (!f) return false;
        fseek(f, 0, 2); long sz = ftell(f); fseek(f, 0, 0);
        ins.resize(sz / 4); (void)fread(ins.data(), 4, ins.size(), f); fclose(f);
        xc = std::make_unique<xrt::xclbin>(std::string(xp));
        d.register_xclbin(*xc);
        hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
        k = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");
        bI = std::make_unique<xrt::bo>(d, ins.size() * 4, XCL_BO_FLAGS_CACHEABLE, k->group_id(1));
        memcpy(bI->map(), ins.data(), ins.size() * 4); bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bA = std::make_unique<xrt::bo>(d, (size_t)MD * KD, XRT_BO_FLAGS_HOST_ONLY, k->group_id(3));
        bC = std::make_unique<xrt::bo>(d, (size_t)MD * ND * 4, XRT_BO_FLAGS_HOST_ONLY, k->group_id(5));
        Am = (int8_t*)bA->map(); Cm = (int32_t*)bC->map();
        bB = std::make_unique<xrt::bo>(d, (size_t)KD * ND, XRT_BO_FLAGS_HOST_ONLY, k->group_id(4));
        return true;
    }

    void load_weight(const int8_t* w, float scale) {
        Bscale = scale;
        memcpy(bB->map(), w, (size_t)KD * ND);
        bB->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    void go(const float* A, int am, int ak, float ascale, float* C, int an) {
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
        bB->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r = (*k)((unsigned)3, *bI, (unsigned)ins.size(), *bA, *bB, *bC);
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

struct TensorEntry { std::string name; size_t offset; int out_f, in_f; float scale; };

static std::string extract_str(const std::string& obj, const std::string& key) {
    auto kp = obj.find("\"" + key + "\"");
    if (kp == std::string::npos) return "";
    auto colon = obj.find(':', kp);
    if (colon == std::string::npos) return "";
    auto start = obj.find_first_not_of(" \t\r\n", colon + 1);
    if (start == std::string::npos) return "";
    if (obj[start] == '"') { auto end = obj.find('"', start + 1); return obj.substr(start + 1, end - start - 1); }
    auto end = obj.find_first_of(",}]", start);
    return obj.substr(start, end - start);
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 3) { printf("Usage: %s wan_weights.bin wan_meta.json [layer=0]\n", argv[0]); return 1; }
    int test_layer = (argc > 3) ? atoi(argv[3]) : 0;

    printf("=== WAN 2.1 DiT NPU Engine (INT32) ===\n\n");

    // Load meta
    int fd = open(argv[2], O_RDONLY); struct stat st; fstat(fd, &st);
    char* meta = (char*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    std::string js(meta, st.st_size); munmap(meta, st.st_size);

    // Parse tensor entries
    std::vector<TensorEntry> tensors;
    auto tp = js.find("\"tensors\""); if (tp == std::string::npos) return 1;
    size_t pos = js.find('[', tp);
    while (pos < js.size()) {
        auto ob = js.find('{', pos); if (ob == std::string::npos) break;
        auto oe = js.find('}', ob); if (oe == std::string::npos) break;
        std::string obj = js.substr(ob, oe - ob + 1);
        TensorEntry te;
        te.name = extract_str(obj, "name");
        te.offset = (size_t)atoll(extract_str(obj, "offset").c_str());
        te.scale = atof(extract_str(obj, "scale").c_str());
        auto sp = obj.find("\"shape\""); if (sp != std::string::npos) {
            auto sb = obj.find('[', sp), se = obj.find(']', sb);
            auto sarr = obj.substr(sb + 1, se - sb - 1);
            auto comma = sarr.find(',');
            te.out_f = atoi(sarr.substr(0, comma).c_str());
            te.in_f = atoi(sarr.substr(comma + 1).c_str());
        }
        if (!te.name.empty()) tensors.push_back(te);
        pos = oe + 1;
    }

    // Load weights
    fd = open(argv[1], O_RDONLY); fstat(fd, &st);
    const uint8_t* wdata = (const uint8_t*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    printf("Weights: %zu MB, %zu tensors\n", st.st_size >> 20, tensors.size());

    auto find_t = [&](const std::string& n) -> const TensorEntry* {
        for (auto& t : tensors) if (t.name == n) return &t;
        return nullptr;
    };

    // Init NPU
    printf("Init NPU xclbins (INT32)...\n"); xrt::device dev(0);
    std::string xd = "/home/bcloud/npu-sandbox/npu-infer/build/wan";
    WAN_GEMM attn_gemm, ffn_gate_gemm, ffn_down_gemm;
    if (!attn_gemm.init(dev, (xd + "/wan_attn_1536_i32.xclbin").c_str(), (xd + "/insts_wan_attn_1536_i32.txt").c_str(), 32, 1536, 1536)) return 1;
    if (!ffn_gate_gemm.init(dev, (xd + "/wan_ffn_gate_i32.xclbin").c_str(), (xd + "/insts_wan_ffn_gate_i32.txt").c_str(), 32, 1536, 8960)) return 1;
    if (!ffn_down_gemm.init(dev, (xd + "/wan_ffn_down_i32.xclbin").c_str(), (xd + "/insts_wan_ffn_down_i32.txt").c_str(), 32, 9216, 1536)) return 1;
    printf("  OK\n");

    // Test
    const int H = 1536;
    char qn[128]; snprintf(qn, 128, "blocks.%d.self_attn.q.weight", test_layer);
    auto* qi = find_t(qn);
    if (!qi) { printf("ERR: %s not found\n", qn); return 1; }
    printf("\n=== %s [%dx%d] scale=%.6f ===\n", qn, qi->out_f, qi->in_f, qi->scale);

    const int8_t* wi8 = (const int8_t*)(wdata + qi->offset);
    attn_gemm.load_weight(wi8, qi->scale);

    // Input: all ones
    std::vector<float> inp(H, 1.0f);
    float amax = 1.0f;
    float ascale = amax / 127.0f;

    // NPU
    std::vector<float> out(H);
    auto t0 = std::chrono::steady_clock::now();
    attn_gemm.go(inp.data(), 1, H, ascale, out.data(), H);
    auto t1 = std::chrono::steady_clock::now();

    float onorm = 0; for (auto v : out) onorm += v*v; onorm = sqrtf(onorm/H);
    printf("NPU: norm=%.4f time=%.2fms\n  first10:", onorm, std::chrono::duration<double,std::milli>(t1-t0).count());
    for (int i = 0; i < 10; i++) printf(" %.4f", out[i]); printf("\n");

    // CPU reference
    std::vector<float> ref(H, 0);
    auto t2 = std::chrono::steady_clock::now();
    for (int o = 0; o < H; o++) { double s = 0; for (int k = 0; k < H; k++) s += inp[k] * (double)wi8[(size_t)k*H+o] * qi->scale; ref[o] = (float)s; }
    auto t3 = std::chrono::steady_clock::now();
    float rnorm = 0; for (auto v : ref) rnorm += v*v; rnorm = sqrtf(rnorm/H);
    printf("CPU: norm=%.4f time=%.2fms\n  first10:", rnorm, std::chrono::duration<double,std::milli>(t3-t2).count());
    for (int i = 0; i < 10; i++) printf(" %.4f", ref[i]); printf("\n");

    // Compare
    double merr = 0, serr = 0;
    for (int i = 0; i < H; i++) { double e = fabsf(out[i] - ref[i]); if (e > merr) merr = e; serr += e; }
    double rerr = (ref[0]-(-ref[0])) > 0 ? merr / (ref[0]-(-ref[0])) * 100 : 0;
    printf("  mean_err=%.6f max_err=%.6f rel_err=%.2f%%\n", serr/H, merr, rerr);
    printf("%s\n\n", rerr < 3.0 ? "MATCH" : "MISMATCH");

    munmap((void*)wdata, st.st_size);
    return 0;
}
