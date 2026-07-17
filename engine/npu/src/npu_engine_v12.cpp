/** NPU Engine v12 - M=32 + OpenMP attention
 *
 *  Standalone INT8 GEMM benchmark harness for the XDNA 2 NPU. The I8Ctx below
 *  owns one MLIR_AIE kernel + its instruction stream and the per-layer B
 *  weight buffers; packB() quantizes a weight tile into int8, go() runs one
 *  M×K×N GEMM on-device and dequantizes the int32 accumulator.
 *
 *  This file used to contain only the struct with no entry point, so
 *  .github/workflows/bench.yml ("g++ -o npu_engine_v12 npu_engine_v12.cpp …")
 *  failed at link time with "undefined reference to main" (issue #234). The
 *  main() at the bottom drives the GEMM path over NC layers with synthetic
 *  weights and prints the "ms/tok" line bench.yml greps for. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
extern "C" float* dequant_i8_to_float(const uint8_t*,int,int*,int*);
static inline float bf16g(uint16_t v){return(v&0x7F80)==0x7F80?0.0f:[&]{uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}();}
static constexpr int H=1024,NC=28,NH=16,NKV=8,HD=128,IM=3072,NV=151936,GQA=2;
static constexpr float EPS=1e-6f; static constexpr int XM=128, BS=64;
static inline void cn(float*x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
static inline float dynamic_ascale(const float* x, int n) {
    float amax = 0;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (std::isfinite(a) && a > amax) amax = a; }
    if (amax < 1e-12f) amax = 1.0f;
    return amax / 127.0f;
}
struct I8Ctx{const char*name;int MD,KD,ND;std::unique_ptr<xrt::xclbin>xc;std::unique_ptr<xrt::hw_context>hc;std::unique_ptr<xrt::kernel>k;std::vector<uint32_t>ins;std::unique_ptr<xrt::bo>bI,bA,bC,layerB[NC];int8_t*Am;int32_t*Cm;
bool init(xrt::device&d,const char*xp,const char*ip,int gid_B){
    FILE*f=fopen(ip,"rb");if(!f)return false;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
    xc=std::make_unique<xrt::xclbin>(std::string(xp));d.register_xclbin(*xc);hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");
    bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));memcpy(bI->map(),ins.data(),ins.size()*4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bA=std::make_unique<xrt::bo>(d,(size_t)MD*KD,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));
    bC=std::make_unique<xrt::bo>(d,(size_t)MD*ND*4,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));
    Am=(int8_t*)bA->map();Cm=(int32_t*)bC->map();
    for(int l=0;l<NC;l++)layerB[l]=std::make_unique<xrt::bo>(d,(size_t)KD*ND,XRT_BO_FLAGS_HOST_ONLY,k->group_id(gid_B));
    return true;
}
void packB(int l,const float*w,int K,int N,float&sout){float amax=0;for(int i=0;i<K*N;i++){float a=fabsf(w[i]);if(std::isfinite(a)&&a>amax)amax=a;}if(amax<1e-12f)amax=1.0f;sout=amax/127.0f;float is=127.0f/amax;auto*Bm=(int8_t*)layerB[l]->map();for(int i=0;i<K*N;i++){float v=w[i];if(!std::isfinite(v))v=0;int x=(int)roundf(v*is);if(x>127)x=127;else if(x<-127)x=-127;Bm[i]=(int8_t)x;}layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);}
inline void go(int l,const float*A,int am,int ak,float ascale,float Bscale,float*C,int an){
    float ais=1.0f/ascale;// memset(Am,0,(size_t)am*KD); // removed: every element is written
    for(int m=0;m<am;m++)for(int k=0;k<ak;k++){float v=A[m*ak+k];int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;Am[m*KD+k]=(int8_t)q;}
    bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);auto r=(*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA,*layerB[l],*bC);r.wait();bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    float cs=ascale*Bscale;
    for(int m=0;m<am;m++)for(int n=0;n<an;n++){float val=(float)Cm[m*ND+n]*cs;if(!std::isfinite(val))val=0;C[m*an+n]=val;}
}
};

// ─── Standalone synthetic GEMM benchmark (entry point for issue #234) ─────────
// bench.yml builds this TU and runs:  ./npu_engine_v12 [prefill_tokens] [decode_tokens]
//   prefill_tokens (default 9, capped at XM=128): batched GEMM pass to warm the NPU
//   decode_tokens  (default 4): single-token-per-step passes timed for ms/tok
//
// xclbin + instruction paths are resolved from env (see engine/npu/src/npu_paths.h):
//   NPU_V12_XCLBIN / NPU_V12_INST  — explicit overrides (highest priority)
//   NPU_XCLBIN_DIR / NPU_INSTS_DIR — directory; file defaults to the Qwen3-1.7B
//                                    down-projection pair shipped in the repo.
static std::string v12_resolve(const char* envvar, const char* dflt) {
    const char* e = getenv(envvar);
    return std::string(e ? e : dflt);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        printf("Usage: %s [prefill_tokens=%d] [decode_tokens=%d]\n", argv[0], 9, 4);
        printf("Env: NPU_V12_XCLBIN / NPU_V12_INST, or NPU_XCLBIN_DIR / NPU_INSTS_DIR\n");
        return 0;
    }
    int M = (argc > 1) ? atoi(argv[1]) : 9;   if (M < 1) M = 1;   if (M > XM) M = XM;
    int N = (argc > 2) ? atoi(argv[2]) : 4;   if (N < 1) N = 1;

    std::string xdir = v12_resolve("NPU_XCLBIN_DIR", "engine/npu/xclbins");
    std::string idir = v12_resolve("NPU_INSTS_DIR", xdir.c_str());
    std::string xp = v12_resolve("NPU_V12_XCLBIN", (xdir + "/final_i8_D_qwen3_1_7b.xclbin").c_str());
    std::string ip = v12_resolve("NPU_V12_INST",   (idir + "/insts_i8_D_qwen3_1_7b.txt").c_str());

    std::unique_ptr<xrt::device> devp;
    try { devp = std::make_unique<xrt::device>(0); }
    catch (const std::exception& e) {
        fprintf(stderr, "NPU v12: no XRT device 0 (%s). Install the AMD XDNA driver + XRT.\n", e.what());
        return 1;
    }
    xrt::device& dev = *devp;

    I8Ctx ctx;
    ctx.name = "v12";
    ctx.MD = XM; ctx.KD = H; ctx.ND = H;
    if (!ctx.init(dev, xp.c_str(), ip.c_str(), /*gid_B=*/3)) {
        fprintf(stderr, "NPU v12: init failed — xclbin=%s instr=%s\n", xp.c_str(), ip.c_str());
        fprintf(stderr, "  set NPU_V12_XCLBIN / NPU_V12_INST (or NPU_XCLBIN_DIR / NPU_INSTS_DIR)\n");
        return 1;
    }

    // Pack a synthetic H×H int8 weight tile for every layer.
    std::vector<float> w((size_t)H * H);
    std::vector<float> bscale(NC, 1.0f);
    unsigned seed = 12345u;
    for (int l = 0; l < NC; ++l) {
        for (size_t i = 0; i < w.size(); ++i)
            w[i] = (float)((int)(rand_r(&seed) % 2001) - 1000) / 1000.0f;  // [-1, 1]
        ctx.packB(l, w.data(), H, H, bscale[l]);
    }

    // Prefill: one batched (M-row) GEMM per layer to warm the engine.
    std::vector<float> hin((size_t)M * H, 0.5f), hout((size_t)M * H, 0.0f);
    {
        float ascale = dynamic_ascale(hin.data(), M * H);
        for (int l = 0; l < NC; ++l)
            ctx.go(l, hin.data(), M, H, ascale, bscale[l], hout.data(), H);
    }

    // Decode: N steps, batch=1, through all NC layers — this is the timed loop.
    std::vector<float> d(H, 0.5f), o(H, 0.0f);
    auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < N; ++s) {
        for (int l = 0; l < NC; ++l) {
            float ascale = dynamic_ascale(d.data(), H);
            ctx.go(l, d.data(), 1, H, ascale, bscale[l], o.data(), H);
            std::swap(d, o);  // feed this layer's output into the next
        }
    }
    double dt_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    double ms_per_tok = dt_ms / std::max(1, N);
    double tok_s = ms_per_tok > 0.0 ? 1000.0 / ms_per_tok : 0.0;
    printf("=== NPU v12 synthetic decode (prefill=%d, decode=%d, %d layers) ===\n", M, N, NC);
    printf("Decode: %.1f ms total, %.2f ms/tok, %.1f tok/s\n", dt_ms, ms_per_tok, tok_s);
    return 0;
}
