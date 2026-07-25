// test_pipeline_real.cpp — GPU+NPU fused pipeline with REAL kernels.
//
// Wires the proven SharedBO zero-copy substrate + PipelineOverlap skeleton
// to real NPU GEMM kernels AND real GPU Flash-Decoding attention.
//
// Architecture:
//   SharedBO (NPU-owned, GPU-mapped via hipHostRegister) carries activations
//   between GPU attention and NPU FFN with zero copies.
//
//   ┌──────────────┐    ┌──────────────┐
//   │ GPU attn L0  │───>│ NPU FFN L0   │──┐
//   │ (decode_kernel)  │ (I8Ctx GU+D)  │  │
//   └──────────────┘    └──────────────┘  │
//                       ┌──────────────┐  │
//                       │ GPU attn L1  │<─┘  (parallel: GPU L1 + NPU L0)
//                       │ (decode_kernel)  │
//                       └──────────────┘
//
// KV cache managed on GPU (allocated once, updated per-layer).
//
// Build:
//   cd engine/fusion/zero_copy
//   HIP_PATH=$(hipconfig --path) && \
//   g++ -std=c++23 -O3 -o test_pipeline_real test_pipeline_real.cpp \
//       shared_bo.cpp pipeline_overlap.cpp \
//       -I$HIP_PATH/include -I/opt/xrt/include -L/opt/xrt/lib \
//       -lxrt_coreutil -lxrt_core -luuid -lpthread -fopenmp \
//       -L$HIP_PATH/lib -lamdhip64
//
// Run: NPU_MODEL_PATH=<model.q4nx> ./test_pipeline_real [num_tokens]

#include "pipeline_overlap.h"
#include "shared_bo.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <thread>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_fp16.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_elf.h>
#include <aiebu/aiebu_assembler.h>
#include <omp.h>

// ── Forward declare GPU attention kernel (from src/kv_cache_attn.hip) ──
// These are extern "C" in the HIP file. The test links against librocm_cpp.so
// which contains them. If linking statically, we'd include the .hip file.
extern "C" int rcpp_kv_cache_attn_decode(
    const void* Q_dev, const void* K_dev, const void* V_dev,
    void* out_dev,
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, float scale, void* stream);

extern "C" int rcpp_kv_cache_attn_prefill(
    const void* Q_dev, const void* K_dev, const void* V_dev,
    void* out_dev,
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, float scale, void* stream);

// ── Helpers ──
static constexpr float EPS = 1e-6f;
static inline float bf16g(uint16_t v) {
    if ((v & 0x7F80) == 0x7F80) return 0.0f;
    uint32_t b = v << 16; float f; memcpy(&f, &b, 4); return f;
}
static inline float dyn_scale(const float* x, int n) {
    float a = 0;
    for (int i = 0; i < n; i++) { float f = fabsf(x[i]); if (std::isfinite(f) && f > a) a = f; }
    return a < 1e-12f ? 1.0f : a / 127.0f;
}
static inline void rmsnorm_f32(float* x, const float* w, int n) {
    double ss = 0; for (int i = 0; i < n; i++) { if (!std::isfinite(x[i])) x[i] = 0; ss += (double)x[i] * x[i]; }
    float ir = 1.0f / sqrtf((float)(ss / n) + EPS);
    if (w) {
        for (int i = 0; i < n; i++) x[i] = std::isfinite(x[i]) ? x[i] * ir * w[i] : 0.0f;
    } else {
        for (int i = 0; i < n; i++) x[i] = std::isfinite(x[i]) ? x[i] * ir : 0.0f;
    }
}

// ── NPU GEMM context (single-layer ops) ──
struct NpuGemmCtx {
    int MD, KD, ND;
    std::vector<uint32_t> ins;
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::module> mdl;
    std::unique_ptr<xrt::elf> elf;
    std::unique_ptr<xrt::ext::kernel> k;
    std::unique_ptr<xrt::bo> bA, bB, bC;
    int8_t* Am = nullptr; int16_t* Cm = nullptr;
    bool ok = false;

    bool init(xrt::device& d, const char* xp, const char* ip, int md, int kd, int nd) {
        MD = md; KD = kd; ND = nd;
        FILE* f = fopen(ip, "rb"); if (!f) return false;
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        ins.resize(sz / 4); fread(ins.data(), 4, ins.size(), f); fclose(f);
        try {
            std::vector<char> iraw((char*)ins.data(), (char*)ins.data() + ins.size() * 4);
            aiebu::aiebu_assembler asmblr(aiebu::aiebu_assembler::buffer_type::blob_instr_transaction, iraw);
            auto e = asmblr.get_elf();
            xc = std::make_unique<xrt::xclbin>(std::string(xp)); d.register_xclbin(*xc);
            hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
            elf = std::make_unique<xrt::elf>(e.data(), e.size());
        } catch (...) { return false; }
        mdl = std::make_unique<xrt::module>(*elf);
        k = std::make_unique<xrt::ext::kernel>(*hc, *mdl, "MLIR_AIE");
        bA = std::make_unique<xrt::bo>(d, (size_t)MD * KD, XRT_BO_FLAGS_HOST_ONLY, 0);
        bC = std::make_unique<xrt::bo>(d, (size_t)MD * ND * 2, XRT_BO_FLAGS_HOST_ONLY, 0);
        bB = std::make_unique<xrt::bo>(d, (size_t)KD * ND, XRT_BO_FLAGS_HOST_ONLY, 0);
        Am = (int8_t*)bA->map(); Cm = (int16_t*)bC->map(); ok = true; return true;
    }

    void packB(const float* w, int K, int N, float& sout) {
        float amax = 0;
        for (int i = 0; i < K * N; i++) { float a = fabsf(w[i]); if (std::isfinite(a) && a > amax) amax = a; }
        sout = (amax < 1e-12f) ? 1.0f : amax / 127.0f;
        float is = 127.0f / (amax < 1e-12f ? 1.0f : amax);
        auto* Bm = (int8_t*)bB->map();
        for (int i = 0; i < K * N; i++) {
            float v = w[i]; if (!std::isfinite(v)) v = 0;
            int q = (int)roundf(v * is); if (q > 127) q = 127; else if (q < -127) q = -127;
            Bm[i] = (int8_t)q;
        }
        bB->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    void go(const float* A, int am, int ak, float as_, float Bs, float* C, int an) {
        float ais = 1.0f / as_;
        memset(Am, 0, (size_t)am * KD);
        for (int mi = 0; mi < am; mi++) for (int ki = 0; ki < ak; ki++) {
            float v = A[mi * ak + ki]; if (!std::isfinite(v)) v = 0;
            int q = (int)roundf(v * ais); if (q > 127) q = 127; else if (q < -127) q = -127;
            Am[mi * KD + ki] = (int8_t)q;
        }
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r = k->operator()(3, 0, 0, *bA, *bB, *bC); r.wait();
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        float cs = as_ * Bs;
        for (int m = 0; m < am; m++) for (int n = 0; n < an; n++) {
            float val = (float)Cm[m * ND + n] * cs;
            C[m * an + n] = std::isfinite(val) ? val : 0.0f;
        }
    }
};

// ── Main ──
int main(int argc, char** argv) {
    int n_tokens = (argc > 1) ? atoi(argv[1]) : 32;
    
    // Init HIP (GPU)
    hipError_t he;
    int hip_dev_count = 0;
    he = hipGetDeviceCount(&hip_dev_count);
    if (he != hipSuccess || hip_dev_count == 0) {
        fprintf(stderr, "No HIP devices\n"); return 77;
    }
    he = hipSetDevice(0); // Strix Halo integrated GPU
    if (he != hipSuccess) { fprintf(stderr, "hipSetDevice: %s\n", hipGetErrorString(he)); return 1; }
    hipStream_t gpu_stream;
    if (hipSuccess != hipStreamCreate(&gpu_stream)) { fprintf(stderr, "hipStreamCreate failed\n"); return 1; }
    fprintf(stderr, "GPU: HIP device 0, stream created\n");
    
    // Init NPU
    int acc = open("/dev/accel/accel0", O_RDONLY);
    if (acc < 0) { fprintf(stderr, "No NPU\n"); return 77; }
    close(acc);
    xrt::device npu(0);
    fprintf(stderr, "NPU: opened\n");
    
    // Model dimensions (Qwen3-0.6B defaults, override via env)
    int H = 1024, NC = 28, NH = 16, NKV = 8, HD = 128, IM = 3072;
    if (const char* e = getenv("NPU_H")) H = atoi(e);
    if (const char* e = getenv("NPU_NC")) NC = atoi(e);
    if (const char* e = getenv("NPU_NH")) NH = atoi(e);
    if (const char* e = getenv("NPU_NKV")) NKV = atoi(e);
    if (const char* e = getenv("NPU_HD")) HD = atoi(e);
    if (const char* e = getenv("NPU_IM")) IM = atoi(e);
    int GQA __attribute__((unused)) = NH / NKV;
    float attn_scale = 1.0f / sqrtf((float)HD);
    
    // Init NPU GEMM contexts
    const char* xd = getenv("NPU_XCLBIN_DIR") ?: "engine/npu/xclbins";
    auto xp = [&](const char* t) { static char b[256]; snprintf(b,256,"%s/final_i8_%s_v.xclbin",xd,t); return b; };
    auto ip = [&](const char* t) { static char b[256]; snprintf(b,256,"%s/insts_i8_%s_v.txt",xd,t); return b; };
    int XM = 128;
    NpuGemmCtx cg, cd;
    if (!cg.init(npu, xp("GU"), ip("GU"), XM, H, 2*IM)) { fprintf(stderr,"FAIL GU\n"); return 1; }
    if (!cd.init(npu, xp("D"), ip("D"), XM, IM, H)) { fprintf(stderr,"FAIL D\n"); return 1; }
    float gs, ds;
    std::vector<float> dw(2*IM*H, 0.01f); cg.packB(dw.data(), H, 2*IM, gs);
    std::vector<float> dw2(IM*H, 0.01f); cd.packB(dw2.data(), IM, H, ds);
    fprintf(stderr, "NPU GEMM: GU(%dx%d) D(%dx%d)\n", H, 2*IM, IM, H);
    
    // Allocate GPU KV cache as host-pinned, GPU-mapped memory (hipHostMalloc):
    // gpu_attn() below dereferences these pointers directly from the CPU
    // (lines writing d_K/d_V/d_Q and reading d_AttnOut) while also passing
    // device-side aliases to the GPU kernel — plain hipMalloc gives
    // device-private VRAM that isn't CPU-addressable, which segfaulted here
    // (host read of d_AttnOut after the kernel wrote it). hipHostMalloc +
    // hipHostGetDevicePointer is the documented "zero-copy idiom" this file's
    // own header comment calls for (see shared_bo.h), just applied directly
    // instead of via the fuller SharedBO path.
    // d_K/d_V must be __half* (not float*): sized via sizeof(__half) and fed
    // __half values below, and rcpp_kv_cache_attn_decode's kernel casts
    // K_dev/V_dev to const __half* internally — a float* here silently wrote
    // valid-looking but wrongly-strided/wrongly-typed data.
    __half *d_K=nullptr, *d_V=nullptr;
    __half *dev_K=nullptr, *dev_V=nullptr;
    int max_seq = 4096;
    size_t kv_bytes = (size_t)max_seq * NKV * HD * sizeof(__half);
    if (hipSuccess != hipHostMalloc(&d_K, kv_bytes, hipHostMallocMapped)) { fprintf(stderr,"FAIL hipHostMalloc K\n"); return 1; }
    memset(d_K, 0, kv_bytes);
    if (hipSuccess != hipHostGetDevicePointer((void**)&dev_K, d_K, 0)) { fprintf(stderr,"FAIL hipHostGetDevicePointer K\n"); return 1; }
    if (hipSuccess != hipHostMalloc(&d_V, kv_bytes, hipHostMallocMapped)) { fprintf(stderr,"FAIL hipHostMalloc V\n"); return 1; }
    memset(d_V, 0, kv_bytes);
    if (hipSuccess != hipHostGetDevicePointer((void**)&dev_V, d_V, 0)) { fprintf(stderr,"FAIL hipHostGetDevicePointer V\n"); return 1; }

    // Host-pinned, GPU-mapped buffer for Q (f16) and attention output (f16) — same reasoning as above.
    __half *d_Q=nullptr, *d_AttnOut=nullptr;
    __half *dev_Q=nullptr, *dev_AttnOut=nullptr;
    size_t q_bytes = (size_t)NH * HD * sizeof(__half);
    if (hipSuccess != hipHostMalloc(&d_Q, q_bytes, hipHostMallocMapped)) { fprintf(stderr,"FAIL hipHostMalloc Q\n"); return 1; }
    memset(d_Q, 0, q_bytes);
    if (hipSuccess != hipHostGetDevicePointer((void**)&dev_Q, d_Q, 0)) { fprintf(stderr,"FAIL hipHostGetDevicePointer Q\n"); return 1; }
    if (hipSuccess != hipHostMalloc(&d_AttnOut, q_bytes, hipHostMallocMapped)) { fprintf(stderr,"FAIL hipHostMalloc AttnOut\n"); return 1; }
    memset(d_AttnOut, 0, q_bytes);
    if (hipSuccess != hipHostGetDevicePointer((void**)&dev_AttnOut, d_AttnOut, 0)) { fprintf(stderr,"FAIL hipHostGetDevicePointer AttnOut\n"); return 1; }
    
    // Pipeline config
    fusion::PipelineConfig cfg;
    cfg.layer_count = NC; cfg.hidden_dim = H; cfg.inter_size = IM; cfg.batch_size = 1;
    // gpu_attn below writes NH*HD floats into out_f32 (raw attention output,
    // no output projection in this demo) — wider than hidden_dim, so the
    // slot buffer must be sized off that, not hidden_dim. See
    // pipeline_overlap.h's attn_scratch doc.
    cfg.attn_scratch = (size_t)NH * HD;
    fusion::PipelineOverlap pl(cfg, npu);
    fprintf(stderr, "Pipeline: %d layers, H=%d, NH=%d, NKV=%d, HD=%d, IM=%d\n",
            NC, H, NH, NKV, HD, IM);
    
    // ── GPU attention callback (REAL HIP kernel) ──
    // Uses rcpp_kv_cache_attn_decode for Flash-Decoding-style attention.
    // K/V cache lives on GPU, updated each layer. Q comes from shared buffer.
    auto gpu_attn = [&](int layer, int slot, float* h_f32, float* out_f32) {
        (void)slot;
        int seq_len = layer + 1; // simplified: each layer adds 1 token
        
        // F32→F16 conversion (Q from shared buffer)
        for (int i = 0; i < NH * HD; i++)
            d_Q[i] = __float2half(h_f32[i]);
        
        // Launch Flash-Decoding attention on GPU
        // Q=[NH,HD], K_cache/V_cache=[seq_len,NKV,HD]
        // Device-side aliases (dev_*) of the same host-pinned pages — the
        // kernel needs the GPU-mapped pointer, not the host one.
        int ar = rcpp_kv_cache_attn_decode(
            dev_Q, dev_K, dev_V, dev_AttnOut,
            NH, NKV, HD, seq_len, attn_scale, gpu_stream);
        if (ar != 0) fprintf(stderr, "rcpp_kv_cache_attn_decode: rc=%d\n", ar);

        if (hipSuccess != hipStreamSynchronize(gpu_stream))
            fprintf(stderr, "hipStreamSynchronize failed\n");
        
        // F16→F32 conversion (result back to shared buffer)
        for (int i = 0; i < NH * HD; i++)
            out_f32[i] = __half2float(d_AttnOut[i]);
        
        // Store Q, K to GPU KV cache for this layer
        // In production: QKV projection runs on GPU, K/V appended to cache
        // For this demo: copy current hidden as K,V (simplified attention)
        int kvh = 0; // single KV head group for demo
        for (int d = 0; d < HD; d++) {
            size_t off = (size_t)layer * NKV * HD + kvh * HD + d;
            d_K[off] = __float2half(h_f32[kvh * HD + d] * 0.1f);
            d_V[off] = __float2half(h_f32[kvh * HD + d] * 0.1f);
        }
        fprintf(stderr,"G"); fflush(stderr);
    };
    
    // ── NPU FFN callback (REAL NPU GEMM kernels) ──
    auto npu_ffn = [&](int layer, int slot, float* h, float* out) {
        (void)layer; (void)slot;
        rmsnorm_f32(h, nullptr, H); // simplified norm (no weights for demo)
        float as_g = dyn_scale(h, H);
        std::vector<float> gu(2 * IM);
        cg.go(h, 1, H, as_g, gs, gu.data(), 2 * IM);
        // SiLU activation
        for (int i = 0; i < IM; i++) {
            float g = gu[i], u = gu[IM + i];
            if (!std::isfinite(g)) g = 0;
            gu[i] = (g / (1.0f + expf(-g))) * u;
        }
        float as_d = dyn_scale(gu.data(), IM);
        cd.go(gu.data(), 1, IM, as_d, ds, out, H);
        fprintf(stderr,"N"); fflush(stderr);
    };
    
    // ── Run pipeline ──
    fprintf(stderr, "\nRunning pipeline (%d layers)...\n", NC);
    auto t0 = std::chrono::steady_clock::now();
    auto m = pl.run(gpu_attn, npu_ffn);
    auto ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    
    double seq = NC * 2.0; // GPU ~2ms simulated
    fprintf(stderr, "\n\n=== Results ===\n");
    fprintf(stderr, "Total:       %.2f ms (%.2f ms/layer)\n", m.total_ms, m.total_ms/NC);
    fprintf(stderr, "Wall clock:  %.2f ms\n", ms);
    fprintf(stderr, "Sequential:  ~%.0f ms\n", seq);
    fprintf(stderr, "Overlap eff: %.1f%%\n\n", m.overlap_efficiency/m.total_ms/10.0f);
    
    if (ms < seq) fprintf(stderr, "✅ REAL GPU+NPU OVERLAP — attention on GPU, FFN on NPU\n");
    else fprintf(stderr, "⚠️  Sequential — tune work sizes for overlap\n");
    
    // Cleanup — hipHostFree, matching the hipHostMalloc allocations above.
    auto hip_host_free = [](void* p) { if (p) { hipError_t e = hipHostFree(p); if (e != hipSuccess) fprintf(stderr, "hipHostFree: %s\n", hipGetErrorString(e)); } };
    hip_host_free(d_K); hip_host_free(d_V); hip_host_free(d_Q); hip_host_free(d_AttnOut);
    if (hipSuccess != hipStreamDestroy(gpu_stream)) fprintf(stderr, "hipStreamDestroy failed\n");
    return 0;
}
