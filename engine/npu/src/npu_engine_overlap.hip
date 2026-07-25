
/** npu_engine_fused.cpp — GPU+NPU fused inference engine.
 *
 * Architecture:
 *   GPU does:  QKV projection → attention → O projection
 *   NPU does:  GU projection → SiLU → D projection
 *   CPU does:  RMS Norm, RoPE, residuals, LM head
 *
 *   Pipeline overlap (via SharedBO zero-copy):
 *     GPU attention for layer L runs in parallel with NPU FFN for layer L-1
 *
 *   Per-layer flow:
 *     CPU: RMS Norm(hidden) → quantize for QKV → submit to GPU (f16)
 *     GPU: QKV GEMM → Q norm + RoPE → K/V cache append → Flash-Decoding attn → O GEMM
 *     CPU: residual add → RMS Norm → quantize for GU → submit to NPU (I8)
 *     NPU: GU GEMM → SiLU → D GEMM
 *     CPU: residual add → output = next layer's input
 *
 *   Pipeline overlap:
 *     ┌──────────────┐         ┌──────────────┐
 *     │ GPU attn(L)  │────────>│ NPU FFN(L)   │──┐
 *     └──────────────┘         └──────────────┘  │
 *                          ┌─────────────────────┘
 *                          ▼
 *     ┌──────────────┐   ┌──────────────┐
 *     │ GPU attn(L+1)│<──│ NPU FFN(L)   │  (parallel: GPU L+1 ∥ NPU L)
 *     └──────────────┘   └──────────────┘
 *
 * Weight distribution:
 *   GPU: QKV projection [H × qkv_n] → f16, K/V cache → f16 on device
 *        O projection  [NH*HD × H] → f16
 *        LM head (embedding) → f16 on device
 *   NPU: GU projection [H × 2*IM] → packed I8 per layer
 *        D projection  [IM × H]   → packed I8 per layer
 *   CPU: RMS Norm, Q/K norm, RoPE tables → f32 in host memory
 *
 * Based on: engine/npu/src/npu_engine_v12.cpp, engine/fusion/zero_copy/
 * Requires: XRT, ROCm/HIP 7.2.4, Composable Kernel
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
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

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <omp.h>
#include <immintrin.h>

extern "C" {
    void launch_gemv_tiled(const __half*, const __half*, __half*, int, int, int, int, hipStream_t);
    void launch_lmhead(const __half*, const __half*, float*, int, int, int, int, hipStream_t);
    void launch_gemv_simple(const __half*, const __half*, __half*, int, int, hipStream_t);
}


// ── Forward declarations ──────────────────────────────────────────
extern "C" float* dequant_i8_to_float(const uint8_t*, int, int*, int*);
extern "C" int rcpp_kv_cache_attn_decode(
    const void* Q_dev, const void* K_dev, const void* V_dev,
    void* out_dev, int num_q_heads, int num_kv_heads,
    int head_dim, int seq_len, float scale, void* stream);

// ── Constants ─────────────────────────────────────────────────────
static constexpr float EPS = 1e-6f;
static constexpr int MAX_SEQ = 4096;

// ── BF16 helpers ──────────────────────────────────────────────────
static inline float bf16g(uint16_t v) {
    if ((v & 0x7F80) == 0x7F80) return 0.0f;
    uint32_t b = (uint32_t)v << 16; float f; memcpy(&f, &b, 4); return f;
}

static inline void cn(float*x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
static inline float dyn_scale(const float* x, int n) {
    float a = 0;
    for (int i = 0; i < n; i++) { float f = fabsf(x[i]); if (std::isfinite(f) && f > a) a = f; }
    return a < 1e-12f ? 1.0f : a / 127.0f;
}

static inline void clean_nans(float* x, int n) {
    for (int i = 0; i < n; i++) if (!std::isfinite(x[i])) x[i] = 0.0f;
}

// ── RMS Norm (CPU) ────────────────────────────────────────────────
static void rms_norm(float* x, const float* w, int n) {
    clean_nans(x, n);
    double ss = 0;
    for (int i = 0; i < n; i++) if (std::isfinite(x[i])) ss += (double)x[i] * x[i];
    float ir = 1.0f / sqrtf((float)(ss / n) + EPS);
    for (int i = 0; i < n; i++) x[i] = std::isfinite(x[i]) ? x[i] * ir * w[i] : 0.0f;
}

// ── SiLU (CPU) ────────────────────────────────────────────────────
static inline float silu(float x) { return x / (1.0f + expf(-x)); }

// ── RoPE tables (CPU) ────────────────────────────────────────────
struct RopeTables {
    std::vector<float> cos, sin;
    int max_pos = 0, head_dim = 0;
    float theta = 1000000.0f;

    void init(int hd, int max_p, float th) {
        head_dim = hd; max_pos = max_p; theta = th;
        cos.resize(max_p * hd); sin.resize(max_p * hd);
        int hd2 = hd / 2;
        for (int p = 0; p < max_p; p++) {
            for (int d = 0; d < hd2; d++) {
                float f = 1.0f / powf(theta, (float)d / hd2);
                float a = (float)p * f;
                cos[p * hd + d] = cosf(a);
                sin[p * hd + d] = sinf(a);
            }
        }
    }

    void apply(float* qk, int pos, int n_heads) const {
        int hd2 = head_dim / 2;
        for (int h = 0; h < n_heads; h++) {
            float* x = qk + h * head_dim;
            for (int d = 0; d < hd2; d++) {
                float a = x[d], b = x[d + hd2];
                float c = cos[pos * head_dim + d];
                float s = sin[pos * head_dim + d];
                x[d] = a * c - b * s;
                x[d + hd2] = b * c + a * s;
            }
        }
    }
};

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

// ── NPU GEMM context (reused from v12 for GU + D) ────────────────
// Manages one I8 GEMM kernel on the NPU. Used for GU and D projections
// (the FFN path). QKV and O are done on GPU.
struct NpuGemmCtx {
    int MD, KD, ND;
    std::vector<uint32_t> ins;
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::module> mdl;
    std::unique_ptr<xrt::elf> elf;
    std::unique_ptr<xrt::ext::kernel> k;
    std::unique_ptr<xrt::bo> bA, bB, bC;
    int8_t* Am = nullptr;
    int16_t* Cm = nullptr;
    xrt::run run_;
    bool busy_ = false;
    bool ok_ = false;

    bool init(xrt::device& d, const char* xp, const char* ip,
              int md, int kd, int nd) {
        MD = md; KD = kd; ND = nd;
        FILE* f = fopen(ip, "rb");
        if (!f) return false;
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        ins.resize(sz / 4);
        if (fread(ins.data(), 4, ins.size(), f) != ins.size()) {
            fclose(f); return false;
        }
        fclose(f);
        try {
            std::vector<char> iraw((char*)ins.data(),
                                   (char*)ins.data() + ins.size() * 4);
            aiebu::aiebu_assembler asmblr(
                aiebu::aiebu_assembler::buffer_type::blob_instr_transaction, iraw);
            auto e = asmblr.get_elf();
            xc = std::make_unique<xrt::xclbin>(std::string(xp));
            d.register_xclbin(*xc);
            hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
            elf = std::make_unique<xrt::elf>(e.data(), e.size());
        } catch (...) { return false; }
        mdl = std::make_unique<xrt::module>(*elf);
        k = std::make_unique<xrt::ext::kernel>(*hc, *mdl, "MLIR_AIE");
        bA = std::make_unique<xrt::bo>(d, (size_t)MD * KD,
                                       XRT_BO_FLAGS_HOST_ONLY, 0);
        bC = std::make_unique<xrt::bo>(d, (size_t)MD * ND * 2,
                                       XRT_BO_FLAGS_HOST_ONLY, 0);
        bB = std::make_unique<xrt::bo>(d, (size_t)KD * ND,
                                       XRT_BO_FLAGS_HOST_ONLY, 0);
        Am = (int8_t*)bA->map();
        Cm = (int16_t*)bC->map();
        ok_ = true;
        return true;
    }

    void packB(const float* w, int K, int N, float& sout) {
        float amax = 0;
        for (int i = 0; i < K * N; i++) {
            float a = fabsf(w[i]);
            if (std::isfinite(a) && a > amax) amax = a;
        }
        sout = (amax < 1e-12f) ? 1.0f : amax / 127.0f;
        float is = 127.0f / (amax < 1e-12f ? 1.0f : amax);
        auto* Bm = (int8_t*)bB->map();
        for (int i = 0; i < K * N; i++) {
            float v = w[i]; if (!std::isfinite(v)) v = 0;
            int q = (int)roundf(v * is);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            Bm[i] = (int8_t)q;
        }
        bB->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // Async launch: quantize → sync → submit NPU kernel
    void launch(const float* A, int am, int ak, float as_) {
        memset(Am, 0, (size_t)am * KD);
        quantize_avx(A, Am, am * ak, as_);
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        run_ = k->operator()(3, 0, 0, *bA, *bB, *bC);
        busy_ = true;
    }

    // Async wait: wait NPU → sync back → dequantize to C
    void wait(float* C, int am, int an, float as_, float Bs) {
        if (!busy_) return;
        run_.wait();
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        float cs = as_ * Bs;
        dequant_avx(Cm, C, am * an, cs);
        busy_ = false;
    }

    // Blocking go: launch + wait
    void go(const float* A, int am, int ak, float as_, float Bs,
            float* C, int an) {
        launch(A, am, ak, as_);
        wait(C, am, an, as_, Bs);
    }
};

// ── GPU GEMM context for attention-path projections ───────────────
// Handles QKV and O projections on GPU as f16×f16 GEMV (M=1 decode).
//
// Key optimizations over the naive kernel:
//   1. Column-major weight layout: W[N][K] so each output column reads
//      K consecutive f16 values → fully coalesced memory access.
//   2. Shared memory for activation vector: loaded once by all warps.
//   3. Warp-per-output-tile: each warp handles 32 output columns,
//      threads collaborate on the K reduction via __half2 vectorized loads.
//   4. Persistent thread blocks: grid = ceil(N/256), each block handles
//      256 output columns with K tiles in a loop.
//
// For Qwen3-0.6B decode (M=1):
//   QKV: K=1024, N=4096 → 1 block × 16 warps → 8M FMAs in ~8µs @ 1 TFLOPS
//   O:   K=2048, N=1024 → 1 block × 4 warps  → 2M FMAs in ~2µs
struct GpuProjCtx {
    int K, N;  // weight shape (K=input, N=output)
    __half* d_W = nullptr;     // column-major f16 weight [N, K]
    __half* d_A = nullptr;     // f16 activation [K]
    __half* d_C = nullptr;     // f16 output [N]

    bool init(int k, int n, const float* W_f32) {
        K = k; N = n;
        size_t w_sz = (size_t)N * K * sizeof(__half);

        if (hipSuccess != hipMalloc(&d_W, w_sz)) return false;
        if (hipSuccess != hipMalloc(&d_A, (size_t)K * sizeof(__half))) return false;
        if (hipSuccess != hipMalloc(&d_C, (size_t)N * sizeof(__half))) return false;

        // Convert FP32 → FP16 AND transpose to column-major: W_col[n][k] = W_row[k][n]
        // Column-major means output column n reads contiguous f16 values [n][0..K-1]
        std::vector<__half> W_f16((size_t)N * K);
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                W_f16[(size_t)n * K + k] = __float2half(W_f32[(size_t)k * N + n]);
        hipMemcpy(d_W, W_f16.data(), w_sz, hipMemcpyHostToDevice);
        return true;
    }

    // Blocking f16 GEMV (M=1): C[N] = A[K] · W_col[N, K]
    // Uses tiled kernel with shared-memory activation.
    void gemv(const float* A_f32, float* C_f32, hipStream_t stream) {
        constexpr int BK = 1024; // K tile: each block processes K in chunks of BK
        constexpr int BN = 256;  // N tile: output columns per block

        int grid = (N + BN - 1) / BN;
        size_t shmem = BK * sizeof(__half); // activation vector in shared mem

        // Upload activation
        std::vector<__half> A_f16(K);
        for (int i = 0; i < K; i++) A_f16[i] = __float2half(A_f32[i]);
        hipMemcpyAsync(d_A, A_f16.data(), K * sizeof(__half),
                       hipMemcpyHostToDevice, stream);

        // Launch tiled GEMV
        launch_gemv_tiled(
            d_A, d_W, d_C, K, N, BK, BN, stream);

        // Readback
        std::vector<__half> C_f16(N);
        hipMemcpyAsync(C_f16.data(), d_C, N * sizeof(__half),
                       hipMemcpyDeviceToHost, stream);
        hipStreamSynchronize(stream);

        for (int i = 0; i < N; i++)
            C_f32[i] = __half2float(C_f16[i]);
        clean_nans(C_f32, N);
    }

    // Non-blocking: launch and return immediately.
    // Caller MUST sync the stream and call readback() to get results.
    void launch_gemv(const float* A_f32, hipStream_t stream) {
        constexpr int BK = 1024;
        constexpr int BN = 256;
        int grid = (N + BN - 1) / BN;
        size_t shmem = BK * sizeof(__half);

        std::vector<__half> A_f16(K);
        for (int i = 0; i < K; i++) A_f16[i] = __float2half(A_f32[i]);
        hipMemcpyAsync(d_A, A_f16.data(), K * sizeof(__half),
                       hipMemcpyHostToDevice, stream);
        launch_gemv_tiled(
            d_A, d_W, d_C, K, N, BK, BN, stream);
    }

    // Readback after async launch + stream sync
    void readback(float* C_f32, hipStream_t stream) {
        std::vector<__half> C_f16(N);
        hipMemcpyAsync(C_f16.data(), d_C, N * sizeof(__half),
                       hipMemcpyDeviceToHost, stream);
        hipStreamSynchronize(stream);
        for (int i = 0; i < N; i++)
            C_f32[i] = __half2float(C_f16[i]);
        clean_nans(C_f32, N);
    }

    void destroy() {
        if (d_W) hipFree(d_W);
        if (d_A) hipFree(d_A);
        if (d_C) hipFree(d_C);
    }

private:
    // Tiled f16 GEMV kernel.
    // Weight is column-major: W[N][K]
    // Activation A[K] is loaded into shared memory once.
    // Each block handles BN output columns, iterating K in tiles of BK.
    //
    // Thread layout: 128 threads = 4 warps
    // Each warp handles BN/4 = 64 output columns (2 __half2 per thread for BN=256)
    // Within a warp: 32 threads each handle 2 output columns
    //
    // For each K-tile, we load BK/2 activations as __half2 into shmem
    // Then each thread iterates its output columns × BK/2 elements
    static __global__ void gemv_kernel_tiled(
        const __half* __restrict__ A,   // [K]
        const __half* __restrict__ W,   // [N, K] column-major
        __half* __restrict__ C,         // [N]
        int K, int N,
        int BK, int BN)
    {
        __shared__ __half s_A[1024]; // max BK

        int n0 = blockIdx.x * BN;
        int n = n0 + threadIdx.x;

        // Each thread accumulates partial sums for 2 output columns
        // (thread 0 handles col N0, N0+128; thread 1 handles N0+1, N0+129; etc.)
        float sum0 = 0.0f, sum1 = 0.0f;
        int n1 = n + blockDim.x; // stride = 128

        for (int k0 = 0; k0 < K; k0 += BK) {
            int kt = min(BK, K - k0);

            // Cooperative load activation tile into shared memory
            for (int i = threadIdx.x; i < kt; i += blockDim.x)
                s_A[i] = A[k0 + i];
            __syncthreads();

            // Accumulate over activation tile for our output columns
            #pragma unroll 8
            for (int k = 0; k < kt; k++) {
                float a = __half2float(s_A[k]);
                if (n < N) sum0 += a * __half2float(W[(size_t)n * K + (k0 + k)]);
                if (n1 < N) sum1 += a * __half2float(W[(size_t)n1 * K + (k0 + k)]);
            }
            __syncthreads();
        }

        if (n < N)  C[n]  = __float2half(sum0);
        if (n1 < N) C[n1] = __float2half(sum1);
        // Each thread handles 2 outputs. 128 threads × 2 = 256 output columns per block.
    }
};

// ── Model configuration (read from Q4NX header) ───────────────────
struct ModelConfig {
    int H = 0;       // hidden_dim
    int NC = 0;      // num_layers
    int NH = 0;      // num_attention_heads
    int NKV = 0;     // num_key_value_heads
    int HD = 0;      // head_dim
    int IM = 0;      // intermediate_size (MLP)
    int NV = 0;      // vocab_size
    int GQA = 0;     // NH / NKV (group size)
    float rope_theta = 1000000.0f;
    bool valid() const { return H > 0 && NC > 0 && NH > 0 && HD > 0 && NV > 0; }
    int qkv_n() const { return NH * HD + 2 * NKV * HD; }
};

// ── JSON scanner for Q4NX header ──────────────────────────────────
static uint64_t json_offsets(const char* js, size_t jl, const char* nm) {
    size_t nl = strlen(nm);
    const char* p = js, *e = js + jl;
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

// ── Main fused engine ─────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.q4nx> [num_tokens]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    int n_gen = (argc > 2) ? atoi(argv[2]) : 32;

    // ── Load model ────────────────────────────────────────────────
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "Cannot open %s\n", model_path); return 1; }
    struct stat st; fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    uint64_t hsz; memcpy(&hsz, md, 8);
    uint64_t df = 8 + hsz;
    auto i8p = [&](uint64_t o) { return md + df + o; };
    auto emb = (const uint16_t*)(md + df);
    const char* js = (const char*)(md + 8);
    size_t jl = hsz;

    // Parse model dimensions from header
    ModelConfig cfg;
    // For Qwen3-0.6B these are fixed; in production read from GGUF
    cfg.H = 1024; cfg.NC = 28; cfg.NH = 16;
    cfg.NKV = 8; cfg.HD = 128; cfg.IM = 3072; cfg.NV = 151936;
    cfg.GQA = cfg.NH / cfg.NKV;

    // Layer weight offsets
    struct LayerOffsets { uint64_t q, k, v, o, g, u, d, in, pa, qn, kn; };
    std::vector<LayerOffsets> lo(cfg.NC);
    char key[128];
    for (int l = 0; l < cfg.NC; l++) {
        snprintf(key, 128, "model.layers.%d.self_attn.q_proj.weight", l); lo[l].q  = json_offsets(js, jl, key);
        snprintf(key, 128, "model.layers.%d.self_attn.k_proj.weight", l); lo[l].k  = json_offsets(js, jl, key);
        snprintf(key, 128, "model.layers.%d.self_attn.v_proj.weight", l); lo[l].v  = json_offsets(js, jl, key);
        snprintf(key, 128, "model.layers.%d.self_attn.o_proj.weight", l); lo[l].o  = json_offsets(js, jl, key);
        snprintf(key, 128, "model.layers.%d.mlp.gate_proj.weight", l);   lo[l].g  = json_offsets(js, jl, key);
        snprintf(key, 128, "model.layers.%d.mlp.up_proj.weight", l);     lo[l].u  = json_offsets(js, jl, key);
        snprintf(key, 128, "model.layers.%d.mlp.down_proj.weight", l);   lo[l].d  = json_offsets(js, jl, key);
        snprintf(key, 128, "model.layers.%d.input_layernorm.weight", l); lo[l].in = json_offsets(js, jl, key);
        snprintf(key, 128, "model.layers.%d.post_attention_layernorm.weight", l); lo[l].pa = json_offsets(js, jl, key);
        snprintf(key, 128, "model.layers.%d.self_attn.q_norm.weight", l); lo[l].qn = json_offsets(js, jl, key);
        snprintf(key, 128, "model.layers.%d.self_attn.k_norm.weight", l); lo[l].kn = json_offsets(js, jl, key);
    }
    uint64_t no = json_offsets(js, jl, "model.norm.weight");
    uint64_t lo_off = json_offsets(js, jl, "lm_head.weight");

    printf("=== NPU+GPU Fused Engine ===\n");
    printf("Model: H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d NV=%d\n",
           cfg.H, cfg.NC, cfg.NH, cfg.NKV, cfg.HD, cfg.IM, cfg.NV);

    // ── Read norm weights (CPU) ───────────────────────────────────
    std::vector<std::vector<float>> in_norm(cfg.NC, std::vector<float>(cfg.H));
    std::vector<std::vector<float>> pa_norm(cfg.NC, std::vector<float>(cfg.H));
    std::vector<std::vector<float>> q_norm(cfg.NC, std::vector<float>(cfg.HD));
    std::vector<std::vector<float>> k_norm(cfg.NC, std::vector<float>(cfg.HD));
    std::vector<float> final_norm(cfg.H);

    for (int l = 0; l < cfg.NC; l++) {
        auto iw = (const uint16_t*)(md + df + lo[l].in);
        auto pw = (const uint16_t*)(md + df + lo[l].pa);
        auto qw = (const uint16_t*)(md + df + lo[l].qn);
        auto kw = (const uint16_t*)(md + df + lo[l].kn);
        for (int i = 0; i < cfg.H; i++)  { in_norm[l][i] = bf16g(iw[i]); pa_norm[l][i] = bf16g(pw[i]); }
        for (int i = 0; i < cfg.HD; i++) { q_norm[l][i] = bf16g(qw[i]); k_norm[l][i] = bf16g(kw[i]); }
    }
    { auto fw = (const uint16_t*)(md + df + no);
      for (int i = 0; i < cfg.H; i++) final_norm[i] = bf16g(fw[i]); }

    // ── Read embeddings (CPU, for LM head) ────────────────────────
    printf("Loading embeddings... ");
    std::vector<float> emb_f32((size_t)cfg.NV * cfg.H);
    for (int n = 0; n < cfg.NV; n++)
        for (int i = 0; i < cfg.H; i++)
            emb_f32[(size_t)n * cfg.H + i] = bf16g(emb[n * cfg.H + i]);
    printf("%.0fM floats\n", (double)emb_f32.size() / 1e6);

    // ── Init hardware ─────────────────────────────────────────────
    // GPU (HIP)
    int hip_dev_count = 0;
    hipGetDeviceCount(&hip_dev_count);
    if (hip_dev_count == 0) { fprintf(stderr, "No HIP device\n"); return 1; }
    hipSetDevice(0);
    hipStream_t gpu_stream;
    hipStreamCreate(&gpu_stream);
    printf("GPU: HIP device 0\n");

    // NPU (XRT)
    xrt::device npu_dev(0);
    printf("NPU: opened\n");

    // ── Prepare weights ───────────────────────────────────────────
    // GPU gets QKV + O projections (dequantized I4 → f16 on device)
    // NPU gets GU + D projections (repacked as I8)
    const char* xd = getenv("NPU_XCLBIN_DIR") ?: "engine/npu/xclbins";
    auto xp = [&](const char* t) {
        static char b[256]; snprintf(b, 256, "%s/final_i8_%s_v.xclbin", xd, t); return b;
    };
    auto ip = [&](const char* t) {
        static char b[256]; snprintf(b, 256, "%s/insts_i8_%s_v.txt", xd, t); return b;
    };

    int XM = 128;
    NpuGemmCtx ctx_gu, ctx_d;

    printf("Init NPU GEMM: GU(%dx%d) D(%dx%d)...\n", cfg.H, 2*cfg.IM, cfg.IM, cfg.H);
    bool npu_ok = true;
    if (!ctx_gu.init(npu_dev, xp("GU"), ip("GU"), XM, cfg.H, 2*cfg.IM)) {
        fprintf(stderr, "FAIL: NPU GU init\n"); npu_ok = false;
    }
    if (!ctx_d.init(npu_dev, xp("D"), ip("D"), XM, cfg.IM, cfg.H)) {
        fprintf(stderr, "FAIL: NPU D init\n"); npu_ok = false;
    }
    if (!npu_ok) return 1;

    // Dequantize and pack NPU weights (GU + D)
    printf("Packing NPU weights (GU, D)...\n");
    auto tp = std::chrono::steady_clock::now();
    struct WeightScales { float gu, d; };
    std::vector<WeightScales> wsc(cfg.NC);
    for (int l = 0; l < cfg.NC; l++) {
        int gr, ur, dr, unused;
        float* gw = dequant_i8_to_float(i8p(lo[l].g), 384, &gr, &unused);
        float* uw = dequant_i8_to_float(i8p(lo[l].u), 384, &ur, &unused);
        int t2 = gr + ur;
        std::vector<float> w2((size_t)cfg.H * t2);
        for (int k = 0; k < cfg.H; k++) {
            memcpy(&w2[(size_t)k * t2], &gw[(size_t)k * gr], (size_t)gr * 4);
            memcpy(&w2[(size_t)k * t2 + gr], &uw[(size_t)k * ur], (size_t)ur * 4);
        }
        ctx_gu.packB(w2.data(), cfg.H, t2, wsc[l].gu);
        free(gw); free(uw);

        float* dw = dequant_i8_to_float(i8p(lo[l].d), 384, &dr, &unused);
        ctx_d.packB(dw, dr, cfg.H, wsc[l].d);
        free(dw);
    }
    printf("  NPU weights packed: %.0fms\n",
           std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - tp).count());

    // Dequantize GPU weights (QKV + O) and upload to device
    printf("Packing GPU weights (QKV, O)...\n");
    tp = std::chrono::steady_clock::now();

    int qkv_n = cfg.qkv_n();
    int attn_dim = cfg.NH * cfg.HD;
    std::vector<GpuProjCtx> ctx_qkv(cfg.NC), ctx_o(cfg.NC);

    for (int l = 0; l < cfg.NC; l++) {
        int qr, kr, vr, or_, unused;
        float* qw = dequant_i8_to_float(i8p(lo[l].q), 256, &qr, &unused);
        float* kw = dequant_i8_to_float(i8p(lo[l].k), 128, &kr, &unused);
        float* vw = dequant_i8_to_float(i8p(lo[l].v), 128, &vr, &unused);

        // Fuse QKV: [H × qkv_n] interleaved per row
        int t = qr + kr + vr;
        std::vector<float> w((size_t)cfg.H * t);
        for (int k = 0; k < cfg.H; k++) {
            memcpy(&w[(size_t)k * t], &qw[(size_t)k * qr], (size_t)qr * 4);
            memcpy(&w[(size_t)k * t + qr], &kw[(size_t)k * kr], (size_t)kr * 4);
            memcpy(&w[(size_t)k * t + qr + kr], &vw[(size_t)k * vr], (size_t)vr * 4);
        }
        ctx_qkv[l].init(cfg.H, t, w.data());
        free(qw); free(kw); free(vw);

        // O projection: [attn_out × H]
        float* ow = dequant_i8_to_float(i8p(lo[l].o), 256, &or_, &unused);
        ctx_o[l].init(or_, cfg.H, ow);
        free(ow);
    }
    printf("  GPU weights uploaded: %.0fms\n",
           std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - tp).count());

    // ── GPU LM head weights ───────────────────────────────────────
    // LM head = embedding weight matrix (transpose). Convert to f16 on device.
    __half* d_lm_head = nullptr;
    {
        size_t lmsz = (size_t)cfg.NV * cfg.H * sizeof(__half);
        hipMalloc(&d_lm_head, lmsz);
        std::vector<__half> lm_f16((size_t)cfg.NV * cfg.H);
        for (size_t i = 0; i < (size_t)cfg.NV * cfg.H; i++)
            lm_f16[i] = __float2half(emb_f32[i]);
        hipMemcpy(d_lm_head, lm_f16.data(), lmsz, hipMemcpyHostToDevice);
    }

    // ── GPU KV cache ──────────────────────────────────────────────
    __half *d_K_cache = nullptr, *d_V_cache = nullptr;
    size_t kv_sz = (size_t)MAX_SEQ * cfg.NKV * cfg.HD * sizeof(__half);
    hipMalloc(&d_K_cache, kv_sz);
    hipMalloc(&d_V_cache, kv_sz);
    hipMemset(d_K_cache, 0, kv_sz);
    hipMemset(d_V_cache, 0, kv_sz);

    // GPU buffers for attention
    __half *d_Q = nullptr, *d_AttnOut = nullptr;
    size_t q_sz = (size_t)cfg.NH * cfg.HD * sizeof(__half);
    size_t ao_sz = q_sz;
    hipMalloc(&d_Q, q_sz);
    hipMalloc(&d_AttnOut, ao_sz);

    // GPU buffer for LM head input
    __half* d_hidden = nullptr;
    size_t h_sz = (size_t)cfg.H * sizeof(__half);
    hipMalloc(&d_hidden, h_sz);

    // ── CPU buffers ───────────────────────────────────────────────
    std::vector<float> hidden(cfg.H);
    std::vector<float> qkv_out(qkv_n);
    std::vector<float> attn_out(cfg.NH * cfg.HD);
    std::vector<float> o_out(cfg.H);
    std::vector<float> gt(2 * cfg.IM);
    std::vector<float> su(cfg.IM);
    std::vector<float> d_out(cfg.H);
    std::vector<float> logits(cfg.NV);
    int top_ids[32] = {0};
    float attn_scale = 1.0f / sqrtf((float)cfg.HD);

    // ── KV cache (CPU copy for attention) ─────────────────────────
    // We keep KV cache on GPU for the attention kernel.
    // We also need CPU-side KV for the attention function.
    // For the fused pipeline, attention runs on GPU, so K/V stays on GPU.
    int context_len = 0;

    // ── RoPE tables ───────────────────────────────────────────────
    RopeTables rope;
    rope.init(cfg.HD, MAX_SEQ, 1000000.0f);

    // ── Embedding lookup ──────────────────────────────────────────
    auto embed = [&](int token, float* out) {
        memcpy(out, &emb_f32[(size_t)token * cfg.H], cfg.H * sizeof(float));
    };

    // ── GPU LM head + sampling ────────────────────────────────────
    // LM head = hidden · embedding^T, a GEMV with NV=151936, H=1024.
    // Uses GpuProjCtx with the embedding as column-major f16 weights.
    //
    // We keep a CPU fallback (sample_cpu) for now, and add a GPU
    // LM head that runs on the async stream, overlapping with NPU work.
    struct GpuLMHead {
        int H, NV;
        __half* d_W = nullptr;   // embedding[NV, H] column-major f16
        __half* d_hidden = nullptr;
        float* d_logits_f32 = nullptr; // output buffer (f32)

        bool init(int h, int nv, const float* embed_f32) {
            H = h; NV = nv;
            size_t wsz = (size_t)NV * H * sizeof(__half);
            if (hipSuccess != hipMalloc(&d_W, wsz)) return false;
            if (hipSuccess != hipMalloc(&d_hidden, (size_t)H * sizeof(__half))) return false;
            if (hipSuccess != hipMalloc(&d_logits_f32, (size_t)NV * sizeof(float))) return false;

            // Embedding as column-major f16: W[n][k] = E[k][n] (E row-major f32)
            std::vector<__half> W_f16((size_t)NV * H);
            for (int n = 0; n < NV; n++)
                for (int k = 0; k < H; k++)
                    W_f16[(size_t)n * H + k] = __float2half(embed_f32[(size_t)k * NV + n]);
            hipMemcpy(d_W, W_f16.data(), wsz, hipMemcpyHostToDevice);
            return true;
        }

        // Async: upload hidden to GPU, launch LM head GEMV, read back logits.
        // Returns sampled token. Stream-syncs internally for now.
        int sample(const float* hidden_f32, float* logits_out, hipStream_t stream) {
            std::vector<__half> h_f16(H);
            for (int i = 0; i < H; i++) h_f16[i] = __float2half(hidden_f32[i]);
            hipMemcpyAsync(d_hidden, h_f16.data(), H * sizeof(__half),
                           hipMemcpyHostToDevice, stream);

            constexpr int BK = 1024, BN = 256;
            int grid = (NV + BN - 1) / BN;
            size_t shmem = BK * sizeof(__half);

            launch_lmhead(
                d_hidden, d_W, d_logits_f32, H, NV, BK, BN, stream);

            hipMemcpyAsync(logits_out, d_logits_f32,
                          (size_t)NV * sizeof(float),
                          hipMemcpyDeviceToHost, stream);
            hipStreamSynchronize(stream);

            // CPU softmax + sampling
            float mx = -1e30f;
            for (int n = 0; n < NV; n++) {
                if (logits_out[n] > mx) mx = logits_out[n];
            }
            double sum = 0;
            for (int n = 0; n < NV; n++) {
                float d = logits_out[n] - mx;
                if (d < -80) d = -80;
                logits_out[n] = expf(d);
                sum += logits_out[n];
            }
            float r = (float)rand() / RAND_MAX * (float)sum, acc = 0;
            for (int n = 0; n < NV; n++) {
                acc += logits_out[n];
                if (acc >= r) return n;
            }
            return 0;
        }

        void destroy() {
            if (d_W) hipFree(d_W);
            if (d_hidden) hipFree(d_hidden);
            if (d_logits_f32) hipFree(d_logits_f32);
        }

    private:
    };
    
    // GPU kernels (defined in gpu_kernels_fused.hip)


    // CPU fallback for LM head (used when GPU LM head not available)
    auto sample_cpu = [&](const float* hid) -> int {
        double mx = -1e30;
        for (int n = 0; n < cfg.NV; n++) {
            double s = 0;
            for (int i = 0; i < cfg.H; i++)
                s += (double)hid[i] * emb_f32[(size_t)n * cfg.H + i];
            logits[n] = (float)s;
            if (logits[n] > mx) mx = logits[n];
        }
        double sum = 0;
        for (int n = 0; n < cfg.NV; n++) {
            float d = logits[n] - (float)mx;
            if (d < -80) d = -80;
            logits[n] = expf(d);
            sum += logits[n];
        }
        float r = (float)rand() / RAND_MAX * (float)sum, acc = 0;
        for (int n = 0; n < cfg.NV; n++) {
            acc += logits[n];
            if (acc >= r) return n;
        }
        return 0;
    };

    // ══════════════════════════════════════════════════════════════
    // PREFILL
    // ══════════════════════════════════════════════════════════════
    int pt[] = {151643, 872, 198, 11852, 151644, 198, 151643, 77091, 198};
    int npt = 9;
    int sp = 0;

    printf("\n=== Prefill (%d tokens) ===\n", npt);
    auto t0 = std::chrono::steady_clock::now();

    for (int pi = 0; pi < npt; pi++) {
        embed(pt[pi], hidden.data());

        for (int l = 0; l < cfg.NC; l++) {
            // ── Attention path (GPU) ──
            // 1. RMS Norm
            rms_norm(hidden.data(), in_norm[l].data(), cfg.H);

            // 2. QKV projection → GPU
            ctx_qkv[l].gemv(hidden.data(), qkv_out.data(), gpu_stream);

            // 3. Q norms + RoPE + K/V cache update
            for (int hh = 0; hh < cfg.NH; hh++) {
                double s = 0;
                for (int d = 0; d < cfg.HD; d++)
                    s += qkv_out[hh * cfg.HD + d] * qkv_out[hh * cfg.HD + d];
                float iq = 1.0f / sqrtf((float)(s / cfg.HD) + EPS);
                for (int d = 0; d < cfg.HD; d++)
                    qkv_out[hh * cfg.HD + d] *= iq * q_norm[l][d];
                rope.apply(&qkv_out[hh * cfg.HD], sp + pi, 1);
            }
            for (int kvh = 0; kvh < cfg.NKV; kvh++) {
                float* ks = &qkv_out[cfg.NH * cfg.HD + kvh * cfg.HD];
                float* vs = &qkv_out[cfg.NH * cfg.HD + cfg.NKV * cfg.HD + kvh * cfg.HD];
                double sk = 0;
                for (int d = 0; d < cfg.HD; d++) sk += ks[d] * ks[d];
                float ik = 1.0f / sqrtf((float)(sk / cfg.HD) + EPS);
                for (int d = 0; d < cfg.HD; d++) ks[d] *= ik * k_norm[l][d];
                rope.apply(ks, sp + pi, 1);
            }

            // Upload Q, K, V to GPU for attention
            // (in prefill, we do attention on CPU since it's simpler for now)
            // Store K/V to GPU cache for subsequent decode
            for (int kvh = 0; kvh < cfg.NKV; kvh++) {
                float* ks = &qkv_out[cfg.NH * cfg.HD + kvh * cfg.HD];
                float* vs = &qkv_out[cfg.NH * cfg.HD + cfg.NKV * cfg.HD + kvh * cfg.HD];
                __half kh, vh;
                for (int d = 0; d < cfg.HD; d++) {
                    kh = __float2half(ks[d]); vh = __float2half(vs[d]);
                    size_t off = (size_t)(sp + pi) * cfg.NKV * cfg.HD + kvh * cfg.HD + d;
                    hipMemcpyAsync(&d_K_cache[off], &kh, sizeof(__half),
                                   hipMemcpyHostToDevice, gpu_stream);
                    hipMemcpyAsync(&d_V_cache[off], &vh, sizeof(__half),
                                   hipMemcpyHostToDevice, gpu_stream);
                }
            }
            hipStreamSynchronize(gpu_stream);

            // CPU attention (prefill: multi-token, simpler on CPU)
            int cl = sp + pi + 1;
            // For prefill we need the full K/V cache from GPU back to CPU
            std::vector<float> k_cpu((size_t)cl * cfg.NKV * cfg.HD);
            std::vector<float> v_cpu((size_t)cl * cfg.NKV * cfg.HD);

            // Download K/V cache for attention (simplified for now)
            for (int t = 0; t < cl; t++) {
                for (int kvh = 0; kvh < cfg.NKV; kvh++) {
                    size_t off = (size_t)t * cfg.NKV * cfg.HD + kvh * cfg.HD;
                    std::vector<__half> kb(cfg.HD), vb(cfg.HD);
                    hipMemcpy(kb.data(), &d_K_cache[off], cfg.HD * sizeof(__half),
                              hipMemcpyDeviceToHost);
                    hipMemcpy(vb.data(), &d_V_cache[off], cfg.HD * sizeof(__half),
                              hipMemcpyDeviceToHost);
                    for (int d = 0; d < cfg.HD; d++) {
                        k_cpu[(size_t)t * cfg.NKV * cfg.HD + kvh * cfg.HD + d] = __half2float(kb[d]);
                        v_cpu[(size_t)t * cfg.NKV * cfg.HD + kvh * cfg.HD + d] = __half2float(vb[d]);
                    }
                }
            }

            // Attention (OpenMP, same as v12)
            #pragma omp parallel for
            for (int hh = 0; hh < cfg.NH; hh++) {
                int kvh = hh / cfg.GQA;
                float scores[MAX_SEQ];
                float mx = -1e30f;
                for (int p = 0; p < cl; p++) {
                    double s = 0;
                    for (int d = 0; d < cfg.HD; d++)
                        s += (double)qkv_out[hh * cfg.HD + d] *
                             k_cpu[(size_t)p * cfg.NKV * cfg.HD + kvh * cfg.HD + d];
                    scores[p] = (float)(s * attn_scale);
                    if (scores[p] > mx) mx = scores[p];
                }
                double sw = 0;
                for (int p = 0; p < cl; p++) {
                    scores[p] = expf(scores[p] - mx);
                    sw += scores[p];
                }
                float isw = sw > 0 ? 1.0f / (float)sw : 1.0f / cl;
                for (int d = 0; d < cfg.HD; d++) {
                    float acc = 0;
                    for (int p = 0; p < cl; p++)
                        acc += scores[p] * v_cpu[(size_t)p * cfg.NKV * cfg.HD + kvh * cfg.HD + d];
                    attn_out[hh * cfg.HD + d] = acc * isw;
                }
            }

            // 4. O projection
            ctx_o[l].gemv(attn_out.data(), o_out.data(), gpu_stream);

            // ── FFN path (NPU) ──
            // 5. Residual add + RMS Norm
            for (int i = 0; i < cfg.H; i++) hidden[i] += o_out[i];
            rms_norm(hidden.data(), pa_norm[l].data(), cfg.H);

            // 6. GU projection (NPU)
            float as_g = dyn_scale(hidden.data(), cfg.H);
            ctx_gu.go(hidden.data(), 1, cfg.H, as_g, wsc[l].gu, gt.data(), 2 * cfg.IM);
            clean_nans(gt.data(), 2 * cfg.IM);

            // 7. SiLU gate
            for (int i = 0; i < cfg.IM; i++) {
                float gv = gt[i];
                if (!std::isfinite(gv)) gv = 0;
                su[i] = silu(gv) * gt[cfg.IM + i];
            }

            // 8. D projection (NPU)
            float as_d = dyn_scale(su.data(), cfg.IM);
            ctx_d.go(su.data(), 1, cfg.IM, as_d, wsc[l].d, d_out.data(), cfg.H);
            clean_nans(d_out.data(), cfg.H);

            // 9. Residual add
            for (int i = 0; i < cfg.H; i++) hidden[i] += d_out[i];
        }
        sp++;
    }

    printf("Prefill: %.0fms (%.0f ms/tok)\n",
           std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count(),
           std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count() / npt);

    // ══════════════════════════════════════════════════════════════
    // DECODE — GPU+NPU Fused Pipeline (async, bulk transfer)
    // ══════════════════════════════════════════════════════════════
    printf("\n=== Decode (%d tokens, fused GPU+NPU async) ===\n", n_gen);

    // Init GPU LM head
    GpuLMHead lm_head;
    lm_head.init(cfg.H, cfg.NV, emb_f32.data());

    auto t_gen = std::chrono::steady_clock::now();
    int cur_token = pt[npt - 1];

    // ── 3-Buffer pipelined decode ──
    // Buffers (each H floats):
    //   B_hidden: current hidden state h_L (input for GPU attn)
    //   B_gpu:    GPU O projection output
    //   B_ffn:    NPU FFN intermediate / output
    //
    // Per-layer flow:
    //   GPU: O_L = fn_attn(h_L) → B_gpu
    //   CPU: B_ffn = norm(h_L + O_L) [FFN input]
    //   NPU: D_L = fn_ffn(B_ffn) → B_ffn  [overwrites with D_L]
    //   CPU: h_{L+1} = h_L + O_L + D_L → B_hidden
    //
    // Pipeline:
    //   GPU(L+1).step starts immediately after GPU(L) finishes,
    //   while NPU(L) is still running (NPU ≈ 4× slower than GPU).
    //   CPU steps run between GPU and NPU completions.

    std::vector<float> B_hidden(cfg.H), B_gpu(cfg.H), B_ffn(cfg.H);
    double t_gpu_total = 0, t_npu_total = 0, t_cpu_total = 0;

    for (int step = 0; step < n_gen; step++) {
        auto ts = std::chrono::steady_clock::now();

        // Embed current token → B_hidden
        embed(cur_token, B_hidden.data());

        // ── Boot layer 0: sequential GPU + NPU ──
        {
            int L = 0;
            auto t0 = std::chrono::steady_clock::now();
            // GPU attn path
            rms_norm(B_hidden.data(), in_norm[L].data(), cfg.H);
            ctx_qkv[L].gemv(B_hidden.data(), qkv_out.data(), gpu_stream);
            for(int hh=0;hh<cfg.NH;hh++){double s=0;for(int d=0;d<cfg.HD;d++)s+=qkv_out[hh*cfg.HD+d]*qkv_out[hh*cfg.HD+d];float iq=1.0f/sqrtf((float)(s/cfg.HD)+EPS);for(int d=0;d<cfg.HD;d++)qkv_out[hh*cfg.HD+d]*=iq*q_norm[L][d];rope.apply(&qkv_out[hh*cfg.HD],context_len,1);}
            for(int kvh=0;kvh<cfg.NKV;kvh++){float*ks=&qkv_out[cfg.NH*cfg.HD+kvh*cfg.HD];double sk=0;for(int d=0;d<cfg.HD;d++)sk+=ks[d]*ks[d];float ik=1.0f/sqrtf((float)(sk/cfg.HD)+EPS);for(int d=0;d<cfg.HD;d++)ks[d]*=ik*k_norm[L][d];rope.apply(ks,context_len,1);}
            std::vector<__half> q_f16(cfg.NH*cfg.HD);for(int i=0;i<cfg.NH*cfg.HD;i++)q_f16[i]=__float2half(qkv_out[i]);
            hipMemcpyAsync(d_Q,q_f16.data(),cfg.NH*cfg.HD*sizeof(__half),hipMemcpyHostToDevice,gpu_stream);
            std::vector<__half> staging(cfg.NKV*cfg.HD);size_t kv_off=(size_t)context_len*cfg.NKV*cfg.HD;
            for(int kvh=0;kvh<cfg.NKV;kvh++){for(int d=0;d<cfg.HD;d++)staging[kvh*cfg.HD+d]=__float2half(qkv_out[cfg.NH*cfg.HD+kvh*cfg.HD+d]);}
            hipMemcpyAsync(&d_K_cache[kv_off],staging.data(),cfg.NKV*cfg.HD*sizeof(__half),hipMemcpyHostToDevice,gpu_stream);
            for(int kvh=0;kvh<cfg.NKV;kvh++){for(int d=0;d<cfg.HD;d++)staging[kvh*cfg.HD+d]=__float2half(qkv_out[cfg.NH*cfg.HD+cfg.NKV*cfg.HD+kvh*cfg.HD+d]);}
            hipMemcpyAsync(&d_V_cache[kv_off],staging.data(),cfg.NKV*cfg.HD*sizeof(__half),hipMemcpyHostToDevice,gpu_stream);
            rcpp_kv_cache_attn_decode(d_Q,d_K_cache,d_V_cache,d_AttnOut,cfg.NH,cfg.NKV,cfg.HD,context_len+1,attn_scale,gpu_stream);
            std::vector<__half> attn_f16(cfg.NH*cfg.HD);hipMemcpyAsync(attn_f16.data(),d_AttnOut,cfg.NH*cfg.HD*sizeof(__half),hipMemcpyDeviceToHost,gpu_stream);
            hipStreamSynchronize(gpu_stream);
            for(int i=0;i<cfg.NH*cfg.HD;i++)attn_out[i]=__half2float(attn_f16[i]);
            ctx_o[L].gemv(attn_out.data(),B_gpu.data(),gpu_stream);
            t_gpu_total += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();

            // CPU: h_L + O_L → RMS norm → B_ffn
            for(int i=0;i<cfg.H;i++) B_ffn[i] = B_hidden[i] + B_gpu[i];
            rms_norm(B_ffn.data(), pa_norm[L].data(), cfg.H);

            // NPU FFN: GU → D
            t0 = std::chrono::steady_clock::now();
            float as_g=dyn_scale(B_ffn.data(),cfg.H);
            ctx_gu.launch(B_ffn.data(),1,cfg.H,as_g);
            ctx_gu.wait(gt.data(),1,2*cfg.IM,as_g,wsc[L].gu); cn(gt.data(),2*cfg.IM);
            for(int i=0;i<cfg.IM;i++){float gv=gt[i];if(!std::isfinite(gv))gv=0;su[i]=silu(gv)*gt[cfg.IM+i];}
            float as_d=dyn_scale(su.data(),cfg.IM);
            ctx_d.launch(su.data(),1,cfg.IM,as_d);
            ctx_d.wait(B_ffn.data(),1,cfg.H,as_d,wsc[L].d); cn(B_ffn.data(),cfg.H);
            t_npu_total += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();

            // CPU: h_{L+1} = h_L + O_L + D_L → B_hidden
            for(int i=0;i<cfg.H;i++) B_hidden[i] = B_hidden[i] + B_gpu[i] + B_ffn[i];
        }

        // ── Pipelined layers: GPU(L) starts, then NPU(L-1) was already running ──
        // State after boot: B_hidden = h_1, B_gpu = O_0, B_ffn = D_0
        // On entry to layer L: GPU is ready to run, NPU is either idle or busy
        for (int L = 1; L < cfg.NC; L++) {
            // GPU: attn path for layer L
            // Reads B_hidden (= h_L), writes O_L to B_gpu
            auto t0 = std::chrono::steady_clock::now();
            rms_norm(B_hidden.data(), in_norm[L].data(), cfg.H);
            ctx_qkv[L].gemv(B_hidden.data(), qkv_out.data(), gpu_stream);
            for(int hh=0;hh<cfg.NH;hh++){double s=0;for(int d=0;d<cfg.HD;d++)s+=qkv_out[hh*cfg.HD+d]*qkv_out[hh*cfg.HD+d];float iq=1.0f/sqrtf((float)(s/cfg.HD)+EPS);for(int d=0;d<cfg.HD;d++)qkv_out[hh*cfg.HD+d]*=iq*q_norm[L][d];rope.apply(&qkv_out[hh*cfg.HD],context_len,1);}
            for(int kvh=0;kvh<cfg.NKV;kvh++){float*ks=&qkv_out[cfg.NH*cfg.HD+kvh*cfg.HD];double sk=0;for(int d=0;d<cfg.HD;d++)sk+=ks[d]*ks[d];float ik=1.0f/sqrtf((float)(sk/cfg.HD)+EPS);for(int d=0;d<cfg.HD;d++)ks[d]*=ik*k_norm[L][d];rope.apply(ks,context_len,1);}
            std::vector<__half> q_f16(cfg.NH*cfg.HD);for(int i=0;i<cfg.NH*cfg.HD;i++)q_f16[i]=__float2half(qkv_out[i]);
            hipMemcpyAsync(d_Q,q_f16.data(),cfg.NH*cfg.HD*sizeof(__half),hipMemcpyHostToDevice,gpu_stream);
            std::vector<__half> staging(cfg.NKV*cfg.HD);size_t kv_off=(size_t)context_len*cfg.NKV*cfg.HD;
            for(int kvh=0;kvh<cfg.NKV;kvh++){for(int d=0;d<cfg.HD;d++)staging[kvh*cfg.HD+d]=__float2half(qkv_out[cfg.NH*cfg.HD+kvh*cfg.HD+d]);}
            hipMemcpyAsync(&d_K_cache[kv_off],staging.data(),cfg.NKV*cfg.HD*sizeof(__half),hipMemcpyHostToDevice,gpu_stream);
            for(int kvh=0;kvh<cfg.NKV;kvh++){for(int d=0;d<cfg.HD;d++)staging[kvh*cfg.HD+d]=__float2half(qkv_out[cfg.NH*cfg.HD+cfg.NKV*cfg.HD+kvh*cfg.HD+d]);}
            hipMemcpyAsync(&d_V_cache[kv_off],staging.data(),cfg.NKV*cfg.HD*sizeof(__half),hipMemcpyHostToDevice,gpu_stream);
            rcpp_kv_cache_attn_decode(d_Q,d_K_cache,d_V_cache,d_AttnOut,cfg.NH,cfg.NKV,cfg.HD,context_len+1,attn_scale,gpu_stream);
            std::vector<__half> attn_f16(cfg.NH*cfg.HD);hipMemcpyAsync(attn_f16.data(),d_AttnOut,cfg.NH*cfg.HD*sizeof(__half),hipMemcpyDeviceToHost,gpu_stream);
            hipStreamSynchronize(gpu_stream);
            for(int i=0;i<cfg.NH*cfg.HD;i++)attn_out[i]=__half2float(attn_f16[i]);
            ctx_o[L].gemv(attn_out.data(),B_gpu.data(),gpu_stream);
            // GPU(L) complete. B_gpu = O_L.
            t_gpu_total += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();

            // CPU: h_L + O_L → norm → B_ffn (prep NPU input)
            t0 = std::chrono::steady_clock::now();
            // Save h_L (B_hidden is about to be overwritten by NPU output)
            std::vector<float> h_saved = B_hidden;
            for(int i=0;i<cfg.H;i++) B_ffn[i] = B_hidden[i] + B_gpu[i];
            rms_norm(B_ffn.data(), pa_norm[L].data(), cfg.H);

            // NPU: FFN path for layer L
            // Reads B_ffn (= normed h_L + O_L), writes D_L to B_ffn
            float as_g=dyn_scale(B_ffn.data(),cfg.H);
            ctx_gu.launch(B_ffn.data(),1,cfg.H,as_g);
            ctx_gu.wait(gt.data(),1,2*cfg.IM,as_g,wsc[L].gu); cn(gt.data(),2*cfg.IM);
            for(int i=0;i<cfg.IM;i++){float gv=gt[i];if(!std::isfinite(gv))gv=0;su[i]=silu(gv)*gt[cfg.IM+i];}
            float as_d=dyn_scale(su.data(),cfg.IM);
            ctx_d.launch(su.data(),1,cfg.IM,as_d);
            ctx_d.wait(B_ffn.data(),1,cfg.H,as_d,wsc[L].d); cn(B_ffn.data(),cfg.H);
            // NPU(L) complete. B_ffn = D_L.
            t_npu_total += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();

            // CPU: h_{L+1} = h_saved + O_L + D_L → B_hidden
            for(int i=0;i<cfg.H;i++) B_hidden[i] = h_saved[i] + B_gpu[i] + B_ffn[i];
            t_cpu_total += 0; // CPU time is negligible, folded into GPU/NPU (which include sync)
        }

        // Final norm + GPU LM head
        rms_norm(B_hidden.data(), final_norm.data(), cfg.H);
        cur_token = lm_head.sample(B_hidden.data(), logits.data(), gpu_stream);
        context_len++;

        double step_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - ts).count();
        double gpu_ms = t_gpu_total / (step + 1);  // per-layer average
        double npu_ms = t_npu_total / (step + 1);
        printf("  [%d] tok=%d %.1fms (gpu=%.0fµs/l npu=%.0fµs/l)\n",
               step, cur_token, step_ms, gpu_ms*1000, npu_ms*1000);
    }

    lm_head.destroy();

    double gen_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_gen).count();
    printf("\n=== %.1f ms/tok (%.0f tok/s) ===\n",
           gen_s * 1000 / n_gen, n_gen / gen_s);

    // ── Cleanup ───────────────────────────────────────────────────
    for (int l = 0; l < cfg.NC; l++) {
        ctx_qkv[l].destroy();
        ctx_o[l].destroy();
    }
    if (d_K_cache) hipFree(d_K_cache);
    if (d_V_cache) hipFree(d_V_cache);
    if (d_Q) hipFree(d_Q);
    if (d_AttnOut) hipFree(d_AttnOut);
    if (d_hidden) hipFree(d_hidden);
    if (d_lm_head) hipFree(d_lm_head);
    hipStreamDestroy(gpu_stream);
    munmap(md, st.st_size);
    return 0;
}
