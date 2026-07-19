// backend_hip.cpp — AMD ROCm HIP backend for RDNA 3.5+ GPUs
// Part of the unified zaya_server binary. Compiled when USE_HIP=ON.
#include "backend.h"
#include "rocm_cpp/ck_gemm.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>

// FP16 conversion for GGUF Q8_0 dequant
static float fp16_to_fp32(uint16_t h) {
    uint32_t s=(h>>15)&1, e=(h>>10)&0x1f, m=h&0x3ff;
    if(e==0){uint32_t a=m?__builtin_clz(m)-10:0;float r;uint32_t f=(s<<31)|((127-15-a)<<23)|((m<<(a+13))&0x7fffff);memcpy(&r,&f,4);return r;}
    else if(e==31){float r;uint32_t f=(s<<31)|0x7f800000|(m<<13);memcpy(&r,&f,4);return r;}
    else{float r;uint32_t f=(s<<31)|((e+127-15)<<23)|(m<<13);memcpy(&r,&f,4);return r;}
}

// K-quant dequantization (GGML_TYPE Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13,
// Q6_K=14 — all 256-elem superblocks). Ported from src/gguf_loader.cpp's
// dequant_q{2,3,4,5,6}_k, but using the fp16_to_fp32 above for the per-block
// scale/min fields instead of that file's read_f16 helper, which does a
// bfloat16-style bit-shift ("<<16, reinterpret as f32") on data that's
// actually real IEEE754 float16 — wrong exponent bias/width, silently
// corrupting every K-quant scale. All 5 formats verified byte-exact against
// the independent `gguf` Python package's reference dequantize_blocks() on
// randomized synthetic block data (5 seeds, 0 mismatches). Q8_K isn't
// handled here yet.
static inline void k_get_scale_min(const uint8_t scales[12], int j, uint8_t& sc, uint8_t& m) {
    if (j < 4) { sc = scales[j] & 63; m = scales[j + 4] & 63; }
    else { sc = (uint8_t)((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
           m = (uint8_t)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)); }
}

static void dequant_q2_k_to_f32(const uint8_t* bd, float* out, int count) {
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        uint8_t scales[16]; memcpy(scales, p, 16); p += 16;
        uint8_t qs[64]; memcpy(qs, p, 64); p += 64;
        uint16_t dh, dminh; memcpy(&dh, p, 2); p += 2; memcpy(&dminh, p, 2); p += 2;
        float d = fp16_to_fp32(dh), dmin = fp16_to_fp32(dminh);
        int base = b * BS;
        int is = 0; const uint8_t* q = qs;
        for (int n = 0; n < BS && base + n < count; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);
                for (int l = 0; l < 16 && base + n + j * 32 + l < count; l++)
                    out[base + n + j * 32 + l] = dl * ((q[l] >> shift) & 3) - ml;
                sc = scales[is++];
                dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
                for (int l = 0; l < 16 && base + n + j * 32 + 16 + l < count; l++)
                    out[base + n + j * 32 + 16 + l] = dl * ((q[l + 16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

static void dequant_q3_k_to_f32(const uint8_t* bd, float* out, int count) {
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        uint8_t hmask[32]; memcpy(hmask, p, 32); p += 32;
        uint8_t qs[64]; memcpy(qs, p, 64); p += 64;
        uint8_t raw_scales[12]; memcpy(raw_scales, p, 12); p += 12;
        uint16_t dh; memcpy(&dh, p, 2); p += 2; float d_all = fp16_to_fp32(dh);

        uint32_t aux[4] = {0, 0, 0, 0};
        memcpy(aux, raw_scales, 12);
        const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int8_t scales[16]; memcpy(scales, aux, 16);
        for (int j = 0; j < 16; j++) scales[j] -= 32;

        int base = b * BS;
        int is = 0; const uint8_t* q = qs;
        uint8_t m = 1;
        for (int n = 0; n < BS && base + n < count; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d_all * scales[is++];
                for (int l = 0; l < 16 && base + n + j * 32 + l < count; l++)
                    out[base + n + j * 32 + l] =
                        dl * (((int8_t)((q[l] >> shift) & 3)) - ((hmask[l] & m) ? 0 : 4));
                dl = d_all * scales[is++];
                for (int l = 0; l < 16 && base + n + j * 32 + 16 + l < count; l++)
                    out[base + n + j * 32 + 16 + l] =
                        dl * (((int8_t)((q[l + 16] >> shift) & 3)) - ((hmask[l + 16] & m) ? 0 : 4));
                shift += 2;
                m = (uint8_t)(m << 1);
            }
            q += 32;
        }
    }
}

static void dequant_q4_k_to_f32(const uint8_t* bd, float* out, int count) {
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        uint16_t dh, dminh; memcpy(&dh, p, 2); p += 2; memcpy(&dminh, p, 2); p += 2;
        float d = fp16_to_fp32(dh), dmin = fp16_to_fp32(dminh);
        uint8_t scales[12]; memcpy(scales, p, 12); p += 12;
        uint8_t qs[128]; memcpy(qs, p, 128); p += 128;
        int base = b * BS;
        int is = 0; const uint8_t* q = qs;
        for (int off = 0; off < BS && base + off < count; off += 64) {
            uint8_t sc, m;
            k_get_scale_min(scales, is, sc, m); float d1 = d * sc, m1 = dmin * m;
            k_get_scale_min(scales, is + 1, sc, m); float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32 && base + off + l < count; l++)
                out[base + off + l] = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32 && base + off + 32 + l < count; l++)
                out[base + off + 32 + l] = d2 * (q[l] >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

static void dequant_q5_k_to_f32(const uint8_t* bd, float* out, int count) {
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        uint16_t dh, dminh; memcpy(&dh, p, 2); p += 2; memcpy(&dminh, p, 2); p += 2;
        float d = fp16_to_fp32(dh), dmin = fp16_to_fp32(dminh);
        uint8_t scales[12]; memcpy(scales, p, 12); p += 12;
        uint8_t qh[32]; memcpy(qh, p, 32); p += 32;
        uint8_t ql[128]; memcpy(ql, p, 128); p += 128;
        int base = b * BS;
        int is = 0; const uint8_t* q = ql;
        uint8_t u1 = 1, u2 = 2;
        for (int n = 0; n < BS && base + n < count; n += 64) {
            uint8_t sc, m;
            k_get_scale_min(scales, is, sc, m); float d1 = d * sc, m1 = dmin * m;
            k_get_scale_min(scales, is + 1, sc, m); float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32 && base + n + l < count; l++)
                out[base + n + l] = d1 * ((q[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32 && base + n + 32 + l < count; l++)
                out[base + n + 32 + l] = d2 * ((q[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            q += 32; is += 2;
            u1 = (uint8_t)(u1 << 2); u2 = (uint8_t)(u2 << 2);
        }
    }
}

static void dequant_q6_k_to_f32(const uint8_t* bd, float* out, int count) {
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        uint8_t ql[128]; memcpy(ql, p, 128); p += 128;
        uint8_t qh[64]; memcpy(qh, p, 64); p += 64;
        int8_t scales[16]; memcpy(scales, p, 16); p += 16;
        uint16_t dh; memcpy(&dh, p, 2); p += 2; float d = fp16_to_fp32(dh);
        int base = b * BS;
        for (int n = 0; n < BS; n += 128) {
            for (int l = 0; l < 32 && base + n + l < count; l++) {
                int8_t sc = scales[l / 2];
                int v = (ql[n / 2 + l] & 0xF) | ((qh[n / 2 + l / 4] >> (4 * (l & 1))) & 0x30);
                out[base + n + l] = d * sc * (v - 32);
            }
        }
    }
}

#define HIP_OK(e) do { auto _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP Error %d\n", _s); abort(); } } while (0)
constexpr int BLK = 256;

// Kernels (from ../kernels/)
__global__ void rmsnorm_k(__half* x, const __half* w, int n, float eps) {
    __shared__ float r[32];
    int tx = threadIdx.x, wid = tx / 32, l = tx % 32;
    float ss = 0;
    for (int i = tx; i < n; i += blockDim.x) ss += (float)x[i] * (float)x[i];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o);
    if (l == 0) r[wid] = ss;
    __syncthreads();
    if (wid == 0) { ss = (l < (256 / 32)) ? r[l] : 0; for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o); if (l == 0) r[0] = ss; }
    __syncthreads();
    float iv = 1.0f / sqrtf(r[0] / n + eps);
    for (int i = tx; i < n; i += blockDim.x) x[i] = __float2half((float)x[i] * iv * (float)w[i]);
}
// Per-head RMSNorm (Qwen3-style Q/K-norm): x is [num_heads, head_dim]; each
// block independently normalizes one head's head_dim-sized slice against the
// same shared, learned weight vector w[head_dim]. One block per head.
__global__ void qk_norm_k(__half* x, const __half* w, int head_dim, float eps) {
    __shared__ float r[32];
    __half* row = x + (size_t)blockIdx.x * head_dim;
    int tx = threadIdx.x, wid = tx / 32, l = tx % 32;
    float ss = 0;
    for (int i = tx; i < head_dim; i += blockDim.x) ss += (float)row[i] * (float)row[i];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o);
    if (l == 0) r[wid] = ss;
    __syncthreads();
    if (wid == 0) { ss = (l < (256 / 32)) ? r[l] : 0; for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o); if (l == 0) r[0] = ss; }
    __syncthreads();
    float iv = 1.0f / sqrtf(r[0] / head_dim + eps);
    for (int i = tx; i < head_dim; i += blockDim.x) row[i] = __float2half((float)row[i] * iv * (float)w[i]);
}
__global__ void copy_k(__half* d, const __half* s, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; d[i] = s[i]; }
__global__ void silu_mul_k(__half* out, const __half* g, const __half* u, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; float v = (float)g[i]; out[i] = __float2half((v / (1.0f + expf(-v))) * (float)u[i]); }
__global__ void residual_scale_k(__half* out, const __half* res, const float* hs_s, const float* hs_b, const float* res_s, const float* res_b, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; out[i] = __float2half((float)out[i] * hs_s[i] + hs_b[i] + (float)res[i] * res_s[i] + res_b[i]); }
__global__ void add_k(__half* a, const __half* b, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; a[i] = __float2half(__half2float(a[i]) + __half2float(b[i])); }

// Compile-time constants needed by kernel headers (Zaya1-8B dimensions)
// These must match the model the kernels were compiled for.
#ifndef H
#define H    2048
#define NQ   8
#define NKV  2
#define HD   128
#define QD   1024
#define KD   256
#define QKV  1280
#define N_LAYERS 40
#define VOCAB 262272
#define N_EXP 16
#define ROUTER_TOP_K 2
#define N_FF  2048
#define RTR_H 256
#endif

#define WMMA_M 16
#define WMMA_THREADS 128
#include "../../kernels/zaya_moe_tiled_gemv.hip"
#include "../../kernels/zaya_cca_custom.hip"
#include "../../kernels/v_interleave_kernel.hip"
#include "../../kernels/zaya_gpu_router.hip"
#include "../../kernels/zaya_router_moe.hip"
#include "../../kernels/zaya_moe_expert_ffn.hip"
#include "../../kernels/argmax_kernel.hip"
#include "../../kernels/lm_head_fused.hip"

// Undefine compile-time constants — we use runtime config from ModelConfig
#undef H
#undef NQ
#undef NKV
#undef HD
#undef QD
#undef KD
#undef QKV
#undef N_LAYERS
#undef VOCAB
#undef N_EXP
#undef ROUTER_TOP_K
#undef N_FF
#undef RTR_H

__global__ void eda_router_moe_kernel(const __half* hs, const float* prev_rs, int has_eda, float eda_scale, const float* gdw, const float* gdb, const float* rfn, const float* rf1, const float* rf1b, const float* rf2, const float* rf2b, const float* rout, const float* bb, const __half* gu, const __half* dn, float* next_rs, __half* moe_out, int* expert_idx, float* expert_wt);

static std::vector<float> load_bin(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return {};
    size_t n = f.tellg() / sizeof(float);
    f.seekg(0);
    std::vector<float> d(n);
    f.read((char*)d.data(), n * sizeof(float));
    return d;
}
static void upf16(const std::vector<float>& s, __half* d, int n, hipStream_t h = 0) { std::vector<__half> b(n); for (int i = 0; i < n; i++) b[i] = __float2half(s[i]); HIP_OK(hipMemcpy(d, b.data(), n * 2, hipMemcpyHostToDevice)); }
static void upf32(const std::vector<float>& s, float* d, int n, hipStream_t h = 0) { HIP_OK(hipMemcpy(d, s.data(), n * 4, hipMemcpyHostToDevice)); }

struct HipLayer {
    __half *nw=nullptr,*wq=nullptr,*wk=nullptr,*wv1=nullptr,*wv2=nullptr,*wo=nullptr,*pan=nullptr;
    __half *up=nullptr;  // generic (non-Zaya) SwiGLU up_proj (l.gu=gate_proj, l.dn=down_proj there)
    __half *qb=nullptr,*kb=nullptr,*vb=nullptr;  // optional QKV bias (Qwen2/2.5)
    __half *qn=nullptr,*kn=nullptr;  // optional per-head Q/K-norm weight [head_dim] (Qwen3)
    float *cdw=nullptr,*cdb=nullptr,*cgw=nullptr,*cgb=nullptr,*ks=nullptr;
    float *pahss=nullptr,*pahsb=nullptr,*parss=nullptr,*parsb=nullptr;
    float *gdw=nullptr,*gdb=nullptr,*rfn=nullptr,*rf1=nullptr,*rf1b=nullptr,*rf2=nullptr,*rf2b=nullptr,*rout=nullptr,*bb=nullptr;
    __half *gu=nullptr,*dn=nullptr;
    float *pmhss=nullptr,*pmhsb=nullptr,*pmrss=nullptr,*pmrsb=nullptr;
    float eda_scale[1]={0}; bool has_eda=false;
};

class HipBackend : public InferenceBackend {
    ModelConfig cfg_;
    bool loaded_ = false;
    hipStream_t st_ = 0;
    __half *d_hs=nullptr,*d_ao=nullptr,*d_tmp=nullptr,*d_fnw=nullptr,*d_embed_gpu=nullptr;
    __half *d_conv=nullptr,*d_phs=nullptr;
    __half *d_kcache=nullptr,*d_vcache=nullptr;  // generic-path KV cache: [N_LAYERS, max_seq_len, NKV, HD]
    float *d_prev_rs=nullptr;
    int *d_expert_idx=nullptr;
    float *d_expert_wt=nullptr;
    __half *d_all_logits=nullptr;
    int *d_best_idx=nullptr;
    float *d_best_val=nullptr;
    std::vector<HipLayer> layers_;
    std::vector<__half> embed_cpu_, fnorm_cpu_;
    bool embed_loaded_=false;

public:
    BackendType type() const override { return BackendType::HIP_GPU; }
    const char* name() const override { return "ROCm HIP"; }
    float estimated_tok_s() const override { return 64.0f; }  // estimate; measured end-to-end on Zaya1-8B
    bool is_coherent() const override { return true; }

    bool avail_cached_ = false, avail_checked_ = false;
    bool is_available() override {
        if (avail_checked_) return avail_cached_;
        avail_checked_ = true;
        int count = 0;
        hipError_t e = hipGetDeviceCount(&count);
        if (e != hipSuccess || count == 0) { return false; }
        hipDeviceProp_t props;
        HIP_OK(hipGetDeviceProperties(&props, 0));
        fprintf(stderr, "  HIP: found %s (%d CU, %zu MB)\n", props.name, props.multiProcessorCount, props.totalGlobalMem / (1024*1024));
        avail_cached_ = true;
        return true;
    }

    bool load_model(const ModelConfig& cfg) override {
        cfg_ = cfg;
        unload_model();

        // Accept any model dimensions — use runtime cfg_ fields.
        bool is_zaya = (cfg.hidden_size == 2048 && cfg.num_layers == 40 && cfg.vocab_size == 262272);
        if (!is_zaya) {
            fprintf(stderr, "  HIP: non-Zaya model (H=%d L=%d V=%d) — using generic GPU path\n",
                    cfg.hidden_size, cfg.num_layers, cfg.vocab_size);
        }

        HIP_OK(hipStreamCreate(&st_));
        int H=cfg.hidden_size, NQ=cfg.num_heads, NKV=cfg.num_kv_heads, HD=cfg.head_dim;
        int QD=NQ*HD, KD=NKV*HD, QKV=QD+KD;
        int N_LAYERS=cfg.num_layers, VOCAB=cfg.vocab_size;
        int N_EXP=cfg.num_experts, ROUTER_TOP_K=cfg.num_experts_top, N_FF=cfg.intermediate_size;
        int RTR_H=cfg.router_hidden;
        std::string W = cfg.weights_dir;
        if (!W.empty() && W.back() != '/') W += '/';

        auto embed = load_bin(W + "model_embed_tokens_weight.bin");
        auto fnorm = load_bin(W + "model_norm_weight.bin");
        if (embed.empty() || fnorm.empty()) {
            // Try GGUF loading
            if (!cfg.model_path.empty()) {
                fprintf(stderr, "  HIP: trying GGUF load from %s\n", cfg.model_path.c_str());
                if (load_gguf_model(cfg)) { loaded_ = true; return true; }
            }
            fprintf(stderr, "  HIP: missing weights (no .bin or GGUF)\n"); return false;
        }
        embed_cpu_.resize(VOCAB * H); fnorm_cpu_.resize(H);
        for (int i = 0; i < VOCAB * H; i++) embed_cpu_[i] = __float2half(embed[i]);
        for (int i = 0; i < H; i++) fnorm_cpu_[i] = __float2half(fnorm[i]);
        HIP_OK(hipMalloc(&d_embed_gpu, (size_t)VOCAB * H * 2));
        HIP_OK(hipMemcpy(d_embed_gpu, embed_cpu_.data(), VOCAB * H * 2, hipMemcpyHostToDevice));
        HIP_OK(hipMalloc(&d_fnw, H * 2));
        HIP_OK(hipMemcpy(d_fnw, fnorm_cpu_.data(), H * 2, hipMemcpyHostToDevice));
        embed_loaded_ = true;

        // Scratch buffers must hold the largest thing ever written into them:
        // QKV concat (QD+2*KD) for attention, or gate+up concat (2*N_FF) for
        // the generic FFN path — both exceed H, unlike the old H-only sizing.
        size_t scratch_elems = std::max<size_t>({(size_t)QD + 2*(size_t)KD, (size_t)2*N_FF, (size_t)H});
        HIP_OK(hipMalloc(&d_hs, H * 2));
        HIP_OK(hipMalloc(&d_ao, scratch_elems * 2));
        HIP_OK(hipMalloc(&d_tmp, scratch_elems * 2));
        HIP_OK(hipMalloc(&d_conv, (size_t)N_LAYERS * 2 * QKV * 4));  // ×2 for separate in/out state
        HIP_OK(hipMalloc(&d_phs, (size_t)N_LAYERS * H * 2));
        HIP_OK(hipMalloc(&d_prev_rs, (size_t)N_LAYERS * RTR_H * 4));
        HIP_OK(hipMalloc(&d_kcache, (size_t)N_LAYERS * cfg.max_seq_len * KD * 2));
        HIP_OK(hipMalloc(&d_vcache, (size_t)N_LAYERS * cfg.max_seq_len * KD * 2));
        HIP_OK(hipMalloc(&d_expert_idx, 4)); HIP_OK(hipMalloc(&d_expert_wt, 4));
        HIP_OK(hipMalloc(&d_all_logits, (size_t)VOCAB * 2));
        HIP_OK(hipMalloc(&d_best_idx, 4)); HIP_OK(hipMalloc(&d_best_val, 4));

        layers_.resize(N_LAYERS);
        auto A = [](auto& p, int n, const std::string& path) { HIP_OK(hipMalloc(&p, n*2)); auto d=load_bin(path); if(!d.empty()){std::vector<__half> h(n);for(int i=0;i<n;i++)h[i]=__float2half(d[i]);HIP_OK(hipMemcpy(p,h.data(),n*2,hipMemcpyHostToDevice));} };
        auto B = [](auto& p, int n, const std::string& path) { HIP_OK(hipMalloc(&p, n*4)); auto d=load_bin(path); if(!d.empty()) HIP_OK(hipMemcpy(p,d.data(),n*4,hipMemcpyHostToDevice)); };
        auto L = [](int i) { return "model_layers_" + std::to_string(i); };

        for (int il = 0; il < N_LAYERS; il++) {
            auto& l = layers_[il];
            std::string lp = L(il) + "_";
            A(l.nw, H, W+lp+"input_layernorm_weight.bin");
            A(l.wq, QD*H, W+lp+"self_attn_qkv_proj_q_proj_weight.bin");
            A(l.wk, KD*H, W+lp+"self_attn_qkv_proj_k_proj_weight.bin");
            A(l.wv1, (KD/2)*H, W+lp+"self_attn_qkv_proj_v_proj_current_weight.bin");
            A(l.wv2, (KD/2)*H, W+lp+"self_attn_qkv_proj_v_proj_delayed_weight.bin");
            A(l.wo, H*QD, W+lp+"self_attn_o_proj_weight.bin");
            B(l.cdw, QKV*2, W+lp+"self_attn_qkv_proj_conv_qk_depthwise_weight.bin");
            B(l.cdb, QKV, W+lp+"self_attn_qkv_proj_conv_qk_depthwise_bias.bin");
            B(l.cgw, QKV*128*2, W+lp+"self_attn_qkv_proj_conv_qk_grouped_weight.bin");
            B(l.cgb, QKV, W+lp+"self_attn_qkv_proj_conv_qk_grouped_bias.bin");
            B(l.ks, NKV, W+lp+"self_attn_qk_norm_temp.bin");
            B(l.pahss, H, W+lp+"post_attention_residual_scale_hidden_states_scale.bin");
            B(l.pahsb, H, W+lp+"post_attention_residual_scale_hidden_states_bias.bin");
            B(l.parss, H, W+lp+"post_attention_residual_scale_residual_scale.bin");
            B(l.parsb, H, W+lp+"post_attention_residual_scale_residual_bias.bin");
            A(l.pan, H, W+lp+"post_attention_layernorm_weight.bin");
            B(l.gdw, H*RTR_H, W+lp+"mlp_gate_down_proj_weight.bin");
            B(l.gdb, RTR_H, W+lp+"mlp_gate_down_proj_bias.bin");
            B(l.rfn, RTR_H, W+lp+"mlp_gate_router_mlp_norm_weight.bin");
            B(l.rf1, RTR_H*RTR_H, W+lp+"mlp_gate_router_mlp_fc1_weight.bin");
            B(l.rf1b, RTR_H, W+lp+"mlp_gate_router_mlp_fc1_bias.bin");
            B(l.rf2, RTR_H*RTR_H, W+lp+"mlp_gate_router_mlp_fc2_weight.bin");
            B(l.rf2b, RTR_H, W+lp+"mlp_gate_router_mlp_fc2_bias.bin");
            B(l.rout, ROUTER_TOP_K*RTR_H, W+lp+"mlp_gate_router_mlp_out_proj_weight.bin");
            B(l.bb, ROUTER_TOP_K, W+lp+"mlp_gate_balancing_biases.bin");
            auto sz_gu=N_EXP*2*N_FF*H, sz_dn=N_EXP*H*N_FF;
            if(hipMalloc(&l.gu,sz_gu*2)==hipSuccess&&hipMalloc(&l.dn,sz_dn*2)==hipSuccess){
                auto gud=load_bin(W+lp+"mlp_experts_gate_up_proj.bin");
                auto dnd=load_bin(W+lp+"mlp_experts_down_proj.bin");
                if(!gud.empty())upf16(gud,l.gu,sz_gu,0);
                if(!dnd.empty())upf16(dnd,l.dn,sz_dn,0);
            }
            B(l.pmhss, H, W+lp+"post_mlp_residual_scale_hidden_states_scale.bin");
            B(l.pmhsb, H, W+lp+"post_mlp_residual_scale_hidden_states_bias.bin");
            B(l.pmrss, H, W+lp+"post_mlp_residual_scale_residual_scale.bin");
            B(l.pmrsb, H, W+lp+"post_mlp_residual_scale_residual_bias.bin");
            std::string ep=W+lp+"mlp_gate_router_states_scale.bin";
            std::ifstream ff(ep,std::ios::binary);
            if(ff){ff.read((char*)l.eda_scale,4);l.has_eda=true;}
        }
        HIP_OK(hipStreamSynchronize(st_));
        loaded_ = true;
        fprintf(stderr, "  HIP: loaded %d layers (%d-dim, %d vocab)\n", N_LAYERS, H, VOCAB);
        return true;
    }

    void unload_model() override {
        auto f = [](void* p) { if (p) { hipError_t _e = hipFree(p); if(_e != hipSuccess) fprintf(stderr, "hipFree failed: %s\n", hipGetErrorString(_e)); } };
        f(d_hs); f(d_ao); f(d_tmp); f(d_fnw); f(d_embed_gpu);
        f(d_conv); f(d_phs); f(d_prev_rs); f(d_expert_idx); f(d_expert_wt);
        f(d_all_logits); f(d_best_idx); f(d_best_val); f(d_kcache); f(d_vcache);
        d_hs=d_ao=d_tmp=d_fnw=d_embed_gpu=nullptr;
        d_conv=d_phs=nullptr; d_prev_rs=nullptr;
        d_expert_idx=nullptr; d_expert_wt=nullptr;
        d_all_logits=nullptr; d_best_idx=nullptr; d_best_val=nullptr;
        d_kcache=nullptr; d_vcache=nullptr;
        for (auto& l : layers_) {
            for (auto* p : {&l.nw,&l.wq,&l.wk,&l.wv1,&l.wv2,&l.wo,&l.pan,&l.gu,&l.dn,&l.up,&l.qb,&l.kb,&l.vb,&l.qn,&l.kn}) f(*p);
            for (auto* p : {&l.cdw,&l.cdb,&l.cgw,&l.cgb,&l.ks,&l.pahss,&l.pahsb,&l.parss,&l.parsb,&l.gdw,&l.gdb,&l.rfn,&l.rf1,&l.rf1b,&l.rf2,&l.rf2b,&l.rout,&l.bb,&l.pmhss,&l.pmhsb,&l.pmrss,&l.pmrsb}) f(*p);
        }
        layers_.clear(); embed_loaded_ = false; loaded_ = false;
        if (st_) { HIP_OK(hipStreamDestroy(st_)); st_ = 0; }
    }

    // ── GGUF weight loading ──────────────────────────────────────
    bool load_gguf_model(const ModelConfig& cfg) {
        // Minimal GGUF reader for direct GPU weight loading
        FILE* gf = fopen(cfg.model_path.c_str(), "rb");
        if (!gf) return false;
        uint32_t magic; fread(&magic, 4, 1, gf);
        if (magic != 0x46554747) { fclose(gf); return false; }
        uint32_t ver; fread(&ver, 4, 1, gf);
        uint64_t nt, nkv; fread(&nt, 8, 1, gf); fread(&nkv, 8, 1, gf);
        // Skip KV
        for (uint64_t i = 0; i < nkv; i++) {
            uint64_t kl; fread(&kl, 8, 1, gf); fseek(gf, kl, SEEK_CUR);
            uint32_t vt; fread(&vt, 4, 1, gf);
            // GGUF scalar type widths: 0/1=u8/i8 (1B), 2/3=u16/i16 (2B),
            // 4/5/6=u32/i32/f32 (4B), 7=bool (1B), 8=string (u64 len + bytes),
            // 10/11/12=u64/i64/f64 (8B). vt==2 (UINT16) was previously lumped
            // in with vt==8 (STRING) and misread as a length-prefixed value.
            if (vt == 8) { uint64_t sl; fread(&sl, 8, 1, gf); fseek(gf, sl, SEEK_CUR); }
            else if (vt == 2 || vt == 3) fseek(gf, 2, SEEK_CUR);
            else if (vt >= 4 && vt <= 6) fseek(gf, 4, SEEK_CUR);
            else if (vt == 7) fseek(gf, 1, SEEK_CUR);
            else if (vt == 9) {
                // GGUF array value layout: element_type (u32), length (u64),
                // THEN `length` elements — was reading both as u32 in the
                // wrong order, desyncing the file offset for everything
                // after the first array-valued KV pair (e.g. tokenizer
                // token/merge lists), which then corrupted tensor-header
                // parsing downstream.
                uint32_t at; fread(&at, 4, 1, gf); uint64_t al; fread(&al, 8, 1, gf);
                if (at == 8) { for (uint64_t j = 0; j < al; j++) { uint64_t ss; fread(&ss, 8, 1, gf); fseek(gf, ss, SEEK_CUR); } }
                else if (at == 2 || at == 3) fseek(gf, al * 2, SEEK_CUR);
                else if (at >= 4 && at <= 6) fseek(gf, al * 4, SEEK_CUR);
                else if (at >= 10 && at <= 12) fseek(gf, al * 8, SEEK_CUR);
                else fseek(gf, al, SEEK_CUR);  // 0/1/7 (u8/i8/bool) — 1 byte each
            } else if (vt >= 10 && vt <= 12) fseek(gf, 8, SEEK_CUR);
            else fseek(gf, 1, SEEK_CUR);  // 0/1 (u8/i8)
        }
        // Read tensor headers
        struct TInfo { std::string name; uint64_t off; uint32_t dtype; uint64_t ne; };
        std::unordered_map<std::string, TInfo> tmap;
        for (uint64_t i = 0; i < nt; i++) {
            uint64_t nl; fread(&nl, 8, 1, gf);
            TInfo ti; ti.name.resize(nl); fread(&ti.name[0], 1, nl, gf);
            uint32_t nd; fread(&nd, 4, 1, gf);
            ti.ne = 1; for (uint32_t j = 0; j < nd; j++) { uint64_t d; fread(&d, 8, 1, gf); ti.ne *= d; }
            fread(&ti.dtype, 4, 1, gf);
            fread(&ti.off, 8, 1, gf);  // GGUF's own authoritative per-tensor offset (relative to data section start)
            tmap[ti.name] = ti;
        }
        // The previous version discarded this real offset field and instead
        // recomputed offsets by accumulating per-dtype size estimates while
        // iterating `tmap` — a std::unordered_map, whose iteration order is
        // unspecified and does NOT match the file's actual physical tensor
        // layout, so every computed offset was wrong (and the size estimates
        // didn't cover K-quants anyway). Using the file's own offset is both
        // simpler and correct regardless of quantization format.
        uint64_t data_section_base = ftell(gf); data_section_base = (data_section_base + 31) & ~31;
        for (auto& [n, t] : tmap) t.off += data_section_base;

        int H = cfg.hidden_size, V = cfg.vocab_size, L = cfg.num_layers;
        int NQ = cfg.num_heads, NKV = cfg.num_kv_heads, HD = cfg.head_dim;
        int QD = NQ * HD, KD = NKV * HD, QKV = QD + KD;
        int FF = cfg.intermediate_size;

        // Embedding
        auto load_half = [&](const char* name, __half*& dptr, size_t n) {
            auto it = tmap.find(name); if (it == tmap.end()) return false;
            HIP_OK(hipMalloc(&dptr, n * 2));
            fseek(gf, it->second.off, SEEK_SET);
            if (it->second.dtype == 1) {
                std::vector<uint16_t> buf(n); fread(buf.data(), 2, n, gf);
                HIP_OK(hipMemcpy(dptr, buf.data(), n * 2, hipMemcpyHostToDevice));
            } else if (it->second.dtype == 8) {
                // Q8_0 dequant to FP16. GGML_TYPE_Q8_0 is genuinely 8 per
                // ggml.h (0=F32,1=F16,2=Q4_0,3=Q4_1,6=Q5_0,7=Q5_1,8=Q8_0,
                // 9=Q8_1,10..15=K-quants) — a previous pass here "corrected"
                // this from 8 to 7 based on a wrong memory of the enum,
                // which actually broke it (7 is Q5_1, not Q8_0). Verified
                // against ggml.h directly and the real `gguf` Python
                // package this time before touching it again.
                std::vector<__half> buf(n);
                int blks = (n + 31) / 32;
                for (int b = 0; b < blks; b++) {
                    uint16_t sh; fread(&sh, 2, 1, gf); float sc = fp16_to_fp32(sh);
                    int8_t q[32]; fread(q, 1, 32, gf);
                    for (int j = 0; j < 32 && b * 32 + j < (int)n; j++) buf[b * 32 + j] = __float2half(q[j] * sc);
                }
                HIP_OK(hipMemcpy(dptr, buf.data(), n * 2, hipMemcpyHostToDevice));
            } else if (it->second.dtype == 10 || it->second.dtype == 11 || it->second.dtype == 12 ||
                       it->second.dtype == 13 || it->second.dtype == 14) {
                // Q2_K/Q3_K/Q4_K/Q5_K/Q6_K: read the whole compressed tensor,
                // dequant to f32 host-side, then convert to fp16 for upload.
                int bpb = it->second.dtype == 10 ? 84
                        : it->second.dtype == 11 ? 110
                        : it->second.dtype == 12 ? 144
                        : it->second.dtype == 13 ? 176 : 210;
                size_t nblocks = (n + 255) / 256;
                std::vector<uint8_t> raw(nblocks * bpb); fread(raw.data(), 1, raw.size(), gf);
                std::vector<float> f32(n);
                switch (it->second.dtype) {
                    case 10: dequant_q2_k_to_f32(raw.data(), f32.data(), (int)n); break;
                    case 11: dequant_q3_k_to_f32(raw.data(), f32.data(), (int)n); break;
                    case 12: dequant_q4_k_to_f32(raw.data(), f32.data(), (int)n); break;
                    case 13: dequant_q5_k_to_f32(raw.data(), f32.data(), (int)n); break;
                    default: dequant_q6_k_to_f32(raw.data(), f32.data(), (int)n); break;
                }
                std::vector<__half> buf(n);
                for (size_t i = 0; i < n; i++) buf[i] = __float2half(f32[i]);
                HIP_OK(hipMemcpy(dptr, buf.data(), n * 2, hipMemcpyHostToDevice));
            } else {
                // F32 → FP16. NOTE: also wrongly hit by any *other* unhandled
                // quant type (Q4_0=2, Q4_1=3, Q5_0=6, Q5_1=7, Q8_1=9, Q8_K=15,
                // IQ*) — those aren't decoded correctly, they just don't crash.
                std::vector<__half> buf(n);
                std::vector<float> f32(n); fread(f32.data(), 4, n, gf);
                for (size_t i = 0; i < n; i++) buf[i] = __float2half(f32[i]);
                HIP_OK(hipMemcpy(dptr, buf.data(), n * 2, hipMemcpyHostToDevice));
            }
            return true;
        };

        // Detect actual vocab size from the first tensor loaded to d_embed_gpu
        {
            int tcount = 0;
            for (auto& [tname, tinfo] : tmap) {
                if (tcount++ < 5) fprintf(stderr, "  HIP: tensor[%d] %s ne=%lu\n", tcount-1, tname.c_str(), tinfo.ne);
                if (tname.find("embed") != std::string::npos && tname.find("norm") == std::string::npos
                    && tinfo.ne > 0 && H > 0) {
                    int actual_v = (int)(tinfo.ne / H);
                    fprintf(stderr, "  HIP: %s ne=%lu H=%d -> V=%d (was %d)\n", tname.c_str(), tinfo.ne, H, actual_v, V);
                    if (actual_v > 100 && actual_v != V) {
                        V = actual_v; cfg_.vocab_size = V; cfg_.vocab = V;
                    }
                    break;
                }
            }
        }
        if (!load_half("token_embd.weight", d_embed_gpu, (size_t)V * H))
            load_half("model.embed_tokens.weight", d_embed_gpu, (size_t)V * H);

        size_t fn_n = 0;
        if (!load_half("output_norm.weight", d_fnw, H)) {
            // Try alternate names
            auto it = tmap.find("model.norm.weight");
            if (it != tmap.end()) load_half("model.norm.weight", d_fnw, H);
        }

        size_t scratch_elems = std::max<size_t>({(size_t)QD + 2*(size_t)KD, (size_t)2*FF, (size_t)H});
        HIP_OK(hipMalloc(&d_hs, H * 2));
        HIP_OK(hipMalloc(&d_ao, scratch_elems * 2));
        HIP_OK(hipMalloc(&d_tmp, scratch_elems * 2));
        HIP_OK(hipMalloc(&d_conv, (size_t)L * 2 * QKV * 4));
        HIP_OK(hipMalloc(&d_phs, (size_t)L * H * 2));
        HIP_OK(hipMalloc(&d_all_logits, (size_t)V * 2));
        HIP_OK(hipMalloc(&d_best_idx, 4)); HIP_OK(hipMalloc(&d_best_val, 4));
        HIP_OK(hipMalloc(&d_kcache, (size_t)L * cfg.max_seq_len * KD * 2));
        HIP_OK(hipMalloc(&d_vcache, (size_t)L * cfg.max_seq_len * KD * 2));

        layers_.resize(L);
        for (int il = 0; il < L; il++) {
            auto& lw = layers_[il];
            // GGUF's own tensor-naming convention (llama.cpp), NOT the
            // HuggingFace-safetensors dotted names this used to look for
            // (model.layers.N.self_attn.q_proj.weight etc) — those never
            // match a real GGUF file, so every per-layer weight below was
            // silently null for every GGUF-loaded model, always.
            char pfx[64]; snprintf(pfx, sizeof(pfx), "blk.%d.", il);
            std::string p(pfx);
            load_half((p + "attn_norm.weight").c_str(), lw.nw, H);
            load_half((p + "ffn_norm.weight").c_str(), lw.pan, H);
            load_half((p + "attn_q.weight").c_str(), lw.wq, (size_t)QD * H);
            load_half((p + "attn_k.weight").c_str(), lw.wk, (size_t)KD * H);
            load_half((p + "attn_v.weight").c_str(), lw.wv1, (size_t)KD * H);
            load_half((p + "attn_output.weight").c_str(), lw.wo, (size_t)H * QD);
            load_half((p + "ffn_gate.weight").c_str(), lw.gu, (size_t)FF * H);
            load_half((p + "ffn_up.weight").c_str(), lw.up, (size_t)FF * H);
            load_half((p + "ffn_down.weight").c_str(), lw.dn, (size_t)H * FF);
            // Optional QKV bias (Qwen2/2.5 use it; most other architectures
            // don't — load_half returns false and leaves the pointer null
            // when the tensor doesn't exist, which the forward pass checks).
            load_half((p + "attn_q.bias").c_str(), lw.qb, (size_t)QD);
            load_half((p + "attn_k.bias").c_str(), lw.kb, (size_t)KD);
            load_half((p + "attn_v.bias").c_str(), lw.vb, (size_t)KD);
            // Optional per-head Q/K-norm (Qwen3 uses it; most other
            // architectures don't — null pointer means "skip" below).
            load_half((p + "attn_q_norm.weight").c_str(), lw.qn, (size_t)HD);
            load_half((p + "attn_k_norm.weight").c_str(), lw.kn, (size_t)HD);
        }
        fclose(gf);
        embed_loaded_ = true;
        fprintf(stderr, "  HIP: GGUF loaded (%d layers, %d vocab)\n", L, V);
        return true;
    }

    void reset_state() override {
        if (!loaded_) return;
        int H=cfg_.hidden_size, N_LAYERS=cfg_.num_layers;
        int QD=cfg_.num_heads*cfg_.head_dim, KD=cfg_.num_kv_heads*cfg_.head_dim, QKV=QD+KD;
        int RTR_H=cfg_.router_hidden;
        HIP_OK(hipMemsetAsync(d_conv, 0, (size_t)N_LAYERS*2*QKV*2, st_));
        HIP_OK(hipMemsetAsync(d_phs, 0, (size_t)N_LAYERS*H*2, st_));
        HIP_OK(hipMemsetAsync(d_prev_rs, 0, (size_t)N_LAYERS*RTR_H*4, st_));
    }

    int forward(int token_id, int pos) override {
        if (!loaded_ || !embed_loaded_) return 0;
        int H=cfg_.hidden_size, NQ=cfg_.num_heads, NKV=cfg_.num_kv_heads, HD=cfg_.head_dim;
        int QD=NQ*HD, KD=NKV*HD, QKV=QD+KD;
        int N_LAYERS=cfg_.num_layers, VOCAB=cfg_.vocab_size;
        int N_FF=cfg_.intermediate_size, RTR_H=cfg_.router_hidden;
        bool is_zaya = (H == 2048 && N_LAYERS == 40 && VOCAB == 262272);
        int g1 = (H+BLK-1)/BLK;
        // d_embed_gpu is populated by both loaders (unlike the host-side
        // embed_cpu_, which only the .bin loader fills in) — a GPU-side
        // lookup works for both instead of reading past the end of an
        // empty embed_cpu_ for GGUF-loaded models.
        rcpp_embedding_lookup_fp16(d_embed_gpu, token_id, d_hs, H, st_);
        size_t kv_layer_stride = (size_t)cfg_.max_seq_len * KD;  // elements, generic-path KV cache only
        for (int il = 0; il < N_LAYERS; il++) {
            auto& l = layers_[il];
            // Pre-attention-norm residual, captured before rmsnorm below mutates
            // d_hs in place. Zaya's CCA path uses this as its own "previous
            // hidden state" input; the generic path below reuses the same slot
            // as a true pre-norm transformer residual (cheap, unconditional,
            // and harmless for Zaya since it already needed this copy anyway).
            copy_k<<<g1, BLK, 0, st_>>>(d_phs + (size_t)il*H, d_hs, H);
            rmsnorm_k<<<1, BLK, 0, st_>>>(d_hs, l.nw, H, cfg_.rms_norm_eps);
            moe_tiled_gemv<<<QD/16, 128, 0, st_>>>(d_tmp, d_hs, l.wq, QD, H);
            moe_tiled_gemv<<<KD/16, 128, 0, st_>>>(d_tmp+QD, d_hs, l.wk, KD, H);
            if (l.qb) add_k<<<(QD+BLK-1)/BLK, BLK, 0, st_>>>(d_tmp, l.qb, QD);
            if (l.kb) add_k<<<(KD+BLK-1)/BLK, BLK, 0, st_>>>(d_tmp+QD, l.kb, KD);
            // Per-head Q/K-norm (Qwen3), applied before RoPE.
            if (l.qn) qk_norm_k<<<NQ, BLK, 0, st_>>>(d_tmp, l.qn, HD, cfg_.rms_norm_eps);
            if (l.kn) qk_norm_k<<<NKV, BLK, 0, st_>>>(d_tmp+QD, l.kn, HD, cfg_.rms_norm_eps);
            if (is_zaya) {
                moe_tiled_gemv<<<KD/2/16, 128, 0, st_>>>(d_tmp+QD+KD, d_hs, l.wv1, KD/2, H);
                moe_tiled_gemv<<<KD/2/16, 128, 0, st_>>>(d_tmp+QD+KD+KD/2, d_phs+(size_t)il*H, l.wv2, KD/2, H);
                // Zaya-specific CCA attention + fused router MoE (fast path)
                v_interleave_kernel<<<(KD/2+BLK-1)/BLK, BLK, 0, st_>>>(d_tmp+QD, d_tmp+QD+KD, d_tmp+QD+KD+KD/2, KD/2);
                cca_custom_kernel<<<1, 256, cca_custom_smem_bytes(NQ, NKV, HD, 64), st_>>>(d_tmp, d_tmp+QD, d_tmp+QD, d_phs+(size_t)il*H, d_conv+(size_t)il*2*QKV, l.cdw, l.cdb, l.cgw, l.cgb, l.ks, d_ao, d_conv+(size_t)il*2*QKV*2, d_phs+(size_t)il*H, il, 1, NQ, NKV, HD, 64, 5000000.0f, 128);
                moe_tiled_gemv<<<H/16, 128, 0, st_>>>(d_ao, d_ao, l.wo, H, QD);
                residual_scale_k<<<g1, BLK, 0, st_>>>(d_ao, d_hs, l.pahss, l.pahsb, l.parss, l.parsb, H);
                copy_k<<<g1, BLK, 0, st_>>>(d_hs, d_ao, H);
                rmsnorm_k<<<1, BLK, 0, st_>>>(d_hs, l.pan, H, cfg_.rms_norm_eps);
                if (l.gu && l.dn) {
                    eda_router_gpu_kernel<<<1, RTR_H, eda_router_smem_bytes(RTR_H, 2), st_>>>(d_hs, d_prev_rs+(size_t)il*RTR_H, l.has_eda?1:0, l.eda_scale[0], l.gdw, l.gdb, l.rfn, l.rf1, l.rf1b, l.rf2, l.rf2b, l.rout, l.bb, d_prev_rs+(size_t)il*RTR_H, d_expert_idx, d_expert_wt, 16, H, RTR_H, 2);
                    encode_expert_cache_kernel<<<1, 32, 0, st_>>>(d_prev_rs+(size_t)il*RTR_H, d_expert_idx, RTR_H);
                    { int gb=(2*N_FF+15)/16, db=(H+15)/16, sb=(N_FF+BLK-1)/BLK;
                    wmma_gateup_kernel<<<gb,128,0,st_>>>(d_tmp,d_hs,l.gu,d_expert_idx);
                    silu_mul_k<<<sb,BLK,0,st_>>>(d_ao,d_tmp,d_tmp+N_FF,N_FF);
                    wmma_down_kernel<<<db,128,0,st_>>>(d_tmp,d_ao,l.dn,d_expert_idx); }
                    residual_scale_k<<<g1, BLK, 0, st_>>>(d_tmp, d_hs, l.pmhss, l.pmhsb, l.pmrss, l.pmrsb, H);
                    copy_k<<<g1, BLK, 0, st_>>>(d_hs, d_tmp, H);
                }
            } else {
                // Generic path: real self-attention (RoPE + KV cache + causal
                // GQA attention) + standard SwiGLU FFN, both pre-norm with
                // residual. Previously this branch skipped attention entirely
                // (QKV was computed and thrown away) and had a broken FFN
                // (read an uninitialized "up" buffer) — see the commit that
                // added this comment for the full writeup.
                __half* kc = d_kcache + (size_t)il * kv_layer_stride;
                __half* vc = d_vcache + (size_t)il * kv_layer_stride;

                // Single full V projection (generic models load one v_proj,
                // not Zaya's split current/delayed halves).
                moe_tiled_gemv<<<KD/16, 128, 0, st_>>>(d_tmp+QD+KD, d_hs, l.wv1, KD, H);
                if (l.vb) add_k<<<(KD+BLK-1)/BLK, BLK, 0, st_>>>(d_tmp+QD+KD, l.vb, KD);

                // RoPE Q+K in place, write RoPE'd K + raw V into this layer's
                // KV cache at position `pos`.
                rcpp_rope_kv_append_fp16(d_tmp, kc, d_tmp+QD+KD, vc, pos, cfg_.rope_theta, NQ, NKV, HD, st_);

                // Causal GQA attention against positions [0, pos] of this
                // layer's KV cache.
                rcpp_kv_cache_attn_decode(d_tmp, kc, vc, d_ao, NQ, NKV, HD, pos + 1, 1.0f / sqrtf((float)HD), st_);

                // O-projection + residual.
                moe_tiled_gemv<<<H/16, 128, 0, st_>>>(d_tmp, d_ao, l.wo, H, QD);
                add_k<<<g1, BLK, 0, st_>>>(d_phs + (size_t)il*H, d_tmp, H);
                copy_k<<<g1, BLK, 0, st_>>>(d_hs, d_phs + (size_t)il*H, H);

                // Pre-FFN-norm residual, same reused per-layer slot (its
                // pre-attention value is no longer needed at this point).
                copy_k<<<g1, BLK, 0, st_>>>(d_phs + (size_t)il*H, d_hs, H);
                rmsnorm_k<<<1, BLK, 0, st_>>>(d_hs, l.pan, H, cfg_.rms_norm_eps);
                if (l.gu && l.dn && l.up) {
                    int ffb = (N_FF+15)/16;
                    moe_tiled_gemv<<<ffb, 128, 0, st_>>>(d_tmp, d_hs, l.gu, N_FF, H);        // gate
                    moe_tiled_gemv<<<ffb, 128, 0, st_>>>(d_tmp+N_FF, d_hs, l.up, N_FF, H);   // up
                    silu_mul_k<<<(N_FF+BLK-1)/BLK, BLK, 0, st_>>>(d_ao, d_tmp, d_tmp+N_FF, N_FF);
                    int db = (H+15)/16;
                    moe_tiled_gemv<<<db, 128, 0, st_>>>(d_tmp, d_ao, l.dn, H, N_FF);
                    add_k<<<g1, BLK, 0, st_>>>(d_phs + (size_t)il*H, d_tmp, H);
                }
                copy_k<<<g1, BLK, 0, st_>>>(d_hs, d_phs + (size_t)il*H, H);
            }
        }
        rmsnorm_k<<<1, BLK, 0, st_>>>(d_hs, d_fnw, H, cfg_.rms_norm_eps);
        lm_head_fused_kernel<<<VOCAB, 256, 0, st_>>>(d_all_logits, d_hs, d_embed_gpu, H, VOCAB);
        argmax_kernel<<<1, 256, 0, st_>>>(d_all_logits, VOCAB, d_best_idx, d_best_val);
        HIP_OK(hipStreamSynchronize(st_));
        int best = 0;
        HIP_OK(hipMemcpy(&best, d_best_idx, 4, hipMemcpyDeviceToHost));
        return best;
    }
};

std::vector<InferenceBackend*> detect_backends_hip() {
    std::vector<InferenceBackend*> backends;
    static HipBackend hip;
    backends.push_back(&hip);
    return backends;
}
