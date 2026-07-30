// backend_fused.cpp — Fused GPU attention ∥ NPU FFN per-layer pipeline.
//
// GPU does everything: RMSNorm → QKV → RoPE → Flash-Decoding → OutProj → FFN
// NPU backfills FFN when available (SharedBO zero-copy).
//
// All GPU operations use custom GEMV kernels (not rocBLAS) — much faster for
// the small- M projections (M ≤ 3072). Everything stays on-device; only one
// sync at the end of forward().

#include "backend.h"
#include "backend_fused_npu.h"
#include "../engine/npu/src/onebp_loader.cpp"
#include "../engine/fusion/zero_copy/shared_bo.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <memory>
#include <chrono>
#include <unistd.h>
#include <fcntl.h>

static constexpr float EPS = 1e-6f;
static constexpr int  BLOCK = 256;

#define HIP_CHECK(call) \
    do { hipError_t _hip_e = (call); \
         if (_hip_e != hipSuccess) { \
             fprintf(stderr, "HIP error %s at %s:%d\n", \
                     hipGetErrorString(_hip_e), __FILE__, __LINE__); \
             std::abort(); } } while(0)
#define HIP_CHECK_D(call) \
    do { hipError_t _hip_e = (call); \
         if (_hip_e != hipSuccess) { \
             fprintf(stderr, "HIP error (dtor) %s at %s:%d\n", \
                     hipGetErrorString(_hip_e), __FILE__, __LINE__); } } while(0)

// ═══════════════════════════════════════════════════════════════════════════
// Device kernels (all use "fused_" prefix — no conflict with hip_1bp_kernels)
// ═══════════════════════════════════════════════════════════════════════════

// ── GEMV: y[M] = W[M,N] @ x[N]  (row-major W) ──
// One block per output row. blockDim.x threads cooperatively reduce.
template<int BLK=BLOCK>
__launch_bounds__(BLK)
__global__ void fused_gemv_kernel(float* __restrict__ y,
                                   const float* __restrict__ W,
                                   const float* __restrict__ x,
                                   int M, int N) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    double sum = 0.0;
    for (int k = threadIdx.x; k < N; k += blockDim.x)
        sum += (double)x[k] * W[(size_t)row * N + k];
    __shared__ double sdata[32][32];
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    sdata[warp][lane] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = sdata[lane][threadIdx.x];
        for (int s = blockDim.x/2; s > 0; s >>= 1) {
            __syncthreads();
            if (threadIdx.x < s) sdata[0][threadIdx.x] += sdata[0][threadIdx.x + s];
        }
        if (threadIdx.x == 0) y[row] = (float)sdata[0][0];
    }
}

// ── Non-template version for external linking ──
__global__ void fused_gemv_plain_kernel(float* y, const float* W, const float* x, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    double sum = 0;
    for (int k = threadIdx.x; k < N; k += blockDim.x)
        sum += (double)x[k] * W[(size_t)row * N + k];
    __shared__ double sdata[BLOCK];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = (float)sdata[0];
}

// ── RMSNorm (in-place) ──
__global__ void fused_rmsnorm_kernel(float* __restrict__ x, const float* __restrict__ w,
                                      int N, float eps) {
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) local += x[i] * x[i];
    __shared__ float sdata[BLOCK];
    sdata[tid] = local;
    for (int s = blockDim.x/2; s > 0; s >>= 1) { __syncthreads(); if (tid < s) sdata[tid] += sdata[tid + s]; }
    __syncthreads();
    float inv = rsqrtf(sdata[0] / N + eps);
    for (int i = tid; i < N; i += blockDim.x) x[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

// ── RoPE ──
__global__ void fused_rope_kernel(float* __restrict__ x, int head_dim, int pos,
                                   float theta_base, int num_heads) {
    int h = blockIdx.x, d = threadIdx.x;
    if (h >= num_heads || d >= head_dim/2) return;
    int hd2 = head_dim/2;
    float f = 1.0f / powf(theta_base, (float)d / (float)hd2);
    float c = cosf(pos * f), s = sinf(pos * f);
    int idx = h * head_dim + d;
    float a = x[idx], b = x[idx + hd2];
    x[idx] = a*c - b*s; x[idx + hd2] = a*s + b*c;
}

// ── Float→Half ──
__global__ void fused_f2h_kernel(__half* dst, const float* src, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = __float2half(src[i]);
}

// ── Half→Float ──
__global__ void fused_h2f_kernel(float* dst, const __half* src, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = __half2float(src[i]);
}

// ── KV store: f32 K,V → f16 KV cache ──
__global__ void fused_kv_store_kernel(__half* __restrict__ dK, __half* __restrict__ dV,
                                       const float* __restrict__ k, const float* __restrict__ v,
                                       int pos, int NKV, int HD, int max_seq) {
    int h = blockIdx.x, d = threadIdx.x;
    if (h >= NKV || d >= HD) return;
    size_t off = (size_t)pos * NKV * HD + (size_t)h * HD + d;
    dK[off] = __float2half(k[(size_t)h * HD + d]);
    dV[off] = __float2half(v[(size_t)h * HD + d]);
}

// ── Output projection: y[H] = Wo[NH*HD, H]^T @ attn[NH*HD] ──
__global__ void fused_out_proj_kernel(float* __restrict__ y,
                                       const float* __restrict__ Wo,
                                       const float* __restrict__ attn,
                                       int H, int NH_HD) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    double sum = 0.0;
    for (int j = 0; j < NH_HD; j++)
        sum += (double)attn[j] * Wo[(size_t)j * H + i];
    y[i] = (float)sum;
}

// ── SiLU: out[i] = sigmoid(gate[i]) * gate[i] * up[i] ──
__global__ void fused_silu_kernel(float* __restrict__ out,
                                   const float* __restrict__ gate,
                                   const float* __restrict__ up, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
}

// ── Element-wise add: x += y ──
__global__ void fused_add_kernel(float* __restrict__ x, const float* __restrict__ y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    x[i] += y[i];
}

// ── Copy: dst = src ──
__global__ void fused_copy_kernel(float* __restrict__ dst, const float* __restrict__ src, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = src[i];
}

// ── Embedding ──
__global__ void fused_embed_kernel(float* __restrict__ dst, const float* __restrict__ embed,
                                    int token_id, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H) dst[i] = embed[(size_t)token_id * H + i];
}

// ── Final RMSNorm (output only, doesn't modify input) ──
__global__ void fused_final_norm_kernel(float* __restrict__ out, const float* __restrict__ x,
                                         const float* __restrict__ w, int H, float eps) {
    int tid = threadIdx.x;
    double local = 0.0;
    for (int i = tid; i < H; i += blockDim.x) local += (double)x[i] * x[i];
    __shared__ float sdata[BLOCK];
    sdata[tid] = (float)local;
    for (int s = blockDim.x/2; s > 0; s >>= 1) { __syncthreads(); if (tid < s) sdata[tid] += sdata[tid + s]; }
    __syncthreads();
    float inv = rsqrtf(sdata[0] / H + eps);
    for (int i = tid; i < H; i += blockDim.x) out[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

// ═══════════════════════════════════════════════════════════════════════════
// FusedBackend
// ═══════════════════════════════════════════════════════════════════════════
struct FusedBackend : Backend {
    int H = 0, NC = 0, NH = 0, NKV = 0, HD_ = 128, IM = 0, VOCAB = 0;
    float rope_theta = 10000.0f;
    int max_seq = 4096;

    hipStream_t stream = nullptr;
    bool gpu_ok = false;

    // GPU weights
    float *d_embed = nullptr, *d_final_norm = nullptr, *d_output = nullptr;
    struct GpuL { float *wq, *wk, *wv, *wo, *w1, *w2, *w3, *pn, *pon; };
    std::vector<GpuL> L;

    // Scratch buffers (pre-allocated)
    float *dh = nullptr;            // [H] — persistent hidden state
    float *datt = nullptr;          // [NH*HD] — Q / attn f32
    float *dgate = nullptr;         // [max(NKV*HD, IM)] — K,V / gate
    float *dup_ = nullptr;          // [max(NKV*HD, IM)] — V / up
    float *doproj = nullptr;        // [H] — Wo @ attn
    float *dffn = nullptr;          // [H] — FFN raw / down output
    float *dlogits = nullptr;       // [VOCAB]
    __half *dQ = nullptr;           // [NH*HD] half
    __half *dAttn = nullptr;        // [NH*HD] half
    __half *dK = nullptr, *dV = nullptr;
    __half *devK = nullptr, *devV = nullptr;
    size_t kvb = 0;

    // NPU (pure C++ module, no HIP context conflict)
    NpuState* npu = nullptr;
    bool npu_ok = false;
    fusion::SharedBO* slot[2] = {};
    // GPU-accessible device pointers into the SharedBO pages.
    // hipHostRegister(host_ptr, hipHostRegisterDefault) exposes the NPU-owned
    // XRT HOST_ONLY pages to the HIP runtime so hipMemcpy can use them as the
    // destination/source without an intermediate CPU bounce buffer (issue #1215).
    void* slot_dev[2] = {};

    // CPU weights (for NPU pack + lm_head)
    std::vector<float> cpu_embed, cpu_final_norm, cpu_output;
    struct CpuL { std::vector<float> w1, w2, w3; };
    std::vector<CpuL> cpu_L;
    int pos = 0;

    FusedBackend() { type = BackendType::GENERIC; name = "Fused GPU+NPU"; }
    ~FusedBackend() override { destroy(); }
    bool can_infer() const override { return true; }

    bool init(const ModelConfig& cfg, const std::string&) override {
        this->cfg = cfg;
        H = cfg.hidden_size; NC = cfg.num_layers; NH = cfg.num_heads;
        NKV = cfg.num_kv_heads; HD_ = cfg.head_dim; IM = cfg.intermediate_size;
        VOCAB = cfg.vocab_size;
        rope_theta = cfg.rope_theta > 0 ? cfg.rope_theta : 10000.0f;
        if (NKV == 0) NKV = NH; if (HD_ == 0) HD_ = 128;
        printf("[fused] H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d V=%d\n",
               H, NC, NH, NKV, HD_, IM, VOCAB);
        int nd = 0;
        if (hipGetDeviceCount(&nd) != hipSuccess || nd == 0) return false;
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipStreamCreate(&stream));

        auto mf = [&](float*& p, size_t n, const char* t) {
            if (n == 0) { p = nullptr; return true; }
            if (hipMalloc(&p, n*4) != hipSuccess) { fprintf(stderr,"[fused] malloc %s fail\n",t); return false; }
            return true;
        };
        auto mh = [&](__half*& p, size_t n, const char* t) {
            if (n == 0) { p = nullptr; return true; }
            if (hipMalloc(&p, n*2) != hipSuccess) { fprintf(stderr,"[fused] malloc %s fail\n",t); return false; }
            return true;
        };
        int s1 = (size_t)NH*HD_, s2 = (size_t)std::max(NKV*HD_, IM), s3 = H;
        if (!mf(dh, H, "dh") || !mf(datt, s1, "datt") || !mf(dgate, s2, "dgate") ||
            !mf(dup_, s2, "dup") || !mf(doproj, H, "doproj") ||
            !mf(dffn, H, "dffn") || !mf(dlogits, VOCAB, "dlogits") ||
            !mh(dQ, s1, "dQ") || !mh(dAttn, s1, "dAttn"))
            return false;

        kvb = (size_t)max_seq * NKV * HD_ * sizeof(__half);
        if (hipHostMalloc(&dK, kvb, hipHostMallocMapped) != hipSuccess ||
            hipHostMalloc(&dV, kvb, hipHostMallocMapped) != hipSuccess) return false;
        memset(dK, 0, kvb); memset(dV, 0, kvb);
        HIP_CHECK(hipHostGetDevicePointer((void**)&devK, dK, 0));
        HIP_CHECK(hipHostGetDevicePointer((void**)&devV, dV, 0));

        // Init NPU (pure C++ module — no HIP context conflict)
        const char* xd = getenv("NPU_XCLBIN_DIR");
        if (!xd) xd = "engine/npu/xclbins";
        npu = npu_state_create(xd, H, IM, NC);
        npu_ok = (npu != nullptr);
        if (!npu_ok) printf("[fused] NPU unavailable — GPU-only\n");

        if (!load_1bp(cfg.model_path)) return false;

        if (npu_ok) {
            // SharedBO needs a persistent xrt::device ref — create once outside
            size_t sb = (size_t)H * sizeof(float) * 2;
            xrt::device npu_for_bo(0);
            slot[0] = fusion::SharedBO::create(npu_for_bo, sb);
            slot[1] = fusion::SharedBO::create(npu_for_bo, sb);
            if (!slot[0] || !slot[1]) {
                fprintf(stderr,"[fused] SharedBO alloc fail — GPU-only\n");
                npu_ok = false;
            } else {
                // Register the NPU-owned pages with HIP so hipMemcpy can use them
                // directly without a bounce buffer.  hipHostRegisterDefault works on
                // Strix Halo (unified memory, APU); hipHostRegisterMapped is not needed
                // because hipMemcpy with these pointers already avoids the CPU copy.
                // On failure we fall back to the vector<float> bounce path (no assert).
                for (int i = 0; i < 2; i++) {
                    void* hp = slot[i]->host_ptr();
                    size_t sz = slot[i]->size();
                    hipError_t reg_err = hipHostRegister(hp, sz, hipHostRegisterDefault);
                    if (reg_err == hipSuccess) {
                        hipError_t gdp_err = hipHostGetDevicePointer(&slot_dev[i], hp, 0);
                        if (gdp_err != hipSuccess) {
                            fprintf(stderr, "[fused] SharedBO hipHostGetDevicePointer slot[%d]: %s\n",
                                    i, hipGetErrorString(gdp_err));
                            (void)hipHostUnregister(hp);
                            slot_dev[i] = nullptr;
                        }
                    } else {
                        fprintf(stderr, "[fused] SharedBO hipHostRegister slot[%d]: %s "
                                "(will use bounce buffer)\n", i, hipGetErrorString(reg_err));
                        slot_dev[i] = nullptr;
                    }
                }
            }
        }

        gpu_ok = true; initialized = true;
        printf(npu_ok ? "[fused] ✅ Fused GPU+NPU\n" : "[fused] ✅ GPU-only\n");
        return true;
    }

    bool load_1bp(const std::string& path) {
        printf("[fused] Loading: %s\n", path.c_str());
        OnebpModel mdl;
        if (!mdl.open(path.c_str())) { fprintf(stderr,"[fused] open fail\n"); return false; }
        auto ld = [&](const char* n, std::vector<float>& v){ return mdl.get_tensor_f32(n,v); };
        ld("token_embd.weight", cpu_embed);
        if (!ld("output_norm.weight", cpu_final_norm)) ld("token_embd_norm.weight", cpu_final_norm);
        if (!ld("output.weight", cpu_output)) ld("lm_head.weight", cpu_output);

        struct Tmp { std::vector<float> wq,wk,wv,wo,w1,w2,w3,pn,pon; };
        std::vector<Tmp> tmp(NC);
        cpu_L.resize(NC);
        char buf[128];
        for (int l = 0; l < NC; l++) {
            auto& t = tmp[l];
            auto gr = [&](const char* blk, const char* leg, std::vector<float>& v, int n) {
                snprintf(buf, sizeof(buf), "blk.%d.%s", l, blk);
                if (!mdl.get_tensor_f32(buf, v)) {
                    snprintf(buf, sizeof(buf), "model.layers.%d.%s", l, leg);
                    mdl.get_tensor_f32(buf, v);
                }
            };
            gr("attn_q.weight","self_attn.q_proj.weight", t.wq, H*NH*HD_);
            gr("attn_k.weight","self_attn.k_proj.weight", t.wk, H*NKV*HD_);
            gr("attn_v.weight","self_attn.v_proj.weight", t.wv, H*NKV*HD_);
            gr("attn_output.weight","self_attn.o_proj.weight", t.wo, NH*HD_*H);
            gr("ffn_gate.weight","mlp.gate_proj.weight", t.w1, H*IM);
            gr("ffn_up.weight","mlp.up_proj.weight", t.w2, H*IM);
            gr("ffn_down.weight","mlp.down_proj.weight", t.w3, IM*H);
            gr("attn_norm.weight","input_layernorm.weight", t.pn, H);
            gr("ffn_norm.weight","post_attention_layernorm.weight", t.pon, H);
            cpu_L[l].w1 = t.w1; cpu_L[l].w2 = t.w2; cpu_L[l].w3 = t.w3;
        }

        auto up = [&](const std::vector<float>& c, float*& g) {
            if (c.empty()) { g = nullptr; return true; }
            if (hipMalloc(&g, c.size()*4) != hipSuccess) return false;
            HIP_CHECK(hipMemcpy(g, c.data(), c.size()*4, hipMemcpyHostToDevice)); return true;
        };
        up(cpu_embed, d_embed); up(cpu_final_norm, d_final_norm); up(cpu_output, d_output);
        L.resize(NC);
        for (int l = 0; l < NC; l++) {
            auto& t = tmp[l]; auto& gl = L[l];
            up(t.wq, gl.wq); up(t.wk, gl.wk); up(t.wv, gl.wv); up(t.wo, gl.wo);
            up(t.w1, gl.w1); up(t.w2, gl.w2); up(t.w3, gl.w3);
            up(t.pn, gl.pn); up(t.pon, gl.pon);
        }

        if (npu_ok && npu) {
            for (int l = 0; l < NC; l++) {
                auto& cl = cpu_L[l];
                if (cl.w1.empty() || cl.w2.empty()) continue;
                npu_state_pack_layer(npu, l, cl.w1.data(), cl.w2.data(), cl.w3.data());
            }
            printf("[fused] NPU weights packed via C++ module\n");
        }
        printf("[fused] 1BP loaded — %d layers\n", NC);
        return true;
    }

    bool reset() override {
        pos = 0; if (dK) memset(dK, 0, kvb); if (dV) memset(dV, 0, kvb); return true;
    }

    static void gemv(float* y, const float* W, const float* x, int M, int N, hipStream_t s) {
        if (!W) return;
        // One block per output row — each block's 256 threads reduce dot product
        fused_gemv_plain_kernel<<<M, BLOCK, 0, s>>>(y, W, x, M, N);
    }

    // ══════════════════════════════════════
    // forward — PURE GPU LOOP
    // ══════════════════════════════════════
    bool forward(int token_id, float* hidden_out) override {
        const int H_ = H, NH_ = NH, NKV_ = NKV, HD_ = this->HD_, IM_ = IM, NC_ = NC;

        // Embedding → GPU
        if (token_id >= 0 && token_id < VOCAB && d_embed)
            fused_embed_kernel<<<(H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dh, d_embed, token_id, H_);
        else
            HIP_CHECK(hipMemset(dh, 0, H_*4));

        for (int l = 0; l < NC_; l++) {
            auto& gl = L[l];
            int s1 = NH_ * HD_, s2 = NKV_ * HD_;

            // ── ATTENTION ──────────────────────────────────
            // 1. RMSNorm (in-place on dh, destroys input — save first)
            fused_copy_kernel<<<(H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(doproj, dh, H_);
            if (gl.pn) fused_rmsnorm_kernel<<<1, BLOCK, 0, stream>>>(dh, gl.pn, H_, EPS);
            else       fused_rmsnorm_kernel<<<1, BLOCK, 0, stream>>>(dh, nullptr, H_, EPS);

            // 2. QKV GEMV (all async on stream)
            if (gl.wq) gemv(datt,  gl.wq, dh, s1, H_, stream);
            if (gl.wk) gemv(dgate, gl.wk, dh, s2, H_, stream);
            if (gl.wv) gemv(dup_,  gl.wv, dh, s2, H_, stream);

            // 3. RoPE
            if (gl.wq) fused_rope_kernel<<<NH_, HD_/2, 0, stream>>>(datt, HD_, pos, rope_theta, NH_);
            if (gl.wk) fused_rope_kernel<<<NKV_, HD_/2, 0, stream>>>(dgate, HD_, pos, rope_theta, NKV_);

            // 4. Q→half + KV store
            if (gl.wo) {
                HIP_CHECK(hipStreamSynchronize(stream));
                fused_f2h_kernel<<<(s1+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dQ, datt, s1);
                fused_kv_store_kernel<<<NKV_, HD_, 0, stream>>>(devK, devV, dgate, dup_, pos, NKV_, HD_, max_seq);
                HIP_CHECK(hipStreamSynchronize(stream));

                // 5. Flash-attention
                float scl = 1.0f / sqrtf((float)HD_);
                rcpp_kv_cache_attn_decode(dQ, devK, devV, dAttn, NH_, NKV_, HD_, pos+1, scl, (void*)stream);

                // 6. attn half→f32 + output projection
                fused_h2f_kernel<<<(s1+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(datt, dAttn, s1);
                gemv(doproj, gl.wo, datt, H_, s1, stream);

                // 7. Residual: doproj += saved dh (original input)
                fused_add_kernel<<<(H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(doproj, dh, H_);
                // doproj = attn_out now. Copy back to dh for FFN.
                fused_copy_kernel<<<(H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dh, doproj, H_);
            }

            // ── FFN (GPU by default, NPU backfill with env flag) ──
            // Set USE_NPU_FFN=1 to use NPU for FFN (slower due to PCIe copies,
            // but lower power). The NPU code runs in a separate C++ translation
            // unit to avoid HIP/XRT context conflicts.
            bool ffn_done = false;
            if (npu && npu_ok && getenv("USE_NPU_FFN")) {
                HIP_CHECK(hipStreamSynchronize(stream));
                int si = l & 1;
                float* host_buf = (float*)slot[si]->host_ptr();
                // Zero-copy path (issue #1215): GPU writes the post-attention hidden
                // state directly into the SharedBO pages via the HIP-registered device
                // pointer.  NPU then reads from host_ptr (same physical pages on Strix
                // Halo unified memory) without an intermediate bounce buffer.
                // If registration failed at init time, fall back to hipMemcpy to CPU.
                if (slot_dev[si]) {
                    HIP_CHECK(hipMemcpy(slot_dev[si], doproj, H_*sizeof(float),
                                        hipMemcpyDeviceToDevice));
                } else {
                    HIP_CHECK(hipMemcpy(host_buf, doproj, H_*sizeof(float),
                                        hipMemcpyDeviceToHost));
                }
                if (npu_state_ffn(npu, l, host_buf, H_)) {
                    // Read result back to the GPU hidden-state buffer.
                    if (slot_dev[si]) {
                        HIP_CHECK(hipMemcpy(dh, slot_dev[si], H_*sizeof(float),
                                            hipMemcpyDeviceToDevice));
                    } else {
                        HIP_CHECK(hipMemcpy(dh, host_buf, H_*sizeof(float),
                                            hipMemcpyHostToDevice));
                    }
                    ffn_done = true;
                } else {
                    fprintf(stderr, "[fused] NPU FFN l=%d failed — fallback GPU\n", l);
                    npu_ok = false;
                }
            }
            if (!ffn_done) {
            // ── GPU FFN ──
            fused_copy_kernel<<<(H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dffn, dh, H_);
            if (gl.pon) fused_rmsnorm_kernel<<<1, BLOCK, 0, stream>>>(dh, gl.pon, H_, EPS);
            else        fused_rmsnorm_kernel<<<1, BLOCK, 0, stream>>>(dh, nullptr, H_, EPS);
            if (gl.w1 && gl.w2 && gl.w3) {
                gemv(dgate, gl.w1, dh, IM_, H_, stream);
                gemv(dup_,  gl.w2, dh, IM_, H_, stream);
                fused_silu_kernel<<<(IM_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(datt, dgate, dup_, IM_);
                gemv(dh, gl.w3, datt, H_, IM_, stream);
                fused_add_kernel<<<(H_+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(dh, dffn, H_);
            }
            } // end if (!ffn_done)
        } // end for (int l = 0; l < NC_; l++)

        // Final RMSNorm + readback (runs after ALL NC_ layers, not inside the loop)
        fused_final_norm_kernel<<<1, BLOCK, 0, stream>>>(dh, dh, d_final_norm, H_, EPS);
        HIP_CHECK(hipMemcpy(hidden_out, dh, H_*4, hipMemcpyDeviceToHost));
        HIP_CHECK(hipStreamSynchronize(stream));
        pos++;
        return true;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        // Upload hidden to GPU, run GEMV, read back
        HIP_CHECK(hipMemcpy(dh, hidden, H*sizeof(float), hipMemcpyHostToDevice));
        if (d_output) {
            gemv(dlogits, d_output, dh, VOCAB, H, stream);
        } else if (d_embed) {
            gemv(dlogits, d_embed, dh, VOCAB, H, stream);
        }
        HIP_CHECK(hipMemcpy(logits, dlogits, VOCAB*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipStreamSynchronize(stream));
        if (argmax) { *argmax=0; float mv=logits[0]; for(int v=1;v<VOCAB;v++){ if(logits[v]>mv){ mv=logits[v]; *argmax=v; } } }
        return true;
    }

    int generate(int token_id) override {
        std::vector<float> h(H);
        if (!forward(token_id, h.data())) return -1;
        std::vector<float> l(VOCAB); int n=-1;
        if (!lm_head(h.data(), l.data(), &n)) return -1;
        return n;
    }

    float benchmark(int tokens) override {
        if (!initialized) return -1;
        reset();
        auto t0 = std::chrono::steady_clock::now();
        int tok = 1;
        for (int i = 0; i < tokens; i++) { tok = generate(tok); if (tok < 0) break; }
        auto t1 = std::chrono::steady_clock::now();
        return (float)(std::chrono::duration<double,std::milli>(t1-t0).count() / tokens);
    }

    void destroy() override {
        // Helper that frees AND nulls the pointer
        auto hf = [](float*& p) { if (p) { HIP_CHECK_D(hipFree(p)); p = nullptr; } };
        auto hfh = [](__half*& p) { if (p) { HIP_CHECK_D(hipFree(p)); p = nullptr; } };
        auto hfhst = [](__half*& p) { if (p) { HIP_CHECK_D(hipHostFree(p)); p = nullptr; } };
        hf(dh); hf(datt); hf(dgate); hf(dup_); hf(doproj); hf(dffn); hf(dlogits);
        hfh(dQ); hfh(dAttn);
        hf(d_embed); hf(d_final_norm); hf(d_output);
        for(auto& l : L){ hf(l.wq);hf(l.wk);hf(l.wv);hf(l.wo);hf(l.w1);hf(l.w2);hf(l.w3);hf(l.pn);hf(l.pon); }
        L.clear();
        hfhst(dK); hfhst(dV);
        devK = devV = nullptr;
        for (int i = 0; i < 2; i++) {
            if (slot_dev[i] && slot[i]) {
                HIP_CHECK_D(hipHostUnregister(slot[i]->host_ptr()));
                slot_dev[i] = nullptr;
            }
            delete slot[i]; slot[i] = nullptr;
        }
        if (stream) { HIP_CHECK_D(hipStreamDestroy(stream)); stream = nullptr; }
        npu_state_destroy(npu); npu = nullptr;
        cpu_L.clear(); cpu_embed.clear(); cpu_final_norm.clear(); cpu_output.clear();
        gpu_ok = false; npu_ok = false; initialized = false;
    }
};

extern "C" Backend* create_fused_backend() {
    return static_cast<Backend*>(new FusedBackend());
}
