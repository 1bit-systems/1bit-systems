// backend_hip.cpp — AMD ROCm HIP backend for RDNA 3.5+ GPUs
// Part of the unified zaya_server binary. Compiled when USE_HIP=ON.
#include "backend.h"
#include "rocm_cpp/ck_gemm.h"
#include "gguf_reader.h"
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

#define HIP_OK_LINE(e, line) do { auto _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP Error %d at line %d: %s\n", _s, line, hipGetErrorString(_s)); abort(); } } while (0)
#define HIP_OK(e) HIP_OK_LINE(e, __LINE__)
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
// fused_qkv_gemv uses H as a parameter name — undef the compile-time
// constant first so it doesn't expand to "int 2048"
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
#include "../../kernels/fused_qkv_gemv.hip"
#include "../../kernels/q4k_gemv.hip"

// These are re-defined by HipBackend::forward() at runtime
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
    // Packed Q4_K/Q6_K weight pointers (non-null = use packed kernel instead of fp16)
    // Packed Q4_K/Q6_K weight pointers (non-null = use packed kernel instead of fp16)
    uint8_t *pk_wq=nullptr, *pk_wk=nullptr, *pk_wv=nullptr, *pk_wo=nullptr;
    uint8_t *pk_wgate=nullptr, *pk_wup=nullptr, *pk_wdown=nullptr;
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
    uint8_t *pk_embed=nullptr;  // packed Q4_K embedding for lm_head
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
    size_t vram_total_mb_ = 0;

    /// Print memory budget estimate for the model.
    void check_vram_budget(const ModelConfig& cfg) {
        int H = cfg.hidden_size, V = cfg.vocab_size, L = cfg.num_layers;
        int NKV = cfg.num_kv_heads, HD = cfg.head_dim;
        int KD = NKV * HD;
        int FF = cfg.intermediate_size;
        int N_EXP = cfg.num_experts;

        // Rough fp16 estimate (weights + KV cache + scratch)
        double fp16_mb = (double)V * H * 2.0 / (1024*1024);
        fp16_mb += (double)L * cfg.max_seq_len * KD * 2.0 / (1024*1024);
        fp16_mb += 32.0;  // scratch
        fp16_mb += (double)L * 4.0 * H * H * 2.0 / (1024*1024);  // ~4*H*H per layer
        if (N_EXP > 0) {
            fp16_mb += (double)N_EXP * 3.0 * FF * H * 2.0 / (1024*1024);
        } else {
            fp16_mb += (double)L * 3.0 * FF * H * 2.0 / (1024*1024);
        }
        double q4k_mb = fp16_mb / 3.6;
        double pct = 100.0 * fp16_mb / vram_total_mb_;
        fprintf(stderr, "  HIP memory: ~%.0f MB needed (fp16) / ~%.0f MB (Q4_K), "
                "%zu MB available (%.1f%%).\n",
                fp16_mb, q4k_mb, vram_total_mb_, pct);
        if (fp16_mb > vram_total_mb_) {
            fprintf(stderr, "  \xe2\x9a\xa0\xef\xb8\x8f  Model exceeds available GPU memory "
                    "(%.0f > %zu MB). Q4_K (~%.0f MB) recommended.\n",
                    fp16_mb, vram_total_mb_, q4k_mb);
        }
    }

    bool is_available() override {
        if (avail_checked_) return avail_cached_;
        avail_checked_ = true;
        int count = 0;
        hipError_t e = hipGetDeviceCount(&count);
        if (e != hipSuccess || count == 0) { return false; }
        hipDeviceProp_t props;
        HIP_OK(hipGetDeviceProperties(&props, 0));
        vram_total_mb_ = props.totalGlobalMem / (1024*1024);
        fprintf(stderr, "  HIP: found %s (%d CU, %zu MB VRAM)\n", props.name, props.multiProcessorCount, vram_total_mb_);
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
        check_vram_budget(cfg);
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
        f(pk_embed); pk_embed=nullptr;
        d_hs=d_ao=d_tmp=d_fnw=d_embed_gpu=nullptr;
        d_conv=d_phs=nullptr; d_prev_rs=nullptr;
        d_expert_idx=nullptr; d_expert_wt=nullptr;
        d_all_logits=nullptr; d_best_idx=nullptr; d_best_val=nullptr;
        d_kcache=nullptr; d_vcache=nullptr;
        for (auto& l : layers_) {
            for (auto* p : {&l.nw,&l.wq,&l.wk,&l.wv1,&l.wv2,&l.wo,&l.pan,&l.gu,&l.dn,&l.up,&l.qb,&l.kb,&l.vb,&l.qn,&l.kn}) f(*p);
            for (auto* p : {&l.cdw,&l.cdb,&l.cgw,&l.cgb,&l.ks,&l.pahss,&l.pahsb,&l.parss,&l.parsb,&l.gdw,&l.gdb,&l.rfn,&l.rf1,&l.rf1b,&l.rf2,&l.rf2b,&l.rout,&l.bb,&l.pmhss,&l.pmhsb,&l.pmrss,&l.pmrsb}) f(*p);
            for (auto* p : {&l.pk_wgate,&l.pk_wup,&l.pk_wdown}) f(*p);
        }
        layers_.clear(); embed_loaded_ = false; loaded_ = false;
        if (st_) { HIP_OK(hipStreamDestroy(st_)); st_ = 0; }
    }

    // ── GGUF weight loading ──────────────────────────────────────
    bool load_gguf_model(const ModelConfig& cfg) {
        GgufReader reader;
        if (!reader.open(cfg.model_path)) return false;

        int H = cfg.hidden_size, V = cfg.vocab_size, L = cfg.num_layers;
        int NQ = cfg.num_heads, NKV = cfg.num_kv_heads, HD = cfg.head_dim;
        int QD = NQ * HD, KD = NKV * HD, QKV = QD + KD;
        int FF = cfg.intermediate_size;

        // Dequantizes to f32 via the shared reader, then converts to fp16
        // for upload — one path for every dtype the shared reader knows
        // about, instead of a per-dtype special case here (which used to
        // leave Q4_0/Q4_1/Q5_0/Q5_1/Q8_K silently un-decoded, falling
        // through to the F32 branch and reading garbage).
        auto load_half = [&](const char* name, __half*& dptr, size_t n) {
            std::vector<float> f32;
            if (!reader.get_tensor_f32(name, f32) || f32.size() != n) return false;
            HIP_OK(hipMalloc(&dptr, n * 2));
            std::vector<__half> buf(n);
            for (size_t i = 0; i < n; i++) buf[i] = __float2half(f32[i]);
            HIP_OK(hipMemcpy(dptr, buf.data(), n * 2, hipMemcpyHostToDevice));
            return true;
        };

        // Load a tensor in native Q4_K/Q6_K packed format (no dequant to fp16).
        // Returns true AND sets dptr to the packed GPU buffer if the tensor
        // has a packed dtype; returns false if the tensor is already fp16/f32.
        // The caller should fall back to load_half on false.
        auto load_packed = [&](const char* name, uint8_t*& dptr, size_t numel) -> bool {
            const GgufTensorInfo* ti = reader.tensor_info(name);
            if (!ti) return false;
            // Only pack Q4_K (dtype 12, 144 bytes/block). Q6_K (dtype 14, 240
            // bytes/block) is NOT packed because q4k_gemv_wmma hardcodes 144.
            if (ti->dtype != 12) return false;
            std::vector<uint8_t> raw;
            uint64_t actual_numel = 0;
            if (!reader.get_tensor_raw(name, 256, 144, raw, &actual_numel)) return false;
            HIP_OK(hipMalloc(&dptr, raw.size()));
            HIP_OK(hipMemcpy(dptr, raw.data(), raw.size(), hipMemcpyHostToDevice));
            fprintf(stderr, "  Packed %s: dtype=%u %zu bytes (%.1f MB)\n",
                    name, ti->dtype, raw.size(), (double)raw.size() / (1024*1024));
            return true;
        };

        // Detect actual vocab size from the first embedding tensor found
        {
            int tcount = 0;
            for (const auto& tname : reader.tensor_names()) {
                const GgufTensorInfo* ti = reader.tensor_info(tname);
                if (tcount++ < 5) fprintf(stderr, "  HIP: tensor[%d] %s ne=%lu\n", tcount-1, tname.c_str(), (unsigned long)ti->numel);
                if (tname.find("embed") != std::string::npos && tname.find("norm") == std::string::npos
                    && ti->numel > 0 && H > 0) {
                    int actual_v = (int)(ti->numel / H);
                    fprintf(stderr, "  HIP: %s ne=%lu H=%d -> V=%d (was %d)\n", tname.c_str(), (unsigned long)ti->numel, H, actual_v, V);
                    if (actual_v > 100 && actual_v != V) {
                        V = actual_v; cfg_.vocab_size = V; cfg_.vocab = V;
                    }
                    break;
                }
            }
        }
        if (!load_half("token_embd.weight", d_embed_gpu, (size_t)V * H))
            load_half("model.embed_tokens.weight", d_embed_gpu, (size_t)V * H);
        // Packed lm_head (Q4_K only for correctness)
        load_packed("token_embd.weight", pk_embed, (size_t)V * H);
        if (!pk_embed) load_packed("output.weight", pk_embed, (size_t)V * H);

        if (!load_half("output_norm.weight", d_fnw, H)) {
            load_half("model.norm.weight", d_fnw, H);
        }

        check_vram_budget(cfg);
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
        // d_prev_rs is needed by reset_state() even for non-Zaya models
        int RTR_H = cfg.router_hidden > 0 ? cfg.router_hidden : 256;
        HIP_OK(hipMalloc(&d_prev_rs, (size_t)L * RTR_H * 4));

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
            // Q/K/V/O packed Q4_K only for Q4_K dtypes (V is often Q6_K)
            load_half((p + "attn_q.weight").c_str(), lw.wq, (size_t)QD * H);
            load_half((p + "attn_k.weight").c_str(), lw.wk, (size_t)KD * H);
            load_half((p + "attn_v.weight").c_str(), lw.wv1, (size_t)KD * H);
            load_half((p + "attn_output.weight").c_str(), lw.wo, (size_t)H * QD);
            // Also try packed versions for Q4_K projections
            load_packed((p + "attn_q.weight").c_str(), lw.pk_wq, (size_t)QD * H);
            load_packed((p + "attn_k.weight").c_str(), lw.pk_wk, (size_t)KD * H);
            load_packed((p + "attn_v.weight").c_str(), lw.pk_wv, (size_t)KD * H);
            load_packed((p + "attn_output.weight").c_str(), lw.pk_wo, (size_t)H * QD);
            // Try packed Q4_K/Q6_K first (3.6x less memory), fall back to fp16
            if (!load_packed((p + "ffn_gate.weight").c_str(), lw.pk_wgate, (size_t)FF * H))
                load_half((p + "ffn_gate.weight").c_str(), lw.gu, (size_t)FF * H);
            if (!load_packed((p + "ffn_up.weight").c_str(), lw.pk_wup, (size_t)FF * H))
                load_half((p + "ffn_up.weight").c_str(), lw.up, (size_t)FF * H);
            if (!load_packed((p + "ffn_down.weight").c_str(), lw.pk_wdown, (size_t)H * FF))
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
        // Check that essential tensors actually loaded.
        // GGUF tensor names may not match (Q2_0 models use different names),
        // so if wq is null for layer 0, no weights were found -> fail.
        if (L > 0 && layers_[0].nw == nullptr) {
            fprintf(stderr, "  HIP: GGUF load failed — no per-layer tensors matched "
                    "(wrong quantization? Q2_0 not supported, use Q4_K).\n");
            unload_model();
            return false;
        }
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
            // Q/K/V: use packed Q4_K for Q4_K projections, fp16 fused for Q6_K
            if (l.pk_wq && l.pk_wk && l.pk_wv) {
                int q64=(QD+63)/64, k64=(KD+63)/64;
                q4k_gemv_wmma<<<q64,128,0,st_>>>(d_tmp,d_hs,l.pk_wq,QD,H);
                q4k_gemv_wmma<<<k64,128,0,st_>>>(d_tmp+QD,d_hs,l.pk_wk,KD,H);
                q4k_gemv_wmma<<<k64,128,0,st_>>>(d_tmp+QD+KD,d_hs,l.pk_wv,KD,H);
            } else if (l.wq && l.wk && l.wv1) {
                fused_qkv_gemv<<<(QD + 2*KD + 255) / 256, 256, 0, st_>>>(
                    d_tmp, d_hs, l.wq, l.wk, l.wv1, QD, KD, H);
            } else {
                fprintf(stderr, "  HIP ERROR: QKV pointers null for layer %d "
                        "(model too large or wrong format)\n", il);
                return 0;
            }
            if (l.qb) add_k<<<(QD+BLK-1)/BLK, BLK, 0, st_>>>(d_tmp, l.qb, QD);
            if (l.kb) add_k<<<(KD+BLK-1)/BLK, BLK, 0, st_>>>(d_tmp+QD, l.kb, KD);
            if (l.vb) add_k<<<(KD+BLK-1)/BLK, BLK, 0, st_>>>(d_tmp+QD+KD, l.vb, KD);
            // Per-head Q/K-norm (Qwen3), applied before RoPE.
            if (l.qn) qk_norm_k<<<NQ, BLK, 0, st_>>>(d_tmp, l.qn, HD, cfg_.rms_norm_eps);
            if (l.kn) qk_norm_k<<<NKV, BLK, 0, st_>>>(d_tmp+QD, l.kn, HD, cfg_.rms_norm_eps);
            if (is_zaya) {
                moe_tiled_gemv<<<KD/2/16, 128, 0, st_>>>(d_tmp+QD+KD, d_hs, l.wv1, KD/2, H);
                moe_tiled_gemv<<<KD/2/16, 128, 0, st_>>>(d_tmp+QD+KD+KD/2, d_phs+(size_t)il*H, l.wv2, KD/2, H);
                // Zaya-specific CCA attention + fused router MoE (fast path)
                v_interleave_kernel<<<(KD/2+BLK-1)/BLK, BLK, 0, st_>>>(d_tmp+QD, d_tmp+QD+KD, d_tmp+QD+KD+KD/2, KD/2);
                cca_custom_kernel<<<1, 256, cca_custom_smem_bytes(NQ, NKV, HD, 64), st_>>>(d_tmp, d_tmp+QD, d_tmp+QD, d_phs+(size_t)il*H, d_conv+(size_t)il*2*QKV, l.cdw, l.cdb, l.cgw, l.cgb, l.ks, d_ao, d_conv+(size_t)il*2*QKV*2, d_phs+(size_t)il*H, il, 1, NQ, NKV, HD, 64, 5000000.0f, 128);
                if (l.pk_wo) { int o64=(H+63)/64; q4k_gemv_wmma<<<o64,128,0,st_>>>(d_ao,d_ao,l.pk_wo,H,QD); }
                else moe_tiled_gemv<<<H/16,128,0,st_>>>(d_ao,d_ao,l.wo,H,QD);
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

                // V is now computed by fused_qkv_gemv above (replaces 3 separate GEMVs).

                // RoPE Q+K in place, write RoPE'd K + raw V into this layer's
                // KV cache at position `pos`.
                rcpp_rope_kv_append_fp16(d_tmp, kc, d_tmp+QD+KD, vc, pos, cfg_.rope_theta, NQ, NKV, HD, st_);

                // Causal GQA attention against positions [0, pos] of this
                // layer's KV cache.
                rcpp_kv_cache_attn_decode(d_tmp, kc, vc, d_ao, NQ, NKV, HD, pos + 1, 1.0f / sqrtf((float)HD), st_);

                // O-projection + residual.
                if (l.pk_wo) { int o64=(H+63)/64; q4k_gemv_wmma<<<o64,128,0,st_>>>(d_tmp,d_ao,l.pk_wo,H,QD); }
                else moe_tiled_gemv<<<H/16,128,0,st_>>>(d_tmp,d_ao,l.wo,H,QD);
                add_k<<<g1, BLK, 0, st_>>>(d_phs + (size_t)il*H, d_tmp, H);
                copy_k<<<g1, BLK, 0, st_>>>(d_hs, d_phs + (size_t)il*H, H);

                // Pre-FFN-norm residual, same reused per-layer slot (its
                // pre-attention value is no longer needed at this point).
                copy_k<<<g1, BLK, 0, st_>>>(d_phs + (size_t)il*H, d_hs, H);
                rmsnorm_k<<<1, BLK, 0, st_>>>(d_hs, l.pan, H, cfg_.rms_norm_eps);
                if (l.gu && l.dn && l.up) {
                    // Use packed Q4_K kernel for gate/up/down (3.6x less memory BW)
                    int gb = (N_FF+15)/16;
                    if (l.pk_wgate && l.pk_wup) {
                        int gb64 = (N_FF + 63) / 64;
                        q4k_gemv_wmma<<<gb64, 128, 0, st_>>>(d_tmp, d_hs, l.pk_wgate, N_FF, H);
                        q4k_gemv_wmma<<<gb64, 128, 0, st_>>>(d_tmp+N_FF, d_hs, l.pk_wup, N_FF, H);
                    } else if (l.gu && l.up) {
                        fused_gate_up_gemv<<<(2*N_FF + 255) / 256, 256, 0, st_>>>(d_tmp, d_hs, l.gu, l.up, N_FF, H);
                    } else {
                        fprintf(stderr, "  HIP ERROR: FFN pointers null for layer %d\n", il);
                        return 0;
                    }
                    silu_mul_k<<<(N_FF+BLK-1)/BLK, BLK, 0, st_>>>(d_ao, d_tmp, d_tmp+N_FF, N_FF);
                    int db = (H+15)/16;
                    if (l.pk_wdown) {
                        int db64 = (H + 63) / 64;
                        q4k_gemv_wmma<<<db64, 128, 0, st_>>>(d_tmp, d_ao, l.pk_wdown, H, N_FF);
                    } else {
                        moe_tiled_gemv<<<db, 128, 0, st_>>>(d_tmp, d_ao, l.dn, H, N_FF);
                    }
                    add_k<<<g1, BLK, 0, st_>>>(d_phs + (size_t)il*H, d_tmp, H);
                }
                copy_k<<<g1, BLK, 0, st_>>>(d_hs, d_phs + (size_t)il*H, H);
            }
        }
        rmsnorm_k<<<1, BLK, 0, st_>>>(d_hs, d_fnw, H, cfg_.rms_norm_eps);
        if (pk_embed) {
            int v64 = (VOCAB + 63) / 64;
            q4k_gemv_wmma<<<v64, 128, 0, st_>>>((__half*)d_all_logits, d_hs, pk_embed, VOCAB, H);
        } else {
            lm_head_fused_kernel<<<VOCAB, 256, 0, st_>>>(d_all_logits, d_hs, d_embed_gpu, H, VOCAB);
        }
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
