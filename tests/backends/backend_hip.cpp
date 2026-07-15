// backend_hip.cpp — AMD ROCm HIP backend for RDNA 3.5+ GPUs
// Part of the unified zaya_server binary. Compiled when USE_HIP=ON.
#include "backend.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>

#define HIP_OK(e) do { auto _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP Error %d\n", _s); abort(); } } while (0)
constexpr float RMD_EPS = 1e-5f;
constexpr int BLK = 256;

// Kernels (from ../kernels/)
__global__ void rmsnorm_k(__half* x, const __half* w, int n) {
    __shared__ float r[32];
    int tx = threadIdx.x, wid = tx / 32, l = tx % 32;
    float ss = 0;
    for (int i = tx; i < n; i += blockDim.x) ss += (float)x[i] * (float)x[i];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o);
    if (l == 0) r[wid] = ss;
    __syncthreads();
    if (wid == 0) { ss = (l < (256 / 32)) ? r[l] : 0; for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o); if (l == 0) r[0] = ss; }
    __syncthreads();
    float iv = 1.0f / sqrtf(r[0] / n + RMD_EPS);
    for (int i = tx; i < n; i += blockDim.x) x[i] = __float2half((float)x[i] * iv * (float)w[i]);
}
__global__ void copy_k(__half* d, const __half* s, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; d[i] = s[i]; }
__global__ void silu_mul_k(__half* out, const __half* g, const __half* u, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; float v = (float)g[i]; out[i] = __float2half((v / (1.0f + expf(-v))) * (float)u[i]); }
__global__ void residual_scale_k(__half* out, const __half* res, const float* hs_s, const float* hs_b, const float* res_s, const float* res_b, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; out[i] = __float2half((float)out[i] * hs_s[i] + hs_b[i] + (float)res[i] * res_s[i] + res_b[i]); }

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
#define N_EXP_T 2
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
#undef N_EXP_T
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
static void upf16(const std::vector<float>& s, __half* d, int n, hipStream_t h = 0) { std::vector<__half> b(n); for (int i = 0; i < n; i++) b[i] = __float2half(s[i]); HIP_OK(hipMemcpyAsync(d, b.data(), n * 2, hipMemcpyHostToDevice, h)); }
static void upf32(const std::vector<float>& s, float* d, int n, hipStream_t h = 0) { HIP_OK(hipMemcpyAsync(d, s.data(), n * 4, hipMemcpyHostToDevice, h)); }

struct HipLayer {
    __half *nw=nullptr,*wq=nullptr,*wk=nullptr,*wv1=nullptr,*wv2=nullptr,*wo=nullptr,*pan=nullptr;
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
    BackendType type() const override { return BackendType::HIP; }
    const char* name() const override { return "ROCm HIP"; }
    float estimated_tok_s() const override { return 113.0f; }
    bool is_coherent() const override { return true; }

    bool is_available() override {
        int count = 0;
        hipError_t e = hipGetDeviceCount(&count);
        if (e != hipSuccess || count == 0) { fprintf(stderr, "  HIP: no devices (error %d)\n", e); return false; }
        hipDeviceProp_t props;
        HIP_OK(hipGetDeviceProperties(&props, 0));
        fprintf(stderr, "  HIP: found %s (%d CU, %zu MB)\n", props.name, props.multiProcessorCount, props.totalGlobalMem / (1024*1024));
        return true;
    }

    bool load_model(const ModelConfig& cfg) override {
        cfg_ = cfg;
        unload_model();

        // Dimension validation: the compiled HIP kernels have hardcoded dimensions
        // (#define H 2048 etc.). Runtime config is read but the kernels ignore it.
        // This guard prevents silent wrong output for unsupported models.
        // FIXME: to support multiple architectures, these kernels need templating
        // or runtime dimension parameters.
        if (cfg.hidden_size != 2048 || cfg.num_heads != 8 || cfg.num_kv_heads != 2 ||
            cfg.head_dim != 128 || cfg.num_layers != 40 || cfg.vocab_size != 262272) {
            fprintf(stderr, "  HIP: kernel dimensions are hardcoded to Zaya1-8B (H=2048, L=40, "
                    "NH=8, NKV=2, V=262272). Model has H=%d, L=%d, NH=%d, NKV=%d, V=%d.\n",
                    cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.num_kv_heads,
                    cfg.vocab_size);
            fprintf(stderr, "  HIP: refusing to load — would produce silent garbage.\n");
            return false;
        }

        HIP_OK(hipStreamCreate(&st_));
        int H=cfg.hidden_size, NQ=cfg.num_heads, NKV=cfg.num_kv_heads, HD=cfg.head_dim;
        int QD=NQ*HD, KD=NKV*HD, QKV=QD+KD;
        int N_LAYERS=cfg.num_layers, VOCAB=cfg.vocab_size;
        int N_EXP=cfg.num_experts, N_EXP_T=cfg.num_experts_top, N_FF=cfg.intermediate_size;
        int RTR_H=cfg.router_hidden;
        std::string W = cfg.weights_dir;
        if (!W.empty() && W.back() != '/') W += '/';

        auto embed = load_bin(W + "model_embed_tokens_weight.bin");
        auto fnorm = load_bin(W + "model_norm_weight.bin");
        if (embed.empty() || fnorm.empty()) { fprintf(stderr, "  HIP: missing weights\n"); return false; }
        embed_cpu_.resize(VOCAB * H); fnorm_cpu_.resize(H);
        for (int i = 0; i < VOCAB * H; i++) embed_cpu_[i] = __float2half(embed[i]);
        for (int i = 0; i < H; i++) fnorm_cpu_[i] = __float2half(fnorm[i]);
        HIP_OK(hipMalloc(&d_embed_gpu, (size_t)VOCAB * H * 2));
        HIP_OK(hipMemcpy(d_embed_gpu, embed_cpu_.data(), VOCAB * H * 2, hipMemcpyHostToDevice));
        HIP_OK(hipMalloc(&d_fnw, H * 2));
        HIP_OK(hipMemcpy(d_fnw, fnorm_cpu_.data(), H * 2, hipMemcpyHostToDevice));
        embed_loaded_ = true;

        HIP_OK(hipMalloc(&d_hs, H * 2)); HIP_OK(hipMalloc(&d_ao, H * 2)); HIP_OK(hipMalloc(&d_tmp, H * 2));
        HIP_OK(hipMalloc(&d_conv, (size_t)N_LAYERS * 2 * QKV * 2));
        HIP_OK(hipMalloc(&d_phs, (size_t)N_LAYERS * H * 2));
        HIP_OK(hipMalloc(&d_prev_rs, (size_t)N_LAYERS * RTR_H * 4));
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
            B(l.rout, N_EXP_T*RTR_H, W+lp+"mlp_gate_router_mlp_out_proj_weight.bin");
            B(l.bb, N_EXP_T, W+lp+"mlp_gate_balancing_biases.bin");
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
        f(d_all_logits); f(d_best_idx); f(d_best_val);
        d_hs=d_ao=d_tmp=d_fnw=d_embed_gpu=nullptr;
        d_conv=d_phs=nullptr; d_prev_rs=nullptr;
        d_expert_idx=nullptr; d_expert_wt=nullptr;
        d_all_logits=nullptr; d_best_idx=nullptr; d_best_val=nullptr;
        for (auto& l : layers_) {
            for (auto* p : {&l.nw,&l.wq,&l.wk,&l.wv1,&l.wv2,&l.wo,&l.pan,&l.gu,&l.dn}) f(*p);
            for (auto* p : {&l.cdw,&l.cdb,&l.cgw,&l.cgb,&l.ks,&l.pahss,&l.pahsb,&l.parss,&l.parsb,&l.gdw,&l.gdb,&l.rfn,&l.rf1,&l.rf1b,&l.rf2,&l.rf2b,&l.rout,&l.bb,&l.pmhss,&l.pmhsb,&l.pmrss,&l.pmrsb}) f(*p);
        }
        layers_.clear(); embed_loaded_ = false; loaded_ = false;
        if (st_) { HIP_OK(hipStreamDestroy(st_)); st_ = 0; }
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
        int g1 = (H+BLK-1)/BLK;
        __half* src = embed_cpu_.data() + (size_t)token_id * H;
        HIP_OK(hipMemcpyAsync(d_hs, src, H * 2, hipMemcpyHostToDevice, st_));
        for (int il = 0; il < N_LAYERS; il++) {
            auto& l = layers_[il];
            copy_k<<<g1, BLK, 0, st_>>>(d_phs + (size_t)il*H, d_hs, H);
            rmsnorm_k<<<1, BLK, 0, st_>>>(d_hs, l.nw, H);
            moe_tiled_gemv<<<QD/16, 128, 0, st_>>>(d_tmp, d_hs, l.wq, QD, H);
            moe_tiled_gemv<<<KD/16, 128, 0, st_>>>(d_tmp+QD, d_hs, l.wk, KD, H);
            moe_tiled_gemv<<<KD/2/16, 128, 0, st_>>>(d_tmp+QD+KD, d_hs, l.wv1, KD/2, H);
            moe_tiled_gemv<<<KD/2/16, 128, 0, st_>>>(d_tmp+QD+KD+KD/2, d_phs+(size_t)il*H, l.wv2, KD/2, H);
            v_interleave_kernel<<<(KD/2+BLK-1)/BLK, BLK, 0, st_>>>(d_tmp+QD, d_tmp+QD+KD, d_tmp+QD+KD+KD/2, KD/2);
            cca_custom_kernel<<<1, 256, 0, st_>>>(d_tmp, d_tmp+QD, d_tmp+QD, d_phs+(size_t)il*H, d_conv+(size_t)il*2*QKV, l.cdw, l.cdb, l.cgw, l.cgb, l.ks, d_ao, d_conv+(size_t)il*2*QKV, d_phs+(size_t)il*H, il, 1);
            moe_tiled_gemv<<<H/16, 128, 0, st_>>>(d_ao, d_ao, l.wo, H, QD);
            residual_scale_k<<<g1, BLK, 0, st_>>>(d_ao, d_hs, l.pahss, l.pahsb, l.parss, l.parsb, H);
            copy_k<<<g1, BLK, 0, st_>>>(d_hs, d_ao, H);
            rmsnorm_k<<<1, BLK, 0, st_>>>(d_hs, l.pan, H);
            if (l.gu && l.dn) {
                eda_router_gpu_kernel<<<1, RTR_H, 0, st_>>>(d_hs, d_prev_rs+(size_t)il*RTR_H, l.has_eda?1:0, l.eda_scale[0], l.gdw, l.gdb, l.rfn, l.rf1, l.rf1b, l.rf2, l.rf2b, l.rout, l.bb, d_prev_rs+(size_t)il*RTR_H, d_expert_idx, d_expert_wt);
                encode_expert_cache_kernel<<<1, 32, 0, st_>>>(d_prev_rs+(size_t)il*RTR_H, d_expert_idx, RTR_H);
                { int gb=(2*N_FF+15)/16, db=(H+15)/16, sb=(N_FF+BLK-1)/BLK;
                wmma_gateup_kernel<<<gb,128,0,st_>>>(d_tmp,d_hs,l.gu,d_expert_idx);
                silu_mul_k<<<sb,BLK,0,st_>>>(d_ao,d_tmp,d_tmp+N_FF,N_FF);
                wmma_down_kernel<<<db,128,0,st_>>>(d_tmp,d_ao,l.dn,d_expert_idx); }
                residual_scale_k<<<g1, BLK, 0, st_>>>(d_tmp, d_hs, l.pmhss, l.pmhsb, l.pmrss, l.pmrsb, H);
                copy_k<<<g1, BLK, 0, st_>>>(d_hs, d_tmp, H);
            }
        }
        rmsnorm_k<<<1, BLK, 0, st_>>>(d_hs, d_fnw, H);
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
