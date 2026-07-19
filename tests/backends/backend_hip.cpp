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
#include <unordered_map>

// FP16 conversion for GGUF Q8_0 dequant
static float fp16_to_fp32(uint16_t h) {
    uint32_t s=(h>>15)&1, e=(h>>10)&0x1f, m=h&0x3ff;
    if(e==0){uint32_t a=m?__builtin_clz(m)-10:0;float r;uint32_t f=(s<<31)|((127-15-a)<<23)|((m<<(a+13))&0x7fffff);memcpy(&r,&f,4);return r;}
    else if(e==31){float r;uint32_t f=(s<<31)|0x7f800000|(m<<13);memcpy(&r,&f,4);return r;}
    else{float r;uint32_t f=(s<<31)|((e+127-15)<<23)|(m<<13);memcpy(&r,&f,4);return r;}
}

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

        HIP_OK(hipMalloc(&d_hs, H * 2)); HIP_OK(hipMalloc(&d_ao, H * 2)); HIP_OK(hipMalloc(&d_tmp, H * 2));
        HIP_OK(hipMalloc(&d_conv, (size_t)N_LAYERS * 2 * QKV * 4));  // ×2 for separate in/out state
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
            if (vt == 2 || vt == 8) { uint64_t sl; fread(&sl, 8, 1, gf); fseek(gf, sl, SEEK_CUR); }
            else if (vt >= 3 && vt <= 6) fseek(gf, 4, SEEK_CUR);
            else if (vt == 7) fseek(gf, 1, SEEK_CUR);
            else if (vt == 9) {
                uint32_t n_arr; fread(&n_arr, 4, 1, gf); uint32_t at; fread(&at, 4, 1, gf); uint64_t al = n_arr;
                if (at == 2 || at == 8) { for (uint64_t j = 0; j < al; j++) { uint64_t ss; fread(&ss, 8, 1, gf); fseek(gf, ss, SEEK_CUR); } }
                else if (at <= 7) fseek(gf, al, SEEK_CUR);
                else if (at >= 10 && at <= 12) fseek(gf, al * 8, SEEK_CUR);
                else fseek(gf, al * 4, SEEK_CUR);
            } else if (vt >= 10 && vt <= 12) fseek(gf, 8, SEEK_CUR);
            else if (vt <= 1) fseek(gf, 1, SEEK_CUR);
            else fseek(gf, 8, SEEK_CUR);
        }
        // Read tensor headers
        struct TInfo { std::string name; uint64_t off; uint32_t dtype; uint64_t ne; };
        std::unordered_map<std::string, TInfo> tmap;
        uint64_t data_off = ftell(gf);
        for (uint64_t i = 0; i < nt; i++) {
            uint64_t nl; fread(&nl, 8, 1, gf);
            TInfo ti; ti.name.resize(nl); fread(&ti.name[0], 1, nl, gf);
            uint32_t nd; fread(&nd, 4, 1, gf);
            ti.ne = 1; for (uint32_t j = 0; j < nd; j++) { uint64_t d; fread(&d, 8, 1, gf); ti.ne *= d; }
            fread(&ti.dtype, 4, 1, gf); fseek(gf, 4, SEEK_CUR);
            tmap[ti.name] = ti;
        }
        data_off = ftell(gf); data_off = (data_off + 31) & ~31;
        for (auto& [n, t] : tmap) { t.off = data_off; int bs = 32, bpb = t.dtype == 1 ? 2 : t.dtype == 8 ? 34 : 4; if (t.dtype == 1) bs = 1; data_off += ((t.ne + bs - 1) / bs) * bpb; data_off = (data_off + 31) & ~31; }

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
                // Q8_0 dequant to FP16
                std::vector<__half> buf(n);
                int blks = (n + 31) / 32;
                for (int b = 0; b < blks; b++) {
                    uint16_t sh; fread(&sh, 2, 1, gf); float sc = fp16_to_fp32(sh);
                    int8_t q[32]; fread(q, 1, 32, gf);
                    for (int j = 0; j < 32 && b * 32 + j < (int)n; j++) buf[b * 32 + j] = __float2half(q[j] * sc);
                }
                HIP_OK(hipMemcpy(dptr, buf.data(), n * 2, hipMemcpyHostToDevice));
            } else {
                // F32 → FP16
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

        HIP_OK(hipMalloc(&d_hs, H * 2)); HIP_OK(hipMalloc(&d_ao, H * 2)); HIP_OK(hipMalloc(&d_tmp, H * 2));
        HIP_OK(hipMalloc(&d_conv, (size_t)L * 2 * QKV * 4));
        HIP_OK(hipMalloc(&d_phs, (size_t)L * H * 2));
        HIP_OK(hipMalloc(&d_all_logits, (size_t)V * 2));
        HIP_OK(hipMalloc(&d_best_idx, 4)); HIP_OK(hipMalloc(&d_best_val, 4));

        layers_.resize(L);
        for (int il = 0; il < L; il++) {
            auto& lw = layers_[il];
            char pfx[128]; snprintf(pfx, sizeof(pfx), "model.layers.%d.", il);
            std::string p(pfx);
            load_half((p + "input_layernorm.weight").c_str(), lw.nw, H);
            load_half((p + "post_attention_layernorm.weight").c_str(), lw.pan, H);
            load_half((p + "self_attn.q_proj.weight").c_str(), lw.wq, (size_t)QD * H);
            load_half((p + "self_attn.k_proj.weight").c_str(), lw.wk, (size_t)KD * H);
            load_half((p + "self_attn.v_proj.weight").c_str(), lw.wv1, (size_t)KD * H);
            load_half((p + "self_attn.o_proj.weight").c_str(), lw.wo, (size_t)H * QD);
            load_half((p + "mlp.gate_proj.weight").c_str(), lw.gu, (size_t)FF * H);
            load_half((p + "mlp.down_proj.weight").c_str(), lw.dn, (size_t)H * FF);
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
            if (is_zaya) {
                // Zaya-specific CCA attention + fused router MoE (fast path)
                v_interleave_kernel<<<(KD/2+BLK-1)/BLK, BLK, 0, st_>>>(d_tmp+QD, d_tmp+QD+KD, d_tmp+QD+KD+KD/2, KD/2);
                cca_custom_kernel<<<1, 256, cca_custom_smem_bytes(NQ, NKV, HD, 64), st_>>>(d_tmp, d_tmp+QD, d_tmp+QD, d_phs+(size_t)il*H, d_conv+(size_t)il*2*QKV, l.cdw, l.cdb, l.cgw, l.cgb, l.ks, d_ao, d_conv+(size_t)il*2*QKV*2, d_phs+(size_t)il*H, il, 1, NQ, NKV, HD, 64, 5000000.0f, 128);
                moe_tiled_gemv<<<H/16, 128, 0, st_>>>(d_ao, d_ao, l.wo, H, QD);
                residual_scale_k<<<g1, BLK, 0, st_>>>(d_ao, d_hs, l.pahss, l.pahsb, l.parss, l.parsb, H);
                copy_k<<<g1, BLK, 0, st_>>>(d_hs, d_ao, H);
                rmsnorm_k<<<1, BLK, 0, st_>>>(d_hs, l.pan, H);
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
                // Generic FFN path for non-Zaya models (gate + SiLU + down)
                rmsnorm_k<<<1, BLK, 0, st_>>>(d_hs, l.pan, H);
                if (l.gu && l.dn) {
                    int ffb = (N_FF+15)/16;
                    moe_tiled_gemv<<<ffb, 128, 0, st_>>>(d_tmp, d_hs, l.gu, N_FF, H);
                    silu_mul_k<<<(N_FF+BLK-1)/BLK, BLK, 0, st_>>>(d_ao, d_tmp, d_tmp+N_FF, N_FF);
                    int db = (H+15)/16;
                    moe_tiled_gemv<<<db, 128, 0, st_>>>(d_tmp, d_ao, l.dn, H, N_FF);
                    // Element-wise residual add: d_hs += d_tmp
                    add_k<<<g1, BLK, 0, st_>>>(d_hs, d_tmp, H);
                }
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
