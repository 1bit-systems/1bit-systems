// zaya_engine.cpp — Zaya inference engine as a C-callable library.
// Compiles into libzaya_engine.a, linked into token_router.
// No main(), no networking — just the model.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <new>
#include "hip_check.h"

#define HIP_OK_R(e, retval) do { \
    hipError_t _s = (e); \
    if (_s != hipSuccess) { \
        fprintf(stderr, "HIP Error %d at %s:%d — %s\n", _s, __FILE__, __LINE__, hipGetErrorString(_s)); \
        return retval; \
    } \
} while(0)
#define HIP_OK_V(e) do { \
    hipError_t _s = (e); \
    if (_s != hipSuccess) { \
        fprintf(stderr, "HIP Error %d at %s:%d — %s\n", _s, __FILE__, __LINE__, hipGetErrorString(_s)); \
        return; \
    } \
} while(0)


// ── Architecture (compile-time constants for kernels ──
static constexpr int H=2048,NQ=8,NKV=2,HD=128,QD=NQ*HD,KD=NKV*HD,QKV=QD+KD;
static constexpr int N_LAYERS=40,VOCAB=262272,N_EXP=16,N_EXP_T=17,N_FF=2048,RTR_H=256; // N_EXP_T=17 = 16 experts + 1 MOD skip token
static constexpr float RMD_EPS=1e-5f; static constexpr int BLK=256;

// ── Runtime config (host code uses these) ──
// Set by zaya_init before calling zaya_forward. Uses separate names
// from the compile-time constants to avoid shadowing.
#define _H H
#define _NQ NQ
#define _NKV NKV
#define _HD HD
#define _QD QD
#define _KD KD
#define _QKV QKV
#define _N_LAYERS N_LAYERS
#define _VOCAB VOCAB
#define _N_EXP N_EXP
#define _N_EXP_T N_EXP_T
#define _N_FF N_FF
#define _RTR_H RTR_H
static struct {
    int h=_H, nq=_NQ, nkv=_NKV, hd=_HD;
    int qd=_QD, kd=_KD, qkv=_QKV;
    int n_layers=_N_LAYERS, vocab=_VOCAB;
    int n_exp=_N_EXP, n_exp_t=_N_EXP_T, n_ff=_N_FF, rtr_h=_RTR_H;
} eng;
#undef _H
#undef _NQ
#undef _NKV
#undef _HD
#undef _QD
#undef _KD
#undef _QKV
#undef _N_LAYERS
#undef _VOCAB
#undef _N_EXP
#undef _N_EXP_T
#undef _N_FF
#undef _RTR_H

// ── Helper kernels ──
__global__ void rmsnorm_k(__half*x,const __half*w,int n){__shared__ float r[32];int tx=threadIdx.x,wid=tx/32,l=tx%32;float ss=0;for(int i=tx;i<n;i+=blockDim.x)ss+=(float)x[i]*(float)x[i];for(int o=16;o>0;o>>=1)ss+=__shfl_xor(ss,o);if(l==0)r[wid]=ss;__syncthreads();if(wid==0){ss=(l<(256/32))?r[l]:0;for(int o=16;o>0;o>>=1)ss+=__shfl_xor(ss,o);if(l==0)r[0]=ss;}__syncthreads();float iv=1.0f/sqrtf(r[0]/n+1e-5f);for(int i=tx;i<n;i+=blockDim.x)x[i]=__float2half((float)x[i]*iv*(float)w[i]);}
__global__ void copy_k(__half*d,const __half*s,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;d[i]=s[i];}
__global__ void mm_k(__half*out,const __half*in,const __half*wt,int M,int K){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=M)return;float s=0;for(int k=0;k<K;k++)s+=(float)in[k]*(float)wt[k*(size_t)M+i];out[i]=__float2half(s);}
__global__ void silu_mul_k(__half*out,const __half*g,const __half*u,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float v=(float)g[i];out[i]=__float2half((v/(1.0f+expf(-v)))*(float)u[i]);}
__global__ void residual_scale_k(__half*out,const __half*res,const float*hs_s,const float*hs_b,const float*res_s,const float*res_b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;out[i]=__float2half((float)out[i]*hs_s[i]+hs_b[i]+(float)res[i]*res_s[i]+res_b[i]);}

// ── Complex kernels from .hip files ──
#include "zaya_cca_attn.hip"
#include "zaya_gpu_router.hip"
#include "zaya_router_moe.hip"
#include "zaya_moe_tiled_gemv.hip"
#include "zaya_moe_expert_ffn.hip"
#include "zaya_moe_batch_union.hip"
#include "argmax_kernel.hip"
#include "zaya_cca_custom.hip"
#include "v_interleave_kernel.hip"

// ── Persistent thread blocks for MoE expert FFN (single grid, all layers) ──
#include "zaya_persistent_moe.hip"

// ── Reference-faithful CCA Q/K/V prep. ──
#include "zaya_cca_prep.hip"

// ── Post-router skip-expert fixup. ──
#include "zaya_skip_fixup.hip"

// ── GQA V-broadcast for batch path. ──
#include "zaya_batch_v_attn.hip"

// ── Flash-decoding KV-cache attention (ensure kv_cache_attn_fd.hip is linked). ──
extern "C" int rcpp_kv_cache_attn_decode_fd(const void* Q,const void* K,const void* V,void* out,
                                            int num_q_heads,int num_kv_heads,int head_dim,
                                            int seq_len,float scale,void* stream);

// ── rocWMMA batched GEMV (requires rocWMMA header library) ──
// Guarded: the CMakeLists.txt sets ROCWMMA_FOUND and the include path.
// If rocWMMA is not available, this kernel is skipped and the engine
// falls back to the scalar-tiled batched GEMV.
#if __has_include(<rocwmma/rocwmma.hpp>)
#include "zaya_moe_wmma_batched.hip"
#endif

// ── Weight loading ──
struct LayerW{__half*nw,*wq,*wk,*wv1,*wv2,*wo,*pan;float*cdw,*cdb,*cgw,*cgb,*ks;float*pahss,*pahsb,*parss,*parsb;float*gdw,*gdb,*rfn,*rf1,*rf1b,*rf2,*rf2b,*rout,*bb;__half*gu,*dn;float*pmhss,*pmhsb,*pmrss,*pmrsb;};

static std::vector<float> load_bin(const std::string& p){
    std::ifstream f(p,std::ios::binary|std::ios::ate);
    if(!f){fprintf(stderr,"Missing: %s\n",p.c_str());return {};}
    size_t n=f.tellg()/sizeof(float);f.seekg(0);
    std::vector<float> d(n);f.read((char*)d.data(),n*sizeof(float));return d;
}
static std::string L(int i){return std::to_string(i);}
static const std::string g_weights_dir = []() -> std::string {
    const char* env = getenv("ZAYA_WEIGHTS_DIR");
    if (env && env[0]) return env;
    const char* xdg = getenv("XDG_DATA_HOME");
    if (xdg && xdg[0]) return std::string(xdg) + "/1bit-systems/weights/";
    const char* home = getenv("HOME");
    if (home && home[0]) return std::string(home) + "/.local/share/1bit-systems/weights/";
    return "/tmp/zaya_weights/";
}();
#define W(N) load_bin(g_weights_dir+N)
static void upf16(const std::vector<float>& s,__half*d,int n,hipStream_t h=0){
    std::vector<__half>b(n);for(int i=0;i<n;i++)b[i]=__float2half(s[i]);
    HIP_OK_V(hipMemcpyAsync(d,b.data(),n*2,hipMemcpyHostToDevice,h));
}
static void upf32(const std::vector<float>& s,float*d,int n,hipStream_t h=0){
    HIP_OK_V(hipMemcpyAsync(d,s.data(),n*4,hipMemcpyHostToDevice,h));
}

extern "C" {

// ── Forward declarations ──
struct ZayaState;
void zaya_destroy(ZayaState* s);

// ── WMMA defines (redefined after zaya_moe_wmma_batched.hip undefs them) ──
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 64
#define WMMA_THREADS 128

// ── Public state ──
struct ZayaState {
    __half *d_hs=nullptr,*d_ao=nullptr,*d_tmp=nullptr,*d_fnw=nullptr,*d_lm_out=nullptr,*d_embed=nullptr;
    __half *d_conv=nullptr,*d_phs=nullptr,*d_lm_vocab=nullptr; float *d_prev_rs=nullptr;
    __half *d_kcache=nullptr,*d_vcache=nullptr,*d_vrec=nullptr;
    __half *d_qout=nullptr,*d_kout=nullptr,*d_vout=nullptr;
    int *d_skip_flag=nullptr;
    int pos=0, max_seq=4096;
    int *d_argmax_idx=nullptr; float *d_argmax_val=nullptr;
    int *d_expert_idx=nullptr; float *d_expert_wt=nullptr;
    int *d_sorted_ids=nullptr,*d_expert_counts=nullptr,*d_expert_offsets=nullptr;
    hipStream_t st=nullptr;
    std::vector<LayerW> lw;
    std::vector<bool> has_eda;
    std::vector<float> eda_scale;
    std::vector<float> embed, ibias, iscale;
};

// ── Init: load weights, allocate GPU memory ──
ZayaState* zaya_init(const char* weights_dir = nullptr) {
    // g_weights_dir is now const — set from ZAYA_WEIGHTS_DIR or XDG_DATA_HOME
    // Use the parameter directly where weights_dir is needed instead
    ZayaState* s = new (std::nothrow) ZayaState();
    if (!s) {
        fprintf(stderr, "zaya_init: failed to allocate ZayaState (OOM)\n");
        return nullptr;
    }
    HIP_OK_R(hipStreamCreate(&s->st), nullptr);
    
    s->embed = W("model_embed_tokens_weight.bin");
    auto fnorm = W("model_norm_weight.bin");
    s->iscale = W("model_input_hidden_states_scale.bin");
    s->ibias = W("model_input_hidden_states_bias.bin");

    // If any of the four initial weight files is missing, abort init gracefully
    // instead of crashing downstream (fixes #61).
    if (s->embed.empty() || fnorm.empty() || s->iscale.empty() || s->ibias.empty()) {
        fprintf(stderr, "zaya_init: failed to load one or more initial weight files — aborting init\n");
        zaya_destroy(s);
        return nullptr;
    }

    // Verify a GPU is available before attempting allocations
    int ndev = 0;
    HIP_OK_R(hipGetDeviceCount(&ndev), nullptr);
    if (ndev < 1) {
        fprintf(stderr, "zaya_init: No HIP-capable GPU found (device count=%d).\n", ndev);
        zaya_destroy(s);
        return nullptr;
    }

    // Dimension validation: check loaded weights match compile-time constants.
    // The engine uses compile-time #{H}x#{NQ}x#{NKV} kernel dimensions and cannot
    // dynamically adapt to different model sizes. (Issue #249)
    size_t expected_embed = (size_t)eng.vocab * eng.h;
    if (s->embed.size() != expected_embed) {
        fprintf(stderr, "zaya_init: model embed size %zu != expected %zu (H=%d, vocab=%d)\n",
                s->embed.size(), expected_embed, eng.h, eng.vocab);
        fprintf(stderr, "  This engine was compiled for H=%d, NQ=%d, NKV=%d, L=%d, V=%d.\n",
                H, NQ, NKV, N_LAYERS, VOCAB);
        fprintf(stderr, "  Refusing to load — would produce silent garbage.\n");
        zaya_destroy(s);
        return nullptr;
    }

    // Allocate GPU buffers with cleanup on failure (fixes #279 — leak on alloc error)
    auto alloc_f16 = [&](auto& p, int n) -> bool {
        hipError_t _e = hipMalloc(&p, n*2);
        if (_e != hipSuccess) { fprintf(stderr,"HIP OOM at %s:%d — %s\n",__FILE__,__LINE__,hipGetErrorString(_e)); return false; }
        return true;
    };
    auto alloc_f32 = [&](auto& p, int n) -> bool {
        hipError_t _e = hipMalloc(&p, n*4);
        if (_e != hipSuccess) { fprintf(stderr,"HIP OOM at %s:%d — %s\n",__FILE__,__LINE__,hipGetErrorString(_e)); return false; }
        return true;
    };
    #define ALLOC_OR_FAIL(s, alloc_fn, ptr, n) do { if (!alloc_fn(ptr, n)) { zaya_destroy(s); return nullptr; } } while(0)
    ALLOC_OR_FAIL(s, alloc_f16, s->d_hs, eng.h);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_ao, eng.h);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_tmp, std::max(eng.h, 2*eng.n_ff));
    ALLOC_OR_FAIL(s, alloc_f16, s->d_fnw, eng.h);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_lm_out, 4096);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_lm_vocab, eng.vocab);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_argmax_idx, 1);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_argmax_val, 1);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_sorted_ids, 8);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_expert_counts, 17);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_expert_offsets, 17);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_embed, eng.vocab * eng.h);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_conv, eng.n_layers * 2 * eng.qkv);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_phs, eng.n_layers * eng.h);
    s->max_seq = 4096;
    ALLOC_OR_FAIL(s, alloc_f16, s->d_kcache, eng.n_layers * s->max_seq * eng.nkv * eng.hd);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_vcache, eng.n_layers * s->max_seq * eng.nkv * eng.hd);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_vrec, eng.n_layers * (eng.kd / 2));
    ALLOC_OR_FAIL(s, alloc_f16, s->d_qout, eng.qd);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_kout, eng.kd);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_vout, eng.kd);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_skip_flag, 1);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_prev_rs, eng.n_layers * eng.rtr_h);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_expert_idx, 1);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_expert_wt, 1);
    #undef ALLOC_OR_FAIL
    
    upf16(s->embed,s->d_embed,eng.vocab*eng.h,s->st);
    std::vector<__half>hf(eng.h);for(int i=0;i<eng.h;i++)hf[i]=__float2half(fnorm[i]);
    HIP_OK_R(hipMemcpy(s->d_fnw,hf.data(),eng.h*2,hipMemcpyHostToDevice), nullptr);
    
    auto A=[&](auto&p,int n)->bool{hipError_t _e=hipMalloc(&p,n*2);if(_e!=hipSuccess){fprintf(stderr,"HIP Error %d at %s:%d — %s\n",_e,__FILE__,__LINE__,hipGetErrorString(_e));return false;}return true;};
    auto B=[&](auto&p,int n)->bool{hipError_t _e=hipMalloc(&p,n*4);if(_e!=hipSuccess){fprintf(stderr,"HIP Error %d at %s:%d — %s\n",_e,__FILE__,__LINE__,hipGetErrorString(_e));return false;}return true;};
    
    s->lw.resize(eng.n_layers);
    s->has_eda.resize(eng.n_layers);
    s->eda_scale.resize(eng.n_layers);
    for(int il=0;il<eng.n_layers;il++){
        auto& l=s->lw[il];
        if(!A(l.nw,eng.h)){zaya_destroy(s);return nullptr;}upf16(W("model_layers_"+L(il)+"_input_layernorm_weight.bin"),l.nw,eng.h,s->st);
        if(!A(l.wq,eng.qd*eng.h)){zaya_destroy(s);return nullptr;}upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_q_proj_weight.bin"),l.wq,eng.qd*eng.h,s->st);
        if(!A(l.wk,eng.kd*eng.h)){zaya_destroy(s);return nullptr;}upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_k_proj_weight.bin"),l.wk,eng.kd*eng.h,s->st);
        if(!A(l.wv1,(eng.kd/2)*eng.h)){zaya_destroy(s);return nullptr;}upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_v_proj_current_weight.bin"),l.wv1,(eng.kd/2)*eng.h,s->st);
        if(!A(l.wv2,(eng.kd/2)*eng.h)){zaya_destroy(s);return nullptr;}upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_v_proj_delayed_weight.bin"),l.wv2,(eng.kd/2)*eng.h,s->st);
        if(!A(l.wo,eng.h*eng.qd)){zaya_destroy(s);return nullptr;}upf16(W("model_layers_"+L(il)+"_self_attn_o_proj_weight.bin"),l.wo,eng.h*eng.qd,s->st);
        if(!B(l.cdw,eng.qkv*2)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_depthwise_weight.bin"),l.cdw,eng.qkv*2,s->st);
        if(!B(l.cdb,eng.qkv)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_depthwise_bias.bin"),l.cdb,eng.qkv,s->st);
        if(!B(l.cgw,eng.qkv*128*2)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_grouped_weight.bin"),l.cgw,eng.qkv*128*2,s->st);
        if(!B(l.cgb,eng.qkv)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_grouped_bias.bin"),l.cgb,eng.qkv,s->st);
        if(!B(l.ks,eng.nkv)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_self_attn_qk_norm_temp.bin"),l.ks,eng.nkv,s->st);
        if(!B(l.pahss,eng.h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_hidden_states_scale.bin"),l.pahss,eng.h,s->st);
        if(!B(l.pahsb,eng.h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_hidden_states_bias.bin"),l.pahsb,eng.h,s->st);
        if(!B(l.parss,eng.h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_residual_scale.bin"),l.parss,eng.h,s->st);
        if(!B(l.parsb,eng.h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_residual_bias.bin"),l.parsb,eng.h,s->st);
        if(!A(l.pan,eng.h)){zaya_destroy(s);return nullptr;}upf16(W("model_layers_"+L(il)+"_post_attention_layernorm_weight.bin"),l.pan,eng.h,s->st);
        // gdw stored transposed [eng.h, eng.rtr_h] for cache-friendly GPU access
        // (each thread reads stride-1 floats in the inner loop instead of stride eng.h).
        // Upstream PyTorch saves it as [eng.rtr_h, eng.h] — we transpose on load.
        if(!B(l.gdw,eng.h*eng.rtr_h)){zaya_destroy(s);return nullptr;}
        {
            auto raw=W("model_layers_"+L(il)+"_mlp_gate_down_proj_weight.bin");
            std::vector<float> tr(eng.h*eng.rtr_h);
            for(int i=0;i<eng.rtr_h;i++)for(int j=0;j<eng.h;j++)tr[j*eng.rtr_h+i]=raw[i*eng.h+j];
            upf32(tr,l.gdw,eng.h*eng.rtr_h,s->st);
        }
        if(!B(l.gdb,eng.rtr_h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_mlp_gate_down_proj_bias.bin"),l.gdb,eng.rtr_h,s->st);
        if(!B(l.rfn,eng.rtr_h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_norm_weight.bin"),l.rfn,eng.rtr_h,s->st);
        if(!B(l.rf1,eng.rtr_h*eng.rtr_h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc1_weight.bin"),l.rf1,eng.rtr_h*eng.rtr_h,s->st);
        if(!B(l.rf1b,eng.rtr_h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc1_bias.bin"),l.rf1b,eng.rtr_h,s->st);
        if(!B(l.rf2,eng.rtr_h*eng.rtr_h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc2_weight.bin"),l.rf2,eng.rtr_h*eng.rtr_h,s->st);
        if(!B(l.rf2b,eng.rtr_h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc2_bias.bin"),l.rf2b,eng.rtr_h,s->st);
        if(!B(l.rout,eng.n_exp_t*eng.rtr_h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_out_proj_weight.bin"),l.rout,eng.n_exp_t*eng.rtr_h,s->st);
        if(!B(l.bb,eng.n_exp_t)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_mlp_gate_balancing_biases.bin"),l.bb,eng.n_exp_t,s->st);
        auto sz_gu=eng.n_exp*2*eng.n_ff*eng.h;auto sz_dn=eng.n_exp*eng.h*eng.n_ff;
        auto e1=hipMalloc(&l.gu,sz_gu*2);auto e2=hipMalloc(&l.dn,sz_dn*2);
        if(e1!=hipSuccess||e2!=hipSuccess){l.gu=nullptr;l.dn=nullptr;}else{
            upf16(W("model_layers_"+L(il)+"_mlp_experts_gate_up_proj.bin"),l.gu,sz_gu,s->st);
            upf16(W("model_layers_"+L(il)+"_mlp_experts_down_proj.bin"),l.dn,sz_dn,s->st);
        }
        if(!B(l.pmhss,eng.h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_hidden_states_scale.bin"),l.pmhss,eng.h,s->st);
        if(!B(l.pmhsb,eng.h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_hidden_states_bias.bin"),l.pmhsb,eng.h,s->st);
        if(!B(l.pmrss,eng.h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_residual_scale.bin"),l.pmrss,eng.h,s->st);
        if(!B(l.pmrsb,eng.h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_residual_bias.bin"),l.pmrsb,eng.h,s->st);
        
        std::string ep=g_weights_dir+"model_layers_"+L(il)+"_mlp_gate_router_states_scale.bin";
        std::ifstream ff(ep,std::ios::binary|std::ios::ate);
        if(ff){
            size_t fsize=ff.tellg();ff.seekg(0);
            if(fsize>=(size_t)eng.rtr_h*4){
                std::vector<float> buf(eng.rtr_h);
                ff.read((char*)buf.data(),eng.rtr_h*4);
                float sum=0; for(auto v:buf) sum+=v;
                s->eda_scale[il]=sum/(float)eng.rtr_h;
            }else{
                ff.read((char*)&s->eda_scale[il],4);
            }
            s->has_eda[il]=true;
        }else{
            s->eda_scale[il]=0;
            s->has_eda[il]=false;
        }
    }
    HIP_OK_R(hipStreamSynchronize(s->st), nullptr);
    return s;
}

// ── Forward: token in, logits out ──
void zaya_forward(ZayaState* s, int token_id, float* logits_out) {
    int g1 = (eng.h+BLK-1)/BLK;
    std::vector<__half> hh(eng.h);
    for(int i=0;i<eng.h;i++){float raw=s->embed[token_id*(size_t)eng.h+i];hh[i]=__float2half((raw+s->ibias[i])*s->iscale[i]);}
    HIP_OK_V(hipMemcpyAsync(s->d_hs,hh.data(),eng.h*2,hipMemcpyHostToDevice,s->st));

    for(int il=0;il<eng.n_layers;il++){
        auto& l=s->lw[il];
        // ── CCA attention: q/k/v proj → cca_prep → KV-cache stash → flash-decode → o_proj ──
        rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,l.nw,eng.h);
        HIP_CHECK(hipGetLastError());
        moe_tiled_gemv<<<eng.qd/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_hs,l.wq,eng.qd,eng.h);                           // q_raw
        HIP_CHECK(hipGetLastError());
        moe_tiled_gemv<<<eng.kd/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd,s->d_hs,l.wk,eng.kd,eng.h);                     // k_raw
        HIP_CHECK(hipGetLastError());
        moe_tiled_gemv<<<eng.kd/2/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd+eng.kd,s->d_hs,l.wv1,eng.kd/2,eng.h);        // v_cur
        HIP_CHECK(hipGetLastError());
        moe_tiled_gemv<<<eng.kd/2/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd+eng.kd+eng.kd/2,s->d_hs,l.wv2,eng.kd/2,eng.h); // v_del
        HIP_CHECK(hipGetLastError());
        cca_prep_kernel<<<1,256,0,s->st>>>(s->d_tmp,s->d_tmp+eng.qd,s->d_tmp+eng.qd+eng.kd,s->d_tmp+eng.qd+eng.kd+eng.kd/2,
            s->d_conv+(size_t)il*2*eng.qkv, s->d_vrec+(size_t)il*(eng.kd/2),
            l.cdw,l.cdb,l.cgw,l.cgb,l.ks,
            s->d_qout,s->d_kout,s->d_vout, s->pos);
        HIP_CHECK(hipGetLastError());
        copy_k<<<(eng.kd+BLK-1)/BLK,BLK,0,s->st>>>(s->d_kcache+(size_t)il*s->max_seq*eng.nkv*eng.hd+(size_t)s->pos*eng.nkv*eng.hd, s->d_kout, eng.kd);
        HIP_CHECK(hipGetLastError());
        copy_k<<<(eng.kd+BLK-1)/BLK,BLK,0,s->st>>>(s->d_vcache+(size_t)il*s->max_seq*eng.nkv*eng.hd+(size_t)s->pos*eng.nkv*eng.hd, s->d_vout, eng.kd);
        HIP_CHECK(hipGetLastError());
        rcpp_kv_cache_attn_decode_fd(s->d_qout,
            s->d_kcache+(size_t)il*s->max_seq*eng.nkv*eng.hd,
            s->d_vcache+(size_t)il*s->max_seq*eng.nkv*eng.hd,
            s->d_ao, eng.nq, eng.nkv, eng.hd, s->pos+1, 1.0f/sqrtf((float)eng.hd), (void*)s->st);
        moe_tiled_gemv<<<eng.h/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_ao,s->d_ao,l.wo,eng.h,eng.qd);                             // o_proj
        HIP_CHECK(hipGetLastError());
        residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_ao,s->d_hs,l.pahss,l.pahsb,l.parss,l.parsb,eng.h);
        HIP_CHECK(hipGetLastError());
        copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_ao,eng.h);
        HIP_CHECK(hipGetLastError());
        rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,l.pan,eng.h);
        HIP_CHECK(hipGetLastError());
        if(l.gu&&l.dn){
            eda_router_gpu_kernel<<<1,eng.rtr_h,0,s->st>>>(s->d_hs,s->d_prev_rs+(size_t)il*eng.rtr_h,s->has_eda[il]?1:0,s->eda_scale[il],l.gdw,l.gdb,l.rfn,l.rf1,l.rf1b,l.rf2,l.rf2b,l.rout,l.bb,s->d_prev_rs+(size_t)il*eng.rtr_h,s->d_expert_idx,s->d_expert_wt);
            HIP_CHECK(hipGetLastError());
            encode_expert_cache_kernel<<<1,32,0,s->st>>>(s->d_prev_rs+(size_t)il*eng.rtr_h,s->d_expert_idx,eng.rtr_h);
            HIP_CHECK(hipGetLastError());
            fixup_skip_expert_kernel<<<1,256,0,s->st>>>(s->d_expert_idx,s->d_tmp,s->d_skip_flag,eng.n_exp,eng.n_exp_t,eng.h);
            HIP_CHECK(hipGetLastError());
            HIP_OK_V(hipStreamSynchronize(s->st));
            int was_skip; HIP_OK_V(hipMemcpy(&was_skip,s->d_skip_flag,4,hipMemcpyDeviceToHost));
            if(!was_skip){
                const int gb=(2*eng.n_ff+WMMA_M-1)/WMMA_M;
                const int db=(eng.h+WMMA_M-1)/WMMA_M;
                const int sb=(eng.n_ff+BLK-1)/BLK;
                wmma_gateup_kernel<<<gb,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_hs,l.gu,s->d_expert_idx);
                HIP_CHECK(hipGetLastError());
                silu_mul_k<<<sb,BLK,0,s->st>>>(s->d_ao,s->d_tmp,s->d_tmp+eng.n_ff,eng.n_ff);
                HIP_CHECK(hipGetLastError());
                wmma_down_kernel<<<db,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_ao,l.dn,s->d_expert_idx);
            HIP_CHECK(hipGetLastError());
            }
            residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,l.pmhss,l.pmhsb,l.pmrss,l.pmrsb,eng.h);
            HIP_CHECK(hipGetLastError());
            copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_tmp,eng.h);
        HIP_CHECK(hipGetLastError());
        }else{
            copy_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,eng.h);
        HIP_CHECK(hipGetLastError());
        }
    }
    rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,s->d_fnw,eng.h);
    HIP_CHECK(hipGetLastError());

    // lm_head — tiled GEMV in a single launch; buffer allocated in zaya_init (fixes #59)
    // No sync needed before the lm_head: both the RMSNorm and the lm_head GEMV are on
    // the same stream, so the GEMV waits for the RMSNorm automatically (fixes perf).
    moe_tiled_gemv<<<(eng.vocab+WMMA_M-1)/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_lm_vocab,s->d_hs,s->d_embed,eng.vocab,eng.h);
    HIP_CHECK(hipGetLastError());
    HIP_OK_V(hipStreamSynchronize(s->st));
    std::vector<__half> lh(eng.vocab);
    HIP_OK_V(hipMemcpy(lh.data(),s->d_lm_vocab,(size_t)eng.vocab*2,hipMemcpyDeviceToHost));
    for(int v=0;v<eng.vocab;v++)logits_out[v]=__half2float(lh[v]);
    if(s->pos < s->max_seq-1) s->pos++;
}

// ── Forward greedy: same as forward but only returns argmax (much faster) ──
int zaya_forward_greedy(ZayaState* s, int token_id) {
    int g1 = (eng.h+BLK-1)/BLK;
    std::vector<__half> hh(eng.h);
    for(int i=0;i<eng.h;i++){float raw=s->embed[token_id*(size_t)eng.h+i];hh[i]=__float2half((raw+s->ibias[i])*s->iscale[i]);}
    HIP_OK_R(hipMemcpyAsync(s->d_hs,hh.data(),eng.h*2,hipMemcpyHostToDevice,s->st), -1);

    for(int il=0;il<eng.n_layers;il++){
        auto& l=s->lw[il];
        // ── CCA attention: q/k/v proj → cca_prep → KV-cache stash → flash-decode → o_proj ──
        rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,l.nw,eng.h);
        HIP_CHECK(hipGetLastError());
        moe_tiled_gemv<<<eng.qd/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_hs,l.wq,eng.qd,eng.h);                           // q_raw
        HIP_CHECK(hipGetLastError());
        moe_tiled_gemv<<<eng.kd/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd,s->d_hs,l.wk,eng.kd,eng.h);                     // k_raw
        HIP_CHECK(hipGetLastError());
        moe_tiled_gemv<<<eng.kd/2/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd+eng.kd,s->d_hs,l.wv1,eng.kd/2,eng.h);        // v_cur
        HIP_CHECK(hipGetLastError());
        moe_tiled_gemv<<<eng.kd/2/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd+eng.kd+eng.kd/2,s->d_hs,l.wv2,eng.kd/2,eng.h); // v_del
        HIP_CHECK(hipGetLastError());
        cca_prep_kernel<<<1,256,0,s->st>>>(s->d_tmp,s->d_tmp+eng.qd,s->d_tmp+eng.qd+eng.kd,s->d_tmp+eng.qd+eng.kd+eng.kd/2,
            s->d_conv+(size_t)il*2*eng.qkv, s->d_vrec+(size_t)il*(eng.kd/2),
            l.cdw,l.cdb,l.cgw,l.cgb,l.ks,
            s->d_qout,s->d_kout,s->d_vout, s->pos);
        HIP_CHECK(hipGetLastError());
        copy_k<<<(eng.kd+BLK-1)/BLK,BLK,0,s->st>>>(s->d_kcache+(size_t)il*s->max_seq*eng.nkv*eng.hd+(size_t)s->pos*eng.nkv*eng.hd, s->d_kout, eng.kd);
        HIP_CHECK(hipGetLastError());
        copy_k<<<(eng.kd+BLK-1)/BLK,BLK,0,s->st>>>(s->d_vcache+(size_t)il*s->max_seq*eng.nkv*eng.hd+(size_t)s->pos*eng.nkv*eng.hd, s->d_vout, eng.kd);
        HIP_CHECK(hipGetLastError());
        rcpp_kv_cache_attn_decode_fd(s->d_qout,
            s->d_kcache+(size_t)il*s->max_seq*eng.nkv*eng.hd,
            s->d_vcache+(size_t)il*s->max_seq*eng.nkv*eng.hd,
            s->d_ao, eng.nq, eng.nkv, eng.hd, s->pos+1, 1.0f/sqrtf((float)eng.hd), (void*)s->st);
        moe_tiled_gemv<<<eng.h/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_ao,s->d_ao,l.wo,eng.h,eng.qd);                             // o_proj
        HIP_CHECK(hipGetLastError());
        residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_ao,s->d_hs,l.pahss,l.pahsb,l.parss,l.parsb,eng.h);
        HIP_CHECK(hipGetLastError());
        copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_ao,eng.h);
        HIP_CHECK(hipGetLastError());
        rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,l.pan,eng.h);
        HIP_CHECK(hipGetLastError());
        if(l.gu&&l.dn){
            eda_router_gpu_kernel<<<1,eng.rtr_h,0,s->st>>>(s->d_hs,s->d_prev_rs+(size_t)il*eng.rtr_h,s->has_eda[il]?1:0,s->eda_scale[il],l.gdw,l.gdb,l.rfn,l.rf1,l.rf1b,l.rf2,l.rf2b,l.rout,l.bb,s->d_prev_rs+(size_t)il*eng.rtr_h,s->d_expert_idx,s->d_expert_wt);
            HIP_CHECK(hipGetLastError());
            encode_expert_cache_kernel<<<1,32,0,s->st>>>(s->d_prev_rs+(size_t)il*eng.rtr_h,s->d_expert_idx,eng.rtr_h);
            HIP_CHECK(hipGetLastError());
            fixup_skip_expert_kernel<<<1,256,0,s->st>>>(s->d_expert_idx,s->d_tmp,s->d_skip_flag,eng.n_exp,eng.n_exp_t,eng.h);
            HIP_CHECK(hipGetLastError());
            HIP_OK_R(hipStreamSynchronize(s->st), -1);
            int was_skip; HIP_OK_R(hipMemcpy(&was_skip,s->d_skip_flag,4,hipMemcpyDeviceToHost), -1);
            if(!was_skip){
                const int gb=(2*eng.n_ff+WMMA_M-1)/WMMA_M;
                const int db=(eng.h+WMMA_M-1)/WMMA_M;
                const int sb=(eng.n_ff+BLK-1)/BLK;
                wmma_gateup_kernel<<<gb,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_hs,l.gu,s->d_expert_idx);
                HIP_CHECK(hipGetLastError());
                silu_mul_k<<<sb,BLK,0,s->st>>>(s->d_ao,s->d_tmp,s->d_tmp+eng.n_ff,eng.n_ff);
                HIP_CHECK(hipGetLastError());
                wmma_down_kernel<<<db,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_ao,l.dn,s->d_expert_idx);
            HIP_CHECK(hipGetLastError());
            }
            residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,l.pmhss,l.pmhsb,l.pmrss,l.pmrsb,eng.h);
            HIP_CHECK(hipGetLastError());
            copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_tmp,eng.h);
        HIP_CHECK(hipGetLastError());
        }else{copy_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,eng.h);}
    HIP_CHECK(hipGetLastError());
    }
    rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,s->d_fnw,eng.h);
    HIP_CHECK(hipGetLastError());

    // lm_head + GPU argmax (no full logit copy); buffers allocated in zaya_init (fixes #59)
    moe_tiled_gemv<<<(eng.vocab+WMMA_M-1)/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_lm_vocab,s->d_hs,s->d_embed,eng.vocab,eng.h);
    HIP_CHECK(hipGetLastError());
    argmax_kernel<<<1,256,0,s->st>>>(s->d_lm_vocab,eng.vocab,s->d_argmax_idx,s->d_argmax_val);
    HIP_CHECK(hipGetLastError());
    HIP_OK_R(hipStreamSynchronize(s->st), -1);
    int best;
    HIP_OK_R(hipMemcpy(&best,s->d_argmax_idx,4,hipMemcpyDeviceToHost), -1);
    if(s->pos < s->max_seq-1) s->pos++;
    return best;
}

// ── Forward batch: process B tokens through all layers ──
// Uses batched router + batch-union MoE for expert dedup.
// B <= 8 recommended (constrained by shared memory in union kernel).
void zaya_forward_batch(ZayaState* s, const int* token_ids, float* logits_out, int B) {
    int g1 = (eng.h+BLK-1)/BLK;
    if (B > 8) {
        fprintf(stderr, "zaya_forward_batch: B=%d > 8, truncating to 8 (tokens %d-%d will NOT be processed)\n", B, 8, B-1);
        B = 8;
    }

    // ── Embedding lookup for all B tokens ──
    std::vector<__half> hh(B * eng.h);
    for (int b = 0; b < B; b++) {
        int tid = token_ids[b];
        for (int i = 0; i < eng.h; i++) {
            float raw = s->embed[tid * (size_t)eng.h + i];
            hh[b * (size_t)eng.h + i] = __float2half((raw + s->ibias[i]) * s->iscale[i]);
        }
    }
    HIP_OK_V(hipMemcpyAsync(s->d_hs, hh.data(), B * eng.h * 2, hipMemcpyHostToDevice, s->st));

    for (int il = 0; il < eng.n_layers; il++) {
        auto& l = s->lw[il];

        // CCA attention: per-token rmsnorm + Q/K/V proj → V interleave → GQA broadcast → o_proj
        for (int b = 0; b < B; b++) {
            __half* hs_b = s->d_hs + (size_t)b * eng.h;
            __half* ao_b = s->d_ao + (size_t)b * eng.h;
            rmsnorm_k<<<1, BLK, 0, s->st>>>(hs_b, l.nw, eng.h);
            HIP_CHECK(hipGetLastError());
            moe_tiled_gemv<<<eng.qd/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp, hs_b, l.wq, eng.qd, eng.h);
            HIP_CHECK(hipGetLastError());
            moe_tiled_gemv<<<eng.kd/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd, hs_b, l.wk, eng.kd, eng.h);
            HIP_CHECK(hipGetLastError());
            moe_tiled_gemv<<<eng.kd/2/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd+eng.kd, hs_b, l.wv1, eng.kd/2, eng.h);
            HIP_CHECK(hipGetLastError());
            moe_tiled_gemv<<<eng.kd/2/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd+eng.kd+eng.kd/2, hs_b, l.wv2, eng.kd/2, eng.h);
            HIP_CHECK(hipGetLastError());
            v_interleave_kernel<<<(eng.kd/2+BLK-1)/BLK,BLK,0,s->st>>>(s->d_tmp+eng.qd, s->d_tmp+eng.qd+eng.kd, s->d_tmp+eng.qd+eng.kd+eng.kd/2, eng.kd/2);
            HIP_CHECK(hipGetLastError());
            gqa_broadcast_k<<<(eng.qd+BLK-1)/BLK,BLK,0,s->st>>>(s->d_tmp+eng.qd, s->d_tmp, eng.nq, eng.nkv, eng.hd);
            HIP_CHECK(hipGetLastError());
            moe_tiled_gemv<<<eng.h/WMMA_M,WMMA_THREADS,0,s->st>>>(ao_b, s->d_tmp, l.wo, eng.h, eng.qd);
        HIP_CHECK(hipGetLastError());
        }

        // Post-attention residual + RMSNorm (per token)
        for (int b = 0; b < B; b++) {
            __half* hs_b  = s->d_hs + (size_t)b * eng.h;
            __half* ao_b  = s->d_ao + (size_t)b * eng.h;
            residual_scale_k<<<g1, BLK, 0, s->st>>>(ao_b, hs_b,
                l.pahss, l.pahsb, l.parss, l.parsb, eng.h);
            HIP_CHECK(hipGetLastError());
            copy_k<<<g1, BLK, 0, s->st>>>(hs_b, ao_b, eng.h);
            HIP_CHECK(hipGetLastError());
            rmsnorm_k<<<1, BLK, 0, s->st>>>(hs_b, l.pan, eng.h);
        HIP_CHECK(hipGetLastError());
        }

        // Batched router + batch-union MoE
        // For B >= 4, use sorted dispatch to group tokens by expert
        // (loads each expert's weights ONCE, avoids L2 thrashing).
        // For B < 4, the fused per-token kernel is simpler and fast enough.
        if (l.gu && l.dn) {
            if (B >= 4) {
                // Phase 1: Route all B tokens (GPU-resident)
                batched_moe_router_kernel<<<B, 256, 0, s->st>>>(
                    s->d_hs, s->d_prev_rs + (size_t)il * eng.rtr_h,
                    s->has_eda[il] ? 1 : 0, s->eda_scale[il],
                    l.gdw, l.gdb, l.rfn, l.rf1, l.rf1b,
                    l.rf2, l.rf2b, l.rout, l.bb,
                    s->d_prev_rs + (size_t)il * eng.rtr_h,
                    s->d_expert_idx, s->d_expert_wt,
                    B);
                HIP_CHECK(hipGetLastError());

                // Phase 2: Sort token IDs by expert (histogram + prefix sum + scatter);
                // buffers allocated in zaya_init (fixes #63).
                moe_sort_histogram_kernel<<<1, 32, 0, s->st>>>(
                    s->d_expert_idx, s->d_expert_counts, s->d_expert_offsets,
                    s->d_sorted_ids, B);
                HIP_CHECK(hipGetLastError());

                // Phase 3: Expert FFN (sorted, one block per expert with count>0)
                moe_sorted_expert_kernel<<<eng.n_exp, 256, 0, s->st>>>(
                    s->d_hs, s->d_sorted_ids, s->d_expert_counts, s->d_expert_offsets,
                    l.gu, l.dn, s->d_tmp, B);
                HIP_CHECK(hipGetLastError());

                // Phase 4: Handle MOD skip tokens (expert_idx == eng.n_exp = 16)
                // These skip the expert FFN entirely (out = hs, identity).
                if (s->d_expert_counts) {  // always true, keeps compiler happy
                    moe_modskip_passthrough_kernel<<<(B + 255) / 256, 256, 0, s->st>>>(
                        s->d_tmp, s->d_hs, s->d_expert_idx, eng.n_exp, B);
                HIP_CHECK(hipGetLastError());
                }
            } else {
                batched_moe_fused_kernel<<<B, 256, 0, s->st>>>(
                    s->d_hs, s->d_prev_rs + (size_t)il * eng.rtr_h,
                    s->has_eda[il] ? 1 : 0, s->eda_scale[il],
                    l.gdw, l.gdb, l.rfn, l.rf1, l.rf1b,
                    l.rf2, l.rf2b, l.rout, l.bb,
                    l.gu, l.dn,
                    s->d_prev_rs + (size_t)il * eng.rtr_h,
                    s->d_tmp, s->d_expert_idx, s->d_expert_wt,
                    B);
            HIP_CHECK(hipGetLastError());
            }

            // Post-MLP residual scale (per token)
            for (int b = 0; b < B; b++) {
                __half* hs_b  = s->d_hs + (size_t)b * eng.h;
                __half* tmp_b = s->d_tmp + (size_t)b * eng.h;
                residual_scale_k<<<g1, BLK, 0, s->st>>>(tmp_b, hs_b,
                    l.pmhss, l.pmhsb, l.pmrss, l.pmrsb, eng.h);
                HIP_CHECK(hipGetLastError());
                copy_k<<<g1, BLK, 0, s->st>>>(hs_b, tmp_b, eng.h);
            HIP_CHECK(hipGetLastError());
            }
        }
    }

    // Final RMSNorm (per token)
    for (int b = 0; b < B; b++) {
        __half* hs_b = s->d_hs + (size_t)b * eng.h;
        rmsnorm_k<<<1, BLK, 0, s->st>>>(hs_b, s->d_fnw, eng.h);
    HIP_CHECK(hipGetLastError());
    }

    // lm_head — tiled GEMV for each token; buffer allocated in zaya_init (fixes #63).
    // No sync needed before lm_head: same-stream ordering guarantees RMSNorm completes
    // before the lm_head GEMV launches.
    {
        const size_t max_need = (size_t)8 * eng.vocab * 2;  // B <= 8, allocated in zaya_init
        #if __has_include(<rocwmma/rocwmma.hpp>)
        if (B >= 2) {
            const int grid_x = (eng.vocab + WMMA_M - 1) / WMMA_M;
            const int grid_y = (B + WMMA_N - 1) / WMMA_N;
            wmma_batched_gemv<<<dim3(grid_x, grid_y, 1), 32, 0, s->st>>>(
                s->d_lm_vocab, s->d_hs, s->d_embed, eng.vocab, eng.h, B);
        HIP_CHECK(hipGetLastError());
        } else {
            moe_tiled_gemv<<<(eng.vocab + WMMA_M - 1) / WMMA_M, WMMA_THREADS, 0, s->st>>>(
                s->d_lm_vocab, s->d_hs, s->d_embed, eng.vocab, eng.h);
        HIP_CHECK(hipGetLastError());
        }
        #else
        for (int b = 0; b < B; b++) {
            __half* hs_b = s->d_hs + (size_t)b * eng.h;
            moe_tiled_gemv<<<(eng.vocab + WMMA_M - 1) / WMMA_M, WMMA_THREADS, 0, s->st>>>(
                s->d_lm_vocab + (size_t)b * eng.vocab, hs_b, s->d_embed, eng.vocab, eng.h);
        HIP_CHECK(hipGetLastError());
        }
        #endif
    }
    HIP_OK_V(hipStreamSynchronize(s->st));

    // Copy logits for all B tokens
    std::vector<__half> lh(B * eng.vocab);
    HIP_OK_V(hipMemcpy(lh.data(), s->d_lm_vocab, (size_t)B * eng.vocab * 2, hipMemcpyDeviceToHost));
    for (int b = 0; b < B; b++)
        for (int v = 0; v < eng.vocab; v++)
            logits_out[b * (size_t)eng.vocab + v] = __half2float(lh[b * (size_t)eng.vocab + v]);
}

// ═══════════════════════════════════════════════════════════════════════
// ── LoRA adapter merge ──
// Reads a .lora file (from merge_lora.py) and merges B*A*scale deltas
// into the GPU-resident weight matrices. Called after zaya_init().
//
// .lora binary format:
//   magic:   b'LORA' (4 bytes)
//   layers:  uint32
//   scale:   float32
//   For each layer:
//     num_mod: uint32
//     For each module:
//       mod_id: uint32 (0=q,1=k,2=v,3=o,4=gate,5=up,6=down)
//       rank:   uint32
//       in_dim: uint32 (input dimension of A = hidden_size)
//       out_dim:uint32 (output dimension of B = projection_size)
//       A[rank][in_dim]:  float32
//       B[out_dim][rank]: float32
// ═══════════════════════════════════════════════════════════════════════

// Per-module LoRA data
struct LoraModule {
    int mod_id;        // 0=q, 1=k, 2=v, 3=o, 4=gate, 5=up, 6=down
    int rank;
    int in_dim;
    int out_dim;
    std::vector<float> A;  // [rank * in_dim]
    std::vector<float> B;  // [out_dim * rank]
};

// Per-layer LoRA data
struct LoraLayer {
    std::vector<LoraModule> modules;
};

// Read a .lora file; returns empty vector on failure
static std::vector<LoraLayer> zaya_read_lora_file(const char* path, float& out_scale) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "zaya_apply_lora: cannot open %s\n", path);
        return {};
    }
    
    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "LORA") {
        fprintf(stderr, "zaya_apply_lora: bad magic in %s\n", path);
        return {};
    }
    
    uint32_t num_layers;
    float scale;
    f.read((char*)&num_layers, 4);
    f.read((char*)&scale, 4);
    out_scale = scale;
    
    std::vector<LoraLayer> layers(num_layers);
    for (uint32_t l = 0; l < num_layers; l++) {
        uint32_t num_mod;
        f.read((char*)&num_mod, 4);
        layers[l].modules.resize(num_mod);
        for (uint32_t m = 0; m < num_mod; m++) {
            LoraModule& mod = layers[l].modules[m];
            f.read((char*)&mod.mod_id, 4);
            f.read((char*)&mod.rank, 4);
            f.read((char*)&mod.in_dim, 4);
            f.read((char*)&mod.out_dim, 4);
            mod.A.resize((size_t)mod.rank * mod.in_dim);
            mod.B.resize((size_t)mod.out_dim * mod.rank);
            f.read((char*)mod.A.data(), mod.A.size() * 4);
            f.read((char*)mod.B.data(), mod.B.size() * 4);
        }
    }
    
    fprintf(stderr, "zaya_apply_lora: read %u layers from %s (scale=%.4f)\n",
            num_layers, path, scale);
    return layers;
}

// Compute delta = B * A * scale for a LoRA module
// Result: [out_dim * in_dim] row-major
static std::vector<float> compute_lora_delta(const LoraModule& mod, float scale) {
    std::vector<float> delta((size_t)mod.out_dim * mod.in_dim, 0.0f);
    // B[out_dim, rank] * A[rank, in_dim] → delta[out_dim, in_dim]
    for (int o = 0; o < mod.out_dim; o++) {
        for (int r = 0; r < mod.rank; r++) {
            float br = mod.B[(size_t)o * mod.rank + r] * scale;
            if (br == 0.0f) continue;
            const float* A_row = mod.A.data() + (size_t)r * mod.in_dim;
            float* delta_row = delta.data() + (size_t)o * mod.in_dim;
            for (int i = 0; i < mod.in_dim; i++) {
                delta_row[i] += br * A_row[i];
            }
        }
    }
    return delta;
}

// Apply LoRA deltas to GPU-resident layer weights
// Supported module -> weight mapping:
//   0 (q_proj)   → lw[].wq   FP16 [QD, H]
//   1 (k_proj)   → lw[].wk   FP16 [KD, H]
//   3 (o_proj)   → lw[].wo   FP16 [H, QD]
//   6 (down)     → lw[].gdw  FP32 [H, RTR_H] (transposed on GPU)
extern "C" int zaya_apply_lora(ZayaState* s, const char* lora_path) {
    if (!s || !lora_path) return -1;
    
    float scale;
    auto layers = zaya_read_lora_file(lora_path, scale);
    if (layers.empty()) {
        fprintf(stderr, "zaya_apply_lora: failed to load %s\n", lora_path);
        return -1;
    }
    
    int total_applied = 0;
    int n_layers = (int)std::min((size_t)eng.n_layers, layers.size());
    
    for (int il = 0; il < n_layers; il++) {
        auto& l = s->lw[il];
        for (auto& mod : layers[il].modules) {
            std::vector<float> delta = compute_lora_delta(mod, scale);
            
            if (mod.mod_id == 0) {  // q_proj → wq [QD, H]
                if ((size_t)mod.out_dim == eng.qd && (size_t)mod.in_dim == eng.h) {
                    // Download GPU weight, add delta, upload back
                    std::vector<__half> gpu_w(eng.qd * eng.h);
                    HIP_OK_R(hipMemcpy(gpu_w.data(), l.wq, eng.qd * eng.h * 2, hipMemcpyDeviceToHost), -1);
                    for (int i = 0; i < eng.qd * eng.h; i++) {
                        float v = __half2float(gpu_w[i]) + delta[i];
                        gpu_w[i] = __float2half(v);
                    }
                    HIP_OK_R(hipMemcpy(l.wq, gpu_w.data(), eng.qd * eng.h * 2, hipMemcpyHostToDevice), -1);
                    total_applied++;
                    fprintf(stderr, "  layer %d q_proj: merged LoRA delta [%dx%d]\n", il, eng.qd, eng.h);
                }
            } else if (mod.mod_id == 1) {  // k_proj → wk [KD, H]
                if ((size_t)mod.out_dim == eng.kd && (size_t)mod.in_dim == eng.h) {
                    std::vector<__half> gpu_w(eng.kd * eng.h);
                    HIP_OK_R(hipMemcpy(gpu_w.data(), l.wk, eng.kd * eng.h * 2, hipMemcpyDeviceToHost), -1);
                    for (int i = 0; i < eng.kd * eng.h; i++) {
                        float v = __half2float(gpu_w[i]) + delta[i];
                        gpu_w[i] = __float2half(v);
                    }
                    HIP_OK_R(hipMemcpy(l.wk, gpu_w.data(), eng.kd * eng.h * 2, hipMemcpyHostToDevice), -1);
                    total_applied++;
                    fprintf(stderr, "  layer %d k_proj: merged LoRA delta [%dx%d]\n", il, eng.kd, eng.h);
                }
            } else if (mod.mod_id == 3) {  // o_proj → wo [H, QD]
                if ((size_t)mod.out_dim == eng.h && (size_t)mod.in_dim == eng.qd) {
                    std::vector<__half> gpu_w(eng.h * eng.qd);
                    HIP_OK_R(hipMemcpy(gpu_w.data(), l.wo, eng.h * eng.qd * 2, hipMemcpyDeviceToHost), -1);
                    for (int i = 0; i < eng.h * eng.qd; i++) {
                        float v = __half2float(gpu_w[i]) + delta[i];
                        gpu_w[i] = __float2half(v);
                    }
                    HIP_OK_R(hipMemcpy(l.wo, gpu_w.data(), eng.h * eng.qd * 2, hipMemcpyHostToDevice), -1);
                    total_applied++;
                    fprintf(stderr, "  layer %d o_proj: merged LoRA delta [%dx%d]\n", il, eng.h, eng.qd);
                }
            } else if (mod.mod_id == 6) {  // down (gate_down_proj) → gdw GPU is [H, RTR_H] transposed
                // File stores raw gate_down_proj weight as [RTR_H, H]. LoRA delta is [RTR_H, H].
                // On GPU, gdw is transposed to [H, RTR_H]. We need to apply delta then retranspose.
                if ((size_t)mod.out_dim == eng.rtr_h && (size_t)mod.in_dim == eng.h) {
                    std::vector<float> gpu_w(eng.h * eng.rtr_h);
                    HIP_OK_R(hipMemcpy(gpu_w.data(), l.gdw, eng.h * eng.rtr_h * 4, hipMemcpyDeviceToHost), -1);
                    // delta is [RTR_H, H]; gpu_w is [H, RTR_H] (transposed)
                    // We add delta^T to gpu_w: gpu_w[j][i] += delta[i][j]
                    for (int i = 0; i < eng.rtr_h; i++) {
                        for (int j = 0; j < eng.h; j++) {
                            gpu_w[(size_t)j * eng.rtr_h + i] += delta[(size_t)i * eng.h + j];
                        }
                    }
                    HIP_OK_R(hipMemcpy(l.gdw, gpu_w.data(), eng.h * eng.rtr_h * 4, hipMemcpyHostToDevice), -1);
                    total_applied++;
                    fprintf(stderr, "  layer %d gate_down: merged LoRA delta [%dx%d]\n", il, eng.rtr_h, eng.h);
                }
            }
        }
    }
    
    HIP_OK_R(hipStreamSynchronize(s->st), -1);
    fprintf(stderr, "zaya_apply_lora: applied %d LoRA deltas\n", total_applied);
    return total_applied > 0 ? 0 : -1;
}

// ── Reset state (new sequence) ──
// ── Set sentinel for expert caching (no previous expert for any layer) ──
__global__ void init_expert_cache_sentinel(float* prev_rs, int n_layers, int rtr_h) {
    // Set prev_rs[layer * rtr_h + rtr_h - 1] = bit-cast eng.n_exp (sentinel)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_layers) return;
    ((int*)&prev_rs[i * rtr_h + rtr_h - 1])[0] = 16; // eng.n_exp = 16 (invalid expert, no previous)
}

void zaya_reset(ZayaState* s) {
    HIP_OK_V(hipMemsetAsync(s->d_conv,0,(size_t)eng.n_layers*2*eng.qkv*2,s->st));
    HIP_OK_V(hipMemsetAsync(s->d_phs,0,(size_t)eng.n_layers*eng.h*2,s->st));
    HIP_OK_V(hipMemsetAsync(s->d_prev_rs,0,(size_t)eng.n_layers*eng.rtr_h*4,s->st));
    HIP_OK_V(hipMemsetAsync(s->d_kcache,0,(size_t)eng.n_layers*s->max_seq*eng.nkv*eng.hd*2,s->st));
    HIP_OK_V(hipMemsetAsync(s->d_vcache,0,(size_t)eng.n_layers*s->max_seq*eng.nkv*eng.hd*2,s->st));
    HIP_OK_V(hipMemsetAsync(s->d_vrec,0,(size_t)eng.n_layers*(eng.kd/2)*2,s->st));
    s->pos=0;
    init_expert_cache_sentinel<<<1, 64, 0, s->st>>>(s->d_prev_rs, eng.n_layers, eng.rtr_h);
HIP_CHECK(hipGetLastError());
}

// ── Destroy ──
void zaya_destroy(ZayaState* s) {
    if (!s) return;
    auto safe = [](auto p) { if (p) (void)hipFree(p); };
    safe(s->d_hs); safe(s->d_ao); safe(s->d_tmp); safe(s->d_fnw);
    safe(s->d_lm_out); safe(s->d_embed); safe(s->d_conv); safe(s->d_phs);
    safe(s->d_prev_rs); safe(s->d_expert_idx); safe(s->d_expert_wt);
    safe(s->d_kcache); safe(s->d_vcache); safe(s->d_vrec);
    safe(s->d_qout); safe(s->d_kout); safe(s->d_vout); safe(s->d_skip_flag);
    for (int i = 0; i < eng.n_layers; i++) {
        auto& l = s->lw[i];
        safe(l.nw); safe(l.wq); safe(l.wk); safe(l.wv1); safe(l.wv2); safe(l.wo); safe(l.pan);
        safe(l.cdw); safe(l.cdb); safe(l.cgw); safe(l.cgb); safe(l.ks);
        safe(l.pahss); safe(l.pahsb); safe(l.parss); safe(l.parsb);
        safe(l.gdw); safe(l.gdb); safe(l.rfn); safe(l.rf1); safe(l.rf1b);
        safe(l.rf2); safe(l.rf2b); safe(l.rout); safe(l.bb);
        safe(l.gu); safe(l.dn);
        safe(l.pmhss); safe(l.pmhsb); safe(l.pmrss); safe(l.pmrsb);
    }
    if (s->st) {
        // Log rather than HIP_OK_V's early-return here — this is cleanup,
        // the remaining frees and delete s below must still run regardless.
        hipError_t _st = hipStreamDestroy(s->st);
        if (_st != hipSuccess)
            fprintf(stderr, "HIP Error %d at %s:%d — %s\n", _st, __FILE__, __LINE__, hipGetErrorString(_st));
    }
    safe(s->d_lm_vocab); safe(s->d_argmax_idx); safe(s->d_argmax_val);
    safe(s->d_sorted_ids); safe(s->d_expert_counts); safe(s->d_expert_offsets);
    delete s;
}

} // extern "C"
