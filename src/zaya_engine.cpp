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

#define HIP_OK(e) do{auto _s=(e);if(_s!=hipSuccess){fprintf(stderr,"HIP Error %d\n",_s);abort();}}while(0)


// ── Architecture (compile-time constants for kernels ──
static constexpr int H=2048,NQ=8,NKV=2,HD=128,QD=NQ*HD,KD=NKV*HD,QKV=QD+KD;
static constexpr int N_LAYERS=40,VOCAB=262272,N_EXP=16,N_EXP_T=17,N_FF=2048,RTR_H=256;
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
__global__ void residual_scale_k(__half*out,const __half*res,const float*hs_s,const float*hs_b,const float*res_s,const float*res_b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float o=(float)out[i]*hs_s[i]+hs_b[i];float r=(float)res[i]*res_s[i]+res_b[i];out[i]=__float2half(o+r);}

// ── Complex kernels from .hip files ──
#include "../kernels/zaya_cca_attn.hip"
#include "../kernels/zaya_gpu_router.hip"
#include "../kernels/zaya_router_moe.hip"
#include "../kernels/zaya_moe_tiled_gemv.hip"
#include "../kernels/zaya_moe_expert_ffn.hip"
#include "../kernels/zaya_moe_batch_union.hip"
#include "../kernels/argmax_kernel.hip"
#include "../kernels/zaya_cca_custom.hip"
#include "../kernels/v_interleave_kernel.hip"

// ── Persistent thread blocks for MoE expert FFN (single grid, all layers) ──
#include "../kernels/zaya_persistent_moe.hip"

// ── rocWMMA batched GEMV (requires rocWMMA header library) ──
// Guarded: the CMakeLists.txt sets ROCWMMA_FOUND and the include path.
// If rocWMMA is not available, this kernel is skipped and the engine
// falls back to the scalar-tiled batched GEMV.
#if __has_include(<rocwmma/rocwmma.hpp>)
#include "../kernels/zaya_moe_wmma_batched.hip"
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
static std::string g_weights_dir = "/tmp/zaya_weights/";
#define W(N) load_bin(g_weights_dir+N)
static void upf16(const std::vector<float>& s,__half*d,int n,hipStream_t h=0){
    std::vector<__half>b(n);for(int i=0;i<n;i++)b[i]=__float2half(s[i]);
    hipMemcpyAsync(d,b.data(),n*2,hipMemcpyHostToDevice,h);
}
static void upf32(const std::vector<float>& s,float*d,int n,hipStream_t h=0){
    hipMemcpyAsync(d,s.data(),n*4,hipMemcpyHostToDevice,h);
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
    if (weights_dir) g_weights_dir = weights_dir;
    ZayaState* s = new ZayaState();
    HIP_OK(hipStreamCreate(&s->st));
    
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

    HIP_OK(hipMalloc(&s->d_hs,eng.h*2)); HIP_OK(hipMalloc(&s->d_ao,eng.h*2));
    HIP_OK(hipMalloc(&s->d_tmp,eng.h*2)); HIP_OK(hipMalloc(&s->d_fnw,eng.h*2));
    HIP_OK(hipMalloc(&s->d_lm_out,4096*2));
    // lm-head / argmax / batching buffers — formerly static locals in forward functions (fixes #59,#63)
    HIP_OK(hipMalloc(&s->d_lm_vocab,(size_t)eng.vocab*2));
    HIP_OK(hipMalloc(&s->d_argmax_idx,4));
    HIP_OK(hipMalloc(&s->d_argmax_val,4));
    HIP_OK(hipMalloc(&s->d_sorted_ids,(size_t)8*4));
    HIP_OK(hipMalloc(&s->d_expert_counts,(size_t)17*4));
    HIP_OK(hipMalloc(&s->d_expert_offsets,(size_t)17*4));
    HIP_OK(hipMalloc(&s->d_embed,(size_t)eng.vocab*eng.h*2));
    HIP_OK(hipMalloc(&s->d_conv,(size_t)eng.n_layers*2*eng.qkv*2));
    HIP_OK(hipMalloc(&s->d_phs,(size_t)eng.n_layers*eng.h*2));
    HIP_OK(hipMalloc(&s->d_prev_rs,(size_t)eng.n_layers*eng.rtr_h*4));
    HIP_OK(hipMalloc(&s->d_expert_idx,4)); HIP_OK(hipMalloc(&s->d_expert_wt,4));
    
    upf16(s->embed,s->d_embed,eng.vocab*eng.h,s->st);
    std::vector<__half>hf(eng.h);for(int i=0;i<eng.h;i++)hf[i]=__float2half(fnorm[i]);
    HIP_OK(hipMemcpy(s->d_fnw,hf.data(),eng.h*2,hipMemcpyHostToDevice));
    
    auto A=[&](auto&p,int n){HIP_OK(hipMalloc(&p,n*2));};
    auto B=[&](auto&p,int n){HIP_OK(hipMalloc(&p,n*4));};
    
    for(int il=0;il<eng.n_layers;il++){
        auto& l=s->lw[il];
        A(l.nw,eng.h);upf16(W("model_layers_"+L(il)+"_input_layernorm_weight.bin"),l.nw,eng.h,s->st);
        A(l.wq,eng.qd*eng.h);upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_q_proj_weight.bin"),l.wq,eng.qd*eng.h,s->st);
        A(l.wk,eng.kd*eng.h);upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_k_proj_weight.bin"),l.wk,eng.kd*eng.h,s->st);
        A(l.wv1,(eng.kd/2)*eng.h);upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_v_proj_current_weight.bin"),l.wv1,(eng.kd/2)*eng.h,s->st);
        A(l.wv2,(eng.kd/2)*eng.h);upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_v_proj_delayed_weight.bin"),l.wv2,(eng.kd/2)*eng.h,s->st);
        A(l.wo,eng.h*eng.qd);upf16(W("model_layers_"+L(il)+"_self_attn_o_proj_weight.bin"),l.wo,eng.h*eng.qd,s->st);
        B(l.cdw,eng.qkv*2);upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_depthwise_weight.bin"),l.cdw,eng.qkv*2,s->st);
        B(l.cdb,eng.qkv);upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_depthwise_bias.bin"),l.cdb,eng.qkv,s->st);
        B(l.cgw,eng.qkv*128*2);upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_grouped_weight.bin"),l.cgw,eng.qkv*128*2,s->st);
        B(l.cgb,eng.qkv);upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_grouped_bias.bin"),l.cgb,eng.qkv,s->st);
        B(l.ks,eng.nkv);upf32(W("model_layers_"+L(il)+"_self_attn_qk_norm_temp.bin"),l.ks,eng.nkv,s->st);
        B(l.pahss,eng.h);upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_hidden_states_scale.bin"),l.pahss,eng.h,s->st);
        B(l.pahsb,eng.h);upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_hidden_states_bias.bin"),l.pahsb,eng.h,s->st);
        B(l.parss,eng.h);upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_residual_scale.bin"),l.parss,eng.h,s->st);
        B(l.parsb,eng.h);upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_residual_bias.bin"),l.parsb,eng.h,s->st);
        A(l.pan,eng.h);upf16(W("model_layers_"+L(il)+"_post_attention_layernorm_weight.bin"),l.pan,eng.h,s->st);
        // gdw stored transposed [eng.h, eng.rtr_h] for cache-friendly GPU access
        // (each thread reads stride-1 floats in the inner loop instead of stride eng.h).
        // Upstream PyTorch saves it as [eng.rtr_h, eng.h] — we transpose on load.
        B(l.gdw,eng.h*eng.rtr_h);
        {
            auto raw=W("model_layers_"+L(il)+"_mlp_gate_down_proj_weight.bin");
            std::vector<float> tr(eng.h*eng.rtr_h);
            for(int i=0;i<eng.rtr_h;i++)for(int j=0;j<eng.h;j++)tr[j*eng.rtr_h+i]=raw[i*eng.h+j];
            upf32(tr,l.gdw,eng.h*eng.rtr_h,s->st);
        }
        B(l.gdb,eng.rtr_h);upf32(W("model_layers_"+L(il)+"_mlp_gate_down_proj_bias.bin"),l.gdb,eng.rtr_h,s->st);
        B(l.rfn,eng.rtr_h);upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_norm_weight.bin"),l.rfn,eng.rtr_h,s->st);
        B(l.rf1,eng.rtr_h*eng.rtr_h);upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc1_weight.bin"),l.rf1,eng.rtr_h*eng.rtr_h,s->st);
        B(l.rf1b,eng.rtr_h);upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc1_bias.bin"),l.rf1b,eng.rtr_h,s->st);
        B(l.rf2,eng.rtr_h*eng.rtr_h);upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc2_weight.bin"),l.rf2,eng.rtr_h*eng.rtr_h,s->st);
        B(l.rf2b,eng.rtr_h);upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc2_bias.bin"),l.rf2b,eng.rtr_h,s->st);
        B(l.rout,eng.n_exp_t*eng.rtr_h);upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_out_proj_weight.bin"),l.rout,eng.n_exp_t*eng.rtr_h,s->st);
        B(l.bb,eng.n_exp_t);upf32(W("model_layers_"+L(il)+"_mlp_gate_balancing_biases.bin"),l.bb,eng.n_exp_t,s->st);
        auto sz_gu=eng.n_exp*2*eng.n_ff*eng.h;auto sz_dn=eng.n_exp*eng.h*eng.n_ff;
        auto e1=hipMalloc(&l.gu,sz_gu*2);auto e2=hipMalloc(&l.dn,sz_dn*2);
        if(e1!=hipSuccess||e2!=hipSuccess){l.gu=nullptr;l.dn=nullptr;}else{
            upf16(W("model_layers_"+L(il)+"_mlp_experts_gate_up_proj.bin"),l.gu,sz_gu,s->st);
            upf16(W("model_layers_"+L(il)+"_mlp_experts_down_proj.bin"),l.dn,sz_dn,s->st);
        }
        B(l.pmhss,eng.h);upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_hidden_states_scale.bin"),l.pmhss,eng.h,s->st);
        B(l.pmhsb,eng.h);upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_hidden_states_bias.bin"),l.pmhsb,eng.h,s->st);
        B(l.pmrss,eng.h);upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_residual_scale.bin"),l.pmrss,eng.h,s->st);
        B(l.pmrsb,eng.h);upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_residual_bias.bin"),l.pmrsb,eng.h,s->st);
        
        std::string ep="/tmp/zaya_weights/model_layers_"+L(il)+"_mlp_gate_router_states_scale.bin";
        std::ifstream ff(ep,std::ios::binary);
        if(ff){ff.read((char*)&s->eda_scale[il],4);s->has_eda[il]=true;}else{s->eda_scale[il]=0;s->has_eda[il]=false;}
    }
    HIP_OK(hipStreamSynchronize(s->st));
    return s;
}

// ── Forward: token in, logits out ──
void zaya_forward(ZayaState* s, int token_id, float* logits_out) {
    int g1 = (eng.h+BLK-1)/BLK;
    std::vector<__half> hh(eng.h);
    for(int i=0;i<eng.h;i++){float raw=s->embed[token_id*(size_t)eng.h+i];hh[i]=__float2half((raw+s->ibias[i])*s->iscale[i]);}
    hipMemcpyAsync(s->d_hs,hh.data(),eng.h*2,hipMemcpyHostToDevice,s->st);
    
    for(int il=0;il<eng.n_layers;il++){
        auto& l=s->lw[il];
        // Tiled CCA: eng.qkv proj (tiled GEMV) + custom (conv+attn) + O_proj
        copy_k<<<g1,BLK,0,s->st>>>(s->d_phs+(size_t)il*eng.h,s->d_hs,eng.h);
        rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,l.nw,eng.h);
        moe_tiled_gemv<<<eng.qd/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_hs,l.wq,eng.qd,eng.h);
        moe_tiled_gemv<<<eng.kd/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd,s->d_hs,l.wk,eng.kd,eng.h);
        moe_tiled_gemv<<<eng.kd/2/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd+eng.kd,s->d_hs,l.wv1,eng.kd/2,eng.h);
        moe_tiled_gemv<<<eng.kd/2/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd+eng.kd+eng.kd/2,s->d_phs+(size_t)il*eng.h,l.wv2,eng.kd/2,eng.h);
        v_interleave_kernel<<<(eng.kd/2+BLK-1)/BLK,BLK,0,s->st>>>(s->d_tmp+eng.qd,s->d_tmp+eng.qd+eng.kd,s->d_tmp+eng.qd+eng.kd+eng.kd/2,eng.kd/2);
        cca_custom_kernel<<<1,256,0,s->st>>>(s->d_tmp,s->d_tmp+eng.qd,s->d_tmp+eng.qd,s->d_phs+(size_t)il*eng.h,s->d_conv+(size_t)il*2*eng.qkv,l.cdw,l.cdb,l.cgw,l.cgb,l.ks,s->d_ao,s->d_conv+(size_t)il*2*eng.qkv,s->d_phs+(size_t)il*eng.h,il,1);
        moe_tiled_gemv<<<eng.h/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_ao,s->d_ao,l.wo,eng.h,eng.qd);
        residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_ao,s->d_hs,l.pahss,l.pahsb,l.parss,l.parsb,eng.h);
        copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_ao,eng.h);
        rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,l.pan,eng.h);
        if(l.gu&&l.dn){
            eda_router_gpu_kernel<<<1,eng.rtr_h,0,s->st>>>(s->d_hs,s->d_prev_rs+(size_t)il*eng.rtr_h,s->has_eda[il]?1:0,s->eda_scale[il],l.gdw,l.gdb,l.rfn,l.rf1,l.rf1b,l.rf2,l.rf2b,l.rout,l.bb,s->d_prev_rs+(size_t)il*eng.rtr_h,s->d_expert_idx,s->d_expert_wt);
            encode_expert_cache_kernel<<<1,32,0,s->st>>>(s->d_prev_rs+(size_t)il*eng.rtr_h,s->d_expert_idx,eng.rtr_h);
            const int gb=(2*eng.n_ff+WMMA_M-1)/WMMA_M;
            const int db=(eng.h+WMMA_M-1)/WMMA_M;
            const int sb=(eng.n_ff+BLK-1)/BLK;
            wmma_gateup_kernel<<<gb,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_hs,l.gu,s->d_expert_idx);
            silu_mul_k<<<sb,BLK,0,s->st>>>(s->d_ao,s->d_tmp,s->d_tmp+eng.n_ff,eng.n_ff);
            wmma_down_kernel<<<db,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_ao,l.dn,s->d_expert_idx);
            residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,l.pmhss,l.pmhsb,l.pmrss,l.pmrsb,eng.h);
            copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_tmp,eng.h);
        }else{
            copy_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,eng.h);
        }
    }
    rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,s->d_fnw,eng.h);
    
    // lm_head — tiled GEMV in a single launch; buffer allocated in zaya_init (fixes #59)
    // No sync needed before the lm_head: both the RMSNorm and the lm_head GEMV are on
    // the same stream, so the GEMV waits for the RMSNorm automatically (fixes perf).
    moe_tiled_gemv<<<(eng.vocab+WMMA_M-1)/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_lm_vocab,s->d_hs,s->d_embed,eng.vocab,eng.h);
    hipStreamSynchronize(s->st);
    std::vector<__half> lh(eng.vocab);
    hipMemcpy(lh.data(),s->d_lm_vocab,(size_t)eng.vocab*2,hipMemcpyDeviceToHost);
    for(int v=0;v<eng.vocab;v++)logits_out[v]=__half2float(lh[v]);
}

// ── Forward greedy: same as forward but only returns argmax (much faster) ──
int zaya_forward_greedy(ZayaState* s, int token_id) {
    int g1 = (eng.h+BLK-1)/BLK;
    std::vector<__half> hh(eng.h);
    for(int i=0;i<eng.h;i++){float raw=s->embed[token_id*(size_t)eng.h+i];hh[i]=__float2half((raw+s->ibias[i])*s->iscale[i]);}
    hipMemcpyAsync(s->d_hs,hh.data(),eng.h*2,hipMemcpyHostToDevice,s->st);
    
    for(int il=0;il<eng.n_layers;il++){
        auto& l=s->lw[il];
        copy_k<<<g1,BLK,0,s->st>>>(s->d_phs+(size_t)il*eng.h,s->d_hs,eng.h);
        rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,l.nw,eng.h);
        moe_tiled_gemv<<<eng.qd/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_hs,l.wq,eng.qd,eng.h);
        moe_tiled_gemv<<<eng.kd/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd,s->d_hs,l.wk,eng.kd,eng.h);
        moe_tiled_gemv<<<eng.kd/2/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd+eng.kd,s->d_hs,l.wv1,eng.kd/2,eng.h);
        moe_tiled_gemv<<<eng.kd/2/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd+eng.kd+eng.kd/2,s->d_phs+(size_t)il*eng.h,l.wv2,eng.kd/2,eng.h);
        v_interleave_kernel<<<(eng.kd/2+BLK-1)/BLK,BLK,0,s->st>>>(s->d_tmp+eng.qd,s->d_tmp+eng.qd+eng.kd,s->d_tmp+eng.qd+eng.kd+eng.kd/2,eng.kd/2);
        cca_custom_kernel<<<1,256,0,s->st>>>(s->d_tmp,s->d_tmp+eng.qd,s->d_tmp+eng.qd,s->d_phs+(size_t)il*eng.h,s->d_conv+(size_t)il*2*eng.qkv,l.cdw,l.cdb,l.cgw,l.cgb,l.ks,s->d_ao,s->d_conv+(size_t)il*2*eng.qkv,s->d_phs+(size_t)il*eng.h,il,1);
        moe_tiled_gemv<<<eng.h/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_ao,s->d_ao,l.wo,eng.h,eng.qd);
        residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_ao,s->d_hs,l.pahss,l.pahsb,l.parss,l.parsb,eng.h);
        copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_ao,eng.h);
        rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,l.pan,eng.h);
        if(l.gu&&l.dn){
            eda_router_gpu_kernel<<<1,eng.rtr_h,0,s->st>>>(s->d_hs,s->d_prev_rs+(size_t)il*eng.rtr_h,s->has_eda[il]?1:0,s->eda_scale[il],l.gdw,l.gdb,l.rfn,l.rf1,l.rf1b,l.rf2,l.rf2b,l.rout,l.bb,s->d_prev_rs+(size_t)il*eng.rtr_h,s->d_expert_idx,s->d_expert_wt);
            encode_expert_cache_kernel<<<1,32,0,s->st>>>(s->d_prev_rs+(size_t)il*eng.rtr_h,s->d_expert_idx,eng.rtr_h);
            const int gb=(2*eng.n_ff+WMMA_M-1)/WMMA_M;
            const int db=(eng.h+WMMA_M-1)/WMMA_M;
            const int sb=(eng.n_ff+BLK-1)/BLK;
            wmma_gateup_kernel<<<gb,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_hs,l.gu,s->d_expert_idx);
            silu_mul_k<<<sb,BLK,0,s->st>>>(s->d_ao,s->d_tmp,s->d_tmp+eng.n_ff,eng.n_ff);
            wmma_down_kernel<<<db,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_ao,l.dn,s->d_expert_idx);
            residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,l.pmhss,l.pmhsb,l.pmrss,l.pmrsb,eng.h);
            copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_tmp,eng.h);
        }else{copy_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,eng.h);}
    }
    rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,s->d_fnw,eng.h);
    
    // lm_head + GPU argmax (no full logit copy); buffers allocated in zaya_init (fixes #59)
    moe_tiled_gemv<<<(eng.vocab+WMMA_M-1)/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_lm_vocab,s->d_hs,s->d_embed,eng.vocab,eng.h);
    argmax_kernel<<<1,256,0,s->st>>>(s->d_lm_vocab,eng.vocab,s->d_argmax_idx,s->d_argmax_val);
    hipStreamSynchronize(s->st);
    int best;
    hipMemcpy(&best,s->d_argmax_idx,4,hipMemcpyDeviceToHost);
    return best;
}

// ── Forward batch: process B tokens through all layers ──
// Uses batched router + batch-union MoE for expert dedup.
// B <= 8 recommended (constrained by shared memory in union kernel).
void zaya_forward_batch(ZayaState* s, const int* token_ids, float* logits_out, int B) {
    int g1 = (eng.h+BLK-1)/BLK;
    if (B > 8) B = 8; // safety cap

    // ── Embedding lookup for all B tokens ──
    std::vector<__half> hh(B * eng.h);
    for (int b = 0; b < B; b++) {
        int tid = token_ids[b];
        for (int i = 0; i < eng.h; i++) {
            float raw = s->embed[tid * (size_t)eng.h + i];
            hh[b * (size_t)eng.h + i] = __float2half((raw + s->ibias[i]) * s->iscale[i]);
        }
    }
    hipMemcpyAsync(s->d_hs, hh.data(), B * eng.h * 2, hipMemcpyHostToDevice, s->st);

    for (int il = 0; il < eng.n_layers; il++) {
        auto& l = s->lw[il];

        // CCA attention — one launch per token in batch (or batched CCA kernel)
        // For now, sequential CCA per token (can be optimized with batched CCA)
        for (int b = 0; b < B; b++) {
            __half* hs_b = s->d_hs + (size_t)b * eng.h;
            cca_attn_kernel<<<1, 256, 0, s->st>>>(
                hs_b, s->d_phs + (size_t)il * eng.h,
                s->d_conv + (size_t)il * 2 * eng.qkv, il,
                l.wq, l.wk, l.wv1, l.wv2, l.wo,
                l.cdw, l.cdb, l.cgw, l.cgb, l.ks, l.nw,
                s->d_ao + (size_t)b * eng.h,
                s->d_conv + (size_t)il * 2 * eng.qkv,
                s->d_phs + (size_t)il * eng.h);
        }

        // Post-attention residual + RMSNorm (per token)
        for (int b = 0; b < B; b++) {
            __half* hs_b  = s->d_hs + (size_t)b * eng.h;
            __half* ao_b  = s->d_ao + (size_t)b * eng.h;
            residual_scale_k<<<g1, BLK, 0, s->st>>>(ao_b, hs_b,
                l.pahss, l.pahsb, l.parss, l.parsb, eng.h);
            copy_k<<<g1, BLK, 0, s->st>>>(hs_b, ao_b, eng.h);
            rmsnorm_k<<<1, BLK, 0, s->st>>>(hs_b, l.pan, eng.h);
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

                // Phase 2: Sort token IDs by expert (histogram + prefix sum + scatter);
                // buffers allocated in zaya_init (fixes #63).
                moe_sort_histogram_kernel<<<1, 32, 0, s->st>>>(
                    s->d_expert_idx, s->d_expert_counts, s->d_expert_offsets,
                    s->d_sorted_ids, B);

                // Phase 3: Expert FFN (sorted, one block per expert with count>0)
                moe_sorted_expert_kernel<<<eng.n_exp, 256, 0, s->st>>>(
                    s->d_hs, s->d_sorted_ids, s->d_expert_counts, s->d_expert_offsets,
                    l.gu, l.dn, s->d_tmp, B);

                // Phase 4: Handle MOD skip tokens (expert_idx == eng.n_exp = 16)
                // These skip the expert FFN entirely (out = hs, identity).
                if (s->d_expert_counts) {  // always true, keeps compiler happy
                    moe_modskip_passthrough_kernel<<<(B + 255) / 256, 256, 0, s->st>>>(
                        s->d_tmp, s->d_hs, s->d_expert_idx, eng.n_exp, B);
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
            }

            // Post-MLP residual scale (per token)
            for (int b = 0; b < B; b++) {
                __half* hs_b  = s->d_hs + (size_t)b * eng.h;
                __half* tmp_b = s->d_tmp + (size_t)b * eng.h;
                residual_scale_k<<<g1, BLK, 0, s->st>>>(tmp_b, hs_b,
                    l.pmhss, l.pmhsb, l.pmrss, l.pmrsb, eng.h);
                copy_k<<<g1, BLK, 0, s->st>>>(hs_b, tmp_b, eng.h);
            }
        }
    }

    // Final RMSNorm (per token)
    for (int b = 0; b < B; b++) {
        __half* hs_b = s->d_hs + (size_t)b * eng.h;
        rmsnorm_k<<<1, BLK, 0, s->st>>>(hs_b, s->d_fnw, eng.h);
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
        } else {
            moe_tiled_gemv<<<(eng.vocab + WMMA_M - 1) / WMMA_M, WMMA_THREADS, 0, s->st>>>(
                s->d_lm_vocab, s->d_hs, s->d_embed, eng.vocab, eng.h);
        }
        #else
        for (int b = 0; b < B; b++) {
            __half* hs_b = s->d_hs + (size_t)b * eng.h;
            moe_tiled_gemv<<<(eng.vocab + WMMA_M - 1) / WMMA_M, WMMA_THREADS, 0, s->st>>>(
                s->d_lm_vocab + (size_t)b * eng.vocab, hs_b, s->d_embed, eng.vocab, eng.h);
        }
        #endif
    }
    hipStreamSynchronize(s->st);

    // Copy logits for all B tokens
    std::vector<__half> lh(B * eng.vocab);
    hipMemcpy(lh.data(), s->d_lm_vocab, (size_t)B * eng.vocab * 2, hipMemcpyDeviceToHost);
    for (int b = 0; b < B; b++)
        for (int v = 0; v < eng.vocab; v++)
            logits_out[b * (size_t)eng.vocab + v] = __half2float(lh[b * (size_t)eng.vocab + v]);
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
    HIP_OK(hipMemsetAsync(s->d_conv,0,(size_t)eng.n_layers*2*eng.qkv*2,s->st));
    HIP_OK(hipMemsetAsync(s->d_phs,0,(size_t)eng.n_layers*eng.h*2,s->st));
    HIP_OK(hipMemsetAsync(s->d_prev_rs,0,(size_t)eng.n_layers*eng.rtr_h*4,s->st));
    init_expert_cache_sentinel<<<1, 64, 0, s->st>>>(s->d_prev_rs, eng.n_layers, eng.rtr_h);
}

// ── Destroy ──
void zaya_destroy(ZayaState* s) {
    if (!s) return;
    auto safe = [](auto p) { if (p) (void)hipFree(p); };
    safe(s->d_hs); safe(s->d_ao); safe(s->d_tmp); safe(s->d_fnw);
    safe(s->d_lm_out); safe(s->d_embed); safe(s->d_conv); safe(s->d_phs);
    safe(s->d_prev_rs); safe(s->d_expert_idx); safe(s->d_expert_wt);
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
    if (s->st) hipStreamDestroy(s->st);
    safe(s->d_lm_vocab); safe(s->d_argmax_idx); safe(s->d_argmax_val);
    safe(s->d_sorted_ids); safe(s->d_expert_counts); safe(s->d_expert_offsets);
    delete s;
}

} // extern "C"
