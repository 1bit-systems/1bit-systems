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

extern "C" {

// ── Architecture ──
static constexpr int H=2048,NQ=8,NKV=2,HD=128,QD=NQ*HD,KD=NKV*HD,QKV=QD+KD;
static constexpr int N_LAYERS=40,VOCAB=262272,N_EXP=16,N_EXP_T=17,N_FF=2048,RTR_H=256;
static constexpr float RMD_EPS=1e-5f; static constexpr int BLK=256;

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

// ── Public state ──
struct ZayaState {
    __half *d_hs,*d_ao,*d_tmp,*d_fnw,*d_lm_out,*d_embed;
    __half *d_conv,*d_phs,*d_lm_vocab; float *d_prev_rs;
    int *d_argmax_idx; float *d_argmax_val;
    int *d_expert_idx,*d_expert_wt;  // MoE dispatch
    int *d_sorted_ids,*d_expert_counts,*d_expert_offsets;  // batched MoE sort (#59,#63)
    hipStream_t st;
    LayerW lw[N_LAYERS];
    bool has_eda[N_LAYERS];
    float eda_scale[N_LAYERS];
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

    HIP_OK(hipMalloc(&s->d_hs,H*2)); HIP_OK(hipMalloc(&s->d_ao,H*2));
    HIP_OK(hipMalloc(&s->d_tmp,H*2)); HIP_OK(hipMalloc(&s->d_fnw,H*2));
    HIP_OK(hipMalloc(&s->d_lm_out,4096*2));
    // lm-head / argmax / batching buffers — formerly static locals in forward functions (fixes #59,#63)
    HIP_OK(hipMalloc(&s->d_lm_vocab,(size_t)VOCAB*2));
    HIP_OK(hipMalloc(&s->d_argmax_idx,4));
    HIP_OK(hipMalloc(&s->d_argmax_val,4));
    HIP_OK(hipMalloc(&s->d_sorted_ids,(size_t)8*4));
    HIP_OK(hipMalloc(&s->d_expert_counts,(size_t)17*4));
    HIP_OK(hipMalloc(&s->d_expert_offsets,(size_t)17*4));
    HIP_OK(hipMalloc(&s->d_embed,(size_t)VOCAB*H*2));
    HIP_OK(hipMalloc(&s->d_conv,(size_t)N_LAYERS*2*QKV*2));
    HIP_OK(hipMalloc(&s->d_phs,(size_t)N_LAYERS*H*2));
    HIP_OK(hipMalloc(&s->d_prev_rs,(size_t)N_LAYERS*RTR_H*4));
    HIP_OK(hipMalloc(&s->d_expert_idx,4)); HIP_OK(hipMalloc(&s->d_expert_wt,4));
    
    upf16(s->embed,s->d_embed,VOCAB*H,s->st);
    std::vector<__half>hf(H);for(int i=0;i<H;i++)hf[i]=__float2half(fnorm[i]);
    HIP_OK(hipMemcpy(s->d_fnw,hf.data(),H*2,hipMemcpyHostToDevice));
    
    auto A=[&](auto&p,int n){HIP_OK(hipMalloc(&p,n*2));};
    auto B=[&](auto&p,int n){HIP_OK(hipMalloc(&p,n*4));};
    
    for(int il=0;il<N_LAYERS;il++){
        auto& l=s->lw[il];
        A(l.nw,H);upf16(W("model_layers_"+L(il)+"_input_layernorm_weight.bin"),l.nw,H,s->st);
        A(l.wq,QD*H);upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_q_proj_weight.bin"),l.wq,QD*H,s->st);
        A(l.wk,KD*H);upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_k_proj_weight.bin"),l.wk,KD*H,s->st);
        A(l.wv1,(KD/2)*H);upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_v_proj_current_weight.bin"),l.wv1,(KD/2)*H,s->st);
        A(l.wv2,(KD/2)*H);upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_v_proj_delayed_weight.bin"),l.wv2,(KD/2)*H,s->st);
        A(l.wo,H*QD);upf16(W("model_layers_"+L(il)+"_self_attn_o_proj_weight.bin"),l.wo,H*QD,s->st);
        B(l.cdw,QKV*2);upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_depthwise_weight.bin"),l.cdw,QKV*2,s->st);
        B(l.cdb,QKV);upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_depthwise_bias.bin"),l.cdb,QKV,s->st);
        B(l.cgw,QKV*128*2);upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_grouped_weight.bin"),l.cgw,QKV*128*2,s->st);
        B(l.cgb,QKV);upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_grouped_bias.bin"),l.cgb,QKV,s->st);
        B(l.ks,NKV);upf32(W("model_layers_"+L(il)+"_self_attn_qk_norm_temp.bin"),l.ks,NKV,s->st);
        B(l.pahss,H);upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_hidden_states_scale.bin"),l.pahss,H,s->st);
        B(l.pahsb,H);upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_hidden_states_bias.bin"),l.pahsb,H,s->st);
        B(l.parss,H);upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_residual_scale.bin"),l.parss,H,s->st);
        B(l.parsb,H);upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_residual_bias.bin"),l.parsb,H,s->st);
        A(l.pan,H);upf16(W("model_layers_"+L(il)+"_post_attention_layernorm_weight.bin"),l.pan,H,s->st);
        // gdw stored transposed [H, RTR_H] for cache-friendly GPU access
        // (each thread reads stride-1 floats in the inner loop instead of stride H).
        // Upstream PyTorch saves it as [RTR_H, H] — we transpose on load.
        B(l.gdw,H*RTR_H);
        {
            auto raw=W("model_layers_"+L(il)+"_mlp_gate_down_proj_weight.bin");
            std::vector<float> tr(H*RTR_H);
            for(int i=0;i<RTR_H;i++)for(int j=0;j<H;j++)tr[j*RTR_H+i]=raw[i*H+j];
            upf32(tr,l.gdw,H*RTR_H,s->st);
        }
        B(l.gdb,RTR_H);upf32(W("model_layers_"+L(il)+"_mlp_gate_down_proj_bias.bin"),l.gdb,RTR_H,s->st);
        B(l.rfn,RTR_H);upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_norm_weight.bin"),l.rfn,RTR_H,s->st);
        B(l.rf1,RTR_H*RTR_H);upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc1_weight.bin"),l.rf1,RTR_H*RTR_H,s->st);
        B(l.rf1b,RTR_H);upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc1_bias.bin"),l.rf1b,RTR_H,s->st);
        B(l.rf2,RTR_H*RTR_H);upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc2_weight.bin"),l.rf2,RTR_H*RTR_H,s->st);
        B(l.rf2b,RTR_H);upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc2_bias.bin"),l.rf2b,RTR_H,s->st);
        B(l.rout,N_EXP_T*RTR_H);upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_out_proj_weight.bin"),l.rout,N_EXP_T*RTR_H,s->st);
        B(l.bb,N_EXP_T);upf32(W("model_layers_"+L(il)+"_mlp_gate_balancing_biases.bin"),l.bb,N_EXP_T,s->st);
        auto sz_gu=N_EXP*2*N_FF*H;auto sz_dn=N_EXP*H*N_FF;
        auto e1=hipMalloc(&l.gu,sz_gu*2);auto e2=hipMalloc(&l.dn,sz_dn*2);
        if(e1!=hipSuccess||e2!=hipSuccess){l.gu=nullptr;l.dn=nullptr;}else{
            upf16(W("model_layers_"+L(il)+"_mlp_experts_gate_up_proj.bin"),l.gu,sz_gu,s->st);
            upf16(W("model_layers_"+L(il)+"_mlp_experts_down_proj.bin"),l.dn,sz_dn,s->st);
        }
        B(l.pmhss,H);upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_hidden_states_scale.bin"),l.pmhss,H,s->st);
        B(l.pmhsb,H);upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_hidden_states_bias.bin"),l.pmhsb,H,s->st);
        B(l.pmrss,H);upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_residual_scale.bin"),l.pmrss,H,s->st);
        B(l.pmrsb,H);upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_residual_bias.bin"),l.pmrsb,H,s->st);
        
        std::string ep="/tmp/zaya_weights/model_layers_"+L(il)+"_mlp_gate_router_states_scale.bin";
        std::ifstream ff(ep,std::ios::binary);
        if(ff){ff.read((char*)&s->eda_scale[il],4);s->has_eda[il]=true;}else{s->eda_scale[il]=0;s->has_eda[il]=false;}
    }
    HIP_OK(hipStreamSynchronize(s->st));
    return s;
}

// ── Forward: token in, logits out ──
void zaya_forward(ZayaState* s, int token_id, float* logits_out) {
    int g1 = (H+BLK-1)/BLK;
    std::vector<__half> hh(H);
    for(int i=0;i<H;i++){float raw=s->embed[token_id*(size_t)H+i];hh[i]=__float2half((raw+s->ibias[i])*s->iscale[i]);}
    hipMemcpyAsync(s->d_hs,hh.data(),H*2,hipMemcpyHostToDevice,s->st);
    
    for(int il=0;il<N_LAYERS;il++){
        auto& l=s->lw[il];
        // Tiled CCA: QKV proj (tiled GEMV) + custom (conv+attn) + O_proj
        copy_k<<<g1,BLK,0,s->st>>>(s->d_phs+(size_t)il*H,s->d_hs,H);
        rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,l.nw,H);
        moe_tiled_gemv<<<QD/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_hs,l.wq,QD,H);
        moe_tiled_gemv<<<KD/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+QD,s->d_hs,l.wk,KD,H);
        moe_tiled_gemv<<<KD/2/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+QD+KD,s->d_hs,l.wv1,KD/2,H);
        moe_tiled_gemv<<<KD/2/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+QD+KD+KD/2,s->d_phs+(size_t)il*H,l.wv2,KD/2,H);
        v_interleave_kernel<<<(KD/2+BLK-1)/BLK,BLK,0,s->st>>>(s->d_tmp+QD,s->d_tmp+QD+KD,s->d_tmp+QD+KD+KD/2,KD/2);
        cca_custom_kernel<<<1,256,0,s->st>>>(s->d_tmp,s->d_tmp+QD,s->d_tmp+QD,s->d_phs+(size_t)il*H,s->d_conv+(size_t)il*2*QKV,l.cdw,l.cdb,l.cgw,l.cgb,l.ks,s->d_ao,s->d_conv+(size_t)il*2*QKV,s->d_phs+(size_t)il*H,il,1);
        moe_tiled_gemv<<<H/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_ao,s->d_ao,l.wo,H,QD);
        residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_ao,s->d_hs,l.pahss,l.pahsb,l.parss,l.parsb,H);
        copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_ao,H);
        rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,l.pan,H);
        if(l.gu&&l.dn){
            eda_router_gpu_kernel<<<1,RTR_H,0,s->st>>>(s->d_hs,s->d_prev_rs+(size_t)il*RTR_H,s->has_eda[il]?1:0,s->eda_scale[il],l.gdw,l.gdb,l.rfn,l.rf1,l.rf1b,l.rf2,l.rf2b,l.rout,l.bb,s->d_prev_rs+(size_t)il*RTR_H,s->d_expert_idx,s->d_expert_wt);
            encode_expert_cache_kernel<<<1,32,0,s->st>>>(s->d_prev_rs+(size_t)il*RTR_H,s->d_expert_idx,RTR_H);
            const int gb=(2*N_FF+WMMA_M-1)/WMMA_M;
            const int db=(H+WMMA_M-1)/WMMA_M;
            const int sb=(N_FF+BLK-1)/BLK;
            wmma_gateup_kernel<<<gb,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_hs,l.gu,s->d_expert_idx);
            silu_mul_k<<<sb,BLK,0,s->st>>>(s->d_ao,s->d_tmp,s->d_tmp+N_FF,N_FF);
            wmma_down_kernel<<<db,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_ao,l.dn,s->d_expert_idx);
            residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,l.pmhss,l.pmhsb,l.pmrss,l.pmrsb,H);
            copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_tmp,H);
        }else{
            copy_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,H);
        }
    }
    rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,s->d_fnw,H);
    hipStreamSynchronize(s->st);
    
    // lm_head — tiled GEMV in a single launch; buffer allocated in zaya_init (fixes #59)
    moe_tiled_gemv<<<(VOCAB+WMMA_M-1)/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_lm_vocab,s->d_hs,s->d_embed,VOCAB,H);
    hipStreamSynchronize(s->st);
    std::vector<__half> lh(VOCAB);
    hipMemcpy(lh.data(),s->d_lm_vocab,(size_t)VOCAB*2,hipMemcpyDeviceToHost);
    for(int v=0;v<VOCAB;v++)logits_out[v]=__half2float(lh[v]);
}

// ── Forward greedy: same as forward but only returns argmax (much faster) ──
int zaya_forward_greedy(ZayaState* s, int token_id) {
    int g1 = (H+BLK-1)/BLK;
    std::vector<__half> hh(H);
    for(int i=0;i<H;i++){float raw=s->embed[token_id*(size_t)H+i];hh[i]=__float2half((raw+s->ibias[i])*s->iscale[i]);}
    hipMemcpyAsync(s->d_hs,hh.data(),H*2,hipMemcpyHostToDevice,s->st);
    
    for(int il=0;il<N_LAYERS;il++){
        auto& l=s->lw[il];
        copy_k<<<g1,BLK,0,s->st>>>(s->d_phs+(size_t)il*H,s->d_hs,H);
        rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,l.nw,H);
        moe_tiled_gemv<<<QD/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_hs,l.wq,QD,H);
        moe_tiled_gemv<<<KD/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+QD,s->d_hs,l.wk,KD,H);
        moe_tiled_gemv<<<KD/2/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+QD+KD,s->d_hs,l.wv1,KD/2,H);
        moe_tiled_gemv<<<KD/2/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+QD+KD+KD/2,s->d_phs+(size_t)il*H,l.wv2,KD/2,H);
        v_interleave_kernel<<<(KD/2+BLK-1)/BLK,BLK,0,s->st>>>(s->d_tmp+QD,s->d_tmp+QD+KD,s->d_tmp+QD+KD+KD/2,KD/2);
        cca_custom_kernel<<<1,256,0,s->st>>>(s->d_tmp,s->d_tmp+QD,s->d_tmp+QD,s->d_phs+(size_t)il*H,s->d_conv+(size_t)il*2*QKV,l.cdw,l.cdb,l.cgw,l.cgb,l.ks,s->d_ao,s->d_conv+(size_t)il*2*QKV,s->d_phs+(size_t)il*H,il,1);
        moe_tiled_gemv<<<H/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_ao,s->d_ao,l.wo,H,QD);
        residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_ao,s->d_hs,l.pahss,l.pahsb,l.parss,l.parsb,H);
        copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_ao,H);
        rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,l.pan,H);
        if(l.gu&&l.dn){
            eda_router_gpu_kernel<<<1,RTR_H,0,s->st>>>(s->d_hs,s->d_prev_rs+(size_t)il*RTR_H,s->has_eda[il]?1:0,s->eda_scale[il],l.gdw,l.gdb,l.rfn,l.rf1,l.rf1b,l.rf2,l.rf2b,l.rout,l.bb,s->d_prev_rs+(size_t)il*RTR_H,s->d_expert_idx,s->d_expert_wt);
            encode_expert_cache_kernel<<<1,32,0,s->st>>>(s->d_prev_rs+(size_t)il*RTR_H,s->d_expert_idx,RTR_H);
            const int gb=(2*N_FF+WMMA_M-1)/WMMA_M;
            const int db=(H+WMMA_M-1)/WMMA_M;
            const int sb=(N_FF+BLK-1)/BLK;
            wmma_gateup_kernel<<<gb,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_hs,l.gu,s->d_expert_idx);
            silu_mul_k<<<sb,BLK,0,s->st>>>(s->d_ao,s->d_tmp,s->d_tmp+N_FF,N_FF);
            wmma_down_kernel<<<db,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_ao,l.dn,s->d_expert_idx);
            residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,l.pmhss,l.pmhsb,l.pmrss,l.pmrsb,H);
            copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_tmp,H);
        }else{copy_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,H);}
    }
    rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,s->d_fnw,H);
    
    // lm_head + GPU argmax (no full logit copy); buffers allocated in zaya_init (fixes #59)
    moe_tiled_gemv<<<(VOCAB+WMMA_M-1)/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_lm_vocab,s->d_hs,s->d_embed,VOCAB,H);
    argmax_kernel<<<1,256,0,s->st>>>(s->d_lm_vocab,VOCAB,s->d_argmax_idx,s->d_argmax_val);
    hipStreamSynchronize(s->st);
    int best;
    hipMemcpy(&best,s->d_argmax_idx,4,hipMemcpyDeviceToHost);
    return best;
}

// ── Forward batch: process B tokens through all layers ──
// Uses batched router + batch-union MoE for expert dedup.
// B <= 8 recommended (constrained by shared memory in union kernel).
void zaya_forward_batch(ZayaState* s, const int* token_ids, float* logits_out, int B) {
    int g1 = (H+BLK-1)/BLK;
    if (B > 8) B = 8; // safety cap

    // ── Embedding lookup for all B tokens ──
    std::vector<__half> hh(B * H);
    for (int b = 0; b < B; b++) {
        int tid = token_ids[b];
        for (int i = 0; i < H; i++) {
            float raw = s->embed[tid * (size_t)H + i];
            hh[b * (size_t)H + i] = __float2half((raw + s->ibias[i]) * s->iscale[i]);
        }
    }
    hipMemcpyAsync(s->d_hs, hh.data(), B * H * 2, hipMemcpyHostToDevice, s->st);

    for (int il = 0; il < N_LAYERS; il++) {
        auto& l = s->lw[il];

        // CCA attention — one launch per token in batch (or batched CCA kernel)
        // For now, sequential CCA per token (can be optimized with batched CCA)
        for (int b = 0; b < B; b++) {
            __half* hs_b = s->d_hs + (size_t)b * H;
            cca_attn_kernel<<<1, 256, 0, s->st>>>(
                hs_b, s->d_phs + (size_t)il * H,
                s->d_conv + (size_t)il * 2 * QKV, il,
                l.wq, l.wk, l.wv1, l.wv2, l.wo,
                l.cdw, l.cdb, l.cgw, l.cgb, l.ks, l.nw,
                s->d_ao + (size_t)b * H,
                s->d_conv + (size_t)il * 2 * QKV,
                s->d_phs + (size_t)il * H);
        }

        // Post-attention residual + RMSNorm (per token)
        for (int b = 0; b < B; b++) {
            __half* hs_b  = s->d_hs + (size_t)b * H;
            __half* ao_b  = s->d_ao + (size_t)b * H;
            residual_scale_k<<<g1, BLK, 0, s->st>>>(ao_b, hs_b,
                l.pahss, l.pahsb, l.parss, l.parsb, H);
            copy_k<<<g1, BLK, 0, s->st>>>(hs_b, ao_b, H);
            rmsnorm_k<<<1, BLK, 0, s->st>>>(hs_b, l.pan, H);
        }

        // Batched router + batch-union MoE
        // For B >= 4, use sorted dispatch to group tokens by expert
        // (loads each expert's weights ONCE, avoids L2 thrashing).
        // For B < 4, the fused per-token kernel is simpler and fast enough.
        if (l.gu && l.dn) {
            if (B >= 4) {
                // Phase 1: Route all B tokens (GPU-resident)
                batched_moe_router_kernel<<<B, 256, 0, s->st>>>(
                    s->d_hs, s->d_prev_rs + (size_t)il * RTR_H,
                    s->has_eda[il] ? 1 : 0, s->eda_scale[il],
                    l.gdw, l.gdb, l.rfn, l.rf1, l.rf1b,
                    l.rf2, l.rf2b, l.rout, l.bb,
                    s->d_prev_rs + (size_t)il * RTR_H,
                    s->d_expert_idx, s->d_expert_wt,
                    B);

                // Phase 2: Sort token IDs by expert (histogram + prefix sum + scatter);
                // buffers allocated in zaya_init (fixes #63).
                moe_sort_histogram_kernel<<<1, 32, 0, s->st>>>(
                    s->d_expert_idx, s->d_expert_counts, s->d_expert_offsets,
                    s->d_sorted_ids, B);

                // Phase 3: Expert FFN (sorted, one block per expert with count>0)
                moe_sorted_expert_kernel<<<N_EXP, 256, 0, s->st>>>(
                    s->d_hs, s->d_sorted_ids, s->d_expert_counts, s->d_expert_offsets,
                    l.gu, l.dn, s->d_tmp, B);

                // Phase 4: Handle MOD skip tokens (expert_idx == N_EXP = 16)
                // These skip the expert FFN entirely (out = hs, identity).
                if (s->d_expert_counts) {  // always true, keeps compiler happy
                    moe_modskip_passthrough_kernel<<<(B + 255) / 256, 256, 0, s->st>>>(
                        s->d_tmp, s->d_hs, s->d_expert_idx, N_EXP, B);
                }
            } else {
                batched_moe_fused_kernel<<<B, 256, 0, s->st>>>(
                    s->d_hs, s->d_prev_rs + (size_t)il * RTR_H,
                    s->has_eda[il] ? 1 : 0, s->eda_scale[il],
                    l.gdw, l.gdb, l.rfn, l.rf1, l.rf1b,
                    l.rf2, l.rf2b, l.rout, l.bb,
                    l.gu, l.dn,
                    s->d_prev_rs + (size_t)il * RTR_H,
                    s->d_tmp, s->d_expert_idx, s->d_expert_wt,
                    B);
            }

            // Post-MLP residual scale (per token)
            for (int b = 0; b < B; b++) {
                __half* hs_b  = s->d_hs + (size_t)b * H;
                __half* tmp_b = s->d_tmp + (size_t)b * H;
                residual_scale_k<<<g1, BLK, 0, s->st>>>(tmp_b, hs_b,
                    l.pmhss, l.pmhsb, l.pmrss, l.pmrsb, H);
                copy_k<<<g1, BLK, 0, s->st>>>(hs_b, tmp_b, H);
            }
        }
    }

    // Final RMSNorm (per token)
    for (int b = 0; b < B; b++) {
        __half* hs_b = s->d_hs + (size_t)b * H;
        rmsnorm_k<<<1, BLK, 0, s->st>>>(hs_b, s->d_fnw, H);
    }
    hipStreamSynchronize(s->st);

    // lm_head — tiled GEMV for each token; buffer allocated in zaya_init (fixes #63)
    {
        const size_t max_need = (size_t)8 * VOCAB * 2;  // B <= 8, allocated in zaya_init
        #if __has_include(<rocwmma/rocwmma.hpp>)
        if (B >= 2) {
            const int grid_x = (VOCAB + WMMA_M - 1) / WMMA_M;
            const int grid_y = (B + WMMA_N - 1) / WMMA_N;
            wmma_batched_gemv<<<dim3(grid_x, grid_y, 1), 32, 0, s->st>>>(
                s->d_lm_vocab, s->d_hs, s->d_embed, VOCAB, H, B);
        } else {
            moe_tiled_gemv<<<(VOCAB + WMMA_M - 1) / WMMA_M, WMMA_THREADS, 0, s->st>>>(
                s->d_lm_vocab, s->d_hs, s->d_embed, VOCAB, H);
        }
        #else
        for (int b = 0; b < B; b++) {
            __half* hs_b = s->d_hs + (size_t)b * H;
            moe_tiled_gemv<<<(VOCAB + WMMA_M - 1) / WMMA_M, WMMA_THREADS, 0, s->st>>>(
                s->d_lm_vocab + (size_t)b * VOCAB, hs_b, s->d_embed, VOCAB, H);
        }
        #endif
    }
    hipStreamSynchronize(s->st);

    // Copy logits for all B tokens
    std::vector<__half> lh(B * VOCAB);
    hipMemcpy(lh.data(), s->d_lm_vocab, (size_t)B * VOCAB * 2, hipMemcpyDeviceToHost);
    for (int b = 0; b < B; b++)
        for (int v = 0; v < VOCAB; v++)
            logits_out[b * (size_t)VOCAB + v] = __half2float(lh[b * (size_t)VOCAB + v]);
}

// ── Reset state (new sequence) ──
// ── Set sentinel for expert caching (no previous expert for any layer) ──
__global__ void init_expert_cache_sentinel(float* prev_rs, int n_layers, int rtr_h) {
    // Set prev_rs[layer * rtr_h + rtr_h - 1] = bit-cast N_EXP (sentinel)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_layers) return;
    ((int*)&prev_rs[i * rtr_h + rtr_h - 1])[0] = 16; // N_EXP = 16 (invalid expert, no previous)
}

void zaya_reset(ZayaState* s) {
    HIP_OK(hipMemsetAsync(s->d_conv,0,(size_t)N_LAYERS*2*QKV*2,s->st));
    HIP_OK(hipMemsetAsync(s->d_phs,0,(size_t)N_LAYERS*H*2,s->st));
    HIP_OK(hipMemsetAsync(s->d_prev_rs,0,(size_t)N_LAYERS*RTR_H*4,s->st));
    init_expert_cache_sentinel<<<1, 64, 0, s->st>>>(s->d_prev_rs, N_LAYERS, RTR_H);
}

// ── Destroy ──
void zaya_destroy(ZayaState* s) {
    if (!s) return;
    hipFree(s->d_hs); hipFree(s->d_ao); hipFree(s->d_tmp); hipFree(s->d_fnw);
    hipFree(s->d_lm_out); hipFree(s->d_embed); hipFree(s->d_conv); hipFree(s->d_phs); 
    hipFree(s->d_prev_rs); hipFree(s->d_expert_idx); hipFree(s->d_expert_wt);
    for (int i = 0; i < N_LAYERS; i++) {
        auto& l = s->lw[i];
        hipFree(l.nw); hipFree(l.wq); hipFree(l.wk); hipFree(l.wv1); hipFree(l.wv2); hipFree(l.wo); hipFree(l.pan);
        hipFree(l.cdw); hipFree(l.cdb); hipFree(l.cgw); hipFree(l.cgb); hipFree(l.ks);
        hipFree(l.pahss); hipFree(l.pahsb); hipFree(l.parss); hipFree(l.parsb);
        hipFree(l.gdw); hipFree(l.gdb); hipFree(l.rfn); hipFree(l.rf1); hipFree(l.rf1b);
        hipFree(l.rf2); hipFree(l.rf2b); hipFree(l.rout); hipFree(l.bb);
        if (l.gu) hipFree(l.gu); if (l.dn) hipFree(l.dn);
        hipFree(l.pmhss); hipFree(l.pmhsb); hipFree(l.pmrss); hipFree(l.pmrsb);
    }
    hipStreamDestroy(s->st);
    hipFree(s->d_lm_vocab); hipFree(s->d_argmax_idx); hipFree(s->d_argmax_val);
    hipFree(s->d_sorted_ids); hipFree(s->d_expert_counts); hipFree(s->d_expert_offsets);
    delete s;
}

} // extern "C"
