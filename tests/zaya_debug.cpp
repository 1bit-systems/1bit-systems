// zaya_debug.cpp — Dump all layer-0 intermediates to compare with Python reference
// Build: cmake --build . --target zaya_debug -j8
// Run:   ./build/zaya_debug && python3 tools/compare_zaya.py

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

#define HIP_OK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP Error %d at %s:%d\n",_s,__FILE__,__LINE__); abort();}} while(0)

constexpr int H=2048, NQ=8, NKV=2, HD=128, QD=NQ*HD, KD=NKV*HD, QKV=QD+KD;
constexpr int GQA=NQ/NKV, N_LAYERS=40, VOCAB=262272;
constexpr int N_EXP=16, N_EXP_TOTAL=17, N_FF=2048, RTR_H=256;
constexpr float RMD_EPS=1e-5f;
constexpr int BLK=256;

static std::vector<float> load_bin(const std::string& p){
    std::ifstream f(p,std::ios::binary|std::ios::ate);
    if(!f){fprintf(stderr,"Missing: %s\n",p.c_str());exit(1);}
    size_t n=f.tellg()/sizeof(float);f.seekg(0);
    std::vector<float> d(n);f.read((char*)d.data(),n*sizeof(float));return d;
}
static std::string L(int i){return std::to_string(i);}
#define W(N) load_bin(std::string("/tmp/zaya_weights/")+N)

static void upf16(const std::vector<float>& s,__half*d,int n,hipStream_t h=0){
    std::vector<__half>b(n);for(int i=0;i<n;i++)b[i]=__float2half(s[i]);
    HIP_OK(hipMemcpyAsync(d,b.data(),n*2,hipMemcpyHostToDevice,h));
}
static void upf32(const std::vector<float>& s,float*d,int n,hipStream_t h=0){
    HIP_OK(hipMemcpyAsync(d,s.data(),n*4,hipMemcpyHostToDevice,h));
}
static void dump(const std::string& name, const __half* d, int n, hipStream_t st){
    std::vector<__half> buf(n);
    HIP_OK(hipMemcpy(buf.data(),d,n*2,hipMemcpyDeviceToHost));
    std::vector<float> fbuf(n);
    for(int i=0;i<n;i++) fbuf[i]=__half2float(buf[i]);
    FILE* fp=fopen((std::string("/tmp/cpp_")+name+".bin").c_str(),"wb");
    if(fp){fwrite(fbuf.data(),4,n,fp);fclose(fp);}
    float norm=0; for(auto v:fbuf) norm+=v*v;
    printf("  dump %s: norm=%.2f [0]=%.4f\n",name.c_str(),sqrtf(norm),fbuf[0]);
}
static void dump_f32(const std::string& name, const float* d, int n){
    FILE* fp=fopen((std::string("/tmp/cpp_")+name+".bin").c_str(),"wb");
    if(fp){fwrite(d,4,n,fp);fclose(fp);}
    float norm=0; for(int i=0;i<n;i++) norm+=d[i]*d[i];
    printf("  dump %s: norm=%.2f [0]=%.4f\n",name.c_str(),sqrtf(norm),d[0]);
}

__global__ void rmsnorm_k(__half* x, const __half* w, int n) {
    __shared__ float red[32]; int tx=threadIdx.x, wid=tx/32, l=tx%32;
    float ss=0; for(int i=tx;i<n;i+=blockDim.x) ss+=(float)x[i]*(float)x[i];
    for(int o=16;o>0;o>>=1) ss+=__shfl_xor(ss,o);
    if(l==0) red[wid]=ss; __syncthreads();
    if(wid==0){ss=(l<(256/32))?red[l]:0; for(int o=16;o>0;o>>=1) ss+=__shfl_xor(ss,o);if(l==0) red[0]=ss;} __syncthreads();
    float r=1.0f/sqrtf(red[0]/n+RMD_EPS);
    for(int i=tx;i<n;i+=blockDim.x) x[i]=__float2half((float)x[i]*r*(float)w[i]);
}
__global__ void copy_k(__half* dst, const __half* src, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; dst[i]=src[i];
}
__global__ void residual_scale_k(__half* out, const __half* res,
    const float* hs_s, const float* hs_b, const float* res_s, const float* res_b, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float o=(float)out[i]*hs_s[i]+hs_b[i];
    float r=(float)res[i]*res_s[i]+res_b[i];
    out[i]=__float2half(o+r);
}
__global__ void mm_k(__half* out, const __half* in, const __half* wt, int M, int K) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=M) return;
    float s=0; for(int k=0;k<K;k++) s+=(float)in[k]*(float)wt[k*(size_t)M+i];
    out[i]=__float2half(s);
}
__global__ void silu_mul_k(__half* out, const __half* gate, const __half* up, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float g=(float)gate[i]; out[i]=__float2half((g/(1.0f+expf(-g)))*(float)up[i]);
}
__global__ void cca_attn_kernel(
    const __half*hs,const __half*phs,const __half*csi,int pos,
    const __half*wq,const __half*wk,const __half*wv1,const __half*wv2,const __half*wo,
    const float*cdw,const float*cdb,const float*cgw,const float*cgb,
    const float*ks,const __half*nw,
    __half*ao,__half*ncs,__half*nph);

struct LayerW {
    __half *nw,*wq,*wk,*wv1,*wv2,*wo;
    float *cdw,*cdb,*cgw,*cgb,*ks;
    float *pahss,*pahsb,*parss,*parsb;
    __half *pan;
    float *gdw,*gdb,*rfn,*rf1,*rf1b,*rf2,*rf2b,*rout,*bb;
    __half *gu,*dn;
    float *pmhss,*pmhsb,*pmrss,*pmrsb;
};

int main() {
    printf("=== Zaya Debug: dump intermediates ===\n");
    auto embed=W("model_embed_tokens_weight.bin");
    auto iscale=W("model_input_hidden_states_scale.bin");
    auto ibias=W("model_input_hidden_states_bias.bin");
    
    __half *d_hs,*d_ao,*d_tmp,*d_conv,*d_prev_hs;
    HIP_OK(hipMalloc(&d_hs,H*2)); HIP_OK(hipMalloc(&d_ao,H*2));
    HIP_OK(hipMalloc(&d_tmp,H*2));
    HIP_OK(hipMalloc(&d_conv,2*QKV*2)); HIP_OK(hipMalloc(&d_prev_hs,H*2));
    HIP_OK(hipMemsetAsync(d_conv,0,2*QKV*2,0));
    HIP_OK(hipMemsetAsync(d_prev_hs,0,H*2,0));
    
    hipStream_t st; HIP_OK(hipStreamCreate(&st));
    
    // Load token 100, apply scale/bias
    std::vector<__half> h_tok(H);
    for(int i=0;i<H;i++) h_tok[i]=__float2half((embed[100*H+i]+ibias[i])*iscale[i]);
    HIP_OK(hipMemcpyAsync(d_hs,h_tok.data(),H*2,hipMemcpyHostToDevice,st));
    HIP_OK(hipStreamSynchronize(st));
    dump("input", d_hs, H, st);
    
    // Load layer 0 weights
    auto A=[&](auto&p,int n){HIP_OK(hipMalloc(&p,n*2));};
    auto B=[&](auto&p,int n){HIP_OK(hipMalloc(&p,n*4));};
    LayerW l;
    A(l.nw,H);  upf16(W("model_layers_0_input_layernorm_weight.bin"),l.nw,H,st);
    A(l.wq,QD*H); upf16(W("model_layers_0_self_attn_qkv_proj_q_proj_weight.bin"),l.wq,QD*H,st);
    A(l.wk,KD*H); upf16(W("model_layers_0_self_attn_qkv_proj_k_proj_weight.bin"),l.wk,KD*H,st);
    A(l.wv1,(KD/2)*H); upf16(W("model_layers_0_self_attn_qkv_proj_v_proj_current_weight.bin"),l.wv1,(KD/2)*H,st);
    A(l.wv2,(KD/2)*H); upf16(W("model_layers_0_self_attn_qkv_proj_v_proj_delayed_weight.bin"),l.wv2,(KD/2)*H,st);
    A(l.wo,H*QD); upf16(W("model_layers_0_self_attn_o_proj_weight.bin"),l.wo,H*QD,st);
    B(l.cdw,QKV*2); upf32(W("model_layers_0_self_attn_qkv_proj_conv_qk_depthwise_weight.bin"),l.cdw,QKV*2,st);
    B(l.cdb,QKV);   upf32(W("model_layers_0_self_attn_qkv_proj_conv_qk_depthwise_bias.bin"),l.cdb,QKV,st);
    B(l.cgw,QKV*128*2); upf32(W("model_layers_0_self_attn_qkv_proj_conv_qk_grouped_weight.bin"),l.cgw,QKV*128*2,st);
    B(l.cgb,QKV);   upf32(W("model_layers_0_self_attn_qkv_proj_conv_qk_grouped_bias.bin"),l.cgb,QKV,st);
    B(l.ks,NKV);    upf32(W("model_layers_0_self_attn_qk_norm_temp.bin"),l.ks,NKV,st);
    B(l.pahss,H); upf32(W("model_layers_0_post_attention_residual_scale_hidden_states_scale.bin"),l.pahss,H,st);
    B(l.pahsb,H); upf32(W("model_layers_0_post_attention_residual_scale_hidden_states_bias.bin"),l.pahsb,H,st);
    B(l.parss,H); upf32(W("model_layers_0_post_attention_residual_scale_residual_scale.bin"),l.parss,H,st);
    B(l.parsb,H); upf32(W("model_layers_0_post_attention_residual_scale_residual_bias.bin"),l.parsb,H,st);
    A(l.pan,H); upf16(W("model_layers_0_post_attention_layernorm_weight.bin"),l.pan,H,st);
    // EDA Router
    B(l.gdw,H*RTR_H); upf32(W("model_layers_0_mlp_gate_down_proj_weight.bin"),l.gdw,H*RTR_H,st);
    B(l.gdb,RTR_H);   upf32(W("model_layers_0_mlp_gate_down_proj_bias.bin"),l.gdb,RTR_H,st);
    B(l.rfn,RTR_H);  upf32(W("model_layers_0_mlp_gate_router_mlp_norm_weight.bin"),l.rfn,RTR_H,st);
    B(l.rf1,RTR_H*RTR_H); upf32(W("model_layers_0_mlp_gate_router_mlp_fc1_weight.bin"),l.rf1,RTR_H*RTR_H,st);
    B(l.rf1b,RTR_H); upf32(W("model_layers_0_mlp_gate_router_mlp_fc1_bias.bin"),l.rf1b,RTR_H,st);
    B(l.rf2,RTR_H*RTR_H); upf32(W("model_layers_0_mlp_gate_router_mlp_fc2_weight.bin"),l.rf2,RTR_H*RTR_H,st);
    B(l.rf2b,RTR_H); upf32(W("model_layers_0_mlp_gate_router_mlp_fc2_bias.bin"),l.rf2b,RTR_H,st);
    B(l.rout,N_EXP_TOTAL*RTR_H); upf32(W("model_layers_0_mlp_gate_router_mlp_out_proj_weight.bin"),l.rout,N_EXP_TOTAL*RTR_H,st);
    B(l.bb,N_EXP_TOTAL); upf32(W("model_layers_0_mlp_gate_balancing_biases.bin"),l.bb,N_EXP_TOTAL,st);
    // Skip giant MoE allocations for now (debugging router only)
    l.gu=nullptr; l.dn=nullptr;
    l.pmhss=nullptr; l.pmhsb=nullptr; l.pmrss=nullptr; l.pmrsb=nullptr;
    HIP_OK(hipStreamSynchronize(st));
    printf("Weights loaded.\n");

    int g1 = (H+BLK-1)/BLK;

    // ── A) CCA Attention ──
    cca_attn_kernel<<<1,128,0,st>>>(d_hs,d_prev_hs,d_conv,0,
        l.wq,l.wk,l.wv1,l.wv2,l.wo,
        l.cdw,l.cdb,l.cgw,l.cgb,l.ks,l.nw,
        d_ao,d_conv,d_prev_hs);
    HIP_OK(hipStreamSynchronize(st));
    dump("attn_out", d_ao, H, st);

    // ── B) Post-attention residual scale ──
    residual_scale_k<<<g1,BLK,0,st>>>(d_ao,d_hs,l.pahss,l.pahsb,l.parss,l.parsb,H);
    copy_k<<<g1,BLK,0,st>>>(d_hs,d_ao,H);
    HIP_OK(hipStreamSynchronize(st));
    dump("post_attn_res", d_hs, H, st);

    // ── C) Post-attention RMSNorm ──
    rmsnorm_k<<<1,BLK,0,st>>>(d_hs,l.pan,H);
    HIP_OK(hipStreamSynchronize(st));
    dump("post_attn_norm", d_hs, H, st);

    // ── D) EDA Router (CPU) ──
    std::vector<__half> h_hs(H);
    HIP_OK(hipMemcpy(h_hs.data(),d_hs,H*2,hipMemcpyDeviceToHost));
    std::vector<float> hs_f32(H);
    for(int i=0;i<H;i++) hs_f32[i]=__half2float(h_hs[i]);

    // Load router host weights
    auto hw_gdw = W("model_layers_0_mlp_gate_down_proj_weight.bin");
    auto hw_gdb = W("model_layers_0_mlp_gate_down_proj_bias.bin");
    auto hw_rfn = W("model_layers_0_mlp_gate_router_mlp_norm_weight.bin");
    auto hw_rf1 = W("model_layers_0_mlp_gate_router_mlp_fc1_weight.bin");
    auto hw_rf1b= W("model_layers_0_mlp_gate_router_mlp_fc1_bias.bin");
    auto hw_rf2 = W("model_layers_0_mlp_gate_router_mlp_fc2_weight.bin");
    auto hw_rf2b= W("model_layers_0_mlp_gate_router_mlp_fc2_bias.bin");
    auto hw_rout= W("model_layers_0_mlp_gate_router_mlp_out_proj_weight.bin");
    auto hw_bb  = W("model_layers_0_mlp_gate_balancing_biases.bin");

    printf("Router weights loaded.\n");

    float rs[RTR_H], rs2[RTR_H];
    
    // 1. gate_down
    for(int i=0;i<RTR_H;i++){
        float s=hw_gdb[i];
        for(int j=0;j<H;j++) s+=hs_f32[j]*hw_gdw[i*(size_t)H+j];
        rs[i]=s;
    }
    dump_f32("router_gate_down", rs, RTR_H);
    
    // 2. RMSNorm
    float ss=0; for(int i=0;i<RTR_H;i++) ss+=rs[i]*rs[i];
    float rinv=1.0f/sqrtf(ss/RTR_H+1e-5f);
    for(int i=0;i<RTR_H;i++) rs[i]=rs[i]*rinv*hw_rfn[i];
    dump_f32("router_norm", rs, RTR_H);
    
    // 3. fc1 + GELU
    for(int i=0;i<RTR_H;i++){
        float s=hw_rf1b[i];
        for(int j=0;j<RTR_H;j++) s+=rs[j]*hw_rf1[i*(size_t)RTR_H+j];
        rs2[i]=s*0.5f*(1.0f+tanhf(0.7978845608f*(s+0.044715f*s*s*s))); // tanh GELU
    }
    // Use tanh-based GELU which is more accurate
    dump_f32("router_fc1_gelu", rs2, RTR_H);
    
    // 4. fc2 + GELU
    for(int i=0;i<RTR_H;i++){
        float s=hw_rf2b[i];
        for(int j=0;j<RTR_H;j++) s+=rs2[j]*hw_rf2[i*(size_t)RTR_H+j];
        rs[i]=s*0.5f*(1.0f+tanhf(0.7978845608f*(s+0.044715f*s*s*s))); // tanh GELU
    }
    dump_f32("router_fc2_gelu", rs, RTR_H);
    
    // 5. out_proj → scores + softmax
    float scores[N_EXP_TOTAL];
    for(int i=0;i<N_EXP_TOTAL;i++){
        float s=0;
        for(int j=0;j<RTR_H;j++) s+=rs[j]*hw_rout[i*(size_t)RTR_H+j];
        scores[i]=s+hw_bb[i];
    }
    dump_f32("router_logits", scores, N_EXP_TOTAL);
    
    // Softmax
    float maxv=scores[0]; for(int i=1;i<N_EXP_TOTAL;i++) if(scores[i]>maxv) maxv=scores[i];
    float sumv=0; for(int i=0;i<N_EXP_TOTAL;i++) sumv+=expf(scores[i]-maxv);
    float inv_sum=1.0f/(sumv+1e-10f);
    float probs[N_EXP_TOTAL];
    for(int i=0;i<N_EXP_TOTAL;i++) probs[i]=expf(scores[i]-maxv)*inv_sum;
    dump_f32("router_probs", probs, N_EXP_TOTAL);
    
    int best=0; float bestv=probs[0];
    for(int i=1;i<N_EXP_TOTAL;i++) if(probs[i]>bestv){bestv=probs[i];best=i;}
    printf("  Router: expert=%d prob=%.4f (N_EXP=%d N_EXP_TOTAL=%d)\n", best, bestv, N_EXP, N_EXP_TOTAL);

    // MoE/Post-MLP skipped (debug router only)
    // Final = post-attn residual (same as post_attn_res since MoE passthrough)
    printf("  (MoE+residual_scale skipped - debug router only)\n");
    printf("\nCompare intermediates:\n  python3 -c \"\nimport numpy as np\nfor name in [%s]:\n  c=np.fromfile('/tmp/cpp_'+name+'.bin',dtype=np.float32)\n  p=np.fromfile('/tmp/py_'+name+'.bin',dtype=np.float32)\n  cos=np.dot(c,p)/(np.linalg.norm(c)*np.linalg.norm(p)+1e-12)\n  print(f'{name}: cpp_norm={np.linalg.norm(c):.2f} py_norm={np.linalg.norm(p):.2f} cos={cos:.4f}')\n\"\n","'attn_out','post_attn_res','post_attn_norm','router_gate_down','router_norm','router_logits','router_probs'");

    printf("\nDone. Compare with:\n  python3 -c \"\nimport numpy as np\nfor name in ['attn_out','post_attn_res','post_attn_norm','router_gate_down','router_norm','router_fc1_gelu','router_fc2_gelu','router_logits','router_probs','moe_gate_up','moe_silu_mul','moe_down','final']:\n  c=np.fromfile('/tmp/cpp_'+name+'.bin',dtype=np.float32)\n  p=np.fromfile('/tmp/py_'+name+'.bin',dtype=np.float32) if name in ['attn_out','post_attn_res','post_attn_norm','moe_out','final'] else None\n  print(f'{name}: cpp_norm={np.linalg.norm(c):.2f}',end='')\n  if p is not None: print(f' py_norm={np.linalg.norm(p):.2f} cos={np.dot(c,p)/(np.linalg.norm(c)*np.linalg.norm(p)+1e-12):.4f}',end='')\n  print()\n\"\n");
    
    hipFree(d_hs); hipFree(d_ao); hipFree(d_tmp); hipFree(d_conv); hipFree(d_prev_hs);
    for(auto p : {l.nw,l.wq,l.wk,l.wv1,l.wv2,l.wo,l.pan}) if(p) hipFree(p);
    for(auto p : {l.cdw,l.cdb,l.cgw,l.cgb,l.ks,l.pahss,l.pahsb,l.parss,l.parsb,
                  l.gdw,l.gdb,l.rfn,l.rf1,l.rf1b,l.rf2,l.rf2b,l.rout,l.bb}) if(p) hipFree(p);
    hipStreamDestroy(st);
    return 0;
}
