// zaya_e2e.cpp — Zaya1-8B complete forward pass: CCA Attention + EDA Router + MoE
// Per-layer: RMSNorm → CCA Attn → res_scale → RMSNorm → EDA Router → MoE → res_scale
// Build: cmake --build . --target zaya_e2e -j8
// Run:   ./build/zaya_e2e [--layers N] [--token T]

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <chrono>
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

// ── Kernels ──
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
__global__ void mm_f32_k(float* out, const float* in, const float* wt, int M, int K) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=M) return;
    float s=0; for(int k=0;k<K;k++) s+=in[k]*wt[k*(size_t)M+i];
    out[i]=s;
}
__global__ void silu_mul_k(__half* out, const __half* gate, const __half* up, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float g=(float)gate[i]; out[i]=__float2half((g/(1.0f+expf(-g)))*(float)up[i]);
}
__global__ void add_k(__half* dst, const __half* a, const __half* b, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    dst[i]=__float2half((float)a[i]+(float)b[i]);
}
__global__ void convert_f16_to_f32_k(float* out, const __half* in, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    out[i]=(float)in[i];
}

// CCA attention kernel
__global__ void cca_attn_kernel(
    const __half*hs,const __half*phs,const __half*csi,int pos,
    const __half*wq,const __half*wk,const __half*wv1,const __half*wv2,const __half*wo,
    const float*cdw,const float*cdb,const float*cgw,const float*cgb,
    const float*ks,const __half*nw,
    __half*ao,__half*ncs,__half*nph);

// ── Full layer weights ──
struct LayerW {
    __half *nw;           // input_layernorm
    __half *wq,*wk,*wv1,*wv2,*wo;
    float *cdw,*cdb,*cgw,*cgb,*ks;
    float *pahss,*pahsb,*parss,*parsb;  // post-attention residual scale
    __half *pan;           // post_attention_layernorm
    float *gdw,*gdb;       // gate_down
    float *eda_scale;      // router_states_scale
    float *rfn;            // router norm weight
    float *rf1,*rf1b;      // router fc1
    float *rf2,*rf2b;      // router fc2
    float *rout,*bb;       // router out_proj + balancing biases
    __half *gu,*dn;        // MoE experts gate_up + down
    float *pmhss,*pmhsb,*pmrss,*pmrsb; // post-MLP residual scale
};

int main(int argc, char** argv) {

    int max_layers = 1, token_id = 100;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--layers")&&i+1<argc) max_layers=atoi(argv[++i]);
        if(!strcmp(argv[i],"--token")&&i+1<argc) token_id=atoi(argv[++i]);
        if(!strcmp(argv[i],"--help")){printf("Usage: %s [--layers N] [--token T]\n",argv[0]);return 0;}
    }
    max_layers = std::min(max_layers, N_LAYERS);
    printf("=== Zaya1-8B End-to-End ===\nLayers: %d, Token: %d\n",max_layers,token_id);

    // Weights
    auto embed=W("model_embed_tokens_weight.bin");
    auto fnorm=W("model_norm_weight.bin");
    auto iscale=W("model_input_hidden_states_scale.bin");
    auto ibias=W("model_input_hidden_states_bias.bin");
    printf("Embed: %.0fM\n",embed.size()/1e6);

    // Device memory
    __half *d_hs,*d_ao,*d_tmp,*d_fnw;
    float *d_fbuf;
    HIP_OK(hipMalloc(&d_hs,H*2)); HIP_OK(hipMalloc(&d_ao,H*2));
    HIP_OK(hipMalloc(&d_tmp,H*2)); HIP_OK(hipMalloc(&d_fnw,H*2));
    HIP_OK(hipMalloc(&d_fbuf,std::max(H*2,RTR_H*8)*4));

    hipStream_t st; HIP_OK(hipStreamCreate(&st));
    
    // Embed
    std::vector<__half> h_tok(H);
    for(int i=0;i<H;i++) h_tok[i]=__float2half((embed[token_id*H+i]+ibias[i])*iscale[i]);
    HIP_OK(hipMemcpyAsync(d_hs,h_tok.data(),H*2,hipMemcpyHostToDevice,st));
    
    // Final norm
    std::vector<__half> h_fn(H);
    for(int i=0;i<H;i++) h_fn[i]=__float2half(fnorm[i]);
    HIP_OK(hipMemcpyAsync(d_fnw,h_fn.data(),H*2,hipMemcpyHostToDevice,st));

    // Load all layer weights
    printf("Loading layers...\n");
    std::vector<LayerW> lw(max_layers);
    auto A=[&](auto&p,int n){HIP_OK(hipMalloc(&p,n*2));};
    auto B=[&](auto&p,int n){HIP_OK(hipMalloc(&p,n*4));};
    
    for(int il=0; il<max_layers; il++){
        auto& l=lw[il];
        A(l.nw,H);  upf16(W("model_layers_"+L(il)+"_input_layernorm_weight.bin"),l.nw,H,st);
        A(l.wq,QD*H); upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_q_proj_weight.bin"),l.wq,QD*H,st);
        A(l.wk,KD*H); upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_k_proj_weight.bin"),l.wk,KD*H,st);
        A(l.wv1,(KD/2)*H); upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_v_proj_current_weight.bin"),l.wv1,(KD/2)*H,st);
        A(l.wv2,(KD/2)*H); upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_v_proj_delayed_weight.bin"),l.wv2,(KD/2)*H,st);
        A(l.wo,H*QD); upf16(W("model_layers_"+L(il)+"_self_attn_o_proj_weight.bin"),l.wo,H*QD,st);
        B(l.cdw,QKV*2); upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_depthwise_weight.bin"),l.cdw,QKV*2,st);
        B(l.cdb,QKV);   upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_depthwise_bias.bin"),l.cdb,QKV,st);
        B(l.cgw,QKV*128*2); upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_grouped_weight.bin"),l.cgw,QKV*128*2,st);
        B(l.cgb,QKV);   upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_grouped_bias.bin"),l.cgb,QKV,st);
        B(l.ks,NKV);    upf32(W("model_layers_"+L(il)+"_self_attn_qk_norm_temp.bin"),l.ks,NKV,st);
        B(l.pahss,H); upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_hidden_states_scale.bin"),l.pahss,H,st);
        B(l.pahsb,H); upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_hidden_states_bias.bin"),l.pahsb,H,st);
        B(l.parss,H); upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_residual_scale.bin"),l.parss,H,st);
        B(l.parsb,H); upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_residual_bias.bin"),l.parsb,H,st);
        A(l.pan,H); upf16(W("model_layers_"+L(il)+"_post_attention_layernorm_weight.bin"),l.pan,H,st);
        // EDA Router
        B(l.gdw,H*RTR_H); upf32(W("model_layers_"+L(il)+"_mlp_gate_down_proj_weight.bin"),l.gdw,H*RTR_H,st);
        B(l.gdb,RTR_H);   upf32(W("model_layers_"+L(il)+"_mlp_gate_down_proj_bias.bin"),l.gdb,RTR_H,st);
        // EDA scale: layer 0 doesn't have it (starts from layer 1)
        B(l.eda_scale,1);
        {
            auto p = "/tmp/zaya_weights/model_layers_"+L(il)+"_mlp_gate_router_states_scale.bin";
            std::ifstream ff(p,std::ios::binary);
            if(ff){ std::vector<float> d(1); ff.read((char*)d.data(),4); upf32(d,l.eda_scale,1,st); }
            else HIP_OK(hipMemsetAsync(l.eda_scale,0,4,st)); // no EDA
        }
        B(l.rfn,RTR_H);  upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_norm_weight.bin"),l.rfn,RTR_H,st);
        B(l.rf1,RTR_H*RTR_H); upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc1_weight.bin"),l.rf1,RTR_H*RTR_H,st);
        B(l.rf1b,RTR_H); upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc1_bias.bin"),l.rf1b,RTR_H,st);
        B(l.rf2,RTR_H*RTR_H); upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc2_weight.bin"),l.rf2,RTR_H*RTR_H,st);
        B(l.rf2b,RTR_H); upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc2_bias.bin"),l.rf2b,RTR_H,st);
        B(l.rout,N_EXP_TOTAL*RTR_H); upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_out_proj_weight.bin"),l.rout,N_EXP_TOTAL*RTR_H,st);
        B(l.bb,N_EXP_TOTAL); upf32(W("model_layers_"+L(il)+"_mlp_gate_balancing_biases.bin"),l.bb,N_EXP_TOTAL,st);
        // MoE experts
        A(l.gu,N_EXP*2*N_FF*H);  upf16(W("model_layers_"+L(il)+"_mlp_experts_gate_up_proj.bin"),l.gu,N_EXP*2*N_FF*H,st);
        A(l.dn,N_EXP*H*N_FF);    upf16(W("model_layers_"+L(il)+"_mlp_experts_down_proj.bin"),l.dn,N_EXP*H*N_FF,st);
        // Post-MLP residual scale
        B(l.pmhss,H); upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_hidden_states_scale.bin"),l.pmhss,H,st);
        B(l.pmhsb,H); upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_hidden_states_bias.bin"),l.pmhsb,H,st);
        B(l.pmrss,H); upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_residual_scale.bin"),l.pmrss,H,st);
        B(l.pmrsb,H); upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_residual_bias.bin"),l.pmrsb,H,st);
    }
    HIP_OK(hipStreamSynchronize(st));
    printf("Loaded.\n");

    // Per-layer state on GPU
    __half *d_conv, *d_prev_hs;
    HIP_OK(hipMalloc(&d_conv, 2*QKV*sizeof(__half)));
    HIP_OK(hipMalloc(&d_prev_hs, H*sizeof(__half)));
    HIP_OK(hipMemsetAsync(d_conv,0,2*QKV*sizeof(__half),st));
    HIP_OK(hipMemsetAsync(d_prev_hs,0,H*sizeof(__half),st));
    std::vector<float> prev_rs(RTR_H,0);
    
    auto t0 = std::chrono::high_resolution_clock::now();
    int g1 = (H+BLK-1)/BLK;
    int g_rtr = (RTR_H+BLK-1)/BLK;

    // ═══ Forward: 40 layers (each does CCA Attn + MoE) ═══
    for(int il=0; il<max_layers; il++){
        auto& l = lw[il];
        
        // ── A) CCA Attention ──
        cca_attn_kernel<<<1,128,0,st>>>(
            d_hs,d_prev_hs,d_conv,il,
            l.wq,l.wk,l.wv1,l.wv2,l.wo,
            l.cdw,l.cdb,l.cgw,l.cgb,l.ks,l.nw,
            d_ao,d_conv,d_prev_hs);
        
        // ── B) Post-attention residual scale ──
        residual_scale_k<<<g1,BLK,0,st>>>(d_ao,d_hs,l.pahss,l.pahsb,l.parss,l.parsb,H);
        copy_k<<<g1,BLK,0,st>>>(d_hs,d_ao,H);
        
        // DEBUG: save intermediates for layer 0
        { 
            int N=H; std::vector<__half> buf(N); HIP_OK(hipMemcpy(buf.data(),d_hs,N*2,hipMemcpyDeviceToHost));
            float nrm=0; for(int i=0;i<N;i++){float v=__half2float(buf[i]);nrm+=v*v;} nrm=sqrtf(nrm);
            printf("  L%d post_attn_res norm=%.2f\n",il,nrm); 
        }
        
        // ── C) Post-attention RMSNorm ──
        rmsnorm_k<<<1,BLK,0,st>>>(d_hs,l.pan,H);
        { int N=H; std::vector<__half> buf(N); HIP_OK(hipMemcpy(buf.data(),d_hs,N*2,hipMemcpyDeviceToHost));
          float nrm=0; for(int i=0;i<N;i++){float v=__half2float(buf[i]);nrm+=v*v;} nrm=sqrtf(nrm);
          printf("  L%d post_attn_norm norm=%.2f\n",il,nrm); }
        
        // ── D) EDA Router (on CPU — router is tiny 256-dim) ──
        HIP_OK(hipStreamSynchronize(st));
        std::vector<__half> h_hs(H);
        HIP_OK(hipMemcpy(h_hs.data(),d_hs,H*2,hipMemcpyDeviceToHost));
        std::vector<float> hs_f32(H);
        for(int i=0;i<H;i++) hs_f32[i]=__half2float(h_hs[i]);
        
        // Load weights from GPU (can't access GPU memory from host)
        // Instead, keep a host-side copy of router weights
        static bool loaded_host_w = false;
        static struct { float gdw[H*RTR_H],gdb[RTR_H],eda_scale[1],rfn[RTR_H],
            rf1[RTR_H*RTR_H],rf1b[RTR_H],rf2[RTR_H*RTR_H],rf2b[RTR_H],
            rout[N_EXP_TOTAL*RTR_H],bb[N_EXP_TOTAL]; } host_w[N_LAYERS];
        
        if(!loaded_host_w){
            for(int j=0;j<max_layers;j++){
                auto w=W("model_layers_"+L(j)+"_mlp_gate_down_proj_weight.bin");
                memcpy(host_w[j].gdw,w.data(),H*RTR_H*4);
                w=W("model_layers_"+L(j)+"_mlp_gate_down_proj_bias.bin");
                memcpy(host_w[j].gdb,w.data(),RTR_H*4);
                {
                    auto p="/tmp/zaya_weights/model_layers_"+L(j)+"_mlp_gate_router_states_scale.bin";
                    std::ifstream ff(p,std::ios::binary);
                    if(ff){ ff.read((char*)host_w[j].eda_scale,4); }
                    else { host_w[j].eda_scale[0]=0; }
                }
                w=W("model_layers_"+L(j)+"_mlp_gate_router_mlp_norm_weight.bin");
                memcpy(host_w[j].rfn,w.data(),RTR_H*4);
                w=W("model_layers_"+L(j)+"_mlp_gate_router_mlp_fc1_weight.bin");
                memcpy(host_w[j].rf1,w.data(),RTR_H*RTR_H*4);
                w=W("model_layers_"+L(j)+"_mlp_gate_router_mlp_fc1_bias.bin");
                memcpy(host_w[j].rf1b,w.data(),RTR_H*4);
                w=W("model_layers_"+L(j)+"_mlp_gate_router_mlp_fc2_weight.bin");
                memcpy(host_w[j].rf2,w.data(),RTR_H*RTR_H*4);
                w=W("model_layers_"+L(j)+"_mlp_gate_router_mlp_fc2_bias.bin");
                memcpy(host_w[j].rf2b,w.data(),RTR_H*4);
                w=W("model_layers_"+L(j)+"_mlp_gate_router_mlp_out_proj_weight.bin");
                memcpy(host_w[j].rout,w.data(),N_EXP_TOTAL*RTR_H*4);
                w=W("model_layers_"+L(j)+"_mlp_gate_balancing_biases.bin");
                memcpy(host_w[j].bb,w.data(),N_EXP_TOTAL*4);
            }
            loaded_host_w=true;
            printf("  Host router weights cached\n");
        }
        
        auto& hw = host_w[il];
        float rs[RTR_H];
        
        // 1. gate_down: rs = hs @ gdw^T + gdb
        for(int i=0;i<RTR_H;i++){
            float s=hw.gdb[i];
            for(int j=0;j<H;j++) s+=hs_f32[j]*hw.gdw[i*(size_t)H+j];
            rs[i]=s;
        }
        // 2. EDA
        for(int i=0;i<RTR_H;i++) rs[i]+=prev_rs[i]*hw.eda_scale[0];
        for(int i=0;i<RTR_H;i++) prev_rs[i]=rs[i]; // save for next layer
        // 3. RMSNorm
        float ss=0; for(int i=0;i<RTR_H;i++) ss+=rs[i]*rs[i];
        float rinv=1.0f/sqrtf(ss/RTR_H+1e-5f);
        for(int i=0;i<RTR_H;i++) rs[i]=rs[i]*rinv*hw.rfn[i];
        // 4. fc1 + GELU
        float rs2[RTR_H];
        for(int i=0;i<RTR_H;i++){
            float s=hw.rf1b[i];
            for(int j=0;j<RTR_H;j++) s+=rs[j]*hw.rf1[i*(size_t)RTR_H+j];
            rs2[i]=s*(1.0f/(1.0f+expf(-1.702f*s))); // GELU approx
        }
        // 5. fc2 + GELU
        for(int i=0;i<RTR_H;i++){
            float s=hw.rf2b[i];
            for(int j=0;j<RTR_H;j++) s+=rs2[j]*hw.rf2[i*(size_t)RTR_H+j];
            rs[i]=s*(1.0f/(1.0f+expf(-1.702f*s))); // GELU approx
        }
        // 6. out_proj → scores + balancing_biases
        float scores[N_EXP_TOTAL];
        for(int i=0;i<N_EXP_TOTAL;i++){
            float s=0;
            for(int j=0;j<RTR_H;j++) s+=rs[j]*hw.rout[i*(size_t)RTR_H+j];
            scores[i]=s+hw.bb[i];
        }
        // 7. Softmax
        float maxv=scores[0]; for(int i=1;i<N_EXP_TOTAL;i++) if(scores[i]>maxv) maxv=scores[i];
        float sumv=0; for(int i=0;i<N_EXP_TOTAL;i++) sumv+=expf(scores[i]-maxv);
        float inv_sum=1.0f/(sumv+1e-10f);
        for(int i=0;i<N_EXP_TOTAL;i++) scores[i]=expf(scores[i]-maxv)*inv_sum;
        // 8. argmax
        int best=0; float bestv=scores[0];
        for(int i=1;i<N_EXP_TOTAL;i++) if(scores[i]>bestv){bestv=scores[i];best=i;}
        
        printf("  L%d EDA: expert=%d (N_EXP=%d) wt=%.4f\n",il,best,N_EXP,bestv); fflush(stdout);
        
        // ── E) MoE Expert FFN ──
        if(best < N_EXP){ // real expert (not MOD skip)
            // Upload hs back to GPU  
            // (already there in d_hs)
            
            // gate_up: gu_result[2*N_FF] = hs[H] @ gu[best, 2*N_FF, H]^T
            mm_k<<<(2*N_FF+BLK-1)/BLK,BLK,0,st>>>(
                d_tmp, d_hs,
                l.gu + (size_t)best*2*N_FF*H,
                2*N_FF, H);
            
            // Split and SiLU-multiply: moe_out[N_FF] = SiLU(gate) * up
            silu_mul_k<<<(N_FF+BLK-1)/BLK,BLK,0,st>>>(d_ao, d_tmp, d_tmp+N_FF, N_FF);
            
            // down: result[H] = moe_out[N_FF] @ dn[best, H, N_FF]^T
            mm_k<<<g1,BLK,0,st>>>(
                d_tmp, d_ao,
                l.dn + (size_t)best*H*N_FF,
                H, N_FF);
        } else {
            // MOD skip: output = hs * prob
            // (simplified — just pass through)
            copy_k<<<g1,BLK,0,st>>>(d_tmp, d_hs, H);
        }
        
        // ── F) Post-MLP residual scale ──
        residual_scale_k<<<g1,BLK,0,st>>>(d_tmp, d_hs, l.pmhss, l.pmhsb, l.pmrss, l.pmrsb, H);
        copy_k<<<g1,BLK,0,st>>>(d_hs, d_tmp, H);
    }

    // ═══ Final RMSNorm ═══
    rmsnorm_k<<<1,BLK,0,st>>>(d_hs, d_fnw, H);
    HIP_OK(hipStreamSynchronize(st));
    
    auto t_end = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float,std::milli>(t_end-t0).count();

    // Read output
    std::vector<__half> h_out(H);
    HIP_OK(hipMemcpy(h_out.data(),d_hs,H*2,hipMemcpyDeviceToHost));
    printf("\n═══ Results ═══\n");
    printf("Forward: %.1f ms total, %.2f ms/layer\n",ms,ms/max_layers);
    printf("Output[0:8]:");
    for(int i=0;i<8;i++) printf(" %.4f",__half2float(h_out[i]));
    printf("\n");

    // Cleanup
    hipFree(d_hs); hipFree(d_ao); hipFree(d_tmp); hipFree(d_fnw); hipFree(d_fbuf);
    hipFree(d_conv); hipFree(d_prev_hs);
    for(auto& l : lw){
        hipFree(l.nw); hipFree(l.wq); hipFree(l.wk); hipFree(l.wv1); hipFree(l.wv2);
        hipFree(l.wo); hipFree(l.cdw); hipFree(l.cdb); hipFree(l.cgw); hipFree(l.cgb);
        hipFree(l.ks); hipFree(l.pahss); hipFree(l.pahsb); hipFree(l.parss); hipFree(l.parsb);
        hipFree(l.pan); hipFree(l.gdw); hipFree(l.gdb); hipFree(l.eda_scale);
        hipFree(l.rfn); hipFree(l.rf1); hipFree(l.rf1b); hipFree(l.rf2); hipFree(l.rf2b);
        hipFree(l.rout); hipFree(l.bb);
        hipFree(l.gu); hipFree(l.dn);
        hipFree(l.pmhss); hipFree(l.pmhsb); hipFree(l.pmrss); hipFree(l.pmrsb);
    }
    hipStreamDestroy(st);
    printf("Done.\n");
    return 0;
}
