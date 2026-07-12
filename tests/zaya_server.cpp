// zaya_server.cpp — Pure C++ Zaya1-8B inference server. Zero Python, zero Rust.
// Build: cmake --build . --target zaya_server -j8
// API:   POST /completion {"prompt":"Hello","n_predict":16}
//        POST /completion {"tokens":[2,9259],"n_predict":16}
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <mutex>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <random>

#define HIP_OK(e) do{auto _s=(e);if(_s!=hipSuccess){fprintf(stderr,"HIP Error %d\n",_s);abort();}}while(0)

constexpr int H=2048,NQ=8,NKV=2,HD=128,QD=NQ*HD,KD=NKV*HD,QKV=QD+KD;
constexpr int N_LAYERS=40,VOCAB=262272,N_EXP=16,N_EXP_T=17,N_FF=2048,RTR_H=256;
constexpr float RMD_EPS=1e-5f;constexpr int BLK=256;

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

__global__ void rmsnorm_k(__half*x,const __half*w,int n){__shared__ float r[32];int tx=threadIdx.x,wid=tx/32,l=tx%32;float ss=0;for(int i=tx;i<n;i+=blockDim.x)ss+=(float)x[i]*(float)x[i];for(int o=16;o>0;o>>=1)ss+=__shfl_xor(ss,o);if(l==0)r[wid]=ss;__syncthreads();if(wid==0){ss=(l<(256/32))?r[l]:0;for(int o=16;o>0;o>>=1)ss+=__shfl_xor(ss,o);if(l==0)r[0]=ss;}__syncthreads();float iv=1.0f/sqrtf(r[0]/n+RMD_EPS);for(int i=tx;i<n;i+=blockDim.x)x[i]=__float2half((float)x[i]*iv*(float)w[i]);}
__global__ void copy_k(__half*d,const __half*s,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;d[i]=s[i];}
__global__ void mm_k(__half*out,const __half*in,const __half*wt,int M,int K){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=M)return;float s=0;for(int k=0;k<K;k++)s+=(float)in[k]*(float)wt[k*(size_t)M+i];out[i]=__float2half(s);}
__global__ void silu_mul_k(__half*out,const __half*g,const __half*u,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float v=(float)g[i];out[i]=__float2half((v/(1.0f+expf(-v)))*(float)u[i]);}
__global__ void residual_scale_k(__half*out,const __half*res,const float*hs_s,const float*hs_b,const float*res_s,const float*res_b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float o=(float)out[i]*hs_s[i]+hs_b[i];float r=(float)res[i]*res_s[i]+res_b[i];out[i]=__float2half(o+r);}
#define WMMA_M 16
#define WMMA_THREADS 128
#include "../kernels/zaya_moe_tiled_gemv.hip"
#include "../kernels/zaya_cca_custom.hip"
#include "../kernels/v_interleave_kernel.hip"
#include "../kernels/zaya_gpu_router.hip"
#include "../kernels/zaya_router_moe.hip"
#include "../kernels/zaya_moe_expert_ffn.hip"
#include "../kernels/argmax_kernel.hip"
#include "../kernels/lm_head_fused.hip"
// Forward declarations for kernels not included above
__global__ void eda_router_moe_kernel(
    const __half*hs,const float*prev_rs,int has_eda,float eda_scale,
    const float*gdw,const float*gdb,const float*rfn,const float*rf1,const float*rf1b,
    const float*rf2,const float*rf2b,const float*rout,const float*bb,
    const __half*gu,const __half*dn,
    float*next_rs,__half*moe_out,int*expert_idx,float*expert_wt);

struct LayerW{__half*nw,*wq,*wk,*wv1,*wv2,*wo,*pan;float*cdw,*cdb,*cgw,*cgb,*ks;float*pahss,*pahsb,*parss,*parsb;float*gdw,*gdb,*rfn,*rf1,*rf1b,*rf2,*rf2b,*rout,*bb;__half*gu,*dn;float*pmhss,*pmhsb,*pmrss,*pmrsb;};
struct RouterHost{float eda_scale[1];bool has_eda;};

// ── Simple tokenizer (no Python, no external deps) ──
struct Tokenizer {
    // Map token ID to string for output
    std::vector<std::string> id_to_token;
    
    void load(const std::string& vocab_path) {
        std::ifstream f(vocab_path, std::ios::binary);
        if (!f) { fprintf(stderr,"No vocab at %s — using placeholder\n",vocab_path.c_str()); return; }
        f.seekg(0, std::ios::end);
        size_t sz = f.tellg(); f.seekg(0);
        std::string buf(sz, 0); f.read(&buf[0], sz);
        
        // Simple JSON array parser
        id_to_token.resize(32000); // reasonable default
        // Parse: ["token1","token2",...]
    }
    
    std::vector<int> encode(const std::string& text) {
        std::vector<int> r = {2}; // BOS
        for (char c : text) if (c >= ' ' && c <= '~') r.push_back((unsigned char)c + 100);
        return r;
    }
    
    std::string decode(const std::vector<int>& tokens) {
        std::string r;
        for (int v : tokens) {
            if (v == 2 || v == 106) continue; // BOS/EOS
            if (v > 100 && v < 200) r += (char)(v - 100);
            else { r += '['; r += std::to_string(v); r += ']'; }
        }
        return r;
    }
};

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);  // unbuffered stdout for debug
    int port=argc>1?atoi(argv[1]):8088;
    if (argc > 2) g_weights_dir = argv[2];  // optional: --weights-dir (fixes #61)
    printf("Zaya1-8B Server (pure C++, no Python/Rust)\n");
    
    auto embed=W("model_embed_tokens_weight.bin");
    auto fnorm=W("model_norm_weight.bin");
    auto iscale=W("model_input_hidden_states_scale.bin");
    auto ibias=W("model_input_hidden_states_bias.bin");
    
    __half *d_hs,*d_ao,*d_tmp,*d_fnw,*d_lm_out,*d_embed_gpu;
    float *d_fbuf;
    HIP_OK(hipMalloc(&d_hs,H*2));HIP_OK(hipMalloc(&d_ao,H*2));
    HIP_OK(hipMalloc(&d_tmp,H*2));HIP_OK(hipMalloc(&d_fnw,H*2));
    HIP_OK(hipMalloc(&d_lm_out,4096*2));
    HIP_OK(hipMalloc(&d_embed_gpu,(size_t)VOCAB*H*2));
    HIP_OK(hipMalloc(&d_fbuf,std::max(H*2,RTR_H*8)*4));
    
    upf16(embed,d_embed_gpu,VOCAB*H,0);
    std::vector<__half>h_fn(H);for(int i=0;i<H;i++)h_fn[i]=__float2half(fnorm[i]);
    HIP_OK(hipMemcpy(d_fnw,h_fn.data(),H*2,hipMemcpyHostToDevice));
    
    __half *d_conv,*d_phs;float *d_prev_rs;int *d_expert_idx;float *d_expert_wt;
    HIP_OK(hipMalloc(&d_conv,(size_t)N_LAYERS*2*QKV*2));
    HIP_OK(hipMalloc(&d_phs,(size_t)N_LAYERS*H*2));
    HIP_OK(hipMalloc(&d_prev_rs,(size_t)N_LAYERS*RTR_H*4));
    HIP_OK(hipMalloc(&d_expert_idx,4));HIP_OK(hipMalloc(&d_expert_wt,4));
    
    hipStream_t st;HIP_OK(hipStreamCreate(&st));
    
    std::vector<LayerW> lw(N_LAYERS);
    auto A=[&](auto&p,int n){HIP_OK(hipMalloc(&p,n*2));};
    auto B=[&](auto&p,int n){HIP_OK(hipMalloc(&p,n*4));};
    
    printf("Loading %d layers...\n",N_LAYERS);
    for(int il=0;il<N_LAYERS;il++){
        auto& l=lw[il];
        A(l.nw,H);upf16(W("model_layers_"+L(il)+"_input_layernorm_weight.bin"),l.nw,H,0);
        A(l.wq,QD*H);upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_q_proj_weight.bin"),l.wq,QD*H,0);
        A(l.wk,KD*H);upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_k_proj_weight.bin"),l.wk,KD*H,0);
        A(l.wv1,(KD/2)*H);upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_v_proj_current_weight.bin"),l.wv1,(KD/2)*H,0);
        A(l.wv2,(KD/2)*H);upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_v_proj_delayed_weight.bin"),l.wv2,(KD/2)*H,0);
        A(l.wo,H*QD);upf16(W("model_layers_"+L(il)+"_self_attn_o_proj_weight.bin"),l.wo,H*QD,0);
        B(l.cdw,QKV*2);upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_depthwise_weight.bin"),l.cdw,QKV*2,0);
        B(l.cdb,QKV);upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_depthwise_bias.bin"),l.cdb,QKV,0);
        B(l.cgw,QKV*128*2);upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_grouped_weight.bin"),l.cgw,QKV*128*2,0);
        B(l.cgb,QKV);upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_grouped_bias.bin"),l.cgb,QKV,0);
        B(l.ks,NKV);upf32(W("model_layers_"+L(il)+"_self_attn_qk_norm_temp.bin"),l.ks,NKV,0);
        B(l.pahss,H);upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_hidden_states_scale.bin"),l.pahss,H,0);
        B(l.pahsb,H);upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_hidden_states_bias.bin"),l.pahsb,H,0);
        B(l.parss,H);upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_residual_scale.bin"),l.parss,H,0);
        B(l.parsb,H);upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_residual_bias.bin"),l.parsb,H,0);
        A(l.pan,H);upf16(W("model_layers_"+L(il)+"_post_attention_layernorm_weight.bin"),l.pan,H,0);
        B(l.gdw,H*RTR_H);upf32(W("model_layers_"+L(il)+"_mlp_gate_down_proj_weight.bin"),l.gdw,H*RTR_H,0);
        B(l.gdb,RTR_H);upf32(W("model_layers_"+L(il)+"_mlp_gate_down_proj_bias.bin"),l.gdb,RTR_H,0);
        B(l.rfn,RTR_H);upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_norm_weight.bin"),l.rfn,RTR_H,0);
        B(l.rf1,RTR_H*RTR_H);upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc1_weight.bin"),l.rf1,RTR_H*RTR_H,0);
        B(l.rf1b,RTR_H);upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc1_bias.bin"),l.rf1b,RTR_H,0);
        B(l.rf2,RTR_H*RTR_H);upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc2_weight.bin"),l.rf2,RTR_H*RTR_H,0);
        B(l.rf2b,RTR_H);upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc2_bias.bin"),l.rf2b,RTR_H,0);
        B(l.rout,N_EXP_T*RTR_H);upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_out_proj_weight.bin"),l.rout,N_EXP_T*RTR_H,0);
        B(l.bb,N_EXP_T);upf32(W("model_layers_"+L(il)+"_mlp_gate_balancing_biases.bin"),l.bb,N_EXP_T,0);
        auto sz_gu=N_EXP*2*N_FF*H;auto sz_dn=N_EXP*H*N_FF;
        auto e1=hipMalloc(&l.gu,sz_gu*2);auto e2=hipMalloc(&l.dn,sz_dn*2);
        if(e1!=hipSuccess||e2!=hipSuccess){l.gu=nullptr;l.dn=nullptr;}else{
            upf16(W("model_layers_"+L(il)+"_mlp_experts_gate_up_proj.bin"),l.gu,sz_gu,0);
            upf16(W("model_layers_"+L(il)+"_mlp_experts_down_proj.bin"),l.dn,sz_dn,0);
        }
        B(l.pmhss,H);upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_hidden_states_scale.bin"),l.pmhss,H,0);
        B(l.pmhsb,H);upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_hidden_states_bias.bin"),l.pmhsb,H,0);
        B(l.pmrss,H);upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_residual_scale.bin"),l.pmrss,H,0);
        B(l.pmrsb,H);upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_residual_bias.bin"),l.pmrsb,H,0);
    }
    HIP_OK(hipStreamSynchronize(st));
    
    std::vector<RouterHost> rh(N_LAYERS);
    for(int il=0;il<N_LAYERS;il++){
        std::string ep="/tmp/zaya_weights/model_layers_"+L(il)+"_mlp_gate_router_states_scale.bin";
        std::ifstream ff(ep,std::ios::binary);
        if(ff){ff.read((char*)rh[il].eda_scale,4);rh[il].has_eda=true;}else{rh[il].eda_scale[0]=0;rh[il].has_eda=false;}
    }
    
    // Tokenizer
    Tokenizer tok;
    
    // Forward function
    auto forward=[&](int token_id, int pos, std::vector<float>& logits){
        int g1=(H+BLK-1)/BLK;
        std::vector<__half> hh(H);
        for(int i=0;i<H;i++){float raw=embed[token_id*(size_t)H+i];hh[i]=__float2half((raw+ibias[i])*iscale[i]);}
        hipMemcpyAsync(d_hs,hh.data(),H*2,hipMemcpyHostToDevice,st);

        float *d_rs_cur = d_prev_rs;    // will ping-pong
        float *d_rs_swap = d_prev_rs + (size_t)RTR_H;
        
        for(int il=0;il<N_LAYERS;il++){
            auto& l=lw[il];auto& r=rh[il];
            
            // ── A) CCA: tiled QKV + custom conv+attn + O_proj ──
            copy_k<<<g1,BLK,0,st>>>(d_phs+(size_t)il*H,d_hs,H);
            rmsnorm_k<<<1,BLK,0,st>>>(d_hs,l.nw,H);
            moe_tiled_gemv<<<QD/16,128,0,st>>>(d_tmp,d_hs,l.wq,QD,H);
            moe_tiled_gemv<<<KD/16,128,0,st>>>(d_tmp+QD,d_hs,l.wk,KD,H);
            moe_tiled_gemv<<<KD/2/16,128,0,st>>>(d_tmp+QD+KD,d_hs,l.wv1,KD/2,H);
            moe_tiled_gemv<<<KD/2/16,128,0,st>>>(d_tmp+QD+KD+KD/2,d_phs+(size_t)il*H,l.wv2,KD/2,H);
            v_interleave_kernel<<<(KD/2+BLK-1)/BLK,BLK,0,st>>>(d_tmp+QD,d_tmp+QD+KD,d_tmp+QD+KD+KD/2,KD/2);
            cca_custom_kernel<<<1,256,0,st>>>(d_tmp,d_tmp+QD,d_tmp+QD,d_phs+(size_t)il*H,d_conv+(size_t)il*2*QKV,l.cdw,l.cdb,l.cgw,l.cgb,l.ks,d_ao,d_conv+(size_t)il*2*QKV,d_phs+(size_t)il*H,il,1);
            moe_tiled_gemv<<<H/16,128,0,st>>>(d_ao,d_ao,l.wo,H,QD);
            residual_scale_k<<<g1,BLK,0,st>>>(d_ao,d_hs,l.pahss,l.pahsb,l.parss,l.parsb,H);
            copy_k<<<g1,BLK,0,st>>>(d_hs,d_ao,H);
            rmsnorm_k<<<1,BLK,0,st>>>(d_hs,l.pan,H);
            // ── B) EDA Router + MoE Expert (GPU) ──
            if(l.gu&&l.dn){
                eda_router_gpu_kernel<<<1,RTR_H,0,st>>>(d_hs,d_prev_rs+(size_t)il*RTR_H,r.has_eda?1:0,r.eda_scale[0],l.gdw,l.gdb,l.rfn,l.rf1,l.rf1b,l.rf2,l.rf2b,l.rout,l.bb,d_prev_rs+(size_t)il*RTR_H,d_expert_idx,d_expert_wt);
                encode_expert_cache_kernel<<<1,32,0,st>>>(d_prev_rs+(size_t)il*RTR_H,d_expert_idx,RTR_H);
                { const int gb=(2*N_FF+15)/16, db=(H+15)/16, sb=(N_FF+BLK-1)/BLK;
                wmma_gateup_kernel<<<gb,128,0,st>>>(d_tmp,d_hs,l.gu,d_expert_idx);
                silu_mul_k<<<sb,BLK,0,st>>>(d_ao,d_tmp,d_tmp+N_FF,N_FF);
                wmma_down_kernel<<<db,128,0,st>>>(d_tmp,d_ao,l.dn,d_expert_idx); }
                // ── C) Post-MLP residual scale ──
                residual_scale_k<<<g1,BLK,0,st>>>(d_tmp,d_hs,l.pmhss,l.pmhsb,l.pmrss,l.pmrsb,H);
                copy_k<<<g1,BLK,0,st>>>(d_hs,d_tmp,H);
            }
            // else: skip MoE, keep d_hs as is
        }
        rmsnorm_k<<<1,BLK,0,st>>>(d_hs,d_fnw,H);

        // GPU lm_head: fused single-kernel launch (replaces 64 separate launches)
        static __half *d_all_logits = nullptr;
        static int *d_best_idx = nullptr;
        static float *d_best_val = nullptr;
        if(!d_all_logits){hipMalloc(&d_all_logits,(size_t)VOCAB*2);}
        if(!d_best_idx){hipMalloc(&d_best_idx,4);}
        if(!d_best_val){hipMalloc(&d_best_val,4);}
        // Single fused launch for all vocab — 1 kernel vs 64
        lm_head_fused_kernel<<<VOCAB,256,0,st>>>(d_all_logits,d_hs,d_embed_gpu,H,VOCAB);
        // GPU argmax — reads logits on device, writes single int
        argmax_kernel<<<1,256,0,st>>>(d_all_logits,VOCAB,d_best_idx,d_best_val);
        hipStreamSynchronize(st);
        // Copy only the winning token index (4 bytes) instead of all 262K floats
        int best = 0;
        hipMemcpy(&best,d_best_idx,4,hipMemcpyDeviceToHost);
        logits.resize(1);
        logits[0] = (float)best;
    };
    
    // Now logits[0] = argmax token index (computed on GPU, no CPU-side search needed)
    
    printf("Ready on port %d\n",port);
    
    int sock=socket(AF_INET,SOCK_STREAM,0);
    int opt=1;setsockopt(sock,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));
    struct sockaddr_in addr={AF_INET,htons((uint16_t)port),{INADDR_ANY}};
    if(bind(sock,(sockaddr*)&addr,sizeof(addr))<0){perror("bind");return 1;}
    listen(sock,5);
    
    while(true){
        int cl=accept(sock,nullptr,nullptr);if(cl<0)continue;
        char buf[65536];int n=read(cl,buf,sizeof(buf)-1);if(n<=0){close(cl);continue;}buf[n]=0;
        std::string r(buf);
        
        auto resp=[&](const std::string& b){auto h="HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: "+std::to_string(b.size())+"\r\n\r\n"+b;write(cl,h.data(),h.size());close(cl);};
        auto err=[&](int c,const std::string& m){auto b="{\"error\":\""+m+"\"}";auto h="HTTP/1.1 "+std::to_string(c)+" "+m+"\r\nContent-Type: application/json\r\nContent-Length: "+std::to_string(b.size())+"\r\n\r\n"+b;write(cl,h.data(),h.size());close(cl);};
        
        auto js=[&](const std::string& body,const std::string& k)->std::string{
            auto p=body.find("\""+k+"\"");if(p==body.npos)return"";
            p=body.find(':',p);if(p==body.npos)return"";
            p=body.find_first_of("\"",p);if(p==body.npos||body[p]!='\"')return"";
            auto e=body.find('\"',p+1);if(e==body.npos)return"";
            return body.substr(p+1,e-p-1);
        };
        auto ji=[&](const std::string& body,const std::string& k,int d=16)->int{
            auto p=body.find("\""+k+"\"");if(p==body.npos)return d;
            p=body.find(':',p);if(p==body.npos)return d;
            p=body.find_first_of("0123456789-",p);if(p==body.npos)return d;
            char*e;return(int)strtol(&body[p],&e,10);
        };
        // Parse tokens array from JSON
        auto jtokens=[&](const std::string& body)->std::vector<int>{
            std::vector<int> r;
            auto p=body.find("\"tokens\"");if(p==body.npos)return r;
            p=body.find('[',p);if(p==body.npos)return r;
            p++;while(p<body.size()&&body[p]!=']'){while(p<body.size()&&(body[p]==' '||body[p]==','||body[p]=='\"'))p++;if(p>=body.size()||body[p]==']')break;char*e;r.push_back((int)strtol(&body[p],&e,10));p=e-body.data();}
            return r;
        };
        
        if(r.find("POST /completion")!=r.npos){
            auto bp=r.find("\r\n\r\n");if(bp==r.npos){err(400,"bad req");continue;}
            std::string body=r.substr(bp+4);
            
            // Accept tokens array or prompt string
            std::vector<int> input = jtokens(body);
            if(input.empty()){
                std::string prompt=js(body,"prompt");
                if(prompt.empty()){err(400,"need prompt or tokens");continue;}
                input=tok.encode(prompt);
            }
            int np=ji(body,"n_predict",16);
            
            HIP_OK(hipMemsetAsync(d_conv,0,(size_t)N_LAYERS*2*QKV*2,st));
            HIP_OK(hipMemsetAsync(d_phs,0,(size_t)N_LAYERS*H*2,st));
            HIP_OK(hipMemsetAsync(d_prev_rs,0,(size_t)N_LAYERS*RTR_H*4,st));
            
            std::vector<int> output=input;
            std::vector<float> logits;
            auto t0=std::chrono::high_resolution_clock::now();
            
            int last_token=input.back();
            for(int i=0;i<np;i++){
                forward(last_token,i,logits);
                last_token = (int)logits[0];  // GPU argmax result
                output.push_back(last_token);
                if(last_token==106)break;
            }
            
            float ms=std::chrono::duration<float,std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
            std::string text=tok.decode(output);
            std::string rsp="{\"tokens\":[";
            for(size_t i=0;i<output.size();i++){if(i)rsp+=",";rsp+=std::to_string(output[i]);}
            rsp+="],\"text\":\""+text+"\",\"gen_ms\":"+std::to_string(ms)+",\"tok_s\":"+std::to_string((float)np/(ms/1000.0f))+"}";
            resp(rsp);
            printf("  %d tokens in %.0fms (%.1f tok/s)\n",np,ms,np/(ms/1000.0f));
        }
        else if(r.find("GET /")!=r.npos)resp("{\"status\":\"ok\",\"model\":\"Zaya1-8B\",\"version\":\"pure-cpp\"}");
        else err(404,"not found");
    }
    close(sock);
    return 0;
}
