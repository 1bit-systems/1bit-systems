#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include "npu_utils/npu_instr_utils.hpp"

// Forward declaration: INT8 NPU instruction generator from gemm_npu_instructions.cpp
void gemm_generate_sequence_i8(
    npu_sequence*           seq,
    uint32_t                M,
    uint32_t                K,
    uint32_t                N,
    uint32_t                weight_offset,
    bool                    add_bias,
    int                     activation,
    uint32_t                bias_offset,
    uint32_t                output_offset
);

extern "C" float* dequant_i8_to_float(const uint8_t*,int,int*,int*);
static inline float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static inline float bf16g(uint16_t v){return(v&0x7F80)==0x7F80?0.0f:bf16f(v);}
static inline uint16_t f2bf(float v){uint32_t b;memcpy(&b,&v,4);return(uint16_t)((b+0x8000)>>16);}
// TODO: Read dimensions from GGUF config instead of hardcoding Qwen3-0.6B defaults.
// H=hidden_size NC=num_layers NH=num_attention_heads NKV=num_key_value_heads HD=head_dim
// IM=intermediate_size NV=vocab_size GQA=num_attention_heads/num_key_value_heads
static constexpr int H=1024,NC=28,NH=16,NKV=8,HD=128,IM=3072,NV=151936,GQA=2;
static constexpr float EPS=1e-6f; static constexpr int XM=128, AW=4, WQH=NH/AW, WKVH=NKV/AW;
static inline void cn(float*x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
// Dynamic per-call activation quantization scale, matching packB()'s amax-based approach for
// weights. The original hardcoded 5.0f/127.0f assumes activations stay within [-5,5] - measured
// post-RMSNorm activations actually range as wide as [-8.24,7.01], so that fixed scale silently
// clips/saturates values past +-5 to +-127, a real (and compounding, since it happens every
// layer) source of error separate from the LM head and weight-transpose bugs.
static inline float dynamic_ascale(const float* x, int n) {
    float amax = 0;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (std::isfinite(a) && a > amax) amax = a; }
    if (amax < 1e-12f) amax = 1.0f;
    return amax / 127.0f;
}
static inline void sm(float*sc,int n){if(n<=0)return;cn(sc,n);float mx=sc[0];for(int i=1;i<n;i++)if(sc[i]>mx)mx=sc[i];double s=0;for(int i=0;i<n;i++){float d=sc[i]-mx;if(d>80)d=80;else if(d<-80)d=-80;sc[i]=expf(d);s+=sc[i];}if(s<=0){float iv=1.0f/n;for(int i=0;i<n;i++)sc[i]=iv;return;}float is=1.0f/(float)s;for(int i=0;i<n;i++)sc[i]*=is;}
static inline void rn_c(float*x,const float*w,int n){cn(x,n);double ss=0;for(int i=0;i<n;i++)if(std::isfinite(x[i]))ss+=(double)x[i]*x[i];float ir=1.0f/sqrtf((float)(ss/n)+EPS);for(int i=0;i<n;i++)x[i]=std::isfinite(x[i])?x[i]*ir*w[i]:0.0f;}
static std::vector<float>rc,rs;static void ri(int hd,float th,int mp){int hd2=hd/2;rc.resize(mp*hd);rs.resize(mp*hd);for(int p=0;p<mp;p++)for(int d=0;d<hd2;d++){float f=1.0f/powf(th,(float)d/hd2),a=p*f;rc[p*hd+d]=cosf(a);rs[p*hd+d]=sinf(a);}}
static inline void ra(float*x,int hd,int p){int hd2=hd/2;for(int d=0;d<hd2;d++){float a=x[d],b=x[d+hd2],c=rc[p*hd+d],s=rs[p*hd+d];x[d]=a*c-b*s;x[d+hd2]=b*c+a*s;}}
static uint64_t jo(const char*js,size_t jl,const char*nm){size_t nl=strlen(nm);const char*p=js,*e=js+jl;while(p<e){auto q=(const char*)memmem(p,e-p,nm,nl);if(!q)return 0;if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){auto o=strstr(q,"\"data_offsets\"");if(o){auto a=strchr(o,'[');if(a)return strtoull(a+1,NULL,10);}}p=q+1;}return 0;}
// Parse tensor shape [rows, cols] from JSON header. Follows the same pattern as jo().
// Returns false if tensor or shape field is not found.
static bool jshape(const char*js,size_t jl,const char*nm,uint64_t&r,uint64_t&c){size_t nl=strlen(nm);const char*p=js,*e=js+jl;while(p<e){auto q=(const char*)memmem(p,e-p,nm,nl);if(!q)return false;if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){auto s=strstr(q,"\"shape\"");if(s){auto a=strchr(s,'[');if(a){r=strtoull(a+1,NULL,10);auto comma=strchr(a,',');if(comma){c=strtoull(comma+1,NULL,10);return true;}}}}p=q+1;}return false;}

struct I8Ctx{const char*name;int MD,KD,ND;std::unique_ptr<xrt::xclbin>xc;std::unique_ptr<xrt::hw_context>hc;std::unique_ptr<xrt::kernel>k;std::vector<uint32_t>ins;std::unique_ptr<xrt::bo>bI,bA,bC,layerB[NC];int8_t*Am;int16_t*Cm;std::vector<float> group_scales[NC];
// Per-group INT8 quantization: K is divided into groups of 32, each with its own scale.
// go() computes the effective Bscale as the average of per-group scales.
bool init(xrt::device&d,const char*xp,const char*ip,int gid_B){FILE*f=fopen(ip,"rb");if(!f)return false;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);xc=std::make_unique<xrt::xclbin>(std::string(xp));d.register_xclbin(*xc);hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));memcpy(bI->map(),ins.data(),ins.size()*4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);bA=std::make_unique<xrt::bo>(d,(size_t)MD*KD,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));bC=std::make_unique<xrt::bo>(d,(size_t)MD*ND*2,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));Am=(int8_t*)bA->map();Cm=(int16_t*)bC->map();for(int l=0;l<NC;l++)layerB[l]=std::make_unique<xrt::bo>(d,(size_t)KD*ND,XRT_BO_FLAGS_HOST_ONLY,k->group_id(gid_B));return true;}
bool init_with_generator(xrt::device&d,const char*xp,int _MD,int _KD,int _ND,int gid_B){MD=_MD;KD=_KD;ND=_ND;npu_sequence seq(device_npu2);gemm_generate_sequence_i8(&seq,MD,MD,ND,0,false,0,0,0);seq.cmds2seq();auto[dp,sz]=seq.dump();ins.assign(dp,dp+sz/sizeof(uint32_t));xc=std::make_unique<xrt::xclbin>(std::string(xp));d.register_xclbin(*xc);hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));memcpy(bI->map(),ins.data(),ins.size()*4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);bA=std::make_unique<xrt::bo>(d,(size_t)MD*KD,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));bC=std::make_unique<xrt::bo>(d,(size_t)MD*ND*2,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));Am=(int8_t*)bA->map();Cm=(int16_t*)bC->map();for(int l=0;l<NC;l++)layerB[l]=std::make_unique<xrt::bo>(d,(size_t)KD*ND,XRT_BO_FLAGS_HOST_ONLY,k->group_id(gid_B));return true;}
void packB(int l,const float*w,int K,int N,float&sout){int num_groups=(K+31)/32;group_scales[l].resize(num_groups);auto*Bm=(int8_t*)layerB[l]->map();for(int g=0;g<num_groups;g++){int g_start=g*32;int g_size=std::min(32,K-g_start);float g_amax=0;for(int j=0;j<N;j++){for(int i=0;i<g_size;i++){float a=fabsf(w[(g_start+i)*N+j]);if(std::isfinite(a)&&a>g_amax)g_amax=a;}}if(g_amax<1e-12f)g_amax=1.0f;group_scales[l][g]=g_amax/127.0f;float g_is=127.0f/g_amax;for(int j=0;j<N;j++){for(int i=0;i<g_size;i++){float v=w[(g_start+i)*N+j];if(!std::isfinite(v))v=0;int x=(int)roundf(v*g_is);if(x>127)x=127;else if(x<-127)x=-127;Bm[(g_start+i)*N+j]=(int8_t)x;}}}layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);float ssum=0;for(int g=0;g<num_groups;g++)ssum+=group_scales[l][g];sout=ssum/num_groups;}
inline void go(int l,const float*A,int am,int ak,float ascale,float Bscale,float*C,int an){float ais=1.0f/ascale;for(int m=0;m<am;m++){for(int k=0;k<ak;k++){float v=A[m*ak+k];if(!std::isfinite(v))v=0;int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;Am[m*KD+k]=(int8_t)q;}if(ak<KD)memset(Am+m*KD+ak,0,(size_t)(KD-ak));}if(am<MD)memset(Am+(size_t)am*KD,0,(size_t)(MD-am)*KD);bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);auto r=(*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA,*layerB[l],*bC);r.wait();bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);float cs=ascale*Bscale;if(!group_scales[l].empty()){float ssum=0;for(float s:group_scales[l])ssum+=s;cs=ascale*(ssum/group_scales[l].size());}for(int m=0;m<am;m++)for(int n=0;n<an;n++){float val=(float)Cm[m*ND+n]*cs;if(!std::isfinite(val))val=0;C[m*an+n]=val;}}
};

// Attention kernel — per-window xclbin, BF16 pre-packed input/output
struct AttnK{
    static constexpr int Q_DW=256,K_DW=2048,V_DW=2048,OUT_DW=256;
    int window; std::unique_ptr<xrt::xclbin>xc; std::unique_ptr<xrt::hw_context>hc; std::unique_ptr<xrt::kernel>k;
    std::vector<uint32_t>ins; std::unique_ptr<xrt::bo>bI,bIn,bOut; int32_t*in_m; bool ready=false;
    bool init(xrt::device&d,int w){window=w;char xp[256],ip[256];snprintf(xp,256,"/home/bcloud/npu-sandbox/npu-infer/build/chess_infer/attn_w%d.xclbin",w);snprintf(ip,256,"/home/bcloud/npu-sandbox/npu-infer/build/chess_infer/attn_w%d.insts",w);FILE*f=fopen(ip,"rb");if(!f)return false;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);xc=std::make_unique<xrt::xclbin>(std::string(xp));d.register_xclbin(*xc);hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");bI=std::make_unique<xrt::bo>(d,sz,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));memcpy(bI->map(),ins.data(),sz);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);bIn=std::make_unique<xrt::bo>(d,(Q_DW+2*K_DW+2*V_DW)*4,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));bOut=std::make_unique<xrt::bo>(d,OUT_DW*4,XRT_BO_FLAGS_HOST_ONLY,k->group_id(4));in_m=(int32_t*)bIn->map();ready=true; return true;}
    // CPU attention fallback for faster low-token counts
    static void cpu_attn(const float*Q,const float*K,const float*V,int n,float*out){
        for(int h=0;h<WQH;h++){
            float scores[32]; float mx=-1e30;
            for(int p=0;p<n;p++){double s=0;for(int d=0;d<HD;d++)s+=Q[h*HD+d]*K[p*NKV*HD+(h/GQA)*HD+d];scores[p]=(float)(s/sqrtf((double)HD));if(scores[p]>mx)mx=scores[p];}
            double sum=0;for(int p=0;p<n;p++){scores[p]=expf(scores[p]-mx);sum+=scores[p];}
            if(sum<=0)sum=1;float is=1.0f/(float)sum;
            for(int d=0;d<HD;d++){float s=0;for(int p=0;p<n;p++)s+=scores[p]*is*V[p*NKV*HD+(h/GQA)*HD+d];out[h*HD+d]=s;}
        }
    }
    // NPU attention: packs Q and K/V into BF16 dwords, submits to NPU
    void run(const float*Q4,const float*K2,const float*V2,int n_tokens,float*out4){
        memset(in_m,0,(Q_DW+2*K_DW+2*V_DW)*4);
        for(int h=0;h<WQH;h++){const float*sq=&Q4[h*HD];int32_t*d=&in_m[h*(HD/2)];for(int dm=0;dm<HD/2;dm++){d[dm]=(f2bf(sq[2*dm])<<16)|f2bf(sq[2*dm+1]);}}
        int kv_dw=WKVH*(HD/2);
        for(int b=0;b<2;b++){int k_off=Q_DW+b*K_DW,v_off=Q_DW+2*K_DW+b*V_DW;
            for(int t=0;t<16;t++){int tt=b*16+t;bool v=(tt<n_tokens);int st=(v?tt:0);
                for(int h=0;h<WKVH;h++){const float*sk=&K2[st*WKVH*HD+h*HD];const float*sv=&V2[st*WKVH*HD+h*HD];
                    int32_t*dk=&in_m[k_off+t*kv_dw+h*(HD/2)],*dv=&in_m[v_off+t*kv_dw+h*(HD/2)];
                    for(int dm=0;dm<HD/2;dm++){dk[dm]=v?((f2bf(sk[2*dm])<<16)|f2bf(sk[2*dm+1])):0;dv[dm]=v?((f2bf(sv[2*dm])<<16)|f2bf(sv[2*dm+1])):0;}
                }
            }
        }
        bIn->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r=(*k)((unsigned)3,*bI,(unsigned)ins.size(),*bIn,*bOut);r.wait();bOut->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        auto*od=(uint32_t*)bOut->map();
        for(int h=0;h<WQH;h++)for(int dm=0;dm<HD/2;dm++){uint32_t p=od[h*(HD/2)+dm];out4[h*HD+2*dm]=bf16f(p>>16);out4[h*HD+2*dm+1]=bf16f(p&0xFFFF);}
        for(int h=0;h<WQH;h++)for(int d=0;d<HD;d++)if(!std::isfinite(out4[h*HD+d]))out4[h*HD+d]=0.0f;
    }
};

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    printf("=== NPU Engine i8 + Attention ===\n\n");
    const char*mp="/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    int fd=open(mp,O_RDONLY);struct stat st;fstat(fd,&st);
    uint8_t*md=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    uint64_t hsz;memcpy(&hsz,md,8);uint64_t df=8+hsz;
    auto i8p=[&](uint64_t o){return md+df+o;};auto emb=(const uint16_t*)(md+df);
    const char*js=(const char*)(md+8);size_t jl=hsz;
    struct LO{uint64_t qp,kp,vp,op,gp,up,dp,in_off,pa_off,qn_off,kn_off;}lo[NC];char b[128];
    for(int l=0;l<NC;l++){snprintf(b,128,"model.layers.%d.self_attn.q_proj.weight",l);lo[l].qp=jo(js,jl,b);snprintf(b,128,"model.layers.%d.self_attn.k_proj.weight",l);lo[l].kp=jo(js,jl,b);snprintf(b,128,"model.layers.%d.self_attn.v_proj.weight",l);lo[l].vp=jo(js,jl,b);snprintf(b,128,"model.layers.%d.self_attn.o_proj.weight",l);lo[l].op=jo(js,jl,b);snprintf(b,128,"model.layers.%d.mlp.gate_proj.weight",l);lo[l].gp=jo(js,jl,b);snprintf(b,128,"model.layers.%d.mlp.up_proj.weight",l);lo[l].up=jo(js,jl,b);snprintf(b,128,"model.layers.%d.mlp.down_proj.weight",l);lo[l].dp=jo(js,jl,b);snprintf(b,128,"model.layers.%d.input_layernorm.weight",l);lo[l].in_off=jo(js,jl,b);snprintf(b,128,"model.layers.%d.post_attention_layernorm.weight",l);lo[l].pa_off=jo(js,jl,b);snprintf(b,128,"model.layers.%d.self_attn.q_norm.weight",l);lo[l].qn_off=jo(js,jl,b);snprintf(b,128,"model.layers.%d.self_attn.k_norm.weight",l);lo[l].kn_off=jo(js,jl,b);}
    uint64_t no=jo(js,jl,"model.norm.weight"),lo_off=jo(js,jl,"lm_head.weight");
    // Validate hardcoded dimensions against GGUF config
    {
        auto jvi = [&](const char* key, int expected) {
            auto p = (const char*)memmem(js, jl, key, strlen(key));
            if (!p) { fprintf(stderr, "WARN: key '%s' not found in GGUF metadata\n", key); return; }
            p = strchr(p, ':');
            if (!p) return;
            int val = (int)strtol(p + 1, NULL, 10);
            if (val != expected)
                fprintf(stderr, "WARN: %s=%d but hardcoded %s=%d — model may misbehave\n", key, val, key, expected);
        };
        jvi("\"model.embedding_length\"", H);
        jvi("\"model.block_count\"", NC);
        jvi("\"model.attention.head_count\"", NH);
        jvi("\"model.attention.head_count_kv\"", NKV);
        jvi("\"model.attention.layer_norm_epsilon\"", (int)(EPS * 1e6));
    }
    float in_n[NC][H],pa_n[NC][H],fin[H],qn_w[NC][HD],kn_w[NC][HD];
    for(int l=0;l<NC;l++){auto iw=(const uint16_t*)(md+df+lo[l].in_off),pw_=(const uint16_t*)(md+df+lo[l].pa_off),qw=(const uint16_t*)(md+df+lo[l].qn_off),kw=(const uint16_t*)(md+df+lo[l].kn_off);for(int i=0;i<H;i++){in_n[l][i]=bf16g(iw[i]);pa_n[l][i]=bf16g(pw_[i]);}for(int i=0;i<HD;i++){qn_w[l][i]=bf16g(qw[i]);kn_w[l][i]=bf16g(kw[i]);}}
    {auto fw=(const uint16_t*)(md+df+no);for(int i=0;i<H;i++)fin[i]=bf16g(fw[i]);}

    printf("Init 8 contexts...\n");xrt::device dev(0);
    #define D "/home/bcloud/npu-sandbox/npu-infer/build/int8"
    I8Ctx cq{"QKV",XM,H,4096},co{"O",XM,NH*HD,H},cg{"GU",XM,H,6144},cd{"D",XM,IM,H};
    cq.init(dev,D"/final_i8_QKV_v.xclbin",D"/insts_i8_QKV_v.txt",4);
    co.init(dev,D"/final_i8_O_v.xclbin",  D"/insts_i8_O_v.txt",  4);
    cg.init(dev,D"/final_i8_GU_v.xclbin", D"/insts_i8_GU_v.txt", 4);
    cd.init(dev,D"/final_i8_D_v.xclbin",  D"/insts_i8_D_v.txt",  4);
    AttnK ak[4]; for(int w=0;w<4;w++)ak[w].init(dev,w);
    printf("Dequant+pack...\n");auto tp=std::chrono::steady_clock::now();
    struct WS{float qk,o_,g_,d_;}wsc[NC];
    for(int l=0;l<NC;l++){int qr,kr,vr,or_,gr,ur,dr,unused;
        float*qw=dequant_i8_to_float(i8p(lo[l].qp),256,&qr,&unused),*kw=dequant_i8_to_float(i8p(lo[l].kp),128,&kr,&unused),*vw=dequant_i8_to_float(i8p(lo[l].vp),128,&vr,&unused);
        int t=qr+kr+vr;std::vector<float>w((size_t)H*t);for(int k=0;k<H;k++){memcpy(&w[k*t],&qw[k*qr],qr*4);memcpy(&w[k*t+qr],&kw[k*kr],kr*4);memcpy(&w[k*t+qr+kr],&vw[k*vr],vr*4);}
        cq.packB(l,w.data(),H,t,wsc[l].qk);free(qw);free(kw);free(vw);
        float*ow=dequant_i8_to_float(i8p(lo[l].op),256,&or_,&unused);co.packB(l,ow,or_,H,wsc[l].o_);free(ow);
        float*gw=dequant_i8_to_float(i8p(lo[l].gp),384,&gr,&unused),*uw=dequant_i8_to_float(i8p(lo[l].up),384,&ur,&unused);
        int t2=gr+ur;std::vector<float>w2((size_t)H*t2);for(int k=0;k<H;k++){memcpy(&w2[k*t2],&gw[k*gr],gr*4);memcpy(&w2[k*t2+gr],&uw[k*ur],ur*4);}
        cg.packB(l,w2.data(),H,t2,wsc[l].g_);free(gw);free(uw);
        float*dw=dequant_i8_to_float(i8p(lo[l].dp),384,&dr,&unused);cd.packB(l,dw,dr,H,wsc[l].d_);free(dw);}
    // Validate lm_head.weight shape; fall back to tied embeddings if wrong.
    {
        uint64_t lm_r = 0, lm_c = 0;
        if (jshape(js, jl, "lm_head.weight", lm_r, lm_c) && lm_r == (uint64_t)NV && lm_c == (uint64_t)H) {
            int lr, lc;
            float* lm_w = dequant_i8_to_float(i8p(lo_off), (int)lm_r, &lr, &lc);
            // TODO: pack lm_head for NPU acceleration — currently all output uses tied embeddings
            free(lm_w);
        } else {
            fprintf(stderr, "WARN: lm_head.weight shape [%lu,%lu] != expected [%d,%d]; using tied embeddings\n",
                    lm_r, lm_c, NV, H);
        }
    }
    printf("  %.0fms\n\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tp).count());
    ri(HD,1000000.0f,4096);
    // KV cache: pre-organized by window for zero-copy NPU attention dispatch
    struct KVCache{std::vector<float>k,v;int n;
        KVCache():k(4096*NKV*HD),v(4096*NKV*HD),n(0){}};std::vector<KVCache>kv(NC);
    std::vector<float>h(H),qo(4096),ko(1024),vo(1024),at(2048),oo(H),gt(6144),su(IM),dwo(H),lg(NV),sb(H),sc(4096);
    int sp=0;
    auto layer=[&](int l){
        memcpy(sb.data(),h.data(),H*4);rn_c(h.data(),in_n[l],H);
        cq.go(l,h.data(),1,H,dynamic_ascale(h.data(),H),wsc[l].qk,qo.data(),4096);cn(qo.data(),4096);
        memcpy(ko.data(),&qo[2048],4096);memcpy(vo.data(),&qo[3072],4096);
        float*qn=qn_w[l],*kn=kn_w[l];
        for(int hh=0;hh<NH;hh++){double sq=0;for(int d=0;d<HD;d++)sq+=qo[hh*HD+d]*qo[hh*HD+d];float iq=1.0f/sqrtf((float)(sq/HD)+EPS);for(int d=0;d<HD;d++)qo[hh*HD+d]*=iq*qn[d];ra(&qo[hh*HD],HD,sp);
            if(hh%GQA==0){int kvh=hh/GQA;double sk=0;for(int d=0;d<HD;d++)sk+=ko[kvh*HD+d]*ko[kvh*HD+d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);for(int d=0;d<HD;d++)ko[kvh*HD+d]*=ik*kn[d];ra(&ko[kvh*HD],HD,sp);memcpy(&kv[l].k[sp*NKV*HD+kvh*HD],&ko[kvh*HD],HD*4);memcpy(&kv[l].v[sp*NKV*HD+kvh*HD],&vo[kvh*HD],HD*4);}}
        kv[l].n=sp+1;int cl=kv[l].n;
        // CPU attention (faster than NPU repack at <100 tokens)
        for(int w=0;w<AW;w++){
            float*Qw=&qo[w*WQH*HD],*Kw=&kv[l].k[w*WKVH*HD],*Vw=&kv[l].v[w*WKVH*HD];
            AttnK::cpu_attn(Qw,Kw,Vw,cl,&at[w*WQH*HD]);
        }
        co.go(l,at.data(),1,NH*HD,dynamic_ascale(at.data(),NH*HD),wsc[l].o_,oo.data(),H);cn(oo.data(),H);for(int i=0;i<H;i++)h[i]=sb[i]+oo[i];
        memcpy(sb.data(),h.data(),H*4);rn_c(h.data(),pa_n[l],H);
        cg.go(l,h.data(),1,H,dynamic_ascale(h.data(),H),wsc[l].g_,gt.data(),6144);cn(gt.data(),6144);for(int i=0;i<IM;i++){float gv=gt[i];if(!std::isfinite(gv))gv=0;su[i]=(gv/(1.0f+expf(-gv)))*gt[IM+i];}
        cd.go(l,su.data(),1,IM,dynamic_ascale(su.data(),IM),wsc[l].d_,dwo.data(),H);cn(dwo.data(),H);for(int i=0;i<H;i++)h[i]=sb[i]+dwo[i];
    };
    int pt[]={151643,872,198,11852,151644,198,151643,77091,198};int npt=(argc>1)?atoi(argv[1]):9;if(npt<1)npt=1;if(npt>9)npt=9;
    printf("=== Prefill %d ===\n",npt);
    for(int pi=0;pi<npt;pi++){for(int i=0;i<H;i++)h[i]=bf16g(emb[pt[pi]*H+i]);for(int l=0;l<NC;l++)layer(l);sp++;if(pi%3==0)printf("  %d/%d\n",pi+1,npt);}
    printf("Done\n\n");
    printf("=== Generate ===\n");int ng=(argc>2)?atoi(argv[2]):8;auto tgs=std::chrono::steady_clock::now();
    for(int st=0;st<ng;st++){auto ts=std::chrono::steady_clock::now();
        for(int l=0;l<NC;l++)layer(l);
        memcpy(sb.data(),h.data(),H*4);rn_c(sb.data(),fin,H);
        for(int n=0;n<NV;n++){double s=0;for(int k=0;k<H;k++){uint16_t r=emb[n*H+k];if((r&0x7F80)!=0x7F80)s+=sb[k]*bf16f(r);}lg[n]=(float)s;}
        float mx=lg[0];for(int i=1;i<NV;i++)if(lg[i]>mx)mx=lg[i];double sum=0;for(int i=0;i<NV;i++){float d=lg[i]-mx;if(d<-80)d=-80;lg[i]=expf(d);sum+=lg[i];}
        float rr=(float)rand()/RAND_MAX*(float)sum,acc=0;int tok=0;for(int i=0;i<NV;i++){acc+=lg[i];if(acc>=rr){tok=i;break;}}
        double mss=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();
        printf("  [%d] %d (%.0fms)\n",st,tok,mss);for(int i=0;i<H;i++)h[i]=bf16g(emb[tok*H+i]);sp++;}
    double tts=std::chrono::duration<double>(std::chrono::steady_clock::now()-tgs).count();
    printf("\n=== %.0f ms/tok ===\n",tts*1000/ng);
    munmap(md,st.st_size);return 0;
}
