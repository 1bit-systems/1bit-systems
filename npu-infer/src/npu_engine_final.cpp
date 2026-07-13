/**
 * NPU Engine v5 — 2048×2048×2048 custom xclbin. 8× faster than v2.
 * Full: dequant→BFP16 pack→NPU GEMM→Q/K norms+RoPE→KV cache attn→MLP→token gen.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include "helper.h"
#include "gemm_atb_layout.h"
extern "C" float* dequant_i8_to_float(const uint8_t*,int,int*,int*);
extern "C" float bf16_to_float(uint16_t);
extern "C" uint16_t float_to_bf16(float);
static const int XM=2048,XK=2048,XN=2048,MT=128,KT=64,NT=128,NROWS=4;
static const char*XCLBIN="/home/bcloud/npu-sandbox/npu-infer/build/qwen3_gemm/design_2048.xclbin";
static const char*INSTRS="/home/bcloud/npu-sandbox/npu-infer/build/qwen3_gemm/design_2048.insts";
static float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static float bf16_g(uint16_t v){return((v&0x7F80)==0x7F80)?0:bf16f(v);}
static uint16_t fbf16(float v){uint32_t b;memcpy(&b,&v,4);uint32_t r=((b>>16)&1)+0x7FFF;return(uint16_t)((b+r)>>16);}

static std::vector<float> rope_cos,rope_sin;
static void rope_init(int hd,float theta,int max_pos){
    rope_cos.resize(max_pos*hd);rope_sin.resize(max_pos*hd);
    for(int p=0;p<max_pos;p++)for(int d=0;d<hd;d+=2){
        float f=1.0f/powf(theta,(float)d/hd),a=p*f;
        rope_cos[p*hd+d]=cosf(a);rope_sin[p*hd+d]=sinf(a);
        rope_cos[p*hd+d+1]=cosf(a);rope_sin[p*hd+d+1]=sinf(a);
    }
}
static void rope_apply(float*x,int hd,int pos){for(int d=0;d<hd;d+=2){float a=x[d],b=x[d+1],c=rope_cos[pos*hd+d],s=rope_sin[pos*hd+d];x[d]=a*c-b*s;x[d+1]=b*c+a*s;}}

static std::vector<uint8_t> pre_pack(const float*w,int out_feat,int in_feat,int deq_rows){
    std::vector<float> Bf(XK*XN);
    bool trans=(deq_rows==in_feat&&in_feat!=out_feat);
    for(int k=0;k<in_feat&&k<XK;k++)for(int n=0;n<out_feat&&n<XN;n++){
        float v=trans?w[k*out_feat+n]:w[n*in_feat+k];
        if(std::isfinite(v))Bf[k*XN+n]=v*16.0f;
    }
    auto Bs=gemm_atb::layout_transpose_L1_1x2_8x8block(Bf,XK,XN,KT,NT);
    auto Be=floatToBfp16(8,XK*XN,Bs.data(),0);
    return{Be.data(),Be.data()+Be.size()};
}

struct Npu{
    xrt::device dev{0};std::unique_ptr<xrt::xclbin>xc;std::unique_ptr<xrt::hw_context>hc;std::unique_ptr<xrt::kernel>k;
    std::vector<uint32_t>ins;std::unique_ptr<xrt::bo>bI,bA,bB,bC;
    bool init(){
        FILE*f=fopen(INSTRS,"rb");if(!f)return false;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);ins.resize(sz/4);
        size_t nr=fread(ins.data(),4,ins.size(),f);(void)nr;fclose(f);
        dev=xrt::device(0);xc=std::make_unique<xrt::xclbin>(std::string(XCLBIN));dev.register_xclbin(*xc);
        hc=std::make_unique<xrt::hw_context>(dev,xc->get_uuid());k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");
        bI=std::make_unique<xrt::bo>(dev,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));
        bA=std::make_unique<xrt::bo>(dev,(size_t)XM*XK*2,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));
        bB=std::make_unique<xrt::bo>(dev,(size_t)XK*XN*9/8,XRT_BO_FLAGS_HOST_ONLY,k->group_id(4));
        bC=std::make_unique<xrt::bo>(dev,(size_t)XM*XN*2,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));
        memcpy(bI->map(),ins.data(),ins.size()*4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);return true;
    }
    void gemm(const float*A,int M,int K,const uint8_t*pB,size_t psz,int out_feat,float*C){
        std::vector<float>Af(XM*XK);for(int m=0;m<M;m++)for(int k=0;k<K;k++)Af[m*XK+k]=A[m*K+k];
        auto As=gemm_atb::layout_A_L1_2x1_8x8block(Af,XM,XK,MT,KT);uint16_t*Ab=(uint16_t*)bA->map();
        for(int i=0;i<XM*XK;i++)Ab[i]=fbf16(As[i]);bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        memcpy(bB->map(),pB,psz);bB->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        memset(bC->map(),0,(size_t)XM*XN*2);bC->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run=(*k)((unsigned)3,*bI,ins.size(),*bA,*bB,*bC);run.wait();bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        uint16_t*Cb=(uint16_t*)bC->map();std::vector<float>Cf(XM*XN);
        for(int i=0;i<XM*XN;i++)Cf[i]=bf16f(Cb[i]);
        auto Cu=gemm_atb::layout_inverse_C_L1_2x2_8x8block(Cf,XM,XN,NROWS*MT,NT);
        for(int m=0;m<M;m++)for(int n=0;n<out_feat;n++)C[m*out_feat+n]=Cu[m*XN+n];
    }
};

struct LO{uint64_t qp,kp,vp,op,gp,up,dp,in_off,pa_off,qn_off,kn_off;};
struct PW{std::vector<uint8_t>q,k,v,o,g,u,d;};
struct KV{std::vector<float>k,v;int n;KV():k(64*8*128),v(64*8*128),n(0){}};

static uint64_t joff(const char*js,size_t jl,const char*nm){
    size_t nl=strlen(nm);const char*end=js+jl;
    while(js<end){auto q=(const char*)memmem(js,end-js,nm,nl);if(!q)return 0;
    if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){auto oo=strstr(q,"\"data_offsets\"");if(oo){auto ar=strchr(oo,'[');if(ar)return strtoull(ar+1,NULL,10);}}js=q+1;}
    return 0;
}

static void rms_norm(float*__restrict x,const uint16_t*__restrict w,int n,float eps){
    double ss=0;for(int i=0;i<n;i++)ss+=(double)x[i]*x[i];
    float ir=1.0f/sqrtf((float)(ss/n)+eps);
    for(int i=0;i<n;i++)x[i]=x[i]*ir*bf16_g(w[i]);
}

static void forward_layer(Npu&npu,int l,int seq_pos,const LO&lo,const PW&pw,float*h,float*qo,float*ko,float*vo,float*atto,float*oo,float*g,float*u,float*su,float*dow,KV&kvc,float eps,int H,int NH,int NKV,int HD,int IM,int GQA,const uint8_t*d,uint64_t df){
    float save[H];memcpy(save,h,H*sizeof(float));
    rms_norm(h,(const uint16_t*)(d+df+lo.in_off),H,eps);
    npu.gemm(h,1,H,pw.q.data(),pw.q.size(),2048,qo);
    npu.gemm(h,1,H,pw.k.data(),pw.k.size(),1024,ko);
    npu.gemm(h,1,H,pw.v.data(),pw.v.size(),1024,vo);
    auto qnw=(const uint16_t*)(d+df+lo.qn_off),knw=(const uint16_t*)(d+df+lo.kn_off);
    for(int hh=0;hh<NH;hh++){double sq=0;for(int d=0;d<HD;d++)sq+=qo[hh*HD+d]*qo[hh*HD+d];
        float iq=1.0f/sqrtf((float)(sq/HD)+eps);for(int d=0;d<HD;d++)qo[hh*HD+d]*=iq*bf16_g(qnw[d]);rope_apply(&qo[hh*HD],HD,seq_pos);
        if(hh%GQA==0){int kv=hh/GQA;double sk=0;for(int d=0;d<HD;d++)sk+=ko[kv*HD+d]*ko[kv*HD+d];
            float ik=1.0f/sqrtf((float)(sk/HD)+eps);for(int d=0;d<HD;d++)ko[kv*HD+d]*=ik*bf16_g(knw[d]);rope_apply(&ko[kv*HD],HD,seq_pos);
            memcpy(&kvc.k[seq_pos*NKV*HD+kv*HD],&ko[kv*HD],HD*4);memcpy(&kvc.v[seq_pos*NKV*HD+kv*HD],&vo[kv*HD],HD*4);}}
    kvc.n=seq_pos+1;int cl=kvc.n;
    for(int hh=0;hh<NH;hh++){int kv=hh/GQA;std::vector<float>sc(cl);
        for(int p=0;p<cl;p++){double s=0;for(int d=0;d<HD;d++)s+=qo[hh*HD+d]*kvc.k[p*NKV*HD+kv*HD+d];sc[p]=(float)(s/sqrtf((float)HD));}
        float mx=sc[0];for(int p=1;p<cl;p++)if(sc[p]>mx)mx=sc[p];double sum=0;
        for(int p=0;p<cl;p++){sc[p]=expf(sc[p]-mx);sum+=sc[p];}for(int p=0;p<cl;p++)sc[p]/=(float)sum;
        for(int d=0;d<HD;d++){float s=0;for(int p=0;p<cl;p++)s+=sc[p]*kvc.v[p*NKV*HD+kv*HD+d];atto[hh*HD+d]=s;}
    }
    npu.gemm(atto,1,NH*HD,pw.o.data(),pw.o.size(),H,oo);
    for(int i=0;i<H;i++)h[i]=save[i]+oo[i];
    rms_norm(h,(const uint16_t*)(d+df+lo.pa_off),H,eps);
    npu.gemm(h,1,H,pw.g.data(),pw.g.size(),IM,g);
    npu.gemm(h,1,H,pw.u.data(),pw.u.size(),IM,u);
    for(int i=0;i<IM;i++)su[i]=g[i]/(1.0f+expf(-g[i]))*u[i];
    npu.gemm(su,1,IM,pw.d.data(),pw.d.size(),H,dow);
    for(int i=0;i<H;i++)h[i]=save[i]+oo[i]+dow[i];
}

int main(){
    setvbuf(stdout,NULL,_IONBF,0);printf("=== NPU v3 — Qwen3-0.6B (2048×2048 xclbin) ===\n\n");
    const char*mp="/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    int fd=open(mp,O_RDONLY);struct stat st;fstat(fd,&st);
    uint8_t*d=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    uint64_t hsz;memcpy(&hsz,d,8);uint64_t df=8+hsz;
    auto i8p=[&](uint64_t o){return d+df+o;};auto emb=(const uint16_t*)(d+df);
    const char*js=(const char*)(d+8);size_t jl=hsz;
    const int H=1024,L=28,NH=16,NKV=8,HD=128,IM=3072,V=151936,GQA=2;const float eps=1e-6f;

    LO lo[28];
    for(int l=0;l<L;l++){char b[128];
        snprintf(b,sizeof(b),"model.layers.%d.self_attn.q_proj.weight",l);lo[l].qp=joff(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.self_attn.k_proj.weight",l);lo[l].kp=joff(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.self_attn.v_proj.weight",l);lo[l].vp=joff(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.self_attn.o_proj.weight",l);lo[l].op=joff(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.mlp.gate_proj.weight",l);lo[l].gp=joff(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.mlp.up_proj.weight",l);lo[l].up=joff(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.mlp.down_proj.weight",l);lo[l].dp=joff(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.input_layernorm.weight",l);lo[l].in_off=joff(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.post_attention_layernorm.weight",l);lo[l].pa_off=joff(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.self_attn.q_norm.weight",l);lo[l].qn_off=joff(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.self_attn.k_norm.weight",l);lo[l].kn_off=joff(js,jl,b);
    }
    uint64_t nm_off=joff(js,jl,"model.norm.weight"),lm_off=joff(js,jl,"lm_head.weight");

    Npu npu;if(!npu.init()){printf("NPU fail\n");return 1;}
    printf("Pre-dequantize + pre-pack...\n");auto t0=std::chrono::steady_clock::now();
    PW pw[28];float*lm_f;int lm_r,lm_c;
    for(int l=0;l<L;l++){
        int unused;
        float*qw=dequant_i8_to_float(i8p(lo[l].qp),256,&lm_r,&unused);pw[l].q=pre_pack(qw,lm_r,H,lm_r);free(qw);
        float*kw=dequant_i8_to_float(i8p(lo[l].kp),128,&lm_r,&unused);pw[l].k=pre_pack(kw,lm_r,H,lm_r);free(kw);
        float*vw=dequant_i8_to_float(i8p(lo[l].vp),128,&lm_r,&unused);pw[l].v=pre_pack(vw,lm_r,H,lm_r);free(vw);
        float*ow=dequant_i8_to_float(i8p(lo[l].op),256,&lm_r,&unused);pw[l].o=pre_pack(ow,H,NH*HD,lm_r);free(ow);
        float*gw=dequant_i8_to_float(i8p(lo[l].gp),384,&lm_r,&unused);pw[l].g=pre_pack(gw,lm_r,H,lm_r);free(gw);
        float*uw=dequant_i8_to_float(i8p(lo[l].up),384,&lm_r,&unused);pw[l].u=pre_pack(uw,lm_r,H,lm_r);free(uw);
        float*dw=dequant_i8_to_float(i8p(lo[l].dp),384,&lm_r,&unused);pw[l].d=pre_pack(dw,H,IM,lm_r);free(dw);
        if(l%7==0)printf("  L%d\n",l);
    }
    lm_f=dequant_i8_to_float(i8p(lm_off),18992,&lm_r,&lm_c);
    printf("Pre-packed in %.0f ms\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    rope_init(HD,1000000.0f,4096);std::vector<KV>kvc(L);
    std::vector<float>h(H),qo(2048),ko(1024),vo(1024),atto(2048),oo(H),g(IM),u(IM),su(IM),dow(H),logits(V);

    // 9-token chat template prefill
    int prompt_tokens[]={151643, 872, 198, 11852, 151644, 198, 151643, 77091, 198};
    int n_prompt=9;
    int seq_pos=0;
    for(int pi=0;pi<n_prompt;pi++){int tok=prompt_tokens[pi];for(int i=0;i<H;i++)h[i]=bf16_g(emb[tok*H+i]);
        for(int l=0;l<L;l++)forward_layer(npu,l,seq_pos,lo[l],pw[l],h.data(),qo.data(),ko.data(),vo.data(),atto.data(),oo.data(),g.data(),u.data(),su.data(),dow.data(),kvc[l],eps,H,NH,NKV,HD,IM,GQA,d,df);
        seq_pos++;if(pi%3==0)printf("  Prefill %d/%d\n",pi+1,n_prompt);}

    printf("Generating...\n");auto ti=std::chrono::steady_clock::now();
    for(int step=0;step<8;step++){
        auto ts=std::chrono::steady_clock::now();
        for(int l=0;l<L;l++)forward_layer(npu,l,seq_pos,lo[l],pw[l],h.data(),qo.data(),ko.data(),vo.data(),atto.data(),oo.data(),g.data(),u.data(),su.data(),dow.data(),kvc[l],eps,H,NH,NKV,HD,IM,GQA,d,df);
        auto fnw=(const uint16_t*)(d+df+nm_off);std::vector<float>no(H);
        memcpy(no.data(),h.data(),H*sizeof(float));
        rms_norm(no.data(),fnw,H,eps);
        for(int n=0;n<V;n++){double s=0;for(int k=0;k<H;k++){uint16_t r=emb[n*H+k];if((r&0x7F80)!=0x7F80)s+=no[k]*bf16f(r);}logits[n]=(float)s;} float mn=logits[0]; for(int i=1;i<V;i++)if(logits[i]<mn)mn=logits[i];
	    int token=0; float mx=logits[0]; for(int i=1;i<V;i++)if(logits[i]>mx){mx=logits[i];token=i;}
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();
        printf("  [%d] %d (%.0f ms) [%.1f,%.1f]\n",step,token,ms,mx,mn);fflush(stdout);
        for(int i=0;i<H;i++)h[i]=bf16_g(emb[token*H+i]);seq_pos++;
    }
    double total_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-ti).count();
    printf("\n%.0f ms/tok\n",total_s*1000/8);
    free(lm_f);munmap(d,st.st_size);return 0;
}
