/**
 * NPU Engine v8 — Chess-compiled, 32-core, 4 resident xclbins.
 * Phase 1: register all 4 xclbins. Phase 2: create all 4 contexts+BOs.
 * NO swapping — all contexts alive simultaneously.
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

static float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static float bf16_g(uint16_t v){return((v&0x7F80)==0x7F80)?0:bf16f(v);}
static uint16_t fbf16(float v){uint32_t b;memcpy(&b,&v,4);uint32_t r=((b>>16)&1)+0x7FFF;return(uint16_t)((b+r)>>16);}
static constexpr int MT=128,KT=64,NT=128,NROWS=4,XM=1024;
static constexpr int H=1024,NC=28,NH=16,NKV=8,HD=128,IM=3072,NV=151936,GQA=2;
static constexpr float EPS=1e-6f;

struct ChessCtx {
    const char* name,*xp,*ip;int MD,KD,ND;
    xrt::device* dev;
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::kernel> k;
    std::vector<uint32_t> ins;
    std::unique_ptr<xrt::bo> bI,bA,bB,bC;
    std::vector<float> Ab,Cb;
    bool alive=false;

    ~ChessCtx(){release();}
    void release(){k.reset();hc.reset();alive=false;}

    // Load + register + create hw_context + kernel + BOs
    bool init(xrt::device& d){
        release();
        dev=&d;
        FILE*f=fopen(ip,"rb");if(!f)return false;
        fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);
        ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
        xc=std::make_unique<xrt::xclbin>(std::string(xp));d.register_xclbin(*xc);
        hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());
        k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");
        bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));
        bA=std::make_unique<xrt::bo>(d,(size_t)MD*KD*2,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));
        bB=std::make_unique<xrt::bo>(d,(size_t)KD*ND*9/8,XRT_BO_FLAGS_HOST_ONLY,k->group_id(4));
        bC=std::make_unique<xrt::bo>(d,(size_t)MD*ND*2,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));
        memcpy(bI->map(),ins.data(),ins.size()*4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        Ab.resize((size_t)MD*KD);Cb.resize((size_t)MD*ND);
        alive=true;return true;
    }

    // Ensure-alive swap: activate this context, release any other active one
    bool sa(xrt::device& d, ChessCtx*& active_ctx) {
        if (alive) { active_ctx = this; return true; }
        if (active_ctx && active_ctx != this) active_ctx->release();
        active_ctx = this;
        return init(d);
    }

    void go(const float*A,int am,int ak,const uint8_t*B,size_t Bsz,float*C,int an){
        int MK=MD*KD;memset(Ab.data(),0,MK*4);
        for(int m=0;m<am;m++)for(int kk=0;kk<ak;kk++)Ab[m*KD+kk]=A[m*ak+kk];
        auto As=gemm_atb::layout_A_L1_2x1_8x8block(Ab,MD,KD,MT,KT);
        uint16_t*Ab2=(uint16_t*)bA->map();
        for(int i=0;i<MK;i++)Ab2[i]=fbf16(As[i]);bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        memcpy(bB->map(),B,Bsz);bB->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        memset(bC->map(),0,(size_t)MD*ND*2);bC->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r=(*k)((unsigned)3,*bI,ins.size(),*bA,*bB,*bC);
        if(r.wait()!=ERT_CMD_STATE_COMPLETED){fprintf(stderr,"[%s] FAIL\n",name);return;}
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        uint16_t*Cc=(uint16_t*)bC->map();int MN=MD*ND;
        for(int i=0;i<MN;i++)Cb[i]=bf16f(Cc[i]);
        auto Cu=gemm_atb::layout_inverse_C_L1_2x2_8x8block(Cb,MD,ND,NROWS*MT,NT);
        for(int m=0;m<am;m++)for(int n=0;n<an;n++)C[m*an+n]=Cu[m*ND+n];
    }
};

static std::vector<uint8_t> pbf(const float*w,int K,int N,int dqr){
    std::vector<float>Bf(K*N,0.0f);bool tr=(dqr==K&&dqr!=N);
    for(int k=0;k<K;k++)for(int n=0;n<N;n++){float v=tr?w[k*N+n]:w[n*K+k];if(std::isfinite(v))Bf[k*N+n]=v;}
    auto Bs=gemm_atb::layout_transpose_L1_1x2_8x8block(Bf,K,N,KT,NT);
    auto Be=floatToBfp16(8,K*N,Bs.data(),0);return{Be.data(),Be.data()+Be.size()};
}
static std::vector<float>rc,rs;
static void ri(int hd,float th,int mp){rc.resize(mp*hd);rs.resize(mp*hd);
    for(int p=0;p<mp;p++)for(int d=0;d<hd;d+=2){float f=1.0f/powf(th,(float)d/hd),a=p*f;
        rc[p*hd+d]=cosf(a);rs[p*hd+d]=sinf(a);rc[p*hd+d+1]=cosf(a);rs[p*hd+d+1]=sinf(a);}}
static void ra(float*x,int hd,int p){for(int d=0;d<hd;d+=2){
    float a=x[d],b=x[d+1],c=rc[p*hd+d],s=rs[p*hd+d];x[d]=a*c-b*s;x[d+1]=b*c+a*s;}}
static void cn(float*x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
static void sm(float*sc,int n){if(n<=0)return;cn(sc,n);
    float mx=sc[0];for(int i=1;i<n;i++)if(sc[i]>mx)mx=sc[i];
    double s=0;for(int i=0;i<n;i++){float d=sc[i]-mx;if(d>80)sc[i]=expf(80);else sc[i]=expf(d);s+=sc[i];}
    if(s<=0){float iv=1.0f/n;for(int i=0;i<n;i++)sc[i]=iv;return;}
    float is=1.0f/(float)s;for(int i=0;i<n;i++)sc[i]*=is;}
static void rn(float*x,const uint16_t*w,int n){cn(x,n);
    double ss=0;for(int i=0;i<n;i++)ss+=(double)x[i]*x[i];
    float ir=1.0f/sqrtf((float)(ss/n)+EPS);for(int i=0;i<n;i++)x[i]=x[i]*ir*bf16_g(w[i]);}
static uint64_t jo(const char*js,size_t jl,const char*nm){
    size_t nl=strlen(nm);const char*p=js,*e=js+jl;
    while(p<e){auto q=(const char*)memmem(p,e-p,nm,nl);if(!q)return 0;
        if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){auto o=strstr(q,"\"data_offsets\"");
            if(o){auto a=strchr(o,'[');if(a)return strtoull(a+1,NULL,10);}}p=q+1;}return 0;}

int main(){
    setvbuf(stdout,NULL,_IONBF,0);
    printf("=== v8 Chess 4-Resident ===\n\n");
    const char*mp="/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    int fd=open(mp,O_RDONLY);struct stat st;fstat(fd,&st);
    uint8_t*md=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    uint64_t hsz;memcpy(&hsz,md,8);uint64_t df=8+hsz;
    auto i8p=[&](uint64_t o){return md+df+o;};auto emb=(const uint16_t*)(md+df);
    const char*js=(const char*)(md+8);size_t jl=hsz;

    struct LO{uint64_t qp,kp,vp,op,gp,up,dp,in_off,pa_off,qn_off,kn_off;}lo[28];char b[128];
    for(int l=0;l<NC;l++){
        snprintf(b,sizeof(b),"model.layers.%d.self_attn.q_proj.weight",l);lo[l].qp=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.self_attn.k_proj.weight",l);lo[l].kp=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.self_attn.v_proj.weight",l);lo[l].vp=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.self_attn.o_proj.weight",l);lo[l].op=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.mlp.gate_proj.weight",l);lo[l].gp=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.mlp.up_proj.weight",l);lo[l].up=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.mlp.down_proj.weight",l);lo[l].dp=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.input_layernorm.weight",l);lo[l].in_off=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.post_attention_layernorm.weight",l);lo[l].pa_off=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.self_attn.q_norm.weight",l);lo[l].qn_off=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.self_attn.k_norm.weight",l);lo[l].kn_off=jo(js,jl,b);
    }
    uint64_t no=jo(js,jl,"model.norm.weight"),lo_=jo(js,jl,"lm_head.weight");

    printf("Init (swap-based, 1 active context)...\n");xrt::device dev(0);
    #define B "/home/bcloud/npu-sandbox/npu-infer/build/chess_infer"
    ChessCtx cq{"QKV",B"/qkv/qwen3_qkv_chess.xclbin",B"/qkv/qwen3_qkv_chess.insts",XM,H,4096};
    ChessCtx co{"O",  B"/o/qwen3_o_chess.xclbin",    B"/o/qwen3_o_chess.insts",    XM,2048,H};
    ChessCtx cg{"GU", B"/gu/qwen3_gu_chess.xclbin",  B"/gu/qwen3_gu_chess.insts",  XM,H,6144};
    ChessCtx cd{"D",  B"/d/qwen3_d_chess.xclbin",    B"/d/qwen3_d_chess.insts",    XM,IM,H};
    ChessCtx* active=nullptr;
    cq.sa(dev,active);
    printf("Ready (QKV loaded, swap on demand).\n\n");

    printf("Dequant+pack...\n");auto t0=std::chrono::steady_clock::now();
    struct PW{std::vector<uint8_t>qk,o_,g_,d_;}pw[28];float*lf;int lr,lc;
    for(int l=0;l<NC;l++){
        int qr,kr,vr,or_,gr,ur,dr,unused;
        float*qw=dequant_i8_to_float(i8p(lo[l].qp),256,&qr,&unused);
        float*kw=dequant_i8_to_float(i8p(lo[l].kp),128,&kr,&unused);
        float*vw=dequant_i8_to_float(i8p(lo[l].vp),128,&vr,&unused);
        std::vector<float>w((size_t)(qr+kr+vr)*H);
        memcpy(w.data(),qw,qr*H*4);memcpy(w.data()+qr*H,kw,kr*H*4);memcpy(w.data()+(qr+kr)*H,vw,vr*H*4);
        pw[l].qk=pbf(w.data(),H,qr+kr+vr,qr+kr+vr);free(qw);free(kw);free(vw);
        float*ow=dequant_i8_to_float(i8p(lo[l].op),256,&or_,&unused);pw[l].o_=pbf(ow,or_,H,or_);free(ow);
        float*gw=dequant_i8_to_float(i8p(lo[l].gp),384,&gr,&unused);
        float*uw=dequant_i8_to_float(i8p(lo[l].up),384,&ur,&unused);
        std::vector<float>w2((size_t)(gr+ur)*H);
        memcpy(w2.data(),gw,gr*H*4);memcpy(w2.data()+gr*H,uw,ur*H*4);
        pw[l].g_=pbf(w2.data(),H,gr+ur,gr+ur);free(gw);free(uw);
        float*dw=dequant_i8_to_float(i8p(lo[l].dp),384,&dr,&unused);pw[l].d_=pbf(dw,dr,H,dr);free(dw);
        if(l%7==0)printf("  L%d\n",l);
    }
    lf=dequant_i8_to_float(i8p(lo_),18992,&lr,&lc);
    printf("Ready %.1fs\n\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count());

    ri(HD,1000000.0f,4096);
    struct KV{std::vector<float>k,v;int n;KV():k(4096*NKV*HD),v(4096*NKV*HD),n(0){}};
    std::vector<KV>kv(NC);
    std::vector<float>h(H),qo(4096),ko(1024),vo(1024),at(2048),oo(H),gt(6144),u(IM),su(IM),dw(H),lg(NV);
    std::vector<float>sb(H),sc(4096);
    int pt[]={151643,872,198,11852,151644,198,151643,77091,198};int npt=9,sp=0;

    auto layer=[&](int l){
        memcpy(sb.data(),h.data(),H*4);
        rn(h.data(),(const uint16_t*)(md+df+lo[l].in_off),H);
        cq.sa(dev,active);cq.go(h.data(),1,H,pw[l].qk.data(),pw[l].qk.size(),qo.data(),4096);cn(qo.data(),4096);
        memcpy(ko.data(),&qo[2048],1024*4);memcpy(vo.data(),&qo[3072],1024*4);
        auto qn=(const uint16_t*)(md+df+lo[l].qn_off),kn=(const uint16_t*)(md+df+lo[l].kn_off);
        for(int hh=0;hh<NH;hh++){double sq=0;for(int d=0;d<HD;d++)sq+=qo[hh*HD+d]*qo[hh*HD+d];
            float iq=1.0f/sqrtf((float)(sq/HD)+EPS);
            for(int d=0;d<HD;d++)qo[hh*HD+d]*=iq*bf16_g(qn[d]);ra(&qo[hh*HD],HD,sp);
            if(hh%GQA==0){int kvh=hh/GQA;double sk=0;for(int d=0;d<HD;d++)sk+=ko[kvh*HD+d]*ko[kvh*HD+d];
                float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
                for(int d=0;d<HD;d++)ko[kvh*HD+d]*=ik*bf16_g(kn[d]);ra(&ko[kvh*HD],HD,sp);
                memcpy(&kv[l].k[sp*NKV*HD+kvh*HD],&ko[kvh*HD],HD*4);
                memcpy(&kv[l].v[sp*NKV*HD+kvh*HD],&vo[kvh*HD],HD*4);}}
        kv[l].n=sp+1;int cl=kv[l].n;
        for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;float*ss=sc.data();
            for(int p=0;p<cl;p++){double s=0;for(int d=0;d<HD;d++)s+=qo[hh*HD+d]*kv[l].k[p*NKV*HD+kvh*HD+d];ss[p]=(float)(s/sqrtf(HD));}
            sm(ss,cl);
            for(int d=0;d<HD;d++){float s=0;for(int p=0;p<cl;p++)s+=ss[p]*kv[l].v[p*NKV*HD+kvh*HD+d];at[hh*HD+d]=s;}}
        co.sa(dev,active);co.go(at.data(),1,NH*HD,pw[l].o_.data(),pw[l].o_.size(),oo.data(),H);cn(oo.data(),H);
        for(int i=0;i<H;i++)h[i]=sb[i]+oo[i];
        memcpy(sb.data(),h.data(),H*4);rn(h.data(),(const uint16_t*)(md+df+lo[l].pa_off),H);
        cg.sa(dev,active);cg.go(h.data(),1,H,pw[l].g_.data(),pw[l].g_.size(),gt.data(),6144);cn(gt.data(),6144);
        memcpy(u.data(),&gt[IM],IM*4);{for(int i=0;i<IM;i++)su[i]=(gt[i]/(1.0f+expf(-gt[i])))*u[i];}
        cd.sa(dev,active);cd.go(su.data(),1,IM,pw[l].d_.data(),pw[l].d_.size(),dw.data(),H);cn(dw.data(),H);
        for(int i=0;i<H;i++)h[i]=sb[i]+dw[i];
    };

    printf("=== Prefill %d ===\n",npt);
    for(int pi=0;pi<npt;pi++){
        for(int i=0;i<H;i++)h[i]=bf16_g(emb[pt[pi]*H+i]);
        for(int l=0;l<NC;l++)layer(l);sp++;
        if(pi%3==0)printf("  %d/%d\n",pi+1,npt);
    }printf("Done (%d tokens, %d pos)\n\n",npt,sp);

    printf("=== Generate ===\n");int ng=8;auto tgs=std::chrono::steady_clock::now();
    for(int st=0;st<ng;st++){auto ts=std::chrono::steady_clock::now();
        for(int l=0;l<NC;l++)layer(l);
        memcpy(sb.data(),h.data(),H*4);
        auto fnw=(const uint16_t*)(md+df+no);rn(sb.data(),fnw,H);
        for(int n=0;n<NV;n++){double s=0;for(int k=0;k<H;k++){uint16_t r=emb[n*H+k];if((r&0x7F80)!=0x7F80)s+=sb[k]*bf16f(r);}lg[n]=(float)s;}
        float mx=lg[0],mn=lg[0];for(int i=1;i<NV;i++){if(lg[i]>mx)mx=lg[i];if(lg[i]<mn)mn=lg[i];}
        double sum=0;for(int i=0;i<NV;i++){lg[i]=expf(lg[i]-mx);sum+=lg[i];}
        float rr=(float)rand()/RAND_MAX*(float)sum,acc=0;int tok=0;for(int i=0;i<NV;i++){acc+=lg[i];if(acc>=rr){tok=i;break;}}
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();
        printf("  [%d] %d (%.0fms) [%.1f,%.1f]\n",st,tok,ms,mn,mx);
        for(int i=0;i<H;i++)h[i]=bf16_g(emb[tok*H+i]);sp++;
    }
    double ts=std::chrono::duration<double>(std::chrono::steady_clock::now()-tgs).count();
    printf("\n=== %.0f ms/tok ===\n",ts*1000/ng);
    free(lf);munmap(md,st.st_size);return 0;
}
