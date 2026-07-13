/**
 * NPU Engine i8 — pre-loaded weight BOs, context-pool, cached norms.
 * Eliminates per-layer weight memcpy: 28 B BOs per slot, packed at init.
 */
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
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
extern "C" float* dequant_i8_to_float(const uint8_t*,int,int*,int*);

static inline float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static inline float bf16g(uint16_t v){return(v&0x7F80)==0x7F80?0.0f:bf16f(v);}
static constexpr int H=1024,NC=28,NH=16,NKV=8,HD=128,IM=3072,NV=151936,GQA=2;
static constexpr float EPS=1e-6f; static constexpr int XM=128;

struct CachedNorms {float in_n[NC][H],pa_n[NC][H],fin[H],qn_w[NC][HD],kn_w[NC][HD];};

// Per-slot: 28 B BOs (one per layer), pre-loaded at init
struct I8Slot {
    const char*name; int MD,KD,ND;
    std::unique_ptr<xrt::xclbin> xc; xrt::uuid uuid;
    std::unique_ptr<xrt::hw_context> hc; std::unique_ptr<xrt::kernel> k;
    std::vector<uint32_t> ins;
    std::unique_ptr<xrt::bo> bI,bA,bC;
    // Per-layer B BOs — written once at init, no runtime memcpy
    std::unique_ptr<xrt::bo> layerB[NC];
    bool init(xrt::device& d,const char*xp,const char*ip,int gid_B){
        FILE*f=fopen(ip,"rb");if(!f)return false;
        fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
        xc=std::make_unique<xrt::xclbin>(std::string(xp));uuid=xc->get_uuid();d.register_xclbin(*xc);
        hc=std::make_unique<xrt::hw_context>(d,uuid);k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");
        bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));
        memcpy(bI->map(),ins.data(),ins.size()*4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bA=std::make_unique<xrt::bo>(d,(size_t)MD*KD,   XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));
        bC=std::make_unique<xrt::bo>(d,(size_t)MD*ND*2, XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));
        // Create all 28 layer B BOs
        for(int l=0;l<NC;l++) layerB[l]=std::make_unique<xrt::bo>(d,(size_t)KD*ND,XRT_BO_FLAGS_HOST_ONLY,k->group_id(gid_B));
        return true;
    }
    void activate(xrt::device& d){hc.reset();hc=std::make_unique<xrt::hw_context>(d,uuid);k.reset();k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");}
    inline void go(int layer,const float*A,int am,int ak,float ascale,float Bscale,float*C,int an){
        auto*Am=(int8_t*)bA->map();memset(Am,0,(size_t)MD*KD);
        float ais=1.0f/ascale;
        for(int m=0;m<am;m++)for(int k=0;k<ak;k++){float v=A[m*ak+k];if(!std::isfinite(v))v=0;int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;Am[m*KD+k]=(int8_t)q;}
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        layerB[layer]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r=(*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA,*layerB[layer],*bC);
        if(r.wait()!=ERT_CMD_STATE_COMPLETED){fprintf(stderr,"[%s:L%d] FAIL\n",name,layer);return;}
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);auto*Cm=(int16_t*)bC->map();float cs=ascale*Bscale;
        for(int m=0;m<am;m++)for(int n=0;n<an;n++){float val=(float)Cm[m*ND+n]*cs;if(!std::isfinite(val))val=0;C[m*an+n]=val;}
    }
    // Write packed weight directly into layer B BO at init
    void packB(int layer,const float*w,int K,int N,float&scale_out){
        float amax=0;for(int i=0;i<K*N;i++){float a=fabsf(w[i]);if(std::isfinite(a)&&a>amax)amax=a;}
        if(amax<1e-12f)amax=1.0f;scale_out=amax/127.0f;float is=127.0f/amax;
        auto*Bm=(int8_t*)layerB[layer]->map();
        for(int i=0;i<K*N;i++){float v=w[i];if(!std::isfinite(v))v=0;int x=(int)roundf(v*is);if(x>127)x=127;else if(x<-127)x=-127;Bm[i]=(int8_t)x;}
        layerB[layer]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
};
static std::vector<float>rc,rs;static void ri(int hd,float th,int mp){rc.resize(mp*hd);rs.resize(mp*hd);for(int p=0;p<mp;p++)for(int d=0;d<hd;d+=2){float f=1.0f/powf(th,(float)d/hd),a=p*f;rc[p*hd+d]=cosf(a);rs[p*hd+d]=sinf(a);rc[p*hd+d+1]=cosf(a);rs[p*hd+d+1]=sinf(a);}}
static inline void ra(float*x,int hd,int p){for(int d=0;d<hd;d+=2){float a=x[d],b=x[d+1],c=rc[p*hd+d],s=rs[p*hd+d];x[d]=a*c-b*s;x[d+1]=b*c+a*s;}}
static inline void cn(float*x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
static inline void sm(float*sc,int n){if(n<=0)return;cn(sc,n);float mx=sc[0];for(int i=1;i<n;i++)if(sc[i]>mx)mx=sc[i];double s=0;for(int i=0;i<n;i++){float d=sc[i]-mx;d=d>80?80:(d<-80?-80:d);sc[i]=expf(d);s+=sc[i];}if(s<=0){float iv=1.0f/n;for(int i=0;i<n;i++)sc[i]=iv;return;}float is=1.0f/(float)s;for(int i=0;i<n;i++)sc[i]*=is;}
static inline void rn_cached(float*x,const float*w,int n){cn(x,n);double ss=0;for(int i=0;i<n;i++)if(std::isfinite(x[i]))ss+=(double)x[i]*x[i];float ir=1.0f/sqrtf((float)(ss/n)+EPS);for(int i=0;i<n;i++)x[i]=std::isfinite(x[i])?x[i]*ir*w[i]:0.0f;}
static uint64_t jo(const char*js,size_t jl,const char*nm){size_t nl=strlen(nm);const char*p=js,*e=js+jl;while(p<e){auto q=(const char*)memmem(p,e-p,nm,nl);if(!q)return 0;if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){auto o=strstr(q,"\"data_offsets\"");if(o){auto a=strchr(o,'[');if(a)return strtoull(a+1,NULL,10);}}p=q+1;}return 0;}

int main(){
    setvbuf(stdout,NULL,_IONBF,0);
    printf("=== NPU Engine i8 — Pre-loaded Weights ===\n\n");
    const char*mp="/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    int fd=open(mp,O_RDONLY);struct stat st;fstat(fd,&st);
    uint8_t*md=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    uint64_t hsz;memcpy(&hsz,md,8);uint64_t df=8+hsz;
    auto i8p=[&](uint64_t o){return md+df+o;};auto emb=(const uint16_t*)(md+df);
    const char*js=(const char*)(md+8);size_t jl=hsz;
    struct LO{uint64_t qp,kp,vp,op,gp,up,dp,in_off,pa_off,qn_off,kn_off;}lo[NC];char b[128];
    for(int l=0;l<NC;l++){
        snprintf(b,128,"model.layers.%d.self_attn.q_proj.weight",l);lo[l].qp=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.self_attn.k_proj.weight",l);lo[l].kp=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.self_attn.v_proj.weight",l);lo[l].vp=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.self_attn.o_proj.weight",l);lo[l].op=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.mlp.gate_proj.weight",l);lo[l].gp=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.mlp.up_proj.weight",l);lo[l].up=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.mlp.down_proj.weight",l);lo[l].dp=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.input_layernorm.weight",l);lo[l].in_off=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.post_attention_layernorm.weight",l);lo[l].pa_off=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.self_attn.q_norm.weight",l);lo[l].qn_off=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.self_attn.k_norm.weight",l);lo[l].kn_off=jo(js,jl,b);
    }
    uint64_t no=jo(js,jl,"model.norm.weight"),lo_off=jo(js,jl,"lm_head.weight");

    auto*nm=new CachedNorms();
    for(int l=0;l<NC;l++){auto iw=(const uint16_t*)(md+df+lo[l].in_off),pw_=(const uint16_t*)(md+df+lo[l].pa_off);auto qw=(const uint16_t*)(md+df+lo[l].qn_off),kw=(const uint16_t*)(md+df+lo[l].kn_off);for(int i=0;i<H;i++){nm->in_n[l][i]=bf16g(iw[i]);nm->pa_n[l][i]=bf16g(pw_[i]);}for(int i=0;i<HD;i++){nm->qn_w[l][i]=bf16g(qw[i]);nm->kn_w[l][i]=bf16g(kw[i]);}}
    {auto fw=(const uint16_t*)(md+df+no);for(int i=0;i<H;i++)nm->fin[i]=bf16g(fw[i]);}

    printf("Init NPU + create 112 layer BOs...\n");
    xrt::device dev(0);
    #define D "/home/bcloud/npu-sandbox/npu-infer/build/int8"
    I8Slot slots[4];
    slots[0].name="QKV";slots[0].MD=XM;slots[0].KD=H;slots[0].ND=4096;slots[0].init(dev,D"/final_i8_QKV_v.xclbin",D"/insts_i8_QKV_v.txt",4);
    slots[1].name="O";  slots[1].MD=XM;slots[1].KD=NH*HD;slots[1].ND=H;slots[1].init(dev,D"/final_i8_O_v.xclbin",D"/insts_i8_O_v.txt",4);
    slots[2].name="GU"; slots[2].MD=XM;slots[2].KD=H;slots[2].ND=6144;slots[2].init(dev,D"/final_i8_GU_v.xclbin",D"/insts_i8_GU_v.txt",4);
    slots[3].name="D";  slots[3].MD=XM;slots[3].KD=IM;slots[3].ND=H;slots[3].init(dev,D"/final_i8_D_v.xclbin",D"/insts_i8_D_v.txt",4);
    int cur=-1;

    printf("Dequant + pack into layer BOs...\n");auto t0=std::chrono::steady_clock::now();
    struct WS{float qk,o_,g_,d_;}wsc[NC];
    for(int l=0;l<NC;l++){
        int qr,kr,vr,or_,gr,ur,dr,unused;
        // QKV
        float*qw=dequant_i8_to_float(i8p(lo[l].qp),256,&qr,&unused),*kw=dequant_i8_to_float(i8p(lo[l].kp),128,&kr,&unused),*vw=dequant_i8_to_float(i8p(lo[l].vp),128,&vr,&unused);
        int t=qr+kr+vr;std::vector<float>w((size_t)H*t);for(int k=0;k<H;k++){memcpy(&w[k*t],&qw[k*qr],qr*4);memcpy(&w[k*t+qr],&kw[k*kr],kr*4);memcpy(&w[k*t+qr+kr],&vw[k*vr],vr*4);}
        slots[0].packB(l,w.data(),H,t,wsc[l].qk);free(qw);free(kw);free(vw);
        // O
        float*ow=dequant_i8_to_float(i8p(lo[l].op),256,&or_,&unused);slots[1].packB(l,ow,or_,H,wsc[l].o_);free(ow);
        // GU
        float*gw=dequant_i8_to_float(i8p(lo[l].gp),384,&gr,&unused),*uw=dequant_i8_to_float(i8p(lo[l].up),384,&ur,&unused);
        int t2=gr+ur;std::vector<float>w2((size_t)H*t2);for(int k=0;k<H;k++){memcpy(&w2[k*t2],&gw[k*gr],gr*4);memcpy(&w2[k*t2+gr],&uw[k*ur],ur*4);}
        slots[2].packB(l,w2.data(),H,t2,wsc[l].g_);free(gw);free(uw);
        // D
        float*dw=dequant_i8_to_float(i8p(lo[l].dp),384,&dr,&unused);slots[3].packB(l,dw,dr,H,wsc[l].d_);free(dw);
    }
    int lr,lc;free(dequant_i8_to_float(i8p(lo_off),18992,&lr,&lc));
    printf("Ready (%.1fs)\n\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count());

    ri(HD,1000000.0f,4096);
    struct KV{std::vector<float>k,v;int n;KV():k(4096*NKV*HD),v(4096*NKV*HD),n(0){}};std::vector<KV>kv(NC);
    std::vector<float>h(H),qo(4096),ko(1024),vo(1024),at(2048),oo(H),gt(6144),su(IM),dwo(H),lg(NV),sb(H),sc(4096);
    int sp=0;

    auto dump_vec=[&](const char* label, const float* v, int n, int fn=8){

        memcpy(sb.data(),h.data(),H*4);rn_cached(h.data(),nm->in_n[l],H);
        // QKV GEMM — no loadB, BO pre-loaded
        if(0!=cur){slots[0].activate(dev);cur=0;}
        slots[0].go(l,h.data(),1,H,5.0f/127.0f,wsc[l].qk,qo.data(),4096);cn(qo.data(),4096);
        memcpy(ko.data(),&qo[2048],1024*4);memcpy(vo.data(),&qo[3072],1024*4);
        float*qn=nm->qn_w[l],*kn=nm->kn_w[l];
        for(int hh=0;hh<NH;hh++){double sq=0;for(int d=0;d<HD;d++)sq+=qo[hh*HD+d]*qo[hh*HD+d];float iq=1.0f/sqrtf((float)(sq/HD)+EPS);for(int d=0;d<HD;d++)qo[hh*HD+d]*=iq*qn[d];ra(&qo[hh*HD],HD,sp);
            if(hh%GQA==0){int kvh=hh/GQA;double sk=0;for(int d=0;d<HD;d++)sk+=ko[kvh*HD+d]*ko[kvh*HD+d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);for(int d=0;d<HD;d++)ko[kvh*HD+d]*=ik*kn[d];ra(&ko[kvh*HD],HD,sp);memcpy(&kv[l].k[sp*NKV*HD+kvh*HD],&ko[kvh*HD],HD*4);memcpy(&kv[l].v[sp*NKV*HD+kvh*HD],&vo[kvh*HD],HD*4);}}
        kv[l].n=sp+1;int cl=kv[l].n;
        for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;float*ss=sc.data();for(int p=0;p<cl;p++){double s=0;for(int d=0;d<HD;d++)s+=qo[hh*HD+d]*kv[l].k[p*NKV*HD+kvh*HD+d];ss[p]=(float)(s/sqrtf(HD));}sm(ss,cl);for(int d=0;d<HD;d++){float s=0;for(int p=0;p<cl;p++)s+=ss[p]*kv[l].v[p*NKV*HD+kvh*HD+d];at[hh*HD+d]=s;}}
        // O GEMM — pre-loaded BO
        if(1!=cur){slots[1].activate(dev);cur=1;}
        slots[1].go(l,at.data(),1,NH*HD,5.0f/127.0f,wsc[l].o_,oo.data(),H);cn(oo.data(),H);for(int i=0;i<H;i++)h[i]=sb[i]+oo[i];
        memcpy(sb.data(),h.data(),H*4);rn_cached(h.data(),nm->pa_n[l],H);
        // GU GEMM — pre-loaded BO
        if(2!=cur){slots[2].activate(dev);cur=2;}
        slots[2].go(l,h.data(),1,H,5.0f/127.0f,wsc[l].g_,gt.data(),6144);cn(gt.data(),6144);for(int i=0;i<IM;i++){float g=gt[i];if(!std::isfinite(g))g=0;su[i]=(g/(1.0f+expf(-g)))*gt[IM+i];}
        // D GEMM — pre-loaded BO
        if(3!=cur){slots[3].activate(dev);cur=3;}
        slots[3].go(l,su.data(),1,IM,5.0f/127.0f,wsc[l].d_,dwo.data(),H);cn(dwo.data(),H);for(int i=0;i<H;i++)h[i]=sb[i]+dwo[i];
    };

    int pt[]={151643,872,198,11852,151644,198,151643,77091,198};int npt=9;
    printf("=== Prefill %d ===\n",npt);
    for(int pi=0;pi<npt;pi++){for(int i=0;i<H;i++)h[i]=bf16g(emb[pt[pi]*H+i]);for(int l=0;l<NC;l++)layer(l);sp++;if(pi%3==0)printf("  %d/%d\n",pi+1,npt);}
    printf("Done\n\n");

    printf("=== Generate ===\n");int ng=8;auto tgs=std::chrono::steady_clock::now();
    for(int st=0;st<ng;st++){auto ts=std::chrono::steady_clock::now();
        for(int l=0;l<NC;l++)layer(l);
        memcpy(sb.data(),h.data(),H*4);rn_cached(sb.data(),nm->fin,H);
        for(int n=0;n<NV;n++){double s=0;for(int k=0;k<H;k++){uint16_t r=emb[n*H+k];if((r&0x7F80)!=0x7F80)s+=sb[k]*bf16f(r);}lg[n]=(float)s;}
        float mx=lg[0];for(int i=1;i<NV;i++)if(lg[i]>mx)mx=lg[i];double sum=0;for(int i=0;i<NV;i++){float d=lg[i]-mx;d=d<-80?-80:d;lg[i]=expf(d);sum+=lg[i];}
        float rr=(float)rand()/RAND_MAX*(float)sum,acc=0;int tok=0;for(int i=0;i<NV;i++){acc+=lg[i];if(acc>=rr){tok=i;break;}}
        double mss=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();
        printf("  [%d] %d (%.0fms)\n",st,tok,mss);for(int i=0;i<H;i++)h[i]=bf16g(emb[tok*H+i]);sp++;}
    double ts=std::chrono::duration<double>(std::chrono::steady_clock::now()-tgs).count();
    printf("\n=== %.0f ms/tok ===\n",ts*1000/ng);
    delete nm;munmap(md,st.st_size);return 0;
}
