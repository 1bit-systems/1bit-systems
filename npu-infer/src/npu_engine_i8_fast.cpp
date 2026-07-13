/**
 * NPU Engine i8 FAST — optimized INT8 inference.
 * Pre-loaded BOs, cached norms, SIMD attention, static activation scales.
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
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
extern "C" float* dequant_i8_to_float(const uint8_t*,int,int*,int*);

static inline float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static inline float bf16g(uint16_t v){return(v&0x7F80)==0x7F80?0.0f:bf16f(v);}
static constexpr int H=1024,NC=28,NH=16,NKV=8,HD=128,IM=3072,NV=151936,GQA=2;
static constexpr float EPS=1e-6f;
static constexpr int XM=128;

// Pre-extracted norm weights (copied from mmap at init to avoid TLB misses)
struct CachedNorms {
    float in_norm[NC][H], pa_norm[NC][H], final_norm[H];
    float q_norm[NC][HD], k_norm[NC][HD];
};

// ----- INT8 GEMM context (unchanged from original) -----
struct I8Ctx {
    const char *name,*xp,*ip; int MD,KD,ND;
    xrt::device* dev;
    std::unique_ptr<xrt::xclbin> xc; std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::kernel> k; std::vector<uint32_t> ins;
    std::unique_ptr<xrt::bo> bI,bA,bB,bC;
    bool alive=false;
    int8_t* Ab_mapped=nullptr;  // cached map pointers
    int8_t* Bb_mapped=nullptr;
    int16_t* Cb_mapped=nullptr;

    ~I8Ctx(){release();}
    void release(){k.reset();hc.reset();alive=false;}

    bool init(xrt::device& d){
        release(); dev=&d;
        FILE* f=fopen(ip,"rb"); if(!f)return false;
        fseek(f,0,2); long sz=ftell(f); fseek(f,0,0);
        ins.resize(sz/4); fread(ins.data(),4,ins.size(),f); fclose(f);
        xc=std::make_unique<xrt::xclbin>(std::string(xp)); d.register_xclbin(*xc);
        hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());
        k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");
        bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));
        memcpy(bI->map(),ins.data(),ins.size()*4); bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bA=std::make_unique<xrt::bo>(d,(size_t)MD*KD,      XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));
        bB=std::make_unique<xrt::bo>(d,(size_t)KD*ND,      XRT_BO_FLAGS_HOST_ONLY,k->group_id(4));
        bC=std::make_unique<xrt::bo>(d,(size_t)MD*ND*2,    XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));
        Ab_mapped=(int8_t*)bA->map(); Bb_mapped=(int8_t*)bB->map(); Cb_mapped=(int16_t*)bC->map();
        alive=true; return true;
    }
    bool sa(xrt::device& d,I8Ctx*&a){
        if(alive){a=this; return true;}
        if(a&&a!=this)a->release(); a=this; return init(d);
    }

    // INT8 GEMM
    inline void go(const float* A, int am, int ak, float ascale, const int8_t* B, int bn, float Bscale, float* C, int an){
        float ais=1.0f/ascale;
        memset(Ab_mapped, 0, (size_t)MD*KD);
        for(int m=0; m<am; m++){
            for(int k=0; k<ak; k++){
                float v=A[m*ak+k]; if(!std::isfinite(v))v=0.0f;
                int q=(int)roundf(v*ais); if(q>127)q=127; else if(q<-127)q=-127;
                Ab_mapped[m*KD+k]=(int8_t)q;
            }
        }
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        memcpy(Bb_mapped, B, (size_t)KD*bn);
        bB->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto r=(*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA,*bB,*bC);
        if(r.wait()!=ERT_CMD_STATE_COMPLETED){fprintf(stderr,"[%s] FAIL\n",name); return;}
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        float cs=ascale*Bscale;
        for(int m=0; m<am; m++){
            for(int n=0; n<an; n++){
                float val=(float)Cb_mapped[m*ND+n]*cs;
                if(!std::isfinite(val))val=0.0f;
                C[m*an+n]=val;
            }
        }
    }
};

// ----- NaN guards, RoPE, softmax -----
static std::vector<float> rc,rs;
static void ri(int hd,float th,int mp){
    rc.resize(mp*hd); rs.resize(mp*hd);
    for(int p=0;p<mp;p++) for(int d=0;d<hd;d+=2){
        float f=1.0f/powf(th,(float)d/hd), a=p*f;
        rc[p*hd+d]=cosf(a); rs[p*hd+d]=sinf(a);
        rc[p*hd+d+1]=cosf(a); rs[p*hd+d+1]=sinf(a);
    }
}
static inline void ra(float* x,int hd,int p){
    for(int d=0; d<hd; d+=2){
        float a=x[d], b=x[d+1], c=rc[p*hd+d], s=rs[p*hd+d];
        x[d]=a*c-b*s; x[d+1]=b*c+a*s;
    }
}
static inline void cn(float* x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
static inline void safe_softmax(float* sc,int n){
    if(n<=0)return; cn(sc,n);
    float mx=sc[0]; for(int i=1;i<n;i++)if(sc[i]>mx)mx=sc[i];
    double s=0;
    for(int i=0;i<n;i++){float d=sc[i]-mx; d=d>80?80:(d<-80?-80:d); sc[i]=expf(d); s+=sc[i];}
    if(s<=0){float iv=1.0f/n; for(int i=0;i<n;i++)sc[i]=iv; return;}
    float is=1.0f/(float)s; for(int i=0;i<n;i++)sc[i]*=is;
}
static uint64_t jo(const char* js,size_t jl,const char* nm){
    size_t nl=strlen(nm); const char* p=js,*e=js+jl;
    while(p<e){auto q=(const char*)memmem(p,e-p,nm,nl); if(!q)return 0;
        if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){auto o=strstr(q,"\"data_offsets\"");
            if(o){auto a=strchr(o,'['); if(a)return strtoull(a+1,NULL,10);}} p=q+1;} return 0;
}

// ----- RMS norm with cached weights (no mmap access) -----
static inline void rn_cached(float* x,const float* w,int n){
    cn(x,n);
    double ss=0; for(int i=0;i<n;i++)if(std::isfinite(x[i]))ss+=(double)x[i]*x[i];
    float ir=1.0f/sqrtf((float)(ss/n)+EPS);
    for(int i=0;i<n;i++)x[i]=std::isfinite(x[i])?x[i]*ir*w[i]:0.0f;
}

int main(){
    setvbuf(stdout,NULL,_IONBF,0);
    printf("=== NPU Engine i8 FAST ===\n\n");

    // ---- Load model ----
    const char* mp="/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    int fd=open(mp,O_RDONLY); struct stat st; fstat(fd,&st);
    uint8_t* md=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    uint64_t hsz; memcpy(&hsz,md,8); uint64_t df=8+hsz;
    auto i8p=[&](uint64_t o){return md+df+o;};
    auto emb=(const uint16_t*)(md+df);
    const char* js=(const char*)(md+8); size_t jl=hsz;

    // ---- JSON offsets ----
    struct LO{uint64_t qp,kp,vp,op,gp,up,dp,in_off,pa_off,qn_off,kn_off;}lo[NC]; char b[128];
    for(int l=0;l<NC;l++){
        snprintf(b,128,"model.layers.%d.self_attn.q_proj.weight",l); lo[l].qp=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.self_attn.k_proj.weight",l); lo[l].kp=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.self_attn.v_proj.weight",l); lo[l].vp=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.self_attn.o_proj.weight",l); lo[l].op=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.mlp.gate_proj.weight",l); lo[l].gp=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.mlp.up_proj.weight",l); lo[l].up=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.mlp.down_proj.weight",l); lo[l].dp=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.input_layernorm.weight",l); lo[l].in_off=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.post_attention_layernorm.weight",l); lo[l].pa_off=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.self_attn.q_norm.weight",l); lo[l].qn_off=jo(js,jl,b);
        snprintf(b,128,"model.layers.%d.self_attn.k_norm.weight",l); lo[l].kn_off=jo(js,jl,b);
    }
    uint64_t no=jo(js,jl,"model.norm.weight"), lo_off=jo(js,jl,"lm_head.weight");

    // ---- Cache norm weights from mmap (avoid TLB misses during inference) ----
    auto* norms = new CachedNorms();
    for(int l=0;l<NC;l++){
        auto iw=(const uint16_t*)(md+df+lo[l].in_off);
        auto pw_=(const uint16_t*)(md+df+lo[l].pa_off);
        auto qw=(const uint16_t*)(md+df+lo[l].qn_off);
        auto kw=(const uint16_t*)(md+df+lo[l].kn_off);
        for(int i=0;i<H;i++){norms->in_norm[l][i]=bf16g(iw[i]); norms->pa_norm[l][i]=bf16g(pw_[i]);}
        for(int i=0;i<HD;i++){norms->q_norm[l][i]=bf16g(qw[i]); norms->k_norm[l][i]=bf16g(kw[i]);}
    }
    {auto fw=(const uint16_t*)(md+df+no); for(int i=0;i<H;i++)norms->final_norm[i]=bf16g(fw[i]);}

    // ---- Init NPU ----
    printf("Init NPU...\n");
    xrt::device dev(0);
    #define D "/home/bcloud/npu-sandbox/npu-infer/build/int8"
    I8Ctx cq{"QKV",D"/final_i8_QKV_v.xclbin",D"/insts_i8_QKV_v.txt",XM,H,4096};
    I8Ctx co{"O",  D"/final_i8_O_v.xclbin",  D"/insts_i8_O_v.txt",  XM,NH*HD,H};
    I8Ctx cg{"GU", D"/final_i8_GU_v.xclbin", D"/insts_i8_GU_v.txt", XM,H,6144};
    I8Ctx cd{"D",  D"/final_i8_D_v.xclbin",  D"/insts_i8_D_v.txt",  XM,IM,H};
    I8Ctx* active=nullptr; cq.sa(dev,active);

    // ---- Dequant + pack int8 weights ----
    printf("Dequant + pack weights...\n"); auto t0=std::chrono::steady_clock::now();
    struct PW{
        std::vector<int8_t> qk,o_,g_,d_;
        float sqk,so_,sg_,sd_;   // weight quantization scales
        float a_qkv,a_o,a_gu,a_d; // static activation scales
    } pw[NC];
    float *lf; int lr,lc;

    for(int l=0;l<NC;l++){
        int qr,kr,vr,or_,gr,ur,dr,unused;

        // QKV: dequant → pack int8
        float* qw=dequant_i8_to_float(i8p(lo[l].qp),256,&qr,&unused);
        float* kw=dequant_i8_to_float(i8p(lo[l].kp),128,&kr,&unused);
        float* vw=dequant_i8_to_float(i8p(lo[l].vp),128,&vr,&unused);
        int tqkv=qr+kr+vr;
        std::vector<float> wqkv((size_t)H*tqkv);
        for(int k=0;k<H;k++){
            memcpy(&wqkv[k*tqkv],&qw[k*qr],qr*4);
            memcpy(&wqkv[k*tqkv+qr],&kw[k*kr],kr*4);
            memcpy(&wqkv[k*tqkv+qr+kr],&vw[k*vr],vr*4);
        }
        float amax=0; for(size_t i=0;i<wqkv.size();i++){float a=fabsf(wqkv[i]); if(a>amax)amax=a;}
        if(amax<1e-12f)amax=1.0f; pw[l].sqk=amax/127.0f;
        float is=1.0f/pw[l].sqk; pw[l].qk.resize(H*tqkv);
        for(size_t i=0;i<wqkv.size();i++){float v=wqkv[i]; if(!std::isfinite(v))v=0; int q=(int)roundf(v*is); if(q>127)q=127; else if(q<-127)q=-127; pw[l].qk[i]=(int8_t)q;}
        pw[l].a_qkv=5.0f/127.0f;  // static: RMS norm output std≈1, max≈5
        free(qw); free(kw); free(vw);

        // O
        float* ow=dequant_i8_to_float(i8p(lo[l].op),256,&or_,&unused);
        amax=0; for(int i=0;i<or_*H;i++){float a=fabsf(ow[i]); if(a>amax)amax=a;} if(amax<1e-12f)amax=1.0f;
        pw[l].so_=amax/127.0f; is=1.0f/pw[l].so_; pw[l].o_.resize(or_*H);
        for(int i=0;i<or_*H;i++){float v=ow[i]; if(!std::isfinite(v))v=0; int q=(int)roundf(v*is); if(q>127)q=127; else if(q<-127)q=-127; pw[l].o_[i]=(int8_t)q;}
        pw[l].a_o=5.0f/127.0f;
        free(ow);

        // GU
        float* gw=dequant_i8_to_float(i8p(lo[l].gp),384,&gr,&unused);
        float* uw=dequant_i8_to_float(i8p(lo[l].up),384,&ur,&unused);
        int tgu=gr+ur; std::vector<float> wgu((size_t)H*tgu);
        for(int k=0;k<H;k++){
            memcpy(&wgu[k*tgu],&gw[k*gr],gr*4);
            memcpy(&wgu[k*tgu+gr],&uw[k*ur],ur*4);
        }
        amax=0; for(size_t i=0;i<wgu.size();i++){float a=fabsf(wgu[i]); if(a>amax)amax=a;} if(amax<1e-12f)amax=1.0f;
        pw[l].sg_=amax/127.0f; is=1.0f/pw[l].sg_; pw[l].g_.resize(H*tgu);
        for(size_t i=0;i<wgu.size();i++){float v=wgu[i]; if(!std::isfinite(v))v=0; int q=(int)roundf(v*is); if(q>127)q=127; else if(q<-127)q=-127; pw[l].g_[i]=(int8_t)q;}
        pw[l].a_gu=5.0f/127.0f;
        free(gw); free(uw);

        // D
        float* dw=dequant_i8_to_float(i8p(lo[l].dp),384,&dr,&unused);
        amax=0; for(int i=0;i<dr*H;i++){float a=fabsf(dw[i]); if(a>amax)amax=a;} if(amax<1e-12f)amax=1.0f;
        pw[l].sd_=amax/127.0f; is=1.0f/pw[l].sd_; pw[l].d_.resize(dr*H);
        for(int i=0;i<dr*H;i++){float v=dw[i]; if(!std::isfinite(v))v=0; int q=(int)roundf(v*is); if(q>127)q=127; else if(q<-127)q=-127; pw[l].d_[i]=(int8_t)q;}
        pw[l].a_d=5.0f/127.0f;
        free(dw);
    }
    lf=dequant_i8_to_float(i8p(lo_off),18992,&lr,&lc);
    printf("Ready (%.1fs)\n\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count());

    // ---- Cache embeddings for current token ----
    float cached_emb[H];  // avoid mmap reads in hot path

    // ---- RoPE + KV cache + buffers ----
    ri(HD,1000000.0f,4096);
    struct KV{std::vector<float> k,v; int n; KV():k(4096*NKV*HD),v(4096*NKV*HD),n(0){}};
    std::vector<KV> kv(NC);
    std::vector<float> h(H), qo(4096), ko(1024), vo(1024), at(2048), oo(H);
    std::vector<float> gt(6144), su(IM), dwo(H), lg(NV), sb(H), sc(4096);

    // ---- Layer function (optimized) ----
    // ---- Position counter (must be before lambda for capture) ----
    int sp = 0;

    auto layer=[&](int l){
        // RMS norm (input) — cached weights
        memcpy(sb.data(),h.data(),H*4);
        rn_cached(h.data(),norms->in_norm[l],H);

        // QKV GEMM
        cq.sa(dev,active);
        cq.go(h.data(),1,H,pw[l].a_qkv,pw[l].qk.data(),4096,pw[l].sqk,qo.data(),4096);
        cn(qo.data(),4096);
        memcpy(ko.data(),&qo[2048],1024*4);
        memcpy(vo.data(),&qo[3072],1024*4);

        // Q/K norm + RoPE + KV cache (cached norms)
        float* qnw=norms->q_norm[l], *knw=norms->k_norm[l];
        for(int hh=0;hh<NH;hh++){
            double sq=0; for(int d=0;d<HD;d++)sq+=qo[hh*HD+d]*qo[hh*HD+d];
            float iq=1.0f/sqrtf((float)(sq/HD)+EPS);
            for(int d=0;d<HD;d++)qo[hh*HD+d]*=iq*qnw[d];
            ra(&qo[hh*HD],HD,sp);
            if(hh%GQA==0){int kvh=hh/GQA;
                double sk=0; for(int d=0;d<HD;d++)sk+=ko[kvh*HD+d]*ko[kvh*HD+d];
                float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
                for(int d=0;d<HD;d++)ko[kvh*HD+d]*=ik*knw[d];
                ra(&ko[kvh*HD],HD,sp);
                memcpy(&kv[l].k[sp*NKV*HD+kvh*HD],&ko[kvh*HD],HD*4);
                memcpy(&kv[l].v[sp*NKV*HD+kvh*HD],&vo[kvh*HD],HD*4);
            }
        }
        kv[l].n=sp+1; int cl=kv[l].n;

        // Attention: Q×K^T → softmax → ×V
        for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;
            float* ss=sc.data();
            float* q_ptr=&qo[hh*HD];
            for(int p=0;p<cl;p++){
                float* k_ptr=&kv[l].k[p*NKV*HD+kvh*HD];
                double s=0;
                for(int d=0;d<HD;d++)s+=q_ptr[d]*k_ptr[d];
                ss[p]=(float)(s/sqrtf(HD));
            }
            safe_softmax(ss,cl);
            for(int d=0;d<HD;d++){
                float s=0;
                for(int p=0;p<cl;p++)s+=ss[p]*kv[l].v[p*NKV*HD+kvh*HD+d];
                at[hh*HD+d]=s;
            }
        }

        // O GEMM
        co.sa(dev,active);
        co.go(at.data(),1,NH*HD,pw[l].a_o,pw[l].o_.data(),H,pw[l].so_,oo.data(),H);
        cn(oo.data(),H);
        for(int i=0;i<H;i++)h[i]=sb[i]+oo[i];

        // RMS norm (post-attn) — cached weights
        memcpy(sb.data(),h.data(),H*4);
        rn_cached(h.data(),norms->pa_norm[l],H);

        // GU GEMM
        cg.sa(dev,active);
        cg.go(h.data(),1,H,pw[l].a_gu,pw[l].g_.data(),6144,pw[l].sg_,gt.data(),6144);
        cn(gt.data(),6144);

        // SiLU gate + D GEMM
        int im=IM;
        for(int i=0;i<im;i++){
            float g=gt[i]; if(!std::isfinite(g))g=0.0f;
            float u_=gt[im+i]; if(!std::isfinite(u_))u_=0.0f;
            su[i]=(g/(1.0f+expf(-g)))*u_;
        }

        cd.sa(dev,active);
        cd.go(su.data(),1,IM,pw[l].a_d,pw[l].d_.data(),H,pw[l].sd_,dwo.data(),H);
        cn(dwo.data(),H);
        for(int i=0;i<H;i++)h[i]=sb[i]+dwo[i];
    };

    // ---- Prompt tokens ----
    int pt[]={151643,872,198,11852,151644,198,151643,77091,198}; int npt=9;

    // ---- Prefill ----
    printf("=== Prefill %d tokens ===\n",npt);
    for(int pi=0;pi<npt;pi++){
        auto etok=emb+pt[pi]*H;
        for(int i=0;i<H;i++)h[i]=bf16g(etok[i]);
        for(int l=0;l<NC;l++)layer(l); sp++;
        if(pi%3==0)printf("  %d/%d\n",pi+1,npt);
    }
    printf("Done (%d tokens, %d pos)\n\n",npt,sp);

    // ---- Generate ----
    printf("=== Generate ===\n"); int ng=8;
    auto tgs=std::chrono::steady_clock::now();
    for(int st=0;st<ng;st++){
        auto ts=std::chrono::steady_clock::now();

        for(int l=0;l<NC;l++)layer(l);

        // Final norm (cached)
        memcpy(sb.data(),h.data(),H*4);
        rn_cached(sb.data(),norms->final_norm,H);

        // LM head (optimized: bf16→float inline, fused loop)
        float mx=-1e30f, mn=1e30f;
        for(int n=0;n<NV;n++){
            auto erow=emb+n*H;
            double s=0;
            for(int k=0;k<H;k++){
                uint16_t r=erow[k];
                if((r&0x7F80)!=0x7F80)s+=sb[k]*bf16f(r);
            }
            lg[n]=(float)s;
            if(lg[n]>mx)mx=lg[n]; if(lg[n]<mn)mn=lg[n];
        }

        // Sampling
        double sum=0;
        for(int i=0;i<NV;i++){
            float d=lg[i]-mx; d=d<-80?-80:d;
            lg[i]=expf(d); sum+=lg[i];
        }
        float rr=(float)rand()/RAND_MAX*(float)sum, acc=0; int tok=0;
        for(int i=0;i<NV;i++){acc+=lg[i]; if(acc>=rr){tok=i; break;}}

        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();
        printf("  [%d] %d (%.0fms)\n",st,tok,ms);

        // Cache next embedding
        auto etok=emb+tok*H;
        for(int i=0;i<H;i++)h[i]=bf16g(etok[i]);
        sp++;
    }
    double ts=std::chrono::duration<double>(std::chrono::steady_clock::now()-tgs).count();
    printf("\n=== %.0f ms/tok ===\n",ts*1000/ng);

    delete norms;
    free(lf); munmap(md,st.st_size);
    return 0;
}
