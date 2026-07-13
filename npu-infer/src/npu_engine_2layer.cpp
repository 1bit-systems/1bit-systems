// NPU Engine - 2-Layer Batch INT8 Decode (SIMPLIFIED)
// Uses N=8192 QKV_2layer xclbin to process 2 layers per QKV call.
// Saves 14 NPU calls vs un-fused (12.5% reduction).
//
// Build:
//   c++ -std=gnu++17 -O3 -mavx2 -mfma -mavx512f -ffast-math -march=native \
//       -funroll-loops -fno-math-errno -Iinclude \
//       -o build/npu_engine_2layer src/npu_engine_2layer.cpp build/deq_std.o -lxrt_coreutil

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
#include <string>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

extern "C" float* dequant_i8_to_float(const uint8_t*,int,int*,int*);
static inline float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static inline float bf16g(uint16_t v){return(v&0x7F80)==0x7F80?0.0f:bf16f(v);}

static const int H=1024,NC=28,NH=16,NKV=8,HD=128,IM=3072,NV=151936,GQA=2,AW=4,WQH=NH/AW,WKVH=NKV/AW;
static const int XM=128;
static const float EPS=1e-6f;
static inline void cn(float*x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
static inline void rn_c(float*x,const float*w,int n,int mm){
    for(int m=0;m<mm;m++){
        double ss=0;for(int i=0;i<n;i++){float v=x[m*n+i];if(std::isfinite(v))ss+=(double)v*v;}
        float ir=1.0f/sqrtf((float)(ss/n)+EPS);
        for(int i=0;i<n;i++){float v=x[m*n+i];x[m*n+i]=std::isfinite(v)?v*ir*w[i]:0.0f;}
    }
}
static std::vector<float>rc,rs;
static void ri(int hd,float th,int mp){rc.resize(mp*hd);rs.resize(mp*hd);
    for(int p=0;p<mp;p++)for(int d=0;d<hd;d+=2){
        float f=1.0f/powf(th,(float)d/hd),a=p*f;
        rc[p*hd+d]=cosf(a);rs[p*hd+d]=sinf(a);
        rc[p*hd+d+1]=cosf(a);rs[p*hd+d+1]=sinf(a);}}
static inline void ra(float*x,int hd,int p){for(int d=0;d<hd;d+=2){
    float a=x[d],b=x[d+1],c=rc[p*hd+d],s=rs[p*hd+d];x[d]=a*c-b*s;x[d+1]=b*c+a*s;}}

static uint64_t jo(const char*js,size_t jl,const char*nm){
    size_t nl=strlen(nm);const char*p=js,*e=js+jl;
    while(p<e){auto q=(const char*)memmem(p,e-p,nm,nl);if(!q)return 0;
        if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){auto o=strstr(q,"\"data_offsets\"");
            if(o){auto a=strchr(o,'[');if(a)return strtoull(a+1,NULL,10);}}p=q+1;}return 0;}

struct I8Ctx{
    int MD,KD,ND;
    std::unique_ptr<xrt::xclbin>xc;
    std::unique_ptr<xrt::hw_context>hc;
    std::unique_ptr<xrt::kernel>k;
    std::vector<uint32_t>ins;
    std::unique_ptr<xrt::bo>bI,bA,bC;
    std::vector<std::unique_ptr<xrt::bo>>layerB;
    int8_t*Am;int16_t*Cm;
    
    bool init(xrt::device&d,const char*xp,const char*ip,int gid_B){
        FILE*f=fopen(ip,"rb");if(!f)return false;
        fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);
        ins.resize(sz/4);
        if(fread(ins.data(),4,ins.size(),f)!=(size_t)ins.size()){fclose(f);return false;}
        fclose(f);
        xc=std::make_unique<xrt::xclbin>(std::string(xp));
        d.register_xclbin(*xc);
        hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());
        k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");
        bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));
        memcpy(bI->map(),ins.data(),ins.size()*4);
        bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bA=std::make_unique<xrt::bo>(d,(size_t)MD*KD,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));
        bC=std::make_unique<xrt::bo>(d,(size_t)MD*ND*2,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));
        Am=(int8_t*)bA->map();Cm=(int16_t*)bC->map();
        for(int l=0;l<NC;l++)
            layerB.emplace_back(std::make_unique<xrt::bo>(d,(size_t)KD*ND,XRT_BO_FLAGS_HOST_ONLY,k->group_id(gid_B)));
        return true;
    }
    void packB(int l,const float*w,int K,int N,float&sout){
        float amax=0;
        for(int i=0;i<K*N;i++){float a=fabsf(w[i]);if(std::isfinite(a)&&a>amax)amax=a;}
        if(amax<1e-12f)amax=1.0f;sout=amax/127.0f;float is=127.0f/amax;
        auto*Bm=(int8_t*)layerB[l]->map();
        for(int i=0;i<K*N;i++){float v=w[i];if(!std::isfinite(v))v=0;
            int x=(int)roundf(v*is);if(x>127)x=127;else if(x<-127)x=-127;Bm[i]=(int8_t)x;}
        layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    void packB_2layer(int l_pair,const float*w0,const float*w1,int K,int N_h,float&s0,float&s1){
        float a0=0,a1=0;
        for(int i=0;i<K*N_h;i++){float fa=fabsf(w0[i]);if(fa>a0)a0=fa;float fb=fabsf(w1[i]);if(fb>a1)a1=fb;}
        if(a0<1e-12f)a0=1.0f;s0=a0/127.0f;float i0=127.0f/a0;
        if(a1<1e-12f)a1=1.0f;s1=a1/127.0f;float i1=127.0f/a1;
        auto*Bm=(int8_t*)layerB[l_pair]->map();
        for(int k=0;k<K;k++)for(int n=0;n<N_h;n++){
            float v0=w0[k*N_h+n];if(!std::isfinite(v0))v0=0;
            int x=(int)roundf(v0*i0);if(x>127)x=127;else if(x<-127)x=-127;
            Bm[k*2*N_h+n]=(int8_t)x;
            float v1=w1[k*N_h+n];if(!std::isfinite(v1))v1=0;
            int y=(int)roundf(v1*i1);if(y>127)y=127;else if(y<-127)y=-127;
            Bm[k*2*N_h+N_h+n]=(int8_t)y;
        }
        layerB[l_pair]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    inline void go(int l,int pm,const float*A,int am,int ak,float as,float bs,float*C,int an){
        float ais=1.0f/as;memset(Am,0,(size_t)MD*KD);
        for(int m=0;m<am;m++)for(int k=0;k<ak;k++){
            float v=A[m*ak+k];if(!std::isfinite(v))v=0;
            int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;
            Am[m*KD+k]=(int8_t)q;}
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r=(*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA,*layerB[l],*bC);r.wait();
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);float cs=as*bs;
        for(int m=0;m<am;m++)for(int n=0;n<an;n++){
            float val=(float)Cm[m*ND+n]*cs;if(!std::isfinite(val))val=0;
            C[(pm+m)*an+n]=val;}
    }
    inline void go_2layer(int lp,int pm,const float*A,int am,int ak,
                          float as,float b0,float b1,float*C0,float*C1,int nh){
        float ais=1.0f/as;memset(Am,0,(size_t)MD*KD);
        for(int m=0;m<am;m++)for(int k=0;k<ak;k++){
            float v=A[m*ak+k];if(!std::isfinite(v))v=0;
            int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;
            Am[m*KD+k]=(int8_t)q;}
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);layerB[lp]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r=(*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA,*layerB[lp],*bC);r.wait();
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        for(int m=0;m<am;m++){
            for(int n=0;n<nh;n++){
                float v0=(float)Cm[m*ND+n]*as*b0;if(!std::isfinite(v0))v0=0;
                C0[(pm+m)*nh+n]=v0;}
            for(int n=0;n<nh;n++){
                float v1=(float)Cm[m*ND+nh+n]*as*b1;if(!std::isfinite(v1))v1=0;
                C1[(pm+m)*nh+n]=v1;}
        }
    }
};

struct AttnCPU{static void run(const float*Q,const float*K,const float*V,int nt,float*out){
    for(int h=0;h<WQH;h++){
        float sc[4096];float mx=-1e30f;
        for(int p=0;p<nt;p++){
            double s=0;for(int d=0;d<HD;d++)s+=Q[h*HD+d]*K[p*NKV*HD+(h/GQA)*HD+d];
            sc[p]=(float)(s/sqrtf((float)HD));if(sc[p]>mx)mx=sc[p];}
        double sum=0;for(int p=0;p<nt;p++){sc[p]=expf(sc[p]-mx);sum+=sc[p];}
        if(sum<=0)sum=1;float is=1.0f/(float)sum;
        for(int d=0;d<HD;d++){double s=0;for(int p=0;p<nt;p++)s+=sc[p]*is*V[p*NKV*HD+(h/GQA)*HD+d];out[h*HD+d]=(float)s;}}}};

struct KV{std::vector<float>k,v;int n;KV():k(4096*NKV*HD),v(4096*NKV*HD),n(0){}};

static void layer_qkv_norm_rope(int l,int M,int wpos,float*qo,float*ko,float*vo,
                                 KV&kv,const float*qn_w,const float*kn_w){
    for(int m=0;m<M;m++){
        float*q=&qo[m*4096],*k=&ko[m*1024],*v=&vo[m*1024];
        for(int hh=0;hh<NH;hh++){
            double sq=0;for(int d=0;d<HD;d++)sq+=q[hh*HD+d]*q[hh*HD+d];
            float iq=1.0f/sqrtf((float)(sq/HD)+EPS);
            for(int d=0;d<HD;d++)q[hh*HD+d]*=iq*qn_w[d];
            ra(&q[hh*HD],HD,wpos+m);
            if(hh%GQA==0){
                int kvh=hh/GQA;
                double sk=0;for(int d=0;d<HD;d++)sk+=k[kvh*HD+d]*k[kvh*HD+d];
                float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
                for(int d=0;d<HD;d++)k[kvh*HD+d]*=ik*kn_w[d];
                ra(&k[kvh*HD],HD,wpos+m);
                memcpy(&kv.k[(wpos+m)*NKV*HD+kvh*HD],&k[kvh*HD],HD*4);
                memcpy(&kv.v[(wpos+m)*NKV*HD+kvh*HD],&v[kvh*HD],HD*4);
            }
        }
    }
    kv.n=wpos+M;
}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"Usage: %s model.q4nx TOKEN1 ...\n",argv[0]);return 1;}
    const char*mp=argv[1];int M=argc-2;
    if(M>128){fprintf(stderr,"Max 128 tokens\n");return 1;}
    std::vector<int>tokens(M);
    for(int i=0;i<M;i++)tokens[i]=atoi(argv[2+i]);

    int fd=open(mp,O_RDONLY);struct stat st;fstat(fd,&st);
    uint8_t*md=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    uint64_t hsz;memcpy(&hsz,md,8);uint64_t df=8+hsz;
    auto i8p=[&](uint64_t o){return md+df+o;};
    auto emb=(const uint16_t*)(md+df);
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
        snprintf(b,128,"model.layers.%d.self_attn.k_norm.weight",l);lo[l].kn_off=jo(js,jl,b);}
    uint64_t no=jo(js,jl,"model.norm.weight");
    float in_n[NC][H],pa_n[NC][H],fin[H],qn_w[NC][HD],kn_w[NC][HD];
    for(int l=0;l<NC;l++){
        auto iw=(const uint16_t*)(md+df+lo[l].in_off),pw_=(const uint16_t*)(md+df+lo[l].pa_off);
        auto qw=(const uint16_t*)(md+df+lo[l].qn_off),kw=(const uint16_t*)(md+df+lo[l].kn_off);
        for(int i=0;i<H;i++){in_n[l][i]=bf16g(iw[i]);pa_n[l][i]=bf16g(pw_[i]);}
        for(int i=0;i<HD;i++){qn_w[l][i]=bf16g(qw[i]);kn_w[l][i]=bf16g(kw[i]);}}
    {auto fw=(const uint16_t*)(md+df+no);for(int i=0;i<H;i++)fin[i]=bf16g(fw[i]);}

    fprintf(stderr,"Init NPU...\n");
    xrt::device dev(0);
    I8Ctx cq2,co,cg,cd;
    cq2.MD=XM;cq2.KD=H;cq2.ND=8192;
    co.MD=XM;co.KD=NH*HD;co.ND=H;
    cg.MD=XM;cg.KD=H;cg.ND=6144;
    cd.MD=XM;cd.KD=IM;cd.ND=H;
#define D "/home/bcloud/npu-sandbox/npu-infer/build/int8"
    if(!cq2.init(dev,D"/final_i8_QKV_2layer.xclbin",D"/insts_i8_QKV_2layer.txt",4)||
       !co.init(dev,D"/final_i8_O_v.xclbin",D"/insts_i8_O_v.txt",4)||
       !cg.init(dev,D"/final_i8_GU_v.xclbin",D"/insts_i8_GU_v.txt",4)||
       !cd.init(dev,D"/final_i8_D_v.xclbin",D"/insts_i8_D_v.txt",4)){fprintf(stderr,"FAIL\n");return 1;}
#undef D

    fprintf(stderr,"Dequant+pack %d layers...\n",NC);
    auto tp=std::chrono::steady_clock::now();
    float ws[NC*4];
    for(int l=0;l<NC;l++){
        int qr,kr,vr,or_,gr,ur,dr,unused;
        float*qw=dequant_i8_to_float(i8p(lo[l].qp),256,&qr,&unused);
        float*kw_=dequant_i8_to_float(i8p(lo[l].kp),128,&kr,&unused);
        float*vw=dequant_i8_to_float(i8p(lo[l].vp),128,&vr,&unused);
        int tq=qr+kr+vr;
        std::vector<float>wq((size_t)H*tq);
        for(int k=0;k<H;k++){memcpy(&wq[k*tq],&qw[k*qr],qr*4);
            memcpy(&wq[k*tq+qr],&kw_[k*kr],kr*4);
            memcpy(&wq[k*tq+qr+kr],&vw[k*vr],vr*4);}
        
        if(l%2==0&&l+1<NC){
            int l2=l+1;
            float*qw2=dequant_i8_to_float(i8p(lo[l2].qp),256,&qr,&unused);
            float*kw2=dequant_i8_to_float(i8p(lo[l2].kp),128,&kr,&unused);
            float*vw2=dequant_i8_to_float(i8p(lo[l2].vp),128,&vr,&unused);
            std::vector<float>w2((size_t)H*tq);
            for(int k=0;k<H;k++){memcpy(&w2[k*tq],&qw2[k*qr],qr*4);
                memcpy(&w2[k*tq+qr],&kw2[k*kr],kr*4);
                memcpy(&w2[k*tq+qr+kr],&vw2[k*vr],vr*4);}
            cq2.packB_2layer(l/2,wq.data(),w2.data(),H,tq,ws[l*4+0],ws[l*4+4+0]);
            free(qw2);free(kw2);free(vw2);
        }else if(l%2==0){
            cq2.packB(l/2,wq.data(),H,tq,ws[l*4+0]);
        }
        free(qw);free(kw_);free(vw);
        
        float*ow=dequant_i8_to_float(i8p(lo[l].op),256,&or_,&unused);co.packB(l,ow,or_,H,ws[l*4+1]);free(ow);
        float*gw=dequant_i8_to_float(i8p(lo[l].gp),384,&gr,&unused);
        float*uw=dequant_i8_to_float(i8p(lo[l].up),384,&ur,&unused);
        int tg=gr+ur;std::vector<float>wg((size_t)H*tg);
        for(int k=0;k<H;k++){memcpy(&wg[k*tg],&gw[k*gr],gr*4);memcpy(&wg[k*tg+gr],&uw[k*ur],ur*4);}
        cg.packB(l,wg.data(),H,tg,ws[l*4+2]);free(gw);free(uw);
        float*dw=dequant_i8_to_float(i8p(lo[l].dp),384,&dr,&unused);cd.packB(l,dw,dr,H,ws[l*4+3]);free(dw);
    }
    fprintf(stderr,"  %.0fms\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tp).count());

    ri(HD,1000000.0f,M+128);
    std::vector<KV>kv(NC);
    std::vector<float>h(M*H),sb(M*H);
    std::vector<float>qo(M*4096),ko(M*1024),vo(M*1024);
    std::vector<float>at(M*NH*HD),oo(M*H),gt(M*6144),su(M*IM),dwo(M*H);
    // For 2-layer QKV output
    std::vector<float>qo_l1(M*4096);

    for(int m=0;m<M;m++)for(int i=0;i<H;i++)h[m*H+i]=bf16g(emb[tokens[m]*H+i]);

    fprintf(stderr,"Inference %d tok x %d layers (2-layer QKV)...\n",M,NC);
    auto ts=std::chrono::steady_clock::now();

    for(int l=0;l<NC;l+=2){
        int wpos=0;
        int l0=l,l1=l+1;
        
        // --- Layer 0: QKV via 2-layer call ---
        memcpy(sb.data(),h.data(),M*H*4);
        rn_c(h.data(),in_n[l0],H,M);
        
        // 2-layer QKV: produces both layers' QKV in one call
        memcpy(qo.data(),h.data(),M*4096*4);
        for(int pass=0;pass<M;pass+=XM){
            int bm=M-pass;if(bm>XM)bm=XM;
            cq2.go_2layer(l0/2,pass,&h[pass*H],bm,H,5.0f/127.0f,
                ws[l0*4+0],ws[l1*4+0],qo.data(),qo_l1.data(),4096);
        }
        for(int m=0;m<M;m++){cn(&qo[m*4096],4096);cn(&qo_l1[m*4096],4096);}
        for(int m=0;m<M;m++){
            memcpy(&ko[m*1024],&qo[m*4096+2048],4096);
            memcpy(&vo[m*1024],&qo[m*4096+3072],4096);
        }
        
        // Q/K norm + RoPE + KV cache for layer 0
        layer_qkv_norm_rope(l0,M,wpos,qo.data(),ko.data(),vo.data(),kv[l0],qn_w[l0],kn_w[l0]);
        
        // Attention for layer 0
        for(int m=0;m<M;m++){
            float*q=&qo[m*4096];float*o=&at[m*NH*HD];
            for(int w=0;w<AW;w++)AttnCPU::run(&q[w*WQH*HD],
                &kv[l0].k[w*WKVH*HD],&kv[l0].v[w*WKVH*HD],kv[l0].n,&o[w*WQH*HD]);
        }
        
        // O projection for layer 0
        for(int pass=0;pass<M;pass+=XM){
            int bm=M-pass;if(bm>XM)bm=XM;
            co.go(l0,pass,&at[pass*NH*HD],bm,NH*HD,5.0f/127.0f,ws[l0*4+1],oo.data(),H);
        }
        for(int m=0;m<M;m++){
            cn(&oo[m*H],H);
            for(int i=0;i<H;i++)h[m*H+i]=sb[m*H+i]+oo[m*H+i];
        }
        
        // MLP for layer 0
        memcpy(sb.data(),h.data(),M*H*4);
        rn_c(h.data(),pa_n[l0],H,M);
        for(int pass=0;pass<M;pass+=XM){
            int bm=M-pass;if(bm>XM)bm=XM;
            cg.go(l0,pass,&h[pass*H],bm,H,5.0f/127.0f,ws[l0*4+2],gt.data(),6144);
        }
        for(int m=0;m<M;m++){
            cn(&gt[m*6144],6144);
            for(int i=0;i<IM;i++){
                float gv=gt[m*6144+i];if(!std::isfinite(gv))gv=0;
                su[m*IM+i]=(gv/(1.0f+expf(-gv)))*gt[m*6144+IM+i];}
        }
        for(int pass=0;pass<M;pass+=XM){
            int bm=M-pass;if(bm>XM)bm=XM;
            cd.go(l0,pass,&su[pass*IM],bm,IM,5.0f/127.0f,ws[l0*4+3],dwo.data(),H);
        }
        for(int m=0;m<M;m++){
            cn(&dwo[m*H],H);
            for(int i=0;i<H;i++)h[m*H+i]=sb[m*H+i]+dwo[m*H+i];
        }
        wpos+=M;
        
        // --- Layer 1 (if exists): process QKV from qo_l1 ---
        if(l1<NC){
            memcpy(sb.data(),h.data(),M*H*4);
            rn_c(h.data(),in_n[l1],H,M);
            
            // Already have QKV from the 2-layer call (qo_l1)
            for(int m=0;m<M;m++){cn(&qo_l1[m*4096],4096);cn(&qo_l1[m*4096],4096);}
            for(int m=0;m<M;m++){
                memcpy(&ko[m*1024],&qo_l1[m*4096+2048],4096);
                memcpy(&vo[m*1024],&qo_l1[m*4096+3072],4096);
            }
            
            layer_qkv_norm_rope(l1,M,wpos,qo_l1.data(),ko.data(),vo.data(),kv[l1],qn_w[l1],kn_w[l1]);
            
            for(int m=0;m<M;m++){
                float*q=&qo_l1[m*4096];float*o=&at[m*NH*HD];
                for(int w=0;w<AW;w++)AttnCPU::run(&q[w*WQH*HD],
                    &kv[l1].k[w*WKVH*HD],&kv[l1].v[w*WKVH*HD],kv[l1].n,&o[w*WQH*HD]);
            }
            
            for(int pass=0;pass<M;pass+=XM){
                int bm=M-pass;if(bm>XM)bm=XM;
                co.go(l1,pass,&at[pass*NH*HD],bm,NH*HD,5.0f/127.0f,ws[l1*4+1],oo.data(),H);
            }
            for(int m=0;m<M;m++){
                cn(&oo[m*H],H);
                for(int i=0;i<H;i++)h[m*H+i]=sb[m*H+i]+oo[m*H+i];
            }
            
            memcpy(sb.data(),h.data(),M*H*4);
            rn_c(h.data(),pa_n[l1],H,M);
            for(int pass=0;pass<M;pass+=XM){
                int bm=M-pass;if(bm>XM)bm=XM;
                cg.go(l1,pass,&h[pass*H],bm,H,5.0f/127.0f,ws[l1*4+2],gt.data(),6144);
            }
            for(int m=0;m<M;m++){
                cn(&gt[m*6144],6144);
                for(int i=0;i<IM;i++){
                    float gv=gt[m*6144+i];if(!std::isfinite(gv))gv=0;
                    su[m*IM+i]=(gv/(1.0f+expf(-gv)))*gt[m*6144+IM+i];}
            }
            for(int pass=0;pass<M;pass+=XM){
                int bm=M-pass;if(bm>XM)bm=XM;
                cd.go(l1,pass,&su[pass*IM],bm,IM,5.0f/127.0f,ws[l1*4+3],dwo.data(),H);
            }
            for(int m=0;m<M;m++){
                cn(&dwo[m*H],H);
                for(int i=0;i<H;i++)h[m*H+i]=sb[m*H+i]+dwo[m*H+i];
            }
            }
        }
    }

    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();
    memcpy(sb.data(),h.data(),M*H*4);
    for(int m=0;m<M;m++)rn_c(&sb[m*H],fin,H,1);
    std::vector<float>lg(M*NV);
    for(int m=0;m<M;m++){
        for(int n=0;n<NV;n++){
            double s=0;for(int k=0;k<H;k++){
                uint16_t r=emb[n*H+k];
                if((r&0x7F80)!=0x7F80)s+=sb[m*H+k]*bf16f(r);}
            lg[m*NV+n]=(float)s;}
    }
    fprintf(stderr,"\n=== M=%d, %.0fms (%.1fms/tok) ===\n",M,ms,ms/M);
    for(int m=0;m<M;m++){
        float mx=lg[m*NV];for(int i=1;i<NV;i++)if(lg[m*NV+i]>mx)mx=lg[m*NV+i];
        double sum=0;float sm[8];int top8[8]={0};
        for(int i=0;i<8;i++){float d=lg[m*NV+i]-mx;if(d<-80)d=-80;sm[i]=expf(d);sum+=sm[i];top8[i]=i;}
        for(int t=0;t<8;t++){
            for(int i=t+1;i<NV;i++){
                float d=lg[m*NV+i]-mx;if(d<-80)d=-80;float ev=expf(d);
                if(ev>sm[t]){for(int k=t;k>0;k--){sm[k]=sm[k-1];top8[k]=top8[k-1];}
                    sm[0]=ev;top8[0]=i;break;}}}
        fprintf(stderr,"  token %3d: top8=[%d,%d,%d,%d,%d,%d,%d,%d]\n",
            m,top8[0],top8[1],top8[2],top8[3],top8[4],top8[5],top8[6],top8[7]);
    }
    munmap(md,st.st_size);return 0;
}
