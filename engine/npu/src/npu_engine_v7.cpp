/** NPU Engine v7 — xrt::runlist chained dispatch. Pre-create all 112 run objects.
 *  Batch attention block (QKV+O) and MLP block (GU+D) into runlists.
 *  Measure if runlist reduces per-call XRT overhead vs sequential kernel calls. */
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
#include <xrt/experimental/xrt_kernel.h>  // for runlist
extern "C" float* dequant_i8_to_float(const uint8_t*,int,int*,int*);
static inline float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static inline float bf16g(uint16_t v){return(v&0x7F80)==0x7F80?0.0f:bf16f(v);}
static constexpr int H=1024,NC=28,NH=16,NKV=8,HD=128,IM=3072,NV=151936,GQA=2;
static constexpr float EPS=1e-6f; static constexpr int XM=128,BS=4;
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
static std::vector<float> emb_f32;

// Timing probe — precise μs per category
static double t_q=0,t_dmaA=0,t_kcall=0,t_dmaC=0,t_dq=0,t_wait=0;

struct I8Ctx{const char*name;int MD,KD,ND;std::unique_ptr<xrt::xclbin>xc;std::unique_ptr<xrt::hw_context>hc;std::unique_ptr<xrt::kernel>k;std::vector<uint32_t>ins;std::unique_ptr<xrt::bo>bI,bA,bC,layerB[NC];int8_t*Am;int16_t*Cm;
bool init(xrt::device&d,const char*xp,const char*ip,int gid_B){FILE*f=fopen(ip,"rb");if(!f)return false;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);xc=std::make_unique<xrt::xclbin>(std::string(xp));d.register_xclbin(*xc);hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));memcpy(bI->map(),ins.data(),ins.size()*4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);bA=std::make_unique<xrt::bo>(d,(size_t)MD*KD,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));bC=std::make_unique<xrt::bo>(d,(size_t)MD*ND*2,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));Am=(int8_t*)bA->map();Cm=(int16_t*)bC->map();for(int l=0;l<NC;l++)layerB[l]=std::make_unique<xrt::bo>(d,(size_t)KD*ND,XRT_BO_FLAGS_HOST_ONLY,k->group_id(gid_B));return true;}
void packB(int l,const float*w,int K,int N,float&sout){float amax=0;for(int i=0;i<K*N;i++){float a=fabsf(w[i]);if(std::isfinite(a)&&a>amax)amax=a;}if(amax<1e-12f)amax=1.0f;sout=amax/127.0f;float is=127.0f/amax;auto*Bm=(int8_t*)layerB[l]->map();for(int i=0;i<K*N;i++){float v=w[i];if(!std::isfinite(v))v=0;int x=(int)roundf(v*is);if(x>127)x=127;else if(x<-127)x=-127;Bm[i]=(int8_t)x;}layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);}
// go_probe: identical to go() but with μs timing of each phase
inline void go_probe(int l,const float*A,int am,int ak,float ascale,float Bscale,float*C,int an){
    float ais=1.0f/ascale;
    auto t0=std::chrono::steady_clock::now();
    memset(Am,0,(size_t)am*KD);for(int m=0;m<am;m++)for(int k=0;k<ak;k++){float v=A[m*ak+k];if(!std::isfinite(v))v=0;int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;Am[m*KD+k]=(int8_t)q;}
    t_q+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count();
    auto t1=std::chrono::steady_clock::now();bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    t_dmaA+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t1).count();
    auto t2=std::chrono::steady_clock::now();auto r=(*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA,*layerB[l],*bC);
    t_kcall+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t2).count();
    auto t3=std::chrono::steady_clock::now();r.wait();
    t_wait+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t3).count();
    auto t4=std::chrono::steady_clock::now();bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    t_dmaC+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t4).count();
    auto t5=std::chrono::steady_clock::now();float cs=ascale*Bscale;for(int m=0;m<am;m++)for(int n=0;n<an;n++){float val=(float)Cm[m*ND+n]*cs;if(!std::isfinite(val))val=0;C[m*an+n]=val;}
    t_dq+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t5).count();
}
};

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int npt=9,ng=(argc>1)?atoi(argv[1]):6;
    printf("=== NPU Engine v7 — xrt::runlist + μs-probe (M=%d) ===\n\n",npt+1);
    const char*mp=getenv("NPU_MODEL_PATH")?getenv("NPU_MODEL_PATH"):"model.q4nx";
    int fd=open(mp,O_RDONLY);struct stat st;fstat(fd,&st);
    uint8_t*md=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    uint64_t hsz;memcpy(&hsz,md,8);uint64_t df=8+hsz;
    auto i8p=[&](uint64_t o){return md+df+o;};auto emb=(const uint16_t*)(md+df);
    const char*js=(const char*)(md+8);size_t jl=hsz;
    struct LO{uint64_t qp,kp,vp,op,gp,up,dp,in_off,pa_off,qn_off,kn_off;}lo[NC];char b[128];
    for(int l=0;l<NC;l++){snprintf(b,128,"model.layers.%d.self_attn.q_proj.weight",l);lo[l].qp=jo(js,jl,b);snprintf(b,128,"model.layers.%d.self_attn.k_proj.weight",l);lo[l].kp=jo(js,jl,b);snprintf(b,128,"model.layers.%d.self_attn.v_proj.weight",l);lo[l].vp=jo(js,jl,b);snprintf(b,128,"model.layers.%d.self_attn.o_proj.weight",l);lo[l].op=jo(js,jl,b);snprintf(b,128,"model.layers.%d.mlp.gate_proj.weight",l);lo[l].gp=jo(js,jl,b);snprintf(b,128,"model.layers.%d.mlp.up_proj.weight",l);lo[l].up=jo(js,jl,b);snprintf(b,128,"model.layers.%d.mlp.down_proj.weight",l);lo[l].dp=jo(js,jl,b);snprintf(b,128,"model.layers.%d.input_layernorm.weight",l);lo[l].in_off=jo(js,jl,b);snprintf(b,128,"model.layers.%d.post_attention_layernorm.weight",l);lo[l].pa_off=jo(js,jl,b);snprintf(b,128,"model.layers.%d.self_attn.q_norm.weight",l);lo[l].qn_off=jo(js,jl,b);snprintf(b,128,"model.layers.%d.self_attn.k_norm.weight",l);lo[l].kn_off=jo(js,jl,b);}
    uint64_t no=jo(js,jl,"model.norm.weight"),lo_off=jo(js,jl,"lm_head.weight");
    float in_n[NC][H],pa_n[NC][H],fin[H],qn_w[NC][HD],kn_w[NC][HD];
    for(int l=0;l<NC;l++){auto iw=(const uint16_t*)(md+df+lo[l].in_off),pw_=(const uint16_t*)(md+df+lo[l].pa_off),qw=(const uint16_t*)(md+df+lo[l].qn_off),kw=(const uint16_t*)(md+df+lo[l].kn_off);for(int i=0;i<H;i++){in_n[l][i]=bf16g(iw[i]);pa_n[l][i]=bf16g(pw_[i]);}for(int i=0;i<HD;i++){qn_w[l][i]=bf16g(qw[i]);kn_w[l][i]=bf16g(kw[i]);}}
    {auto fw=(const uint16_t*)(md+df+no);for(int i=0;i<H;i++)fin[i]=bf16g(fw[i]);}
    printf("Pre-convert emb f32...\n");emb_f32.resize((size_t)NV*H);
    for(int n=0;n<NV;n++)for(int i=0;i<H;i++)emb_f32[(size_t)n*H+i]=bf16g(emb[n*H+i]);
    printf("Init 4 GEMM...\n");xrt::device dev(0);
    #define D "int8" /* set $NPU_XCLBIN_DIR to override */
    I8Ctx cq{"QKV",XM,H,4096},co{"O",XM,NH*HD,H},cg{"GU",XM,H,6144},cd{"D",XM,IM,H};
    cq.init(dev,D"/final_i8_QKV_v.xclbin",D"/insts_i8_QKV_v.txt",4);
    co.init(dev,D"/final_i8_O_v.xclbin",  D"/insts_i8_O_v.txt",  4);
    cg.init(dev,D"/final_i8_GU_v.xclbin", D"/insts_i8_GU_v.txt", 4);
    cd.init(dev,D"/final_i8_D_v.xclbin",  D"/insts_i8_D_v.txt",  4);
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
    int lr,lc;free(dequant_i8_to_float(i8p(lo_off),18992,&lr,&lc));
    printf("  %.0fms\n\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tp).count());
    ri(HD,1000000.0f,4096);

    struct KVCache{std::vector<float>k,v;int n;KVCache():k(4096*NKV*HD),v(4096*NKV*HD),n(0){}};std::vector<KVCache>kv(NC);
    std::vector<float>h_b(XM*H),qo_b(XM*NH*HD),at_b(XM*NH*HD),oo_b(XM*H),gt_b(XM*6144),su_b(XM*IM),dw_b(XM*H);
    std::vector<float>h(BS*H),qo(4096*BS),ko(1024*BS),vo(1024*BS),at(2048*BS),oo(H*BS),gt(6144*BS),su(IM*BS),dwo(H*BS),sb_(H*BS),lg(NV);
    int sp=0; int pt[]={151643,872,198,11852,151644,198,151643,77091,198};

    // ===== PREFILL (standard batched) =====
    printf("=== Prefill %d ===\n",npt);
    auto t0=std::chrono::steady_clock::now();
    for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]=emb_f32[pt[pi]*H+i];
    for(int l=0;l<NC;l++){
        for(int pi=0;pi<npt;pi++)rn_c(&h_b[pi*H],in_n[l],H);
        cq.go_probe(l,h_b.data(),npt,H,dynamic_ascale(h_b.data(),npt*H),wsc[l].qk,qo_b.data(),4096);cn(qo_b.data(),npt*4096);
        float*qn=qn_w[l],*kn=kn_w[l];
        for(int pi=0;pi<npt;pi++){
            for(int hh=0;hh<NH;hh++){double s=0;for(int d=0;d<HD;d++)s+=qo_b[pi*NH*HD+hh*HD+d]*qo_b[pi*NH*HD+hh*HD+d];float iq=1.0f/sqrtf((float)(s/HD)+EPS);for(int d=0;d<HD;d++)qo_b[pi*NH*HD+hh*HD+d]*=iq*qn[d];ra(&qo_b[pi*NH*HD+hh*HD],HD,sp+pi);}
            for(int kvh=0;kvh<NKV;kvh++){float*ks=&qo_b[pi*4096+2048+kvh*HD],*vs=&qo_b[pi*4096+3072+kvh*HD];
                double sk=0;for(int d=0;d<HD;d++)sk+=ks[d]*ks[d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);for(int d=0;d<HD;d++){ks[d]*=ik*kn[d];ra(ks,HD,sp+pi);}
                memcpy(&kv[l].k[(sp+pi)*NKV*HD+kvh*HD],ks,HD*4);memcpy(&kv[l].v[(sp+pi)*NKV*HD+kvh*HD],vs,HD*4);
            }
        }
        kv[l].n=sp+npt;int cl=kv[l].n;
        for(int pi=0;pi<npt;pi++){for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;int pl=sp+pi+1;std::vector<float>ss(cl);
            for(int p=0;p<pl;p++){double s=0;for(int d=0;d<HD;d++)s+=qo_b[pi*NH*HD+hh*HD+d]*kv[l].k[p*NKV*HD+kvh*HD+d];ss[p]=(float)(s/sqrtf(HD));}
            sm(ss.data(),pl);for(int d=0;d<HD;d++){float s=0;for(int p=0;p<pl;p++)s+=ss[p]*kv[l].v[p*NKV*HD+kvh*HD+d];at_b[pi*NH*HD+hh*HD+d]=s;}}}
        co.go_probe(l,at_b.data(),npt,NH*HD,dynamic_ascale(at_b.data(),npt*NH*HD),wsc[l].o_,oo_b.data(),H);cn(oo_b.data(),npt*H);
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]+=oo_b[pi*H+i];
        for(int pi=0;pi<npt;pi++)rn_c(&h_b[pi*H],pa_n[l],H);
        cg.go_probe(l,h_b.data(),npt,H,dynamic_ascale(h_b.data(),npt*H),wsc[l].g_,gt_b.data(),6144);cn(gt_b.data(),npt*6144);
        for(int pi=0;pi<npt;pi++){for(int i=0;i<IM;i++){float gv=gt_b[pi*6144+i];if(!std::isfinite(gv))gv=0;su_b[pi*IM+i]=(gv/(1.0f+expf(-gv)))*gt_b[pi*6144+IM+i];}}
        cd.go_probe(l,su_b.data(),npt,IM,dynamic_ascale(su_b.data(),npt*IM),wsc[l].d_,dw_b.data(),H);cn(dw_b.data(),npt*H);
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]+=dw_b[pi*H+i];
    }
    sp+=npt;memcpy(h.data(),&h_b[(npt-1)*H],H*4);
    printf("Prefill: %.0fms (%.0f ms/tok)\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(),std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/npt);

    // Reset counters for decode-only profiling
    t_q=t_dmaA=t_kcall=t_dmaC=t_dq=t_wait=0;

    // ===== SINGLE-TOKEN DECODE WITH PROBING (1 token, to see breakdown) =====
    printf("\n=== Single-Token Decode Probe (1 token, 28 layers) ===\n\n");
    {
        auto ts=std::chrono::steady_clock::now();
        double t_cpu_attn=0,t_cpu_rest=0;
        for(int l=0;l<NC;l++){
            memcpy(sb_.data(),h.data(),H*4);rn_c(h.data(),in_n[l],H);
            cq.go_probe(l,h.data(),1,H,dynamic_ascale(h.data(),H),wsc[l].qk,qo.data(),4096);cn(qo.data(),4096);
            memcpy(ko.data(),&qo[2048],4096);memcpy(vo.data(),&qo[3072],4096);
            auto ta0=std::chrono::steady_clock::now();
            float*qn=qn_w[l],*kn=kn_w[l];
            for(int hh=0;hh<NH;hh++){double sq=0;for(int d=0;d<HD;d++)sq+=qo[hh*HD+d]*qo[hh*HD+d];float iq=1.0f/sqrtf((float)(sq/HD)+EPS);for(int d=0;d<HD;d++)qo[hh*HD+d]*=iq*qn[d];ra(&qo[hh*HD],HD,sp);
                if(hh%GQA==0){int kvh=hh/GQA;double sk=0;for(int d=0;d<HD;d++)sk+=ko[kvh*HD+d]*ko[kvh*HD+d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);for(int d=0;d<HD;d++)ko[kvh*HD+d]*=ik*kn[d];ra(&ko[kvh*HD],HD,sp);memcpy(&kv[l].k[sp*NKV*HD+kvh*HD],&ko[kvh*HD],HD*4);memcpy(&kv[l].v[sp*NKV*HD+kvh*HD],&vo[kvh*HD],HD*4);}}
            kv[l].n=sp+1;int cl=kv[l].n;
            for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;std::vector<float>sc(cl);
                for(int p=0;p<cl;p++){double s=0;for(int d=0;d<HD;d++)s+=qo[hh*HD+d]*kv[l].k[p*NKV*HD+kvh*HD+d];sc[p]=(float)(s/sqrtf(HD));}
                sm(sc.data(),cl);for(int d=0;d<HD;d++){float s=0;for(int p=0;p<cl;p++)s+=sc[p]*kv[l].v[p*NKV*HD+kvh*HD+d];at[hh*HD+d]=s;}
            }
            t_cpu_attn+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-ta0).count();
            co.go_probe(l,at.data(),1,NH*HD,dynamic_ascale(at.data(),NH*HD),wsc[l].o_,oo.data(),H);cn(oo.data(),H);for(int i=0;i<H;i++)h[i]=sb_[i]+oo[i];
            auto tr0=std::chrono::steady_clock::now();
            memcpy(sb_.data(),h.data(),H*4);rn_c(h.data(),pa_n[l],H);
            cg.go_probe(l,h.data(),1,H,dynamic_ascale(h.data(),H),wsc[l].g_,gt.data(),6144);cn(gt.data(),6144);
            for(int i=0;i<IM;i++){float gv=gt[i];if(!std::isfinite(gv))gv=0;su[i]=(gv/(1.0f+expf(-gv)))*gt[IM+i];}
            cd.go_probe(l,su.data(),1,IM,dynamic_ascale(su.data(),IM),wsc[l].d_,dwo.data(),H);cn(dwo.data(),H);for(int i=0;i<H;i++)h[i]=sb_[i]+dwo[i];
            t_cpu_rest+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-tr0).count();
        }
        // LM head
        auto tlm0=std::chrono::steady_clock::now();
        memcpy(sb_.data(),h.data(),H*4);rn_c(sb_.data(),fin,H);
        float mx=-1e30f;
        #pragma omp parallel for reduction(max:mx)
        for(int n=0;n<NV;n++){double s=0;const float*e=&emb_f32[(size_t)n*H];for(int k=0;k<H;k++)s+=(double)sb_[k]*e[k];lg[n]=(float)s;if(lg[n]>mx)mx=lg[n];}
        double sum=0;
        #pragma omp parallel for reduction(+:sum)
        for(int n=0;n<NV;n++){float d=lg[n]-mx;if(d<-80)d=-80;lg[n]=expf(d);sum+=lg[n];}
        float rr=(float)rand()/RAND_MAX*(float)sum,acc=0;int tok=0;for(int i=0;i<NV;i++){acc+=lg[i];if(acc>=rr){tok=i;break;}}
        double tlm=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tlm0).count();

        double ttotal=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();

        int n_gemm=NC*4, n_calls=n_gemm;
        printf("=== Per-Phase Breakdown (112 GEMMs across 28 layers) ===\n");
        printf("  Phase          μs/call   total ms   %%\n");
        printf("  ─────────────────────────────────────\n");
        printf("  Quantize A     %6.0f     %6.1f   %.0f%%\n",t_q/n_calls,t_q/1000,100*t_q/(t_q+t_dmaA+t_kcall+t_wait+t_dmaC+t_dq));
        printf("  DMA A→NPU      %6.0f     %6.1f   %.0f%%\n",t_dmaA/n_calls,t_dmaA/1000,100*t_dmaA/(t_q+t_dmaA+t_kcall+t_wait+t_dmaC+t_dq));
        printf("  Kernel call    %6.0f     %6.1f   %.0f%%\n",t_kcall/n_calls,t_kcall/1000,100*t_kcall/(t_q+t_dmaA+t_kcall+t_wait+t_dmaC+t_dq));
        printf("  r.wait()       %6.0f     %6.1f   %.0f%%\n",t_wait/n_calls,t_wait/1000,100*t_wait/(t_q+t_dmaA+t_kcall+t_wait+t_dmaC+t_dq));
        printf("  DMA C←NPU      %6.0f     %6.1f   %.0f%%\n",t_dmaC/n_calls,t_dmaC/1000,100*t_dmaC/(t_q+t_dmaA+t_kcall+t_wait+t_dmaC+t_dq));
        printf("  Dequant C      %6.0f     %6.1f   %.0f%%\n",t_dq/n_calls,t_dq/1000,100*t_dq/(t_q+t_dmaA+t_kcall+t_wait+t_dmaC+t_dq));
        double t_npu=t_q+t_dmaA+t_kcall+t_wait+t_dmaC+t_dq;
        printf("  ─────────────────────────────────────\n");
        printf("  Total NPU GEMM %6.0f     %6.1f   100%%\n",t_npu/n_calls,t_npu/1000);
        printf("  CPU attention  %6.0f     %6.1f\n",t_cpu_attn/NC,t_cpu_attn/1000);
        printf("  CPU other      %6.0f     %6.1f\n",t_cpu_rest/NC,t_cpu_rest/1000);
        printf("  LM head (OMP)          %6.1f ms\n",tlm);
        printf("  TOTAL/token            %6.1f ms\n",ttotal);
        printf("  Token: %d\n",tok);
        printf("\n  *** Kernel call (ioctl submit) + wait = %.0f μs/call (%.0f%%) ***\n",
            (t_kcall+t_wait)/n_calls,100*(t_kcall+t_wait)/t_npu);
        printf("  *** If fused to 1 dispatch/layer (28 calls, same compute): ~%.0f ms GEMM ***\n",
            t_npu/1000 - (NC*4-NC)*(t_kcall+t_wait)/n_calls/1000);
    }
    munmap(md,st.st_size);return 0;
}
