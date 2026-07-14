/** NPU Engine v4 — No-redundant-sync. Weights synced once at startup, never re-DMA'd.
 *  Also: detailed per-GEMM profiling to find remaining overhead.
 *  Target: prove dispatch overhead, drive toward <100ms NPU GEMM. */
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
static constexpr float EPS=1e-6f; static constexpr int XM=128;
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

// F32 pre-converted embeddings for fast LM head
static std::vector<float> emb_f32;

// ===== I8Ctx v4: single weight sync at startup, never re-synced =====
struct I8Ctx{const char*name;int MD,KD,ND;std::unique_ptr<xrt::xclbin>xc;std::unique_ptr<xrt::hw_context>hc;std::unique_ptr<xrt::kernel>k;std::vector<uint32_t>ins;std::unique_ptr<xrt::bo>bI,bA,bC,layerB[NC];int8_t*Am;int16_t*Cm;
bool init(xrt::device&d,const char*xp,const char*ip,int gid_B){FILE*f=fopen(ip,"rb");if(!f)return false;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);xc=std::make_unique<xrt::xclbin>(std::string(xp));d.register_xclbin(*xc);hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));memcpy(bI->map(),ins.data(),ins.size()*4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);bA=std::make_unique<xrt::bo>(d,(size_t)MD*KD,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));bC=std::make_unique<xrt::bo>(d,(size_t)MD*ND*2,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));Am=(int8_t*)bA->map();Cm=(int16_t*)bC->map();for(int l=0;l<NC;l++)layerB[l]=std::make_unique<xrt::bo>(d,(size_t)KD*ND,XRT_BO_FLAGS_HOST_ONLY,k->group_id(gid_B));return true;}
// packB: quantize weights + sync ONCE. Weights are read-only from NPU perspective — never modified.
void packB(int l,const float*w,int K,int N,float&sout){float amax=0;for(int i=0;i<K*N;i++){float a=fabsf(w[i]);if(std::isfinite(a)&&a>amax)amax=a;}if(amax<1e-12f)amax=1.0f;sout=amax/127.0f;float is=127.0f/amax;auto*Bm=(int8_t*)layerB[l]->map();for(int i=0;i<K*N;i++){float v=w[i];if(!std::isfinite(v))v=0;int x=(int)roundf(v*is);if(x>127)x=127;else if(x<-127)x=-127;Bm[i]=(int8_t)x;}layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);}
// go v4: NO weight re-sync. Weights already on device from packB().
inline void go(int l,const float*A,int am,int ak,float ascale,float Bscale,float*C,int an){float ais=1.0f/ascale;memset(Am,0,(size_t)am*KD);for(int m=0;m<am;m++)for(int k=0;k<ak;k++){float v=A[m*ak+k];if(!std::isfinite(v))v=0;int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;Am[m*KD+k]=(int8_t)q;}bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);auto r=(*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA,*layerB[l],*bC);r.wait();bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);float cs=ascale*Bscale;for(int m=0;m<am;m++)for(int n=0;n<an;n++){float val=(float)Cm[m*ND+n]*cs;if(!std::isfinite(val))val=0;C[m*an+n]=val;}}
};

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int npt=9,ng=(argc>1)?atoi(argv[1]):8;
    printf("=== NPU Engine v4 — No Redundant Sync (M=%d) ===\n\n",npt+1);
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

    printf("Pre-convert emb f32...\n");auto t_emb=std::chrono::steady_clock::now();
    emb_f32.resize((size_t)NV*H);
    for(int n=0;n<NV;n++)for(int i=0;i<H;i++)emb_f32[(size_t)n*H+i]=bf16g(emb[n*H+i]);
    printf("  %.0fms\n\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_emb).count());

    printf("Init 4 GEMM...\n");xrt::device dev(0);
    #define D "int8" /* set $NPU_XCLBIN_DIR to override */
    I8Ctx cq{"QKV",XM,H,4096},co{"O",XM,NH*HD,H},cg{"GU",XM,H,6144},cd{"D",XM,IM,H};
    cq.init(dev,D"/final_i8_QKV_v.xclbin",D"/insts_i8_QKV_v.txt",4);
    co.init(dev,D"/final_i8_O_v.xclbin",  D"/insts_i8_O_v.txt",  4);
    cg.init(dev,D"/final_i8_GU_v.xclbin", D"/insts_i8_GU_v.txt", 4);
    cd.init(dev,D"/final_i8_D_v.xclbin",  D"/insts_i8_D_v.txt",  4);

    printf("Dequant+pack (weights synced ONCE)...\n");auto tp=std::chrono::steady_clock::now();
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
    std::vector<float>h(H),qo(4096),ko(1024),vo(1024),at(2048),oo(H),gt(6144),su(IM),dwo(H),sb(H),sc(4096),lg(NV);
    int sp=0;
    int pt[]={151643,872,198,11852,151644,198,151643,77091,198};

    // ===== PREFILL =====
    printf("=== Prefill %d (batched) ===\n",npt);
    auto t0=std::chrono::steady_clock::now();
    for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]=bf16g(emb[pt[pi]*H+i]);
    for(int l=0;l<NC;l++){
        for(int pi=0;pi<npt;pi++)rn_c(&h_b[pi*H],in_n[l],H);
        cq.go(l,h_b.data(),npt,H,dynamic_ascale(h_b.data(),npt*H),wsc[l].qk,qo_b.data(),4096);cn(qo_b.data(),npt*4096);
        float*qn=qn_w[l],*kn=kn_w[l];
        for(int pi=0;pi<npt;pi++){
            for(int hh=0;hh<NH;hh++){double s=0;for(int d=0;d<HD;d++)s+=qo_b[pi*NH*HD+hh*HD+d]*qo_b[pi*NH*HD+hh*HD+d];float iq=1.0f/sqrtf((float)(s/HD)+EPS);for(int d=0;d<HD;d++)qo_b[pi*NH*HD+hh*HD+d]*=iq*qn[d];ra(&qo_b[pi*NH*HD+hh*HD],HD,sp+pi);}
            for(int kvh=0;kvh<NKV;kvh++){float*ks=&qo_b[pi*4096+2048+kvh*HD],*vs=&qo_b[pi*4096+3072+kvh*HD];
                double sk=0;for(int d=0;d<HD;d++)sk+=ks[d]*ks[d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
                for(int d=0;d<HD;d++){ks[d]*=ik*kn[d];ra(ks,HD,sp+pi);}
                memcpy(&kv[l].k[(sp+pi)*NKV*HD+kvh*HD],ks,HD*4);memcpy(&kv[l].v[(sp+pi)*NKV*HD+kvh*HD],vs,HD*4);
            }
        }
        kv[l].n=sp+npt;int cl=kv[l].n;
        for(int pi=0;pi<npt;pi++){for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;int pl=sp+pi+1;std::vector<float>ss(cl);
            for(int p=0;p<pl;p++){double s=0;for(int d=0;d<HD;d++)s+=qo_b[pi*NH*HD+hh*HD+d]*kv[l].k[p*NKV*HD+kvh*HD+d];ss[p]=(float)(s/sqrtf(HD));}
            sm(ss.data(),pl);for(int d=0;d<HD;d++){float s=0;for(int p=0;p<pl;p++)s+=ss[p]*kv[l].v[p*NKV*HD+kvh*HD+d];at_b[pi*NH*HD+hh*HD+d]=s;}}}
        co.go(l,at_b.data(),npt,NH*HD,dynamic_ascale(at_b.data(),npt*NH*HD),wsc[l].o_,oo_b.data(),H);cn(oo_b.data(),npt*H);
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]+=oo_b[pi*H+i];
        for(int pi=0;pi<npt;pi++)rn_c(&h_b[pi*H],pa_n[l],H);
        cg.go(l,h_b.data(),npt,H,dynamic_ascale(h_b.data(),npt*H),wsc[l].g_,gt_b.data(),6144);cn(gt_b.data(),npt*6144);
        for(int pi=0;pi<npt;pi++){for(int i=0;i<IM;i++){float gv=gt_b[pi*6144+i];if(!std::isfinite(gv))gv=0;su_b[pi*IM+i]=(gv/(1.0f+expf(-gv)))*gt_b[pi*6144+IM+i];}}
        cd.go(l,su_b.data(),npt,IM,dynamic_ascale(su_b.data(),npt*IM),wsc[l].d_,dw_b.data(),H);cn(dw_b.data(),npt*H);
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]+=dw_b[pi*H+i];
    }
    sp+=npt;memcpy(h.data(),&h_b[(npt-1)*H],H*4);
    printf("Prefill: %.0fms (%.0f ms/tok)\n\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(),std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/npt);

    // ===== PROFILED DECODE =====
    printf("=== Profiled Decode v4 (%d tokens) ===\n",ng);
    printf("Each GEMM: q=quantize A, s=sync A in, k=kernel+wait, c=sync C out, d=dequant\n\n");
    double t_q=0,t_syncA=0,t_kern=0,t_syncC=0,t_dq=0;
    double t_attn_total=0,t_lm_total=0;

    for(int step=0;step<ng;step++){auto ts=std::chrono::steady_clock::now();
        for(int l=0;l<NC;l++){
            memcpy(sb.data(),h.data(),H*4);rn_c(h.data(),in_n[l],H);

            // === QKV GEMM ===
            {auto t0g=std::chrono::steady_clock::now();
            float*iA=h.data(); float ais=dynamic_ascale(iA,H);
            memset(cq.Am,0,(size_t)1*cq.KD);
            for(int k=0;k<H;k++){float v=iA[k];if(!std::isfinite(v))v=0;int q=(int)roundf(v/ais);if(q>127)q=127;else if(q<-127)q=-127;cq.Am[k]=(int8_t)q;}
            t_q+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0g).count();
            auto t1g=std::chrono::steady_clock::now();cq.bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            t_syncA+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t1g).count();
            auto t2g=std::chrono::steady_clock::now();auto r=(*cq.k)((unsigned)3,*cq.bI,(unsigned)cq.ins.size(),*cq.bA,*cq.layerB[l],*cq.bC);r.wait();
            t_kern+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t2g).count();
            auto t3g=std::chrono::steady_clock::now();cq.bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            t_syncC+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t3g).count();
            auto t4g=std::chrono::steady_clock::now();float cs=ais*wsc[l].qk;for(int n=0;n<4096;n++){float val=(float)cq.Cm[n]*cs;if(!std::isfinite(val))val=0;qo[n]=val;}
            t_dq+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t4g).count();}

            cn(qo.data(),4096);memcpy(ko.data(),&qo[2048],4096);memcpy(vo.data(),&qo[3072],4096);
            float*qn=qn_w[l],*kn=kn_w[l];

            auto ta0=std::chrono::steady_clock::now();
            for(int hh=0;hh<NH;hh++){double sq=0;for(int d=0;d<HD;d++)sq+=qo[hh*HD+d]*qo[hh*HD+d];float iq=1.0f/sqrtf((float)(sq/HD)+EPS);for(int d=0;d<HD;d++)qo[hh*HD+d]*=iq*qn[d];ra(&qo[hh*HD],HD,sp);
                if(hh%GQA==0){int kvh=hh/GQA;double sk=0;for(int d=0;d<HD;d++)sk+=ko[kvh*HD+d]*ko[kvh*HD+d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);for(int d=0;d<HD;d++)ko[kvh*HD+d]*=ik*kn[d];ra(&ko[kvh*HD],HD,sp);memcpy(&kv[l].k[sp*NKV*HD+kvh*HD],&ko[kvh*HD],HD*4);memcpy(&kv[l].v[sp*NKV*HD+kvh*HD],&vo[kvh*HD],HD*4);}}
            kv[l].n=sp+1;int cl=kv[l].n;
            for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;std::vector<float>sc(cl);
                for(int p=0;p<cl;p++){double s=0;for(int d=0;d<HD;d++)s+=qo[hh*HD+d]*kv[l].k[p*NKV*HD+kvh*HD+d];sc[p]=(float)(s/sqrtf(HD));}
                sm(sc.data(),cl);for(int d=0;d<HD;d++){float s=0;for(int p=0;p<cl;p++)s+=sc[p]*kv[l].v[p*NKV*HD+kvh*HD+d];at[hh*HD+d]=s;}
            }
            t_attn_total+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-ta0).count();

            // === O GEMM ===
            {auto t0g=std::chrono::steady_clock::now();
            float*iA=at.data(); float ais=dynamic_ascale(iA,NH*HD);
            memset(co.Am,0,(size_t)1*co.KD);
            for(int k=0;k<NH*HD;k++){float v=iA[k];if(!std::isfinite(v))v=0;int q=(int)roundf(v/ais);if(q>127)q=127;else if(q<-127)q=-127;co.Am[k]=(int8_t)q;}
            t_q+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0g).count();
            auto t1g=std::chrono::steady_clock::now();co.bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            t_syncA+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t1g).count();
            auto t2g=std::chrono::steady_clock::now();auto r=(*co.k)((unsigned)3,*co.bI,(unsigned)co.ins.size(),*co.bA,*co.layerB[l],*co.bC);r.wait();
            t_kern+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t2g).count();
            auto t3g=std::chrono::steady_clock::now();co.bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            t_syncC+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t3g).count();
            auto t4g=std::chrono::steady_clock::now();float cs=ais*wsc[l].o_;for(int n=0;n<H;n++){float val=(float)co.Cm[n]*cs;if(!std::isfinite(val))val=0;oo[n]=val;}
            t_dq+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t4g).count();}

            cn(oo.data(),H);for(int i=0;i<H;i++)h[i]=sb[i]+oo[i];

            memcpy(sb.data(),h.data(),H*4);rn_c(h.data(),pa_n[l],H);

            // === GU GEMM ===
            {auto t0g=std::chrono::steady_clock::now();
            float*iA=h.data(); float ais=dynamic_ascale(iA,H);
            memset(cg.Am,0,(size_t)1*cg.KD);
            for(int k=0;k<H;k++){float v=iA[k];if(!std::isfinite(v))v=0;int q=(int)roundf(v/ais);if(q>127)q=127;else if(q<-127)q=-127;cg.Am[k]=(int8_t)q;}
            t_q+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0g).count();
            auto t1g=std::chrono::steady_clock::now();cg.bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            t_syncA+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t1g).count();
            auto t2g=std::chrono::steady_clock::now();auto r=(*cg.k)((unsigned)3,*cg.bI,(unsigned)cg.ins.size(),*cg.bA,*cg.layerB[l],*cg.bC);r.wait();
            t_kern+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t2g).count();
            auto t3g=std::chrono::steady_clock::now();cg.bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            t_syncC+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t3g).count();
            auto t4g=std::chrono::steady_clock::now();float cs=ais*wsc[l].g_;for(int n=0;n<6144;n++){float val=(float)cg.Cm[n]*cs;if(!std::isfinite(val))val=0;gt[n]=val;}
            t_dq+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t4g).count();}

            cn(gt.data(),6144);for(int i=0;i<IM;i++){float gv=gt[i];if(!std::isfinite(gv))gv=0;su[i]=(gv/(1.0f+expf(-gv)))*gt[IM+i];}

            // === D GEMM ===
            {auto t0g=std::chrono::steady_clock::now();
            float*iA=su.data(); float ais=dynamic_ascale(iA,IM);
            memset(cd.Am,0,(size_t)1*cd.KD);
            for(int k=0;k<IM;k++){float v=iA[k];if(!std::isfinite(v))v=0;int q=(int)roundf(v/ais);if(q>127)q=127;else if(q<-127)q=-127;cd.Am[k]=(int8_t)q;}
            t_q+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0g).count();
            auto t1g=std::chrono::steady_clock::now();cd.bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            t_syncA+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t1g).count();
            auto t2g=std::chrono::steady_clock::now();auto r=(*cd.k)((unsigned)3,*cd.bI,(unsigned)cd.ins.size(),*cd.bA,*cd.layerB[l],*cd.bC);r.wait();
            t_kern+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t2g).count();
            auto t3g=std::chrono::steady_clock::now();cd.bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            t_syncC+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t3g).count();
            auto t4g=std::chrono::steady_clock::now();float cs=ais*wsc[l].d_;for(int n=0;n<H;n++){float val=(float)cd.Cm[n]*cs;if(!std::isfinite(val))val=0;dwo[n]=val;}
            t_dq+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t4g).count();}

            cn(dwo.data(),H);for(int i=0;i<H;i++)h[i]=sb[i]+dwo[i];
        }

        auto tlm0=std::chrono::steady_clock::now();
        memcpy(sb.data(),h.data(),H*4);rn_c(sb.data(),fin,H);
        float mx=-1e30f;
        for(int n=0;n<NV;n++){double s=0;const float*e=&emb_f32[(size_t)n*H];
            for(int k=0;k<H;k++)s+=(double)sb[k]*e[k];lg[n]=(float)s;if(lg[n]>mx)mx=lg[n];}
        double sum=0;for(int i=0;i<NV;i++){float d=lg[i]-mx;if(d<-80)d=-80;lg[i]=expf(d);sum+=lg[i];}
        float rr=(float)rand()/RAND_MAX*(float)sum,acc=0;int tok=0;for(int i=0;i<NV;i++){acc+=lg[i];if(acc>=rr){tok=i;break;}}
        t_lm_total+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tlm0).count();

        double mss=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();
        printf("  [%d] %d (%.0fms)\n",step,tok,mss);
        for(int i=0;i<H;i++)h[i]=emb_f32[(size_t)tok*H+i];sp++;
    }

    int n_ge=(ng*NC*4); // ng tokens × 28 layers × 4 GEMMs
    printf("\n=== Per-GEMM Dispatch Profile (v4, no weight re-sync) ===\n");
    printf("  Quantize A:   %.0f μs/call (%.1f ms total)\n",t_q/n_ge,t_q/1000);
    printf("  Sync A→NPU:   %.0f μs/call (%.1f ms total)\n",t_syncA/n_ge,t_syncA/1000);
    printf("  Kernel+wait:  %.0f μs/call (%.1f ms total)\n",t_kern/n_ge,t_kern/1000);
    printf("  Sync C←NPU:   %.0f μs/call (%.1f ms total)\n",t_syncC/n_ge,t_syncC/1000);
    printf("  Dequant C:    %.0f μs/call (%.1f ms total)\n",t_dq/n_ge,t_dq/1000);
    double t_op=t_q+t_syncA+t_kern+t_syncC+t_dq;
    printf("  ─────────────────────────────────────\n");
    printf("  Total GEMM:   %.0f μs/call (%.1f ms = %.0f%% of decode)\n",t_op/n_ge,t_op/1000,100*t_op/(t_op+t_attn_total*ng/1000));
    printf("  NPU attention: %.0f μs/layer (%.1f ms total)\n",t_attn_total/(ng*NC),t_attn_total/1000);
    printf("  LM head:      %.0f ms/token\n",t_lm_total/ng);
    printf("  TOTAL/token:  %.0f ms\n",(t_op+t_attn_total+t_lm_total*1000)/ng/1000);
    munmap(md,st.st_size);return 0;
}
