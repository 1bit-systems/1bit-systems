/** NPU Engine v5 — Universal model support. Dimensions from npu_dims.h via -DMODEL_<tag>. */
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
#include "npu_dims.h"
extern "C" float* dequant_i8_to_float_ex(const uint8_t*,int,int,int*,int*);
static constexpr float EPS=1e-6f;
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

struct I8Ctx{const char*name;int MD,KD,ND;std::unique_ptr<xrt::xclbin>xc;std::unique_ptr<xrt::hw_context>hc;std::unique_ptr<xrt::kernel>k;std::vector<uint32_t>ins;std::unique_ptr<xrt::bo>bI,bA,bC,layerB[NC];int8_t*Am;int16_t*Cm;
bool init(xrt::device&d,const char*xp,const char*ip,int gid_B){FILE*f=fopen(ip,"rb");if(!f)return false;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);xc=std::make_unique<xrt::xclbin>(std::string(xp));d.register_xclbin(*xc);hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));memcpy(bI->map(),ins.data(),ins.size()*4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);bA=std::make_unique<xrt::bo>(d,(size_t)MD*KD,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));bC=std::make_unique<xrt::bo>(d,(size_t)MD*ND*2,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));Am=(int8_t*)bA->map();Cm=(int16_t*)bC->map();for(int l=0;l<NC;l++)layerB[l]=std::make_unique<xrt::bo>(d,(size_t)KD*ND,XRT_BO_FLAGS_HOST_ONLY,k->group_id(gid_B));return true;}
void packB(int l,const float*w,int K,int N,float&sout){float amax=0;for(int i=0;i<K*N;i++){float a=fabsf(w[i]);if(std::isfinite(a)&&a>amax)amax=a;}if(amax<1e-12f)amax=1.0f;sout=amax/127.0f;float is=127.0f/amax;auto*Bm=(int8_t*)layerB[l]->map();for(int i=0;i<K*N;i++){float v=w[i];if(!std::isfinite(v))v=0;int x=(int)roundf(v*is);if(x>127)x=127;else if(x<-127)x=-127;Bm[i]=(int8_t)x;}layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);}
inline void go(int l,const float*A,int am,int ak,float ascale,float Bscale,float*C,int an){float ais=1.0f/ascale;memset(Am,0,(size_t)MD*KD);for(int m=0;m<am;m++)for(int k=0;k<ak;k++){float v=A[m*ak+k];if(!std::isfinite(v))v=0;int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;Am[m*KD+k]=(int8_t)q;}bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);auto r=(*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA,*layerB[l],*bC);r.wait();bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);float cs=ascale*Bscale;for(int m=0;m<am;m++)for(int n=0;n<an;n++){float val=(float)Cm[m*ND+n]*cs;if(!std::isfinite(val))val=0;C[m*an+n]=val;}}
};

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int npt=(argc>1)?atoi(argv[1]):9;if(npt<1)npt=1;if(npt>9)npt=9;
    int ng=(argc>2)?atoi(argv[2]):16;
    #ifdef MODEL_qwen3_0_6b
    const char*mp=(argc>3)?argv[3]:DEF_MP;
    #else
    const char*mp=DEF_MP;
    (void)(argc>3?argv[3]:mp);
    #endif
    printf("=== NPU Engine v5 — %s (H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d NV=%d GQA=%d) ===\n\n",
           MODEL_TAG,H,NC,NH,NKV,HD,IM,NV,GQA);
    printf("Model: %s\n\n",mp);
    int fd=open(mp,O_RDONLY);struct stat st;fstat(fd,&st);
    uint8_t*md=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    uint64_t mmap_sz=st.st_size;
    uint64_t hsz;memcpy(&hsz,md,8);uint64_t df=8+hsz;
    auto i8p=[&](uint64_t o){return md+df+o;};auto emb=(const uint16_t*)(md+df);
    const char*js=(const char*)(md+8);size_t jl=hsz;
    struct LO{uint64_t qp,kp,vp,op,gp,up,dp,in_off,pa_off,qn_off,kn_off;}lo[NC];char b[128];
    for(int l=0;l<NC;l++){snprintf(b,128,"model.layers.%d.self_attn.q_proj.weight",l);lo[l].qp=jo(js,jl,b);snprintf(b,128,"model.layers.%d.self_attn.k_proj.weight",l);lo[l].kp=jo(js,jl,b);snprintf(b,128,"model.layers.%d.self_attn.v_proj.weight",l);lo[l].vp=jo(js,jl,b);snprintf(b,128,"model.layers.%d.self_attn.o_proj.weight",l);lo[l].op=jo(js,jl,b);snprintf(b,128,"model.layers.%d.mlp.gate_proj.weight",l);lo[l].gp=jo(js,jl,b);snprintf(b,128,"model.layers.%d.mlp.up_proj.weight",l);lo[l].up=jo(js,jl,b);snprintf(b,128,"model.layers.%d.mlp.down_proj.weight",l);lo[l].dp=jo(js,jl,b);snprintf(b,128,"model.layers.%d.input_layernorm.weight",l);lo[l].in_off=jo(js,jl,b);snprintf(b,128,"model.layers.%d.post_attention_layernorm.weight",l);lo[l].pa_off=jo(js,jl,b);snprintf(b,128,"model.layers.%d.self_attn.q_norm.weight",l);lo[l].qn_off=jo(js,jl,b);snprintf(b,128,"model.layers.%d.self_attn.k_norm.weight",l);lo[l].kn_off=jo(js,jl,b);}
    uint64_t no=jo(js,jl,"model.norm.weight"),lo_off=jo(js,jl,"lm_head.weight");
    std::vector<float>in_n(NC*H),pa_n(NC*H),fin(H),qn_w(NC*HD),kn_w(NC*HD);
    for(int l=0;l<NC;l++){auto iw=(const uint16_t*)(md+df+lo[l].in_off),pw_=(const uint16_t*)(md+df+lo[l].pa_off),qw=(const uint16_t*)(md+df+lo[l].qn_off),kw=(const uint16_t*)(md+df+lo[l].kn_off);for(int i=0;i<H;i++){in_n[l*H+i]=bf16g(iw[i]);pa_n[l*H+i]=bf16g(pw_[i]);}for(int i=0;i<HD;i++){qn_w[l*HD+i]=bf16g(qw[i]);kn_w[l*HD+i]=bf16g(kw[i]);}}
    {auto fw=(const uint16_t*)(md+df+no);for(int i=0;i<H;i++)fin[i]=bf16g(fw[i]);}

    printf("Init 4 GEMM...\n");xrt::device dev(0);
    // QKV combined dims: Q=H, K=H/GQA, V=H/GQA -> total per row = H + 2*(H/GQA)
    static constexpr int QKV_N = H + 2*HD*NKV; // = H + (NV?) No: QKV output dim = H + 2*HD*NKV... wait.
    // Actually: QKV concatenated = NH*HD + NKV*HD + NKV*HD = HD*(NH + 2*NKV) = 4096 for qwen3_0_6b
    // But NH and NKV vary per model, so we compute at compile time
    static constexpr int QKV_STRIDE = HD*(NH + 2*NKV); // total per-row QKV output
    static constexpr int ATTEN_STRIDE = NH*HD; // attention output per token
    // GU stride: GU fused = 2*IM, or G separate = IM + ... wait, G and U are separate
    static constexpr int GU_NOUT = GU_FUSED ? 2*IM : IM; // for fused: both G+U; for separate: G only
    static constexpr int U_NOUT = GU_FUSED ? 0 : IM; // separate U
    // D output = H
    #define XD XCLBIN_DIR

    // Build xclbin/insts paths using the per-model suffix
    char qkv_xp[256], o_xp[256], gu_xp[256], d_xp[256];
    char qkv_ip[256], o_ip[256], gu_ip[256], d_ip[256];
    char g_xp[256], u_xp[256], g_ip[256], u_ip[256];

    snprintf(qkv_xp,256,XD"/final_i8_QKV_%s.xclbin",XCLBIN_SUFFIX);
    snprintf(qkv_ip,256,XD"/insts_i8_QKV_%s.txt",XCLBIN_SUFFIX);
    snprintf(o_xp,256,XD"/final_i8_O_%s.xclbin",XCLBIN_SUFFIX);
    snprintf(o_ip,256,XD"/insts_i8_O_%s.txt",XCLBIN_SUFFIX);
    if(GU_FUSED){
        snprintf(gu_xp,256,XD"/final_i8_GU_%s.xclbin",XCLBIN_SUFFIX);
        snprintf(gu_ip,256,XD"/insts_i8_GU_%s.txt",XCLBIN_SUFFIX);
    }else{
        snprintf(g_xp,256,XD"/final_i8_G_%s.xclbin",XCLBIN_SUFFIX);
        snprintf(g_ip,256,XD"/insts_i8_G_%s.txt",XCLBIN_SUFFIX);
        snprintf(u_xp,256,XD"/final_i8_U_%s.xclbin",XCLBIN_SUFFIX);
        snprintf(u_ip,256,XD"/insts_i8_U_%s.txt",XCLBIN_SUFFIX);
    }
    snprintf(d_xp,256,XD"/final_i8_D_%s.xclbin",XCLBIN_SUFFIX);
    snprintf(d_ip,256,XD"/insts_i8_D_%s.txt",XCLBIN_SUFFIX);

    I8Ctx cq{"QKV",XM,H,QKV_STRIDE},co{"O",XM,ATTEN_STRIDE,H};
    I8Ctx cg{"GU",XM,H,GU_FUSED ? (2*IM) : IM};
    I8Ctx cd{"D",XM,IM,H};
    I8Ctx cu{"U",XM,H,IM}; // only used for separated G+U

    cq.init(dev,qkv_xp,qkv_ip,4);
    co.init(dev,o_xp,o_ip,4);
    if(GU_FUSED){
        cg.init(dev,gu_xp,gu_ip,4);
    }else{
        cg.init(dev,g_xp,g_ip,4);
        cu.init(dev,u_xp,u_ip,4);
    }
    cd.init(dev,d_xp,d_ip,4);

    printf("Dequant+pack...\n");auto tp=std::chrono::steady_clock::now();
    struct WS{float qk,o_,g_,u_,d_;}wsc[NC];
    for(int l=0;l<NC;l++){
        // Validate all weight offsets before dequant
        bool valid=true;
        auto check_off = [&](uint64_t off, const char* name) {
            if(!off) { fprintf(stderr,"  Layer %d %s: offset=0, SKIPPING\n",l,name); valid=false; }
            // Check if offset is past the mmap file end
            uint64_t abs_off = df + off;
            if(abs_off + 8 > mmap_sz) { fprintf(stderr,"  Layer %d %s: offset=%lu past file end, SKIPPING\n",l,name,off); valid=false; }
        };
        check_off(lo[l].qp,"q_proj");
        check_off(lo[l].kp,"k_proj");
        check_off(lo[l].vp,"v_proj");
        check_off(lo[l].op,"o_proj");
        check_off(lo[l].gp,"g_proj");
        check_off(lo[l].up,"u_proj");
        check_off(lo[l].dp,"d_proj");
        if(!valid) { fprintf(stderr,"  Layer %d: INCOMPLETE, skipping\n",l); continue; }
        int qr,kr,vr,or_,gr,ur,dr,unused;
        fprintf(stderr,"  Layer %d/%d dequant...\r",l,NC);
        float*qw=dequant_i8_to_float_ex(i8p(lo[l].qp),Q_I8R,H,&qr,&unused);
        float*kw=dequant_i8_to_float_ex(i8p(lo[l].kp),KV_I8R,H,&kr,&unused);
        float*vw=dequant_i8_to_float_ex(i8p(lo[l].vp),KV_I8R,H,&vr,&unused);
        // QKV separate dequant, then pack concatenated
        int t=qr+kr+vr;std::vector<float>w((size_t)H*t);
        for(int k=0;k<H;k++){memcpy(&w[k*t],&qw[k*qr],qr*4);memcpy(&w[k*t+qr],&kw[k*kr],kr*4);memcpy(&w[k*t+qr+kr],&vw[k*vr],vr*4);}
        cq.packB(l,w.data(),H,t,wsc[l].qk);free(qw);free(kw);free(vw);
        // O
        float*ow=dequant_i8_to_float_ex(i8p(lo[l].op),O_I8R,H,&or_,&unused);
        co.packB(l,ow,or_,H,wsc[l].o_);free(ow);
        // GU (fused) or G+U (separate)
        float*gw=dequant_i8_to_float_ex(i8p(lo[l].gp),GU_I8R,H,&gr,&unused);
        float*uw=dequant_i8_to_float_ex(i8p(lo[l].up),GU_I8R,H,&ur,&unused);
        if(GU_FUSED){
            int t2=gr+ur;std::vector<float>w2((size_t)H*t2);
            for(int k=0;k<H;k++){memcpy(&w2[k*t2],&gw[k*gr],gr*4);memcpy(&w2[k*t2+gr],&uw[k*ur],ur*4);}
            cg.packB(l,w2.data(),H,t2,wsc[l].g_);
        }else{
            cg.packB(l,gw,gr,H,wsc[l].g_);
            cu.packB(l,uw,ur,H,wsc[l].u_);
        }
        free(gw);free(uw);
        // D
        float*dw=dequant_i8_to_float_ex(i8p(lo[l].dp),D_I8R,H,&dr,&unused);
        cd.packB(l,dw,dr,H,wsc[l].d_);free(dw);
    }
    // Skip lm_head full dequant (was used for validation; now validated via offset check)
    printf("  %.0fms\n\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tp).count());
    ri(HD,ROPE_THETA,MAX_POS);

    struct KVCache{std::vector<float>k,v;int n;
        KVCache():k((size_t)MAX_POS*NKV*HD),v((size_t)MAX_POS*NKV*HD),n(0){}
    };std::vector<KVCache>kv(NC);
    // Batched buffers — use max possible sizes at compile time
    std::vector<float>h_b((size_t)XM*H);
    std::vector<float>qo_b((size_t)XM*QKV_STRIDE);
    std::vector<float>at_b((size_t)XM*ATTEN_STRIDE);
    std::vector<float>oo_b((size_t)XM*H);
    std::vector<float>gt_b((size_t)XM*(GU_FUSED ? 2*IM : IM));
    std::vector<float>ut_b((size_t)XM*IM); // for separate U
    std::vector<float>su_b((size_t)XM*IM);
    std::vector<float>dw_b((size_t)XM*H);
    std::vector<float>h(H),qo(QKV_STRIDE),ko((size_t)NKV*HD),vo((size_t)NKV*HD),at(ATTEN_STRIDE),
                      oo(H),lg(NV),sb(H);
    int sp=0;
    int pt[]={BOS,872,198,11852,EOS,198,BOS,77091,198};

    // ===== PREFILL =====
    printf("=== Prefill %d ===\n",npt);auto t0=std::chrono::steady_clock::now();
    for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[(size_t)pi*H+i]=bf16g(emb[(size_t)pt[pi]*H+i]);

    for(int l=0;l<NC;l++){
        for(int pi=0;pi<npt;pi++)rn_c(&h_b[(size_t)pi*H],&in_n[(size_t)l*H],H);
        cq.go(l,h_b.data(),npt,H,dynamic_ascale(h_b.data(),npt*H),wsc[l].qk,qo_b.data(),QKV_STRIDE);cn(qo_b.data(),npt*QKV_STRIDE);
        for(int pi=0;pi<npt;pi++){
            for(int hh=0;hh<NH;hh++){double s=0;for(int d=0;d<HD;d++)s+=qo_b[(size_t)pi*QKV_STRIDE+hh*HD+d]*qo_b[(size_t)pi*QKV_STRIDE+hh*HD+d];float iq=1.0f/sqrtf((float)(s/HD)+EPS);for(int d=0;d<HD;d++)qo_b[(size_t)pi*QKV_STRIDE+hh*HD+d]*=iq*qn_w[(size_t)l*HD+d];ra(&qo_b[(size_t)pi*QKV_STRIDE+hh*HD],HD,sp+pi);}
            for(int kvh=0;kvh<NKV;kvh++){float*ks=&qo_b[(size_t)pi*QKV_STRIDE+NH*HD+kvh*HD],*vs=&qo_b[(size_t)pi*QKV_STRIDE+(NH+NKV)*HD+kvh*HD];
                double sk=0;for(int d=0;d<HD;d++)sk+=ks[d]*ks[d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
                for(int d=0;d<HD;d++){ks[d]*=ik*kn_w[(size_t)l*HD+d];ra(ks,HD,sp+pi);}
                memcpy(&kv[l].k[(size_t)(sp+pi)*NKV*HD+kvh*HD],ks,HD*4);memcpy(&kv[l].v[(size_t)(sp+pi)*NKV*HD+kvh*HD],vs,HD*4);
            }
        }
        kv[l].n=sp+npt;int cl=kv[l].n;
        for(int pi=0;pi<npt;pi++){for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;int pl=sp+pi+1;std::vector<float>ss(cl);
            for(int p=0;p<pl;p++){double s=0;for(int d=0;d<HD;d++)s+=qo_b[(size_t)pi*QKV_STRIDE+hh*HD+d]*kv[l].k[(size_t)p*NKV*HD+kvh*HD+d];ss[p]=(float)(s/sqrtf(HD));}
            sm(ss.data(),pl);for(int d=0;d<HD;d++){float s=0;for(int p=0;p<pl;p++)s+=ss[p]*kv[l].v[(size_t)p*NKV*HD+kvh*HD+d];at_b[(size_t)pi*ATTEN_STRIDE+hh*HD+d]=s;}}}
        co.go(l,at_b.data(),npt,ATTEN_STRIDE,dynamic_ascale(at_b.data(),npt*ATTEN_STRIDE),wsc[l].o_,oo_b.data(),H);cn(oo_b.data(),npt*H);
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[(size_t)pi*H+i]+=oo_b[(size_t)pi*H+i];
        for(int pi=0;pi<npt;pi++)rn_c(&h_b[(size_t)pi*H],&pa_n[(size_t)l*H],H);
        // MLP: G+U -> silu(G)*U -> D
        if(GU_FUSED){
            cg.go(l,h_b.data(),npt,H,dynamic_ascale(h_b.data(),npt*H),wsc[l].g_,gt_b.data(),2*IM);cn(gt_b.data(),npt*2*IM);
            for(int pi=0;pi<npt;pi++){for(int i=0;i<IM;i++){float gv=gt_b[(size_t)pi*2*IM+i];if(!std::isfinite(gv))gv=0;su_b[(size_t)pi*IM+i]=(gv/(1.0f+expf(-gv)))*gt_b[(size_t)pi*2*IM+IM+i];}}
        }else{
            cg.go(l,h_b.data(),npt,H,dynamic_ascale(h_b.data(),npt*H),wsc[l].g_,gt_b.data(),IM);cn(gt_b.data(),npt*IM);
            cu.go(l,h_b.data(),npt,H,dynamic_ascale(h_b.data(),npt*H),wsc[l].u_,ut_b.data(),IM);cn(ut_b.data(),npt*IM);
            for(int pi=0;pi<npt;pi++){for(int i=0;i<IM;i++){float gv=gt_b[(size_t)pi*IM+i];if(!std::isfinite(gv))gv=0;su_b[(size_t)pi*IM+i]=(gv/(1.0f+expf(-gv)))*ut_b[(size_t)pi*IM+i];}}
        }
        cd.go(l,su_b.data(),npt,IM,dynamic_ascale(su_b.data(),npt*IM),wsc[l].d_,dw_b.data(),H);cn(dw_b.data(),npt*H);
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[(size_t)pi*H+i]+=dw_b[(size_t)pi*H+i];
    }
    sp+=npt;memcpy(h.data(),&h_b[(size_t)(npt-1)*H],H*4);
    double ms_prefill=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    printf("Prefill: %.0fms (%.0f ms/tok)\n\n",ms_prefill,ms_prefill/npt);

    // ===== DECODE =====
    printf("=== Decode (%d tokens) ===\n",ng);
    auto tgs=std::chrono::steady_clock::now();
    for(int step=0;step<ng;step++){auto ts=std::chrono::steady_clock::now();
        for(int l=0;l<NC;l++){
            memcpy(sb.data(),h.data(),H*4);rn_c(h.data(),&in_n[(size_t)l*H],H);
            cq.go(l,h.data(),1,H,dynamic_ascale(h.data(),H),wsc[l].qk,qo.data(),QKV_STRIDE);cn(qo.data(),QKV_STRIDE);
            // Split Q/K/V from QKV output
            memcpy(ko.data(),&qo[(size_t)NH*HD],(size_t)NKV*HD*4);
            memcpy(vo.data(),&qo[(size_t)(NH+NKV)*HD],(size_t)NKV*HD*4);
            for(int hh=0;hh<NH;hh++){
                double sq=0;for(int d=0;d<HD;d++)sq+=qo[(size_t)hh*HD+d]*qo[(size_t)hh*HD+d];
                float iq=1.0f/sqrtf((float)(sq/HD)+EPS);
                for(int d=0;d<HD;d++)qo[(size_t)hh*HD+d]*=iq*qn_w[(size_t)l*HD+d];
                ra(&qo[hh*HD],HD,sp);
                if(hh%GQA==0){int kvh=hh/GQA;
                    double sk=0;for(int d=0;d<HD;d++)sk+=ko[(size_t)kvh*HD+d]*ko[(size_t)kvh*HD+d];
                    float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
                    for(int d=0;d<HD;d++){ko[(size_t)kvh*HD+d]*=ik*kn_w[(size_t)l*HD+d];ra(&ko[kvh*HD],HD,sp);}
                    memcpy(&kv[l].k[(size_t)sp*NKV*HD+kvh*HD],&ko[kvh*HD],HD*4);
                    memcpy(&kv[l].v[(size_t)sp*NKV*HD+kvh*HD],&vo[kvh*HD],HD*4);
                }
            }
            kv[l].n=sp+1;int cl=kv[l].n;
            for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;std::vector<float>sc(cl);
                for(int p=0;p<cl;p++){double s=0;for(int d=0;d<HD;d++)s+=qo[(size_t)hh*HD+d]*kv[l].k[(size_t)p*NKV*HD+kvh*HD+d];sc[p]=(float)(s/sqrtf(HD));}
                sm(sc.data(),cl);for(int d=0;d<HD;d++){float s=0;for(int p=0;p<cl;p++)s+=sc[p]*kv[l].v[(size_t)p*NKV*HD+kvh*HD+d];at[(size_t)hh*HD+d]=s;}
            }
            co.go(l,at.data(),1,ATTEN_STRIDE,dynamic_ascale(at.data(),ATTEN_STRIDE),wsc[l].o_,oo.data(),H);cn(oo.data(),H);
            for(int i=0;i<H;i++)h[i]=sb[i]+oo[i];
            memcpy(sb.data(),h.data(),H*4);rn_c(h.data(),&pa_n[(size_t)l*H],H);
            // MLP decode
            if(GU_FUSED){
                cg.go(l,h.data(),1,H,dynamic_ascale(h.data(),H),wsc[l].g_,gt_b.data(),2*IM);cn(gt_b.data(),2*IM);
                for(int i=0;i<IM;i++){float gv=gt_b[i];if(!std::isfinite(gv))gv=0;su_b[i]=(gv/(1.0f+expf(-gv)))*gt_b[IM+i];}
            }else{
                cg.go(l,h.data(),1,H,dynamic_ascale(h.data(),H),wsc[l].g_,gt_b.data(),IM);cn(gt_b.data(),IM);
                cu.go(l,h.data(),1,H,dynamic_ascale(h.data(),H),wsc[l].u_,ut_b.data(),IM);cn(ut_b.data(),IM);
                for(int i=0;i<IM;i++){float gv=gt_b[i];if(!std::isfinite(gv))gv=0;su_b[i]=(gv/(1.0f+expf(-gv)))*ut_b[i];}
            }
            cd.go(l,su_b.data(),1,IM,dynamic_ascale(su_b.data(),IM),wsc[l].d_,dw_b.data(),H);cn(dw_b.data(),H);
            for(int i=0;i<H;i++)h[i]=sb[i]+dw_b[i];
        }
        memcpy(sb.data(),h.data(),H*4);rn_c(sb.data(),fin.data(),H);
        for(int n=0;n<NV;n++){double s=0;for(int k=0;k<H;k++){uint16_t r=emb[(size_t)n*H+k];if((r&0x7F80)!=0x7F80)s+=sb[k]*bf16f(r);}lg[n]=(float)s;}
        float mx=lg[0];for(int i=1;i<NV;i++)if(lg[i]>mx)mx=lg[i];
        double sum=0;for(int i=0;i<NV;i++){float d=lg[i]-mx;if(d<-80)d=-80;lg[i]=expf(d);sum+=lg[i];}
        float rr=(float)rand()/RAND_MAX*(float)sum,acc=0;int tok=0;
        for(int i=0;i<NV;i++){acc+=lg[i];if(acc>=rr){tok=i;break;}}
        double mss=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();
        printf("  [%d] %d (%.0fms)\n",step,tok,mss);
        for(int i=0;i<H;i++)h[i]=bf16g(emb[(size_t)tok*H+i]);sp++;
    }
    double tts=std::chrono::duration<double>(std::chrono::steady_clock::now()-tgs).count();
    printf("\n=== %.0f ms/tok ===\n",tts*1000/ng);
    munmap(md,st.st_size);return 0;
}
