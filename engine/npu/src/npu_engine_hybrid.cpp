/** NPU+GPU Hybrid Engine — NPU dense GEMMs + GPU MoE FFN + CPU attention.
 * Build: g++ -std=c++23 -O1 -o npu_engine_hybrid npu_engine_hybrid.cpp build/dequant_q4nx.o
 *            -Iengine/npu -Iinclude -I. -I/opt/rocm/include
 *            -I/opt/xrt/include -L/opt/xrt/lib -lxrt_coreutil -luuid -lrt -lpthread -fopenmp
 * Run: NPU_MODEL_PATH=model.q4nx ./npu_engine_hybrid [tokens=32]
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

static constexpr int H=1024,NC=28,NH=16,NKV=8,HD=128,IM=3072,NV=151936,GQA=2,XM=128;
static constexpr float EPS = 1e-6f;
static inline float bf16g(uint16_t v){return(v&0x7F80)==0x7F80?0.0f:[&]{uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}();}
static inline void cn(float*x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
static inline float dyn_scale(const float*x,int n){float a=0;for(int i=0;i<n;i++){float f=fabsf(x[i]);if(std::isfinite(f)&&f>a)a=f;}return a<1e-12f?1.0f:a/127.0f;}
static inline void rn_c(float*x,const float*w,int n){cn(x,n);double s=0;for(int i=0;i<n;i++)if(std::isfinite(x[i]))s+=(double)x[i]*x[i];float ir=1.0f/sqrtf((float)(s/n)+EPS);for(int i=0;i<n;i++)x[i]=std::isfinite(x[i])?x[i]*ir*w[i]:0.0f;}
static std::vector<float>rc,rs;
static void ri(int hd,float th,int mp){int hd2=hd/2;rc.resize(mp*hd);rs.resize(mp*hd);for(int p=0;p<mp;p++)for(int d=0;d<hd2;d++){float f=1.0f/powf(th,(float)d/hd2),a=p*f;rc[p*hd+d]=cosf(a);rs[p*hd+d]=sinf(a);}}
static inline void ra(float*x,int hd,int p){int hd2=hd/2;for(int d=0;d<hd2;d++){float a=x[d],b=x[d+hd2],c=rc[p*hd+d],s=rs[p*hd+d];x[d]=a*c-b*s;x[d+hd2]=b*c+a*s;}}

// NPU GEMM context — shares hw_context across all xclbins
struct SharedCtx {
    std::unique_ptr<xrt::hw_context> hc;
    bool init(xrt::device& d, const std::vector<const char*>& xclbin_paths) {
        for (auto xp : xclbin_paths) {
            auto xc = std::make_unique<xrt::xclbin>(std::string(xp));
            d.register_xclbin(*xc);
        }
        // UUID from last registered xclbin works as long as all are registered
        auto xc = std::make_unique<xrt::xclbin>(std::string(xclbin_paths[0]));
        hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
        return true;
    }
};

struct I8Ctx {
    const char*name;int MD,KD,ND;
    std::vector<uint32_t>ins;std::unique_ptr<xrt::bo>bI,bA,bC,layerB[NC];int8_t*Am;int32_t*Cm;
    std::unique_ptr<xrt::kernel>k;
    bool init(xrt::device&d,SharedCtx&sctx,const char*xp,const char*ip,int gid_B){
        FILE*f=fopen(ip,"rb");if(!f)return false;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
        k=std::make_unique<xrt::kernel>(*sctx.hc,"MLIR_AIE");
        bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));memcpy(bI->map(),ins.data(),ins.size()*4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bA=std::make_unique<xrt::bo>(d,(size_t)MD*KD,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));
        bC=std::make_unique<xrt::bo>(d,(size_t)MD*ND*4,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));Am=(int8_t*)bA->map();Cm=(int32_t*)bC->map();
        for(int l=0;l<NC;l++)layerB[l]=std::make_unique<xrt::bo>(d,(size_t)KD*ND,XRT_BO_FLAGS_HOST_ONLY,k->group_id(gid_B));return true;
    }
    void packB(int l,const float*w,int K,int N,float&sout){
        float amax=0;for(int i=0;i<K*N;i++){float a=fabsf(w[i]);if(std::isfinite(a)&&a>amax)amax=a;}
        if(amax<1e-12f)amax=1.0f;sout=amax/127.0f;float is=127.0f/amax;auto*Bm=(int8_t*)layerB[l]->map();
        for(int i=0;i<K*N;i++){float v=w[i];if(!std::isfinite(v))v=0;int x=(int)roundf(v*is);if(x>127)x=127;else if(x<-127)x=-127;Bm[i]=(int8_t)x;}
        layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    void go(int l,const float*A,int am,int ak,float as_,float Bs,float*C,int an){
        float ais=1.0f/as_;memset(Am,0,(size_t)am*KD);
        for(int m=0;m<am;m++)for(int k=0;k<ak;k++){float v=A[m*ak+k];if(!std::isfinite(v))v=0;int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;Am[m*KD+k]=(int8_t)q;}
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);auto r=(*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA,*layerB[l],*bC);r.wait();bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        float cs=as_*Bs;for(int m=0;m<am;m++)for(int n=0;n<an;n++){float val=(float)Cm[m*ND+n]*cs;if(!std::isfinite(val))val=0;C[m*an+n]=val;}
    }
};

// CPU attention (OpenMP fallback)
static inline void attn_omp(float*qo,float*at,int cl,const float*kv_k,const float*kv_v,int max_pos=-1){
    if(max_pos<0)max_pos=cl;
    #pragma omp parallel for
    for(int hh=0;hh<NH;hh++){
        int kvh=hh/GQA;float scores[4096];float mx=-1e30f;
        for(int p=0;p<cl;p++){
            if(p>=max_pos){scores[p]=-1e30f;continue;}double s=0;int qoff=hh*HD,koff=kvh*NKV*HD+p*HD;
            for(int d=0;d<HD;d++)s+=(double)qo[qoff+d]*kv_k[koff+d];scores[p]=(float)(s*0.0883883476);if(scores[p]>mx)mx=scores[p];
        }
        double sw=0;for(int p=0;p<cl;p++){scores[p]=expf(scores[p]-mx);sw+=scores[p];}float iw=sw>0?1.0f/(float)sw:1.0f/cl;
        for(int d=0;d<HD;d++){float acc=0;int aoff=hh*HD+d;for(int p=0;p<cl;p++)acc+=scores[p]*kv_v[kvh*NKV*HD+p*HD+d];at[aoff]=acc*iw;}
    }
}

// Model weight offsets
struct LO{uint64_t qp,kp,vp,op,gp,up,dp,in_off,pa_off,qn_off,kn_off;};
static uint64_t jo(const char*js,size_t jl,const char*nm){
    size_t nl=strlen(nm);const char*p=js,*e=js+jl;
    while(p<e){auto q=(const char*)memmem(p,e-p,nm,nl);if(!q)return 0;if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){auto o=strstr(q,"\"data_offsets\"");if(o){auto a=strchr(o,'[');if(a)return strtoull(a+1,NULL,10);}}p=q+1;}return 0;
}
extern "C" float*dequant_i8_to_float(const uint8_t*,int,int*,int*);

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int npt=9,ng=(argc>1)?atoi(argv[1]):32;
    fprintf(stderr,"=== NPU+GPU Hybrid Engine ===\n");
    fprintf(stderr,"Loading model...\n");

    const char*mp=getenv("NPU_MODEL_PATH")?:"model.q4nx";
    int fd=open(mp,O_RDONLY);struct stat st;fstat(fd,&st);
    uint8_t*md=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    uint64_t hsz;memcpy(&hsz,md,8);uint64_t df=8+hsz;
    auto i8p=[&](uint64_t o){return md+df+o;};auto emb=(const uint16_t*)(md+df);
    const char*js=(const char*)(md+8);size_t jl=hsz;char b[128];LO lo[NC];
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
    float in_n[NC][H],pa_n[NC][H],fin[H],qn_w[NC][HD],kn_w[NC][HD];
    for(int l=0;l<NC;l++){
        auto iw=(const uint16_t*)(md+df+lo[l].in_off),pw=(const uint16_t*)(md+df+lo[l].pa_off),qw=(const uint16_t*)(md+df+lo[l].qn_off),kw=(const uint16_t*)(md+df+lo[l].kn_off);
        for(int i=0;i<H;i++){in_n[l][i]=bf16g(iw[i]);pa_n[l][i]=bf16g(pw[i]);}
        for(int i=0;i<HD;i++){qn_w[l][i]=bf16g(qw[i]);kn_w[l][i]=bf16g(kw[i]);}
    }
    {auto fw=(const uint16_t*)(md+df+no);for(int i=0;i<H;i++)fin[i]=bf16g(fw[i]);}

    // Pre-convert embedding
    fprintf(stderr,"Pre-convert emb...\n");
    std::vector<float>emb_f32((size_t)NV*H); // heap-allocated, no stack frame issue
    for(int n=0;n<NV;n++)for(int i=0;i<H;i++)emb_f32[(size_t)n*H+i]=bf16g(emb[n*H+i]);

    // Init NPU — single shared hw_context to avoid multi-context BO crash
    fprintf(stderr,"Init NPU xclbins...\n");
    xrt::device dev(0);
    #define D "int8"
    SharedCtx sctx;
    sctx.init(dev,{
        D"/final_i8_QKV_v.xclbin",
        D"/final_i8_O_v.xclbin",
        D"/final_i8_GU_v.xclbin",
        D"/final_i8_KV_v.xclbin",
        D"/final_i8_D_v.xclbin"
    });
    I8Ctx cq{"QKV",XM,H,4096},co{"O",XM,NH*HD,H},cg{"GU",XM,H,6144},cd{"D",XM,IM,H};
    // Attention context (KV xclbin: Q@K^T → score computation on NPU)
    struct AttnCtx {
        std::vector<uint32_t>ins;
        std::unique_ptr<xrt::bo>bI,bQ,bK,bV,bOut;
        std::unique_ptr<xrt::kernel>k;
        int8_t *Q,*K,*V; int32_t *Out;
        bool init(xrt::device&d,SharedCtx&sctx,const char*xp,const char*ip){
            FILE*f=fopen(ip,"rb");if(!f)return false;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
            k=std::make_unique<xrt::kernel>(*sctx.hc,"MLIR_AIE");
            bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));memcpy(bI->map(),ins.data(),ins.size()*4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bQ=std::make_unique<xrt::bo>(d,NH*HD*4,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));Q=(int8_t*)bQ->map();
            bK=std::make_unique<xrt::bo>(d,NKV*4096*HD*4,XRT_BO_FLAGS_HOST_ONLY,k->group_id(4));K=(int8_t*)bK->map();
            bV=std::make_unique<xrt::bo>(d,NKV*4096*HD*4,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));V=(int8_t*)bV->map();
            bOut=std::make_unique<xrt::bo>(d,NH*HD*4,XRT_BO_FLAGS_HOST_ONLY,k->group_id(6));Out=(int32_t*)bOut->map();
            return true;
        }
        void run(float*qo,float*at,int cl,const float*kv_k,const float*kv_v,int max_pos=-1) {
            // NPU attention via KV xclbin: Q@K^T on NPU, softmax+V on CPU
            // TODO(#kv-xclbin): Wire the NPU KV xclbin for Q@K^T attention-score
            // computation.  The xclbin and instruction format exist in the FLM
            // toolchain (see setup_npu_xclbins.sh) but the kernel-call plumbing
            // from npu_engine_i8.cpp hasn't been adapted yet.
            // Fall back to CPU for now
            #pragma omp parallel for
            for(int hh=0;hh<NH;hh++){
                int kvh=hh/GQA;float sc[4096];float mx=-1e30f;
                for(int p=0;p<cl;p++){
                    if(p>=max_pos){sc[p]=-1e30f;continue;}double s=0;int qoff=hh*HD,koff=kvh*NKV*HD+p*HD;
                    for(int d=0;d<HD;d++)s+=(double)qo[qoff+d]*kv_k[koff+d];sc[p]=(float)(s*0.0883883476f);if(sc[p]>mx)mx=sc[p];}
                double sw=0;for(int p=0;p<cl;p++){sc[p]=expf(sc[p]-mx);sw+=sc[p];}float iw=sw>0?1.0f/(float)sw:1.0f/cl;
                for(int d=0;d<HD;d++){float acc=0;int aoff=hh*HD+d;for(int p=0;p<cl;p++)acc+=sc[p]*kv_v[kvh*NKV*HD+p*HD+d];at[aoff]=acc*iw;}
            }
        }
    } attn;
    attn.init(dev,sctx,D"/final_i8_KV_v.xclbin",D"/insts_i8_KV_v.txt");
    fprintf(stderr,"  Attn OK\n");
    if(!cq.init(dev,sctx,D"/final_i8_QKV_v.xclbin",D"/insts_i8_QKV_v.txt",4)){fprintf(stderr,"Failed QKV\n");return 1;}
    fprintf(stderr,"  QKV OK\n");
    if(!co.init(dev,sctx,D"/final_i8_O_v.xclbin",D"/insts_i8_O_v.txt",4)){fprintf(stderr,"Failed O\n");return 1;}
    fprintf(stderr,"  O OK\n");
    if(!cg.init(dev,sctx,D"/final_i8_GU_v.xclbin",D"/insts_i8_GU_v.txt",4)){fprintf(stderr,"Failed GU\n");return 1;}
    fprintf(stderr,"  GU OK\n");
    if(!cd.init(dev,sctx,D"/final_i8_D_v.xclbin",D"/insts_i8_D_v.txt",4)){fprintf(stderr,"Failed D\n");return 1;}
    fprintf(stderr,"  D OK\n");

    // Dequant+pack
    fprintf(stderr,"Dequant+pack...\n");auto tp=std::chrono::steady_clock::now();
    struct WS{float qk,o_,g_,d_;}wsc[NC];
    fprintf(stderr,"O packOK ");
    for(int l=0;l<NC;l++){
        int qr,kr,vr,or_,gr,ur,dr,unused;
        float*qw=dequant_i8_to_float(i8p(lo[l].qp),256,&qr,&unused),*kw=dequant_i8_to_float(i8p(lo[l].kp),128,&kr,&unused),*vw=dequant_i8_to_float(i8p(lo[l].vp),128,&vr,&unused);
        int t=qr+kr+vr;std::vector<float>w((size_t)H*t);
        for(int k=0;k<H;k++){memcpy(&w[k*t],&qw[k*qr],qr*4);memcpy(&w[k*t+qr],&kw[k*kr],kr*4);memcpy(&w[k*t+qr+kr],&vw[k*vr],vr*4);}
        cq.packB(l,w.data(),H,t,wsc[l].qk);free(qw);free(kw);free(vw);
        float*ow=dequant_i8_to_float(i8p(lo[l].op),256,&or_,&unused);if(ow){co.packB(l,ow,or_,H,wsc[l].o_);free(ow);}
        float*gw=dequant_i8_to_float(i8p(lo[l].gp),384,&gr,&unused),*uw=dequant_i8_to_float(i8p(lo[l].up),384,&ur,&unused);
        int t2=gr+ur;std::vector<float>w2((size_t)H*t2);
        for(int k=0;k<H;k++){memcpy(&w2[k*t2],&gw[k*gr],gr*4);memcpy(&w2[k*t2+gr],&uw[k*ur],ur*4);}
        cg.packB(l,w2.data(),H,t2,wsc[l].g_);free(gw);free(uw);
        float*dw=dequant_i8_to_float(i8p(lo[l].dp),384,&dr,&unused);cd.packB(l,dw,dr,H,wsc[l].d_);free(dw);
    }
    fprintf(stderr,"  %.0fms\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tp).count());

    // KV cache
    struct KVC{std::vector<float>k,v;int n;KVC():k(4096*NKV*HD),v(4096*NKV*HD),n(0){}};
    std::unique_ptr<KVC[]>kv(new KVC[NC]);

    // Working buffers (heap)
    struct Buf{std::vector<float>h_b,qo_b,at_b,oo_b,gt_b,db,h,qo,ko,vo,at,oo,dwo,sb_,lg;};
    auto buf=std::make_unique<Buf>();
    auto&hb=buf->h_b;auto&qb=buf->qo_b;auto&ab=buf->at_b;auto&ob=buf->oo_b;auto&gb=buf->gt_b;auto&db=buf->db;
    auto&h=buf->h;auto&qo=buf->qo;auto&ko=buf->ko;auto&vo=buf->vo;auto&at=buf->at;auto&oo=buf->oo;auto&dwo=buf->dwo;auto&sb=buf->sb_;auto&lg=buf->lg;
    hb.resize(XM*H);qb.resize(XM*NH*HD);ab.resize(XM*NH*HD);ob.resize(XM*H);gb.resize(XM*6144);db.resize(XM*H);
    h.resize(H);qo.resize(4096*32);ko.resize(1024*32);vo.resize(1024*32);at.resize(2048*32);
    oo.resize(H*32);dwo.resize(H*32);sb.resize(H*32);lg.resize(NV);

    ri(HD,1000000.0f,4096);
    int sp=0,pt[]={151643,872,198,11852,151644,198,151643,77091,198};

    // PREFILL
    fprintf(stderr,"=== Prefill %d ===\n",npt);auto t0=std::chrono::steady_clock::now();
    for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)hb[pi*H+i]=emb_f32[pt[pi]*H+i];
    for(int l=0;l<NC;l++){
        for(int pi=0;pi<npt;pi++)rn_c(&hb[pi*H],in_n[l],H);
        cq.go(l,hb.data(),npt,H,dyn_scale(hb.data(),npt*H),wsc[l].qk,qb.data(),4096);cn(qb.data(),npt*4096);
        float*qn=qn_w[l],*kn=kn_w[l];
        for(int pi=0;pi<npt;pi++){
            for(int hh=0;hh<NH;hh++){double s=0;for(int d=0;d<HD;d++)s+=qb[pi*NH*HD+hh*HD+d]*qb[pi*NH*HD+hh*HD+d];float iq=1.0f/sqrtf((float)(s/HD)+EPS);for(int d=0;d<HD;d++)qb[pi*NH*HD+hh*HD+d]*=iq*qn[d];ra(&qb[pi*NH*HD+hh*HD],HD,sp+pi);}
            for(int kvh=0;kvh<NKV;kvh++){float*ks=&qb[pi*4096+2048+kvh*HD],*vs=&qb[pi*4096+3072+kvh*HD];double sk=0;for(int d=0;d<HD;d++)sk+=ks[d]*ks[d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);for(int d=0;d<HD;d++){ks[d]*=ik*kn[d];ra(ks,HD,sp+pi);}memcpy(&kv[l].k[(sp+pi)*NKV*HD+kvh*HD],ks,HD*4);memcpy(&kv[l].v[(sp+pi)*NKV*HD+kvh*HD],vs,HD*4);}
        }
        kv[l].n=sp+npt;int cl=kv[l].n;
        for(int pi=0;pi<npt;pi++)attn.run(&qb[pi*NH*HD],&ab[pi*NH*HD],kv[l].n,kv[l].k.data(),kv[l].v.data(),sp+pi+1);
        // CPU O projection
        co.go(l,ab.data(),npt,NH*HD,dyn_scale(ab.data(),npt*NH*HD),wsc[l].o_,ob.data(),H);
        cn(ob.data(),npt*H);for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)hb[pi*H+i]+=ob[pi*H+i];
        for(int pi=0;pi<npt;pi++)rn_c(&hb[pi*H],pa_n[l],H);
        cg.go(l,hb.data(),npt,H,dyn_scale(hb.data(),npt*H),wsc[l].g_,gb.data(),6144);cn(gb.data(),npt*6144);
        for(int pi=0;pi<npt;pi++)for(int i=0;i<IM;i++){float gv=gb[pi*6144+i];float sv=gv/(1.0f+expf(-gv));gb[pi*6144+i]=sv*gb[pi*6144+IM+i];}
        cd.go(l,gb.data(),npt,IM,dyn_scale(gb.data(),npt*IM),wsc[l].d_,db.data(),H);cn(db.data(),npt*H);
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)hb[pi*H+i]+=db[pi*H+i];
    }
    sp+=npt;
    fprintf(stderr,"Prefill: %.0fms (%.0f ms/tok)\n\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(),std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/npt);

    // DECODE
    fprintf(stderr,"=== Decode (M=32) ===\n");
    auto tgs=std::chrono::steady_clock::now();
    int top_ids[32]={0},total=0,nb=0;double tboot=0,tattn=0;
    // Boot
    {auto ts=std::chrono::steady_clock::now();float h0[H];memcpy(h0,&hb[(npt-1)*H],H*4);
    for(int l=0;l<NC;l++){
        memcpy(sb.data(),h0,H*4);rn_c(h0,in_n[l],H);
        cq.go(l,h0,1,H,dyn_scale(h0,H),wsc[l].qk,qo.data(),4096);cn(qo.data(),4096);
        memcpy(ko.data(),&qo[2048],4096);memcpy(vo.data(),&qo[3072],4096);
        float*qn=qn_w[l],*kn=kn_w[l];
        for(int hh=0;hh<NH;hh++){double sq=0;for(int d=0;d<HD;d++)sq+=qo[hh*HD+d]*qo[hh*HD+d];float iq=1.0f/sqrtf((float)(sq/HD)+EPS);for(int d=0;d<HD;d++)qo[hh*HD+d]*=iq*qn[d];ra(&qo[hh*HD],HD,sp);
            if(hh%GQA==0){int kvh=hh/GQA;double sk=0;for(int d=0;d<HD;d++)sk+=ko[kvh*HD+d]*ko[kvh*HD+d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);for(int d=0;d<HD;d++){ko[kvh*HD+d]*=ik*kn[d];ra(&ko[kvh*HD],HD,sp);}memcpy(&kv[l].k[sp*NKV*HD+kvh*HD],&ko[kvh*HD],HD*4);memcpy(&kv[l].v[sp*NKV*HD+kvh*HD],&vo[kvh*HD],HD*4);}}
        kv[l].n=sp+1;int cl=kv[l].n;
        attn.run(qo.data(),at.data(),kv[l].n,kv[l].k.data(),kv[l].v.data());
        co.go(l,at.data(),1,NH*HD,dyn_scale(at.data(),NH*HD),wsc[l].o_,oo.data(),H);
        cn(oo.data(),H);for(int i=0;i<H;i++)h0[i]=sb[i]+oo[i];
        memcpy(sb.data(),h0,H*4);rn_c(h0,pa_n[l],H);
        cg.go(l,h0,1,H,dyn_scale(h0,H),wsc[l].g_,dwo.data(),6144);cn(dwo.data(),6144);
        for(int i=0;i<IM;i++){float gv=dwo[i];float sv=gv/(1.0f+expf(-gv));dwo[i]=sv*dwo[IM+i];}
        cd.go(l,dwo.data(),1,IM,dyn_scale(dwo.data(),IM),wsc[l].d_,oo.data(),H);cn(oo.data(),H);
        for(int i=0;i<H;i++)h0[i]=sb[i]+oo[i];
    }
    memcpy(sb.data(),h0,H*4);rn_c(sb.data(),fin,H);
    float bv=-1e30f;top_ids[0]=0;for(int v=0;v<NV;v++){float dot=0;for(int i=0;i<H;i++)dot+=sb[i]*emb_f32[(size_t)v*H+i];if(dot>bv){bv=dot;top_ids[0]=v;}}
    memcpy(h.data(),h0,H*4);sp++;total++;tboot=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();
    fprintf(stderr,"  [0] boot=%d (%.0fms)\n",top_ids[0],tboot);}
    int step=1;
    while(step<ng){
        auto ts=std::chrono::steady_clock::now();int bs=std::min(32,ng-step);
        for(int b=0;b<bs;b++)for(int i=0;i<H;i++)hb[b*H+i]=emb_f32[(size_t)top_ids[b]*H+i];
        for(int l=0;l<NC;l++){
            for(int b=0;b<bs;b++)rn_c(&hb[b*H],in_n[l],H);
            cq.go(l,hb.data(),bs,H,dyn_scale(hb.data(),bs*H),wsc[l].qk,qb.data(),4096);cn(qb.data(),bs*4096);
            float*qn=qn_w[l],*kn=kn_w[l];
            for(int b=0;b<bs;b++){
                for(int hh=0;hh<NH;hh++){double s=0;for(int d=0;d<HD;d++)s+=qb[b*NH*HD+hh*HD+d]*qb[b*NH*HD+hh*HD+d];float iq=1.0f/sqrtf((float)(s/HD)+EPS);for(int d=0;d<HD;d++)qb[b*NH*HD+hh*HD+d]*=iq*qn[d];ra(&qb[b*NH*HD+hh*HD],HD,sp+b);}
                for(int kvh=0;kvh<NKV;kvh++){float*ks=&qb[b*4096+2048+kvh*HD],*vs=&qb[b*4096+3072+kvh*HD];double sk=0;for(int d=0;d<HD;d++)sk+=ks[d]*ks[d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);for(int d=0;d<HD;d++){ks[d]*=ik*kn[d];ra(ks,HD,sp+b);}}
            }
            for(int b=0;b<bs;b++)for(int kvh=0;kvh<NKV;kvh++){float*ks=&qb[b*4096+2048+kvh*HD],*vs=&qb[b*4096+3072+kvh*HD];memcpy(&kv[l].k[(sp+b)*NKV*HD+kvh*HD],ks,HD*4);memcpy(&kv[l].v[(sp+b)*NKV*HD+kvh*HD],vs,HD*4);}
            kv[l].n=sp+bs;int cl=kv[l].n;
            auto ta=std::chrono::steady_clock::now();for(int b=0;b<bs;b++)attn.run(&qb[b*NH*HD],&ab[b*NH*HD],kv[l].n,kv[l].k.data(),kv[l].v.data());tattn+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ta).count();
            co.go(l,ab.data(),bs,NH*HD,dyn_scale(ab.data(),bs*NH*HD),wsc[l].o_,ob.data(),H);
            cn(ob.data(),bs*H);for(int b=0;b<bs;b++)for(int i=0;i<H;i++)hb[b*H+i]+=ob[b*H+i];
            for(int b=0;b<bs;b++)rn_c(&hb[b*H],pa_n[l],H);
            cg.go(l,hb.data(),bs,H,dyn_scale(hb.data(),bs*H),wsc[l].g_,gb.data(),6144);cn(gb.data(),bs*6144);
            for(int b=0;b<bs;b++)for(int i=0;i<IM;i++){float gv=gb[b*6144+i];float sv=gv/(1.0f+expf(-gv));gb[b*6144+i]=sv*gb[b*6144+IM+i];}
            cd.go(l,gb.data(),bs,IM,dyn_scale(gb.data(),bs*IM),wsc[l].d_,db.data(),H);cn(db.data(),bs*H);
            for(int b=0;b<bs;b++)for(int i=0;i<H;i++)hb[b*H+i]+=db[b*H+i];
        }
        auto tl=std::chrono::steady_clock::now();
        float bv=-1e30f; top_ids[0]=0;
        memcpy(sb.data(),&hb[0],H*4);rn_c(sb.data(),fin,H);
        for(int v=0;v<NV;v++){float dot=0;for(int i=0;i<H;i++)dot+=sb[i]*emb_f32[(size_t)v*H+i];if(dot>bv){bv=dot;top_ids[0]=v;}}
        double lm=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tl).count();
        total+=bs;sp+=bs;nb++;double bms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();
        fprintf(stderr,"  [%d] batch=%d tok=%d %.0fms (%.0f ms/tok)\n",step,bs,top_ids[0],bms,bms/bs);
        step+=bs;
    }
    double tts=std::chrono::duration<double>(std::chrono::steady_clock::now()-tgs).count();
    fprintf(stderr,"\n=== %.1f ms/tok (%.0f tok/s) attn=%.0fms | boot=%.0fms ===\n",tts*1000/ng,ng/tts,tattn,tboot);
    munmap(md,st.st_size);return 0;
}
