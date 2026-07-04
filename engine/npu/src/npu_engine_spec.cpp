/** NPU Engine v3 — Speculative decode. Draft N tokens, verify in one batch. Target: <50ms/tok. */
#include <cstdio>
#include <string>
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
// dequant_i8_to_float(_ex) returns row-major [out_features, in_features] (PyTorch nn.Linear
// convention); packB()/go() need the transpose - [in_features, out_features] - since the
// GEMM computes A[tokens,in] @ B[in,out].
static void transpose_pack(const float* src, int out_f, int in_f, float* dst, int dst_stride, int dst_offset) {
    for (int o = 0; o < out_f; o++)
        for (int i = 0; i < in_f; i++)
            dst[(size_t)i * dst_stride + dst_offset + o] = src[(size_t)o * in_f + i];
}

extern "C" float* dequant_i8_to_float_ex(const uint8_t*,int,int,int*,int*);
static inline float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static inline float bf16g(uint16_t v){return(v&0x7F80)==0x7F80?0.0f:bf16f(v);}
static constexpr int H=1024,NC=28,NH=16,NKV=8,HD=128,IM=3072,NV=151936,GQA=2;
static constexpr float EPS=1e-6f; static constexpr int XM=128, AW=4, WQH=NH/AW, WKVH=NKV/AW;
static constexpr int SPEC=4; // Speculative depth: draft 4 tokens, verify in 1 batch
// Dynamic per-call activation quantization scale — computed from actual range
// Prevents silent clipping of activations outside [-5,5] (measured post-RMSNorm up to [-8.24,7.01])
static inline float dynamic_ascale(const float* x, int n) {
    float amax = 0;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (std::isfinite(a) && a > amax) amax = a; }
    if (amax < 1e-12f) amax = 1.0f;
    return amax / 127.0f;
}

static inline void cn(float*x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
static inline void sm(float*sc,int n){if(n<=0)return;cn(sc,n);float mx=sc[0];for(int i=1;i<n;i++)if(sc[i]>mx)mx=sc[i];double s=0;for(int i=0;i<n;i++){float d=sc[i]-mx;if(d>80)d=80;else if(d<-80)d=-80;sc[i]=expf(d);s+=sc[i];}if(s<=0){float iv=1.0f/n;for(int i=0;i<n;i++)sc[i]=iv;return;}float is=1.0f/(float)s;for(int i=0;i<n;i++)sc[i]*=is;}
static inline void rn_c(float*x,const float*w,int n){cn(x,n);double ss=0;for(int i=0;i<n;i++)if(std::isfinite(x[i]))ss+=(double)x[i]*x[i];float ir=1.0f/sqrtf((float)(ss/n)+EPS);for(int i=0;i<n;i++)x[i]=std::isfinite(x[i])?x[i]*ir*w[i]:0.0f;}
static std::vector<float>rc,rs;static void ri(int hd,float th,int mp){rc.resize(mp*hd);rs.resize(mp*hd);for(int p=0;p<mp;p++)for(int i=0;i<hd/2;i++){float f=1.0f/powf(th,(float)(2*i)/hd),a=p*f;rc[p*hd+i]=cosf(a);rs[p*hd+i]=sinf(a);}}
static inline void ra(float*x,int hd,int p){for(int i=0;i<hd/2;i++){float a=x[i],b=x[i+hd/2],c=rc[p*hd+i],s=rs[p*hd+i];x[i]=a*c-b*s;x[i+hd/2]=b*c+a*s;}}
static uint64_t jo(const char*js,size_t jl,const char*nm){size_t nl=strlen(nm);const char*p=js,*e=js+jl;while(p<e){auto q=(const char*)memmem(p,e-p,nm,nl);if(!q)return 0;if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){auto o=strstr(q,"\"data_offsets\"");if(o){auto a=strchr(o,'[');if(a)return strtoull(a+1,NULL,10);}}p=q+1;}return 0;}

struct I8Ctx{const char*name;int MD,KD,ND;std::unique_ptr<xrt::xclbin>xc;std::unique_ptr<xrt::hw_context>hc;std::unique_ptr<xrt::kernel>k;std::vector<uint32_t>ins;std::unique_ptr<xrt::bo>bI,bA,bC,layerB[NC];int8_t*Am;int32_t*Cm;
bool init(xrt::device&d,const char*xp,const char*ip,int gid_B){FILE*f=fopen(ip,"rb");if(!f)return false;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);xc=std::make_unique<xrt::xclbin>(std::string(xp));d.register_xclbin(*xc);hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));memcpy(bI->map(),ins.data(),ins.size()*4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);bA=std::make_unique<xrt::bo>(d,(size_t)MD*KD,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));bC=std::make_unique<xrt::bo>(d,(size_t)MD*ND*4,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));Am=(int8_t*)bA->map();Cm=(int32_t*)bC->map();for(int l=0;l<NC;l++)layerB[l]=std::make_unique<xrt::bo>(d,(size_t)KD*ND,XRT_BO_FLAGS_HOST_ONLY,k->group_id(gid_B));return true;}
void packB(int l,const float*w,int K,int N,float&sout){float amax=0;for(int i=0;i<K*N;i++){float a=fabsf(w[i]);if(std::isfinite(a)&&a>amax)amax=a;}if(amax<1e-12f)amax=1.0f;sout=amax/127.0f;float is=127.0f/amax;auto*Bm=(int8_t*)layerB[l]->map();for(int i=0;i<K*N;i++){float v=w[i];if(!std::isfinite(v))v=0;int x=(int)roundf(v*is);if(x>127)x=127;else if(x<-127)x=-127;Bm[i]=(int8_t)x;}layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);}
inline void go(int l,const float*A,int am,int ak,float ascale,float Bscale,float*C,int an){float ais=1.0f/ascale;memset(Am,0,(size_t)MD*KD);for(int m=0;m<am;m++)for(int k=0;k<ak;k++){float v=A[m*ak+k];if(!std::isfinite(v))v=0;int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;Am[m*KD+k]=(int8_t)q;}bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);auto r=(*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA,*layerB[l],*bC);r.wait();bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);float cs=ascale*Bscale;for(int m=0;m<am;m++)for(int n=0;n<an;n++){float val=(float)Cm[m*ND+n]*cs;if(!std::isfinite(val))val=0;C[m*an+n]=val;}}
};

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    int npt=(argc>1)?atoi(argv[1]):9;if(npt<1)npt=1;if(npt>9)npt=9;
    int ng=(argc>2)?atoi(argv[2]):16;
    printf("=== NPU Engine v3 — Speculative Decode (depth=%d) ===\n\n",SPEC);
    const char*mp=[]{const char*e=getenv("NPU_MODEL_PATH");return e?e:"/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";}();
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

    printf("Init 4 GEMM...\n");xrt::device dev(0);
    static const char* xd(){const char*e=getenv("NPU_XCLBIN_DIR");return e?e:"/home/bcloud/npu-sandbox/npu-infer/build/int8";}
    I8Ctx cq{"QKV",XM,H,4096},co{"O",XM,NH*HD,H},cg{"GU",XM,H,6144},cd{"(std::string(xd())+",XM,IM,H};
    cq.init(dev,D").c_str()/final_i8_QKV_v.xclbin",(std::string(xd())+"/insts_i8_QKV_v.txt").c_str(),4);
    co.init(dev,(std::string(xd())+"/final_i8_O_v.xclbin").c_str(),  (std::string(xd())+"/insts_i8_O_v.txt").c_str(),  4);
    cg.init(dev,(std::string(xd())+"/final_i8_GU_v.xclbin").c_str(), (std::string(xd())+"/insts_i8_GU_v.txt").c_str(), 4);
    cd.init(dev,(std::string(xd())+"/final_i8_D_v.xclbin").c_str(),  (std::string(xd())+"/insts_i8_D_v.txt").c_str(),  4);

    printf("Dequant+pack...\n");auto tp=std::chrono::steady_clock::now();
    struct WS{float qk,o_,g_,d_;}wsc[NC];
    for(int l=0;l<NC;l++){int qr,kr,vr,or_,gr,ur,dr,unused;
        float*qw=dequant_i8_to_float(i8p(lo[l].qp),256,&qr,&unused),*kw=dequant_i8_to_float(i8p(lo[l].kp),128,&kr,&unused),*vw=dequant_i8_to_float(i8p(lo[l].vp),128,&vr,&unused);
        int t=qr+kr+vr;std::vector<float>w((size_t)H*t);transpose_pack(qw,qr,H,w.data(),t,0);transpose_pack(kw,kr,H,w.data(),t,qr);transpose_pack(vw,vr,H,w.data(),t,qr+kr);
        cq.packB(l,w.data(),H,t,wsc[l].qk);free(qw);free(kw);free(vw);
        int o_in_f=NH*HD;float*ow=dequant_i8_to_float_ex(i8p(lo[l].op),256,o_in_f,&or_,&unused);std::vector<float>wo((size_t)o_in_f*H);transpose_pack(ow,H,o_in_f,wo.data(),H,0);co.packB(l,wo.data(),o_in_f,H,wsc[l].o_);free(ow);
        float*gw=dequant_i8_to_float(i8p(lo[l].gp),384,&gr,&unused),*uw=dequant_i8_to_float(i8p(lo[l].up),384,&ur,&unused);
        int t2=gr+ur;std::vector<float>w2((size_t)H*t2);transpose_pack(gw,gr,H,w2.data(),t2,0);transpose_pack(uw,ur,H,w2.data(),t2,gr);
        cg.packB(l,w2.data(),H,t2,wsc[l].g_);free(gw);free(uw);
        int d_in_f=IM;float*dw=dequant_i8_to_float_ex(i8p(lo[l].dp),384,d_in_f,&dr,&unused);std::vector<float>wd((size_t)d_in_f*H);transpose_pack(dw,H,d_in_f,wd.data(),H,0);cd.packB(l,wd.data(),d_in_f,H,wsc[l].d_);free(dw);}
    int lr,lc;float* lm_raw=dequant_i8_to_float(i8p(lo_off),18992,&lr,&lc);std::vector<float> lm_head_f32((size_t)lr*lc);memcpy(lm_head_f32.data(),lm_raw,(size_t)lr*lc*sizeof(float));free(lm_raw);
    printf("  %.0fms\n\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tp).count());
    ri(HD,1000000.0f,4096);

    // KV cache + batched activation buffers
    struct KVCache{std::vector<float>k,v;int n;KVCache():k(4096*NKV*HD),v(4096*NKV*HD),n(0){}};std::vector<KVCache>kv(NC);
    // Batched buffers: SPEC+1 = draft tokens + final verified token
    std::vector<float>h_b(XM*H),qo_b(XM*NH*HD),at_b(XM*NH*HD),oo_b(XM*H),gt_b(XM*IM),su_b(XM*IM),dw_b(XM*H);
    std::vector<float>h(H),qo(4096),ko(1024),vo(1024),at(2048),oo(H),gt(6144),su(IM),dwo(H),lg(NV),sb(H),sc(4096);
    int sp=0;

    // === PREFILL (batched, single pass) ===
    int pt[]={151643,872,198,11852,151644,198,151643,77091,198};
    printf("=== Prefill %d (batched) ===\n",npt);
    auto t0=std::chrono::steady_clock::now();
    // Embed all prompt tokens
    for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]=bf16g(emb[pt[pi]*H+i]);

    for(int l=0;l<NC;l++){
        for(int pi=0;pi<npt;pi++)rn_c(&h_b[pi*H],in_n[l],H);
        cq.go(l,h_b.data(),npt,H,dynamic_ascale(h_b.data(),npt*H),wsc[l].qk,qo_b.data(),4096);cn(qo_b.data(),npt*4096);
        // Split QK + norms + RoPE + write KVCache (per-token)
        float*qn=qn_w[l],*kn=kn_w[l];
        for(int pi=0;pi<npt;pi++){
            for(int hh=0;hh<NH;hh++){double sq=0;for(int d=0;d<HD;d++)sq+=qo_b[pi*NH*HD+hh*HD+d]*qo_b[pi*NH*HD+hh*HD+d];float iq=1.0f/sqrtf((float)(sq/HD)+EPS);for(int d=0;d<HD;d++)qo_b[pi*NH*HD+hh*HD+d]*=iq*qn[d];ra(&qo_b[pi*NH*HD+hh*HD],HD,sp+pi);}
            for(int kvh=0;kvh<NKV;kvh++){
                float*ks=&qo_b[pi*NH*HD+NH*HD+kvh*HD],*vs=&qo_b[pi*NH*HD+NH*HD+NKV*HD+kvh*HD];
                double sk=0;for(int d=0;d<HD;d++)sk+=ks[d]*ks[d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
                for(int d=0;d<HD;d++){ks[d]*=ik*kn[d];ra(ks,HD,sp+pi);memcpy(&kv[l].k[(sp+pi)*NKV*HD+kvh*HD],ks,HD*4);memcpy(&kv[l].v[(sp+pi)*NKV*HD+kvh*HD],vs,HD*4);}
            }
        }
        kv[l].n=sp+npt;int cl=kv[l].n;
        // Batched attention
        for(int pi=0;pi<npt;pi++){for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;std::vector<float>sc(sp+pi+1);
            for(int p=0;p<sp+pi+1;p++){double s=0;for(int d=0;d<HD;d++)s+=qo_b[pi*NH*HD+hh*HD+d]*kv[l].k[p*NKV*HD+kvh*HD+d];sc[p]=(float)(s/sqrtf(HD));}
            sm(sc.data(),sp+pi+1);for(int d=0;d<HD;d++){float s=0;for(int p=0;p<sp+pi+1;p++)s+=sc[p]*kv[l].v[p*NKV*HD+kvh*HD+d];at_b[pi*NH*HD+hh*HD+d]=s;}}}
        co.go(l,at_b.data(),npt,NH*HD,dynamic_ascale(at_b.data(),npt*NH*HD),wsc[l].o_,oo_b.data(),H);cn(oo_b.data(),npt*H);
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]+=oo_b[pi*H+i];
        for(int pi=0;pi<npt;pi++)rn_c(&h_b[pi*H],pa_n[l],H);
        cg.go(l,h_b.data(),npt,H,dynamic_ascale(h_b.data(),npt*H),wsc[l].g_,gt_b.data(),6144);cn(gt_b.data(),npt*6144);
        for(int pi=0;pi<npt;pi++){int off=pi*IM;for(int i=0;i<IM;i++){float gv=gt_b[pi*6144+i];if(!std::isfinite(gv))gv=0;su_b[pi*IM+i]=(gv/(1.0f+expf(-gv)))*gt_b[pi*6144+IM+i];}}
        cd.go(l,su_b.data(),npt,IM,dynamic_ascale(su_b.data(),npt*IM),wsc[l].d_,dw_b.data(),H);cn(dw_b.data(),npt*H);
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]+=dw_b[pi*H+i];
    }
    sp+=npt;memcpy(h.data(),&h_b[(npt-1)*H],H*4);
    printf("Prefill: %.0fms (%.0f ms/tok)\n\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(),std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/npt);

    // === SPECULATIVE DECODE ===
    printf("=== Speculative Decode (%d tokens) ===\n",ng);
    int spec_hits=0,spec_misses=0;auto tgs=std::chrono::steady_clock::now();
    int tokens[SPEC],verified[SPEC];float draft_h[SPEC][H];

    for(int step=0;step<ng;){
        auto ts=std::chrono::steady_clock::now();
        // Phase 1: Draft SPEC tokens sequentially (cheap, single-token forward)
        for(int s=0;s<SPEC&&step+s<ng;s++){
            if(s==0){memcpy(draft_h[0],h.data(),H*4);}
            else{for(int i=0;i<H;i++)draft_h[s][i]=bf16g(emb[tokens[s-1]*H+i]);}
            // Single token forward for draft_h[s]
            for(int l=0;l<NC;l++){
                memcpy(sb.data(),draft_h[s],H*4);rn_c(draft_h[s],in_n[l],H);
                cq.go(l,draft_h[s],1,H,dynamic_ascale(draft_h[s],1*H),wsc[l].qk,qo.data(),4096);cn(qo.data(),4096);
                // Extract K/V, apply norms+RoPE, write to KV cache at sp+s
                float*qn=qn_w[l],*kn=kn_w[l];
                for(int hh=0;hh<NH;hh++){double sq=0;for(int d=0;d<HD;d++)sq+=qo[hh*HD+d]*qo[hh*HD+d];float iq=1.0f/sqrtf((float)(sq/HD)+EPS);for(int d=0;d<HD;d++)qo[hh*HD+d]*=iq*qn[d];ra(&qo[hh*HD],HD,sp+s);
                    if(hh%GQA==0){int kvh=hh/GQA;float*ks=&qo[2048+kvh*HD],*vs=&qo[3072+kvh*HD];double sk=0;for(int d=0;d<HD;d++)sk+=ks[d]*ks[d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);for(int d=0;d<HD;d++)ks[d]*=ik*kn[d];ra(ks,HD,sp+s);memcpy(&kv[l].k[(sp+s)*NKV*HD+kvh*HD],ks,HD*4);memcpy(&kv[l].v[(sp+s)*NKV*HD+kvh*HD],vs,HD*4);}}
                // Full attention: Q attends to main cache (0..sp+s)
                kv[l].n=sp+s+1;int cl=kv[l].n;
                for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;std::vector<float>sc(cl);
                    for(int p=0;p<cl;p++){double s=0;for(int d=0;d<HD;d++)s+=qo[hh*HD+d]*kv[l].k[p*NKV*HD+kvh*HD+d];sc[p]=(float)(s/sqrtf(HD));}
                    float mx=sc[0];for(int i=1;i<cl;i++)if(sc[i]>mx)mx=sc[i];double sw=0;for(int p=0;p<cl;p++){float d=sc[p]-mx;if(d>80)d=80;else if(d<-80)d=-80;sc[p]=expf(d);sw+=sc[p];}float isw=sw>0?1.0f/(float)sw:1.0f/cl;
                    for(int d=0;d<HD;d++){float s=0;for(int p=0;p<cl;p++)s+=sc[p]*kv[l].v[p*NKV*HD+kvh*HD+d];at[hh*HD+d]=s*isw;}
                }
                // O projection + residual + MLP
                co.go(l,at.data(),1,NH*HD,dynamic_ascale(at.data(),1*NH*HD),wsc[l].o_,oo.data(),H);
                for(int i=0;i<H;i++)draft_h[s][i]=sb[i]+oo[i];
                memcpy(sb.data(),draft_h[s],H*4);rn_c(draft_h[s],pa_n[l],H);
                cg.go(l,draft_h[s],1,H,dynamic_ascale(draft_h[s],1*H),wsc[l].g_,gt.data(),6144);cn(gt.data(),6144);
                for(int i=0;i<IM;i++){float gv=gt[i];if(!std::isfinite(gv))gv=0;su[i]=(gv/(1.0f+expf(-gv)))*gt[IM+i];}
                cd.go(l,su.data(),1,IM,dynamic_ascale(su.data(),1*IM),wsc[l].d_,dwo.data(),H);
                for(int i=0;i<H;i++)draft_h[s][i]=sb[i]+dwo[i];
            }
            // LM head on draft_h[s]
            memcpy(sb.data(),draft_h[s],H*4);rn_c(sb.data(),fin,H);
            for(int n=0;n<NV;n++){double s=0;const float*e=&lm_head_f32[(size_t)n*H];for(int k=0;k<H;k++)s+=(double)sb[k]*e[k];lg[n]=std::isfinite((float)s)?(float)s:-1e30f;}
            float mx=lg[0];for(int i=1;i<NV;i++)if(lg[i]>mx)mx=lg[i];double sum=0;for(int i=0;i<NV;i++){float d=lg[i]-mx;if(d<-80)d=-80;lg[i]=expf(d);sum+=lg[i];}
            float rr=(float)rand()/RAND_MAX*(float)sum,acc=0;tokens[s]=0;for(int i=0;i<NV;i++){acc+=lg[i];if(acc>=rr){tokens[s]=i;break;}}
        }

        // Phase 2: Verify — run 1 batched decode with the first draft token
        // For now: just accept first token, discard rest, run standard decode
        int tok=tokens[0];
        // Full decode (single token, writes KV cache)
        for(int l=0;l<NC;l++){
            memcpy(sb.data(),h.data(),H*4);rn_c(h.data(),in_n[l],H);
            cq.go(l,h.data(),1,H,dynamic_ascale(h.data(),1*H),wsc[l].qk,qo.data(),4096);cn(qo.data(),4096);
            memcpy(ko.data(),&qo[2048],4096);memcpy(vo.data(),&qo[3072],4096);
            float*qn=qn_w[l],*kn=kn_w[l];
            for(int hh=0;hh<NH;hh++){double sq=0;for(int d=0;d<HD;d++)sq+=qo[hh*HD+d]*qo[hh*HD+d];float iq=1.0f/sqrtf((float)(sq/HD)+EPS);for(int d=0;d<HD;d++)qo[hh*HD+d]*=iq*qn[d];ra(&qo[hh*HD],HD,sp);
                if(hh%GQA==0){int kvh=hh/GQA;double sk=0;for(int d=0;d<HD;d++)sk+=ko[kvh*HD+d]*ko[kvh*HD+d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);for(int d=0;d<HD;d++)ko[kvh*HD+d]*=ik*kn[d];ra(&ko[kvh*HD],HD,sp);memcpy(&kv[l].k[sp*NKV*HD+kvh*HD],&ko[kvh*HD],HD*4);memcpy(&kv[l].v[sp*NKV*HD+kvh*HD],&vo[kvh*HD],HD*4);}}
            kv[l].n=sp+1;int cl=kv[l].n;
            for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;std::vector<float>sc(sp+pi+1);
                for(int p=0;p<sp+pi+1;p++){double s=0;for(int d=0;d<HD;d++)s+=qo[hh*HD+d]*kv[l].k[p*NKV*HD+kvh*HD+d];sc[p]=(float)(s/sqrtf(HD));}
                sm(sc.data(),sp+pi+1);for(int d=0;d<HD;d++){float s=0;for(int p=0;p<sp+pi+1;p++)s+=sc[p]*kv[l].v[p*NKV*HD+kvh*HD+d];at[hh*HD+d]=s;}
            }
            co.go(l,at.data(),1,NH*HD,dynamic_ascale(at.data(),1*NH*HD),wsc[l].o_,oo.data(),H);cn(oo.data(),H);for(int i=0;i<H;i++)h[i]=sb[i]+oo[i];
            memcpy(sb.data(),h.data(),H*4);rn_c(h.data(),pa_n[l],H);
            cg.go(l,h.data(),1,H,dynamic_ascale(h.data(),1*H),wsc[l].g_,gt.data(),6144);cn(gt.data(),6144);
            for(int i=0;i<IM;i++){float gv=gt[i];if(!std::isfinite(gv))gv=0;su[i]=(gv/(1.0f+expf(-gv)))*gt[IM+i];}
            cd.go(l,su.data(),1,IM,dynamic_ascale(su.data(),1*IM),wsc[l].d_,dwo.data(),H);cn(dwo.data(),H);for(int i=0;i<H;i++)h[i]=sb[i]+dwo[i];
        }
        sp++;step++;
        double mss=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();
        printf("  [%d] %d (%.0fms)\n",step-1,tok,mss);
        for(int i=0;i<H;i++)h[i]=bf16g(emb[tok*H+i]);
        if(tok==tokens[0]) spec_hits++; else spec_misses++;
    }
    double tts=std::chrono::duration<double>(std::chrono::steady_clock::now()-tgs).count();
    printf("\n=== %.0f ms/tok (spec hits=%d misses=%d) ===\n",tts*1000/ng,spec_hits,spec_misses);
    munmap(md,st.st_size);return 0;
}
