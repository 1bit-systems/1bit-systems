/** npu_engine_spec.cpp — GPU-accelerated speculative decoding.
 *
 * GPU runs Eagle3 draft model (1 layer, 8.5M params) → N candidate tokens
 * NPU verifies all candidates in one batched forward pass (28 layers)
 *
 * Timeline:
 *   GPU draft N=4: ~50µs (1 layer, small)
 *   NPU verify N+1: ~6.2ms (28 layers, full model)
 *   Acceptance: ~1µs
 *   Effective: 69 tok/s × (N+1) / 1 = ~345 tok/s at 100% accept
 *              ~250 tok/s at 70% accept (typical)
 *
 * Build:
 *   g++ -std=c++17 -O3 -mavx512f -march=native \
 *       npu_engine_spec.cpp dequant_q4nx.cpp \
 *       -I/opt/xrt/include -L/opt/xrt/lib \
 *       -I$(hipconfig --path)/include -L$(hipconfig --path)/lib \
 *       -lxrt_coreutil -lxrt_core -luuid -lamdhip64 \
 *       -lpthread -fopenmp -laiebu -o npu_engine_spec
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_elf.h>
#include <aiebu/aiebu_assembler.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <omp.h>
#include <immintrin.h>

// GPU kernel wrappers
extern "C" {
    void launch_gemv_tiled(const __half*, const __half*, __half*, int, int, int, int, hipStream_t);
    void launch_lmhead(const __half*, const __half*, float*, int, int, int, int, hipStream_t);
    void launch_gemv_simple(const __half*, const __half*, __half*, int, int, hipStream_t);
    void launch_rmsnorm(__half*, const __half*, int, float, hipStream_t);
    void launch_add(__half*, const __half*, int, hipStream_t);
    void launch_copy(const __half*, __half*, int, hipStream_t);
    void launch_silu_mul(const __half*, const __half*, __half*, int, hipStream_t);
    void launch_rope(__half*, const __half*, const __half*, int, int, int, hipStream_t);
}


extern "C" float* dequant_i8_to_float(const uint8_t*,int,int*,int*);
extern "C" int rcpp_kv_cache_attn_decode(
    const void*,const void*,const void*,void*,int,int,int,int,float,void*);

static constexpr float EPS = 1e-6f;
static constexpr int MAX_SEQ = 4096;

static inline float bf16g(uint16_t v){if((v&0x7F80)==0x7F80)return 0.0f;uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f;}
static inline float dyn_scale(const float*x,int n){float a=0;for(int i=0;i<n;i++){float f=fabsf(x[i]);if(std::isfinite(f)&&f>a)a=f;}return a<1e-12f?1.0f:a/127.0f;}
static inline void cn(float*x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
static void rms_norm(float*x,const float*w,int n){cn(x,n);double ss=0;for(int i=0;i<n;i++)if(std::isfinite(x[i]))ss+=(double)x[i]*x[i];float ir=1.0f/sqrtf((float)(ss/n)+EPS);for(int i=0;i<n;i++)x[i]=std::isfinite(x[i])?x[i]*ir*w[i]:0.0f;}
static inline float silu(float x){return x/(1.0f+expf(-x));}

struct RopeTables{std::vector<float>cos,sin;int mp=0,hd=0;float th=1000000.0f;
    void init(int hd_,int mp_,float th_){hd=hd_;mp=mp_;th=th_;cos.resize(mp*hd);sin.resize(mp*hd);int hd2=hd/2;
        for(int p=0;p<mp;p++)for(int d=0;d<hd2;d++){float f=1.0f/powf(th,(float)d/hd2),a=p*f;cos[p*hd+d]=cosf(a);sin[p*hd+d]=sinf(a);}}
    void apply(float*qk,int pos,int nh)const{int hd2=hd/2;for(int h=0;h<nh;h++){float*x=qk+h*hd;for(int d=0;d<hd2;d++){float a=x[d],b=x[d+hd2],c=cos[pos*hd+d],s=sin[pos*hd+d];x[d]=a*c-b*s;x[d+hd2]=b*c+a*s;}}}
};

static void quantize_avx(const float*A,int8_t*D,int n,float s){float is=1.0f/s;__m512 v_is=_mm512_set1_ps(is),v_max=_mm512_set1_ps(127.0f),v_min=_mm512_set1_ps(-127.0f);int i=0;for(;i+16<=n;i+=16){__m512 v=_mm512_loadu_ps(&A[i]);v=_mm512_mul_ps(v,v_is);v=_mm512_max_ps(v_min,_mm512_min_ps(v_max,v));__m512i vi=_mm512_cvtps_epi32(v);__m256i lo=_mm512_cvtepi32_epi16(vi);__m128i i8=_mm256_cvtepi16_epi8(lo);_mm_storeu_si128((__m128i*)&D[i],i8);}for(;i<n;i++){int q=(int)roundf(A[i]*is);if(q>127)q=127;else if(q<-127)q=-127;D[i]=(int8_t)q;}}
static void dequant_avx(const int16_t*Cm,float*C,int n,float s){__m512 v_s=_mm512_set1_ps(s);int i=0;for(;i+16<=n;i+=16){__m256i vi16=_mm256_loadu_si256((__m256i*)&Cm[i]);__m512i vi32=_mm512_cvtepi16_epi32(vi16);__m512 v=_mm512_cvtepi32_ps(vi32);v=_mm512_mul_ps(v,v_s);_mm512_storeu_ps(&C[i],v);}for(;i<n;i++)C[i]=(float)Cm[i]*s;}

struct NpuGemmCtx{int MD,KD,ND;std::vector<uint32_t>ins;std::unique_ptr<xrt::xclbin>xc;std::unique_ptr<xrt::hw_context>hc;std::unique_ptr<xrt::module>mdl;std::unique_ptr<xrt::elf>elf;std::unique_ptr<xrt::ext::kernel>k;std::unique_ptr<xrt::bo>bA,bB,bC;int8_t*Am=nullptr;int16_t*Cm=nullptr;xrt::run run_;bool busy_=false;
    bool init(xrt::device&d,const char*xp,const char*ip,int md,int kd,int nd){
        MD=md;KD=kd;ND=nd;FILE*f=fopen(ip,"rb");if(!f)return false;fseek(f,0,SEEK_END);long sz=ftell(f);fseek(f,0,SEEK_SET);ins.resize(sz/4);if(fread(ins.data(),4,ins.size(),f)!=ins.size()){fclose(f);return false;}fclose(f);
        try{std::vector<char>iraw((char*)ins.data(),(char*)ins.data()+ins.size()*4);aiebu::aiebu_assembler asmblr(aiebu::aiebu_assembler::buffer_type::blob_instr_transaction,iraw);auto e=asmblr.get_elf();xc=std::make_unique<xrt::xclbin>(std::string(xp));d.register_xclbin(*xc);hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());elf=std::make_unique<xrt::elf>(e.data(),e.size());}catch(...){return false;}
        mdl=std::make_unique<xrt::module>(*elf);k=std::make_unique<xrt::ext::kernel>(*hc,*mdl,"MLIR_AIE");
        bA=std::make_unique<xrt::bo>(d,(size_t)MD*KD,XRT_BO_FLAGS_HOST_ONLY,0);bC=std::make_unique<xrt::bo>(d,(size_t)MD*ND*2,XRT_BO_FLAGS_HOST_ONLY,0);bB=std::make_unique<xrt::bo>(d,(size_t)KD*ND,XRT_BO_FLAGS_HOST_ONLY,0);Am=(int8_t*)bA->map();Cm=(int16_t*)bC->map();return true;}
    void packB(const float*w,int K,int N,float&sout){float amax=0;for(int i=0;i<K*N;i++){float a=fabsf(w[i]);if(std::isfinite(a)&&a>amax)amax=a;}sout=(amax<1e-12f)?1.0f:amax/127.0f;float is=127.0f/(amax<1e-12f?1.0f:amax);auto*Bm=(int8_t*)bB->map();for(int i=0;i<K*N;i++){float v=w[i];if(!std::isfinite(v))v=0;int q=(int)roundf(v*is);if(q>127)q=127;else if(q<-127)q=-127;Bm[i]=(int8_t)q;}bB->sync(XCL_BO_SYNC_BO_TO_DEVICE);}
    void launch(const float*A,int am,int ak,float as_){memset(Am,0,(size_t)am*KD);quantize_avx(A,Am,am*ak,as_);bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);run_=k->operator()(3,0,0,*bA,*bB,*bC);busy_=true;}
    void wait(float*C,int am,int an,float as_,float Bs){if(!busy_)return;run_.wait();bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);float cs=as_*Bs;dequant_avx(Cm,C,am*an,cs);busy_=false;}
    void go(int l,const float*A,int am,int ak,float as_,float Bs,float*C,int an){(void)l;launch(A,am,ak,as_);wait(C,am,an,as_,Bs);}
};

// ── Tiled f16 GEMV (column-major weights) ─────────────────────────
struct GpuGemvCtx{int K,N;__half*d_W=nullptr;
    bool init(int k,int n,const float*W_f32){K=k;N=n;size_t sz=(size_t)N*K*sizeof(__half);if(hipSuccess!=hipMalloc(&d_W,sz))return false;std::vector<__half>buf((size_t)N*K);for(int ni=0;ni<N;ni++)for(int ki=0;ki<K;ki++)buf[(size_t)ni*K+ki]=__float2half(W_f32[(size_t)ki*N+ni]);hipMemcpy(d_W,buf.data(),sz,hipMemcpyHostToDevice);return true;}
    void run(const float*A_f32,__half*C_f16,hipStream_t s,int grid,int block,size_t shmem){
        __half*d_A;hipMalloc(&d_A,(size_t)K*sizeof(__half));std::vector<__half>buf(K);for(int i=0;i<K;i++)buf[i]=__float2half(A_f32[i]);hipMemcpyAsync(d_A,buf.data(),K*sizeof(__half),hipMemcpyHostToDevice,s);
        launch_gemv_simple(d_A,d_W,C_f16,K,N,s);hipFreeAsync(d_A,s);}
    void destroy(){if(d_W)hipFree(d_W);}
};




// ── JSON ──────────────────────────────────────────────────────────
static uint64_t json_off(const char*js,size_t jl,const char*nm){size_t nl=strlen(nm);const char*p=js,*e=js+jl;
    while(p<e){auto q=(const char*)memmem(p,e-p,nm,nl);if(!q)return 0;if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){auto o=strstr(q,"\"data_offsets\"");if(o){auto a=strchr(o,'[');if(a)return strtoull(a+1,NULL,10);}}p=q+1;}return 0;}
struct MdlCfg{int H=0,NC=0,NH=0,NKV=0,HD=0,IM=0,NV=0,GQA=0;int qkv_n()const{return NH*HD+2*NKV*HD;}};
struct LOff{uint64_t q,k,v,o,g,u,d,in,pa,qn,kn;};

// ══════════════════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════════════════
int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"Usage: %s <model.q4nx> [draft.bin] [tokens]\n",argv[0]);return 1;}
    const char*mp=argv[1],*dp=argc>2?argv[2]:"spec-decode/checkpoints/eagle3_draft_v2.bin";
    int n_gen=argc>3?atoi(argv[3]):64;

    // ── Load model ────────────────────────────────────────────────
    int fd=open(mp,O_RDONLY);struct stat st;fstat(fd,&st);
    uint8_t*md=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    uint64_t hsz;memcpy(&hsz,md,8);uint64_t df=8+hsz;
    auto i8p=[&](uint64_t o){return md+df+o;};auto emb_ptr=(const uint16_t*)(md+df);
    const char*js=(const char*)(md+8);size_t jl=hsz;
    MdlCfg cfg;cfg.H=1024;cfg.NC=28;cfg.NH=16;cfg.NKV=8;cfg.HD=128;cfg.IM=3072;cfg.NV=151936;cfg.GQA=2;
    std::vector<LOff>lo(cfg.NC);char k[128];
    for(int l=0;l<cfg.NC;l++){
        snprintf(k,128,"model.layers.%d.self_attn.q_proj.weight",l);lo[l].q=json_off(js,jl,k);
        snprintf(k,128,"model.layers.%d.self_attn.k_proj.weight",l);lo[l].k=json_off(js,jl,k);
        snprintf(k,128,"model.layers.%d.self_attn.v_proj.weight",l);lo[l].v=json_off(js,jl,k);
        snprintf(k,128,"model.layers.%d.self_attn.o_proj.weight",l);lo[l].o=json_off(js,jl,k);
        snprintf(k,128,"model.layers.%d.mlp.gate_proj.weight",l);lo[l].g=json_off(js,jl,k);
        snprintf(k,128,"model.layers.%d.mlp.up_proj.weight",l);lo[l].u=json_off(js,jl,k);
        snprintf(k,128,"model.layers.%d.mlp.down_proj.weight",l);lo[l].d=json_off(js,jl,k);
        snprintf(k,128,"model.layers.%d.input_layernorm.weight",l);lo[l].in=json_off(js,jl,k);
        snprintf(k,128,"model.layers.%d.post_attention_layernorm.weight",l);lo[l].pa=json_off(js,jl,k);
        snprintf(k,128,"model.layers.%d.self_attn.q_norm.weight",l);lo[l].qn=json_off(js,jl,k);
        snprintf(k,128,"model.layers.%d.self_attn.k_norm.weight",l);lo[l].kn=json_off(js,jl,k);
    }
    uint64_t no=json_off(js,jl,"model.norm.weight"),lm_off=json_off(js,jl,"lm_head.weight");
    std::vector<std::vector<float>>in_norm(cfg.NC,std::vector<float>(cfg.H)),pa_norm(cfg.NC,std::vector<float>(cfg.H));
    std::vector<std::vector<float>>q_norm(cfg.NC,std::vector<float>(cfg.HD)),k_norm(cfg.NC,std::vector<float>(cfg.HD));
    std::vector<float>fn(cfg.H),ef32((size_t)cfg.NV*cfg.H);
    for(int l=0;l<cfg.NC;l++){auto iw=(const uint16_t*)(md+df+lo[l].in),pw=(const uint16_t*)(md+df+lo[l].pa),qw=(const uint16_t*)(md+df+lo[l].qn),kw=(const uint16_t*)(md+df+lo[l].kn);for(int i=0;i<cfg.H;i++){in_norm[l][i]=bf16g(iw[i]);pa_norm[l][i]=bf16g(pw[i]);}for(int i=0;i<cfg.HD;i++){q_norm[l][i]=bf16g(qw[i]);k_norm[l][i]=bf16g(kw[i]);}}
    {auto fw=(const uint16_t*)(md+df+no);for(int i=0;i<cfg.H;i++)fn[i]=bf16g(fw[i]);}
    for(int ni=0;ni<cfg.NV;ni++)for(int i=0;i<cfg.H;i++)ef32[(size_t)ni*cfg.H+i]=bf16g(emb_ptr[ni*cfg.H+i]);
    printf("=== Spec Decode === H=%d NC=%d NV=%d\n",cfg.H,cfg.NC,cfg.NV);

    // ── HW ────────────────────────────────────────────────────────
    hipSetDevice(0);hipStream_t gs;hipStreamCreate(&gs);
    xrt::device nd(0);
    const char*xd=getenv("NPU_XCLBIN_DIR")?:"/home/bcloud/npu-sandbox/npu-infer/build/int8";
    auto xp=[&](const char*t){static char b[256];snprintf(b,256,"%s/final_i8_%s_v.xclbin",xd,t);return b;};
    auto ip_=[&](const char*t){static char b[256];snprintf(b,256,"%s/insts_i8_%s_v.txt",xd,t);return b;};
    int XM=128;
    NpuGemmCtx cq,co,cg,cd;
    if(!cq.init(nd,xp("QKV"),ip_("QKV"),XM,cfg.H,cfg.qkv_n())){fprintf(stderr,"FAIL QKV\n");return 1;}
    if(!co.init(nd,xp("O"),ip_("O"),XM,cfg.NH*cfg.HD,cfg.H)){fprintf(stderr,"FAIL O\n");return 1;}
    if(!cg.init(nd,xp("GU"),ip_("GU"),XM,cfg.H,2*cfg.IM)){fprintf(stderr,"FAIL GU\n");return 1;}
    if(!cd.init(nd,xp("D"),ip_("D"),XM,cfg.IM,cfg.H)){fprintf(stderr,"FAIL D\n");return 1;}
    struct WS{float qk,o_,g_,d_;};std::vector<WS>ws(cfg.NC);
    for(int l=0;l<cfg.NC;l++){int qr,kr,vr,or_,gr,ur,dr,u;
        float*qw=dequant_i8_to_float(i8p(lo[l].q),256,&qr,&u),*kw=dequant_i8_to_float(i8p(lo[l].k),128,&kr,&u),*vw=dequant_i8_to_float(i8p(lo[l].v),128,&vr,&u);
        int t=qr+kr+vr;std::vector<float>w((size_t)cfg.H*t);for(int kk=0;kk<cfg.H;kk++){memcpy(&w[(size_t)kk*t],&qw[(size_t)kk*qr],(size_t)qr*4);memcpy(&w[(size_t)kk*t+qr],&kw[(size_t)kk*kr],(size_t)kr*4);memcpy(&w[(size_t)kk*t+qr+kr],&vw[(size_t)kk*vr],(size_t)vr*4);}
        cq.packB(w.data(),cfg.H,t,ws[l].qk);free(qw);free(kw);free(vw);
        float*ow=dequant_i8_to_float(i8p(lo[l].o),256,&or_,&u);co.packB(ow,or_,cfg.H,ws[l].o_);free(ow);
        float*gw=dequant_i8_to_float(i8p(lo[l].g),384,&gr,&u),*uw=dequant_i8_to_float(i8p(lo[l].u),384,&ur,&u);
        int t2=gr+ur;std::vector<float>w2((size_t)cfg.H*t2);for(int kk=0;kk<cfg.H;kk++){memcpy(&w2[(size_t)kk*t2],&gw[(size_t)kk*gr],(size_t)gr*4);memcpy(&w2[(size_t)kk*t2+gr],&uw[(size_t)kk*ur],(size_t)ur*4);}
        cg.packB(w2.data(),cfg.H,t2,ws[l].g_);free(gw);free(uw);
        float*dw=dequant_i8_to_float(i8p(lo[l].d),384,&dr,&u);cd.packB(dw,dr,cfg.H,ws[l].d_);free(dw);}

    // ── GPU KV cache + attn buffers ────────────────────────────────
    __half *dK=nullptr,*dV=nullptr,*dQ=nullptr,*dAO=nullptr;
    hipMalloc(&dK,(size_t)MAX_SEQ*cfg.NKV*cfg.HD*2);hipMemset(dK,0,(size_t)MAX_SEQ*cfg.NKV*cfg.HD*2);
    hipMalloc(&dV,(size_t)MAX_SEQ*cfg.NKV*cfg.HD*2);hipMemset(dV,0,(size_t)MAX_SEQ*cfg.NKV*cfg.HD*2);
    hipMalloc(&dQ,(size_t)cfg.NH*cfg.HD*2);hipMalloc(&dAO,(size_t)cfg.NH*cfg.HD*2);

    // ── Load draft model weights onto GPU ─────────────────────────
    int H=cfg.H,V=cfg.NV,NH=cfg.NH,NKV=cfg.NKV,HD=cfg.HD,IM=cfg.IM,NTL=5;
    struct DWS{__half *embed,*fc,*qP,*kP,*vP,*oP,*gate,*up,*dp,*lmH;
              __half *hN,*iLN,*paN,*fN,*qN,*kN;};
    DWS dws={};bool dl=false;
    {
        FILE*f=fopen(dp,"rb");if(!f){printf("NO DRAFT at %s\n",dp);goto no_draft;}
        auto rv=[&](auto&v,size_t n){v.resize(n);return fread(v.data(),4,n,f)==n;};
        auto gpu_up=[&](__half*&gpu,const std::vector<float>&cpu,int n){
            hipMalloc(&gpu,(size_t)n*2);std::vector<__half>buf(n);
            for(int i=0;i<n;i++)buf[i]=__float2half(cpu[i]);
            hipMemcpy(gpu,buf.data(),(size_t)n*2,hipMemcpyHostToDevice);};
        std::vector<float>et,fc,qpk,kpk,vpk,opk,gpk,upk,dpk,lmk;
        std::vector<float>hn,iln,pan,fn_,qn,kn;
        bool ok=true;
        ok&=rv(et,(size_t)V*H);ok&=rv(fc,(size_t)H*NTL*H);ok&=rv(hn,H);ok&=rv(iln,H);
        ok&=rv(qpk,(size_t)(NH*HD)*(2*H));ok&=rv(kpk,(size_t)(NKV*HD)*(2*H));ok&=rv(vpk,(size_t)(NKV*HD)*(2*H));
        ok&=rv(opk,(size_t)H*(NH*HD));ok&=rv(qn,HD);ok&=rv(kn,HD);ok&=rv(pan,H);
        ok&=rv(gpk,(size_t)IM*H);ok&=rv(upk,(size_t)IM*H);ok&=rv(dpk,(size_t)H*IM);
        ok&=rv(fn_,H);ok&=rv(lmk,(size_t)V*H);
        fclose(f);
        if(!ok){printf("DRAFT corrupt\n");goto no_draft;}
        // Upload weights to GPU (column-major for GEMV)
        // For GEMV: W is [out, in] row-major in file, need [out, in] column-major on GPU
        auto gpu_proj=[&](__half*&gpu,const std::vector<float>&cpu,int out_d,int in_d){
            size_t n=(size_t)out_d*in_d;hipMalloc(&gpu,n*2);
            std::vector<__half>buf(n);for(int o=0;o<out_d;o++)for(int i=0;i<in_d;i++)buf[(size_t)o*in_d+i]=__float2half(cpu[(size_t)o*in_d+i]);
            hipMemcpy(gpu,buf.data(),n*2,hipMemcpyHostToDevice);};
        gpu_up(dws.embed,et,(size_t)V*H);gpu_up(dws.hN,hn,H);gpu_up(dws.iLN,iln,H);
        gpu_up(dws.paN,pan,H);gpu_up(dws.fN,fn_,H);gpu_up(dws.qN,qn,HD);gpu_up(dws.kN,kn,HD);
        gpu_proj(dws.fc,fc,H,NTL*H);
        gpu_proj(dws.qP,qpk,NH*HD,2*H);gpu_proj(dws.kP,kpk,NKV*HD,2*H);gpu_proj(dws.vP,vpk,NKV*HD,2*H);
        gpu_proj(dws.oP,opk,H,NH*HD);
        gpu_proj(dws.gate,gpk,IM,H);gpu_proj(dws.up,upk,IM,H);gpu_proj(dws.dp,dpk,H,IM);
        gpu_proj(dws.lmH,lmk,V,H);
        dl=true;
        printf("Draft: loaded (%d weights)\n",(int)(et.size()+fc.size()+qpk.size()+kpk.size()+vpk.size()+opk.size()+gpk.size()+upk.size()+dpk.size()+lmk.size()));
    }
    no_draft:;

    // ── GPU draft model forward pass ──────────────────────────────
    // All operations chain on GPU stream gs. No CPU sync until the final readback.
    __half *dK_d=nullptr,*dV_d=nullptr; // draft KV cache (block_size positions)
    __half *d_hid=nullptr,*d_emb=nullptr,*d_con=nullptr,*d_scr=nullptr;
    __half *d_attn=nullptr,*d_lgt=nullptr; // attention buffer, logits (f32)
    hipMalloc(&dK_d,(size_t)7*NKV*HD*2);hipMemset(dK_d,0,(size_t)7*NKV*HD*2);
    hipMalloc(&dV_d,(size_t)7*NKV*HD*2);hipMemset(dV_d,0,(size_t)7*NKV*HD*2);
    hipMalloc(&d_hid,(size_t)H*2);hipMalloc(&d_emb,(size_t)H*2);
    hipMalloc(&d_con,(size_t)(2*H)*2);hipMalloc(&d_attn,(size_t)(NH*HD)*2);
    hipMalloc(&d_scr,(size_t)max(IM,2*H)*2);
    hipMalloc(&d_lgt,(size_t)V*4); // f32 logits

    // RoPE tables on GPU
    RopeTables rope;rope.init(HD,MAX_SEQ,1000000.0f);
    __half*d_cos=nullptr,*d_sin=nullptr;
    hipMalloc(&d_cos,(size_t)MAX_SEQ*HD*2);hipMalloc(&d_sin,(size_t)MAX_SEQ*HD*2);
    {std::vector<__half>b(MAX_SEQ*HD);for(int i=0;i<MAX_SEQ*HD;i++)b[i]=__float2half(rope.cos[i]);hipMemcpy(d_cos,b.data(),(size_t)MAX_SEQ*HD*2,hipMemcpyHostToDevice);}
    {std::vector<__half>b(MAX_SEQ*HD);for(int i=0;i<MAX_SEQ*HD;i++)b[i]=__float2half(rope.sin[i]);hipMemcpy(d_sin,b.data(),(size_t)MAX_SEQ*HD*2,hipMemcpyHostToDevice);}

    // ── GPU kernel abstractions ──
    auto gv=[&](__half*in,__half*W,__half*out,int K,int N){
        launch_gemv_simple(in,W,out,K,N,gs);};
    auto nr=[&](__half*x,const __half*w,int n){launch_rmsnorm(x,w,n,EPS,gs);};
    auto ad=[&](__half*x,const __half*y,int n){launch_add(x,y,n,gs);};
    auto cp=[&](const __half*s,__half*d,int n){launch_copy(s,d,n,gs);};
    auto sl=[&](const __half*g,const __half*u,__half*o,int n){launch_silu_mul(g,u,o,n,gs);};
    auto rp=[&](__half*qk,int pos,int nh){launch_rope(qk,d_cos,d_sin,pos,HD,nh,gs);};
    auto lm=[&](__half*h,__half*W,float*L,int H_,int V_){
        launch_lmhead(h,W,L,H_,V_,1024,256,gs);};

    // Readback f16 buffer → float (syncs stream)
    auto rb16=[&](__half*src,float*dst,int n){
        std::vector<__half>b(n);hipMemcpyAsync(b.data(),src,(size_t)n*2,hipMemcpyDeviceToHost,gs);
        hipStreamSynchronize(gs);for(int i=0;i<n;i++)dst[i]=__half2float(b[i]);};

    // Full draft forward pass: trunk_feat[5*H] + token_id → logits[V] (f32 on CPU)
    // No CPU sync until the final lmhead_k readback below.
    auto draft_forward=[&](const float*trunk_feat,int token_id,int pos,int)->int{
        // Upload trunk (f32→f16, H2D, synchronous for simplicity)
        std::vector<__half>tb(NTL*H);for(int i=0;i<NTL*H;i++)tb[i]=__float2half(trunk_feat[i]);
        hipMemcpyAsync(d_scr,tb.data(),(size_t)NTL*H*2,hipMemcpyHostToDevice,gs);

        // FC: [5*H]×[H,5*H]→[H]; save to d_hid, embed token
        // FC: [5*H]×[H,5*H]→[H], save fc_out to d_con for residual
        gv(d_scr,dws.fc,d_hid,NTL*H,H);
        hipMemcpyAsync(d_emb,&dws.embed[(size_t)token_id*H],(size_t)H*2,hipMemcpyDeviceToDevice,gs);
        cp(d_hid,d_con,H); // save fc_out in d_con

        // Norm both, concat [hid_norm | emb_norm] → d_con (now free)
        nr(d_hid,dws.hN,H); nr(d_emb,dws.iLN,H);
        cp(d_hid,d_scr,H); cp(d_emb,&d_scr[H],H); // concat in d_scr

        // QKV: concat→q/k/v, norm+rope on Q, K to cache
        gv(d_scr,dws.qP,d_attn,2*H,NH*HD); // Q in d_attn
        gv(d_scr,dws.kP,d_emb,2*H,NKV*HD); // K in d_emb
        gv(d_scr,dws.vP,d_hid,2*H,NKV*HD); // V in d_hid
        cp(d_emb,&dK_d[(size_t)pos*NKV*HD],NKV*HD); // K to cache
        cp(d_hid,&dV_d[(size_t)pos*NKV*HD],NKV*HD); // V to cache
        nr(d_attn,dws.qN,NH*HD); rp(d_attn,pos,NH);
        nr(&dK_d[(size_t)pos*NKV*HD],dws.kN,NKV*HD); rp(&dK_d[(size_t)pos*NKV*HD],pos,NKV);

        // Attention + O projection
        rcpp_kv_cache_attn_decode(d_attn,dK_d,dV_d,d_scr,NH,NKV,HD,pos+1,0.0883883476f,gs);
        gv(d_scr,dws.oP,d_hid,NH*HD,H);
        ad(d_hid,d_con,H); // residual: hid += fc_out (saved in d_con before norm)

        // FFN: norm→gate/up→silu→down→residual
        // d_scr is now free (attention output was consumed by O projection)
        cp(d_hid,d_scr,H); // save hid in d_scr for residual
        nr(d_hid,dws.paN,H);
        gv(d_hid,dws.gate,d_emb,H,IM); // gate in d_emb
        gv(d_hid,dws.up,d_con,H,IM);   // up in d_con
        sl(d_emb,d_con,d_emb,IM);       // silu_out in d_emb
        gv(d_emb,dws.dp,d_con,IM,H);   // down_out in d_con
        cp(d_scr,d_hid,H);              // d_hid = pre-norm value (was in d_scr)
        ad(d_hid,d_con,H);              // hid += down_out (FFN residual, d_scr has pre-norm)

        // Final norm + LM head → f32 logits
        nr(d_hid,dws.fN,H);
        lm(d_hid,dws.lmH,(float*)d_lgt,H,V);

        // Readback logits and argmax
        std::vector<float>lb(V);
        hipMemcpyAsync(lb.data(),d_lgt,(size_t)V*4,hipMemcpyDeviceToHost,gs);
        hipStreamSynchronize(gs);
        return (int)std::distance(lb.begin(),std::max_element(lb.begin(),lb.end()));
    };

    // LM head kernel: f16[H] × f16[V,H]col-major → f32[V]
    auto amax=[&](const float*l,int n){return(int)std::distance(l,std::max_element(l,l+n));};
    auto emb=[&](int t,float*o){memcpy(o,&ef32[(size_t)t*cfg.H],cfg.H*4);};

    // ── Batched NPU verify: QKV/O/GU/D run once per layer for ALL positions ──
    // h_batch[B, H] = input embeddings for each verify position
    // qkv_batch[B, qkv_n], attn_batch[B, NH*HD], hid_batch[B, H]
    // For each layer: QKV→split→norm/RoPE/cache/attn/pos→O→residual→norm→GU→SiLU→D→residual
    int maxB=7; // max block_size + 1
    std::vector<float>h_b((size_t)maxB*cfg.H),qv_b((size_t)maxB*cfg.qkv_n()),at_b((size_t)maxB*cfg.NH*cfg.HD);
    std::vector<float>kvk((size_t)cfg.NC*MAX_SEQ*cfg.NKV*cfg.HD,0);
    std::vector<float>kvv((size_t)cfg.NC*MAX_SEQ*cfg.NKV*cfg.HD,0);
    std::vector<float>oo_b((size_t)maxB*cfg.H),gt_b((size_t)maxB*2*cfg.IM),su_b((size_t)maxB*cfg.IM),dw_b((size_t)maxB*cfg.H);

    // Target layers for EAGLE3 trunk features
    const int trunk_layers[5] = {1, 6, 12, 18, 24};
    std::vector<float> trunk(5 * cfg.H);
    auto verify_batch=[&](int npos,int past_len,std::vector<float>&logits_out,float* trunk_out=nullptr){
        // logits_out[B, NV] filled for each position
        for(int l=0;l<cfg.NC;l++){
            // Phase 1: QKV GEMM (batched across all positions)
            float as_qk=dyn_scale(h_b.data(),npos*cfg.H);
            cq.go(l,h_b.data(),npos,cfg.H,as_qk,ws[l].qk,qv_b.data(),cfg.qkv_n());
            cn(qv_b.data(),npos*cfg.qkv_n());

            // Phase 2: Per-position norms/RoPE/cache/attention
            for(int pi=0;pi<npos;pi++){
                float*qo=&qv_b[(size_t)pi*cfg.qkv_n()];
                int pos=past_len+pi;
                // Q norms + RoPE
                for(int hh=0;hh<cfg.NH;hh++){double ss=0;for(int d=0;d<cfg.HD;d++)ss+=qo[hh*cfg.HD+d]*qo[hh*cfg.HD+d];float iq=1.0f/sqrtf((float)(ss/cfg.HD)+EPS);for(int d=0;d<cfg.HD;d++)qo[hh*cfg.HD+d]*=iq*q_norm[l][d];rope.apply(&qo[hh*cfg.HD],pos,1);}
                // K norms + RoPE + KV cache update
                for(int kvh=0;kvh<cfg.NKV;kvh++){
                    float*ks=&qo[cfg.NH*cfg.HD+kvh*cfg.HD];
                    float*vs=&qo[cfg.NH*cfg.HD+cfg.NKV*cfg.HD+kvh*cfg.HD];
                    double ss=0;for(int d=0;d<cfg.HD;d++)ss+=ks[d]*ks[d];
                    float ik=1.0f/sqrtf((float)(ss/cfg.HD)+EPS);
                    for(int d=0;d<cfg.HD;d++){ks[d]*=ik*k_norm[l][d];rope.apply(ks,pos,1);}
                    memcpy(&kvk[(size_t)l*MAX_SEQ*cfg.NKV*cfg.HD+(size_t)pos*cfg.NKV*cfg.HD+kvh*cfg.HD],ks,cfg.HD*4);
                    memcpy(&kvv[(size_t)l*MAX_SEQ*cfg.NKV*cfg.HD+(size_t)pos*cfg.NKV*cfg.HD+kvh*cfg.HD],vs,cfg.HD*4);
                }
                // CPU attention (attends to KV cache[0..pos])
                int cl=pos+1;
                float*attn_out=&at_b[(size_t)pi*cfg.NH*cfg.HD];
                #pragma omp parallel for
                for(int hh=0;hh<cfg.NH;hh++){int kvh=hh/cfg.GQA;float sc[4096];float mx=-1e30f;
                    for(int p=0;p<cl;p++){double ss=0;for(int d=0;d<cfg.HD;d++)ss+=(double)qo[hh*cfg.HD+d]*kvk[(size_t)l*MAX_SEQ*cfg.NKV*cfg.HD+(size_t)p*cfg.NKV*cfg.HD+kvh*cfg.HD+d];sc[p]=(float)(ss*0.0883883476f);if(sc[p]>mx)mx=sc[p];}
                    double sw=0;for(int p=0;p<cl;p++){sc[p]=expf(sc[p]-mx);sw+=sc[p];}float isw=sw>0?1.0f/(float)sw:1.0f/cl;
                    for(int d=0;d<cfg.HD;d++){float acc=0;for(int p=0;p<cl;p++)acc+=sc[p]*kvv[(size_t)l*MAX_SEQ*cfg.NKV*cfg.HD+(size_t)p*cfg.NKV*cfg.HD+kvh*cfg.HD+d];attn_out[hh*cfg.HD+d]=acc*isw;}}
            }

            // Phase 3: O GEMM (batched)
            float as_o=dyn_scale(at_b.data(),npos*cfg.NH*cfg.HD);
            co.go(l,at_b.data(),npos,cfg.NH*cfg.HD,as_o,ws[l].o_,oo_b.data(),cfg.H);
            cn(oo_b.data(),npos*cfg.H);
            for(int pi=0;pi<npos;pi++)for(int i=0;i<cfg.H;i++)h_b[(size_t)pi*cfg.H+i]+=oo_b[(size_t)pi*cfg.H+i];

            // Phase 4: RMS norm (batched) + GU GEMM (batched)
            for(int pi=0;pi<npos;pi++)rms_norm(&h_b[(size_t)pi*cfg.H],pa_norm[l].data(),cfg.H);
            float as_g=dyn_scale(h_b.data(),npos*cfg.H);
            cg.go(l,h_b.data(),npos,cfg.H,as_g,ws[l].g_,gt_b.data(),2*cfg.IM);
            cn(gt_b.data(),npos*2*cfg.IM);
            for(int pi=0;pi<npos;pi++){
                for(int i=0;i<cfg.IM;i++){float gv=gt_b[(size_t)pi*2*cfg.IM+i];if(!std::isfinite(gv))gv=0;su_b[(size_t)pi*cfg.IM+i]=(gv/(1.0f+expf(-gv)))*gt_b[(size_t)pi*2*cfg.IM+cfg.IM+i];}}

            // Phase 5: D GEMM (batched)
            float as_d=dyn_scale(su_b.data(),npos*cfg.IM);
            cd.go(l,su_b.data(),npos,cfg.IM,as_d,ws[l].d_,dw_b.data(),cfg.H);
            cn(dw_b.data(),npos*cfg.H);
            for(int pi=0;pi<npos;pi++)for(int i=0;i<cfg.H;i++)h_b[(size_t)pi*cfg.H+i]+=dw_b[(size_t)pi*cfg.H+i];

            if (trunk_out && npos >= 1) {
                for (int tl = 0; tl < 5; tl++) {
                    if (l == trunk_layers[tl]) {
                        memcpy(&trunk_out[(size_t)tl * cfg.H], &h_b[0], cfg.H * sizeof(float));
                        break;
                    }
                }
            }
        }

        // Final norm + LM head
        for(int pi=0;pi<npos;pi++){
            rms_norm(&h_b[(size_t)pi*cfg.H],fn.data(),cfg.H);
            for(int n=0;n<cfg.NV;n++){double ss=0;for(int i=0;i<cfg.H;i++)ss+=(double)h_b[(size_t)pi*cfg.H+i]*ef32[(size_t)n*cfg.H+i];logits_out[(size_t)pi*cfg.NV+n]=(float)ss;}
        }
    };

    // ── Prefill ──────────────────────────────────────────────────
    int pt[]={151643,872,198,11852,151644,198,151643,77091,198},npt=9,sp=0;
    printf("Prefill %d...\n",npt);auto t0=std::chrono::steady_clock::now();
    for(int pi=0;pi<npt;pi++){
        emb(pt[pi],h_b.data());
        std::vector<float>tmp((size_t)cfg.NV);
        verify_batch(1,sp,tmp);
        sp++;
    }
    printf("  %.0fms\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    // ══════════════════════════════════════════════════════════════
    // DECODE — Batched NPU Verify + GPU Draft
    // ══════════════════════════════════════════════════════════════
    printf("Decode %d tokens...\n",n_gen);
    auto tg=std::chrono::steady_clock::now();
    int ctx=sp,block=dl?4:1;
    std::vector<int>out;for(int i=0;i<sp;i++)out.push_back(pt[i]);

    int64_t na=0,np=0,nv=0;
    std::vector<float>vl((size_t)(1+block)*cfg.NV);
    std::vector<int32_t>dt(block);

    // First token (baseline, no speculation)
    {
        emb(out.back(),h_b.data());
        vl.resize((size_t)cfg.NV);
        verify_batch(1,ctx,vl,trunk.data());
        int tok=amax(vl.data(),cfg.NV);
        out.push_back(tok);ctx++;
    }

    while((int)out.size()<sp+n_gen){
        auto ts=std::chrono::steady_clock::now();

        if(!dl||block<1){
            // No draft: single-token decode
            emb(out.back(),h_b.data());
            vl.resize((size_t)cfg.NV);
            verify_batch(1,ctx,vl);
            int tok=amax(vl.data(),cfg.NV);
            out.push_back(tok);ctx++;
        }else{
            // ── Draft: GPU generates block candidate tokens ──
            for(int di=0;di<block;di++){
                float tf[5*H];for(int i=0;i<5*H;i++)tf[i]=h_b[i%H]; // trunk from last verify output
                int tok=draft_forward(tf,out.back()+di,di,block);
                dt[di]=tok;
            }
            np+=block;

            // ── Verify: batched NPU forward for all positions ──
            int past=ctx-1;
            int npos=1+block;
            // Embed all tokens into h_b[0..npos*H]
            emb(out.back(),&h_b[0]);
            for(int di=0;di<block;di++)emb(dt[di],&h_b[(size_t)(1+di)*cfg.H]);
            vl.resize((size_t)npos*cfg.NV);
            verify_batch(npos,past,vl,trunk.data());
            nv++;

            // ── Acceptance ──
            int nacc=0;bool rej=false;
            for(int di=0;di<block;di++){
                int tt=amax(vl.data()+(size_t)di*cfg.NV,cfg.NV);
                if(dt[di]==tt){out.push_back(dt[di]);nacc++;ctx++;}else{out.push_back(tt);ctx++;rej=true;break;}
            }
            if(!rej){
                int bn=amax(vl.data()+(size_t)block*cfg.NV,cfg.NV);
                out.push_back(bn);ctx++;
            }
            na+=nacc;
            // Copy last accepted hidden state for next draft
            int last_idx=rej?0:block;
            memcpy(h_b.data(),&h_b[(size_t)last_idx*cfg.H],cfg.H*4);
        }

        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();
        printf("  [%zu] %dms\n",out.size(),(int)ms);
        if(out.back()==151645)break;
    }
    double gt_=std::chrono::duration<double>(std::chrono::steady_clock::now()-tg).count();
    int tg_=(int)out.size()-sp;
    printf("\n=== %d tok %.1fs = %.0f tok/s ===\n",tg_,gt_,tg_/gt_);
    printf("Draft: %ld prop, %ld acc (%.0f%%), %ld ver\n",np,na,na*100.0/np,nv);
    if(nv)printf("Speedup: %.1f×\n",tg_/(double)nv);

    hipFree(dK);hipFree(dV);hipFree(dQ);hipFree(dAO);
    hipFree(dK_d);hipFree(dV_d);hipFree(d_hid);hipFree(d_emb);
    hipFree(d_con);hipFree(d_attn);hipFree(d_scr);hipFree(d_lgt);
    hipFree(d_cos);hipFree(d_sin);
    if(dl){hipFree(dws.embed);hipFree(dws.fc);hipFree(dws.qP);hipFree(dws.kP);hipFree(dws.vP);hipFree(dws.oP);hipFree(dws.gate);hipFree(dws.up);hipFree(dws.dp);hipFree(dws.lmH);hipFree(dws.hN);hipFree(dws.iLN);hipFree(dws.paN);hipFree(dws.fN);hipFree(dws.qN);hipFree(dws.kN);}
    hipStreamDestroy(gs);munmap(md,st.st_size);
    return 0;
}
