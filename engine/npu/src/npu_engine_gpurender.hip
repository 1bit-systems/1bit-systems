/** npu_engine_gpurender.hip — All-ops-on-GPU with f32 hidden state.
 *
 * Architecture:
 *   d_hid (f32) — hidden state in f32 (no overflow)
 *   d_tmp (f16) — temp buffer for QKV/norms/attention (f16 ops)
 *   All GPU ops chained on stream gs with ONE sync per layer for NPU.
 *
 * Per-layer pipeline:
 *   1. Save d_hid→d_res (f32 copy)
 *   2. d_hid→f32→f16→d_tmp, RMS norm, QKV, Q/K norms+RoPE, attn, O gemv
 *   3. O→f16→f32→d_o_f32, residual: d_hid = d_res + d_o_f32 (f32)
 *   4. d_hid→f32→f16→d_tmp, RMS norm, readback→NPU, upload D→f32, residual
 *   Sync once per layer (after NPU D readback upload)
 *
 * Target: 15-20 tok/s (vs 5 tok/s for overlap)
 *
 * Build:
 *   hipcc -O3 -mavx512f -mavx512bw -mavx512vl -mavx512dq \
 *       -D__HIP_PLATFORM_AMD__=1 -I src -I include -I/opt/rocm/include \
 *       src/npu_engine_gpurender.hip src/gpu_kernels_fused.hip \
 *       src/dequant_q4nx.cpp src/attn_gpu.hip \
 *       --offload-arch=gfx1151 \
 *       -lxrt_coreutil -lxrt_core -luuid -lpthread -laiebu -lm -ldl -fopenmp \
 *       -o build/npu_engine_gpurender
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
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

extern "C" float* dequant_i8_to_float(const uint8_t*,int,int*,int*);

static constexpr float EPS=1e-6f;
static constexpr int MAX_SEQ=4096;

static inline float bf16g(uint16_t v){if((v&0x7F80)==0x7F80)return 0.0f;uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f;}
static inline void cn(float*x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
static inline float silu(float x){return x/(1.0f+expf(-x));}
static inline float dyn_scale(const float*x,int n){float a=0;for(int i=0;i<n;i++){float f=fabsf(x[i]);if(std::isfinite(f)&&f>a)a=f;}return a<1e-12f?1.0f:a/127.0f;}

static void quantize_avx(const float*A,int8_t*D,int n,float s){float is=1.0f/s;__m512 v_is=_mm512_set1_ps(is),v_max=_mm512_set1_ps(127.0f),v_min=_mm512_set1_ps(-127.0f);int i=0;for(;i+16<=n;i+=16){__m512 v=_mm512_loadu_ps(&A[i]);v=_mm512_mul_ps(v,v_is);v=_mm512_max_ps(v_min,_mm512_min_ps(v_max,v));__m512i vi=_mm512_cvtps_epi32(v);__m256i lo=_mm512_cvtepi32_epi16(vi);__m128i i8=_mm256_cvtepi16_epi8(lo);_mm_storeu_si128((__m128i*)&D[i],i8);}for(;i<n;i++){int q=(int)roundf(A[i]*is);if(q>127)q=127;else if(q<-127)q=-127;D[i]=(int8_t)q;}}
static void dequant_avx(const int16_t*Cm,float*C,int n,float s){__m512 v_s=_mm512_set1_ps(s);int i=0;for(;i+16<=n;i+=16){__m256i vi16=_mm256_loadu_si256((__m256i*)&Cm[i]);__m512i vi32=_mm512_cvtepi16_epi32(vi16);__m512 v=_mm512_cvtepi32_ps(vi32);v=_mm512_mul_ps(v,v_s);_mm512_storeu_ps(&C[i],v);}for(;i<n;i++)C[i]=(float)Cm[i]*s;}

// GPU kernel declarations
extern "C"{
void launch_gemv_tiled(const __half*,const __half*,__half*,int,int,int,int,hipStream_t);
void launch_gemv_simple(const __half*,const __half*,__half*,int,int,hipStream_t);
void launch_rmsnorm(__half*,const __half*,int,float,hipStream_t);
void launch_add(__half*,const __half*,int,hipStream_t);
void launch_copy(const __half*,__half*,int,hipStream_t);
void launch_rope(__half*,const __half*,const __half*,int,int,int,hipStream_t);
void launch_rmsnorm_f32(float*,const float*,int,float,hipStream_t);
void launch_add_f32(float*,const float*,int,hipStream_t);
void launch_copy_f32(const float*,float*,int,hipStream_t);
void launch_f32_to_f16(const float*,__half*,int,hipStream_t);
void launch_f16_to_f32(const __half*,float*,int,hipStream_t);
int rcpp_kv_cache_attn_decode(const void*,const void*,const void*,void*,int,int,int,int,float,void*);
}

struct NpuGemmCtx{
    int MD,KD,ND;std::vector<uint32_t>ins;
    std::unique_ptr<xrt::xclbin>xc;std::unique_ptr<xrt::hw_context>hc;
    std::unique_ptr<xrt::module>mdl;std::unique_ptr<xrt::elf>elf;
    std::unique_ptr<xrt::ext::kernel>k;
    std::unique_ptr<xrt::bo>bA,bB,bC;int8_t*Am=nullptr;int16_t*Cm=nullptr;
    xrt::run run_;bool busy_=false;
    bool init(xrt::device&d,const char*xp,const char*ip,int md,int kd,int nd){
        MD=md;KD=kd;ND=nd;FILE*f=fopen(ip,"rb");if(!f)return false;fseek(f,0,SEEK_END);long sz=ftell(f);fseek(f,0,SEEK_SET);
        ins.resize(sz/4);if(fread(ins.data(),4,ins.size(),f)!=ins.size()){fclose(f);return false;}fclose(f);
        try{std::vector<char>iraw((char*)ins.data(),(char*)ins.data()+ins.size()*4);
            aiebu::aiebu_assembler asmblr(aiebu::aiebu_assembler::buffer_type::blob_instr_transaction,iraw);
            auto e=asmblr.get_elf();
            xc=std::make_unique<xrt::xclbin>(std::string(xp));d.register_xclbin(*xc);
            hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());
            elf=std::make_unique<xrt::elf>(e.data(),e.size());
        }catch(...){return false;}
        mdl=std::make_unique<xrt::module>(*elf);
        k=std::make_unique<xrt::ext::kernel>(*hc,*mdl,"MLIR_AIE");
        bA=std::make_unique<xrt::bo>(d,(size_t)MD*KD,XRT_BO_FLAGS_HOST_ONLY,0);
        bC=std::make_unique<xrt::bo>(d,(size_t)MD*ND*2,XRT_BO_FLAGS_HOST_ONLY,0);
        bB=std::make_unique<xrt::bo>(d,(size_t)KD*ND,XRT_BO_FLAGS_HOST_ONLY,0);
        Am=(int8_t*)bA->map();Cm=(int16_t*)bC->map();return true;}
    void packB(const float*w,int K,int N,float&sout){
        float amax=0;for(int i=0;i<K*N;i++){float a=fabsf(w[i]);if(std::isfinite(a)&&a>amax)amax=a;}
        sout=(amax<1e-12f)?1.0f:amax/127.0f;float is=127.0f/(amax<1e-12f?1.0f:amax);
        auto*Bm=(int8_t*)bB->map();for(int i=0;i<K*N;i++){float v=w[i];if(!std::isfinite(v))v=0;int q=(int)roundf(v*is);if(q>127)q=127;else if(q<-127)q=-127;Bm[i]=(int8_t)q;}
        bB->sync(XCL_BO_SYNC_BO_TO_DEVICE);}
    void launch(const float*A,int am,int ak,float as_){memset(Am,0,(size_t)am*KD);quantize_avx(A,Am,am*ak,as_);bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);run_=k->operator()(3,0,0,*bA,*bB,*bC);busy_=true;}
    void wait(float*C,int am,int an,float as_,float Bs){if(!busy_)return;run_.wait();bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);float cs=as_*Bs;dequant_avx(Cm,C,am*an,cs);busy_=false;}
};

struct GpuGemvCtx{
    int K,N;__half*d_W=nullptr;
    bool init(int k,int n,const float*W_f32){
        K=k;N=n;size_t sz=(size_t)N*K*2;
        if(hipSuccess!=hipMalloc(&d_W,sz))return false;
        std::vector<__half>buf((size_t)N*K);
        for(int ni=0;ni<N;ni++)for(int ki=0;ki<K;ki++)buf[(size_t)ni*K+ki]=__float2half(W_f32[(size_t)ki*N+ni]);
        hipMemcpy(d_W,buf.data(),sz,hipMemcpyHostToDevice);return true;}
    void gemv_gpu(__half*d_in,__half*d_out,hipStream_t s){launch_gemv_tiled(d_in,d_W,d_out,K,N,1024,256,s);}
    void destroy(){if(d_W)hipFree(d_W);}
};

static uint64_t json_off(const char*js,size_t jl,const char*nm){
    size_t nl=strlen(nm);const char*p=js,*e=js+jl;
    while(p<e){auto q=(const char*)memmem(p,e-p,nm,nl);if(!q)return 0;
    if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){auto o=strstr(q,"\"data_offsets\"");
    if(o){auto a=strchr(o,'[');if(a)return strtoull(a+1,NULL,10);}}p=q+1;}return 0;}
struct MdlCfg{int H=0,NC=0,NH=0,NKV=0,HD=0,IM=0,NV=0,GQA=0;int qkv_n()const{return NH*HD+2*NKV*HD;}};
struct LOff{uint64_t q,k,v,o,g,u,d,in,pa,qn,kn;};

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"Usage: %s <model.q4nx> [tokens]\n",argv[0]);return 1;}
    const char*mp=argv[1];int n_gen=argc>2?atoi(argv[2]):64;

    // Load model
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
    printf("=== GPU Render Engine === H=%d NC=%d NV=%d\n",cfg.H,cfg.NC,cfg.NV);

    // Init HW
    hipSetDevice(0);hipStream_t gs;hipStreamCreate(&gs);
    xrt::device nd(0);
    const char*xd=getenv("NPU_XCLBIN_DIR")?:"/home/bcloud/npu-sandbox/npu-infer/build/int8";
    auto xp=[&](const char*t){static char b[256];snprintf(b,256,"%s/final_i8_%s_v.xclbin",xd,t);return b;};
    auto ip=[&](const char*t){static char b[256];snprintf(b,256,"%s/insts_i8_%s_v.txt",xd,t);return b;};
    int XM=128;

    // NPU: GU + D
    NpuGemmCtx ctx_gu,ctx_d;
    if(!ctx_gu.init(nd,xp("GU"),ip("GU"),XM,cfg.H,2*cfg.IM)){fprintf(stderr,"FAIL GU\n");return 1;}
    if(!ctx_d.init(nd,xp("D"),ip("D"),XM,cfg.IM,cfg.H)){fprintf(stderr,"FAIL D\n");return 1;}
    struct WS{float g_,d_;};std::vector<WS>ws(cfg.NC);
    for(int l=0;l<cfg.NC;l++){int gr,ur,dr,u;
        float*gw=dequant_i8_to_float(i8p(lo[l].g),384,&gr,&u),*uw=dequant_i8_to_float(i8p(lo[l].u),384,&ur,&u);
        int t2=gr+ur;std::vector<float>w2((size_t)cfg.H*t2);for(int kk=0;kk<cfg.H;kk++){memcpy(&w2[(size_t)kk*t2],&gw[(size_t)kk*gr],(size_t)gr*4);memcpy(&w2[(size_t)kk*t2+gr],&uw[(size_t)kk*ur],(size_t)ur*4);}
        ctx_gu.packB(w2.data(),cfg.H,t2,ws[l].g_);free(gw);free(uw);
        float*dw=dequant_i8_to_float(i8p(lo[l].d),384,&dr,&u);ctx_d.packB(dw,dr,cfg.H,ws[l].d_);free(dw);}

    // GPU: QKV + O
    std::vector<GpuGemvCtx> ctx_qkv(cfg.NC),ctx_o(cfg.NC);
    for(int l=0;l<cfg.NC;l++){int qr,kr,vr,or_,u;
        float*qw=dequant_i8_to_float(i8p(lo[l].q),256,&qr,&u),*kw=dequant_i8_to_float(i8p(lo[l].k),128,&kr,&u),*vw=dequant_i8_to_float(i8p(lo[l].v),128,&vr,&u);
        int t=qr+kr+vr;std::vector<float>w((size_t)cfg.H*t);for(int kk=0;kk<cfg.H;kk++){memcpy(&w[(size_t)kk*t],&qw[(size_t)kk*qr],(size_t)qr*4);memcpy(&w[(size_t)kk*t+qr],&kw[(size_t)kk*kr],(size_t)kr*4);memcpy(&w[(size_t)kk*t+qr+kr],&vw[(size_t)kk*vr],(size_t)vr*4);}
        ctx_qkv[l].init(cfg.H,t,w.data());free(qw);free(kw);free(vw);
        float*ow=dequant_i8_to_float(i8p(lo[l].o),256,&or_,&u);ctx_o[l].init(or_,cfg.H,ow);free(ow);}

    // GPU buffers: f32 hidden state + f16 intermediates
    float *d_hid=nullptr,*d_res=nullptr,*d_o_f32=nullptr;
    __half *d_tmp=nullptr,*d_qkv=nullptr,*d_attn=nullptr,*d_o_f16=nullptr;
    __half *d_K=nullptr,*d_V=nullptr,*d_cos=nullptr,*d_sin=nullptr;
    __half *d_in_w=nullptr,*d_pa_w=nullptr,*d_qn=nullptr,*d_kn=nullptr;
    hipMalloc(&d_hid,(size_t)cfg.H*4);hipMalloc(&d_res,(size_t)cfg.H*4);
    hipMalloc(&d_o_f32,(size_t)cfg.H*4);
    hipMalloc(&d_tmp,(size_t)cfg.H*2);hipMalloc(&d_qkv,(size_t)cfg.qkv_n()*2);
    hipMalloc(&d_attn,(size_t)cfg.NH*cfg.HD*2);hipMalloc(&d_o_f16,(size_t)cfg.H*2);
    hipMalloc(&d_K,(size_t)MAX_SEQ*cfg.NKV*cfg.HD*2);hipMemset(d_K,0,(size_t)MAX_SEQ*cfg.NKV*cfg.HD*2);
    hipMalloc(&d_V,(size_t)MAX_SEQ*cfg.NKV*cfg.HD*2);hipMemset(d_V,0,(size_t)MAX_SEQ*cfg.NKV*cfg.HD*2);
    hipMalloc(&d_in_w,(size_t)cfg.H*2);hipMalloc(&d_pa_w,(size_t)cfg.H*2);
    hipMalloc(&d_qn,(size_t)cfg.HD*2);hipMalloc(&d_kn,(size_t)cfg.HD*2);
    hipMalloc(&d_cos,(size_t)MAX_SEQ*cfg.HD*2);hipMalloc(&d_sin,(size_t)MAX_SEQ*cfg.HD*2);
    // Upload RoPE tables
    {std::vector<__half>b(MAX_SEQ*cfg.HD);
        for(int i=0;i<MAX_SEQ*cfg.HD;i++){float f=1.0f/powf(1000000.0f,(float)(i%cfg.HD)/(cfg.HD/2));float a=(float)(i/cfg.HD)*f;b[i]=__float2half(cosf(a));}
        hipMemcpy(d_cos,b.data(),(size_t)MAX_SEQ*cfg.HD*2,hipMemcpyHostToDevice);}
    {std::vector<__half>b(MAX_SEQ*cfg.HD);
        for(int i=0;i<MAX_SEQ*cfg.HD;i++){float f=1.0f/powf(1000000.0f,(float)(i%cfg.HD)/(cfg.HD/2));float a=(float)(i/cfg.HD)*f;b[i]=__float2half(sinf(a));}
        hipMemcpy(d_sin,b.data(),(size_t)MAX_SEQ*cfg.HD*2,hipMemcpyHostToDevice);}

    float attn_scale=1.0f/sqrtf((float)cfg.HD);
    std::vector<float>hidden(cfg.H),gt(2*cfg.IM),su(cfg.IM),d_out(cfg.H),lg(cfg.NV);
    auto emb=[&](int t,float*o){memcpy(o,&ef32[(size_t)t*cfg.H],cfg.H*4);};
    auto amax=[&](const float*l,int n){return(int)std::distance(l,std::max_element(l,l+n));};

    // Prefill (simplified: uses CPU attention)
    int pt[]={151643,872,198,11852,151644,198,151643,77091,198},npt=9,sp=0;
    printf("Prefill %d...\n",npt);auto t0=std::chrono::steady_clock::now();
    for(int pi=0;pi<npt;pi++){
        emb(pt[pi],hidden.data());
        // Upload to f32 d_hid
        hipMemcpyAsync(d_hid,hidden.data(),cfg.H*4,hipMemcpyHostToDevice,gs);hipStreamSynchronize(gs);
        for(int l=0;l<cfg.NC;l++){
            // Upload norm weights (f16, sync copy — at init time only)
            {__half buf[1024];for(int i=0;i<cfg.H;i++)buf[i]=__float2half(in_norm[l][i]);hipMemcpyAsync(d_in_w,buf,cfg.H*2,hipMemcpyHostToDevice,gs);hipStreamSynchronize(gs);}
            {__half buf[1024];for(int i=0;i<cfg.H;i++)buf[i]=__float2half(pa_norm[l][i]);hipMemcpyAsync(d_pa_w,buf,cfg.H*2,hipMemcpyHostToDevice,gs);hipStreamSynchronize(gs);}
            {__half buf[128];for(int i=0;i<cfg.HD;i++)buf[i]=__float2half(q_norm[l][i]);hipMemcpyAsync(d_qn,buf,cfg.HD*2,hipMemcpyHostToDevice,gs);hipStreamSynchronize(gs);}
            {__half buf[128];for(int i=0;i<cfg.HD;i++)buf[i]=__float2half(k_norm[l][i]);hipMemcpyAsync(d_kn,buf,cfg.HD*2,hipMemcpyHostToDevice,gs);hipStreamSynchronize(gs);}

            // GPU pipeline: f32 hidden → f16 temp → QKV → norms → attn → O → f32 residual
            launch_copy_f32(d_hid,d_res,cfg.H,gs); // save for residual
            launch_f32_to_f16(d_hid,d_tmp,cfg.H,gs);
            launch_rmsnorm(d_tmp,d_in_w,cfg.H,EPS,gs);
            ctx_qkv[l].gemv_gpu(d_tmp,d_qkv,gs);
            launch_rmsnorm(d_qkv,d_qn,cfg.NH*cfg.HD,EPS,gs);
            launch_rope(d_qkv,d_cos,d_sin,sp+pi,cfg.HD,cfg.NH,gs);
            launch_rmsnorm(&d_qkv[cfg.NH*cfg.HD],d_kn,cfg.NKV*cfg.HD,EPS,gs);
            launch_rope(&d_qkv[cfg.NH*cfg.HD],d_cos,d_sin,sp+pi,cfg.HD,cfg.NKV,gs);
            launch_copy(&d_qkv[cfg.NH*cfg.HD],&d_K[(size_t)(sp+pi)*cfg.NKV*cfg.HD],cfg.NKV*cfg.HD,gs);
            launch_copy(&d_qkv[cfg.NH*cfg.HD+cfg.NKV*cfg.HD],&d_V[(size_t)(sp+pi)*cfg.NKV*cfg.HD],cfg.NKV*cfg.HD,gs);
            rcpp_kv_cache_attn_decode(d_qkv,d_K,d_V,d_attn,cfg.NH,cfg.NKV,cfg.HD,sp+pi+1,attn_scale,gs);
            ctx_o[l].gemv_gpu(d_attn,d_o_f16,gs);
            launch_f16_to_f32(d_o_f16,d_o_f32,cfg.H,gs);
            launch_copy_f32(d_res,d_hid,cfg.H,gs);
            launch_add_f32(d_hid,d_o_f32,cfg.H,gs);
            // Post-attention norm on the current (post-residual) hidden state
            // before feeding the NPU FFN -- d_tmp otherwise still holds the
            // stale pre-attention d_in_w-normed value from above. Convert
            // back to f32 (reusing the idle d_o_f32 scratch buffer) before
            // readback -- a raw f16-byte memcpy into a float host buffer
            // would just reinterpret half the bit pattern as garbage.
            launch_f32_to_f16(d_hid,d_tmp,cfg.H,gs);
            launch_rmsnorm(d_tmp,d_pa_w,cfg.H,EPS,gs);
            launch_f16_to_f32(d_tmp,d_o_f32,cfg.H,gs);
            hipMemcpyAsync(hidden.data(),d_o_f32,cfg.H*4,hipMemcpyDeviceToHost,gs);
            hipStreamSynchronize(gs);

            float as_g=dyn_scale(hidden.data(),cfg.H);
            ctx_gu.launch(hidden.data(),1,cfg.H,as_g);
            ctx_gu.wait(gt.data(),1,2*cfg.IM,as_g,ws[l].g_);cn(gt.data(),2*cfg.IM);
            for(int i=0;i<cfg.IM;i++){float gv=gt[i];if(!std::isfinite(gv))gv=0;su[i]=silu(gv)*gt[cfg.IM+i];}
            float as_d=dyn_scale(su.data(),cfg.IM);
            ctx_d.launch(su.data(),1,cfg.IM,as_d);
            ctx_d.wait(d_out.data(),1,cfg.H,as_d,ws[l].d_);cn(d_out.data(),cfg.H);

            __half dbuf[1024];for(int i=0;i<cfg.H;i++)dbuf[i]=__float2half(d_out[i]);
            hipMemcpyAsync(d_o_f16,dbuf,cfg.H*2,hipMemcpyHostToDevice,gs);hipStreamSynchronize(gs);
            launch_f16_to_f32(d_o_f16,d_o_f32,cfg.H,gs);
            launch_add_f32(d_hid,d_o_f32,cfg.H,gs);
            hipStreamSynchronize(gs);
        }
    }
    printf("  %.0fms\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    // Decode — all GPU ops on one stream
    printf("Decode %d...\n",n_gen);
    auto tg=std::chrono::steady_clock::now();
    int ctx=sp,cur=pt[npt-1];

    // Persistent host buffers for norm weights
    __half h_in_w[1024],h_pa_w[1024],h_qn[128],h_kn[128];

    for(int step=0;step<n_gen;step++){
        auto ts=std::chrono::steady_clock::now();

        // Embed → upload to f32 d_hid
        emb(cur,hidden.data());
        hipMemcpyAsync(d_hid,hidden.data(),cfg.H*4,hipMemcpyHostToDevice,gs);hipStreamSynchronize(gs);

        for(int l=0;l<cfg.NC;l++){
            // Upload norm weights
            for(int i=0;i<cfg.H;i++)h_in_w[i]=__float2half(in_norm[l][i]);
            for(int i=0;i<cfg.H;i++)h_pa_w[i]=__float2half(pa_norm[l][i]);
            for(int i=0;i<cfg.HD;i++)h_qn[i]=__float2half(q_norm[l][i]);
            for(int i=0;i<cfg.HD;i++)h_kn[i]=__float2half(k_norm[l][i]);
            hipMemcpyAsync(d_in_w,h_in_w,cfg.H*2,hipMemcpyHostToDevice,gs);
            hipMemcpyAsync(d_pa_w,h_pa_w,cfg.H*2,hipMemcpyHostToDevice,gs);
            hipMemcpyAsync(d_qn,h_qn,cfg.HD*2,hipMemcpyHostToDevice,gs);
            hipMemcpyAsync(d_kn,h_kn,cfg.HD*2,hipMemcpyHostToDevice,gs);

            // ─── ALL GPU OPS ON STREAM gs ───
            // Save f32 hidden for residual
            launch_copy_f32(d_hid,d_res,cfg.H,gs);
            // Convert f32→f16 for QKV pipeline
            launch_f32_to_f16(d_hid,d_tmp,cfg.H,gs);
            // RMS norm → QKV gemv → Q norms+RoPE → K norms+RoPE → KV cache
            launch_rmsnorm(d_tmp,d_in_w,cfg.H,EPS,gs);
            ctx_qkv[l].gemv_gpu(d_tmp,d_qkv,gs);
            launch_rmsnorm(d_qkv,d_qn,cfg.NH*cfg.HD,EPS,gs);
            launch_rope(d_qkv,d_cos,d_sin,ctx,cfg.HD,cfg.NH,gs);
            launch_rmsnorm(&d_qkv[cfg.NH*cfg.HD],d_kn,cfg.NKV*cfg.HD,EPS,gs);
            launch_rope(&d_qkv[cfg.NH*cfg.HD],d_cos,d_sin,ctx,cfg.HD,cfg.NKV,gs);
            launch_copy(&d_qkv[cfg.NH*cfg.HD],&d_K[(size_t)ctx*cfg.NKV*cfg.HD],cfg.NKV*cfg.HD,gs);
            launch_copy(&d_qkv[cfg.NH*cfg.HD+cfg.NKV*cfg.HD],&d_V[(size_t)ctx*cfg.NKV*cfg.HD],cfg.NKV*cfg.HD,gs);
            // Attention → O projection (all f16)
            rcpp_kv_cache_attn_decode(d_qkv,d_K,d_V,d_attn,cfg.NH,cfg.NKV,cfg.HD,ctx+1,attn_scale,gs);
            ctx_o[l].gemv_gpu(d_attn,d_o_f16,gs);
            // Convert O to f32, residual: d_hid = d_res + O
            launch_f16_to_f32(d_o_f16,d_o_f32,cfg.H,gs);
            launch_copy_f32(d_res,d_hid,cfg.H,gs);
            launch_add_f32(d_hid,d_o_f32,cfg.H,gs);
            // Post-attention norm on the current (post-residual) hidden state,
            // converted back to f32 (reusing the idle d_o_f32 scratch buffer)
            // before readback -- see matching comment in the prefill loop above.
            launch_f32_to_f16(d_hid,d_tmp,cfg.H,gs);
            launch_rmsnorm(d_tmp,d_pa_w,cfg.H,EPS,gs);
            launch_f16_to_f32(d_tmp,d_o_f32,cfg.H,gs);
            // Readback: ONE sync per layer
            hipMemcpyAsync(hidden.data(),d_o_f32,cfg.H*4,hipMemcpyDeviceToHost,gs);
            hipStreamSynchronize(gs);
            cn(hidden.data(),cfg.H);

            // NPU GU → D
            float as_g=dyn_scale(hidden.data(),cfg.H);
            ctx_gu.launch(hidden.data(),1,cfg.H,as_g);
            ctx_gu.wait(gt.data(),1,2*cfg.IM,as_g,ws[l].g_);cn(gt.data(),2*cfg.IM);
            for(int i=0;i<cfg.IM;i++){float gv=gt[i];if(!std::isfinite(gv))gv=0;su[i]=silu(gv)*gt[cfg.IM+i];}
            float as_d=dyn_scale(su.data(),cfg.IM);
            ctx_d.launch(su.data(),1,cfg.IM,as_d);
            ctx_d.wait(d_out.data(),1,cfg.H,as_d,ws[l].d_);cn(d_out.data(),cfg.H);

            // Upload D output → f32 residual add
            __half dbuf[1024];for(int i=0;i<cfg.H;i++)dbuf[i]=__float2half(d_out[i]);
            hipMemcpyAsync(d_o_f16,dbuf,cfg.H*2,hipMemcpyHostToDevice,gs);hipStreamSynchronize(gs);
            launch_f16_to_f32(d_o_f16,d_o_f32,cfg.H,gs);
            launch_add_f32(d_hid,d_o_f32,cfg.H,gs);
            hipStreamSynchronize(gs);
        }

        // Final norm + LM head
        hipMemcpyAsync(hidden.data(),d_hid,cfg.H*4,hipMemcpyDeviceToHost,gs);hipStreamSynchronize(gs);
                cn(hidden.data(),cfg.H);
        double mx=-1e30;for(int n=0;n<cfg.NV;n++){double ss=0;for(int i=0;i<cfg.H;i++)ss+=(double)hidden[i]*ef32[(size_t)n*cfg.H+i];lg[n]=(float)ss;if(lg[n]>mx)mx=lg[n];}
        cur=amax(lg.data(),cfg.NV);
        ctx++;

        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();
        printf("  [%d] tok=%d %.1fms\n",step,cur,ms);
    }
    double gen_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-tg).count();
    printf("\n=== %d tok %.1fs = %.0f tok/s ===\n",n_gen,gen_s,n_gen/gen_s);

    hipFree(d_hid);hipFree(d_res);hipFree(d_o_f32);
    hipFree(d_tmp);hipFree(d_qkv);hipFree(d_attn);hipFree(d_o_f16);
    hipFree(d_K);hipFree(d_V);hipFree(d_cos);hipFree(d_sin);
    hipFree(d_in_w);hipFree(d_pa_w);hipFree(d_qn);hipFree(d_kn);
    for(int l=0;l<cfg.NC;l++){ctx_qkv[l].destroy();ctx_o[l].destroy();}
    hipStreamDestroy(gs);munmap(md,st.st_size);
    return 0;
}
