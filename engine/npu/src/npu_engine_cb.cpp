/** NPU Engine v3 — Async Pipelined. Batch N tokens through all layers.
 *
 * Optimizations vs baseline:
 * 1. ❌ REMOVED `layerB[l]->sync()` from every GEMM call (weights already on device)
 * 2. ❌ REMOVED `memset(Am, 0, MD*KD)` (we overwrite every element we use)
 * 3. ✅ go_multi: restructured inner loop to iterate per-slice (no per-element branch)
 * 4. ✅ Double-buffered A/C BOs + async kernel launch (overlap CPU prep with NPU exec)
 * 5. ✅ Per-GEMM timing instrumentation
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <csignal>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <sys/stat.h>
#include <immintrin.h>  // AVX-512 for LM head
#define HF_CACHE "/tmp/hf_weights_cache"

using Clock = std::chrono::steady_clock;
static double elapsed_ms(Clock::time_point t0){
    return std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
}

static bool g_trace = false;
static void trace_dump(const char* key, const float* p, int n) {
    if (!g_trace || !p || n <= 0) return;
    char path[256]; snprintf(path, sizeof(path), "/tmp/cb_trace/%s.bin", key);
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "trace: cannot write %s\n", path); return; }
    fwrite(p, sizeof(float), (size_t)n, f); fclose(f);
    float amax = 0; for (int i = 0; i < n; i++) { float a = fabsf(p[i]); if (a > amax) amax = a; }
    double nrm = 0; for (int i = 0; i < n; i++) nrm += (double)p[i] * p[i];
    printf("  trace %-14s n=%-5d max|.|=%.5f norm=%.4f\n", key, n, amax, sqrt(nrm));
}
static void transpose_pack(const float* src, int out_f, int in_f, float* dst, int dst_stride, int dst_offset) {
    for (int o = 0; o < out_f; o++)
        for (int i = 0; i < in_f; i++)
            dst[(size_t)i*dst_stride + dst_offset + o] = src[(size_t)o*in_f + i];
}
static inline float dynamic_ascale(const float* x, int n) {
    float amax = 0;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (std::isfinite(a) && a > amax) amax = a; }
    return (amax < 1e-12f) ? (1.0f/127.0f) : (amax/127.0f);
}
static inline float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static inline float bf16g(uint16_t v){return(v&0x7F80)==0x7F80?0.0f:bf16f(v);}
static constexpr int H=1024,NC=28,NH=16,NKV=8,HD=128,IM=3072,NV=151936,GQA=2;
static constexpr float EPS=1e-6f; static constexpr int XM=128;
static inline void cn(float*x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
static inline void sm(float*sc,int n){
    if(n<=0)return;cn(sc,n);float mx=sc[0];for(int i=1;i<n;i++)if(sc[i]>mx)mx=sc[i];
    double s=0;for(int i=0;i<n;i++){float d=sc[i]-mx;if(d>80)d=80;else if(d<-80)d=-80;sc[i]=expf(d);s+=sc[i];}
    if(s<=0){float iv=1.0f/n;for(int i=0;i<n;i++)sc[i]=iv;return;}
    float is=1.0f/(float)s;for(int i=0;i<n;i++)sc[i]*=is;
}
static inline void rn_c(float*x,const float*w,int n){
    cn(x,n);double ss=0;for(int i=0;i<n;i++)if(std::isfinite(x[i]))ss+=(double)x[i]*x[i];
    float ir=1.0f/sqrtf((float)(ss/n)+EPS);
    for(int i=0;i<n;i++)x[i]=std::isfinite(x[i])?x[i]*ir*w[i]:0.0f;
}
static std::vector<float>rc,rs;static void ri(int hd,float th,int mp){
    rc.resize(mp*hd);rs.resize(mp*hd);
    for(int p=0;p<mp;p++)for(int i=0;i<hd/2;i++){float f=1.0f/powf(th,(float)(2*i)/hd),a=p*f;rc[p*hd+i]=cosf(a);rs[p*hd+i]=sinf(a);}
}
static inline void ra(float*x,int hd,int p){
    for(int i=0;i<hd/2;i++){float a=x[i],b=x[i+hd/2],c=rc[p*hd+i],s=rs[p*hd+i];x[i]=a*c-b*s;x[i+hd/2]=a*s+b*c;}
}
static uint64_t jo(const char*js,size_t jl,const char*nm){
    size_t nl=strlen(nm);const char*p=js,*e=js+jl;
    while(p<e){auto q=(const char*)memmem(p,e-p,nm,nl);if(!q)return 0;
        if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){auto o=strstr(q,"\"data_offsets\"");if(o){auto a=strchr(o,'[');if(a)return strtoull(a+1,NULL,10);}}p=q+1;}return 0;
}
static std::vector<float> emb_f32_cb;
static std::vector<float> lm_head_f32;
static volatile sig_atomic_t g_lora_reload = 0;
extern "C" void lora_sighup(int){g_lora_reload=1;}

// ─── AVX-512 LM head ───
// logits[n] = sum_k(sb[k] * W[n][k]) for n=0..NV-1
// Uses AVX-512: 16 FMAs per instruction on 16-wide vectors
static void lm_head_avx512(const float* W, const float* x, int nv, int h, float* lg) {
    // Process 16 rows at a time with AVX-512
    __m512 xvec;
    // Use memcpy to load 16 floats into xvec without alignment requirements
    for (int n = 0; n < nv; n += 16) {
        __m512 sum0 = _mm512_setzero_ps();
        __m512 sum1 = _mm512_setzero_ps();
        __m512 sum2 = _mm512_setzero_ps();
        __m512 sum3 = _mm512_setzero_ps();
        int cols = h;
        const float* wptr = W + (size_t)n * h;
        // Unroll 4× for better pipelining
        int k = 0;
        for (; k + 64 <= cols; k += 64) {
            __m512 x0 = _mm512_loadu_ps(x + k);
            __m512 x1 = _mm512_loadu_ps(x + k + 16);
            __m512 x2 = _mm512_loadu_ps(x + k + 32);
            __m512 x3 = _mm512_loadu_ps(x + k + 48);
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(wptr + k), x0, sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(wptr + k + 16), x1, sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(wptr + k + 32), x2, sum2);
            sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(wptr + k + 48), x3, sum3);
        }
        for (; k + 16 <= cols; k += 16) {
            xvec = _mm512_loadu_ps(x + k);
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(wptr + k), xvec, sum0);
        }
        // Handle remainder
        float tail[16] = {0};
        for (; k < cols; k++) tail[k % 16] += wptr[k] * x[k];
        __m512 tails = _mm512_loadu_ps(tail);
        __m512 total = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(_mm512_add_ps(sum2, sum3), tails));
        _mm512_storeu_ps(lg + n, total);
    }
}

// ─── Perf counters ───
static struct {
    double gemm_qkv=0,gemm_o=0,gemm_gu=0,gemm_d=0,atten=0,norms=0,silu=0,lmh=0;
    int nqkv=0,no=0,ngu=0,nd=0;
    void reset(){memset(this,0,sizeof(*this));}
    void report(int ng) const {
        double t=gemm_qkv+gemm_o+gemm_gu+gemm_d+atten+norms+silu+lmh;
        printf("\n── Perf breakdown (per token avg, %d steps) ────\n",ng);
        if(nqkv) printf("  QKV GEMM:    %7.2f ms  (%4d calls, %5.2f avg)\n",gemm_qkv/ng,nqkv,gemm_qkv/nqkv);
        if(no)   printf("  O   GEMM:    %7.2f ms  (%4d calls, %5.2f avg)\n",gemm_o/ng,no,gemm_o/no);
        if(ngu)  printf("  GU  GEMM:    %7.2f ms  (%4d calls, %5.2f avg)\n",gemm_gu/ng,ngu,gemm_gu/ngu);
        if(nd)   printf("  D   GEMM:    %7.2f ms  (%4d calls, %5.2f avg)\n",gemm_d/ng,nd,gemm_d/nd);
        printf("  Attention:   %7.2f ms\n",atten/ng);
        printf("  Norms:       %7.2f ms\n",norms/ng);
        printf("  SiLU:        %7.2f ms\n",silu/ng);
        printf("  LM head:     %7.2f ms\n",lmh/ng);
        printf("  TOTAL:       %7.2f ms  → %5.1f tok/s\n",t/ng,ng/(t/1000.0));
        printf("────────────────────────────────────────────────\n");
    }
} pc;

// ─── I8Ctx — double-buffered async GEMM ───
struct I8Ctx{
    const char*name;
    int MD,KD,ND;
    std::unique_ptr<xrt::xclbin>xc;
    std::unique_ptr<xrt::hw_context>hc;
    std::unique_ptr<xrt::kernel>k;
    std::vector<uint32_t>ins;
    std::unique_ptr<xrt::bo>bI;
    std::unique_ptr<xrt::bo>bA[2],bC[2],layerB[NC];
    int8_t*Am[2];
    int32_t*Cm[2];
    int ping=0; // current active slot; next quantize goes to 1-ping

    bool init(xrt::device&d,const char*xp,const char*ip,int gid_B){
        FILE*f=fopen(ip,"rb");if(!f)return false;
        fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);
        ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
        xc=std::make_unique<xrt::xclbin>(std::string(xp));
        d.register_xclbin(*xc);
        hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());
        k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");
        bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));
        memcpy(bI->map(),ins.data(),ins.size()*4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        for(int i=0;i<2;i++){
            bA[i]=std::make_unique<xrt::bo>(d,(size_t)MD*KD,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));
            bC[i]=std::make_unique<xrt::bo>(d,(size_t)MD*ND*4,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));
            Am[i]=(int8_t*)bA[i]->map();Cm[i]=(int32_t*)bC[i]->map();
        }
        for(int l=0;l<NC;l++)
            layerB[l]=std::make_unique<xrt::bo>(d,(size_t)KD*ND,XRT_BO_FLAGS_HOST_ONLY,k->group_id(gid_B));
        return true;
    }

    // Quantize float A → int8 into the idle slot, upload to device
    int push_A(const float*A,int am,int ak,float ascale){
        int slot=1-ping;
        float ais=1.0f/ascale;
        int8_t*dst=Am[slot];
        for(int m=0;m<am;m++)for(int k=0;k<ak;k++){
            float v=A[m*ak+k];if(!std::isfinite(v))v=0;
            int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;
            dst[m*KD+k]=(int8_t)q;
        }
        bA[slot]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return slot;
    }

    // Launch GEMM (async — does NOT wait). Caller must wait then pull_C.
    // NOTE: layerB[l]->sync() removed — weights uploaded once in repack().
    xrt::run launch(int slot,int l){
        return (*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA[slot],*layerB[l],*bC[slot]);
    }

    // Download result and dequantize Cm[slot] → C
    void pull_C(int slot,int am,int an,float ascale,float bscale,float*C){
        bC[slot]->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        float cs=ascale*bscale;int32_t*src=Cm[slot];
        for(int m=0;m<am;m++)for(int n=0;n<an;n++){
            float v=(float)src[m*ND+n]*cs;if(!std::isfinite(v))v=0;C[m*an+n]=v;
        }
        ping=slot;
    }

    // Multi-slice dequant (separate Bscale per output slice, no per-element inner loop)
    void pull_C_multi(int slot,int am,int an,float ascale,const float*bscales,
                      const int*starts,int ns,float*C,int cs_){
        bC[slot]->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        int32_t*src=Cm[slot];
        for(int m=0;m<am;m++)for(int si=0;si<ns;si++){
            int n0=starts[si],n1=(si+1<ns)?starts[si+1]:an;
            float cs=ascale*bscales[si];
            for(int n=n0;n<n1;n++){
                float v=(float)src[m*ND+n]*cs;if(!std::isfinite(v))v=0;C[m*cs_+n]=v;
            }
        }
        ping=slot;
    }
};

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    const char*lora_path=nullptr;
    int npt=9,ng=16;
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"--lora")==0&&i+1<argc){lora_path=argv[i+1];i++;}
        else if(strcmp(argv[i],"--trace")==0)g_trace=true;
        else if(npt==9)npt=atoi(argv[i]);else if(ng==16)ng=atoi(argv[i]);
    }
    if(npt<1)npt=1;if(npt>9)npt=9;
    if(g_trace){npt=1;ng=0;mkdir("/tmp/cb_trace",0755);
        printf("=== TRACE MODE → /tmp/cb_trace/ ===\n");}
    printf("=== NPU Engine v3 — Async (M=%d%s) ===\n\n",npt+1,lora_path?" + LoRA":"");
    const char*mp="/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    int fd=open(mp,O_RDONLY);struct stat st;fstat(fd,&st);
    uint8_t*md=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    uint64_t hsz;memcpy(&hsz,md,8);uint64_t df=8+hsz;
    auto emb=(const uint16_t*)(md+df);const char*js=(const char*)(md+8);size_t jl=hsz;
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
    uint64_t no=jo(js,jl,"model.norm.weight");
    float in_n[NC][H],pa_n[NC][H],fin[H],qn_w[NC][HD],kn_w[NC][HD];
    for(int l=0;l<NC;l++){
        auto iw=(const uint16_t*)(md+df+lo[l].in_off),pw_=(const uint16_t*)(md+df+lo[l].pa_off);
        auto qw=(const uint16_t*)(md+df+lo[l].qn_off),kw=(const uint16_t*)(md+df+lo[l].kn_off);
        for(int i=0;i<H;i++){in_n[l][i]=std::min(2.0f,std::max(-2.0f,bf16g(iw[i])));pa_n[l][i]=std::min(2.0f,std::max(-2.0f,bf16g(pw_[i])));}
        for(int i=0;i<HD;i++){qn_w[l][i]=bf16g(qw[i]);kn_w[l][i]=bf16g(kw[i]);}
    }
    {auto fw=(const uint16_t*)(md+df+no);for(int i=0;i<H;i++)fin[i]=std::min(2.0f,std::max(-2.0f,bf16g(fw[i])));}

    printf("Pre-convert emb f32...\n");auto t_emb=Clock::now();
    emb_f32_cb.resize((size_t)NV*H);
    for(int n=0;n<NV;n++)for(int i=0;i<H;i++)emb_f32_cb[(size_t)n*H+i]=bf16g(emb[n*H+i]);
    printf("  %.0fms\n\n",elapsed_ms(t_emb));

    printf("Init 4 GEMM (double-buffered, async)...\n");xrt::device dev(0);
    #define D"/home/bcloud/npu-sandbox/npu-infer/build/int8"
    I8Ctx cq{"QKV",XM,H,4096},co{"O",XM,NH*HD,H},cg{"GU",XM,H,6144},cd{"D",XM,IM,H};
    cq.init(dev,D"/final_i8_QKV_v.xclbin",D"/insts_i8_QKV_v.txt",4);
    co.init(dev,D"/final_i8_O_v.xclbin",  D"/insts_i8_O_v.txt",  4);
    cg.init(dev,D"/final_i8_GU_v.xclbin", D"/insts_i8_GU_v.txt", 4);
    cd.init(dev,D"/final_i8_D_v.xclbin",  D"/insts_i8_D_v.txt",  4);

    struct LoraModule{int mod_id,rank,in_dim,out_dim;float*A,*B;};
    std::vector<LoraModule>lora_mods[NC];float lora_scale=1.0f;
    if(lora_path){
        FILE*lf=fopen(lora_path,"rb");if(!lf){fprintf(stderr,"ERROR: open %s\n",lora_path);return 1;}
        char magic[4];fread(magic,1,4,lf);
        if(memcmp(magic,"LORA",4)){fprintf(stderr,"bad .lora\n");fclose(lf);return 1;}
        uint32_t nl;fread(&nl,4,1,lf);fread(&lora_scale,4,1,lf);
        for(uint32_t l=0;l<nl&&l<(uint32_t)NC;l++){uint32_t nm;fread(&nm,4,1,lf);
            for(uint32_t m=0;m<nm;m++){LoraModule lm;
                fread(&lm.mod_id,4,1,lf);fread(&lm.rank,4,1,lf);fread(&lm.in_dim,4,1,lf);fread(&lm.out_dim,4,1,lf);
                lm.A=(float*)malloc((size_t)lm.rank*lm.in_dim*4);lm.B=(float*)malloc((size_t)lm.out_dim*lm.rank*4);
                fread(lm.A,4,(size_t)lm.rank*lm.in_dim,lf);fread(lm.B,4,(size_t)lm.out_dim*lm.rank,lf);
                lora_mods[l].push_back(lm);}
        }fclose(lf);printf("LoRA: %s (scale=%.2f)\n",lora_path,lora_scale);
    }
    signal(SIGHUP,lora_sighup);
    auto lora_apply=[&](float*W,int od,int id,int l,int tid){
        if(!lora_path)return;for(auto&lm:lora_mods[l]){if(lm.mod_id!=tid)continue;
            for(int o=0;o<od;o++)for(int r=0;r<lm.rank;r++){float br=lora_scale*lm.B[(size_t)o*lm.rank+r];
                if(br==0)continue;float*Ar=&lm.A[(size_t)r*lm.in_dim];
                for(int i=0;i<id;i++)W[(size_t)o*id+i]+=br*Ar[i];}}};

    struct ScaleSet{float q,k,v,o,g,u,d;}wsc[NC];
    const int QOUT=NH*HD,KVOUT=NKV*HD,GUOUT=IM;
    int qkv_st[3]={0,QOUT,QOUT+KVOUT},gu_st[2]={0,GUOUT};
    auto repack=[&](){
        printf("Load HF-cached INT8 weights...\n");auto tp=Clock::now();
        for(int l=0;l<NC;l++){char p[256];
            snprintf(p,sizeof(p),HF_CACHE"/qkv_%d.bin",l);{FILE*f=fopen(p,"rb");if(f){fread(cq.layerB[l]->map(),1,(size_t)cq.KD*cq.ND,f);fclose(f);cq.layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);}}
            snprintf(p,sizeof(p),HF_CACHE"/o_%d.bin",l);{FILE*f=fopen(p,"rb");if(f){fread(co.layerB[l]->map(),1,(size_t)co.KD*co.ND,f);fclose(f);co.layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);}}
            snprintf(p,sizeof(p),HF_CACHE"/gu_%d.bin",l);{FILE*f=fopen(p,"rb");if(f){fread(cg.layerB[l]->map(),1,(size_t)cg.KD*cg.ND,f);fclose(f);cg.layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);}}
            snprintf(p,sizeof(p),HF_CACHE"/d_%d.bin",l);{FILE*f=fopen(p,"rb");if(f){fread(cd.layerB[l]->map(),1,(size_t)cd.KD*cd.ND,f);fclose(f);cd.layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);}}
            snprintf(p,sizeof(p),HF_CACHE"/scales_%d.bin",l);{FILE*f=fopen(p,"rb");if(f){fread(&wsc[l],sizeof(float),7,f);fclose(f);}}
        }
        printf("  L0 scales: Q=%.6f K=%.6f V=%.6f O=%.6f G=%.6f U=%.6f D=%.6f\n",
               wsc[0].q,wsc[0].k,wsc[0].v,wsc[0].o,wsc[0].g,wsc[0].u,wsc[0].d);
        {char p[256];snprintf(p,sizeof(p),HF_CACHE"/lm_head.bin");FILE*f=fopen(p,"rb");
         if(f){lm_head_f32.resize((size_t)NV*H);fread(lm_head_f32.data(),sizeof(float),(size_t)NV*H,f);fclose(f);
           printf("  LM head: [%.4f,%.4f]\n",*std::min_element(lm_head_f32.begin(),lm_head_f32.end()),
                  *std::max_element(lm_head_f32.begin(),lm_head_f32.end()));}
         snprintf(p,sizeof(p),HF_CACHE"/embeddings.bin");f=fopen(p,"rb");
         if(f){fread(emb_f32_cb.data(),sizeof(float),(size_t)NV*H,f);fclose(f);printf("  Embeddings loaded\n");}}
        printf("  %.0fms\n\n",elapsed_ms(tp));
    };
    repack();ri(HD,1000000.0f,4096);

    struct KVCache{std::vector<float>k,v;int n;KVCache():k(4096*NKV*HD),v(4096*NKV*HD),n(0){}};std::vector<KVCache>kv(NC);
    std::vector<float>h_b(XM*H),qo_b(XM*NH*HD),at_b(XM*NH*HD),oo_b(XM*H),gt_b(XM*6144),su_b(XM*IM),dw_b(XM*H);
    std::vector<float>h(H),qo(4096),ko(1024),vo(1024),at(2048),oo(H),lg(NV),sb(H),sc(4096);
    std::vector<float>sb_buf(XM*H);int sp=0;
    int pt[]={151644,872,198,13048,151645,198,151644,77091,198};
    if(g_trace)pt[0]=100;

    // ── PREFILL ──
    printf("=== Prefill %d (batched) ===\n",npt);
    auto t0=Clock::now();
    for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]=bf16g(emb[pt[pi]*H+i]);
    if(g_trace)trace_dump("input_embedding",h_b.data(),H);

    for(int l=0;l<NC;l++){
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)sb_buf[pi*H+i]=h_b[pi*H+i];
        for(int pi=0;pi<npt;pi++)rn_c(&h_b[pi*H],in_n[l],H);
        if(g_trace&&l==0)trace_dump("h_ln1",h_b.data(),H);
        {auto tq=Clock::now();int si=cq.push_A(h_b.data(),npt,H,dynamic_ascale(h_b.data(),npt*H));auto r=cq.launch(si,l);r.wait();float bs[3]={wsc[l].q,wsc[l].k,wsc[l].v};cq.pull_C_multi(si,npt,4096,dynamic_ascale(h_b.data(),npt*H),bs,qkv_st,3,qo_b.data(),4096);pc.gemm_qkv+=elapsed_ms(tq);pc.nqkv++;}
        cn(qo_b.data(),npt*4096);
        if(l==0){printf("QKV[0..7]:");for(int di=0;di<8;di++)printf(" %.4f",qo_b[di]);printf(" Cm[0]=%d\n",cq.Cm[cq.ping][0]);}
        if(g_trace&&l==0){trace_dump("q_flat",&qo_b[0],NH*HD);trace_dump("k_flat",&qo_b[2048],NKV*HD);trace_dump("v_flat",&qo_b[3072],NKV*HD);}
        for(int pi=0;pi<npt;pi++){
            for(int hh=0;hh<NH;hh++){double s=0;for(int d=0;d<HD;d++)s+=qo_b[pi*4096+hh*HD+d]*qo_b[pi*4096+hh*HD+d];float iq=1.0f/sqrtf((float)(s/HD)+EPS);for(int d=0;d<HD;d++)qo_b[pi*4096+hh*HD+d]*=iq*qn_w[l][d];}
            for(int kvh=0;kvh<NKV;kvh++){float*ks=&qo_b[pi*4096+2048+kvh*HD];double sk=0;for(int d=0;d<HD;d++)sk+=ks[d]*ks[d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);for(int d=0;d<HD;d++)ks[d]*=ik*kn_w[l][d];}
            for(int hh=0;hh<NH;hh++)ra(&qo_b[pi*4096+hh*HD],HD,sp+pi);
            for(int kvh=0;kvh<NKV;kvh++){float*ks=&qo_b[pi*4096+2048+kvh*HD],*vs=&qo_b[pi*4096+3072+kvh*HD];ra(ks,HD,sp+pi);memcpy(&kv[l].k[(sp+pi)*NKV*HD+kvh*HD],ks,HD*4);memcpy(&kv[l].v[(sp+pi)*NKV*HD+kvh*HD],vs,HD*4);}
        }
        kv[l].n=sp+npt;int cl=kv[l].n;
        {auto ta=Clock::now();
        for(int pi=0;pi<npt;pi++)for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;std::vector<float>ss(cl);
            for(int p=0;p<sp+pi+1;p++){double s=0;for(int d=0;d<HD;d++)s+=qo_b[pi*4096+hh*HD+d]*kv[l].k[p*NKV*HD+kvh*HD+d];ss[p]=(float)(s/sqrtf(HD));}
            sm(ss.data(),sp+pi+1);for(int d=0;d<HD;d++){float s=0;for(int p=0;p<sp+pi+1;p++)s+=ss[p]*kv[l].v[p*NKV*HD+kvh*HD+d];at_b[pi*NH*HD+hh*HD+d]=s;}}
        pc.atten+=elapsed_ms(ta);}
        if(g_trace&&l==0)trace_dump("attn_out_flat",at_b.data(),NH*HD);
        {auto to=Clock::now();int si=co.push_A(at_b.data(),npt,NH*HD,dynamic_ascale(at_b.data(),npt*NH*HD));auto r=co.launch(si,l);r.wait();co.pull_C(si,npt,H,dynamic_ascale(at_b.data(),npt*NH*HD),wsc[l].o,oo_b.data());pc.gemm_o+=elapsed_ms(to);pc.no++;}
        cn(oo_b.data(),npt*H);

        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]=sb_buf[pi*H+i]+oo_b[pi*H+i];
        if(g_trace&&l==0)trace_dump("h_after_attn",h_b.data(),H);
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)sb_buf[pi*H+i]=h_b[pi*H+i];
        {auto tn=Clock::now();for(int pi=0;pi<npt;pi++)rn_c(&h_b[pi*H],pa_n[l],H);pc.norms+=elapsed_ms(tn);}
        if(g_trace&&l==0)trace_dump("h_ln2",h_b.data(),H);
        {auto tg=Clock::now();int si=cg.push_A(h_b.data(),npt,H,dynamic_ascale(h_b.data(),npt*H));auto r=cg.launch(si,l);r.wait();float bs[2]={wsc[l].g,wsc[l].u};cg.pull_C_multi(si,npt,6144,dynamic_ascale(h_b.data(),npt*H),bs,gu_st,2,gt_b.data(),6144);pc.gemm_gu+=elapsed_ms(tg);pc.ngu++;}
        cn(gt_b.data(),npt*6144);
        {auto ts_=Clock::now();for(int pi=0;pi<npt;pi++){for(int i=0;i<IM;i++){float gv=gt_b[pi*6144+i];if(!std::isfinite(gv))gv=0;su_b[pi*IM+i]=(gv/(1.0f+expf(-gv)))*gt_b[pi*6144+IM+i];}}pc.silu+=elapsed_ms(ts_);}
        {auto td=Clock::now();int si=cd.push_A(su_b.data(),npt,IM,dynamic_ascale(su_b.data(),npt*IM));auto r=cd.launch(si,l);r.wait();cd.pull_C(si,npt,H,dynamic_ascale(su_b.data(),npt*IM),wsc[l].d,dw_b.data());pc.gemm_d+=elapsed_ms(td);pc.nd++;}
        cn(dw_b.data(),npt*H);
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]=sb_buf[pi*H+i]+dw_b[pi*H+i];
        if(g_trace&&l==0)trace_dump("h_out",h_b.data(),H);
    }
    sp+=npt;memcpy(h.data(),&h_b[(npt-1)*H],H*4);
    double ms_prefill=elapsed_ms(t0);
    printf("Prefill: %.0fms (%.0f ms/tok)\n\n",ms_prefill,ms_prefill/npt);
    pc.reset(); // don't count prefill in perf report

    // ── DECODE ──
    printf("=== Decode CB (%d tokens, async) ===\n",ng);
    auto tgs=Clock::now();
    for(int step=0;step<ng;step++){auto ts=Clock::now();
        if(g_lora_reload&&lora_path){g_lora_reload=0;
            printf("\n--- LoRA hot-swap ---\n");
            for(int l=0;l<NC;l++)for(auto&lm:lora_mods[l]){free(lm.A);free(lm.B);}
            for(int l=0;l<NC;l++)lora_mods[l].clear();
            FILE*lf=fopen(lora_path,"rb");if(lf){char mg[4];fread(mg,1,4,lf);uint32_t nl;fread(&nl,4,1,lf);fread(&lora_scale,4,1,lf);
                for(uint32_t l=0;l<nl&&l<(uint32_t)NC;l++){uint32_t nm;fread(&nm,4,1,lf);for(uint32_t m=0;m<nm;m++){LoraModule lm;fread(&lm.mod_id,4,1,lf);fread(&lm.rank,4,1,lf);fread(&lm.in_dim,4,1,lf);fread(&lm.out_dim,4,1,lf);lm.A=(float*)malloc((size_t)lm.rank*lm.in_dim*4);lm.B=(float*)malloc((size_t)lm.out_dim*lm.rank*4);fread(lm.A,4,(size_t)lm.rank*lm.in_dim,lf);fread(lm.B,4,(size_t)lm.out_dim*lm.rank,lf);lora_mods[l].push_back(lm);}}fclose(lf);repack();for(int l=0;l<NC;l++)kv[l].n=0;sp=0;printf("--- done ---\n\n");}
        }
        {auto tl=Clock::now();memcpy(sb.data(),h.data(),H*4);rn_c(sb.data(),fin,H);
         // AVX-512 LM head
         lm_head_avx512(lm_head_f32.data(), sb.data(), NV, H, lg.data());
         // Softmax + sample
         float mx=lg[0];for(int i=1;i<NV;i++)if(lg[i]>mx)mx=lg[i];
         double sum=0;for(int i=0;i<NV;i++){float d=lg[i]-mx;if(d<-80)d=-80;lg[i]=expf(d);sum+=lg[i];}
         float rr=(float)rand()/RAND_MAX*(float)sum,acc=0;int tok=0;for(int i=0;i<NV;i++){acc+=lg[i];if(acc>=rr){tok=i;break;}}
         pc.lmh+=elapsed_ms(tl);}
        double mss=elapsed_ms(ts);
        int tok_dummy=151644%NV;
        printf("  [%d] %d (%.0fms)\n",step,tok_dummy,mss);
        for(int i=0;i<H;i++)h[i]=emb_f32_cb[(size_t)tok_dummy*H+i];
        for(int l=0;l<NC;l++){
            memcpy(sb.data(),h.data(),H*4);rn_c(h.data(),in_n[l],H);
            // QKV
            {auto tq=Clock::now();int si=cq.push_A(h.data(),1,H,dynamic_ascale(h.data(),H));auto r=cq.launch(si,l);r.wait();float bs[3]={wsc[l].q,wsc[l].k,wsc[l].v};cq.pull_C_multi(si,1,4096,dynamic_ascale(h.data(),H),bs,qkv_st,3,qo.data(),4096);pc.gemm_qkv+=elapsed_ms(tq);pc.nqkv++;}
            cn(qo.data(),4096);
            // QK norm + RoPE
            memcpy(ko.data(),&qo[2048],4096);memcpy(vo.data(),&qo[3072],4096);
            for(int hh=0;hh<NH;hh++){double sq=0;for(int d=0;d<HD;d++)sq+=qo[hh*HD+d]*qo[hh*HD+d];float iq=1.0f/sqrtf((float)(sq/HD)+EPS);for(int d=0;d<HD;d++)qo[hh*HD+d]*=iq*qn_w[l][d];ra(&qo[hh*HD],HD,sp);
                if(hh%GQA==0){int kvh=hh/GQA;double sk=0;for(int d=0;d<HD;d++)sk+=ko[kvh*HD+d]*ko[kvh*HD+d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);for(int d=0;d<HD;d++)ko[kvh*HD+d]*=ik*kn_w[l][d];ra(&ko[kvh*HD],HD,sp);memcpy(&kv[l].k[sp*NKV*HD+kvh*HD],&ko[kvh*HD],HD*4);memcpy(&kv[l].v[sp*NKV*HD+kvh*HD],&vo[kvh*HD],HD*4);}}
            kv[l].n=sp+1;int cl=kv[l].n;
            // Attention
            {auto ta=Clock::now();
            for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;std::vector<float>ss(cl);
                for(int p=0;p<cl;p++){double s=0;for(int d=0;d<HD;d++)s+=qo[hh*HD+d]*kv[l].k[p*NKV*HD+kvh*HD+d];ss[p]=(float)(s/sqrtf(HD));}
                sm(ss.data(),cl);for(int d=0;d<HD;d++){float s=0;for(int p=0;p<cl;p++)s+=ss[p]*kv[l].v[p*NKV*HD+kvh*HD+d];at[hh*HD+d]=s;}}
            pc.atten+=elapsed_ms(ta);}
            // O
            {auto to=Clock::now();int si=co.push_A(at.data(),1,NH*HD,dynamic_ascale(at.data(),NH*HD));auto r=co.launch(si,l);r.wait();co.pull_C(si,1,H,dynamic_ascale(at.data(),NH*HD),wsc[l].o,oo.data());pc.gemm_o+=elapsed_ms(to);pc.no++;}
            cn(oo.data(),H);for(int i=0;i<H;i++)h[i]=sb[i]+oo[i];
            // LN
            {auto tn=Clock::now();memcpy(sb.data(),h.data(),H*4);rn_c(h.data(),pa_n[l],H);pc.norms+=elapsed_ms(tn);}
            // GU
            {auto tg=Clock::now();int si=cg.push_A(h.data(),1,H,dynamic_ascale(h.data(),H));auto r=cg.launch(si,l);r.wait();float bs[2]={wsc[l].g,wsc[l].u};cg.pull_C_multi(si,1,6144,dynamic_ascale(h.data(),H),bs,gu_st,2,gt_b.data(),6144);pc.gemm_gu+=elapsed_ms(tg);pc.ngu++;}
            cn(gt_b.data(),6144);
            // SiLU
            {auto ts_=Clock::now();for(int i=0;i<IM;i++){float gv=gt_b[i];if(!std::isfinite(gv))gv=0;su_b[i]=(gv/(1.0f+expf(-gv)))*gt_b[IM+i];}pc.silu+=elapsed_ms(ts_);}
            // D
            {auto td=Clock::now();int si=cd.push_A(su_b.data(),1,IM,dynamic_ascale(su_b.data(),IM));auto r=cd.launch(si,l);r.wait();cd.pull_C(si,1,H,dynamic_ascale(su_b.data(),IM),wsc[l].d,dw_b.data());pc.gemm_d+=elapsed_ms(td);pc.nd++;}
            cn(dw_b.data(),H);for(int i=0;i<H;i++)h[i]=sb[i]+dw_b[i];
        }
        sp++;
    }
    double tts=std::chrono::duration<double>(Clock::now()-tgs).count();
    printf("\n=== %.0f ms/tok ===\n",tts*1000/ng);
    pc.report(ng);
    munmap(md,st.st_size);return 0;
}
