/** NPU Engine — Universal Fast. Model-agnostic auto-detect + v12 speed.
 *  M=32 batched decode, OpenMP attention, OpenMP LM head, f32 embeddings.
 *  Supports ALL models with tagged xclbins. Target: >80 tok/s on any model. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <exception>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include "model_config.h"
extern "C" float* dequant_i8_to_float_ex(const uint8_t*,int,int,int*,int*);
static inline float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static inline float bf16g(uint16_t v){return(v&0x7F80)==0x7F80?0.0f:bf16f(v);}
static constexpr float EPS=1e-6f;
static inline void cn(float*x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
static inline void sm(float*sc,int n){if(n<=0)return;cn(sc,n);float mx=sc[0];
    for(int i=1;i<n;i++)if(sc[i]>mx)mx=sc[i];double s=0;
    for(int i=0;i<n;i++){float d=sc[i]-mx;if(d>80)d=80;else if(d<-80)d=-80;sc[i]=expf(d);s+=sc[i];}
    if(s<=0){float iv=1.0f/n;for(int i=0;i<n;i++)sc[i]=iv;return;}
    float is=1.0f/(float)s;for(int i=0;i<n;i++)sc[i]*=is;}
static inline void rn_c(float*x,const float*w,int n){cn(x,n);double ss=0;
    for(int i=0;i<n;i++)if(std::isfinite(x[i]))ss+=(double)x[i]*x[i];
    float ir=1.0f/sqrtf((float)(ss/n)+EPS);for(int i=0;i<n;i++)x[i]=std::isfinite(x[i])?x[i]*ir*w[i]:0.0f;}
static std::vector<float>rc,rs;
static void ri(int hd,float th,int mp){int hd2=hd/2;rc.resize(mp*hd);rs.resize(mp*hd);
    for(int p=0;p<mp;p++)for(int d=0;d<hd2;d++){
        float f=1.0f/powf(th,(float)d/hd2),a=p*f;
        rc[p*hd+d]=cosf(a);rs[p*hd+d]=sinf(a);}}
static inline void ra(float*x,int hd,int p){int hd2=hd/2;for(int d=0;d<hd2;d++){
    float a=x[d],b=x[d+hd2],c=rc[p*hd+d],s=rs[p*hd+d];x[d]=a*c-b*s;x[d+hd2]=b*c+a*s;}}
static std::vector<float> emb_f32; // f32 embeddings for fast LM head
static std::vector<float> lm_head_f32; // f32 lm_head weights (separate from emb)
// dequant_i8_to_float(_ex) returns row-major [out_features, in_features] (PyTorch nn.Linear);
// packB()/go() need the transpose — [in_features, out_features] — since the GEMM computes
// A[tokens,in] @ B[in,out].
static void transpose_pack(const float* src, int out_f, int in_f, float* dst, int dst_stride, int dst_offset) {
    for (int o = 0; o < out_f; o++)
        for (int i = 0; i < in_f; i++)
            dst[(size_t)i * dst_stride + dst_offset + o] = src[(size_t)o * in_f + i];
}
// Dynamic per-call activation quantization scale.
// Hardcoded 5.0f/127.0f assumes activations stay in [-5,5], but measured post-RMSNorm
// activations range as wide as [-8.24,7.01], silently clipping every layer.
static inline float dynamic_ascale(const float* x, int n) {
    float amax = 0;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (std::isfinite(a) && a > amax) amax = a; }
    if (amax < 1e-12f) amax = 1.0f;
    return amax / 127.0f;
}

static uint64_t jo(const char*js,size_t jl,const char*nm){size_t nl=strlen(nm);
    const char*p=js,*e=js+jl;while(p<e){auto q=(const char*)memmem(p,e-p,nm,nl);
        if(!q)return 0;if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){
            auto o=strstr(q,"\"data_offsets\"");if(o){auto a=strchr(o,'[');if(a)return strtoull(a+1,NULL,10);}}p=q+1;}return 0;}

struct I8Ctx{int MD,KD,ND,NL;std::unique_ptr<xrt::xclbin>xc;std::unique_ptr<xrt::hw_context>hc;
    std::unique_ptr<xrt::kernel>k;std::vector<uint32_t>ins;std::unique_ptr<xrt::bo>bI,bA,bC;
    std::vector<std::unique_ptr<xrt::bo>>layerB;int8_t*Am;int16_t*Cm;
    bool initialized=false;
    ~I8Ctx(){/* Am/Cm are mapped from bA/bC — destroyed by unique_ptr dtors */}
    bool isReady(){return initialized&&k&&bA&&bC;}
    bool init(xrt::device&d,const char*xp,const char*ip,int gid_B,int nlayers){
        NL=nlayers;FILE*f=fopen(ip,"rb");if(!f)return false;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);
        ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
        xc=std::make_unique<xrt::xclbin>(std::string(xp));d.register_xclbin(*xc);
        hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");
        bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));
        memcpy(bI->map(),ins.data(),ins.size()*4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bA=std::make_unique<xrt::bo>(d,(size_t)MD*KD,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));
        bC=std::make_unique<xrt::bo>(d,(size_t)MD*ND*2,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));
        Am=(int8_t*)bA->map();Cm=(int16_t*)bC->map();
        for(int l=0;l<NL;l++)layerB.emplace_back(std::make_unique<xrt::bo>(d,(size_t)KD*ND,XRT_BO_FLAGS_HOST_ONLY,k->group_id(gid_B)));
        initialized=true;return true;}
    void packB(int l,const float*w,int K,int N,float&sout){float amax=0;
        for(int i=0;i<K*N;i++){float a=fabsf(w[i]);if(std::isfinite(a)&&a>amax)amax=a;}
        if(amax<1e-12f)amax=1.0f;sout=amax/127.0f;float is=127.0f/amax;auto*Bm=(int8_t*)layerB[l]->map();
        for(int i=0;i<K*N;i++){float v=w[i];if(!std::isfinite(v))v=0;
            int x=(int)roundf(v*is);if(x>127)x=127;else if(x<-127)x=-127;Bm[i]=(int8_t)x;}layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);}
    // Async quantize: packs float activations into the A buffer without syncing
    // Returns the quantized buffer pointer for later sync_and_launch.
    inline int8_t* quantize_async(const float*A,int am,int ak,float ascale){
        float ais=1.0f/ascale;
        memset(Am,0,(size_t)am*KD);
        for(int m=0;m<am;m++)for(int k=0;k<ak;k++){
            float v=A[m*ak+k];if(!std::isfinite(v))v=0;
            int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;
            Am[m*KD+k]=(int8_t)q;}
        return Am;
    }
    // Sync A to device (non-blocking DMA, can overlap with NPU compute).
    inline void sync_A(int l){
        (void)l;
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    // Launch kernel without sync (buffer must already be synced). Returns run handle.
    inline xrt::run launch(int l){
        return (*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA,*layerB[l],*bC);
    }
    // Sync A to device and launch kernel. Returns run handle.
    inline xrt::run sync_and_launch(int l){
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return (*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA,*layerB[l],*bC);
    }
    // Wait for run, sync C back, and dequantize.
    inline void dequantize(xrt::run& r,float*C,int am,int an,float ascale,float Bscale){
        r.wait();
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        float cs=ascale*Bscale;
        for(int m=0;m<am;m++)for(int n=0;n<an;n++){
            float val=(float)Cm[m*ND+n]*cs;if(!std::isfinite(val))val=0;
            C[m*an+n]=val;}
    }
    // Wait for NPU kernel completion without readback.
    // Returns immediately after kernel finishes. Call sync_back_and_dequant() later.
    inline void wait_kernel(xrt::run& r){
        r.wait();
    }
    // Sync C back from device and dequantize (call after wait_kernel).
    // This is CPU-only work that CAN overlap with the next kernel's NPU execution.
    inline void sync_back_and_dequant(float*C,int am,int an,float ascale,float Bscale){
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        float cs=ascale*Bscale;
        for(int m=0;m<am;m++)for(int n=0;n<an;n++){
            float val=(float)Cm[m*ND+n]*cs;if(!std::isfinite(val))val=0;
            C[m*an+n]=val;}
    }
    // Synchronous go() — simple, always works
    inline bool go(int l,const float*A,int am,int ak,float ascale,float Bscale,float*C,int an){
        quantize_async(A,am,ak,ascale);
        auto r=sync_and_launch(l);
        r.wait();
        dequantize(r,C,am,an,ascale,Bscale);
        return true;
    }
    // Fast path: launch, return run handle for later wait+dequant
    inline xrt::run launch_async(int l,const float*A,int am,int ak,float ascale){
        quantize_async(A,am,ak,ascale);
        return sync_and_launch(l);
    }
    // Complete an async launch: wait + dequant
    inline void finish_async(xrt::run& r,float*C,int am,int an,float ascale,float Bscale){
        r.wait();
        dequantize(r,C,am,an,ascale,Bscale);
    }
};

// v12: OpenMP attention — parallelize across heads, with optional causal mask
static inline void attn_omp(float*qo,float*at,int cl,const float*kv_k,const float*kv_v,int NH,int NKV,int HD,int GQA,int max_pos=-1){
    if(max_pos<0)max_pos=cl;
    #pragma omp parallel for
    for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;
        std::vector<float> scores(cl);float mx=-1e30f;
        for(int p=0;p<cl;p++){if(p>=max_pos){scores[p]=-1e30f;continue;}
            double s=0;int qoff=hh*HD,koff=p*NKV*HD+kvh*HD;
            #pragma omp simd reduction(+:s)
            for(int d=0;d<HD;d++)s+=(double)qo[qoff+d]*kv_k[koff+d];scores[p]=(float)(s/sqrtf((float)HD));if(scores[p]>mx)mx=scores[p];}
        double sw=0;for(int p=0;p<cl;p++){scores[p]=expf(scores[p]-mx);sw+=scores[p];}
        float isw=sw>0?1.0f/(float)sw:1.0f/cl;
        for(int d=0;d<HD;d++){float acc=0;int aoff=hh*HD+d;
            #pragma omp simd reduction(+:acc)
            for(int p=0;p<cl;p++)acc+=scores[p]*kv_v[p*NKV*HD+kvh*HD+d];at[aoff]=acc*isw;}}
}

// v12: OpenMP LM head with f32 embeddings — top-K sampling
// emb: embedding/lm_head table (row-major [vocab_size, hidden_size])
inline void lm_topk_omp(const float*hidden,float*lg,int*top_ids,int K,int NV,int H,const float*emb,float mx=-1e30f){
    #pragma omp parallel for reduction(max:mx)
    for(int n=0;n<NV;n++){double s=0;const float*e=&emb[(size_t)n*H];const float*h=hidden;
        #pragma omp simd reduction(+:s)
        for(int k=0;k<H;k++)s+=(double)h[k]*e[k];lg[n]=(float)s;if(lg[n]>mx)mx=lg[n];}
    double sum=0;
    #pragma omp parallel for reduction(+:sum)
    for(int n=0;n<NV;n++){float d=lg[n]-mx;if(d<-80)d=-80;lg[n]=expf(d);sum+=lg[n];}
    float r=(float)rand()/RAND_MAX*(float)sum,acc=0;
    for(int n=0;n<NV;n++){acc+=lg[n];if(acc>=r){top_ids[0]=n;break;}}
    struct TI{int id;float v;};TI top[32];
    for(int b=0;b<K;b++){top[b].id=-1;top[b].v=-1e30f;}
    for(int n=0;n<NV;n++){float v=lg[n];for(int b=0;b<K;b++){if(v>top[b].v){memmove(&top[b+1],&top[b],(K-1-b)*sizeof(TI));top[b].id=n;top[b].v=v;break;}}}
    for(int b=0;b<K;b++)top_ids[b]=top[b].id;
}

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    if(argc<2){fprintf(stderr,"Usage: %s model.q4nx [decode_tokens] [input_tokens_file|-]\n",argv[0]);return 1;}
    // Check for --worker flag (subprocess protocol mode)
    bool worker_mode=false;
    for(int i=2;i<argc;i++){if(strcmp(argv[i],"--worker")==0){worker_mode=true;break;}}
    const char*mp=argv[1];int ng=(argc>2&&!worker_mode)?atoi(argv[2]):32;if(ng<1)ng=1;if(ng>4096)ng=4096; // cap to KV cache size (issue #112)
    const char*input_tok_file=(argc>3&&!worker_mode&&argv[3][0]!='\0')?argv[3]:nullptr;

    // Model tag
    std::string mp_s(mp),model_tag;auto ls=mp_s.rfind('/');auto sl=mp_s.rfind('/',ls-1);
    model_tag=(sl!=std::string::npos&&ls!=std::string::npos)?mp_s.substr(sl+1,ls-sl-1):mp_s.substr(ls+1);
    for(auto&c:model_tag){c=tolower(c);if(c=='-'||c=='.')c='_';}
    const char*sfxs[]={"_npu2","_instruct","_it","_it_npu2"};
    for(auto sf:sfxs){size_t sl=strlen(sf);if(model_tag.size()>sl&&model_tag.substr(model_tag.size()-sl)==sf)model_tag=model_tag.substr(0,model_tag.size()-sl);}

    // Parse config
    ModelConfig cfg=parse_q4nx_header(mp,model_tag.c_str());
    if(!cfg.valid()){fprintf(stderr,"ERR: invalid model config\n");return 1;}
    int H=cfg.H,NC=cfg.NC,NH=cfg.NH,NKV=cfg.NKV,HD=cfg.HD,IM=cfg.IM,NV=cfg.NV,GQA=cfg.GQA,XM=cfg.XM;
    fprintf(stderr,"=== NPU Engine Universal — %s ===\n",model_tag.c_str());
    fprintf(stderr,"H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d NV=%d GU_split=%d\n",H,NC,NH,NKV,HD,IM,NV,cfg.gu_split);

    // Open model
    int fd=open(mp,O_RDONLY);struct stat st;fstat(fd,&st);
    uint8_t*md=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    uint64_t hsz;memcpy(&hsz,md,8);uint64_t df=8+hsz;
    auto i8p=[&](uint64_t o){return md+df+o;};auto emb=(const uint16_t*)(md+df);
    const char*js=(const char*)(md+8);size_t jl=hsz;

    // Pre-convert embeddings f32 (v12 optimization)
    fprintf(stderr,"Pre-convert emb f32...\n");auto te=std::chrono::steady_clock::now();
    emb_f32.resize((size_t)NV*H);
    for(int n=0;n<NV;n++)for(int i=0;i<H;i++)emb_f32[(size_t)n*H+i]=bf16g(emb[n*H+i]);
    fprintf(stderr,"  %.0fms\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-te).count());

    // Norm weights
    std::vector<uint64_t> in_off(NC),pa_off(NC),qn_off(NC),kn_off(NC),qp(NC),kp(NC),vp(NC),op(NC),gp(NC),up(NC),dp(NC);
    char bn[128];
    for(int l=0;l<NC;l++){
        snprintf(bn,128,"model.layers.%d.self_attn.q_proj.weight",l);qp[l]=jo(js,jl,bn);
        snprintf(bn,128,"model.layers.%d.self_attn.k_proj.weight",l);kp[l]=jo(js,jl,bn);
        snprintf(bn,128,"model.layers.%d.self_attn.v_proj.weight",l);vp[l]=jo(js,jl,bn);
        snprintf(bn,128,"model.layers.%d.self_attn.o_proj.weight",l);op[l]=jo(js,jl,bn);
        snprintf(bn,128,"model.layers.%d.mlp.gate_proj.weight",l);gp[l]=jo(js,jl,bn);
        snprintf(bn,128,"model.layers.%d.mlp.up_proj.weight",l);up[l]=jo(js,jl,bn);
        snprintf(bn,128,"model.layers.%d.mlp.down_proj.weight",l);dp[l]=jo(js,jl,bn);
        snprintf(bn,128,"model.layers.%d.input_layernorm.weight",l);in_off[l]=jo(js,jl,bn);
        snprintf(bn,128,"model.layers.%d.post_attention_layernorm.weight",l);pa_off[l]=jo(js,jl,bn);
        snprintf(bn,128,"model.layers.%d.self_attn.q_norm.weight",l);qn_off[l]=jo(js,jl,bn);
        snprintf(bn,128,"model.layers.%d.self_attn.k_norm.weight",l);kn_off[l]=jo(js,jl,bn);}
    uint64_t no=jo(js,jl,"model.norm.weight");
    uint64_t lo=jo(js,jl,"lm_head.weight");
    std::vector<std::vector<float>> in_n(NC,std::vector<float>(H)),pa_n(NC,std::vector<float>(H)),qn_w(NC,std::vector<float>(HD)),kn_w(NC,std::vector<float>(HD));
    std::vector<float> fin_v(H);
    for(int l=0;l<NC;l++){auto iw=(const uint16_t*)(md+df+in_off[l]),pw=(const uint16_t*)(md+df+pa_off[l]);
        for(int i=0;i<H;i++){in_n[l][i]=bf16g(iw[i]);pa_n[l][i]=bf16g(pw[i]);}
        if(cfg.has_q_norm&&qn_off[l]){auto qq=(const uint16_t*)(md+df+qn_off[l]);for(int i=0;i<HD;i++)qn_w[l][i]=bf16g(qq[i]);}
        if(cfg.has_k_norm&&kn_off[l]){auto kk=(const uint16_t*)(md+df+kn_off[l]);for(int i=0;i<HD;i++)kn_w[l][i]=bf16g(kk[i]);}}
    {auto fw=(const uint16_t*)(md+df+no);for(int i=0;i<H;i++)fin_v[i]=bf16g(fw[i]);}

    // I8 tile rows
    auto gi8=[&](const char*k)->int{int r=0;find_tensor_info(js,jl,k,&r);return r;};
    int q_i8=gi8("model.layers.0.self_attn.q_proj.weight"),k_i8=gi8("model.layers.0.self_attn.k_proj.weight"),v_i8=gi8("model.layers.0.self_attn.v_proj.weight");
    int o_i8=gi8("model.layers.0.self_attn.o_proj.weight"),g_i8=gi8("model.layers.0.mlp.gate_proj.weight"),u_i8=gi8("model.layers.0.mlp.up_proj.weight"),d_i8=gi8("model.layers.0.mlp.down_proj.weight");
    int lm_i8=gi8("lm_head.weight");

    // Load lm_head.weight separately — NOT tied to embed_tokens.weight for this model
    if(lo&&lm_i8>0){int lr,lc;float*lm_raw=dequant_i8_to_float_ex(i8p(lo),lm_i8,H,&lr,&lc);if(lm_raw){
        lm_head_f32.assign(lm_raw,lm_raw+(size_t)lr*lc);free(lm_raw);
        fprintf(stderr,"  lm_head: %dx%d (loaded from JSON), using for final logits\n",lr,lc);
    }else{fprintf(stderr,"  lm_head: dequant failed, falling back to emb\n");}}
    if(lm_head_f32.empty()){fprintf(stderr,"  lm_head: using emb_f32 (tied embeddings)\n");}
    const float* lm_emb = lm_head_f32.empty() ? emb_f32.data() : lm_head_f32.data();

    // Init NPU
    fprintf(stderr,"Init NPU...\n");xrt::device dev(0);
    // Xclbin directory: respect NPU_XCLBIN_DIR env var, fall back to repo-relative path
    const char* env_xd = getenv("NPU_XCLBIN_DIR");
    std::string xd = env_xd ? env_xd : "engine/npu/xclbins";
    auto xp=[&](const char*t){return xd+"/final_i8_"+t+"_"+cfg.model_tag+".xclbin";};
    auto ip=[&](const char*t){return xd+"/insts_i8_"+t+"_"+cfg.model_tag+".txt";};

    I8Ctx cq,co,cg,cd;cq.MD=XM;cq.KD=cfg.xclbin_qkv_k;cq.ND=cfg.xclbin_qkv_n;co.MD=XM;co.KD=cfg.xclbin_o_k;co.ND=cfg.xclbin_o_n;cd.MD=XM;cd.KD=cfg.xclbin_d_k;cd.ND=cfg.xclbin_d_n;
    if(cfg.gu_split){cg.MD=XM;cg.KD=cfg.xclbin_g_k;cg.ND=cfg.xclbin_g_n;}else{cg.MD=XM;cg.KD=cfg.xclbin_gu_k;cg.ND=cfg.xclbin_gu_n;}
    if(!cq.init(dev,xp("QKV").c_str(),ip("QKV").c_str(),4,NC)){fprintf(stderr,"FAIL QKV\n");return 1;}
    if(!co.init(dev,xp("O").c_str(),ip("O").c_str(),4,NC)){fprintf(stderr,"FAIL O\n");return 1;}
    if(cfg.gu_split){if(!cg.init(dev,xp("G").c_str(),ip("G").c_str(),4,NC)){fprintf(stderr,"FAIL G\n");return 1;}}else{if(!cg.init(dev,xp("GU").c_str(),ip("GU").c_str(),4,NC)){fprintf(stderr,"FAIL GU\n");return 1;}}
    if(!cd.init(dev,xp("D").c_str(),ip("D").c_str(),4,NC)){fprintf(stderr,"FAIL D\n");return 1;}
    std::unique_ptr<I8Ctx> cu_ptr;
    if(cfg.gu_split){cu_ptr=std::make_unique<I8Ctx>();cu_ptr->MD=XM;cu_ptr->KD=cfg.xclbin_u_k;cu_ptr->ND=cfg.xclbin_u_n;if(!cu_ptr->init(dev,xp("U").c_str(),ip("U").c_str(),4,NC)){fprintf(stderr,"FAIL U\n");return 1;}}

    fprintf(stderr,"Dequant+pack...\n");auto tp=std::chrono::steady_clock::now();
    std::vector<float> qsc(NC),osc(NC),gsc(NC),dsc(NC),usc(NC);
    const int QOUT=NH*HD,KVOUT=NKV*HD;   // QKV out_features, in_features=H (default dequant correct)
    const int OOUT=H,OIN=NH*HD;          // O: out=H, in=NH*HD — dequant needs OIN
    const int GUOUT=IM;                   // Gate/Up: out=IM, in=H
    const int DOUT=H,DIN=IM;              // Down: out=H, in=IM — dequant needs DIN
    for(int l=0;l<NC;l++){int qr,kr,vr,unused;
        float*qw=dequant_i8_to_float_ex(i8p(qp[l]),q_i8,H,&qr,&unused),*kw=dequant_i8_to_float_ex(i8p(kp[l]),k_i8,H,&kr,&unused),*vw=dequant_i8_to_float_ex(i8p(vp[l]),v_i8,H,&vr,&unused);
        int t=QOUT+KVOUT+KVOUT;std::vector<float>w((size_t)H*t);
        transpose_pack(qw,QOUT,H,w.data(),t,0); transpose_pack(kw,KVOUT,H,w.data(),t,QOUT); transpose_pack(vw,KVOUT,H,w.data(),t,QOUT+KVOUT);
        cq.packB(l,w.data(),H,t,qsc[l]);free(qw);free(kw);free(vw);
        int or2,oc2;float*ow=dequant_i8_to_float_ex(i8p(op[l]),o_i8,OIN,&or2,&oc2);
        std::vector<float>wo((size_t)OIN*OOUT);transpose_pack(ow,OOUT,OIN,wo.data(),OOUT,0);
        co.packB(l,wo.data(),OIN,OOUT,osc[l]);free(ow);
        int gr,ur;float*gw=dequant_i8_to_float_ex(i8p(gp[l]),g_i8,H,&gr,&unused),*uw=dequant_i8_to_float_ex(i8p(up[l]),u_i8,H,&ur,&unused);
        if(cfg.gu_split){
            std::vector<float>wg((size_t)H*gr);transpose_pack(gw,GUOUT,H,wg.data(),gr,0);
            cg.packB(l,wg.data(),H,gr,gsc[l]);
            std::vector<float>wu((size_t)H*ur);transpose_pack(uw,GUOUT,H,wu.data(),ur,0);
            cu_ptr->packB(l,wu.data(),H,ur,usc[l]);
        }else{
            int t2=gr+ur;std::vector<float>w2((size_t)H*t2);
            transpose_pack(gw,GUOUT,H,w2.data(),t2,0);transpose_pack(uw,GUOUT,H,w2.data(),t2,GUOUT);
            cg.packB(l,w2.data(),H,t2,gsc[l]);
        }free(gw);free(uw);
        int dr2,dc2;float*dw=dequant_i8_to_float_ex(i8p(dp[l]),d_i8,DIN,&dr2,&dc2);
        std::vector<float>wd((size_t)DIN*DOUT);transpose_pack(dw,DOUT,DIN,wd.data(),DOUT,0);
        cd.packB(l,wd.data(),DIN,DOUT,dsc[l]);free(dw);}
    fprintf(stderr,"  %.0fms\n\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tp).count());

    // RoPE
    ri(HD,cfg.rope_theta,4096);
    int kv_dwords=NKV*HD/2;

    // Decode batch width.
    //
    // WARNING (issue #111): the "M=32 batched decode" path is NOT a correct
    // decoding algorithm. It embeds the 32 top-K candidates for a *single*
    // next position as if they were 32 *sequential* tokens (see the loop that
    // does h_b[b*H+i]=emb_f32[top_ids[b]*H+i]), writes all 32 into the KV cache
    // at consecutive positions, and runs attention with cl=sp+batch_size --
    // i.e. every position attends over 31 not-yet-decoded, mutually-exclusive
    // "future" positions (non-causal). This corrupts even position 0's output,
    // so the reported 32x throughput described tokens that were never valid.
    //
    // Until a real speculative draft+verify is implemented (accept only the
    // longest matching prefix, roll the KV cache back on a miss), BS is pinned
    // to 1 -> plain causal single-token greedy decode, which is correct.
    // Do not raise this without implementing verification.
    int BS=1;
    struct KVCache{std::vector<float>k,v;int n;KVCache(int size):k(size),v(size),n(0){}};
    int kv_size=4096*NKV*HD;
    std::vector<KVCache> kv_caches;for(int i=0;i<NC;i++)kv_caches.emplace_back(kv_size);
    int qkv_n=cfg.qkv_total;
    std::vector<float> h_b(XM*H), qo_b(XM*qkv_n), at_b(XM*NH*HD), oo_b(XM*H), gt_b(XM*(cfg.gu_split?IM:2*IM)), su_b(XM*IM), dw_b(XM*H);
    std::vector<float> h_data(H), qo_data(qkv_n*BS), ko_data(NKV*HD*BS), vo_data(NKV*HD*BS), at_data(NH*HD*BS), oo_data(H*BS);
    std::vector<float> gt_data((cfg.gu_split?IM:2*IM)*BS), su_data(IM*BS), dwo_data(H*BS), sb_data(XM*H), lg_buf(NV);
    int sp=0;
    // ===== WORKER MODE (subprocess protocol) =====
    // The Zig fused executor (fused_execute.zig) sends individual GEMM
    // operations (QKV, OPROJ, GATEUP, DOWN) via this protocol. Each request
    // is header[4] (op, layer, batch, in_dim) followed by float input data.
    // Response is header[2] (0=ok, out_dim) followed by float output data.
    if(worker_mode){
        fprintf(stderr,"WORKER_READY\n");
        fflush(stderr);
        uint32_t hdr[4];
        while(fread(hdr,sizeof(uint32_t),4,stdin)==4){
            uint32_t op=hdr[0],layer=hdr[1],batch=hdr[2],in_dim=hdr[3];
            if(op==0) break; // QUIT

            // Input validation: batch and in_dim must be reasonable
            if(batch==0||batch>XM||in_dim==0||in_dim>4096||layer>=(uint32_t)NC){
                uint32_t resp[2]={1,0};
                fwrite(resp,sizeof(uint32_t),2,stdout);
                fflush(stdout);
                // Drain input payload
                std::vector<float> drain(batch*in_dim);
                fread(drain.data(),sizeof(float),batch*in_dim,stdin);
                continue;
            }

            std::vector<float> in_data(batch*in_dim);
            if(fread(in_data.data(),sizeof(float),batch*in_dim,stdin)!=(size_t)(batch*in_dim)) break;

            uint32_t out_dim=0;
            std::vector<float> out_data;
            bool ok=true;

            try{
                if(op==1&&cq.isReady()){ // QKV projection
                    out_dim=cfg.qkv_total;
                    out_data.resize(batch*out_dim,0);
                    float ascale=dynamic_ascale(in_data.data(),batch*in_dim);
                    cq.go(layer,in_data.data(),batch,(int)in_dim,ascale,qsc[layer],out_data.data(),(int)out_dim);
                }else if(op==2&&co.isReady()){ // O projection
                    out_dim=H;
                    out_data.resize(batch*out_dim,0);
                    float ascale=dynamic_ascale(in_data.data(),batch*in_dim);
                    co.go(layer,in_data.data(),batch,(int)in_dim,ascale,osc[layer],out_data.data(),(int)out_dim);
                }else if(op==3&&cg.isReady()){ // Gate+Up
                    out_dim=cfg.gu_split?IM:(2*IM);
                    out_data.resize(batch*out_dim,0);
                    float ascale=dynamic_ascale(in_data.data(),batch*in_dim);
                    cg.go(layer,in_data.data(),batch,(int)in_dim,ascale,gsc[layer],out_data.data(),(int)out_dim);
                }else if(op==4&&cfg.gu_split&&cu_ptr&&cu_ptr->isReady()){ // Up
                    out_dim=IM;
                    out_data.resize(batch*out_dim,0);
                    float ascale=dynamic_ascale(in_data.data(),batch*in_dim);
                    cu_ptr->go(layer,in_data.data(),batch,(int)in_dim,ascale,usc[layer],out_data.data(),(int)out_dim);
                }else if(op==5&&cd.isReady()){ // Down
                    out_dim=H;
                    out_data.resize(batch*out_dim,0);
                    float ascale=dynamic_ascale(in_data.data(),batch*in_dim);
                    cd.go(layer,in_data.data(),batch,(int)in_dim,ascale,dsc[layer],out_data.data(),(int)out_dim);
                }else{
                    ok=false;
                }
            }catch(std::exception&e){
                fprintf(stderr,"NPU worker op %u layer %u batch %u: %s\n",op,layer,batch,e.what());
                fflush(stderr);
                ok=false;
            }catch(...){
                fprintf(stderr,"NPU worker op %u layer %u batch %u: unknown error\n",op,layer,batch);
                fflush(stderr);
                ok=false;
            }

            if(!ok){
                uint32_t resp[2]={1,0}; // error
                fwrite(resp,sizeof(uint32_t),2,stdout);
                fflush(stdout);
                continue;
            }

            // Success: send response code + output
            uint32_t resp[2]={0,out_dim};
            fwrite(resp,sizeof(uint32_t),2,stdout);
            fwrite(out_data.data(),sizeof(float),batch*out_dim,stdout);
            fflush(stdout);
        }
        // Use _exit() to skip destructor cleanup — XRT's BO destructors can
        // corrupt glibc's heap when vectors containing GB-scale weight data
        // (emb_f32 ~594MB, lm_head_f32 ~594MB, kv_caches ~896MB) race with
        // XRT dma-buf teardown during normal exit() destructor chain.
        _exit(0);
    }

    // Load input tokens from file or use default hardcoded sequence
    std::vector<int> pt_vec;
    if(input_tok_file){
        FILE* tf;
        if(strcmp(input_tok_file,"-")==0) tf=stdin;  // stdin convention must precede fopen (fixes #88)
        else {
            tf=fopen(input_tok_file,"r");
            if(!tf){ fprintf(stderr,"Cannot open input tokens: %s\n",input_tok_file); return 1; }
        }
        int tid;
        while(fscanf(tf,"%d",&tid)==1) pt_vec.push_back(tid);
        if(tf!=stdin) fclose(tf);
        if(pt_vec.empty()){ fprintf(stderr,"Empty input token file: %s\n",input_tok_file); return 1; }
        if((int)pt_vec.size() > 4095) pt_vec.resize(4095);
    }else{
        pt_vec={151644,872,198,13048,151645,198,151644,77091,198};
    }
    int npt=(int)pt_vec.size(); if(npt<1)npt=1;
    if(input_tok_file && npt > XM) npt = XM;

    // ===== PREFILL (pipelined: parallel QKV+GU launch, overlapped dequant) =====
    printf("=== Prefill %d ===\n",npt);auto t0=std::chrono::steady_clock::now();fflush(stdout);
    for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]=emb_f32[pt_vec[pi]*H+i];
    xrt::run pending_gu; bool has_pending=false;
    for(int l=0;l<NC;l++){
        fprintf(stderr,"  L%d",l);fflush(stderr);
        // Save pre-norm residuals
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)sb_data[pi*H+i]=h_b[pi*H+i];
        for(int pi=0;pi<npt;pi++)rn_c(&h_b[pi*H],in_n[l].data(),H);
        // Phase 1: Launch QKV on NPU
        float qkv_ascale=dynamic_ascale(h_b.data(),npt*H);
        auto r_qkv=cq.launch_async(l,h_b.data(),npt,H,qkv_ascale);
        // Phase 2: Wait QKV + dequant (CPU attention runs after)
        cq.finish_async(r_qkv,qo_b.data(),npt,qkv_n,qkv_ascale,qsc[l]);cn(qo_b.data(),npt*qkv_n);
        fprintf(stderr,"q");fflush(stderr);
        float*qn=qn_w[l].data(),*kn=kn_w[l].data();
        for(int pi=0;pi<npt;pi++){
            for(int hh=0;hh<NH;hh++){double s=0;for(int d=0;d<HD;d++)s+=qo_b[pi*qkv_n+hh*HD+d]*qo_b[pi*qkv_n+hh*HD+d];float iq=1.0f/sqrtf((float)(s/HD)+EPS);
                for(int d=0;d<HD;d++)qo_b[pi*qkv_n+hh*HD+d]*=iq*(cfg.has_q_norm?qn[d]:1.0f);ra(&qo_b[pi*qkv_n+hh*HD],HD,sp+pi);}
            for(int kvh=0;kvh<NKV;kvh++){float*ks=&qo_b[pi*qkv_n+cfg.qkv_k_offset+kvh*HD],*vs=&qo_b[pi*qkv_n+cfg.qkv_v_offset+kvh*HD];
                double sk=0;for(int d=0;d<HD;d++)sk+=ks[d]*ks[d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
                for(int d=0;d<HD;d++)ks[d]*=ik*(cfg.has_k_norm?kn[d]:1.0f);ra(ks,HD,sp+pi);
                memcpy(&kv_caches[l].k[(sp+pi)*NKV*HD+kvh*HD],ks,HD*4);memcpy(&kv_caches[l].v[(sp+pi)*NKV*HD+kvh*HD],vs,HD*4);}}
        kv_caches[l].n=sp+npt;int cl=kv_caches[l].n;
        // Causal attention: token pi attends only to positions [0, sp+pi]
        for(int pi=0;pi<npt;pi++){fprintf(stderr,"a");fflush(stderr);attn_omp(&qo_b[pi*qkv_n],&at_b[pi*NH*HD],cl,kv_caches[l].k.data(),kv_caches[l].v.data(),NH,NKV,HD,GQA,sp+pi+1);}
        // Phase 3: Launch O + GU in parallel on NPU
        int mlp_out=cfg.gu_split?IM:2*IM;
        float o_ascale=dynamic_ascale(at_b.data(),npt*NH*HD);
        float gu_ascale=dynamic_ascale(h_b.data(),npt*H);
        auto r_o=co.launch_async(l,at_b.data(),npt,NH*HD,o_ascale);
        auto r_gu=cg.launch_async(l,h_b.data(),npt,H,gu_ascale);
        // Phase 4: Wait O, apply residual
        co.finish_async(r_o,oo_b.data(),npt,H,o_ascale,osc[l]);cn(oo_b.data(),npt*H);
        fprintf(stderr,"o");fflush(stderr);
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]=sb_data[pi*H+i]+oo_b[pi*H+i];
        // Save pre-FFN residuals
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)sb_data[pi*H+i]=h_b[pi*H+i];
        for(int pi=0;pi<npt;pi++)rn_c(&h_b[pi*H],pa_n[l].data(),H);
        // Phase 5: Wait GU (was launched in parallel with O), SiLU, launch D
        cg.finish_async(r_gu,gt_b.data(),npt,mlp_out,gu_ascale,gsc[l]);cn(gt_b.data(),npt*mlp_out);
        fprintf(stderr,"g");fflush(stderr);
        if(cfg.gu_split){cu_ptr->go(l,h_b.data(),npt,H,dynamic_ascale(h_b.data(),npt*H),usc[l],su_b.data(),IM);cn(su_b.data(),npt*IM);
            for(int pi=0;pi<npt;pi++){for(int i=0;i<IM;i++){float gv=gt_b[pi*IM+i];if(!std::isfinite(gv))gv=0;su_b[pi*IM+i]=(gv/(1.0f+expf(-gv)))*su_b[pi*IM+i];}}}
        else{for(int pi=0;pi<npt;pi++){for(int i=0;i<IM;i++){float gv=gt_b[pi*mlp_out+i];if(!std::isfinite(gv))gv=0;su_b[pi*IM+i]=(gv/(1.0f+expf(-gv)))*gt_b[pi*mlp_out+IM+i];}}}
        fprintf(stderr,"d");fflush(stderr);cd.go(l,su_b.data(),npt,IM,dynamic_ascale(su_b.data(),npt*IM),dsc[l],dw_b.data(),H);cn(dw_b.data(),npt*H);
        // Residual add: use saved pre-FFN values
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]=sb_data[pi*H+i]+dw_b[pi*H+i];
        fprintf(stderr,"\n");fflush(stderr);
    }sp+=npt;memcpy(h_data.data(),&h_b[(npt-1)*H],H*4);
    printf("Prefill: %.0fms (%.0f ms/tok)\n\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(),std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/npt);

    // ===== v12: M=32 BATCHED DECODE =====
    printf("=== M=%d Batch Decode (%d tokens) ===\n",BS,ng);
    auto tgs=std::chrono::steady_clock::now();
    // NOTE: greedy batched decode — runs batch_size tokens per step, no draft verification.
    // (fixes #95). total_verified tracks all tokens processed.
    int top_ids[BS]={0},total_generated=0,total_verified=0,n_batches=0;double t_boot=0;

    // Boot: single-token decode → top-32 token IDs
    {
        auto ts_boot=std::chrono::steady_clock::now();
        float h0[H];memcpy(h0,h_data.data(),H*4);
        for(int l=0;l<NC;l++){
            memcpy(sb_data.data(),h0,H*4);rn_c(h0,in_n[l].data(),H);
            cq.go(l,h0,1,H,dynamic_ascale(h0,H),qsc[l],qo_data.data(),qkv_n);cn(qo_data.data(),qkv_n);
            memcpy(ko_data.data(),&qo_data[cfg.qkv_k_offset],NKV*HD*4);memcpy(vo_data.data(),&qo_data[cfg.qkv_v_offset],NKV*HD*4);
            float*qn=qn_w[l].data(),*kn=kn_w[l].data();
            for(int hh=0;hh<NH;hh++){double sq=0;for(int d=0;d<HD;d++)sq+=qo_data[hh*HD+d]*qo_data[hh*HD+d];float iq=1.0f/sqrtf((float)(sq/HD)+EPS);
                for(int d=0;d<HD;d++)qo_data[hh*HD+d]*=iq*(cfg.has_q_norm?qn[d]:1.0f);ra(&qo_data[hh*HD],HD,sp);
                if(hh%GQA==0){int kvh=hh/GQA;double sk=0;for(int d=0;d<HD;d++)sk+=ko_data[kvh*HD+d]*ko_data[kvh*HD+d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
                for(int d=0;d<HD;d++)ko_data[kvh*HD+d]*=ik*(cfg.has_k_norm?kn[d]:1.0f);ra(&ko_data[kvh*HD],HD,sp);
                memcpy(&kv_caches[l].k[sp*NKV*HD+kvh*HD],&ko_data[kvh*HD],HD*4);memcpy(&kv_caches[l].v[sp*NKV*HD+kvh*HD],&vo_data[kvh*HD],HD*4);}}
            kv_caches[l].n=sp+1;int cl=kv_caches[l].n;
            attn_omp(qo_data.data(),at_data.data(),cl,kv_caches[l].k.data(),kv_caches[l].v.data(),NH,NKV,HD,GQA);
            co.go(l,at_data.data(),1,NH*HD,dynamic_ascale(at_data.data(),NH*HD),osc[l],oo_data.data(),H);cn(oo_data.data(),H);for(int i=0;i<H;i++)h0[i]=sb_data[i]+oo_data[i];
            memcpy(sb_data.data(),h0,H*4);rn_c(h0,pa_n[l].data(),H);
            int mlp_out=cfg.gu_split?IM:2*IM;
            cg.go(l,h0,1,H,dynamic_ascale(h0,H),gsc[l],gt_data.data(),mlp_out);cn(gt_data.data(),mlp_out);
            if(cfg.gu_split){cu_ptr->go(l,h0,1,H,dynamic_ascale(h0,H),usc[l],su_data.data(),IM);cn(su_data.data(),IM);
                for(int i=0;i<IM;i++){float gv=gt_data[i];if(!std::isfinite(gv))gv=0;su_data[i]=(gv/(1.0f+expf(-gv)))*su_data[i];}}
            else{for(int i=0;i<IM;i++){float gv=gt_data[i];if(!std::isfinite(gv))gv=0;su_data[i]=(gv/(1.0f+expf(-gv)))*gt_data[IM+i];}}
            cd.go(l,su_data.data(),1,IM,dynamic_ascale(su_data.data(),IM),dsc[l],dwo_data.data(),H);cn(dwo_data.data(),H);for(int i=0;i<H;i++)h0[i]=sb_data[i]+dwo_data[i];
        }
        memcpy(sb_data.data(),h0,H*4);rn_c(sb_data.data(),fin_v.data(),H);
        lm_topk_omp(sb_data.data(),lg_buf.data(),top_ids,BS,NV,H,lm_emb);
        memcpy(h_data.data(),h0,H*4);sp++;total_generated++;
        t_boot=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts_boot).count();
        printf("  [0] boot=%d (%.0fms)\n",top_ids[0],t_boot);
    }

    int step=1;
    while(step<ng){
        auto ts_batch=std::chrono::steady_clock::now();
        int batch_size=std::min(BS,ng-step);
        for(int b=0;b<batch_size;b++)for(int i=0;i<H;i++)h_b[b*H+i]=emb_f32[(size_t)top_ids[b]*H+i];
        // ===== PIPELINED LAYER LOOP =====
        // Overlaps CPU quantize with NPU kernel execution.
        // Pattern: launch(N) → quantize(N+1) → wait(N) → dequantize(N) → sync+launch(N+1) → ...
        // co and cg are independent (different inputs) → quantize cg WHILE co runs on NPU.
        for(int l=0;l<NC;l++){
            // Save pre-norm residuals before rn_c
            for(int b=0;b<batch_size;b++)for(int i=0;i<H;i++)sb_data[b*H+i]=h_b[b*H+i];
            for(int b=0;b<batch_size;b++)rn_c(&h_b[b*H],in_n[l].data(),H);

            // ── QKV GEMM ──
            float cq_ascale=dynamic_ascale(h_b.data(),batch_size*H);
            cq.quantize_async(h_b.data(),batch_size,H,cq_ascale);
            auto r_cq=cq.sync_and_launch(l);
            cq.dequantize(r_cq,qo_b.data(),batch_size,qkv_n,cq_ascale,qsc[l]);
            cn(qo_b.data(),batch_size*qkv_n);

            // ── Attention + RoPE + KV cache ──
            float*qn=qn_w[l].data(),*kn=kn_w[l].data();
            for(int b=0;b<batch_size;b++){
                for(int hh=0;hh<NH;hh++){double s=0;for(int d=0;d<HD;d++)s+=qo_b[b*qkv_n+hh*HD+d]*qo_b[b*qkv_n+hh*HD+d];float iq=1.0f/sqrtf((float)(s/HD)+EPS);
                    for(int d=0;d<HD;d++)qo_b[b*qkv_n+hh*HD+d]*=iq*(cfg.has_q_norm?qn[d]:1.0f);ra(&qo_b[b*qkv_n+hh*HD],HD,sp+b);}
                for(int kvh=0;kvh<NKV;kvh++){float*ks=&qo_b[b*qkv_n+cfg.qkv_k_offset+kvh*HD],*vs=&qo_b[b*qkv_n+cfg.qkv_v_offset+kvh*HD];
                    double sk=0;for(int d=0;d<HD;d++)sk+=ks[d]*ks[d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
                    for(int d=0;d<HD;d++)ks[d]*=ik*(cfg.has_k_norm?kn[d]:1.0f);ra(ks,HD,sp+b);}
            }
            for(int b=0;b<batch_size;b++)for(int kvh=0;kvh<NKV;kvh++){
                float*ks=&qo_b[b*qkv_n+cfg.qkv_k_offset+kvh*HD],*vs=&qo_b[b*qkv_n+cfg.qkv_v_offset+kvh*HD];
                memcpy(&kv_caches[l].k[(sp+b)*NKV*HD+kvh*HD],ks,HD*4);memcpy(&kv_caches[l].v[(sp+b)*NKV*HD+kvh*HD],vs,HD*4);}
            kv_caches[l].n=sp+batch_size;int cl=kv_caches[l].n;
            for(int b=0;b<batch_size;b++){attn_omp(&qo_b[b*qkv_n],&at_b[b*NH*HD],cl,kv_caches[l].k.data(),kv_caches[l].v.data(),NH,NKV,HD,GQA);}

            // ── O GEMM ──
            // Launch O, then quantize GU input WHILE O runs (overlapped)
            float co_ascale=dynamic_ascale(at_b.data(),batch_size*NH*HD);
            co.quantize_async(at_b.data(),batch_size,NH*HD,co_ascale);
            auto r_co=co.sync_and_launch(l);

            // ── GU GEMM: independent of O! Quantize GU input while O runs on NPU ──
            int mlp_out=cfg.gu_split?IM:2*IM;
            float cg_ascale=dynamic_ascale(h_b.data(),batch_size*H);
            cg.quantize_async(h_b.data(),batch_size,H,cg_ascale);

            // ── CO-GU FULLY PARALLEL ──
            // Phase 1: Submit GU's DMA sync WHILE O runs on NPU
            //   bA->sync(to_device) uses MM2S DMA channel (independent of NPU compute)
            //   This hides the sync latency behind O's NPU time.
            cg.sync_A(l);  // non-blocking: cg.bA sync starts, DMA runs parallel to NPU

            // Phase 2: Wait for O kernel completion (minimal)
            co.wait_kernel(r_co);

            // Phase 3: Submit GU kernel to NPU + start co readback SIMULTANEOUSLY
            //   cg.launch() submits the kernel (queued behind O on NPU's compute)
            //   co.bC->sync uses S2MM DMA channel (independent of MM2S for cg.bA)
            auto r_cg=cg.launch(l);
            co.sync_back_and_dequant(oo_b.data(),batch_size,H,co_ascale,osc[l]);
            cn(oo_b.data(),batch_size*H);

            // Phase 4: Residual add + rn_c (CPU work, overlaps with cg's NPU execution)
            for(int b=0;b<batch_size;b++)for(int i=0;i<H;i++)h_b[b*H+i]=sb_data[b*H+i]+oo_b[b*H+i];
            for(int b=0;b<batch_size;b++)for(int i=0;i<H;i++)sb_data[b*H+i]=h_b[b*H+i];
            for(int b=0;b<batch_size;b++)rn_c(&h_b[b*H],pa_n[l].data(),H);
            if(cfg.gu_split){
                cu_ptr->quantize_async(h_b.data(),batch_size,H,cg_ascale);
            }

            // Phase 5: Wait for GU, read back, dequant
            cg.wait_kernel(r_cg);
            cg.sync_back_and_dequant(gt_b.data(),batch_size,mlp_out,cg_ascale,gsc[l]);
            cn(gt_b.data(),batch_size*mlp_out);

            // SiLU gate + U GEMM (gu_split) or combined gate*up
            if(cfg.gu_split){
                auto r_cu=cu_ptr->sync_and_launch(l);
                cu_ptr->dequantize(r_cu,su_b.data(),batch_size,IM,cg_ascale,usc[l]);
                cn(su_b.data(),batch_size*IM);
                for(int b=0;b<batch_size;b++){for(int i=0;i<IM;i++){float gv=gt_b[b*IM+i];if(!std::isfinite(gv))gv=0;su_b[b*IM+i]=(gv/(1.0f+expf(-gv)))*su_b[b*IM+i];}}}
            else{for(int b=0;b<batch_size;b++){for(int i=0;i<IM;i++){float gv=gt_b[b*mlp_out+i];if(!std::isfinite(gv))gv=0;su_b[b*IM+i]=(gv/(1.0f+expf(-gv)))*gt_b[b*mlp_out+IM+i];}}}

            // ── D GEMM ──
            float cd_ascale=dynamic_ascale(su_b.data(),batch_size*IM);
            cd.quantize_async(su_b.data(),batch_size,IM,cd_ascale);
            auto r_cd=cd.sync_and_launch(l);
            cd.dequantize(r_cd,dw_b.data(),batch_size,H,cd_ascale,dsc[l]);
            cn(dw_b.data(),batch_size*H);

            // Residual add
            for(int b=0;b<batch_size;b++)for(int i=0;i<H;i++)h_b[b*H+i]=sb_data[b*H+i]+dw_b[b*H+i];
        }

        // LM head on the (single, BS=1) decoded position -> greedy next token.
        // total_verified == total_generated because every emitted token is a
        // real causal decode, not a speculative candidate (issue #111).
        memcpy(sb_data.data(),&h_b[0],H*4);rn_c(sb_data.data(),fin_v.data(),H);
        lm_topk_omp(sb_data.data(),lg_buf.data(),top_ids,BS,NV,H,lm_emb);

        total_generated+=batch_size;total_verified+=batch_size;sp+=batch_size;n_batches++;
        double batch_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts_batch).count();
        printf("  [%d] batch=%d tok=%d %.0fms (%.0f ms/tok)\n",step,batch_size,top_ids[0],batch_ms,batch_ms/batch_size);
        step+=batch_size;
    }

    double tts=std::chrono::duration<double>(std::chrono::steady_clock::now()-tgs).count();
    printf("\n=== %.1f ms/tok (%.0f tok/s) | boot=%.0fms batches=%d tokens=%d ===\n",tts*1000/ng,ng/tts,t_boot,n_batches,total_generated);

    // Graceful exit: the XRT BO destructors (unique_ptr cleanup) can corrupt
    // glibc's heap when GB-scale vectors race with dma-buf teardown.
    // Use _exit() to skip the destructor chain entirely — the OS reclaims
    // all resources on process exit anyway.
    munmap(md,st.st_size);fflush(stdout);fflush(stderr);_exit(0);
}
