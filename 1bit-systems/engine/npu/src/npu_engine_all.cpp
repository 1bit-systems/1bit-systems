/** NPU Engine — ALL Models, v12 Speed. M=32 batch, OpenMP attn+LM, f32 emb.
 *  Auto-detects model from Q4NX header. Works on all 5 model families. */
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
#include "model_config.h"

extern "C" float* dequant_i8_to_float_ex(const uint8_t*,int,int,int*,int*);
// dequant_i8_to_float_ex returns row-major [out_features, in_features]
// (PyTorch nn.Linear layout). packB()/the NPU GEMM need the transpose
// [in_features, out_features] since it computes A[tokens,in] @ B[in,out].
// Ported from npu_engine_universal.cpp to fix issue #109 (this binary used
// to pack QKV/O/Gate-Up/Down without transposing -> scrambled weights).
static void transpose_pack(const float* src,int out_f,int in_f,float* dst,int dst_stride,int dst_offset){
    for(int o=0;o<out_f;o++)
        for(int i=0;i<in_f;i++)
            dst[(size_t)i*dst_stride+dst_offset+o]=src[(size_t)o*in_f+i];
}
static inline float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static inline float bf16g(uint16_t v){return(v&0x7F80)==0x7F80?0.0f:bf16f(v);}
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
static inline void sm(float*sc,int n){if(n<=0)return;cn(sc,n);float mx=sc[0];for(int i=1;i<n;i++)if(sc[i]>mx)mx=sc[i];
    double s=0;for(int i=0;i<n;i++){float d=sc[i]-mx;if(d>80)d=80;else if(d<-80)d=-80;sc[i]=expf(d);s+=sc[i];}
    if(s<=0){float iv=1.0f/n;for(int i=0;i<n;i++)sc[i]=iv;return;}float is=1.0f/(float)s;for(int i=0;i<n;i++)sc[i]*=is;}
static inline void rn_c(float*x,const float*w,int n){cn(x,n);double ss=0;
    for(int i=0;i<n;i++)if(std::isfinite(x[i]))ss+=x[i]*x[i];float ir=1.0f/sqrtf((float)(ss/n)+EPS);
    for(int i=0;i<n;i++)x[i]=std::isfinite(x[i])?x[i]*ir*w[i]:0.0f;}
static std::vector<float>rc,rs;
static void ri(int hd,float th,int mp){int hd2=hd/2;rc.resize(mp*hd);rs.resize(mp*hd);
    for(int p=0;p<mp;p++)for(int d=0;d<hd2;d++){float f=1.0f/powf(th,(float)d/hd2),a=p*f;
        rc[p*hd+d]=cosf(a);rs[p*hd+d]=sinf(a);}}
static inline void ra(float*x,int hd,int p){int hd2=hd/2;for(int d=0;d<hd2;d++){
    float a=x[d],b=x[d+hd2],c=rc[p*hd+d],s=rs[p*hd+d];x[d]=a*c-b*s;x[d+hd2]=b*c+a*s;}}
static std::vector<float> emb_f32;
static uint64_t jo(const char*js,size_t jl,const char*nm){size_t nl=strlen(nm);const char*p=js,*e=js+jl;
    while(p<e){auto q=(const char*)memmem(p,e-p,nm,nl);if(!q)return 0;
        if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){auto o=strstr(q,"\"data_offsets\"");
            if(o){auto a=strchr(o,'[');if(a)return strtoull(a+1,NULL,10);}}p=q+1;}return 0;}

struct I8Ctx{int MD,KD,ND,NL;std::unique_ptr<xrt::xclbin>xc;std::unique_ptr<xrt::hw_context>hc;
    std::unique_ptr<xrt::kernel>k;std::vector<uint32_t>ins;std::unique_ptr<xrt::bo>bI,bA,bC;
    std::vector<std::unique_ptr<xrt::bo>>layerB;int8_t*Am;int16_t*Cm;
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
        return true;}
    void packB(int l,const float*w,int K,int N,float&sout){float amax=0;
        for(int i=0;i<K*N;i++){float a=fabsf(w[i]);if(std::isfinite(a)&&a>amax)amax=a;}
        if(amax<1e-12f)amax=1.0f;sout=amax/127.0f;float is=127.0f/amax;auto*Bm=(int8_t*)layerB[l]->map();
        // Transpose from safe-tensor [N,K] to NPU kernel [K,N] layout (fixes #109)
        for(int k=0;k<K;k++)for(int n=0;n<N;n++){
            float v=w[n*K+k];if(!std::isfinite(v))v=0;int x=(int)roundf(v*is);
            if(x>127)x=127;else if(x<-127)x=-127;Bm[k*N+n]=(int8_t)x;}layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);}
    inline void go(int l,const float*A,int am,int ak,float ascale,float Bscale,float*C,int an){float ais=1.0f/ascale;
        memset(Am,0,(size_t)am*KD);for(int m=0;m<am;m++)for(int k=0;k<ak;k++){
            float v=A[m*ak+k];if(!std::isfinite(v))v=0;int q=(int)roundf(v*ais);
            if(q>127)q=127;else if(q<-127)q=-127;Am[m*KD+k]=(int8_t)q;}bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r=(*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA,*layerB[l],*bC);r.wait();bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        float cs=ascale*Bscale;for(int m=0;m<am;m++)for(int n=0;n<an;n++){
            float val=(float)Cm[m*ND+n]*cs;if(!std::isfinite(val))val=0;C[m*an+n]=val;}}
};

// ── BF16 Attention context (runs attention on NPU, fixes perf) ──
struct AttnCtx{int NH,NKV,HD,block_sz,scratch_sz;
    std::unique_ptr<xrt::xclbin>xc;std::unique_ptr<xrt::hw_context>hc;
    std::unique_ptr<xrt::kernel>k;std::vector<uint32_t>ins;
    std::unique_ptr<xrt::bo>bI,bQ,bK,bV,bO,bS;
    uint8_t*qm,*km,*vm,*om,*sm;
    bool init(xrt::device&d,const char*xp,int nh,int nkv,int hd){
        NH=nh;NKV=nkv;HD=hd;block_sz=2048;scratch_sz=560;
        xc=std::make_unique<xrt::xclbin>(std::string(xp));d.register_xclbin(*xc);
        hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());
        k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");
        // Attention xclbin has instructions embedded; no separate .txt file needed.
        ins.resize(1);ins[0]=0;  // dummy instruction for kernel interface
        bI=std::make_unique<xrt::bo>(d,4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));
        memcpy(bI->map(),ins.data(),4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        size_t qb=(size_t)NH*HD*2;size_t kvb=(size_t)block_sz*NKV*HD*2;size_t scb=(size_t)scratch_sz*4;
        bQ=std::make_unique<xrt::bo>(d,qb,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));
        bK=std::make_unique<xrt::bo>(d,kvb,XRT_BO_FLAGS_HOST_ONLY,k->group_id(4));
        bV=std::make_unique<xrt::bo>(d,kvb,XRT_BO_FLAGS_HOST_ONLY,k->group_id(4));
        bO=std::make_unique<xrt::bo>(d,qb,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));
        bS=std::make_unique<xrt::bo>(d,scb,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));
        qm=(uint8_t*)bQ->map();km=(uint8_t*)bK->map();
        vm=(uint8_t*)bV->map();om=(uint8_t*)bO->map();sm=(uint8_t*)bS->map();
        return true;}
    static inline uint16_t f32b(float f){uint32_t u;memcpy(&u,&f,4);return u>>16;}
    static inline float bf32(uint16_t b){uint32_t u=(uint32_t)b<<16;float f;memcpy(&f,&u,4);return f;}
    void go(const float*Q,const float*Kb,const float*Vb,int bs,float*out){
        auto cp=[&](uint8_t*d,const float*s,int n){for(int i=0;i<n;i++){((uint16_t*)d)[i]=f32b(s[i]);}};
        cp(qm,Q,NH*HD);cp(km,Kb,bs*NKV*HD);cp(vm,Vb,bs*NKV*HD);
        bQ->sync(XCL_BO_SYNC_BO_TO_DEVICE);bK->sync(XCL_BO_SYNC_BO_TO_DEVICE);bV->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r=(*k)((unsigned)3,*bI,(unsigned)ins.size(),*bQ,*bK,*bV,*bO,*bS);r.wait();
        bO->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        for(int i=0;i<NH*HD;i++)out[i]=bf32(((uint16_t*)om)[i]);}
};

// ── Forward declarations ──
static inline void attn_omp(float*qo,float*at,int cl,const float*kv_k,const float*kv_v,
    int NH,int NKV,int HD,int GQA,int max_pos);

// ── NPU attention dispatch with CPU fallback ──
// Uses NPU attention xclbin when available and seq_len fits in one block.
// Falls back to CPU attn_omp for longer sequences or when NPU is unavailable.
static inline void attn_dispatch(AttnCtx*ca,bool ha,float*qo,float*at,int cl,
    const float*kv_k,const float*kv_v,int NH,int NKV,int HD,int GQA){
    if(ha&&ca&&cl>=256&&cl<=ca->block_sz){
        ca->go(qo,kv_k,kv_v,cl,at);
    }else{
        attn_omp(qo,at,cl,kv_k,kv_v,NH,NKV,HD,GQA,cl);
    }
}

static inline void attn_omp(float*qo,float*at,int cl,const float*kv_k,const float*kv_v,
    int NH,int NKV,int HD,int GQA,int max_pos=-1){
    if(max_pos<0)max_pos=cl;
    #pragma omp parallel for
    for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;
        std::vector<float> scores(cl);float mx=-1e30f;
        for(int p=0;p<cl;p++){if(p>=max_pos){scores[p]=-1e30f;continue;}
            double s=0;int qoff=hh*HD,koff=p*NKV*HD+kvh*HD;
            #pragma omp simd reduction(+:s)
            for(int d=0;d<HD;d++)s+=qo[qoff+d]*kv_k[koff+d];scores[p]=(float)(s*0.0883883476);if(scores[p]>mx)mx=scores[p];}
        double sw=0;for(int p=0;p<cl;p++){scores[p]=expf(scores[p]-mx);sw+=scores[p];}
        float isw=sw>0?1.0f/(float)sw:1.0f/cl;
        for(int d=0;d<HD;d++){float acc=0;int aoff=hh*HD+d;
            #pragma omp simd reduction(+:acc)
            for(int p=0;p<cl;p++)acc+=scores[p]*kv_v[p*NKV*HD+kvh*HD+d];at[aoff]=acc*isw;}}
}

inline void lm_topk_omp(const float*hidden,float*lg,int*top_ids,int K,int NV,int H){
    float mx=-1e30f;
    #pragma omp parallel for reduction(max:mx)
    for(int n=0;n<NV;n++){double s=0;const float*e=&emb_f32[(size_t)n*H];const float*h=hidden;
        #pragma omp simd reduction(+:s)
        for(int k=0;k<H;k++)s+=h[k]*e[k];lg[n]=(float)s;if(lg[n]>mx)mx=lg[n];}
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
    if(argc<2){printf("Usage: %s model.q4nx [decode_tokens]\n",argv[0]);return 1;}
    const char*mp=argv[1];int ng=(argc>2)?atoi(argv[2]):32;if(ng<1)ng=1;if(ng>4096)ng=4096; // cap to KV cache size (issue #112/#142)
    std::string mp_s(mp),model_tag,orig_model_name;auto ls=mp_s.rfind('/');auto sl=mp_s.rfind('/',ls-1);
    orig_model_name=(sl!=std::string::npos&&ls!=std::string::npos)?mp_s.substr(sl+1,ls-sl-1):mp_s.substr(ls+1);
    model_tag=orig_model_name;for(auto&c:model_tag){c=tolower(c);if(c=='-'||c=='.')c='_';}
    const char*sfxs[]={"_npu2","_instruct","_it","_it_npu2"};
    for(auto sf:sfxs){size_t l=strlen(sf);if(model_tag.size()>l&&model_tag.substr(model_tag.size()-l)==sf)model_tag=model_tag.substr(0,model_tag.size()-l);}

    ModelConfig cfg=parse_q4nx_header(mp,model_tag.c_str());
    if(!cfg.valid()){printf("ERR: invalid model config\n");return 1;}
    int H=cfg.H,NC=cfg.NC,NH=cfg.NH,NKV=cfg.NKV,HD=cfg.HD,IM=cfg.IM,NV=cfg.NV,GQA=cfg.GQA,XM=cfg.XM;
    printf("=== NPU Engine ALL — %s ===\n",model_tag.c_str());
    printf("H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d NV=%d GU_split=%d\n\n",H,NC,NH,NKV,HD,IM,NV,cfg.gu_split);

    int fd=open(mp,O_RDONLY);struct stat st;fstat(fd,&st);
    uint8_t*md=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    uint64_t hsz;memcpy(&hsz,md,8);uint64_t df=8+hsz;
    auto i8p=[&](uint64_t o){return md+df+o;};auto emb=(const uint16_t*)(md+df);
    const char*js=(const char*)(md+8);size_t jl=hsz;

    printf("Pre-convert emb f32...\n");auto te=std::chrono::steady_clock::now();
    emb_f32.resize((size_t)NV*H);
    for(int n=0;n<NV;n++)for(int i=0;i<H;i++)emb_f32[(size_t)n*H+i]=bf16g(emb[n*H+i]);
    printf("  %.0fms\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-te).count());

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

    std::vector<float> in_n(NC*H),pa_n(NC*H),qn_w(NC*HD),kn_w(NC*HD),fin_v(H);
    for(int l=0;l<NC;l++){auto iw=(const uint16_t*)(md+df+in_off[l]),pw=(const uint16_t*)(md+df+pa_off[l]);
        for(int i=0;i<H;i++){in_n[l*H+i]=bf16g(iw[i]);pa_n[l*H+i]=bf16g(pw[i]);}
        if(cfg.has_q_norm&&qn_off[l]){auto qq=(const uint16_t*)(md+df+qn_off[l]);for(int i=0;i<HD;i++)qn_w[l*HD+i]=bf16g(qq[i]);}
        if(cfg.has_k_norm&&kn_off[l]){auto kk=(const uint16_t*)(md+df+kn_off[l]);for(int i=0;i<HD;i++)kn_w[l*HD+i]=bf16g(kk[i]);}}
    {auto fw=(const uint16_t*)(md+df+no);for(int i=0;i<H;i++)fin_v[i]=bf16g(fw[i]);}

    auto gi8=[&](const char*k)->int{int r=0;find_tensor_info(js,jl,k,&r);return r;};
    int q_i8=gi8("model.layers.0.self_attn.q_proj.weight"),k_i8=gi8("model.layers.0.self_attn.k_proj.weight"),v_i8=gi8("model.layers.0.self_attn.v_proj.weight");
    int o_i8=gi8("model.layers.0.self_attn.o_proj.weight"),g_i8=gi8("model.layers.0.mlp.gate_proj.weight"),u_i8=gi8("model.layers.0.mlp.up_proj.weight"),d_i8=gi8("model.layers.0.mlp.down_proj.weight");

    printf("Init NPU...\n");xrt::device dev(0);
    const char* d=getenv("NPU_XCLBIN_DIR"); std::string xd=d?d:"int8";
    auto xp=[&](const char*t){return xd+"/final_i8_"+t+"_"+cfg.model_tag+".xclbin";};
    auto ip=[&](const char*t){return xd+"/insts_i8_"+t+"_"+cfg.model_tag+".txt";};

    I8Ctx cq,co,cg,cd;cq.MD=XM;cq.KD=cfg.xclbin_qkv_k;cq.ND=cfg.xclbin_qkv_n;
    co.MD=XM;co.KD=cfg.xclbin_o_k;co.ND=cfg.xclbin_o_n;
    cd.MD=XM;cd.KD=cfg.xclbin_d_k;cd.ND=cfg.xclbin_d_n;
    if(cfg.gu_split){cg.MD=XM;cg.KD=cfg.xclbin_g_k;cg.ND=cfg.xclbin_g_n;}else{cg.MD=XM;cg.KD=cfg.xclbin_gu_k;cg.ND=cfg.xclbin_gu_n;}
    if(!cq.init(dev,xp("QKV").c_str(),ip("QKV").c_str(),4,NC)){printf("FAIL QKV: %s\n",xp("QKV").c_str());return 1;}
    if(!co.init(dev,xp("O").c_str(),ip("O").c_str(),4,NC)){printf("FAIL O\n");return 1;}
    if(cfg.gu_split){if(!cg.init(dev,xp("G").c_str(),ip("G").c_str(),4,NC)){printf("FAIL G\n");return 1;}}else{if(!cg.init(dev,xp("GU").c_str(),ip("GU").c_str(),4,NC)){printf("FAIL GU\n");return 1;}}
    if(!cd.init(dev,xp("D").c_str(),ip("D").c_str(),4,NC)){printf("FAIL D\n");return 1;}
    std::unique_ptr<I8Ctx> cu_ptr;

    // ── NPU Attention (optional, replaces CPU attn_omp) ──
    const char* ffx=getenv("FFLM_XCLBIN_DIR"); std::string attn_xd=ffx?ffx:"fastflowlm-build/src/xclbins/"; attn_xd+=orig_model_name+"/attn.xclbin";
    std::string attn_inst=std::string(d?d:"int8")+"/../chess_infer/attn_06b.o";
    AttnCtx ca;bool have_attn=false;
    if(ca.init(dev,attn_xd.c_str(),NH,NKV,HD)){
        printf("  NPU Attention: available (block_sz=%d) — use for seq_len>256\n",ca.block_sz);
    }else{printf("  NPU Attention: unavailable, using CPU fallback\n");}
    if(cfg.gu_split){cu_ptr=std::make_unique<I8Ctx>();cu_ptr->MD=XM;cu_ptr->KD=cfg.xclbin_u_k;cu_ptr->ND=cfg.xclbin_u_n;
        if(!cu_ptr->init(dev,xp("U").c_str(),ip("U").c_str(),4,NC)){printf("FAIL U\n");return 1;}}

    printf("Dequant+pack...\n");auto tp=std::chrono::steady_clock::now();
    auto dq=[&](uint64_t off,int i8r,int in_f,int*or_,int*oc){
        return dequant_i8_to_float_ex(i8p(off),i8r,in_f,or_,oc);};
    int o_in_f=NH*HD;
    std::vector<float> qsc(NC),osc(NC),gsc(NC),dsc(NC),usc(NC);
    for(int l=0;l<NC;l++){int qr,kr,vr,or_,gr,ur,dr,unused;
        // QKV: dequant returns [out, in=H]; transpose to [H, out] and concat.
        float*qw=dq(qp[l],q_i8,H,&qr,&unused);
        float*kw=dq(kp[l],k_i8,H,&kr,&unused);
        float*vw=dq(vp[l],v_i8,H,&vr,&unused);
        int t=qr+kr+vr;std::vector<float>w((size_t)H*t);
        // QKV pack with proper transpose (fixes #109)
        transpose_pack(qw,qr,H,w.data(),t,0);
        transpose_pack(kw,kr,H,w.data(),t,qr);
        transpose_pack(vw,vr,H,w.data(),t,qr+kr);
        cq.packB(l,w.data(),H,t,qsc[l]);free(qw);free(kw);free(vw);
        // O-proj: dequant [out, in=o_in_f]; transpose to [o_in_f, out].
        float*ow=dq(op[l],o_i8,o_in_f,&or_,&unused);
        std::vector<float>wo((size_t)o_in_f*or_);transpose_pack(ow,or_,o_in_f,wo.data(),or_,0);
        co.packB(l,wo.data(),o_in_f,or_,osc[l]);free(ow);
        // Gate/Up: dequant [out, in=H]; transpose to [H, out].
        float*gw=dq(gp[l],g_i8,H,&gr,&unused);
        if(cfg.gu_split){float*uw=dq(up[l],u_i8,H,&ur,&unused);
            std::vector<float>wg((size_t)H*gr);transpose_pack(gw,gr,H,wg.data(),gr,0);
            cg.packB(l,wg.data(),H,gr,gsc[l]);
            std::vector<float>wu((size_t)H*ur);transpose_pack(uw,ur,H,wu.data(),ur,0);
            cu_ptr->packB(l,wu.data(),H,ur,usc[l]);free(uw);}
        else{float*uw=dq(up[l],u_i8,H,&ur,&unused);
            int t2=gr+ur;std::vector<float>w2((size_t)H*t2);
            transpose_pack(gw,gr,H,w2.data(),t2,0);
            transpose_pack(uw,ur,H,w2.data(),t2,gr);
            cg.packB(l,w2.data(),H,t2,gsc[l]);free(uw);}free(gw);
        // Down-proj: dequant [out, in=IM]; transpose to [IM, out].
        float*dw=dq(dp[l],d_i8,IM,&dr,&unused);
        std::vector<float>wd((size_t)IM*dr);transpose_pack(dw,dr,IM,wd.data(),dr,0);
        cd.packB(l,wd.data(),IM,dr,dsc[l]);free(dw);}
    printf("  %.0fms\n\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tp).count());
    ri(HD,cfg.rope_theta,4096);

    // Decode batch width. Pinned to 1 (correct causal single-token greedy).
    // The old M=32 "batch decode" here has the same defect as
    // npu_engine_universal.cpp (issue #111): it embeds the 32 top-K candidates
    // for one position as 32 sequential tokens and attends over them
    // non-causally, corrupting output. Do not raise without real draft-verify.
    int BS=1,qkv_n=cfg.qkv_total;
    struct KVCache{std::vector<float>k,v;int n;KVCache(int s):k(s),v(s),n(0){}};
    int kv_sz=4096*NKV*HD;std::vector<KVCache> kv_c;for(int i=0;i<NC;i++)kv_c.emplace_back(kv_sz);
    std::vector<float> h_b(XM*H),qo_b(XM*qkv_n),at_b(XM*NH*HD),oo_b(XM*H);
    std::vector<float> gt_b(XM*(cfg.gu_split?IM:2*IM)),su_b(XM*IM),dw_b(XM*H);
    std::vector<float> h_data(H),qo_d(qkv_n*BS),ko_d(NKV*HD*BS),vo_d(NKV*HD*BS),at_d(NH*HD*BS),oo_d(H*BS);
    std::vector<float> gt_d((cfg.gu_split?IM:2*IM)*BS),su_d(IM*BS),dwo_d(H*BS),sb_d(H*BS),lg_b(NV);
    int sp=0,npt=9,pt[]={151643,872,198,11852,151644,198,151643,77091,198};

    // Prefill
    printf("=== Prefill %d ===\n",npt);auto t0=std::chrono::steady_clock::now();
    for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]=emb_f32[pt[pi]*H+i];
    for(int l=0;l<NC;l++){
        for(int pi=0;pi<npt;pi++)rn_c(&h_b[pi*H],&in_n[l*H],H);
        cq.go(l,h_b.data(),npt,H,dynamic_ascale(h_b.data(),npt*H),qsc[l],qo_b.data(),qkv_n);cn(qo_b.data(),npt*qkv_n);
        float*qn=&qn_w[l*HD],*kn=&kn_w[l*HD];
        for(int pi=0;pi<npt;pi++){
            for(int hh=0;hh<NH;hh++){double s=0;for(int d=0;d<HD;d++)s+=qo_b[pi*qkv_n+hh*HD+d]*qo_b[pi*qkv_n+hh*HD+d];
                float iq=1.0f/sqrtf((float)(s/HD)+EPS);for(int d=0;d<HD;d++)qo_b[pi*qkv_n+hh*HD+d]*=iq*(cfg.has_q_norm?qn[d]:1.0f);
                ra(&qo_b[pi*qkv_n+hh*HD],HD,sp+pi);}
            for(int kvh=0;kvh<NKV;kvh++){float*ks=&qo_b[pi*qkv_n+cfg.qkv_k_offset+kvh*HD],*vs=&qo_b[pi*qkv_n+cfg.qkv_v_offset+kvh*HD];
                double sk=0;for(int d=0;d<HD;d++)sk+=ks[d]*ks[d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
                for(int d=0;d<HD;d++){ks[d]*=ik*(cfg.has_k_norm?kn[d]:1.0f);ra(ks,HD,sp+pi);}
                memcpy(&kv_c[l].k[(sp+pi)*NKV*HD+kvh*HD],ks,HD*4);memcpy(&kv_c[l].v[(sp+pi)*NKV*HD+kvh*HD],vs,HD*4);}}
        kv_c[l].n=sp+npt;int cl=kv_c[l].n;
        for(int pi=0;pi<npt;pi++){attn_omp(&qo_b[pi*qkv_n],&at_b[pi*NH*HD],cl,kv_c[l].k.data(),kv_c[l].v.data(),NH,NKV,HD,GQA,sp+pi+1);}
        co.go(l,at_b.data(),npt,NH*HD,dynamic_ascale(at_b.data(),npt*NH*HD),osc[l],oo_b.data(),H);cn(oo_b.data(),npt*H);
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]+=oo_b[pi*H+i];
        for(int pi=0;pi<npt;pi++)rn_c(&h_b[pi*H],&pa_n[l*H],H);
        int mlp_o=cfg.gu_split?IM:2*IM;
        cg.go(l,h_b.data(),npt,H,dynamic_ascale(h_b.data(),npt*H),gsc[l],gt_b.data(),mlp_o);cn(gt_b.data(),npt*mlp_o);
        if(cfg.gu_split){cu_ptr->go(l,h_b.data(),npt,H,dynamic_ascale(h_b.data(),npt*H),usc[l],su_b.data(),IM);cn(su_b.data(),npt*IM);
            for(int pi=0;pi<npt;pi++){for(int i=0;i<IM;i++){float gv=gt_b[pi*IM+i];if(!std::isfinite(gv))gv=0;su_b[pi*IM+i]=(gv/(1.0f+expf(-gv)))*su_b[pi*IM+i];}}}
        else{for(int pi=0;pi<npt;pi++){for(int i=0;i<IM;i++){float gv=gt_b[pi*mlp_o+i];if(!std::isfinite(gv))gv=0;su_b[pi*IM+i]=(gv/(1.0f+expf(-gv)))*gt_b[pi*mlp_o+IM+i];}}}
        cd.go(l,su_b.data(),npt,IM,dynamic_ascale(su_b.data(),npt*IM),dsc[l],dw_b.data(),H);cn(dw_b.data(),npt*H);
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]+=dw_b[pi*H+i];
    }sp+=npt;memcpy(h_data.data(),&h_b[(npt-1)*H],H*4);
    printf("Prefill: %.0fms (%.0f ms/tok)\n\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(),std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/npt);

    // v12 M=32 batched decode
    printf("=== M=%d Batch Decode (%d tokens) ===\n",BS,ng);
    auto tgs=std::chrono::steady_clock::now();
    int top_ids[BS]={0},total_acc=0,n_bat=0;double t_boot=0;

    // Boot
    {auto ts_b=std::chrono::steady_clock::now();float h0[H];memcpy(h0,h_data.data(),H*4);
    for(int l=0;l<NC;l++){
        memcpy(sb_d.data(),h0,H*4);rn_c(h0,&in_n[l*H],H);
        cq.go(l,h0,1,H,dynamic_ascale(h0,H),qsc[l],qo_d.data(),qkv_n);cn(qo_d.data(),qkv_n);
        memcpy(ko_d.data(),&qo_d[cfg.qkv_k_offset],NKV*HD*4);memcpy(vo_d.data(),&qo_d[cfg.qkv_v_offset],NKV*HD*4);
        float*qn=&qn_w[l*HD],*kn=&kn_w[l*HD];
        for(int hh=0;hh<NH;hh++){double sq=0;for(int d=0;d<HD;d++)sq+=qo_d[hh*HD+d]*qo_d[hh*HD+d];float iq=1.0f/sqrtf((float)(sq/HD)+EPS);
            for(int d=0;d<HD;d++)qo_d[hh*HD+d]*=iq*(cfg.has_q_norm?qn[d]:1.0f);ra(&qo_d[hh*HD],HD,sp);
            if(hh%GQA==0){int kvh=hh/GQA;double sk=0;for(int d=0;d<HD;d++)sk+=ko_d[kvh*HD+d]*ko_d[kvh*HD+d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
            for(int d=0;d<HD;d++)ko_d[kvh*HD+d]*=ik*(cfg.has_k_norm?kn[d]:1.0f);ra(&ko_d[kvh*HD],HD,sp);
            memcpy(&kv_c[l].k[sp*NKV*HD+kvh*HD],&ko_d[kvh*HD],HD*4);memcpy(&kv_c[l].v[sp*NKV*HD+kvh*HD],&vo_d[kvh*HD],HD*4);}}
        kv_c[l].n=sp+1;int cl=kv_c[l].n;
        attn_dispatch(&ca,have_attn,qo_d.data(),at_d.data(),cl,kv_c[l].k.data(),kv_c[l].v.data(),NH,NKV,HD,GQA);
        co.go(l,at_d.data(),1,NH*HD,dynamic_ascale(at_d.data(),NH*HD),osc[l],oo_d.data(),H);cn(oo_d.data(),H);for(int i=0;i<H;i++)h0[i]=sb_d[i]+oo_d[i];
        memcpy(sb_d.data(),h0,H*4);rn_c(h0,&pa_n[l*H],H);
        int mlp_o=cfg.gu_split?IM:2*IM;
        cg.go(l,h0,1,H,dynamic_ascale(h0,H),gsc[l],gt_d.data(),mlp_o);cn(gt_d.data(),mlp_o);
        if(cfg.gu_split){cu_ptr->go(l,h0,1,H,dynamic_ascale(h0,H),usc[l],su_d.data(),IM);cn(su_d.data(),IM);
            for(int i=0;i<IM;i++){float gv=gt_d[i];if(!std::isfinite(gv))gv=0;su_d[i]=(gv/(1.0f+expf(-gv)))*su_d[i];}}
        else{for(int i=0;i<IM;i++){float gv=gt_d[i];if(!std::isfinite(gv))gv=0;su_d[i]=(gv/(1.0f+expf(-gv)))*gt_d[IM+i];}}
        cd.go(l,su_d.data(),1,IM,dynamic_ascale(su_d.data(),IM),dsc[l],dwo_d.data(),H);cn(dwo_d.data(),H);for(int i=0;i<H;i++)h0[i]=sb_d[i]+dwo_d[i];
    }
    memcpy(sb_d.data(),h0,H*4);rn_c(sb_d.data(),fin_v.data(),H);
    lm_topk_omp(sb_d.data(),lg_b.data(),top_ids,BS,NV,H);
    memcpy(h_data.data(),h0,H*4);sp++;total_acc++;
    t_boot=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts_b).count();
    printf("  [0] boot=%d (%.0fms)\n",top_ids[0],t_boot);}

    int step=1;
    while(step<ng){
        auto ts_b=std::chrono::steady_clock::now();int bs=std::min(BS,ng-step);
        for(int b=0;b<bs;b++)for(int i=0;i<H;i++)h_b[b*H+i]=emb_f32[(size_t)top_ids[b]*H+i];
        for(int l=0;l<NC;l++){
            for(int b=0;b<bs;b++)rn_c(&h_b[b*H],&in_n[l*H],H);
            cq.go(l,h_b.data(),bs,H,dynamic_ascale(h_b.data(),bs*H),qsc[l],qo_b.data(),qkv_n);cn(qo_b.data(),bs*qkv_n);
            float*qn=&qn_w[l*HD],*kn=&kn_w[l*HD];
            for(int b=0;b<bs;b++){
                for(int hh=0;hh<NH;hh++){double s=0;for(int d=0;d<HD;d++)s+=qo_b[b*qkv_n+hh*HD+d]*qo_b[b*qkv_n+hh*HD+d];
                    float iq=1.0f/sqrtf((float)(s/HD)+EPS);for(int d=0;d<HD;d++)qo_b[b*qkv_n+hh*HD+d]*=iq*(cfg.has_q_norm?qn[d]:1.0f);
                    ra(&qo_b[b*qkv_n+hh*HD],HD,sp+b);}
                for(int kvh=0;kvh<NKV;kvh++){float*ks=&qo_b[b*qkv_n+cfg.qkv_k_offset+kvh*HD],*vs=&qo_b[b*qkv_n+cfg.qkv_v_offset+kvh*HD];
                    double sk=0;for(int d=0;d<HD;d++)sk+=ks[d]*ks[d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
                    for(int d=0;d<HD;d++){ks[d]*=ik*(cfg.has_k_norm?kn[d]:1.0f);ra(ks,HD,sp+b);}}}
            for(int b=0;b<bs;b++)for(int kvh=0;kvh<NKV;kvh++){
                float*ks=&qo_b[b*qkv_n+cfg.qkv_k_offset+kvh*HD],*vs=&qo_b[b*qkv_n+cfg.qkv_v_offset+kvh*HD];
                memcpy(&kv_c[l].k[(sp+b)*NKV*HD+kvh*HD],ks,HD*4);memcpy(&kv_c[l].v[(sp+b)*NKV*HD+kvh*HD],vs,HD*4);}
            kv_c[l].n=sp+bs;int cl=kv_c[l].n;
            for(int b=0;b<bs;b++){attn_dispatch(&ca,have_attn,&qo_b[b*qkv_n],&at_b[b*NH*HD],cl,kv_c[l].k.data(),kv_c[l].v.data(),NH,NKV,HD,GQA);}
            co.go(l,at_b.data(),bs,NH*HD,dynamic_ascale(at_b.data(),bs*NH*HD),osc[l],oo_b.data(),H);cn(oo_b.data(),bs*H);
            for(int b=0;b<bs;b++)for(int i=0;i<H;i++)h_b[b*H+i]+=oo_b[b*H+i];
            for(int b=0;b<bs;b++)rn_c(&h_b[b*H],&pa_n[l*H],H);
            int mlp_o=cfg.gu_split?IM:2*IM;
            cg.go(l,h_b.data(),bs,H,dynamic_ascale(h_b.data(),bs*H),gsc[l],gt_b.data(),mlp_o);cn(gt_b.data(),bs*mlp_o);
            if(cfg.gu_split){cu_ptr->go(l,h_b.data(),bs,H,dynamic_ascale(h_b.data(),bs*H),usc[l],su_b.data(),IM);cn(su_b.data(),bs*IM);
                for(int b=0;b<bs;b++){for(int i=0;i<IM;i++){float gv=gt_b[b*IM+i];if(!std::isfinite(gv))gv=0;su_b[b*IM+i]=(gv/(1.0f+expf(-gv)))*su_b[b*IM+i];}}}
            else{for(int b=0;b<bs;b++){for(int i=0;i<IM;i++){float gv=gt_b[b*mlp_o+i];if(!std::isfinite(gv))gv=0;su_b[b*IM+i]=(gv/(1.0f+expf(-gv)))*gt_b[b*mlp_o+IM+i];}}
            cd.go(l,su_b.data(),bs,IM,dynamic_ascale(su_b.data(),bs*IM),dsc[l],dw_b.data(),H);cn(dw_b.data(),bs*H);
            for(int b=0;b<bs;b++)for(int i=0;i<H;i++)h_b[b*H+i]+=dw_b[b*H+i];
        }}
        memcpy(sb_d.data(),&h_b[0],H*4);rn_c(sb_d.data(),fin_v.data(),H);
        lm_topk_omp(sb_d.data(),lg_b.data(),top_ids,BS,NV,H);
        total_acc+=bs;sp+=bs;n_bat++;
        double bms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts_b).count();
        printf("  [%d] batch=%d tok=%d %.0fms (%.0f ms/tok)\n",step,bs,top_ids[0],bms,bms/bs);
        step+=bs;
    }
    double tts=std::chrono::duration<double>(std::chrono::steady_clock::now()-tgs).count();
    printf("\n=== %.1f ms/tok (%.0f tok/s) | boot=%.0fms batches=%d ===\n",tts*1000/ng,ng/tts,t_boot,n_bat);
    // Use _exit() to skip the C++ destructor chain (vectors ~896MB KV cache,
    // XRT BO dma-buf teardown). XRT's BO destructors can corrupt glibc's heap
    // by racing dma-buf release with vector heap free() during normal exit().
    // The OS reclaims all resources on process exit regardless.
    munmap(md,st.st_size);fflush(stdout);fflush(stderr);_exit(0);
}
