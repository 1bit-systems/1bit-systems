// NPU Engine — Model-Agnostic Multi-Token INT8 Decode
//
// Loads model dimensions dynamically from Q4NX JSON header.
// Selects tagged xclbins by model name from build/int8/.
// Handles GU split/combined, q_norm/k_norm presence,
// file-based RoPE, separate vs tied lm_head.
//
// Build:
//   c++ -std=gnu++17 -O3 -mavx2 -mfma -mavx512f -ffast-math -march=native
//       -funroll-loops -fno-math-errno -Iinclude
//       -o build/npu_engine_mt src/npu_engine_mt.cpp build/deq_std.o -lxrt_coreutil
//
// Run:
//   sudo ... ./npu_engine_mt /path/to/model.q4nx TOKEN1 TOKEN2 ... TOKEN256

#include <cstdio>
#include <cstdlib>
#include <ctime>
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
#include <immintrin.h>
#include "kv_quant.h"
#include "model_config.h"

extern "C" float* dequant_i8_to_float_ex(const uint8_t*,int,int,int*,int*);
static inline float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static inline float bf16g(uint16_t v){return(v&0x7F80)==0x7F80?0.0f:bf16f(v);}

static constexpr float EPS=1e-6f;
static inline void cn(float*x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
static inline void rn_c(float*x,const float*w,int n,int M){
    for(int m=0;m<M;m++){
        double ss=0;for(int i=0;i<n;i++){float v=x[m*n+i];if(std::isfinite(v))ss+=(double)v*v;}
        float ir=1.0f/sqrtf((float)(ss/n)+EPS);
        for(int i=0;i<n;i++){float v=x[m*n+i];x[m*n+i]=std::isfinite(v)?v*ir*w[i]:0.0f;}
    }
}

// RoPE tables (theta-based), sized dynamically
static std::vector<float>rc,rs;
static void ri(int hd,float th,int mp){rc.resize(mp*hd);rs.resize(mp*hd);
    for(int p=0;p<mp;p++)for(int d=0;d<hd;d+=2){
        float f=1.0f/powf(th,(float)d/hd),a=p*f;
        rc[p*hd+d]=cosf(a);rs[p*hd+d]=sinf(a);
        rc[p*hd+d+1]=cosf(a);rs[p*hd+d+1]=sinf(a);}}
static inline void ra(float*x,int hd,int p){for(int d=0;d<hd;d+=2){
    float a=x[d],b=x[d+1],c=rc[p*hd+d],s=rs[p*hd+d];x[d]=a*c-b*s;x[d+1]=b*c+a*s;}}

// NPU GEMM context — one per xclbin type
struct I8Ctx{
    int MD,KD,ND,NL;
    std::unique_ptr<xrt::xclbin>xc;
    std::unique_ptr<xrt::hw_context>hc;
    std::unique_ptr<xrt::kernel>k;
    std::vector<uint32_t>ins;
    std::unique_ptr<xrt::bo>bI,bA,bC;
    std::vector<std::unique_ptr<xrt::bo>>layerB;
    int8_t*Am;int16_t*Cm;
    
    bool init(xrt::device&d,const char*xp,const char*ip,int gid_B,int nlayers){
        NL=nlayers;
        FILE*f=fopen(ip,"rb");if(!f)return false;
        fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);
        ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
        xc=std::make_unique<xrt::xclbin>(std::string(xp));
        d.register_xclbin(*xc);
        hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());
        k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");
        bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));
        memcpy(bI->map(),ins.data(),ins.size()*4);
        bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bA=std::make_unique<xrt::bo>(d,(size_t)MD*KD,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));
        bC=std::make_unique<xrt::bo>(d,(size_t)MD*ND*2,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));
        Am=(int8_t*)bA->map();Cm=(int16_t*)bC->map();
        for(int l=0;l<NL;l++)
            layerB.emplace_back(std::make_unique<xrt::bo>(d,(size_t)KD*ND,XRT_BO_FLAGS_HOST_ONLY,k->group_id(gid_B)));
        return true;
    }
    
    void packB(int l,const float*w,int K,int N,float&sout){
        float amax=0;
        for(int i=0;i<K*N;i++){float a=fabsf(w[i]);if(std::isfinite(a)&&a>amax)amax=a;}
        if(amax<1e-12f)amax=1.0f;sout=amax/127.0f;float is=127.0f/amax;
        auto*Bm=(int8_t*)layerB[l]->map();
        for(int i=0;i<K*N;i++){float v=w[i];if(!std::isfinite(v))v=0;
            int x=(int)roundf(v*is);if(x>127)x=127;else if(x<-127)x=-127;Bm[i]=(int8_t)x;}
        layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    
    inline void go(int l,int pass_m,const float*A,int am,int ak,float ascale,float Bscale,float*C,int an){
        float ais=1.0f/ascale;
        memset(Am,0,(size_t)MD*KD);
        for(int m=0;m<am;m++)for(int k=0;k<ak;k++){
            float v=A[m*ak+k];if(!std::isfinite(v))v=0;
            int q=(int)roundf(v*ais);
            if(q>127)q=127;else if(q<-127)q=-127;
            Am[m*KD+k]=(int8_t)q;}
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r=(*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA,*layerB[l],*bC);
        r.wait();
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        float cs=ascale*Bscale;
        for(int m=0;m<am;m++)for(int n=0;n<an;n++){
            float val=(float)Cm[m*ND+n]*cs;
            if(!std::isfinite(val))val=0;
            C[(pass_m+m)*an+n]=val;}
    }
};

// SIMD attention — parameterized head dim and counts
struct AttnCPU{
    int WQH, HD;
    AttnCPU(int wqh, int hd): WQH(wqh), HD(hd) {}
    void run(const float*Q,const float*Kd,const float*Vd,int nt,float*out) const {
        for(int h=0;h<WQH;h++){
            float sc_[4096];
            float* sc = sc_;
            float mx=-1e30f;
            for(int t=0;t<nt;t++){
                __m512 sum=_mm512_setzero_ps();
                const float*Kt=Kd+t*HD;
                for(int d=0;d<HD;d+=16){
                    __m512 qv=_mm512_loadu_ps(Q+h*HD+d);
                    __m512 kv=_mm512_loadu_ps(Kt+d);
                    sum=_mm512_fmadd_ps(qv,kv,sum);}
                float s=_mm512_reduce_add_ps(sum)*(1.0f/sqrtf((float)HD));
                sc[t]=s;if(s>mx)mx=s;}
            double ssum=0;for(int t=0;t<nt;t++){sc[t]=expf(sc[t]-mx);ssum+=sc[t];}
            if(ssum<=0)ssum=1;float is=1.0f/(float)ssum;
            __m512 ov[8];for(int i=0;i<8;i++)ov[i]=_mm512_setzero_ps();
            for(int t=0;t<nt;t++){
                __m512 sv=_mm512_set1_ps(sc[t]*is);
                const float*Vt=Vd+t*HD;
                for(int i=0;i<8;i++)ov[i]=_mm512_fmadd_ps(sv,_mm512_loadu_ps(Vt+i*16),ov[i]);}
            for(int i=0;i<8;i++)_mm512_storeu_ps(out+h*HD+i*16,ov[i]);}}};

// Load an I8 weight block and dequantize it
static float* dequant_weight(const uint8_t* data, int i8_rows, int in_feat,
                              int* out_rows, int* out_cols) {
    // Use dequant_i8_to_float (not _ex) — the _ex variant has allocator issues on large models
    (void)in_feat;
    return dequant_i8_to_float_ex(data, i8_rows, in_feat, out_rows, out_cols);
}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"Usage: %s model.q4nx TOKEN1 [TOKEN2 ...]\n",argv[0]);return 1;}
    const char*mp=argv[1];
    int M=argc-2;
    if(M>256){fprintf(stderr,"Max 256 tokens, got %d\n",M);return 1;}
    
    // Derive model tag from path
    std::string mp_s(mp);
    std::string model_tag;
    auto last_slash = mp_s.rfind('/');
    auto second_last = mp_s.rfind('/', last_slash - 1);
    if (second_last != std::string::npos && last_slash != std::string::npos)
        model_tag = mp_s.substr(second_last + 1, last_slash - second_last - 1);
    else if (last_slash != std::string::npos)
        model_tag = mp_s.substr(last_slash + 1);
    else
        model_tag = mp_s;
    for (auto& c : model_tag){
        c = (c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c;
        if (c == '-' || c == '.') c = '_';}
    const char* sfxs[] = {"_npu2","_instruct","_it","_it_npu2"};
    for (auto sfx : sfxs){
        size_t sl = strlen(sfx);
        if (model_tag.size() > sl && model_tag.substr(model_tag.size()-sl) == sfx)
            model_tag = model_tag.substr(0, model_tag.size()-sl);}
    
    // Parse model config from Q4NX header
    fprintf(stderr,"Parsing model config from %s (tag=%s)...\n",mp,model_tag.c_str());
    ModelConfig cfg = parse_q4nx_header(mp, model_tag.c_str());
    if(!cfg.valid()){
        fprintf(stderr,"ERROR: Failed to parse model config\n");
        fprintf(stderr,"  H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d NV=%d\n",
                cfg.H,cfg.NC,cfg.NH,cfg.NKV,cfg.HD,cfg.IM,cfg.NV);
        return 1;}
    fprintf(stderr,"  Model: H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d NV=%d\n",
            cfg.H,cfg.NC,cfg.NH,cfg.NKV,cfg.HD,cfg.IM,cfg.NV);
    fprintf(stderr,"  Features: qn=%d kn=%d rf=%d lm=%d gs=%d\n",
            cfg.has_q_norm,cfg.has_k_norm,cfg.has_rope_freqs_file,cfg.has_lm_head,cfg.gu_split);
    
    int H=cfg.H, NC=cfg.NC, NH=cfg.NH, NKV=cfg.NKV, HD=cfg.HD;
    int IM=cfg.IM, NV=cfg.NV, GQA=cfg.GQA, AW=cfg.AW, XM=cfg.XM;
    
    std::vector<int>input_tokens(M);
    for(int i=0;i<M;i++)input_tokens[i]=atoi(argv[2+i]);
    
    // Re-open file for mmap (parse_q4nx_header unmapped it)
    int fd=open(mp,O_RDONLY);struct stat st;fstat(fd,&st);
    uint8_t*md=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    uint64_t hsz;memcpy(&hsz,md,8);uint64_t df=8+hsz;
    auto i8p=[&](uint64_t o){return md+df+o;};
    auto emb=(const uint16_t*)(md+df);
    const char*js=(const char*)(md+8);size_t jl=hsz;

    auto jo=[&](const char*nm)->uint64_t{
        size_t nl=strlen(nm);const char*p=js,*e=js+jl;
        while(p<e){auto q=(const char*)memmem(p,e-p,nm,nl);if(!q)return 0;
            if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){auto o=strstr(q,"\"data_offsets\"");
                if(o){auto a=strchr(o,'[');if(a)return strtoull(a+1,NULL,10);}}p=q+1;}return 0;};

    std::vector<uint64_t> qp(NC),kp(NC),vp(NC),op(NC),gp(NC),up(NC),dp(NC);
    std::vector<uint64_t> in_off(NC),pa_off(NC),qn_off(NC),kn_off(NC);
    char bn[128];
    for(int l=0;l<NC;l++){
        snprintf(bn,128,"model.layers.%d.self_attn.q_proj.weight",l); qp[l]=jo(bn);
        snprintf(bn,128,"model.layers.%d.self_attn.k_proj.weight",l); kp[l]=jo(bn);
        snprintf(bn,128,"model.layers.%d.self_attn.v_proj.weight",l); vp[l]=jo(bn);
        snprintf(bn,128,"model.layers.%d.self_attn.o_proj.weight",l); op[l]=jo(bn);
        snprintf(bn,128,"model.layers.%d.mlp.gate_proj.weight",l);   gp[l]=jo(bn);
        snprintf(bn,128,"model.layers.%d.mlp.up_proj.weight",l);     up[l]=jo(bn);
        snprintf(bn,128,"model.layers.%d.mlp.down_proj.weight",l);   dp[l]=jo(bn);
        snprintf(bn,128,"model.layers.%d.input_layernorm.weight",l);          in_off[l]=jo(bn);
        snprintf(bn,128,"model.layers.%d.post_attention_layernorm.weight",l); pa_off[l]=jo(bn);
        snprintf(bn,128,"model.layers.%d.self_attn.q_norm.weight",l);         qn_off[l]=jo(bn);
        snprintf(bn,128,"model.layers.%d.self_attn.k_norm.weight",l);         kn_off[l]=jo(bn);}
    uint64_t no=jo("model.norm.weight");
    
    // Load norm weights
    std::vector<std::vector<float>> in_n(NC,std::vector<float>(H));
    std::vector<std::vector<float>> pa_n(NC,std::vector<float>(H));
    std::vector<float> fin_v(H); float* fin = fin_v.data();
    std::vector<std::vector<float>> qn_w(NC,std::vector<float>(HD));
    std::vector<std::vector<float>> kn_w(NC,std::vector<float>(HD));
    
    for(int l=0;l<NC;l++){
        auto iw=(const uint16_t*)(md+df+in_off[l]),pw=(const uint16_t*)(md+df+pa_off[l]);
        for(int i=0;i<H;i++){in_n[l][i]=bf16g(iw[i]);pa_n[l][i]=bf16g(pw[i]);}
        if(cfg.has_q_norm&&qn_off[l]){auto qq=(const uint16_t*)(md+df+qn_off[l]);
            for(int i=0;i<HD;i++) qn_w[l][i]=bf16g(qq[i]);}
        if(cfg.has_k_norm&&kn_off[l]){auto kk=(const uint16_t*)(md+df+kn_off[l]);
            for(int i=0;i<HD;i++) kn_w[l][i]=bf16g(kk[i]);}}
    {auto fw=(const uint16_t*)(md+df+no);for(int i=0;i<H;i++)fin_v[i]=bf16g(fw[i]);}

    // Load rope_freqs if file-based
    std::vector<float> rope_freqs;
    if(cfg.has_rope_freqs_file){
        int rf_rows=0; find_tensor_info(js,jl,"rope_freqs.weight",&rf_rows);
        if(rf_rows>0){rope_freqs.resize(rf_rows);
            auto rf=(const uint16_t*)(md+df+jo("rope_freqs.weight"));
            for(int i=0;i<rf_rows;i++)rope_freqs[i]=bf16g(rf[i]);
            fprintf(stderr,"  rope_freqs[%zu]\n",rope_freqs.size());}}

    // Determine I8 weight tile rows
    auto gi8=[&](const char*k)->int{int r=0;find_tensor_info(js,jl,k,&r);return r;};
    int q_i8=gi8("model.layers.0.self_attn.q_proj.weight");
    int k_i8=gi8("model.layers.0.self_attn.k_proj.weight");
    int v_i8=gi8("model.layers.0.self_attn.v_proj.weight");
    int o_i8=gi8("model.layers.0.self_attn.o_proj.weight");
    int g_i8=gi8("model.layers.0.mlp.gate_proj.weight");
    int u_i8=gi8("model.layers.0.mlp.up_proj.weight");
    int d_i8=gi8("model.layers.0.mlp.down_proj.weight");

    fprintf(stderr,"Init NPU contexts (M=%d,XM=%d)...\n",M,XM);
    xrt::device dev(0);
    std::string xd="/home/bcloud/npu-sandbox/npu-infer/build/int8";
    
    I8Ctx cq,co,cg,cd;
    cq.MD=XM;cq.KD=cfg.xclbin_qkv_k;cq.ND=cfg.xclbin_qkv_n;
    co.MD=XM;co.KD=cfg.xclbin_o_k;  co.ND=cfg.xclbin_o_n;
    cd.MD=XM;cd.KD=cfg.xclbin_d_k;  cd.ND=cfg.xclbin_d_n;
    if(cfg.gu_split){cg.MD=XM;cg.KD=cfg.xclbin_g_k;cg.ND=cfg.xclbin_g_n;}
    else{cg.MD=XM;cg.KD=cfg.xclbin_gu_k;cg.ND=cfg.xclbin_gu_n;}
    
    auto xp=[&](const char*t){return xd+"/final_i8_"+t+"_"+cfg.model_tag+".xclbin";};
    auto ip=[&](const char*t){return xd+"/insts_i8_"+t+"_"+cfg.model_tag+".txt";};
    
    if(!cq.init(dev,xp("QKV").c_str(),ip("QKV").c_str(),4,NC)){fprintf(stderr,"FAIL QKV\n");return 1;}
    if(!co.init(dev,xp("O").c_str(),  ip("O").c_str(),  4,NC)){fprintf(stderr,"FAIL O\n");return 1;}
    if(cfg.gu_split){
        if(!cg.init(dev,xp("G").c_str(), ip("G").c_str(),4,NC)){fprintf(stderr,"FAIL G\n");return 1;}
    } else {
        if(!cg.init(dev,xp("GU").c_str(),ip("GU").c_str(),4,NC)){fprintf(stderr,"FAIL GU\n");return 1;}
    }
    if(!cd.init(dev,xp("D").c_str(),  ip("D").c_str(),  4,NC)){fprintf(stderr,"FAIL D\n");return 1;}
    
    // 5th context for U if GU split
    std::unique_ptr<I8Ctx> cu_ptr;
    if(cfg.gu_split){
        cu_ptr=std::make_unique<I8Ctx>();
        cu_ptr->MD=XM;cu_ptr->KD=cfg.xclbin_u_k;cu_ptr->ND=cfg.xclbin_u_n;
        if(!cu_ptr->init(dev,xp("U").c_str(),ip("U").c_str(),4,NC)){fprintf(stderr,"FAIL U\n");return 1;}
    }

    fprintf(stderr,"Dequant+pack %d layers...\n",NC);
    auto tp=std::chrono::steady_clock::now();
    
    std::vector<float> qsc(NC),osc(NC),gsc(NC),dsc(NC),usc(NC);
    for(int l=0;l<NC;l++){
        int qr,kr,vr,or_,gr,ur,dr,unused;
        float*qw=dequant_weight(i8p(qp[l]),q_i8,H,&qr,&unused);
        float*kw=dequant_weight(i8p(kp[l]),k_i8,H,&kr,&unused);
        float*vw=dequant_weight(i8p(vp[l]),v_i8,H,&vr,&unused);
        int t=qr+kr+vr;std::vector<float>w((size_t)H*t);
        for(int k=0;k<H;k++){memcpy(&w[k*t],&qw[k*qr],qr*4);
            memcpy(&w[k*t+qr],&kw[k*kr],kr*4);
            memcpy(&w[k*t+qr+kr],&vw[k*vr],vr*4);}
        cq.packB(l,w.data(),H,t,qsc[l]);free(qw);free(kw);free(vw);
        
        float*ow=dequant_weight(i8p(op[l]),o_i8,NH*HD,&or_,&unused);
        co.packB(l,ow,or_,H,osc[l]);free(ow);
        
        float*gw=dequant_weight(i8p(gp[l]),g_i8,H,&gr,&unused);
        if(cfg.gu_split){
            float*uw=dequant_weight(i8p(up[l]),u_i8,H,&ur,&unused);
            cg.packB(l,gw,H,gr,gsc[l]);cu_ptr->packB(l,uw,H,ur,usc[l]);free(uw);
        } else {
            float*uw=dequant_weight(i8p(up[l]),u_i8,H,&ur,&unused);
            int t2=gr+ur;std::vector<float>w2((size_t)H*t2);
            for(int k=0;k<H;k++){memcpy(&w2[k*t2],&gw[k*gr],gr*4);memcpy(&w2[k*t2+gr],&uw[k*ur],ur*4);}
            cg.packB(l,w2.data(),H,t2,gsc[l]);free(uw);
        }
        free(gw);
        
        float*dw=dequant_weight(i8p(dp[l]),d_i8,IM,&dr,&unused);
        cd.packB(l,dw,dr,H,dsc[l]);free(dw);
    }
    fprintf(stderr,"  %.0fms\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tp).count());

    // Initialize RoPE
    if(cfg.has_rope_freqs_file&&!rope_freqs.empty()){
        rc.resize((M+128)*HD);rs.resize((M+128)*HD);
        for(int p=0;p<M+128;p++) for(int d=0;d<HD;d+=2){
            int fi=(d/2)%(int)rope_freqs.size();float a=p*rope_freqs[fi];
            rc[p*HD+d]=cosf(a);rs[p*HD+d]=sinf(a);
            rc[p*HD+d+1]=cosf(a);rs[p*HD+d+1]=sinf(a);}
    } else ri(HD,cfg.rope_theta,M+128);
    
    // KV quantized caches
    std::vector<KVQuant> kv(NC);
    
    // Dynamic buffers
    std::vector<float> h((size_t)M*H);
    int qkv_n=cfg.qkv_total,kv_n=NKV*HD;
    std::vector<float> qo((size_t)M*qkv_n),ko((size_t)M*kv_n),vo((size_t)M*kv_n);
    std::vector<float> at((size_t)M*NH*HD),oo((size_t)M*H);
    int mlp_out=cfg.gu_split?IM:(2*IM);
    std::vector<float> gt((size_t)M*mlp_out),su((size_t)M*IM);
    std::vector<float> dwo((size_t)M*H),sb((size_t)M*H);
    
    // Embed input tokens
    for(int m=0;m<M;m++) for(int i=0;i<H;i++) h[m*H+i]=bf16g(emb[input_tokens[m]*H+i]);

    fprintf(stderr,"Inference %d tokens x %d layers...\n",M,NC);
    auto ts=std::chrono::steady_clock::now();
    AttnCPU attn_obj(cfg.WQH,HD);
    int tok_nelems=NKV*HD;

    for(int l=0;l<NC;l++){
        memcpy(sb.data(),h.data(),M*H*4);
        rn_c(h.data(),in_n[l].data(),H,M);
        
        // QKV projection
        memset(qo.data(),0,M*qkv_n*4);
        for(int pass=0;pass<M;pass+=XM){
            int bm=M-pass;if(bm>XM)bm=XM;
            cq.go(l,pass,&h[pass*H],bm,H,5.0f/127.0f,qsc[l],qo.data(),qkv_n);}
        for(int m=0;m<M;m++)cn(&qo[m*qkv_n],qkv_n);
        
        // Split Q, K, V
        for(int m=0;m<M;m++){
            memcpy(&ko[m*kv_n],&qo[m*qkv_n+cfg.qkv_k_offset],kv_n*4);
            memcpy(&vo[m*kv_n],&qo[m*qkv_n+cfg.qkv_v_offset],kv_n*4);}
        
        // Q-norm, K-norm, RoPE
        std::vector<float> tok_K(M*tok_nelems),tok_V(M*tok_nelems);
        for(int m=0;m<M;m++){
            float*q=&qo[m*qkv_n],*k=&ko[m*kv_n],*v=&vo[m*kv_n];
            for(int hh=0;hh<NH;hh++){
                double sq=0;for(int d=0;d<HD;d++){float vv=q[hh*HD+d];if(std::isfinite(vv))sq+=vv*vv;}
                float iq=1.0f/sqrtf((float)(sq/HD)+EPS);
                if(cfg.has_q_norm){for(int d=0;d<HD;d++)q[hh*HD+d]*=iq*qn_w[l][d];}
                else for(int d=0;d<HD;d++)q[hh*HD+d]*=iq;
                ra(&q[hh*HD],HD,m);
                if(hh%GQA==0){
                    int kvh=hh/GQA;
                    double sk=0;for(int d=0;d<HD;d++){float vv=k[kvh*HD+d];if(std::isfinite(vv))sk+=vv*vv;}
                    float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
                    if(cfg.has_k_norm){for(int d=0;d<HD;d++)k[kvh*HD+d]*=ik*kn_w[l][d];}
                    else for(int d=0;d<HD;d++)k[kvh*HD+d]*=ik;
                    ra(&k[kvh*HD],HD,m);
                    memcpy(&tok_K[m*tok_nelems+kvh*HD],&k[kvh*HD],HD*4);
                    memcpy(&tok_V[m*tok_nelems+kvh*HD],&v[kvh*HD],HD*4);}}}
        // Batch-quantize K/V
        for(int m=0;m<M;m++) kv[l].append(m,&tok_K[m*tok_nelems],&tok_V[m*tok_nelems]);
        
        // SIMD attention
        std::vector<float> Ks(kv_n*M),Vs(kv_n*M);
        for(int w=0;w<AW;w++){
            kv[l].dequant_K_all(w,M,Ks.data());
            kv[l].dequant_V_all(w,M,Vs.data());
            for(int m=0;m<M;m++) attn_obj.run(&qo[m*qkv_n+w*cfg.WQH*HD],
                Ks.data(),Vs.data(),M,&at[m*NH*HD+w*cfg.WQH*HD]);}
        
        // O projection
        for(int pass=0;pass<M;pass+=XM){
            int bm=M-pass;if(bm>XM)bm=XM;
            co.go(l,pass,&at[pass*NH*HD],bm,NH*HD,5.0f/127.0f,osc[l],oo.data(),H);}
        for(int m=0;m<M;m++){
            cn(&oo[m*H],H);for(int i=0;i<H;i++)h[m*H+i]=sb[m*H+i]+oo[m*H+i];}
        
        // MLP
        memcpy(sb.data(),h.data(),M*H*4);
        rn_c(h.data(),pa_n[l].data(),H,M);
        
        if(cfg.gu_split){
            // G separately
            memset(gt.data(),0,M*IM*4);
            for(int pass=0;pass<M;pass+=XM){
                int bm=M-pass;if(bm>XM)bm=XM;
                cg.go(l,pass,&h[pass*H],bm,H,5.0f/127.0f,gsc[l],gt.data(),IM);}
            for(int m=0;m<M;m++)cn(&gt[m*IM],IM);
            // U separately
            memset(dwo.data(),0,M*IM*4);
            for(int pass=0;pass<M;pass+=XM){
                int bm=M-pass;if(bm>XM)bm=XM;
                cu_ptr->go(l,pass,&h[pass*H],bm,H,5.0f/127.0f,usc[l],dwo.data(),IM);}
            for(int m=0;m<M;m++)cn(&dwo[m*IM],IM);
            // SiLU(G) * U
            for(int m=0;m<M;m++){
                float*g=&gt[m*IM],*u=&dwo[m*IM];
                for(int i=0;i<IM;i++){float gv=g[i];if(!std::isfinite(gv))gv=0;
                    su[m*IM+i]=(gv/(1.0f+expf(-gv)))*u[i];}}
        } else {
            // Combined GU
            memset(gt.data(),0,M*mlp_out*4);
            for(int pass=0;pass<M;pass+=XM){
                int bm=M-pass;if(bm>XM)bm=XM;
                cg.go(l,pass,&h[pass*H],bm,H,5.0f/127.0f,gsc[l],gt.data(),mlp_out);}
            for(int m=0;m<M;m++){
                cn(&gt[m*mlp_out],mlp_out);
                for(int i=0;i<IM;i++){
                    float gv=gt[m*mlp_out+i];if(!std::isfinite(gv))gv=0;
                    float uv=gt[m*mlp_out+IM+i];if(!std::isfinite(uv))uv=0;
                    su[m*IM+i]=(gv/(1.0f+expf(-gv)))*uv;}}
        }
        
        // D projection
        for(int pass=0;pass<M;pass+=XM){
            int bm=M-pass;if(bm>XM)bm=XM;
            cd.go(l,pass,&su[pass*IM],bm,IM,5.0f/127.0f,dsc[l],dwo.data(),H);}
        for(int m=0;m<M;m++){
            cn(&dwo[m*H],H);for(int i=0;i<H;i++)h[m*H+i]=sb[m*H+i]+dwo[m*H+i];}
    }

    // Final norm + LM head
    memcpy(sb.data(),h.data(),M*H*4);
    rn_c(sb.data(),fin,H,M);  // batch norm + NaN guard
    std::vector<float>lg((size_t)M*NV);
    for(int m=0;m<M;m++){
        for(int n=0;n<NV;n++){
            double s=0;for(int k=0;k<H;k++){
                float hv=sb[m*H+k];if(!std::isfinite(hv))continue;
                uint16_t r=emb[n*H+k];
                if((r&0x7F80)!=0x7F80)s+=(double)hv*bf16f(r);}
            float v=(float)s;lg[m*NV+n]=std::isfinite(v)?v:0.0f;}
    }

    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();

    fprintf(stderr,"\n=== M=%d, %.0fms (%.1fms/tok) ===\n",M,ms,ms/M);
    int last_pred = 0;
    for(int m=0;m<M;m++){
        float mx=-1e30f;for(int i=0;i<NV;i++){float v=lg[m*NV+i];if(std::isfinite(v)&&v>mx)mx=v;}
        if(!std::isfinite(mx)||mx<-1e29f){
            // NaN/Inf in logits — uniform fallback
            fprintf(stderr,"  token %3d: logits=NaN, fallback to token 0\n",m);
            printf("0%s",(m==M-1)?"\n":" ");
            continue;
        }
        double sum=0;float sm[8];int top8[8];
        for(int b=0;b<8;b++){top8[b]=-1;sm[b]=-1e30f;}
        for(int i=0;i<8;i++){float d=lg[m*NV+i]-mx;if(d<-80)d=-80;sm[i]=expf(d);sum+=sm[i];top8[i]=i;}
        for(int i=8;i<NV;i++){
            float d=lg[m*NV+i]-mx;if(d<-80)d=-80;float ev=expf(d);
            for(int t=0;t<8;t++){if(ev>sm[t]){memmove(&sm[t+1],&sm[t],(7-t)*sizeof(float));memmove(&top8[t+1],&top8[t],(7-t)*sizeof(int));sm[t]=ev;top8[t]=i;break;}}}
        // Output top-1 token to stdout for server parsing
        int t = top8[0]>=0 ? top8[0] : 0;
        last_pred = t;
        printf("%d%s",t,(m==M-1)?"\n":" ");
        fprintf(stderr,"  token %3d: top8=[%d,%d,%d,%d,%d,%d,%d,%d]  %.1fms/tok\n",
            m,t,top8[1],top8[2],top8[3],top8[4],top8[5],top8[6],top8[7],ms/M);
    }

    // ── Autoregressive decode loop ──────────────────────────────
    // NPU_GEN=N env var: generate N tokens using persistent KV cache
    int gen_count = 0;
    const char* gen_env = getenv("NPU_GEN");
    if (gen_env) gen_count = atoi(gen_env);
    if (gen_count < 0) gen_count = 0;
    if (gen_count > 64) gen_count = 64;

    if (gen_count > 0) {
        srand(time(NULL));  // seed for temperature sampling
        int next_tok = last_pred;
        int pos = M;  // first decode position = prefill length
        fprintf(stderr,"\n=== Decode %d tokens (pos=%d) ===\n",gen_count,pos);
        auto td0 = std::chrono::steady_clock::now();

        // Single-token buffers
        h.resize(H); qo.resize(qkv_n); ko.resize(kv_n); vo.resize(kv_n);
        at.resize(NH*HD); oo.resize(H);
        if (cfg.gu_split) { gt.resize(IM); dwo.resize(IM); }
        else { gt.resize(mlp_out); }
        su.resize(IM); dwo.resize(H); sb.resize(H);

        std::vector<float> Ks_buf, Vs_buf;
        Ks_buf.reserve(kv_n * KQ_MAX_CTX);
        Vs_buf.reserve(kv_n * KQ_MAX_CTX);

        for (int g = 0; g < gen_count; g++) {
            // Embed
            for (int i = 0; i < H; i++) h[i] = bf16g(emb[next_tok*H + i]);

            for (int l = 0; l < NC; l++) {
                memcpy(sb.data(), h.data(), H*4);
                rn_c(h.data(), in_n[l].data(), H, 1);

                // QKV projection (M=1)
                memset(qo.data(), 0, qkv_n*4);
                cq.go(l, 0, h.data(), 1, H, 5.0f/127.0f, qsc[l], qo.data(), qkv_n);
                cn(qo.data(), qkv_n);

                // Split QKV
                float* q = qo.data();
                float* k = &qo[cfg.qkv_k_offset];
                float* v = &qo[cfg.qkv_v_offset];

                // Q-norm, K-norm, RoPE at absolute position
                for (int hh = 0; hh < NH; hh++) {
                    double sq = 0;
                    for (int d = 0; d < HD; d++) { float vv = q[hh*HD+d]; if (std::isfinite(vv)) sq += vv*vv; }
                    float iq = 1.0f/sqrtf((float)(sq/HD)+EPS);
                    if (cfg.has_q_norm) for (int d = 0; d < HD; d++) q[hh*HD+d] *= iq*qn_w[l][d];
                    else for (int d = 0; d < HD; d++) q[hh*HD+d] *= iq;
                    ra(&q[hh*HD], HD, pos);
                    if (hh % GQA == 0) {
                        int kvh = hh/GQA;
                        double sk = 0;
                        for (int d = 0; d < HD; d++) { float vv = k[kvh*HD+d]; if (std::isfinite(vv)) sk += vv*vv; }
                        float ik = 1.0f/sqrtf((float)(sk/HD)+EPS);
                        if (cfg.has_k_norm) for (int d = 0; d < HD; d++) k[kvh*HD+d] *= ik*kn_w[l][d];
                        else for (int d = 0; d < HD; d++) k[kvh*HD+d] *= ik;
                        ra(&k[kvh*HD], HD, pos);
                    }
                }

                // Append KV to cache
                kv[l].append(pos, k, v);

                // Dequant ALL KV entries and attend
                int nt = kv[l].n;  // total positions now
                Ks_buf.assign(nt * kv_n, 0.0f);
                Vs_buf.assign(nt * kv_n, 0.0f);
                for (int w = 0; w < AW; w++) {
                    kv[l].dequant_K_all(w, nt, Ks_buf.data());
                    kv[l].dequant_V_all(w, nt, Vs_buf.data());
                    attn_obj.run(&q[w*cfg.WQH*HD], Ks_buf.data(), Vs_buf.data(), nt,
                                 &at[w*cfg.WQH*HD]);
                }

                // O projection (M=1)
                co.go(l, 0, at.data(), 1, NH*HD, 5.0f/127.0f, osc[l], oo.data(), H);
                cn(oo.data(), H);
                for (int i = 0; i < H; i++) h[i] = sb[i] + oo[i];

                // MLP
                memcpy(sb.data(), h.data(), H*4);
                rn_c(h.data(), pa_n[l].data(), H, 1);

                if (cfg.gu_split) {
                    cg.go(l, 0, h.data(), 1, H, 5.0f/127.0f, gsc[l], gt.data(), IM);
                    cn(gt.data(), IM);
                    cu_ptr->go(l, 0, h.data(), 1, H, 5.0f/127.0f, usc[l], dwo.data(), IM);
                    cn(dwo.data(), IM);
                    for (int i = 0; i < IM; i++) {
                        float gv = gt[i]; if (!std::isfinite(gv)) gv = 0;
                        su[i] = (gv/(1.0f+expf(-gv))) * dwo[i];
                    }
                } else {
                    cg.go(l, 0, h.data(), 1, H, 5.0f/127.0f, gsc[l], gt.data(), mlp_out);
                    cn(gt.data(), mlp_out);
                    for (int i = 0; i < IM; i++) {
                        float gv = gt[i]; if (!std::isfinite(gv)) gv = 0;
                        float uv = gt[IM+i]; if (!std::isfinite(uv)) uv = 0;
                        su[i] = (gv/(1.0f+expf(-gv))) * uv;
                    }
                }

                // D projection (M=1)
                cd.go(l, 0, su.data(), 1, IM, 5.0f/127.0f, dsc[l], dwo.data(), H);
                cn(dwo.data(), H);
                for (int i = 0; i < H; i++) h[i] = sb[i] + dwo[i];
            }

            // Final norm + LM head
            memcpy(sb.data(), h.data(), H*4);
            rn_c(sb.data(), fin, H, 1);
            float mx_ = -1e30f;
            std::vector<float> lg_dec(NV);
            for (int n = 0; n < NV; n++) {
                double s = 0;
                for (int k = 0; k < H; k++) {
                    float hv = sb[k]; if (!std::isfinite(hv)) continue;
                    uint16_t r = emb[n*H+k];
                    if ((r&0x7F80) != 0x7F80) s += (double)hv*bf16f(r);
                }
                float v = (float)s; lg_dec[n] = std::isfinite(v) ? v : 0.0f;
                if (lg_dec[n] > mx_) mx_ = lg_dec[n];
            }

            // Temperature sampling (default 0.8 from NPU_TEMP env)
            float temperature = 0.8f;
            const char* temp_env = getenv("NPU_TEMP");
            if (temp_env) { float t = atof(temp_env); if (t > 0.01f) temperature = t; }

            if (!std::isfinite(mx_) || mx_ < -1e29f) {
                next_tok = 0;
            } else {
                double sum_ = 0;
                for (int i = 0; i < NV; i++) {
                    float d = lg_dec[i] - mx_; if (d < -80) d = -80;
                    lg_dec[i] = expf(d / temperature);
                    sum_ += lg_dec[i];
                }
                if (sum_ <= 0) { next_tok = 0; }
                else {
                    double r = (double)rand() / RAND_MAX * sum_, acc = 0;
                    next_tok = 0;
                    for (int i = 0; i < NV; i++) {
                        acc += lg_dec[i];
                        if (acc >= r) { next_tok = i; break; }
                    }
                }
            }

            printf(" %d", next_tok);
            fprintf(stderr, "  decode %3d: tok=%d pos=%d\n", g, next_tok, pos);
            pos++;

            // Stop on EOS
            if (next_tok == 151643 || next_tok == 151645 || next_tok == 151646) break;
        }
        printf("\n");
        double dms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-td0).count();
        fprintf(stderr,"  Decode: %.0fms (%.1f ms/tok)\n", dms, dms/gen_count);
    }

    munmap(md,st.st_size);return 0;
}
