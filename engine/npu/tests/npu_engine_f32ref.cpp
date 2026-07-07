/** npu_engine_f32ref.cpp — Pure-float CPU forward of Qwen3-0.6B from a Q4NX file.
 *
 *  Bisect oracle for the NPU engine coherence bug. Mirrors the transformer math
 *  of npu_engine_universal.cpp VERBATIM (same RoPE ri/ra, same RMSNorm rn_c,
 *  same attn_omp, same residual save/restore, same lm_head) — the ONLY change is
 *  the GEMM path: float matmul on dequantized f32 weights instead of INT8-xclbin.
 *  The deployed xclbins are verified bit-exact (see XCLBIN-VERIFIED-2026-07-07.md),
 *  so any coherence difference between this binary and universal.cpp isolates the
 *  bug to (a) the quantize/rescale integration in I8Ctx or (b) shared transformer
 *  math — by diffing per-layer hidden states.
 *
 *  Greedy argmax (temperature 0) for determinism. Set DEBUG_LAYERS=1 to dump
 *  the last-prefill-token hidden state after each layer.
 *
 *  Build:
 *    g++ -std=gnu++23 -O3 -I src -I/usr/include -o tests/npu_engine_f32ref \
 *        tests/npu_engine_f32ref.cpp build/dequant_q4nx.o -lm
 *  Run:
 *    ./tests/npu_engine_f32ref <model.q4nx> [decode_tokens] [DEBUG_LAYERS=1]
 */
#include "platform.h"
#include "model_config.h"
extern "C" float* dequant_i8_to_float_ex(const uint8_t*,int,int,int*,int*);
static constexpr float EPS=1e-6f;

// ── copied verbatim from npu_engine_universal.cpp (transformer math) ──
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
static std::vector<float> emb_f32, lm_head_f32;

static void transpose_pack(const float* src, int out_f, int in_f, float* dst, int dst_stride, int dst_offset) {
    for (int o = 0; o < out_f; o++)
        for (int i = 0; i < in_f; i++)
            dst[(size_t)i * dst_stride + dst_offset + o] = src[(size_t)o * in_f + i];
}

// f32 GEMM: C[M,N] = A[M,K] @ B[K,N]   (B already in [in,out]=[K,N] row-major)
struct F32Layer {
    std::vector<float> W;  // [K, N] row-major
    int K=0, N=0;
    void set(const float* w, int out_f, int in_f) {
        K=in_f; N=out_f; W.assign((size_t)in_f*out_f, 0.0f);
        transpose_pack(w, out_f, in_f, W.data(), out_f, 0);
    }
};
static inline void fgo(const F32Layer& L, const float* A, int M, int K, int N, float* C) {
    for (int m=0;m<M;m++) {
        const float* am = A + (size_t)m*K;
        float* cm = C + (size_t)m*N;
        for (int n=0;n<N;n++) cm[n]=0.0f;
        for (int k=0;k<K;k++) {
            float a = am[k]; if (!std::isfinite(a)) a=0;
            const float* bk = L.W.data() + (size_t)k*N;
            #pragma omp simd
            for (int n=0;n<N;n++){ float v=a*bk[n]; if (std::isfinite(v)) cm[n]+=v; }
        }
    }
}

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

// greedy argmax lm head (deterministic; replaces lm_topk_omp's rand() sampling)
static int lm_argmax(const float*hidden,int NV,int H,const float*emb){
    int best=0; float bv=-1e30f;
    #pragma omp parallel
    { int tbest=0; float tbv=-1e30f;
      #pragma omp for nowait
      for(int n=0;n<NV;n++){double s=0;const float*e=&emb[(size_t)n*H];for(int k=0;k<H;k++)s+=(double)hidden[k]*e[k];float v=(float)s;if(v>tbv){tbv=v;tbest=n;}}
      #pragma omp critical
      { if (tbv>bv){bv=tbv;best=tbest;} }
    }
    return best;
}

static uint64_t jo(const char*js,size_t jl,const char*nm){size_t nl=strlen(nm);
    const char*p=js,*e=js+jl;while(p<e){auto q=(const char*)platform_memmem(p,e-p,nm,nl);
        if(!q)return 0;if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){
            auto o=strstr(q,"\"data_offsets\"");if(o){auto a=strchr(o,'[');if(a)return strtoull(a+1,NULL,10);}}p=q+1;}return 0;}

static bool debug_layers(){const char*e=getenv("DEBUG_LAYERS");return e&&*e;}

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    if(argc<2){printf("Usage: %s model.q4nx [decode_tokens] (DEBUG_LAYERS=1 env)\n",argv[0]);return 1;}
    const char*mp=argv[1];int ng=(argc>2)?atoi(argv[2]):16;if(ng<1)ng=1;
    std::string mp_s(mp),model_tag;auto ls=mp_s.rfind('/');auto sl=mp_s.rfind('/',ls-1);
    model_tag=(sl!=std::string::npos&&ls!=std::string::npos)?mp_s.substr(sl+1,ls-sl-1):mp_s.substr(ls+1);
    for(auto&c:model_tag){c=tolower(c);if(c=='-'||c=='.')c='_';}
    const char*sfxs[]={"_npu2","_instruct","_it","_it_npu2"};
    for(auto sf:sfxs){size_t sl2=strlen(sf);if(model_tag.size()>sl2&&model_tag.substr(model_tag.size()-sl2)==sf)model_tag=model_tag.substr(0,model_tag.size()-sl2);}

    ModelConfig cfg=parse_q4nx_header(mp,model_tag.c_str());
    if(!cfg.valid()){printf("ERR: invalid model config\n");return 1;}
    int H=cfg.H,NC=cfg.NC,NH=cfg.NH,NKV=cfg.NKV,HD=cfg.HD,IM=cfg.IM,NV=cfg.NV,GQA=cfg.GQA,XM=cfg.XM;
    printf("=== f32 REFERENCE — %s ===\n",model_tag.c_str());
    printf("H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d NV=%d gu_split=%d q_norm=%d k_norm=%d rope_theta=%.0f\n",
        H,NC,NH,NKV,HD,IM,NV,cfg.gu_split,(int)cfg.has_q_norm,(int)cfg.has_k_norm,cfg.rope_theta);

    auto fd=platform_open_read(mp);platform_stat st;platform_fstat(fd,&st);
    uint8_t*md=(uint8_t*)platform_mmap((size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0);platform_close(fd);
    uint64_t hsz;memcpy(&hsz,md,8);uint64_t df=8+hsz;
    auto i8p=[&](uint64_t o){return md+df+o;};auto emb=(const uint16_t*)(md+df);
    const char*js=(const char*)(md+8);size_t jl=hsz;

    printf("Pre-convert emb f32...\n");
    emb_f32.resize((size_t)NV*H);
    for(int n=0;n<NV;n++)for(int i=0;i<H;i++)emb_f32[(size_t)n*H+i]=bf16g(emb[n*H+i]);

    std::vector<uint64_t> qp(NC),kp(NC),vp(NC),op(NC),gp(NC),up(NC),dp(NC),in_off(NC),pa_off(NC),qn_off(NC),kn_off(NC);
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
    uint64_t no=jo(js,jl,"model.norm.weight"),lo=jo(js,jl,"lm_head.weight");
    std::vector<std::vector<float>> in_n(NC,std::vector<float>(H)),pa_n(NC,std::vector<float>(H)),qn_w(NC,std::vector<float>(HD)),kn_w(NC,std::vector<float>(HD));
    std::vector<float> fin_v(H);
    // NOTE: do NOT clamp norm weights — HF Qwen3 final_norm has 99.6% of weights
    // outside [-2,2] (max 15.31, mean 3.84). The prior std::min(2.0f,max(-2.0f,...))
    // clamp (inherited in this oracle and removed from npu_engine_universal.cpp in
    // commit 2a9d770cc) destroys final RMSNorm scaling → hidden-state explosion.
    for(int l=0;l<NC;l++){auto iw=(const uint16_t*)(md+df+in_off[l]),pw=(const uint16_t*)(md+df+pa_off[l]);
        for(int i=0;i<H;i++){in_n[l][i]=bf16g(iw[i]);pa_n[l][i]=bf16g(pw[i]);}
        if(cfg.has_q_norm&&qn_off[l]){auto qq=(const uint16_t*)(md+df+qn_off[l]);for(int i=0;i<HD;i++)qn_w[l][i]=bf16g(qq[i]);}
        if(cfg.has_k_norm&&kn_off[l]){auto kk=(const uint16_t*)(md+df+kn_off[l]);for(int i=0;i<HD;i++)kn_w[l][i]=bf16g(kk[i]);}}
    {auto fw=(const uint16_t*)(md+df+no);for(int i=0;i<H;i++)fin_v[i]=bf16g(fw[i]);}

    auto gi8=[&](const char*k)->int{int r=0;find_tensor_info(js,jl,k,&r);return r;};
    int q_i8=gi8("model.layers.0.self_attn.q_proj.weight"),k_i8=gi8("model.layers.0.self_attn.k_proj.weight"),v_i8=gi8("model.layers.0.self_attn.v_proj.weight");
    int o_i8=gi8("model.layers.0.self_attn.o_proj.weight"),g_i8=gi8("model.layers.0.mlp.gate_proj.weight"),u_i8=gi8("model.layers.0.mlp.up_proj.weight"),d_i8=gi8("model.layers.0.mlp.down_proj.weight");
    int lm_i8=gi8("lm_head.weight");

    bool tied=true; // Qwen3-0.6B: tie_word_embeddings=true (verified)
    if(lo&&lm_i8>0&&!tied){int lr,lc;float*lm_raw=dequant_i8_to_float_ex(i8p(lo),lm_i8,H,&lr,&lc);if(lm_raw){lm_head_f32.assign(lm_raw,lm_raw+(size_t)lr*lc);free(lm_raw);
        printf("  lm_head: %dx%d (separate)\n",lr,lc);}else printf("  lm_head: dequant fail, emb\n");}
    if(lm_head_f32.empty())printf("  lm_head: using emb_f32 (tied)\n");
    const float* lm_emb = lm_head_f32.empty() ? emb_f32.data() : lm_head_f32.data();

    // Dequant all weights to f32 and pack per-layer (transpose to [K,N])
    printf("Dequant+pack f32 weights...\n");auto tp=std::chrono::steady_clock::now();
    std::vector<F32Layer> Lq(NC),Lk(NC),Lv(NC),Lo(NC),Lg(NC),Lu(NC),Ld(NC);
    const int QOUT=NH*HD,KVOUT=NKV*HD,OOUT=H,OIN=NH*HD,GUOUT=IM,DOUT=H,DIN=IM;
    for(int l=0;l<NC;l++){
        int r1,r2;
        float*qw=dequant_i8_to_float_ex(i8p(qp[l]),q_i8,H,&r1,&r2);
        float*kw=dequant_i8_to_float_ex(i8p(kp[l]),k_i8,H,&r1,&r2);
        float*vw=dequant_i8_to_float_ex(i8p(vp[l]),v_i8,H,&r1,&r2);
        Lq[l].set(qw,QOUT,H);free(qw);
        Lk[l].set(kw,KVOUT,H);free(kw);
        Lv[l].set(vw,KVOUT,H);free(vw);
        float*ow=dequant_i8_to_float_ex(i8p(op[l]),o_i8,OIN,&r1,&r2);Lo[l].set(ow,OOUT,OIN);free(ow);
        float*gw=dequant_i8_to_float_ex(i8p(gp[l]),g_i8,H,&r1,&r2);Lg[l].set(gw,GUOUT,H);free(gw);
        float*uw=dequant_i8_to_float_ex(i8p(up[l]),u_i8,H,&r1,&r2);Lu[l].set(uw,GUOUT,H);free(uw);
        float*dw=dequant_i8_to_float_ex(i8p(dp[l]),d_i8,DIN,&r1,&r2);Ld[l].set(dw,DOUT,DIN);free(dw);
        if(l==0)printf("  L0 shapes: Q %dx%d O %dx%d G %dx%d D %dx%d\n",
            Lq[l].K,Lq[l].N,Lo[l].K,Lo[l].N,Lg[l].K,Lg[l].N,Ld[l].K,Ld[l].N);
    }
    printf("  %.0fms\n\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tp).count());

    ri(HD,cfg.rope_theta,4096);
    int kv_size=4096*NKV*HD;
    std::vector<std::vector<float>> kvk(NC),kvv(NC);
    for(int l=0;l<NC;l++){kvk[l].resize(kv_size,0);kvv[l].resize(kv_size,0);}
    int qkv_n=cfg.qkv_total;
    std::vector<float> h_b(XM*H), qo_b(XM*qkv_n), at_b(XM*NH*HD), oo_b(XM*H), gt_b(XM*2*IM), su_b(XM*IM), dw_b(XM*H), sb_data(XM*H), h_data(H), lg_buf(NV);
    int sp=0;
    std::vector<int> pt_vec={151644,872,198,13048,151645,198,151644,77091,198}; // "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    int npt=(int)pt_vec.size();

    // ===== PREFILL =====
    printf("=== Prefill %d ===\n",npt);auto t0=std::chrono::steady_clock::now();
    for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]=emb_f32[(size_t)pt_vec[pi]*H+i];
    for(int l=0;l<NC;l++){
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)sb_data[pi*H+i]=h_b[pi*H+i];
        for(int pi=0;pi<npt;pi++)rn_c(&h_b[pi*H],in_n[l].data(),H);
        // QKV via separate float GEMMs, then SCATTER into universal's per-token-interleaved
        // layout that the attention loop reads: for each token pi, qo_b[pi*qkv_n + 0..QOUT]=Q,
        // [.. + qkv_k_offset]=K, [.. + qkv_v_offset]=V. (universal's fused GEMM produces
        // this layout directly; separate GEMMs do not, so we scatter.)
        { static thread_local std::vector<float> tq,tk,tv;
          tq.assign((size_t)npt*QOUT,0.0f); tk.assign((size_t)npt*KVOUT,0.0f); tv.assign((size_t)npt*KVOUT,0.0f);
          fgo(Lq[l],h_b.data(),npt,H,QOUT,tq.data());  cn(tq.data(),npt*QOUT);
          fgo(Lk[l],h_b.data(),npt,H,KVOUT,tk.data());  cn(tk.data(),npt*KVOUT);
          fgo(Lv[l],h_b.data(),npt,H,KVOUT,tv.data());  cn(tv.data(),npt*KVOUT);
          for(int pi=0;pi<npt;pi++){
            memcpy(&qo_b[(size_t)pi*qkv_n],            &tq[(size_t)pi*QOUT],QOUT*4);
            memcpy(&qo_b[(size_t)pi*qkv_n+cfg.qkv_k_offset],&tk[(size_t)pi*KVOUT],KVOUT*4);
            memcpy(&qo_b[(size_t)pi*qkv_n+cfg.qkv_v_offset],&tv[(size_t)pi*KVOUT],KVOUT*4);
          }
          cn(qo_b.data(),npt*qkv_n); }
        float*qn=qn_w[l].data(),*kn=kn_w[l].data();
        for(int pi=0;pi<npt;pi++){
            for(int hh=0;hh<NH;hh++){double s=0;for(int d=0;d<HD;d++)s+=qo_b[pi*qkv_n+hh*HD+d]*qo_b[pi*qkv_n+hh*HD+d];float iq=1.0f/sqrtf((float)(s/HD)+EPS);
                for(int d=0;d<HD;d++)qo_b[pi*qkv_n+hh*HD+d]*=iq*(cfg.has_q_norm?qn[d]:1.0f);ra(&qo_b[pi*qkv_n+hh*HD],HD,sp+pi);}
            for(int kvh=0;kvh<NKV;kvh++){float*ks=&qo_b[pi*qkv_n+cfg.qkv_k_offset+kvh*HD],*vs=&qo_b[pi*qkv_n+cfg.qkv_v_offset+kvh*HD];
                double sk=0;for(int d=0;d<HD;d++)sk+=ks[d]*ks[d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
                for(int d=0;d<HD;d++)ks[d]*=ik*(cfg.has_k_norm?kn[d]:1.0f);ra(ks,HD,sp+pi);
                memcpy(&kvk[l][(sp+pi)*NKV*HD+kvh*HD],ks,HD*4);
                memcpy(&kvv[l][(sp+pi)*NKV*HD+kvh*HD],vs,HD*4);}
        }
        int cl=sp+npt;
        for(int pi=0;pi<npt;pi++)attn_omp(&qo_b[pi*qkv_n],&at_b[pi*NH*HD],cl,kvk[l].data(),kvv[l].data(),NH,NKV,HD,GQA,sp+pi+1);
        fgo(Lo[l],at_b.data(),npt,NH*HD,H,oo_b.data());cn(oo_b.data(),npt*H);
        if(debug_layers()&&l==0){double nq=0,nr=0,na=0,no=0;for(int i=0;i<H;i++){double v=qo_b[(npt-1)*qkv_n+i];if(std::isfinite(v))nq+=v*v;v=h_b[(npt-1)*H+i];if(std::isfinite(v))nr+=v*v;v=at_b[(npt-1)*NH*HD+i%NH*HD];na+=v*v;v=oo_b[(npt-1)*H+i];if(std::isfinite(v))no+=v*v;}printf("  L0sub prenorm|h|=%.4f postQKV|q|=%.4f attn|a|=%.4f o|oo|=%.4f\n",sqrtf((float)nr),sqrtf((float)nq),sqrtf((float)na),sqrtf((float)no));}
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]=sb_data[pi*H+i]+oo_b[pi*H+i];
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)sb_data[pi*H+i]=h_b[pi*H+i];
        for(int pi=0;pi<npt;pi++)rn_c(&h_b[pi*H],pa_n[l].data(),H);
        fgo(Lg[l],h_b.data(),npt,H,IM,gt_b.data());cn(gt_b.data(),npt*IM);
        fgo(Lu[l],h_b.data(),npt,H,IM,&gt_b[npt*IM]);cn(&gt_b[npt*IM],npt*IM);
        for(int pi=0;pi<npt;pi++)for(int i=0;i<IM;i++){float gv=gt_b[pi*IM+i];if(!std::isfinite(gv))gv=0;su_b[pi*IM+i]=(gv/(1.0f+expf(-gv)))*gt_b[pi*IM+IM+i];}
        fgo(Ld[l],su_b.data(),npt,IM,H,dw_b.data());cn(dw_b.data(),npt*H);
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]=sb_data[pi*H+i]+dw_b[pi*H+i];

        if(debug_layers()){
            double nm=0;for(int i=0;i<H;i++)nm+=(double)h_b[(npt-1)*H+i]*h_b[(npt-1)*H+i];
            printf("LAYR %02d post |h|=%.4f  h[0..7]=",(int)l,sqrtf((float)nm));
            for(int q=0;q<8;q++)printf("%.4f ",h_b[(npt-1)*H+q]);printf("\n");
        }
    }
    sp+=npt;memcpy(h_data.data(),&h_b[(npt-1)*H],H*4);
    printf("Prefill: %.0fms\n\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    // ===== GREEDY DECODE (argmax) =====
    // BUG #2 (in npu_engine_universal.cpp AND this oracle before this rewrite):
    // the original "boot" step re-ran all 28 transformer layers on h_data — but
    // h_data is the post-prefill final RESIDUAL (input_rmsnorm/head), not an
    // embedding. Forwarding a residual through RMSNorm + attention + GEMM again
    // is invalid; it should ONLY receive final-norm + lm_head. Subsequent steps
    // embed the predicted token and run the full layer stack.
    printf("=== Greedy Decode (%d tokens) ===\n",ng);
    // Step 0: predict next token from prefill's last residual (NO layer loop).
    {   float h0[H]; memcpy(h0,h_data.data(),H*4);
        rn_c(h0,fin_v.data(),H);
        int tid=lm_argmax(h0,NV,H,lm_emb);
        printf("  [0] tok=%d\n",tid);
        // next decode step forwards embed(tid) through all layers
        for(int i=0;i<H;i++)h_data[i]=emb_f32[(size_t)tid*H+i];
    }
    for(int step=1;step<ng;step++){
        float h0[H];memcpy(h0,h_data.data(),H*4);
        for(int l=0;l<NC;l++){
            memcpy(sb_data.data(),h0,H*4);rn_c(h0,in_n[l].data(),H);
            std::vector<float> qo(qkv_n);
            fgo(Lq[l],h0,1,H,QOUT,qo.data());cn(qo.data(),QOUT);
            fgo(Lk[l],h0,1,H,KVOUT,&qo[QOUT]);cn(&qo[QOUT],KVOUT);
            fgo(Lv[l],h0,1,H,KVOUT,&qo[QOUT+KVOUT]);cn(&qo[QOUT+KVOUT],KVOUT);
            std::vector<float> ko(NKV*HD),vo(NKV*HD);
            memcpy(ko.data(),&qo[QOUT],NKV*HD*4);memcpy(vo.data(),&qo[QOUT+KVOUT],NKV*HD*4);
            float*qn=qn_w[l].data(),*kn=kn_w[l].data();
            for(int hh=0;hh<NH;hh++){double sq=0;for(int d=0;d<HD;d++)sq+=qo[hh*HD+d]*qo[hh*HD+d];float iq=1.0f/sqrtf((float)(sq/HD)+EPS);
                for(int d=0;d<HD;d++)qo[hh*HD+d]*=iq*(cfg.has_q_norm?qn[d]:1.0f);ra(&qo[hh*HD],HD,sp);}
            for(int kvh=0;kvh<NKV;kvh++){double sk=0;for(int d=0;d<HD;d++)sk+=ko[kvh*HD+d]*ko[kvh*HD+d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
                for(int d=0;d<HD;d++)ko[kvh*HD+d]*=ik*(cfg.has_k_norm?kn[d]:1.0f);ra(&ko[kvh*HD],HD,sp);
                memcpy(&kvk[l][sp*NKV*HD+kvh*HD],&ko[kvh*HD],HD*4);memcpy(&kvv[l][sp*NKV*HD+kvh*HD],&vo[kvh*HD],HD*4);}
            int cl=sp+1;std::vector<float>at(NH*HD);
            attn_omp(qo.data(),at.data(),cl,kvk[l].data(),kvv[l].data(),NH,NKV,HD,GQA);
            fgo(Lo[l],at.data(),1,NH*HD,H,oo_b.data());cn(oo_b.data(),H);for(int i=0;i<H;i++)h0[i]=sb_data[i]+oo_b[i];
            memcpy(sb_data.data(),h0,H*4);rn_c(h0,pa_n[l].data(),H);
            std::vector<float>gt(IM*2);
            fgo(Lg[l],h0,1,H,IM,gt.data());cn(gt.data(),IM);
            fgo(Lu[l],h0,1,H,IM,&gt[IM]);cn(&gt[IM],IM);
            for(int i=0;i<IM;i++){float gv=gt[i];if(!std::isfinite(gv))gv=0;su_b[i]=(gv/(1.0f+expf(-gv)))*gt[IM+i];}
            fgo(Ld[l],su_b.data(),1,IM,H,dw_b.data());cn(dw_b.data(),H);for(int i=0;i<H;i++)h0[i]=sb_data[i]+dw_b[i];
        }
        memcpy(sb_data.data(),h0,H*4);rn_c(sb_data.data(),fin_v.data(),H);
        int tid=lm_argmax(sb_data.data(),NV,H,lm_emb);
        printf("  [%d] tok=%d\n",step,tid);
        sp++;
        // next decode step forwards embed(tid) through all layers
        for(int i=0;i<H;i++)h_data[i]=emb_f32[(size_t)tid*H+i];
    }
    (void)md;

    platform_munmap(md,(size_t)st.st_size);return 0;
}

