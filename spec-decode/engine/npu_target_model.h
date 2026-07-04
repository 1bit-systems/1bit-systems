#pragma once
/** NPUQwen3Target — real NPU dispatch for the speculative-decode TargetModelInterface.
 *  Ported from the proven engine/npu/src/npu_engine_cb.cpp dispatch pattern (4 INT8 GEMM
 *  xclbins: QKV/O/GU/D, hand-rolled RMSNorm/RoPE/attention/SwiGLU) into a reusable class.
 *  Unlike engine/npu/src/npu_engine_spec_decode.cpp (a same-model self-draft prototype that
 *  doesn't actually save compute), this only implements the TARGET side — draft comes from
 *  the separate tiny MTP head in draft/mtp_draft.h. */

#include "spec_decode.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

extern "C" float* dequant_i8_to_float(const uint8_t*, int, int*, int*);
extern "C" float* dequant_i8_to_float_ex(const uint8_t*, int, int, int*, int*);

namespace npu_target_detail {

// See docs/V12-CORRECTNESS-BLOCKER.md. dequant_i8_to_float(_ex) returns row-major
// [out_features, in_features] (PyTorch nn.Linear convention); packB()/go() need the
// transpose - [in_features, out_features] - since the GEMM computes A[tokens,in] @ B[in,out].
static inline void transpose_pack(const float* src, int out_f, int in_f, float* dst, int dst_stride, int dst_offset) {
    for (int o = 0; o < out_f; o++)
        for (int i = 0; i < in_f; i++)
            dst[(size_t)i * dst_stride + dst_offset + o] = src[(size_t)o * in_f + i];
}
// Dynamic per-call activation quantization scale (see docs/V12-CORRECTNESS-BLOCKER.md) -
// a hardcoded 5.0f/127.0f assumes activations stay within [-5,5], but measured post-RMSNorm
// activations range as wide as [-8.24,7.01], silently clipping to +-127 every layer.
static inline float dynamic_ascale(const float* x, int n) {
    float amax = 0;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (std::isfinite(a) && a > amax) amax = a; }
    if (amax < 1e-12f) amax = 1.0f;
    return amax / 127.0f;
}

static inline float bf16f(uint16_t v){uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f;}
static inline float bf16g(uint16_t v){return(v&0x7F80)==0x7F80?0.0f:bf16f(v);}
static inline void cn(float*x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
static inline void sm(float*sc,int n){
    if(n<=0)return; cn(sc,n);
    float mx=sc[0]; for(int i=1;i<n;i++) if(sc[i]>mx) mx=sc[i];
    double s=0; for(int i=0;i<n;i++){float d=sc[i]-mx; if(d>80)d=80; else if(d<-80)d=-80; sc[i]=expf(d); s+=sc[i];}
    if(s<=0){float iv=1.0f/n; for(int i=0;i<n;i++)sc[i]=iv; return;}
    float is=1.0f/(float)s; for(int i=0;i<n;i++)sc[i]*=is;
}
static inline void rn_c(float*x,const float*w,int n){
    cn(x,n); double ss=0; for(int i=0;i<n;i++) if(std::isfinite(x[i])) ss+=(double)x[i]*x[i];
    float ir=1.0f/sqrtf((float)(ss/n)+1e-6f);
    for(int i=0;i<n;i++) x[i]=std::isfinite(x[i])?x[i]*ir*w[i]:0.0f;
}
static uint64_t jo(const char*js,size_t jl,const char*nm){
    size_t nl=strlen(nm); const char*p=js,*e=js+jl;
    while(p<e){
        auto q=(const char*)memmem(p,e-p,nm,nl); if(!q) return 0;
        if(q>js && *(q-1)=='"' && *(q+nl)=='"'){
            auto o=strstr(q,"\"data_offsets\""); if(o){auto a=strchr(o,'['); if(a) return strtoull(a+1,NULL,10);}
        }
        p=q+1;
    }
    return 0;
}

struct I8Ctx {
    const char* name; int MD,KD,ND;
    std::unique_ptr<xrt::xclbin> xc; std::unique_ptr<xrt::hw_context> hc; std::unique_ptr<xrt::kernel> k;
    std::vector<uint32_t> ins;
    std::unique_ptr<xrt::bo> bI,bA,bC;
    std::vector<std::unique_ptr<xrt::bo>> layerB;
    int8_t* Am; int16_t* Cm;

    bool init(xrt::device& d, const char* xp, const char* ip, int gid_B, int num_layers) {
        FILE* f = fopen(ip, "rb"); if (!f) return false;
        fseek(f,0,2); long sz=ftell(f); fseek(f,0,0);
        ins.resize(sz/4); size_t rd = fread(ins.data(),4,ins.size(),f); (void)rd; fclose(f);
        xc = std::make_unique<xrt::xclbin>(std::string(xp));
        d.register_xclbin(*xc);
        hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
        k = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");
        bI = std::make_unique<xrt::bo>(d, ins.size()*4, XCL_BO_FLAGS_CACHEABLE, k->group_id(1));
        memcpy(bI->map(), ins.data(), ins.size()*4);
        bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bA = std::make_unique<xrt::bo>(d, (size_t)MD*KD, XRT_BO_FLAGS_HOST_ONLY, k->group_id(3));
        bC = std::make_unique<xrt::bo>(d, (size_t)MD*ND*2, XRT_BO_FLAGS_HOST_ONLY, k->group_id(5));
        Am = (int8_t*)bA->map(); Cm = (int16_t*)bC->map();
        layerB.resize(num_layers);
        for (int l=0;l<num_layers;l++)
            layerB[l] = std::make_unique<xrt::bo>(d, (size_t)KD*ND, XRT_BO_FLAGS_HOST_ONLY, k->group_id(gid_B));
        return true;
    }
    void packB(int l, const float* w, int K, int N, float& sout) {
        float amax=0; for(int i=0;i<K*N;i++){float a=fabsf(w[i]); if(std::isfinite(a)&&a>amax)amax=a;}
        if(amax<1e-12f)amax=1.0f; sout=amax/127.0f; float is=127.0f/amax;
        auto* Bm=(int8_t*)layerB[l]->map();
        for(int i=0;i<K*N;i++){float v=w[i]; if(!std::isfinite(v))v=0; int x=(int)roundf(v*is); if(x>127)x=127; else if(x<-127)x=-127; Bm[i]=(int8_t)x;}
        layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    void go(int l, const float* A, int am, int ak, float ascale, float Bscale, float* C, int an) {
        float ais=1.0f/ascale; memset(Am,0,(size_t)MD*KD);
        for(int m=0;m<am;m++) for(int kk=0;kk<ak;kk++){
            float v=A[m*ak+kk]; if(!std::isfinite(v))v=0;
            int q=(int)roundf(v*ais); if(q>127)q=127; else if(q<-127)q=-127;
            Am[m*KD+kk]=(int8_t)q;
        }
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r = (*k)((unsigned)3, *bI, (unsigned)ins.size(), *bA, *layerB[l], *bC);
        r.wait();
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        float cs = ascale*Bscale;
        for(int m=0;m<am;m++) for(int n=0;n<an;n++){
            float val=(float)Cm[m*ND+n]*cs; if(!std::isfinite(val))val=0;
            C[m*an+n]=val;
        }
    }
};

} // namespace npu_target_detail

// Qwen3-0.6B target model running on real NPU hardware via the proven 4-xclbin INT8 dispatch.
// forward()/forward_with_kv() both route through batch_forward(), parameterized by start_pos:
// start_pos=0 resets the KV cache (fresh generation); start_pos=past_len continues it (verify
// step for speculative decoding). Rejected draft tokens are rolled back for free — kv[l].n is
// only advanced past accepted positions, so stale written-but-unaccepted KV entries are simply
// never read by later attention passes.
class NPUQwen3Target : public TargetModelInterface {
public:
    static constexpr int H=1024, NC=28, NH=16, NKV=8, HD=128, IM=3072, NV=151936, GQA=2, XM=128;

    NPUQwen3Target(const char* model_path,
                   const char* xclbin_dir,
                   const int32_t* target_layer_ids, int32_t num_target_layers)
        : target_layer_ids_(target_layer_ids, target_layer_ids + num_target_layers) {
        using namespace npu_target_detail;

        int fd = open(model_path, O_RDONLY);
        struct stat st; fstat(fd, &st);
        md_ = (uint8_t*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        md_size_ = st.st_size;
        close(fd);
        uint64_t hsz; memcpy(&hsz, md_, 8);
        df_ = 8 + hsz;
        const char* js = (const char*)(md_ + 8);
        size_t jl = hsz;
        emb_ = (const uint16_t*)(md_ + df_);
        auto i8p = [&](uint64_t o) { return md_ + df_ + o; };

        struct LO { uint64_t qp,kp,vp,op,gp,up,dp,in_off,pa_off,qn_off,kn_off; };
        std::vector<LO> lo(NC);
        char b[128];
        for (int l = 0; l < NC; l++) {
            snprintf(b,128,"model.layers.%d.self_attn.q_proj.weight",l); lo[l].qp=jo(js,jl,b);
            snprintf(b,128,"model.layers.%d.self_attn.k_proj.weight",l); lo[l].kp=jo(js,jl,b);
            snprintf(b,128,"model.layers.%d.self_attn.v_proj.weight",l); lo[l].vp=jo(js,jl,b);
            snprintf(b,128,"model.layers.%d.self_attn.o_proj.weight",l); lo[l].op=jo(js,jl,b);
            snprintf(b,128,"model.layers.%d.mlp.gate_proj.weight",l);    lo[l].gp=jo(js,jl,b);
            snprintf(b,128,"model.layers.%d.mlp.up_proj.weight",l);      lo[l].up=jo(js,jl,b);
            snprintf(b,128,"model.layers.%d.mlp.down_proj.weight",l);    lo[l].dp=jo(js,jl,b);
            snprintf(b,128,"model.layers.%d.input_layernorm.weight",l);  lo[l].in_off=jo(js,jl,b);
            snprintf(b,128,"model.layers.%d.post_attention_layernorm.weight",l); lo[l].pa_off=jo(js,jl,b);
            snprintf(b,128,"model.layers.%d.self_attn.q_norm.weight",l); lo[l].qn_off=jo(js,jl,b);
            snprintf(b,128,"model.layers.%d.self_attn.k_norm.weight",l); lo[l].kn_off=jo(js,jl,b);
        }
        uint64_t no = jo(js,jl,"model.norm.weight");
        uint64_t lo_off = jo(js,jl,"lm_head.weight");

        in_n_.assign(NC, std::vector<float>(H));
        pa_n_.assign(NC, std::vector<float>(H));
        qn_w_.assign(NC, std::vector<float>(HD));
        kn_w_.assign(NC, std::vector<float>(HD));
        for (int l = 0; l < NC; l++) {
            auto iw=(const uint16_t*)(md_+df_+lo[l].in_off), pw=(const uint16_t*)(md_+df_+lo[l].pa_off);
            auto qw=(const uint16_t*)(md_+df_+lo[l].qn_off), kw=(const uint16_t*)(md_+df_+lo[l].kn_off);
            for (int i=0;i<H;i++){in_n_[l][i]=bf16g(iw[i]); pa_n_[l][i]=bf16g(pw[i]);}
            for (int i=0;i<HD;i++){qn_w_[l][i]=bf16g(qw[i]); kn_w_[l][i]=bf16g(kw[i]);}
        }
        fin_.resize(H);
        { auto fw=(const uint16_t*)(md_+df_+no); for(int i=0;i<H;i++) fin_[i]=bf16g(fw[i]); }

        dev_ = std::make_unique<xrt::device>(0);
        char path[512];
        snprintf(path,512,"%s/final_i8_QKV_v.xclbin",xclbin_dir); char ipath[512];
        snprintf(ipath,512,"%s/insts_i8_QKV_v.txt",xclbin_dir);
        cq_ = {"QKV",XM,H,4096}; cq_.init(*dev_, path, ipath, 4, NC);
        snprintf(path,512,"%s/final_i8_O_v.xclbin",xclbin_dir); snprintf(ipath,512,"%s/insts_i8_O_v.txt",xclbin_dir);
        co_ = {"O",XM,NH*HD,H}; co_.init(*dev_, path, ipath, 4, NC);
        snprintf(path,512,"%s/final_i8_GU_v.xclbin",xclbin_dir); snprintf(ipath,512,"%s/insts_i8_GU_v.txt",xclbin_dir);
        cg_ = {"GU",XM,H,6144}; cg_.init(*dev_, path, ipath, 4, NC);
        snprintf(path,512,"%s/final_i8_D_v.xclbin",xclbin_dir); snprintf(ipath,512,"%s/insts_i8_D_v.txt",xclbin_dir);
        cd_ = {"D",XM,IM,H}; cd_.init(*dev_, path, ipath, 4, NC);

        wsc_.resize(NC);
        const int QOUT = NH*HD, KVOUT = NKV*HD;   // in_features=H (default dequant is correct)
        const int OOUT = H, OIN = NH*HD;          // in_features=2048, NOT the default 1024
        const int GUOUT = IM;                      // in_features=H (default dequant is correct)
        const int DOUT = H, DIN = IM;              // in_features=3072, NOT the default 1024
        for (int l = 0; l < NC; l++) {
            int qr,kr,vr,unused;
            float* qw=dequant_i8_to_float(i8p(lo[l].qp),256,&qr,&unused);
            float* kw=dequant_i8_to_float(i8p(lo[l].kp),128,&kr,&unused);
            float* vw=dequant_i8_to_float(i8p(lo[l].vp),128,&vr,&unused);
            int t=QOUT+KVOUT+KVOUT; std::vector<float> w((size_t)H*t);
            npu_target_detail::transpose_pack(qw, QOUT, H, w.data(), t, 0);
            npu_target_detail::transpose_pack(kw, KVOUT, H, w.data(), t, QOUT);
            npu_target_detail::transpose_pack(vw, KVOUT, H, w.data(), t, QOUT+KVOUT);
            cq_.packB(l, w.data(), H, t, wsc_[l].qk); free(qw); free(kw); free(vw);

            int or2,oc2;
            float* ow=dequant_i8_to_float_ex(i8p(lo[l].op),256,OIN,&or2,&oc2);
            std::vector<float> wo((size_t)OIN*OOUT);
            npu_target_detail::transpose_pack(ow, OOUT, OIN, wo.data(), OOUT, 0);
            co_.packB(l, wo.data(), OIN, OOUT, wsc_[l].o_); free(ow);

            int gr,ur;
            float* gw=dequant_i8_to_float(i8p(lo[l].gp),384,&gr,&unused);
            float* uw=dequant_i8_to_float(i8p(lo[l].up),384,&ur,&unused);
            int t2=GUOUT+GUOUT; std::vector<float> w2((size_t)H*t2);
            npu_target_detail::transpose_pack(gw, GUOUT, H, w2.data(), t2, 0);
            npu_target_detail::transpose_pack(uw, GUOUT, H, w2.data(), t2, GUOUT);
            cg_.packB(l, w2.data(), H, t2, wsc_[l].g_); free(gw); free(uw);

            int dr2,dc2;
            float* dw=dequant_i8_to_float_ex(i8p(lo[l].dp),384,DIN,&dr2,&dc2);
            std::vector<float> wd((size_t)DIN*DOUT);
            npu_target_detail::transpose_pack(dw, DOUT, DIN, wd.data(), DOUT, 0);
            cd_.packB(l, wd.data(), DIN, DOUT, wsc_[l].d_); free(dw);
        }

        // lm_head.weight is NOT tied to embed_tokens.weight for this model (separate storage,
        // separate quantization). See docs/V12-CORRECTNESS-BLOCKER.md.
        {
            int lr, lc;
            float* lm_raw = dequant_i8_to_float(i8p(lo_off), 18992, &lr, &lc);
            lm_head_f32_.resize((size_t)lr * lc);
            memcpy(lm_head_f32_.data(), lm_raw, (size_t)lr * lc * sizeof(float));
            free(lm_raw);
        }

        rope_cos_.resize(4096*HD); rope_sin_.resize(4096*HD);
        for (int p=0;p<4096;p++) for (int i=0;i<HD/2;i++) {
            float f=1.0f/powf(1000000.0f,(float)(2*i)/HD), a=p*f;
            rope_cos_[p*HD+2*i]=cosf(a); rope_sin_[p*HD+2*i]=sinf(a);
            rope_cos_[p*HD+2*i+1]=cosf(a); rope_sin_[p*HD+2*i+1]=sinf(a);
        }

        kv_.resize(NC);
        for (int l=0;l<NC;l++){ kv_[l].k.resize(4096*NKV*HD); kv_[l].v.resize(4096*NKV*HD); kv_[l].n=0; }
        layer_hidden_snapshot_.assign(NC, std::vector<float>(H));
    }

    // start_pos=0 => fresh generation (resets KV cache). start_pos>0 => continue from
    // existing KV cache (used for speculative-decode verification against past context).
    // logits_for_all_positions=false writes only the LAST position's logits into out_logits
    // (a single NV-sized buffer — used by forward()/prefill, which only needs the next-token
    // distribution). true writes n*NV (used by forward_with_kv()/verify, which needs every
    // draft position's distribution to check acceptance).
    void batch_forward(const int32_t* tokens, int n, int32_t start_pos,
                        float* out_logits /* NV or n*NV depending on logits_for_all_positions */,
                        float* out_hidden /* n*H, last-layer hidden per position */,
                        bool logits_for_all_positions = true) {
        using namespace npu_target_detail;
        if (start_pos == 0) for (auto& c : kv_) c.n = 0;
        int sp = start_pos;

        std::vector<float> h_b((size_t)n*H), res_b((size_t)n*H), qo_b((size_t)n*4096), at_b((size_t)n*NH*HD);
        std::vector<float> oo_b((size_t)n*H), gt_b((size_t)n*6144), su_b((size_t)n*IM), dw_b((size_t)n*H);
        for (int pi=0;pi<n;pi++) for (int i=0;i<H;i++) h_b[pi*H+i]=bf16g(emb_[tokens[pi]*H+i]);

        for (int l=0;l<NC;l++) {
            for (int pi=0;pi<n;pi++) memcpy(&res_b[pi*H], &h_b[pi*H], H*4);
            for (int pi=0;pi<n;pi++) rn_c(&h_b[pi*H], in_n_[l].data(), H);
            cq_.go(l, h_b.data(), n, H, npu_target_detail::dynamic_ascale(h_b.data(), n * H), wsc_[l].qk, qo_b.data(), 4096);
            cn(qo_b.data(), n*4096);
            const float* qn = qn_w_[l].data(); const float* kn = kn_w_[l].data();
            for (int pi=0;pi<n;pi++) {
                for (int hh=0;hh<NH;hh++) {
                    // QK RMSNorm FIRST, then RoPE (matching HuggingFace Qwen3 order)
                    double s=0; for (int d=0;d<HD;d++) s+=(double)qo_b[pi*4096+hh*HD+d]*qo_b[pi*4096+hh*HD+d];
                    float iq=1.0f/sqrtf((float)(s/HD)+1e-6f);
                    for (int d=0;d<HD;d++) qo_b[pi*4096+hh*HD+d]*=iq*qn[d];
                    apply_rope(&qo_b[pi*4096+hh*HD], sp+pi);
                }
                for (int kvh=0;kvh<NKV;kvh++) {
                    float* ks=&qo_b[pi*4096+2048+kvh*HD]; float* vs=&qo_b[pi*4096+3072+kvh*HD];
                    // QK RMSNorm FIRST, then RoPE (matching HuggingFace Qwen3 order)
                    double sk=0; for (int d=0;d<HD;d++) sk+=(double)ks[d]*ks[d];
                    float ik=1.0f/sqrtf((float)(sk/HD)+1e-6f);
                    for (int d=0;d<HD;d++) ks[d]*=ik*kn[d];
                    apply_rope(ks, sp+pi);
                    memcpy(&kv_[l].k[(size_t)(sp+pi)*NKV*HD+kvh*HD], ks, HD*4);
                    memcpy(&kv_[l].v[(size_t)(sp+pi)*NKV*HD+kvh*HD], vs, HD*4);
                }
            }
            kv_[l].n = sp+n; int cl = kv_[l].n;
            for (int pi=0;pi<n;pi++) for (int hh=0;hh<NH;hh++) {
                int kvh=hh/GQA; std::vector<float> ss(cl);
                for (int p=0;p<=sp+pi;p++) {
                    double s=0; for (int d=0;d<HD;d++) s+=(double)qo_b[pi*4096+hh*HD+d]*kv_[l].k[(size_t)p*NKV*HD+kvh*HD+d];
                    ss[p]=(float)(s/sqrtf((float)HD));
                }
                sm(ss.data(), sp+pi+1);
                for (int d=0;d<HD;d++) {
                    float s=0; for (int p=0;p<=sp+pi;p++) s+=ss[p]*kv_[l].v[(size_t)p*NKV*HD+kvh*HD+d];
                    at_b[pi*NH*HD+hh*HD+d]=s;
                }
            }
            co_.go(l, at_b.data(), n, NH*HD, npu_target_detail::dynamic_ascale(at_b.data(), n * NH * HD), wsc_[l].o_, oo_b.data(), H);
            cn(oo_b.data(), n*H);
            for (int pi=0;pi<n;pi++) for (int i=0;i<H;i++) h_b[pi*H+i] = res_b[pi*H+i] + oo_b[pi*H+i];
            for (int pi=0;pi<n;pi++) memcpy(&res_b[pi*H], &h_b[pi*H], H*4);
            for (int pi=0;pi<n;pi++) rn_c(&h_b[pi*H], pa_n_[l].data(), H);
            cg_.go(l, h_b.data(), n, H, npu_target_detail::dynamic_ascale(h_b.data(), n * H), wsc_[l].g_, gt_b.data(), 6144);
            cn(gt_b.data(), n*6144);
            for (int pi=0;pi<n;pi++) for (int i=0;i<IM;i++) {
                float gv=gt_b[pi*6144+i]; if(!std::isfinite(gv))gv=0;
                su_b[pi*IM+i]=(gv/(1.0f+expf(-gv)))*gt_b[pi*6144+IM+i];
            }
            cd_.go(l, su_b.data(), n, IM, npu_target_detail::dynamic_ascale(su_b.data(), n * IM), wsc_[l].d_, dw_b.data(), H);
            cn(dw_b.data(), n*H);
            for (int pi=0;pi<n;pi++) for (int i=0;i<H;i++) h_b[pi*H+i] = res_b[pi*H+i] + dw_b[pi*H+i];

            // Snapshot last position's hidden state at this layer, for MTP draft feature taps.
            memcpy(layer_hidden_snapshot_[l].data(), &h_b[(size_t)(n-1)*H], H*4);
        }

        if (out_hidden) memcpy(out_hidden, h_b.data(), (size_t)n*H*4);

        if (out_logits) {
            std::vector<float> sb(H);
            int start_pi = logits_for_all_positions ? 0 : (n - 1);
            for (int pi = start_pi; pi < n; pi++) {
                memcpy(sb.data(), &h_b[(size_t)pi*H], H*4);
                rn_c(sb.data(), fin_.data(), H);
                float* lg = out_logits + (size_t)(logits_for_all_positions ? pi : 0)*NV;
                for (int v=0; v<NV; v++) {
                    double s=0; const float* wrow = &lm_head_f32_[(size_t)v*H];
                    for (int kk=0;kk<H;kk++) s+=(double)sb[kk]*wrow[kk];
                    lg[v]=(float)s;
                }
            }
        }
    }

    // Prefill: only the next-token distribution is needed, so logits is a single NV-sized
    // buffer (not n*NV) — matches TargetModelInterface::forward's documented contract.
    void forward(const int32_t* input_ids, int32_t seq_len, float* logits, float* hidden_states) override {
        batch_forward(input_ids, seq_len, 0, logits, nullptr, /*logits_for_all_positions=*/false);
        get_layer_hidden_internal(hidden_states);
    }

    // Verify: every draft position's distribution is needed to check acceptance, so logits
    // must be n_tokens*NV (matches how spec_decode.h sizes its verify_logits buffer).
    void forward_with_kv(const int32_t* input_ids, int32_t n_tokens, int32_t past_len,
                          float* logits, float* hidden_states) override {
        batch_forward(input_ids, n_tokens, past_len, logits, nullptr, /*logits_for_all_positions=*/true);
        get_layer_hidden_internal(hidden_states);
    }

    void get_layer_hidden(const float* /*all_hidden*/, int32_t /*num_layers*/,
                           const int32_t* target_layer_ids, int32_t num_target_layers,
                           float* out) override {
        // Reads directly from the snapshot captured during the last batch_forward() call
        // (all_hidden param is unused — real per-layer states live in layer_hidden_snapshot_,
        // which the generic float* buffer in TargetModelInterface can't represent cleanly).
        for (int i = 0; i < num_target_layers; i++) {
            int layer = target_layer_ids[i];
            memcpy(out + (size_t)i*H, layer_hidden_snapshot_[layer].data(), H*4);
        }
    }

    // Accept the first `n_accept` positions of the most recent batch_forward() call,
    // discarding any later speculative positions that were rejected. Free: KV cache entries
    // past n_accept were written but kv_[l].n is simply not advanced past them.
    void commit_accepted(int32_t start_pos, int32_t n_accept) override {
        for (auto& c : kv_) c.n = start_pos + n_accept;
    }

private:
    struct WS { float qk, o_, g_, d_; };
    struct KVCache { std::vector<float> k, v; int n; };

    void get_layer_hidden_internal(float* hidden_states) {
        // Populate all NC layers into a flat buffer for callers expecting the full stack
        // (TargetModelInterface::forward's hidden_states arg is [num_layers, hidden_size]).
        if (!hidden_states) return;
        for (int l = 0; l < NC; l++) memcpy(hidden_states + (size_t)l*H, layer_hidden_snapshot_[l].data(), H*4);
    }

    void apply_rope(float* x, int pos) {
        for (int d=0; d<HD; d+=2) {
            float a=x[d], b=x[d+1];
            float c=rope_cos_[(size_t)pos*HD+d], s=rope_sin_[(size_t)pos*HD+d];
            x[d]=a*c-b*s; x[d+1]=b*c+a*s;
        }
    }

    uint8_t* md_ = nullptr; size_t md_size_ = 0; uint64_t df_ = 0;
    const uint16_t* emb_ = nullptr;
    std::vector<float> lm_head_f32_;
    std::vector<std::vector<float>> in_n_, pa_n_, qn_w_, kn_w_;
    std::vector<float> fin_;
    std::unique_ptr<xrt::device> dev_;
    npu_target_detail::I8Ctx cq_, co_, cg_, cd_;
    std::vector<WS> wsc_;
    std::vector<float> rope_cos_, rope_sin_;
    std::vector<KVCache> kv_;
    std::vector<std::vector<float>> layer_hidden_snapshot_;
    std::vector<int32_t> target_layer_ids_;
};
