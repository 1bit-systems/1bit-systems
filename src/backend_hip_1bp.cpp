// backend_hip_1bp.cpp — Fast GPU inference for 1BP models.
// Custom GEMV kernel replaces rocBLAS. All ops on-device.

#include "backend.h"
#include "../engine/npu/src/onebp_loader.cpp"
#include "rocm_cpp/ck_gemm.h"
#include <hip/hip_runtime_api.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <memory>

__global__ void rmsnorm_kernel(float*,const float*,int,float);
__global__ void rope_kernel(float*,int,int,float,int);
__global__ void silu_kernel(float*,const float*,const float*,int);
__global__ void add_kernel(float*,const float*,int);
__global__ void f2h_kernel(__half*,const float*,int);
__global__ void h2f_kernel(float*,const __half*,int);
__global__ void kv_store_kernel(__half*,__half*,const float*,const float*,int,int,int,int);
__global__ void embed_copy_kernel(float*,const float*,int,int);
__global__ void gemv_kernel(float* y, const float* W, const float* x, int M, int N);

// gemv_kernel defined in hip_1bp_kernels.hip
static constexpr float EPS = 1e-6f;

#define HIP_CHECK(call) \
    do { hipError_t _hip_e = (call); \
         if (_hip_e != hipSuccess) { \
             fprintf(stderr, "HIP error %s at %s:%d\n", \
                     hipGetErrorString(_hip_e), __FILE__, __LINE__); \
             std::abort(); } } while(0)
#define HIP_CHECK_D(call) \
    do { hipError_t _hip_e = (call); \
         if (_hip_e != hipSuccess) { \
             fprintf(stderr, "HIP error (dtor) %s at %s:%d\n", \
                     hipGetErrorString(_hip_e), __FILE__, __LINE__); } } while(0)

struct Hip1bpBackend : Backend {
    int H=0,NC=0,NH=0,NKV=0,HD=128,IM=0,VOCAB=0;
    float rope_theta=10000.0f; int max_seq=4096;
    hipStream_t stream=nullptr; bool gpu_ok=false;

    // GPU weights
    float *d_embed=nullptr,*d_final_norm=nullptr,*d_output=nullptr;
    struct GL{float*wq,*wk,*wv,*wo,*w1,*w2,*w3,*pn,*pon;};
    std::vector<GL> L;

    // GPU scratch (persistent, device-only)
    float *dh=nullptr,*datt=nullptr,*dgate=nullptr,*dup=nullptr;
    float *dsilu=nullptr,*doproj=nullptr,*dffn=nullptr,*datt2=nullptr;
    float *dlogits=nullptr; // [VOCAB] — pre-allocated lm_head output

    // KV cache (__half device)
    __half *dK=nullptr,*dV=nullptr,*dQ=nullptr,*dAttn=nullptr;
    size_t kvb=0; int pos=0;
    std::vector<float> cpu_final_norm;

    Hip1bpBackend(){type=BackendType::HIP_GPU;name="HIP 1BP GPU";}
    ~Hip1bpBackend()override{destroy();}
    bool can_infer()const override{return true;}

    bool init(const ModelConfig& cfg,const std::string&) override {
        this->cfg=cfg; H=cfg.hidden_size; NC=cfg.num_layers;
        NH=cfg.num_heads; NKV=cfg.num_kv_heads; HD=cfg.head_dim;
        IM=cfg.intermediate_size; VOCAB=cfg.vocab_size;
        rope_theta=cfg.rope_theta>0?cfg.rope_theta:10000.0f;
        if(NKV==0)NKV=NH; if(HD==0)HD=128;
        printf("[hip1bp] H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d V=%d\n",H,NC,NH,NKV,HD,IM,VOCAB);

        int nd=0;
        if(hipGetDeviceCount(&nd)!=hipSuccess||nd==0)return false;
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipStreamCreate(&stream));

        // Device-only allocations
        kvb=(size_t)max_seq*NKV*HD*sizeof(__half);
        HIP_CHECK(hipMalloc(&dK,kvb));  HIP_CHECK(hipMemset(dK,0,kvb));
        HIP_CHECK(hipMalloc(&dV,kvb));  HIP_CHECK(hipMemset(dV,0,kvb));
        size_t qb=(size_t)NH*HD*sizeof(__half);
        HIP_CHECK(hipMalloc(&dQ,qb));   HIP_CHECK(hipMalloc(&dAttn,qb));
        HIP_CHECK(hipMalloc(&dh,H*4));  HIP_CHECK(hipMalloc(&datt,NH*HD*4));
        HIP_CHECK(hipMalloc(&datt2,NH*HD*4));
        HIP_CHECK(hipMalloc(&dgate,NKV*HD*4)); HIP_CHECK(hipMalloc(&dup,NKV*HD*4));
        HIP_CHECK(hipMalloc(&dsilu,IM*4)); HIP_CHECK(hipMalloc(&doproj,H*4));
        HIP_CHECK(hipMalloc(&dffn,H*4));
        HIP_CHECK(hipMalloc(&dlogits,VOCAB*4));

        if(cfg.format!=ModelFormat::ONEBP||cfg.model_path.empty())return false;
        printf("[hip1bp] Loading: %s\n",cfg.model_path.c_str());
        OnebpModel mdl;
        if(!mdl.open(cfg.model_path.c_str()))return false;

        std::vector<float> emb,fn,out;
        auto ld=[&](const char* n,std::vector<float>& v){return mdl.get_tensor_f32(n,v);};
        if(!ld("token_embd.weight",emb))return false;
        if(!ld("output_norm.weight",fn))ld("token_embd_norm.weight",fn);
        if(!ld("output.weight",out))ld("lm_head.weight",out);
        cpu_final_norm=fn;

        auto up=[&](std::vector<float>& c,float*& g){
            if(c.empty()){g=nullptr;return true;}
            if(hipMalloc(&g,c.size()*4)!=hipSuccess)return false;
            HIP_CHECK(hipMemcpy(g,c.data(),c.size()*4,hipMemcpyHostToDevice));
            c.clear();c.shrink_to_fit();return true;
        };
        up(emb,d_embed);up(fn,d_final_norm);up(out,d_output);

        L.resize(NC);char buf[128];
        for(int l=0;l<NC;l++){
            auto& ll=L[l];std::vector<float> w;
            auto gr=[&](const char* bk,const char* lg,float*& gp,int n){
                snprintf(buf,sizeof(buf),"blk.%d.%s",l,bk);w.clear();
                if(!mdl.get_tensor_f32(buf,w)){
                    snprintf(buf,sizeof(buf),"model.layers.%d.%s",l,lg);
                    mdl.get_tensor_f32(buf,w);
                }
                if((int)w.size()==n){HIP_CHECK(hipMalloc(&gp,n*4));HIP_CHECK(hipMemcpy(gp,w.data(),n*4,hipMemcpyHostToDevice));}
                else gp=nullptr;
            };
            gr("attn_q.weight","self_attn.q_proj.weight",ll.wq,H*NH*HD);
            gr("attn_k.weight","self_attn.k_proj.weight",ll.wk,H*NKV*HD);
            gr("attn_v.weight","self_attn.v_proj.weight",ll.wv,H*NKV*HD);
            gr("attn_output.weight","self_attn.o_proj.weight",ll.wo,NH*HD*H);
            gr("ffn_gate.weight","mlp.gate_proj.weight",ll.w1,H*IM);
            gr("ffn_up.weight","mlp.up_proj.weight",ll.w2,H*IM);
            gr("ffn_down.weight","mlp.down_proj.weight",ll.w3,IM*H);
            gr("attn_norm.weight","input_layernorm.weight",ll.pn,H);
            gr("ffn_norm.weight","post_attention_layernorm.weight",ll.pon,H);
        }
        gpu_ok=true;initialized=true;
        printf("[hip1bp] ✅ GPU 1BP ready\n");
        return true;
    }

    bool reset()override{pos=0;HIP_CHECK(hipMemset(dK,0,kvb));HIP_CHECK(hipMemset(dV,0,kvb));return true;}

    bool forward(int token_id,float* hidden_out)override{
        if(!gpu_ok)return false;
        int H_=H,NH_=NH,NKV_=NKV,HD_=HD,IM_=IM,NC_=NC;
        int block=256;

        // Embedding
        if(token_id>=0&&token_id<VOCAB&&d_embed)
            embed_copy_kernel<<<(H_+block-1)/block,block,0,stream>>>(dh,d_embed,token_id,H_);
        else
            HIP_CHECK(hipMemset(dh,0,H_*4));

        for(int l=0;l<NC_;l++){
            auto& ll=L[l];

            // 1. Pre-attention RMSNorm
            if(ll.pn)rmsnorm_kernel<<<1,256,0,stream>>>(dh,ll.pn,H_,EPS);
            else rmsnorm_kernel<<<1,256,0,stream>>>(dh,nullptr,H_,EPS);

            // 2. QKV via custom GEMV (device-only, no CPU copies)
            if(ll.wq) gemv_kernel<<<NH_*HD_,256,0,stream>>>(datt,ll.wq,dh,NH_*HD_,H_);
            if(ll.wk) gemv_kernel<<<NKV_*HD_,256,0,stream>>>(dgate,ll.wk,dh,NKV_*HD_,H_);
            if(ll.wv) gemv_kernel<<<NKV_*HD_,256,0,stream>>>(dup,ll.wv,dh,NKV_*HD_,H_);

            // 3. RoPE
            if(ll.wq) rope_kernel<<<NH_,HD_/2,0,stream>>>(datt,HD_,pos,rope_theta,NH_);
            if(ll.wk) rope_kernel<<<NKV_,HD_/2,0,stream>>>(dgate,HD_,pos,rope_theta,NKV_);

            // 4. Attention — all on stream, no syncs needed
            if(ll.wo){
                f2h_kernel<<<(NH_*HD_+255)/256,256,0,stream>>>(dQ,datt,NH_*HD_);
                kv_store_kernel<<<NKV_,HD_,0,stream>>>(dK,dV,dgate,dup,pos,NKV_,HD_,max_seq);

                float scl=1.0f/sqrtf((float)HD_);
                rcpp_kv_cache_attn_decode(dQ,dK,dV,dAttn,NH_,NKV_,HD_,pos+1,scl,stream);

                // Use separate datt2 for attn output — avoids RAW hazard with datt (used by Q GEMV next layer)
                h2f_kernel<<<(NH_*HD_+255)/256,256,0,stream>>>(datt2,dAttn,NH_*HD_);
                gemv_kernel<<<H_,256,0,stream>>>(doproj,ll.wo,datt2,H_,NH_*HD_);
                add_kernel<<<(H_+255)/256,256,0,stream>>>(dh,doproj,H_);
            }

            // 5. Post-attention RMSNorm
            if(ll.pon)rmsnorm_kernel<<<1,256,0,stream>>>(dh,ll.pon,H_,EPS);
            else rmsnorm_kernel<<<1,256,0,stream>>>(dh,nullptr,H_,EPS);

            // 6. FFN (all on-device)
            if(ll.w1&&ll.w2&&ll.w3){
                gemv_kernel<<<IM_,256,0,stream>>>(dgate,ll.w1,dh,IM_,H_);
                gemv_kernel<<<IM_,256,0,stream>>>(dup,ll.w2,dh,IM_,H_);
                // Gemv on same stream → ordering guaranteed, no sync needed
                // Use datt2 for SiLU output (separate from dgate/dup which are used in Q/K GEMV next layer)
                silu_kernel<<<(IM_+255)/256,256,0,stream>>>(datt2,dgate,dup,IM_);
                gemv_kernel<<<H_,256,0,stream>>>(dffn,ll.w3,datt2,H_,IM_);
                add_kernel<<<(H_+255)/256,256,0,stream>>>(dh,dffn,H_);
            }
        }

        // Final RMSNorm
        rmsnorm_kernel<<<1,256,0,stream>>>(dh,d_final_norm,H_,EPS);
        HIP_CHECK(hipMemcpy(hidden_out,dh,H_*4,hipMemcpyDeviceToHost));
        pos++;
        return true;
    }

    bool lm_head(const float* hidden,float* logits,int* argmax)override{
        if(!d_output){memset(logits,0,VOCAB*4);logits[0]=1;if(argmax)*argmax=0;return true;}
        HIP_CHECK(hipMemcpy(dh,hidden,H*4,hipMemcpyHostToDevice));
        gemv_kernel<<<VOCAB,256,0,stream>>>(dlogits,d_output,dh,VOCAB,H);
        HIP_CHECK(hipMemcpy(logits,dlogits,VOCAB*4,hipMemcpyDeviceToHost));
        if(argmax){*argmax=0;for(int v=1;v<VOCAB;v++)if(logits[v]>logits[*argmax])*argmax=v;}
        return true;
    }

    int generate(int token_id)override{
        std::vector<float>hidden(H);
        if(!forward(token_id,hidden.data()))return-1;
        std::vector<float>logits(VOCAB);
        int n=-1;lm_head(hidden.data(),logits.data(),&n);return n;
    }

    float benchmark(int tokens)override{
        if(!initialized)return-1;reset();
        auto t0=std::chrono::steady_clock::now();int tok=1;
        for(int i=0;i<tokens;i++){tok=generate(tok);if(tok<0)break;}
        auto t1=std::chrono::steady_clock::now();
        return std::chrono::duration<float,std::milli>(t1-t0).count()/tokens;
    }

    void destroy()override{
        auto hf=[](void*p){if(p)HIP_CHECK_D(hipFree(p));};
        hf(d_embed);hf(d_final_norm);hf(d_output);
        for(auto&ll:L){hf(ll.wq);hf(ll.wk);hf(ll.wv);hf(ll.wo);hf(ll.w1);hf(ll.w2);hf(ll.w3);hf(ll.pn);hf(ll.pon);}
        L.clear();hf(dh);hf(datt);hf(dgate);hf(dup);hf(dsilu);hf(doproj);hf(dffn);hf(dlogits);
        hf(datt2);hf(dK);hf(dV);hf(dQ);hf(dAttn);
        if(stream){HIP_CHECK_D(hipStreamDestroy(stream));stream=nullptr;}
        gpu_ok=false;initialized=false;
    }
};

extern"C" Backend* create_hip_1bp_backend(){return static_cast<Backend*>(new Hip1bpBackend());}
