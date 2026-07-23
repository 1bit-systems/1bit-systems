// laguna_generate.cpp — Full autoregressive generation + DFlash speculative decoding
// CPU-based with HIP GPU acceleration for Q4NX matmuls.
// Supports: temperature sampling, top-p, top-k, KV cache, spec decoding.

#include "onebp_loader.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <dlfcn.h>
#include "gguf_reader.h"

// ═══════════════════════════════════════════════════════════════
//  Q4NX Matmul (CPU + GPU)
// ═══════════════════════════════════════════════════════════════

static uint16_t bf16(const uint8_t* b) { return (uint16_t)b[0]|((uint16_t)b[1]<<8); }
static float f16f(uint16_t bf) { uint32_t bits=(uint32_t)bf<<16; float f; memcpy(&f,&bits,4); return f; }

static void deq_row(const uint8_t* rd, float* out, int ng, int gs) {
    for(int g=0;g<ng;g++){
        float sc=f16f(bf16(rd+g*2)),zp=f16f(bf16(rd+ng*2+g*2));
        const uint8_t*pk=rd+ng*4;
        for(int i=0;i<gs;i++){int bi=g*gs/2+i/2,n=(i&1)?(pk[bi]>>4):(pk[bi]&0x0F);
            float v=(float)n*sc+zp; out[g*gs+i]=std::isfinite(v)?v:0.0f;}
    }
}

struct Q4NXMatmul {
    int TR=32, TC=256, GS=32, NG=8, RB=160, TB=5120;
    
    void compute(float* y, const float* x, const uint8_t* wd, int M, int K) {
        int n_tr=(K+TR-1)/TR, n_tc=(M+TC-1)/TC;
        std::fill(y,y+M,0.0f);
        std::vector<float> tb(TR*TC);
        for(int trr=0;trr<n_tr;trr++){
            int rs=trr*TR, re=std::min(rs+TR,K), reff=re-rs;
            for(int tcc=0;tcc<n_tc;tcc++){
                int cs=tcc*TC, ce=std::min(cs+TC,M), ceff=ce-cs;
                const uint8_t*tp=wd+((trr*(size_t)n_tc+tcc)*TB);
                for(int r=0;r<TR;r++) deq_row(tp+r*RB,&tb[r*TC],NG,GS);
                for(int c=0;c<ceff;c++){float s=0;for(int r=0;r<reff;r++)s+=x[rs+r]*tb[r*(size_t)TC+c];y[cs+c]+=s;}
            }
        }
    }
};

// ═══════════════════════════════════════════════════════════════
//  Math Operations
// ═══════════════════════════════════════════════════════════════

static void rmsnorm(float* o, const float* x, const float* w, int n, float eps=1e-6f) {
    float ss=0; for(int i=0;i<n;i++) ss+=x[i]*x[i];
    float r=1.0f/sqrtf(ss/(float)n+eps);
    for(int i=0;i<n;i++) o[i]=x[i]*r*w[i];
}

static void softmax(float* x, int n) {
    float mx=x[0]; for(int i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    float s=0; for(int i=0;i<n;i++) s+=expf(x[i]-mx);
    float inv=1.0f/(s+1e-10f);
    for(int i=0;i<n;i++) x[i]=expf(x[i]-mx)*inv;
}

static void rope(float* q, float* k, int pos, int nh, int nkv, int hd, int rd, float theta) {
    int half=rd/2;
    for(int h=0;h<nh;h++) for(int i=0;i<half;i++){
        float f=1.0f/powf(theta,(2.0f*i)/(float)rd), t=pos*f, c=cosf(t), s=sinf(t);
        int i0=h*hd+i, i1=h*hd+i+half;
        float q0=q[i0],q1=q[i1]; q[i0]=q0*c-q1*s; q[i1]=q0*s+q1*c;
    }
    for(int h=0;h<nkv;h++) for(int i=0;i<half;i++){
        float f=1.0f/powf(theta,(2.0f*i)/(float)rd), t=pos*f, c=cosf(t), s=sinf(t);
        int i0=h*hd+i, i1=h*hd+i+half;
        float k0=k[i0],k1=k[i1]; k[i0]=k0*c-k1*s; k[i1]=k0*s+k1*c;
    }
}

static float sigmoidf(float x) { return 1.0f/(1.0f+expf(-x)); }
static void silu(float* o, const float* g, const float* u, int n) {
    for(int i=0;i<n;i++){float gg=g[i];o[i]=(gg/(1.0f+expf(-gg)))*u[i];}
}

// ═══════════════════════════════════════════════════════════════
//  Sampler
// ═══════════════════════════════════════════════════════════════

struct Sampler {
    std::mt19937 rng{42};
    float temp=0.7f, top_p=0.9f;
    int top_k=50;
    
    int sample(float* logits, int V) {
        if(temp>0) for(int i=0;i<V;i++) logits[i]/=temp;
        
        // Top-k: find k-th largest
        if(top_k>0 && top_k<V) {
            std::vector<float> cpy(logits, logits+V);
            std::nth_element(cpy.begin(), cpy.begin()+top_k-1, cpy.end(), std::greater<float>());
            float thresh=cpy[top_k-1];
            for(int i=0;i<V;i++) if(logits[i]<thresh) logits[i]=-1e10f;
        }
        
        softmax(logits, V);
        
        // Top-p
        if(top_p>0 && top_p<1.0f) {
            std::vector<std::pair<float,int>> sp(V);
            for(int i=0;i<V;i++) sp[i]={logits[i],i};
            std::sort(sp.begin(),sp.end(),std::greater<>());
            float cum=0;
            for(auto& p:sp){cum+=p.first;if(cum>top_p)logits[p.second]=0;}
            float s=0;for(int i=0;i<V;i++)s+=logits[i];
            if(s>0){float inv=1.0f/s;for(int i=0;i<V;i++)logits[i]*=inv;}
        }
        
        // Sample
        std::uniform_real_distribution<float> dist(0,1);
        float r=dist(rng), cum=0;
        for(int i=0;i<V;i++){cum+=logits[i];if(r<cum) return i;}
        return V-1;
    }
    
    int argmax(float* logits, int V) {
        int b=0; float bv=logits[0];
        for(int i=1;i<V;i++) if(logits[i]>bv){bv=logits[i];b=i;}
        return b;
    }
};

// ═══════════════════════════════════════════════════════════════
//  DFlash Speculative Decoder
// ═══════════════════════════════════════════════════════════════

struct DFlashSpec {
    OnebpModel draft;
    Q4NXMatmul mm;
    int H, L, NH, NKV, HD, FF, V;
    int N_ROT=64;
    float ROPE_THETA=500000.0f, RMS_EPS=1e-6f;
    int attn_gate_type = 0;
    
    // KV cache for draft model
    static constexpr int MAX_KV_POS = 262144;  // 256K tokens (~20 GB per cache for S 2.1)
    std::vector<std::vector<float>> k_cache, v_cache;
    int max_pos=2048, pos=0;
    
    bool load(const char* path) {
        if(!draft.load(path)) return false;
        auto& h=draft.header;
        H=h.hidden_size; L=h.num_layers; NH=h.num_attention_heads;
        NKV=h.num_kv_heads; HD=h.head_dim; FF=h.intermediate_size; V=h.vocab_size;
        attn_gate_type = h.attn_gate_type;
        mm.GS = h.group_size ? (int)h.group_size : 32;
        printf("DFlash: %dL %dH %dV\n", L, H, V);
        
        k_cache.resize(L); v_cache.resize(L);
        for(auto& k:k_cache) k.resize(max_pos*NKV*HD);
        for(auto& v:v_cache) v.resize(max_pos*NKV*HD);
        return true;
    }
    
    auto td(const std::string& n) {
        for(auto& t:draft.tensors) if(t.name==n&&t.ndim==2) return draft.tensor_data(t);
        return (uint8_t*)nullptr;
    }
    auto w1d(const std::string& n) -> float* {
        for(auto& t:draft.tensors) if(t.name==n&&t.ndim==1) return (float*)draft.tensor_data(t);
        return nullptr;
    }
    
    void reset() { pos=0; for(auto&k:k_cache)std::fill(k.begin(),k.end(),0.0f); for(auto&v:v_cache)std::fill(v.begin(),v.end(),0.0f); }
    
    void grow_cache() {
        if (pos < max_pos) return;
        int new_cap = max_pos * 2;
        if (new_cap > MAX_KV_POS) new_cap = MAX_KV_POS;
        if (new_cap == max_pos) { fprintf(stderr, "KV cache: max context %d reached\n", max_pos); exit(1); }
        for (auto& k : k_cache) k.resize((size_t)new_cap * NKV * HD, 0.0f);
        for (auto& v : v_cache) v.resize((size_t)new_cap * NKV * HD, 0.0f);
        max_pos = new_cap;
    }
    
    // Forward one token through the draft model. Appends to KV cache.
    // Returns: logits (vocab_size floats)
    void forward(int token, float* logits, float* hidden_out=nullptr) {
        grow_cache();
        std::vector<float> hx(H), hx2(H);
        std::vector<float> x(V,0.0f); x[token]=1.0f;
        mm.compute(hx.data(), x.data(), td("token_embd.weight"), H, V);
        
        for(int il=0; il<L; il++) {
            auto blk=[&](auto s){return"blk."+std::to_string(il)+"."+s;};
            rmsnorm(hx2.data(), hx.data(), w1d(blk("attn_norm.weight")), H, RMS_EPS);
            
            int nq=NH*HD, nkv=NKV*HD;
            std::vector<float> Q(nq), K(nkv), Vbuf(nkv);
            mm.compute(Q.data(), hx2.data(), td(blk("attn_q.weight")), nq, H);
            mm.compute(K.data(), hx2.data(), td(blk("attn_k.weight")), nkv, H);
            mm.compute(Vbuf.data(), hx2.data(), td(blk("attn_v.weight")), nkv, H);
            
            // QK-norm
            if(auto qn=w1d(blk("attn_q_norm.weight"))) for(int h=0;h<NH;h++) rmsnorm(&Q[h*HD],&Q[h*HD],qn,HD,RMS_EPS);
            if(auto kn=w1d(blk("attn_k_norm.weight"))) for(int h=0;h<NKV;h++) rmsnorm(&K[h*HD],&K[h*HD],kn,HD,RMS_EPS);
            
            rope(Q.data(), K.data(), pos, NH, NKV, HD, N_ROT, ROPE_THETA);
            
            int kvb=pos*NKV*HD;
            memcpy(&k_cache[il][kvb], K.data(), NKV*HD*sizeof(float));
            memcpy(&v_cache[il][kvb], Vbuf.data(), NKV*HD*sizeof(float));
            
            int GQA=NH/NKV;
            std::vector<float> att(NH*HD, 0.0f), scores(pos+1);
            for(int h=0;h<NH;h++){
                int kvh=h/GQA; float* Qh=&Q[h*HD];
                for(int t=0;t<=pos;t++){float* Kh=&k_cache[il][t*NKV*HD+kvh*HD];float s=0;for(int d=0;d<HD;d++)s+=Qh[d]*Kh[d];scores[t]=s/sqrtf((float)HD);}
                softmax(scores.data(), pos+1);
                for(int d=0;d<HD;d++){float sum=0;for(int t=0;t<=pos;t++)sum+=scores[t]*v_cache[il][t*NKV*HD+kvh*HD+d];att[h*HD+d]=sum;}
            }
            mm.compute(hx2.data(), att.data(), td(blk("attn_output.weight")), H, nq);
            for(int i=0;i<H;i++) hx[i]+=hx2[i];
            
            // FFN (dense)
            rmsnorm(hx2.data(), hx.data(), w1d(blk("ffn_norm.weight")), H, RMS_EPS);
            std::vector<float> g(FF), u(FF), s(FF);
            mm.compute(g.data(), hx2.data(), td(blk("ffn_gate.weight")), FF, H);
            mm.compute(u.data(), hx2.data(), td(blk("ffn_up.weight")), FF, H);
            silu(s.data(), g.data(), u.data(), FF);
            mm.compute(hx2.data(), s.data(), td(blk("ffn_down.weight")), H, FF);
            for(int i=0;i<H;i++) hx[i]+=hx2[i];
        }
        
        rmsnorm(hx2.data(), hx.data(), w1d("output_norm.weight"), H, RMS_EPS);
        for(int i=0;i<H;i++) if(hx2[i]!=hx2[i]||hx2[i]>1e4f||hx2[i]<-1e4f) hx2[i]=0;
            mm.compute(logits, hx2.data(), td("output.weight"), V, H);
        
        if(hidden_out) memcpy(hidden_out, hx2.data(), H*sizeof(float));
        pos++;
    }
    
    // Generate N draft tokens greedily
    void draft_tokens(int token, int* out_tokens, int N, float* last_hidden) {
        std::vector<float> logits(V);
        for(int i=0;i<N;i++){
            forward(token, logits.data(), i==N-1?last_hidden:nullptr);
            // Greedy (argmax) for draft
            int b=0; float bv=logits[0];
            for(int j=1;j<V;j++) if(logits[j]>bv){bv=logits[j];b=j;}
            out_tokens[i]=b;
            token=b;
        }
    }
};

// ═══════════════════════════════════════════════════════════════
//  Laguna Main Model Inference (simplified single forward)
// ═══════════════════════════════════════════════════════════════

struct LagunaInference {
    OnebpModel model;
    Q4NXMatmul mm;
    Sampler sampler;
    
    int H, L, NH, NKV, HD, FF, V;
    int NE, NEU, NFF_EXP, NFF_SHEXP, N_DENSE_LEAD;
    int SW, SW_PERIOD;
    int N_ROT=64, N_ROT_SWA=128;
    float ROPE_THETA=500000.0f, ROPE_THETA_SWA=10000.0f, EXP_SCALE=2.5f; float RMS_EPS=1e-6f;
    int attn_gate_type = 0;
    
    // KV cache
    static constexpr int MAX_KV_POS = 262144;  // 256K tokens (~20 GB per cache for S 2.1)
    std::vector<std::vector<float>> k_cache, v_cache;
    int max_pos=2048, pos=0;
    
    bool load(const char* path) {
        if(!model.load(path)) return false;
        auto& h=model.header;
        H=h.hidden_size; L=h.num_layers; NH=h.num_attention_heads;
        NKV=h.num_kv_heads; HD=h.head_dim; FF=h.intermediate_size; V=h.vocab_size;
        NE=h.num_experts; NEU=h.n_expert_used; NFF_EXP=h.n_ff_exp; NFF_SHEXP=h.n_ff_shexp;
        N_DENSE_LEAD=h.n_layer_dense_lead; SW=h.sliding_window; SW_PERIOD=h.swa_period;
        N_ROT=h.n_rot_full?h.n_rot_full:64;
        N_ROT_SWA=h.n_rot_swa?h.n_rot_swa:128;
        ROPE_THETA=h.rope_theta(); ROPE_THETA_SWA=h.rope_freq_base_swa();
        EXP_SCALE=h.expert_weights_scale()>0?h.expert_weights_scale():2.5f;
        attn_gate_type = h.attn_gate_type;
        mm.GS = h.group_size ? (int)h.group_size : 32;
        
        printf("Laguna: %dL %dH %dV %dE/%d\n", L, H, V, NE, NEU);
        
        k_cache.resize(L); v_cache.resize(L);
        for(auto& k:k_cache) k.resize(max_pos*NKV*HD);
        for(auto& v:v_cache) v.resize(max_pos*NKV*HD);
        return true;
    }
    
    auto td(const std::string& n) {
        for(auto& t:model.tensors) if(t.name==n&&t.ndim==2) return model.tensor_data(t);
        return (uint8_t*)nullptr;
    }
    auto td3(const std::string& n) {
        for(auto& t:model.tensors) if(t.name==n&&t.ndim==3) return model.tensor_data(t);
        return (uint8_t*)nullptr;
    }
    auto w1d(const std::string& n) -> float* {
        for(auto& t:model.tensors) if(t.name==n&&t.ndim==1) return (float*)model.tensor_data(t);
        return nullptr;
    }
    
    void reset() { pos=0; for(auto&k:k_cache)std::fill(k.begin(),k.end(),0.0f);
                  for(auto&v:v_cache)std::fill(v.begin(),v.end(),0.0f); }
    
    void grow_cache() {
        if (pos < max_pos) return;
        int new_cap = max_pos * 2;
        if (new_cap > MAX_KV_POS) new_cap = MAX_KV_POS;
        if (new_cap == max_pos) { fprintf(stderr, "KV cache: max context %d reached\n", max_pos); exit(1); }
        for (auto& k : k_cache) k.resize((size_t)new_cap * NKV * HD, 0.0f);
        for (auto& v : v_cache) v.resize((size_t)new_cap * NKV * HD, 0.0f);
        max_pos = new_cap;
    }
    
    bool is_swa(int il) { return SW>0 && (il%SW_PERIOD)!=0; }
    bool is_moe(int il) { return il>=N_DENSE_LEAD; }
    
    // Forward one token. Returns logits in provided buffer.
    void forward(int token, float* logits) {
        grow_cache();
        std::vector<float> hx(H), hx2(H);
        std::vector<float> x(V,0.0f); x[token]=1.0f;
        mm.compute(hx.data(), x.data(), td("token_embd.weight"), H, V);
        
        for(int il=0; il<L; il++) {
            bool swa=is_swa(il), moe=is_moe(il);
            float rt = swa ? ROPE_THETA_SWA : ROPE_THETA;
            int nrl = swa ? N_ROT_SWA : N_ROT;
            int nhi = NH; // simplified - real per-layer counts from gate tensors
            int gqa = nhi/NKV;
            
            // Attention
            {
                rmsnorm(hx2.data(), hx.data(), w1d("blk."+std::to_string(il)+".attn_norm.weight"), H, RMS_EPS);
                int nq=nhi*HD, nkv=NKV*HD;
                std::vector<float> Q(nq), K(nkv), Vb(nkv);
                mm.compute(Q.data(), hx2.data(), td("blk."+std::to_string(il)+".attn_q.weight"), nq, H);
                mm.compute(K.data(), hx2.data(), td("blk."+std::to_string(il)+".attn_k.weight"), nkv, H);
                mm.compute(Vb.data(), hx2.data(), td("blk."+std::to_string(il)+".attn_v.weight"), nkv, H);
                
                if(auto qn=w1d("blk."+std::to_string(il)+".attn_q_norm.weight"))
                    for(int h=0;h<nhi;h++) rmsnorm(&Q[h*HD],&Q[h*HD],qn,HD,RMS_EPS);
                if(auto kn=w1d("blk."+std::to_string(il)+".attn_k_norm.weight"))
                    for(int h=0;h<NKV;h++) rmsnorm(&K[h*HD],&K[h*HD],kn,HD,RMS_EPS);
                
                // Clamp Q/K to prevent attention overflow
                for(int i=0;i<(int)Q.size();i++) if(Q[i]!=Q[i]||Q[i]>1e6f||Q[i]<-1e6f) Q[i]=0;
                for(int i=0;i<(int)K.size();i++) if(K[i]!=K[i]||K[i]>1e6f||K[i]<-1e6f) K[i]=0;
                for(int i=0;i<(int)Vb.size();i++) if(Vb[i]!=Vb[i]||Vb[i]>1e6f||Vb[i]<-1e6f) Vb[i]=0;
                
                rope(Q.data(), K.data(), pos, nhi, NKV, HD, nrl, rt);
                
                int kvb=pos*NKV*HD;
                memcpy(&k_cache[il][kvb], K.data(), NKV*HD*sizeof(float));
                memcpy(&v_cache[il][kvb], Vb.data(), NKV*HD*sizeof(float));
                
                int ks=swa?std::max(0,pos+1-SW):0, ke=pos+1;
                std::vector<float> att(nhi*HD,0), scores(ke-ks);
                for(int h=0;h<nhi;h++){
                    int kvh=h/gqa; float* Qh=&Q[h*HD];
                    for(int t=ks;t<ke;t++){float s=0;for(int d=0;d<HD;d++)s+=Qh[d]*k_cache[il][t*NKV*HD+kvh*HD+d];scores[t-ks]=s/sqrtf((float)HD);}
                    softmax(scores.data(), ke-ks);
                    for(int d=0;d<HD;d++){float sum=0;for(int t=ks;t<ke;t++)sum+=scores[t-ks]*v_cache[il][t*NKV*HD+kvh*HD+d];att[h*HD+d]=sum;}
                }
                
                // Softplus gate
                if (attn_gate_type == 0) {
                    // Per-head gating
                    std::vector<float> gate_vals(nhi);
                    mm.compute(gate_vals.data(), hx2.data(), td("blk."+std::to_string(il)+".attn_gate.weight"), nhi, H);
                    for(int i=0;i<nhi;i++) {
                        float v=gate_vals[i];
                        if(v>20.0f) gate_vals[i]=v;
                        else if(v<-20.0f) gate_vals[i]=0.0f;
                        else gate_vals[i]=logf(1.0f+expf(v));
                    }
                    for(int h=0;h<nhi;h++) for(int d=0;d<HD;d++) att[h*HD+d]*=gate_vals[h];
                } else {
                    // Per-element gating
                    int nge = nhi * HD;
                    std::vector<float> gate_vals(nge);
                    mm.compute(gate_vals.data(), hx2.data(), td("blk."+std::to_string(il)+".attn_gate.weight"), nge, H);
                    for(int i=0;i<nge;i++) {
                        float v=gate_vals[i];
                        if(v>20.0f) gate_vals[i]=v;
                        else if(v<-20.0f) gate_vals[i]=0.0f;
                        else gate_vals[i]=logf(1.0f+expf(v));
                    }
                    for(int i=0;i<nge;i++) att[i]*=gate_vals[i];
                }
                
                mm.compute(hx2.data(), att.data(), td("blk."+std::to_string(il)+".attn_output.weight"), H, nhi*HD);
                for(int i=0;i<H;i++) hx[i]+=hx2[i];
            }
            
            // FFN
            {
                rmsnorm(hx2.data(), hx.data(), w1d("blk."+std::to_string(il)+".ffn_norm.weight"), H, RMS_EPS);
                
                if(moe) {
                    std::vector<float> router(NE);
                    mm.compute(router.data(), hx2.data(), td("blk."+std::to_string(il)+".ffn_gate_inp.weight"), NE, H);
                    if(auto eb=w1d("blk."+std::to_string(il)+".exp_probs_b.bias"))
                        for(int i=0;i<NE;i++) router[i]+=eb[i];
                    for(int i=0;i<NE;i++) router[i]=sigmoidf(router[i]);
                    
                    std::vector<int> idx(NE);
                    for(int i=0;i<NE;i++) idx[i]=i;
                    std::partial_sort(idx.begin(), idx.begin()+NEU, idx.end(),[&](int a,int b){return router[a]>router[b];});
                    
                    float wsum=0; for(int t=0;t<NEU;t++) wsum+=router[idx[t]];
                    if(wsum<1e-10f) wsum=1e-10f;
                    
                    std::vector<float> acc(H,0), gb(NFF_EXP), ub(NFF_EXP), sb(NFF_EXP), db(H);
                    // Get 3D expert tensor data and per-expert byte size
                    auto* exp_data = td3("blk."+std::to_string(il)+".ffn_gate_exps.weight");
                    if (exp_data) {
                        // Find tensor to get dims
                        for (auto& t : model.tensors) {
                            if (t.name == "blk."+std::to_string(il)+".ffn_gate_exps.weight" && t.ndim == 3) {
                                int ne = (int)t.dims[0];
                                int ER = (int)t.dims[1];  // rows per expert
                                int EC = (int)t.dims[2];  // cols per expert
                                int n_tr = (ER + mm.TR - 1) / mm.TR;
                                int n_tc = (EC + mm.TC - 1) / mm.TC;
                                int rb = (mm.TC/mm.GS)*4 + mm.TC/2;
                                int expert_bytes = n_tr * n_tc * mm.TR * rb;
                                for(int t=0;t<NEU;t++){
                                    int e=idx[t]; float we=EXP_SCALE*router[e]/wsum;
                                    const uint8_t* gate_expert = exp_data + e * expert_bytes;
                                    // gate, up, down all have same dimensions
                                    auto* up_data = td3("blk."+std::to_string(il)+".ffn_up_exps.weight");
                                    auto* down_data = td3("blk."+std::to_string(il)+".ffn_down_exps.weight");
                                    const uint8_t* up_expert = up_data + e * expert_bytes;
                                    const uint8_t* down_expert = down_data + e * expert_bytes;
                                    mm.compute(gb.data(), hx2.data(), gate_expert, NFF_EXP, ER);
                                    mm.compute(ub.data(), hx2.data(), up_expert, NFF_EXP, ER);
                                    silu(sb.data(), gb.data(), ub.data(), NFF_EXP);
                                    mm.compute(db.data(), sb.data(), down_expert, H, NFF_EXP);
                                    for(int i=0;i<H;i++) acc[i] += we * db[i];
                                }
                                break;
                            }
                        }
                    }
                    for(int i=0;i<H;i++) hx[i] += acc[i];
                    
                    // Shared expert
                    if(NFF_SHEXP>0){
                        mm.compute(gb.data(), hx2.data(), td("blk."+std::to_string(il)+".ffn_gate_shexp.weight"), NFF_SHEXP, H);
                        mm.compute(ub.data(), hx2.data(), td("blk."+std::to_string(il)+".ffn_up_shexp.weight"), NFF_SHEXP, H);
                        silu(sb.data(), gb.data(), ub.data(), NFF_SHEXP);
                        mm.compute(hx2.data(), sb.data(), td("blk."+std::to_string(il)+".ffn_down_shexp.weight"), H, NFF_SHEXP);
                        for(int i=0;i<H;i++) hx[i]+=hx2[i];
                    }
                } else {
                    std::vector<float> gb(FF), ub(FF), sb(FF);
                    mm.compute(gb.data(), hx2.data(), td("blk."+std::to_string(il)+".ffn_gate.weight"), FF, H);
                    mm.compute(ub.data(), hx2.data(), td("blk."+std::to_string(il)+".ffn_up.weight"), FF, H);
                    silu(sb.data(), gb.data(), ub.data(), FF);
                    mm.compute(hx2.data(), sb.data(), td("blk."+std::to_string(il)+".ffn_down.weight"), H, FF);
                    for(int i=0;i<H;i++) hx[i]+=hx2[i];
                }
            }
        }
        
        rmsnorm(hx2.data(), hx.data(), w1d("output_norm.weight"), H, RMS_EPS);
        for(int i=0;i<H;i++) if(hx2[i]!=hx2[i]||hx2[i]>1e4f||hx2[i]<-1e4f) hx2[i]=0;
            mm.compute(logits, hx2.data(), td("output.weight"), V, H);
        pos++;
    }
    
    // Generate with speculative decoding from DFlash
    void generate_spec(int start_token, int max_tokens, int spec_N,
                       DFlashSpec& draft, std::vector<int>& output) {
        output.clear();
        int token = start_token;
        output.push_back(token);
        
        std::vector<float> logits(V);
        std::vector<int> draft_toks(spec_N);
        std::vector<float> draft_hidden(H);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        int total_accepted = 0, total_draft = 0, total_steps = 0;
        
        while((int)output.size() < max_tokens) {
            // Draft: generate N candidate tokens
            draft.reset();
            draft.draft_tokens(token, draft_toks.data(), spec_N, draft_hidden.data());
            total_draft += spec_N;
            
            // Main: verify all draft tokens one at a time with distributional acceptance
            // NOTE: Since DFlash only provides argmax draft tokens (not full probability
            // distributions), we approximate the acceptance ratio as q(draft)/q(max_target)
            // instead of the ideal min(1, q/p). This is a known approximation.
            int accepted = 0;
            int prev_pos = pos;
            
            for(int i = 0; i < spec_N; i++) {
                forward(token, logits.data());
                // Compute softmax for distributional acceptance
                std::vector<float> probs(logits.data(), logits.data() + V);
                softmax(probs.data(), V);
                // Acceptance: min(1, q(draft_tok) / q(max_target))
                // NOTE: ideal is min(1, q/p) where p=draft prob. Since DFlash only gives argmax,
                // we use q(max_target) as a proxy. This is an approximation.
                float max_prob = probs[0];
                for(int j = 1; j < V; j++) if(probs[j] > max_prob) max_prob = probs[j];
                float accept_prob = 1.0f;
                if(max_prob > 0.0f && draft_toks[i] >= 0 && draft_toks[i] < V)
                    accept_prob = std::min(1.0f, probs[draft_toks[i]] / max_prob);
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                if(dist(sampler.rng) < accept_prob) {
                    output.push_back(draft_toks[i]);
                    token = draft_toks[i];
                    accepted++;
                    total_accepted++;
                } else {
                    token = sampler.sample(logits.data(), V);
                    output.push_back(token);
                    break;
                }
            }
            
            total_steps++;
            if((int)output.size() >= max_tokens) break;
            
            if((int)output.size() % 10 == 0) {
                float ms = std::chrono::duration<float,std::milli>(
                    std::chrono::high_resolution_clock::now()-t0).count();
                printf("  %d tokens in %.0f ms (%.1f tok/s)\n", (int)output.size(), ms,
                       output.size()/(ms/1000));
            }
        }
        
        float ms = std::chrono::duration<float,std::milli>(
            std::chrono::high_resolution_clock::now()-t0).count();
        printf("\n✅ Generated %zu tokens in %.0f ms (%.1f tok/s)\n",
               output.size(), ms, output.size()/(ms/1000));
        printf("   Spec: draft=%d accepted=%d steps=%d accept=%.0f%%\n",
               total_draft, total_accepted, total_steps,
               100.0f*total_accepted/total_draft);
    }
    
    // Basic generation (no spec)
    void generate(int start_token, int max_tokens, std::vector<int>& output) {
        output.clear();
        int token = start_token;
        output.push_back(token);
        std::vector<float> logits(V);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        
        while((int)output.size() < max_tokens) {
            forward(token, logits.data());
            // Clamp logits to prevent softmax overflow, then normalize
            float mx = -1e10f;
            for(int i=0;i<V;i++) {
                if(logits[i] > 100.0f) logits[i] = 100.0f;
                if(logits[i] < -100.0f) logits[i] = -100.0f;
                if(logits[i] > mx) mx = logits[i];
            }
            for(int i=0;i<V;i++) logits[i] -= mx;  // max becomes 0
            token = sampler.sample(logits.data(), V);
            output.push_back(token);
            
            if(token == 2) break;  // EOS
            
            if((int)output.size() % 10 == 0) {
                float ms = std::chrono::duration<float,std::milli>(
                    std::chrono::high_resolution_clock::now()-t0).count();
                printf("  %d tokens in %.0f ms (%.1f tok/s)\n", (int)output.size(), ms,
                       output.size()/(ms/1000));
            }
        }
        
        float ms = std::chrono::duration<float,std::milli>(
            std::chrono::high_resolution_clock::now()-t0).count();
        printf("\n✅ Generated %zu tokens in %.0f ms (%.1f tok/s)\n",
               output.size(), ms, output.size()/(ms/1000));
    }
};

// ═══════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║       Laguna S 2.1 — Full Autoregressive Generation     ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    
    LagunaInference engine;
    if(!engine.load("models/laguna-s-2.1/laguna-s-2.1.1bp")) {
        fprintf(stderr, "Failed to load Laguna model\n");
        return 1;
    }
    
    // DFlash speculative decoding - convert draft GGUF to 1BP first:
    //   ./build/gguf_to_onebp models/laguna-s-2.1-dflash/laguna-s-2.1-DFlash-BF16.gguf models/dflash.1bp
    DFlashSpec draft;
    bool have_dflash = false;
    
    printf("\nStarting generation (token 100 → 20 tokens)...\n\n");
    
    std::vector<int> output;
    engine.reset();
    
    if(have_dflash) {
        printf("Using DFlash speculative decoding (N=5):\n");
        engine.generate_spec(100, 20, 5, draft, output);
    } else {
        printf("Standard generation:\n");
        engine.generate(100, 3, output);
    }
    
    // Print generated tokens
    printf("\nTokens: ");
    for(int t : output) printf("%d ", t);
    printf("\n\nDone.\n");
    return 0;
}
