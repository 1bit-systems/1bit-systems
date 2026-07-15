// tools/dspark_bench.cpp — DSpark speculative decoding benchmark
// Draft: CPU TRG (configurable layer count, fast)
// Verify: CPU TRG (all layers, batch verify)
//
// Build: g++ -O3 -march=native -fopenmp -std=c++17 -Iengine/fusion \
//        -o tools/dspark_bench tools/dspark_bench.cpp \
//        engine/fusion/cpu_layer.cpp -lm
//
// Run:   ./tools/dspark_bench model.q4nx [draft_layers=4] [M=8]

#include "cpu_layer.h"
#include "cpu_q4nx_loader.h"
#include "dspark.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

// ── FP32 → ternary packer ─────────────────────────────────────
static void pack(const float* src, uint32_t* pk, float* sc, int O, int I) {
    int pp = (I + 15) / 16;
    for (int r = 0; r < O; r++) {
        double sa = 0; int nz = 0;
        for (int c = 0; c < I; c++) { float v = src[r*I+c]; sa += fabs(v); if(v!=0)nz++; }
        sc[r] = (nz > 0) ? (float)(sa / nz) : 0;
        for (int u = 0; u < pp; u++) {
            uint32_t w = 0;
            for (int v = 0; v < 16; v++) {
                int c = u*16+v; float val = (c < I) ? src[r*I+c] : 0;
                w |= ((val>0?0x1u:(val<0?0x2u:0)) << (v*2));
            }
            pk[r*pp+u] = w;
        }
    }
}

// ── Loaded model ──────────────────────────────────────────────
struct Model {
    int H,IM,NH,NKV,HD,GQA,V,L;
    std::vector<float> emb, fn, lm;
    std::vector<float> in_norm, pa_norm, q_norm, k_norm;
    struct Ly { std::vector<uint32_t> q,k,v,o,g,u,d; std::vector<float> qs,ks,vs,os,gs,us,ds; };
    std::vector<Ly> layers;

    bool load(const char* path) {
        Q4nxModel qm; if (!qm.load(path)) return false;
        H=qm.hidden_dim; IM=qm.inter_size; NH=qm.n_heads; NKV=qm.n_kv_heads;
        HD=qm.head_dim; GQA=NH/NKV; V=qm.vocab_size; L=qm.n_layers;
        auto gt=[&](auto n){auto i=qm.tensors.find(n);return i==qm.tensors.end()?std::vector<float>():i->second.fp32;};
        emb=gt("model.embed_tokens.weight"); fn=gt("model.norm.weight"); lm=gt("lm_head.weight");
        if(lm.empty())lm=emb;
        in_norm.resize(L*H); pa_norm.resize(L*H); q_norm.resize(L*HD); k_norm.resize(L*HD);
        layers.resize(L);
        #pragma omp parallel for
        for(int l=0;l<L;l++){
            auto&ly=layers[l]; char k[256];
            auto gl=[&](auto f){snprintf(k,256,f,l);return gt(k);};
            auto inn=gl("model.layers.%d.input_layernorm.weight");
            auto pan=gl("model.layers.%d.post_attention_layernorm.weight");
            auto qn=gl("model.layers.%d.self_attn.q_norm.weight");
            auto kn=gl("model.layers.%d.self_attn.k_norm.weight");
            if(!inn.empty())memcpy(&in_norm[l*H],inn.data(),H*4);
            if(!pan.empty())memcpy(&pa_norm[l*H],pan.data(),H*4);
            if(!qn.empty())memcpy(&q_norm[l*HD],qn.data(),HD*4);
            if(!kn.empty())memcpy(&k_norm[l*HD],kn.data(),HD*4);
            auto pk=[&](auto n,auto&p,auto&s,int O,int I){
                auto w=gl(n); if(!w.empty()){ p.resize(O*((I+15)/16));s.resize(O);::pack(w.data(),p.data(),s.data(),O,I); }};
            pk("model.layers.%d.self_attn.q_proj.weight",ly.q,ly.qs,NH*HD,H);
            pk("model.layers.%d.self_attn.k_proj.weight",ly.k,ly.ks,NKV*HD,H);
            pk("model.layers.%d.self_attn.v_proj.weight",ly.v,ly.vs,NKV*HD,H);
            pk("model.layers.%d.self_attn.o_proj.weight",ly.o,ly.os,H,NH*HD);
            pk("model.layers.%d.mlp.gate_proj.weight",ly.g,ly.gs,IM,H);
            pk("model.layers.%d.mlp.up_proj.weight",ly.u,ly.us,IM,H);
            pk("model.layers.%d.mlp.down_proj.weight",ly.d,ly.ds,H,IM);
        }
        return true;
    }

    // Forward: runs layers [start, start+n) on hidden state
    void forward(float* hidden, int pos, int start, int n,
                 float* kc, float* vc,
                 float* sq, float* sa, float* sf, float* sa2,
                 float* sin_tab, float* cos_tab) {
        for (int l = start; l < start + n && l < L; l++) {
            auto& ly = layers[l];
            float res[4096]; memcpy(res, hidden, H*4);
            cpu_rmsnorm(hidden, &in_norm[l*H], hidden, H, 1e-6f);
            cpu_ternary_gemv(ly.q.data(),hidden,ly.qs.data(),sq,NH*HD,H);
            cpu_ternary_gemv(ly.k.data(),hidden,ly.ks.data(),sq+NH*HD,NKV*HD,H);
            cpu_ternary_gemv(ly.v.data(),hidden,ly.vs.data(),sq+NH*HD+NKV*HD,NKV*HD,H);
            for(int h=0;h<NH;h++)cpu_rmsnorm(sq+h*HD,&q_norm[l*HD],sq+h*HD,HD,1e-6f);
            cpu_rope(sq,pos,NH,HD,sin_tab,cos_tab);
            for(int h=0;h<NKV;h++)cpu_rmsnorm(sq+NH*HD+h*HD,&k_norm[l*HD],sq+NH*HD+h*HD,HD,1e-6f);
            cpu_rope(sq+NH*HD,pos,NKV,HD,sin_tab,cos_tab);
            for(int h=0;h<NKV;h++){
                memcpy(&kc[l*4096*NKV*HD+pos*NKV*HD+h*HD],sq+NH*HD+h*HD,HD*4);
                memcpy(&vc[l*4096*NKV*HD+pos*NKV*HD+h*HD],sq+NH*HD+NKV*HD+h*HD,HD*4);
            }
            cpu_attention(sq,&kc[l*4096*NKV*HD],&vc[l*4096*NKV*HD],sa,NH,NKV,HD,pos+1,GQA);
            cpu_ternary_gemv(ly.o.data(),sa,ly.os.data(),hidden,H,NH*HD);
            for(int i=0;i<H;i++)hidden[i]=res[i]+hidden[i];
            memcpy(res,hidden,H*4);
            cpu_rmsnorm(hidden,&pa_norm[l*H],hidden,H,1e-6f);
            cpu_ternary_gemv(ly.g.data(),hidden,ly.gs.data(),sf,IM,H);
            cpu_ternary_gemv(ly.u.data(),hidden,ly.us.data(),sf+IM,IM,H);
            cpu_silu_glu(sf,sf+IM,sa2,IM);
            cpu_ternary_gemv(ly.d.data(),sa2,ly.ds.data(),hidden,H,IM);
            for(int i=0;i<H;i++)hidden[i]=res[i]+hidden[i];
        }
    }

    void lm_head(const float* h, float* logits) {
        cpu_lm_head(h, lm.data(), logits, V, H);
    }
};

// ── C-callable wrappers for DSpark callbacks ──────────────────
struct Ctx { Model* m; float* kc; float* vc; float* sq; float* sa; float* sf; float* sa2; float* st; float* ct; };

extern "C" {
static void draft_forward(void* ctx, int start, int n, float* h, int pos,
                           float* kc, float* vc, float* sq, float* sa,
                           float* sf, float* sa2, float* st, float* ct,
                           const float* in, const float* pa,
                           const float* qn, const float* kn, const float* fn,
                           const void** lw, int H, int IM, int NH, int NKV, int HD, int GQA) {
    auto c = (Ctx*)ctx;
    c->m->forward(h, pos, start, n, c->kc, c->vc, c->sq, c->sa, c->sf, c->sa2, c->st, c->ct);
}
static float* draft_lm(void* ctx, const float* h, float* logits, int V, int H) {
    ((Ctx*)ctx)->m->lm_head(h, logits); return logits;
}
static void verify_forward(void* ctx, int start, int n, float* h, int pos,
                            float* kc, float* vc, float* sq, float* sa,
                            float* sf, float* sa2, float* st, float* ct,
                            const float* in, const float* pa,
                            const float* qn, const float* kn, const float* fn,
                            const void** lw, int H, int IM, int NH, int NKV, int HD, int GQA) {
    auto c = (Ctx*)ctx;
    c->m->forward(h, pos, start, n, c->kc, c->vc, c->sq, c->sa, c->sf, c->sa2, c->st, c->ct);
}
static float* verify_lm(void* ctx, const float* h, float* logits, int V, int H) {
    ((Ctx*)ctx)->m->lm_head(h, logits); return logits;
}
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1]
        : std::string(getenv("HOME") ? getenv("HOME") : "/tmp") +"/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    int draft_layers = argc > 2 ? atoi(argv[2]) : 4;
    int M = argc > 3 ? atoi(argv[3]) : 8;

    printf("=== DSpark Speculative Decode Benchmark ===\n\n");
    printf("  Draft layers: %d\n", draft_layers);
    printf("  Verifier layers: full (%d)\n", 28);
    printf("  Candidates per round (M): %d\n\n", M);

    Model m;
    if (!m.load(path)) return 1;
    const int H=m.H, IM=m.IM, NH=m.NH, NKV=m.NKV, HD=m.HD, L=m.L, V=m.V, GQA=m.GQA;

    // RoPE tables
    std::vector<float> st(4096*HD), ct(4096*HD);
    for (int p = 0; p < 4096; p++)
        for (int d = 0; d < HD; d++) {
            float th = p / pow(10000.0f, (2.0f*(d/2))/HD);
            st[p*HD+d]=sin(th); ct[p*HD+d]=cos(th);
        }

    // Scratch
    std::vector<float> hidden(H), sq(NH*HD+2*NKV*HD), sa(NH*HD), sf(2*IM), sa2(IM);
    std::vector<float> kc_d(L*4096*NKV*HD,0), vc_d(L*4096*NKV*HD,0);
    std::vector<float> kc_f(L*4096*NKV*HD,0), vc_f(L*4096*NKV*HD,0);
    std::vector<float> logits(V);

    // Prefill first token
    memcpy(hidden.data(), m.emb.data()+1*H, H*4);
    m.forward(hidden.data(), 0, 0, L, kc_f.data(), vc_f.data(),
              sq.data(), sa.data(), sf.data(), sa2.data(), st.data(), ct.data());

    m.lm_head(hidden.data(), logits.data());
    int cur_tok = 0;
    for (int i = 1; i < V; i++) if (logits[i] > logits[cur_tok]) cur_tok = i;
    printf("  First token: %d\n\n", cur_tok);

    // ── Measure draft-only speed (draft_layers) ──
    printf("── 1. Draft-only (%d layers) ──\n", draft_layers);
    int warmup = 10, runs = 50;
    for (int i = 0; i < warmup; i++) {
        memcpy(hidden.data(), m.emb.data()+1*H, H*4);
        m.forward(hidden.data(), i, 0, draft_layers, kc_d.data(), vc_d.data(),
                  sq.data(), sa.data(), sf.data(), sa2.data(), st.data(), ct.data());
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; i++) {
        memcpy(hidden.data(), m.emb.data()+1*H, H*4);
        m.forward(hidden.data(), i, 0, draft_layers, kc_d.data(), vc_d.data(),
                  sq.data(), sa.data(), sf.data(), sa2.data(), st.data(), ct.data());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double draft_ms = std::chrono::duration<double,std::milli>(t1-t0).count() / runs;
    double draft_tok_s = 1000.0 / draft_ms;
    printf("  %.2f ms/tok  = %.0f tok/s\n", draft_ms, draft_tok_s);

    // ── Measure verify-only (full model) ──
    printf("\n── 2. Verify-only (%d layers, single) ──\n", L);
    for (int i = 0; i < warmup; i++) {
        memcpy(hidden.data(), m.emb.data()+1*H, H*4);
        m.forward(hidden.data(), i, 0, L, kc_f.data(), vc_f.data(),
                  sq.data(), sa.data(), sf.data(), sa2.data(), st.data(), ct.data());
    }
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; i++) {
        memcpy(hidden.data(), m.emb.data()+1*H, H*4);
        m.forward(hidden.data(), i, 0, L, kc_f.data(), vc_f.data(),
                  sq.data(), sa.data(), sf.data(), sa2.data(), st.data(), ct.data());
    }
    t1 = std::chrono::high_resolution_clock::now();
    double verify_ms = std::chrono::duration<double,std::milli>(t1-t0).count() / runs;
    double verify_tok_s = 1000.0 / verify_ms;
    printf("  %.2f ms/tok  = %.0f tok/s\n", verify_ms, verify_tok_s);

    // ── DSpark simulation ──
    printf("\n── 3. DSpark (M=%d, acceptance=%.0f%%) ──\n", M, 0.75*100.0);
    double acceptance = 0.75; // typical spec-decode acceptance rate
    int n_rounds = 50;
    
    // Simulate: each round generates M draft tokens, verifies all at once
    // For simplicity: measure 1 draft pass + 1 verify pass per round
    // In reality: draft generates M tokens sequentially (M * draft_ms),
    // then verify runs once (verify_ms). Net per round:
    //   time = M * draft_ms + verify_ms
    //   tokens = 1 + acceptance * M (1 verified prefix token + accepted drafts)
    double draft_seq_ms = M * draft_ms;
    double round_ms = draft_seq_ms + verify_ms;
    double tok_per_round = 1.0 + acceptance * M;
    double dspark_tok_s = tok_per_round / (round_ms / 1000.0);
    double speedup = dspark_tok_s / verify_tok_s;
    
    printf("  Draft %d tokens sequentially: %.1f ms\n", M, draft_seq_ms);
    printf("  Verify batch:                 %.1f ms\n", verify_ms);
    printf("  Per round:                    %.1f ms\n", round_ms);
    printf("  Tokens per round:             %.1f\n", tok_per_round);
    printf("  Effective throughput:         %.0f tok/s\n", dspark_tok_s);
    printf("  Speedup vs verify-only:       %.1fx\n", speedup);

    // ── Parameter sweep ──
    printf("\n── 4. Sweep: draft_layers vs M ──\n");
    printf("  %10s %10s %12s %10s %8s\n", "Draft L", "M", "ms/round", "tok/s", "speedup");
    int draft_sweep[] = {2, 4, 8, 12, 16, 20};
    int m_sweep[] = {4, 8, 16};
    
    for (int dl : draft_sweep) {
        if (dl > L) continue;
        // Measure draft speed for this layer count
        double dl_ms;
        {
            for (int i = 0; i < warmup; i++) {
                memcpy(hidden.data(), m.emb.data()+1*H, H*4);
                m.forward(hidden.data(), i, 0, dl, kc_d.data(), vc_d.data(),
                          sq.data(), sa.data(), sf.data(), sa2.data(), st.data(), ct.data());
            }
            auto x0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < runs; i++) {
                memcpy(hidden.data(), m.emb.data()+1*H, H*4);
                m.forward(hidden.data(), i, 0, dl, kc_d.data(), vc_d.data(),
                          sq.data(), sa.data(), sf.data(), sa2.data(), st.data(), ct.data());
            }
            auto x1 = std::chrono::high_resolution_clock::now();
            dl_ms = std::chrono::duration<double,std::milli>(x1-x0).count() / runs;
        }
        double dl_tok_s = 1000.0 / dl_ms;
        
        for (int mm : m_sweep) {
            double ds_ms = mm * dl_ms;          // sequential draft time
            double r_ms = ds_ms + verify_ms;    // total round time
            double tpr = 1.0 + acceptance * mm;  // tokens per round
            double ts = tpr / (r_ms / 1000.0);  // tok/s
            double sp = ts / verify_tok_s;       // speedup
            printf("  L=%5d  M=%5d  %8.1f ms  %7.0f  %5.1fx\n", dl, mm, r_ms, ts, sp);
        }
    }

    // ── Best configuration ──
    printf("\n── 5. Best configuration ──\n");
    double best_sp = 0; int best_dl = 0, best_m = 0;
    for (int dl : draft_sweep) {
        if (dl > L) continue;
        double dl_ms;
        {
            memcpy(hidden.data(), m.emb.data()+1*H, H*4);
            m.forward(hidden.data(), 0, 0, dl, kc_d.data(), vc_d.data(),
                      sq.data(), sa.data(), sf.data(), sa2.data(), st.data(), ct.data());
            auto x0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 20; i++) {
                m.forward(hidden.data(), i, 0, dl, kc_d.data(), vc_d.data(),
                          sq.data(), sa.data(), sf.data(), sa2.data(), st.data(), ct.data());
            }
            auto x1 = std::chrono::high_resolution_clock::now();
            dl_ms = std::chrono::duration<double,std::milli>(x1-x0).count() / 20;
        }
        for (int mm : m_sweep) {
            double ds_ms = mm * dl_ms;
            double r_ms = ds_ms + verify_ms;
            double tpr = 1.0 + acceptance * mm;
            double ts = tpr / (r_ms / 1000.0);
            double sp = ts / verify_tok_s;
            if (sp > best_sp) { best_sp = sp; best_dl = dl; best_m = mm; }
        }
    }
    printf("  Draft L=%d, M=%d: %.1fx speedup = %.0f tok/s\n",
           best_dl, best_m, best_sp,
           verify_tok_s * best_sp);

    printf("\n=== DSpark Benchmark Complete ===\n");
    return 0;
}
