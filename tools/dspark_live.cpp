// tools/dspark_live.cpp — Live DSpark: real draft→verify→accept pipeline
// Measures actual acceptance rate and tok/s, not projections.
//
// Build: g++ -O3 -march=native -std=c++17 -Iengine/fusion \
//        -o tools/dspark_live tools/dspark_live.cpp engine/fusion/cpu_layer.cpp -lm
// Run:   ./tools/dspark_live model.trg [draft_layers=2] [M=8] [rounds=20]

#include "cpu_layer.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// ── Load .trg model ──────────────────────────────────────────
struct TrgModel {
    int H,IM,NH,NKV,HD,V,L,GQA,per_layer;
    const float *emb,*fn,*lm,*inorm,*pan,*qn,*kn,*sc;
    const uint32_t *pk;
    int rows[7],KK[7],ps[7];

    bool load(const char* path) {
        int fd=open(path,O_RDONLY);
        size_t fsz=lseek(fd,0,SEEK_END);
        auto p=(const char*)mmap(0,fsz,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
        if (!p||p==MAP_FAILED||memcmp(p,"TRG1",4)) return false;
        auto r4=[&](int o){uint32_t v;memcpy(&v,p+o,4);return(int)v;};
        auto r8=[&](int o){uint64_t v;memcpy(&v,p+o,8);return v;};
        H=r4(4);IM=r4(8);NH=r4(12);NKV=r4(16);HD=r4(20);V=r4(24);L=r4(28);GQA=r4(32);
        for(int i=7;i--;) ps[i]=r4(36+i*4);
        uint64_t o_emb=r8(64),o_fn=r8(72),o_lm=r8(80),o_norms=r8(88),o_pk=r8(96),o_sc=r8(104);
        auto F=[&](auto oo){return(const float*)(p+oo);};
        auto U=[&](auto oo){return(const uint32_t*)(p+oo);};
        emb=F(o_emb);fn=F(o_fn);lm=F(o_lm);
        inorm=F(o_norms);pan=F(o_norms+L*H*4);
        qn=F(o_norms+2*L*H*4);kn=F(o_norms+2*L*H*4+L*HD*4);
        pk=U(o_pk);sc=F(o_sc);
        per_layer=0; for(int i=7;i--;)per_layer+=ps[i];
        int r[7]={NH*HD,NKV*HD,NKV*HD,H,IM,IM,H},kk[7]={H,H,H,NH*HD,H,H,IM};
        for(int i=7;i--;){rows[i]=r[i];KK[i]=kk[i];}
        return true;
    }

    // Forward: runs n_layers of the model, updates hidden in place
    void forward(float* hd, int pos, int n_layers,
                 float* kc, float* vc,
                 float* st, float* ct,
                 float* qkv, float* at, float* ff, float* ac) {
        for(int l=0;l<n_layers;l++){
            std::vector<float> res(H); memcpy(res.data(),hd,H*4);
            auto pw=pk+l*per_layer;
            auto sw=sc+l*(NH*HD+NKV*HD+NKV*HD+H+IM+IM+H);
            cpu_rmsnorm(hd,inorm+l*H,hd,H,1e-6f);
            cpu_ternary_gemv(pw,hd,sw,qkv,rows[0],KK[0]);pw+=ps[0];sw+=NH*HD;
            cpu_ternary_gemv(pw,hd,sw,qkv+NH*HD,rows[1],KK[1]);pw+=ps[1];sw+=NKV*HD;
            cpu_ternary_gemv(pw,hd,sw,qkv+NH*HD+NKV*HD,rows[2],KK[2]);pw+=ps[2];sw+=NKV*HD;
            for(int hh=0;hh<NH;hh++) cpu_rmsnorm(qkv+hh*HD,qn+l*HD,qkv+hh*HD,HD,1e-6f);
            cpu_rope(qkv,pos,NH,HD,st,ct);
            for(int hh=0;hh<NKV;hh++) cpu_rmsnorm(qkv+NH*HD+hh*HD,kn+l*HD,qkv+NH*HD+hh*HD,HD,1e-6f);
            cpu_rope(qkv+NH*HD,pos,NKV,HD,st,ct);
            for(int hh=0;hh<NKV;hh++){
                memcpy(&kc[l*4096*NKV*HD+pos*NKV*HD+hh*HD],qkv+NH*HD+hh*HD,HD*4);
                memcpy(&vc[l*4096*NKV*HD+pos*NKV*HD+hh*HD],qkv+NH*HD+NKV*HD+hh*HD,HD*4);
            }
            cpu_attention(qkv,&kc[l*4096*NKV*HD],&vc[l*4096*NKV*HD],at,NH,NKV,HD,pos+1,GQA);
            cpu_ternary_gemv(pw,at,sw,hd,rows[3],KK[3]);pw+=ps[3];sw+=H;
            for(int i=0;i<H;i++)hd[i]=res[i]+hd[i];
            memcpy(res.data(),hd,H*4);
            cpu_rmsnorm(hd,pan+l*H,hd,H,1e-6f);
            cpu_ternary_gemv(pw,hd,sw,ff,rows[4],KK[4]);pw+=ps[4];sw+=IM;
            cpu_ternary_gemv(pw,hd,sw,ff+IM,rows[5],KK[5]);pw+=ps[5];sw+=IM;
            cpu_silu_glu(ff,ff+IM,ac,IM);
            cpu_ternary_gemv(pw,ac,sw,hd,rows[6],KK[6]);
            for(int i=0;i<H;i++)hd[i]=res[i]+hd[i];
        }
    }

    // Get logits from hidden state (after final norm on a copy)
    void get_logits(const float* hd, float* logits) {
        std::vector<float> tmp(H);
        memcpy(tmp.data(),hd,H*4);
        cpu_rmsnorm(tmp.data(),fn,tmp.data(),H,1e-6f);
        cpu_lm_head(tmp.data(),lm,logits,V,H);
    }

    // Get probability of a specific token
    float get_prob(const float* hd, int token) {
        std::vector<float> logits(V);
        get_logits(hd, logits.data());
        float max_v=logits[0]; for(int i=1;i<V;i++)if(logits[i]>max_v)max_v=logits[i];
        double sum=0,target=0;
        for(int i=0;i<V;i++){
            double e=exp((double)(logits[i]-max_v));
            sum+=e;
            if(i==token)target=e;
        }
        return (float)(target/sum);
    }

    // Greedy token from hidden state
    int argmax(const float* hd) {
        std::vector<float> logits(V);
        get_logits(hd, logits.data());
        int best=0; for(int i=1;i<V;i++)if(logits[i]>logits[best])best=i;
        return best;
    }
};

int main(int argc, char** argv) {
    const char* path = argc>1?argv[1]:"/tmp/model.trg";
    int draft_L = argc>2?atoi(argv[2]):2;
    int M = argc>3?atoi(argv[3]):8;
    int n_rounds = argc>4?atoi(argv[4]):20;

    TrgModel m;
    if (!m.load(path)) return fprintf(stderr,"load failed\n"),1;
    printf("=== DSpark Live ===\n");
    printf("  Model: H=%d L=%d, Draft %dL, M=%d, %d rounds\n\n",
           m.H,m.L,draft_L,M,n_rounds);

    // Buffers
    std::vector<float> hd(m.H),qkv(m.NH*m.HD+2*m.NKV*m.HD),at(m.NH*m.HD);
    std::vector<float> ff(2*m.IM),ac(m.IM);
    std::vector<float> st(4096*m.HD),ct(4096*m.HD);
    for(int p=0;p<4096;p++)for(int d=0;d<m.HD;d++){
        float th=p/pow(10000.f,(2.f*(d/2))/m.HD);
        st[p*m.HD+d]=sin(th);ct[p*m.HD+d]=cos(th);
    }

    // One KV cache shared between draft and verify (both use same model layers)
    auto kc_buf = std::vector<float>(m.L*4096*m.NKV*m.HD,0.0f);
    auto vc_buf = std::vector<float>(m.L*4096*m.NKV*m.HD,0.0f);
    float* kc = kc_buf.data();
    float* vc = vc_buf.data();

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> uniform(0,1);

    int total_draft = 0, total_accepted = 0;
    double total_draft_ms = 0, total_verify_ms = 0;
    int pos = 0;

    // Start with first token
    float hd_saved[4096];
    memcpy(hd_saved, m.emb+1*m.H, m.H*4);
    memcpy(hd.data(), hd_saved, m.H*4);

    // Warmup
    m.forward(hd.data(), 0, m.L, kc, vc,
              st.data(), ct.data(), qkv.data(), at.data(), ff.data(), ac.data());
    int prev_token = m.argmax(hd.data());
    pos = 1;

    printf("── Results ──\n");
    for(int round=0;round<n_rounds;round++){
        // ── Draft phase: generate M tokens using draft model ──
        float draft_hd[4096];
        memcpy(draft_hd, hd_saved, m.H*4);
        int draft_tokens[32];
        float draft_probs[32];

        auto t0 = std::chrono::high_resolution_clock::now();
        for(int i=0;i<M;i++){
            m.forward(draft_hd, pos+i, draft_L, kc, vc,
                      st.data(), ct.data(), qkv.data(), at.data(), ff.data(), ac.data());
            draft_tokens[i] = m.argmax(draft_hd);
            draft_probs[i] = m.get_prob(draft_hd, draft_tokens[i]);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double draft_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        total_draft_ms += draft_ms;

        // ── Verify phase: run full model for each candidate sequentially ──
        // Each candidate builds on the previous (verified prefix + accepted drafts)
        float verify_hd[4096];
        memcpy(verify_hd, hd_saved, m.H*4);
        int n_accepted = 0;

        auto t2 = std::chrono::high_resolution_clock::now();
        for(int i=0;i<M;i++){
            // Run full model forward from current accumulated state
            m.forward(verify_hd, pos+i, m.L, kc, vc,
                      st.data(), ct.data(), qkv.data(), at.data(), ff.data(), ac.data());

            // Get target probability of draft token at this position
            float target_prob = m.get_prob(verify_hd, draft_tokens[i]);

            // Acceptance threshold: min(1, p_target / p_draft)
            float threshold = (draft_probs[i] > 1e-10f) ? target_prob / draft_probs[i] : 1.0f;
            if (threshold > 1.0f) threshold = 1.0f;
            float r = uniform(rng);

            if (r <= threshold) {
                // Accept draft token
                n_accepted++;
                prev_token = draft_tokens[i];
            } else {
                // Reject — sample from target distribution (greedy)
                prev_token = m.argmax(verify_hd);
                break;
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double verify_ms = std::chrono::duration<double,std::milli>(t3-t2).count();
        total_verify_ms += verify_ms;

        // Update state
        total_draft += M;
        total_accepted += n_accepted;

        if (n_accepted > 0) {
            // Copy the last accepted hidden state as the new prefix
            memcpy(hd_saved, verify_hd, m.H*4);
            pos += n_accepted;
        }

        // Progress
        double acc_rate = (float)total_accepted / total_draft;
        double round_ms = draft_ms + verify_ms;
        double tok_per_round = 1.0 + n_accepted; // verified token + accepted drafts
        double tok_s = (round > 0) ? (pos - round + 1) / ((total_draft_ms + total_verify_ms) / 1000.0) : 0;

        if (round % 5 == 0 || round == n_rounds-1) {
            printf("  Round %2d: draft=%d accept=%d (%.0f%%)  ms/rnd=%.1f  tok/s=%.0f  pos=%d\n",
                   round, M, n_accepted, acc_rate*100, round_ms, tok_s, pos);
        }
        fflush(stdout);
    }

    double total_ms = total_draft_ms + total_verify_ms;
    float accept_rate = (float)total_accepted / total_draft;
    double tok_s = pos / (total_ms / 1000.0);

    printf("\n── Summary ──\n");
    printf("  Draft %dL, M=%d, %d rounds\n", draft_L, M, n_rounds);
    printf("  Draft time:   %.0f ms (%.1f ms/round)\n", total_draft_ms, total_draft_ms/n_rounds);
    printf("  Verify time:  %.0f ms (%.1f ms/round)\n", total_verify_ms, total_verify_ms/n_rounds);
    printf("  Total time:   %.0f ms\n", total_ms);
    printf("  Tokens:       %d\n", pos);
    printf("  Acceptance:   %.0f%% (%d/%d)\n", accept_rate*100, total_accepted, total_draft);
    printf("  Throughput:   %.0f tok/s\n", tok_s);
    printf("  Baseline:     %.0f tok/s (full model only)\n", 1000.0/(total_verify_ms/n_rounds));
    printf("  Speedup:      %.1fx\n", tok_s / (1000.0/(total_verify_ms/n_rounds)));
    printf("\n=== DSpark Live Complete ===\n");
    return 0;
}
