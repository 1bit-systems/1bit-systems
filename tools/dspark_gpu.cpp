// tools/dspark_gpu.cpp — DSpark with GPU verification
// Draft on CPU (fast), verify on GPU (flash attention = batch efficient)
//
// Build:
//   g++ -O3 -march=native -std=c++17 -Iengine/fusion \
//        -o tools/dspark_gpu tools/dspark_gpu.cpp \
//        engine/fusion/cpu_layer.cpp -lm \
//        -L~/1bit/build -lgpu_verify -Wl,-rpath,~/1bit/build
//
// Run:   ./tools/dspark_gpu model.trg [draft_layers=2] [M=8] [rounds=10]

#include "cpu_layer.h"
#include "gpu_verify.h"
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

    void forward_cpu(float* hd, int pos, int nL, float* kc, float* vc,
                     float* st, float* ct, float* qkv, float* at, float* ff, float* ac) {
        for(int l=0;l<nL;l++){
            float res[4096]; memcpy(res,hd,H*4);
            auto pw=pk+l*per_layer; auto sw=sc+l*(rows[0]+rows[1]+rows[2]+rows[3]+rows[4]+rows[5]+rows[6]);
            cpu_rmsnorm(hd,inorm+l*H,hd,H,1e-6f);
            cpu_ternary_gemv(pw,hd,sw,qkv,rows[0],KK[0]);pw+=ps[0];sw+=rows[0];
            cpu_ternary_gemv(pw,hd,sw,qkv+NH*HD,rows[1],KK[1]);pw+=ps[1];sw+=rows[1];
            cpu_ternary_gemv(pw,hd,sw,qkv+NH*HD+NKV*HD,rows[2],KK[2]);pw+=ps[2];sw+=rows[2];
            for(int hh=0;hh<NH;hh++) cpu_rmsnorm(qkv+hh*HD,qn+l*HD,qkv+hh*HD,HD,1e-6f);
            cpu_rope(qkv,pos,NH,HD,st,ct);
            for(int hh=0;hh<NKV;hh++) cpu_rmsnorm(qkv+NH*HD+hh*HD,kn+l*HD,qkv+NH*HD+hh*HD,HD,1e-6f);
            cpu_rope(qkv+NH*HD,pos,NKV,HD,st,ct);
            for(int hh=0;hh<NKV;hh++){
                memcpy(&kc[l*4096*NKV*HD+pos*NKV*HD+hh*HD],qkv+NH*HD+hh*HD,HD*4);
                memcpy(&vc[l*4096*NKV*HD+pos*NKV*HD+hh*HD],qkv+NH*HD+NKV*HD+hh*HD,HD*4);
            }
            cpu_attention(qkv,&kc[l*4096*NKV*HD],&vc[l*4096*NKV*HD],at,NH,NKV,HD,pos+1,GQA);
            cpu_ternary_gemv(pw,at,sw,hd,rows[3],KK[3]);pw+=ps[3];sw+=rows[3];
            for(int i=0;i<H;i++)hd[i]=res[i]+hd[i];
            memcpy(res,hd,H*4);
            cpu_rmsnorm(hd,pan+l*H,hd,H,1e-6f);
            cpu_ternary_gemv(pw,hd,sw,ff,rows[4],KK[4]);pw+=ps[4];sw+=rows[4];
            cpu_ternary_gemv(pw,hd,sw,ff+IM,rows[5],KK[5]);pw+=ps[5];sw+=rows[5];
            cpu_silu_glu(ff,ff+IM,ac,IM);
            cpu_ternary_gemv(pw,ac,sw,hd,rows[6],KK[6]);
            for(int i=0;i<H;i++)hd[i]=res[i]+hd[i];
        }
    }

    int argmax_cpu(const float* hd) {
        float tmp[4096]; memcpy(tmp,hd,H*4);
        cpu_rmsnorm(tmp,fn,tmp,H,1e-6f);
        std::vector<float> logits(V);
        cpu_lm_head(tmp,lm,logits.data(),V,H);
        int b=0; for(int i=1;i<V;i++) if(logits[i]>logits[b]) b=i;
        return b;
    }
};

int main(int argc, char** argv) {
    const char* path = argc>1?argv[1]:"/tmp/model.trg";
    int draft_L = argc>2?atoi(argv[2]):2;
    int M = argc>3?atoi(argv[3]):8;
    int n_rounds = argc>4?atoi(argv[4]):10;

    TrgModel m;
    if (!m.load(path)) return fprintf(stderr,"load failed\n"),1;
    printf("=== DSpark GPU Verify ===\n  H=%d L=%d Draft=%dL M=%d\n\n",m.H,m.L,draft_L,M);

    // Init GPU
    GpuVerifyCtx* gpu = gpu_verify_init(path);
    if (!gpu) return fprintf(stderr,"GPU init failed\n"),1;

    // CPU buffers
    std::vector<float> hd(m.H), qkv(m.NH*m.HD+2*m.NKV*m.HD), at(m.NH*m.HD), ff(2*m.IM), ac(m.IM);
    std::vector<float> st(4096*m.HD), ct(4096*m.HD);
    for(int p=0;p<4096;p++)for(int d=0;d<m.HD;d++){
        float th=p/pow(10000.f,(2.f*(d/2))/m.HD);
        st[p*m.HD+d]=sin(th);ct[p*m.HD+d]=cos(th);
    }

    // KV caches (shared between draft and verify on CPU)
    auto kcv = std::vector<float>(m.L*4096*m.NKV*m.HD,0.0f);
    auto vcv = std::vector<float>(m.L*4096*m.NKV*m.HD,0.0f);

    float hd_saved[4096], hd_prefix[4096];
    memcpy(hd_saved, m.emb+1*m.H, m.H*4);
    memcpy(hd.data(), hd_saved, m.H*4);

    // Warmup: run full model on CPU to establish prefix
    m.forward_cpu(hd.data(), 0, m.L, kcv.data(), vcv.data(),
                  st.data(), ct.data(), qkv.data(), at.data(), ff.data(), ac.data());
    int first_tok = m.argmax_cpu(hd.data());
    memcpy(hd_saved, hd.data(), m.H*4);
    int pos = 1;
    printf("  Prefix token: %d\n\n", first_tok);

    // ── Verify GPU vs CPU produce same output ──
    printf("── GPU vs CPU correctness ──\n");
    float cpu_hd[4096], gpu_hd[4096];
    memcpy(cpu_hd, hd_saved, m.H*4);
    memcpy(gpu_hd, hd_saved, m.H*4);

    auto t0 = std::chrono::high_resolution_clock::now();
    m.forward_cpu(cpu_hd, pos, m.L, kcv.data(), vcv.data(),
                  st.data(), ct.data(), qkv.data(), at.data(), ff.data(), ac.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double,std::milli>(t1-t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    gpu_verify_forward(gpu, gpu_hd, pos);
    t1 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double,std::milli>(t1-t0).count();

    int cpu_tok = m.argmax_cpu(cpu_hd);
    int gpu_tok = m.argmax_cpu(gpu_hd);  // argmax on CPU (same logic)

    float diff = 0;
    for(int i=0;i<m.H;i++) diff += fabs(cpu_hd[i] - gpu_hd[i]);
    printf("  CPU: %d (%.1f ms)\n", cpu_tok, cpu_ms);
    printf("  GPU: %d (%.1f ms)\n", gpu_tok, gpu_ms);
    printf("  Diff: %.4f  Tokens match: %s\n", diff, cpu_tok==gpu_tok?"YES":"NO");
    printf("  GPU speedup: %.1fx\n\n", cpu_ms/gpu_ms);

    // ── DSpark: CPU draft + GPU verify ──
    printf("── DSpark (CPU draft %dL + GPU verify) M=%d %d rounds ──\n", draft_L, M, n_rounds);

    // Fresh KV cache for this run
    std::fill(kcv.begin(), kcv.end(), 0);
    std::fill(vcv.begin(), vcv.end(), 0);
    memcpy(hd.data(), hd_saved, m.H*4);

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> uniform(0,1);
    int total_draft=0, total_accepted=0;
    double total_draft_ms=0, total_verify_ms=0;

    // Restore prefix in GPU's KV cache by running one verify at position 0
    float tmp_hd[4096]; memcpy(tmp_hd, m.emb+1*m.H, m.H*4);
    gpu_verify_forward(gpu, tmp_hd, 0);

    // Also restore CPU KV cache
    memcpy(hd.data(), hd_saved, m.H*4);
    std::fill(kcv.begin(), kcv.end(), 0);
    m.forward_cpu(hd.data(), 0, m.L, kcv.data(), vcv.data(),
                  st.data(), ct.data(), qkv.data(), at.data(), ff.data(), ac.data());

    float hd_cpu_draft[4096], hd_gpu_verify[4096];
    memcpy(hd_cpu_draft, hd_saved, m.H*4);
    memcpy(hd_gpu_verify, hd_saved, m.H*4);
    pos = 1;

    for(int round=0;round<n_rounds;round++){
        // ── Draft: CPU generates M tokens ──
        int draft_tokens[32];
        float draft_probs[32];
        memcpy(hd_cpu_draft, hd_saved, m.H*4);

        t0 = std::chrono::high_resolution_clock::now();
        for(int i=0;i<M;i++){
            m.forward_cpu(hd_cpu_draft, pos+i, draft_L, kcv.data(), vcv.data(),
                          st.data(), ct.data(), qkv.data(), at.data(), ff.data(), ac.data());
            draft_tokens[i] = m.argmax_cpu(hd_cpu_draft);
        }
        t1 = std::chrono::high_resolution_clock::now();
        double draft_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        total_draft_ms += draft_ms;

        // ── Verify: GPU checks each candidate ──
        memcpy(hd_gpu_verify, hd_saved, m.H*4);
        int n_accepted = 0;

        t0 = std::chrono::high_resolution_clock::now();
        for(int i=0;i<M;i++){
            gpu_verify_forward(gpu, hd_gpu_verify, pos+i); // runs forward + updates hidden
            // The hidden state is now in FP16 from GPU, convert for argmax
            int target_tok = m.argmax_cpu(hd_gpu_verify);

            // Acceptance: does the target agree?
            if (target_tok == draft_tokens[i]) {
                n_accepted++;
            } else {
                break;
            }
        }
        t1 = std::chrono::high_resolution_clock::now();
        double verify_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        total_verify_ms += verify_ms;

        total_draft += M;
        total_accepted += n_accepted;

        if (n_accepted > 0) {
            memcpy(hd_saved, hd_gpu_verify, m.H*4);
            pos += n_accepted;
        }

        float acc = (float)total_accepted / total_draft;
        double tot_ms = total_draft_ms + total_verify_ms;
        double tok_s = pos / (tot_ms / 1000.0);

        if (round % 5 == 0 || round == n_rounds-1) {
            printf("  Rnd %2d: acc=%d/%d (%.0f%%)  draft=%.0fms  verify=%.0fms  tok/s=%.0f  pos=%d\n",
                   round, n_accepted, M, acc*100, draft_ms, verify_ms, tok_s, pos);
        }
        fflush(stdout);
    }

    double tot_ms = total_draft_ms + total_verify_ms;
    float acc = (float)total_accepted / total_draft;
    printf("\n── Summary ──\n");
    printf("  Draft %dL (CPU) + Verify (GPU): M=%d\n", draft_L, M);
    printf("  Draft:  %.0f ms total (%.1f ms/round)\n", total_draft_ms, total_draft_ms/n_rounds);
    printf("  Verify: %.0f ms total (%.1f ms/round)\n", total_verify_ms, total_verify_ms/n_rounds);
    printf("  Acceptance: %.0f%% (%d/%d)\n", acc*100, total_accepted, total_draft);
    printf("  Throughput: %.0f tok/s\n", 1000.0*pos/tot_ms);
    printf("  Speedup vs CPU-only (%.0f ms): %.1fx\n",
           cpu_ms, (1000.0*pos/tot_ms) / (1000.0/cpu_ms));

    gpu_verify_free(gpu);
    printf("\n=== DSpark GPU Complete ===\n");
    return 0;
}
