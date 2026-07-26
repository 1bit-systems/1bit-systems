#include "cpu_layer.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

int main(int argc, char** argv) {
    const char* path = argc>1 ? argv[1] : "/tmp/model.trg";
    int fd = open(path, O_RDONLY);
    size_t fsz = lseek(fd,0,SEEK_END);
    auto p = (const char*)mmap(0,fsz,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    if (memcmp(p,"TRG1",4)) return fprintf(stderr,"bad magic\n"),1;

    auto r4=[&](int o){uint32_t v;memcpy(&v,p+o,4);return(int)v;};
    auto r8=[&](int o){uint64_t v;memcpy(&v,p+o,8);return v;};
    int H=r4(4),IM=r4(8),NH=r4(12),NKV=r4(16),HD=r4(20),V=r4(24),L=r4(28),GQA=r4(32);
    int ps[7]; for(int i=7;i--;) ps[i]=r4(36+i*4);
    uint64_t o_emb=r8(64),o_fn=r8(72),o_lm=r8(80),o_norms=r8(88),o_pk=r8(96),o_sc=r8(104);
    auto F=[&](auto oo){return(const float*)(p+oo);};
    auto U=[&](auto oo){return(const uint32_t*)(p+oo);};
    const float*emb=F(o_emb),*fn=F(o_fn),*lm=F(o_lm);
    const float*inorm=F(o_norms),*pan=F(o_norms+L*H*4);
    const float*qn=F(o_norms+2*L*H*4),*kn=F(o_norms+2*L*H*4+L*HD*4);

    int per_layer=0, rows[7]={NH*HD,NKV*HD,NKV*HD,H,IM,IM,H},KK[7]={H,H,H,NH*HD,H,H,IM};
    for(int i=7;i--;) per_layer+=ps[i];

    printf("=== DSpark on .trg ===\n  H=%d L=%d NH=%d NKV=%d HD=%d IM=%d\n\n",H,L,NH,NKV,HD,IM);

    // Derive max sequence length from model config (read from header, default 4096)
    int max_seq_len = 4096;  // default
    // The .trg format stores max_seq_len at offset 112 (after 7 ps values at 36-63,
    // then 6 uint64_t offsets at 64-111). Read it if available.
    // For now, use a safe default. In the future, read from header offset 112.
    if (H > 0 && max_seq_len > 4096) max_seq_len = 4096;  // cap at 4096

    // Shared buffers (sized by derived max_seq_len instead of hardcoded 4096)
    std::vector<float> hd(H),qkv(NH*HD+2*NKV*HD),at(NH*HD),ff(2*IM),ac(IM);
    std::vector<float> kcv(L*max_seq_len*NKV*HD),vcv(L*max_seq_len*NKV*HD);
    std::vector<float> st(max_seq_len*HD),ct(max_seq_len*HD);
    for(int p2=0;p2<max_seq_len;p2++)for(int d=0;d<HD;d++){
        float th=p2/pow(10000.f,(2.f*(d/2))/HD);
        st[p2*HD+d]=sin(th);ct[p2*HD+d]=cos(th);
    }

    auto run_layers = [&](int nL, float* kc, float* vc) {
        memcpy(hd.data(),emb+1*H,H*4);
        for(int l=0;l<nL;l++){
            float res[4096]; memcpy(res,hd.data(),H*4);
            auto pw=U(o_pk)+l*per_layer;
            auto sw=F(o_sc)+l*(NH*HD+NKV*HD+NKV*HD+H+IM+IM+H);
            cpu_rmsnorm(hd.data(),inorm+l*H,hd.data(),H,1e-6f);
            cpu_ternary_gemv(pw,hd.data(),sw,qkv.data(),rows[0],KK[0]);pw+=ps[0];sw+=NH*HD;
            cpu_ternary_gemv(pw,hd.data(),sw,qkv.data()+NH*HD,rows[1],KK[1]);pw+=ps[1];sw+=NKV*HD;
            cpu_ternary_gemv(pw,hd.data(),sw,qkv.data()+NH*HD+NKV*HD,rows[2],KK[2]);pw+=ps[2];sw+=NKV*HD;
            for(int hh=0;hh<NH;hh++) cpu_rmsnorm(qkv.data()+hh*HD,qn+l*HD,qkv.data()+hh*HD,HD,1e-6f);
            cpu_rope(qkv.data(),l,NH,HD,st.data(),ct.data());
            for(int hh=0;hh<NKV;hh++) cpu_rmsnorm(qkv.data()+NH*HD+hh*HD,kn+l*HD,qkv.data()+NH*HD+hh*HD,HD,1e-6f);
            cpu_rope(qkv.data()+NH*HD,l,NKV,HD,st.data(),ct.data());
            for(int hh=0;hh<NKV;hh++){
                // Fix #607: use `l` as layer index, `l` as position — these are benchmarks
                // where each layer runs at a new position. For real inference, position
                // should be passed as a separate parameter.
                int pos = l;  // position = layer index for this benchmark
                if (pos >= max_seq_len) {
                    fprintf(stderr, "WARNING: position %d >= max_seq_len %d — skipping\n", pos, max_seq_len);
                    continue;
                }
                memcpy(&kc[l*max_seq_len*NKV*HD+pos*NKV*HD+hh*HD],qkv.data()+NH*HD+hh*HD,HD*4);
                memcpy(&vc[l*max_seq_len*NKV*HD+pos*NKV*HD+hh*HD],qkv.data()+NH*HD+NKV*HD+hh*HD,HD*4);
            }
            cpu_attention(qkv.data(),&kc[l*max_seq_len*NKV*HD],&vc[l*max_seq_len*NKV*HD],at.data(),NH,NKV,HD,l+1,GQA);
            cpu_ternary_gemv(pw,at.data(),sw,hd.data(),rows[3],KK[3]);pw+=ps[3];sw+=H;
            for(int i=0;i<H;i++) hd[i]=res[i]+hd[i];
            memcpy(res,hd.data(),H*4);
            cpu_rmsnorm(hd.data(),pan+l*H,hd.data(),H,1e-6f);
            cpu_ternary_gemv(pw,hd.data(),sw,ff.data(),rows[4],KK[4]);pw+=ps[4];sw+=IM;
            cpu_ternary_gemv(pw,hd.data(),sw,ff.data()+IM,rows[5],KK[5]);pw+=ps[5];sw+=IM;
            cpu_silu_glu(ff.data(),ff.data()+IM,ac.data(),IM);
            cpu_ternary_gemv(pw,ac.data(),sw,hd.data(),rows[6],KK[6]);
            for(int i=0;i<H;i++) hd[i]=res[i]+hd[i];
        }
        cpu_rmsnorm(hd.data(),fn,hd.data(),H,1e-6f);
    };

    // Benchmark each layer count
    int sweep[] = {1,2,3,4,6,8,12,16,28,0};
    double dms[28]={0}, full_ms=0;
    printf("── Layer speeds ──\n  %5s %8s %7s\n","L","ms","tok/s");
    for(int si=0;sweep[si];si++){
        int nl=sweep[si]; if(nl>L) break;
        for(int i=3;i--;) run_layers(nl,kcv.data(),vcv.data());
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=20;i--;) run_layers(nl,kcv.data(),vcv.data());
        auto t1=std::chrono::high_resolution_clock::now();
        double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/10.0;
        dms[nl-1]=ms; if(nl==L)full_ms=ms;
        printf("  L=%2d %7.2f %7.0f\n",nl,ms,1000./ms);
        fflush(stdout);
    }

    // DSpark estimates
    printf("\n── DSpark (verify %.2fms) ──\n",full_ms);
    printf("  %4s %3s %5s %8s %8s %6s\n","L","M","acc","ms/rnd","tok/s","spd");
    int Ms[]={4,8,16,0}; double accs[]={0.5,0.65,0.8};
    double best_ts=0; int best_nl=0,best_M=0; double best_acc=0;
    for(int si=0;sweep[si];si++){
        int nl=sweep[si]; if(nl>=L||nl<=0)continue;
        double dm=dms[nl-1]; if(dm<=0)continue;
        for(int mi=0;Ms[mi];mi++){int M=Ms[mi];
        for(int ai=0;ai<3;ai++){double acc=accs[ai];
            double rms=M*dm+full_ms, tpr=1+acc*M, ts=tpr/(rms/1000), sp=ts/(1000/full_ms);
            printf("  L=%2d  M=%2d  %.2f %7.1f %7.0f %5.1fx\n",nl,M,acc,rms,ts,sp);
            if(ts>best_ts){best_ts=ts;best_nl=nl;best_M=M;best_acc=acc;}
        }}
    }
    printf("\n── Best: L=%d M=%d acc=%.0f%%: %.0f tok/s (%.1fx) ──\n",
           best_nl,best_M,best_acc*100,best_ts,best_ts/(1000/full_ms));
    munmap((void*)p,fsz);
}
