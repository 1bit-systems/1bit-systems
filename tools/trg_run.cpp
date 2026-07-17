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
    if (argc < 2) return printf("Usage: %s model.trg [layers]\n", argv[0]), 1;
    int fd = open(argv[1], O_RDONLY);
    size_t fsz = lseek(fd, 0, SEEK_END);
    auto p = (const char*)mmap(0, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (memcmp(p, "TRG1", 4)) return fprintf(stderr, "bad magic\n"), 1;

    // Read header via memcpy (unaligned-safe)
    auto r4=[&](int o){uint32_t v;memcpy(&v,p+o,4);return(int)v;};
    auto r8=[&](int o){uint64_t v;memcpy(&v,p+o,8);return v;};
    int H=r4(4),IM=r4(8),NH=r4(12),NKV=r4(16),HD=r4(20),V=r4(24),L=r4(28),GQA=r4(32);
    // ps[i] = TOTAL uint32 count for the entire projection, NOT per-row
    int ps[7]; for(int i=7;i--;) ps[i]=r4(36+i*4);
    uint64_t o_emb=r8(64),o_fn=r8(72),o_lm=r8(80),o_norms=r8(88),o_pk=r8(96),o_sc=r8(104);
    int nL=argc>2?atoi(argv[2]):L; if(nL>L)nL=L;
    auto F=[&](auto o){return(const float*)(p+o);};
    auto U=[&](auto o){return(const uint32_t*)(p+o);};

    printf("=== TRG: %d/%dL H=%d ===\n",nL,L,H);

    std::vector<float> hd(H),qkv(NH*HD+2*NKV*HD),at(NH*HD),ff(2*IM),ac(IM);
    std::vector<float> kcv(L*4096*NKV*HD,0),vcv(L*4096*NKV*HD,0);
    float*kc=kcv.data(),*vc=vcv.data();
    std::vector<float> st(4096*HD),ct(4096*HD);
    for(int p2=0;p2<4096;p2++)for(int d=0;d<HD;d++){
        float th=p2/pow(10000.f,(2.f*(d/2))/HD);
        st[p2*HD+d]=sin(th);ct[p2*HD+d]=cos(th);
    }
    memcpy(hd.data(),F(o_emb)+1*H,H*4);

    // Total uint32 per layer
    int per_layer = 0;
    for(int i=0;i<7;i++) per_layer += ps[i];

    int rows[7]={NH*HD,NKV*HD,NKV*HD,H,IM,IM,H};
    int KK[7]={H,H,H,NH*HD,H,H,IM};

    auto t0 = std::chrono::high_resolution_clock::now();
    for(int l=0;l<nL;l++){
        float res[4096]; memcpy(res,hd.data(),H*4);
        const uint32_t* pw = U(o_pk) + l * per_layer;
        const float* sw = F(o_sc) + l * (NH*HD+NKV*HD+NKV*HD+H+IM+IM+H);

        // QKV
        cpu_rmsnorm(hd.data(), F(o_norms)+l*H, hd.data(), H, 1e-6f);
        cpu_ternary_gemv(pw, hd.data(), sw, qkv.data(), rows[0], KK[0]); pw += ps[0]; sw += NH*HD;
        cpu_ternary_gemv(pw, hd.data(), sw, qkv.data()+NH*HD, rows[1], KK[1]); pw += ps[1]; sw += NKV*HD;
        cpu_ternary_gemv(pw, hd.data(), sw, qkv.data()+NH*HD+NKV*HD, rows[2], KK[2]); pw += ps[2]; sw += NKV*HD;

        // RoPE + KV
        for(int hh=0;hh<NH;hh++) cpu_rmsnorm(qkv.data()+hh*HD,F(o_norms)+2*L*H+l*HD,qkv.data()+hh*HD,HD,1e-6f);
        cpu_rope(qkv.data(),l,NH,HD,st.data(),ct.data());
        for(int hh=0;hh<NKV;hh++) cpu_rmsnorm(qkv.data()+NH*HD+hh*HD,F(o_norms)+2*L*H+L*HD+l*HD,qkv.data()+NH*HD+hh*HD,HD,1e-6f);
        cpu_rope(qkv.data()+NH*HD,l,NKV,HD,st.data(),ct.data());
        for(int hh=0;hh<NKV;hh++){
            memcpy(&kc[l*4096*NKV*HD+l*NKV*HD+hh*HD],qkv.data()+NH*HD+hh*HD,HD*4);
            memcpy(&vc[l*4096*NKV*HD+l*NKV*HD+hh*HD],qkv.data()+NH*HD+NKV*HD+hh*HD,HD*4);
        }

        // Attention
        cpu_attention(qkv.data(),&kc[l*4096*NKV*HD],&vc[l*4096*NKV*HD],at.data(),NH,NKV,HD,l+1,GQA);

        // O
        cpu_ternary_gemv(pw, at.data(), sw, hd.data(), rows[3], KK[3]); pw += ps[3]; sw += H;
        for(int i=0;i<H;i++) hd[i]=res[i]+hd[i];
        memcpy(res,hd.data(),H*4);

        // FFN
        cpu_rmsnorm(hd.data(),F(o_norms)+L*H+l*H,hd.data(),H,1e-6f);
        cpu_ternary_gemv(pw, hd.data(), sw, ff.data(), rows[4], KK[4]); pw += ps[4]; sw += IM;
        cpu_ternary_gemv(pw, hd.data(), sw, ff.data()+IM, rows[5], KK[5]); pw += ps[5]; sw += IM;
        cpu_silu_glu(ff.data(),ff.data()+IM,ac.data(),IM);
        cpu_ternary_gemv(pw, ac.data(), sw, hd.data(), rows[6], KK[6]);
        for(int i=0;i<H;i++) hd[i]=res[i]+hd[i];
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    cpu_rmsnorm(hd.data(),F(o_fn),hd.data(),H,1e-6f);
    std::vector<float> logits(V);
    cpu_lm_head(hd.data(),F(o_lm),logits.data(),V,H);
    int best=0; for(int i=1;i<V;i++) if(logits[i]>logits[best]) best=i;
    printf("  %d layers: %.1f ms (%.2f ms/l) = %.0f tok/s\n",nL,ms,ms/nL,1000./ms);
    printf("  Token: %d\n",best);
    munmap((void*)p,fsz);
}
