// tools/draft_data_gen.cpp — Generate training data for draft adapter
// Runs the model on random tokens, outputs (draft_hidden, full_hidden) pairs
//
// Build: g++ -O3 -march=native -std=c++17 -Iengine/fusion \
//        -o tools/draft_data_gen tools/draft_data_gen.cpp \
//        engine/fusion/cpu_layer.cpp -lm
//
// Run:   ./tools/draft_data_gen model.trg [draft_L=4] [n_samples=1000] [output=data.bin]

#include "cpu_layer.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

struct Model {
    int H,IM,NH,NKV,HD,V,L,GQA,per_layer,rows[7],KK[7],ps[7],per_sc;
    const float *emb,*fn,*lm,*inorm,*pan,*qn,*kn,*sc;
    const uint32_t *pk;

    bool load(const char* path) {
        int fd=open(path,O_RDONLY);
        size_t fsz=lseek(fd,0,SEEK_END);
        auto p=(const char*)mmap(0,fsz,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
        if(!p||p==MAP_FAILED||memcmp(p,"TRG1",4))return false;
        auto r4=[&](int o){uint32_t v;memcpy(&v,p+o,4);return(int)v;};
        auto r8=[&](int o){uint64_t v;memcpy(&v,p+o,8);return v;};
        H=r4(4);IM=r4(8);NH=r4(12);NKV=r4(16);HD=r4(20);V=r4(24);L=r4(28);GQA=r4(32);
        for(int i=7;i--;)ps[i]=r4(36+i*4);
        uint64_t o_emb=r8(64),o_fn=r8(72),o_lm=r8(80),o_norms=r8(88),o_pk=r8(96),o_sc=r8(104);
        auto F=[&](auto oo){return(const float*)(p+oo);};
        auto U=[&](auto oo){return(const uint32_t*)(p+oo);};
        emb=F(o_emb);fn=F(o_fn);lm=F(o_lm);
        inorm=F(o_norms);pan=F(o_norms+L*H*4);
        qn=F(o_norms+2*L*H*4);kn=F(o_norms+2*L*H*4+L*HD*4);
        pk=U(o_pk);sc=F(o_sc);
        per_layer=0;for(int i=7;i--;)per_layer+=ps[i];
        int r[7]={NH*HD,NKV*HD,NKV*HD,H,IM,IM,H},kk[7]={H,H,H,NH*HD,H,H,IM};
        per_sc=0;for(int i=7;i--;){rows[i]=r[i];KK[i]=kk[i];per_sc+=rows[i];}
        return true;
    }

    void forward(float* hd, int pos, int nL, float* kc, float* vc,
                 float* st, float* ct, float* qkv, float* at, float* ff, float* ac) {
        for(int l=0;l<nL;l++){
            float res[4096]; memcpy(res,hd,H*4);
            auto pw=pk+l*per_layer; auto sw=sc+l*per_sc;
            cpu_rmsnorm(hd,inorm+l*H,hd,H,1e-6f);
            cpu_ternary_gemv(pw,hd,sw,qkv,rows[0],KK[0]);pw+=ps[0];sw+=rows[0];
            cpu_ternary_gemv(pw,hd,sw,qkv+NH*HD,rows[1],KK[1]);pw+=ps[1];sw+=rows[1];
            cpu_ternary_gemv(pw,hd,sw,qkv+NH*HD+NKV*HD,rows[2],KK[2]);pw+=ps[2];sw+=rows[2];
            for(int h=0;h<NH;h++) cpu_rmsnorm(qkv+h*HD,qn+l*HD,qkv+h*HD,HD,1e-6f);
            cpu_rope(qkv,pos,NH,HD,st,ct);
            for(int h=0;h<NKV;h++) cpu_rmsnorm(qkv+NH*HD+h*HD,kn+l*HD,qkv+NH*HD+h*HD,HD,1e-6f);
            cpu_rope(qkv+NH*HD,pos,NKV,HD,st,ct);
            for(int h=0;h<NKV;h++){
                memcpy(&kc[l*4096*NKV*HD+pos*NKV*HD+h*HD],qkv+NH*HD+h*HD,HD*4);
                memcpy(&vc[l*4096*NKV*HD+pos*NKV*HD+h*HD],qkv+NH*HD+NKV*HD+h*HD,HD*4);
            }
            cpu_attention(qkv,&kc[l*4096*NKV*HD],&vc[l*4096*NKV*HD],at,NH,NKV,HD,pos+1,GQA);
            cpu_ternary_gemv(pw,at,sw,hd,rows[3],KK[3]);pw+=ps[3];sw+=rows[3];
            for(int i=0;i<H;i++)hd[i]=res[i]+hd[i];memcpy(res,hd,H*4);
            cpu_rmsnorm(hd,pan+l*H,hd,H,1e-6f);
            cpu_ternary_gemv(pw,hd,sw,ff,rows[4],KK[4]);pw+=ps[4];sw+=rows[4];
            cpu_ternary_gemv(pw,hd,sw,ff+IM,rows[5],KK[5]);pw+=ps[5];sw+=rows[5];
            cpu_silu_glu(ff,ff+IM,ac,IM);
            cpu_ternary_gemv(pw,ac,sw,hd,rows[6],KK[6]);
            for(int i=0;i<H;i++)hd[i]=res[i]+hd[i];
        }
    }
};

int main(int argc, char** argv) {
    const char* path=argc>1?argv[1]:"/tmp/model.trg";
    int draft_L=argc>2?atoi(argv[2]):4;
    int n_samples=argc>3?atoi(argv[3]):1000;
    const char* output=argc>4?argv[4]:"train_data.bin";

    Model m;
    if(!m.load(path)) return 1;
    printf("Generating %d training samples (draft=%dL)...\n",n_samples,draft_L);

    std::vector<float> qkv(m.NH*m.HD+2*m.NKV*m.HD),at(m.NH*m.HD),ff(2*m.IM),ac(m.IM);
    std::vector<float> st(4096*m.HD),ct(4096*m.HD);
    for(int p=0;p<4096;p++)for(int d=0;d<m.HD;d++){
        float th=p/pow(10000.f,(2.f*(d/2))/m.HD);
        st[p*m.HD+d]=sin(th);ct[p*m.HD+d]=cos(th);
    }
    auto kcv=std::vector<float>(m.L*4096*m.NKV*m.HD,0);
    auto vcv=std::vector<float>(m.L*4096*m.NKV*m.HD,0);

    FILE* fout=fopen(output,"wb");
    if(!fout){perror("fopen");return 1;}
    
    // Write header: H, draft_L, n_samples
    fwrite(&m.H,sizeof(int),1,fout);
    fwrite(&draft_L,sizeof(int),1,fout);
    fwrite(&n_samples,sizeof(int),1,fout);

    std::mt19937 rng(42);
    int total_tokens=0;

    for(int s=0;s<n_samples;s++){
        // Random input token
        int tok = rng() % 1000 + 1;
        
        // Run draft forward
        float hd[4096];
        memcpy(hd, m.emb + tok * m.H, m.H*4);
        m.forward(hd, s, draft_L, kcv.data(), vcv.data(),
                  st.data(), ct.data(), qkv.data(), at.data(), ff.data(), ac.data());
        fwrite(hd, sizeof(float), m.H, fout); // draft hidden
        
        // Continue to full model
        m.forward(hd, s, m.L - draft_L, kcv.data(), vcv.data(),
                  st.data(), ct.data(), qkv.data(), at.data(), ff.data(), ac.data());
        fwrite(hd, sizeof(float), m.H, fout); // full hidden
        
        total_tokens++;
        if(s%100==0){printf("  %d/%d\r",s,n_samples);fflush(stdout);}
    }
    fclose(fout);
    printf("\nGenerated %d pairs -> %s (%lld bytes)\n",total_tokens,output,
           (long long)(n_samples*m.H*8));
    printf("Run: source /tmp/ds_env/bin/activate && python3 tools/train_adapter.py %s\n",output);
}
