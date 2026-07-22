// tools/train_draft.cpp — Train a linear draft adapter for DSpark
// Learns a matrix W such that W * h_draft ≈ h_full for any input.
// This lets the draft model (fewer layers) predict what the full model
// would output, making speculative decoding actually work.
//
// Usage:
//   1. Generate training data: first run the model on many random inputs
//   2. Train: finds the best linear mapping from draft→full hidden states
//   3. Apply: during DSpark, transform draft output before checking acceptance
//
// Build: g++ -O3 -march=native -std=c++17 -Iengine/fusion \
//        -o tools/train_draft tools/train_draft.cpp \
//        engine/fusion/cpu_layer.cpp -lm
//
// Run:   ./tools/train_draft model.trg [draft_L=4] [n_samples=1000]

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

    // Run N layers from a starting hidden state, updating in place
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

    int argmax(const float* hd) {
        float tmp[4096];memcpy(tmp,hd,H*4);cpu_rmsnorm(tmp,fn,tmp,H,1e-6f);
        std::vector<float> lg(V);cpu_lm_head(tmp,lm,lg.data(),V,H);
        int b=0;for(int i=1;i<V;i++)if(lg[i]>lg[b])b=i;return b;
    }
};

int main(int argc, char** argv) {
    const char* path=argc>1?argv[1]:"/tmp/model.trg";
    int draft_L=argc>2?atoi(argv[2]):4;
    int n_samples=argc>3?atoi(argv[3]):1000;

    Model m;
    if(!m.load(path)) return 1;
    const int H=m.H;
    printf("=== Train Draft Adapter ===\n  H=%d L=%d draft=%dL samples=%d\n\n",H,m.L,draft_L,n_samples);

    // Buffers
    std::vector<float> qkv(m.NH*m.HD+2*m.NKV*m.HD),at(m.NH*m.HD),ff(2*m.IM),ac(m.IM);
    std::vector<float> st(4096*m.HD),ct(4096*m.HD);
    for(int p=0;p<4096;p++)for(int d=0;d<m.HD;d++){
        float th=p/pow(10000.f,(2.f*(d/2))/m.HD);
        st[p*m.HD+d]=sin(th);ct[p*m.HD+d]=cos(th);
    }
    auto kcv=std::vector<float>(m.L*4096*m.NKV*m.HD,0);
    auto vcv=std::vector<float>(m.L*4096*m.NKV*m.HD,0);

    // Generate training data: run model on random token sequences
    // For each token, record (draft_hidden, full_hidden) pair
    printf("Generating %d training samples...\n", n_samples);
    std::vector<float> X(n_samples * H); // draft hidden states
    std::vector<float> Y(n_samples * H); // full hidden states
    std::mt19937 rng(42);

    for(int s=0;s<n_samples;s++){
        // Random token
        int tok = rng() % 1000 + 1; // use first 1000 tokens
        float hd[4096];
        memcpy(hd, m.emb + tok * H, H*4);

        // Save draft hidden state (after draft_L layers)
        m.forward(hd, s, draft_L, kcv.data(), vcv.data(),
                  st.data(), ct.data(), qkv.data(), at.data(), ff.data(), ac.data());
        memcpy(&X[s*H], hd, H*4);

        // Continue to full model to get full hidden state
        m.forward(hd, s, m.L - draft_L, kcv.data(), vcv.data(),
                  st.data(), ct.data(), qkv.data(), at.data(), ff.data(), ac.data());
        memcpy(&Y[s*H], hd, H*4);

        if(s%200==0){printf("  %d/%d\r",s,n_samples);fflush(stdout);}
    }
    printf("\n");

    // Train: find W = Y X^T (X X^T)^{-1}
    // Compute X * X^T (H×H matrix)
    printf("Training linear adapter (H=%d)...\n", H);
    const size_t szH = (size_t)H;
    std::vector<double> XtX(szH*H, 0.0);
    std::vector<double> XtY(szH*H, 0.0);

    for(int s=0;s<n_samples;s++){
        for(int i=0;i<H;i++){
            float xi = X[s*H+i];
            for(int j=0;j<H;j++){
                XtX[i*H+j] += (double)xi * X[s*H+j];
                XtY[i*H+j] += (double)xi * Y[s*H+j];
            }
        }
    }

    // Solve: W = XtY * inv(XtX)
    // Simple Gauss-Jordan inversion
    std::vector<double> inv(szH*H);
    for(int i=0;i<H;i++) for(int j=0;j<H;j++) inv[i*H+j] = (i==j) ? 1.0 : 0.0;

    // Gauss-Jordan elimination
    for(int col=0;col<H;col++){
        // Find pivot
        int pivot=col;
        double max_val=fabs(XtX[col*H+col]);
        for(int row=col+1;row<H;row++){
            double v=fabs(XtX[row*H+col]);
            if(v>max_val){max_val=v;pivot=row;}
        }
        if(max_val<1e-10){printf("  Singular matrix at col %d\n",col);continue;}

        // Swap rows
        if(pivot!=col){
            for(int j=0;j<H;j++){
                std::swap(XtX[col*H+j], XtX[pivot*H+j]);
                std::swap(inv[col*H+j], inv[pivot*H+j]);
            }
        }

        double diag = XtX[col*H+col];
        for(int j=0;j<H;j++){
            XtX[col*H+j] /= diag;
            inv[col*H+j] /= diag;
        }

        for(int row=0;row<H;row++){
            if(row==col) continue;
            double factor = XtX[row*H+col];
            for(int j=0;j<H;j++){
                XtX[row*H+j] -= factor * XtX[col*H+j];
                inv[row*H+j] -= factor * inv[col*H+j];
            }
        }
    }

    // W = inv * XtY
    std::vector<float> W(szH*H, 0.0f);
    for(int i=0;i<H;i++) for(int j=0;j<H;j++) for(int k=0;k<H;k++)
        W[i*H+j] += (float)(inv[i*H+k] * XtY[k*H+j]);

    // Save adapter to file
    FILE* f=fopen("draft_adapter.bin","wb");
    if(f){
        fwrite(&H,4,1,f);
        fwrite(&draft_L,4,1,f);
        fwrite(W.data(),4,H*H,f);
        fclose(f);
        printf("  Saved to draft_adapter.bin (%.1f MB)\n", H*H*4.0/1024/1024);
    }

    // Validate: run a few test samples and check if adapter improves acceptance
    printf("\nValidating...\n");
    kcv.assign(kcv.size(),0); vcv.assign(vcv.size(),0);
    int ok_before=0, ok_after=0, n_test=100;
    for(int t=0;t<n_test;t++){
        int tok = rng() % 1000 + 1;
        float hd[4096];
        memcpy(hd, m.emb + tok * H, H*4);
        m.forward(hd, 1000+t, draft_L, kcv.data(), vcv.data(),
                  st.data(), ct.data(), qkv.data(), at.data(), ff.data(), ac.data());
        int draft_tok = m.argmax(hd);

        // Check what adapter-predicted model would output
        float adapted[4096];
        for(int i=0;i<H;i++){
            double val=0;
            for(int j=0;j<H;j++) val += W[i*H+j] * hd[j];
            adapted[i] = (float)val;
        }

        // Continue draft to full model and compare
        m.forward(hd, 1000+t, m.L-draft_L, kcv.data(), vcv.data(),
                  st.data(), ct.data(), qkv.data(), at.data(), ff.data(), ac.data());
        int full_tok = m.argmax(hd);
        if(draft_tok==full_tok) ok_before++;

        // Check if adapter output predicts a better token
        int adapted_tok = m.argmax(adapted);
        if(adapted_tok==full_tok) ok_after++;
    }
    printf("  Draft top-1 matches full: %d/%d (%.0f%%)\n",ok_before,n_test,100.*ok_before/n_test);
    printf("  Adapter top-1 matches full: %d/%d (%.0f%%)\n",ok_after,n_test,100.*ok_after/n_test);

    printf("\n=== Training Complete ===\n");
}
