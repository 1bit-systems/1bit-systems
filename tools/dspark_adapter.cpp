// DSpark with linear adapter — CPU draft + adapter + GPU batch verify
// The adapter predicts full model output from truncated-layer hidden states.
//
// Build: hipcc -O3 --offload-arch=gfx1151 -Iinclude -Iengine/fusion \
//   -o tools/dspark_adapter tools/dspark_adapter.cpp \
//   engine/fusion/cpu_layer.cpp -lm -Lbuild -lrocm_cpp -lbatch_attn
// Run:  LD_LIBRARY_PATH=build ./tools/dspark_adapter model.trg [draft_L=4] [M=4] [rounds=5]

#include "rocm_cpp/ck_gemm.h"
#include "cpu_layer.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

extern "C" void batch_attn_launch(const void*, const void*, const void*, void*,
    int M, int NH, int NKV, int HD, int pos, hipStream_t s);
#define SYNC hipStreamSynchronize(s)

// ── Linear adapter ──
struct Adapter {
    float W[1024*1024];
    float Xm[1024], Xs[1024], Ym[1024], Ys[1024];
    
    bool load(const char* dir) {
        auto load = [&](const char* name, float* data, int n) {
            char path[256]; snprintf(path,256,"%s/adapter_%s.bin",dir,name);
            FILE* f=fopen(path,"rb");
            if(!f) return false;
            fread(data,4,n,f);
            fclose(f); return true;
        };
        return load("W",W,1024*1024) && load("X_mean",Xm,1024) && 
               load("X_std",Xs,1024) && load("Y_mean",Ym,1024) && load("Y_std",Ys,1024);
    }
    
    void apply(const float* h_in, float* h_out) {
        // adapted = Y_std * ((h_in - X_mean) / X_std @ W) + Y_mean
        for(int i=0;i<1024;i++){
            double val = 0;
            for(int j=0;j<1024;j++){
                val += (double)(h_in[j] - Xm[j]) / Xs[j] * W[i*1024+j];
            }
            h_out[i] = (float)(val * Ys[i] + Ym[i]);
        }
    }
    
    int predict_token(const float* h_in, const float* lm, int V, int H) {
        float adapted[1024];
        apply(h_in, adapted);
        float tmp[1024]; memcpy(tmp,adapted,1024*4);
        cpu_rmsnorm(tmp, lm + V*H, tmp, H, 1e-6f); // final norm doesn't exist here
        // Actually the LM head needs the correct final_norm. We don't have it in adapter.
        // For now, just use argmax on the adapted hidden directly (no final norm)
        // This is an approximation
        std::vector<float> lg(V);
        cpu_lm_head(adapted, lm, lg.data(), V, H);
        int b=0; for(int i=1;i<V;i++)if(lg[i]>lg[b])b=i;
        return b;
    }
};

int main(int argc, char** argv) {
    const char* path=argc>1?argv[1]:"/tmp/model.trg";
    int draft_L=argc>2?atoi(argv[2]):4, M=argc>3?atoi(argv[3]):4, rounds=argc>4?atoi(argv[4]):5;
    
    int fd=open(path,O_RDONLY);
    size_t fsz=lseek(fd,0,SEEK_END);
    auto p=(const char*)mmap(0,fsz,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    auto r4=[&](int o){uint32_t v;memcpy(&v,p+o,4);return(int)v;};
    auto r8=[&](int o){uint64_t v;memcpy(&v,p+o,8);return v;};
    int H=r4(4),IM=r4(8),NH=r4(12),NKV=r4(16),HD=r4(20),V=r4(24),L=r4(28),GQA=r4(32);
    int ps[7];for(int i=7;i--;)ps[i]=r4(36+i*4);
    uint64_t o_emb=r8(64),o_fn=r8(72),o_lm=r8(80),o_norms=r8(88),o_pk=r8(96),o_sc=r8(104);
    auto F=[&](auto oo){return(const float*)(p+oo);};
    auto U=[&](auto oo){return(const uint32_t*)(p+oo);};
    int per_layer=0,rows[7]={NH*HD,NKV*HD,NKV*HD,H,IM,IM,H},KK[7]={H,H,H,NH*HD,H,H,IM};
    for(int i=7;i--;)per_layer+=ps[i];
    int per_sc=0;for(int i=7;i--;)per_sc+=rows[i];
    
    // Load adapter
    Adapter ada;
    if(!ada.load(".")){
        // Try tools directory
        ada.load("tools");
    }
    printf("=== DSpark Adapter === H=%d L=%d draft=%dL M=%d\n\n",H,L,draft_L,M);

    // GPU init
    hipStream_t s; hipStreamCreate(&s);
    float *d_pk,*d_sc,*d_inorm,*d_pan,*d_qn,*d_kn,*d_fn,*d_hf,*d_xs; _Float16 *d_kc,*d_vc;
    _Float16 *d_h[32],*d_q[32],*d_at,*d_ff[32],*d_ac[32]; int8_t *d_i8; float xsh;
    auto ml=[&](auto&p_,size_t b){hipMalloc((void**)&p_,b);hipMemset(p_,0,b);SYNC;};
    ml(d_pk,L*per_layer*4);ml(d_sc,L*per_sc*4);
    ml(d_inorm,L*H*4);ml(d_pan,L*H*4);ml(d_qn,L*HD*4);ml(d_kn,L*HD*4);
    ml(d_fn,H*4);ml(d_hf,8192*32*4);ml(d_xs,4);
    ml(d_at,(size_t)M*NH*HD*2*2);
    for(int m=0;m<M;m++){ml(d_h[m],H*2);ml(d_q[m],(NH*HD+2*NKV*HD)*2);ml(d_ff[m],2*IM*2);ml(d_ac[m],IM*2);}
    ml(d_i8,H); int MP=4096;
    ml(d_kc,(size_t)L*MP*NKV*HD*2);ml(d_vc,(size_t)L*MP*NKV*HD*2);
    auto up=[&](auto d,auto h,size_t b){hipMemcpy(d,h,b,hipMemcpyHostToDevice);SYNC;};
    up(d_pk,U(o_pk),L*per_layer*4);up(d_sc,F(o_sc),L*per_sc*4);
    up(d_inorm,F(o_norms),L*H*4);up(d_pan,F(o_norms)+L*H,L*H*4);
    up(d_qn,F(o_norms)+2*L*H,L*HD*4);up(d_kn,F(o_norms)+2*L*H+L*HD,L*HD*4);
    up(d_fn,F(o_fn),H*4);

    // Batch forward
    auto bf=[&](float** hd, int pos){
        for(int m=0;m<M;m++){hipMemcpy(d_hf+m*8192,hd[m],H*4,hipMemcpyHostToDevice);SYNC;rcpp_fp32_to_fp16(d_hf+m*8192,d_h[m],H,s);SYNC;}
        for(int l=0;l<L;l++){
            auto wp=(const uint32_t*)((const char*)d_pk+l*per_layer*4);
            auto ws=d_sc+l*per_sc;
            for(int m=0;m<M;m++){
                _Float16*dh=d_h[m];_Float16*qv=d_q[m];_Float16*ffm=d_ff[m];
                hipMemcpy(ffm,dh,H*2,hipMemcpyDeviceToDevice);SYNC;
                rcpp_rmsnorm_fp16(dh,d_inorm+l*H,dh,1e-6f,H,s);SYNC;
                rcpp_quantize_fp16_to_i8(dh,d_i8,d_xs,H,s);SYNC;
                hipMemcpy(&xsh,d_xs,4,hipMemcpyDeviceToHost);SYNC;
                auto wq=wp;auto ws2=ws;
                rcpp_ternary_gemv(wq,d_i8,xsh,ws2,d_hf+m*8192,NH*HD,H,s);SYNC;wq+=ps[0];ws2+=rows[0];
                rcpp_fp32_to_fp16(d_hf+m*8192,qv,NH*HD,s);SYNC;
                rcpp_ternary_gemv(wq,d_i8,xsh,ws2,d_hf+m*8192,NKV*HD,H,s);SYNC;wq+=ps[1];ws2+=rows[1];
                rcpp_fp32_to_fp16(d_hf+m*8192,qv+NH*HD,NKV*HD,s);SYNC;
                rcpp_ternary_gemv(wq,d_i8,xsh,ws2,d_hf+m*8192,NKV*HD,H,s);SYNC;wq+=ps[2];ws2+=rows[2];
                rcpp_fp32_to_fp16(d_hf+m*8192,qv+NH*HD+NKV*HD,NKV*HD,s);SYNC;
                for(int h=0;h<NH;h++){rcpp_rmsnorm_fp16(qv+h*HD,d_qn+l*HD,qv+h*HD,1e-6f,HD,s);SYNC;}
                rcpp_rope_fp16(qv,pos+m,1000000.0f,NH,HD,s);SYNC;
                for(int h=0;h<NKV;h++){rcpp_rmsnorm_fp16(qv+NH*HD+h*HD,d_kn+l*HD,qv+NH*HD+h*HD,1e-6f,HD,s);SYNC;}
                rcpp_rope_fp16(qv+NH*HD,pos+m,1000000.0f,NKV,HD,s);SYNC;
                _Float16*kl=d_kc+(size_t)l*MP*NKV*HD+(size_t)(pos+m)*NKV*HD;
                _Float16*vl=d_vc+(size_t)l*MP*NKV*HD+(size_t)(pos+m)*NKV*HD;
                hipMemcpy(kl,qv+NH*HD,NKV*HD*2,hipMemcpyDeviceToDevice);SYNC;
                hipMemcpy(vl,qv+NH*HD+NKV*HD,NKV*HD*2,hipMemcpyDeviceToDevice);SYNC;
                hipMemcpy(d_at+(size_t)m*NH*HD,qv,NH*HD*2,hipMemcpyDeviceToDevice);SYNC;
            }
            _Float16* ao=d_at+(size_t)M*NH*HD;
            batch_attn_launch(d_at,d_kc+(size_t)l*MP*NKV*HD,d_vc+(size_t)l*MP*NKV*HD,ao,M,NH,NKV,HD,pos,s);SYNC;
            for(int m=0;m<M;m++){
                _Float16*dh=d_h[m];_Float16*ffm=d_ff[m];_Float16*acm=d_ac[m];
                rcpp_quantize_fp16_to_i8(ao+(size_t)m*NH*HD,d_i8,d_xs,NH*HD,s);SYNC;
                hipMemcpy(&xsh,d_xs,4,hipMemcpyDeviceToHost);SYNC;
                auto wq=wp+ps[0]+ps[1]+ps[2];auto ws2=ws+rows[0]+rows[1]+rows[2];
                rcpp_ternary_gemv(wq,d_i8,xsh,ws2,d_hf+m*8192,H,NH*HD,s);SYNC;
                rcpp_fp32_to_fp16(d_hf+m*8192,dh,H,s);SYNC;
                rcpp_residual_add_fp16(dh,ffm,H,s);SYNC;
                rcpp_rmsnorm_fp16(dh,d_pan+l*H,dh,1e-6f,H,s);SYNC;
                rcpp_quantize_fp16_to_i8(dh,d_i8,d_xs,H,s);SYNC;
                hipMemcpy(&xsh,d_xs,4,hipMemcpyDeviceToHost);SYNC;
                wq=wp+ps[0]+ps[1]+ps[2]+ps[3];ws2=ws+rows[0]+rows[1]+rows[2]+rows[3];
                rcpp_ternary_gemv(wq,d_i8,xsh,ws2,d_hf+m*8192,IM,H,s);SYNC;wq+=ps[4];ws2+=rows[4];
                rcpp_fp32_to_fp16(d_hf+m*8192,ffm,IM,s);SYNC;
                rcpp_ternary_gemv(wq,d_i8,xsh,ws2,d_hf+m*8192,IM,H,s);SYNC;wq+=ps[5];ws2+=rows[5];
                rcpp_fp32_to_fp16(d_hf+m*8192,ffm+IM,IM,s);SYNC;
                rcpp_silu_glu_fp16(ffm+IM,ffm,acm,IM,s);SYNC;
                rcpp_quantize_fp16_to_i8(acm,d_i8,d_xs,IM,s);SYNC;
                hipMemcpy(&xsh,d_xs,4,hipMemcpyDeviceToHost);SYNC;
                wq+=ps[6];ws2+=rows[6];
                rcpp_ternary_gemv(wq,d_i8,xsh,ws2,d_hf+m*8192,H,IM,s);SYNC;
                rcpp_fp32_to_fp16(d_hf+m*8192,dh,H,s);SYNC;
                rcpp_residual_add_fp16(dh,ffm,H,s);SYNC;
            }
        }
        for(int m=0;m<M;m++){rcpp_rmsnorm_fp16(d_h[m],d_fn,d_h[m],1e-6f,H,s);SYNC;rcpp_fp16_to_fp32(d_h[m],d_hf+m*8192,H,s);SYNC;hipMemcpy(hd[m],d_hf+m*8192,H*4,hipMemcpyDeviceToHost);SYNC;}
    };

    // CPU draft with adapter
    auto kcv=std::vector<float>(L*4096*NKV*HD,0),vcv=std::vector<float>(L*4096*NKV*HD,0);
    std::vector<float> qkvv(NH*HD+2*NKV*HD),atv(NH*HD),ffv(2*IM),acv(IM);
    std::vector<float> st(4096*HD),ct(4096*HD);
    for(int p2=0;p2<4096;p2++)for(int d=0;d<HD;d++){float th=p2/pow(10000.f,(2.f*(d/2))/HD);st[p2*HD+d]=sin(th);ct[p2*HD+d]=cos(th);}
    
    auto cd=[&](float*hd,int p,int nL){
        for(int l=0;l<nL;l++){
            float res[4096];memcpy(res,hd,H*4);
            auto pw=U(o_pk)+l*per_layer;auto sw=F(o_sc)+l*per_sc;
            cpu_rmsnorm(hd,F(o_norms)+l*H,hd,H,1e-6f);
            cpu_ternary_gemv(pw,hd,sw,qkvv.data(),rows[0],KK[0]);pw+=ps[0];sw+=rows[0];
            cpu_ternary_gemv(pw,hd,sw,qkvv.data()+NH*HD,rows[1],KK[1]);pw+=ps[1];sw+=rows[1];
            cpu_ternary_gemv(pw,hd,sw,qkvv.data()+NH*HD+NKV*HD,rows[2],KK[2]);pw+=ps[2];sw+=rows[2];
            for(int h=0;h<NH;h++)cpu_rmsnorm(qkvv.data()+h*HD,F(o_norms)+2*L*H+l*HD,qkvv.data()+h*HD,HD,1e-6f);
            cpu_rope(qkvv.data(),p,NH,HD,st.data(),ct.data());
            for(int h=0;h<NKV;h++)cpu_rmsnorm(qkvv.data()+NH*HD+h*HD,F(o_norms)+2*L*H+L*HD+l*HD,qkvv.data()+NH*HD+h*HD,HD,1e-6f);
            cpu_rope(qkvv.data()+NH*HD,p,NKV,HD,st.data(),ct.data());
            for(int h=0;h<NKV;h++){memcpy(&kcv[l*4096*NKV*HD+p*NKV*HD+h*HD],qkvv.data()+NH*HD+h*HD,HD*4);memcpy(&vcv[l*4096*NKV*HD+p*NKV*HD+h*HD],qkvv.data()+NH*HD+NKV*HD+h*HD,HD*4);}
            cpu_attention(qkvv.data(),&kcv[l*4096*NKV*HD],&vcv[l*4096*NKV*HD],atv.data(),NH,NKV,HD,p+1,GQA);
            cpu_ternary_gemv(pw,atv.data(),sw,hd,rows[3],KK[3]);pw+=ps[3];sw+=rows[3];
            for(int i=0;i<H;i++)hd[i]=res[i]+hd[i];memcpy(res,hd,H*4);
            cpu_rmsnorm(hd,F(o_norms)+L*H+l*H,hd,H,1e-6f);
            cpu_ternary_gemv(pw,hd,sw,ffv.data(),rows[4],KK[4]);pw+=ps[4];sw+=rows[4];
            cpu_ternary_gemv(pw,hd,sw,ffv.data()+IM,rows[5],KK[5]);pw+=ps[5];sw+=rows[5];
            cpu_silu_glu(ffv.data(),ffv.data()+IM,acv.data(),IM);
            cpu_ternary_gemv(pw,acv.data(),sw,hd,rows[6],KK[6]);
            for(int i=0;i<H;i++)hd[i]=res[i]+hd[i];
        }
        return 0; // token computed externally via adapter
    };

    float ph[4096];memcpy(ph,F(o_emb)+1*H,H*4);
    int pos=1,total_acc=0,total_raw=0,total_adapted=0;
    double t_d=0,t_v=0;
    float pr[4096];memcpy(pr,F(o_emb)+1*H,H*4);float*pp[1]={pr};bf(pp,0);
    memcpy(ph,F(o_emb)+1*H,H*4);

    for(int r=0;r<rounds;r++){
        std::vector<float> dh_vec(32 * 4096); float (*dh)[4096] = (float (*)[4096])dh_vec.data();
        int dt[32], dta[32];
        auto t0=std::chrono::high_resolution_clock::now();
        memcpy(dh[0],ph,H*4);
        for(int i=0;i<M;i++){
            if(i>0)memcpy(dh[i],dh[i-1],H*4);
            cd(dh[i],pos+i,draft_L);
            // Raw draft token (argmax of draft hidden without adapter)
            float tmph[4096];memcpy(tmph,dh[i],H*4);cpu_rmsnorm(tmph,F(o_fn),tmph,H,1e-6f);
            std::vector<float>lg(V);cpu_lm_head(tmph,F(o_lm),lg.data(),V,H);
            dt[i]=0;for(int j=1;j<V;j++)if(lg[j]>lg[dt[i]])dt[i]=j;
            // Adapted token
            dta[i]=ada.predict_token(dh[i],F(o_lm),V,H);
        }
        auto t1=std::chrono::high_resolution_clock::now();
        t_d+=std::chrono::duration<double,std::milli>(t1-t0).count();

        float*ghs[32]; std::vector<float> gh_vec(32 * 4096); float (*gh)[4096] = (float (*)[4096])gh_vec.data();
        for(int i=0;i<M;i++){memcpy(gh[i],ph,H*4);ghs[i]=gh[i];}
        t0=std::chrono::high_resolution_clock::now();
        bf(ghs,pos);
        t1=std::chrono::high_resolution_clock::now();
        t_v+=std::chrono::duration<double,std::milli>(t1-t0).count();

        int nac=0,nac_raw=0;
        for(int i=0;i<M;i++){
            float th[4096];memcpy(th,gh[i],H*4);cpu_rmsnorm(th,F(o_fn),th,H,1e-6f);
            std::vector<float>lg(V);cpu_lm_head(th,F(o_lm),lg.data(),V,H);
            int b=0;for(int j=1;j<V;j++)if(lg[j]>lg[b])b=j;
            if(b==dt[i])nac_raw++;
            if(b==dta[i])nac++;else break;
        }
        total_raw+=nac_raw;total_adapted+=nac;
        if(nac>0){memcpy(ph,gh[nac-1],H*4);pos+=nac;}
        double ts=pos/((t_d+t_v)/1000);
        printf("  R%2d: raw=%d/%d adapt=%d/%d tok/s=%.0f\n",r,nac_raw,M,nac,M,ts);
    }
    double ts=pos/((t_d+t_v)/1000);
    printf("\n── Summary ──\n  Raw acceptance: %.0f%%\n  Adapted acceptance: %.0f%%\n  Throughput: %.0f tok/s\n",
           100.*total_raw/(M*rounds),100.*total_adapted/(M*rounds),ts);
}
