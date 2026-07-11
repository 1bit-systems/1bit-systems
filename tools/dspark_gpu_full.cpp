// Full DSpark: CPU draft + GPU verify — compiled with hipcc
#include "rocm_cpp/ck_gemm.h"
#include "cpu_layer.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

int main(int argc, char** argv) {
    const char* path=argc>1?argv[1]:"/tmp/model.trg";
    int draft_L=argc>2?atoi(argv[2]):2;
    int M=argc>3?atoi(argv[3]):8;
    int n_rounds=argc>4?atoi(argv[4]):10;

    // Load .trg header
    int fd=open(path,O_RDONLY);
    size_t fsz=lseek(fd,0,SEEK_END);
    auto p=(const char*)mmap(0,fsz,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    if(memcmp(p,"TRG1",4)) return 1;
    auto r4=[&](int o){uint32_t v;memcpy(&v,p+o,4);return(int)v;};
    auto r8=[&](int o){uint64_t v;memcpy(&v,p+o,8);return v;};
    int H=r4(4),IM=r4(8),NH=r4(12),NKV=r4(16),HD=r4(20),V=r4(24),L=r4(28),GQA=r4(32);
    int ps[7];for(int i=7;i--;)ps[i]=r4(36+i*4);
    uint64_t o_emb=r8(64),o_fn=r8(72),o_lm=r8(80),o_norms=r8(88),o_pk=r8(96),o_sc=r8(104);
    auto F=[&](auto oo){return(const float*)(p+oo);};
    auto U=[&](auto oo){return(const uint32_t*)(p+oo);};
    auto emb=F(o_emb),fn=F(o_fn),lm=F(o_lm);
    auto inorm=F(o_norms),pan=F(o_norms+L*H*4),qn=F(o_norms+2*L*H*4),kn=F(o_norms+2*L*H*4+L*HD*4);

    int per_layer=0, rows[7]={NH*HD,NKV*HD,NKV*HD,H,IM,IM,H},KK[7]={H,H,H,NH*HD,H,H,IM};
    for(int i=7;i--;)per_layer+=ps[i];
    int per_sc=NH*HD+NKV*HD+NKV*HD+H+IM+IM+H;

    printf("=== DSpark GPU ===\n  H=%d L=%d Draft=%dL M=%d\n\n",H,L,draft_L,M);

    // GPU init
    hipStream_t s; hipStreamCreate(&s);
    float *d_emb,*d_fn,*d_lm,*d_inorm,*d_pan,*d_qn,*d_kn,*d_pk,*d_sc,*d_hf,*d_logits;
    _Float16 *d_h,*d_qkv,*d_at,*d_ff,*d_ac; int8_t *d_i8; float *d_xs; int *d_tok;
    auto ml=[&](auto&p_,size_t b){hipMalloc(&p_,b);};
    size_t kv_sz = L*4096*NKV*HD*2;
    float *d_kc, *d_vc;
    ml(d_emb,V*H*4);ml(d_fn,H*4);ml(d_lm,V*H*4);ml(d_inorm,L*H*4);ml(d_pan,L*H*4);
    ml(d_qn,L*HD*4);ml(d_kn,L*HD*4);ml(d_pk,L*per_layer*4);ml(d_sc,L*per_sc*4);
    ml(d_hf,H*4);ml(d_h,H*2);ml(d_qkv,(NH*HD+2*NKV*HD)*2);ml(d_at,NH*HD*2);
    ml(d_ff,2*IM*2);ml(d_ac,IM*2);ml(d_i8,H);ml(d_xs,4);
    ml(d_kc,kv_sz);hipMemset(d_kc,0,kv_sz);
    ml(d_vc,kv_sz);hipMemset(d_vc,0,kv_sz);
    ml(d_logits,V*4);ml(d_tok,4);

    hipMemcpy(d_emb,emb,V*H*4,hipMemcpyHostToDevice);
    hipMemcpy(d_fn,fn,H*4,hipMemcpyHostToDevice);
    hipMemcpy(d_lm,lm,V*H*4,hipMemcpyHostToDevice);
    hipMemcpy(d_inorm,inorm,L*H*4,hipMemcpyHostToDevice);
    hipMemcpy(d_pan,pan,L*H*4,hipMemcpyHostToDevice);
    hipMemcpy(d_qn,qn,L*HD*4,hipMemcpyHostToDevice);
    hipMemcpy(d_kn,kn,L*HD*4,hipMemcpyHostToDevice);
    hipMemcpy(d_pk,U(o_pk),L*per_layer*4,hipMemcpyHostToDevice);
    hipMemcpy(d_sc,F(o_sc),L*per_sc*4,hipMemcpyHostToDevice);
    printf("[GPU] Weights uploaded\n");

    // CPU buffers
    std::vector<float> hd_cpu(H),qkv(NH*HD+2*NKV*HD),at_b(NH*HD),ff_b(2*IM),ac_b(IM);
    std::vector<float> st(4096*HD),ct(4096*HD);
    for(int p2=4096;p2--;)for(int d=HD;d--;){float th=p2/pow(10000.f,(2.f*(d/2))/HD);st[p2*HD+d]=sin(th);ct[p2*HD+d]=cos(th);}
    auto kcv=std::vector<float>(L*4096*NKV*HD,0),vcv=std::vector<float>(L*4096*NKV*HD,0);
    auto cpu_fwd=[&](float*hd,int pos,int nL){
        for(int l=0;l<nL;l++){
            float res[4096];memcpy(res,hd,H*4);
            auto pw=U(o_pk)+l*per_layer;auto sw=F(o_sc)+l*per_sc;
            cpu_rmsnorm(hd,inorm+l*H,hd,H,1e-6f);
            cpu_ternary_gemv(pw,hd,sw,qkv.data(),rows[0],KK[0]);pw+=ps[0];sw+=rows[0];
            cpu_ternary_gemv(pw,hd,sw,qkv.data()+NH*HD,rows[1],KK[1]);pw+=ps[1];sw+=rows[1];
            cpu_ternary_gemv(pw,hd,sw,qkv.data()+NH*HD+NKV*HD,rows[2],KK[2]);pw+=ps[2];sw+=rows[2];
            for(int h=0;h<NH;h++)cpu_rmsnorm(qkv.data()+h*HD,qn+l*HD,qkv.data()+h*HD,HD,1e-6f);
            cpu_rope(qkv.data(),pos,NH,HD,st.data(),ct.data());
            for(int h=0;h<NKV;h++)cpu_rmsnorm(qkv.data()+NH*HD+h*HD,kn+l*HD,qkv.data()+NH*HD+h*HD,HD,1e-6f);
            cpu_rope(qkv.data()+NH*HD,pos,NKV,HD,st.data(),ct.data());
            for(int h=0;h<NKV;h++){memcpy(&kcv[l*4096*NKV*HD+pos*NKV*HD+h*HD],qkv.data()+NH*HD+h*HD,HD*4);memcpy(&vcv[l*4096*NKV*HD+pos*NKV*HD+h*HD],qkv.data()+NH*HD+NKV*HD+h*HD,HD*4);}
            cpu_attention(qkv.data(),&kcv[l*4096*NKV*HD],&vcv[l*4096*NKV*HD],at_b.data(),NH,NKV,HD,pos+1,GQA);
            cpu_ternary_gemv(pw,at_b.data(),sw,hd,rows[3],KK[3]);pw+=ps[3];sw+=rows[3];for(int i=0;i<H;i++)hd[i]=res[i]+hd[i];memcpy(res,hd,H*4);
            cpu_rmsnorm(hd,pan+l*H,hd,H,1e-6f);
            cpu_ternary_gemv(pw,hd,sw,ff_b.data(),rows[4],KK[4]);pw+=ps[4];sw+=rows[4];
            cpu_ternary_gemv(pw,hd,sw,ff_b.data()+IM,rows[5],KK[5]);pw+=ps[5];sw+=rows[5];
            cpu_silu_glu(ff_b.data(),ff_b.data()+IM,ac_b.data(),IM);
            cpu_ternary_gemv(pw,ac_b.data(),sw,hd,rows[6],KK[6]);
            for(int i=0;i<H;i++)hd[i]=res[i]+hd[i];
        }
    };
    auto cpu_argmax=[&](const float*hd){
        float tmp[4096];memcpy(tmp,hd,H*4);cpu_rmsnorm(tmp,fn,tmp,H,1e-6f);
        std::vector<float>lg(V);cpu_lm_head(tmp,lm,lg.data(),V,H);
        int b=0;for(int i=1;i<V;i++)if(lg[i]>lg[b])b=i;return b;
    };

    // GPU forward lambda
    auto gpu_fwd = [&](float* hd, int pos) {
        hipMemcpy(d_hf, hd, H*4, hipMemcpyHostToDevice);
        rcpp_fp32_to_fp16(d_hf, d_h, H, s);
        for(int l=0;l<L;l++){
            hipMemcpy(d_ff, d_h, H*2, hipMemcpyDeviceToDevice);
            rcpp_rmsnorm_fp16(d_h, d_inorm+l*H, d_h, 1e-6f, H, s);
            rcpp_quantize_fp16_to_i8(d_h, d_i8, d_xs, H, s);
            float xs; hipMemcpy(&xs, d_xs, 4, hipMemcpyDeviceToHost); hipStreamSynchronize(s);
            auto wp=(const uint32_t*)((const char*)d_pk+l*per_layer*4);
            auto ws=d_sc+l*per_sc;
            auto g=[&](int pi,_Float16*dst,int M,int K){
                rcpp_ternary_gemv(wp,d_i8,xs,ws,d_hf,M,K,s);
                rcpp_fp32_to_fp16(d_hf,dst,M,s);
                wp+=ps[pi];ws+=rows[pi];
            };
            g(0,d_qkv,NH*HD,H);
            g(1,d_qkv+NH*HD,NKV*HD,H);
            g(2,d_qkv+NH*HD+NKV*HD,NKV*HD,H);
            for(int h=0;h<NH;h++)rcpp_rmsnorm_fp16(d_qkv+h*HD,d_qn+l*HD,d_qkv+h*HD,1e-6f,HD,s);
            rcpp_rope_fp16(d_qkv,pos,1000000.0f,NH,HD,s);
            for(int h=0;h<NKV;h++)rcpp_rmsnorm_fp16(d_qkv+NH*HD+h*HD,d_kn+l*HD,d_qkv+NH*HD+h*HD,1e-6f,HD,s);
            rcpp_rope_fp16(d_qkv+NH*HD,pos,1000000.0f,NKV,HD,s);
            size_t kvo=l*4096*NKV*HD+pos*NKV*HD;
            hipMemcpy(d_kc+kvo,d_qkv+NH*HD,NKV*HD*2,hipMemcpyDeviceToDevice);
            hipMemcpy(d_vc+kvo,d_qkv+NH*HD+NKV*HD,NKV*HD*2,hipMemcpyDeviceToDevice);
            rcpp_kv_cache_attn_decode_fd(d_qkv,d_kc,d_vc,d_at,NH,NKV,HD,pos+1,1.0f/sqrtf(HD),s);
            rcpp_quantize_fp16_to_i8(d_at,d_i8,d_xs,NH*HD,s);
            hipMemcpy(&xs,d_xs,4,hipMemcpyDeviceToHost);hipStreamSynchronize(s);
            rcpp_ternary_gemv(wp,d_i8,xs,ws,d_hf,H,NH*HD,s);wp+=ps[3];ws+=rows[3];
            rcpp_fp32_to_fp16(d_hf,d_h,H,s);
            rcpp_residual_add_fp16(d_h,d_ff,H,s);
            rcpp_rmsnorm_fp16(d_h,d_pan+l*H,d_h,1e-6f,H,s);
            rcpp_quantize_fp16_to_i8(d_h,d_i8,d_xs,H,s);
            hipMemcpy(&xs,d_xs,4,hipMemcpyDeviceToHost);hipStreamSynchronize(s);
            g(4,d_ff,IM,H);
            g(5,d_ff+IM,IM,H);
            rcpp_silu_glu_fp16(d_ff+IM,d_ff,d_ac,IM,s);
            rcpp_quantize_fp16_to_i8(d_ac,d_i8,d_xs,IM,s);
            hipMemcpy(&xs,d_xs,4,hipMemcpyDeviceToHost);hipStreamSynchronize(s);
            rcpp_ternary_gemv(wp,d_i8,xs,ws,d_hf,H,IM,s);
            rcpp_fp32_to_fp16(d_hf,d_h,H,s);
            rcpp_residual_add_fp16(d_h,d_ff,H,s);
        }
        rcpp_rmsnorm_fp16(d_h,d_fn,d_h,1e-6f,H,s);
        rcpp_fp16_gemv(d_lm,d_h,d_logits,V,H,s);
        hipStreamSynchronize(s);
        hipMemcpy(hd,d_hf,H*4,hipMemcpyDeviceToHost);
    };

    // Verify GPU vs CPU match
    printf("── GPU vs CPU ──\n");
    float cpu_hd[4096],gpu_hd[4096];
    memcpy(cpu_hd,emb+1*H,H*4); memcpy(gpu_hd,emb+1*H,H*4);
    auto t0=std::chrono::high_resolution_clock::now(); cpu_fwd(cpu_hd,0,L); auto t1=std::chrono::high_resolution_clock::now();
    double cpu_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    t0=std::chrono::high_resolution_clock::now(); gpu_fwd(gpu_hd,0); t1=std::chrono::high_resolution_clock::now();
    double gpu_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    int cpu_tok=cpu_argmax(cpu_hd), gpu_tok=cpu_argmax(gpu_hd);
    float diff=0;for(int i=0;i<H;i++)diff+=fabs(cpu_hd[i]-gpu_hd[i]);
    printf("  CPU: %d (%.1f ms)\n  GPU: %d (%.1f ms)\n  Diff=%.4f Match=%s\n",
           cpu_tok,cpu_ms,gpu_tok,gpu_ms,diff,cpu_tok==gpu_tok?"YES":"NO");

    // DSpark loop
    printf("\n── DSpark (draft=%dL M=%d %d rounds) ──\n",draft_L,M,n_rounds);
    std::fill(kcv.begin(),kcv.end(),0);std::fill(vcv.begin(),vcv.end(),0);
    float hd_saved[4096]; memcpy(hd_saved,emb+1*H,H*4);
    int pos=0, total_accepted=0;
    double total_draft_ms=0,total_verify_ms=0;
    std::mt19937_64 rng(42);

    // Prime CPU cache
    memcpy(hd_cpu.data(),hd_saved,H*4); cpu_fwd(hd_cpu.data(),0,L);
    pos=1; memcpy(hd_saved,hd_cpu.data(),H*4);
    // Prime GPU cache
    memcpy(gpu_hd,emb+1*H,H*4); gpu_fwd(gpu_hd,0);

    for(int round=0;round<n_rounds;round++){
        float dh[4096]; memcpy(dh,hd_saved,H*4);
        int dtok[32];
        t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<M;i++){cpu_fwd(dh,pos+i,draft_L);dtok[i]=cpu_argmax(dh);}
        t1=std::chrono::high_resolution_clock::now(); double dms=std::chrono::duration<double,std::milli>(t1-t0).count();
        total_draft_ms+=dms;

        memcpy(gpu_hd,hd_saved,H*4);
        int nac=0;
        t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<M;i++){
            gpu_fwd(gpu_hd,pos+i);
            int ttok=cpu_argmax(gpu_hd);
            if(ttok==dtok[i]){nac++;}else break;
        }
        t1=std::chrono::high_resolution_clock::now(); double vms=std::chrono::duration<double,std::milli>(t1-t0).count();
        total_verify_ms+=vms;

        total_accepted+=nac;
        if(nac>0){memcpy(hd_saved,gpu_hd,H*4);pos+=nac;}
        double ts=pos/((total_draft_ms+total_verify_ms)/1000);
        if(round%5==0||round==n_rounds-1)
            printf("  R%2d: acc=%d/%d d=%.0fms v=%.0fms %.0f tok/s\n",
                   round,nac,M,dms,vms,ts);
    }
    double ts=pos/((total_draft_ms+total_verify_ms)/1000);
    printf("\n── Summary ──\n  Draft %dL CPU + Verify GPU\n  %.0f tok/s\n  %.0f%% acceptance (%d/%d)\n",
           draft_L, ts, 100.*total_accepted/(M*n_rounds),total_accepted,M*n_rounds);

    hipFree(d_emb);hipFree(d_fn);hipFree(d_lm);hipFree(d_inorm);hipFree(d_pan);
    hipFree(d_qn);hipFree(d_kn);hipFree(d_pk);hipFree(d_sc);hipFree(d_hf);
    hipFree(d_h);hipFree(d_qkv);hipFree(d_at);hipFree(d_ff);hipFree(d_ac);
    hipFree(d_i8);hipFree(d_xs);hipFree(d_logits);hipFree(d_tok);
    hipFree(d_kc);hipFree(d_vc);hipStreamDestroy(s);
}
