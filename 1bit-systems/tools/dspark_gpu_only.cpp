// DSpark GPU-only: draft (N layers) + verify (all layers), both on GPU in FP16
#include "cpu_layer.h"
// Build: hipcc -O3 --offload-arch=gfx1151 -Iinclude \
//   -o tools/dspark_gpu_only tools/dspark_gpu_only.cpp -Lbuild -lrocm_cpp
// Run:  LD_LIBRARY_PATH=build ./tools/dspark_gpu_only model.trg [draft_L] [M] [rounds]

#include "rocm_cpp/ck_gemm.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#define SYNC hipStreamSynchronize(s)

int main(int argc, char** argv) {
    const char* path=argc>1?argv[1]:"/tmp/model.trg";
    int draft_L=argc>2?atoi(argv[2]):4;
    int M=argc>3?atoi(argv[3]):8;
    int n_rounds=argc>4?atoi(argv[4]):10;

    int fd=open(path,O_RDONLY);
    size_t fsz=lseek(fd,0,SEEK_END);
    auto p=(const char*)mmap(0,fsz,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    if(memcmp(p,"TRG1",4))return 1;
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
    printf("=== DSpark GPU-Only ===\n  H=%d L=%d draft=%dL M=%d\n\n",H,L,draft_L,M);

    // GPU init — full weights + KV cache
    hipStream_t s; hipStreamCreate(&s);
    float *d_pk,*d_sc,*d_inorm,*d_pan,*d_qn,*d_kn,*d_fn,*d_lm,*d_hf,*d_xs;
    _Float16 *d_h,*d_q,*d_at,*d_ffg,*d_ag,*d_kc,*d_vc,*d_lm16; int8_t *d_i8; float xsh;
    auto ml=[&](auto&p_,size_t b){hipMalloc(&p_,b);SYNC;};
    ml(d_pk,L*per_layer*4);ml(d_sc,L*per_sc*4);
    ml(d_inorm,L*H*4);ml(d_pan,L*H*4);ml(d_qn,L*HD*4);ml(d_kn,L*HD*4);
    ml(d_fn,H*4);ml(d_hf,8192*4);ml(d_h,H*2);
    ml(d_q,(NH*HD+2*NKV*HD)*2);ml(d_at,NH*HD*2);ml(d_ffg,2*IM*2);ml(d_ag,IM*2);
    ml(d_i8,H);ml(d_xs,4);int MP=4096;
    ml(d_kc,L*MP*NKV*HD*2);hipMemset(d_kc,0,L*MP*NKV*HD*2);
    ml(d_vc,L*MP*NKV*HD*2);hipMemset(d_vc,0,L*MP*NKV*HD*2);
    // LM head as FP16 (for rcpp_fp16_gemv)
    ml(d_lm16,V*H*2); // FP16 LM head
    ml(d_lm,V*4); // FP32 version as well (for reference)

    auto up=[&](auto d,auto h,size_t b){hipMemcpy(d,h,b,hipMemcpyHostToDevice);SYNC;};
    up(d_pk,U(o_pk),L*per_layer*4);up(d_sc,F(o_sc),L*per_sc*4);
    up(d_inorm,F(o_norms),L*H*4);up(d_pan,F(o_norms)+L*H,L*H*4);
    up(d_qn,F(o_norms)+2*L*H,L*HD*4);up(d_kn,F(o_norms)+2*L*H+L*HD,L*HD*4);
    up(d_fn,F(o_fn),H*4);
    // Upload LM head, convert FP32→FP16 on host
    std::vector<_Float16> lm16(V*H);
    for(int i=0;i<V*H;i++) lm16[i] = (_Float16)F(o_lm)[i]; 
    up(d_lm16,lm16.data(),V*H*2);

    // GPU forward for N layers, returns token via GPU argmax
    auto gpu_fwd=[&](float* result_hidden, int pos, int nL) {
        hipMemcpy(d_hf,result_hidden,H*4,hipMemcpyHostToDevice);SYNC;
        rcpp_fp32_to_fp16(d_hf,d_h,H,s);SYNC;
        for(int l=0;l<nL;l++){
            hipMemcpy(d_ffg,d_h,H*2,hipMemcpyDeviceToDevice);SYNC;
            rcpp_rmsnorm_fp16(d_h,d_inorm+l*H,d_h,1e-6f,H,s);SYNC;
            rcpp_quantize_fp16_to_i8(d_h,d_i8,d_xs,H,s);SYNC;
            hipMemcpy(&xsh,d_xs,4,hipMemcpyDeviceToHost);SYNC;
            auto wp=(const uint32_t*)((const char*)d_pk+l*per_layer*4);
            auto ws=d_sc+l*per_sc;
            rcpp_ternary_gemv(wp,d_i8,xsh,ws,d_hf,NH*HD,H,s);SYNC;wp+=ps[0];ws+=rows[0];
            rcpp_fp32_to_fp16(d_hf,d_q,NH*HD,s);SYNC;
            rcpp_ternary_gemv(wp,d_i8,xsh,ws,d_hf,NKV*HD,H,s);SYNC;wp+=ps[1];ws+=rows[1];
            rcpp_fp32_to_fp16(d_hf,d_q+NH*HD,NKV*HD,s);SYNC;
            rcpp_ternary_gemv(wp,d_i8,xsh,ws,d_hf,NKV*HD,H,s);SYNC;wp+=ps[2];ws+=rows[2];
            rcpp_fp32_to_fp16(d_hf,d_q+NH*HD+NKV*HD,NKV*HD,s);SYNC;
            for(int h=0;h<NH;h++){rcpp_rmsnorm_fp16(d_q+h*HD,d_qn+l*HD,d_q+h*HD,1e-6f,HD,s);SYNC;}
            rcpp_rope_fp16(d_q,pos,1000000.0f,NH,HD,s);SYNC;
            for(int h=0;h<NKV;h++){rcpp_rmsnorm_fp16(d_q+NH*HD+h*HD,d_kn+l*HD,d_q+NH*HD+h*HD,1e-6f,HD,s);SYNC;}
            rcpp_rope_fp16(d_q+NH*HD,pos,1000000.0f,NKV,HD,s);SYNC;
            _Float16*kl=d_kc+l*MP*NKV*HD+pos*NKV*HD;
            _Float16*vl=d_vc+l*MP*NKV*HD+pos*NKV*HD;
            hipMemcpy(kl,d_q+NH*HD,NKV*HD*2,hipMemcpyDeviceToDevice);SYNC;
            hipMemcpy(vl,d_q+NH*HD+NKV*HD,NKV*HD*2,hipMemcpyDeviceToDevice);SYNC;
            rcpp_kv_cache_attn_decode_fd(d_q,d_kc+l*MP*NKV*HD,d_vc+l*MP*NKV*HD,d_at,NH,NKV,HD,pos+1,1.0f/sqrtf(HD),s);SYNC;
            rcpp_quantize_fp16_to_i8(d_at,d_i8,d_xs,NH*HD,s);SYNC;
            hipMemcpy(&xsh,d_xs,4,hipMemcpyDeviceToHost);SYNC;
            rcpp_ternary_gemv(wp,d_i8,xsh,ws,d_hf,H,NH*HD,s);SYNC;wp+=ps[3];ws+=rows[3];
            rcpp_fp32_to_fp16(d_hf,d_h,H,s);SYNC;
            rcpp_residual_add_fp16(d_h,d_ffg,H,s);SYNC;
            rcpp_rmsnorm_fp16(d_h,d_pan+l*H,d_h,1e-6f,H,s);SYNC;
            rcpp_quantize_fp16_to_i8(d_h,d_i8,d_xs,H,s);SYNC;
            hipMemcpy(&xsh,d_xs,4,hipMemcpyDeviceToHost);SYNC;
            rcpp_ternary_gemv(wp,d_i8,xsh,ws,d_hf,IM,H,s);SYNC;wp+=ps[4];ws+=rows[4];
            rcpp_fp32_to_fp16(d_hf,d_ffg,IM,s);SYNC;
            rcpp_ternary_gemv(wp,d_i8,xsh,ws,d_hf,IM,H,s);SYNC;wp+=ps[5];ws+=rows[5];
            rcpp_fp32_to_fp16(d_hf,d_ffg+IM,IM,s);SYNC;
            rcpp_silu_glu_fp16(d_ffg+IM,d_ffg,d_ag,IM,s);SYNC;
            rcpp_quantize_fp16_to_i8(d_ag,d_i8,d_xs,IM,s);SYNC;
            hipMemcpy(&xsh,d_xs,4,hipMemcpyDeviceToHost);SYNC;
            rcpp_ternary_gemv(wp,d_i8,xsh,ws,d_hf,H,IM,s);SYNC;
            rcpp_fp32_to_fp16(d_hf,d_h,H,s);SYNC;
            rcpp_residual_add_fp16(d_h,d_ffg,H,s);SYNC;
        }
        rcpp_rmsnorm_fp16(d_h,d_fn,d_h,1e-6f,H,s);SYNC;
        // Return FP32 hidden state for CPU argmax
        rcpp_fp16_to_fp32(d_h,d_hf,H,s);SYNC;
        hipMemcpy(result_hidden,d_hf,H*4,hipMemcpyDeviceToHost);SYNC;
    };

    // ── DSpark loop ──
    float hd_saved[4096]; memcpy(hd_saved,F(o_emb)+1*H,H*4);
    int pos=1,total_acc=0;
    double t_draft=0,t_verify=0;

    // Prime GPU KV cache with first token
    gpu_fwd(hd_saved,0,L); // first token through full model
    memcpy(hd_saved, F(o_emb)+1*H, H*4); // reset to start

    for(int r=0;r<n_rounds;r++){
        // Draft: generate M tokens on GPU (draft_L layers)
        float dh[4096]; memcpy(dh,hd_saved,H*4);
        int dtok[32];
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<M;i++){
            gpu_fwd(dh,pos+i,draft_L);
            // CPU argmax on hidden state  
            float fh[4096]; memcpy(fh,dh,H*4); cpu_rmsnorm(fh,F(o_fn),fh,H,1e-6f);
            std::vector<float> lg(V); cpu_lm_head(fh,F(o_lm),lg.data(),V,H);
            int b=0; for(int j=1;j<V;j++) if(lg[j]>lg[b]) b=j;
            dtok[i]=b;
        }
        auto t1=std::chrono::high_resolution_clock::now();
        t_draft+=std::chrono::duration<double,std::milli>(t1-t0).count();

        // Verify: full model on GPU — check acceptance
        float vh[4096]; memcpy(vh,hd_saved,H*4);
        int nac=0;
        t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<M;i++){
            // Prime: fill KV cache for ALL layers at position pos+i
            // by running a full forward for M-i tokens
            memcpy(vh,hd_saved,H*4);
            gpu_fwd(vh,pos+i,L); // full model, writes all L layers to KV cache
            float fh[4096]; memcpy(fh,vh,H*4); cpu_rmsnorm(fh,F(o_fn),fh,H,1e-6f);
            std::vector<float> lg(V); cpu_lm_head(fh,F(o_lm),lg.data(),V,H);
            int b=0; for(int j=1;j<V;j++) if(lg[j]>lg[b]) b=j;
            if(b==dtok[i]) nac++; else break;
        }
        t1=std::chrono::high_resolution_clock::now();
        t_verify+=std::chrono::duration<double,std::milli>(t1-t0).count();
        total_acc+=nac;
        if(nac>0){memcpy(hd_saved,vh,H*4);pos+=nac;}
        double ts=pos/((t_draft+t_verify)/1000);
        if(r%5==0||r==n_rounds-1)printf("  R%2d: acc=%d/%d %.0f tok/s\n",r,nac,M,ts);
    }
    double ts=pos/((t_draft+t_verify)/1000);
    printf("\n── Summary ──\n  Draft %dL GPU + Verify GPU (same FP16)\n  %.0f tok/s (%d tok)\n  %.0f%% acceptance\n",
           draft_L,ts,pos,100.*total_acc/(M*n_rounds));
}
