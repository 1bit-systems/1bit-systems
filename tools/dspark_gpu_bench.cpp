// DSpark: CPU draft + GPU verify — full pipeline benchmark
// Build: hipcc -O3 --offload-arch=gfx1151 -Iinclude -Iengine/fusion -Ispec-decode/draft \
//   -o tools/dspark_gpu_bench tools/dspark_gpu_bench.cpp \
//   engine/fusion/cpu_layer.cpp -lm -Lbuild -lrocm_cpp
// Run:  LD_LIBRARY_PATH=build ./tools/dspark_gpu_bench model.trg [M] [rounds] [draft_path]

#include "rocm_cpp/ck_gemm.h"
#include "cpu_layer.h"
#include "../spec-decode/draft/mtp_draft.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cerrno>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <hipblas/hipblas.h>

#define HIP_CHECK(e) do { hipError_t _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP Error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_s)); abort(); } } while(0)
#define SYNC HIP_CHECK(hipStreamSynchronize(s))

int main(int argc, char** argv) {
    const char* path=argc>1?argv[1]:"/tmp/model.trg";
    int M=argc>2?atoi(argv[2]):8;
    int n_rounds=argc>3?atoi(argv[3]):10;

    int fd=open(path,O_RDONLY);
    if (fd<0) { fprintf(stderr,"Cannot open: %s\n",path); return 1; }
    size_t fsz=lseek(fd,0,SEEK_END);
    auto p=(const char*)mmap(0,fsz,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    if(p==MAP_FAILED){perror("mmap");fprintf(stderr,"mmap failed for %s\n",path);return 1;}
    if(memcmp(p,"TRG1",4) && memcmp(p,"TRG2",4)){fprintf(stderr,"Bad magic: %.4s\n",p);return 1;}
    auto r4=[&](int o){uint32_t v;memcpy(&v,p+o,4);return(int)v;};
    auto r8=[&](int o){uint64_t v;memcpy(&v,p+o,8);return v;};
    int H=r4(4),IM=r4(8),NH=r4(12),NKV=r4(16),HD=r4(20),V=r4(24),L=r4(28),GQA=r4(32);
    int ps[7];for(int i=7;i--;)ps[i]=r4(36+i*4);
    int trg_version = memcmp(p,"TRG1",4)==0 ? 1 : 2;
    int offset_base = (trg_version == 2) ? 92 : 64;
    uint64_t o_emb=r8(offset_base),o_fn=r8(offset_base+8),o_lm=r8(offset_base+16),o_norms=r8(offset_base+24),o_pk=r8(offset_base+32),o_sc=r8(offset_base+40);
    auto F=[&](auto oo){return(const float*)(p+oo);};
    auto U=[&](auto oo){return(const uint32_t*)(p+oo);};
    int per_layer=0,rows[7]={NH*HD,NKV*HD,NKV*HD,H,IM,IM,H},KK[7]={H,H,H,NH*HD,H,H,IM};
    for(int i=7;i--;)per_layer+=ps[i];
    // For TRG1: per_sc = sum(rows). For TRG2: per_sc = sum(M * block_sizes[i]) from header
    int ps7[7]; for(int i=7;i--;) ps7[i] = r4(36 + i*4); // pack_size (same as ps)
    int bs7[7]; for(int i=7;i--;) bs7[i] = r4(64 + i*4); // block_sizes (TRG2 only, garbage for TRG1)
    int per_sc = 0;
    if (trg_version == 2) {
        // TRG2: per-block scales, each block = 256 values
        // Scales are [M * block_size] floats per projection
        int sc_rows[7] = {NH*HD, NKV*HD, NKV*HD, H, IM, IM, H};
        for(int i=7;i--;) per_sc += sc_rows[i] * bs7[i];
    } else {
        // TRG1: per-row scales, one scale per output row
        for(int i=7;i--;) per_sc += rows[i];
    }

    printf("=== DSpark GPU Bench ===\n  H=%d L=%d M=%d\n\n",H,L,M);

    // ── Load trained Eagle3 draft checkpoint ──
    const char* draft_path;
    std::string fallback_draft_path;
    if(argc>4){
        draft_path=argv[4];
    }else{
        const char* home=getenv("HOME");
        fallback_draft_path=std::string(home?home:"/tmp")+"/spec-decode/checkpoints/eagle3_draft_trained_420.bin";
        draft_path=fallback_draft_path.c_str();
    }
    // GPU warmup
    HIP_CHECK(hipSetDevice(0));
    {hipStream_t ws;HIP_CHECK(hipStreamCreate(&ws));float*wb;HIP_CHECK(hipMalloc(&wb,64<<20));HIP_CHECK(hipStreamSynchronize(ws));HIP_CHECK(hipFree(wb));HIP_CHECK(hipStreamDestroy(ws));}

    MTPDraftConfig draft_cfg;
    MTPDraftModel draft(draft_cfg);
    MTPDraftState draft_state;
    draft_state.resize(draft_cfg.num_kv_heads, draft_cfg.head_dim, draft_cfg.max_seq);
    if(!draft.load_weights(draft_path)){
        fprintf(stderr,"Failed to load draft checkpoint: %s\n",draft_path);
        fprintf(stderr,"Specify a path as argv[5] or place a trained .bin at ~/spec-decode/checkpoints/\n");
        return 1;
    }
    fprintf(stderr,"Draft checkpoint loaded: %s (vocab=%d hidden=%d target_layers=%d)\n",
            draft_path,draft_cfg.vocab_size,draft_cfg.hidden_size,draft_cfg.num_target_layers);

    fprintf(stderr, "Starting GPU init...\n");
    // ── GPU init ──
    hipStream_t s; HIP_CHECK(hipStreamCreate(&s));
    float *d_pk,*d_sc,*d_inorm,*d_pan,*d_qn,*d_kn,*d_fn,*d_lm,*d_hf,*d_xs;
    _Float16 *d_h,*d_q,*d_at,*d_ffg,*d_ag,*d_kc,*d_vc; int8_t *d_i8; float xsh;
    #define HIPM(p,s) do{hipError_t _e=hipMalloc(&(p),(size_t)(s));if(_e!=hipSuccess){fprintf(stderr,"HIP OOM %s:%d\n",__FILE__,__LINE__);exit(1);}}while(0)
    #define HIPMS() do{hipError_t _e=hipStreamSynchronize(s);if(_e!=hipSuccess){fprintf(stderr,"HIP SYNC %s:%d\n",__FILE__,__LINE__);exit(1);}}while(0)
    HIPM(d_pk,(size_t)L*per_layer*4);
    HIPM(d_sc,(size_t)L*per_sc*4);
    HIPM(d_inorm,(size_t)L*H*4);
    HIPM(d_pan,(size_t)L*H*4);
    HIPM(d_qn,(size_t)L*HD*4);
    HIPM(d_kn,(size_t)L*HD*4);
    HIPM(d_fn,(size_t)H*4);
    HIPM(d_lm,(size_t)V*H*4);
    HIPM(d_hf,8192*4);
    float* d_logits; HIPM(d_logits,(size_t)V*4);
    HIPM(d_h,(size_t)H*2);
    HIPM(d_q,(size_t)(NH*HD+2*NKV*HD)*2);
    HIPM(d_at,(size_t)NH*HD*2);
    HIPM(d_ffg,(size_t)2*IM*2);
    HIPM(d_ag,(size_t)IM*2);
    HIPM(d_i8,(size_t)H);
    HIPM(d_xs,4);
    int MP=4096;
    HIPM(d_kc,(size_t)L*MP*NKV*HD*2); HIP_CHECK(hipMemset(d_kc,0,(size_t)L*MP*NKV*HD*2));
    HIPM(d_vc,(size_t)L*MP*NKV*HD*2); HIP_CHECK(hipMemset(d_vc,0,(size_t)L*MP*NKV*HD*2));
    HIPMS();
    // Aligned temp buffer for GPU DMA (page-aligned source avoids GPU memcpy issues)
    #define ALIGNED_UPLOAD(dst,src,nbytes) do{ \
        size_t _n=(size_t)(nbytes); \
        size_t _a=(_n+4095)&~4095; \
        void* _b=aligned_alloc(4096,_a?_a:4096); \
        if(!_b){fprintf(stderr,"aligned_alloc(%zu) failed\\n",_a);exit(1);} \
        memcpy(_b,(src),_n); \
        HIP_CHECK(hipMemcpy((dst),_b,_n,hipMemcpyHostToDevice)); \
        HIP_CHECK(hipStreamSynchronize(s)); \
        free(_b); \
    }while(0)
    ALIGNED_UPLOAD(d_pk,U(o_pk),(size_t)L*per_layer*4);
    ALIGNED_UPLOAD(d_sc,F(o_sc),(size_t)L*per_sc*4);
    ALIGNED_UPLOAD(d_inorm,F(o_norms),(size_t)L*H*4);
    ALIGNED_UPLOAD(d_pan,F(o_norms)+L*H,(size_t)L*H*4);
    ALIGNED_UPLOAD(d_qn,F(o_norms)+2*L*H,(size_t)L*HD*4);
    ALIGNED_UPLOAD(d_kn,F(o_norms)+2*L*H+L*HD,(size_t)L*HD*4);
    ALIGNED_UPLOAD(d_fn,F(o_fn),(size_t)H*4);
    ALIGNED_UPLOAD(d_lm,F(o_lm),(size_t)V*H*4);

    // GPU forward lambda — runs full model on GPU, returns logits in d_logits
    hipblasHandle_t blas; hipblasCreate(&blas);
    auto gpu_fwd=[&](float*hd,int pos){
            HIP_CHECK(hipMemcpy(d_hf,hd,H*4,hipMemcpyHostToDevice));HIP_CHECK(hipStreamSynchronize(s));rcpp_fp32_to_fp16(d_hf,d_h,H,s);SYNC;
        for(int l=0;l<L;l++){
            HIP_CHECK(hipMemcpy(d_ffg,d_h,H*2,hipMemcpyDeviceToDevice));SYNC;
            rcpp_rmsnorm_fp16(d_h,d_inorm+l*H,d_h,1e-6f,H,s);SYNC;
            rcpp_quantize_fp16_to_i8(d_h,d_i8,d_xs,H,s);SYNC;
            HIP_CHECK(hipMemcpy(&xsh,d_xs,4,hipMemcpyDeviceToHost));SYNC;
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
            HIP_CHECK(hipMemcpy(kl,d_q+NH*HD,NKV*HD*2,hipMemcpyDeviceToDevice));SYNC;
            HIP_CHECK(hipMemcpy(vl,d_q+NH*HD+NKV*HD,NKV*HD*2,hipMemcpyDeviceToDevice));SYNC;
            rcpp_kv_cache_attn_decode_fd(d_q,d_kc+l*MP*NKV*HD,d_vc+l*MP*NKV*HD,d_at,NH,NKV,HD,pos+1,1.0f/sqrtf(HD),s);SYNC;
            rcpp_quantize_fp16_to_i8(d_at,d_i8,d_xs,NH*HD,s);SYNC;
            HIP_CHECK(hipMemcpy(&xsh,d_xs,4,hipMemcpyDeviceToHost));SYNC;
            rcpp_ternary_gemv(wp,d_i8,xsh,ws,d_hf,H,NH*HD,s);SYNC;wp+=ps[3];ws+=rows[3];
            rcpp_fp32_to_fp16(d_hf,d_h,H,s);SYNC;
            rcpp_residual_add_fp16(d_h,d_ffg,H,s);SYNC;
            rcpp_rmsnorm_fp16(d_h,d_pan+l*H,d_h,1e-6f,H,s);SYNC;
            rcpp_quantize_fp16_to_i8(d_h,d_i8,d_xs,H,s);SYNC;
            HIP_CHECK(hipMemcpy(&xsh,d_xs,4,hipMemcpyDeviceToHost));SYNC;
            rcpp_ternary_gemv(wp,d_i8,xsh,ws,d_hf,IM,H,s);SYNC;wp+=ps[4];ws+=rows[4];
            rcpp_fp32_to_fp16(d_hf,d_ffg,IM,s);SYNC;
            rcpp_ternary_gemv(wp,d_i8,xsh,ws,d_hf,IM,H,s);SYNC;wp+=ps[5];ws+=rows[5];
            rcpp_fp32_to_fp16(d_hf,d_ffg+IM,IM,s);SYNC;
            rcpp_silu_glu_fp16(d_ffg+IM,d_ffg,d_ag,IM,s);SYNC;
            rcpp_quantize_fp16_to_i8(d_ag,d_i8,d_xs,IM,s);SYNC;
            HIP_CHECK(hipMemcpy(&xsh,d_xs,4,hipMemcpyDeviceToHost));SYNC;
            rcpp_ternary_gemv(wp,d_i8,xsh,ws,d_hf,H,IM,s);SYNC;
            rcpp_fp32_to_fp16(d_hf,d_h,H,s);SYNC;
            rcpp_residual_add_fp16(d_h,d_ffg,H,s);SYNC;
        }
        // Final RMS norm + LM head on GPU via hipBLAS
        rcpp_rmsnorm_fp16(d_h,d_fn,d_h,1e-6f,H,s);SYNC;
        rcpp_fp16_to_fp32(d_h,d_hf,H,s);SYNC;
        // d_logits = d_hf @ d_lm^T  (row-vector * matrix = [H] @ [V×H] → [V])
        // hipblasSgemv: y = α·op(A)·x + β·y  (column-major)
        // Our lm_head is [V, H] row-major. We want: logits[v] = sum_h(hidden[h] * lm_head[v*H+h])
        // In column-major terms: same layout, so op(A)=N, A is V×H, x is H, y is V
        float _a=1.0f,_b=0.0f;
        hipblasSgemv(blas,HIPBLAS_OP_N,V,H,&_a,d_lm,V,d_hf,1,&_b,d_logits,1);
        HIP_CHECK(hipStreamSynchronize(s));
    };

    // ── CPU draft forward (real Eagle3) ──
    int n_target = draft_cfg.num_target_layers;
    auto kcv=std::vector<float>(L*4096*NKV*HD,0),vcv=std::vector<float>(L*4096*NKV*HD,0);
    std::vector<float> qkvv(NH*HD+2*NKV*HD),atv(NH*HD),ffv(2*IM),acv(IM);
    std::vector<float> st(4096*HD),ct(4096*HD);
    for(int p2=0;p2<4096;p2++)for(int d=0;d<HD;d++){float th=p2/pow(10000.f,(2.f*(d/2))/HD);st[p2*HD+d]=sin(th);ct[p2*HD+d]=cos(th);}
    // Run first n_target target layers on CPU, collect intermediate hidden states,
    // then feed them to the trained Eagle3 draft model for real spec-decode (fixes #57).
    auto cpu_draft=[&](float*hd,int pos,int input_id)->int{
        std::vector<float> layer_out(n_target*H);
        float hidden[4096];memcpy(hidden,hd,H*4);
        for(int l=0;l<n_target;l++){
            float res[4096];memcpy(res,hidden,H*4);
            auto pw=U(o_pk)+l*per_layer;auto sw=F(o_sc)+l*per_sc;
            cpu_rmsnorm(hidden,F(o_norms)+l*H,hidden,H,1e-6f);
            cpu_ternary_gemv(pw,hidden,sw,qkvv.data(),rows[0],KK[0]);pw+=ps[0];sw+=rows[0];
            cpu_ternary_gemv(pw,hidden,sw,qkvv.data()+NH*HD,rows[1],KK[1]);pw+=ps[1];sw+=rows[1];
            cpu_ternary_gemv(pw,hidden,sw,qkvv.data()+NH*HD+NKV*HD,rows[2],KK[2]);pw+=ps[2];sw+=rows[2];
            for(int h=0;h<NH;h++)cpu_rmsnorm(qkvv.data()+h*HD,F(o_norms)+2*L*H+l*HD,qkvv.data()+h*HD,HD,1e-6f);
            cpu_rope(qkvv.data(),pos,NH,HD,st.data(),ct.data());
            for(int h=0;h<NKV;h++)cpu_rmsnorm(qkvv.data()+NH*HD+h*HD,F(o_norms)+2*L*H+L*HD+l*HD,qkvv.data()+NH*HD+h*HD,HD,1e-6f);
            cpu_rope(qkvv.data()+NH*HD,pos,NKV,HD,st.data(),ct.data());
            for(int h=0;h<NKV;h++){memcpy(&kcv[l*4096*NKV*HD+pos*NKV*HD+h*HD],qkvv.data()+NH*HD+h*HD,HD*4);memcpy(&vcv[l*4096*NKV*HD+pos*NKV*HD+h*HD],qkvv.data()+NH*HD+NKV*HD+h*HD,HD*4);}
            cpu_attention(qkvv.data(),&kcv[l*4096*NKV*HD],&vcv[l*4096*NKV*HD],atv.data(),NH,NKV,HD,pos+1,GQA);
            cpu_ternary_gemv(pw,atv.data(),sw,hidden,rows[3],KK[3]);pw+=ps[3];sw+=rows[3];
            for(int i=0;i<H;i++)hidden[i]=res[i]+hidden[i];memcpy(res,hidden,H*4);
            cpu_rmsnorm(hidden,F(o_norms)+L*H+l*H,hidden,H,1e-6f);
            cpu_ternary_gemv(pw,hidden,sw,ffv.data(),rows[4],KK[4]);pw+=ps[4];sw+=rows[4];
            cpu_ternary_gemv(pw,hidden,sw,ffv.data()+IM,rows[5],KK[5]);pw+=ps[5];sw+=rows[5];
            cpu_silu_glu(ffv.data(),ffv.data()+IM,acv.data(),IM);
            cpu_ternary_gemv(pw,acv.data(),sw,hidden,rows[6],KK[6]);
            for(int i=0;i<H;i++)hidden[i]=res[i]+hidden[i];
            memcpy(&layer_out[(size_t)l*H],hidden,H*4);
        }
        std::vector<float> draft_hidden(draft_cfg.hidden_size),draft_logits(draft_cfg.vocab_size);
        draft.forward(layer_out.data(),input_id,pos,draft_state,draft_logits.data(),draft_hidden.data());
        int best=0;for(int i=1;i<draft_cfg.vocab_size;i++)if(draft_logits[i]>draft_logits[best])best=i;
        return best;
    };

    // ── DSpark loop ──
    float hd_saved[4096];memcpy(hd_saved,F(o_emb)+1*H,H*4);
    int pos=1,total_acc=0,current_token=1;
    double t_draft=0,t_verify=0;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> u(0,1);

    // Prime GPU cache with first token
    float prime[4096];memcpy(prime,F(o_emb)+1*H,H*4);
    gpu_fwd(prime,0);
    // Prime CPU cache
    float cpu_p[4096];memcpy(cpu_p,F(o_emb)+1*H,H*4);
    for(int l=0;l<L;l++){
        auto pw=U(o_pk)+l*per_layer;auto sw=F(o_sc)+l*per_sc;
        cpu_rmsnorm(cpu_p,F(o_norms)+l*H,cpu_p,H,1e-6f);
        cpu_ternary_gemv(pw,cpu_p,sw,qkvv.data(),NH*HD,H);
    }

    for(int r=0;r<n_rounds;r++){
        float dh[4096];memcpy(dh,hd_saved,H*4);
        int dtok[32];
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<M;i++){
            int tok=cpu_draft(dh,pos+i,current_token);
            dtok[i]=tok;
            current_token=tok;
        }
        auto t1=std::chrono::high_resolution_clock::now();
        t_draft+=std::chrono::duration<double,std::milli>(t1-t0).count();

        float gh[4096];memcpy(gh,hd_saved,H*4);
        int nac=0,last_acc_tok=current_token;
        t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<M;i++){
            gpu_fwd(gh,pos+i);
            // Read GPU-computed logits directly (avoids CPU lm_head, 311M matmul)
            std::vector<float> lg(V);
            HIP_CHECK(hipMemcpy(lg.data(),d_logits,(size_t)V*4,hipMemcpyDeviceToHost));HIP_CHECK(hipStreamSynchronize(s));
            int b=0;for(int j=1;j<V;j++)if(lg[j]>lg[b])b=j;
            if(b==dtok[i]){nac++;last_acc_tok=b;}else break;
        }
        t1=std::chrono::high_resolution_clock::now();
        t_verify+=std::chrono::duration<double,std::milli>(t1-t0).count();
        total_acc+=nac;
        if(nac>0){memcpy(hd_saved,gh,H*4);pos+=nac;current_token=last_acc_tok;}
        double ts=pos/((t_draft+t_verify)/1000);
        if(r%5==0||r==n_rounds-1)printf("  R%2d: acc=%d/%d %.0f tok/s\n",r,nac,M,ts);
    }
    double ts=pos/((t_draft+t_verify)/1000);
    printf("\n── Summary ──\n  Eagle3 Draft (CPU) + Verify (GPU)\n  %.0f tok/s (%d tok in %.0f ms)\n  %.0f%% acceptance\n",
           ts,pos,t_draft+t_verify,100.*total_acc/(M*n_rounds));
}
