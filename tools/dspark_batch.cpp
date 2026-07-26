// DSpark Batch — CPU draft + GPU batch verify (M candidates in ONE forward pass)
// The key insight: rcpp_kv_cache_attn_decode_fd processes N_QUERIES in one GPU
// kernel. We set num_q_heads = M * NH and process all candidates at once.
//
// Build: hipcc -O3 --offload-arch=gfx1151 -Iinclude -Iengine/fusion \
//   -o tools/dspark_batch tools/dspark_batch.cpp \
//   engine/fusion/cpu_layer.cpp -lm -Lbuild -lrocm_cpp
// Run:  LD_LIBRARY_PATH=build ./tools/dspark_batch model.trg [draft_L=2] [M=8] [rounds=10]

#include "rocm_cpp/ck_gemm.h"
#include "cpu_layer.h"
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
    int draft_L=argc>2?atoi(argv[2]):2;
    int M=argc>3?atoi(argv[3]):8;
    int rounds=argc>4?atoi(argv[4]):10;

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
    printf("=== DSpark Batch GPU ===\n  H=%d L=%d draft=%dL M=%d\n\n",H,L,draft_L,M);

    // GPU buffers (large enough for batch M)
    hipStream_t s; hipStreamCreate(&s);
    float *d_pk,*d_sc,*d_inorm,*d_pan,*d_qn,*d_kn,*d_fn,*d_lm;
    float *d_hf,*d_logits,*d_xs;
    _Float16 *d_h[32],*d_qkv[32],*d_at[32],*d_ff[32],*d_ac[32];
    _Float16 *d_kc,*d_vc; int8_t *d_i8; float xsh;
    auto ml=[&](auto&p_,size_t b){hipMalloc(&p_,b);SYNC;};

    ml(d_pk,L*per_layer*4);ml(d_sc,L*per_sc*4);
    ml(d_inorm,L*H*4);ml(d_pan,L*H*4);ml(d_qn,L*HD*4);ml(d_kn,L*HD*4);
    ml(d_fn,H*4);ml(d_lm,V*4);ml(d_hf,M*8192*4);ml(d_logits,V*4);ml(d_xs,4);
    for(int m=0;m<M;m++){
        ml(d_h[m],H*2);ml(d_qkv[m],(NH*HD+2*NKV*HD)*2);
        ml(d_at[m],NH*HD*2);ml(d_ff[m],2*IM*2);ml(d_ac[m],IM*2);
    }
    ml(d_i8,H); int MP=4096;
    ml(d_kc,L*MP*NKV*HD*2);hipMemset(d_kc,0,L*MP*NKV*HD*2);
    ml(d_vc,L*MP*NKV*HD*2);hipMemset(d_vc,0,L*MP*NKV*HD*2);

    auto up=[&](auto d,auto h,size_t b){hipMemcpy(d,h,b,hipMemcpyHostToDevice);SYNC;};
    up(d_pk,U(o_pk),L*per_layer*4);up(d_sc,F(o_sc),L*per_sc*4);
    up(d_inorm,F(o_norms),L*H*4);up(d_pan,F(o_norms)+L*H,L*H*4);
    up(d_qn,F(o_norms)+2*L*H,L*HD*4);up(d_kn,F(o_norms)+2*L*H+L*HD,L*HD*4);
    up(d_fn,F(o_fn),H*4);up(d_lm,F(o_lm),V*4);

    // Batch GPU forward: process M draft tokens in one pass
    auto batch_fwd=[&](float** hd, int pos){
        // Upload all M hidden states  
        for(int m=0;m<M;m++){
            hipMemcpy(d_hf+m*8192, hd[m], H*4, hipMemcpyHostToDevice); SYNC;
            rcpp_fp32_to_fp16(d_hf+m*8192, d_h[m], H, s); SYNC;
        }

        for(int l=0;l<L;l++){
            auto wp=(const uint32_t*)((const char*)d_pk+l*per_layer*4);
            auto ws=d_sc+l*per_sc;

            // Process all M candidates through this layer
            for(int m=0;m<M;m++){
                _Float16* dh = d_h[m];
                _Float16* q = d_qkv[m];
                _Float16* at_m = d_at[m];
                _Float16* ff_m = d_ff[m];
                _Float16* ac_m = d_ac[m];

                // Save residual
                hipMemcpy(ff_m, dh, H*2, hipMemcpyDeviceToDevice); SYNC;

                // RMSNorm + Quant + QKV
                rcpp_rmsnorm_fp16(dh, d_inorm+l*H, dh, 1e-6f, H, s); SYNC;
                rcpp_quantize_fp16_to_i8(dh, d_i8, d_xs, H, s); SYNC;
                hipMemcpy(&xsh, d_xs, 4, hipMemcpyDeviceToHost); SYNC;

                auto wq=wp; auto wsc=ws;
                rcpp_ternary_gemv(wq,d_i8,xsh,wsc,d_hf+m*8192,NH*HD,H,s); SYNC; wq+=ps[0];wsc+=rows[0];
                rcpp_fp32_to_fp16(d_hf+m*8192, q, NH*HD, s); SYNC;
                rcpp_ternary_gemv(wq,d_i8,xsh,wsc,d_hf+m*8192,NKV*HD,H,s); SYNC; wq+=ps[1];wsc+=rows[1];
                rcpp_fp32_to_fp16(d_hf+m*8192, q+NH*HD, NKV*HD, s); SYNC;
                rcpp_ternary_gemv(wq,d_i8,xsh,wsc,d_hf+m*8192,NKV*HD,H,s); SYNC; wq+=ps[2];wsc+=rows[2];
                rcpp_fp32_to_fp16(d_hf+m*8192, q+NH*HD+NKV*HD, NKV*HD, s); SYNC;

                // Q/K norm + RoPE
                for(int h=0;h<NH;h++){rcpp_rmsnorm_fp16(q+h*HD,d_qn+l*HD,q+h*HD,1e-6f,HD,s);SYNC;}
                rcpp_rope_fp16(q, pos+m, 1000000.0f, NH, HD, s); SYNC;
                for(int h=0;h<NKV;h++){rcpp_rmsnorm_fp16(q+NH*HD+h*HD,d_kn+l*HD,q+NH*HD+h*HD,1e-6f,HD,s);SYNC;}
                rcpp_rope_fp16(q+NH*HD, pos+m, 1000000.0f, NKV, HD, s); SYNC;

                // KV cache write (each candidate has its own position pos+m)
                _Float16* kl=d_kc+l*MP*NKV*HD+(pos+m)*NKV*HD;
                _Float16* vl=d_vc+l*MP*NKV*HD+(pos+m)*NKV*HD;
                hipMemcpy(kl, q+NH*HD, NKV*HD*2, hipMemcpyDeviceToDevice); SYNC;
                hipMemcpy(vl, q+NH*HD+NKV*HD, NKV*HD*2, hipMemcpyDeviceToDevice); SYNC;
            }

            // BATCHED attention: process all M queries in ONE kernel call!
            // Concatenate all M queries into one buffer
            _Float16* all_q = d_qkv[0]; // reuse first candidate's buffer for concat
            // Actually we need a larger buffer. Let me use d_hf for concatenation
            float* all_q_f32 = d_hf + M*8192; // temp space after hidden states
            for(int m=0;m<M;m++){
                hipMemcpy(all_q_f32+m*NH*HD, d_qkv[m], NH*HD*2, hipMemcpyDeviceToDevice); SYNC;
            }
            // BATCHED attention: M * NH queries, NKV key-value heads
            rcpp_kv_cache_attn_decode_fd(
                all_q_f32,                    // Q: [M*NH, HD] FP16
                d_kc+l*MP*NKV*HD,            // K cache for layer l
                d_vc+l*MP*NKV*HD,            // V cache for layer l
                all_q_f32,                    // reuse as output
                M*NH, NKV, HD, pos+M,         // seq_len = prefix + M tokens
                1.0f/sqrtf(HD), s); SYNC;

            // Scatter attention outputs + O projection + FFN for each candidate
            for(int m=0;m<M;m++){
                _Float16* attn_out = d_at[m];
                hipMemcpy(attn_out, all_q_f32+m*NH*HD, NH*HD*2, hipMemcpyDeviceToDevice); SYNC;

                _Float16* dh = d_h[m];
                _Float16* ff_m = d_ff[m];
                _Float16* ac_m = d_ac[m];
                _Float16* q = d_qkv[m];

                // O projection
                rcpp_quantize_fp16_to_i8(attn_out,d_i8,d_xs,NH*HD,s); SYNC;
                hipMemcpy(&xsh,d_xs,4,hipMemcpyDeviceToHost); SYNC;
                auto wq=wp+ps[0]+ps[1]+ps[2]; auto wsc=ws+rows[0]+rows[1]+rows[2];
                rcpp_ternary_gemv(wq,d_i8,xsh,wsc,d_hf+m*8192,H,NH*HD,s); SYNC;
                rcpp_fp32_to_fp16(d_hf+m*8192, dh, H, s); SYNC;

                // Residual + post-attn RMSNorm
                rcpp_residual_add_fp16(dh, ff_m, H, s); SYNC;
                rcpp_rmsnorm_fp16(dh, d_pan+l*H, dh, 1e-6f, H, s); SYNC;

                // Gate/Up
                rcpp_quantize_fp16_to_i8(dh,d_i8,d_xs,H,s); SYNC;
                hipMemcpy(&xsh,d_xs,4,hipMemcpyDeviceToHost); SYNC;
                wq=wp+ps[0]+ps[1]+ps[2]+ps[3]; wsc=ws+rows[0]+rows[1]+rows[2]+rows[3];
                rcpp_ternary_gemv(wq,d_i8,xsh,wsc,d_hf+m*8192,IM,H,s); SYNC;
                rcpp_fp32_to_fp16(d_hf+m*8192, ff_m, IM, s); SYNC;
                wq+=ps[4]; wsc+=rows[4];
                rcpp_ternary_gemv(wq,d_i8,xsh,wsc,d_hf+m*8192,IM,H,s); SYNC;
                rcpp_fp32_to_fp16(d_hf+m*8192, ff_m+IM, IM, s); SYNC;

                // SiLU GLU
                rcpp_silu_glu_fp16(ff_m+IM, ff_m, ac_m, IM, s); SYNC;

                // Down
                rcpp_quantize_fp16_to_i8(ac_m,d_i8,d_xs,IM,s); SYNC;
                hipMemcpy(&xsh,d_xs,4,hipMemcpyDeviceToHost); SYNC;
                wq+=ps[5]; wsc+=rows[5];
                rcpp_ternary_gemv(wq,d_i8,xsh,wsc,d_hf+m*8192,H,IM,s); SYNC;
                rcpp_fp32_to_fp16(d_hf+m*8192, dh, H, s); SYNC;

                // FFN residual
                rcpp_residual_add_fp16(dh, ff_m, H, s); SYNC;
            }
        }

        // Final norm + LM head for each candidate
        for(int m=0;m<M;m++){
            rcpp_rmsnorm_fp16(d_h[m], d_fn, d_h[m], 1e-6f, H, s); SYNC;
            rcpp_fp16_to_fp32(d_h[m], d_hf+m*8192, H, s); SYNC;
        }
        // Copy all M hidden states back
        for(int m=0;m<M;m++){
            hipMemcpy(hd[m], d_hf+m*8192, H*4, hipMemcpyDeviceToHost); SYNC;
        }
    };

    // ── DSpark loop ──
    float prefix_hd[4096]; memcpy(prefix_hd, F(o_emb)+1*H, H*4);
    int pos=1, total_acc=0;
    double t_draft=0, t_verify=0;

    // Prime GPU cache
    float prime[4096]; memcpy(prime, F(o_emb)+1*H, H*4);
    float* pp[1] = {prime};
    batch_fwd(pp, 0);
    memcpy(prefix_hd, F(o_emb)+1*H, H*4);

    // CPU draft buffers
    auto kcv=std::vector<float>(L*4096*NKV*HD,0),vcv=std::vector<float>(L*4096*NKV*HD,0);
    std::vector<float> qkvv(NH*HD+2*NKV*HD),atv(NH*HD),ffv(2*IM),acv(IM);
    std::vector<float> st(4096*HD),ct(4096*HD);
    for(int p2=0;p2<4096;p2++)for(int d=0;d<HD;d++){float th=p2/pow(10000.f,(2.f*(d/2))/HD);st[p2*HD+d]=sin(th);ct[p2*HD+d]=cos(th);}
    auto cpu_draft=[&](float*hd,int p,int nL){
        for(int l=0;l<nL;l++){
            float res[4096];memcpy(res.data(),hd,H*4);
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
            for(int i=0;i<H;i++)hd[i]=res[i]+hd[i];memcpy(res.data(),hd,H*4);
            cpu_rmsnorm(hd,F(o_norms)+L*H+l*H,hd,H,1e-6f);
            cpu_ternary_gemv(pw,hd,sw,ffv.data(),rows[4],KK[4]);pw+=ps[4];sw+=rows[4];
            cpu_ternary_gemv(pw,hd,sw,ffv.data()+IM,rows[5],KK[5]);pw+=ps[5];sw+=rows[5];
            cpu_silu_glu(ffv.data(),ffv.data()+IM,acv.data(),IM);
            cpu_ternary_gemv(pw,acv.data(),sw,hd,rows[6],KK[6]);
            for(int i=0;i<H;i++)hd[i]=res[i]+hd[i];
        }
        float th[4096];memcpy(th,hd,H*4);cpu_rmsnorm(th,F(o_fn),th,H,1e-6f);
        std::vector<float>lg(V);cpu_lm_head(th,F(o_lm),lg.data(),V,H);
        int b=0;for(int i=1;i<V;i++)if(lg[i]>lg[b])b=i;return b;
    };

    for(int r=0;r<rounds;r++){
        // CPU draft: generate M sequential tokens
        float dh[32][4096];
        int dtok[32];
        auto t0=std::chrono::high_resolution_clock::now();
        memcpy(dh[0],prefix_hd,H*4);
        for(int i=0;i<M;i++){
            if(i>0)memcpy(dh[i],dh[i-1],H*4); // each token builds on previous
            dtok[i]=cpu_draft(dh[i],pos+i,draft_L);
        }
        auto t1=std::chrono::high_resolution_clock::now();
        t_draft+=std::chrono::duration<double,std::milli>(t1-t0).count();

        // GPU batch verify: process ALL M candidates in ONE forward pass
        float* gpu_hds[32];
        float gh[32][4096];
        for(int i=0;i<M;i++){memcpy(gh[i],prefix_hd,H*4);gpu_hds[i]=gh[i];}

        t0=std::chrono::high_resolution_clock::now();
        batch_fwd(gpu_hds, pos);
        t1=std::chrono::high_resolution_clock::now();
        t_verify+=std::chrono::duration<double,std::milli>(t1-t0).count();

        // Check acceptance: compare GPU top token with draft token for each position
        int nac=0;
        for(int i=0;i<M;i++){
            float th[4096];memcpy(th,gh[i],H*4);cpu_rmsnorm(th,F(o_fn),th,H,1e-6f);
            std::vector<float>lg(V);cpu_lm_head(th,F(o_lm),lg.data(),V,H);
            int b=0;for(int j=1;j<V;j++)if(lg[j]>lg[b])b=j;
            if(b==dtok[i])nac++;else break;
        }
        total_acc+=nac;
        if(nac>0){memcpy(prefix_hd,gh[nac-1],H*4);pos+=nac;}
        double ts=pos/((t_draft+t_verify)/1000);
        if(r%3==0||r==rounds-1)printf("  R%2d: acc=%d/%d %.0f tok/s\n",r,nac,M,ts);
    }
    double ts=pos/((t_draft+t_verify)/1000);
    printf("\n── Summary ──\n  Batch DSpark (M=%d): %.0f tok/s  %.0f%% acceptance\n",
           M,ts,100.*total_acc/(M*rounds));
}
