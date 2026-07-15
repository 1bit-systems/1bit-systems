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
extern "C" void batch_attn_launch(const void*, const void*, const void*, void*,
    int M, int NH, int NKV, int HD, int pos, hipStream_t s);
#define SYNC hipStreamSynchronize(s)
#define CK(msg) do{hipError_t e=hipGetLastError();if(e)printf("ERR %s: %s\n",msg,hipGetErrorString(e));}while(0)

int main(int argc, char** argv) {
    const char* path=argc>1?argv[1]:"/tmp/model.trg";
    int draft_L=argc>2?atoi(argv[2]):2, M=argc>3?atoi(argv[3]):4, rounds=argc>4?atoi(argv[4]):3;
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

    hipStream_t s; hipStreamCreate(&s);CK("init");
    float *d_pk,*d_sc,*d_inorm,*d_pan,*d_qn,*d_kn,*d_fn,*d_hf,*d_xs; _Float16 *d_kc,*d_vc;
    _Float16 *d_h[32],*d_q[32],*d_at,*d_ff[32],*d_ac[32]; int8_t *d_i8; float xsh;
    auto ml=[&](auto&p_,size_t b){hipMalloc((void**)&p_,b);CK("ml");hipMemset(p_,0,b);SYNC;};
    ml(d_pk,L*per_layer*4);ml(d_sc,L*per_sc*4);
    ml(d_inorm,L*H*4);ml(d_pan,L*H*4);ml(d_qn,L*HD*4);ml(d_kn,L*HD*4);
    ml(d_fn,H*4);ml(d_hf,8192*32*4);ml(d_xs,4);
    ml(d_at,(size_t)M*NH*HD*2*2);
    for(int m=0;m<M;m++){ml(d_h[m],H*2);ml(d_q[m],(NH*HD+2*NKV*HD)*2);ml(d_ff[m],2*IM*2);ml(d_ac[m],IM*2);}
    ml(d_i8,H); int MP=4096;
    ml(d_kc,(size_t)L*MP*NKV*HD*2);CK("kc");
    ml(d_vc,(size_t)L*MP*NKV*HD*2);CK("vc");
    auto up=[&](auto d,auto h,size_t b){hipMemcpy(d,h,b,hipMemcpyHostToDevice);SYNC;CK("up");};
    up(d_pk,U(o_pk),L*per_layer*4);up(d_sc,F(o_sc),L*per_sc*4);
    up(d_inorm,F(o_norms),L*H*4);up(d_pan,F(o_norms)+L*H,L*H*4);
    up(d_qn,F(o_norms)+2*L*H,L*HD*4);up(d_kn,F(o_norms)+2*L*H+L*HD,L*HD*4);
    up(d_fn,F(o_fn),H*4);CK("upload");

    // Run ONE layer batch forward
    printf("Running single-layer batch...\n");
    float hd_base[4096]; memcpy(hd_base,F(o_emb)+1*H,H*4);
    for(int m=0;m<M;m++){
        hipMemcpy(d_hf+m*8192,hd_base,H*4,hipMemcpyHostToDevice);SYNC;
        rcpp_fp32_to_fp16(d_hf+m*8192,d_h[m],H,s);SYNC;CK("f2h");
    }
    for(int l=0;l<L;l++){ // just ONE layer
        auto wp=(const uint32_t*)((const char*)d_pk+l*per_layer*4);
        auto ws=d_sc+l*per_sc;
        for(int m=0;m<M;m++){
            _Float16* dh=d_h[m];_Float16* qv=d_q[m];_Float16* ffm=d_ff[m];
            hipMemcpy(ffm,dh,H*2,hipMemcpyDeviceToDevice);SYNC;CK("res");
            rcpp_rmsnorm_fp16(dh,d_inorm+l*H,dh,1e-6f,H,s);SYNC;CK("rn");
            rcpp_quantize_fp16_to_i8(dh,d_i8,d_xs,H,s);SYNC;CK("qi8");
            hipMemcpy(&xsh,d_xs,4,hipMemcpyDeviceToHost);SYNC;
            auto wq=wp;auto ws2=ws;
            rcpp_ternary_gemv(wq,d_i8,xsh,ws2,d_hf+m*8192,NH*HD,H,s);SYNC;wq+=ps[0];ws2+=rows[0];CK("qg");
            rcpp_fp32_to_fp16(d_hf+m*8192,qv,NH*HD,s);SYNC;CK("q2h");
            rcpp_ternary_gemv(wq,d_i8,xsh,ws2,d_hf+m*8192,NKV*HD,H,s);SYNC;wq+=ps[1];ws2+=rows[1];CK("kg");
            rcpp_fp32_to_fp16(d_hf+m*8192,qv+NH*HD,NKV*HD,s);SYNC;
            rcpp_ternary_gemv(wq,d_i8,xsh,ws2,d_hf+m*8192,NKV*HD,H,s);SYNC;wq+=ps[2];ws2+=rows[2];CK("vg");
            rcpp_fp32_to_fp16(d_hf+m*8192,qv+NH*HD+NKV*HD,NKV*HD,s);SYNC;
            for(int h=0;h<NH;h++){rcpp_rmsnorm_fp16(qv+h*HD,d_qn+l*HD,qv+h*HD,1e-6f,HD,s);SYNC;}
            rcpp_rope_fp16(qv,0+m,1000000.0f,NH,HD,s);SYNC;CK("rope");
            for(int h=0;h<NKV;h++){rcpp_rmsnorm_fp16(qv+NH*HD+h*HD,d_kn+l*HD,qv+NH*HD+h*HD,1e-6f,HD,s);SYNC;}
            rcpp_rope_fp16(qv+NH*HD,0+m,1000000.0f,NKV,HD,s);SYNC;
            _Float16* kl=d_kc+(size_t)l*MP*NKV*HD+(size_t)(0+m)*NKV*HD;
            _Float16* vl=d_vc+(size_t)l*MP*NKV*HD+(size_t)(0+m)*NKV*HD;
            hipMemcpy(kl,qv+NH*HD,NKV*HD*2,hipMemcpyDeviceToDevice);SYNC;CK("kcpy");
            hipMemcpy(vl,qv+NH*HD+NKV*HD,NKV*HD*2,hipMemcpyDeviceToDevice);SYNC;
            hipMemcpy(d_at+(size_t)m*NH*HD,qv,NH*HD*2,hipMemcpyDeviceToDevice);SYNC;CK("qcpy");
        }
        // BATCH ATTENTION
        printf("  Launching batch_attn l=%d...\n",l);fflush(stdout);
        _Float16* attn_out = d_at + (size_t)M*NH*HD;
        batch_attn_launch(d_at, d_kc+(size_t)l*MP*NKV*HD, d_vc+(size_t)l*MP*NKV*HD,
                        attn_out, M, NH, NKV, HD, 0, s);
        SYNC;CK("batch_attn");
        printf("  batch_attn OK\n");fflush(stdout);
    }
    printf("ALL OK\n");
}
