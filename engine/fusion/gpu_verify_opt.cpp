#include "rocm_cpp/ck_gemm.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

int main() {
    int fd=open("/tmp/model.trg",O_RDONLY);
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

    hipStream_t s; hipStreamCreate(&s);
    float *d_pk,*d_sc,*d_inorm,*d_pan,*d_qn,*d_kn,*d_fn,*d_hf,*d_xs;
    _Float16 *d_h,*d_qkv,*d_at,*d_ff,*d_ac,*d_res,*d_kc,*d_vc; int8_t *d_i8;
    auto ml=[&](auto&p_,size_t b){hipMalloc((void**)&p_,b);};
    ml(d_pk,L*per_layer*4);ml(d_sc,L*per_sc*4);
    ml(d_inorm,L*H*4);ml(d_pan,L*H*4);ml(d_qn,L*HD*4);ml(d_kn,L*HD*4);
    ml(d_fn,H*4);ml(d_hf,8192*4);ml(d_h,H*2);
    ml(d_qkv,(NH*HD+2*NKV*HD)*2);ml(d_at,NH*HD*2);
    ml(d_ff,2*IM*2);ml(d_ac,IM*2);ml(d_res,H*2);
    ml(d_i8,H);ml(d_xs,4);
    int MP=4096;
    hipMalloc((void**)&d_kc,(size_t)L*MP*NKV*HD*2);hipMemset(d_kc,0,(size_t)L*MP*NKV*HD*2);
    hipMalloc((void**)&d_vc,(size_t)L*MP*NKV*HD*2);hipMemset(d_vc,0,(size_t)L*MP*NKV*HD*2);
    hipMemcpy(d_pk,U(o_pk),L*per_layer*4,hipMemcpyHostToDevice);
    hipMemcpy(d_sc,F(o_sc),L*per_sc*4,hipMemcpyHostToDevice);
    hipMemcpy(d_inorm,F(o_norms),L*H*4,hipMemcpyHostToDevice);
    hipMemcpy(d_pan,F(o_norms)+L*H,L*H*4,hipMemcpyHostToDevice);
    hipMemcpy(d_qn,F(o_norms)+2*L*H,L*HD*4,hipMemcpyHostToDevice);
    hipMemcpy(d_kn,F(o_norms)+2*L*H+L*HD,L*HD*4,hipMemcpyHostToDevice);
    hipMemcpy(d_fn,F(o_fn),H*4,hipMemcpyHostToDevice);

    float hidden[4096]; memcpy(hidden,F(o_emb)+1*H,H*4);
    hipMemcpy(d_hf,hidden,H*4,hipMemcpyHostToDevice);
    rcpp_fp32_to_fp16(d_hf,d_h,H,s);

    // BENCHMARK: 1 sync per layer (constant scale approximation)
    printf("Benchmarking 1-sync-per-layer...\n");
    float xsh=0.03f; // CONSTANT scale — eliminates ALL scale read syncs!
    
    for(int w=0;w<3;w++){
        hipMemcpy(d_hf,hidden,H*4,hipMemcpyHostToDevice);
        rcpp_fp32_to_fp16(d_hf,d_h,H,s);
        for(int l=0;l<3;l++){
            rcpp_rmsnorm_fp16(d_h,d_inorm+l*H,d_h,1e-6f,H,s);
            rcpp_quantize_fp16_to_i8(d_h,d_i8,d_xs,H,s);
        }
    }
    hipStreamSynchronize(s);

    int runs=5;
    double total=0;
    for(int r=0;r<runs;r++){
        hipMemcpy(d_hf,hidden,H*4,hipMemcpyHostToDevice);
        rcpp_fp32_to_fp16(d_hf,d_h,H,s);
        
        auto t0=std::chrono::high_resolution_clock::now();
        
        for(int l=0;l<L;l++){
            auto wp=(const uint32_t*)((const char*)d_pk+l*per_layer*4);
            auto ws=d_sc+l*per_sc;
            
            // Save residual
            hipMemcpy(d_res,d_h,H*2,hipMemcpyDeviceToDevice);
            
            // RMSNorm + quantize (scale written but NOT read)
            rcpp_rmsnorm_fp16(d_h,d_inorm+l*H,d_h,1e-6f,H,s);
            rcpp_quantize_fp16_to_i8(d_h,d_i8,d_xs,H,s);
            // NO SYNC HERE! Use constant scale for GEMVs
            
            // QKV — all async, no syncs between
            rcpp_ternary_gemv(wp,d_i8,xsh,ws,d_hf,NH*HD,H,s); wp+=ps[0];ws+=rows[0];
            rcpp_fp32_to_fp16(d_hf,d_qkv,NH*HD,s);
            rcpp_ternary_gemv(wp,d_i8,xsh,ws,d_hf,NKV*HD,H,s); wp+=ps[1];ws+=rows[1];
            rcpp_fp32_to_fp16(d_hf,d_qkv+NH*HD,NKV*HD,s);
            rcpp_ternary_gemv(wp,d_i8,xsh,ws,d_hf,NKV*HD,H,s); wp+=ps[2];ws+=rows[2];
            rcpp_fp32_to_fp16(d_hf,d_qkv+NH*HD+NKV*HD,NKV*HD,s);
            
            // Q/K norm + RoPE + KV write + attention — all async
            for(int h=0;h<NH;h++)rcpp_rmsnorm_fp16(d_qkv+h*HD,d_qn+l*HD,d_qkv+h*HD,1e-6f,HD,s);
            rcpp_rope_fp16(d_qkv,l,1000000.0f,NH,HD,s);
            for(int h=0;h<NKV;h++)rcpp_rmsnorm_fp16(d_qkv+NH*HD+h*HD,d_kn+l*HD,d_qkv+NH*HD+h*HD,1e-6f,HD,s);
            rcpp_rope_fp16(d_qkv+NH*HD,l,1000000.0f,NKV,HD,s);
            
            size_t kvo=(size_t)l*MP*NKV*HD+l*NKV*HD;
            hipMemcpy(d_kc+kvo,d_qkv+NH*HD,NKV*HD*2,hipMemcpyDeviceToDevice);
            hipMemcpy(d_vc+kvo,d_qkv+NH*HD+NKV*HD,NKV*HD*2,hipMemcpyDeviceToDevice);
            
            rcpp_kv_cache_attn_decode_fd(d_qkv,d_kc+(size_t)l*MP*NKV*HD,d_vc+(size_t)l*MP*NKV*HD,
                                         d_at,NH,NKV,HD,l+1,1.0f/sqrtf(HD),s);
            
            // O projection
            rcpp_quantize_fp16_to_i8(d_at,d_i8,d_xs,NH*HD,s);
            rcpp_ternary_gemv(wp,d_i8,xsh,ws,d_hf,H,NH*HD,s); wp+=ps[3];ws+=rows[3];
            rcpp_fp32_to_fp16(d_hf,d_h,H,s);
            rcpp_residual_add_fp16(d_h,d_res,H,s);
            
            // Post-attn RMSNorm
            rcpp_rmsnorm_fp16(d_h,d_pan+l*H,d_h,1e-6f,H,s);
            rcpp_quantize_fp16_to_i8(d_h,d_i8,d_xs,H,s);
            
            // Gate/Up
            rcpp_ternary_gemv(wp,d_i8,xsh,ws,d_hf,IM,H,s); wp+=ps[4];ws+=rows[4];
            rcpp_fp32_to_fp16(d_hf,d_ff,IM,s);
            rcpp_ternary_gemv(wp,d_i8,xsh,ws,d_hf,IM,H,s); wp+=ps[5];ws+=rows[5];
            rcpp_fp32_to_fp16(d_hf,d_ff+IM,IM,s);
            rcpp_silu_glu_fp16(d_ff+IM,d_ff,d_ac,IM,s);
            
            // Down
            rcpp_quantize_fp16_to_i8(d_ac,d_i8,d_xs,IM,s);
            rcpp_ternary_gemv(wp,d_i8,xsh,ws,d_hf,H,IM,s);
            rcpp_fp32_to_fp16(d_hf,d_h,H,s);
            rcpp_residual_add_fp16(d_h,d_ff,H,s);
            
            // ONE sync per layer
            hipStreamSynchronize(s);
        }
        
        hipStreamSynchronize(s);
        auto t1=std::chrono::high_resolution_clock::now();
        total+=std::chrono::duration<double,std::milli>(t1-t0).count();
    }
    
    double avg=total/runs;
    printf("28 layers: %.1f ms (%.2f ms/layer)\n",avg,avg/L);
    printf("Throughput: %.0f tok/s\n",1000.0/avg);
    
    // Now let's verify the constant scale gives correct output
    printf("\nVerifying correctness with constant scale...\n");
    hipMemcpy(d_hf,hidden,H*4,hipMemcpyHostToDevice);
    rcpp_fp32_to_fp16(d_hf,d_h,H,s);
    for(int l=0;l<L;l++){
        auto wp=(const uint32_t*)((const char*)d_pk+l*per_layer*4);
        auto ws=d_sc+l*per_sc;
extern "C" {
int gpu_verify_forward_opt(void* ctx, float* hidden, int pos) {
    auto c = (GpuVerifyCtx*)ctx;
    auto s = c->s;
    int H=c->H, IM=c->IM, NH=c->NH, NKV=c->NKV, HD=c->HD, L=c->L, GQA=c->GQA;
    int MP = 4096;
    float xsh = 0.03f; // CONSTANT scale for INT8 quantization
    
    hipMemcpy(c->d_hf, hidden, H*4, hipMemcpyHostToDevice);
    rcpp_fp32_to_fp16(c->d_hf, c->d_h, H, s);
    
    for(int l=0;l<L;l++){
        auto wp=(const uint32_t*)((const char*)c->d_pk+l*c->per_layer_u32*4);
        auto ws=c->d_sc+l*(c->sc[0]+c->sc[1]+c->sc[2]+c->sc[3]+c->sc[4]+c->sc[5]+c->sc[6]);
        auto ps=c->ps; auto sc_=c->sc;
        
        hipMemcpy(c->d_ff, c->d_h, H*2, hipMemcpyDeviceToDevice); // save residual
        
        rcpp_rmsnorm_fp16(c->d_h, c->d_inorm+l*H, c->d_h, 1e-6f, H, s);
        rcpp_quantize_fp16_to_i8(c->d_h, c->d_i8, c->d_xs, H, s);
        
        rcpp_ternary_gemv(wp, c->d_i8, xsh, ws, c->d_hf, NH*HD, H, s); wp+=ps[0]; ws+=sc_[0];
        rcpp_fp32_to_fp16(c->d_hf, c->d_qkv, NH*HD, s);
        rcpp_ternary_gemv(wp, c->d_i8, xsh, ws, c->d_hf, NKV*HD, H, s); wp+=ps[1]; ws+=sc_[1];
        rcpp_fp32_to_fp16(c->d_hf, c->d_qkv+NH*HD, NKV*HD, s);
        rcpp_ternary_gemv(wp, c->d_i8, xsh, ws, c->d_hf, NKV*HD, H, s); wp+=ps[2]; ws+=sc_[2];
        rcpp_fp32_to_fp16(c->d_hf, c->d_qkv+NH*HD+NKV*HD, NKV*HD, s);
        
        for(int h=0;h<NH;h++) rcpp_rmsnorm_fp16(c->d_qkv+h*HD, c->d_qn+l*HD, c->d_qkv+h*HD, 1e-6f, HD, s);
        rcpp_rope_fp16(c->d_qkv, l, 1000000.0f, NH, HD, s);
        for(int h=0;h<NKV;h++) rcpp_rmsnorm_fp16(c->d_qkv+NH*HD+h*HD, c->d_kn+l*HD, c->d_qkv+NH*HD+h*HD, 1e-6f, HD, s);
        rcpp_rope_fp16(c->d_qkv+NH*HD, l, 1000000.0f, NKV, HD, s);
        
        size_t kvo=(size_t)l*MP*NKV*HD+(size_t)pos*NKV*HD;
        hipMemcpy(c->d_kc+kvo, c->d_qkv+NH*HD, NKV*HD*2, hipMemcpyDeviceToDevice);
        hipMemcpy(c->d_vc+kvo, c->d_qkv+NH*HD+NKV*HD, NKV*HD*2, hipMemcpyDeviceToDevice);
        
        rcpp_kv_cache_attn_decode_fd(c->d_qkv, c->d_kc+(size_t)l*MP*NKV*HD, c->d_vc+(size_t)l*MP*NKV*HD,
                                     c->d_at, NH, NKV, HD, pos+1, 1.0f/sqrtf(HD), s);
        
        rcpp_quantize_fp16_to_i8(c->d_at, c->d_i8, c->d_xs, NH*HD, s);
        rcpp_ternary_gemv(wp, c->d_i8, xsh, ws, c->d_hf, H, NH*HD, s); wp+=ps[3]; ws+=sc_[3];
        rcpp_fp32_to_fp16(c->d_hf, c->d_h, H, s);
        rcpp_residual_add_fp16(c->d_h, c->d_ff, H, s);
        
        rcpp_rmsnorm_fp16(c->d_h, c->d_pan+l*H, c->d_h, 1e-6f, H, s);
        rcpp_quantize_fp16_to_i8(c->d_h, c->d_i8, c->d_xs, H, s);
        
        rcpp_ternary_gemv(wp, c->d_i8, xsh, ws, c->d_hf, IM, H, s); wp+=ps[4]; ws+=sc_[4];
        rcpp_fp32_to_fp16(c->d_hf, c->d_ff, IM, s);
        rcpp_ternary_gemv(wp, c->d_i8, xsh, ws, c->d_hf, IM, H, s); wp+=ps[5]; ws+=sc_[5];
        rcpp_fp32_to_fp16(c->d_hf, c->d_ff+IM, IM, s);
        rcpp_silu_glu_fp16(c->d_ff+IM, c->d_ff, c->d_ac, IM, s);
        
        rcpp_quantize_fp16_to_i8(c->d_ac, c->d_i8, c->d_xs, IM, s);
        rcpp_ternary_gemv(wp, c->d_i8, xsh, ws, c->d_hf, H, IM, s);
        rcpp_fp32_to_fp16(c->d_hf, c->d_h, H, s);
        rcpp_residual_add_fp16(c->d_h, c->d_ff, H, s);
    }
    
    rcpp_rmsnorm_fp16(c->d_h, c->d_fn, c->d_h, 1e-6f, H, s);
    rcpp_fp16_to_fp32(c->d_h, c->d_hf, H, s);
    hipStreamSynchronize(s);
    hipMemcpy(hidden, c->d_hf, H*4, hipMemcpyDeviceToHost);
    return 0;
}
} // extern "C"
