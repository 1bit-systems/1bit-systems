// engine/fusion/gpu_verify.hip — GPU verification for DSpark
// Uses librocm_cpp.so for all GPU operations. No custom kernels needed.
//
// Build:
//   hipcc -O3 --offload-arch=gfx1151 -shared -fPIC \
//         -I/home/bcloud/1bit/include \
//         -o build/libgpu_verify.so engine/fusion/gpu_verify.hip \
//         -L/home/bcloud/1bit/build -lrocm_cpp

#include "gpu_verify.h"
#include "rocm_cpp/ck_gemm.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

extern "C" {

struct GpuVerifyCtx {
    int H,IM,NH,NKV,HD,V,L,GQA;
    int ps[7], sc[7], per_pk, per_sc;

    float *d_emb, *d_fn, *d_lm;
    float *d_pk, *d_sc;          // packed weights + scales (device)
    float *d_inorm, *d_pan;
    float *d_qn, *d_kn;
    float *d_h_f32;              // hidden state (FP32 residual stream)
    _Float16 *d_h, *d_qkv, *d_at, *d_ff, *d_ac;  // FP16 buffers
    int8_t *d_i8;                // INT8 quantized activations
    float *d_xs;                 // activation scale
    float *d_logits;
    int *d_token;
    float *d_kc, *d_vc;         // KV cache (FP16)
    hipStream_t s;
};

GpuVerifyCtx* gpu_verify_init(const char* trg_path) {
    int fd = open(trg_path, O_RDONLY);
    size_t fsz = lseek(fd, 0, SEEK_END);
    auto p = (const char*)mmap(0, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (!p || p == MAP_FAILED || memcmp(p, "TRG1", 4)) return nullptr;

    auto r4=[&](int o){uint32_t v;memcpy(&v,p+o,4);return(int)v;};
    auto r8=[&](int o){uint64_t v;memcpy(&v,p+o,8);return v;};
    auto*c=new GpuVerifyCtx();
    c->H=r4(4);c->IM=r4(8);c->NH=r4(12);c->NKV=r4(16);
    c->HD=r4(20);c->V=r4(24);c->L=r4(28);c->GQA=r4(32);
    for(int i=7;i--;)c->ps[i]=r4(36+i*4);
    uint64_t o_emb=r8(64),o_fn=r8(72),o_lm=r8(80),o_norms=r8(88),o_pk=r8(96),o_sc=r8(104);
    auto F=[&](auto o){return(const float*)(p+o);};
    auto U=[&](auto o){return(const uint32_t*)(p+o);};

    c->sc[0]=c->NH*c->HD;c->sc[1]=c->NKV*c->HD;c->sc[2]=c->NKV*c->HD;
    c->sc[3]=c->H;c->sc[4]=c->IM;c->sc[5]=c->IM;c->sc[6]=c->H;
    for(int i=7;i--;){c->per_pk+=c->ps[i];c->per_sc+=c->sc[i];}

    hipStreamCreate(&c->s);
    auto M=[&](auto&p_,size_t b){hipMalloc(&p_,b);};
    auto kv=[&](auto&p_,auto*sp,size_t b){M(p_,b);hipMemcpyAsync(p_,sp,b,hipMemcpyHostToDevice,c->s);};

    M(c->d_pk,c->L*c->per_pk*4); M(c->d_sc,c->L*c->per_sc*4);
    M(c->d_inorm,c->L*c->H*4); M(c->d_pan,c->L*c->H*4);
    M(c->d_qn,c->L*c->HD*4); M(c->d_kn,c->L*c->HD*4);
    M(c->d_emb,c->V*c->H*4); M(c->d_fn,c->H*4); M(c->d_lm,c->V*c->H*4);

    M(c->d_h_f32,c->H*4); M(c->d_h,c->H*2);
    M(c->d_qkv,(c->NH*c->HD+2*c->NKV*c->HD)*2);
    M(c->d_at,c->NH*c->HD*2); M(c->d_ff,2*c->IM*2); M(c->d_ac,c->IM*2);
    M(c->d_i8,c->H); M(c->d_xs,4);

    size_t kv_sz=c->L*4096*c->NKV*c->HD*2;
    M(c->d_kc,kv_sz);hipMemset(c->d_kc,0,kv_sz);
    M(c->d_vc,kv_sz);hipMemset(c->d_vc,0,kv_sz);
    M(c->d_logits,c->V*4); M(c->d_token,4);

    auto s=c->s;
    hipMemcpyAsync(c->d_pk,U(o_pk),c->L*c->per_pk*4,hipMemcpyHostToDevice,s);
    hipMemcpyAsync(c->d_sc,F(o_sc),c->L*c->per_sc*4,hipMemcpyHostToDevice,s);
    hipMemcpyAsync(c->d_inorm,F(o_norms),c->L*c->H*4,hipMemcpyHostToDevice,s);
    hipMemcpyAsync(c->d_pan,F(o_norms)+c->L*c->H,c->L*c->H*4,hipMemcpyHostToDevice,s);
    hipMemcpyAsync(c->d_qn,F(o_norms)+2*c->L*c->H,c->L*c->HD*4,hipMemcpyHostToDevice,s);
    hipMemcpyAsync(c->d_kn,F(o_norms)+2*c->L*c->H+c->L*c->HD,c->L*c->HD*4,hipMemcpyHostToDevice,s);
    hipMemcpyAsync(c->d_emb,F(o_emb),c->V*c->H*4,hipMemcpyHostToDevice,s);
    hipMemcpyAsync(c->d_fn,F(o_fn),c->H*4,hipMemcpyHostToDevice,s);
    hipMemcpyAsync(c->d_lm,F(o_lm),c->V*c->H*4,hipMemcpyHostToDevice,s);
    hipStreamSynchronize(s);
    munmap((void*)p,fsz);
    printf("[GPU] Init: H=%d L=%d NH=%d NKV=%d HD=%d IM=%d V=%d\n",
           c->H,c->L,c->NH,c->NKV,c->HD,c->IM,c->V);
    return c;
}

int gpu_verify_forward(GpuVerifyCtx* c, float* hidden, int pos) {
    auto s=c->s; int H=c->H,IM=c->IM,NH=c->NH,NKV=c->NKV,HD=c->HD,GQA=c->GQA;

    // Upload hidden to GPU, convert to FP16 (sync copy for unpinned host memory)
    hipMemcpy(c->d_h_f32, hidden, H*4, hipMemcpyHostToDevice);
    rcpp_fp32_to_fp16(c->d_h_f32, c->d_h, H, s);

    for(int l=0;l<c->L;l++){
        // Weight ptrs for this layer (byte offsets over void*)
        auto pk = (const uint32_t*)((const char*)c->d_pk + l*c->per_pk*4);
        auto sc = (const float*)((const char*)c->d_sc + l*c->per_sc*4);

        // Save residual (FP16 copy)
        hipMemcpy(c->d_ff, c->d_h, H*2, hipMemcpyDeviceToDevice);

        // RMSNorm → INT8 quant
        rcpp_rmsnorm_fp16(c->d_h, c->d_inorm+l*H, c->d_h, 1e-6f, H, s);
        rcpp_quantize_fp16_to_i8(c->d_h, c->d_i8, c->d_xs, H, s);

        // Read activation scale from device
        float xs_val=0;
        auto get_xs = [&](){ hipMemcpy(&xs_val, c->d_xs, 4, hipMemcpyDeviceToHost); hipStreamSynchronize(s); };

        // QKV ternary GEMV (FP32 output → convert to FP16)
        auto gemv = [&](const uint32_t* wp, const float* wsc, _Float16* dst, int M){
            rcpp_ternary_gemv(wp, c->d_i8, xs_val, wsc, (float*)c->d_qkv, M, H, s);
            rcpp_fp32_to_fp16(c->d_qkv, dst, M, s);
        };

        get_xs();
        auto wp = pk;
        auto wsc = sc;
        gemv(wp, wsc, c->d_qkv, NH*HD);
        wp += c->ps[0]; wsc += c->sc[0];
        gemv(wp, wsc, c->d_qkv+NH*HD, NKV*HD);
        wp += c->ps[1]; wsc += c->sc[1];
        gemv(wp, wsc, c->d_qkv+NH*HD+NKV*HD, NKV*HD);
        wp += c->ps[2]; wsc += c->sc[2];

        // Q/K norm → RoPE (using theta=1000000, standard for Qwen3)
        for(int h=0;h<NH;h++) rcpp_rmsnorm_fp16(c->d_qkv+h*HD,c->d_qn+l*HD,c->d_qkv+h*HD,1e-6f,HD,s);
        rcpp_rope_fp16(c->d_qkv, pos, 1000000.0f, NH, HD, s);
        for(int h=0;h<NKV;h++) rcpp_rmsnorm_fp16(c->d_qkv+NH*HD+h*HD,c->d_kn+l*HD,c->d_qkv+NH*HD+h*HD,1e-6f,HD,s);
        rcpp_rope_fp16(c->d_qkv+NH*HD, pos, 1000000.0f, NKV, HD, s);

        // KV cache write
        size_t kvo = l*4096*NKV*HD + pos*NKV*HD;
        hipMemcpy(c->d_kc+kvo, c->d_qkv+NH*HD, NKV*HD*2, hipMemcpyDeviceToDevice);
        hipMemcpy(c->d_vc+kvo, c->d_qkv+NH*HD+NKV*HD, NKV*HD*2, hipMemcpyDeviceToDevice);

        // Flash attention (GPU!)
        rcpp_kv_cache_attn_decode_fd(c->d_qkv, c->d_kc, c->d_vc, c->d_at,
                                      NH, NKV, HD, pos+1, 1.0f/sqrtf(HD), s);

        // O projection
        rcpp_quantize_fp16_to_i8(c->d_at, c->d_i8, c->d_xs, NH*HD, s);
        get_xs();
        rcpp_ternary_gemv(wp, c->d_i8, xs_val, wsc, (float*)c->d_qkv, H, NH*HD, s);
        rcpp_fp32_to_fp16(c->d_qkv, c->d_h, H, s);
        wp += c->ps[3]; wsc += c->sc[3];

        // Residual add (FP16) + post-attn RMSNorm
        // c->d_ff = saved residual
        rcpp_residual_add_fp16(c->d_h, c->d_ff, H, s);
        rcpp_rmsnorm_fp16(c->d_h, c->d_pan+l*H, c->d_h, 1e-6f, H, s);

        // Gate/Up
        rcpp_quantize_fp16_to_i8(c->d_h, c->d_i8, c->d_xs, H, s);
        get_xs();
        gemv(wp, wsc, c->d_ff, IM); wp += c->ps[4]; wsc += c->sc[4];
        gemv(wp, wsc, c->d_ff+IM, IM); wp += c->ps[5]; wsc += c->sc[5];

        // SiLU GLU
        rcpp_silu_glu_fp16(c->d_ff+IM, c->d_ff, c->d_ac, IM, s); // up, gate → out

        // Down projection
        rcpp_quantize_fp16_to_i8(c->d_ac, c->d_i8, c->d_xs, IM, s);
        get_xs();
        rcpp_ternary_gemv(wp, c->d_i8, xs_val, wsc, (float*)c->d_qkv, H, IM, s);
        rcpp_fp32_to_fp16(c->d_qkv, c->d_h, H, s);

        // FFN residual add
        rcpp_residual_add_fp16(c->d_h, c->d_ff, H, s);
    }

    // Final norm + LM head (on GPU)
    rcpp_rmsnorm_fp16(c->d_h, c->d_fn, c->d_h, 1e-6f, H, s);
    rcpp_fp16_gemv(c->d_lm, c->d_h, c->d_logits, c->V, H, s);
    rcpp_argmax_fp32(c->d_logits, c->d_token, c->V, s);

    // Convert FP16→FP32, copy back (sync copy for unpinned memory)
    rcpp_fp16_to_fp32(c->d_h, c->d_h_f32, H, s);
    hipStreamSynchronize(s);
    hipMemcpy(hidden, c->d_h_f32, H*4, hipMemcpyDeviceToHost);
    return 0;
}

void gpu_verify_free(GpuVerifyCtx* c) {
    if(!c)return;
    hipFree(c->d_emb); hipFree(c->d_fn); hipFree(c->d_lm);
    hipFree(c->d_pk); hipFree(c->d_sc);
    hipFree(c->d_inorm); hipFree(c->d_pan); hipFree(c->d_qn); hipFree(c->d_kn);
    hipFree(c->d_h_f32); hipFree(c->d_h); hipFree(c->d_qkv);
    hipFree(c->d_at); hipFree(c->d_ff); hipFree(c->d_ac);
    hipFree(c->d_i8); hipFree(c->d_xs);
    hipFree(c->d_logits); hipFree(c->d_token);
    hipFree(c->d_kc); hipFree(c->d_vc);
    hipStreamDestroy(c->s);
    delete c;
}

int gpu_verify_hidden_dim(GpuVerifyCtx* c) { return c->H; }
int gpu_verify_num_layers(GpuVerifyCtx* c) { return c->L; }
int gpu_verify_vocab_size(GpuVerifyCtx* c) { return c->V; }

} // extern "C"
