// laguna_matmul.h — Shared Q4NX matmul for CPU + GPU inference
// Single implementation used by both backend_laguna.cpp and laguna_generate.cpp
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>

// BF16→FP32 conversion relies on little-endian byte order for the
// (bf16 << 16) bit pattern. x86 and AMD GPU are little-endian.
// NOTE: This is a practical constraint, not a theoretical limitation.
// On big-endian hosts, a manual sign/exp/mantissa reconstruction would be needed.
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "1bit-systems requires little-endian host (x86/AMD64)");

// Byte-order guard — 1bit-systems requires little-endian host
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    // little-endian (x86, AMD64) — OK
#elif defined(_MSC_VER) || defined(__i386__) || defined(__x86_64__) || defined(__amd64__)
    // MSVC or x86 target — always little-endian
#else
    #error "1bit-systems requires little-endian host (x86/AMD64). Big-endian not supported."
#endif

inline uint16_t laguna_bf16(const uint8_t* b) { return (uint16_t)b[0]|((uint16_t)b[1]<<8); }
inline float laguna_bf16_to_f32(uint16_t bf) { uint32_t bits=(uint32_t)bf<<16; float f; memcpy(&f,&bits,4); return f; }

// NaN-safe dequant for one Q4NX row (32 elements)
inline void laguna_deq_row(const uint8_t* rd, float* out, int ng, int gs) {
    for(int g=0;g<ng;g++){
        float sc=laguna_bf16_to_f32(laguna_bf16(rd+g*2)),zp=laguna_bf16_to_f32(laguna_bf16(rd+ng*2+g*2));
        const uint8_t*pk=rd+ng*4;
        for(int i=0;i<gs;i++){int bi=g*gs/2+i/2,n=(i&1)?(pk[bi]>>4):(pk[bi]&0x0F);
            float v=(float)n*sc+zp;out[g*gs+i]=(v==v&&v<1e10f&&v>-1e10f)?v:0.0f;}
    }
}

// CPU Q4NX matmul: y[M] = W[K×M] · x[K]
inline void laguna_matmul(float* y, const float* x, const uint8_t* wd, int M, int K,
                          int tr=32, int tc=256, int gs=32) {
    int ng=tc/gs, rb=ng*4+tc/2, tby=tr*rb;
    int n_tr=(K+tr-1)/tr, n_tc=(M+tc-1)/tc;
    std::fill(y,y+M,0.0f);
    std::vector<float> tb(tr*tc);
    for(int trr=0;trr<n_tr;trr++){int rs=trr*tr,re=std::min(rs+tr,K),reff=re-rs;
        for(int tcc=0;tcc<n_tc;tcc++){int cs=tcc*tc,ce=std::min(cs+tc,M),ceff=ce-cs;
            const uint8_t*tp=wd+((trr*(size_t)n_tc+tcc)*tby);
            for(int r=0;r<tr;r++)laguna_deq_row(tp+r*rb,&tb[r*tc],ng,gs);
            for(int c=0;c<ceff;c++){float s=0;for(int r=0;r<reff;r++)s+=x[rs+r]*tb[r*(size_t)tc+c];y[cs+c]+=s;}}}
}

// Math ops
// NOTE: These CPU implementations don't need softplus/silu clamping for activations
// since CPU code uses the numerically stable softplus_inplace in backend_laguna.cpp.
// Any future GPU kernel porting these ops should clamp inputs to prevent exp() overflow.
inline void laguna_rmsnorm(float* o, const float* x, const float* w, int n, float eps=1e-6f) {
    float ss=0;for(int i=0;i<n;i++)ss+=x[i]*x[i];
    float r=1.0f/sqrtf(ss/(float)n+eps);
    for(int i=0;i<n;i++)o[i]=x[i]*r*w[i];
}

inline void laguna_softmax(float* x, int n) {
    float mx=x[0];for(int i=1;i<n;i++)if(x[i]>mx)mx=x[i];
    if (mx == -INFINITY || std::isinf(mx)) {
        float inv = 1.0f / n;
        for (int i = 0; i < n; i++) x[i] = inv;
        return;
    }
    float s=0;for(int i=0;i<n;i++)s+=expf(x[i]-mx);
    float inv=1.0f/(s+1e-10f);
    for(int i=0;i<n;i++)x[i]=expf(x[i]-mx)*inv;
}

inline void laguna_rope(float* q, float* k, int pos, int nh, int nkv, int hd, int rd, float theta) {
    int half=rd/2;
    for(int h=0;h<nh;h++)for(int i=0;i<half;i++){
        float f=1.0f/powf(theta,(2.0f*i)/(float)rd),t=pos*f,c=cosf(t),s=sinf(t);
        int i0=h*hd+i,i1=h*hd+i+half;
        float q0=q[i0],q1=q[i1];q[i0]=q0*c-q1*s;q[i1]=q0*s+q1*c;}
    for(int h=0;h<nkv;h++)for(int i=0;i<half;i++){
        float f=1.0f/powf(theta,(2.0f*i)/(float)rd),t=pos*f,c=cosf(t),s=sinf(t);
        int i0=h*hd+i,i1=h*hd+i+half;
        float k0=k[i0],k1=k[i1];k[i0]=k0*c-k1*s;k[i1]=k0*s+k1*c;}
}

inline float laguna_sigmoid(float x) { return 1.0f/(1.0f+expf(-x)); }

inline void laguna_silu(float* o, const float* g, const float* u, int n) {
    for(int i=0;i<n;i++){float gg=g[i];o[i]=(gg/(1.0f+expf(-gg)))*u[i];}
}
