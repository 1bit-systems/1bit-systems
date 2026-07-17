// Zaya CCA custom ops kernel — conv_qk + qk_means + L2 + RoPE + GQA
// Expects Q[QD], K[KD], V[KD] already computed via tiled GEMV
// Outputs attn_out[H] (pre-O_proj)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

#define BS 256
#define NH 8
#define NK 2
#define HD 128
#define H 2048
#define GQA 4
#define QD 1024
#define KD 256
#define QKV 1280
#define GC 128
#define DC 2
#define NROT 64
#define EPS 1e-5f
#define ROPE_BASE 5000000.0f

__device__ inline float wsum_cca(float v){for(int o=16;o>0;o>>=1)v+=__shfl_xor(v,o);return v;}
__device__ inline float bsum_cca(float v,float*r){
    int tx=threadIdx.x,w=tx/32,l=tx%32;v=wsum_cca(v);if(!l)r[w]=v;
    __syncthreads();float s=0;
    if(!w){s=(l<(BS/32))?r[l]:0;s=wsum_cca(s);}if(!w&&!l)r[0]=s;
    __syncthreads();return r[0];
}

// ── Batched CCA custom kernel ──
// Each block handles one token. blockIdx.x = token index in batch.
// Q, K, V, attn_out are [batch × dim] strided arrays
__global__ void cca_custom_kernel(
    const __half*q, const __half*k, const __half*v,
    const __half*phs, const __half*csi,
    const float*cdw, const float*cdb, const float*cgw, const float*cgb,
    const float*ks,
    __half*attn_out, __half*ncs, __half*nph,
    int pos, int batch)
{
    int b = blockIdx.x;        // token index
    if(b >= batch) return;
    int tx = threadIdx.x;

    // Per-token pointers (batch stride = dim)
    const __half*q_b = q + (size_t)b * QD;
    const __half*k_b = k + (size_t)b * KD;
    const __half*v_b = v + (size_t)b * KD;
    __half*attn_b = attn_out + (size_t)b * H;

    __shared__ float rbuf[32];
    __shared__ float sqk[QKV];
    __shared__ float dw0[QKV], dw1[QKV];
    __shared__ float rc[64], rs[64];

    for(int i=tx;i<QKV;i+=BS) sqk[i] = (i<QD) ? __half2float(q_b[i]) : __half2float(k_b[i-QD]);
    __syncthreads();

    // Read conv state BEFORE updating it (csi and ncs point to same buffer at same offset)
    // Otherwise the shift+store at lines 52-53 would clobber the old state (issue #250).
    for(int c=tx;c<QKV;c+=BS){
        float p0=__half2float(csi[c]),p1=__half2float(csi[QKV+c]),p2=sqk[c];
        dw0[c]=cdw[c*2]*p0+cdw[c*2+1]*p1+cdb[c];
        dw1[c]=cdw[c*2]*p1+cdw[c*2+1]*p2+cdb[c];
    }
    __syncthreads();

    // Update conv state AFTER reading (shift old second half to front, store new sqk at back)
    for(int c=tx;c<QKV;c+=BS){ ncs[c]=csi[QKV+c]; ncs[QKV+c]=__float2half(sqk[c]); }
    __syncthreads();

    for(int oc=tx;oc<QKV;oc+=BS){
        int g=oc/GC,base=g*GC;const float*w=cgw+oc*(size_t)(GC*DC);float a=0;
        for(int j=0;j<GC;j++) a+=w[j*2]*dw0[base+j]+w[j*2+1]*dw1[base+j];
        sqk[oc]=a+cgb[oc];
    }
    __syncthreads();

    for(int h=0;h<NH;h++){int kv=h/GQA;for(int d=0;d<HD;d++) sqk[h*HD+d]=sqk[h*HD+d]+0.5f*__half2float(q_b[h*HD+d])+0.5f*__half2float(k_b[kv*HD+d]);}
    for(int khv=0;khv<NK;khv++){for(int d=0;d<HD;d++){float sm=0;for(int g=0;g<GQA;g++)sm+=__half2float(q_b[(khv*GQA+g)*HD+d]);sqk[QD+khv*HD+d]=sqk[QD+khv*HD+d]+0.5f*sm/(float)GQA+0.5f*__half2float(k_b[khv*HD+d]);}}
    __syncthreads();

    float shd=sqrtf((float)HD);
    for(int h=0;h<NH;h++){float s=0;for(int d=tx;d<HD;d+=BS)s+=sqk[h*HD+d]*sqk[h*HD+d];float iv=shd/(sqrtf(bsum_cca(s,rbuf))+1e-12f);for(int d=tx;d<HD;d+=BS)sqk[h*HD+d]*=iv;__syncthreads();}
    for(int khv=0;khv<NK;khv++){float s=0;for(int d=tx;d<HD;d+=BS)s+=sqk[QD+khv*HD+d]*sqk[QD+khv*HD+d];float iv=shd*ks[khv]/(sqrtf(bsum_cca(s,rbuf))+1e-12f);for(int d=tx;d<HD;d+=BS)sqk[QD+khv*HD+d]*=iv;__syncthreads();}

    // RoPE with position
    for(int i=tx;i<32;i+=BS){float th=pos*powf(ROPE_BASE,-2.0f*(float)i/(float)NROT);rc[i]=cosf(th);rs[i]=sinf(th);rc[32+i]=rc[i];rs[32+i]=rs[i];}
    __syncthreads();
    for(int idx=tx;idx<(NH+NK)*NROT;idx+=BS){
        int hh=idx/NROT,d=idx%NROT;float*base=(hh<NH)?(sqk+hh*HD):(sqk+QD+(hh-NH)*HD);
        int d2=d<32?d+32:d-32;float xv=base[d],xw=base[d2];float rh=d<32?-xw:xw;
        base[d]=xv*rc[d]+rh*rs[d];
    }
    __syncthreads();

    for(int h=0;h<NH;h++){int kv=h/GQA;for(int d=tx;d<HD;d+=BS)attn_b[h*HD+d]=v_b[kv*HD+d];}
}
