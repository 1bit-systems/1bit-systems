// Zaya CCA custom ops kernel — conv_qk + qk_means + L2 + RoPE + GQA
// Runtime-configurable: model dimensions passed as kernel parameters.
// Expects Q[qd], K[kd], V[kd] already computed via tiled GEMV
// Outputs attn_out[h_dim] (pre-O_proj)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

#define BS 256

__device__ inline float wsum_cca(float v){for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffffULL, v, o);return v;}
=======
>>>>>>> origin/main
__device__ inline float bsum_cca(float v,float*r){
    int tx=threadIdx.x,w=tx/32,l=tx%32;v=wsum_cca(v);if(!l)r[w]=v;
    __syncthreads();float s=0;
    if(!w){s=(l<(BS/32))?r[l]:0;s=wsum_cca(s);}if(!w&&!l)r[0]=s;
    __syncthreads();return r[0];
}

// Shared memory layout:
//   [0..31]             rbuf[32]
//   [32..32+qkv-1]      sqk[qkv]
//   [32+qkv..32+2*qkv-1]      dw0[qkv]
//   [32+2*qkv..32+3*qkv-1]    dw1[qkv]
//   [32+3*qkv..32+3*qkv+nrot-1]     rc[nrot]
//   [32+3*qkv+nrot..32+3*qkv+2*nrot-1]  rs[nrot]
// Total = (32 + 3*qkv + 2*nrot) * sizeof(float)

__global__ void cca_custom_kernel(
    const __half*q, const __half*k, const __half*v,
    const __half*phs, const __half*csi,
    const float*cdw, const float*cdb, const float*cgw, const float*cgb,
    const float*ks,
    __half*attn_out, __half*ncs, __half*nph,
    int pos, int batch,
    int nq, int nkv, int hd, int nrot, float rope_base, int gc)
{
    int b = blockIdx.x;
    if(b >= batch) return;
    int tx = threadIdx.x;
    const int qd  = nq * hd;
    const int kd  = nkv * hd;
    const int qkv = qd + kd;
    const int gqa = nq / nkv;
    const int h_dim = qd; // hidden = qd for this architecture (tied dims)

    const __half*q_b = q + (size_t)b * qd;
    const __half*k_b = k + (size_t)b * kd;
    const __half*v_b = v + (size_t)b * kd;
    __half*attn_b = attn_out + (size_t)b * h_dim;

    extern __shared__ float smem[];
    float* rbuf = smem;
    float* sqk  = smem + 32;
    float* dw0  = smem + 32 + qkv;
    float* dw1  = smem + 32 + qkv * 2;
    float* rc   = smem + 32 + qkv * 3;
    float* rs   = smem + 32 + qkv * 3 + nrot;

    for(int i=tx;i<qkv;i+=BS) sqk[i] = (i<qd) ? __half2float(q_b[i]) : __half2float(k_b[i-qd]);
    __syncthreads();

    for(int c=tx;c<qkv;c+=BS){
        float p0=__half2float(csi[c]),p1=__half2float(csi[qkv+c]),p2=sqk[c];
        dw0[c]=cdw[c*2]*p0+cdw[c*2+1]*p1+cdb[c];
        dw1[c]=cdw[c*2]*p1+cdw[c*2+1]*p2+cdb[c];
    }
    __syncthreads();

    for(int c=tx;c<qkv;c+=BS){ ncs[c]=csi[qkv+c]; ncs[qkv+c]=__float2half(sqk[c]); }
    __syncthreads();

    for(int oc=tx;oc<qkv;oc+=BS){
        int g=oc/gc,base=g*gc;const float*w=cgw+oc*(size_t)(gc*2);float a=0;
        for(int j=0;j<gc;j++) a+=w[j*2]*dw0[base+j]+w[j*2+1]*dw1[base+j];
        sqk[oc]=a+cgb[oc];
    }
    __syncthreads();

    for(int d=tx;d<hd;d+=BS){for(int h=0;h<nq;h++){int kv=h/gqa;sqk[h*hd+d]=sqk[h*hd+d]+0.5f*__half2float(q_b[h*hd+d])+0.5f*__half2float(k_b[kv*hd+d]);}}
    for(int khv=0;khv<nkv;khv++){for(int d=0;d<hd;d++){float sm=0;for(int g=0;g<gqa;g++)sm+=__half2float(q_b[(khv*gqa+g)*hd+d]);sqk[qd+khv*hd+d]=sqk[qd+khv*hd+d]+0.5f*sm/(float)gqa+0.5f*__half2float(k_b[khv*hd+d]);}}
    __syncthreads();

    float shd=sqrtf((float)hd);
    for(int h=0;h<nq;h++){float s=0;for(int d=tx;d<hd;d+=BS)s+=sqk[h*hd+d]*sqk[h*hd+d];float iv=shd/(sqrtf(bsum_cca(s,rbuf))+1e-12f);for(int d=tx;d<hd;d+=BS)sqk[h*hd+d]*=iv;__syncthreads();}
    for(int khv=0;khv<nkv;khv++){float s=0;for(int d=tx;d<hd;d+=BS)s+=sqk[qd+khv*hd+d]*sqk[qd+khv*hd+d];float iv=shd*ks[khv]/(sqrtf(bsum_cca(s,rbuf))+1e-12f);for(int d=tx;d<hd;d+=BS)sqk[qd+khv*hd+d]*=iv;__syncthreads();}

    const int nrot_half = nrot / 2;
    for(int i=tx;i<nrot_half;i+=BS){float th=pos*powf(rope_base,-2.0f*(float)i/(float)nrot);rc[i]=cosf(th);rs[i]=sinf(th);rc[nrot_half+i]=rc[i];rs[nrot_half+i]=rs[i];}
    __syncthreads();
    for(int idx=tx;idx<(nq+nkv)*nrot;idx+=BS){
        int hh=idx/nrot,d=idx%nrot;float*base=(hh<nq)?(sqk+hh*hd):(sqk+qd+(hh-nq)*hd);
        int d2=d<nrot_half?d+nrot_half:d-nrot_half;float xv=base[d],xw=base[d2];float rh=d<nrot_half?-xw:xw;
        base[d]=xv*rc[d]+rh*rs[d];
    }
    __syncthreads();

    for(int h=0;h<nq;h++){int kv=h/gqa;for(int d=tx;d<hd;d+=BS)attn_b[h*hd+d]=v_b[kv*hd+d];}
}

static inline size_t cca_custom_smem_bytes(int nq, int nkv, int hd, int nrot) {
    int qkv = nq * hd + nkv * hd;
    return (size_t)(32 + 3 * qkv + 2 * nrot) * sizeof(float);
}
