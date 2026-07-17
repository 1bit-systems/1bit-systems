// zaya_cca_prep.hip — Reference-faithful CCA Q/K/V preparation for single-token decode.
#define CCP_NH    8
#define CCP_NKV   2
#define CCP_HD    128
#define CCP_GQA   4
#define CCP_QD    (CCP_NH  * CCP_HD)
#define CCP_KD    (CCP_NKV * CCP_HD)
#define CCP_QKV   (CCP_QD + CCP_KD)
#define CCP_GC    128
#define CCP_NROT  64
#define CCP_BS    256
#define CCP_ROPE_BASE 5000000.0f

__device__ __forceinline__ float ccp_wsum(float v){for(int o=16;o>0;o>>=1)v+=__shfl_xor(v,o);return v;}
__device__ __forceinline__ float ccp_bsum(float v,float*r){
    int tx=threadIdx.x,w=tx/32,l=tx%32; v=ccp_wsum(v); if(!l) r[w]=v; __syncthreads();
    float s=0; if(!w){ s=(l<(CCP_BS/32))?r[l]:0; s=ccp_wsum(s);} if(!w&&!l) r[0]=s;
    __syncthreads(); return r[0];
}

__global__ void cca_prep_kernel(
    const __half* __restrict__ q_raw,   // [QD]
    const __half* __restrict__ k_raw,   // [KD]
    const __half* __restrict__ v_cur,   // [KD/2]
    const __half* __restrict__ v_del,   // [KD/2]
    __half* __restrict__ conv_state,    // [2*QKV] in/out
    __half* __restrict__ vrec,          // [KD/2]  in/out
    const float*  __restrict__ cdw,const float*  __restrict__ cdb,
    const float*  __restrict__ cgw,const float*  __restrict__ cgb,
    const float*  __restrict__ ks,
    __half* __restrict__ q_out, __half* __restrict__ k_out, __half* __restrict__ v_out,
    int pos)
{
    const int tx = threadIdx.x;
    __shared__ float rbuf[32];
    __shared__ float sqk[CCP_QKV];
    __shared__ float dw0[CCP_QKV], dw1[CCP_QKV];

    for (int i = tx; i < CCP_QKV; i += CCP_BS)
        sqk[i] = (i < CCP_QD) ? __half2float(q_raw[i]) : __half2float(k_raw[i - CCP_QD]);
    __syncthreads();

    for (int c = tx; c < CCP_QKV; c += CCP_BS) {
        float s0=__half2float(conv_state[c]), s1=__half2float(conv_state[CCP_QKV+c]), cur=sqk[c];
        dw0[c]=cdw[c*2+0]*s0+cdw[c*2+1]*s1+cdb[c];
        dw1[c]=cdw[c*2+0]*s1+cdw[c*2+1]*cur+cdb[c];
    }
    __syncthreads();
    for (int c = tx; c < CCP_QKV; c += CCP_BS) {
        float olds1=__half2float(conv_state[CCP_QKV+c]);
        conv_state[c]=__float2half(olds1); conv_state[CCP_QKV+c]=__float2half(sqk[c]);
    }
    __syncthreads();

    for (int oc = tx; oc < CCP_QKV; oc += CCP_BS) {
        int g=oc/CCP_GC,base=g*CCP_GC; const float* w=cgw+(size_t)oc*(CCP_GC*2); float a=0.f;
        for (int j=0;j<CCP_GC;j++) a+=w[j*2+0]*dw0[base+j]+w[j*2+1]*dw1[base+j];
        sqk[oc]=a+cgb[oc];
    }
    __syncthreads();

    for (int h = tx; h < CCP_NH*CCP_HD; h += CCP_BS) {
        int hh=h/CCP_HD,d=h%CCP_HD,kv=hh/CCP_GQA;
        sqk[hh*CCP_HD+d] += 0.5f*__half2float(q_raw[hh*CCP_HD+d])+0.5f*__half2float(k_raw[kv*CCP_HD+d]);
    }
    __syncthreads();
    for (int h = tx; h < CCP_NKV*CCP_HD; h += CCP_BS) {
        int khv=h/CCP_HD,d=h%CCP_HD; float sm=0.f;
        for (int gq=0;gq<CCP_GQA;gq++) sm+=__half2float(q_raw[(khv*CCP_GQA+gq)*CCP_HD+d]);
        sqk[CCP_QD+khv*CCP_HD+d] += 0.5f*(sm/(float)CCP_GQA)+0.5f*__half2float(k_raw[khv*CCP_HD+d]);
    }
    __syncthreads();

    const float shd=sqrtf((float)CCP_HD);
    for (int h=0;h<CCP_NH;h++){float s=0;for(int d=tx;d<CCP_HD;d+=CCP_BS)s+=sqk[h*CCP_HD+d]*sqk[h*CCP_HD+d];
        float iv=shd/(sqrtf(ccp_bsum(s,rbuf))+1e-12f); for(int d=tx;d<CCP_HD;d+=CCP_BS)sqk[h*CCP_HD+d]*=iv; __syncthreads();}
    for (int khv=0;khv<CCP_NKV;khv++){float s=0;for(int d=tx;d<CCP_HD;d+=CCP_BS)s+=sqk[CCP_QD+khv*CCP_HD+d]*sqk[CCP_QD+khv*CCP_HD+d];
        float iv=shd*ks[khv]/(sqrtf(ccp_bsum(s,rbuf))+1e-12f); for(int d=tx;d<CCP_HD;d+=CCP_BS)sqk[CCP_QD+khv*CCP_HD+d]*=iv; __syncthreads();}

    __shared__ float rc[CCP_NROT], rs[CCP_NROT];
    for (int i=tx;i<CCP_NROT/2;i+=CCP_BS){float th=pos*powf(CCP_ROPE_BASE,-2.0f*(float)i/(float)CCP_NROT);rc[i]=cosf(th);rs[i]=sinf(th);}
    __syncthreads();
    for (int d=CCP_NROT/2+tx;d<CCP_NROT;d+=CCP_BS){rc[d]=rc[d-CCP_NROT/2];rs[d]=rs[d-CCP_NROT/2];}
    __syncthreads();
    const int nrot_total=(CCP_NH+CCP_NKV)*CCP_NROT;
    for (int idx=tx;idx<nrot_total;idx+=CCP_BS){int hh=idx/CCP_NROT,d=idx%CCP_NROT;
        float* base=(hh<CCP_NH)?(sqk+hh*CCP_HD):(sqk+CCP_QD+(hh-CCP_NH)*CCP_HD);
        int d2=(d<CCP_NROT/2)?(d+CCP_NROT/2):(d-CCP_NROT/2);
        float xv=base[d],xw=base[d2],rh=(d<CCP_NROT/2)?-xw:xw; base[d]=xv*rc[d]+rh*rs[d];}
    __syncthreads();

    for(int i=tx;i<CCP_QD;i+=CCP_BS) q_out[i]=__float2half(sqk[i]);
    for(int i=tx;i<CCP_KD;i+=CCP_BS) k_out[i]=__float2half(sqk[CCP_QD+i]);
    __syncthreads();

    for(int i=tx;i<CCP_KD/2;i+=CCP_BS){ v_out[i]=v_cur[i]; v_out[CCP_KD/2+i]=vrec[i]; vrec[i]=v_del[i]; }
}
