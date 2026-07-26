// Zaya EDA Router — pure GPU, no host sync needed
// Runtime-configurable: dimensions passed as kernel parameters.
// Shared memory dynamically sized via extern __shared__.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

static __device__ inline float wsum_f(float v){for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffffULL, v,o);return v;}
static __device__ inline float bsum_f(float v,float*r){
    int tx=threadIdx.x,w=tx/32,l=tx%32;
    v=wsum_f(v);if(!l)r[w]=v;
    __syncthreads();float s=0;
    if(!w){s=(l<8)?r[l]:0;s=wsum_f(s);}if(!w&&!l)r[0]=s;
    __syncthreads();return r[0];
}
__device__ inline float gelu_f(float x){
    float t=tanhf(0.79788456f*(x+0.044715f*x*x*x));
    return 0.5f*x*(1.0f+t);
}

// Shared memory layout (all float except sh_prev_exp which is int in the same block):
//   [0..31]           rbuf[32]
//   [32..32+rtr_h-1]  rs[rtr_h]
//   [32+rtr_h..32+2*rtr_h-1]  r2[rtr_h]
//   [32+2*rtr_h..32+2*rtr_h+rtk-1]  probs[router_top_k]
//   aligned int:      sh_prev_exp (overlaps end of float region, 1 int)
// Total smem bytes = max((32 + 2*rtr_h + rtk)*sizeof(float), (32+2*rtr_h)*sizeof(float)+sizeof(int))

__global__ void eda_router_gpu_kernel(
    const __half*hs, const float*prev_rs, int has_eda, float eda_scale,
    const float*gdw, const float*gdb,
    const float*rfn, const float*rf1, const float*rf1b,
    const float*rf2, const float*rf2b,
    const float*rout, const float*bb,
    float*next_rs, int*expert_idx, float*expert_wt,
    int n_exp, int h_dim, int rtr_h, int router_top_k)
{
    int tx=threadIdx.x;

    extern __shared__ float smem[];
    float* rbuf  = smem;
    float* rs    = smem + 32;
    float* r2    = smem + 32 + rtr_h;
    float* probs = smem + 32 + rtr_h * 2;
    // sh_prev_exp stored as float-bitcast int at the end
    int* sh_prev_exp = (int*)(smem + 32 + rtr_h * 2 + router_top_k);

    // 1. gate_down
    for(int i=tx;i<rtr_h;i+=blockDim.x){
        float s=gdb[i];
        // gdw is transposed [h_dim, rtr_h] for stride-1 access
        for(int j=0;j<h_dim;j++) s+=__half2float(hs[j])*gdw[(size_t)j*rtr_h+i];
        rs[i]=s;
    }
    __syncthreads();
    if(has_eda&&prev_rs){for(int i=tx;i<rtr_h;i+=blockDim.x)rs[i]+=prev_rs[i]*eda_scale;}
    __syncthreads();
    for(int i=tx;i<rtr_h;i+=blockDim.x)next_rs[i]=rs[i];
    __syncthreads();

    // ── Expert caching: reuse previous layer's expert within 90% of best ──
    int reuse_expert = -1;
    if (has_eda && prev_rs) {
        int prev_exp;
        if (tx == 0) {
            prev_exp = *(int*)&prev_rs[rtr_h - 1];
        }
        if (tx == 0) *sh_prev_exp = prev_exp;
        __syncthreads();
        prev_exp = *sh_prev_exp;
        if (prev_exp >= 0 && prev_exp < n_exp) {
            reuse_expert = prev_exp;
        }
    }

    // 2. RMSNorm
    float ss=0; for(int i=tx;i<rtr_h;i+=blockDim.x)ss+=rs[i]*rs[i];
    float rr=1.0f/sqrtf(bsum_f(ss,rbuf)/(float)rtr_h+1e-5f);
    for(int i=tx;i<rtr_h;i+=blockDim.x)rs[i]=rs[i]*rr*rfn[i];
    __syncthreads();

    // 3. fc1+GELU
    for(int i=tx;i<rtr_h;i+=blockDim.x){
        float s=rf1b[i]; for(int j=0;j<rtr_h;j++)s+=rs[j]*rf1[i*(size_t)rtr_h+j];
        r2[i]=gelu_f(s);
    }
    __syncthreads();

    // 4. fc2+GELU
    for(int i=tx;i<rtr_h;i+=blockDim.x){
        float s=rf2b[i]; for(int j=0;j<rtr_h;j++)s+=r2[j]*rf2[i*(size_t)rtr_h+j];
        rs[i]=gelu_f(s);
    }
    __syncthreads();

    // 5. out_proj → scores + bb
    for(int i=tx;i<router_top_k;i+=blockDim.x){
        float s=0; for(int j=0;j<rtr_h;j++)s+=rs[j]*rout[i*(size_t)rtr_h+j];
        probs[i]=s+bb[i];
    }
    __syncthreads();

    // 6. softmax
    float mv=probs[0];for(int i=1;i<router_top_k;i++)if(probs[i]>mv)mv=probs[i];
    float sv=0;for(int i=tx;i<router_top_k;i+=blockDim.x){probs[i]=expf(probs[i]-mv);sv+=probs[i];}
    sv=bsum_f(sv,rbuf);float is=1.0f/(sv+1e-10f);
    for(int i=tx;i<router_top_k;i+=blockDim.x)probs[i]*=is;
    __syncthreads();

    // 7. argmax with expert caching
    if(tx==0){
        int best=0;float bv=probs[0];
        for(int i=1;i<router_top_k;i++)if(probs[i]>bv){bv=probs[i];best=i;}

        float cache_threshold = 0.90f;
        if (reuse_expert >= 0 && reuse_expert < n_exp && best != reuse_expert) {
            float cached_score = probs[reuse_expert];
            if (cached_score >= bv * cache_threshold) {
                best = reuse_expert;
                bv = cached_score;
            }
        }

        *expert_idx=best; *expert_wt=bv;
    }
}

// Precomputed shared memory size helper
static inline size_t eda_router_smem_bytes(int rtr_h, int router_top_k) {
    // rbuf[32] + rs[rtr_h] + r2[rtr_h] + probs[router_top_k] + sh_prev_exp(int)
    size_t floats = 32 + (size_t)rtr_h * 2 + router_top_k;
    size_t bytes = floats * sizeof(float) + sizeof(int); // pad for int alignment
    return bytes;
}

// ── Encode expert_idx into prev_rs[RTR_H-1] for cross-layer expert caching ──
__global__ void encode_expert_cache_kernel(float* prev_rs, const int* expert_idx, int rtr_h) {
    if (threadIdx.x == 0) {
        ((int*)&prev_rs[rtr_h - 1])[0] = expert_idx[0];
    }
}
