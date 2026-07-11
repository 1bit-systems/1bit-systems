// Albert-MoE-13 GPU kernels — single-token forward with KV cache
// Architecture: 31 blocks × 2 streams, H=256, 4 heads, 12 experts (top-3)
//
// Build: cmake --build . --target albert_moe_gpu -j8

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

#define H        256
#define N_HEADS  4
#define HD       64
#define N_EXP    12
#define TOP_K    3
#define FFN_HID  (H*4)   // 1024
#define VOCAB    32000
#define MAX_SEQ  256
#define RO_THETA 10000.0f

// ── Helpers ──
__device__ inline float wsum(float v){for(int o=16;o>0;o>>=1)v+=__shfl_xor(v,o);return v;}
__device__ inline float bsum(float v,float*r){
    int tx=threadIdx.x,w=tx/32,l=tx%32;v=wsum(v);if(!l)r[w]=v;
    __syncthreads();float s=0;
    if(!w){s=(l<(256/32))?r[l]:0;s=wsum(s);}if(!w&&!l)r[0]=s;
    __syncthreads();return r[0];
}

// ── LayerNorm [H=256] ──
__global__ void layernorm_k(float*x,const float*w,const float*b,int n){
    __shared__ float red[32];int tx=threadIdx.x;
    float mn=0;for(int i=tx;i<n;i+=blockDim.x)mn+=x[i];mn=bsum(mn,red)/(float)n;
    float vr=0;for(int i=tx;i<n;i+=blockDim.x){float d=x[i]-mn;vr+=d*d;}vr=sqrtf(bsum(vr,red)/(float)n+1e-5f);
    for(int i=tx;i<n;i+=blockDim.x)x[i]=(x[i]-mn)/vr*w[i]+b[i];
}

// ── Matmul: out[M] = in[K] @ wt[M,K]^T ──
__global__ void mm_k(float*out,const float*in,const float*wt,int M,int K){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=M)return;
    float s=0;for(int k=0;k<K;k++)s+=in[k]*wt[i*(size_t)K+k];
    out[i]=s;
}

// ── Matmul transposed: out[M] = in[K] @ wt[K,M] ──
__global__ void mm_t_k(float*out,const float*in,const float*wt,int M,int K){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=M)return;
    float s=0;for(int k=0;k<K;k++)s+=in[k]*wt[k*(size_t)M+i];
    out[i]=s;
}

// ── GELU ──
__global__ void gelu_k(float*x,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    float v=x[i];x[i]=0.5f*v*(1.0f+tanhf(0.79788456f*(v+0.044715f*v*v*v)));
}

// ── Copy ──
__global__ void copy_k(float*dst,const float*src,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;dst[i]=src[i];
}

// ── Add ──
__global__ void add_k(float*dst,const float*a,const float*b,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;dst[i]=a[i]+b[i];
}

// ── Embed + scale ──
__global__ void embed_k(float*h,const float*embed,int token){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=H)return;
    h[i]=embed[token*(size_t)H+i];
}

// ── Attention (single token, KV-cached) ──
// Input:  q[H], k[H], v[H] (pre-projected, before RoPE)
//         k_cache[pos][n_heads*HD], v_cache[pos][n_heads*HD]
//         pos, cos_cache[pos][HD/2], sin_cache[pos][HD/2]
// Output: attn_out[H]
// Side:   updates k_cache[pos] with RoPE'd K, v_cache[pos] with V
__global__ void attn_k(float*attn_out,float*q,float*k,float*v,
    float*k_cache,float*v_cache,int pos,const float*cos_,const float*sin_)
{
    int tx=threadIdx.x;
    __shared__ float scores[MAX_SEQ];
    __shared__ float red[32];

    // ── Apply RoPE to Q and K (first HD/2 dims of each head) ──
    for(int h=0;h<N_HEADS;h++){
        float*qh=q+h*HD,*kh=k+h*HD;
        if(tx<HD/2){float q1=qh[tx],q2=qh[tx+HD/2];float c=cos_[pos*HD/2+tx],s=sin_[pos*HD/2+tx];
            qh[tx]=q1*c-q2*s;qh[tx+HD/2]=q1*s+q2*c;
            float k1=kh[tx],k2=kh[tx+HD/2];kh[tx]=k1*c-k2*s;kh[tx+HD/2]=k1*s+k2*c;}
    }
    __syncthreads();

    // Store K (rope'd) and V in cache
    if(tx<N_HEADS*HD){k_cache[pos*N_HEADS*HD+tx]=k[tx];v_cache[pos*N_HEADS*HD+tx]=v[tx];}
    __syncthreads();

    // Attention scores: q[h] @ k_cache[t][h] for all t ≤ pos
    float inv_sqrt_hd = 1.0f/sqrtf((float)HD);
    for(int h=0;h<N_HEADS;h++){
        float*qh=q+h*HD;
        for(int t=tx;t<=pos;t+=blockDim.x){
            float*kh=&k_cache[t*N_HEADS*HD+h*HD];
            float s=0;for(int d=0;d<HD;d++)s+=qh[d]*kh[d];
            scores[t]=s*inv_sqrt_hd;
        }
        __syncthreads();

        // Softmax
        float mv=scores[0];for(int t=1;t<=pos;t++)if(scores[t]>mv)mv=scores[t];
        float su=0;for(int t=tx;t<=pos;t+=blockDim.x){scores[t]=expf(scores[t]-mv);su+=scores[t];}
        su=bsum(su,red);float is=1.0f/(su+1e-10f);
        for(int t=tx;t<=pos;t+=blockDim.x)scores[t]*=is;
        __syncthreads();

        // Weighted sum of V
        float*out=attn_out+h*HD;
        for(int d=tx;d<HD;d+=blockDim.x){
            float s=0;for(int t=0;t<=pos;t++)s+=scores[t]*v_cache[t*N_HEADS*HD+h*HD+d];
            out[d]=s;
        }
        __syncthreads();
    }
}

// ── MoE Router + Top-3 + Expert FFN ──
// Input:  h[H] (post-LN2), expert weights (fp16) for selected expert block
// Output: moe_out[H]
__global__ void moe_k(float*moe_out,const float*h,
    const float*gate_w,       // [N_EXP×H]
    const float*fc_w,         // [H×FFN_HID] (selected expert)
    const float*fc_b,         // [FFN_HID]
    const float*proj_w,       // [FFN_HID×H]
    const float*proj_b        // [H]
){
    int tx=threadIdx.x;
    __shared__ float gs[N_EXP];
    __shared__ float probs[N_EXP];
    __shared__ float sh[32];
    __shared__ float fc[FFN_HID];
    __shared__ float proj[H];

    // ── Router: gs[e] = h @ gate_w[e]^T ──
    for(int e=tx;e<N_EXP;e+=blockDim.x){
        float s=0;for(int j=0;j<H;j++)s+=h[j]*gate_w[e*(size_t)H+j];
        gs[e]=s;
    }
    __syncthreads();

    // Softmax
    float mv=gs[0];for(int e=1;e<N_EXP;e++)if(gs[e]>mv)mv=gs[e];
    float su=0;for(int e=tx;e<N_EXP;e+=blockDim.x){probs[e]=expf(gs[e]-mv);su+=probs[e];}
    su=bsum(su,sh);float is=1.0f/(su+1e-10f);
    for(int e=tx;e<N_EXP;e+=blockDim.x)probs[e]*=is;
    __syncthreads();

    // Top-3 selection (serial on tx==0)
    int top_i[TOP_K];float top_v[TOP_K];
    if(tx==0){
        for(int k=0;k<TOP_K;k++){top_i[k]=0;top_v[k]=-1e9f;}
        for(int e=0;e<N_EXP;e++)
            for(int k=0;k<TOP_K;k++)
                if(probs[e]>top_v[k]){
                    for(int k2=TOP_K-1;k2>k;k2--){top_i[k2]=top_i[k2-1];top_v[k2]=top_v[k2-1];}
                    top_i[k]=e;top_v[k]=probs[e];break;
                }
    }
    // Broadcast top_i to all threads
    __shared__ int s_top_i[TOP_K];__shared__ float s_top_v[TOP_K];
    if(tx==0){for(int k=0;k<TOP_K;k++){s_top_i[k]=top_i[k];s_top_v[k]=top_v[k];}}
    __syncthreads();

    // ── Expert FFN (accumulate top-3 results) ──
    for(int j=tx;j<H;j+=blockDim.x)moe_out[j]=0;__syncthreads();

    for(int k=0;k<TOP_K;k++){
        int e=s_top_i[k];float w=s_top_v[k];

        // c_fc: fc[FFN_HID] = h @ fc_w[e]^T (using shared weights for this block)
        for(int j=tx;j<FFN_HID;j+=blockDim.x){
            float s=fc_b[j];for(int d=0;d<H;d++)s+=h[d]*fc_w[j*(size_t)H+d];
            fc[j]=s;
        }
        __syncthreads();
        // GELU in place
        for(int j=tx;j<FFN_HID;j+=blockDim.x){float v=fc[j];fc[j]=0.5f*v*(1.0f+tanhf(0.79788456f*(v+0.044715f*v*v*v)));}
        __syncthreads();
        // c_proj: proj[H] = fc @ proj_w[e]^T
        for(int j=tx;j<H;j+=blockDim.x){
            float s=proj_b[j];for(int d=0;d<FFN_HID;d++)s+=fc[d]*proj_w[j*(size_t)FFN_HID+d];
            proj[j]=s*w; // weight by top-3 score
        }
        __syncthreads();
        // Accumulate
        for(int j=tx;j<H;j+=blockDim.x)moe_out[j]+=proj[j];
        __syncthreads();
    }
}

// ── lm_head: logits[VOCAB] = h[H] @ lm_head_w[VOCAB×H]^T, then argmax ──
__global__ void lm_head_k(float*logits,int*argmax,const float*h,const float*lm_head_w){
    int tx=threadIdx.x;
    for(int v=tx;v<VOCAB;v+=blockDim.x){
        float s=0;for(int j=0;j<H;j++)s+=h[j]*lm_head_w[v*(size_t)H+j];
        logits[v]=s;
    }
    __syncthreads();
    if(tx==0){int b=0;for(int v=1;v<VOCAB;v++)if(logits[v]>logits[b])b=v;*argmax=b;}
}
