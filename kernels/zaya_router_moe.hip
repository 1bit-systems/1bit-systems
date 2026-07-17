// Zaya EDA Router + MoE Expert — fused GPU kernel
// Eliminates CPU sync by running router on GPU.
// One kernel launch per layer: router → expert selection → MoE FFN

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

#define H    2048
#define QD   1024
#define KD   256
#define QKV  1280
#define NKV  2
#define NQ   8
#define HD   128
#define GQA  4
#define N_EXP    16
#define N_EXPERT_SLOTS  17
#define N_FF  2048
#define RTR_H 256
#define BLK   256

// Warp reductions
__device__ inline float warp_sum(float v){for(int o=16;o>0;o>>=1)v+=__shfl_xor(v,o);return v;}
__device__ inline float block_sum(float v,float*sh){
    int tx=threadIdx.x,w=tx/32,l=tx%32;
    v=warp_sum(v);if(l==0)sh[w]=v;
    __syncthreads();float s=0;
    if(w==0){s=(l<8)?sh[l]:0;s=warp_sum(s);}if(w==0&&l==0)sh[0]=s;
    __syncthreads();return sh[0];
}

// tanh-based GELU
__device__ inline float gelu_tanh(float x){
    float t=tanhf(0.79788456f*(x+0.044715f*x*x*x));
    return 0.5f*x*(1.0f+t);
}

// ─── EDA Router Kernel ───
// Input:  hs[H] — post-attention RMSNorm'd hidden states (fp16)
//         prev_rs[RTR_H] — previous router states (or zeros)
//         has_eda, eda_scale
// Output: moe_out[H] — expert FFN output
//         next_rs[RTR_H] — next router states
//         expert_idx, expert_wt
__global__ void eda_router_moe_kernel(
    const __half*hs,           // [H] post-attn RMSNorm'd input
    const float*prev_rs,       // [RTR_H] prev router states
    int has_eda,
    float eda_scale,
    // Router weights
    const float*gdw,           // [RTR_H, H]
    const float*gdb,           // [RTR_H]
    const float*rfn,           // [RTR_H] RMSNorm weight
    const float*rf1,           // [RTR_H, RTR_H] fc1
    const float*rf1b,          // [RTR_H]
    const float*rf2,           // [RTR_H, RTR_H] fc2
    const float*rf2b,          // [RTR_H]
    const float*rout,          // [N_EXPERT_SLOTS, RTR_H]
    const float*bb,            // [N_EXPERT_SLOTS] balancing biases
    // MoE weights (fp16)
    const __half*gu,           // [N_EXP, 2*N_FF, H]
    const __half*dn,           // [N_EXP, H, N_FF]
    // Outputs
    float*next_rs,             // [RTR_H]
    __half*moe_out,            // [H] expert output
    int*expert_idx,            // scalar
    float*expert_wt            // scalar
) {
    int tx = threadIdx.x;
    __shared__ float sh[32];
    __shared__ float rs[RTR_H];
    __shared__ float tmp[RTR_H];
    __shared__ float scores[N_EXPERT_SLOTS];
    __shared__ float probs[N_EXPERT_SLOTS];

    // ── 1. gate_down: rs[RTR_H] = hs[H] @ gdw^T + gdb ──
    for(int i=tx; i<RTR_H; i+=blockDim.x){
        float s = gdb[i];
        for(int j=0;j<H;j++) s+=__half2float(hs[j])*gdw[i*(size_t)H+j];
        rs[i] = s;
    }
    __syncthreads();

    // ── 2. EDA ──
    if(has_eda && prev_rs){
        for(int i=tx; i<RTR_H; i+=blockDim.x) rs[i] += prev_rs[i] * eda_scale;
    }
    __syncthreads();

    // Save next router states
    for(int i=tx; i<RTR_H; i+=blockDim.x) next_rs[i] = rs[i];
    __syncthreads();

    // ── 3. RMSNorm ──
    float ss = 0;
    for(int i=tx; i<RTR_H; i+=blockDim.x) ss += rs[i] * rs[i];
    float rms_r = 1.0f / sqrtf(block_sum(ss,sh)/(float)RTR_H + 1e-5f);
    for(int i=tx; i<RTR_H; i+=blockDim.x) rs[i] = rs[i] * rms_r * rfn[i];
    __syncthreads();

    // ── 4. fc1 + GELU ──
    for(int i=tx; i<RTR_H; i+=blockDim.x){
        float s = rf1b[i];
        for(int j=0;j<RTR_H;j++) s += rs[j]*rf1[i*(size_t)RTR_H+j];
        tmp[i] = gelu_tanh(s);
    }
    __syncthreads();

    // ── 5. fc2 + GELU ──
    for(int i=tx; i<RTR_H; i+=blockDim.x){
        float s = rf2b[i];
        for(int j=0;j<RTR_H;j++) s += tmp[j]*rf2[i*(size_t)RTR_H+j];
        rs[i] = gelu_tanh(s);
    }
    __syncthreads();

    // ── 6. out_proj → scores + balancing_biases ──
    for(int i=tx; i<N_EXPERT_SLOTS; i+=blockDim.x){
        float s = 0;
        for(int j=0;j<RTR_H;j++) s += rs[j]*rout[i*(size_t)RTR_H+j];
        scores[i] = s + bb[i];
    }
    __syncthreads();

    // ── 7. softmax ──
    float maxv = scores[0];
    for(int i=1;i<N_EXPERT_SLOTS;i++) if(scores[i]>maxv) maxv=scores[i];
    float sumv = 0;
    for(int i=tx; i<N_EXPERT_SLOTS; i+=blockDim.x){
        probs[i] = expf(scores[i]-maxv);
        sumv += probs[i];
    }
    sumv = block_sum(sumv, sh);
    float inv_sum = 1.0f/(sumv+1e-10f);
    for(int i=tx; i<N_EXPERT_SLOTS; i+=blockDim.x) probs[i] *= inv_sum;
    __syncthreads();

    // ── 8. top-1 ──
    // Use shared memory for parallel reduction
    if(tx == 0){
        int best = 0;
        float bestv = probs[0];
        for(int i=1;i<N_EXPERT_SLOTS;i++){
            if(probs[i] > bestv){ bestv = probs[i]; best = i; }
        }
        *expert_idx = best;
        *expert_wt = bestv;
    }

    // ── 9. MoE Expert FFN ──
    // Need expert_idx from step 8. Since only tx==0 knows it, broadcast.
    __syncthreads();
    int expert = (tx == 0) ? *expert_idx : 0;
    // Broadcast expert to all threads
    if(blockDim.x > 1){
        // Use shared memory for broadcast
        __shared__ int expert_sh;
        if(tx == 0) expert_sh = *expert_idx;
        __syncthreads();
        expert = expert_sh;
    }

    if(expert < N_EXP){  // Real expert (not MOD skip)
        // gate_up: tmp2[2*N_FF] = hs[H] @ gu[expert, 2*N_FF, H]^T
        __shared__ __half gate_up[2*N_FF];
        for(int i=tx; i<2*N_FF; i+=blockDim.x){
            float s = 0;
            const __half*w = gu + ((size_t)expert*2*N_FF + i)*(size_t)H;
            for(int j=0;j<H;j++) s += __half2float(hs[j]) * __half2float(w[j]);
            gate_up[i] = __float2half(s);
        }
        __syncthreads();

        // SiLU(gate) * up → moe_out[N_FF]
        __shared__ __half moe_buf[N_FF];
        for(int i=tx; i<N_FF; i+=blockDim.x){
            float g = __half2float(gate_up[i]);
            float u = __half2float(gate_up[N_FF+i]);
            moe_buf[i] = __float2half((g/(1.0f+expf(-g))) * u);
        }
        __syncthreads();

        // down: moe_out[H] = moe_buf[N_FF] @ dn[expert, H, N_FF]^T
        for(int i=tx; i<H; i+=blockDim.x){
            float s = 0;
            const __half*w = dn + ((size_t)expert*H + i)*(size_t)N_FF;
            for(int j=0;j<N_FF;j++) s += __half2float(moe_buf[j]) * __half2float(w[j]);
            moe_out[i] = __float2half(s);
        }
    } else {
        // MOD skip: output = hs (passthrough)
        for(int i=tx; i<H; i+=blockDim.x) moe_out[i] = hs[i];
    }
}
