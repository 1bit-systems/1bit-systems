// lm_head_fused.hip — Fused lm_head: logits[v] = hs @ embed[v]^T for all vocab
// Single kernel launch replaces N_VOCAB/256 separate mm_k launches.
//
// Launch: lm_head_fused_kernel<<<VOCAB, 256, 0, stream>>>(
//            d_logits, d_hs, d_embed, HID, N_VOCAB);
//
// Each block computes logits[v] = sum_i hs[i] * embed[v*H + i]
// 256 threads, each does H/256 = 8 elements, warp reduction.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void lm_head_fused_kernel(
    __half* __restrict__ d_logits,
    const __half* __restrict__ d_hs,
    const __half* __restrict__ d_embed,
    int HID,
    int N_VOCAB)
{
    int v = blockIdx.x;
    if (v >= N_VOCAB) return;
    
    int tx = threadIdx.x;
    int lane = tx % 32;
    int warp = tx / 32;
    
    float sum = 0.0f;
    const __half* emb = d_embed + (size_t)v * HID;
    
    for (int i = tx; i < HID; i += blockDim.x) {
        sum += (float)d_hs[i] * (float)emb[i];
    }
    
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_xor_sync(0xffffffffULL, sum, o);
    
    __shared__ float s_vals[8];
    if (lane == 0)
        s_vals[warp] = sum;
    __syncthreads();
    
    if (warp == 0) {
        float val = (lane < (blockDim.x / 32)) ? s_vals[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1)
            val += __shfl_xor_sync(0xffffffffULL, val, o);
        if (tx == 0)
            d_logits[v] = __float2half(val);
    }
}
