// GPU Flash-Decoding attention for Qwen3-0.6B decode.
// Block=128 threads, each handles 1 of 128 HD elements.
// Block reduction via shared memory tree.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <float.h>

extern "C" {

__global__ void attn_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ out,
    int NH, int NKV, int HD, int seq_len, float scale) 
{
    int h = blockIdx.x;
    if (h >= NH) return;
    int tid = threadIdx.x;
    int kvh = h / (NH / NKV);
    if (tid >= HD) return; // only HD threads needed
    
    __shared__ float s_red[128]; // reduction buffer
    
    float m = -FLT_MAX, l = 0.0f, acc = 0.0f;
    int d = tid; // this thread handles HD element d
    
    for (int t = 0; t < seq_len; t++) {
        // Dot product: Q[d] * K[t, kvh, d]
        float dot = __half2float(Q[(size_t)h * HD + d]) * 
                    __half2float(K[((size_t)t * NKV + kvh) * HD + d]);
        
        // Tree reduction to sum dot across all HD threads
        s_red[tid] = dot;
        for (int s = HD / 2; s > 0; s >>= 1) {
            __syncthreads();
            if (tid < s) s_red[tid] += s_red[tid + s];
        }
        __syncthreads();
        float s = s_red[0] * scale; // full dot product
        
        // Online softmax update (all threads get same s)
        float m_new = fmaxf(s, m);
        float a = expf(m - m_new), b = expf(s - m_new);
        l = l * a + b;
        acc = acc * a + b * __half2float(V[((size_t)t * NKV + kvh) * HD + d]);
        m = m_new;
    }
    
    out[(size_t)h * HD + d] = __float2half(acc / l);
}

int rcpp_kv_cache_attn_decode(
    const void* Q_dev, const void* K_dev, const void* V_dev,
    void* out_dev, int NH, int NKV, int HD, 
    int seq_len, float scale, void* stream)
{
    if (!Q_dev || !K_dev || !V_dev || !out_dev) return 1;
    if (NH % NKV != 0 || HD > 128) return 1;
    
    int block_sz = HD; // one thread per HD element
    attn_kernel<<<NH, block_sz, HD * sizeof(float), (hipStream_t)stream>>>(
        (const __half*)Q_dev, (const __half*)K_dev, (const __half*)V_dev,
        (__half*)out_dev, NH, NKV, HD, seq_len, scale);
    
    return 0;
}

}
