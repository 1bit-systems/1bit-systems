// laguna_ops.hip — Fused attention: QK-norm + RoPE + score + softmax + output
// All GPU, no CPU roundtrips for the entire attention block.

#include <hip/hip_runtime.h>

// ── RMS Norm (per vector of length N) ──
// Each block handles one vector. N must be <= 1024.
__global__ void rms_norm_kernel(float* __restrict__ out, const float* __restrict__ x,
                                 const float* __restrict__ w, int N, float eps) {
    int i = threadIdx.x;
    __shared__ float s_ss;
    if (i == 0) s_ss = 0;
    __syncthreads();
    
    float val = i < N ? x[i] : 0;
    atomicAdd(&s_ss, val * val);
    __syncthreads();
    
    float r = rsqrtf(s_ss / (float)N + eps);
    if (i < N) out[i] = x[i] * r * w[i];
}

// ── Batch RMS Norm (for QK-norm: N heads × HD dims) ──
// grid.x = n_heads, block = HD (max 128)
__global__ void batch_rms_norm(float* __restrict__ data, const float* __restrict__ w,
                                int n_heads, int HD, float eps) {
    int h = blockIdx.x;
    int i = threadIdx.x;
    float* vec = data + h * HD;
    __shared__ float s_ss;
    if (i == 0) s_ss = 0;
    __syncthreads();
    float val = i < HD ? vec[i] : 0;
    atomicAdd(&s_ss, val * val);
    __syncthreads();
    float r = rsqrtf(s_ss / (float)HD + eps);
    if (i < HD) vec[i] = val * r * w[i];
}

// ── RoPE (NeoX-style) ──
__global__ void rope_kernel(float* __restrict__ q, float* __restrict__ k,
                             int pos, int n_heads, int n_kv, int HD, int rot_dim, float theta) {
    int h = blockIdx.x;
    int i = threadIdx.x;
    int half = rot_dim / 2;
    float* vec = (h < n_heads) ? (q + h * HD) : (k + (h - n_heads) * HD);
    int n_total = n_heads + n_kv;
    
    if (h < n_total && i < half) {
        float freq = 1.0f / powf(theta, (2.0f * i) / (float)rot_dim);
        float t = pos * freq;
        float c = cosf(t), s = sinf(t);
        int i0 = i, i1 = i + half;
        float v0 = vec[i0], v1 = vec[i1];
        vec[i0] = v0 * c - v1 * s;
        vec[i1] = v0 * s + v1 * c;
    }
}

// ── Fused GQA Attention ──
// Input: Q[n_heads×HD], K[n_kv×HD], V[n_kv×HD], gate[n_heads], k_cache[], v_cache[]
// Output: att[n_heads×HD]
__global__ void attention_kernel(
    const float* __restrict__ Q,    // [n_heads, HD]
    const float* __restrict__ K,    // [n_kv, HD]
    const float* __restrict__ V,    // [n_kv, HD]
    const float* __restrict__ gate, // [n_heads]
    const float* __restrict__ k_cache, // [max_pos, n_kv, HD]
    const float* __restrict__ v_cache, // [max_pos, n_kv, HD]
    float* __restrict__ att,        // [n_heads, HD]
    int n_heads, int n_kv, int HD, int GQA, int pos, float scale,
    int kv_start  // sliding window start, 0 for full attention
) {
    int h = blockIdx.x;       // query head
    int i = threadIdx.x;      // position within HD
    
    int kv_h = h / GQA;       // corresponding KV head
    int kv_end = pos + 1;
    
    // Each thread handles one element of the output
    __shared__ float scores[2048];  // max context length
    
    // Compute scores over KV positions
    if (i < (kv_end - kv_start)) {
        int t = kv_start + i;
        float* K_t = (float*)k_cache + (t * n_kv + kv_h) * HD;
        float* Qh = (float*)Q + h * HD;
        float s = 0;
        for (int d = 0; d < HD; d++) s += Qh[d] * K_t[d];
        scores[i] = s * scale;
    }
    __syncthreads();
    
    // Softmax
    int kv_len = kv_end - kv_start;
    if (i < kv_len) {
        // Find max
        float mx = scores[0];
        for (int j = 1; j < kv_len; j++) if (scores[j] > mx) mx = scores[j];
        // Exp sum
        float sum = 0;
        for (int j = 0; j < kv_len; j++) sum += expf(scores[j] - mx);
        scores[i] = expf(scores[i] - mx) / (sum + 1e-10f);
    }
    __syncthreads();
    
    // Weighted sum of V
    if (i < HD) {
        float sum = 0;
        for (int t = 0; t < kv_len; t++) {
            sum += scores[t] * v_cache[(kv_start + t) * n_kv * HD + kv_h * HD + i];
        }
        att[h * HD + i] = sum * (gate ? gate[h] : 1.0f);
    }
}

// Wrappers
extern "C" void launch_rms_norm(float* out, const float* x, const float* w, int N, float eps, void* stream) {
    hipLaunchKernelGGL(rms_norm_kernel, 1, 256, 0, (hipStream_t)stream, out, x, w, N, eps);
}

extern "C" void launch_batch_rms_norm(float* data, const float* w, int n_heads, int HD, float eps, void* stream) {
    hipLaunchKernelGGL(batch_rms_norm, n_heads, HD > 256 ? 256 : HD, 0, (hipStream_t)stream, data, w, n_heads, HD, eps);
}

extern "C" void launch_rope(float* q, float* k, int pos, int nh, int nkv, int hd, int rd, float theta, void* stream) {
    hipLaunchKernelGGL(rope_kernel, nh + nkv, rd/2, 0, (hipStream_t)stream, q, k, pos, nh, nkv, hd, rd, theta);
}

extern "C" void launch_attention(
    const float* Q, const float* K, const float* V, const float* gate,
    const float* k_cache, const float* v_cache,
    float* att, int nh, int nkv, int HD, int GQA, int pos, float scale, int kv_start, void* stream
) {
    hipLaunchKernelGGL(attention_kernel, nh, HD > 256 ? 256 : HD, 0, (hipStream_t)stream,
        Q, K, V, gate, k_cache, v_cache, att, nh, nkv, HD, GQA, pos, scale, kv_start);
}
