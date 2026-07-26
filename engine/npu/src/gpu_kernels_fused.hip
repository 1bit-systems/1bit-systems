// GPU kernels for fused engine — compiled by hipcc. 
// Includes: tiled GEMV, LM head, RMS norm, add, SiLU, RoPE, copy.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Tiled f16 GEMV: C[N] = A[K] · W_col[N,K] (column-major weight)
// 128 threads/block, each thread handles 2 output columns
__global__ void gemv_kernel_tiled(
    const __half *__restrict__ A, const __half *__restrict__ W,
    __half *__restrict__ C, int K, int N, int BK, int BN)
{
    __shared__ __half s_A[1024];
    int n0 = blockIdx.x * BN;
    int tid = threadIdx.x;
    int n = n0 + tid, n1 = n + blockDim.x;
    float s0 = 0, s1 = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        int kt = min(BK, K - k0);
        for (int i = tid; i < kt; i += blockDim.x) s_A[i] = A[k0 + i];
        __syncthreads();
        for (int k = 0; k < kt; k++) {
            float a = __half2float(s_A[k]);
            if (n < N)  s0 += a * __half2float(W[(size_t)n * K + (k0 + k)]);
            if (n1 < N) s1 += a * __half2float(W[(size_t)n1 * K + (k0 + k)]);
        }
        __syncthreads();
    }
    if (n < N)  C[n] = __float2half(s0);
    if (n1 < N) C[n1] = __float2half(s1);
}

// Simple GEMV: one thread per output column
__global__ void gemv_k(const __half *__restrict__ A, const __half *__restrict__ W,
                       __half *__restrict__ C, int K, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float s = 0;
    for (int k = 0; k < K; k++)
        s += __half2float(A[k]) * __half2float(W[(size_t)n * K + k]);
    C[n] = __float2half(s);
}

// LM head: f16[H] × f16[V,H] → f32[V]
__global__ void lmhead_k(const __half *__restrict__ h, const __half *__restrict__ W,
                         float *__restrict__ L, int H_, int V_) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V_) return;
    float s = 0;
    for (int i = 0; i < H_; i++)
        s += __half2float(h[i]) * __half2float(W[(size_t)v * H_ + i]);
    L[v] = s;
}

// RMS Norm (in-place)
__global__ void rms_norm_k(__half *__restrict__ x, const __half *__restrict__ w,
                           int n, float eps) {
    __shared__ float ss_buf[256];
    __shared__ float s_ir;
    float l = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float v = __half2float(x[j]); l += v * v;
    }
    ss_buf[threadIdx.x] = l;
    __syncthreads();
    if (threadIdx.x == 0) {
        double ss = 0; for (int j = 0; j < blockDim.x; j++) ss += ss_buf[j];
        ss /= n; s_ir = 1.0f / sqrtf((float)ss + eps);
    }
    __syncthreads();
    // All threads participate to avoid early-return race on AMD
    float ir = s_ir;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        x[j] = __float2half(__half2float(x[j]) * ir * __half2float(w[j]));
}

// Element-wise add (in-place)
__global__ void add_k(__half *__restrict__ a, const __half *__restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

// SiLU gate: o = silu(g) * u
__global__ void silu_mul_k(const __half *__restrict__ g, const __half *__restrict__ u,
                          __half *__restrict__ o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float gv = __half2float(g[i]);
        o[i] = __float2half((gv / (1.0f + expf(-gv))) * __half2float(u[i]));
    }
}

// RoPE (in-place)
__global__ void rope_k(__half *__restrict__ qk, const __half *__restrict__ cos,
                       const __half *__restrict__ sin, int pos, int hd, int nh) {
    int h = blockIdx.x, d = threadIdx.x;
    if (h >= nh || d >= hd / 2) return;
    int i = h * hd + d, j = h * hd + d + hd / 2;
    float a = __half2float(qk[i]), b = __half2float(qk[j]);
    float c = __half2float(cos[pos * hd + d]), s = __half2float(sin[pos * hd + d]);
    qk[i] = __float2half(a * c - b * s);
    qk[j] = __float2half(b * c + a * s);
}

// Copy
__global__ void copy_k(const __half *__restrict__ src, __half *__restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}


// ── Wrapper functions (extern "C" for linking with g++ compiled code) ──
extern "C" {

// Launch tiled GEMV: C[N] = A[K] · W_col[N,K]
// Returns after kernel launch (async on stream)
void launch_gemv_tiled(const __half* A, const __half* W, __half* C,
                       int K, int N, int BK, int BN, hipStream_t stream) {
    int grid = (N + BN - 1) / BN;
    size_t shmem = BK * sizeof(__half);
    gemv_kernel_tiled<<<grid, 128, shmem, stream>>>(A, W, C, K, N, BK, BN);
}

// Launch LM head: f32 logits = hidden @ W^T
void launch_lmhead(const __half* hidden, const __half* W, float* logits,
                   int H, int NV, int BK, int BN, hipStream_t stream) {
    int grid = (NV + BN - 1) / BN;
    size_t shmem = BK * sizeof(__half);
    lmhead_k<<<grid, 128, shmem, stream>>>(hidden, W, logits, H, NV);
}

// Launch simple GEMV: one thread per output column
void launch_gemv_simple(const __half* A, const __half* W, __half* C,
                        int K, int N, hipStream_t stream) {
    int grid = (N + 255) / 256;
    gemv_k<<<grid, 256, 0, stream>>>(A, W, C, K, N);
}

} // extern "C"

// ── Additional wrappers for spec engine ──
extern "C" {

void launch_rmsnorm(__half* x, const __half* w, int n, float eps, hipStream_t stream) {
    rms_norm_k<<<1, 256, n*2, stream>>>(x, w, n, eps);
}

void launch_add(__half* x, const __half* y, int n, hipStream_t stream) {
    int g = (n + 255) / 256;
    add_k<<<g, 256, 0, stream>>>(x, y, n);
}

void launch_copy(const __half* src, __half* dst, int n, hipStream_t stream) {
    int g = (n + 255) / 256;
    copy_k<<<g, 256, 0, stream>>>(src, dst, n);
}

void launch_silu_mul(const __half* g, const __half* u, __half* o, int n, hipStream_t stream) {
    int g2 = (n + 255) / 256;
    silu_mul_k<<<g2, 256, 0, stream>>>(g, u, o, n);
}

void launch_rope(__half* qk, const __half* cos, const __half* sin, int pos, int hd, int nh, hipStream_t stream) {
    rope_k<<<nh, hd/2, 0, stream>>>(qk, cos, sin, pos, hd, nh);
}

} // extern "C"

// ── f32 kernels for hidden state accumulation (avoids f16 overflow) ──
extern "C" {

// RMS norm on f32: x[i] = x[i] * inv_rms * w[i]
__global__ void rms_norm_f32_k(float *__restrict__ x, const float *__restrict__ w,
                               int n, float eps) {
    __shared__ float ss_buf[256];
    float l = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x) l += x[j] * x[j];
    ss_buf[threadIdx.x] = l;
    __syncthreads();
    if (threadIdx.x == 0) {
        double ss = 0;
        for (int j = 0; j < blockDim.x; j++) ss += ss_buf[j];
        ss /= n;
        float ir = 1.0f / sqrtf((float)ss + eps);
        for (int j = 0; j < n; j++) x[j] = x[j] * ir * w[j];
    }
}

// f32 add: x[i] += y[i]
__global__ void add_f32_k(float *__restrict__ x, const float *__restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// f32 copy: dst[i] = src[i]
__global__ void copy_f32_k(const float *__restrict__ src, float *__restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// f32→f16 conversion: dst[i] = __float2half(src[i]), clamped to f16 range
__global__ void f32_to_f16_k(const float *__restrict__ src, __half *__restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = src[i];
    // Clamp to f16 range to prevent overflow
    if (v > 65000.0f) v = 65000.0f;
    if (v < -65000.0f) v = -65000.0f;
    dst[i] = __float2half(v);
}

// f16→f32 conversion: dst[i] = __half2float(src[i])
__global__ void f16_to_f32_k(const __half *__restrict__ src, float *__restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __half2float(src[i]);
}

// Wrapper functions
void launch_rmsnorm_f32(float* x, const float* w, int n, float eps, hipStream_t stream) {
    rms_norm_f32_k<<<1, 256, 0, stream>>>(x, w, n, eps);
}

void launch_add_f32(float* x, const float* y, int n, hipStream_t stream) {
    int g = (n + 255) / 256;
    add_f32_k<<<g, 256, 0, stream>>>(x, y, n);
}

void launch_copy_f32(const float* src, float* dst, int n, hipStream_t stream) {
    int g = (n + 255) / 256;
    copy_f32_k<<<g, 256, 0, stream>>>(src, dst, n);
}

void launch_f32_to_f16(const float* src, __half* dst, int n, hipStream_t stream) {
    int g = (n + 255) / 256;
    f32_to_f16_k<<<g, 256, 0, stream>>>(src, dst, n);
}

void launch_f16_to_f32(const __half* src, float* dst, int n, hipStream_t stream) {
    int g = (n + 255) / 256;
    f16_to_f32_k<<<g, 256, 0, stream>>>(src, dst, n);
}

} // extern "C"
