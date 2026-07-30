// hip_1bp_kernels.hip — GPU kernels for 1BP inference.
// All ops stay on-device. Custom GEMV kernel replaces rocBLAS for small M.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cfloat>
#include <cstdint>

// ── Custom GEMV: y[M] = W[M,N] @ x[N], row-major W ──
// One thread per row, block-wide reduction over columns.
__global__ void gemv_kernel(float* y, const float* W, const float* x, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    double sum = 0;
    for (int k = threadIdx.x; k < N; k += blockDim.x)
        sum += (double)x[k] * W[(size_t)row * N + k];
    __shared__ double sdata[256];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = (float)sdata[0];
}

// ── RMSNorm (GPU cooperative reduction) ──
static constexpr int RMS_BLOCK = 256;
__launch_bounds__(RMS_BLOCK)
__global__ void rmsnorm_kernel(float* __restrict__ x, const float* __restrict__ w,
                                int N, float eps) {
    __shared__ float ssum[32];
    int tid = threadIdx.x;
    int wid = tid / warpSize;
    int lane = tid & (warpSize - 1);

    float local = 0.0f;
    for (int i = tid; i < N; i += RMS_BLOCK)
        local += x[i] * x[i];

    for (int offset = warpSize/2; offset > 0; offset /= 2)
        local += __shfl_down(local, offset);

    if (lane == 0) { ssum[wid] = local; }
    __syncthreads();

    if (wid == 0) {
        local = (tid < (RMS_BLOCK + warpSize - 1) / warpSize) ? ssum[lane] : 0.0f;
        for (int offset = warpSize/2; offset > 0; offset /= 2)
            local += __shfl_down(local, offset);
        if (tid == 0) ssum[0] = rsqrtf(local / N + eps);
    }
    __syncthreads();

    float inv_rms = ssum[0];
    if (w) {
        for (int i = tid; i < N; i += RMS_BLOCK)
            x[i] = x[i] * inv_rms * w[i];
    } else {
        for (int i = tid; i < N; i += RMS_BLOCK)
            x[i] = x[i] * inv_rms;
    }
}

// ── RoPE ──
__global__ void rope_kernel(float* __restrict__ x, int head_dim, int pos,
                             float theta_base, int num_heads) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (h >= num_heads || d >= head_dim/2) return;
    int hd2 = head_dim / 2;
    float freq = 1.0f / powf(theta_base, (float)d / (float)hd2);
    float t = (float)pos * freq;
    float c = cosf(t), s = sinf(t);
    int idx = h * head_dim + d;
    float a = x[idx], b = x[idx + hd2];
    x[idx]       = a * c - b * s;
    x[idx + hd2] = a * s + b * c;
}

// ── SiLU ──
__global__ void silu_kernel(float* __restrict__ out,
                             const float* __restrict__ gate,
                             const float* __restrict__ up, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
}

// ── Element-wise add ──
__global__ void add_kernel(float* __restrict__ x, const float* __restrict__ y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    x[i] += y[i];
}

// ── Float→Half conversion (device-to-device) ──
__global__ void f2h_kernel(__half* dst, const float* src, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N) dst[i]=__float2half(src[i]);
}

// ── Half→Float conversion (device-to-device) ──
__global__ void h2f_kernel(float* dst, const __half* src, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N) dst[i]=__half2float(src[i]);
}

// ── Fused K,V store: convert f32→f16 and store in KV cache ──
__global__ void kv_store_kernel(__half* __restrict__ dK, __half* __restrict__ dV,
                                 const float* __restrict__ k, const float* __restrict__ v,
                                 int pos, int NKV, int HD, int max_seq) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (h >= NKV || d >= HD) return;
    size_t off = (size_t)pos * NKV * HD + (size_t)h * HD + d;
    dK[off] = __float2half(k[(size_t)h * HD + d]);
    dV[off] = __float2half(v[(size_t)h * HD + d]);
}

// ── Copy embedding row to output ──
__global__ void embed_copy_kernel(float* __restrict__ dst,
                                   const float* __restrict__ embed,
                                   int token_id, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H) dst[i] = embed[(size_t)token_id * H + i];
}
