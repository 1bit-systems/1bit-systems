// whisper_kernels.hip — GPU-accelerated Whisper inference kernels
// Conv1D, self-attention, cross-attention, and FFT-based mel spectrogram
// All kernels are HIP (AMD ROCm) compatible.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

// ─── FFT-based STFT kernel ─────────────────────────────────────────
// Computes magnitude spectrogram via real FFT.
// Input: audio[n_samples] float
// Output: mag[n_fft_bins × n_frames] float (column-major: [bin][frame])
// Uses a simple radix-2 FFT per frame.

#define STFT_BLOCK 256

__global__ void stft_magnitude_kernel(
    const float* __restrict__ audio,
    float* __restrict__ mag,
    const float* __restrict__ window,
    int n_samples, int n_fft, int hop, int n_frames)
{
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames) return;
    
    int n_fft_bins = n_fft / 2 + 1;
    int base = frame * hop;
    
    // Load frame into shared memory (real values)
    extern __shared__ float s_frame[];
    for (int i = threadIdx.x; i < n_fft; i += blockDim.x) {
        int idx = base + i;
        s_frame[i] = (idx < n_samples) ? audio[idx] * window[i] : 0.0f;
    }
    __syncthreads();
    
    // In-place real FFT using Cooley-Tukey radix-2
    // For simplicity, do a direct DFT for each bin this thread handles
    // (This is parallelized across bins, each thread does one bin)
    int bin = threadIdx.x; // each thread handles one frequency bin
    if (bin < n_fft_bins) {
        float re = 0.0f, im = 0.0f;
        for (int i = 0; i < n_fft; i++) {
            float angle = -2.0f * M_PI * bin * i / n_fft;
            re += s_frame[i] * __cosf(angle);
            im += s_frame[i] * __sinf(angle);
        }
        mag[(size_t)bin * n_frames + frame] = sqrtf(re * re + im * im);
    }
}

// ─── Mel filterbank application kernel ────────────────────────────
// Applies pre-computed mel filterbank weights to STFT magnitude.
// mel_w[f][bin] = weight of bin for mel band f (sparse, stored as offsets + weights)
__global__ void mel_filterbank_kernel(
    const float* __restrict__ stft_mag,
    float* __restrict__ mel_out,
    const int* __restrict__ mel_bins,    // [n_mels, 2] — (bin_lo, bin_hi) per mel
    const float* __restrict__ mel_weights, // [n_mels, 2] — weights for lo and hi
    int n_mels, int n_frames, int n_fft_bins)
{
    int fid = blockIdx.x * blockDim.x + threadIdx.x;
    int frame = blockIdx.y;
    if (fid >= n_mels || frame >= n_frames) return;
    
    int bin_lo = mel_bins[fid * 2];
    int bin_hi = mel_bins[fid * 2 + 1];
    float w_lo = mel_weights[fid * 2];
    float w_hi = mel_weights[fid * 2 + 1];
    
    float val = 0.0f;
    if (bin_lo >= 0 && bin_lo < n_fft_bins)
        val += stft_mag[(size_t)bin_lo * n_frames + frame] * (1.0f - w_lo);
    if (bin_hi >= 0 && bin_hi < n_fft_bins && bin_hi != bin_lo)
        val += stft_mag[(size_t)bin_hi * n_frames + frame] * w_hi;
    
    val = fmaxf(val, 1e-10f);
    mel_out[(size_t)fid * n_frames + frame] = log10f(val) * 10.0f;
}

// ─── Conv1D kernel ────────────────────────────────────────────────
// Implements 1D convolution: out[C_out, L_out] = in[C_in, L_in] * w[C_out, C_in, K]
// Each block handles one (C_out, L_out) pair
__global__ void conv1d_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    int C_in, int L_in, int C_out, int K, int stride)
{
    int co = blockIdx.x;
    int lo = blockIdx.y;
    int L_out = (L_in - K) / stride + 1;
    if (co >= C_out || lo >= L_out) return;
    
    float sum = bias ? bias[co] : 0.0f;
    for (int ci = 0; ci < C_in; ci++) {
        for (int ki = 0; ki < K; ki++) {
            int li = lo * stride + ki;
            sum += in[(size_t)ci * L_in + li] * w[(size_t)co * C_in * K + (size_t)ci * K + ki];
        }
    }
    out[(size_t)co * L_out + lo] = sum;
}

// ─── GELU activation kernel ───────────────────────────────────────
__global__ void gelu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float c = 0.7978845608f;
    float v = x[i];
    x[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
}

// ─── Self-attention kernel (single head through batch) ────────────
// Q, K, V are [N, D] where D = n_heads * head_dim
// Output: attn_out[N, D]
__global__ void self_attention_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int N, int D, int n_heads, float scale)
{
    int t = blockIdx.x;   // query position
    int h = blockIdx.y;   // head
    if (t >= N || h >= n_heads) return;
    
    int H = D / n_heads;
    extern __shared__ float s_scores[];
    float* scores = s_scores;  // N floats
    
    // Compute attention scores for this head
    float max_score = -1e30f;
    for (int s = 0; s < N; s++) {
        float acc = 0.0f;
        for (int d = threadIdx.x; d < H; d += blockDim.x) {
            acc += q[(size_t)t * D + (size_t)h * H + d] * k[(size_t)s * D + (size_t)h * H + d];
        }
        // Warp-level reduction
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffULL, acc, o);
        if (threadIdx.x == 0) {
            scores[s] = acc * scale;
            max_score = fmaxf(max_score, scores[s]);
        }
        __syncthreads();
    }
    
    // Softmax
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int s = 0; s < N; s++) {
            scores[s] = expf(scores[s] - max_score);
            sum += scores[s];
        }
        float inv_sum = 1.0f / sum;
        for (int s = 0; s < N; s++) scores[s] *= inv_sum;
    }
    __syncthreads();
    
    // Weighted sum of values
    for (int d = threadIdx.x; d < H; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < N; s++) {
            acc += scores[s] * v[(size_t)s * D + (size_t)h * H + d];
        }
        out[(size_t)t * D + (size_t)h * H + d] = acc;
    }
}

// ─── Cross-attention kernel ──────────────────────────────────────
// Q from decoder [N_dec, D], K/V from encoder [N_enc, D]
__global__ void cross_attention_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int N_dec, int N_enc, int D, int n_heads, float scale)
{
    int t = blockIdx.x;   // decoder position
    int h = blockIdx.y;   // head
    if (t >= N_dec || h >= n_heads) return;
    
    int H = D / n_heads;
    extern __shared__ float s_scores[];
    float* scores = s_scores;
    
    float max_score = -1e30f;
    for (int s = 0; s < N_enc; s++) {
        float acc = 0.0f;
        for (int d = threadIdx.x; d < H; d += blockDim.x) {
            acc += q[(size_t)t * D + (size_t)h * H + d] * k[(size_t)s * D + (size_t)h * H + d];
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffULL, acc, o);
        if (threadIdx.x == 0) {
            scores[s] = acc * scale;
            max_score = fmaxf(max_score, scores[s]);
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int s = 0; s < N_enc; s++) {
            scores[s] = expf(scores[s] - max_score);
            sum += scores[s];
        }
        float inv_sum = 1.0f / sum;
        for (int s = 0; s < N_enc; s++) scores[s] *= inv_sum;
    }
    __syncthreads();
    
    for (int d = threadIdx.x; d < H; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < N_enc; s++) {
            acc += scores[s] * v[(size_t)s * D + (size_t)h * H + d];
        }
        out[(size_t)t * D + (size_t)h * H + d] = acc;
    }
}

// ─── LayerNorm kernel (with bias) ────────────────────────────────
__global__ void layernorm_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    int N, int D, float eps)
{
    int t = blockIdx.x;
    if (t >= N) return;
    
    extern __shared__ float s_buf[];
    float* vals = s_buf;  // D floats
    
    // Load
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        vals[d] = x[(size_t)t * D + d];
    __syncthreads();
    
    // Mean
    float mean = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) mean += vals[d];
    for (int o = 16; o > 0; o >>= 1) mean += __shfl_xor_sync(0xffffffffULL, mean, o);
    mean /= D;
    __syncthreads();
    
    // Variance
    float var = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float diff = vals[d] - mean;
        var += diff * diff;
    }
    for (int o = 16; o > 0; o >>= 1) var += __shfl_xor_sync(0xffffffffULL, var, o);
    var /= D;
    __syncthreads();
    
    float inv_std = rsqrtf(var + eps);
    
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        out[(size_t)t * D + d] = (vals[d] - mean) * inv_std * w[d] + (b ? b[d] : 0.0f);
    }
}

// ─── Matmul (GEMM) kernel for small matrices ─────────────────────
// out[M] = in[K] @ w[M, K] (row-major weight)
__global__ void gemv_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ w,
    int M, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    
    float sum = 0.0f;
    for (int j = 0; j < K; j++) {
        sum += in[j] * w[(size_t)i * K + j];
    }
    out[i] = sum;
}
