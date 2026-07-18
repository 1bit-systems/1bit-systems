// kernels/oscar_quant.hip — OSCAR INT2 quantization + attention decode kernels
//
// Pipeline (per KV row):
//   1. Load fp16 vector
//   2. Apply precomputed OSCAR rotation R (HD×HD matrix from calibration)
//   3. INT2 quantize with per-token symmetric clipping
//   4. Pack 4 values per byte + store fp16 scale
//
// Attention decode:
//   1. Load INT2 packed indices + fp16 scale
//   2. Dequantize to fp16
//   3. Apply inverse rotation R^T
//   4. Compute attention with Q
//
// HD must be multiple of 4 (pack 4×2bit = 1 byte). HD=128 OK.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cmath>
#include <cfloat>

#include "rocm_cpp/oscar.h"

#define HIP_CHECK(e) do { hipError_t _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP Error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_s)); abort(); } } while(0)

namespace {

constexpr int WARP         = 32;
constexpr int BLOCK        = 128;    // one warp per (pos, head), HD <= 128
constexpr int MAX_HD       = 256;
constexpr int TILE         = 128;    // flash-decoding KV tile

// ── INT2 quantize helpers ──────────────────────────────────────────────────
// Symmetric: find max |v|, scale to [-2,2], quantize to {-3,-1,1,3}/scale
// Actually simpler: 2-bit values {0,1,2,3} -> dequant {-max, -max/3, max/3, max}
// But the paper uses symmetric: decode as s * {0, 1, 2, 3} - s*1.5 → {-1.5s, -0.5s, 0.5s, 1.5s}

__device__ inline float int2_dequant(uint32_t code, float scale) {
    // 2-bit code 0..3 → symmetric values: -3/3, -1/3, 1/3, 3/3 = -1.5, -0.5, 0.5, 1.5
    // But scale is max|v|, so range is scale * {-1.5, -0.5, 0.5, 1.5}/1.5 = scale * {-1, -1/3, 1/3, 1}
    // Simplest: scale = max_abs, 4 levels: {-scale, -scale/3, scale/3, scale}
    static const float LUT[4] = {-1.0f, -0.3333333f, 0.3333333f, 1.0f};
    return LUT[code & 3] * scale;
}

__device__ inline uint32_t quantize_to_int2(float v, float max_abs, float* scale_out) {
    float abs_v = fabsf(v);
    // Use max_abs as the scale. Clip to max_abs if larger.
    float s = max_abs > 1e-10f ? max_abs : 1e-10f;
    float ratio = v / s; // now in [-1, 1]
    // Map to 4 levels: -1 → 0, -1/3 → 1, 1/3 → 2, 1 → 3
    uint32_t code;
    if (ratio < -0.6666667f) code = 0;
    else if (ratio < 0.0f)    code = 1;
    else if (ratio < 0.6666667f) code = 2;
    else                       code = 3;
    *scale_out = s;
    return code;
}

// ── Apply rotation R (HD×HD row-major) to vector x (HD) → y (HD) ──
__device__ inline void apply_rotation(const float* __restrict__ R,
                                       const float* __restrict__ x,
                                       float* y, int HD) {
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        double s = 0.0;
        for (int j = 0; j < HD; j++)
            s += (double)R[i * HD + j] * x[j];
        y[i] = (float)s;
    }
}

// ── Apply R^T (transpose = inverse for orthogonal) ──
__device__ inline void apply_rotation_T(const float* __restrict__ R,
                                          const float* __restrict__ x,
                                          float* y, int HD) {
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        double s = 0.0;
        for (int j = 0; j < HD; j++)
            s += (double)R[j * HD + i] * x[j];
        y[i] = (float)s;
    }
}

// ── INT2 quantize kernel ──────────────────────────────────────────────────
// grid = (seq_len, num_kv_heads), block = (WARP, 1, 1)
__global__ __launch_bounds__(WARP)
void oscar_quantize_kernel(
    const __half* __restrict__ in_fp16,   // [seq_len, num_kv_heads, HD]
    uint8_t*      __restrict__ out_idx,    // [seq_len, num_kv_heads, HD/4]
    __half*       __restrict__ out_scale,  // [seq_len, num_kv_heads]
    int seq_len, int num_kv_heads, int HD, int layer_idx,
    const float* __restrict__ R_layer)     // [HD*HD] rotation matrix for this layer
{
    int pos     = blockIdx.x;
    int kv_head = blockIdx.y;
    if (pos >= seq_len || kv_head >= num_kv_heads) return;
    int lane = threadIdx.x;

    extern __shared__ float s_buf[];
    float* s_row  = s_buf;               // [HD] post-rotation
    float* s_rot  = s_buf + MAX_HD;      // [HD] pre-rotation

    // Load fp16 → fp32
    size_t base = ((size_t)pos * num_kv_heads + kv_head) * HD;
    for (int i = lane; i < HD; i += WARP)
        s_rot[i] = (float)in_fp16[base + i];
    __syncthreads();

    // Apply rotation R
    apply_rotation(R_layer, s_rot, s_row, HD);
    __syncthreads();

    // Find max absolute value
    float max_abs = 0.0f;
    for (int i = lane; i < HD; i += WARP)
        max_abs = fmaxf(max_abs, fabsf(s_row[i]));
    // Warp reduce
    for (int off = WARP/2; off > 0; off /= 2)
        max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
    
    // Quantize to INT2 (4 values per byte)
    for (int i = lane; i < HD/4; i += WARP) {
        float s;
        uint32_t c0 = quantize_to_int2(s_row[i*4+0], max_abs, &s);
        uint32_t c1 = quantize_to_int2(s_row[i*4+1], max_abs, &s);
        uint32_t c2 = quantize_to_int2(s_row[i*4+2], max_abs, &s);
        uint32_t c3 = quantize_to_int2(s_row[i*4+3], max_abs, &s);
        out_idx[base/4 + i] = (uint8_t)(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6));
    }

    // Store scale (lane 0)
    if (lane == 0)
        out_scale[pos * num_kv_heads + kv_head] = (__half)max_abs;
}

// ── INT2 dequant + inverse rotation + attention decode ─────────────────────
// Flash-decoding style: block per (q_head, kv_tile)
__global__ __launch_bounds__(BLOCK)
void oscar_attn_decode_kernel(
    const __half* __restrict__ Q_dev,         // [num_q_heads, HD]
    const uint8_t* __restrict__ K_idx,        // [seq_len, num_kv_heads, HD/4]
    const __half* __restrict__ K_scale,       // [seq_len, num_kv_heads]
    const uint8_t* __restrict__ V_idx,        // [seq_len, num_kv_heads, HD/4]
    const __half* __restrict__ V_scale,       // [seq_len, num_kv_heads]
    void*          __restrict__ out_dev,       // [num_q_heads, HD] fp16
    int num_q_heads, int num_kv_heads, int HD, int seq_len,
    int layer_idx, float attn_scale,
    const float* __restrict__ R_K,            // [HD*HD] rotation for this layer
    const float* __restrict__ R_V)            // [HD*HD] rotation for this layer
{
    __shared__ float s_q[MAX_HD];
    __shared__ float s_k[MAX_HD];
    __shared__ float s_v[MAX_HD];
    __shared__ float s_k_rot[MAX_HD];
    __shared__ float s_v_rot[MAX_HD];
    __shared__ float s_m[2], s_l[2]; // online-softmax

    int q_head = blockIdx.x;
    int tile   = blockIdx.y;
    if (q_head >= num_q_heads) return;

    // Load Q vector
    int lane = threadIdx.x;
    for (int i = lane; i < HD; i += BLOCK)
        s_q[i] = (float)Q_dev[q_head * HD + i];
    __syncthreads();

    int kvh = q_head % num_kv_heads; // simple GQA mapping
    int t_start = tile * TILE;
    int t_end   = min(t_start + TILE, seq_len);

    float m = -FLT_MAX, l = 0.0f;
    for (int t = t_start + lane; t < t_end; t += BLOCK) {
        // Load INT2 K[ t, kvh ] → fp32 → dequant → inverse rotate
        float max_abs_k = (float)K_scale[t * num_kv_heads + kvh];
        for (int i = 0; i < HD/4; i++) {
            uint8_t packed = K_idx[((size_t)t * num_kv_heads + kvh) * HD/4 + i];
            s_k[i*4+0] = int2_dequant(packed, max_abs_k);
            s_k[i*4+1] = int2_dequant(packed >> 2, max_abs_k);
            s_k[i*4+2] = int2_dequant(packed >> 4, max_abs_k);
            s_k[i*4+3] = int2_dequant(packed >> 6, max_abs_k);
        }
        __syncthreads();
        apply_rotation_T(R_K, s_k, s_k_rot, HD);
        __syncthreads();

        // Score: q · k
        double score = 0.0;
        for (int i = lane; i < HD; i += BLOCK)
            score += (double)s_q[i] * s_k_rot[i];
        // Warp reduce score
        for (int off = WARP/2; off > 0; off /= 2)
            score += __shfl_xor((float)score, off);
        score *= attn_scale;

        if (lane == 0) {
            float new_m = fmaxf(m, (float)score);
            float alpha = __expf(m - new_m);
            float beta  = __expf((float)score - new_m);
            l = l * alpha + beta;
            m = new_m;
            s_m[0] = m; s_l[0] = l;
        }
        // Broadcast
        if (lane < 2) { s_m[lane] = s_m[0]; s_l[lane] = s_l[0]; }
        __syncthreads();
        m = s_m[0]; l = s_l[0];

        // Accumulate V
        float max_abs_v = (float)V_scale[t * num_kv_heads + kvh];
        for (int i = 0; i < HD/4; i++) {
            uint8_t packed = V_idx[((size_t)t * num_kv_heads + kvh) * HD/4 + i];
            s_v[i*4+0] = int2_dequant(packed, max_abs_v);
            s_v[i*4+1] = int2_dequant(packed >> 2, max_abs_v);
            s_v[i*4+2] = int2_dequant(packed >> 4, max_abs_v);
            s_v[i*4+3] = int2_dequant(packed >> 6, max_abs_v);
        }
        __syncthreads();
        apply_rotation_T(R_V, s_v, s_v_rot, HD);
        __syncthreads();

        // Online softmax: o += exp(score - m) * v / l
        float p = __expf((float)score - m);
        for (int i = lane; i < HD; i += BLOCK)
            s_v_rot[i] = p * s_v_rot[i] / l;
        __syncthreads();
    }

    // Write output
    for (int i = lane; i < HD; i += BLOCK)
        ((__half*)out_dev)[q_head * HD + i] = (__half)s_v_rot[i];
}

} // anonymous namespace

// ── Host-side launcher: quantize K ─────────────────────────────────────────
extern "C" void rcpp_oscar_quantize_launch(
    const void* in_fp16, void* out_idx, void* out_scale,
    int seq_len, int num_kv_heads, int HD, int layer_idx,
    const float* R_host,  // CPU-side rotation matrix [HD*HD]
    void* stream)
{
    if (!in_fp16 || !out_idx || !out_scale || !R_host) return;
    if (HD > MAX_HD) return;
    if ((HD & 3) != 0) return; // must be multiple of 4

    // Upload rotation matrix to GPU constant memory (statically allocated)
    // For now, pass as kernel argument (fits in constant cache for HD<=128)
    float* R_dev = nullptr;
    HIP_CHECK(hipMalloc(&R_dev, HD * HD * sizeof(float)));
    HIP_CHECK(hipMemcpy(R_dev, R_host, HD * HD * sizeof(float), hipMemcpyHostToDevice));

    dim3 grid(seq_len, num_kv_heads);
    int smem = 2 * MAX_HD * sizeof(float); // s_row + s_rot
    hipLaunchKernelGGL(oscar_quantize_kernel, grid, WARP, smem, (hipStream_t)stream,
                       (const __half*)in_fp16, (uint8_t*)out_idx, (__half*)out_scale,
                       seq_len, num_kv_heads, HD, layer_idx, R_dev);

    HIP_CHECK(hipStreamSynchronize((hipStream_t)stream));
    HIP_CHECK(hipFree(R_dev));
}

// ── Host-side launcher: attention decode ────────────────────────────────────
extern "C" int rcpp_oscar_attn_decode_launch(
    const void* Q_dev, const void* K_idx, const void* K_scale,
    const void* V_idx, const void* V_scale,
    void* out_dev,
    int num_q_heads, int num_kv_heads, int HD, int seq_len,
    int layer_idx, float scale,
    const float* R_K_host, const float* R_V_host,
    void* stream)
{
    if (!Q_dev || !K_idx || !V_idx || !out_dev) return -1;
    if (HD > MAX_HD) return -1;

    float *R_K_dev, *R_V_dev;
    HIP_CHECK(hipMalloc(&R_K_dev, HD * HD * sizeof(float)));
    HIP_CHECK(hipMalloc(&R_V_dev, HD * HD * sizeof(float)));
    HIP_CHECK(hipMemcpy(R_K_dev, R_K_host, HD * HD * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(R_V_dev, R_V_host, HD * HD * sizeof(float), hipMemcpyHostToDevice));

    int num_tiles = (seq_len + TILE - 1) / TILE;
    dim3 grid(num_q_heads, num_tiles);
    hipLaunchKernelGGL(oscar_attn_decode_kernel, grid, BLOCK, 0, (hipStream_t)stream,
                       (const __half*)Q_dev,
                       (const uint8_t*)K_idx, (const __half*)K_scale,
                       (const uint8_t*)V_idx, (const __half*)V_scale,
                       out_dev,
                       num_q_heads, num_kv_heads, HD, seq_len,
                       layer_idx, scale,
                       R_K_dev, R_V_dev);

    HIP_CHECK(hipStreamSynchronize((hipStream_t)stream));
    HIP_CHECK(hipFree(R_K_dev));
    HIP_CHECK(hipFree(R_V_dev));
    return 0;
}
