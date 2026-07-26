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

#include "hip_check.h"

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
        max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffffULL, max_abs, off));
    
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
// Flash-decoding style: block per (q_head, kv_tile).
// All threads cooperatively process each KV position to avoid data races.
// Uses proper online softmax with accumulation (not overwrite).
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
    __shared__ float s_o[MAX_HD];     // accumulated output (online softmax)
    __shared__ float s_m, s_l;        // online-softmax running state (block-wide, not per-thread)
    __shared__ float s_m_old;         // previous m for rescaling
    __shared__ float s_warp_score[BLOCK / WARP]; // per-warp partial dot products, reduced below

    int q_head = blockIdx.x;
    int tile   = blockIdx.y;
    if (q_head >= num_q_heads) return;

    int tid = threadIdx.x;
    int warp_id = tid / WARP;
    int lane_id = tid & 31;

    // Load Q vector (cooperative)
    for (int i = tid; i < HD; i += BLOCK)
        s_q[i] = (float)Q_dev[q_head * HD + i];

    // Initialize output accumulator
    for (int i = tid; i < HD; i += BLOCK)
        s_o[i] = 0.0f;

    int kvh = q_head % num_kv_heads; // simple GQA mapping
    int t_start = tile * TILE;
    int t_end   = min(t_start + TILE, seq_len);

    // m/l used to live as per-thread locals, only ever updated by tid==0 —
    // every other thread read back its own perpetually-unchanged -FLT_MAX/0,
    // silently corrupting p_over_l (and thus the whole output) with inf/nan
    // for every thread's assigned output indices except tid==0's. Both must
    // be the single block-wide shared value.
    if (tid == 0) { s_m = -FLT_MAX; s_l = 0.0f; }
    __syncthreads();

    // Process each KV position cooperatively
    for (int t = t_start; t < t_end; t++) {
        // ── Load and dequant K[t, kvh] (cooperative) ──
        float max_abs_k = (float)K_scale[t * num_kv_heads + kvh];
        for (int i = tid; i < HD/4; i += BLOCK) {
            uint8_t packed = K_idx[((size_t)t * num_kv_heads + kvh) * HD/4 + i];
            s_k[i*4+0] = int2_dequant(packed, max_abs_k);
            s_k[i*4+1] = int2_dequant(packed >> 2, max_abs_k);
            s_k[i*4+2] = int2_dequant(packed >> 4, max_abs_k);
            s_k[i*4+3] = int2_dequant(packed >> 6, max_abs_k);
        }
        __syncthreads();
        apply_rotation_T(R_K, s_k, s_k_rot, HD);
        __syncthreads();

        // ── Score: q · k (cooperative dot product) ──
        // Each thread sums the i%BLOCK==tid slice of the dot product, then
        // __shfl_xor reduces WITHIN each warp — leaving every warp holding
        // only its own 32-lane partial sum, not the true full-HD dot
        // product whenever HD > WARP (i.e. whenever more than one warp is
        // needed). The old code fed that per-warp partial straight into the
        // softmax as if it were the complete score, silently computing
        // wrong attention weights for any HD > 32. Reduce across warps too
        // via shared memory before using it.
        float partial = 0.0f;
        for (int i = tid; i < HD; i += BLOCK)
            partial += s_q[i] * s_k_rot[i];
        for (int off = WARP/2; off > 0; off /= 2)
            partial += __shfl_xor_sync(0xffffffffULL, partial, off);
        if (lane_id == 0) s_warp_score[warp_id] = partial;
        __syncthreads();
        float score = 0.0f;
        for (int w = 0; w < BLOCK / WARP; w++) score += s_warp_score[w];
        score *= attn_scale;

        // ── Online softmax update — single thread, block-wide state ──
        if (tid == 0) {
            s_m_old = s_m;
            float new_m = fmaxf(s_m, score);
            float alpha = __expf(s_m - new_m);
            float beta  = __expf(score - new_m);
            s_l = s_l * alpha + beta;
            s_m = new_m;
        }
        __syncthreads();
        float m_new = s_m, m_old = s_m_old;
        float alpha = __expf(m_old - m_new);

        // ── Rescale existing accumulator: o *= alpha ──
        for (int i = tid; i < HD; i += BLOCK)
            s_o[i] *= alpha;

        // ── Load, dequant, and rotate V[t, kvh] (cooperative) ──
        float max_abs_v = (float)V_scale[t * num_kv_heads + kvh];
        for (int i = tid; i < HD/4; i += BLOCK) {
            uint8_t packed = V_idx[((size_t)t * num_kv_heads + kvh) * HD/4 + i];
            s_v[i*4+0] = int2_dequant(packed, max_abs_v);
            s_v[i*4+1] = int2_dequant(packed >> 2, max_abs_v);
            s_v[i*4+2] = int2_dequant(packed >> 4, max_abs_v);
            s_v[i*4+3] = int2_dequant(packed >> 6, max_abs_v);
        }
        __syncthreads();
        apply_rotation_T(R_V, s_v, s_v_rot, HD);
        __syncthreads();

        // ── Accumulate: o += exp(score - m_new) * v (UNnormalized) ──
        // Dividing by the running s_l on every iteration (the old code) is
        // wrong: s_l keeps growing as later terms are added, but the
        // rescale step above only corrects the existing accumulator for the
        // shift in m, never for the fact that l has grown since earlier
        // terms were folded in — the earlier terms end up permanently
        // under-weighted relative to the true softmax denominator. Standard
        // flash-attention accumulates unnormalized like this and divides by
        // the FINAL l exactly once, after the loop. Verified empirically:
        // dividing every iteration produced a systematic ~15-20% output
        // bias against a quantize-aware fp32 reference; dividing once at
        // the end matches it closely.
        float p = __expf(score - m_new);
        for (int i = tid; i < HD; i += BLOCK)
            s_o[i] += p * s_v_rot[i];
        __syncthreads();
    }

    // Normalize by the final softmax denominator exactly once.
    float inv_l = 1.0f / s_l;
    for (int i = tid; i < HD; i += BLOCK)
        ((__half*)out_dev)[q_head * HD + i] = (__half)(s_o[i] * inv_l);
}

} // anonymous namespace

// ── Host-side: pre-allocated rotation matrix buffers (reused across calls) ──
namespace {
    float* g_R_dev = nullptr;
    float* g_RK_dev = nullptr;
    float* g_RV_dev = nullptr;
    int g_R_capacity = 0;   // current capacity in floats (= HD*HD)
    int g_RK_capacity = 0;
    int g_RV_capacity = 0;

    float* ensure_R_buffer(float*& buf, int& capacity, int needed) {
        if (capacity < needed) {
            if (buf) HIP_CHECK(hipFree(buf));
            HIP_CHECK(hipMalloc(&buf, needed * sizeof(float)));
            capacity = needed;
        }
        return buf;
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

    // Pre-allocate or reuse rotation matrix buffer
    int needed = HD * HD;
    float* R_dev = ensure_R_buffer(g_R_dev, g_R_capacity, needed);
    HIP_CHECK(hipMemcpy(R_dev, R_host, needed * sizeof(float), hipMemcpyHostToDevice));

    dim3 grid(seq_len, num_kv_heads);
    int smem = 2 * MAX_HD * sizeof(float); // s_row + s_rot
    hipLaunchKernelGGL(oscar_quantize_kernel, grid, WARP, smem, (hipStream_t)stream,
                       (const __half*)in_fp16, (uint8_t*)out_idx, (__half*)out_scale,
                       seq_len, num_kv_heads, HD, layer_idx, R_dev);
    HIP_CHECK(hipGetLastError());
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

    // Pre-allocate or reuse rotation matrix buffers
    int needed = HD * HD;
    float* R_K_dev = ensure_R_buffer(g_RK_dev, g_RK_capacity, needed);
    float* R_V_dev = ensure_R_buffer(g_RV_dev, g_RV_capacity, needed);
    HIP_CHECK(hipMemcpy(R_K_dev, R_K_host, needed * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(R_V_dev, R_V_host, needed * sizeof(float), hipMemcpyHostToDevice));

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
    HIP_CHECK(hipGetLastError());
    return 0;
}
