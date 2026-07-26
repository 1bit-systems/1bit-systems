// VitaLLM-style Predictive Sparse Attention (ISCAS 2026)
// Paper: VitaLLM — A Versatile and Tiny Accelerator for Mixed-Precision LLM Inference
//
// Key insight: Leading-One (LO) surrogate avoids expensive dot products
//   ŝ(q,k) = Σᵢ sgn(qᵢ)·sgn(kᵢ)·2^{LO(qᵢ)+LO(kᵢ)}
//   where LO(x) = ⌊log₂|x|⌋
//
// Comparison-free top-K via bucketization:
//   1. Bucketize scores into discrete ranges
//   2. Prefix scan to find cut bin
//   3. Priority encoder to emit K indices
//
// Result: ~1-K/M KV traffic reduction (typically 88%+ at K=32, M=256)

#include <hip/hip_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 256
#define WARP_SIZE 64  // RDNA 3.5: wave64
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define N_BUCKETS 64

#include "hip_check.h"

// Leading-One: ⌊log₂|x|⌋
// Uses AMD's flbit_u32 (find last bit) instruction
__device__ inline int leading_one(float x) {
    uint32_t u;
    __builtin_memcpy(&u, &x, 4);
    uint32_t abs_val = u & 0x7FFFFFFF;
    if (abs_val == 0) return -31;  // special case for 0
    // AMD GPU has __ffs (find first set) or we use flbit
    // flbit_u32 returns bit position of most significant 1
    int exp = (u >> 23) & 0xFF;
    int mant = (int)(u & 0x007FFFFF) | 0x00800000;
    return exp - 127;  // approximate log2 via exponent
}

// LO surrogate score: efficient, multiplier-free
// Each term = shift + XOR sign → hardware-friendly
__device__ inline float lo_surrogate_score(
    const float* __restrict__ q,
    const float* __restrict__ k,
    int d)
{
    float score = 0.0f;
    for (int i = 0; i < d; i++) {
        float qv = q[i], kv = k[i];
        int lo_q = leading_one(qv);
        int lo_k = leading_one(kv);
        int sign = (qv > 0) == (kv > 0) ? 1 : -1;
        // 2^{LO(q)+LO(k)} = shift 1.0 by (LO(q)+LO(k))
        // Handle negative shifts (values < 1.0)
        int shift = lo_q + lo_k;
        float term;
        if (shift >= 0) {
            term = (float)sign * (1 << shift);
        } else {
            term = (float)sign / (float)(1 << (-shift));
        }
        score += term;
    }
    return score;
}

// Compute surrogate scores for all KV pairs
__global__ void compute_lo_surrogate_scores(
    const float* __restrict__ query,    // [d]
    const float* __restrict__ keys,     // [M, d]
    float* __restrict__ scores,         // [M] output scores
    int M, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;

    scores[idx] = lo_surrogate_score(query, &keys[idx * d], d);
}

// Comparison-free top-K via bucketization + prefix scan
// Returns number of selected indices in selected_indices
__global__ void vitallm_topk_select(
    const float* __restrict__ scores,
    int* __restrict__ selected_indices,
    int M, int K)
{
    __shared__ int bucket_counts[N_BUCKETS];
    __shared__ int bucket_prefix[N_BUCKETS + 1];
    __shared__ int cut_bin;
    __shared__ int cut_remainder;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Initialize buckets
    if (tid < N_BUCKETS) bucket_counts[tid] = 0;
    __syncthreads();

    // Find score range
    __shared__ float s_min, s_max;
    if (tid == 0) {
        s_min = s_max = scores[0];
    }
    __syncthreads();

    // Parallel reduction for min/max
    float local_min = (idx < M) ? scores[idx] : 1e10f;
    float local_max = (idx < M) ? scores[idx] : -1e10f;

    // Warp-level reduction (assume wave64)
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) {
        local_min = fminf(local_min, __shfl_xor_sync(0xffffffffULL, local_min, o));
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffULL, local_max, o));
    }

    if (tid % WARP_SIZE == 0) {
        // Atomic across warps
        atomicMinf(&s_min, local_min);
        atomicMaxf(&s_max, local_max);
    }
    __syncthreads();

    // Bucketize scores
    float range = s_max - s_min;
    if (range < 1e-10f) range = 1.0f;

    if (idx < M) {
        int bucket = (int)((scores[idx] - s_min) / range * N_BUCKETS);
        if (bucket >= N_BUCKETS) bucket = N_BUCKETS - 1;
        if (bucket < 0) bucket = 0;
        atomicAdd(&bucket_counts[bucket], 1);
    }
    __syncthreads();

    // Prefix scan (exclusive)
    if (tid < N_BUCKETS + 1) {
        bucket_prefix[tid] = 0;
    }
    __syncthreads();

    if (tid < N_BUCKETS) {
        bucket_prefix[tid + 1] = bucket_counts[tid];
    }
    __syncthreads();

    // Parallel prefix sum
    for (int i = 1; i <= N_BUCKETS; i <<= 1) {
        if (tid >= i) {
            bucket_prefix[tid] += bucket_prefix[tid - i];
        }
        __syncthreads();
    }

    // Find cut bin: first bucket where cumulative >= K
    if (tid == 0) {
        cut_bin = N_BUCKETS - 1;
        cut_remainder = 0;
        for (int i = 0; i < N_BUCKETS; i++) {
            if (bucket_prefix[i + 1] >= K) {
                cut_bin = i;
                cut_remainder = K - bucket_prefix[i];
                break;
            }
        }
    }
    __syncthreads();

    // Output selected indices
    if (idx < M) {
        int bucket = (int)((scores[idx] - s_min) / range * N_BUCKETS);
        if (bucket >= N_BUCKETS) bucket = N_BUCKETS - 1;
        if (bucket < 0) bucket = 0;

        if (bucket > cut_bin ||
            (bucket == cut_bin && (bucket_prefix[cut_bin] + atomicAdd(&cut_remainder, -1) > 0))) {
            int pos = atomicAdd((int*)&selected_indices[0], 0);  // will fix
            // Simplified: just write at bucket position
            if (bucket > cut_bin) {
                selected_indices[bucket_prefix[bucket]] = idx;
            }
        }
    }
}

// Simplified top-K: just take K highest-scoring indices via sorting
// For production, use the bucketization approach above
__global__ void vitallm_topk_simple(
    const float* __restrict__ scores,
    int* __restrict__ indices,
    int M, int K)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= M) return;

    // For each index, check if it's in the top K
    float my_score = scores[idx];
    int count = 0;
    for (int i = 0; i < M; i++) {
        if (scores[i] > my_score || (scores[i] == my_score && i < idx))
            count++;
    }
    if (count < K) {
        indices[count] = idx;
    }
}
