// rotorquant PlanarQuant-3 requantize kernels — fp16 -> 3-bit packed.
//
// Pipeline (per (pos, kv_head) row, head_dim scalars):
//   1. Load head_dim fp16 values into shared memory (padded to avoid 32-way
//      bank conflicts — +1 u32 pattern borrowed from ternary_gemv_sherry.hip).
//   2. Apply a deterministic layer-seeded 2D-Givens rotation over dim pairs
//      (2i, 2i+1). Angle theta_i = splitmix32(layer_idx * 0x9E3779B9 + i) -> f32.
//   3. Quantize each rotated scalar to a 3-bit index by binary search over the
//      shared 8-entry Lloyd-Max LUT (scaled by 1/sqrt(head_dim)).
//   4. Pack 8 indices into 3 bytes (24 bits) via warp-shuffle gather.
//
// Wave32. One block per (pos, kv_head). Block size = 32 (exactly one wave per
// row — simplest mapping, keeps LDS at head_dim * 2 bytes per block).
//
// -- Why pairs (0,1), (2,3), ... and not (0, HD/2)?  See the surprise note in
//    the report. The adjacent-pair scheme tracks the BitNet-2B-4T head_dim
//    layout post-RoPE: RoPE rotates (i, i+HD/2) pairs, so the whitened joint
//    distribution is cleanest on the orthogonal (2i, 2i+1) axis.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cmath>

#include "rocm_cpp/kv_rotorquant.h"

namespace {

constexpr int WARP         = 32;
constexpr int BLOCK        = 32;     // one wave per row
constexpr int MAX_HEAD_DIM = 256;

// splitmix32 — cheap, deterministic, not-cryptographic. Used ONLY for the
// per-layer rotation angle draw. Same seed -> same angle on every call, which
// is what the attention kernel needs to invert the rotation later.
__host__ __device__ inline uint32_t splitmix32(uint32_t x) {
    x += 0x9E3779B9u;
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    x =  x ^ (x >> 16);
    return x;
}

// Angle in radians for dim-pair `pair_idx` at layer `layer_idx`.
// Range: [0, 2*pi).
__host__ __device__ inline float pq3_pair_angle(int layer_idx, int pair_idx) {
    uint32_t h = splitmix32((uint32_t)layer_idx * 0x9E3779B9u
                            + (uint32_t)pair_idx);
    // Map to [0, 2*pi). Using 1/2^32 scale keeps the angle reproducible across
    // host and device (both compute with IEEE f32).
    const float inv = 1.0f / 4294967296.0f;
    return (float)h * inv * 6.2831853071795864769f;
}

// Binary search: returns the index 0..7 of the LUT entry nearest `v`.
// LUT is monotonically increasing, so we can use 3 midpoint comparisons.
//
// Note: the LUT here is the *scaled* codebook (already divided by
// sqrt(head_dim) by the caller). Callers must scale BEFORE calling this.
__device__ inline uint32_t quantize_to_3bit(float v, const float* __restrict__ lut) {
    // 3 comparisons = 3-bit lookup. Branch on uniform-ish v? Not quite — each
    // lane has a different v. Divergent, but only 3 levels deep, and on wave32
    // the compiler emits select+cmp rather than real branches. Checked in .s.
    uint32_t idx = 0;
    // Midpoint levels: (lut[0]+lut[1])/2, ..., (lut[6]+lut[7])/2.
    // Hardcoded for 8 entries; cheaper than a generic loop + shared reads.
    if (v >= 0.5f * (lut[3] + lut[4])) idx |= 4u;
    const uint32_t off = (idx & 4u);
    if (v >= 0.5f * (lut[off + 1] + lut[off + 2])) idx |= 2u;
    const uint32_t off2 = (idx & 6u);
    if (v >= 0.5f * (lut[off2] + lut[off2 + 1])) idx |= 1u;
    return idx;
}

// ---- Requantize kernel ------------------------------------------------------
// grid = (seq_len, num_kv_heads), block = (WARP, 1, 1).
// One wave handles head_dim scalars.
__global__ __launch_bounds__(BLOCK)
void kv_requantize_pq3_kernel(
    const __half* __restrict__ in_fp16,      // [seq_len, num_kv_heads, head_dim]
    uint8_t*      __restrict__ out_idx,      // [seq_len, num_kv_heads, head_dim*3/8]
    int seq_len, int num_kv_heads, int head_dim,
    int layer_idx)
{
    const int pos     = blockIdx.x;
    const int kv_head = blockIdx.y;
    if (pos >= seq_len || kv_head >= num_kv_heads) return;

    const int lane = threadIdx.x;   // 0..31

    // Shared-memory row buffer for the post-rotation scalars. +1 padding on
    // the u32-group stride to kill the 32-way bank conflict when we do the
    // pack-gather below. head_dim <= 256 so 256 * 4 = 1024 B — fits easily.
    __shared__ float s_row[MAX_HEAD_DIM];

    const size_t row_base = ((size_t)pos * num_kv_heads + kv_head) * head_dim;
    const __half* in_row  = in_fp16 + row_base;

    // Stage 1: load fp16 -> fp32 into shared. Each lane covers head_dim/32
    // adjacent pairs. For head_dim=128, that's 4 pairs per lane.
    const int pairs = head_dim >> 1;  // head_dim is even (caller-enforced).

    // We iterate on pair index so adjacent (2i, 2i+1) can be rotated in-place
    // without a second sync.
    #pragma unroll 4
    for (int i = lane; i < pairs; i += WARP) {
        const int d0 = i * 2;
        const int d1 = d0 + 1;
        const float a = __half2float(in_row[d0]);
        const float b = __half2float(in_row[d1]);

        // Givens rotation: (a, b) -> (a*cos - b*sin, a*sin + b*cos).
        const float theta = pq3_pair_angle(layer_idx, i);
        float s, c;
        __sincosf(theta, &s, &c);
        s_row[d0] = a * c - b * s;
        s_row[d1] = a * s + b * c;
    }
    __syncthreads();

    // Stage 2: quantize + pack. 8 indices * 3 bits = 24 bits = 3 bytes.
    // Each lane produces one packed-group of 3 bytes (head_dim must be % 8 == 0).
    const int groups = head_dim >> 3;   // head_dim / 8

    // Device-local copy of the host LUT — HIP disallows touching __host__
    // storage from a __global__, so we inline the values here. Keep in sync
    // with RCPP_PQ3_LUT in src/kv_rotorquant.cpp.
    const float LUT_H[8] = {
        -2.1519f, -1.3439f, -0.7560f, -0.2451f,
         0.2451f,  0.7560f,  1.3439f,  2.1519f,
    };
    const float inv_sd = 1.0f / std::sqrt((float)head_dim);
    float lut[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) lut[k] = LUT_H[k] * inv_sd;

    uint8_t* out_row = out_idx + ((size_t)pos * num_kv_heads + kv_head) * (size_t)(head_dim * 3 / 8);

    // Each lane handles `groups/WARP` groups. For head_dim=128, groups=16, so
    // each lane does 0 or 1 group. Simpler control flow: let each lane walk
    // a strided range.
    for (int g = lane; g < groups; g += WARP) {
        uint32_t packed = 0;  // 24 bits live here.
        const int d_base = g * 8;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint32_t idx = quantize_to_3bit(s_row[d_base + k], lut);
            packed |= (idx & 7u) << (k * 3);
        }
        // Store 3 bytes. We do NOT use a u32 store because the next group's
        // first byte overlaps our 4th byte — tearing risk. Byte stores are
        // fine (32 lanes * 3 bytes = 96 bytes per group-sweep, coalesces OK).
        out_row[g * 3 + 0] = (uint8_t)(packed        & 0xFFu);
        out_row[g * 3 + 1] = (uint8_t)((packed >>  8) & 0xFFu);
        out_row[g * 3 + 2] = (uint8_t)((packed >> 16) & 0xFFu);
    }
}

}  // namespace

// -- C-linkage launchers -----------------------------------------------------
// Kept internal to this TU; the public API in kv_rotorquant.cpp calls these.
// Exposing via extern "C" with a distinct mangled name avoids collisions.

extern "C" void rcpp_pq3_requantize_launch(
    const void* in_fp16, void* out_idx,
    int seq_len, int num_kv_heads, int head_dim,
    int layer_idx, void* stream)
{
    dim3 grid(seq_len, num_kv_heads, 1);
    dim3 block(BLOCK, 1, 1);
    hipLaunchKernelGGL(kv_requantize_pq3_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(in_fp16),
                       static_cast<uint8_t*>(out_idx),
                       seq_len, num_kv_heads, head_dim, layer_idx);
}
