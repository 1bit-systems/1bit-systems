#ifndef BLOCK_SCALED_TERNARY_H
#define BLOCK_SCALED_TERNARY_H

#include <cstdint>
#include <cmath>
#include <cstring>
#include <cassert>

// __host__ __device__ annotation for HIP/CUDA dual-use functions.
// On non-GPU compilers, this expands to nothing.
#if defined(__HIP__) || defined(__HIPCC__) || defined(__CUDACC__)
#define BST_HOST_DEVICE __host__ __device__
#else
#define BST_HOST_DEVICE
#endif

static constexpr int BST_BLOCK_K      = 16;
static constexpr int BST_BLOCK_BYTES  = 5;
static constexpr int BST_BITS_PER_VAL = 2;
static constexpr uint8_t FP8_E4M3_NAN = 0xFF;

// Precomputed FP32 normal bit patterns for FP8 E4M3 subnormal mantissas (e=0).
// For m in 1..7: value = m/8 * 2^(-6). These are encoded as FP32 normals.
static const uint32_t FP8_SUBNORM_FP32[8] = {
    0,           // m=0 -> 0.0
    0x3B000000u, // m=1 -> 1/8 * 2^-6 = 2^-9
    0x3B800000u, // m=2 -> 2/8 * 2^-6 = 2^-8
    0x3BC00000u, // m=3 -> 3/8 * 2^-6 = 3*2^-9
    0x3C000000u, // m=4 -> 4/8 * 2^-6 = 2^-7
    0x3C200000u, // m=5 -> 5/8 * 2^-6 = 5*2^-9
    0x3C400000u, // m=6 -> 6/8 * 2^-6 = 3*2^-8
    0x3C600000u, // m=7 -> 7/8 * 2^-6 = 7*2^-9
};

// Single source of truth for FP8 E4M3 -> FP32 conversion.
// Used by both host code (CPU reference) and device code (GPU kernel).
BST_HOST_DEVICE inline float fp8e4m3_to_fp32(uint8_t fp8) {
    if (fp8 == FP8_E4M3_NAN) {
        uint32_t bits = 0x7FC00000u;  // quiet NaN
        float r;
        __builtin_memcpy(&r, &bits, sizeof(r));
        return r;
    }
    uint32_t s = (fp8 >> 7) & 1, e = (fp8 >> 3) & 0xF, m = fp8 & 0x7;
    uint32_t bits;
    if (e == 0) {
        bits = FP8_SUBNORM_FP32[m] | (s << 31);
    } else {
        bits = (s << 31) | ((e + 120) << 23) | (m << 20);
    }
    float r;
    __builtin_memcpy(&r, &bits, sizeof(r));
    return r;
}

// Host-only: FP32 -> FP8 E4M3 with RNE rounding.
// Uses std::isnan which is not available in GPU device code.
inline uint8_t fp32_to_fp8e4m3(float v) {
    // Use bit-level NaN check to avoid UB under -ffast-math (issue #404)
    uint32_t vbits; __builtin_memcpy(&vbits, &v, sizeof(vbits));
    if ((vbits & 0x7F800000u) == 0x7F800000u && (vbits & 0x007FFFFFu) != 0) return FP8_E4M3_NAN;
    if (v > 448.0f) v = 448.0f;
    if (v < -448.0f) v = -448.0f;
    if (v > -0.0009765625f && v < 0.0009765625f) v = 0.0f;

    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    uint32_t s        = (bits >> 31) & 1;
    int32_t  exp      = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant_rne = (bits >> 19) & 0xF;

    uint32_t lsb    = (bits >> 20) & 1;
    uint32_t round  = (bits >> 19) & 1;
    uint32_t sticky = (bits & 0x7FFFF) != 0 ? 1 : 0;
    int rne = (int)(round & (lsb | sticky));
    mant_rne = (mant_rne + rne);
    if (mant_rne > 15) { mant_rne >>= 1; exp++; }
    mant_rne >>= 1;
    if (mant_rne > 7) { mant_rne = 0; exp++; }

    if (exp < -6) return (s << 7);
    if (exp > 8) { exp = 8; mant_rne = 7; }

    return (s << 7) | ((uint32_t)(exp + 7) << 3) | (mant_rne & 0x7);
}

inline uint32_t ternary_pack_16(const int8_t values[16]) {
    uint32_t word = 0;
    for (int i = 0; i < 16; ++i) {
        uint32_t code;
        if      (values[i] ==  1) code = 1;
        else if (values[i] == -1) code = 2;
        else                      code = 0;
        word |= (code << (i * 2));
    }
    return word;
}

inline void ternary_unpack_16(uint32_t packed, int8_t out[16]) {
    for (int i = 0; i < 16; ++i) {
        uint32_t bits = (packed >> (i * 2)) & 0x3;
        out[i] = (bits == 1) ? 1 : (bits == 2) ? -1 : 0;
    }
}

inline float block_scaled_ternary_dequant(
    const uint8_t block[BST_BLOCK_BYTES], int elem_idx)
{
    assert(elem_idx >= 0 && elem_idx < 16);
    uint32_t packed;
    __builtin_memcpy(&packed, block, sizeof(packed));
    uint32_t bits = (packed >> (elem_idx * 2)) & 0x3;
    int8_t tv = (bits == 1) ? 1 : (bits == 2) ? -1 : 0;
    return (float)tv * fp8e4m3_to_fp32(block[4]);
}

inline int block_scaled_ternary_pack_row(
    const float* row, uint8_t* blocks, int cols)
{
    int n_blocks = (cols + BST_BLOCK_K - 1) / BST_BLOCK_K;
    for (int b = 0; b < n_blocks; ++b) {
        int start = b * BST_BLOCK_K;
        int end = (start + BST_BLOCK_K <= cols) ? start + BST_BLOCK_K : cols;
        float amax = 0.0f;
        for (int i = start; i < end; ++i) {
            float absv = std::abs(row[i]);
            if (absv > amax) amax = absv;
        }
        float scale = (amax > 0.0f) ? amax : 1.0f;
        int8_t vals[16] = {};
        for (int i = start; i < end; ++i) {
            float q = row[i] / scale;
            if (q > 0.5f)       vals[i - start] = 1;
            else if (q < -0.5f) vals[i - start] = -1;
            else                vals[i - start] = 0;
        }
        uint32_t packed = ternary_pack_16(vals);
        __builtin_memcpy(blocks + b * BST_BLOCK_BYTES, &packed, 4);
        blocks[b * BST_BLOCK_BYTES + 4] = fp32_to_fp8e4m3(scale);
    }
    return n_blocks;
}

inline void block_scaled_ternary_dequant_row(
    const uint8_t* blocks, float* row, int cols)
{
    int n_blocks = (cols + BST_BLOCK_K - 1) / BST_BLOCK_K;
    for (int b = 0; b < n_blocks; ++b) {
        int start = b * BST_BLOCK_K;
        int end = (start + BST_BLOCK_K <= cols) ? start + BST_BLOCK_K : cols;
        float scale = fp8e4m3_to_fp32(blocks[b * BST_BLOCK_BYTES + 4]);
        uint32_t packed;
        __builtin_memcpy(&packed, blocks + b * BST_BLOCK_BYTES, 4);
        for (int i = start; i < end; ++i) {
            uint32_t bits = (packed >> ((i - start) * 2)) & 0x3;
            row[i] = (float)((bits == 1) ? 1 : (bits == 2) ? -1 : 0) * scale;
        }
    }
}

#undef BST_HOST_DEVICE
#endif
