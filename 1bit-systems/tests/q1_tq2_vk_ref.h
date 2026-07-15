// q1_tq2_vk_ref.h — CPU reference math for TQ2_0_g128 / Q1_0_g128, the
// oracle for tests/test_vulkan_gemv.cpp. Mirrors src/bonsai_tq2_gemv.hip
// and src/bonsai_q1_gemv.hip exactly (same math already validated this
// session in zinc's tq2_ref.zig / q1_ref.zig).
#ifndef Q1_TQ2_VK_REF_H
#define Q1_TQ2_VK_REF_H

#include <cstdint>
#include <cstring>
#include <vector>

namespace vkref {

// ── TQ2_0_g128: 34 bytes / 128 weights, d-first, linear (code-1)*d ─────────
// [fp16 d : 2 bytes][qs[32] : 32 bytes, 2-bit codes LSB-first, 4/byte]
// value(code) = code - 1 -> {-1, 0, +1, +2}. Verified bit-exact vs F16.

constexpr uint32_t kTq2BlockWeights = 128;
constexpr uint32_t kTq2BlockBytes = 34;

inline float fp16ToFloat(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign << 31; }
        else {
            // subnormal
            int e = -1;
            uint32_t m = mant;
            do { m <<= 1; e++; } while (!(m & 0x400));
            m &= 0x3FF;
            bits = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        bits = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline float tq2CodeValue(uint32_t code) {
    return (float)code - 1.0f;
}

// Dot product of one packed TQ2 row ((k/128)*34 bytes) against a k-length
// fp16 activation vector, accumulated in f32.
inline float tq2RowDot(const uint8_t* row, const uint16_t* act, uint32_t k) {
    float sum = 0.0f;
    uint32_t num_blocks = k / kTq2BlockWeights;
    for (uint32_t block = 0; block < num_blocks; block++) {
        const uint8_t* blk = row + (size_t)block * kTq2BlockBytes;
        uint16_t d_bits;
        std::memcpy(&d_bits, blk, 2);
        float d = fp16ToFloat(d_bits);

        float partial = 0.0f;
        for (uint32_t byte_idx = 0; byte_idx < 32; byte_idx++) {
            uint8_t qbyte = blk[2 + byte_idx];
            for (uint32_t j = 0; j < 4; j++) {
                uint32_t code = (qbyte >> (j * 2)) & 3u;
                uint32_t w = block * kTq2BlockWeights + byte_idx * 4 + j;
                partial += tq2CodeValue(code) * fp16ToFloat(act[w]);
            }
        }
        sum += d * partial;
    }
    return sum;
}

// weights: m packed rows of (k/128)*34 bytes each.
inline void tq2Gemv(const uint8_t* weights, const uint16_t* act, float* out, uint32_t m, uint32_t k) {
    size_t row_bytes = (size_t)(k / kTq2BlockWeights) * kTq2BlockBytes;
    for (uint32_t row = 0; row < m; row++) {
        out[row] = tq2RowDot(weights + row * row_bytes, act, k);
    }
}

// ── Q1_0_g128: 18 bytes / 128 weights, sign bit LSB-first ──────────────────
// [fp16 d : 2 bytes][qs[16] : 16 bytes, 1 sign bit/weight, LSB-first]
// w[i] = (qs[i/8] >> (i%8)) & 1 ? +d : -d.

constexpr uint32_t kQ1BlockWeights = 128;
constexpr uint32_t kQ1BlockBytes = 18;

inline float q1SignValue(uint32_t bit) {
    return bit ? 1.0f : -1.0f;
}

inline float q1RowDot(const uint8_t* row, const uint16_t* act, uint32_t k) {
    float sum = 0.0f;
    uint32_t num_blocks = k / kQ1BlockWeights;
    for (uint32_t block = 0; block < num_blocks; block++) {
        const uint8_t* blk = row + (size_t)block * kQ1BlockBytes;
        uint16_t d_bits;
        std::memcpy(&d_bits, blk, 2);
        float d = fp16ToFloat(d_bits);

        float partial = 0.0f;
        for (uint32_t i = 0; i < kQ1BlockWeights; i++) {
            uint32_t byte_idx = i >> 3;
            uint32_t bit_idx = i & 7u;
            uint32_t bit = (blk[2 + byte_idx] >> bit_idx) & 1u;
            uint32_t w = block * kQ1BlockWeights + i;
            partial += q1SignValue(bit) * fp16ToFloat(act[w]);
        }
        sum += d * partial;
    }
    return sum;
}

inline void q1Gemv(const uint8_t* weights, const uint16_t* act, float* out, uint32_t m, uint32_t k) {
    size_t row_bytes = (size_t)(k / kQ1BlockWeights) * kQ1BlockBytes;
    for (uint32_t row = 0; row < m; row++) {
        out[row] = q1RowDot(weights + row * row_bytes, act, k);
    }
}

} // namespace vkref

#endif // Q1_TQ2_VK_REF_H
