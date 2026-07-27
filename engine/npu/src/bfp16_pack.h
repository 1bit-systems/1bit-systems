//===- bfp16_pack.h ------------------------------------------------*- C++ -*-===//
// Convert f32 weights to v8bfp16ebs8 block-float format for the BF16 GEMM xclbins.
// Packing: 8 f32 values → BF16 → 1 exponent byte + 5 mantissa bytes = 6 bytes
//===----------------------------------------------------------------------===//

#ifndef BFP16_PACK_H
#define BFP16_PACK_H

#include <cstdint>
#include <cmath>
#include <cstring>

// Convert f32 to bfloat16 (truncate lower 16 bits)
inline uint16_t f32_to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    // Round to nearest even
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    bits += rounding_bias;
    return (uint16_t)(bits >> 16);
}

// Pack 8 BF16 values into v8bfp16ebs8 format (6 bytes: 1 exponent + 5 mantissas)
// Each mantissa is 5 bits signed (-16 to +15)
// Output layout: [exp][m4:0][m9:5][m14:10][m19:15][m24:20][m29:25][m34:30][m39:35]
// Which is: [exp][byte0][byte1][byte2][byte3][byte4]
// where byte0 = mant0[4:0] | mant1[4:0]<<5 | mant2[4:0]<<... 
// Actually: each 5-bit mantissa packed across 5 bytes (bit 0 of each mantissa in byte 0, etc.)
inline void pack_bfp16_block(const uint16_t bf16_vals[8], uint8_t out[6]) {
    // Find shared exponent
    float max_abs = 0;
    for (int i = 0; i < 8; i++) {
        uint32_t bits = (uint32_t)bf16_vals[i] << 16;
        float f;
        memcpy(&f, &bits, 4);
        float a = fabsf(f);
        if (std::isfinite(a) && a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-12f) {
        // All zeros
        out[0] = 0;
        for (int i = 0; i < 5; i++) out[1+i] = 0;
        return;
    }

    // Exponent: ceil(log2(max_abs))
    uint32_t max_bits;
    memcpy(&max_bits, &max_abs, 4);
    int exp = ((max_bits >> 23) & 0xFF) - 127 + 1;  // ceil(log2)
    if (exp < -126) exp = -126;

    float scale = (float)(1 << (exp + 4));  // 2^(exp+4), so mantissa range is [-16, 15]
    float inv_scale = 1.0f / scale;

    // Quantize each value to 5-bit signed mantissa
    int8_t mants[8];
    for (int i = 0; i < 8; i++) {
        uint32_t bits = (uint32_t)bf16_vals[i] << 16;
        float f;
        memcpy(&f, &bits, 4);
        int q = (int)roundf(f * inv_scale);
        if (q > 15) q = 15;
        else if (q < -16) q = -16;
        mants[i] = (int8_t)(q & 0x1F);  // 5-bit signed value
    }

    // Exponent byte (bias by 127 to match f32 exponent format)
    out[0] = (uint8_t)(exp + 127);

    // Pack 8 × 5-bit values into 5 bytes
    // byte 0: mant0[0] | mant1[0]<<1 | mant2[0]<<2 | mant3[0]<<3 | mant4[0]<<4 | mant5[0]<<5 | mant6[0]<<6 | mant7[0]<<7
    // ...
    // byte 4: mant0[4] | mant1[4]<<1 | mant2[4]<<2 | mant3[4]<<3 | mant4[4]<<4 | mant5[4]<<5 | mant6[4]<<6 | mant7[4]<<7
    for (int bit = 0; bit < 5; bit++) {
        uint8_t byte = 0;
        for (int i = 0; i < 8; i++) {
            byte |= ((mants[i] >> bit) & 1) << i;
        }
        out[1 + bit] = byte;
    }
}

// Pack a weight matrix from f32 to v8bfp16ebs8 format
// Input: w is K*N f32 values in row-major order
// Output: packed buffer of size (K*N*6 + 7) / 8 bytes (rounded up)
inline void pack_bfp16_weights(const float* w, int K, int N, uint8_t* packed_buf, size_t packed_size) {
    // Process in blocks of 8 values
    // For a K×N matrix, we pack along the inner dimension (N)
    size_t idx = 0;
    const size_t total_vals = (size_t)K * N;
    for (size_t i = 0; i < total_vals; i += 8) {
        uint16_t bf16_block[8];
        int n = 8;
        if (i + 8 > total_vals) n = total_vals - i;
        for (int j = 0; j < n; j++) {
            bf16_block[j] = f32_to_bf16(w[i + j]);
        }
        for (int j = n; j < 8; j++) {
            bf16_block[j] = 0;  // Pad with zeros
        }
        uint8_t packed[6];
        pack_bfp16_block(bf16_block, packed);
        for (int j = 0; j < 6 && idx < packed_size; j++) {
            packed_buf[idx++] = packed[j];
        }
    }
}

// Dequantize BF16 output from the NPU to f32
// bC buffer contains BF16 values (uint16_t), convert to float
inline void dequant_bf16_output(const uint16_t* bf16_data, float* f32_out, int count) {
    for (int i = 0; i < count; i++) {
        uint32_t bits = (uint32_t)bf16_data[i] << 16;
        float val;
        memcpy(&val, &bits, 4);
        if (!std::isfinite(val)) val = 0.0f;
        f32_out[i] = val;
    }
}
#endif // BFP16_PACK_H
