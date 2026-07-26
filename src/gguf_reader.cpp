#include "gguf_reader.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

// Portable fseeko: MSVC uses _fseeki64, POSIX uses fseeko
#ifdef _WIN32
#define fseeko _fseeki64
#endif

namespace {

// Reads a real IEEE754 half-precision float16 (5-bit exponent, bias 15).
// NOT the naive `(uint32_t)bits << 16` trick — that's only valid for
// bfloat16 (whose bits are float32's truncated upper half) and silently
// produces the wrong value for every real float16 field (see issue #473).
inline float read_f16(const uint8_t* p) {
    uint16_t h; memcpy(&h, p, 2);
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float sign = s ? -1.0f : 1.0f;
    if (e == 0) return sign * (float)m * 5.9604644775390625e-08f;  // subnormal: m * 2^-24
    if (e == 31) return m ? NAN : sign * INFINITY;
    return sign * (1.0f + (float)m / 1024.0f) * powf(2.0f, (float)((int)e - 15));
}

// Shared 6-bit scale/min unpacking scheme used by Q4_K and Q5_K: 8
// (scale,min) pairs packed into 12 bytes (12*8/8/2 = 6 bits each).
inline void k_get_scale_min(const uint8_t scales[12], int j, uint8_t& sc, uint8_t& m) {
    if (j < 4) { sc = scales[j] & 63; m = scales[j + 4] & 63; }
    else { sc = (uint8_t)((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
           m = (uint8_t)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)); }
}

bool dequant_q4_0(const uint8_t* bd, float* out, int count) {
    int blocks = (count + 31) / 32;
    // Port of llama.cpp's dequantize_row_q4_0: elements are NOT interleaved
    // consecutive-pair nibbles (low,high,low,high...) — the low nibble of
    // byte j is element j, the HIGH nibble of byte j is element j+16 (first
    // half of the block is all low nibbles, second half all high nibbles).
    // The previous interleaved-pair version was never synthetically
    // verified and silently produced wrong values for every real Q4_0/Q4_1/
    // Q5_0/Q5_1 tensor.
    for (int b = 0; b < blocks; b++) {
        const uint8_t* p = bd + (size_t)b * 18;
        float scale = read_f16(p);
        const uint8_t* q = p + 2;
        for (int j = 0; j < 16; j++) {
            if (b * 32 + j < count)      out[b * 32 + j]      = ((q[j] & 0xF) - 8) * scale;
            if (b * 32 + j + 16 < count) out[b * 32 + j + 16] = ((q[j] >> 4) - 8) * scale;
        }
    }
    return true;
}

bool dequant_q4_1(const uint8_t* bd, float* out, int count) {
    int blocks = (count + 31) / 32;
    for (int b = 0; b < blocks; b++) {
        const uint8_t* p = bd + (size_t)b * 20;
        float scale = read_f16(p), min = read_f16(p + 2);
        const uint8_t* q = p + 4;
        for (int j = 0; j < 16; j++) {
            if (b * 32 + j < count)      out[b * 32 + j]      = (q[j] & 0xF) * scale + min;
            if (b * 32 + j + 16 < count) out[b * 32 + j + 16] = (q[j] >> 4) * scale + min;
        }
    }
    return true;
}

bool dequant_q5_0(const uint8_t* bd, float* out, int count) {
    int blocks = (count + 31) / 32;
    for (int b = 0; b < blocks; b++) {
        const uint8_t* p = bd + (size_t)b * 22;
        float scale = read_f16(p);
        uint32_t qh; memcpy(&qh, p + 2, 4);
        const uint8_t* q = p + 6;
        for (int j = 0; j < 16; j++) {
            uint8_t xh_0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
            uint8_t xh_1 = (uint8_t)((qh >> (j + 12)) & 0x10);
            if (b * 32 + j < count)      out[b * 32 + j]      = (int32_t)(((q[j] & 0xF) | xh_0) - 16) * scale;
            if (b * 32 + j + 16 < count) out[b * 32 + j + 16] = (int32_t)(((q[j] >> 4) | xh_1) - 16) * scale;
        }
    }
    return true;
}

bool dequant_q5_1(const uint8_t* bd, float* out, int count) {
    int blocks = (count + 31) / 32;
    for (int b = 0; b < blocks; b++) {
        const uint8_t* p = bd + (size_t)b * 24;
        float scale = read_f16(p), min = read_f16(p + 2);
        uint32_t qh; memcpy(&qh, p + 4, 4);
        const uint8_t* q = p + 8;
        for (int j = 0; j < 16; j++) {
            uint8_t xh_0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
            uint8_t xh_1 = (uint8_t)((qh >> (j + 12)) & 0x10);
            if (b * 32 + j < count)      out[b * 32 + j]      = ((q[j] & 0xF) | xh_0) * scale + min;
            if (b * 32 + j + 16 < count) out[b * 32 + j + 16] = ((q[j] >> 4) | xh_1) * scale + min;
        }
    }
    return true;
}

bool dequant_q8_0(const uint8_t* bd, float* out, int count) {
    int blocks = (count + 31) / 32;
    for (int b = 0; b < blocks; b++) {
        const uint8_t* p = bd + (size_t)b * 34;
        float scale = read_f16(p);
        const int8_t* q = (const int8_t*)(p + 2);
        for (int j = 0; j < 32 && b * 32 + j < count; j++) out[b * 32 + j] = q[j] * scale;
    }
    return true;
}

bool dequant_q2_k(const uint8_t* bd, float* out, int count) {
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        uint8_t scales[16]; memcpy(scales, p, 16); p += 16;
        uint8_t qs[64]; memcpy(qs, p, 64); p += 64;
        float d = read_f16(p); p += 2;
        float dmin = read_f16(p); p += 2;
        int base = b * BS;
        int is = 0; const uint8_t* q = qs;
        for (int n = 0; n < BS && base + n < count; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);
                for (int l = 0; l < 16 && base + n + j * 32 + l < count; l++)
                    out[base + n + j * 32 + l] = dl * ((q[l] >> shift) & 3) - ml;
                sc = scales[is++];
                dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
                for (int l = 0; l < 16 && base + n + j * 32 + 16 + l < count; l++)
                    out[base + n + j * 32 + 16 + l] = dl * ((q[l + 16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
    return true;
}

bool dequant_q3_k(const uint8_t* bd, float* out, int count) {
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        uint8_t hmask[32]; memcpy(hmask, p, 32); p += 32;
        uint8_t qs[64]; memcpy(qs, p, 64); p += 64;
        uint8_t raw_scales[12]; memcpy(raw_scales, p, 12); p += 12;
        float d_all = read_f16(p); p += 2;

        uint32_t aux[4] = {0, 0, 0, 0};
        memcpy(aux, raw_scales, 12);
        const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int8_t scales[16]; memcpy(scales, aux, 16);
        for (int j = 0; j < 16; j++) scales[j] -= 32;

        int base = b * BS;
        int is = 0; const uint8_t* q = qs;
        uint8_t m = 1;
        for (int n = 0; n < BS && base + n < count; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d_all * scales[is++];
                for (int l = 0; l < 16 && base + n + j * 32 + l < count; l++)
                    out[base + n + j * 32 + l] =
                        dl * (((int8_t)((q[l] >> shift) & 3)) - ((hmask[l] & m) ? 0 : 4));
                dl = d_all * scales[is++];
                for (int l = 0; l < 16 && base + n + j * 32 + 16 + l < count; l++)
                    out[base + n + j * 32 + 16 + l] =
                        dl * (((int8_t)((q[l + 16] >> shift) & 3)) - ((hmask[l + 16] & m) ? 0 : 4));
                shift += 2;
                m = (uint8_t)(m << 1);
            }
            q += 32;
        }
    }
    return true;
}

bool dequant_q4_k(const uint8_t* bd, float* out, int count) {
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        float d = read_f16(p); p += 2;
        float dmin = read_f16(p); p += 2;
        uint8_t scales[12]; memcpy(scales, p, 12); p += 12;
        uint8_t qs[128]; memcpy(qs, p, 128); p += 128;
        int base = b * BS;
        int is = 0; const uint8_t* q = qs;
        for (int off = 0; off < BS && base + off < count; off += 64) {
            uint8_t sc, m;
            k_get_scale_min(scales, is, sc, m);
            float d1 = d * sc, m1 = dmin * m;
            k_get_scale_min(scales, is + 1, sc, m);
            float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32 && base + off + l < count; l++)
                out[base + off + l] = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32 && base + off + 32 + l < count; l++)
                out[base + off + 32 + l] = d2 * (q[l] >> 4) - m2;
            q += 32; is += 2;
        }
    }
    return true;
}

bool dequant_q5_k(const uint8_t* bd, float* out, int count) {
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        float d = read_f16(p); p += 2;
        float dmin = read_f16(p); p += 2;
        uint8_t scales[12]; memcpy(scales, p, 12); p += 12;
        uint8_t qh[32]; memcpy(qh, p, 32); p += 32;
        uint8_t ql[128]; memcpy(ql, p, 128); p += 128;
        int base = b * BS;
        int is = 0; const uint8_t* q = ql;
        uint8_t u1 = 1, u2 = 2;
        for (int n = 0; n < BS && base + n < count; n += 64) {
            uint8_t sc, m;
            k_get_scale_min(scales, is, sc, m);
            float d1 = d * sc, m1 = dmin * m;
            k_get_scale_min(scales, is + 1, sc, m);
            float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32 && base + n + l < count; l++)
                out[base + n + l] = d1 * ((q[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32 && base + n + 32 + l < count; l++)
                out[base + n + 32 + l] = d2 * ((q[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            q += 32; is += 2;
            u1 = (uint8_t)(u1 << 2); u2 = (uint8_t)(u2 << 2);
        }
    }
    return true;
}

// Port of llama.cpp's dequantize_row_q6_K (ggml-quants.c) — the previous
// version of this function only ever wrote indices [0,32) of each 128-
// element half (192 of every 256 elements silently left as zero); this
// was never caught because it was only spot-checked against real files,
// never synthetically verified against the Python `gguf` reference the
// way Q2_K/Q3_K/Q5_K were. Verified byte-exact against `gguf.quants.Q6_K`
// with seeded random blocks before this fix landed.
bool dequant_q6_k(const uint8_t* bd, float* out, int count) {
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        const uint8_t* ql = p; p += 128;
        const uint8_t* qh = p; p += 64;
        const int8_t* sc = (const int8_t*)p; p += 16;
        float d = read_f16(p); p += 2;
        int base = b * BS;
        float* y = out + base;
        for (int n = 0; n < BS; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                if (base + n + l < count)      y[l]      = d * sc[is + 0] * q1;
                if (base + n + l + 32 < count) y[l + 32] = d * sc[is + 2] * q2;
                if (base + n + l + 64 < count) y[l + 64] = d * sc[is + 4] * q3;
                if (base + n + l + 96 < count) y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
    return true;
}

bool dequant_q8_k(const uint8_t* bd, float* out, int count) {
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        float d; memcpy(&d, p, 4); p += 4;
        int8_t qs[256]; memcpy(qs, p, 256); p += 256;
        p += 32; // bsums — not needed for dequant
        int base = b * BS;
        for (int l = 0; l < BS && base + l < count; l++) out[base + l] = d * qs[l];
    }
    return true;
}

} // namespace

inline float bf16_to_fp32(uint16_t bf16) {
    // bfloat16 -> float32: reinterpret the bits by shifting left 16.
    // No special handling needed for NaN/Inf — the bit pattern is identical.
    float result;
    uint32_t f32 = (uint32_t)bf16 << 16;
    memcpy(&result, &f32, 4);
    return result;
}

GgufBlockInfo gguf_block_info(uint32_t dtype) {
    switch (dtype) {
        case GGUF_DTYPE_F32:    return {1, 4};
        case GGUF_DTYPE_F16:    return {1, 2};
        case GGUF_DTYPE_BF16:   return {1, 2};
        case GGUF_DTYPE_Q4_0:   return {32, 18};
        case GGUF_DTYPE_Q4_1:   return {32, 20};
        case GGUF_DTYPE_Q5_0:   return {32, 22};
        case GGUF_DTYPE_Q5_1:   return {32, 24};
        case GGUF_DTYPE_Q8_0:   return {32, 34};
        case GGUF_DTYPE_Q8_1:   return {32, 36};
        case GGUF_DTYPE_Q2_K:   return {256, 84};
        case GGUF_DTYPE_Q3_K:   return {256, 110};
        case GGUF_DTYPE_Q4_K:   return {256, 144};
        case GGUF_DTYPE_Q5_K:   return {256, 176};
        case GGUF_DTYPE_Q6_K:   return {256, 210};
        case GGUF_DTYPE_Q8_K:   return {256, 292};
        // IQ format block sizes (for correct file offset computation).
        // Dequantization is not implemented here — these return false
        // from gguf_dequant; callers can use get_tensor_raw() for
        // custom dequant.
        case GGUF_DTYPE_IQ1_S:  return {256, 206};
        case GGUF_DTYPE_IQ1_M:  return {256, 230};
        case GGUF_DTYPE_IQ2_XXS: return {256, 166};
        case GGUF_DTYPE_IQ2_S:  return {256, 214};
        case GGUF_DTYPE_IQ3_XXS: return {256, 198};
        case GGUF_DTYPE_IQ3_S:  return {256, 238};
        case GGUF_DTYPE_IQ4_NL: return {32, 22};
        case GGUF_DTYPE_IQ4_XS: return {256, 214};
        case GGUF_DTYPE_Q4_0_4_4: return {256, 208};
        case GGUF_DTYPE_Q4_0_4_8: return {256, 208};
        case GGUF_DTYPE_Q4_0_8_8: return {256, 272};
        // Project-specific ternary/binary formats (h1b weight format)
        // TQ2_0_g128: ternary, 2-bit packed, group=128 → blocks of 128 el, 33 bytes
        // TQ2_0 ternary: fp16 scale (2) + 2-bit codes (128*2/8=32) = 34 bytes
        case GGUF_DTYPE_TQ2_0_G128: return {128, 34};
        // Q1_0 binary: fp16 scale (2) + 1-bit codes (128/8=16) = 18 bytes
        case GGUF_DTYPE_Q1_0_G128: return {128, 18};
        default: return {0, 0};
    }
}

bool gguf_dequant(uint32_t dtype, const uint8_t* data, float* out, int count) {
    switch (dtype) {
        case GGUF_DTYPE_F32: memcpy(out, data, (size_t)count * 4); return true;
        case GGUF_DTYPE_F16:
            for (int i = 0; i < count; i++) out[i] = read_f16(data + (size_t)i * 2);
            return true;
        case GGUF_DTYPE_BF16:
            for (int i = 0; i < count; i++) out[i] = bf16_to_fp32(((const uint16_t*)data)[i]);
            return true;
        case GGUF_DTYPE_Q4_0: return dequant_q4_0(data, out, count);
        case GGUF_DTYPE_Q4_1: return dequant_q4_1(data, out, count);
        case GGUF_DTYPE_Q5_0: return dequant_q5_0(data, out, count);
        case GGUF_DTYPE_Q5_1: return dequant_q5_1(data, out, count);
        case GGUF_DTYPE_Q8_0: return dequant_q8_0(data, out, count);
        case GGUF_DTYPE_Q8_1: {
            // Q8_1: fp16 d (scale) + fp16 s (unused for dequant) + int8[32]
            for (int i = 0; i < count; i++) {
                int bi = i / 32, ei = i % 32;
                const uint8_t* blk = data + (size_t)bi * 36;
                float d = read_f16(blk);
                out[i] = (float)((int8_t)blk[4 + ei]) * d;
            }
            return true;
        }
        case GGUF_DTYPE_Q2_K: return dequant_q2_k(data, out, count);
        case GGUF_DTYPE_Q3_K: return dequant_q3_k(data, out, count);
        case GGUF_DTYPE_Q4_K: return dequant_q4_k(data, out, count);
        case GGUF_DTYPE_Q5_K: return dequant_q5_k(data, out, count);
        case GGUF_DTYPE_Q6_K: return dequant_q6_k(data, out, count);
        case GGUF_DTYPE_Q8_K: return dequant_q8_k(data, out, count);
        case GGUF_DTYPE_Q1_0_G128: {
            // Q1_0 binary: fp16 scale + sign bits (1 bit per element)
            for (int i = 0; i < count; i++) {
                int bi = i / 128, ei = i % 128;
                const uint8_t* blk = data + (size_t)bi * 18;
                float sc = read_f16(blk);
                const uint8_t* bits = blk + 2;
                out[i] = (bits[ei / 8] >> (ei % 8)) & 1 ? sc : -sc;
            }
            return true;
        }
        case GGUF_DTYPE_TQ2_0_G128: {
            // TQ2_0 ternary: fp16 scale + 2-bit codes (0=-s, 1=0, 2=+s, 3=0)
            for (int i = 0; i < count; i++) {
                int bi = i / 128, ei = i % 128;
                const uint8_t* blk = data + (size_t)bi * 34;
                float sc = read_f16(blk);
                uint8_t c = (blk[2 + ei/4] >> ((ei%4)*2)) & 3;
                if (c == 0) out[i] = -sc;
                else if (c == 2) out[i] = sc;
                else out[i] = 0.0f;
            }
            return true;
        }
        default: return false;
    }
}

GgufReader::~GgufReader() { if (f_) fclose(f_); }

std::string GgufReader::read_string() {
    uint64_t len = 0;
    if (fread(&len, 8, 1, f_) != 1) return {};
    static constexpr uint64_t MAX_STRING_LEN = 1ULL * 1024 * 1024;
    if (len > MAX_STRING_LEN) { fseeko(f_, (off_t)len, SEEK_CUR); return "truncated"; }
    std::string s(len, '\0');
    if (len > 0 && fread(&s[0], 1, len, f_) != len) return {};
    return s;
}

// GGUF KV value types: 0=u8 1=i8 2=u16 3=i16 4=u32 5=i32 6=f32 7=bool
// 8=string 9=array 10=u64 11=i64 12=f64
bool GgufReader::read_kv_value(uint32_t vtype, KV& out) {
    out.vtype = vtype;
    switch (vtype) {
        case 0: { uint8_t v; if (fread(&v, 1, 1, f_) != 1) return false; out.u = v; return true; }
        case 1: { int8_t v; if (fread(&v, 1, 1, f_) != 1) return false; out.u = (uint64_t)(int64_t)v; return true; }
        case 2: { uint16_t v; if (fread(&v, 2, 1, f_) != 1) return false; out.u = v; return true; }
        case 3: { int16_t v; if (fread(&v, 2, 1, f_) != 1) return false; out.u = (uint64_t)(int64_t)v; return true; }
        case 4: { uint32_t v; if (fread(&v, 4, 1, f_) != 1) return false; out.u = v; return true; }
        case 5: { int32_t v; if (fread(&v, 4, 1, f_) != 1) return false; out.u = (uint64_t)(int64_t)v; return true; }
        case 6: { float v; if (fread(&v, 4, 1, f_) != 1) return false; out.f = v; return true; }
        case 7: { uint8_t v; if (fread(&v, 1, 1, f_) != 1) return false; out.u = v; return true; }
        case 8: { out.s = read_string(); return true; }
        case 9: {
            uint32_t at; fread(&at, 4, 1, f_);
            uint64_t an; fread(&an, 8, 1, f_);
            static constexpr uint64_t MAX_ARRAY_COUNT = 1000000;
            if (an > MAX_ARRAY_COUNT) {
                for (uint64_t j = 0; j < an; j++) skip_kv_value(at);
                an = 0;
            }
            if (at == 8) {
                out.arr_str.resize(an);
                for (uint64_t j = 0; j < an; j++) out.arr_str[j] = read_string();
            } else {
                for (uint64_t j = 0; j < an; j++) skip_kv_value(at);
            }
            return true;
        }
        case 10: { uint64_t v; if (fread(&v, 8, 1, f_) != 1) return false; out.u = v; return true; }
        case 11: { int64_t v; if (fread(&v, 8, 1, f_) != 1) return false; out.u = (uint64_t)v; return true; }
        case 12: { double v; if (fread(&v, 8, 1, f_) != 1) return false; out.f = v; return true; }
        default: return false;
    }
}

void GgufReader::skip_kv_value(uint32_t vtype) {
    switch (vtype) {
        case 0: case 1: case 7: fseeko(f_, 1, SEEK_CUR); break;
        case 2: case 3: fseeko(f_, 2, SEEK_CUR); break;
        case 4: case 5: case 6: fseeko(f_, 4, SEEK_CUR); break;
        case 8: { read_string(); break; }
        case 9: {
            uint32_t at; fread(&at, 4, 1, f_);
            uint64_t an; fread(&an, 8, 1, f_);
            static constexpr uint64_t MAX_ARRAY_COUNT = 1000000;
            if (an > MAX_ARRAY_COUNT) an = 0;
            if (at == 8) { for (uint64_t j = 0; j < an; j++) read_string(); }
            else { for (uint64_t j = 0; j < an; j++) skip_kv_value(at); }
            break;
        }
        case 10: case 11: case 12: fseeko(f_, 8, SEEK_CUR); break;
        default: break;
    }
}

bool GgufReader::open(const std::string& path) {
    f_ = fopen(path.c_str(), "rb");
    if (!f_) return false;
    char magic[4];
    if (fread(magic, 1, 4, f_) != 4 || memcmp(magic, "GGUF", 4) != 0) { fclose(f_); f_ = nullptr; return false; }
    uint32_t version; if (fread(&version, 4, 1, f_) != 1) { fclose(f_); f_ = nullptr; return false; }
    if (version != 2 && version != 3) { fclose(f_); f_ = nullptr; return false; }
    uint64_t tensor_count, kv_count;
    if (fread(&tensor_count, 8, 1, f_) != 1 || fread(&kv_count, 8, 1, f_) != 1) { fclose(f_); f_ = nullptr; return false; }
    static constexpr uint64_t MAX_TENSOR_COUNT = 200000, MAX_KV_COUNT = 200000;
    if (tensor_count > MAX_TENSOR_COUNT || kv_count > MAX_KV_COUNT) { fclose(f_); f_ = nullptr; return false; }

    for (uint64_t i = 0; i < kv_count; i++) {
        std::string key = read_string();
        uint32_t vtype; fread(&vtype, 4, 1, f_);
        KV kv;
        if (!read_kv_value(vtype, kv)) { fclose(f_); f_ = nullptr; return false; }
        kv_[key] = std::move(kv);
    }
    if (auto it = kv_.find("general.architecture"); it != kv_.end() && it->second.vtype == 8) arch_ = it->second.s;

    uint64_t alignment = 32;
    { uint32_t a; if (get_u32("general.alignment", a) && a > 0) alignment = a; }

    static constexpr uint32_t MAX_NDIM = 16;
    static constexpr uint64_t MAX_DIM_SIZE = 1ULL << 24;
    for (uint64_t i = 0; i < tensor_count; i++) {
        std::string name = read_string();
        uint32_t ndim; fread(&ndim, 4, 1, f_);
        if (ndim > MAX_NDIM) { fclose(f_); f_ = nullptr; return false; }
        GgufTensorInfo ti;
        ti.shape.resize(ndim);
        for (uint32_t d = 0; d < ndim; d++) {
            fread(&ti.shape[d], 8, 1, f_);
            if (ti.shape[d] > MAX_DIM_SIZE) { fclose(f_); f_ = nullptr; return false; }
        }
        fread(&ti.dtype, 4, 1, f_);
        uint64_t rel_offset; fread(&rel_offset, 8, 1, f_);
        ti.numel = 1;
        for (auto s : ti.shape) ti.numel *= (s ? s : 1);
        ti.abs_offset = rel_offset; // fixed up to absolute after the loop
        tensor_order_.push_back(name);
        tensors_[name] = ti; // abs_offset placeholder for now
        // stash rel_offset in abs_offset temporarily; fixed up below
        tensors_[name].abs_offset = rel_offset;
    }

    uint64_t data_start = (uint64_t)ftell(f_);
    uint64_t rem = data_start % alignment;
    if (rem) data_start += alignment - rem;
    for (auto& name : tensor_order_) tensors_[name].abs_offset += data_start;

    return true;
}

bool GgufReader::has_tensor(const std::string& name) const { return tensors_.count(name) != 0; }

const GgufTensorInfo* GgufReader::tensor_info(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

bool GgufReader::get_u32(const std::string& key, uint32_t& out) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) return false;
    const KV& kv = it->second;
    if (kv.vtype <= 5 || kv.vtype == 7 || kv.vtype == 10 || kv.vtype == 11) { out = (uint32_t)kv.u; return true; }
    return false;
}

bool GgufReader::get_f32(const std::string& key, float& out) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) return false;
    const KV& kv = it->second;
    if (kv.vtype == 6 || kv.vtype == 12) { out = (float)kv.f; return true; }
    if (kv.vtype <= 5 || kv.vtype == 10 || kv.vtype == 11) { out = (float)(int64_t)kv.u; return true; }
    return false;
}

bool GgufReader::get_string(const std::string& key, std::string& out) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.vtype != 8) return false;
    out = it->second.s;
    return true;
}

bool GgufReader::get_string_array(const std::string& key, std::vector<std::string>& out) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.vtype != 9) return false;
    out = it->second.arr_str;
    return true;
}

std::vector<std::string> GgufReader::kv_keys() const {
    std::vector<std::string> keys;
    keys.reserve(kv_.size());
    for (auto& [k, v] : kv_) keys.push_back(k);
    return keys;
}

bool GgufReader::get_tensor_raw(const std::string& name, int block_size, int block_bytes,
                                 std::vector<uint8_t>& out, uint64_t* out_numel) {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !f_ || block_size <= 0 || block_bytes <= 0) return false;
    const GgufTensorInfo& ti = it->second;
    if (out_numel) *out_numel = ti.numel;
    uint64_t n_blocks = (ti.numel + block_size - 1) / block_size;
    out.resize(n_blocks * (uint64_t)block_bytes);
    fseeko(f_, (off_t)ti.abs_offset, SEEK_SET);
    return fread(out.data(), 1, out.size(), f_) == out.size();
}

bool GgufReader::get_tensor_f32(const std::string& name, std::vector<float>& out, size_t* out_n) {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !f_) return false;
    const GgufTensorInfo& ti = it->second;
    static constexpr uint64_t MAX_TENSOR_ELEMENTS = 1ULL << 30;
    if (ti.numel > MAX_TENSOR_ELEMENTS) return false;
    out.resize(ti.numel);
    if (out_n) *out_n = ti.numel;

    GgufBlockInfo bi = gguf_block_info(ti.dtype);
    if (bi.block_bytes <= 0) return false;
    fseeko(f_, (off_t)ti.abs_offset, SEEK_SET);
    uint64_t n_blocks = (ti.numel + bi.block_size - 1) / bi.block_size;
    std::vector<uint8_t> block_buf((size_t)bi.block_bytes);
    for (uint64_t b = 0; b < n_blocks; b++) {
        uint64_t start = b * bi.block_size;
        uint64_t count = std::min<uint64_t>(bi.block_size, ti.numel - start);
        if (fread(block_buf.data(), (size_t)bi.block_bytes, 1, f_) != 1) return false;
        if (!gguf_dequant(ti.dtype, block_buf.data(), out.data() + start, (int)count)) return false;
    }
    return true;
}
