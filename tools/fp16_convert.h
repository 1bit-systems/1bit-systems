// Proper float32 → fp16 conversion
static inline uint16_t float_to_half(float val) {
    uint32_t bits;
    memcpy(&bits, &val, 4);
    uint16_t sign = (bits >> 16) & 0x8000;
    int exp = ((bits >> 23) & 0xFF) - 127 + 15;  // FP32 exp to FP16 exp bias
    uint16_t mant = (bits >> 13) & 0x03FF;        // Top 10 bits of 23-bit mantissa
    
    if (exp <= 0) {
        // Denormal/zero in FP32 — flush to zero in FP16
        return sign;
    } else if (exp >= 31) {
        // Infinity or NaN
        return sign | 0x7C00 | (mant ? 0x0200 : 0);  // Preserve NaN sign
    }
    return sign | ((uint16_t)exp << 10) | mant;
}
