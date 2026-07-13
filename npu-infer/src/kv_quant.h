#pragma once
#include <cstdint>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

static constexpr int KQ_GROUP = 32;
static constexpr int KQ_HD = 128;
static constexpr int KQ_NKV = 8;
static constexpr int KQ_ELEM = KQ_NKV * KQ_HD;        // 1024
static constexpr int KQ_GROUPS = KQ_ELEM / KQ_GROUP;   // 32
static constexpr int KQ_MAX_CTX = 4096;

struct KQScale { uint16_t scale, zp; }; // BF16

struct KVQuant {
    std::vector<uint8_t> k_data, v_data;  // packed INT4
    std::vector<KQScale> k_scales, v_scales;
    int n;
    KVQuant();
    void append(int pos, const float* K, const float* V);
    void dequant_K_all(int kvh, int nt, float* out) const;
    void dequant_V_all(int kvh, int nt, float* out) const;
    static void quantize_group(const float* s, uint8_t* p, KQScale* gs);
    static void dequant_group(const uint8_t* p, const KQScale& gs, float* d);
    static inline uint16_t f32bf16(float f) { uint32_t b; memcpy(&b,&f,4); return b>>16; }
    static inline float bf16f32(uint16_t v) { uint32_t b=v<<16; float f; memcpy(&f,&b,4); return f; }
};
