#ifndef ENGINE_NPU_HELPER_H
#define ENGINE_NPU_HELPER_H

#include <stdint.h>
#include <stddef.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <cstdio>
#include <algorithm>

// ── NPU GEMM dimensions (torch2aie AI Engine block sizes) ──
static constexpr int XM = 128;          // Activation batch / tile rows
static constexpr int XK = 1024;         // Reduction dimension per tile
static constexpr int XN = 4096;         // Output columns per tile

// ATB (AI Engine Block) layout tile sizes
static constexpr int M_TILE = 128;
static constexpr int K_TILE = 8;
static constexpr int N_TILE = 8;
static constexpr int N_AIE_ROWS = 4;   // from comment: N_AIE_ROWS * M_TILE = 512

// XRT kernel argument group IDs
static constexpr int INSTR_GROUP_ID  = 1;
static constexpr int ACT_GROUP_ID    = 3;
static constexpr int WEIGHT_GROUP_ID = 4;
static constexpr int OUT_GROUP_ID    = 5;

// ── BF16 conversion (inline, matches engine/npu/src/npu_engine_universal.cpp) ──
inline float bf16_to_float(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f; memcpy(&f, &bits, 4); return f;
}
inline uint16_t float_to_bf16(float f) {
    uint32_t bits; memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}
inline float bf16_to_float_safe(uint16_t v) {
    // NaN/Inf guard: BF16 exponent 0x7F80 → float exponent 0xFF
    return (v & 0x7F80) == 0x7F80 ? 0.0f : bf16_to_float(v);
}

// ── Packed BFP16 weight format (for MAI AI Engine xclbin) ──
struct BfpPackedWeight {
    uint8_t* data;
    size_t   size;
    int      rows;   // K dimension
    int      cols;   // N dimension
    int      block_k;
    int      block_n;
};

// BFP16 weight packer — packs raw BF16 weights into the ATB-shuffled format
// that the NPU kernel consumes. The packing is column-major within each
// (block_k × block_n) tile, interleaved for the AI Engine data mover.
// NOTE: This is a placeholder. The real packer depends on the specific
//       xclbin's data layout and is provided via torch2aie's weight tools.
static inline BfpPackedWeight* pack_weight_bfp(
    const uint16_t* data, int rows, int cols,
    int block_k, int block_n)
{
    // For now: this is documented as requiring torch2aie's weight packer
    // to produce correctly-formatted BFP16 for the MAI AI Engine xclbin.
    // This stub creates a flat copy as a placeholder.
    (void)block_k; (void)block_n;
    BfpPackedWeight* pw = (BfpPackedWeight*)calloc(1, sizeof(BfpPackedWeight));
    if (!pw) { fprintf(stderr, "pack_weight_bfp: OOM\n"); return nullptr; }
    size_t bytes = (size_t)rows * cols * 2; // BF16 = 2 bytes per value
    pw->data = (uint8_t*)malloc(bytes);
    if (!pw->data) { free(pw); fprintf(stderr, "pack_weight_bfp: OOM\n"); return nullptr; }
    memcpy(pw->data, data, bytes);
    pw->size  = bytes;
    pw->rows  = rows;
    pw->cols  = cols;
    pw->block_k = block_k;
    pw->block_n = block_n;
    return pw;
}
static inline void free_packed_weight(BfpPackedWeight* pw) {
    if (pw) { free(pw->data); free(pw); }
}

// ── RMS Normalization (BF16) ──
static inline void rms_norm_bf16(uint16_t* out, const uint16_t* input,
                                  const uint16_t* weight, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) {
        float v = bf16_to_float_safe(input[i]);
        ss += (double)v * v;
    }
    float inv_rms = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) {
        float v = bf16_to_float_safe(input[i]) * inv_rms * bf16_to_float(weight[i]);
        out[i] = float_to_bf16(v);
    }
}

// ── Softmax (float) ──
static inline void softmax_f32(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / (float)sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

// ── SiLU (BF16) ──
static inline void silu_bf16(uint16_t* x, int n) {
    for (int i = 0; i < n; i++) {
        float v = bf16_to_float(x[i]);
        x[i] = float_to_bf16(v / (1.0f + expf(-v)));
    }
}

// ── Argmax ──
static inline int argmax_f32(const float* x, int n) {
    int mi = 0; float mv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mv) { mv = x[i]; mi = i; }
    return mi;
}

// ── Element-wise multiply (BF16) ──
static inline void elem_mul_bf16(uint16_t* out, const uint16_t* a,
                                  const uint16_t* b, int n) {
    for (int i = 0; i < n; i++)
        out[i] = float_to_bf16(bf16_to_float(a[i]) * bf16_to_float(b[i]));
}

// ── Element-wise add (BF16) ──
static inline void elem_add_bf16(uint16_t* out, const uint16_t* a,
                                  const uint16_t* b, int n) {
    for (int i = 0; i < n; i++)
        out[i] = float_to_bf16(bf16_to_float(a[i]) + bf16_to_float(b[i]));
}

#endif // ENGINE_NPU_HELPER_H
