// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// mm_ternary_32x64x128 — native ternary AIE2P matrix multiply kernel
//
// 4× memory density: 2-bit packed ternary weights vs 8-bit INT8.
// Decodes on-the-fly and multiplies against BF16 activations.
//
// Dimensions:  M=32 output rows, K=64 (→ 256 ternary), N=128 output cols
//
// Weight format (packed uint8):
//   Each byte holds 4 ternary weights in 2-bit fields:
//     bits[0:1]=val0, bits[2:3]=val1, bits[4:5]=val2, bits[6:7]=val3
//   Mapping: 00 → -1.0,  01 → 0.0,  10 → +1.0,  11 → -1.0
//
// Memory layout (all in one input buffer):
//   [0..WEIGHT_BYTES):         packed weights [M][K/4] uint8
//   [WEIGHT_BYTES..SCALE_OFF): scales [M] bf16
//   [SCALE_OFF..END):          activations [K*4] bf16
//
// Output: [M] bf16 (accumulated per N-slice)

#include <aie_api/aie.hpp>
#include <stdint.h>

// ── Compile-time dimensions (overridable via -D flags) ─────────
#ifndef DIM_M
#define DIM_M 32
#endif
#ifndef DIM_K_PACKED
#define DIM_K_PACKED 64    // packed bytes per tile: 64 bytes → 256 ternary
#endif
#ifndef DIM_N
#define DIM_N 128
#endif

// ── Derived constants ──────────────────────────────────────────
constexpr int32_t kM = DIM_M;                 // 32 output rows
constexpr int32_t kKPacked = DIM_K_PACKED;    // 64 packed bytes → 256 ternary
constexpr int32_t kN = DIM_N;                 // 128 output columns per tile

// AIE2P vector width for BF16 operations
constexpr int32_t kVLen = 32;

// Each inner iteration processes 8 bytes → 32 ternary values
constexpr int32_t kBytesPerIter = 8;
constexpr int32_t kInnerIters = kKPacked / kBytesPerIter;  // 64/8 = 8
constexpr int32_t kTernaryTotal = kKPacked * 4;             // 256

// Buffer offset calculations
constexpr int32_t kWeightBytes = kM * kKPacked;           // M*64 bytes
constexpr int32_t kScaleOffset = kWeightBytes;            // scales start after weights
constexpr int32_t kScaleBytes = kM * 2;                   // M bf16 = M*2 bytes
constexpr int32_t kActOffset = kScaleOffset + kScaleBytes;
constexpr int32_t kActBytes = kTernaryTotal * 2;          // 256*2 bytes

// ── Decode 8 packed bytes → 32 BF16 ternary values ────────────
__attribute__((always_inline)) static inline aie::vector<bfloat16, kVLen>
decode_ternary_8bytes(const uint8_t *__restrict packed) {
  aie::vector<bfloat16, kVLen> result;

  for (int32_t i = 0; i < 8; i++) {
    uint8_t byte = packed[i];

    for (int32_t b = 0; b < 4; b++) {
      uint8_t code = (byte >> (b * 2)) & 3;
      bfloat16 val;
      if (code == 2)       val = bfloat16(1.0f);
      else if (code == 1)  val = bfloat16(0.0f);
      else                 val = bfloat16(-1.0f);

      result[i * 4 + b] = val;
    }
  }

  return result;
}

// ── Accumulate over full K for one output row ──────────────────
__attribute__((always_inline)) static inline aie::vector<bfloat16, kVLen>
compute_one_row(const uint8_t *__restrict weight_row,
                const bfloat16 *__restrict activation,
                bfloat16 scale) {
  aie::accum<accfloat, kVLen> acc = aie::zeros<accfloat, kVLen>();

  // 8 iterations, each handling 8 packed bytes → 32 ternary values
  for (int32_t iter = 0; iter < kInnerIters; iter++) {
    int32_t byte_offset = iter * kBytesPerIter;
    int32_t act_offset  = iter * kVLen;

    aie::vector<bfloat16, kVLen> tern = decode_ternary_8bytes(weight_row + byte_offset);
    aie::vector<bfloat16, kVLen> act  = aie::load_v<kVLen>(activation + act_offset);

    auto prod = aie::mul(tern, act);
    acc = aie::add(acc, prod);
  }

  // Apply per-row scale: convert accum→vector, scale, convert back
  aie::vector<bfloat16, kVLen> acc_vec   = aie::to_vector<bfloat16>(acc);
  aie::vector<bfloat16, kVLen> scale_vec = aie::broadcast<bfloat16, kVLen>(scale);
  auto scaled_acc = aie::mul(acc_vec, scale_vec);
  return aie::to_vector<bfloat16>(scaled_acc);
}

// ── Main kernel entry point ─────────────────────────────────────
//
// All data lives in a single flat i32 buffer.
// The kernel casts sub-regions to appropriate types.
//
// input:  [weights (ui8)] [scales (bf16)] [activations (bf16)]
//         Buffer sized for DIM_M rows (compile-time).
// output: [num_rows] bf16 (0-indexed, independent of row_start)
//
// row_start, num_rows: process only rows [row_start, row_start+num_rows).
//   Multi-core tiling: all cores in a row share one broadcast buffer
//   (containing all M rows), each picks its slice via row_start.
//   Single-core use: row_start=0, num_rows=DIM_M.
//
// The caller tiles over N_cols externally; this processes one N-slice.

extern "C" {

void mm_ternary_32x64x128(int32_t *__restrict input,
                          bfloat16 *__restrict output) {
  // Cast sub-regions
  const uint8_t  *weights = reinterpret_cast<const uint8_t *>(input);
  const bfloat16 *scales  = reinterpret_cast<const bfloat16 *>(input + kScaleOffset / 4);
  const bfloat16 *acts    = reinterpret_cast<const bfloat16 *>(input + kActOffset / 4);

  // Zero output
  for (int32_t i = 0; i < kM; i++) {
    output[i] = 0.0f;
  }

  // Process each output row
  for (int32_t row = 0; row < kM; row++) {
    const uint8_t *wt_row = weights + row * kKPacked;
    bfloat16 scale = scales[row];

    aie::vector<bfloat16, kVLen> row_result = compute_one_row(wt_row, acts, scale);
    output[row] += aie::reduce_add(row_result);
  }
}

// ── Scalar reference for validation ────────────────────────────
void mm_ternary_reference(const uint8_t *__restrict packed_weights,
                          const bfloat16 *__restrict scales,
                          const bfloat16 *__restrict activation,
                          bfloat16 *__restrict output,
                          int32_t m, int32_t k_packed, int32_t n) {
  for (int32_t row = 0; row < m; row++) {
    float acc = 0.0f;
    const uint8_t *wt_row = packed_weights + row * k_packed;
    bfloat16 scale = scales[row];

    for (int32_t i = 0; i < k_packed; i++) {
      uint8_t packed = wt_row[i];
      for (int32_t b = 0; b < 4; b++) {
        uint8_t code = (packed >> (b * 2)) & 3;
        float ternary;
        if (code == 2)       ternary = 1.0f;
        else if (code == 1)  ternary = 0.0f;
        else                 ternary = -1.0f;
        acc += ternary * (float)activation[i * 4 + b];
      }
    }

    output[row] += (bfloat16)(acc * (float)scale);
  }
}

} // extern "C"
