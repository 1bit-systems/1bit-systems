// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// mm_ternary_32x64x128 — native ternary AIE2P matrix multiply kernel
//
// 4× memory density: 2-bit packed ternary weights vs 8-bit INT8.
// Decodes on-the-fly and multiplies against BF16 activations.
//
// Dimensions:  M=32 output rows, K_packed=512 (→2048 ternary), N=128
// Block structure: K is divided into blocks of 256 ternary values,
//   each block has its own BF16 scale (8 scales per row for K=2048).
//
// Weight format (packed uint8):
//   Each byte holds 4 ternary weights in 2-bit fields:
//     bits[0:1]=val0, bits[2:3]=val1, bits[4:5]=val2, bits[6:7]=val3
//   Mapping: 00 → -1.0,  01 → 0.0,  10 → +1.0,  11 → -1.0
//
// Memory layout (all in one flat i32 buffer):
//   [0..WEIGHT_BYTES):          packed weights [M][K/4] uint8
//   [WEIGHT_BYTES..BLK_SC_OFF): per-block scales [M][n_blocks] bf16
//   [BLK_SC_OFF..END):          activations [K*4] bf16
//
// Output: [num_rows] bf16 (accumulated per N-slice)

#include <aie_api/aie.hpp>
#include <stdint.h>

// ── Compile-time dimensions ─────────────────────────
#ifndef DIM_M
#define DIM_M 32
#endif
#ifndef DIM_K_PACKED
#define DIM_K_PACKED 512    // packed bytes → 2048 ternary
#endif
#ifndef DIM_N
#define DIM_N 128
#endif

// ── Derived constants ──────────────────────────────
constexpr int32_t kM = DIM_M;
constexpr int32_t kKPacked = DIM_K_PACKED;
constexpr int32_t kN = DIM_N;

constexpr int32_t kVLen = 32;               // AIE2P vector width
constexpr int32_t kBytesPerIter = 8;        // 8 bytes → 32 ternary per iter
constexpr int32_t kInnerIters = kKPacked / kBytesPerIter;  // 512/8 = 64
constexpr int32_t kTernaryTotal = kKPacked * 4;            // 2048

// Block: 256 ternary values = 64 packed bytes = 8 inner iters
constexpr int32_t kBlockIters = 8;           // iters per block
constexpr int32_t kTernaryPerBlock = kBlockIters * kVLen;  // 256
constexpr int32_t kPackedPerBlock = kTernaryPerBlock / 4;   // 64 bytes
constexpr int32_t kNumBlocks = kInnerIters / kBlockIters;   // 64/8 = 8

// Buffer offsets
constexpr int32_t kWeightBytes = kM * kKPacked;            // M*512
constexpr int32_t kScaleOffset = kWeightBytes;
constexpr int32_t kScaleBytes = kM * kNumBlocks * 2;       // M*8*2 bf16
constexpr int32_t kActOffset = kScaleOffset + kScaleBytes;
constexpr int32_t kActBytes = kTernaryTotal * 2;           // 2048*2

// ── Decode 8 packed bytes → 32 BF16 ternary values ──
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

// ── Accumulate one block (256 ternary, 8 iters) ─────
__attribute__((always_inline)) static inline aie::accum<accfloat, kVLen>
accumulate_one_block(const uint8_t *__restrict weight_row,
                     const bfloat16 *__restrict activation) {
  aie::accum<accfloat, kVLen> acc = aie::zeros<accfloat, kVLen>();
  for (int32_t iter = 0; iter < kBlockIters; iter++) {
    int32_t byte_off = iter * kBytesPerIter;
    int32_t act_off  = iter * kVLen;
    aie::vector<bfloat16, kVLen> tern = decode_ternary_8bytes(weight_row + byte_off);
    aie::vector<bfloat16, kVLen> act  = aie::load_v<kVLen>(activation + act_off);
    auto prod = aie::mul(tern, act);
    acc = aie::add(acc, prod);
  }
  return acc;
}

// ── Compute one row (all blocks, per-block scales) ──
__attribute__((always_inline)) static inline bfloat16
compute_one_row(const uint8_t *__restrict weight_row,
                const bfloat16 *__restrict activation,
                const bfloat16 *__restrict block_scales) {
  float total = 0.0f;

  for (int32_t blk = 0; blk < kNumBlocks; blk++) {
    // Accumulate this block
    aie::accum<accfloat, kVLen> acc = accumulate_one_block(
        weight_row + blk * kPackedPerBlock,
        activation + blk * kTernaryPerBlock);

    // Convert to BF16, apply per-block scale, reduce-add
    aie::vector<bfloat16, kVLen> acc_vec = aie::to_vector<bfloat16>(acc);
    aie::vector<bfloat16, kVLen> scale_vec = aie::broadcast<bfloat16, kVLen>(block_scales[blk]);
    auto scaled = aie::mul(acc_vec, scale_vec);
    aie::vector<bfloat16, kVLen> scaled_bf16 = aie::to_vector<bfloat16>(scaled);

    // Reduce-add: sum the 32 BF16 values to a single scalar
    total += (float)aie::reduce_add(scaled_bf16);
  }

  return bfloat16(total);
}

// ── Main kernel entry point ──────────────────────────
//
// input:  [weights (ui8)] [block_scales (bf16)] [activations (bf16)]
//         Buffer sized for DIM_M rows (compile-time).
// output: [num_rows] bf16
//
// row_start, num_rows: process slice [row_start, row_start+num_rows).

extern "C" {

void mm_ternary_32x64x128(int32_t *__restrict input,
                          bfloat16 *__restrict output,
                          int32_t row_start,
                          int32_t num_rows) {
  const uint8_t  *weights      = reinterpret_cast<const uint8_t *>(input);
  const bfloat16 *block_scales = reinterpret_cast<const bfloat16 *>(input + kScaleOffset / 4);
  const bfloat16 *acts         = reinterpret_cast<const bfloat16 *>(input + kActOffset / 4);

  for (int32_t i = 0; i < num_rows; i++) {
    output[i] = 0.0f;
  }

  for (int32_t ri = 0; ri < num_rows; ri++) {
    int32_t row = row_start + ri;
    if (row >= kM) break;

    const uint8_t *wt_row = weights + row * kKPacked;
    const bfloat16 *blk_sc = block_scales + row * kNumBlocks;

    output[ri] += compute_one_row(wt_row, acts, blk_sc);
  }
}

// ── Scalar reference for validation ──────────────────
void mm_ternary_reference(const uint8_t *__restrict packed_weights,
                          const bfloat16 *__restrict block_scales,
                          const bfloat16 *__restrict activation,
                          bfloat16 *__restrict output,
                          int32_t m, int32_t k_packed, int32_t n,
                          int32_t row_start, int32_t num_rows) {
  int32_t num_blocks = k_packed / 64;  // 64 packed bytes per block (256 ternary = 64 bytes)
  for (int32_t ri = 0; ri < num_rows; ri++) {
    int32_t row = row_start + ri;
    float acc = 0.0f;
    const uint8_t *wt_row = packed_weights + row * k_packed;

    for (int32_t blk = 0; blk < num_blocks; blk++) {
      float blk_acc = 0.0f;
      for (int32_t i = 0; i < 64; i++) {
        uint8_t packed = wt_row[blk * 64 + i];
        for (int32_t b = 0; b < 4; b++) {
          uint8_t code = (packed >> (b * 2)) & 3;
          float t = (code == 2) ? 1.0f : (code == 1) ? 0.0f : -1.0f;
          blk_acc += t * (float)activation[blk * 256 + i * 4 + b];
        }
      }
      acc += blk_acc * (float)block_scales[row * num_blocks + blk];
    }

    output[ri] += (bfloat16)acc;
  }
}

} // extern "C"
