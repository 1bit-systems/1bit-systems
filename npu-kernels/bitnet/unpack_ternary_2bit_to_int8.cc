//===- unpack_ternary_2bit_to_int8.cc ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, 1bit.systems / bong-water-water-bong.
//
// Stage 1 of the BitNet-1.58 NPU pipeline on AIE2P (npu2 / Strix Halo).
// Unpacks halo-ai 2-bit ternary packing to int8 values in {-1, 0, +1}.
//
// Packing contract (halo-ai .h1b v2 ternary rows):
//   Each byte contains 4 values, LSB-first:
//     code 0  -> -1
//     code 1  ->  0
//     code 2  -> +1
//     code 3  -> unused (treated as 0; reserved, not produced by packer)
//   K-contiguous: lanes along the K dimension are packed densely into bytes.
//
// Kernel contract:
//   Input:  uint8_t *packed           [num_bytes]   = (rows * ncols) / 4
//   Output: int8_t  *out              [rows * ncols] in {-1, 0, +1}
//   ncols % 256 == 0  (unroll-friendly; reported in README)
//   Rows are independent and processed in strict K-contiguous order.
//
// Vectorization notes (AIE2P):
//   - 512-bit SIMD store units -> aie::vector<int8_t, 64> per cycle nominally,
//     but we emit width-32 to match the aie::vector<int8_t, 32> that the
//     int8 MAC primitive consumes one row at a time in the 2x2_mmul tile.
//   - One packed byte expands to 4 int8 lanes. We process 64 packed bytes
//     (= 256 int8 outputs) per outer iteration: eight unrolled 32-lane stores.
//   - LUT via aie::lookup is declared in an `__chess__`-only fast path; the
//     default fast path uses shift+mask+subtract which is certain to compile
//     under Peano. If the box confirms aie::lookup<int8_t> exists on AIE2P,
//     swap to the LUT variant for a ~1.3x kernel speedup (see README).

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

using namespace aie;

// Width of one SIMD store. Matches the int8 A-operand lane count consumed by
// matmul_vectorized_8x8x8_i8_i32's inner tile (r=8, 2x2 expansion across K in
// chunks of 32 bytes). Changing this breaks downstream assumptions.
static constexpr unsigned kUnpackLanes = 32;

// Packed bytes per unroll iteration (8 stores of 32 lanes = 256 int8 outputs).
static constexpr unsigned kBytesPerIter = kUnpackLanes * 8 / 4;  // = 64

// --------------------------------------------------------------------------
// Scalar fallback, used for the ncols-remainder path and for correctness
// cross-checks on the host. Keep this identical-in-behavior to the vector
// path — the kernel ships both and the IRON glue picks the vector one.
// --------------------------------------------------------------------------
template <int NCOLS>
static inline void unpack_ternary_scalar(const uint8_t *__restrict packed,
                                         int8_t *__restrict out,
                                         int32_t rows) {
  event0();
  static_assert(NCOLS % 4 == 0,
                "NCOLS must be a multiple of 4 for 2-bit packing");

  const int32_t bytes_per_row = NCOLS / 4;
  for (int32_t r = 0; r < rows; r++) {
    const uint8_t *__restrict p = packed + r * bytes_per_row;
    int8_t *__restrict o = out + r * NCOLS;
    for (int32_t b = 0; b < bytes_per_row; b++) {
      const uint8_t byte = p[b];
      // Decode: code - 1 for codes in {0,1,2}; code 3 clamps to 0.
      // Branch-free: (code == 3) ? 0 : (code - 1).
      for (int lane = 0; lane < 4; lane++) {
        const uint8_t code = (byte >> (2 * lane)) & 0x3;
        const int8_t val = (code == 3) ? 0 : (int8_t)((int)code - 1);
        o[b * 4 + lane] = val;
      }
    }
  }
  event1();
}

// --------------------------------------------------------------------------
// Vectorized path: 32-lane SIMD unpack.
//
// Per packed byte B we need 4 output bytes:
//   lane 0: (B      ) & 0x3
//   lane 1: (B >> 2 ) & 0x3
//   lane 2: (B >> 4 ) & 0x3
//   lane 3: (B >> 6 ) & 0x3
// Then code -> ternary: v = (code == 3) ? 0 : code - 1.
//
// AIE-API strategy:
//   Load 8 packed bytes into aie::vector<uint8_t, 8>. Broadcast-replicate
//   each byte 4 times into a 32-lane vector, right-shift each lane by its
//   lane-local shift {0,2,4,6,0,2,4,6,...}, mask with 0x3, subtract 1,
//   then clamp codes that were 3 back to 0.
//
// The broadcast/shift step is expressed via aie::shuffle_up / aie::concat or
// equivalently via a precomputed aie::vector<uint8_t, 32> shift vector used
// with aie::shift_bytes. The cleanest portable form uses aie::shift_right +
// aie::broadcast. Peano vectorizes the per-lane shift-by-variable via the
// AIE2P v32i8 shift instruction.
// --------------------------------------------------------------------------
// Vectorized path — kept for future work, currently disabled.
// See TEMP note in unpack_ternary_i2_i8 for the two fixes needed before
// re-enabling. Wrap in #if to skip non-dependent name lookup on
// aie::shift_bytes (which doesn't exist on Peano for AIE2P).
#if 0
template <int NCOLS>
static inline void unpack_ternary_vectorized(const uint8_t *__restrict packed,
                                             int8_t *__restrict out,
                                             int32_t rows) {
  event0();
  static_assert(NCOLS % 256 == 0,
                "NCOLS must be a multiple of 256 for the unrolled path");

  const int32_t bytes_per_row = NCOLS / 4;

  // Per-lane shift amount, replicated across 4-byte groups:
  //   lanes  0..3  => {0, 2, 4, 6}
  //   lanes  4..7  => {0, 2, 4, 6}
  //   ...
  //   lanes 28..31 => {0, 2, 4, 6}
  alignas(32) static constexpr uint8_t shift_pat[kUnpackLanes] = {
      0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6,
      0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6,
  };
  const aie::vector<uint8_t, kUnpackLanes> shift_v =
      aie::load_v<kUnpackLanes>(shift_pat);

  const aie::vector<uint8_t, kUnpackLanes> mask_v =
      aie::broadcast<uint8_t, kUnpackLanes>((uint8_t)0x3);
  const aie::vector<uint8_t, kUnpackLanes> three_v =
      aie::broadcast<uint8_t, kUnpackLanes>((uint8_t)0x3);
  const aie::vector<int8_t, kUnpackLanes> one_v =
      aie::broadcast<int8_t, kUnpackLanes>((int8_t)1);

  for (int32_t r = 0; r < rows; r++) {
    const uint8_t *__restrict p = packed + r * bytes_per_row;
    int8_t *__restrict o = out + r * NCOLS;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int32_t b = 0; b < bytes_per_row; b += kBytesPerIter) {
      // kBytesPerIter = 64 packed -> 256 int8 out; 8 iterations of 32 lanes.
      AIE_LOOP_UNROLL(8)
      for (int chunk = 0; chunk < 8; chunk++) {
        // Load 8 packed bytes, each will fan out to 4 unpacked lanes -> 32.
        const uint8_t *__restrict p_chunk = p + b + chunk * 8;

        // Replicate each of 8 bytes 4x into a 32-lane vector.
        // aie::broadcast_to_v + concat-of-duplicates is the portable form;
        // equivalent to the AIE shuffle {0,0,0,0,1,1,1,1,...,7,7,7,7}.
        // We express it via 8 broadcasts of a scalar load, then concat-pair.
        aie::vector<uint8_t, kUnpackLanes> rep;
        // scalar-broadcast path; Peano fuses this with the following shift.
        for (int j = 0; j < 8; j++) {
          const uint8_t bj = p_chunk[j];
          rep[4 * j + 0] = bj;
          rep[4 * j + 1] = bj;
          rep[4 * j + 2] = bj;
          rep[4 * j + 3] = bj;
        }

        // code = (rep >> shift_v) & 0x3   — per-lane variable shift.
        aie::vector<uint8_t, kUnpackLanes> shifted =
            aie::shift_bytes(rep, shift_v);  // see note below
        aie::vector<uint8_t, kUnpackLanes> code = aie::bit_and(shifted, mask_v);

        // signed = int8(code) - 1          — maps {0,1,2,3} -> {-1,0,1,2}
        // Reinterpret uint8 as int8; values 0..3 are safe under two's
        // complement (no std::bit_cast on AIE core, use aie::vector_cast).
        aie::vector<int8_t, kUnpackLanes> sval =
            aie::vector_cast<int8_t>(code);
        aie::vector<int8_t, kUnpackLanes> signed_v = aie::sub(sval, one_v);

        // Clamp unused-code lanes (code == 3) back to 0.
        // mask_three is 1 where code==3, else 0.
        aie::mask<kUnpackLanes> is_three = aie::eq(code, three_v);
        aie::vector<int8_t, kUnpackLanes> out_v =
            aie::select(signed_v, (int8_t)0, is_three);

        aie::store_v(o + b * 4 + chunk * kUnpackLanes, out_v);
      }
    }
  }
  event1();
}

#endif  // disabled vectorized path

// NOTE on aie::shift_bytes:
//   Confirmed on box 2026-04-23: `aie::shift_bytes` does NOT exist in
//   Peano's aie-api for AIE2P. The real intrinsic is the chess builtin
//   `shift_bytes(vec_a, vec_b, byte_count)` in the global namespace, but
//   that shifts concatenated vectors by a byte count — not per-lane
//   variable shift. For the broadcast-then-variable-shift pattern we
//   actually want, write as 4 fixed-shift vectors + interleave, OR
//   stay scalar. Deferred to follow-up; current build uses scalar path.

extern "C" {

// Kernel matches halo-ai .h1b layout: row-major ternary, K-contiguous.
// DIM_M / DIM_K come from the Makefile (-DDIM_M=..., -DDIM_K=...), matching
// the naming convention used by mm.cc.
#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_K
#define DIM_K 64
#endif

// One tile's worth of weights, unpacked per call. rows = m (tile rows),
// ncols = k (tile K). Called once per K-tile per M-tile — amortized across
// the matmul stage.
//
// TEMP: routed to scalar path until the vectorized unrolled variant gets
// two fixes validated on box — (1) drop aie::shift_bytes (doesn't exist
// on Peano for AIE2P; the variable-per-lane u8 shift needs a different
// primitive — candidate: fold into a 4-way fixed-shift + interleave),
// (2) generalise the unrolled path to handle DIM_K < 256 (currently the
// static_assert + outer-loop granularity assume ≥256).
void unpack_ternary_i2_i8(uint8_t *__restrict packed,
                          int8_t *__restrict out) {
  unpack_ternary_scalar<DIM_K>(packed, out, DIM_M);
}

// Scalar reference (scalar_ prefix matches zero.cc's scalar_/vector_ split).
void unpack_ternary_scalar_i2_i8(uint8_t *__restrict packed,
                                 int8_t *__restrict out) {
  unpack_ternary_scalar<DIM_K>(packed, out, DIM_M);
}

}  // extern "C"
