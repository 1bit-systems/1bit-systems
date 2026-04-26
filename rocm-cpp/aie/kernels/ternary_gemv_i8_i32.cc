//===- ternary_gemv_i8_i32.cc -----------------------------------*- C++ -*-===//
//
// First ternary GEMV kernel for AMD XDNA2 NPU2 (Strix Halo / RyzenAI-npu5).
//
// V0 scope (2026-04-25): scalar correctness over speed. The vectorized AIE2P
// MAC path will land in v1 once we have a bit-exact baseline pinned down.
//
// Inputs (per invocation, single-tile):
//   A_packed : [m, k_pack] uint8, k_pack = k / 4 — four 2-bit ternary trits
//              per byte, packed LSB-first:
//                  bits[1:0] → trit 0 (column k+0)
//                  bits[3:2] → trit 1 (column k+1)
//                  bits[5:4] → trit 2 (column k+2)
//                  bits[7:6] → trit 3 (column k+3)
//              Trit codes:
//                  0b00 → 0
//                  0b01 → +1
//                  0b10 → -1
//                  0b11 → reserved (decoded as 0; valid encoders never emit)
//   b        : [k]      int8  activations
//   c        : [m]      int32 accumulator (this kernel ADDS into c, mirroring
//              the matmul/matvec convention where the host inits c via zero())
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (c) 2026 Daniel <d1r7yman@gmail.com>

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

// Headers come from $MLIR_AIE_INSTALL_DIR/include (added via -I in the
// Makefile). The Peano build does not see the mlir-aie source tree.
#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>

// zero.cc is header-only (template definitions); it lives under the same
// include root as aie_kernel_utils.h.
#include "aie_kernels/aie2/zero.cc"

// Decode the i-th packed trit (i in [0..3]) of `byte` as int8 ∈ {-1, 0, +1}.
// Branchless: shift, mask, and convert via sign-extension of (lo - hi).
//   lo = bit0 of nibble  (== +1 contribution)
//   hi = bit1 of nibble  (== -1 contribution)
//   trit = lo - hi  ∈  { 0-0,  1-0,  0-1,  1-1 } = { 0, +1, -1, 0 }
// The 0b11 reserved code therefore decodes to 0, which is the safe default.
static inline int8_t unpack_trit(uint8_t byte, int i) {
  uint8_t shift = static_cast<uint8_t>(i * 2);
  uint8_t nibble = static_cast<uint8_t>((byte >> shift) & 0x3u);
  int8_t lo = static_cast<int8_t>(nibble & 0x1u);
  int8_t hi = static_cast<int8_t>((nibble >> 1) & 0x1u);
  return static_cast<int8_t>(lo - hi);
}

// Scalar ternary GEMV. M rows, K input columns; A is packed at K/4 bytes/row.
// Produces an additive update on c: c[row] += sum_k(trit(A[row,k]) * b[k]).
// The runtime sequence calls zero_*_i32(c) before the first call per output
// row so no internal init is required — matches the matvec_scalar contract.
template <int M, int K_PACK>
void ternary_gemv_scalar_i8_i32(uint8_t *__restrict A_packed,
                                int8_t *__restrict b,
                                int32_t *__restrict c) {
  static_assert(K_PACK > 0, "K_PACK must be > 0");
  event0();
  for (int row = 0; row < M; ++row) {
    int32_t running = 0;
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int kp = 0; kp < K_PACK; ++kp) {
      uint8_t byte = A_packed[row * K_PACK + kp];
      int k_base = kp * 4;
      // Unrolled by hand so the compiler can keep b loads in registers.
      int8_t t0 = unpack_trit(byte, 0);
      int8_t t1 = unpack_trit(byte, 1);
      int8_t t2 = unpack_trit(byte, 2);
      int8_t t3 = unpack_trit(byte, 3);
      running += static_cast<int32_t>(t0) * static_cast<int32_t>(b[k_base + 0]);
      running += static_cast<int32_t>(t1) * static_cast<int32_t>(b[k_base + 1]);
      running += static_cast<int32_t>(t2) * static_cast<int32_t>(b[k_base + 2]);
      running += static_cast<int32_t>(t3) * static_cast<int32_t>(b[k_base + 3]);
    }
    c[row] += running;
  }
  event1();
}

extern "C" {

// If the host build wants different inner-tile shapes it must redefine these
// at compile time via -DDIM_M=... -DDIM_K_PACK=... . Defaults are tuned for
// the v0 single-core M=64, K=64 (=> K_PACK=16) smoke configuration.
#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_K_PACK
#define DIM_K_PACK 16
#endif

// Symbol exposed to the MLIR-AIE design. The matrix_vector design's
// `external_func` declaration must match this name.
void ternary_gemv_scalar_u8packed_i8_i32(uint8_t *A_in, int8_t *b_in,
                                          int32_t *c_out) {
  ternary_gemv_scalar_i8_i32<DIM_M, DIM_K_PACK>(A_in, b_in, c_out);
}

// Reuse the stock zero.cc instantiation under a stable name so the design
// can link a single .o for both the gemv and its output reset.
void zero_scalar_i32_ternary(int32_t *c_out) {
  zero_scalar<int32_t, DIM_M, 1>(c_out);
}

} // extern "C"
