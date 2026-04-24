// bitnet_gemv_aie2p.cc — tile-local AIE2P C++ entry for BitNet-1.58 GEMV.
//
// Status (2026-04-24): SKELETON ONLY. This file is the single-file bring-up
// seat for the ternary-decode GEMV on the XDNA2 NPU. It pins the tile-side
// entry signature and documents the three-stage pipeline we wrap, but the
// vectorized body is stubbed.  The real per-stage implementations live in:
//
//   ../../../npu-kernels/bitnet/unpack_ternary_2bit_to_int8.cc
//   /home/bcloud/repos/mlir-aie/aie_kernels/aie2p/mm.cc (stock upstream,
//       matmul_vectorized_8x8x8_i8_i32 — we LINK against this, do not ship
//       a copy)
//   ../../../npu-kernels/bitnet/scale_i32_fp16.cc
//
// The earlier "fused" design at ../halo_ternary_mm.{h,cpp} is kept as a
// parity oracle — its scalar reference matmul is the bit-exact target for
// the pipelined path below. Once the pipelined kernel lands and passes
// parity vs halo_ternary_mm_core on a small M/N/K sweep, we retire the
// fused design.
//
// Compile site: Peano clang, target aie2p-none-unknown-elf.
//   /opt/peano/bin/clang --target=aie2p-none-unknown-elf -O2 \
//       -c bitnet_gemv_aie2p.cc -o bitnet_gemv_aie2p.o
//
// The xclbin packer (IRON + aiecc) wires this object into an ObjectFifo
// graph; see ../scripts/build_aie.sh for the build wrapper.

#include <stdint.h>

// The entry symbol the xclbin packer will expose as `bitnet_gemv_core`.
// Public C ABI so the MLIR-AIE wrapper can reference it by name without
// name-mangling surprises. The argument list mirrors the tile-memory
// contract documented in docs/wiki/NPU-Kernel-Handoff.md §"Memory layout":
//
//   packed_W : uint8_t*    — ternary weights, 2 bits/elem, 4 elems/byte.
//                            Packed layout matches the halo v2 on-disk
//                            order (row-major, K-inner); for BitNet-1.58
//                            decode this is M rows × (K/4) bytes.
//   x        : int8_t*     — unquantized activations, pre-scaled + clamped
//                            host-side to int8. For decode N=1 is padded
//                            up to N=16 (tile constraint).
//   scales   : int16_t*    — per-row bf16 scales, cast to int16 so the
//                            C ABI stays integer-typed. One code per output
//                            row. Host-side loader converts halo fp16 to
//                            bf16 before dispatch (see README).
//   out      : int16_t*    — per-output-row bf16 result, stride N.
//   M, K, N  : int32_t     — logical dims. Enforced: M%16==0, K%8==0,
//                            K%4==0 (unpack), N%16==0.
//
// This function assumes all four buffers are already resident in the L1
// ping-pong banks — the host xclbin drives the L3 -> L2 -> L1 DMAs and
// scheduling. No L2/L3 pointers cross this boundary.
extern "C" void bitnet_gemv_core(const uint8_t*  packed_W,
                                 const int8_t*   x,
                                 const int16_t*  scales,
                                 int16_t*        out,
                                 int32_t M, int32_t K, int32_t N);

// ---------------------------------------------------------------------------
// Stub body. Real implementation lands in a follow-up — see the effort
// breakdown in docs/wiki/Roadmap.md §v0.1.2 and the handoff doc's FFI
// section for the per-stage decomposition.
//
// The stub is a *scalar reference* matmul that compiles on AIE2P without
// needing aie_api/aie.hpp. It is NOT efficient, it is NOT tile-aware, and
// it does NOT exploit the 8x8x8 mmul unit. Its role is twofold:
//   (1) give the xclbin build something real to link while the pipelined
//       path is in flight, so the rest of the toolchain (packer, ND-DMA
//       descriptors, host xrt::bo plumbing) can be shaken out in parallel.
//   (2) serve as a bit-exact parity oracle for the pipelined path during
//       bring-up — feed identical inputs, diff the int32 accumulator before
//       the bf16 scale step, verify max_abs_err == 0.
//
// When the pipelined path lands, the stub body is replaced with a call
// chain to the three micro-kernels (unpack → stock mm → scale) driven by
// ObjectFifo worker placement from the IRON emitter
// (npu-kernels/bitnet/bitnet_gemv.py).
// ---------------------------------------------------------------------------

extern "C" void bitnet_gemv_core(const uint8_t*  packed_W,
                                 const int8_t*   x,
                                 const int16_t*  scales,
                                 int16_t*        out,
                                 int32_t M, int32_t K, int32_t N)
{
    // Shape checks at the entry — cheap, and catch requantizer-side mis-packs
    // before we touch any L1 scratch. Constants match mm.cc static_asserts.
    // (On real hardware we'd surface these via an xrt::error return value;
    // for now an invalid-shape early-exit is the least-bad diagnostic.)
    if ((M % 16) != 0) return;
    if ((K %  8) != 0) return;
    if ((K %  4) != 0) return;  // 2-bit packing: 4 codes/byte.
    if ((N % 16) != 0) return;

    // Unused-arg silencer — all five pointers are contractually valid by
    // the time the host dispatches, but the scalar stub doesn't touch them
    // yet. Real body goes here.
    (void) packed_W;
    (void) x;
    (void) scales;
    (void) out;
}

// Note: no `int main` and no `aie::` imports on purpose. The AIE tile
// runtime scheduler invokes `bitnet_gemv_core` once per ND-DMA descriptor
// dispatch from the host xclbin; there is no tile-local entry point.
