// xdna_ternary_gemv.h — libxrt wrapper for the v0 ternary GEMV xclbin.
//
// V0 scope (2026-04-25): single-tile ternary matrix-vector. The xclbin baked
// in by the build is M=64, K=64. Larger M/K need a fresh xclbin (the dims are
// compile-time constants in the AIE design and the kernel object).
//
// The NPU produces a per-row int32 accumulator. The wrapper applies per-row
// fp32 scales and a single uniform x_scale on the host CPU and casts to fp16.
// On-NPU fold of the scale is a v1 task (would need bf16/fp32 mac path).
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (c) 2026 Daniel <d1r7yman@gmail.com>

#ifndef ROCM_CPP_XDNA_TERNARY_GEMV_H
#define ROCM_CPP_XDNA_TERNARY_GEMV_H

#include <cstdint>
#include <memory>
#include <string>

namespace rocm_cpp {

// Ternary trit packing — must mirror the kernel-side decoder in
// aie/kernels/ternary_gemv_i8_i32.cc:unpack_trit:
//   bits[1:0] of byte → trit at column k+0
//   bits[3:2]         → trit at column k+1
//   bits[5:4]         → trit at column k+2
//   bits[7:6]         → trit at column k+3
//   nibble code: 00 → 0, 01 → +1, 10 → -1, 11 → reserved (decoded as 0)
class XdnaTernaryGemv {
public:
    // Loads the xclbin + companion `insts_*.txt` instruction stream.
    // Constructor probes the xclbin's first kernel name; mlir-aie pads
    // "MLIR_AIE" with a hash so prefix-match is sufficient.
    XdnaTernaryGemv(const std::string& xclbin_path);
    XdnaTernaryGemv(const std::string& xclbin_path,
                    const std::string& insts_path);
    ~XdnaTernaryGemv();

    XdnaTernaryGemv(const XdnaTernaryGemv&) = delete;
    XdnaTernaryGemv& operator=(const XdnaTernaryGemv&) = delete;

    // True after successful construction.
    bool is_loaded() const noexcept;

    // Resolved kernel name (e.g. "MLIR_AIE_<hash>"). Useful for diagnostics.
    const std::string& kernel_name() const noexcept;

    // Compile-time M/K baked into the loaded xclbin. Throw if mismatched.
    int M() const noexcept;
    int K() const noexcept;

    // Run one ternary GEMV.
    //   A_packed  : [M, K/4] uint8 packed weights
    //   x_i8      : [K]      int8 activations
    //   scales    : [M]      fp32 per-row scales
    //   x_scale   : scalar fp32 activation scale (Sherry/BitNet uniform)
    //   y_out     : [M]      fp16 output (= fp16(scales[m] * x_scale * acc))
    // Throws std::runtime_error on shape mismatch or libxrt error.
    void run(const uint8_t* A_packed,
             const int8_t* x_i8,
             const float* scales,
             float x_scale,
             _Float16* y_out,
             int M_in, int K_in);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rocm_cpp

#endif // ROCM_CPP_XDNA_TERNARY_GEMV_H
