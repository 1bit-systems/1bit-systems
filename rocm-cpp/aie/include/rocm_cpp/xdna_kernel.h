// xdna_kernel.h — thin C++ wrapper around libxrt for AIE2P (XDNA2) NPU kernels.
//
// V0 scope (2026-04-25): prove an in-process libxrt dispatch path. Loads an
// MLIR-AIE-compiled .xclbin + its instruction stream, reuses host-only BOs
// across calls, and exposes a single-shape int8×int8→int32 matmul entry.
//
// The shape and tile sizes are baked into the xclbin at compile time (see
// `mlir-aie/programming_examples/basic/matrix_multiplication/single_core/`).
// V0 hard-codes the M=K=N=512 / 64×64×64-tile build because that's the
// stock-mlir-aie xclbin we have on disk; runtime M/K/N must match. A future
// generic `dispatch()` taking arbitrary BOs will land alongside our own
// authored ternary GEMM kernel — see Impl::matmul_512_i8_i32 for the contract.
//
// Design notes:
//   - libxrt's xrt::* objects are pure RAII so we hide them behind Impl to keep
//     this header free of /usr/include/xrt/* (and therefore safe to include
//     from non-XRT callers behind LEMON_HAS_XDNA / equivalent guards).
//   - Construction is heavyweight (load_axlf + register_xclbin + hw_context +
//     BO allocation). Designed to be done ONCE per loaded model, after which
//     matmul_512_i8_i32() is just memcpy + sync + run.wait.
//   - All errors surface as std::runtime_error.
//   - Not thread-safe: one in-flight call per instance. Caller must serialize.

#ifndef ROCM_CPP_XDNA_KERNEL_H
#define ROCM_CPP_XDNA_KERNEL_H

#include <cstdint>
#include <memory>
#include <string>

namespace rocm_cpp {

class XdnaKernel {
public:
    // Load `xclbin_path` and resolve the kernel by name-prefix match
    // (mlir-aie xclbins typically expose a single kernel named "MLIR_AIE").
    // Pass an empty string to grab the first kernel in the xclbin.
    //
    // For the M=K=N=512 / m=k=n=64 stock matmul, the companion instruction
    // stream is expected at <xclbin without ".xclbin"> -> swap "final_" for
    // "insts_" and use ".txt" — i.e. for
    //   .../build/final_512x512x512_64x64x64.xclbin
    // we load
    //   .../build/insts_512x512x512_64x64x64.txt
    // The constructor derives the path itself; pass an explicit override via
    // the second overload if your build directory deviates.
    XdnaKernel(const std::string& xclbin_path, const std::string& kernel_name);
    XdnaKernel(const std::string& xclbin_path,
               const std::string& kernel_name,
               const std::string& insts_path);
    ~XdnaKernel();

    XdnaKernel(const XdnaKernel&) = delete;
    XdnaKernel& operator=(const XdnaKernel&) = delete;

    // Returns true once the xclbin has been registered and BOs are allocated.
    // Always true after a successful constructor; false only if a future
    // lazy-load mode is added.
    bool is_loaded() const noexcept;

    // Resolved kernel name as reported by the xclbin (after optional prefix
    // match). Useful for logs.
    const std::string& kernel_name() const noexcept;

    // Single fixed-shape matmul: A[M,K] × B[K,N] -> C[M,N], int8 × int8 -> int32,
    // with M = K = N = 512 baked into the loaded xclbin. The pointers must
    // address contiguous row-major buffers of the right size; we memcpy in,
    // sync-to-device, run, sync-from-device, memcpy out. Throws if the shape
    // doesn't match what the xclbin was compiled for.
    void matmul_512_i8_i32(const int8_t* A_host,
                           const int8_t* B_host,
                           int32_t* C_host);

    // Generic shape-checked matmul wrapper. v0 only accepts M=K=N=512;
    // anything else throws "shape not supported by loaded xclbin (compiled
    // for ...)". Kept extensible for the day we ship a parameterized
    // dispatch path or multiple xclbins in one wrapper.
    void matmul_i8_i32(const int8_t* A_host,
                       const int8_t* B_host,
                       int32_t* C_host,
                       int M, int K, int N);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rocm_cpp

#endif // ROCM_CPP_XDNA_KERNEL_H
