// engine_npu_dispatch.h — C-ABI bridge between Engine::ternary_gemv (C++17,
// HIP-compiled) and onebit::aie::BitnetGemmAIE2P (C++23, libxrt direct-link).
//
// Why a C-ABI shim?  engine.cpp lives in the rocm-cpp/ subtree which pins
// CMAKE_CXX_STANDARD=17 (HIP kernel TUs share the same standard).  The AIE
// wrapper exposes std::expected (C++23) on its public surface.  Rather than
// drag the whole rocm-cpp tree to C++23 — which cascades through CK headers
// and TheRock's clang HIP frontend — we isolate the C++23 island in a single
// TU (engine_npu_dispatch.cpp, compiled with cxx_std_23) and present a plain
// C interface to engine.cpp.
//
// The whole header is gated on ROCM_CPP_HAVE_XDNA.  When XDNA is OFF this
// file should not be included; CMake won't compile the .cpp side either.
//
// Phase-2 contract (per project_npu_bitnet_gemm_authored.md):
//   * AIE2P kernel ingests bf16 acts (M*K) + HALO_V2-packed ternary weights
//     (K*N/16 u32) and emits bf16 outputs (M*N).  No 3rd input channel for
//     row-scales — the kernel has only 2 host-fed DMA streams.
//   * The engine's row-scale (per-output-row of W, length N) is folded into
//     C *post-mmul* on the host.  This is mathematically equivalent to
//     pre-scaling A only when M==1 (decode); for M>1 we'd need a per-row-of-A
//     fold, which doesn't apply on this lane.
//   * Compiled tile is 64x64x64 today.  Dispatch is gated on N%64==0 &&
//     K%64==0 in engine.cpp; this shim splits the work into 64x64x64 blocks
//     and zero-pads the M axis up from 1 to 64 (decode).  98% of the M
//     compute is wasted; smoke-test only until production tile widening
//     lands.

#ifndef ROCM_CPP_ENGINE_NPU_DISPATCH_H
#define ROCM_CPP_ENGINE_NPU_DISPATCH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle.  Owns the BitnetGemmAIE2P engine + scratch buffers used
// across dispatches.  Created once on first call, freed at Engine teardown.
typedef struct rcpp_npu_dispatch_t rcpp_npu_dispatch_t;

// Status codes.  Mirror onebit::aie::ErrorKind by intent; the integer
// values are local to this shim (engine.cpp doesn't peek inside).
typedef enum {
    RCPP_NPU_OK              = 0,
    RCPP_NPU_NOT_INITIALIZED = 1,
    RCPP_NPU_LOAD_FAILED     = 2,  // xclbin/insts missing or bad
    RCPP_NPU_DISPATCH_FAILED = 3,  // libxrt rejected the kernel run
    RCPP_NPU_HIP_ERROR       = 4,  // hipMemcpy failed
    RCPP_NPU_SHAPE_MISMATCH  = 5,  // N%64 or K%64 != 0 reached the shim
} rcpp_npu_status_t;

// Create + load the AIE2P engine from the given xclbin + insts files.
// `*out` is set to a fresh handle on success; NULL on failure.
//
// The error_buf (if non-null, capacity error_buf_size) gets a NUL-terminated
// reason string — useful for the engine to fprintf and disable the lane.
rcpp_npu_status_t
rcpp_npu_dispatch_create(const char*  xclbin_path,
                         const char*  insts_path,
                         char*        error_buf,
                         size_t       error_buf_size,
                         rcpp_npu_dispatch_t** out);

// Free the handle and its scratch.  Safe to call with NULL.
void rcpp_npu_dispatch_free(rcpp_npu_dispatch_t* h);

// Run a tiled bf16-acts × ternary-W GEMV.  All device pointers; the shim
// owns host scratch internally.
//
// Inputs (device memory):
//   packed_dev      : HALO_V2-packed weights, K*N/4 bytes (4 ternary codes
//                     per byte == 16 codes per u32).  Layout matches
//                     pack_halo_v2 in run_pyxrt_bitnet.py.
//   row_scales_dev  : fp32 [N] — per-output-row scale of W, applied
//                     post-mmul to C on the host.
//   normed_fp16_dev : fp16 [K] — single-row activation.  Cast to bf16
//                     on the host before submission.
//
// Output (device memory):
//   out_fp16_dev    : fp16 [N] — written via hipMemcpy(H2D) after the
//                     bf16 outputs are downcast to fp16.
//
// Caller pre-checks N%64==0 && K%64==0; shim returns RCPP_NPU_SHAPE_MISMATCH
// otherwise (defensive — should never fire).
rcpp_npu_status_t
rcpp_npu_dispatch_gemv(rcpp_npu_dispatch_t* h,
                       const void* packed_dev,        // u8* / u32* on device
                       const float* row_scales_dev,   // fp32 [N]
                       const void* normed_fp16_dev,   // fp16 [K]
                       void*       out_fp16_dev,      // fp16 [N]
                       int N, int K);

// Last-error string for the most recent gemv() call on this handle.
// Returned pointer is owned by the handle and stable until the next call.
// May return "" on success.
const char* rcpp_npu_dispatch_last_error(rcpp_npu_dispatch_t* h);

#ifdef __cplusplus
}
#endif

#endif // ROCM_CPP_ENGINE_NPU_DISPATCH_H
